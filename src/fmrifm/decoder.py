"""Cross-attention decoder: learned target queries read the encoder's context."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fmrifm.blocks import add_dropped_residual, take_batch_rows
from fmrifm.position import RoPEnD, apply_rope


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.qk_head_dim = query_dim // n_heads
        self.query = nn.Linear(query_dim, query_dim, bias=False)
        self.key = nn.Linear(context_dim, query_dim, bias=False)
        self.value = nn.Linear(context_dim, query_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.qk_head_dim)
        self.k_norm = nn.RMSNorm(self.qk_head_dim)
        self.output = nn.Linear(query_dim, query_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_rope: torch.Tensor,
        context_rope: torch.Tensor,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, n_queries, _ = query.shape
        context_patches = context.shape[1]
        query = self.query(query).reshape(batch_size, n_queries, self.n_heads, self.qk_head_dim)
        key = self.key(context).reshape(batch_size, context_patches, self.n_heads, self.qk_head_dim)
        value = self.value(context).reshape(batch_size, context_patches, self.n_heads, self.qk_head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        query, key = self.q_norm(query), self.k_norm(key)
        query = apply_rope(query, query_rope)
        key = apply_rope(key, context_rope)
        value = value.transpose(1, 2)

        attention_mask = None
        if key_valid is not None:
            attention_mask = key_valid[:, None, None, :]
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        return self.output(attended.transpose(1, 2).flatten(2))


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        n_heads: int,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.query_norm = nn.RMSNorm(dim)
        self.context_norm = nn.RMSNorm(context_dim)
        self.attention = CrossAttention(dim, context_dim, n_heads)
        self.mlp_norm = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * dim, dim, bias=False),
        )
        self.drop_prob = float(drop_path_rate)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_rope: torch.Tensor,
        context_rope: torch.Tensor,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = query.shape[0]

        def attention(subset, index):
            return self.attention(
                self.query_norm(subset),
                self.context_norm(take_batch_rows(context, index, batch_size)),
                take_batch_rows(query_rope, index, batch_size),
                take_batch_rows(context_rope, index, batch_size),
                take_batch_rows(key_valid, index, batch_size),
            )

        query = add_dropped_residual(query, attention, self.drop_prob, self.training)

        def feedforward(subset, _index):
            return self.mlp(self.mlp_norm(subset))

        return add_dropped_residual(query, feedforward, self.drop_prob, self.training)


class Decoder(nn.Module):
    """Independent target queries cross-attend to the encoder context over space-time."""

    def __init__(
        self,
        encoder_dim: int,
        dim: int,
        n_heads: int,
        depth: int,
        output_dim: int,
        drop_path_rate: float = 0.0,
        **band,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.mask_token = nn.Parameter(torch.empty(1, 1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)
        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(dim, encoder_dim, n_heads, drop_path_rate=rate)
                for rate in rates
            ]
        )
        self.norm = nn.RMSNorm(dim)
        self.output_proj = nn.Linear(dim, output_dim, bias=False)
        self.rope = RoPEnD(head_dim=dim // n_heads, n_heads=n_heads, **band)

    def forward(
        self,
        context: torch.Tensor,
        context_coord: torch.Tensor,
        target_coord: torch.Tensor,
        context_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``[B, T, M, D_out]`` target predictions."""
        batch_size, time_tokens, context_patches, context_dim = context.shape
        if context_dim != self.encoder_dim:
            raise ValueError(f"context must have width {self.encoder_dim}, got {context_dim}")
        if context_coord.shape != (batch_size, context_patches, 3):
            raise ValueError(
                f"context_coord must have shape {(batch_size, context_patches, 3)}, "
                f"got {tuple(context_coord.shape)}"
            )
        if context_valid is not None:
            context_valid = context_valid.to(device=context.device, dtype=torch.bool)
            invalid = ~context_valid
            context = context.masked_fill(invalid[:, None, :, None], 0)
            context_coord = context_coord.masked_fill(invalid[..., None], 0)
        if target_coord.ndim != 3 or target_coord.shape[0] != batch_size:
            raise ValueError(
                f"target_coord must have shape [{batch_size}, M, 3], got {tuple(target_coord.shape)}"
            )
        target_patches = target_coord.shape[1]

        query = self.mask_token.expand(batch_size, time_tokens, target_patches, -1).to(
            dtype=context.dtype
        )
        query = rearrange(query, "b t n d -> b (t n) d")
        keys = rearrange(context.to(dtype=query.dtype), "b t n d -> b (t n) d")
        query_rope = rearrange(self.rope(target_coord, time_tokens), "b h t n d -> b h (t n) d")
        context_rope = rearrange(self.rope(context_coord, time_tokens), "b h t n d -> b h (t n) d")
        key_valid = None
        if context_valid is not None:
            key_valid = repeat(context_valid, "b n -> b (t n)", t=time_tokens)

        for block in self.blocks:
            query = block(query, keys, query_rope, context_rope, key_valid)

        query = self.output_proj(self.norm(query))
        return rearrange(query, "b (t n) d -> b t n d", t=time_tokens, n=target_patches)
