"""Transformer blocks with rotary attention and row-wise stochastic depth."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp

from fmrifm.position import apply_rope


def take_batch_rows(
    tensor: torch.Tensor | None,
    batch_index: torch.Tensor | None,
    batch_size: int,
):
    if tensor is None or batch_index is None or tensor.shape[0] != batch_size:
        return tensor
    return tensor[batch_index]


def add_dropped_residual(x, branch, drop_prob: float, training: bool):
    """Run ``branch(subset, index)`` on kept rows, preserving expectation."""
    if not training or drop_prob <= 0.0:
        return x + branch(x, None)
    keep = torch.rand(x.shape[0], device=x.device) >= drop_prob
    index = keep.nonzero(as_tuple=True)[0]
    # Empty index_add is a no-op; branching on `index` would break compilation.
    residual = branch(x[index], index)
    return x.index_add(0, index, residual.to(dtype=x.dtype), alpha=1.0 / (1.0 - drop_prob))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        norm_layer: type[nn.Module] = nn.RMSNorm,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = norm_layer(self.head_dim)
        self.k_norm = norm_layer(self.head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, n_tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, n_tokens, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query, key = self.q_norm(query), self.k_norm(key)

        query = apply_rope(query, rope).type_as(value)
        key = apply_rope(key, rope).type_as(value)

        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        # Empty stochastic-depth batches require an explicit output width.
        attended = attended.transpose(1, 2).reshape(batch_size, n_tokens, self.n_heads * self.head_dim)
        return self.proj(attended)


class Block(nn.Module):
    """Pre-norm attention and MLP, each on its own residual."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.RMSNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, n_heads, norm_layer=norm_layer)
        self.drop_prob = float(drop_path)

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        def attention(subset, index):
            return self.attn(
                self.norm1(subset),
                rope=take_batch_rows(rope, index, batch_size),
                attn_mask=take_batch_rows(attn_mask, index, batch_size),
            )

        x = add_dropped_residual(x, attention, self.drop_prob, self.training)

        def feedforward(subset, _index):
            return self.mlp(self.norm2(subset))

        return add_dropped_residual(x, feedforward, self.drop_prob, self.training)
