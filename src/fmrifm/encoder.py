"""ViT encoder over one flattened space-time sequence of patch tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange, repeat

from fmrifm.blocks import Block
from fmrifm.position import RoPEnD


def fold_time(x: torch.Tensor, temporal_patch_size: int) -> torch.Tensor:
    """Fold adjacent frames into the feature dimension of each spatial patch."""
    if x.ndim != 4:
        raise ValueError(f"x must have shape [B, T, N, P], got {tuple(x.shape)}")
    if x.shape[1] % temporal_patch_size:
        raise ValueError(
            f"the number of frames ({x.shape[1]}) must be divisible by "
            f"temporal_patch_size ({temporal_patch_size})"
        )
    return rearrange(x, "b (t pt) n ps -> b t n (pt ps)", pt=temporal_patch_size)


class Encoder(nn.Module):
    """Flattened space-time ViT; every token carries a world-millimetre coordinate."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        depth: int,
        patch_size: int,
        temporal_patch_size: int,
        drop_path_rate: float = 0.0,
        **band,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.patch_dim = patch_size**3 * temporal_patch_size

        self.input_proj = nn.Linear(self.patch_dim, dim, bias=False)

        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    n_heads=n_heads,
                    mlp_ratio=4.0,
                    drop_path=rate,
                    norm_layer=nn.RMSNorm,
                )
                for rate in rates
            ]
        )
        self.norm = nn.RMSNorm(dim)
        self.rope = RoPEnD(head_dim=dim // n_heads, n_heads=n_heads, **band)

    def _validate_inputs(self, x: torch.Tensor, coord: torch.Tensor, valid: torch.Tensor | None):
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B, T, N, P], got {tuple(x.shape)}")
        if x.shape[-1] != self.patch_size**3:
            raise ValueError(f"x has patch width {x.shape[-1]}, expected {self.patch_size**3}")
        batch_size, _, n_patches, _ = x.shape
        if coord.shape != (batch_size, n_patches, 3):
            raise ValueError(f"coord must have shape {(batch_size, n_patches, 3)}, got {tuple(coord.shape)}")
        if valid is not None and valid.shape != (batch_size, n_patches):
            raise ValueError(f"valid must have shape {(batch_size, n_patches)}, got {tuple(valid.shape)}")
        return n_patches

    def forward(
        self,
        x: torch.Tensor,
        coord: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode ``x [B,T,N,P]`` at ``coord [B,N,3]`` to patch tokens ``[B,T',N,D]``."""
        n_patches = self._validate_inputs(x, coord, valid)

        if valid is not None:
            valid = valid.to(device=x.device, dtype=torch.bool)
            invalid = ~valid
            x = x.masked_fill(invalid[:, None, :, None], 0)
            coord = coord.masked_fill(invalid[..., None], 0)

        hidden = self.input_proj(fold_time(x, self.temporal_patch_size))
        time_tokens = hidden.shape[1]
        sequence = rearrange(hidden, "b t n d -> b (t n) d")

        attention_mask = None
        if valid is not None:
            attention_mask = repeat(valid, "b n -> b (t n)", t=time_tokens)[:, None, None, :]

        rope = rearrange(self.rope(coord, time_tokens), "b h t n d -> b h (t n) d")
        for block in self.blocks:
            sequence = block(sequence, rope=rope, attn_mask=attention_mask)
        sequence = self.norm(sequence)

        patches = rearrange(sequence, "b (t n) d -> b t n d", t=time_tokens, n=n_patches)
        if valid is not None:
            patches = patches.masked_fill(~valid[:, None, :, None], 0)
        return patches
