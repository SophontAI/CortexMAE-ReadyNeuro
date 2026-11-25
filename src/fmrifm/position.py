"""Rotary position code over 4D (x, y, z, t) patch coordinates."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPEnD(nn.Module):
    """Simplex 4D-RoPE with complete scales and a position-free channel tail.

    All four axes share one frequency ladder: the temporal band is the spatial
    ladder restated in TRs, so ``config`` fixes the shape and nothing here is tunable.
    """

    def __init__(
        self,
        head_dim: int,
        n_heads: int,
        min_wavelength: float,
        max_wavelength: float,
    ):
        super().__init__()
        n_dims = 4
        n_directions = n_dims + 1

        self.head_dim = head_dim
        self.n_heads = n_heads
        self.min_wavelength = float(min_wavelength)
        self.max_wavelength = float(max_wavelength)
        # One temporal token is 2*pi temporal patches of the shortest spatial wavelength.
        self.time_scale = self.min_wavelength / (2 * math.pi)
        n_pairs = head_dim // 2
        n_scales = n_pairs // n_directions
        self.rotary_pairs = n_scales * n_directions

        simplex = torch.eye(n_directions, dtype=torch.float32)
        simplex -= torch.ones_like(simplex) / n_directions
        _, _, vh = torch.linalg.svd(simplex)
        simplex = F.normalize(simplex @ vh.T[:, :n_dims], dim=-1)

        generator = torch.Generator(device="cpu").manual_seed(0)
        rotations = []
        for _ in range(n_heads):
            matrix = torch.randn(n_dims, n_dims, generator=generator)
            rotation, triangular = torch.linalg.qr(matrix)
            signs = torch.sign(torch.diagonal(triangular))
            signs[signs == 0] = 1
            rotation = rotation * signs
            if torch.linalg.det(rotation) < 0:
                rotation[:, 0] *= -1
            rotations.append(rotation)
        rotations = torch.stack(rotations)
        wave_vectors = torch.einsum("md,hkd->hmk", simplex, rotations)
        if n_scales == 1:
            wavelengths = torch.tensor([math.sqrt(min_wavelength * max_wavelength)], dtype=torch.float32)
        else:
            steps = torch.arange(n_scales, dtype=torch.float32) / (n_scales - 1)
            wavelengths = min_wavelength * (max_wavelength / min_wavelength) ** steps

        self.register_buffer("wave_vectors", wave_vectors)
        self.register_buffer("wavelengths", wavelengths)

    def forward(self, coord: torch.Tensor, time_tokens: int) -> torch.Tensor:
        """Return concatenated sin/cos as ``[B, H, T, N, 2D]``."""
        if coord.ndim != 3 or coord.shape[-1] != 3:
            raise ValueError(f"coord must have shape [B, N, 3], got {tuple(coord.shape)}")
        if time_tokens <= 0:
            raise ValueError(f"time_tokens must be positive, got {time_tokens}")

        # Compute fixed features in fp32 even if the module was cast.
        dtype = torch.float32
        batch_size, n_patches = coord.shape[:2]
        spatial = coord.to(dtype=dtype)[:, None].expand(-1, time_tokens, -1, -1)
        time = self.time_scale * torch.arange(
            time_tokens, device=self.wave_vectors.device, dtype=dtype
        )
        time = time[None, :, None, None].expand(batch_size, -1, n_patches, -1)
        position = torch.cat((spatial, time), dim=-1)
        projection = torch.einsum("btnd,hmd->bhtnm", position, self.wave_vectors.float())

        angle = (2 * math.pi * projection[..., None] / self.wavelengths.float()).flatten(-2)
        unused_pairs = self.head_dim // 2 - self.rotary_pairs
        if unused_pairs:
            angle = F.pad(angle, (0, unused_pairs))
        sin = angle.sin().repeat_interleave(2, dim=-1)
        cos = angle.cos().repeat_interleave(2, dim=-1)
        return torch.cat((sin, cos), dim=-1)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Rotate ``[B, H, N, D]`` by the sin/cos pair in ``rope``."""
    sin, cos = rope.chunk(2, dim=-1)
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    # Keep sin/cos in fp32 and round once after rotation.
    return (x * cos + rotated * sin).to(dtype=x.dtype)
