"""Masked reconstruction of held-out fMRI patches from an encoded context."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import repeat

from fmrifm.decoder import Decoder
from fmrifm.encoder import Encoder, fold_time


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared error over real target voxels only."""
    prediction, target = prediction.float(), target.float()
    if loss_mask is None:
        return (prediction - target).square().mean()
    loss_mask = loss_mask.to(device=prediction.device, dtype=torch.bool)
    prediction = prediction.masked_fill(~loss_mask, 0)
    target = target.masked_fill(~loss_mask, 0)
    return (prediction - target).square().sum() / loss_mask.sum()


class MaskedAutoencoder(nn.Module):
    """Encode loader-selected context patches and predict its targets."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        depth: int,
        patch_size: int,
        temporal_patch_size: int,
        decoder_dim: int,
        decoder_heads: int,
        decoder_depth: int,
        drop_path_rate: float,
        min_wavelength: float,
        max_wavelength: float,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size

        band = dict(min_wavelength=min_wavelength, max_wavelength=max_wavelength)
        self.encoder = Encoder(
            dim=dim,
            n_heads=n_heads,
            depth=depth,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            drop_path_rate=drop_path_rate,
            **band,
        )
        self.decoder = Decoder(
            encoder_dim=dim,
            dim=decoder_dim,
            n_heads=decoder_heads,
            depth=decoder_depth,
            output_dim=patch_size**3 * temporal_patch_size,
            drop_path_rate=drop_path_rate,
            **band,
        )

    def _reconstruction_loss_mask(
        self,
        target: torch.Tensor,
        target_voxel_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if target_voxel_mask is None:
            return None
        expected_shape = (target.shape[0], target.shape[2], self.patch_size**3)
        if target_voxel_mask.shape != expected_shape:
            raise ValueError(
                f"target_voxel_mask must have shape {expected_shape}, got {tuple(target_voxel_mask.shape)}"
            )
        return repeat(
            target_voxel_mask.to(device=target.device, dtype=torch.bool),
            "b n p -> b t n (pt p)",
            t=target.shape[1],
            pt=self.temporal_patch_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        coord: torch.Tensor,
        target: torch.Tensor | None = None,
        target_coord: torch.Tensor | None = None,
        target_voxel_mask: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        compute_loss: bool | None = None,
    ):
        """Encode, or encode and score the reconstruction.

        ``x [B,T,N,P]`` uses ``coord [B,N,3]`` and optional ``valid [B,N]``.
        Targets are ``[B,T,M,P]`` at ``target_coord [B,M,3]`` with an optional
        spatial-voxel mask ``[B,M,P]``.
        """
        compute_loss = self.training if compute_loss is None else compute_loss
        if not compute_loss:
            return self.encoder(x, coord, valid=valid)
        if target is None or target_coord is None:
            raise ValueError("reconstruction needs the loader's target patches")

        hidden = self.encoder(x, coord, valid=valid)
        prediction = self.decoder(hidden, coord, target_coord, context_valid=valid)
        target = fold_time(target, self.temporal_patch_size)
        loss = _masked_mse(prediction, target, self._reconstruction_loss_mask(target, target_voxel_mask))
        return loss, hidden, prediction, target


def build_model(config) -> MaskedAutoencoder:
    cfg = config.model
    return MaskedAutoencoder(
        dim=cfg.dim,
        n_heads=cfg.n_heads,
        depth=cfg.depth,
        patch_size=cfg.patch_size,
        temporal_patch_size=cfg.temporal_patch_size,
        decoder_dim=cfg.decoder_dim,
        decoder_heads=cfg.decoder_heads,
        decoder_depth=cfg.decoder_depth,
        drop_path_rate=cfg.drop_path_rate,
        min_wavelength=cfg.resolved_pos_min_wavelength,
        max_wavelength=cfg.pos_max_wavelength,
    )
