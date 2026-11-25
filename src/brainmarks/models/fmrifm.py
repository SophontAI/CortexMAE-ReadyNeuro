"""Brainmarks transform and frozen-encoder wrapper for the pretrained model."""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from brainmarks.models.base import Embeddings
from brainmarks.models.registry import register_model
from einops import rearrange, repeat

from fmrifm.checkpoint import load_weights
from fmrifm.config import parse_config
from fmrifm.model import MaskedAutoencoder, build_model
from fmrifm.pretrain_data import build_patch_grid, standardize_frames, world_coordinates

_GEOMETRY_FIELDS = ("idx", "cortical_gm", "shape")
_OUTPUT_META = ("target", "cond_id", "n_frames", "tr")
_CACHE_ENTRIES = 32
# Headroom over the largest training-split subject, so an unseen test subject
# with a slightly bigger mask still fits the padded budget.
FIT_MARGIN = 1.1


def _bold_fields(sample: dict) -> dict:
    return {field: sample[f"bold_{field}"] for field in _GEOMETRY_FIELDS}


def _layout(decoded: dict, *, patch_size: int, threshold: float):
    keep = np.asarray(decoded["cortical_gm"], dtype=bool)
    mask = np.zeros(tuple(decoded["shape"]), dtype=bool)
    mask.flat[np.asarray(decoded["idx"], dtype=np.int64)[keep]] = True
    return keep, build_patch_grid(mask, patch_size=patch_size, threshold=threshold)


def _prepare(sample: dict, layout, *, clip_value: float):
    keep, patch_grid = layout
    frame_by_voxel = standardize_frames(np.asarray(sample["bold_values"])[keep].T.astype(np.float32))
    patches = patch_grid.gather(frame_by_voxel)
    return (
        np.clip(patches, -clip_value, clip_value).astype(np.float32),
        world_coordinates(patch_grid.centers, sample["bold_affine"]),
    )


def _pad(patches: np.ndarray, centers: np.ndarray, max_patches: int) -> dict:
    n_patches = len(centers)
    if n_patches > max_patches:
        raise ValueError(f"sample has {n_patches} patches, max_patches is {max_patches}")
    x = np.zeros((patches.shape[0], max_patches, patches.shape[2]), dtype=np.float32)
    coord = np.zeros((max_patches, 3), dtype=np.float32)
    valid = np.zeros(max_patches, dtype=bool)
    x[:, :n_patches] = patches
    coord[:n_patches] = centers
    valid[:n_patches] = True
    return {
        "x": torch.from_numpy(x),
        "coord": torch.from_numpy(coord),
        "valid": torch.from_numpy(valid),
    }


class EvalTransform:
    """Apply training preprocessing to packed evaluation samples."""

    def __init__(
        self,
        *,
        patch_size: int,
        clip_value: float,
        threshold: float,
    ) -> None:
        self.patch_size = int(patch_size)
        self.threshold = float(threshold)
        self.clip_value = float(clip_value)
        self.max_patches: int | None = None
        self._layouts: OrderedDict[tuple, tuple] = OrderedDict()

    def _make_layout(self, sample: dict):
        return _layout(
            _bold_fields(sample),
            patch_size=self.patch_size,
            threshold=self.threshold,
        )

    def fit(self, train_dataset) -> None:
        """Size the padded patch budget from the TRAIN split alone."""
        geometries = train_dataset.func_geometries()
        counts = [len(self._make_layout(geometry)[1]) for geometry in geometries]
        if not counts:
            raise RuntimeError("no training subjects probed")
        if max(counts) < 1:
            raise RuntimeError("training geometries produced no usable patches")
        self.max_patches = math.ceil(max(counts) * FIT_MARGIN)

    def __call__(self, sample: dict) -> dict:
        if self.max_patches is None:
            raise RuntimeError("max_patches unset; call transform.fit() first")

        # NSD can have run-specific geometry, represented by ``task`` as ``ses_run``.
        cache_key = (sample["dataset"], sample["space"], sample["sub"], sample["task"])
        if cache_key not in self._layouts:
            if len(self._layouts) >= _CACHE_ENTRIES:
                self._layouts.popitem(last=False)
            self._layouts[cache_key] = self._make_layout(sample)
        patches, centers = _prepare(sample, self._layouts[cache_key], clip_value=self.clip_value)
        output = _pad(patches, centers, self.max_patches)
        output.update({name: sample[name] for name in _OUTPUT_META})
        return output


class EvalWrapper(nn.Module):
    """Encode a padded scan in fixed-length clips for Brainmarks probes."""

    def __init__(self, model: MaskedAutoencoder, clip_frames: int) -> None:
        super().__init__()
        if clip_frames % model.temporal_patch_size:
            raise ValueError(
                f"clip_frames ({clip_frames}) must be divisible by temporal_patch_size "
                f"({model.temporal_patch_size})"
            )
        # Frozen by eval() plus torch.inference_mode() in forward rather than by
        # requires_grad_: no autograd graph is built at all, and stochastic depth is off.
        self.model = model.eval()
        self.clip_frames = int(clip_frames)

    def forward(self, batch: dict) -> Embeddings:
        x = batch["x"]
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B, T, N, P], got {tuple(x.shape)}")
        batch_size, frames = x.shape[0], x.shape[1]
        if frames < self.clip_frames:
            raise ValueError(f"scan has {frames} frames, needs at least {self.clip_frames}")

        # Scan lengths are not generally multiples of clip_frames -- PPMI, ABIDE and
        # ADHD-200 are 300 frames against clip_frames=16 (AABC's 400 already divides).
        # Reflect the tail back on itself rather than discarding it: unlike zero or edge
        # padding this leaves no step discontinuity or constant run to read as signal.
        pad_frames = (-frames) % self.clip_frames
        if pad_frames:
            # A true reflection starts at frames-2, so the final frame is never repeated.
            mirror = torch.arange(frames - 2, frames - 2 - pad_frames, -1, device=x.device)
            x = torch.cat([x, x.index_select(1, mirror)], dim=1)
            frames = frames + pad_frames

        num_clips = frames // self.clip_frames
        x = rearrange(x, "b (c t) n p -> (b c) t n p", c=num_clips)
        coord = repeat(batch["coord"], "b n d -> (b c) n d", c=num_clips)
        valid = batch["valid"].to(dtype=torch.bool)
        clip_valid = repeat(valid, "b n -> (b c) n", c=num_clips)

        # fp16, not the bf16 the model trains in: this is what produced the paper's
        # benchmark numbers, so changing it would make them not comparable.
        with torch.inference_mode(), torch.autocast(x.device.type, dtype=torch.float16):
            hidden = self.model(x.float(), coord.float(), valid=clip_valid, compute_loss=False)

        time_tokens, dim = hidden.shape[1], hidden.shape[-1]
        hidden = rearrange(hidden, "(b c) t n d -> b (c t n) d", b=batch_size, c=num_clips)
        token_valid = repeat(valid, "b n -> b (c t n)", c=num_clips, t=time_tokens)
        hidden = (hidden * token_valid.unsqueeze(-1).to(hidden.dtype)).float()
        # Fourth field is the padding mask; the attention probe honours it.
        return Embeddings(None, None, hidden.reshape(batch_size, -1, dim), ~token_valid)


def _make(space: str, *, ckpt_path=None, cfg_path=None, compile_model: bool = True):
    if space not in {"native", "mni"}:
        raise ValueError(f"space must be 'native' or 'mni', got {space!r}")
    resolved_cfg = Path(cfg_path) if cfg_path is not None else None
    if resolved_cfg is None and ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        # An exported encoder carries its config beside it; a training checkpoint, in its run.
        if ckpt_path.suffix == ".safetensors":
            resolved_cfg = ckpt_path.with_name("config.yaml")
        else:
            resolved_cfg = ckpt_path.parent.parent / "config.yaml"
        if not resolved_cfg.exists():
            raise FileNotFoundError(f"no model config supplied and {resolved_cfg} does not exist")
    config = parse_config(["--config", str(resolved_cfg)] if resolved_cfg is not None else [])

    model = build_model(config)
    if ckpt_path is not None:
        load_weights(ckpt_path, model)
    if compile_model:
        # Keep the Brainmarks wrapper and its Embeddings result eager.
        model.encoder = torch.compile(model.encoder)
    wrapper = EvalWrapper(model, clip_frames=config.preproc.clip_frames)
    # Read upstream to pick the matching evaluation store.
    wrapper.__space__ = space

    transform = EvalTransform(
        patch_size=config.model.patch_size,
        clip_value=config.preproc.clip_value,
        threshold=config.preproc.patch_valid_fraction_threshold,
    )
    return transform, wrapper


@register_model
def fmrifm_native(**kwargs):
    return _make("native", **kwargs)


@register_model
def fmrifm_mni(**kwargs):
    return _make("mni", **kwargs)
