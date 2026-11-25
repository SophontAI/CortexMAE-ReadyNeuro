"""Shapes out of the encoder/decoder pair, and the 96/32 context/target draw."""

import numpy as np
import pytest
import torch

from fmrifm.config import parse_config
from fmrifm.masking import sample_context_and_target
from fmrifm.model import build_model

# A ViT-L is too big for CI; the shapes under test do not depend on width.
TINY = [
    "--model.dim=40",
    "--model.n_heads=2",
    "--model.depth=2",
    "--model.decoder_dim=20",
    "--model.decoder_head_dim=10",
    "--model.decoder_depth=1",
    "--model.patch_size=2",
    "--model.temporal_patch_size=2",
    "--preproc.clip_frames=4",
]


def tiny_model():
    return build_model(parse_config(TINY))


def batch(batch_size=2, frames=4, context=5, target=3, patch=8):
    return {
        "x": torch.randn(batch_size, frames, context, patch),
        "coord": torch.randn(batch_size, context, 3),
        "target": torch.randn(batch_size, frames, target, patch),
        "target_coord": torch.randn(batch_size, target, 3),
    }


def test_reconstruction_predicts_one_spatiotemporal_patch_per_target():
    model = tiny_model()

    loss, hidden, prediction, target = model(**batch())

    assert loss.ndim == 0 and torch.isfinite(loss)
    # Frames collapse by temporal_patch_size; context patches survive one for one.
    assert hidden.shape == (2, 2, 5, 40)
    # Each target token carries patch_size**3 voxels x temporal_patch_size frames.
    assert prediction.shape == target.shape == (2, 2, 3, 16)

    loss.backward()
    assert model.encoder.input_proj.weight.grad is not None
    assert model.decoder.output_proj.weight.grad is not None


def test_inference_returns_one_token_per_context_patch_and_no_loss():
    model = tiny_model().eval()
    sample = batch()

    hidden = model(sample["x"], sample["coord"])

    assert isinstance(hidden, torch.Tensor)
    assert hidden.shape == (2, 2, 5, 40)


def test_frame_count_must_be_divisible_by_the_temporal_patch_size():
    with pytest.raises(ValueError):
        tiny_model()(**batch(frames=3))


def test_the_draw_takes_96_context_and_32_disjoint_target_patches():
    coords = np.random.default_rng(0).random((3000, 3)) * 60

    context, target = sample_context_and_target(coords, 96, 32, torch.Generator().manual_seed(7))

    assert context.shape == (96,) and target.shape == (32,)
    assert len(set(context.tolist())) == 96
    assert len(set(target.tolist())) == 32
    assert not set(context.tolist()) & set(target.tolist())


def test_the_draw_is_reproducible_and_resamples_per_call():
    coords = np.random.default_rng(4).random((1500, 3)) * 40

    first = sample_context_and_target(coords, 96, 32, torch.Generator().manual_seed(11))
    again = sample_context_and_target(coords, 96, 32, torch.Generator().manual_seed(11))
    generator = torch.Generator().manual_seed(11)
    a = sample_context_and_target(coords, 96, 32, generator)
    b = sample_context_and_target(coords, 96, 32, generator)

    assert all(np.array_equal(x, y) for x, y in zip(first, again, strict=True))
    assert not np.array_equal(a[0], b[0])


def test_a_subject_smaller_than_the_budget_is_rejected_rather_than_padded():
    coords = np.random.default_rng(5).random((100, 3))

    with pytest.raises(ValueError, match="cannot supply"):
        sample_context_and_target(coords, 96, 32, torch.Generator().manual_seed(0))
