"""EvalWrapper packs a scan into clips and tells the probe which tokens are padding.

The resting-state evaluation sets are all 300 frames against clip_frames=16. The tail
used to be discarded; padding keeps every acquired frame, and getting the reflection
wrong (edge or zero padding, or repeating the final frame) would show up only as
slightly worse numbers on five of the seven targets. Subjects also differ in patch
count, so a mask that does not line up with the flattened tokens pools the wrong ones.

``conftest.py`` stubs Brainmarks when the optional ``eval`` extra is absent, so these
run on a stock checkout.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from brainmarks.models.fmrifm import EvalWrapper


class StubModel(nn.Module):
    """Records the frame count each clip actually carried."""

    def __init__(self, temporal_patch_size=4, dim=8):
        super().__init__()
        self.temporal_patch_size = temporal_patch_size
        self.dim = dim
        self.seen = []

    def forward(self, x, coord, *, valid=None, compute_loss=False):
        clips, frames, patches, _ = x.shape
        self.seen.append((clips, frames, patches))
        self.last_x = x
        # A distinct nonzero value per (clip, frame, patch), so the flattening order
        # and the clip-to-subject split can both be read off the output.
        time = frames // self.temporal_patch_size
        ids = torch.arange(1, clips * time * patches + 1, dtype=torch.float32)
        return ids.reshape(clips, time, patches, 1).expand(-1, -1, -1, self.dim).contiguous()


def _batch(frames, n_patches=3, batch_size=1):
    # The patch feature is the frame index, so the reflection can be read back out.
    x = torch.arange(frames, dtype=torch.float32)
    x = x.reshape(1, frames, 1, 1).expand(batch_size, frames, n_patches, 2).contiguous()
    return {
        "x": x,
        "coord": torch.zeros(batch_size, n_patches, 3),
        "valid": torch.ones(batch_size, n_patches, dtype=torch.bool),
    }


def _frame_sequence(model):
    return model.last_x.reshape(-1, model.last_x.shape[2], 2)[:, 0, 0].numpy()


@pytest.mark.parametrize("frames,clips", [(16, 1), (400, 25)])
def test_divisible_scans_are_untouched(frames, clips):
    """hcpya/nsd (16 frames) and aabc (400) must be bit-identical to no padding."""
    model = StubModel()
    EvalWrapper(model, clip_frames=16)(_batch(frames))

    assert model.seen[0][:2] == (clips, 16)
    assert np.array_equal(_frame_sequence(model), np.arange(frames, dtype=np.float32))


def test_300_frames_become_19_clips_by_true_reflection():
    model = StubModel()
    EvalWrapper(model, clip_frames=16)(_batch(300))

    assert model.seen[0][:2] == (19, 16)
    seen = _frame_sequence(model)
    assert np.array_equal(seen, np.pad(np.arange(300, dtype=np.float32), (0, 4), mode="reflect"))
    assert np.array_equal(seen[:300], np.arange(300)), "no acquired frame may be dropped"
    assert seen[300] == 298, "a true reflection does not repeat the final frame"


def test_a_scan_shorter_than_one_clip_is_refused():
    with pytest.raises(ValueError, match="needs at least"):
        EvalWrapper(StubModel(), clip_frames=16)(_batch(8))


def test_the_padding_mask_lines_up_with_the_flattened_tokens():
    """The probe pools by this mask; a shifted one silently pools padded tokens."""
    model = StubModel()
    batch = _batch(32, n_patches=3, batch_size=2)
    batch["valid"][1, 2] = False  # the second subject's mask is one patch smaller

    _, _, tokens, pad = EvalWrapper(model, clip_frames=16)(batch)

    per_subject = tokens.shape[1]  # 2 clips x 4 temporal tokens x 3 patches
    assert pad.shape == (2, per_subject) and per_subject == 24
    # Tokens are emitted subject-major, then clip, then frame, then patch.
    ids = tokens[..., 0]
    assert torch.equal(ids[0], torch.arange(1, per_subject + 1, dtype=ids.dtype))
    second = torch.arange(per_subject + 1, 2 * per_subject + 1, dtype=ids.dtype)
    second[pad[1]] = 0
    assert torch.equal(ids[1], second)
    # The mask marks a token exactly when that token was zeroed.
    assert torch.equal(pad, ids == 0)
    assert not pad[0].any() and int(pad[1].sum()) == per_subject // 3
