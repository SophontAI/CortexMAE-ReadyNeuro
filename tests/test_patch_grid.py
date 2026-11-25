"""The three preprocessing claims the paper rests on, none of which fail loudly.

Flip the z-score axis, roll the patch centres off their contents, or read whole-brain
grey matter instead of cortical, and the run still produces finite, plausible numbers.
"""

import numpy as np

from fmrifm.pretrain_data import PatchSampler, build_patch_grid, standardize_frames, world_coordinates

AFFINE = np.array([[2.0, 0, 0, 10.0], [0, 2.0, 0, 20.0], [0, 0, 2.0, 30.0], [0, 0, 0, 1.0]])


def test_each_patch_center_is_the_world_mean_of_the_voxels_it_gathers():
    """A rolled or transposed centre invalidates the whole native-vs-MNI argument."""
    mask = np.ones((4, 4, 4), bool)
    grid = build_patch_grid(mask, patch_size=2, threshold=0.0)
    world = world_coordinates(np.argwhere(mask), AFFINE)

    np.testing.assert_allclose(
        grid.gather(world.T).mean(axis=-1).T, world_coordinates(grid.centers, AFFINE), rtol=1e-5
    )


def test_frames_are_z_scored_across_voxels_not_over_time():
    """The pipeline already z-scored each voxel over time; this is the other axis."""
    clip = np.random.default_rng(0).normal(size=(8, 50)) + np.arange(8)[:, None] * 100.0

    standardized = standardize_frames(clip)

    np.testing.assert_allclose(standardized.mean(axis=1), 0.0, atol=1e-4)
    np.testing.assert_allclose(standardized.std(axis=1), 1.0, atol=1e-4)


def test_the_layout_is_built_from_cortical_grey_matter_only():
    """Whole-brain grey matter is present in every bundle and trains just as happily."""
    bundle = {
        "bold_idx": np.arange(64),
        "bold_gm": np.ones(64, np.uint8),
        "bold_cortical_gm": np.array([1] * 8 + [0] * 56, np.uint8),
        "bold_affine": AFFINE,
        "bold_shape": (4, 4, 4),
    }
    sampler = PatchSampler(
        context_patches=1,
        target_patches=1,
        patch_size=2,
        clip_frames=16,
        clip_value=3.0,
        patch_valid_fraction_threshold=0.1,
    )

    assert len(sampler.build_layout(bundle).bold_grid) == 2
