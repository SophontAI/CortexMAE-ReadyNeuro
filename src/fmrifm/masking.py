"""Uniform context patches with targets uniform over their complement."""

import numpy as np
import torch


def sample_context_and_target(
    coords,
    context_patches: int,
    target_patches: int,
    generator: torch.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``context_patches`` cells uniformly, then targets from the complement."""
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"coords must be [N, 3], got {tuple(coords.shape)}")
    n_patches = coords.shape[0]
    if context_patches + target_patches > n_patches:
        raise ValueError(
            f"a subject with {n_patches} patches cannot supply {context_patches} context "
            f"plus {target_patches} target patches"
        )

    noise = torch.rand(1, n_patches, generator=generator)
    context = noise.argsort(dim=1)[:, :context_patches]

    # Push context to the back of a fresh random ordering of the complement.
    taken = torch.zeros(1, n_patches, dtype=torch.bool)
    taken.scatter_(1, context, True)
    noise = torch.rand(1, n_patches, generator=generator).masked_fill(taken, float("inf"))
    target = noise.argsort(dim=1)[:, :target_patches]
    return context[0].numpy(), target[0].numpy()
