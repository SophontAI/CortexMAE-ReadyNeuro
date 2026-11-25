import torch
import torch.nn as nn


class OnlineLinearProbe(nn.Module):
    """Detached linear age/sex probe."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 2)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        if representation.ndim < 3:
            raise ValueError(
                "online probe expects representation shaped [batch, ..., dim], "
                f"got {tuple(representation.shape)}"
            )
        token_axes = tuple(range(1, representation.ndim - 1))
        return self.linear(representation.detach().mean(dim=token_axes))
