"""Detection feature encoder."""
from __future__ import annotations

import torch
from torch import nn


class DetectionEncoder(nn.Module):
    """Encode detection attributes into a fixed dimensional embedding."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.output_dim = hidden_dim

    def forward(self, dets: torch.Tensor) -> torch.Tensor:
        if dets.size(0) == 0:
            return dets.new_zeros(0, self.output_dim)
        return self.net(dets)


__all__ = ["DetectionEncoder"]
