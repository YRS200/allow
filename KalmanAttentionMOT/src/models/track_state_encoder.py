"""Track history encoding module."""
from __future__ import annotations

import torch
from torch import nn


class TrackStateEncoder(nn.Module):
    """Encode per-track history tensors using a GRU stack."""

    def __init__(self, input_dim: int = 21, hidden_dim: int = 128, bidirectional: bool = True) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.output_dim = hidden_dim

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Encode a batch of track histories."""

        if history.size(0) == 0:
            return history.new_zeros(0, self.output_dim)
        _, hidden = self.gru(history)
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden = hidden[-1]
        return self.proj(hidden)


__all__ = ["TrackStateEncoder"]
