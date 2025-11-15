"""Geometric helper functions for KalmanAttentionMOT."""
from __future__ import annotations

from typing import List

import torch


def bbox_center(box: torch.Tensor) -> torch.Tensor:
    """Return bounding box centers."""

    return box[..., :2]


def bbox_area(box: torch.Tensor) -> torch.Tensor:
    """Return bounding box areas."""

    return box[..., 2] * box[..., 3]


def xywh_to_cxcywh(box: torch.Tensor) -> torch.Tensor:
    """Convert ``[x, y, w, h]`` boxes to ``[cx, cy, w, h]`` format."""

    cx = box[..., 0] + box[..., 2] / 2.0
    cy = box[..., 1] + box[..., 3] / 2.0
    wh = box[..., 2:4]
    return torch.cat([cx.unsqueeze(-1), cy.unsqueeze(-1), wh], dim=-1)


def cxcywh_to_xywh(box: torch.Tensor) -> torch.Tensor:
    """Convert ``[cx, cy, w, h]`` boxes to ``[x, y, w, h]`` format."""

    x = box[..., 0] - box[..., 2] / 2.0
    y = box[..., 1] - box[..., 3] / 2.0
    wh = box[..., 2:4]
    return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), wh], dim=-1)


def pad_sequences(sequences: List[torch.Tensor], padding_value: float = 0.0) -> torch.Tensor:
    """Pad a list of variable length sequences along the temporal dimension."""

    if len(sequences) == 0:
        return torch.zeros(0)
    max_len = max(seq.size(0) for seq in sequences)
    feature_dim = sequences[0].size(-1)
    device = sequences[0].device
    padded = torch.full((len(sequences), max_len, feature_dim), padding_value, device=device)
    for i, seq in enumerate(sequences):
        padded[i, : seq.size(0)] = seq
    return padded


__all__ = [
    "bbox_center",
    "bbox_area",
    "xywh_to_cxcywh",
    "cxcywh_to_xywh",
    "pad_sequences",
]
