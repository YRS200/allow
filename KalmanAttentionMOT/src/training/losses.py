"""Loss functions for KalmanAttentionMOT."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


def state_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between predicted and ground truth state."""

    return torch.mean((pred - target) ** 2)


def association_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross entropy for association decisions."""

    return nn.functional.cross_entropy(logits, target)


def total_loss(
    pred_state: torch.Tensor,
    target_state: torch.Tensor,
    assoc_logits: torch.Tensor,
    assoc_target: torch.Tensor,
    lambda_state: float = 1.0,
    lambda_assoc: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    l_state = state_loss(pred_state, target_state)
    l_assoc = association_loss(assoc_logits, assoc_target)
    loss = lambda_state * l_state + lambda_assoc * l_assoc
    return loss, {"state": l_state.detach(), "assoc": l_assoc.detach()}


__all__ = ["state_loss", "association_loss", "total_loss"]
