"""Gating utilities used during association."""
from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - only for typing
    from ..tracker.track import Track


def mahalanobis_gating(
    tracks: Sequence["Track"],
    detections: torch.Tensor,
    H: torch.Tensor,
    threshold: float = 9.4877,
    default_measurement_diag: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a Mahalanobis distance gating mask."""

    num_tracks = len(tracks)
    num_dets = detections.size(0)
    device = detections.device if num_dets > 0 else H.device
    mask = torch.ones(num_tracks, num_dets, dtype=torch.bool, device=device)
    if num_tracks == 0 or num_dets == 0:
        return mask

    H = H.to(device)
    I = torch.eye(H.size(0), device=device)

    for idx, track in enumerate(tracks):
        state = track.state.to(device)
        covariance = track.covariance.to(device)
        mean = H @ state
        diag = getattr(track, "measurement_noise", None)
        if diag is None:
            if default_measurement_diag is None:
                default_measurement_diag = torch.ones(H.size(0), device=device)
            diag = default_measurement_diag.to(device)
        else:
            diag = diag.to(device)
        R = torch.diag(diag)
        S = H @ covariance @ H.t() + R
        chol, info = torch.linalg.cholesky_ex(S + 1e-6 * I)
        if torch.any(info != 0):
            S = S + 1e-3 * I
            chol = torch.linalg.cholesky(S)
        diff = detections - mean.unsqueeze(0)
        sol = torch.cholesky_solve(diff.unsqueeze(-1), chol)
        d2 = torch.sum(diff.unsqueeze(-1) * sol, dim=(1, 2))
        mask[idx] = d2 <= threshold
    return mask


__all__ = ["mahalanobis_gating"]
