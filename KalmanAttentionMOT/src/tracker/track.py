"""Track representation used by the KalmanAttention tracker."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch

from ..models.motion_model import STATE_DIM


@dataclass
class Track:
    """Container storing per-object Kalman filter state."""

    track_id: int
    state: torch.Tensor
    covariance: torch.Tensor
    kf_hidden: torch.Tensor
    process_noise: torch.Tensor | None = None
    measurement_noise: torch.Tensor | None = None
    age: int = 1
    hits: int = 1
    misses: int = 0
    history_features: List[torch.Tensor] = field(default_factory=list)

    def innovation(self, measurement: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Return the innovation vector for a measurement."""

        return measurement - H @ self.state

    def predict(self, F: torch.Tensor, process_noise_diag: torch.Tensor) -> None:
        """Propagate the state through the motion model."""

        device = self.state.device
        Q = torch.diag(process_noise_diag.to(device))
        self.state = F.to(device) @ self.state
        self.covariance = F.to(device) @ self.covariance @ F.to(device).t() + Q
        self.process_noise = process_noise_diag.detach().clone()

    def apply_kalman_update(
        self,
        innovation: torch.Tensor,
        kalman_gain: torch.Tensor,
        measurement_noise_diag: torch.Tensor,
        H: torch.Tensor,
    ) -> None:
        """Apply the Joseph form covariance update."""

        device = self.state.device
        self.state = self.state + kalman_gain @ innovation
        R = torch.diag(measurement_noise_diag.to(device))
        KH = kalman_gain @ H.to(device)
        I = torch.eye(STATE_DIM, device=device)
        updated = (I - KH) @ self.covariance @ (I - KH).t() + kalman_gain @ R @ kalman_gain.t()
        self.covariance = 0.5 * (updated + updated.t())
        self.measurement_noise = measurement_noise_diag.detach().clone()

    def feature_vector(self) -> torch.Tensor:
        """Return the 21-dimensional feature vector for history encoding."""

        diag = torch.diagonal(self.covariance, 0)
        speed = torch.norm(self.state[2:4])
        aspect = torch.clamp(self.state[4] / (self.state[5] + 1e-6), min=0.0, max=10.0)
        scale = self.state[4] * self.state[5]
        meta = torch.tensor(
            [float(self.age), float(self.misses), float(speed), float(aspect), float(scale)],
            device=self.state.device,
        )
        return torch.cat([self.state, diag, meta])

    def as_measurement(self) -> torch.Tensor:
        """Return the observable subset of the state."""

        return self.state[[0, 1, 4, 5]]

    def append_history(self, feature: torch.Tensor, max_history: int) -> None:
        feature = feature.detach().to(self.state.device)
        self.history_features.append(feature)
        if len(self.history_features) > max_history:
            self.history_features = self.history_features[-max_history:]

    def history_tensor(self, max_history: int, feature_dim: int = 21) -> torch.Tensor:
        history = self.state.new_zeros(max_history, feature_dim)
        if not self.history_features:
            return history
        features = torch.stack(self.history_features[-max_history:], dim=0)
        history[-features.size(0) :] = features
        return history

    def clone(self) -> "Track":
        return Track(
            track_id=self.track_id,
            state=self.state.clone(),
            covariance=self.covariance.clone(),
            kf_hidden=self.kf_hidden.clone(),
            process_noise=self.process_noise.clone() if self.process_noise is not None else None,
            measurement_noise=self.measurement_noise.clone() if self.measurement_noise is not None else None,
            age=self.age,
            hits=self.hits,
            misses=self.misses,
            history_features=[h.clone() for h in self.history_features],
        )


__all__ = ["Track"]
