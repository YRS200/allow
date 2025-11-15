"""Motion model utilities for KalmanAttentionMOT.

This module implements a constant velocity motion model with an 8 dimensional
state vector:
    [cx, cy, vx, vy, w, h, dw, dh].

The corresponding observation space is four dimensional:
    [cx, cy, w, h].

The helpers exposed here are intentionally lightweight so they can be reused
across classic Kalman filtering code and the learnable KalmanNet gain module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


STATE_DIM = 8
MEAS_DIM = 4


@dataclass
class MotionModelConfig:
    """Configuration describing the kinematics of the motion model.

    Attributes:
        dt: Time delta between consecutive frames in seconds.
        process_std_pos: Base process noise standard deviation for position.
        process_std_vel: Base process noise standard deviation for velocity.
        process_std_size: Base process noise standard deviation for width/height.
        process_std_growth: Base process noise for dw/dh.
    """

    dt: float = 1.0
    process_std_pos: float = 1.5
    process_std_vel: float = 3.0
    process_std_size: float = 0.5
    process_std_growth: float = 0.2


class ConstantVelocityModel:
    """Constant velocity motion model.

    The model is intentionally stateless; the methods simply construct matrices
    that can be used inside the KalmanNet pipeline.
    """

    def __init__(self, config: MotionModelConfig | None = None) -> None:
        self.config = config or MotionModelConfig()

    def F(self) -> torch.Tensor:
        """Return the state transition matrix of shape (8, 8).

        The matrix performs a simple constant velocity update on position and
        size related components. The update is differentiable and static.
        """

        dt = self.config.dt
        F = torch.eye(STATE_DIM)
        F[0, 2] = dt
        F[1, 3] = dt
        F[4, 6] = dt
        F[5, 7] = dt
        return F

    def H(self) -> torch.Tensor:
        """Return the observation matrix of shape (4, 8)."""

        H = torch.zeros(MEAS_DIM, STATE_DIM)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 4] = 1.0
        H[3, 5] = 1.0
        return H

    def base_process_noise(self) -> torch.Tensor:
        """Return a diagonal process noise covariance matrix (8, 8).

        The diagonal entries encode uncertainty on position, velocity, size and
        size growth parameters. These serve as a starting point for the
        learnable KalmanNet module which refines them on-line.
        """

        cfg = self.config
        q = torch.tensor(
            [
                cfg.process_std_pos,
                cfg.process_std_pos,
                cfg.process_std_vel,
                cfg.process_std_vel,
                cfg.process_std_size,
                cfg.process_std_size,
                cfg.process_std_growth,
                cfg.process_std_growth,
            ],
            dtype=torch.float32,
        )
        return torch.diag(q**2)

    def base_measurement_noise(self) -> torch.Tensor:
        """Return a diagonal measurement noise covariance matrix (4, 4)."""

        r = torch.tensor([1.5, 1.5, 2.0, 2.0], dtype=torch.float32)
        return torch.diag(r**2)

    def init_state(self, measurement: torch.Tensor) -> torch.Tensor:
        """Initialise a state vector from a measurement.

        Args:
            measurement: Tensor with shape (..., 4) representing [cx, cy, w, h].

        Returns:
            Tensor with shape (..., 8) with zero velocities and growth rates.
        """

        zeros = torch.zeros(*measurement.shape[:-1], STATE_DIM, device=measurement.device)
        zeros[..., 0] = measurement[..., 0]
        zeros[..., 1] = measurement[..., 1]
        zeros[..., 4] = measurement[..., 2]
        zeros[..., 5] = measurement[..., 3]
        return zeros

    def init_covariance(self) -> torch.Tensor:
        """Return an initial covariance matrix (8, 8)."""

        # Start with fairly large uncertainty to let KalmanNet adapt quickly.
        std = torch.tensor([50.0, 50.0, 10.0, 10.0, 20.0, 20.0, 5.0, 5.0])
        return torch.diag(std**2)


def get_motion_model(config: MotionModelConfig | None = None) -> ConstantVelocityModel:
    """Factory helper returning a configured motion model."""

    return ConstantVelocityModel(config=config)


__all__ = [
    "STATE_DIM",
    "MEAS_DIM",
    "MotionModelConfig",
    "ConstantVelocityModel",
    "get_motion_model",
]
