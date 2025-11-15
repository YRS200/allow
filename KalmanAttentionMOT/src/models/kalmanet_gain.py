"""KalmanNet gain network implementation.

The network learns to adapt Kalman filter parameters (Kalman gain, process and
measurement noise) conditioned on the current prediction, measurements and
innovation. The implementation follows the high-level idea presented in
"KalmanNet: Neural network aided Kalman filtering" but is adapted for multi
object tracking and compact enough for integration with the rest of the repo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .motion_model import MEAS_DIM, STATE_DIM


@dataclass
class KalmanNetConfig:
    """Configuration options for :class:`KalmanNetGain`."""

    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0


class KalmanNetGain(nn.Module):
    """GRU based gain adapter for Kalman filtering.

    The module processes per-track features sequentially. During filtering the
    tracker maintains a hidden state for each track which is fed back into the
    GRU on every step, allowing the network to adapt its parameters based on the
    motion history of the target.

    Inputs are concatenated as [x_pred, z, innovation] and projected before
    entering the recurrent stack. The outputs parametrise diagonal process and
    measurement noise matrices as well as the full Kalman gain.
    """

    def __init__(self, config: KalmanNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or KalmanNetConfig()
        input_dim = STATE_DIM + MEAS_DIM + MEAS_DIM

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.gru = nn.GRU(
            input_size=self.config.hidden_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )

        gain_dim = STATE_DIM * MEAS_DIM
        self.gain_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, gain_dim),
        )

        self.q_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, STATE_DIM),
        )
        self.r_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, MEAS_DIM),
        )

    def forward(
        self,
        x_pred: torch.Tensor,
        z: torch.Tensor,
        innovation: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute adaptive gain parameters.

        Args:
            x_pred: Predicted state tensor with shape (T, B, 8) or (1, B, 8).
            z: Measurement tensor with shape (T, B, 4).
            innovation: Innovation tensor with shape (T, B, 4).
            hidden: Optional recurrent hidden state with shape
                (num_layers, B, hidden_dim).

        Returns:
            gain: Kalman gain matrix tensor with shape (T, B, 8, 4).
            q_diag: Process noise diagonal tensor with shape (T, B, 8).
            r_diag: Measurement noise diagonal tensor with shape (T, B, 4).
            hidden: Updated hidden state with shape (num_layers, B, hidden_dim).
        """

        if x_pred.dim() == 2:
            x_pred = x_pred.unsqueeze(0)
            z = z.unsqueeze(0)
            innovation = innovation.unsqueeze(0)

        inp = torch.cat([x_pred, z, innovation], dim=-1)
        seq = self.input_proj(inp)
        output, hidden = self.gru(seq, hidden)

        gain = self.gain_head(output)
        gain = gain.view(*gain.shape[:-1], STATE_DIM, MEAS_DIM)

        q_diag = torch.relu(self.q_head(output)) + 1e-3
        r_diag = torch.relu(self.r_head(output)) + 1e-3

        return gain, q_diag, r_diag, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return an initial hidden state filled with zeros."""

        return torch.zeros(
            self.config.num_layers,
            batch_size,
            self.config.hidden_dim,
            device=device,
        )


__all__ = ["KalmanNetGain", "KalmanNetConfig"]
