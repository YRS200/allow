"""Synthetic dataset for training and testing KalmanAttentionMOT."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from ..models.motion_model import STATE_DIM


@dataclass
class SyntheticSequenceConfig:
    num_objects: int = 10
    num_frames: int = 150
    frame_size: Tuple[int, int] = (1280, 720)
    measurement_noise: float = 2.0
    velocity_noise: float = 0.2
    size_noise: float = 1.0


def _random_state(device: torch.device, frame_size: Tuple[int, int]) -> torch.Tensor:
    w, h = frame_size
    state = torch.zeros(STATE_DIM, device=device)
    state[0] = random.uniform(0.1 * w, 0.9 * w)
    state[1] = random.uniform(0.1 * h, 0.9 * h)
    state[2] = random.uniform(-10.0, 10.0)
    state[3] = random.uniform(-10.0, 10.0)
    state[4] = random.uniform(40.0, 120.0)
    state[5] = random.uniform(80.0, 200.0)
    state[6] = random.uniform(-5.0, 5.0)
    state[7] = random.uniform(-5.0, 5.0)
    return state


class SyntheticMOTDataset(Dataset):
    """Simple synthetic MOT dataset with constant velocity dynamics."""

    def __init__(self, config: SyntheticSequenceConfig | None = None, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.config = config or SyntheticSequenceConfig()
        self.device = torch.device(device)
        self.sequences = self._generate_sequences()

    def _generate_sequences(self) -> List[Dict[str, torch.Tensor]]:
        cfg = self.config
        sequences: List[Dict[str, torch.Tensor]] = []
        for _ in range(cfg.num_objects):
            state = _random_state(self.device, cfg.frame_size)
            states = []
            measurements = []
            for _ in range(cfg.num_frames):
                states.append(state.clone())
                meas = state[[0, 1, 4, 5]] + torch.randn(4, device=self.device) * cfg.measurement_noise
                measurements.append(meas)
                state = state.clone()
                state[0] += state[2]
                state[1] += state[3]
                state[4] = max(5.0, state[4] + state[6])
                state[5] = max(5.0, state[5] + state[7])
                state[2] += torch.randn(1, device=self.device) * cfg.velocity_noise
                state[3] += torch.randn(1, device=self.device) * cfg.velocity_noise
                state[6] += torch.randn(1, device=self.device) * cfg.size_noise * 0.1
                state[7] += torch.randn(1, device=self.device) * cfg.size_noise * 0.1
            sequences.append(
                {
                    "states": torch.stack(states, dim=0),
                    "measurements": torch.stack(measurements, dim=0),
                }
            )
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        return {
            "states": seq["states"].clone(),
            "measurements": seq["measurements"].clone(),
        }


__all__ = ["SyntheticMOTDataset", "SyntheticSequenceConfig"]
