"""Generic trainer utilities (optional)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..utils.logger import get_logger


@dataclass
class TrainingConfig:
    epochs: int = 5
    lr: float = 1e-3
    device: str = "cpu"
    log_interval: int = 10


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
        config: TrainingConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.logger = get_logger(__name__)
        self.device = torch.device(config.device)
        self.model.to(self.device)

    def fit(self, dataloader: DataLoader) -> None:
        for epoch in range(self.config.epochs):
            self.model.train()
            for step, batch in enumerate(dataloader):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = self.model(batch)
                loss = self.loss_fn(outputs)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                if step % self.config.log_interval == 0:
                    self.logger.info("Epoch %d Step %d Loss %.4f", epoch, step, float(loss))


__all__ = ["Trainer", "TrainingConfig"]
