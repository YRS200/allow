"""Data module wrapping dataset creation and loaders."""
from __future__ import annotations

from dataclasses import dataclass, field

from torch.utils.data import DataLoader, Dataset

from .kitti_dataset import KITTIDatasetConfig, KITTIMOTDataset
from .synthetic_dataset import SyntheticMOTDataset, SyntheticSequenceConfig


@dataclass
class DataModuleConfig:
    """Configuration for :class:`MOTDataModule`."""

    dataset: str = "synthetic"
    batch_size: int = 1
    num_workers: int = 0
    shuffle: bool = True
    device: str = "cpu"
    synthetic: SyntheticSequenceConfig = field(default_factory=SyntheticSequenceConfig)
    kitti_root: str = ""
    kitti_split: str = "training"


class MOTDataModule:
    """Lightweight data module handling train/val loaders."""

    def __init__(self, config: DataModuleConfig) -> None:
        self.config = config
        self._train_dataset: Dataset | None = None
        self._val_dataset: Dataset | None = None

    def setup(self) -> None:
        if self.config.dataset == "synthetic":
            self._train_dataset = SyntheticMOTDataset(self.config.synthetic, device=self.config.device)
            self._val_dataset = self._train_dataset
        elif self.config.dataset == "kitti":
            if not self.config.kitti_root:
                raise ValueError("kitti_root must be provided for KITTI dataset")
            cfg = KITTIDatasetConfig(root=self.config.kitti_root, split=self.config.kitti_split)
            dataset = KITTIMOTDataset(cfg)
            self._train_dataset = dataset
            self._val_dataset = dataset
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            self.setup()
        assert self._train_dataset is not None
        return DataLoader(
            self._train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            self.setup()
        assert self._val_dataset is not None
        return DataLoader(
            self._val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )


__all__ = ["MOTDataModule", "DataModuleConfig"]
