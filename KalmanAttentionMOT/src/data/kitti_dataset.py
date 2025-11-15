"""KITTI dataset loader for KalmanAttentionMOT."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from ..utils.geometry import xywh_to_cxcywh


@dataclass
class KITTIDatasetConfig:
    root: str
    split: str = "training"
    sequences: Optional[List[int]] = None


class KITTIMOTDataset(Dataset):
    """Dataset wrapper for KITTI Tracking benchmark."""

    def __init__(self, config: KITTIDatasetConfig) -> None:
        super().__init__()
        self.config = config
        self.root = Path(config.root)
        self.label_dir = self.root / "data_tracking_label_02" / config.split / "label_02"
        self.detection_dir = self.root / "data_tracking_det_02" / config.split / "det_02"
        self.sequence_ids = config.sequences or self._discover_sequences()
        self.samples = self._collect_samples()

    def _discover_sequences(self) -> List[int]:
        seqs = sorted(int(p.stem) for p in self.label_dir.glob("*.txt"))
        return seqs

    def _collect_samples(self) -> List[Tuple[int, int]]:
        samples: List[Tuple[int, int]] = []
        for seq_id in self.sequence_ids:
            label_file = self.label_dir / f"{seq_id:04d}.txt"
            if not label_file.exists():
                continue
            frames = set()
            with open(label_file, "r") as f:
                reader = csv.reader(f, delimiter=" ")
                for row in reader:
                    frame = int(row[0])
                    frames.add(frame)
            for frame in sorted(frames):
                samples.append((seq_id, frame))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_labels(self, seq_id: int, frame: int) -> List[Dict[str, torch.Tensor]]:
        label_file = self.label_dir / f"{seq_id:04d}.txt"
        labels = []
        with open(label_file, "r") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                if int(row[0]) != frame:
                    continue
                obj_type = row[2]
                track_id = int(row[1])
                bbox = torch.tensor([float(v) for v in row[6:10]])
                bbox = xywh_to_cxcywh(bbox.unsqueeze(0)).squeeze(0)
                labels.append({
                    "track_id": track_id,
                    "class": obj_type,
                    "bbox": bbox,
                })
        return labels

    def _load_detections(self, seq_id: int, frame: int) -> torch.Tensor:
        det_file = self.detection_dir / f"{seq_id:04d}.txt"
        detections: List[torch.Tensor] = []
        if det_file.exists():
            with open(det_file, "r") as f:
                reader = csv.reader(f, delimiter=" ")
                for row in reader:
                    if int(row[0]) != frame:
                        continue
                    bbox = torch.tensor([float(v) for v in row[6:10]])
                    bbox = xywh_to_cxcywh(bbox.unsqueeze(0)).squeeze(0)
                    score = float(row[5])
                    cls = 0.0
                    detections.append(torch.cat([bbox, torch.tensor([score, cls])]))
        if len(detections) == 0:
            return torch.zeros(0, 6)
        return torch.stack(detections, dim=0)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        seq_id, frame = self.samples[idx]
        labels = self._load_labels(seq_id, frame)
        detections = self._load_detections(seq_id, frame)
        return {
            "sequence": seq_id,
            "frame": frame,
            "detections": detections,
            "labels": labels,
        }


__all__ = ["KITTIMOTDataset", "KITTIDatasetConfig"]
