"""Utilities for exporting tracker results in KITTI format."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from ..utils.geometry import cxcywh_to_xywh


def write_kitti_results(output_dir: str | Path, sequence_id: int, frame: int, tracks: Iterable[Dict[str, torch.Tensor]]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{sequence_id:04d}.txt"
    rows: List[List[str]] = []
    for track in tracks:
        bbox = cxcywh_to_xywh(track["bbox"].unsqueeze(0)).squeeze(0)
        row = [
            str(frame),
            str(track["track_id"]),
            track.get("class", "Car"),
            "-1", "-1", "-1", "-10",
            f"{bbox[0]:.2f}",
            f"{bbox[1]:.2f}",
            f"{bbox[2]:.2f}",
            f"{bbox[3]:.2f}",
            "-1", "-1", "-1", "-1000", "-1000", "-1000", "-10",
        ]
        rows.append(row)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(rows)


__all__ = ["write_kitti_results"]
