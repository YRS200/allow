"""CLI entry point for KalmanAttentionMOT."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from .src.data.synthetic_dataset import SyntheticMOTDataset
from .src.evaluation.kitti_eval import write_kitti_results
from .src.training.train import build_datamodule, build_tracker, train as run_training
from .src.utils.configs import ExperimentConfig


def run_demo(steps: int = 20) -> None:
    """Run a small synthetic demo showing tracker outputs."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = build_tracker({}, device)
    dataset = SyntheticMOTDataset(device=device)
    sample = dataset[0]

    tracker.eval()
    with torch.no_grad():
        for t in range(min(steps, sample["measurements"].size(0))):
            measurement = sample["measurements"][t]
            detection = torch.cat([measurement, measurement.new_tensor([1.0, 0.0])]).unsqueeze(0)
            tracks = tracker.step(detection)
            print(f"Step {t:02d}: {len(tracks)} active tracks")


def run_evaluation(config_path: str, output_dir: str) -> None:
    """Run the tracker on the validation split and write KITTI results."""

    cfg = ExperimentConfig.from_file(config_path)
    device = torch.device(cfg.training.get("device", "cpu"))
    tracker = build_tracker(cfg.tracker, device)
    datamodule = build_datamodule(cfg.data, device)
    dataloader = datamodule.val_dataloader()
    tracker.eval()
    output_dir = Path(output_dir)
    current_sequence: int | None = None

    def _to_int(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.squeeze().item())
        if isinstance(value, (list, tuple)):
            return int(value[0])
        return int(value)

    with torch.no_grad():
        for batch in dataloader:
            sequence = _to_int(batch["sequence"])
            frame = _to_int(batch["frame"])
            detections = batch["detections"]
            if isinstance(detections, torch.Tensor) and detections.dim() == 3:
                detections = detections[0]
            detections = detections.to(device)
            if current_sequence != sequence:
                tracker.reset()
                current_sequence = sequence
            tracks = tracker.step(detections)
            export = [
                {"track_id": track.track_id, "bbox": track.as_measurement().cpu(), "class": "Car"}
                for track in tracks
            ]
            write_kitti_results(output_dir, sequence, frame, export)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KalmanAttentionMOT CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("--config", type=str, required=True, help="Path to YAML config")

    demo_parser = subparsers.add_parser("demo", help="Run synthetic demo")
    demo_parser.add_argument("--steps", type=int, default=20)

    eval_parser = subparsers.add_parser("eval", help="Export KITTI-style tracking results")
    eval_parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    eval_parser.add_argument("--output", type=str, required=True, help="Output directory for KITTI files")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_training(args.config)
    elif args.command == "demo":
        run_demo(args.steps)
    elif args.command == "eval":
        run_evaluation(args.config, args.output)
    else:
        raise ValueError(f"Unsupported command {args.command}")


if __name__ == "__main__":
    main()
