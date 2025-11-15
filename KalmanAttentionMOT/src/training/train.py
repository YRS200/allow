"""Training entry point for KalmanAttentionMOT."""
from __future__ import annotations

import argparse

import torch

from ..association.cross_attention_da import CrossAttentionAssociator
from ..association.matching import AssociationMatcher
from ..data.datamodule import DataModuleConfig, MOTDataModule
from ..models.detection_encoder import DetectionEncoder
from ..models.kalmanet_gain import KalmanNetGain
from ..models.motion_model import ConstantVelocityModel, MotionModelConfig
from ..models.track_state_encoder import TrackStateEncoder
from ..tracker.tracker import KalmanAttentionTracker, TrackerConfig
from ..training.losses import state_loss
from ..utils.configs import ExperimentConfig
from ..utils.logger import get_logger


def build_tracker(config: dict, device: torch.device) -> KalmanAttentionTracker:
    motion_cfg = MotionModelConfig()
    motion_model = ConstantVelocityModel(motion_cfg)
    embedding_dim = config.get("embedding_dim", 128)
    num_heads = config.get("num_heads", 4)
    kalmanet = KalmanNetGain()
    track_encoder = TrackStateEncoder(hidden_dim=embedding_dim)
    detection_encoder = DetectionEncoder(hidden_dim=embedding_dim)
    associator = CrossAttentionAssociator(dim=embedding_dim, num_heads=num_heads)
    matcher = AssociationMatcher()
    tracker_cfg = TrackerConfig(
        max_age=config.get("max_age", 30),
        max_history=config.get("history", 5),
        detection_threshold=config.get("detection_threshold", 0.3),
        gating_threshold=config.get("gating_threshold", 9.4877),
        device=str(device),
    )
    tracker = KalmanAttentionTracker(
        motion_model=motion_model,
        kalmanet=kalmanet,
        track_encoder=track_encoder,
        detection_encoder=detection_encoder,
        associator=associator,
        matcher=matcher,
        config=tracker_cfg,
    )
    return tracker.to(device)


def build_datamodule(config: dict, device: torch.device) -> MOTDataModule:
    data_cfg = DataModuleConfig(
        dataset=config.get("dataset", "synthetic"),
        batch_size=config.get("batch_size", 1),
        num_workers=config.get("num_workers", 0),
        shuffle=config.get("shuffle", True),
        device=str(device),
    )
    if data_cfg.dataset == "kitti":
        data_cfg.kitti_root = config.get("kitti_root", "")
        data_cfg.kitti_split = config.get("kitti_split", "training")
    return MOTDataModule(data_cfg)


def train(config_path: str) -> None:
    cfg = ExperimentConfig.from_file(config_path)
    device = torch.device(cfg.training.get("device", "cpu"))
    logger = get_logger(__name__)

    tracker = build_tracker(cfg.tracker, device)
    datamodule = build_datamodule(cfg.data, device)
    dataloader = datamodule.train_dataloader()

    optimizer = torch.optim.Adam(tracker.parameters(), lr=cfg.training.get("lr", 1e-3))
    epochs = cfg.training.get("epochs", 1)

    for epoch in range(epochs):
        tracker.train()
        epoch_loss = 0.0
        sequences = 0
        for batch in dataloader:
            measurements = batch["measurements"].to(device)
            states = batch["states"].to(device)
            tracker.reset()
            optimizer.zero_grad()
            step_losses = []
            for t in range(measurements.size(0)):
                meas = measurements[t]
                det = torch.cat([meas, meas.new_tensor([1.0, 0.0])]).unsqueeze(0)
                tracks = tracker.step(det)
                if len(tracks) == 0:
                    continue
                pred_states = torch.stack([trk.state for trk in tracks])
                target = states[t].unsqueeze(0).expand_as(pred_states)
                step_losses.append(state_loss(pred_states, target))
            if not step_losses:
                continue
            loss = torch.stack(step_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tracker.parameters(), 10.0)
            optimizer.step()
            epoch_loss += float(loss)
            sequences += 1
        if sequences > 0:
            logger.info("Epoch %d loss %.4f", epoch + 1, epoch_loss / sequences)
        else:
            logger.warning("Epoch %d produced no sequences with associations", epoch + 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KalmanAttentionMOT")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
