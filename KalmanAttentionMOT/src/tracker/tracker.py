"""Kalman attention based multi-object tracker."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from ..association.cross_attention_da import CrossAttentionAssociator
from ..association.matching import AssociationMatcher
from ..models.detection_encoder import DetectionEncoder
from ..models.kalmanet_gain import KalmanNetGain
from ..models.motion_model import ConstantVelocityModel, MEAS_DIM
from ..models.track_state_encoder import TrackStateEncoder
from ..utils.gating import mahalanobis_gating
from ..utils.logger import get_logger
from .track import Track


@dataclass
class TrackerConfig:
    """Runtime configuration for :class:`KalmanAttentionTracker`."""

    max_age: int = 30
    max_history: int = 5
    detection_threshold: float = 0.3
    gating_threshold: float = 9.4877
    device: str = "cpu"


class KalmanAttentionTracker(nn.Module):
    """High-level tracker coordinating prediction, association and updates."""

    def __init__(
        self,
        motion_model: ConstantVelocityModel,
        kalmanet: KalmanNetGain,
        track_encoder: TrackStateEncoder,
        detection_encoder: DetectionEncoder,
        associator: CrossAttentionAssociator,
        matcher: AssociationMatcher | None = None,
        config: TrackerConfig | None = None,
    ) -> None:
        super().__init__()
        self.motion_model = motion_model
        self.kalmanet = kalmanet
        self.track_encoder = track_encoder
        self.detection_encoder = detection_encoder
        self.associator = associator
        self.matcher = matcher or AssociationMatcher()
        self.config = config or TrackerConfig()
        self.device = torch.device(self.config.device)
        self.logger = get_logger(__name__)

        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

        self.register_buffer("F", self.motion_model.F())
        self.register_buffer("H", self.motion_model.H())
        self.register_buffer("base_process_noise_diag", self.motion_model.base_process_noise().diag())
        self.register_buffer("base_measurement_noise_diag", self.motion_model.base_measurement_noise().diag())

    @property
    def tracks(self) -> List[Track]:
        return list(self._tracks.values())

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def _create_track(self, measurement: torch.Tensor) -> Track:
        state = self.motion_model.init_state(measurement).to(self.device)
        covariance = self.motion_model.init_covariance().to(self.device)
        hidden = self.kalmanet.init_hidden(1, self.device)
        process_noise = self.base_process_noise_diag.to(self.device).clone()
        measurement_noise = self.base_measurement_noise_diag.to(self.device).clone()
        track = Track(
            track_id=self._next_id,
            state=state.squeeze(0) if state.dim() > 1 else state,
            covariance=covariance,
            kf_hidden=hidden.detach(),
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )
        track.append_history(track.feature_vector(), self.config.max_history)
        self._tracks[self._next_id] = track
        self._next_id += 1
        return track

    def _build_history_tensor(self, tracks: List[Track]) -> torch.Tensor:
        if len(tracks) == 0:
            return torch.zeros(0, self.config.max_history, 21, device=self.device)
        histories = [trk.history_tensor(self.config.max_history).to(self.device) for trk in tracks]
        return torch.stack(histories, dim=0)

    def _associate(self, tracks: List[Track], detections: torch.Tensor) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if len(tracks) == 0:
            return [], [], list(range(detections.size(0)))
        history = self._build_history_tensor(tracks)
        track_emb = self.track_encoder(history)
        det_emb = self.detection_encoder(detections)
        pair_scores, no_match = self.associator(track_emb, det_emb)
        gating_mask = None
        if detections.size(0) > 0:
            gating_mask = mahalanobis_gating(
                tracks,
                detections[:, :MEAS_DIM],
                self.H,
                threshold=self.config.gating_threshold,
                default_measurement_diag=self.base_measurement_noise_diag,
            )
        matches, unmatched_tracks, unmatched_dets = self.matcher.match(pair_scores, no_match, gating_mask)
        return matches, unmatched_tracks, unmatched_dets

    def step(self, detections: torch.Tensor) -> List[Track]:
        detections = detections.to(self.device)
        tracks = self.tracks
        for track in tracks:
            process_diag = track.process_noise if track.process_noise is not None else self.base_process_noise_diag
            track.predict(self.F.to(self.device), process_diag.to(self.device))
            track.age += 1
            track.misses += 1

        matches, unmatched_tracks, unmatched_dets = self._associate(tracks, detections)

        for t_idx, d_idx in matches:
            track = tracks[t_idx]
            det = detections[d_idx]
            measurement = det[:MEAS_DIM]
            innovation = track.innovation(measurement, self.H.to(self.device))
            gain, q_diag, r_diag, hidden = self.kalmanet(
                track.state.unsqueeze(0),
                measurement.unsqueeze(0),
                innovation.unsqueeze(0),
                track.kf_hidden,
            )
            kalman_gain = gain.squeeze(0).squeeze(0)
            q_diag = q_diag.squeeze(0).squeeze(0)
            r_diag = r_diag.squeeze(0).squeeze(0)
            track.kf_hidden = hidden.detach()
            track.apply_kalman_update(innovation, kalman_gain, r_diag, self.H)
            track.process_noise = q_diag.detach()
            track.measurement_noise = r_diag.detach()
            track.hits += 1
            track.misses = 0
            track.append_history(track.feature_vector(), self.config.max_history)

        for idx in unmatched_dets:
            det = detections[idx]
            if det.size(0) >= 5 and det[4] < self.config.detection_threshold:
                continue
            self._create_track(det[:MEAS_DIM])

        for idx in unmatched_tracks:
            track = tracks[idx]
            if track.misses > self.config.max_age:
                self._tracks.pop(track.track_id, None)
            else:
                track.append_history(track.feature_vector(), self.config.max_history)

        return self.tracks

    def get_active_tracks(self) -> List[Tuple[int, torch.Tensor]]:
        return [(track.track_id, track.as_measurement().detach().cpu()) for track in self.tracks]


__all__ = ["KalmanAttentionTracker", "TrackerConfig"]
