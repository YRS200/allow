"""Tracking metrics implementation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from ..utils.iou import bbox_iou


@dataclass
class MOTAccumulator:
    matches: int = 0
    misses: int = 0
    false_positives: int = 0
    id_switches: int = 0
    total_distance: float = 0.0
    total_overlaps: float = 0.0
    total_associations: int = 0


def mota(acc: MOTAccumulator) -> float:
    denom = acc.matches + acc.misses + acc.false_positives
    if denom == 0:
        return 0.0
    return 1.0 - (acc.misses + acc.false_positives + acc.id_switches) / denom


def motp(acc: MOTAccumulator) -> float:
    if acc.matches == 0:
        return 0.0
    return acc.total_distance / acc.matches


def idf1(idtp: int, idfp: int, idfn: int) -> float:
    if idtp + 0.5 * (idfp + idfn) == 0:
        return 0.0
    return idtp / (idtp + 0.5 * (idfp + idfn))


def hota(acc: MOTAccumulator) -> float:
    if acc.total_associations == 0:
        return 0.0
    return acc.total_overlaps / acc.total_associations


def evaluate_frame(gt_boxes: torch.Tensor, pred_boxes: torch.Tensor, gt_ids: torch.Tensor, pred_ids: torch.Tensor) -> MOTAccumulator:
    acc = MOTAccumulator()
    if gt_boxes.numel() == 0 and pred_boxes.numel() == 0:
        return acc
    if gt_boxes.numel() == 0:
        acc.false_positives = int(pred_boxes.size(0))
        return acc
    if pred_boxes.numel() == 0:
        acc.misses = int(gt_boxes.size(0))
        return acc

    ious = bbox_iou(gt_boxes, pred_boxes)
    gt_matches = torch.argmax(ious, dim=1)
    pred_taken = set()
    for i, pred_idx in enumerate(gt_matches.tolist()):
        if pred_idx in pred_taken:
            acc.misses += 1
            continue
        pred_taken.add(pred_idx)
        iou = float(ious[i, pred_idx])
        if iou < 0.1:
            acc.misses += 1
            continue
        acc.matches += 1
        acc.total_overlaps += iou
        acc.total_associations += 1
        acc.total_distance += 1.0 - iou
        if int(gt_ids[i]) != int(pred_ids[pred_idx]):
            acc.id_switches += 1
    acc.false_positives = int(pred_boxes.size(0) - len(pred_taken))
    acc.misses += int(gt_boxes.size(0) - acc.matches)
    return acc


__all__ = ["MOTAccumulator", "mota", "motp", "idf1", "hota", "evaluate_frame"]
