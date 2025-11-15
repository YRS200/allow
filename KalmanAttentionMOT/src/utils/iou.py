"""IoU computation utilities."""
from __future__ import annotations

import torch


def bbox_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between bounding boxes in [cx, cy, w, h] format."""

    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))

    b1 = boxes1.unsqueeze(1)
    b2 = boxes2.unsqueeze(0)
    x1_min = b1[..., 0] - b1[..., 2] / 2.0
    y1_min = b1[..., 1] - b1[..., 3] / 2.0
    x1_max = b1[..., 0] + b1[..., 2] / 2.0
    y1_max = b1[..., 1] + b1[..., 3] / 2.0

    x2_min = b2[..., 0] - b2[..., 2] / 2.0
    y2_min = b2[..., 1] - b2[..., 3] / 2.0
    x2_max = b2[..., 0] + b2[..., 2] / 2.0
    y2_max = b2[..., 1] + b2[..., 3] / 2.0

    inter_x1 = torch.max(x1_min, x2_min)
    inter_y1 = torch.max(y1_min, y2_min)
    inter_x2 = torch.min(x1_max, x2_max)
    inter_y2 = torch.min(y1_max, y2_max)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter_area + 1e-6

    return inter_area / union


__all__ = ["bbox_iou"]
