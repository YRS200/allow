"""Association utilities built on top of Hungarian matching."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from ..utils.hungarian import hungarian


@dataclass
class MatchingConfig:
    """Configuration governing the matching behaviour."""

    invalid_cost: float = 1e4


class AssociationMatcher:
    """Turn attention scores into discrete associations."""

    def __init__(self, config: MatchingConfig | None = None) -> None:
        self.config = config or MatchingConfig()

    def build_cost_matrix(
        self,
        pair_scores: torch.Tensor,
        no_match_scores: torch.Tensor,
        gating_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert logits into a cost matrix for Hungarian matching."""

        if pair_scores.size(0) == 0:
            return pair_scores
        cost = -pair_scores
        if gating_mask is not None:
            invalid = (~gating_mask).to(cost.dtype) * self.config.invalid_cost
            cost = cost + invalid
        no_match_cost = (-no_match_scores).unsqueeze(-1)
        cost = torch.cat([cost, no_match_cost], dim=1)
        return cost

    def match(
        self,
        pair_scores: torch.Tensor,
        no_match_scores: torch.Tensor,
        gating_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Return matches, unmatched track indices and unmatched detections."""

        num_tracks, num_dets = pair_scores.shape
        if num_tracks == 0:
            return [], [], list(range(num_dets))

        cost = self.build_cost_matrix(pair_scores, no_match_scores, gating_mask)
        cost_np = cost.detach().cpu().numpy()
        rows, cols = hungarian(cost_np)

        matches: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(num_tracks))
        unmatched_dets = set(range(num_dets))

        for r, c in zip(rows, cols):
            if r >= num_tracks:
                continue
            if c == num_dets or c >= cost_np.shape[1]:
                unmatched_tracks.discard(r)
                continue
            if gating_mask is not None and not gating_mask[r, c]:
                continue
            matches.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)

        return matches, sorted(unmatched_tracks), sorted(unmatched_dets)


__all__ = ["AssociationMatcher", "MatchingConfig"]
