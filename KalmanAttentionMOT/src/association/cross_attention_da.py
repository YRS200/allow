"""Cross-attention module for track-to-detection association."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class CrossAttentionAssociator(nn.Module):
    """Transformer-style cross-attention that yields association logits."""

    def __init__(self, dim: int = 128, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embedding dimension must be divisible by number of heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.no_match_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 1),
        )

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), self.num_heads, self.head_dim)

    def forward(self, track_emb: torch.Tensor, det_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute association logits for tracks and detections."""

        if track_emb.size(0) == 0:
            return track_emb.new_zeros(0, det_emb.size(0)), track_emb.new_zeros(0)
        if det_emb.size(0) == 0:
            return track_emb.new_zeros(track_emb.size(0), 0), self.no_match_head(track_emb).squeeze(-1)

        queries = self._reshape_heads(self.q_proj(track_emb))
        keys = self._reshape_heads(self.k_proj(det_emb))
        values = self._reshape_heads(self.v_proj(det_emb))

        attn_logits = torch.einsum("qhd,khd->qkh", queries, keys) * self.scale
        attn_weights = torch.softmax(attn_logits, dim=1)
        attn_output = torch.einsum("qkh,khd->qhd", attn_weights, values).reshape(track_emb.size(0), self.dim)
        attn_output = self.out_proj(attn_output)
        pair_scores = torch.einsum("qd,kd->qk", attn_output, det_emb)

        no_match = self.no_match_head(track_emb).squeeze(-1)
        return pair_scores, no_match


__all__ = ["CrossAttentionAssociator"]
