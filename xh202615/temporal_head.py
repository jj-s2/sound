"""Small trainable temporal heads over frozen speaker-evidence features."""

from __future__ import annotations

import torch
from torch import nn


class TemporalSpeakerHead(nn.Module):
    """Predict target presence and overlap from a sequence of frozen features.

    The upstream speaker encoder is intentionally outside this module.  This
    keeps the trainable experiment small and makes the GRU/MLP ablation
    comparable on exactly the same cached features.
    """

    def __init__(self, *, input_dim: int, hidden_dim: int = 128, mode: str = "gru") -> None:
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if mode not in {"gru", "mlp", "fused"}:
            raise ValueError("mode must be 'gru', 'mlp', or 'fused'")
        self.mode = mode
        self.hidden_dim = hidden_dim
        if mode == "gru":
            self.temporal = nn.GRU(input_dim, hidden_dim, batch_first=True)
        elif mode == "mlp":
            self.temporal = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        else:
            self.temporal = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.summary = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.fuse = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        self.presence = nn.Linear(hidden_dim, 1)
        self.overlap = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape (batch, time, input_dim)")
        if self.mode == "gru":
            _, hidden = self.temporal(features)
            summary = hidden[-1]
        elif self.mode == "mlp":
            summary = self.temporal(features).mean(dim=1)
        else:
            _, hidden = self.temporal(features)
            global_summary = self.summary(
                torch.cat((features.mean(dim=1), features.amax(dim=1)), dim=1)
            )
            summary = self.fuse(torch.cat((hidden[-1], global_summary), dim=1))
        return self.presence(summary), self.overlap(summary)
