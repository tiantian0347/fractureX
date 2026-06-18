# fracturex/learn/models/multioutput_fno.py
"""Multi-field-output FNO for the T3 main task (plan §M2 Stage B).

Predicts ``n_fields`` channels per time step — for Stage B the 4 fields are
``(d, σ_xx, σ_yy, σ_xy)``. Hu-Zhang stress supervision is what differentiates
this surrogate from existing phase-field DeepONet/FNO papers.

Contract: ``forward(x): (B, C_in, H, W) → (B, T, n_fields, H, W)`` (scheme A,
all T steps as a reshaped channel block). The optional 5th field (𝓗) for the
plan §3.5 (b) variant is selected via ``n_fields=5``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .fno_2d import FNO2d


class MultiOutputFNO2d(nn.Module):
    """FNO backbone emitting ``T·n_fields`` channels, reshaped to (B, T, n_fields, H, W)."""

    def __init__(self, in_channels: int, *,
                 T: int,
                 n_fields: int = 4,
                 hidden_channels: int = 64,
                 n_modes: tuple[int, int] = (16, 16),
                 n_layers: int = 4,
                 activation: str = "gelu",
                 include_history: bool = False) -> None:
        super().__init__()
        if include_history and n_fields < 5:
            n_fields = 5
        self.T = T
        self.n_fields = n_fields
        self.backbone = FNO2d(
            in_channels, T * n_fields,
            hidden_channels=hidden_channels, n_modes=n_modes,
            n_layers=n_layers, activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        y = self.backbone(x)                       # (B, T*n_fields, H, W)
        return y.view(B, self.T, self.n_fields, H, W)
