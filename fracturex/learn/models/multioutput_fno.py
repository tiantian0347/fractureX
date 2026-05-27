# fracturex/learn/models/multioutput_fno.py
"""4-channel-output FNO for the T3 main task: (d, σ_xx, σ_yy, σ_xy).

Plan §M2 Stage B: this is the paper's main surrogate — Hu-Zhang stress
supervision is what differentiates this work from existing phase-field
DeepONet/FNO papers.
"""
from __future__ import annotations
from typing import Optional


class MultiOutputFNO2d:
    """FNO2d with 4 output channels per time step.

    Optional 5-channel mode (include 𝓗) for plan §3.5 (b) variant.
    """

    def __init__(self, in_channels: int,
                 *,
                 T: int,
                 hidden_channels: int = 64,
                 n_modes: tuple[int, int] = (16, 16),
                 n_layers: int = 4,
                 include_history: bool = False,
                 stress_scale: Optional[float] = None) -> None:
        raise NotImplementedError("M2 Stage B: 4-or-5 output channels per t, broadcast over T")

    def forward(self, x):
        raise NotImplementedError
