# fracturex/learn/models/fno_2d.py
"""FNO2d-global (scheme A in plan §3.9).

One-shot prediction of all T time steps as output channels:
    (C_in, H, W) → (T·C_out, H, W).

Reference: Li et al., "Fourier Neural Operator for Parametric PDEs",
ICLR 2021. Equation (3.11) of plan §3.4.
"""
from __future__ import annotations
from typing import Sequence


class FNO2d:
    """Stacked spectral conv blocks with truncation to n_modes Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 64,
                 n_modes: tuple[int, int] = (12, 12),
                 n_layers: int = 4,
                 activation: str = "gelu") -> None:
        raise NotImplementedError("M1 task: build P → L_1..L_L → Q with SpectralConv2d")

    def forward(self, x):
        raise NotImplementedError
