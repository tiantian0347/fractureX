# fracturex/learn/models/unet.py
"""U-Net baseline (strong sharp-front competitor to FNO).

Plan §3.9 Baselines: U-Net must be in the M1 comparison table; on local
sharp fronts it often beats FNO. Default config: 4 down + 4 up + skip.
"""
from __future__ import annotations
from typing import Optional


class UNet2d:
    """4-level U-Net for (C_in, H, W) → (C_out, H, W).

    Time handling defaults to scheme A of plan §3.9 (predict all T steps
    as channels). Use multioutput_fno when stress channels are needed.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 base_channels: int = 32, depth: int = 4,
                 activation: str = "gelu") -> None:
        raise NotImplementedError("M1 task: build encoder/decoder + skip connections")

    def forward(self, x):
        raise NotImplementedError
