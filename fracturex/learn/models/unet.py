# fracturex/learn/models/unet.py
"""U-Net baseline (strong sharp-front competitor to FNO).

Plan §3.9 Baselines: U-Net must be in the M1 comparison table; on local
sharp fronts it often beats FNO. Scheme A: predict all T steps as channels,
``(B, C_in, H, W) → (B, T, H, W)``.

Size-robust: the decoder upsamples to each skip's spatial size, so the net
works on small / non-power-of-two grids (e.g. the 16×16 smoke grid).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_ACT = {"gelu": nn.GELU, "relu": nn.ReLU}


def _conv_block(cin: int, cout: int, act) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        act(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(min(8, cout), cout),
        act(),
    )


class UNet2d(nn.Module):
    """4-level U-Net for (C_in, H, W) → (C_out, H, W)."""

    def __init__(self, in_channels: int, out_channels: int,
                 base_channels: int = 32, depth: int = 4,
                 activation: str = "gelu") -> None:
        super().__init__()
        act = _ACT.get(activation, nn.GELU)
        self.depth = depth
        chs = [base_channels * (2 ** i) for i in range(depth)]

        self.downs = nn.ModuleList()
        prev = in_channels
        for c in chs:
            self.downs.append(_conv_block(prev, c, act))
            prev = c
        self.bottleneck = _conv_block(prev, prev * 2, act)

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        prev = prev * 2
        for c in reversed(chs):
            self.up_convs.append(nn.Conv2d(prev, c, 1))   # channel reduce after upsample
            self.ups.append(_conv_block(c + c, c, act))    # concat skip (c) + up (c)
            prev = c
        self.head = nn.Conv2d(prev, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for down in self.downs:
            h = down(h)
            skips.append(h)
            h = F.max_pool2d(h, 2, ceil_mode=True)
        h = self.bottleneck(h)
        for up_conv, up, skip in zip(self.up_convs, self.ups, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_conv(h)
            h = up(torch.cat([h, skip], dim=1))
        return self.head(h)
