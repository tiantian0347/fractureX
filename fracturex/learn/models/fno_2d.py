# fracturex/learn/models/fno_2d.py
"""FNO2d-global (scheme A in plan §3.9).

One-shot prediction of all T time steps as output channels:
    (C_in, H, W) → (T·C_out, H, W), here C_out = 1 (damage), so → (T, H, W).

Reference: Li et al., "Fourier Neural Operator for Parametric PDEs",
ICLR 2021. Equation (3.11) of plan §3.4.
"""
from __future__ import annotations

import torch
import torch.nn as nn

_ACT = {"gelu": nn.GELU, "relu": nn.ReLU}


class SpectralConv2d(nn.Module):
    """2D spectral convolution: truncate to (modes_h, modes_w) Fourier modes."""

    def __init__(self, in_ch: int, out_ch: int, modes_h: int, modes_w: int):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes_h, self.modes_w = modes_h, modes_w
        scale = 1.0 / (in_ch * out_ch)
        # Two corner blocks of the rfft2 spectrum carry the low modes.
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes_h, modes_w, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes_h, modes_w, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        mh = min(self.modes_h, H)
        mw = min(self.modes_w, W // 2 + 1)
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :mh, :mw] = torch.einsum(
            "bihw,iohw->bohw", x_ft[:, :, :mh, :mw], self.w1[:, :, :mh, :mw]
        )
        out_ft[:, :, -mh:, :mw] = torch.einsum(
            "bihw,iohw->bohw", x_ft[:, :, -mh:, :mw], self.w2[:, :, :mh, :mw]
        )
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNO2d(nn.Module):
    """Stacked spectral conv blocks with truncation to n_modes Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 64,
                 n_modes: tuple[int, int] = (12, 12),
                 n_layers: int = 4,
                 activation: str = "gelu") -> None:
        super().__init__()
        act = _ACT.get(activation, nn.GELU)
        self.lift = nn.Conv2d(in_channels, hidden_channels, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(hidden_channels, hidden_channels, *n_modes) for _ in range(n_layers)]
        )
        self.local = nn.ModuleList(
            [nn.Conv2d(hidden_channels, hidden_channels, 1) for _ in range(n_layers)]
        )
        self.act = act()
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            act(),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lift(x)
        for spec, loc in zip(self.spectral, self.local):
            h = h + self.act(spec(h) + loc(h))
        return self.project(h)
