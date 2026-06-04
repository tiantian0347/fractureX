# fracturex/learn/models/deeponet.py
"""DeepONet branch-trunk baseline.

Plan §3.4 eq. (3.12): G(a)(y) ≈ Σ_j br_j(a) · tr_j(y).

Branch ingests the input function sampled on a fixed sensor grid (SDF + mask +
material planes); trunk takes the query coordinates (x, y). Mesh-flexible —
included as a theoretical baseline and a sanity check that the chosen
representation isn't accidentally over-fit to FNO's strengths.

M1 scheme A: one shared trunk basis over the grid, one branch coefficient set
per time step, giving ``(B, C_in, H, W) → (B, T, H, W)`` through ``forward(x)``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

_ACT = {"gelu": nn.GELU, "relu": nn.ReLU}


def _mlp(sizes: tuple[int, ...], act) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *,
                 grid_hw: tuple[int, int],
                 sensor: int = 8,
                 branch_hidden: tuple[int, ...] = (256, 256),
                 trunk_hidden: tuple[int, ...] = (256, 256, 256),
                 latent_dim: int = 128,
                 activation: str = "gelu") -> None:
        super().__init__()
        act = _ACT.get(activation, nn.GELU)
        self.in_channels = in_channels
        self.out_channels = out_channels          # T
        self.latent = latent_dim
        self.sensor = sensor
        self.pool = nn.AdaptiveAvgPool2d(sensor)   # fixed sensor grid (sensor×sensor)

        branch_in = in_channels * sensor * sensor
        # Branch outputs T·latent coefficients (one latent vector per time step).
        self.branch = _mlp((branch_in, *branch_hidden, out_channels * latent_dim), act)
        self.trunk = _mlp((2, *trunk_hidden, latent_dim), act)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self._grid_hw = grid_hw

    def _trunk_coords(self, H: int, W: int, device) -> torch.Tensor:
        ys = torch.linspace(0.0, 1.0, H, device=device)
        xs = torch.linspace(0.0, 1.0, W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (H*W, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        b = self.pool(x).reshape(B, -1)                       # (B, C*sensor^2)
        coeff = self.branch(b).reshape(B, self.out_channels, self.latent)  # (B, T, p)
        coords = self._trunk_coords(H, W, x.device)           # (H*W, 2)
        basis = self.trunk(coords)                            # (H*W, p)
        out = torch.einsum("btp,np->btn", coeff, basis)       # (B, T, H*W)
        out = out + self.bias[None, :, None]
        return out.reshape(B, self.out_channels, H, W)
