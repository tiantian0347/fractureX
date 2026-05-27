# fracturex/learn/models/deeponet.py
"""DeepONet branch-trunk baseline.

Plan §3.4 eq. (3.12): G(a)(y) ≈ Σ_j br_j(a) · tr_j(y).

Branch ingests (SDF + mask + material + load_history); trunk takes (x, y, t).
Mesh-flexible — included for theoretical baseline + sanity that the chosen
representation isn't accidentally over-fit to FNO's strengths.
"""
from __future__ import annotations
from typing import Optional


class DeepONet:
    def __init__(self,
                 branch_in: int, branch_hidden: tuple[int, ...] = (256, 256, 256),
                 trunk_in: int = 3, trunk_hidden: tuple[int, ...] = (256, 256, 256, 256),
                 latent_dim: int = 128,
                 activation: str = "gelu") -> None:
        raise NotImplementedError("M1 task: branch + trunk MLP, inner-product head")

    def forward(self, branch_input, trunk_input):
        raise NotImplementedError
