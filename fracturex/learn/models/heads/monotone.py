# fracturex/learn/models/heads/monotone.py
"""Irreversibility-preserving head (plan §3.5 eq. 3.17b).

Network emits per-step increments z_n; the head returns
    Δd_n = softplus(z_n) ≥ 0
    d_n  = clip(d_0 + Σ_{k≤n} Δd_k, 0, 1)

By construction d_n ≥ d_{n-1} and d_n ∈ [0, 1], so the irreversibility
constraint is hard rather than a soft penalty. Plan §9.1 lists this as
one of the paper's three core contributions.
"""
from __future__ import annotations
from typing import Optional


class MonotoneIncrementHead:
    """Maps per-step latent z (B, T, C, H, W) to cumulative d_n (B, T, 1, H, W)."""

    def __init__(self, *, d0_strategy: str = "from_input",
                 softplus_beta: float = 1.0) -> None:
        """d0_strategy: 'from_input' takes d_0 from the input channel;
                       'zero' starts from undamaged."""
        raise NotImplementedError("M2 Stage E: implement cumulative softplus head")

    def forward(self, z_seq, d0=None):
        raise NotImplementedError
