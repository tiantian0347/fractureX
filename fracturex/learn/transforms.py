# fracturex/learn/transforms.py
"""Target transforms for heavy-tailed fields (Stage B σ).

The Hu-Zhang stress field has a crack-tip singularity: after per-sample
``stress_scale`` normalization it is still heavy-tailed (median ~0.1, p95 ~14,
max ~340 on the m1_pilot dataset). A plain relative-L² loss is then dominated
by the few singular pixels, so a smooth network under-predicts the peak
(see docs/operator_learning/m2_stageB_results.md §3.2).

``arcsinh`` compresses the tail while staying ~linear and sign-preserving near
zero: asinh(0.1)≈0.1, asinh(14)≈3.3, asinh(344)≈6.5. We train σ in this space
and invert with ``sinh`` for physical-unit evaluation. Works on both numpy
arrays and torch tensors (dispatch by duck-typing).
"""
from __future__ import annotations

import numpy as np


def _is_torch(x) -> bool:
    return hasattr(x, "detach") and hasattr(x, "device")


def sigma_forward(x, kind: str = "arcsinh", scale: float = 1.0):
    """Map physical(normalized) σ → training space."""
    if kind == "none":
        return x
    if kind == "arcsinh":
        if _is_torch(x):
            import torch
            return torch.asinh(x / scale)
        return np.arcsinh(np.asarray(x) / scale)
    raise ValueError(f"unknown sigma transform {kind!r}")


def sigma_inverse(y, kind: str = "arcsinh", scale: float = 1.0):
    """Invert :func:`sigma_forward` (training space → physical/normalized σ)."""
    if kind == "none":
        return y
    if kind == "arcsinh":
        if _is_torch(y):
            import torch
            return torch.sinh(y) * scale
        return np.sinh(np.asarray(y)) * scale
    raise ValueError(f"unknown sigma transform {kind!r}")
