# fracturex/learn/eval/reconstruct.py
"""Reconstruction operator R_h: structured grid → continuous function space.

Plan §3.3 defines R_h as the inverse of the tensorization step (E_h^out).
R_h is used ONLY in error analysis (plan §3.6 eq. 3.18) — it does not
participate in training. Implementations:

  - piecewise_bilinear:    cheap, O(h^2) accurate.
  - piecewise_polynomial:  match the Lagrange degree of the source field.

Both return a callable f(x) defined on Ω for evaluation against the
original FE field σ_h / d_h.
"""
from __future__ import annotations
from typing import Callable

import numpy as np


def piecewise_bilinear_reconstruct(field_grid: np.ndarray, grid_bbox) -> Callable:
    """Bilinear interpolation back to continuous function space.

    Args:
        field_grid: (C, H, W) field on the structured grid.
        grid_bbox:  ((x_lo, x_hi), (y_lo, y_hi)).

    Returns:
        f: (x, y) → (C,) field value; outside bbox is NaN.
    """
    raise NotImplementedError("Evaluation-only: bilinear interpolation")


def piecewise_polynomial_reconstruct(
    field_grid: np.ndarray, grid_bbox, *, degree: int = 1,
) -> Callable:
    """Higher-order reconstruction matching source Lagrange degree.

    For degree=1 reduces to bilinear; degree=2 uses biquadratic over
    overlapping 3x3 patches.
    """
    raise NotImplementedError("Evaluation-only: polynomial reconstruction")
