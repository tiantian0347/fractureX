# fracturex/postprocess/dataset_export/grid.py
"""Structured-grid spec and grid-side input channels (model-agnostic).

Defines the target regular grid (E_h^in / E_h^out lattice) and the purely
geometric input channels ``sdf`` / ``mask`` / ``coords`` (schema §3.1). None
of this depends on the solver — only on a :data:`GeometryLike`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import GeometryLike, signed_distance


@dataclass(frozen=True)
class GridSpec:
    """Structured grid for E_h^in / E_h^out."""

    H: int
    W: int
    bbox: tuple[tuple[float, float], tuple[float, float]]  # ((x_lo, x_hi), (y_lo, y_hi))


def grid_points(grid: GridSpec) -> np.ndarray:
    """(H, W, 2) physical (x, y) coordinates of the structured grid centers."""
    (x_lo, x_hi), (y_lo, y_hi) = grid.bbox
    xs = np.linspace(x_lo, x_hi, grid.W, dtype=np.float64)
    ys = np.linspace(y_lo, y_hi, grid.H, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")  # X,Y: (H, W)
    return np.stack([X, Y], axis=-1)


# Backwards-compatible private alias.
_grid_points = grid_points


def compute_sdf(grid: GridSpec, geometry: GeometryLike) -> np.ndarray:
    """Signed distance to ∂Ω on the structured grid.

    Convention (schema §3.1): **positive inside Ω, negative outside, zero on ∂Ω**.

    Returns:
        (1, H, W) float32 SDF in length units.
    """
    pts = grid_points(grid)  # (H, W, 2)
    sdf_hw = signed_distance(geometry, pts)
    return sdf_hw.astype(np.float32)[None, ...]


def compute_valid_mask(grid: GridSpec, geometry: GeometryLike) -> np.ndarray:
    """Boolean inside-Ω mask; 1 = inside (incl. boundary), 0 = outside.

    Computed from the same SDF as :func:`compute_sdf`, so the two are
    consistent by construction (mask ≡ sdf >= 0).

    Returns:
        (1, H, W) uint8.
    """
    pts = grid_points(grid)
    sdf_hw = signed_distance(geometry, pts)
    mask = (sdf_hw >= 0.0).astype(np.uint8)
    return mask[None, ...]


def compute_coords(grid: GridSpec) -> np.ndarray:
    """Normalized (x, y) ∈ [0,1]^2 on grid bbox.

    coords[0, i, j] = x normalized; coords[1, i, j] = y normalized.
    For a single-pixel row/col, the corresponding axis is fixed at 0.

    Returns:
        (2, H, W) float32.
    """
    if grid.W == 1:
        x_norm = np.zeros(grid.W, dtype=np.float64)
    else:
        x_norm = np.linspace(0.0, 1.0, grid.W, dtype=np.float64)
    if grid.H == 1:
        y_norm = np.zeros(grid.H, dtype=np.float64)
    else:
        y_norm = np.linspace(0.0, 1.0, grid.H, dtype=np.float64)
    X, Y = np.meshgrid(x_norm, y_norm, indexing="xy")  # (H, W)
    return np.stack([X, Y], axis=0).astype(np.float32)
