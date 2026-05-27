# fracturex/postprocess/dataset_export.py
"""Encode a RunRecorder output dir into a training-ready npz sample.

External protocol: docs/SURROGATE_DATA_SCHEMA.md.
Math: docs/plan_operator_learning.md §3.3 (E_h^in, E_h^out, masks).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class GridSpec:
    """Structured grid for E_h^in / E_h^out."""
    H: int
    W: int
    bbox: tuple[tuple[float, float], tuple[float, float]]  # ((x_lo, x_hi), (y_lo, y_hi))


@dataclass(frozen=True)
class ExportConfig:
    """Knobs that change schema content; written to metadata.json."""
    grid: GridSpec
    formulation: str = "standard"               # 'standard' | 'effective_stress'
    interpolation: str = "I2_L2_projection"     # 'I1_nearest_quad' | 'I2_L2_projection'
    include_history: bool = False               # plan §3.5 (b) variant
    include_material_field: bool = False        # heterogeneous material
    stress_scale: Optional[float] = None        # None → dataset-level autoscale later
    u_scale: Optional[float] = None
    boundary_codes: Optional[dict[int, str]] = None


# ---------------------------------------------------------------------------
# Inputs encoding  (E_h^in)
# ---------------------------------------------------------------------------

def compute_sdf(grid: GridSpec, geometry) -> np.ndarray:
    """Signed distance to ∂Ω on the structured grid; <0 outside Ω.

    Returns:
        (1, H, W) float32 SDF in length units.
    """
    raise NotImplementedError("M0 task: signed-distance to domain (notch-aware)")


def compute_valid_mask(grid: GridSpec, geometry) -> np.ndarray:
    """Boolean inside-Ω mask; 1 = inside (incl. boundary), 0 = outside.

    Returns:
        (1, H, W) uint8.
    """
    raise NotImplementedError("M0 task: mask from geometry (consistent with sdf>=0)")


def compute_coords(grid: GridSpec) -> np.ndarray:
    """Normalized (x, y) ∈ [0,1]^2 on grid bbox.

    Returns:
        (2, H, W) float32. coords[0] = x normalized, coords[1] = y.
    """
    raise NotImplementedError("M0 task: trivial linspace meshgrid")


def encode_inputs(recorder_dir: Path, cfg: ExportConfig) -> dict[str, np.ndarray]:
    """Build the input portion of one sample's npz payload.

    Returns dict keyed by SURROGATE_DATA_SCHEMA §3.1 field names.
    """
    raise NotImplementedError("M0 task: read recorder meta + emit input tensors")


# ---------------------------------------------------------------------------
# Outputs encoding  (E_h^out)
# ---------------------------------------------------------------------------

def sample_field_nearest_quad(
    field_qp: np.ndarray,
    quad_coords: np.ndarray,
    grid: GridSpec,
) -> np.ndarray:
    """𝓘₁: nearest-quadrature-point scatter to structured grid.

    Args:
        field_qp:    (NC, NQ, ...) field on quadrature points.
        quad_coords: (NC, NQ, 2) physical coords of those points.
        grid:        target grid.
    Returns:
        (..., H, W) float32 sampled field; out-of-Ω points are zero-filled
        (mask applied by caller).
    """
    raise NotImplementedError("M0 task: KDTree nearest-neighbor, mask out-of-Ω")


def sample_field_l2_projection(
    field_qp: np.ndarray,
    discr,           # huzhang_discretization-like object exposing space_d / quadrature info
    grid: GridSpec,
) -> np.ndarray:
    """𝓘₂: L^2-project to nodal Lagrange space, then evaluate on grid.

    See plan §3.3, m0_interpolation_error.md §2.2.
    """
    raise NotImplementedError("M0 task: assemble mass matrix, solve, evaluate Lagrange basis")


def sample_huzhang_stress_on_grid(
    sigma_dofs: np.ndarray,
    discr,
    grid: GridSpec,
) -> np.ndarray:
    """Evaluate Hu-Zhang σ_h directly via FE basis (no projection).

    Returns:
        (3, H, W) float32 with channel order (σ_xx, σ_yy, σ_xy).
    """
    raise NotImplementedError("M0 task: locate (x,y) in mesh, evaluate Hu-Zhang basis")


def encode_outputs(recorder_dir: Path, cfg: ExportConfig) -> dict[str, np.ndarray]:
    """Build the output portion of one sample's npz payload.

    Iterates over checkpoint steps in recorder_dir/checkpoints/step_XXX.npz,
    samples damage / stress / (optional history) onto the grid, stacks along
    time axis, and applies normalization per cfg / dataset-level scaling.
    """
    raise NotImplementedError("M0 task: iterate checkpoints, sample fields, normalize")


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def export_recorder_to_sample(
    recorder_dir: Path,
    out_npz: Path,
    out_meta: Path,
    cfg: ExportConfig,
    *,
    sample_id: Optional[str] = None,
    git_commit: Optional[str] = None,
    config_hash: Optional[str] = None,
) -> None:
    """Encode one recorder run into <sample>.npz + <sample>.meta.json.

    Writes per SURROGATE_DATA_SCHEMA.md §2-§4. Side effect: out_npz and
    out_meta are written atomically (write to .tmp then rename) so partial
    failures don't leave half-written samples.
    """
    raise NotImplementedError("M0 task: orchestrate encode_inputs/outputs + write meta")
