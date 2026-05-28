# fracturex/postprocess/dataset_export.py
"""Encode a RunRecorder output dir into a training-ready npz sample.

External protocol: docs/SURROGATE_DATA_SCHEMA.md.
Math: docs/plan_operator_learning.md §3.3 (E_h^in, E_h^out, masks).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import os
import subprocess
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np

SCHEMA_VERSION = "0.1"

# Voigt convention used by HuZhangFESpace2d: stress channels [σxx, σxy, σyy].
# Schema §3.2 stores stress as (σxx, σyy, σxy) -- conversion happens at the
# very end of `encode_outputs` so the FE-side code stays in HuZhang order.
_HZ_TO_SCHEMA_STRESS = (0, 2, 1)  # [xx, xy, yy] -> [xx, yy, xy]


@dataclass(frozen=True)
class GridSpec:
    """Structured grid for E_h^in / E_h^out."""
    H: int
    W: int
    bbox: tuple[tuple[float, float], tuple[float, float]]  # ((x_lo, x_hi), (y_lo, y_hi))


# ---------------------------------------------------------------------------
# Geometry abstraction.  A "geometry" describes Ω ⊂ R² for SDF/mask building.
# Two representations are accepted:
#   1. A callable signed_distance(points) → array of same leading shape, with
#      *positive inside Ω* (matches schema §3.1 sdf convention).
#   2. A primitive descriptor we know how to handle (model0_circular_notch).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CircularNotchDomain:
    """Ω = box [x0,x1]×[y0,y1] minus a disk of radius r at (cx, cy).

    Matches `Model0CircularNotchCase` (cases/model0_circular_notch.py).
    """
    box: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    cx: float = 0.5
    cy: float = 0.5
    r: float = 0.2

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """SDF (positive inside Ω) for points (..., 2)."""
        pts = np.asarray(points, dtype=np.float64)
        x0, x1, y0, y1 = self.box
        x = pts[..., 0]
        y = pts[..., 1]
        # box SDF (positive inside)
        d_box = np.minimum.reduce([x - x0, x1 - x, y - y0, y1 - y])
        # disk SDF: dist_to_circle = r - radius;  inside disk ⇒ positive.
        # The disk is a *hole*, so Ω-inside ⇔ outside disk ⇔ -d_disk_inside.
        d_to_center = np.sqrt((x - self.cx) ** 2 + (y - self.cy) ** 2)
        d_outside_disk = d_to_center - self.r  # positive when outside disk
        return np.minimum(d_box, d_outside_disk)


GeometryLike = Union[CircularNotchDomain, Callable[[np.ndarray], np.ndarray]]


def _signed_distance(geometry: GeometryLike, points: np.ndarray) -> np.ndarray:
    if hasattr(geometry, "signed_distance"):
        return np.asarray(geometry.signed_distance(points), dtype=np.float64)
    if callable(geometry):
        return np.asarray(geometry(points), dtype=np.float64)
    raise TypeError(
        f"geometry must expose signed_distance(points) or be callable; got {type(geometry)!r}"
    )


def _grid_points(grid: GridSpec) -> np.ndarray:
    """(H, W, 2) physical (x, y) coordinates of the structured grid centers."""
    (x_lo, x_hi), (y_lo, y_hi) = grid.bbox
    xs = np.linspace(x_lo, x_hi, grid.W, dtype=np.float64)
    ys = np.linspace(y_lo, y_hi, grid.H, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")  # X,Y: (H, W)
    return np.stack([X, Y], axis=-1)


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

def compute_sdf(grid: GridSpec, geometry: GeometryLike) -> np.ndarray:
    """Signed distance to ∂Ω on the structured grid.

    Convention (schema §3.1): **positive inside Ω, negative outside, zero on ∂Ω**.

    Returns:
        (1, H, W) float32 SDF in length units.
    """
    pts = _grid_points(grid)  # (H, W, 2)
    sdf_hw = _signed_distance(geometry, pts)
    return sdf_hw.astype(np.float32)[None, ...]


def compute_valid_mask(grid: GridSpec, geometry: GeometryLike) -> np.ndarray:
    """Boolean inside-Ω mask; 1 = inside (incl. boundary), 0 = outside.

    Computed from the same SDF as `compute_sdf`, so the two are consistent
    by construction (mask ≡ sdf >= 0).

    Returns:
        (1, H, W) uint8.
    """
    pts = _grid_points(grid)
    sdf_hw = _signed_distance(geometry, pts)
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


def encode_inputs(
    recorder_dir: Path,
    cfg: ExportConfig,
    geometry: GeometryLike,
    *,
    load_history: Optional[np.ndarray] = None,
    material_order: Sequence[str] = ("lambda", "mu", "Gc", "l0", "eta"),
    material_overrides: Optional[dict] = None,
) -> dict[str, np.ndarray]:
    """Build the input portion of one sample's npz payload.

    Reads ``recorder_dir/meta.json`` for material parameters and load history
    (the latter is reconstructed from ``history.csv`` when not supplied).
    Computes ``sdf``, ``mask``, ``coords`` from ``geometry`` on ``cfg.grid``.

    Returns dict keyed by SURROGATE_DATA_SCHEMA §3.1 field names.
    """
    recorder_dir = Path(recorder_dir)
    meta = _read_recorder_meta(recorder_dir)
    material_meta = dict(meta.get("material") or {})
    if material_overrides:
        material_meta.update(material_overrides)
    material_vec = _material_vector(material_meta, material_order, eta_default=1e-9)

    if load_history is None:
        load_history = _read_history_csv_loads(recorder_dir)
    load_history = np.asarray(load_history, dtype=np.float32)
    if load_history.ndim == 1:
        load_history = load_history[:, None]
    elif load_history.ndim != 2:
        raise ValueError(
            f"load_history must be 1D or 2D, got shape {load_history.shape}"
        )

    out: dict[str, np.ndarray] = {
        "sdf": compute_sdf(cfg.grid, geometry),
        "mask": compute_valid_mask(cfg.grid, geometry),
        "coords": compute_coords(cfg.grid),
        "material": material_vec.astype(np.float32),
        "load_history": load_history.astype(np.float32),
        "time": _normalized_time(load_history.shape[0]),
    }
    return out


def _read_recorder_meta(recorder_dir: Path) -> dict:
    meta_path = recorder_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing meta.json under {recorder_dir}")
    with meta_path.open("r") as f:
        return json.load(f)


def _read_history_csv_loads(recorder_dir: Path) -> np.ndarray:
    """Pull the per-step prescribed load from history.csv.

    The driver writes a ``load`` column on every step (see
    ``HuZhangPhaseFieldStaggeredDriver.solve_one_step``); other columns are
    ignored here. Returns a (T,) float array.
    """
    import csv

    csv_path = recorder_dir / "history.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing history.csv under {recorder_dir}")
    loads: list[float] = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        if "load" not in reader.fieldnames:
            raise ValueError(
                f"history.csv at {csv_path} has no 'load' column; "
                f"available columns: {reader.fieldnames}"
            )
        for row in reader:
            loads.append(float(row["load"]))
    return np.asarray(loads, dtype=np.float32)


def _material_vector(
    material: dict,
    order: Sequence[str],
    *,
    eta_default: float,
) -> np.ndarray:
    aliases = {
        "lambda": ("lambda", "lam", "lambda0"),
        "mu": ("mu", "lambda1"),
        "Gc": ("Gc", "G_c"),
        "l0": ("l0", "ell0", "ell_0"),
        "eta": ("eta",),
    }
    vec = np.zeros(len(order), dtype=np.float64)
    for i, key in enumerate(order):
        candidates = aliases.get(key, (key,))
        for c in candidates:
            if c in material:
                vec[i] = float(material[c])
                break
        else:
            if key == "eta":
                vec[i] = float(eta_default)
            else:
                raise KeyError(
                    f"material[{key!r}] missing in recorder meta (looked for "
                    f"{candidates}); cannot build material vector for schema."
                )
    return vec


def _normalized_time(T: int) -> np.ndarray:
    if T <= 0:
        return np.zeros((0,), dtype=np.float32)
    if T == 1:
        return np.zeros((1,), dtype=np.float32)
    return np.linspace(0.0, 1.0, T, dtype=np.float32)


# ---------------------------------------------------------------------------
# Pixel → mesh location & per-cell evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PixelLocator:
    """Map structured-grid pixels onto the unstructured mesh.

    Attributes:
        cell_id: (H*W,) int64; -1 for pixels with no triangle (outside the
            mesh hull / inside notch holes).
        bary:    (H*W, 3) float64 barycentric coords; rows where cell_id == -1
            are zero-filled.
        H, W:    grid resolution.
    """

    cell_id: np.ndarray
    bary: np.ndarray
    H: int
    W: int


def _build_pixel_locator(mesh, grid: "GridSpec") -> _PixelLocator:
    """Locate every grid pixel in the FE mesh (matplotlib trifinder + bary)."""
    from matplotlib.tri import Triangulation, TrapezoidMapTriFinder

    node = np.asarray(mesh.entity("node"), dtype=np.float64)
    cell = np.asarray(mesh.entity("cell"), dtype=np.int64)
    if cell.shape[1] != 3:
        raise ValueError(
            f"_build_pixel_locator expects a triangle mesh; got cell shape {cell.shape}"
        )

    pts = _grid_points(grid).reshape(-1, 2)  # (HW, 2)

    tri = Triangulation(node[:, 0], node[:, 1], cell)
    finder = TrapezoidMapTriFinder(tri)
    cell_id = finder(pts[:, 0], pts[:, 1]).astype(np.int64)  # -1 outside

    bary = np.zeros((pts.shape[0], 3), dtype=np.float64)
    inside = cell_id >= 0
    if np.any(inside):
        cid = cell_id[inside]
        v = node[cell[cid]]  # (n_in, 3, 2)
        x0, x1, x2 = v[:, 0], v[:, 1], v[:, 2]
        # 2A * λ_i derived from triangle edge cross products.
        det = (x1[:, 0] - x0[:, 0]) * (x2[:, 1] - x0[:, 1]) - (
            x2[:, 0] - x0[:, 0]
        ) * (x1[:, 1] - x0[:, 1])
        p = pts[inside]
        l1 = ((x1[:, 0] - p[:, 0]) * (x2[:, 1] - p[:, 1])
              - (x2[:, 0] - p[:, 0]) * (x1[:, 1] - p[:, 1])) / det
        l2 = ((x2[:, 0] - p[:, 0]) * (x0[:, 1] - p[:, 1])
              - (x0[:, 0] - p[:, 0]) * (x2[:, 1] - p[:, 1])) / det
        l3 = 1.0 - l1 - l2
        bary[inside, 0] = l1
        bary[inside, 1] = l2
        bary[inside, 2] = l3

    return _PixelLocator(cell_id=cell_id, bary=bary, H=grid.H, W=grid.W)


def _group_pixels_by_cell(locator: _PixelLocator) -> dict[int, np.ndarray]:
    """Return ``cell_id → array of pixel indices`` (only inside pixels)."""
    cid = locator.cell_id
    inside = cid >= 0
    pix_idx = np.flatnonzero(inside)
    sort = np.argsort(cid[inside], kind="stable")
    sorted_cells = cid[inside][sort]
    sorted_pix = pix_idx[sort]
    boundaries = np.flatnonzero(np.diff(sorted_cells)) + 1
    splits = np.split(sorted_pix, boundaries)
    cell_ids_unique = np.split(sorted_cells, boundaries)
    return {int(group[0]): pixels for group, pixels in zip(cell_ids_unique, splits)}


def _evaluate_huzhang_on_grid(
    space, dofs: np.ndarray, locator: _PixelLocator
) -> np.ndarray:
    """Evaluate a HuZhang stress function at every inside pixel.

    Args:
        space: ``HuZhangFESpace2d`` instance.
        dofs:  (gdof,) DOF vector (one snapshot, single field).
        locator: pixel→cell mapping.

    Returns:
        (3, H, W) float32 in HuZhang Voigt order [σxx, σxy, σyy]. Outside-Ω
        pixels are zero.
    """
    HW = locator.H * locator.W
    out = np.zeros((HW, 3), dtype=np.float64)
    cell_to_dof = np.asarray(space.dof.cell_to_dof())
    groups = _group_pixels_by_cell(locator)
    for cid, pix in groups.items():
        bc = locator.bary[pix].astype(np.float64)  # (n, 3)
        idx = np.array([cid], dtype=np.int64)
        phi = np.asarray(space.basis(bc, index=idx))  # (1, n, ldof, 3)
        local_dofs = np.asarray(dofs)[cell_to_dof[cid]]  # (ldof,)
        out[pix] = np.einsum("qld,l->qd", phi[0], local_dofs)
    return out.reshape(locator.H, locator.W, 3).transpose(2, 0, 1).astype(np.float32)


def _evaluate_lagrange_on_grid(
    space, dofs: np.ndarray, locator: _PixelLocator
) -> np.ndarray:
    """Evaluate a continuous Lagrange scalar function on every inside pixel.

    Returns:
        (1, H, W) float32. Outside-Ω pixels are zero.
    """
    HW = locator.H * locator.W
    out = np.zeros(HW, dtype=np.float64)
    cell_to_dof = np.asarray(space.dof.cell_to_dof())
    groups = _group_pixels_by_cell(locator)
    for cid, pix in groups.items():
        bc = locator.bary[pix].astype(np.float64)
        idx = np.array([cid], dtype=np.int64)
        phi = np.asarray(space.basis(bc, index=idx))  # (1, n, ldof) or (1, n, ldof, 1)
        if phi.ndim == 4 and phi.shape[-1] == 1:
            phi = phi[..., 0]
        local_dofs = np.asarray(dofs)[cell_to_dof[cid]]
        out[pix] = np.einsum("ql,l->q", phi[0], local_dofs)
    return out.reshape(1, locator.H, locator.W).astype(np.float32)

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
    *,
    locator: Optional[_PixelLocator] = None,
) -> np.ndarray:
    """Evaluate Hu-Zhang σ_h directly via FE basis (no projection).

    Returns:
        (3, H, W) float32 in **HuZhang Voigt order [σxx, σxy, σyy]**.
        Channel reordering to schema (σxx, σyy, σxy) happens in
        :func:`encode_outputs`.
    """
    space = discr.space_sigma
    if space is None:
        raise RuntimeError("discr.space_sigma is None; build() the discretization first.")
    if locator is None:
        locator = _build_pixel_locator(discr.mesh, grid)
    return _evaluate_huzhang_on_grid(space, np.asarray(sigma_dofs), locator)


def encode_outputs(
    recorder_dir: Path,
    cfg: ExportConfig,
    discr,
    *,
    geometry: Optional[GeometryLike] = None,
    locator: Optional[_PixelLocator] = None,
    stress_scale: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """Build the output portion of one sample's npz payload.

    Iterates ``recorder_dir/checkpoints/step_XXX.npz`` in order, evaluates
    ``damage`` and ``stress`` on the grid via FE basis, masks pixels outside
    Ω to zero, and packs ``(T, C, H, W)`` tensors per schema §3.2.

    The returned ``stress`` is in **schema channel order (σxx, σyy, σxy)**
    and is divided by ``stress_scale`` (cfg.stress_scale, falls back to
    auto-computed 95th percentile of |σ_xx|+|σ_yy|+2|σ_xy| on the first frame
    if neither is given).
    """
    recorder_dir = Path(recorder_dir)
    ckpt_dir = recorder_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"missing {ckpt_dir}")
    ckpts = sorted(ckpt_dir.glob("step_*.npz"))
    if not ckpts:
        raise FileNotFoundError(f"no step_*.npz under {ckpt_dir}")

    if locator is None:
        locator = _build_pixel_locator(discr.mesh, cfg.grid)

    # In-Ω mask: prefer geometry-based valid_mask; fall back to locator.cell_id
    # (mesh hull). The mesh hull is conservative — pixels inside the notch
    # but mistakenly meshed (shouldn't happen with current cases) would leak.
    if geometry is not None:
        mask_hw = compute_valid_mask(cfg.grid, geometry)[0].astype(bool)  # (H, W)
    else:
        mask_hw = (locator.cell_id.reshape(cfg.grid.H, cfg.grid.W) >= 0)

    T = len(ckpts)
    H, W = cfg.grid.H, cfg.grid.W
    damage_t = np.zeros((T, 1, H, W), dtype=np.float32)
    stress_hz_t = np.zeros((T, 3, H, W), dtype=np.float32)  # HuZhang order

    space_d = discr.space_d
    if space_d is None:
        raise RuntimeError("discr.space_d is None; build() the discretization first.")

    # Track step_iters / step_converged from history.csv if available; the
    # checkpoint itself doesn't carry these flags.
    step_iters = np.zeros(T, dtype=np.int32)
    step_converged = np.ones(T, dtype=np.uint8)
    try:
        meta_iters, meta_conv = _read_history_csv_iter_status(recorder_dir, T)
        step_iters = meta_iters
        step_converged = meta_conv
    except (FileNotFoundError, KeyError, ValueError):
        pass

    for t, ckpt in enumerate(ckpts):
        z = np.load(ckpt, allow_pickle=False)
        d_dofs = np.asarray(z["d"])
        sigma_dofs = np.asarray(z["sigma"])

        d_grid = _evaluate_lagrange_on_grid(space_d, d_dofs, locator)  # (1, H, W)
        sigma_grid_hz = _evaluate_huzhang_on_grid(
            discr.space_sigma, sigma_dofs, locator
        )  # (3, H, W) in [xx, xy, yy]

        d_grid[0][~mask_hw] = 0.0
        sigma_grid_hz[:, ~mask_hw] = 0.0

        damage_t[t] = d_grid
        stress_hz_t[t] = sigma_grid_hz

    # Reorder HuZhang [xx, xy, yy] → schema [xx, yy, xy].
    stress_t = stress_hz_t[:, _HZ_TO_SCHEMA_STRESS, :, :].copy()

    # Stress scale.
    if stress_scale is None:
        stress_scale = cfg.stress_scale
    if stress_scale is None:
        # Conservative auto-scale using last frame to avoid 0-stress at t=0.
        last = stress_t[-1]
        magnitude = np.percentile(np.abs(last[:, mask_hw]), 95.0)
        stress_scale = float(magnitude) if magnitude > 0 else 1.0
    stress_scale = float(stress_scale)
    stress_t = (stress_t / stress_scale).astype(np.float32)

    out: dict[str, np.ndarray] = {
        "damage": damage_t,
        "stress": stress_t,
        "step_iters": step_iters,
        "step_converged": step_converged,
        "_stress_scale": np.float32(stress_scale),  # consumed by export_*; popped.
    }
    return out


def _read_history_csv_iter_status(
    recorder_dir: Path, T: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read ``iters`` and ``converged`` columns from history.csv (if present)."""
    import csv

    csv_path = recorder_dir / "history.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    iters: list[int] = []
    converged: list[int] = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        if "iters" not in reader.fieldnames or "converged" not in reader.fieldnames:
            raise KeyError("history.csv missing iters / converged column")
        for row in reader:
            iters.append(int(row["iters"]))
            cv = str(row["converged"]).strip().lower()
            converged.append(1 if cv in ("1", "true", "yes") else 0)
    if len(iters) != T:
        # Recorder may have written more rows than checkpoints when
        # save_every>1; truncate / pad.
        iters = iters[:T] + [0] * max(0, T - len(iters))
        converged = converged[:T] + [1] * max(0, T - len(converged))
    return (
        np.asarray(iters, dtype=np.int32),
        np.asarray(converged, dtype=np.uint8),
    )


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def load_discr_from_dir(recorder_dir: Path):
    """Rebuild a HuZhangDiscretization from ``<recorder_dir>/mesh.npz``.

    Mirrors :meth:`fracturex.postprocess.recorder.RunRecorder.save_mesh`.
    Avoids ``case.make_mesh`` (distmesh isn't reproducible) and avoids
    ``case.isD_bd`` by reusing the persisted Neumann/Dirichlet edge mask.

    Returns:
        A built ``HuZhangDiscretization`` whose ``mesh`` matches the one used
        to produce the run's checkpoints. ``state`` fields are zero-initialized.
    """
    from fealpy.backend import backend_manager as bm
    from fealpy.functionspace.huzhang_fe_space_2d import HuZhangFESpace2d
    from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
    from fealpy.mesh import TriangleMesh

    from fracturex.cases.base import CaseBase
    from fracturex.discretization.huzhang_discretization import (
        HuZhangDiscretization,
        HuZhangState,
    )

    recorder_dir = Path(recorder_dir)
    mesh_path = recorder_dir / "mesh.npz"
    if not mesh_path.exists():
        raise FileNotFoundError(
            f"missing {mesh_path}; run with a recorder that calls "
            "RunRecorder.save_mesh (driver auto-emits since 2026-05-28). "
            "Older runs need to be rerun or reconstructed from `case` "
            "directly."
        )

    z = np.load(mesh_path, allow_pickle=False)
    node = np.asarray(z["node"], dtype=np.float64)
    cell = np.asarray(z["cell"], dtype=np.int64)
    p_sigma = int(z["p_sigma"])
    damage_p = int(z["damage_p"])
    u_space_order = int(z["u_space_order"])
    use_relaxation = bool(z["use_relaxation"])
    be_aug = np.asarray(z["boundary_edge_flag_aug"], dtype=bool)
    is_n_bd = np.asarray(z["is_neumann_edge"], dtype=bool)

    mesh = TriangleMesh(bm.asarray(node), bm.asarray(cell))

    # Patch boundary_edge_flag so HuZhangFESpace2d sees the augmented set.
    def _be_aug():
        return bm.asarray(be_aug)

    mesh.boundary_edge_flag = _be_aug
    if hasattr(mesh, "boundary_face_index"):
        def _bfi_aug():
            return bm.where(_be_aug())[0]
        mesh.boundary_face_index = _bfi_aug

    space_sigma = HuZhangFESpace2d(
        mesh, p=p_sigma, use_relaxation=use_relaxation, bd_stress=is_n_bd
    )
    u_scalar = LagrangeFESpace(mesh, p=u_space_order, ctype="D")
    space_u = TensorFunctionSpace(u_scalar, shape=(2, -1))
    space_d = LagrangeFESpace(mesh, p=damage_p, ctype="C")

    sigma = space_sigma.function()
    u = space_u.function()
    d = space_d.function()
    r_hist = space_d.function()

    # Use a dummy case purely to satisfy CaseBase typing; it is never invoked
    # because we hand-build mesh + spaces below.
    class _RebuiltCase(CaseBase):
        name = "rebuilt_from_recorder"
        def make_mesh(self, **kw):
            raise RuntimeError("rebuilt discr already has a mesh")
        def isD_bd(self, points):
            raise RuntimeError("rebuilt discr already has Neumann edges")
        def model(self):
            raise RuntimeError("rebuilt discr has no material model attached")

    discr = HuZhangDiscretization(
        _RebuiltCase(),
        p=p_sigma,
        use_relaxation=use_relaxation,
        damage_p=damage_p,
        u_space_order=u_space_order,
    )
    discr.mesh = mesh
    discr.space_sigma = space_sigma
    discr.space_u = space_u
    discr.space_d = space_d
    discr.state = HuZhangState(sigma=sigma, u=u, d=d, r_hist=r_hist, H=None)
    return discr


def export_recorder_to_sample(
    recorder_dir: Path,
    out_npz: Path,
    out_meta: Path,
    cfg: ExportConfig,
    discr,
    geometry: GeometryLike,
    *,
    sample_id: Optional[str] = None,
    git_commit: Optional[str] = None,
    config_hash: Optional[str] = None,
    load_history: Optional[np.ndarray] = None,
    material_overrides: Optional[dict] = None,
    extra_meta: Optional[dict] = None,
) -> dict:
    """Encode one recorder run into <sample>.npz + <sample>.meta.json.

    Writes per SURROGATE_DATA_SCHEMA.md §2-§4. Side effect: out_npz and
    out_meta are written atomically (write to .tmp then rename) so partial
    failures don't leave half-written samples.

    Returns the metadata dict that was written (for caller-side bookkeeping).
    """
    recorder_dir = Path(recorder_dir)
    out_npz = Path(out_npz)
    out_meta = Path(out_meta)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    locator = _build_pixel_locator(discr.mesh, cfg.grid)

    inputs = encode_inputs(
        recorder_dir,
        cfg,
        geometry,
        load_history=load_history,
        material_overrides=material_overrides,
    )
    outputs = encode_outputs(
        recorder_dir,
        cfg,
        discr,
        geometry=geometry,
        locator=locator,
        stress_scale=cfg.stress_scale,
    )

    stress_scale = float(outputs.pop("_stress_scale"))
    valid_mask = inputs["mask"].copy()

    payload = {
        "sdf": inputs["sdf"],
        "mask": inputs["mask"],
        "coords": inputs["coords"],
        "material": inputs["material"],
        "load_history": inputs["load_history"],
        "time": inputs["time"],
        "damage": outputs["damage"],
        "stress": outputs["stress"],
        "step_iters": outputs["step_iters"],
        "step_converged": outputs["step_converged"],
        "valid_mask": valid_mask,
    }

    npz_tmp = out_npz.with_suffix(out_npz.suffix + ".tmp")
    # np.savez_compressed appends '.npz' to the path if it doesn't end in
    # '.npz'; pass an open file handle to bypass that auto-suffix and keep
    # the atomic-rename contract.
    with npz_tmp.open("wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(npz_tmp, out_npz)

    meta = _build_sample_meta(
        recorder_dir=recorder_dir,
        cfg=cfg,
        discr=discr,
        sample_id=sample_id or out_npz.stem,
        git_commit=git_commit or _git_commit_short(),
        config_hash=config_hash,
        stress_scale=stress_scale,
        damage=outputs["damage"],
        load_history=inputs["load_history"],
        valid_mask=valid_mask,
        material_overrides=material_overrides,
        extra_meta=extra_meta,
    )
    meta_tmp = out_meta.with_suffix(out_meta.suffix + ".tmp")
    with meta_tmp.open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    os.replace(meta_tmp, out_meta)

    return meta


def _git_commit_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _build_sample_meta(
    *,
    recorder_dir: Path,
    cfg: ExportConfig,
    discr,
    sample_id: str,
    git_commit: str,
    config_hash: Optional[str],
    stress_scale: float,
    damage: np.ndarray,
    load_history: np.ndarray,
    valid_mask: np.ndarray,
    material_overrides: Optional[dict],
    extra_meta: Optional[dict],
) -> dict:
    rec_meta = _read_recorder_meta(recorder_dir)
    material_meta = dict(rec_meta.get("material") or {})
    if material_overrides:
        material_meta.update(material_overrides)
    mesh = discr.mesh

    # Mesh stats; h_min/h_max from edge lengths if available.
    try:
        node = np.asarray(mesh.entity("node"))
        cell = np.asarray(mesh.entity("cell"))
        edge_vec = node[cell[:, [1, 2, 0]]] - node[cell[:, [0, 1, 2]]]
        edge_len = np.linalg.norm(edge_vec, axis=-1)
        h_min = float(edge_len.min())
        h_max = float(edge_len.max())
    except Exception:
        h_min = float("nan")
        h_max = float("nan")

    converged_steps = int(np.sum(damage[..., 0, 0, 0] >= 0))  # placeholder
    n_steps = int(damage.shape[0])
    n_converged = int((load_history.shape[0]))  # use checkpoint count as proxy

    meta = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "geometry_params": _geometry_meta_dict(cfg, recorder_dir, rec_meta),
        "material_params": {
            k: float(material_meta.get(k))
            for k in material_meta
            if isinstance(material_meta.get(k), (int, float))
        },
        "material_order": ["lambda", "mu", "Gc", "l0", "eta"],
        "formulation": cfg.formulation,
        "interpolation": cfg.interpolation,
        "mesh_info": {
            "NC": int(mesh.number_of_cells()),
            "NN": int(mesh.number_of_nodes()),
            "h_min": h_min,
            "h_max": h_max,
            "p_sigma": int(getattr(discr, "p", -1)),
            "p_d": int(getattr(discr, "damage_p", -1)),
            "p_u": int(getattr(discr, "u_space_order", -1)),
        },
        "grid": {
            "H": int(cfg.grid.H),
            "W": int(cfg.grid.W),
            "domain_bbox": [list(cfg.grid.bbox[0]), list(cfg.grid.bbox[1])],
        },
        "load": {
            "kind": rec_meta.get("load_kind", "monotone"),
            "u_max": float(np.max(np.abs(load_history))) if load_history.size else 0.0,
            "N_steps": n_steps,
            "load_surfaces": rec_meta.get("load_surfaces", []),
        },
        "scaling": {
            "stress_scale": stress_scale,
            "u_scale": float(cfg.u_scale) if cfg.u_scale is not None else 1.0,
            "length_scale": 1.0,
            "time_scale": 1.0,
        },
        "boundary_codes": cfg.boundary_codes or {},
        "solver_config": rec_meta.get("solver", {}),
        "git_commit": git_commit,
        "config_hash": config_hash or "unknown",
        "run_paths": {"recorder_dir": str(recorder_dir)},
        "stats": {
            "max_damage": float(damage.max()) if damage.size else 0.0,
            "n_inside_pixels": int(valid_mask.sum()),
            "n_valid_steps": int(n_converged),
            "converged_step_ratio": 1.0 if n_steps == 0 else float(n_converged) / n_steps,
        },
    }
    if extra_meta:
        meta.update(extra_meta)
    return meta


def _geometry_meta_dict(
    cfg: ExportConfig, recorder_dir: Path, rec_meta: dict
) -> dict:
    return {
        "case": rec_meta.get("case", "unknown"),
        "domain_bbox": [list(cfg.grid.bbox[0]), list(cfg.grid.bbox[1])],
    }
