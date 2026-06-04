# fracturex/postprocess/dataset_export/sampling.py
"""Pixel↔mesh location and field sampling onto the structured grid.

These utilities are **FE-model-agnostic**: they work for any triangle mesh and
any continuous Lagrange space, and underpin both the 𝓘₁ (nearest-quadrature)
and 𝓘₂ (L²-projection) sampling schemes (plan §3.3, m0_interpolation_error.md).
Model-specific stress evaluation (e.g. Hu-Zhang basis) lives in the relevant
solver adapter, not here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .grid import GridSpec, grid_points


@dataclass(frozen=True)
class PixelLocator:
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


def build_pixel_locator(mesh, grid: GridSpec) -> PixelLocator:
    """Locate every grid pixel in the FE mesh (matplotlib trifinder + bary)."""
    from matplotlib.tri import Triangulation, TrapezoidMapTriFinder

    node = np.asarray(mesh.entity("node"), dtype=np.float64)
    cell = np.asarray(mesh.entity("cell"), dtype=np.int64)
    if cell.shape[1] != 3:
        raise ValueError(
            f"build_pixel_locator expects a triangle mesh; got cell shape {cell.shape}"
        )

    pts = grid_points(grid).reshape(-1, 2)  # (HW, 2)

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

    return PixelLocator(cell_id=cell_id, bary=bary, H=grid.H, W=grid.W)


def group_pixels_by_cell(locator: PixelLocator) -> dict[int, np.ndarray]:
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


def evaluate_lagrange_on_grid(space, dofs: np.ndarray, locator: PixelLocator) -> np.ndarray:
    """Evaluate a continuous Lagrange scalar function on every inside pixel.

    Works for any FEALPy Lagrange space exposing ``basis`` and
    ``dof.cell_to_dof`` — not tied to a particular physical model.

    Returns:
        (1, H, W) float32. Outside-Ω pixels are zero.
    """
    HW = locator.H * locator.W
    out = np.zeros(HW, dtype=np.float64)
    cell_to_dof = np.asarray(space.dof.cell_to_dof())
    groups = group_pixels_by_cell(locator)
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
    *,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """𝓘₁: nearest-quadrature-point scatter to structured grid.

    For every grid pixel x_ij take the value of the qp closest in Euclidean
    distance: (𝓘₁ f)(x_ij) = f(argmin_q ‖x_ij − x_q‖). See plan §3.3 and
    docs/operator_learning/m0_interpolation_error.md §2.1. Cost is O((NC·NQ + H·W) log) via
    a KD-tree.

    Args:
        field_qp:    (NC, NQ, *trailing) field on quadrature points.
        quad_coords: (NC, NQ, 2) physical coords of those points.
        grid:        target structured grid.
        mask:        optional (1, H, W) or (H, W) uint8 / bool inside-Ω mask;
                     where 0 the output is zero-filled.
    Returns:
        (*trailing, H, W) float32 sampled field.
    """
    from scipy.spatial import cKDTree

    field_qp = np.asarray(field_qp)
    quad_coords = np.asarray(quad_coords, dtype=np.float64)
    if quad_coords.ndim != 3 or quad_coords.shape[-1] != 2:
        raise ValueError(
            f"quad_coords must be (NC, NQ, 2); got {quad_coords.shape}"
        )
    if field_qp.shape[:2] != quad_coords.shape[:2]:
        raise ValueError(
            f"field_qp[:2]={field_qp.shape[:2]} must match "
            f"quad_coords[:2]={quad_coords.shape[:2]}"
        )

    NC, NQ = quad_coords.shape[:2]
    trailing = field_qp.shape[2:]
    flat_pts = quad_coords.reshape(NC * NQ, 2)
    flat_field = field_qp.reshape(NC * NQ, *trailing)

    H, W = grid.H, grid.W
    grid_pts = grid_points(grid).reshape(H * W, 2)

    tree = cKDTree(flat_pts)
    _, nn = tree.query(grid_pts, k=1)
    sampled = flat_field[nn].reshape(H, W, *trailing)

    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        m_bool = m.astype(bool)
        sampled[~m_bool] = 0

    if trailing:
        axes = tuple(range(2, 2 + len(trailing))) + (0, 1)
        sampled = np.transpose(sampled, axes)
    return np.ascontiguousarray(sampled, dtype=np.float32)


def sample_field_l2_projection(
    field_qp: np.ndarray,
    discr,
    grid: GridSpec,
    *,
    locator: Optional[PixelLocator] = None,
    mask: Optional[np.ndarray] = None,
    quadrature_order: int = 5,
) -> np.ndarray:
    """𝓘₂: L²-project quadrature-point field to ``discr.space_d`` and evaluate on grid.

    Solves (Π_h f, w_h) = (f, w_h) for w_h ∈ space_d, then samples Π_h f on
    the structured grid via Lagrange basis. See plan §3.3 and
    docs/operator_learning/m0_interpolation_error.md §2.2. Cost is dominated by one mass-matrix
    solve (sparse SPD); reuses that mass matrix across trailing channels.

    Args:
        field_qp:    (NC, NQ) or (NC, NQ, C) field on quadrature points.
                     Must be evaluated at the **same quadrature order**
                     used here (``quadrature_order``).
        discr:       built discretization exposing ``mesh`` and ``space_d``
                     (continuous Lagrange).
        grid:        target structured grid.
        locator:     optional precomputed pixel locator; built on demand.
        mask:        optional inside-Ω mask; zero-fill outside.
        quadrature_order: must match the order used to generate ``field_qp``.

    Returns:
        (H, W) for scalar input, else (C, H, W). float32. Outside-Ω is 0.
    """
    from fealpy.fem import BilinearForm, ScalarMassIntegrator
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve

    space = discr.space_d
    if space is None:
        raise RuntimeError("discr.space_d is None; build() the discretization first.")
    mesh = discr.mesh

    field_qp = np.asarray(field_qp)
    if field_qp.ndim == 2:
        scalar = True
        field_qp = field_qp[..., None]   # (NC, NQ, 1)
    elif field_qp.ndim == 3:
        scalar = False
    else:
        raise ValueError(
            f"field_qp must be (NC,NQ) or (NC,NQ,C); got shape {field_qp.shape}"
        )
    NC, NQ, C = field_qp.shape

    bform = BilinearForm(space)
    bform.add_integrator(ScalarMassIntegrator(coef=1.0, q=quadrature_order))
    M = bform.assembly()
    Msp = csr_matrix(
        (np.asarray(M.values), np.asarray(M.col), np.asarray(M.crow)),
        shape=tuple(M.shape),
    )

    qf = mesh.quadrature_formula(quadrature_order)
    bcs, ws = qf.get_quadrature_points_and_weights()
    bcs = np.asarray(bcs)
    ws = np.asarray(ws)
    if ws.shape[0] != NQ:
        raise ValueError(
            f"field_qp has NQ={NQ} but quadrature_formula(q={quadrature_order}) "
            f"yields NQ={ws.shape[0]}; pass matching quadrature_order."
        )
    area = np.asarray(mesh.entity_measure("cell"))
    phi = np.asarray(space.basis(bcs))                            # (1,NQ,ldof) or (NC,NQ,ldof)
    if phi.ndim == 3 and phi.shape[0] == 1:
        phi = np.broadcast_to(phi, (NC, NQ, phi.shape[2]))
    elif phi.ndim != 3 or phi.shape[:2] != (NC, NQ):
        raise ValueError(f"unexpected basis shape: {phi.shape}")
    c2d = np.asarray(space.cell_to_dof())                         # (NC, ldof)
    gdof = space.number_of_global_dofs()

    out_grid = np.zeros((C, grid.H, grid.W), dtype=np.float32)
    if locator is None:
        locator = build_pixel_locator(mesh, grid)
    for ch in range(C):
        rhs_local = area[:, None] * np.einsum(
            "q,cq,cqd->cd", ws, field_qp[..., ch], phi
        )
        rhs = np.zeros(gdof, dtype=np.float64)
        np.add.at(rhs, c2d, rhs_local)
        coef = spsolve(Msp, rhs)
        out_grid[ch] = evaluate_lagrange_on_grid(space, coef, locator)[0]

    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        out_grid[:, ~m.astype(bool)] = 0.0

    return out_grid[0] if scalar else out_grid


# Backwards-compatible private aliases (old module-level names).
_PixelLocator = PixelLocator
_build_pixel_locator = build_pixel_locator
_group_pixels_by_cell = group_pixels_by_cell
_evaluate_lagrange_on_grid = evaluate_lagrange_on_grid
