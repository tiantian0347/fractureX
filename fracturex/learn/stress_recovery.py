"""Displacement-recovered stress σ_h^rec = g(d)·C·ε(u_h) — paper_thesis §F.

The T3.M3b contrast experiment (`paper_thesis.md §F.3 / §G`) trains two
identical surrogates, one supervised on the Hu--Zhang stress σ_h ∈ H(div,S)
and the other on the displacement-recovered stress σ_h^rec = g(d)·C·ε(u_h)
that is *not* in H(div,S). The reconstruction utility here is the offline
step that converts a stored displacement field u_h on the grid into the
σ_h^rec supervision target consumed by :func:`fracturex.learn.datasets.target_stress_rec`.

Grid conventions match the rest of ``fracturex.learn``: shape (..., 2, H, W)
for displacements (u_x, u_y), (..., H, W) for damage, and channel order
(σ_xx, σ_yy, σ_xy) for the produced stress. Row axis (-2) is y (i), column
axis (-1) is x (j), so ε_xx = ∂u_x/∂x is a central difference along axis -1.

Backend policy: this module uses ``fealpy.backend.backend_manager`` so the
same code path runs on numpy for offline preprocessing and on torch/jax for
future GPU / autograd usage — no ``import numpy as np``.
"""
from __future__ import annotations

from fealpy.backend import backend_manager as bm


def plane_strain_C(E: float, nu: float):
    """3×3 plane-strain stiffness matrix in Voigt order (σ_xx, σ_yy, σ_xy)."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return bm.array(
        [
            [lam + 2.0 * mu, lam, 0.0],
            [lam, lam + 2.0 * mu, 0.0],
            [0.0, 0.0, mu],  # engineering shear: σ_xy = μ·γ_xy = 2μ·ε_xy
        ],
        dtype=bm.float64,
    )


def plane_stress_C(E: float, nu: float):
    """3×3 plane-stress stiffness matrix in Voigt order (σ_xx, σ_yy, σ_xy)."""
    fac = E / (1.0 - nu * nu)
    return bm.array(
        [
            [fac, fac * nu, 0.0],
            [fac * nu, fac, 0.0],
            [0.0, 0.0, fac * (1.0 - nu) / 2.0],
        ],
        dtype=bm.float64,
    )


def _asarray(x):
    """Bring an ndarray-like or torch tensor into the current bm backend."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return bm.asarray(x, dtype=bm.float64)


def _central_diff_x(u, dx: float):
    """∂u/∂x via central differences along axis -1; interior only, edges = 0."""
    du = bm.zeros_like(u)
    du[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / (2.0 * dx)
    return du


def _central_diff_y(u, dy: float):
    """∂u/∂y via central differences along axis -2; interior only, edges = 0."""
    du = bm.zeros_like(u)
    du[..., 1:-1, :] = (u[..., 2:, :] - u[..., :-2, :]) / (2.0 * dy)
    return du


def strain_from_displacement(u_grid, dx: float, dy: float):
    """Small-strain ε(u_h) on the grid via central differences.

    Parameters
    ----------
    u_grid
        Array of shape (..., 2, H, W). Channels are (u_x, u_y).
    dx, dy
        Grid spacings along axis -1 (x) and axis -2 (y).

    Returns
    -------
    Strain array of shape (..., 3, H, W) in Voigt engineering order
    (ε_xx, ε_yy, γ_xy) where γ_xy = 2·ε_xy = ∂u_x/∂y + ∂u_y/∂x. This is
    the form that pairs with the C matrices returned by :func:`plane_strain_C`
    and :func:`plane_stress_C` to give σ_xy directly (rather than 2·σ_xy).
    """
    u = _asarray(u_grid)
    if u.ndim < 3 or u.shape[-3] != 2:
        raise ValueError(f"u_grid must have shape (..., 2, H, W); got {tuple(u.shape)}")
    ux = u[..., 0, :, :]
    uy = u[..., 1, :, :]
    eps_xx = _central_diff_x(ux, dx)
    eps_yy = _central_diff_y(uy, dy)
    gamma_xy = _central_diff_y(ux, dy) + _central_diff_x(uy, dx)
    return bm.stack([eps_xx, eps_yy, gamma_xy], axis=-3)


def stress_recovered_from_displacement(
    u_grid,
    d_grid,
    C,
    dx: float,
    dy: float,
    kres: float = 1.0e-6,
    stress_scale: float = 1.0,
):
    """σ_h^rec = g(d)·C·ε(u_h) on the grid, normalized by ``stress_scale``.

    Uses the AT2 quadratic degradation g(d) = (1 − k_res)(1 − d)² + k_res
    (paper_thesis §A; equilibrated_aposteriori.tex eq. (2)). The output is
    scaled by 1/``stress_scale`` to match the on-disk convention of
    :func:`fracturex.learn.datasets.target_stress` (schema §3.2: stress is
    stored per-sample-normalised), so the recovered target lives in the same
    O(1) training space as σ_h.

    Parameters
    ----------
    u_grid
        (..., 2, H, W) displacement in physical units.
    d_grid
        (..., H, W) damage in [0, 1].
    C
        (3, 3) plane-strain or plane-stress stiffness matrix in Voigt order.
    dx, dy
        Grid spacings.
    kres
        Residual stiffness floor.
    stress_scale
        Per-sample normalisation used in the dataset schema.
    """
    strain = strain_from_displacement(u_grid, dx, dy)           # (..., 3, H, W)
    C_arr = _asarray(C)
    if C_arr.shape != (3, 3):
        raise ValueError(f"C must be (3, 3); got {tuple(C_arr.shape)}")
    # Contract along the strain-channel axis: σ_voigt = C · ε_voigt.
    sigma = bm.einsum("ij,...jhw->...ihw", C_arr, strain)
    d = _asarray(d_grid)
    # Broadcast damage to the (..., 1, H, W) channel slot.
    d_expand = d[..., None, :, :]
    g = (1.0 - kres) * (1.0 - d_expand) ** 2 + kres
    sigma = sigma * g
    if stress_scale != 1.0:
        sigma = sigma / float(stress_scale)
    return bm.astype(sigma, bm.float32)
