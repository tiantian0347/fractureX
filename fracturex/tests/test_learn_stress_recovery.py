"""Tests for σ_h^rec displacement-recovery utility.

Backend-agnostic via ``fealpy.backend.backend_manager`` — no ``import numpy``.
"""
from __future__ import annotations

import pytest
from fealpy.backend import backend_manager as bm

from fracturex.learn.stress_recovery import (
    plane_strain_C,
    plane_stress_C,
    strain_from_displacement,
    stress_recovered_from_displacement,
)


def _linear_ux_uy_grid(H=32, W=32):
    """u = (x, y) so ε_xx = ε_yy = 1, γ_xy = 0."""
    xs = bm.linspace(0.0, 1.0, W)
    ys = bm.linspace(0.0, 1.0, H)
    X, Y = bm.meshgrid(xs, ys)                          # indexing='xy'
    u = bm.zeros((2, H, W), dtype=bm.float64)
    u[0] = X
    u[1] = Y
    return u, 1.0 / (W - 1), 1.0 / (H - 1)


def test_plane_stress_C_symmetric_and_pd():
    C = plane_stress_C(E=210.0, nu=0.3)
    assert bm.allclose(C, C.T)
    # smallest eigenvalue > 0 (backend-neutral via numpy view for eig)
    import numpy as _np
    eigs = _np.linalg.eigvalsh(_np.asarray(C))
    assert eigs.min() > 0.0


def test_plane_strain_C_matches_lame():
    E, nu = 210.0, 0.3
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = plane_strain_C(E, nu)
    assert abs(float(C[0, 0]) - (lam + 2 * mu)) < 1e-10
    assert abs(float(C[0, 1]) - lam) < 1e-10
    assert abs(float(C[2, 2]) - mu) < 1e-10


def test_strain_of_uniform_stretch_is_identity():
    u, dx, dy = _linear_ux_uy_grid(32, 32)
    eps = strain_from_displacement(u, dx, dy)         # (3, H, W)
    eps_int = eps[..., 1:-1, 1:-1]
    assert bm.allclose(eps_int[0], bm.ones_like(eps_int[0]), atol=1e-10)   # ε_xx = 1
    assert bm.allclose(eps_int[1], bm.ones_like(eps_int[1]), atol=1e-10)   # ε_yy = 1
    assert bm.allclose(eps_int[2], bm.zeros_like(eps_int[2]), atol=1e-10)  # γ_xy = 0


def test_strain_of_pure_shear():
    """u_x = y, u_y = 0 → ε_xx = ε_yy = 0, γ_xy = 1."""
    H = W = 32
    xs = bm.linspace(0.0, 1.0, W)
    ys = bm.linspace(0.0, 1.0, H)
    _X, Y = bm.meshgrid(xs, ys)
    u = bm.zeros((2, H, W), dtype=bm.float64)
    u[0] = Y
    eps = strain_from_displacement(u, 1.0 / (W - 1), 1.0 / (H - 1))
    eps_int = eps[..., 1:-1, 1:-1]
    assert bm.allclose(eps_int[0], bm.zeros_like(eps_int[0]), atol=1e-10)
    assert bm.allclose(eps_int[1], bm.zeros_like(eps_int[1]), atol=1e-10)
    assert bm.allclose(eps_int[2], bm.ones_like(eps_int[2]), atol=1e-10)


def test_recovered_stress_shape():
    H = W = 16
    u = bm.zeros((5, 2, H, W), dtype=bm.float64)
    d = bm.zeros((5, H, W), dtype=bm.float64)
    C = plane_stress_C(210.0, 0.3)
    sigma = stress_recovered_from_displacement(u, d, C, 1.0, 1.0)
    assert tuple(sigma.shape) == (5, 3, H, W)
    # Zero displacement → zero recovered stress.
    assert float(bm.max(bm.abs(sigma))) == 0.0


def test_recovered_stress_uniform_stretch():
    """u = (x, y) plane-strain: C·ε in Voigt = (λ+2μ+λ, λ+λ+2μ, 0) = 2·(λ+μ, λ+μ, 0)."""
    u, dx, dy = _linear_ux_uy_grid(16, 16)
    d = bm.zeros((16, 16), dtype=bm.float64)
    E, nu = 210.0, 0.3
    C = plane_strain_C(E, nu)
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    expected_sxx = 2.0 * (lam + mu)
    sigma = stress_recovered_from_displacement(u, d, C, dx, dy)
    interior = sigma[..., 1:-1, 1:-1]
    ones = bm.ones_like(interior[0])
    assert bm.allclose(interior[0], expected_sxx * ones, rtol=1e-4)   # σ_xx
    assert bm.allclose(interior[1], expected_sxx * ones, rtol=1e-4)   # σ_yy
    assert bm.allclose(interior[2], bm.zeros_like(interior[2]), atol=1e-4)


def test_damage_degradation_scaling():
    """d=1 (broken) should drop σ_rec to kres × the intact value."""
    u, dx, dy = _linear_ux_uy_grid(16, 16)
    C = plane_stress_C(210.0, 0.3)
    d_zero = bm.zeros((16, 16), dtype=bm.float64)
    d_full = bm.ones((16, 16), dtype=bm.float64)
    kres = 1e-6
    s_intact = stress_recovered_from_displacement(u, d_zero, C, dx, dy, kres=kres)
    s_broken = stress_recovered_from_displacement(u, d_full, C, dx, dy, kres=kres)
    # Compare on the σ_xx channel where the intact value is O(1) after normalization.
    ratio = float(s_broken[0, 8, 8]) / (float(s_intact[0, 8, 8]) + 1e-30)
    assert abs(ratio - kres) < 1e-3 * kres or abs(ratio - kres) < 1e-9


def test_stress_scale_normalization():
    u, dx, dy = _linear_ux_uy_grid(16, 16)
    d = bm.zeros((16, 16), dtype=bm.float64)
    C = plane_stress_C(210.0, 0.3)
    s_unit = stress_recovered_from_displacement(u, d, C, dx, dy, stress_scale=1.0)
    s_scaled = stress_recovered_from_displacement(u, d, C, dx, dy, stress_scale=100.0)
    assert bm.allclose(s_scaled, s_unit / 100.0, rtol=1e-4)


def test_shape_validation_u():
    with pytest.raises(ValueError):
        strain_from_displacement(bm.zeros((3, 16, 16), dtype=bm.float64), 1.0, 1.0)


def test_shape_validation_C():
    u = bm.zeros((2, 8, 8), dtype=bm.float64)
    d = bm.zeros((8, 8), dtype=bm.float64)
    with pytest.raises(ValueError):
        stress_recovered_from_displacement(u, d, bm.eye(4), 1.0, 1.0)
