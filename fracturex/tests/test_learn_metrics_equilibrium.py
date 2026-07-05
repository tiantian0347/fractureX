"""Tests for equilibrium_residual_l2 (paper_thesis §C, R̃_h).

Verifies the discrete central-difference divergence, mask semantics,
damage exclusion, batch handling, and shape validation.
"""
from __future__ import annotations

import numpy as np
import pytest

from fracturex.learn.eval.metrics import equilibrium_residual_l2


def _linear_sxx_grid(H=32, W=32):
    """σ_xx(x,y) = x, σ_yy = σ_xy = 0. div σ = (1, 0)."""
    xs = np.linspace(0.0, 1.0, W)
    ys = np.linspace(0.0, 1.0, H)
    X, _Y = np.meshgrid(xs, ys)
    sigma = np.zeros((3, H, W))
    sigma[0] = X
    return sigma, 1.0 / (W - 1), 1.0 / (H - 1)


def test_zero_stress_returns_zero():
    H = W = 16
    sigma = np.zeros((3, H, W))
    mask = np.ones((H, W), dtype=bool)
    r = equilibrium_residual_l2(sigma, mask, 1.0 / (W - 1), 1.0 / (H - 1))
    assert r == 0.0


def test_constant_stress_is_divergence_free():
    H = W = 16
    sigma = np.zeros((3, H, W))
    sigma[0] = 1.0
    sigma[1] = 2.0
    sigma[2] = 0.5
    mask = np.ones((H, W), dtype=bool)
    r = equilibrium_residual_l2(sigma, mask, 1.0 / (W - 1), 1.0 / (H - 1))
    assert r < 1e-12


def test_linear_sxx_matches_analytic():
    """For σ_xx = x on [0,1]², div σ = (1, 0), so
    ‖(r_x, r_y)‖_L2 ≈ √|Ω| = 1 and ‖σ‖_L2 ≈ √(1/3).
    L ≈ √2 (bbox diagonal of unit square). Analytic R̃_h ≈ √2 / √(1/3) ≈ 2.449.
    On a 32×32 grid the boundary trim shifts this to ≈2.55; we allow a 5% band.
    """
    sigma, dx, dy = _linear_sxx_grid(32, 32)
    mask = np.ones(sigma.shape[-2:], dtype=bool)
    r = equilibrium_residual_l2(sigma, mask, dx, dy)
    assert 2.3 < r < 2.7


def test_body_force_cancels_divergence():
    sigma, dx, dy = _linear_sxx_grid(32, 32)
    mask = np.ones(sigma.shape[-2:], dtype=bool)
    fx = -np.ones(sigma.shape[-2:])
    fy = np.zeros(sigma.shape[-2:])
    f = np.stack([fx, fy])
    r = equilibrium_residual_l2(sigma, mask, dx, dy, f=f)
    assert r < 1e-10


def test_batched_leading_axes():
    H = W = 16
    B, T = 2, 3
    sigma_b = np.zeros((B, T, 3, H, W))
    mask_b = np.ones((B, T, H, W), dtype=bool)
    r = equilibrium_residual_l2(sigma_b, mask_b, 1.0 / (W - 1), 1.0 / (H - 1))
    assert r == 0.0


def test_damage_exclusion_reduces_noise_contribution():
    """A noisy stress patch in the fully damaged half should inflate R̃_h; the
    d>d_c cell exclusion must drop it back to the smooth-region baseline."""
    sigma, dx, dy = _linear_sxx_grid(32, 32)
    H, W = sigma.shape[-2:]
    rng = np.random.default_rng(1)
    sigma[0, H // 2 :, :] += rng.standard_normal((H - H // 2, W))
    d = np.zeros((H, W))
    d[H // 2 :, :] = 1.0
    mask = np.ones((H, W), dtype=bool)

    r_noisy = equilibrium_residual_l2(sigma, mask, dx, dy)
    r_clean = equilibrium_residual_l2(sigma, mask, dx, dy, d=d, d_c=0.9)
    assert r_clean < 0.2 * r_noisy


def test_shape_validation():
    mask = np.ones((16, 16), dtype=bool)
    with pytest.raises(ValueError):
        equilibrium_residual_l2(np.zeros((5, 16, 16)), mask, 1.0, 1.0)


def test_explicit_L_overrides_bbox():
    sigma, dx, dy = _linear_sxx_grid(32, 32)
    mask = np.ones(sigma.shape[-2:], dtype=bool)
    r_auto = equilibrium_residual_l2(sigma, mask, dx, dy)
    r_L2 = equilibrium_residual_l2(sigma, mask, dx, dy, L=2.0)
    # Auto-L ≈ √2 · H/(H-1); setting L=2 should scale R̃_h by 2/(√2·H/(H-1)).
    factor = 2.0 / (np.sqrt(2.0) * 32.0 / 31.0)
    assert abs(r_L2 / r_auto - factor) < 1e-6
