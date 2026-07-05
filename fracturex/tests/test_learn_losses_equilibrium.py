"""Tests for the differentiable Stage-D equilibrium residual loss.

Mirrors the numpy metric tests (`test_learn_metrics_equilibrium`) with the
addition of an autograd check — the training loss must be differentiable in
sigma_pred.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fracturex.learn.losses import equilibrium_residual_fd


def _linear_sxx(H=16, W=16):
    xs = torch.linspace(0.0, 1.0, W)
    ys = torch.linspace(0.0, 1.0, H)
    # indexing="ij" -> first output is ys[i], second is xs[j]; we want X = xs[j].
    _Y, X = torch.meshgrid(ys, xs, indexing="ij")
    sigma = torch.zeros(1, 3, H, W)
    sigma[0, 0] = X
    dx = 1.0 / (W - 1)
    dy = 1.0 / (H - 1)
    return sigma, dx, dy


def test_zero_stress_returns_zero():
    H = W = 8
    sigma = torch.zeros(2, 3, H, W)
    mask = torch.ones(2, 1, H, W)
    r = equilibrium_residual_fd(sigma, mask, dx=1.0, dy=1.0)
    assert float(r) < 1e-10


def test_constant_stress_is_divergence_free():
    H = W = 8
    sigma = torch.zeros(1, 3, H, W)
    sigma[0, 0] = 1.0
    sigma[0, 1] = -0.5
    sigma[0, 2] = 0.25
    mask = torch.ones(1, 1, H, W)
    r = equilibrium_residual_fd(sigma, mask, dx=1.0, dy=1.0)
    assert float(r) < 1e-6


def test_linear_sxx_matches_expected_magnitude():
    """For σ_xx = x, div σ = (1, 0), so R_h ≈ √|Ω_h^∘| ≈ √((1-2dx)(1-2dy)) ≈ 0.87 on 16×16."""
    sigma, dx, dy = _linear_sxx(16, 16)
    mask = torch.ones(1, 1, 16, 16)
    r = float(equilibrium_residual_fd(sigma, mask, dx=dx, dy=dy))
    assert 0.7 < r < 1.0


def test_body_force_cancels_divergence():
    sigma, dx, dy = _linear_sxx(16, 16)
    mask = torch.ones(1, 1, 16, 16)
    f = torch.zeros(1, 2, 16, 16)
    f[0, 0] = -1.0
    r = float(equilibrium_residual_fd(sigma, mask, dx=dx, dy=dy, body_force=f))
    assert r < 1e-5


def test_batched_time_axis():
    H = W = 8
    sigma = torch.zeros(2, 3, 3, H, W)
    mask = torch.ones(2, 1, H, W)
    r = equilibrium_residual_fd(sigma, mask, dx=1.0, dy=1.0)
    assert float(r) < 1e-10


def test_autograd_flows_to_sigma():
    """A loss term must produce non-zero gradients w.r.t. sigma_pred."""
    sigma, dx, dy = _linear_sxx(16, 16)
    sigma = sigma.detach().requires_grad_(True)
    mask = torch.ones(1, 1, 16, 16)
    r = equilibrium_residual_fd(sigma, mask, dx=dx, dy=dy)
    r.backward()
    assert sigma.grad is not None
    # Non-boundary interior cells should carry gradient.
    assert float(sigma.grad.abs().sum()) > 0.0


def test_damage_exclusion_masks_out_broken_cells():
    sigma, dx, dy = _linear_sxx(32, 32)
    H, W = sigma.shape[-2:]
    torch.manual_seed(0)
    sigma[0, 0, H // 2 :, :] = sigma[0, 0, H // 2 :, :] + torch.randn(H - H // 2, W)
    d = torch.zeros(1, 1, H, W)
    d[0, 0, H // 2 :, :] = 1.0
    mask = torch.ones(1, 1, H, W)

    r_noisy = float(equilibrium_residual_fd(sigma, mask, dx=dx, dy=dy))
    r_clean = float(equilibrium_residual_fd(sigma, mask, dx=dx, dy=dy, d=d, d_c=0.9))
    assert r_clean < 0.3 * r_noisy


def test_shape_validation():
    mask = torch.ones(1, 1, 8, 8)
    with pytest.raises(ValueError):
        equilibrium_residual_fd(torch.zeros(1, 5, 8, 8), mask, dx=1.0, dy=1.0)


def test_consistency_with_numpy_metric():
    """Torch R_h and numpy R_h (unnormalized) should agree bit-close on the
    same inputs.  We compare the absolute norm before the R̃_h normalization:
    per-sample per_res of the numpy metric equals the returned value of the
    torch loss (on batch-size 1).
    """
    from fracturex.learn.eval.metrics import equilibrium_residual_l2

    sigma, dx, dy = _linear_sxx(16, 16)
    mask = torch.ones(1, 1, 16, 16)
    r_torch = float(equilibrium_residual_fd(sigma, mask, dx=dx, dy=dy))

    # Recover the same absolute R_h from the numpy scale-free metric R̃_h by
    # multiplying out the (L / ‖σ‖) factor.
    sigma_np = sigma[0].numpy()
    mask_np = mask[0, 0].numpy().astype(bool)
    r_tilde = equilibrium_residual_l2(sigma_np, mask_np, dx, dy)
    # Rebuild denominator on the same interior mask used inside the numpy fn.
    H, W = sigma_np.shape[-2:]
    m_int = np.zeros_like(mask_np)
    m_int[1:-1, 1:-1] = (
        mask_np[1:-1, 1:-1] & mask_np[2:, 1:-1] & mask_np[:-2, 1:-1]
        & mask_np[1:-1, 2:] & mask_np[1:-1, :-2]
    )
    sig_sq = ((sigma_np[0] ** 2 + sigma_np[1] ** 2 + sigma_np[2] ** 2) * m_int).sum() * dx * dy
    per_sig = float(np.sqrt(sig_sq))
    L = float(np.sqrt(2.0) * 16.0 / 15.0)   # bbox diagonal for full-mask on unit square
    per_res_np = r_tilde * (per_sig + 1e-12) / L
    assert abs(r_torch - per_res_np) < 5e-6
