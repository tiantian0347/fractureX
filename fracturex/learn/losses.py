# fracturex/learn/losses.py
"""Masked loss functions for the operator-learning surrogate.

Equations and stage table: docs/operator_learning/plan_operator_learning.md §3.5.
All losses are mask-weighted; out-of-Ω points are NOT supervised.

Tensors follow ``(B, T, H, W)`` (damage) or ``(B, T, C, H, W)`` (stress);
``mask`` is ``(B, 1, H, W)`` and broadcasts over the time / channel axes.
PyTorch is required (training-only module).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

# Default stage weights (plan §3.5; λ_irr=1 fixed there, others are sane starts).
_DEFAULT_LAMBDAS = {
    "lambda_sigma": 1.0,
    "lambda_h1": 0.1,
    "lambda_front": 1.0,
    "lambda_irr": 1.0,
}


def _align_mask(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast a (B,1,H,W) mask to ``ref``'s shape (B,T,H,W) or (B,T,C,H,W)."""
    m = mask
    while m.dim() < ref.dim():
        m = m.unsqueeze(1)            # insert axes after batch until ranks match
    return m.expand_as(ref)


# ---------------------------------------------------------------------------
# Data-fidelity losses (Stage A / B)
# ---------------------------------------------------------------------------

def masked_relative_l2(pred, target, mask, eps: float = 1e-8):
    """L_d / L_σ (eq. 3.13): relative L² with mask, resolution-invariant."""
    m = _align_mask(mask, target)
    diff = (pred - target) * m
    denom = torch.linalg.vector_norm((target * m).flatten(1), dim=1) + eps
    num = torch.linalg.vector_norm(diff.flatten(1), dim=1)
    return (num / denom).mean()


def _central_grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Central differences along the last two (H, W) axes; edges via one-sided."""
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[..., 1:-1, :] = (x[..., 2:, :] - x[..., :-2, :]) * 0.5
    gy[..., :, 1:-1] = (x[..., :, 2:] - x[..., :, :-2]) * 0.5
    return gx, gy


def masked_relative_h1(pred, target, mask, eps: float = 1e-8):
    """L_{H¹} (eq. 3.14a): masked Sobolev (value + gradient) relative error."""
    m = _align_mask(mask, target)
    px, py = _central_grad(pred)
    tx, ty = _central_grad(target)
    num_val = ((pred - target) * m).flatten(1)
    num_gx = ((px - tx) * m).flatten(1)
    num_gy = ((py - ty) * m).flatten(1)
    num = torch.sqrt(
        num_val.pow(2).sum(1) + num_gx.pow(2).sum(1) + num_gy.pow(2).sum(1)
    )
    den_val = (target * m).flatten(1)
    den = torch.sqrt(
        den_val.pow(2).sum(1) + (tx * m).flatten(1).pow(2).sum(1)
        + (ty * m).flatten(1).pow(2).sum(1)
    ) + eps
    return (num / den).mean()


def peak_weighted_relative_l2(pred, target, mask, alpha: float = 4.0, eps: float = 1e-8):
    """Peak-aware σ loss: up-weight high-stress (crack-tip) pixels in rel-L².

    A plain relative-L² lets a smooth network regress the singular crack-tip
    peak toward the mean (under-prediction; see m2_stageB_results.md §3.2/§6).
    Here each pixel is weighted ``1 + α·(‖σ_tgt‖ / peak)`` where ‖σ_tgt‖ is the
    per-pixel stress magnitude (Frobenius over the 3 channels) and ``peak`` is
    the per-sample max — so tip pixels get up to (1+α)× the gradient pull.

    pred/target: (B, T, 3, H, W); mask: (B, 1, H, W).
    """
    m = _align_mask(mask, target)                                  # (B,T,3,H,W)
    smag = torch.sqrt((target ** 2).sum(dim=2, keepdim=True) + eps)  # (B,T,1,H,W)
    B = smag.shape[0]
    peak = smag.reshape(B, -1).amax(dim=1).reshape(B, 1, 1, 1, 1) + eps
    w = m * (1.0 + alpha * (smag / peak))                          # broadcast over channel
    num = torch.sqrt((w * (pred - target) ** 2).flatten(1).sum(1))
    den = torch.sqrt((w * target ** 2).flatten(1).sum(1)) + eps
    return (num / den).mean()


def front_weighted_l2(pred, target, mask, alpha: float = 1.0, eps: float = 1e-8):
    """L_front (eq. 3.14b): emphasize the [0.1, 0.9] transition band of d.

    Weight w(d) = 1 + α·𝟙_{[0.1, 0.9]}(target).
    """
    m = _align_mask(mask, target)
    band = ((target >= 0.1) & (target <= 0.9)).to(target.dtype)
    w = (1.0 + alpha * band) * m
    num = torch.linalg.vector_norm(((pred - target) * w).flatten(1), dim=1)
    denom = torch.linalg.vector_norm((target * w).flatten(1), dim=1) + eps
    return (num / denom).mean()


# ---------------------------------------------------------------------------
# Physics-consistency losses (Stage D) — deferred to M2
# ---------------------------------------------------------------------------

def equilibrium_residual_fd(
    sigma_pred,
    mask,
    dx: float = 1.0,
    dy: float = 1.0,
    body_force=None,
    d=None,
    d_c: float = 0.9,
):
    """Differentiable grid equilibrium residual R_h(σ̂) — paper_thesis §C.

    Training-side companion to :func:`fracturex.learn.eval.metrics.equilibrium_residual_l2`.
    The Stage D composite loss L_D = E² + λ R_h² (paper_thesis §G.2)
    should square the value returned here; this function returns R_h itself,
    consistent with the other loss primitives in this module.

    Central-difference divergence of the stress on a Cartesian grid restricted to
    the interior mask Ω_h^∘ = {(i,j): m=1 and all four central-difference
    neighbours in m and (if ``d`` is given) d ≤ d_c}. Channel axis (-3) order is
    (σ_xx, σ_yy, σ_xy); row axis (-2) is y (i), column axis (-1) is x (j).

        r_x[i,j] = (σ_xx[i,j+1] − σ_xx[i,j−1])/(2Δx)
                   + (σ_xy[i+1,j] − σ_xy[i−1,j])/(2Δy) + f_x
        r_y[i,j] = (σ_xy[i,j+1] − σ_xy[i,j−1])/(2Δx)
                   + (σ_yy[i+1,j] − σ_yy[i−1,j])/(2Δy) + f_y

    Returns the mean over the batch axis of the per-sample absolute L²(Ω_h^∘)
    norm of (r_x, r_y). Fully autograd-compatible in ``sigma_pred``.

    Parameters
    ----------
    sigma_pred
        Tensor of shape (B, T, 3, H, W) or (B, 3, H, W). Channel order
        (σ_xx, σ_yy, σ_xy).
    mask
        Mask indicating Ω on the grid; must broadcast to
        ``sigma_pred[..., 0, :, :]`` (typically ``(B, 1, H, W)`` like other
        losses in this module).
    dx, dy
        Grid spacings along axis -1 (x) and axis -2 (y).
    body_force
        Optional body force. ``None`` or ``0`` for the canonical fracture setting;
        an array of shape ``(B, T, 2, H, W)`` (or broadcastable) with channel
        order (f_x, f_y).
    d
        Optional damage field with the same leading shape as ``sigma_pred[..., 0, :, :]``;
        cells with ``d > d_c`` are removed from Ω_h^∘.
    d_c
        Damage cutoff excluded from the residual (paper_thesis §C).
    """
    if sigma_pred.dim() < 3 or sigma_pred.shape[-3] != 3:
        raise ValueError(
            f"sigma_pred must have shape (..., 3, H, W); got {tuple(sigma_pred.shape)}"
        )
    sxx = sigma_pred[..., 0, :, :]
    syy = sigma_pred[..., 1, :, :]
    sxy = sigma_pred[..., 2, :, :]

    m_bool = mask.to(torch.bool)
    while m_bool.dim() > sxx.dim():
        if m_bool.shape[1] != 1:
            raise ValueError(
                f"mask has non-singleton extra dim vs sigma spatial slice: mask {tuple(m_bool.shape)}, sigma slice {tuple(sxx.shape)}"
            )
        m_bool = m_bool.squeeze(1)
    while m_bool.dim() < sxx.dim():
        m_bool = m_bool.unsqueeze(1)
    m_bool = m_bool.expand_as(sxx)
    m_int = torch.zeros_like(m_bool)
    inner = (
        m_bool[..., 1:-1, 1:-1]
        & m_bool[..., 2:, 1:-1]
        & m_bool[..., :-2, 1:-1]
        & m_bool[..., 1:-1, 2:]
        & m_bool[..., 1:-1, :-2]
    )
    m_int[..., 1:-1, 1:-1] = inner
    if d is not None:
        d_t = d
        while d_t.dim() > sxx.dim():
            if d_t.shape[1] != 1:
                raise ValueError(
                    f"d has non-singleton extra dim vs sigma spatial slice: d {tuple(d_t.shape)}"
                )
            d_t = d_t.squeeze(1)
        while d_t.dim() < sxx.dim():
            d_t = d_t.unsqueeze(1)
        d_t = d_t.expand_as(sxx)
        m_int = m_int & (d_t <= d_c)
    m_int_f = m_int.to(sxx.dtype)

    d_sxx_dx = torch.zeros_like(sxx)
    d_sxy_dy = torch.zeros_like(sxx)
    d_sxy_dx = torch.zeros_like(sxx)
    d_syy_dy = torch.zeros_like(sxx)
    d_sxx_dx[..., 1:-1, 1:-1] = (sxx[..., 1:-1, 2:] - sxx[..., 1:-1, :-2]) / (2.0 * dx)
    d_sxy_dy[..., 1:-1, 1:-1] = (sxy[..., 2:, 1:-1] - sxy[..., :-2, 1:-1]) / (2.0 * dy)
    d_sxy_dx[..., 1:-1, 1:-1] = (sxy[..., 1:-1, 2:] - sxy[..., 1:-1, :-2]) / (2.0 * dx)
    d_syy_dy[..., 1:-1, 1:-1] = (syy[..., 2:, 1:-1] - syy[..., :-2, 1:-1]) / (2.0 * dy)

    r_x = d_sxx_dx + d_sxy_dy
    r_y = d_sxy_dx + d_syy_dy
    if body_force is not None:
        if body_force.shape[-3] != 2:
            raise ValueError(
                f"body_force must have shape (..., 2, H, W); got {tuple(body_force.shape)}"
            )
        r_x = r_x + body_force[..., 0, :, :]
        r_y = r_y + body_force[..., 1, :, :]

    cell_area = dx * dy
    res_sq = (r_x * r_x + r_y * r_y) * m_int_f            # (..., H, W)
    per_sample = torch.sqrt(res_sq.flatten(1).sum(dim=1) * cell_area + 1e-30)
    return per_sample.mean()


def equilibrium_residual_weak(sigma_pred, test_functions, traction, body_force, mask):
    """L_eq^weak (eq. 3.15b): residual against a fixed test-function basis."""
    raise NotImplementedError("M2 Stage D: assemble ∫ σ:∇^s φ dx − ∫ t·φ ds − ∫ f·φ dx")


def phase_field_residual(d_pred, history_field, mask, *, Gc: float, l0: float):
    """L_pf (eq. 3.16): phase-field PDE residual on the grid."""
    raise NotImplementedError("M2 Stage D: (G_c/l_0)d − G_c l_0 Δ_h d − 2(1-d)𝓗")


# ---------------------------------------------------------------------------
# Irreversibility (Stage E) — soft variant; hard variant lives in models/heads
# ---------------------------------------------------------------------------

def irreversibility_penalty(d_seq, mask):
    """L_irr (eq. 3.17a): mean ReLU(d_{n-1} − d_n)² over masked points and time."""
    if d_seq.shape[1] < 2:
        return d_seq.new_zeros(())
    m = _align_mask(mask, d_seq)
    backslide = F.relu(d_seq[:, :-1] - d_seq[:, 1:]) * m[:, 1:]
    return backslide.pow(2).mean()


# ---------------------------------------------------------------------------
# Stage scheduler
# ---------------------------------------------------------------------------

def stage_loss(stage: str, **components):
    """Combine the stage-table losses (plan §3.5):

       A: L_d
       B: L_d + λ_σ L_σ
       C: B + λ_{H¹} L_{H¹} + λ_front L_front
       D/E: deferred to M2 (loss terms above raise NotImplementedError).

    ``components`` provides the computed terms (``l_d`` required; ``l_sigma``,
    ``l_h1``, ``l_front`` as needed) and optional lambda overrides
    (``lambda_sigma`` etc.); missing lambdas fall back to plan defaults.
    """
    lam = {**_DEFAULT_LAMBDAS}
    for k in list(_DEFAULT_LAMBDAS):
        if components.get(k) is not None:
            lam[k] = float(components[k])

    stage = stage.upper()
    if "l_d" not in components:
        raise KeyError("stage_loss requires component 'l_d'")
    total = components["l_d"]
    if stage == "A":
        return total
    if stage in ("B", "C", "D", "E"):
        total = total + lam["lambda_sigma"] * components["l_sigma"]
    if stage in ("C", "D", "E"):
        total = total + lam["lambda_h1"] * components["l_h1"]
        total = total + lam["lambda_front"] * components["l_front"]
    if stage in ("D", "E"):
        raise NotImplementedError("Stage D/E physics + irreversibility land in M2")
    if stage not in ("A", "B", "C"):
        raise ValueError(f"unknown stage {stage!r}")
    return total
