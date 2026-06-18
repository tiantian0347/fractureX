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

def equilibrium_residual_fd(sigma_pred, body_force, mask):
    """L_eq^FD (eq. 3.15a): ‖m ⊙ (∇_h · σ̂ + f)‖² with central differences."""
    raise NotImplementedError("M2 Stage D: central-diff divergence on grid")


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
