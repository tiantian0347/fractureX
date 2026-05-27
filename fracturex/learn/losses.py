# fracturex/learn/losses.py
"""Masked loss functions for the operator-learning surrogate.

Equations and stage table: docs/plan_operator_learning.md §3.5.
All losses MUST be mask-weighted; out-of-Ω points are NOT supervised.
"""
from __future__ import annotations
from typing import Optional


# ---------------------------------------------------------------------------
# Data-fidelity losses (Stage A / B)
# ---------------------------------------------------------------------------

def masked_relative_l2(pred, target, mask, eps: float = 1e-8):
    """L_d / L_σ (eq. 3.13): relative L^2 with mask, resolution-invariant.

    Args:
        pred, target: (B, ..., C, H, W).
        mask: (B, 1, H, W) broadcastable to channels and time.
    """
    raise NotImplementedError("M1 task: ||m⊙(pred-target)||_2 / (||m⊙target||_2 + eps)")


def masked_relative_h1(pred, target, mask, eps: float = 1e-8):
    """L_{H^1} (eq. 3.14a): masked Sobolev relative error.

    Suppresses front-smearing; central differences inside the domain.
    """
    raise NotImplementedError("M2 Stage C: finite-difference gradient + masked L2")


def front_weighted_l2(pred, target, mask, alpha: float = 1.0):
    """L_front (eq. 3.14b): emphasize the [0.1, 0.9] transition band of d.

    Weight w(d) = 1 + α · 𝟙_{[0.1, 0.9]}(target) by default.
    """
    raise NotImplementedError("M2 Stage C: front mask × masked L^2")


# ---------------------------------------------------------------------------
# Physics-consistency losses (Stage D)
# ---------------------------------------------------------------------------

def equilibrium_residual_fd(sigma_pred, body_force, mask):
    """L_eq^FD (eq. 3.15a): ‖m ⊙ (∇_h · σ̂ + f)‖_2^2 with central differences.

    sigma_pred: (B, T, 3, H, W) in (xx, yy, xy) channel order.
    body_force: same shape or broadcastable.
    """
    raise NotImplementedError("M2 Stage D: central-diff divergence on grid")


def equilibrium_residual_weak(sigma_pred, test_functions, traction, body_force, mask):
    """L_eq^weak (eq. 3.15b): residual against a fixed test-function basis.

    More robust to masked / irregular domains than FD; M2 ablation control.
    """
    raise NotImplementedError("M2 Stage D: assemble ∫ σ:∇^s φ dx − ∫ t·φ ds − ∫ f·φ dx")


def phase_field_residual(d_pred, history_field, mask, *, Gc: float, l0: float):
    """L_pf (eq. 3.16): phase-field PDE residual on the grid.

    history_field is the supervision-side 𝓗̃ (option (a) in plan §3.5)
    or the predicted 𝓗̂ (option (b), 5-channel network output).
    """
    raise NotImplementedError("M2 Stage D: (G_c/l_0)d − G_c l_0 Δ_h d − 2(1-d)𝓗")


# ---------------------------------------------------------------------------
# Irreversibility (Stage E) — soft variant; hard variant lives in models/heads
# ---------------------------------------------------------------------------

def irreversibility_penalty(d_seq, mask):
    """L_irr (eq. 3.17a): ReLU(d_{n-1} - d_n) penalty over time.

    Use this ONLY when the network does not already use a monotone head
    (see models/heads/monotone.py for the architectural alternative).
    """
    raise NotImplementedError("M2 Stage E: time-wise positive part penalty")


# ---------------------------------------------------------------------------
# Stage scheduler
# ---------------------------------------------------------------------------

def stage_loss(stage: str, **components):
    """Combine the stage-table losses (plan §3.5):
       A: L_d
       B: L_d + λ_σ L_σ
       C: B + λ_{H^1} L_{H^1} + λ_front L_front
       D: C + λ_eq L_eq^{FD or weak}
       E: D + L_irr (soft) OR replace with monotone head (hard)
    """
    raise NotImplementedError("M1+: dispatch by stage with default lambdas from plan §3.5")
