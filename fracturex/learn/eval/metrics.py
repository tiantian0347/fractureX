# fracturex/learn/eval/metrics.py
"""Evaluation metrics for the operator-learning surrogate.

Plan §M1 evaluation table (every baseline must report these):
  - relative L^2, relative H^1   (smoothness / fidelity)
  - SSIM                          (perceptual similarity for damage)
  - crack-set IoU at d_c = 0.5
  - crack-front Hausdorff distance
  - peak-load error               (most persuasive single number, plan §9.2 Q4)
  - equilibrium residual L^2      (physics, added in M2 Stage D)
"""
from __future__ import annotations
from typing import Optional


def relative_l2(pred, target, mask, eps: float = 1e-8) -> float:
    raise NotImplementedError("M1: ||m⊙(pred-target)||_2 / (||m⊙target||_2 + eps)")


def relative_linf(pred, target, mask) -> float:
    raise NotImplementedError("M1: max-norm restricted to mask==1")


def relative_h1(pred, target, mask, eps: float = 1e-8) -> float:
    raise NotImplementedError("M1: relative L^2 of central-difference gradient")


def crack_set_iou(d_pred, d_target, mask, d_c: float = 0.5) -> float:
    raise NotImplementedError("M1: IoU of {d > d_c} sets, restricted to mask")


def crack_front_hausdorff(d_pred, d_target, mask, d_c: float = 0.5) -> float:
    raise NotImplementedError("M1: Hausdorff distance between {d ≈ d_c} contours")


def ssim_masked(pred, target, mask, window: int = 11) -> float:
    raise NotImplementedError("M1: structural similarity index, mask-aware")


def peak_load_error(reaction_pred, reaction_target) -> float:
    """e_peak = |F_max^pred − F_max^ref| / |F_max^ref|.  Plan §9.2 Q4."""
    raise NotImplementedError("M1: max over time axis, then relative error")


def equilibrium_residual_l2(sigma_pred, body_force, mask) -> float:
    raise NotImplementedError("M2 Stage D: ‖m⊙(∇_h·σ̂ + f)‖_2")
