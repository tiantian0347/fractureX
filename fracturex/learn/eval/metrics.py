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
from typing import Optional, Union

import numpy as np

ArrayLike = Union[np.ndarray, "torch.Tensor"]  # noqa: F821


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _broadcast_mask(mask: ArrayLike, ref_shape: tuple) -> np.ndarray:
    m = _to_numpy(mask).astype(bool)
    if m.shape == ref_shape:
        return m
    return np.broadcast_to(m, ref_shape)


def relative_l2(pred, target, mask, eps: float = 1e-8) -> float:
    """Mask-weighted relative L² error: ‖m⊙(pred−target)‖₂ / (‖m⊙target‖₂ + eps).

    Numpy / torch tensors of any matching shape; ``mask`` may broadcast.
    """
    p = _to_numpy(pred).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs target {t.shape}")
    m = _broadcast_mask(mask, p.shape)
    diff = (p - t) * m
    denom = np.linalg.norm(t * m) + eps
    return float(np.linalg.norm(diff) / denom)


def relative_linf(pred, target, mask) -> float:
    """Mask-restricted L^∞: max|pred−target| / (max|target|) over m==1."""
    p = _to_numpy(pred).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs target {t.shape}")
    m = _broadcast_mask(mask, p.shape)
    if not m.any():
        return 0.0
    num = float(np.abs(p[m] - t[m]).max())
    den = float(np.abs(t[m]).max())
    return num / den if den > 0 else num


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
