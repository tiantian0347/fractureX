# fracturex/learn/eval/metrics.py
"""Evaluation metrics for the operator-learning surrogate.

Plan §M1 evaluation table (every baseline must report these):
  - relative L^2, relative H^1   (smoothness / fidelity)
  - SSIM                          (perceptual similarity for damage)
  - crack-set IoU at d_c = 0.5
  - crack-front Hausdorff distance
  - peak-load error               (most persuasive single number, plan §9.2 Q4)
  - equilibrium residual L^2      (physics, added in M2 Stage D)

All metrics are numpy-based and torch-free (accept numpy or torch tensors),
so they can be exercised without a training backend installed.
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
    # Insert singleton axes after the batch axis until ranks match, then broadcast.
    while m.ndim < len(ref_shape):
        m = m[:, None, ...] if m.ndim >= 1 else m[None, ...]
    return np.broadcast_to(m, ref_shape)


def relative_l2(pred, target, mask, eps: float = 1e-8) -> float:
    """Mask-weighted relative L² error: ‖m⊙(pred−target)‖₂ / (‖m⊙target‖₂ + eps)."""
    p = _to_numpy(pred).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs target {t.shape}")
    m = _broadcast_mask(mask, p.shape)
    diff = (p - t) * m
    denom = np.linalg.norm(t * m) + eps
    return float(np.linalg.norm(diff) / denom)


def relative_linf(pred, target, mask) -> float:
    """Mask-restricted L^∞: max|pred−target| / max|target| over m==1."""
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


def _central_grad_np(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    gx[..., 1:-1, :] = (x[..., 2:, :] - x[..., :-2, :]) * 0.5
    gy[..., :, 1:-1] = (x[..., :, 2:] - x[..., :, :-2]) * 0.5
    return gx, gy


def relative_h1(pred, target, mask, eps: float = 1e-8) -> float:
    """Relative H¹ error (value + central-difference gradient), mask-weighted."""
    p = _to_numpy(pred).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs target {t.shape}")
    m = _broadcast_mask(mask, p.shape).astype(np.float64)
    px, py = _central_grad_np(p)
    tx, ty = _central_grad_np(t)
    num = np.sqrt(
        np.sum(((p - t) * m) ** 2)
        + np.sum(((px - tx) * m) ** 2)
        + np.sum(((py - ty) * m) ** 2)
    )
    den = np.sqrt(
        np.sum((t * m) ** 2) + np.sum((tx * m) ** 2) + np.sum((ty * m) ** 2)
    ) + eps
    return float(num / den)


def crack_set_iou(d_pred, d_target, mask, d_c: float = 0.5) -> float:
    """IoU of the crack sets {d > d_c}, restricted to the in-Ω mask."""
    p = _to_numpy(d_pred)
    t = _to_numpy(d_target)
    m = _broadcast_mask(mask, p.shape)
    pp = (p > d_c) & m
    tt = (t > d_c) & m
    inter = np.logical_and(pp, tt).sum()
    union = np.logical_or(pp, tt).sum()
    if union == 0:
        return 1.0  # both empty ⇒ perfect agreement
    return float(inter) / float(union)


def crack_front_hausdorff(d_pred, d_target, mask, d_c: float = 0.5) -> float:
    """Symmetric Hausdorff distance (in pixels) between {d ≈ d_c} crack fronts.

    Uses the boundary of the {d > d_c} set as the front. Returns 0.0 when both
    fronts are empty, and ``inf`` when exactly one is empty.
    """
    from scipy.ndimage import binary_erosion

    def _front(d):
        d = _to_numpy(d).astype(bool) if d.dtype == bool else (_to_numpy(d) > d_c)
        # Collapse any leading (T, C, ...) axes; use the last frame's 2D plane set.
        if d.ndim > 2:
            d = d.reshape(-1, d.shape[-2], d.shape[-1]).any(axis=0)
        eroded = binary_erosion(d, border_value=0)
        return np.argwhere(d & ~eroded)  # boundary pixel coords (n, 2)

    a = _front(_to_numpy(d_pred) > d_c)
    b = _front(_to_numpy(d_target) > d_c)
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    from scipy.spatial import cKDTree

    ta, tb = cKDTree(a), cKDTree(b)
    d_ab = ta.query(b)[0].max()
    d_ba = tb.query(a)[0].max()
    return float(max(d_ab, d_ba))


def ssim_masked(pred, target, mask, window: int = 11, eps: float = 1e-8) -> float:
    """Global SSIM over masked damage maps (single-window approximation).

    A lightweight, dependency-free SSIM computed over in-Ω pixels with the
    standard constants; sufficient for the M1 comparison table.
    """
    p = _to_numpy(pred).astype(np.float64)
    t = _to_numpy(target).astype(np.float64)
    m = _broadcast_mask(mask, p.shape)
    if not m.any():
        return 1.0
    pv, tv = p[m], t[m]
    L = max(float(tv.max()) - float(tv.min()), 1.0)  # dynamic range; ≥1 for d∈[0,1]
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu_p, mu_t = pv.mean(), tv.mean()
    var_p, var_t = pv.var(), tv.var()
    cov = ((pv - mu_p) * (tv - mu_t)).mean()
    num = (2 * mu_p * mu_t + c1) * (2 * cov + c2)
    den = (mu_p ** 2 + mu_t ** 2 + c1) * (var_p + var_t + c2)
    return float(num / (den + eps))


def stress_relative_l2(sigma_pred, sigma_target, mask, eps: float = 1e-8) -> float:
    """Mask-weighted relative L² over the 3 stress channels (σxx, σyy, σxy).

    Just :func:`relative_l2` applied to the stress block; named for clarity in
    the Stage B report. Inputs are (B, T, 3, H, W) (or any matching shape).
    """
    return relative_l2(sigma_pred, sigma_target, mask, eps=eps)


def _max_principal_stress(sigma: np.ndarray) -> np.ndarray:
    """σ₁ = (σxx+σyy)/2 + sqrt(((σxx−σyy)/2)² + σxy²).

    ``sigma`` has the stress channels on axis -3 in (σxx, σyy, σxy) order
    (shape (..., 3, H, W)); returns (..., H, W).
    """
    sxx = sigma[..., 0, :, :]
    syy = sigma[..., 1, :, :]
    sxy = sigma[..., 2, :, :]
    mean = 0.5 * (sxx + syy)
    rad = np.sqrt((0.5 * (sxx - syy)) ** 2 + sxy ** 2)
    return mean + rad


def principal_stress_relative_l2(sigma_pred, sigma_target, mask, eps: float = 1e-8) -> float:
    """Relative L² of the max principal stress σ₁ (mask-weighted).

    Captures the physically salient stress concentration near the crack tip,
    which a per-component L² can under-weight. Channel axis is -3 (σxx,σyy,σxy).
    """
    p = _to_numpy(sigma_pred).astype(np.float64)
    t = _to_numpy(sigma_target).astype(np.float64)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs target {t.shape}")
    s1p = _max_principal_stress(p)   # drops channel axis → (..., H, W)
    s1t = _max_principal_stress(t)
    m = _broadcast_mask(mask, s1t.shape)
    num = np.linalg.norm((s1p - s1t) * m)
    den = np.linalg.norm(s1t * m) + eps
    return float(num / den)


def sigma_peak_relative_l2(sigma_pred, sigma_target, mask, q: float = 95.0,
                           eps: float = 1e-8) -> float:
    """Relative L² restricted to the crack-tip peak region (top-(100−q)% ‖σ‖).

    Isolates how well the singular stress concentration is captured — the part
    a global rel-L² hides (m2_stageB_results.md §6.3). Channel axis is -3
    (σxx, σyy, σxy); the peak mask is the pixels whose target stress magnitude
    ‖σ‖ exceeds its q-th percentile within Ω.
    """
    p = _to_numpy(sigma_pred).astype(np.float64)
    t = _to_numpy(sigma_target).astype(np.float64)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs target {t.shape}")
    smag = np.sqrt((t ** 2).sum(axis=-3))                  # (..., H, W)
    m = _broadcast_mask(mask, smag.shape).astype(bool)
    vals = smag[m]
    if vals.size == 0:
        return 0.0
    thr = np.percentile(vals, q)
    peak = (m & (smag >= thr))[..., None, :, :]            # (..., 1, H, W)
    num = np.linalg.norm((p - t) * peak)
    den = np.linalg.norm(t * peak) + eps
    return float(num / den)


def _grid_reaction_y(sigma, mask) -> np.ndarray:
    """Top-boundary reaction proxy R_y(t) from grid σ (scale-free, normalized).

    On the top row (y=ymax, n=(0,1)) the y-traction is σ_yy, so
    R_y(t) ≈ Σ_j σ_yy(top_row, j)·mask. Channel axis -3 is (σxx,σyy,σxy);
    the spatial top row is index -2 == H-1 (rows increase with y).
    Returns shape (..., T).
    """
    s = _to_numpy(sigma).astype(np.float64)
    m = _to_numpy(mask).astype(bool)
    syy_top = s[..., 1, -1, :]                 # (..., T, W)
    # mask top row → (..., W); collapse a leading channel axis of size 1 if present.
    mtop = m[..., -1, :]
    if mtop.shape[-2:] != syy_top.shape[-1:] and mtop.ndim >= 2 and mtop.shape[-2] == 1:
        mtop = mtop[..., 0, :]
    return (syy_top * mtop[..., None, :]).sum(axis=-1)   # (..., T)


def peak_load_error(reaction_pred, reaction_target) -> float:
    """e_peak = |F_max^pred − F_max^ref| / |F_max^ref|.  Plan §9.2 Q4.

    Inputs are reaction curves (..., T) (or (..., T, r)); the peak is taken
    over the time axis of the magnitude.
    """
    rp = _to_numpy(reaction_pred).astype(np.float64)
    rt = _to_numpy(reaction_target).astype(np.float64)
    fmax_p = float(np.abs(rp).max()) if rp.size else 0.0
    fmax_t = float(np.abs(rt).max()) if rt.size else 0.0
    if fmax_t == 0.0:
        return 0.0 if fmax_p == 0.0 else float("inf")
    return abs(fmax_p - fmax_t) / fmax_t


def peak_load_error_grid(sigma_pred, sigma_target, mask) -> float:
    """Peak-load error from grid σ: |max_t R_pred − max_t R_true| / |max_t R_true|.

    Both reaction curves are integrated from the (normalized) grid stress on the
    top boundary via :func:`_grid_reaction_y`, so the per-sample stress_scale
    cancels and the metric is scale-free. sigma_*: (B, T, 3, H, W).
    """
    Rp = _grid_reaction_y(sigma_pred, mask)     # (B, T)
    Rt = _grid_reaction_y(sigma_target, mask)
    fp = np.abs(Rp).max(axis=-1)                 # (B,)
    ft = np.abs(Rt).max(axis=-1)
    denom = np.where(ft > 0, ft, 1.0)
    err = np.abs(fp - ft) / denom
    return float(err.mean())


def equilibrium_residual_l2(sigma_pred, body_force, mask) -> float:
    raise NotImplementedError("M2 Stage D: ‖m⊙(∇_h·σ̂ + f)‖_2")
