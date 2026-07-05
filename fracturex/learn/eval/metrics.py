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


def equilibrium_residual_l2(
    sigma_grid,
    mask,
    dx: float = 1.0,
    dy: float = 1.0,
    f=0.0,
    d=None,
    d_c: float = 0.9,
    L: Optional[float] = None,
    eps: float = 1e-12,
) -> float:
    """Discrete grid equilibrium residual R̃_h(σ̂) — paper_thesis §C.

    Central-difference divergence of the stress on a Cartesian grid, restricted
    to the interior mask Ω_h^∘ = {(i,j): m=1 and all four central-difference
    neighbours are in m}; if ``d`` is provided, cells with d > d_c are also
    excluded from Ω_h^∘ to keep the crack band from dominating through
    finite-difference noise (paper_thesis §C, "一致性" clause).

    Sigma channel order (axis -3) is (σ_xx, σ_yy, σ_xy); the row axis (-2) is
    the y direction (i-index) and the column axis (-1) is the x direction
    (j-index), matching :func:`_grid_reaction_y` and the paper's convention
    "行 i↔y, 列 j↔x".

        r_x[i,j] = (σ_xx[i,j+1] − σ_xx[i,j−1])/(2Δx)
                   + (σ_xy[i+1,j] − σ_xy[i−1,j])/(2Δy) + f_x
        r_y[i,j] = (σ_xy[i,j+1] − σ_xy[i,j−1])/(2Δx)
                   + (σ_yy[i+1,j] − σ_yy[i−1,j])/(2Δy) + f_y

    The scale-free reported value is

        R̃_h = L · ‖(r_x, r_y)‖_{L²(Ω_h^∘)} / ‖σ̂‖_{L²(Ω_h^∘)},

    with L = diam(Ω); when ``L`` is not supplied it is taken as the diagonal
    of the mask's bounding box in physical units. Sigma is assumed already
    normalised (stress_scale removed), so the ratio is O(1). The returned
    scalar is the mean of R̃_h over any leading batch/time axes.

    Parameters
    ----------
    sigma_grid
        Array of shape (..., 3, H, W). Channel order (σ_xx, σ_yy, σ_xy).
    mask
        Boolean/0-1 mask indicating Ω on the grid, broadcastable to (..., H, W).
    dx, dy
        Grid spacings along axis -1 (x, cols) and axis -2 (y, rows).
    f
        Body force. Scalar (default 0 for the fracture benchmark) or an array
        broadcastable to (..., 2, H, W); channels (f_x, f_y).
    d
        Optional damage field of shape (..., H, W). Cells with d > d_c are
        removed from Ω_h^∘.
    d_c
        Damage threshold beyond which cells are excluded from Ω_h^∘.
    L
        Domain diameter; if ``None``, taken from the mask's bounding box.
    eps
        Numerical floor added to the denominator ‖σ̂‖.
    """
    s = _to_numpy(sigma_grid).astype(np.float64)
    if s.ndim < 3 or s.shape[-3] != 3:
        raise ValueError(f"sigma_grid must have shape (..., 3, H, W); got {s.shape}")
    H, W = s.shape[-2], s.shape[-1]
    sxx = s[..., 0, :, :]
    syy = s[..., 1, :, :]
    sxy = s[..., 2, :, :]

    m = _broadcast_mask(mask, sxx.shape).astype(bool)

    # Interior mask: central-difference neighbours all present in Ω, and the
    # centre itself is in Ω. We compare shifted copies of the mask; cells at
    # the boundary of the array (i or j at 0 or H-1 / W-1) automatically fail
    # because their shifted counterparts wrap in with zero — build via slicing.
    m_int = np.zeros_like(m)
    inner = (
        m[..., 1:-1, 1:-1]
        & m[..., 2:, 1:-1]
        & m[..., :-2, 1:-1]
        & m[..., 1:-1, 2:]
        & m[..., 1:-1, :-2]
    )
    m_int[..., 1:-1, 1:-1] = inner

    if d is not None:
        d_arr = _to_numpy(d).astype(np.float64)
        d_arr = np.broadcast_to(d_arr, sxx.shape)
        m_int = m_int & (d_arr <= d_c)

    # Central differences, defined only on the interior.
    d_sxx_dx = np.zeros_like(sxx)
    d_sxy_dy = np.zeros_like(sxx)
    d_sxy_dx = np.zeros_like(sxx)
    d_syy_dy = np.zeros_like(sxx)
    d_sxx_dx[..., 1:-1, 1:-1] = (sxx[..., 1:-1, 2:] - sxx[..., 1:-1, :-2]) / (2.0 * dx)
    d_sxy_dy[..., 1:-1, 1:-1] = (sxy[..., 2:, 1:-1] - sxy[..., :-2, 1:-1]) / (2.0 * dy)
    d_sxy_dx[..., 1:-1, 1:-1] = (sxy[..., 1:-1, 2:] - sxy[..., 1:-1, :-2]) / (2.0 * dx)
    d_syy_dy[..., 1:-1, 1:-1] = (syy[..., 2:, 1:-1] - syy[..., :-2, 1:-1]) / (2.0 * dy)

    if np.isscalar(f) or (isinstance(f, np.ndarray) and f.ndim == 0):
        fx = np.float64(f)
        fy = np.float64(f)
    else:
        f_arr = _to_numpy(f).astype(np.float64)
        if f_arr.shape[-3] != 2:
            raise ValueError(
                f"f must be scalar or shape (..., 2, H, W); got {f_arr.shape}"
            )
        fx = f_arr[..., 0, :, :]
        fy = f_arr[..., 1, :, :]

    r_x = d_sxx_dx + d_sxy_dy + fx
    r_y = d_sxy_dx + d_syy_dy + fy

    m_int_f = m_int.astype(np.float64)
    cell_area = dx * dy
    res_sq = (r_x * r_x + r_y * r_y) * m_int_f
    sig_sq = (sxx * sxx + syy * syy + sxy * sxy) * m_int_f

    spatial_axes = (-2, -1)
    per_res = np.sqrt(res_sq.sum(axis=spatial_axes) * cell_area)
    per_sig = np.sqrt(sig_sq.sum(axis=spatial_axes) * cell_area)

    if L is None:
        # Bounding-box diagonal of the passed-in mask, in physical units.
        # Use the outer mask (not m_int) so the domain scale is stable
        # regardless of whether the crack-band exclusion is active.
        rows_any = m.any(axis=-1)  # (..., H)
        cols_any = m.any(axis=-2)  # (..., W)
        # Per-sample bbox extent — reduce over leading dims by taking the
        # max span (a stable per-sample scale, same across leading axes when
        # the mask is shared, which is the common case).
        rows_span = rows_any.sum(axis=-1).astype(np.float64) * dy
        cols_span = cols_any.sum(axis=-1).astype(np.float64) * dx
        L_arr = np.sqrt(rows_span * rows_span + cols_span * cols_span)
        # Guard: if a sample has an empty mask, fall back to grid diagonal.
        grid_diag = np.sqrt((H * dy) ** 2 + (W * dx) ** 2)
        L_arr = np.where(L_arr > 0.0, L_arr, grid_diag)
    else:
        L_arr = np.float64(L)

    R_tilde = L_arr * per_res / (per_sig + eps)
    return float(np.asarray(R_tilde).mean())

