"""Spectral labels + evaluation for the learned coarse space (D13 L2-beta).

Two roles, deliberately separated:

1. **Training label** (supervised, stable): ``ideal_interface_amplitude`` turns per-node
   features into a target per-node amplitude for the enrichment template. It is derived
   from the *high-contrast residual* signature the geometric P1 coarse space cannot
   resolve (plan command 0): nodes where ``g`` jumps across a ring (large
   ``|log(g/g_bar)|``) and the damage gradient is steep get a large target amplitude.
   Regressing to this avoids back-propagating through an eigensolver (unstable; plan
   §5.2 notes A1 is offline-spectral, A2 non-differentiable).

2. **Evaluation metric** (the real objective): ``two_level_kappa`` measures the actual
   condition number of the two-level (coarse + optional enrichment) preconditioned
   Schur block, i.e. what the enrichment must improve. Dense path for small/test blocks
   (exact), power-iteration ``lambda_max`` for large operators.

Backend policy: Schur / prolongation arrive as scipy CSR (solver boundary, exempt);
feature inputs are bm/numpy. No solver and no torch import here.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from fealpy.backend import backend_manager as bm

from fracturex.ml.coarse_features import FEATURE_NAMES


# ---------------------------------------------------------------------------
# 1. Supervised training label
# ---------------------------------------------------------------------------

def ideal_interface_amplitude(
    features,
    *,
    contrast_scale: float = 6.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Per-node target amplitude for the enrichment template (supervised label).

    The label is large exactly where the geometric P1 coarse space struggles: sharp
    high-contrast interfaces. We combine the two plan-command-0 signatures —
    ``|log(g/g_bar)|`` (degradation contrast across a node ring) and ``gradd_l0``
    (dimensionless damage gradient) — into a smooth target in ``[0, 1]``:

        a = ( |log(g/g_bar)| / contrast_scale )^gamma  *  tanh(gradd_l0 / median_band)

    normalized so the most-localized node maps near 1 and smooth regions map to ~0.

    Args:
        features: ``(NN, n_feat)`` per-node features (bm/numpy), columns per
            ``coarse_features.FEATURE_NAMES``.
        contrast_scale: log-contrast that maps to amplitude ~1 (eps_g=1e-6 -> ~6).
        gamma: sharpening exponent on the contrast term.
    Returns:
        ``(NN,)`` numpy float64 target amplitudes in ``[0, 1]``.
    """
    phi = np.asarray(bm.to_numpy(features) if not isinstance(features, np.ndarray)
                     else features, dtype=np.float64)
    j_log = FEATURE_NAMES.index("log_g_over_gbar")
    j_grad = FEATURE_NAMES.index("gradd_l0")
    contrast = np.abs(phi[:, j_log]) / max(contrast_scale, 1e-30)
    contrast = np.clip(contrast, 0.0, 1.0) ** max(gamma, 1e-6)
    grad = phi[:, j_grad]
    scale = np.median(grad[grad > 0]) if np.any(grad > 0) else 1.0
    grad_term = np.tanh(grad / max(scale, 1e-30))
    a = contrast * grad_term
    amax = float(np.max(a)) if a.size else 0.0
    return a / amax if amax > 0 else a


def worst_mode_amplitude(
    S_b: "sp.csr_matrix",
    PI_s: "sp.csr_matrix",
    *,
    iters: int = 50,
    seed: int = 0,
    smoother_omega: float = 0.5,
) -> np.ndarray:
    """Per-node target amplitude from the SPECTRALLY-WORST mode of the baseline solver.

    Unlike :func:`ideal_interface_amplitude` (a pure feature heuristic), this label is
    derived from the actual two-level error-propagation operator ``E = I - M0^{-1} S_b``
    (geometric coarse + Jacobi smoother): its dominant eigenvector is the slowest-
    converging mode — exactly what the enrichment must remove (plan command 0/6). We
    power-iterate ``E`` (operator-action only, no densification), restrict the worst
    mode to the P1 coarse space via ``PI_s^T``, and take per-node magnitude (normalized
    to ``[0, 1]``) as the supervised amplitude target. Training to this connects the
    learned amplitude to the real spectral objective.

    Args:
        S_b: ``(n, n)`` SPD Schur block (one displacement component).
        PI_s: ``(n, nc)`` P1 prolongation (nc == NN coarse dofs).
        iters: power-iteration steps on the error-propagation operator.
        seed: RNG seed for the start vector.
        smoother_omega: Jacobi damping in the baseline ``M0^{-1}`` (match the metric).
    Returns:
        ``(nc,)`` numpy float64 target amplitudes in ``[0, 1]`` (per coarse node).
    """
    n = S_b.shape[0]
    M0 = make_two_level_minv(S_b, PI_s, smoother_omega=smoother_omega)

    def E(v):  # error-propagation operator I - M0^{-1} S_b
        return v - M0.matvec(S_b @ v)

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x /= max(np.linalg.norm(x), 1e-30)
    for _ in range(max(1, iters)):
        y = E(x)
        ny = float(np.linalg.norm(y))
        if ny < 1e-30:
            break
        x = y / ny
    # restrict worst fine-space mode to the coarse (P1) space; per-node magnitude.
    coarse = np.abs(np.asarray(PI_s.T @ x).reshape(-1))
    cmax = float(np.max(coarse)) if coarse.size else 0.0
    return coarse / cmax if cmax > 0 else coarse


def top_k_worst_modes(
    S_b: "sp.csr_matrix",
    PI_s: "sp.csr_matrix",
    k: int,
    *,
    iters: int = 40,
    seed: int = 0,
    smoother_omega: float = 0.5,
) -> np.ndarray:
    """Top-k slowest-converging modes of the baseline two-level solver (GenEO basis).

    Block (subspace) iteration on the error-propagation operator ``E = I - M0^{-1} S_b``:
    its k dominant eigenvectors span the worst-conditioned subspace the geometric coarse
    + smoother cannot resolve (plan command 0/6). Returned as FINE-space columns (sgdof),
    to be used DIRECTLY as the deflation subspace (they are genuinely outside
    ``range(PI_s)``, unlike ``PI_s @ coarse``). This is the multi-mode generalization of
    :func:`worst_mode_amplitude`; a single rank-1 mode cannot fix a high-dimensional
    interface bad-subspace, hence ``k >> 1`` (D13_IMPL §6.5).

    Args:
        S_b: ``(n, n)`` SPD Schur block (one displacement component), n == sgdof.
        PI_s: ``(n, nc)`` P1 prolongation (only used to build the baseline M0).
        k: number of worst modes to return.
        iters: subspace-iteration steps.
        seed: RNG seed for the random start block.
        smoother_omega: Jacobi damping in the baseline M0. MUST keep ``M0`` convergent
            (``rho(I - M0^{-1} S_b) < 1``); the undamped ``omega=1`` additive two-level
            DIVERGES on the localized Schur block (|lambda(E)|~2.3, D13_IMPL §6.7), which
            would make the "worst modes" garbage. ``0.5`` restores convergence here.
    Returns:
        ``(n, k)`` numpy float64 orthonormal basis of the k dominant ``E`` eigenvectors.
    """
    n = S_b.shape[0]
    k = int(max(1, k))
    M0 = make_two_level_minv(S_b, PI_s, smoother_omega=smoother_omega)

    def E_block(X):  # apply E = I - M0^{-1} S_b to each column
        out = np.empty_like(X)
        for j in range(X.shape[1]):
            v = X[:, j]
            out[:, j] = v - M0.matvec(S_b @ v)
        return out

    rng = np.random.default_rng(seed)
    X, _ = np.linalg.qr(rng.standard_normal((n, k)))
    for _ in range(max(1, iters)):
        Y = E_block(X)
        X, _ = np.linalg.qr(Y)  # re-orthonormalize the subspace
    return np.ascontiguousarray(X)


# ---------------------------------------------------------------------------
# 2. Two-level condition-number evaluation (the real objective)
# ---------------------------------------------------------------------------

def make_two_level_minv(
    S_b: sp.csr_matrix,
    PI_s: sp.csr_matrix,
    *,
    enrich_W: Optional[np.ndarray] = None,
    enrich_pinv: Optional[np.ndarray] = None,
    coarse_reg: float = 0.0,
    smoother_omega: float = 1.0,
) -> LinearOperator:
    """Additive two-level preconditioner ``M^{-1}`` for one Schur block (SPD).

    ``M^{-1} r = omega diag(S_b)^{-1} r  +  PI_s Sc^{-1} PI_s^T r  [+ W S_Phi^+ W^T r]``

    with coarse matrix ``Sc = PI_s^T S_b PI_s`` (exact Galerkin, densely solved). The
    **Jacobi smoother term is essential**: coarse correction alone is rank-deficient
    (rank ``nc << n``), so its ``M^{-1} S_b`` condition number is meaningless. The
    additive smoother makes ``M^{-1}`` SPD on all of ``R^n`` and the resulting kappa
    reflects the runtime two-level operator (smoother + coarse + enrichment). A
    high-contrast interface jump mode is low-energy yet outside ``range(PI_s)`` and not
    damped by Jacobi -> it inflates kappa, and enriching with it is what brings kappa
    down (plan command 0 / command 6).

    Args:
        S_b: ``(n, n)`` SPD Schur block (one displacement component).
        PI_s: ``(n, nc)`` P1 prolongation.
        enrich_W: optional ``(n, k)`` prolonged enrichment basis ``PI_s @ Phi``.
        enrich_pinv: optional ``(k, k)`` regularized inverse of ``W^T S_b W``.
        coarse_reg: ridge on the coarse Galerkin matrix before factorization.
        smoother_omega: damping for the additive Jacobi smoother term.
    Returns:
        a scipy ``LinearOperator`` applying ``M^{-1}``.
    """
    n = S_b.shape[0]
    PI = PI_s.tocsr()
    PIT = PI.T.tocsr()
    Sc = (PIT @ S_b @ PI).toarray()
    Sc = 0.5 * (Sc + Sc.T)
    if coarse_reg > 0:
        Sc = Sc + coarse_reg * np.trace(Sc) / max(Sc.shape[0], 1) * np.eye(Sc.shape[0])
    Sc_inv = np.linalg.pinv(Sc)
    diag = np.asarray(S_b.diagonal(), dtype=np.float64)
    dinv = np.where(np.abs(diag) > 1e-30, 1.0 / diag, 0.0)
    omega = float(smoother_omega)

    def base_solve(r):
        # Keep PI_s SPARSE: PI @ (Sc_inv @ (PI^T r)) avoids densifying the (sgdof x NN)
        # prolongation (~200MB) and the memory-bandwidth-bound dense matmuls -- important
        # on a shared box. Numerically identical to PId @ ... @ PId^T.
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        return omega * (dinv * r) + (PI @ (Sc_inv @ (PIT @ r)))

    if enrich_W is None or enrich_pinv is None:
        return LinearOperator((n, n), matvec=base_solve, dtype=np.float64)

    # Enrichment via DEFLATION (not additive): wraps the base two-level solve.
    # M_def^{-1} r = Q r + (I - Q S_b)(base_solve((I - S_b Q) r)),  Q = W (W^T S_b W)^+ W^T.
    Wd = np.ascontiguousarray(enrich_W)
    SbWd = np.ascontiguousarray((S_b @ Wd))
    pinv = np.ascontiguousarray(enrich_pinv)

    def matvec(r):
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        coeff = pinv @ (Wd.T @ r)          # (W^T S_b W)^+ (W^T r)
        coarse = Wd @ coeff                # Q r
        rhs = r - SbWd @ coeff             # (I - S_b Q) r
        y = base_solve(rhs)
        y = y - Wd @ (pinv @ (SbWd.T @ y))  # (I - Q S_b) y
        return coarse + y

    return LinearOperator((n, n), matvec=matvec, dtype=np.float64)


def two_level_kappa_dense(S_b: sp.csr_matrix, Minv: LinearOperator) -> float:
    """Exact condition number of ``M^{-1} S_b`` by densification (small/test blocks).

    ``M^{-1} S_b`` is SPD-similar (product of two SPD operators), so its eigenvalues are
    real positive; we symmetrize numerically and use ``eigvalsh``.

    Args:
        S_b: ``(n, n)`` SPD Schur block.
        Minv: two-level preconditioner operator (e.g. from :func:`make_two_level_minv`).
    Returns:
        ``lambda_max / lambda_min`` (float). Use only for modest ``n`` (densifies).
    """
    n = S_b.shape[0]
    Sd = S_b.toarray()
    T = np.column_stack([Minv.matvec(Sd[:, j]) for j in range(n)])  # M^{-1} S_b
    T = 0.5 * (T + T.T)
    ev = np.linalg.eigvalsh(T)
    ev = ev[ev > 1e-14 * float(ev.max())]
    return float(ev.max() / ev.min())


def power_lambda_max(apply: Callable[[np.ndarray], np.ndarray], n: int,
                     *, iters: int = 30, seed: int = 0) -> float:
    """Largest eigenvalue magnitude of an operator via power iteration (any size).

    Args:
        apply: ``v -> A v`` operator action (e.g. ``M^{-1} S_b``).
        n: operator dimension.
        iters: power-iteration steps.
        seed: RNG seed for the start vector (Date/random unavailable in workflows; ok in tests).
    Returns:
        estimated ``lambda_max`` (Rayleigh quotient at the final iterate).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x /= max(np.linalg.norm(x), 1e-30)
    lam = 1.0
    for _ in range(max(1, iters)):
        y = np.asarray(apply(x)).reshape(-1)
        ny = float(np.linalg.norm(y))
        if ny < 1e-30:
            return max(lam, 1e-30)
        x = y / ny
        lam = float(x @ np.asarray(apply(x)).reshape(-1))
    return max(lam, 1e-30)
