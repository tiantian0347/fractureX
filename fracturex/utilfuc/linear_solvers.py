"""Krylov solvers and block/auxiliary-space preconditioners for the Hu-Zhang mixed system.

This module provides two layers:

1. Generic preconditioned Krylov wrappers (``solve_gmres_ilu``, ``solve_lgmres_amg``,
   ``solve_minres_diag``, ...) returning ``(x, KrylovInfo)``.
2. Saddle-point solvers for the Hu-Zhang stress/displacement system
   ``K_h = [[A(d_h), B^T], [B, 0]]`` with block-triangular preconditioners whose Schur
   complement ``S(d_h) = B A(d_h)^{-1} B^T`` is approximated by ``B diag(A)^{-1} B^T`` and
   accelerated by Gauss-Seidel / Chebyshev smoothing plus a P1 auxiliary-space coarse
   correction (``solve_huzhang_block_gmres_fast`` / ``..._auxspace``).

Optional backends FEALPy (``FEALPY_AVAILABLE``) and pyamg (``PYAMG_AVAILABLE``) are detected
at import; the block solvers degrade or raise with an explicit message when they are absent.

Module-level ``_*_CACHE`` dicts memoize geometry-only operators (P1 prolongation ``PI_s``,
divergence block ``B``, pyamg hierarchies) keyed by ``id(mesh)`` so they survive across
staggered iterations where only the damaged coefficients in ``A`` change.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import importlib.util
import inspect
import numpy as np
from scipy.sparse import block_diag, csr_matrix, spdiags
from scipy.sparse.linalg import LinearOperator, gmres, lgmres, minres, spilu

try:
    from fealpy.backend import backend_manager as bm
    from fealpy.functionspace import LagrangeFESpace
    from fealpy.fem import BilinearForm, DirichletBC, ScalarDiffusionIntegrator

    FEALPY_AVAILABLE = True
except ImportError:
    FEALPY_AVAILABLE = False

try:
    from pyamg.relaxation.relaxation import gauss_seidel
    PYAMG_AVAILABLE = True
except ImportError:
    PYAMG_AVAILABLE = False


_AUXSPACE_STATIC_CACHE: Dict[Tuple[int, int], Dict[str, Any]] = {}
_AUXSPACE_SCHUR_CACHE: Dict[Tuple[int, int], Dict[str, Any]] = {}
# Divergence block B_div = A_[m:, :m] and its transpose. Geometric constant for the
# `standard` formulation (B2 = TM^T B is d-independent); keyed by (mesh_id, gdof_sigma).
_AUXSPACE_B_CACHE: Dict[Tuple[int, int], Tuple[Any, Any]] = {}
_AUXSPACE_COARSE_CACHE: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
# P1 isotropic Laplacian -> pyamg hierarchy (mesh_id, q); independent of elastic A updates.
_FAST_PYAMG_P1_ISO_CACHE: Dict[Tuple[int, int], Any] = {}
# fast-path setup amortization: reuse the EXPENSIVE per-solve objects (Schur S triple
# product; weighted P1-coarse diffusion + pyamg hierarchy) across up to
# `precond_rebuild_interval` consecutive solves. The preconditioner is approximate, so a
# slightly stale S/ml only affects GMRES iteration count, NEVER the solution (GMRES is run
# on the fresh A_,b_). Keyed per (mesh_id, ..., kind); each entry tracks calls_since_build.
_FAST_PRECOND_CACHE: Dict[Tuple, Dict[str, Any]] = {}


def _auxspace_log(msg: str) -> None:
    """Progress logging for aux-space GMRES (set FRACTUREX_AUXSPACE_DEBUG=1)."""
    if os.environ.get("FRACTUREX_AUXSPACE_DEBUG", "").strip() not in ("1", "true", "True", "yes", "YES"):
        return
    t = time.perf_counter()
    print(f"[aux-gmres {t:.3f}s] {msg}", flush=True)


def _get_or_build_auxspace_pi_operators(mesh, vspace, q: int) -> Dict[str, Any]:
    """
    Cached P1 coarse space + prolongation PI_s from scalar displacement space to P1 nodes.

    Shared by ``solve_huzhang_block_gmres_auxspace`` and ``solve_huzhang_block_gmres_fast``.
    """
    if not FEALPY_AVAILABLE:
        raise RuntimeError("PI_s operators require FEALPy (FEALPY_AVAILABLE is False).")
    mesh_id = id(mesh)
    cache_key = (mesh_id, int(q))
    cached = _AUXSPACE_STATIC_CACHE.get(cache_key, None)
    if cached is not None:
        _auxspace_log("static cache hit")
        return cached

    _auxspace_log("static cache miss: building P1 cspace + PI_s operators")
    gdim = mesh.geo_dimension()
    cspace = LagrangeFESpace(mesh, 1)
    NC = mesh.number_of_cells()

    sspace = getattr(vspace, "scalar_space", None)
    if sspace is None or not hasattr(sspace, "dof"):
        raise RuntimeError("TensorFunctionSpace.scalar_space/dof is unavailable")

    bc_ref = sspace.dof.multiIndex / sspace.p
    entries_s = bm.tile(bc_ref, (NC, 1))

    fldof_s = sspace.number_of_local_dofs()
    sgdof = sspace.number_of_global_dofs()
    cldof = cspace.number_of_local_dofs()
    cgdof = cspace.number_of_global_dofs()

    I_s = bm.broadcast_to(sspace.cell_to_dof()[:, :, None], (NC, fldof_s, gdim + 1))
    J_s = bm.broadcast_to(cspace.cell_to_dof()[:, None, :], (NC, fldof_s, cldof))

    PI_s = csr_matrix(
        (np.asarray(entries_s).ravel(), (np.asarray(I_s).ravel(), np.asarray(J_s).ravel())),
        shape=(sgdof, cgdof),
    )
    PI_s_T = PI_s.T.tocsr()
    cached = {
        "gdim": int(gdim),
        "sgdof": int(sgdof),
        "cspace": cspace,
        "PI_s": PI_s.tocsr(),
        "PI_s_T": PI_s_T,
    }
    _AUXSPACE_STATIC_CACHE[cache_key] = cached
    _auxspace_log("static cache stored")
    return cached


def _estimate_lambda_max_dinv_s_numpy(S: csr_matrix, dinv: np.ndarray, *, iters: int = 10) -> float:
    """Power iteration for largest eigenvalue of diag(S)^{-1} S (for Chebyshev upper bound)."""
    n = int(S.shape[0])
    x = np.ones(n, dtype=np.float64)
    nx = float(np.linalg.norm(x))
    if nx < 1e-30:
        return 1.0
    x = x / nx
    lam = 1.0
    for _ in range(max(1, int(iters))):
        y = dinv * (S @ x)
        ny = float(np.linalg.norm(y))
        if ny < 1e-30:
            return max(lam, 1e-12)
        x = y / ny
        Ax = dinv * (S @ x)
        den = float(np.dot(x, x))
        lam = float(np.dot(x, Ax) / den) if den > 0 else lam
    return max(lam, 1e-12)


def _chebyshev_smooth_numpy(
    S: csr_matrix,
    x: np.ndarray,
    b: np.ndarray,
    dinv: np.ndarray,
    lmax: float,
    *,
    degree: int = 2,
    lower_ratio: float = 0.1,
) -> np.ndarray:
    """Chebyshev iteration for (D^{-1} S) on residual; x updated in-place semantics via return."""
    if degree <= 0:
        return np.asarray(x, dtype=np.float64).copy()

    x = np.asarray(x, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    lmin = max(lower_ratio * lmax, 1e-12 * lmax)
    theta = 0.5 * (lmax + lmin)
    delta = 0.5 * (lmax - lmin)

    if abs(delta) < 1e-30:
        r = b - S @ x
        return x + (dinv / theta) * r

    sigma = theta / delta
    rho = 1.0 / sigma

    r = b - S @ x
    z = dinv * r
    p = z / theta
    x = x + p

    for _ in range(1, int(degree)):
        r = b - S @ x
        z = dinv * r
        rho_new = 1.0 / (2.0 * sigma - rho)
        p = rho_new * rho * p + (2.0 * rho_new / delta) * z
        x = x + p
        rho = rho_new
    return x


def _normalize_elastic_formulation(formulation: str) -> str:
    """Map user-facing elastic formulation name to ``standard`` or ``effective_stress``."""
    f = str(formulation).strip().lower()
    if f in ("standard", "std", "stress", "m", "sigma"):
        return "standard"
    if f in ("effective_stress", "effective", "effective-stress", "b", "coupling", "mix"):
        return "effective_stress"
    raise ValueError(
        f"Unknown elastic_formulation={formulation!r}; use 'standard' or 'effective_stress'."
    )


def _coarse_diffusion_uses_stress_weight(formulation: str, weighted_aux: bool) -> bool:
    """
    Auxiliary P1 diffusion is weighted by g(d) only for the standard formulation
    (degradation on the stress block M; Chen et al. 2017 §5: g-weighted lambda=0 elasticity
    / vector Poisson on the Schur complement). For effective_stress, g(d) lives on B and
    the coarse Laplacian stays unweighted; damage enters the Schur block via A.
    """
    return bool(weighted_aux) and _normalize_elastic_formulation(formulation) == "standard"


def _assemble_p1_diffusion_pyamg(mesh, q: int, coef=None):
    """P1 scalar diffusion + Dirichlet BC; optional FE coefficient (rebuilt every call)."""
    if not FEALPY_AVAILABLE:
        raise RuntimeError("P1 Laplacian assembly requires FEALPy.")
    try:
        import pyamg
    except ImportError as ex:
        raise RuntimeError("P1 diffusion pyamg requires pyamg.") from ex

    cspace = LagrangeFESpace(mesh, 1)
    bform = BilinearForm(cspace)
    if coef is None:
        bform.add_integrator(ScalarDiffusionIntegrator(q=int(q)))
    else:
        bform.add_integrator(ScalarDiffusionIntegrator(coef=coef, q=int(q)))
    S_ = bform.assembly()
    bc = DirichletBC(cspace, bm.zeros(S_.shape[0], dtype=S_.dtype))
    S_, _ = bc.apply(S_, bm.zeros(S_.shape[0], dtype=S_.dtype))
    S_scipy = as_scipy_csr(S_)
    return pyamg.smoothed_aggregation_solver(S_scipy)


def _get_or_build_p1_isotropic_pyamg(mesh, q: int):
    """Smoothed-aggregation hierarchy on P1 scalar Laplacian (Dirichlet); cached per mesh,q."""
    mesh_id = id(mesh)
    key = (mesh_id, int(q))
    ml = _FAST_PYAMG_P1_ISO_CACHE.get(key, None)
    if ml is not None:
        return ml
    ml = _assemble_p1_diffusion_pyamg(mesh, int(q), coef=None)
    _FAST_PYAMG_P1_ISO_CACHE[key] = ml
    return ml


def _make_weighted_g_coef(*, d_fun, degradation_fun):
    """
    Legacy coef(bcs,index)->g(d) from raw d and degradation(d).
    Prefer :func:`_make_coarse_diffusion_coef` with damage.coef_bary for consistency with assembly.
    """
    if d_fun is None or degradation_fun is None:
        return None

    def coef_g(bcs, index=None):
        dval = d_fun(bcs, index=index)
        gd = degradation_fun(dval)
        return bm.asarray(gd)

    return coef_g


def _make_coarse_diffusion_coef(
    *,
    formulation: str,
    weighted_aux: bool,
    damage=None,
    state=None,
    d_fun=None,
    degradation_fun=None,
):
    """
    Coarse auxiliary diffusion coefficient (Chen et al. 2017 §5, phase-field extension):

    - ``standard``: coef = g(d) from ``damage.coef_bary`` (g-weighted vector Poisson /
      lambda=0 elasticity on the Schur complement; matches stress-block 1/g without an
      extra floor, so both sides see the same g(d)).
    - ``effective_stress``: None (unweighted Laplacian; g(d) is on the B block only).
    """
    if not _coarse_diffusion_uses_stress_weight(formulation, weighted_aux):
        return None

    if damage is not None and state is not None:
        from fracturex.damage.base import DamageStateView

        def coef_g(bcs, index=None):
            view = DamageStateView(
                d=state.d,
                sigma=state.sigma,
                u=state.u,
                r_hist=state.r_hist,
                H=state.H,
            )
            return bm.asarray(damage.coef_bary(view, bcs, index=index))

        return coef_g

    return _make_weighted_g_coef(d_fun=d_fun, degradation_fun=degradation_fun)


def _extract_mechanical_blocks(A_, gdof_sigma: int):
    """
    Blocks of the saddle-point system (unknown ordering ``[sigma; u]``):

        K_h = [[A_sigma, B_div^T],
               [B_div,     0    ]]

    ``HuZhangElasticAssembler`` assembles ``bmat([[M2, B2], [B2.T, None]])``, so
    ``A_sigma = M2`` is the (possibly damaged) stress operator ``A(d_h)``, and
    ``B_div = B2.T`` is the discrete divergence ``B``; ``B_div.T = B2``.
    """
    m = int(gdof_sigma)
    A_sigma = A_[:m, :m].tocsr()
    B_div = A_[m:, :m].tocsr()
    return m, A_sigma, B_div


def _diag_inv_stress_block(A_sigma, *, floor: float = 1e-30) -> np.ndarray:
    """Diagonal ``D^{-1} ≈ A(d_h)^{-1}`` used in the Schur approximation."""
    diag = np.asarray(A_sigma.diagonal(), dtype=float)
    diag = np.where(np.abs(diag) > floor, diag, 1.0)
    return 1.0 / diag


def _approximate_schur_spd(A_sigma, B_div, D_inv: np.ndarray):
    r"""
    SPD Schur complement used in preconditioners:

        S(d_h) = B\,A(d_h)^{-1}\,B^\top

    with ``A(d_h)^{-1}`` approximated by ``\mathrm{diag}(A(d_h))^{-1}`` (``D_inv``).
    This matches the sign convention where the minus in ``-B A^{-1} B^\top`` is
    absorbed so that ``S`` is positive definite under the discrete inf-sup condition.
    """
    # ``A_sigma`` is used only for its size; allow None (matrix-free path) and
    # derive m from D_inv instead.
    m = int(A_sigma.shape[0]) if A_sigma is not None else int(len(D_inv))
    return (B_div @ spdiags(D_inv, 0, m, m) @ B_div.T).tocsr()


def _schur_div_blocks(A_, m: int, mesh_id: int, form: str):
    """Return ``(B_div, B_div^T)`` for the Schur approximation.

    For the ``standard`` formulation ``B_div = A_[m:, :m]`` is d-independent
    (``B2 = TM^T B`` is geometric), so cache it per mesh and reuse across staggered
    iterations. For ``effective_stress`` it carries ``g(d)`` and is re-sliced each call.
    """
    if form == "standard":
        cached = _AUXSPACE_B_CACHE.get((mesh_id, m))
        if cached is not None:
            return cached
    B = A_[m:, :m].tocsr()
    BT = B.T.tocsr()
    if form == "standard":
        _AUXSPACE_B_CACHE[(mesh_id, m)] = (B, BT)
    return B, BT


def _schur_spd_from_blocks(B_div, BT, D_inv: np.ndarray):
    """``S = B diag(D_inv) B^T`` with the ``B @ diag`` step done as an O(nnz) column
    scaling (bit-identical to ``B @ spdiags(D_inv)``), leaving a single sparse product.
    """
    BD = B_div.copy()
    BD.data = BD.data * D_inv[BD.indices]
    return (BD @ BT).tocsr()


@dataclass
class KrylovInfo:
    """Convergence summary returned alongside the solution by every ``solve_*`` entry point.

    Attributes:
        solver: Tag identifying the method/preconditioner, e.g. ``"gmres-auxspace"``.
        niter: Number of Krylov iterations actually performed.
        converged: Whether the requested tolerance was met (``True`` for a zero RHS).
        residual_norm: Final residual norm (preconditioned ``||r||/||b||`` for GMRES paths,
            true ``||b - A x||`` for MINRES; ``nan`` when unavailable).
        atol: Absolute tolerance passed to the Krylov solver.
        rtol: Relative tolerance passed to the Krylov solver.
    """

    solver: str
    niter: int
    converged: bool
    residual_norm: float
    atol: float
    rtol: float


def as_scipy_csr(A):
    """Coerce a FEALPy/SciPy sparse matrix to ``scipy.sparse.csr_matrix``.

    Args:
        A: A matrix exposing ``to_scipy()`` (FEALPy) or ``tocsr()`` (SciPy), or an object
            already usable as-is.

    Returns:
        The matrix in CSR form (or ``A`` unchanged if it has neither hook).
    """
    from fracturex.utilfuc.matfree_elastic import MatrixFreeElasticOperator

    if isinstance(A, MatrixFreeElasticOperator):
        return A  # matrix-free: never materialize; GMRES/preconditioner use matvec only
    if hasattr(A, "to_scipy"):
        return A.to_scipy().tocsr()
    if hasattr(A, "tocsr"):
        return A.tocsr()
    return A


def make_ilu_preconditioner(A, *, drop_tol: float = 1e-4, fill_factor: float = 10.0):
    """Build an incomplete-LU preconditioner as a ``LinearOperator``.

    Args:
        A: System matrix (FEALPy or SciPy sparse); internally converted to CSC for ``spilu``.
        drop_tol: ILU drop tolerance (smaller -> denser, stronger factor).
        fill_factor: Upper bound on fill relative to ``A``.

    Returns:
        A ``scipy.sparse.linalg.LinearOperator`` applying ``(LU)^{-1}``, or ``None`` if the
        factorization fails (e.g. singular block).
    """
    A_ = as_scipy_csr(A)
    try:
        # spilu internally prefers CSC; convert explicitly to avoid warning/copy.
        ilu = spilu(A_.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
    except Exception:
        return None

    n = A_.shape[0]

    def matvec(x):
        return ilu.solve(np.asarray(x).reshape(-1))

    return LinearOperator((n, n), matvec=matvec, dtype=A_.dtype)


def make_amg_preconditioner(A):
    """Try to build an AMG preconditioner via pyamg, fallback to None."""
    A_ = as_scipy_csr(A)
    try:
        import pyamg

        ml = pyamg.smoothed_aggregation_solver(A_)
    except Exception:
        return None

    n = A_.shape[0]

    def matvec(x):
        return ml.solve(np.asarray(x).reshape(-1), tol=1e-12, maxiter=1, cycle="V")

    return LinearOperator((n, n), matvec=matvec, dtype=A_.dtype)


def _extract_niter_from_info(info) -> int:
    """Best-effort iteration count from a solver's heterogeneous ``info`` return.

    Handles dict (``niter``/``iter``/...), attribute-bearing objects, and SciPy's integer
    convergence flag (positive == iteration count). Returns 0 when nothing usable is found.
    """
    try:
        if isinstance(info, dict):
            for k in ("niter", "iter", "iterations", "num_iter"):
                if k in info:
                    return int(info[k])
        if hasattr(info, "niter"):
            return int(getattr(info, "niter"))
        if hasattr(info, "iterations"):
            return int(getattr(info, "iterations"))
        if isinstance(info, (int, np.integer)):
            iv = int(info)
            if iv > 0:
                return iv
    except Exception:
        pass
    return 0


def _extract_converged_from_info(info, *, rtol: float = 1e-8, bnorm: float = 1.0) -> bool:
    """Best-effort convergence flag from a solver's heterogeneous ``info`` return.

    Args:
        info: dict, integer SciPy flag (0 == converged), or object with a residual.
        rtol: Relative tolerance used when only a raw residual is available.
        bnorm: ``||b||`` for turning ``rtol`` into an absolute residual threshold.

    Returns:
        ``True`` if the solver is judged to have converged, else ``False``.
    """
    try:
        if isinstance(info, dict):
            for k in ("converged", "success"):
                if k in info:
                    return bool(info[k])
            if "info" in info:
                return int(info["info"]) == 0
            if "residual" in info:
                res = float(info["residual"])
                return res <= max(float(rtol) * max(float(bnorm), 1e-30), 1e-30)
        if isinstance(info, (int, np.integer)):
            return int(info) == 0
    except Exception:
        pass
    return False


def _krylov_stats(callback_residuals, solver: str, info, atol, rtol, *, bnorm: float = 1.0, cb_count_hint: int = 0) -> KrylovInfo:
    """Assemble a :class:`KrylovInfo` from callback residuals plus the solver ``info``.

    Args:
        callback_residuals: Per-iteration residual norms collected by the callback (may be empty).
        solver: Solver/preconditioner tag stored in the result.
        info: Raw convergence info from the Krylov routine.
        atol, rtol: Tolerances echoed into the result.
        bnorm: ``||b||`` used for convergence/residual interpretation.
        cb_count_hint: Fallback iteration count when neither residuals nor ``info`` give one.

    Returns:
        A populated :class:`KrylovInfo`.
    """
    niter = len(callback_residuals)
    if niter == 0:
        niter = _extract_niter_from_info(info)
    if niter == 0 and cb_count_hint > 0:
        niter = int(cb_count_hint)
    residual_norm = float(callback_residuals[-1]) if callback_residuals else float("nan")
    if callback_residuals:
        converged = _extract_converged_from_info(info, rtol=rtol, bnorm=bnorm)
    elif isinstance(info, dict) and "residual" in info:
        residual_norm = float(info["residual"])
        converged = _extract_converged_from_info(info, rtol=rtol, bnorm=bnorm)
    else:
        converged = _extract_converged_from_info(info, rtol=rtol, bnorm=bnorm)
    return KrylovInfo(
        solver=solver,
        niter=int(niter),
        converged=bool(converged),
        residual_norm=residual_norm,
        atol=float(atol),
        rtol=float(rtol),
    )


def _zero_rhs_stats(solver: str, atol: float, rtol: float) -> KrylovInfo:
    """Trivial :class:`KrylovInfo` for a zero right-hand side (solution is exactly 0)."""
    return KrylovInfo(
        solver=solver,
        niter=0,
        converged=True,
        residual_norm=0.0,
        atol=float(atol),
        rtol=float(rtol),
    )


def _is_zero_rhs(b, *, atol: float = 0.0) -> bool:
    """Return ``True`` when ``||b|| <= max(atol, 1e-30)`` so the solve can be short-circuited."""
    b_ = np.asarray(b, dtype=float).reshape(-1)
    tol = max(float(atol), 1e-30)
    return float(np.linalg.norm(b_)) <= tol


def _fealpy_krylov(name: str):
    """Return FEALPy krylov callable if available, else None."""
    # Prefer loading `fealpy/solver/<name>.py` directly: importing `fealpy.solver.*` executes
    # `fealpy/solver/__init__.py`, which may hard-require optional deps (e.g. pyamg) in some builds.
    try:
        import fealpy as _fp

        root = pathlib.Path(_fp.__file__).parent
        path = root / "solver" / f"{name}.py"
        if path.exists():
            mod_name = f"_fealpy_solver_{name}_shim"
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fn = getattr(mod, name, None)
            if fn is not None:
                return fn
    except Exception:
        pass

    try:
        if name == "gmres":
            import fealpy.solver.gmres as _fgmres  # type: ignore

            return _fgmres.gmres
        if name == "minres":
            import fealpy.solver.minres as _fminres  # type: ignore

            return _fminres.minres
    except Exception:
        return None


def _import_fealpy_gamg_solver():
    """Import FEALPy's GAMGSolver without going through `fealpy.solver` package __init__."""
    import importlib
    import types
    import fealpy as _fp

    # Fast path: standard package import (works on modern FEALPy layouts).
    try:
        mod = importlib.import_module("fealpy.solver.gamg_solver")
        if hasattr(mod, "GAMGSolver"):
            return getattr(mod, "GAMGSolver")
    except Exception:
        pass

    root = pathlib.Path(_fp.__file__).parent
    path = root / "solver" / "gamg_solver.py"
    if not path.exists():
        raise ImportError(f"Cannot locate fealpy GAMGSolver at {path}")
    # Fallback: load by file, but keep package context so `from ..backend import ...` works.
    pkg_name = "fealpy.solver"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(root / "solver")]  # mark as package path
        sys.modules[pkg_name] = pkg

    mod_name = "fealpy.solver.gamg_solver"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Invalid importlib spec for fealpy GAMGSolver")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_name
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, "GAMGSolver")


def _import_fealpy_csr_tensor():
    """Import FEALPy CSRTensor class."""
    try:
        from fealpy.sparse.csr_tensor import CSRTensor  # type: ignore

        return CSRTensor
    except Exception as ex:
        raise ImportError("Cannot import fealpy.sparse.csr_tensor.CSRTensor") from ex


def _call_fealpy_gmres(fgmres, A, b, *, M, restart: int, maxit: int, atol: float, rtol: float):
    """
    Call FEALPy gmres with a SciPy-like argument set, adapting to minor signature differences.
    """
    sig = None
    try:
        sig = inspect.signature(fgmres)
    except Exception:
        sig = None

    kwargs = {}
    if sig is not None:
        params = sig.parameters
        if "restart" in params:
            kwargs["restart"] = int(restart)
        if "maxit" in params:
            kwargs["maxit"] = int(maxit)
        elif "maxiter" in params:
            kwargs["maxiter"] = int(maxit)
        if "atol" in params:
            kwargs["atol"] = float(atol)
        if "rtol" in params:
            kwargs["rtol"] = float(rtol)
        elif "tol" in params and "rtol" not in params:
            kwargs["tol"] = float(rtol)
        if "M" in params and M is not None:
            kwargs["M"] = M
    else:
        kwargs = {"restart": int(restart), "maxit": int(maxit), "atol": float(atol), "rtol": float(rtol), "M": M}

    return fgmres(A, b, **kwargs)


def solve_lgmres_ilu(A, b, *, rtol: float = 1e-8, atol: float = 0.0, maxit: int = 200, drop_tol: float = 1e-4, fill_factor: float = 10.0):
    """Solve ``A x = b`` with LGMRES preconditioned by ILU.

    Args:
        A: System matrix (FEALPy or SciPy sparse, ``(n, n)``).
        b: Right-hand side, shape ``(n,)`` (flattened internally).
        rtol, atol: Relative/absolute convergence tolerances.
        maxit: Maximum iterations.
        drop_tol, fill_factor: ILU parameters forwarded to :func:`make_ilu_preconditioner`.

    Returns:
        ``(x, info)`` where ``x`` is the solution ``(n,)`` and ``info`` is a :class:`KrylovInfo`.
    """
    A_ = as_scipy_csr(A)
    b_ = np.asarray(b, dtype=float).reshape(-1)
    if _is_zero_rhs(b_, atol=atol):
        return np.zeros_like(b_), _zero_rhs_stats("lgmres", atol, rtol)
    M = make_ilu_preconditioner(A_, drop_tol=drop_tol, fill_factor=fill_factor)
    # scipy ``lgmres`` does NOT accept ``callback_type`` and its callback receives the
    # current solution vector (not the residual). Count outer iterations and compute the
    # final relative residual from x.
    n_calls = {"k": 0}

    def callback(_xk):
        n_calls["k"] += 1

    x, info = lgmres(A_, b_, M=M, atol=atol, rtol=rtol, maxiter=maxit, callback=callback)
    bnorm = max(float(np.linalg.norm(b_)), 1e-30)
    rrel = float(np.linalg.norm(A_ @ x - b_) / bnorm)
    return x, KrylovInfo(
        solver="lgmres-ilu",
        niter=max(int(n_calls["k"]), 1),
        converged=bool(info == 0 and rrel <= max(rtol * 10.0, 1e-6)),
        residual_norm=rrel,
        atol=atol,
        rtol=rtol,
    )


def solve_gmres_ilu(A, b, *, rtol: float = 1e-8, atol: float = 0.0, restart: int = 50, maxit: int = 200, drop_tol: float = 1e-4, fill_factor: float = 10.0):
    """Solve ``A x = b`` with restarted GMRES preconditioned by ILU.

    Args:
        A: System matrix ``(n, n)``.
        b: Right-hand side ``(n,)``.
        rtol, atol: Convergence tolerances.
        restart: GMRES restart length.
        maxit: Maximum (outer) iterations.
        drop_tol, fill_factor: ILU parameters.

    Returns:
        ``(x, info)`` with solution ``(n,)`` and :class:`KrylovInfo`.
    """
    A_ = as_scipy_csr(A)
    b_ = np.asarray(b, dtype=float).reshape(-1)
    if _is_zero_rhs(b_, atol=atol):
        return np.zeros_like(b_), _zero_rhs_stats("gmres", atol, rtol)
    M = make_ilu_preconditioner(A_, drop_tol=drop_tol, fill_factor=fill_factor)
    residuals = []

    def callback(rk):
        residuals.append(float(rk))

    x, info = gmres(
        A_,
        b_,
        M=M,
        restart=restart,
        maxiter=maxit,
        atol=atol,
        rtol=rtol,
        callback=callback,
        callback_type="pr_norm",
    )
    return x, _krylov_stats(residuals, "gmres", info, atol, rtol)


def solve_gmres_amg(A, b, *, rtol: float = 1e-8, atol: float = 0.0, restart: int = 50, maxit: int = 200):
    """Solve ``A x = b`` with restarted GMRES preconditioned by smoothed-aggregation AMG (pyamg).

    Falls back to no preconditioner if pyamg is unavailable.

    Args:
        A: System matrix ``(n, n)``.
        b: Right-hand side ``(n,)``.
        rtol, atol: Convergence tolerances.
        restart: GMRES restart length.
        maxit: Maximum (outer) iterations.

    Returns:
        ``(x, info)`` with solution ``(n,)`` and :class:`KrylovInfo`.
    """
    A_ = as_scipy_csr(A)
    b_ = np.asarray(b, dtype=float).reshape(-1)
    if _is_zero_rhs(b_, atol=atol):
        return np.zeros_like(b_), _zero_rhs_stats("gmres-amg", atol, rtol)
    M = make_amg_preconditioner(A_)
    residuals = []

    def callback(rk):
        residuals.append(float(rk))

    x, info = gmres(
        A_,
        b_,
        M=M,
        restart=restart,
        maxiter=maxit,
        atol=atol,
        rtol=rtol,
        callback=callback,
        callback_type="pr_norm",
    )
    return x, _krylov_stats(residuals, "gmres-amg", info, atol, rtol)


def solve_lgmres_amg(A, b, *, rtol: float = 1e-8, atol: float = 0.0, maxit: int = 200):
    """Solve ``A x = b`` with LGMRES preconditioned by smoothed-aggregation AMG (pyamg).

    Falls back to no preconditioner if pyamg is unavailable.

    Args:
        A: System matrix ``(n, n)``.
        b: Right-hand side ``(n,)``.
        rtol, atol: Convergence tolerances.
        maxit: Maximum iterations.

    Returns:
        ``(x, info)`` with solution ``(n,)`` and :class:`KrylovInfo`.
    """
    A_ = as_scipy_csr(A)
    b_ = np.asarray(b, dtype=float).reshape(-1)
    if _is_zero_rhs(b_, atol=atol):
        return np.zeros_like(b_), _zero_rhs_stats("lgmres-amg", atol, rtol)
    M = make_amg_preconditioner(A_)
    residuals = []

    def callback(rk):
        residuals.append(float(rk))

    x, info = lgmres(
        A_,
        b_,
        M=M,
        atol=atol,
        rtol=rtol,
        maxiter=maxit,
        callback=callback,
        callback_type="pr_norm",
    )
    return x, _krylov_stats(residuals, "lgmres-amg", info, atol, rtol)


def solve_minres_diag(A, b, *, rtol: float = 1e-8, atol: float = 0.0, maxit: int = 200):
    """Solve a symmetric system ``A x = b`` with unpreconditioned MINRES.

    The callback recomputes the true residual ``||b - A x_k||`` each iteration for an
    accurate convergence history.

    Args:
        A: Symmetric system matrix ``(n, n)``.
        b: Right-hand side ``(n,)``.
        rtol, atol: Convergence tolerances.
        maxit: Maximum iterations.

    Returns:
        ``(x, info)`` with solution ``(n,)`` and :class:`KrylovInfo`.
    """
    A_ = as_scipy_csr(A)
    b_ = np.asarray(b, dtype=float).reshape(-1)
    residuals = []

    def callback(xk):
        rk = b_ - A_ @ xk
        residuals.append(float(np.linalg.norm(rk)))

    x, info = minres(A_, b_, rtol=rtol, atol=atol, maxiter=maxit, callback=callback)
    return x, _krylov_stats(residuals, "minres", info, atol, rtol)


def solve_huzhang_block_gmres(
    A,
    b,
    *,
    gdof_sigma: int,
    rtol: float = 1e-8,
    atol: float = 0.0,
    restart: int = 50,
    maxit: int = 200,
    schur_drop_tol: float = 1e-4,
    schur_fill_factor: float = 10.0,
):
    """
    Block preconditioned GMRES for Hu-Zhang mixed system.

    Uses a block lower-triangular preconditioner with an ILU approximation
    for the scalar Schur complement on the displacement block.
    """
    A_ = as_scipy_csr(A)
    b_ = np.asarray(b, dtype=float).reshape(-1)
    if _is_zero_rhs(b_, atol=atol):
        return np.zeros_like(b_), _zero_rhs_stats("gmres-block", atol, rtol)

    m, M, B = _extract_mechanical_blocks(A_, gdof_sigma)
    D_inv = _diag_inv_stress_block(M)
    S = _approximate_schur_spd(M, B, D_inv)

    ilu_s = make_ilu_preconditioner(S, drop_tol=schur_drop_tol, fill_factor=schur_fill_factor)

    def solve_s(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        if ilu_s is not None:
            return np.asarray(ilu_s.matvec(x)).reshape(-1)
        try:
            return np.linalg.solve(S.toarray(), x)
        except Exception:
            return x

    def precondition(r):
        r = np.asarray(r, dtype=float).reshape(-1)
        r0 = r[:m]
        r1 = r[m:]

        u0 = D_inv * r0
        rhs1 = r1 - B @ u0
        u1 = solve_s(rhs1)
        u0 = u0 - D_inv * (B.T @ u1)
        return np.concatenate([u0, u1])

    P = LinearOperator(A_.shape, matvec=precondition, dtype=A_.dtype)
    residuals = []

    def callback(rk):
        residuals.append(float(rk))

    x, info = gmres(
        A_,
        b_,
        M=P,
        restart=restart,
        maxiter=maxit,
        atol=atol,
        rtol=rtol,
        callback=callback,
        callback_type="pr_norm",
    )
    return x, _krylov_stats(residuals, "gmres-block", info, atol, rtol)


def solve_huzhang_block_krylov(
    A,
    b,
    *,
    gdof_sigma: int,
    solver: str = "gmres",
    rtol: float = 1e-8,
    atol: float = 0.0,
    restart: int = 50,
    maxit: int = 200,
    schur_drop_tol: float = 1e-4,
    schur_fill_factor: float = 10.0,
):
    """
    Hu-Zhang mixed system with block preconditioning.

    Parameters
    ----------
    solver : {'gmres', 'minres'}
        GMRES uses an unsymmetric block-triangular preconditioner.
        MINRES uses a symmetric block-diagonal variant.
    """
    A_ = as_scipy_csr(A)
    b_ = np.asarray(b, dtype=float).reshape(-1)
    s = solver.lower()
    if _is_zero_rhs(b_, atol=atol):
        return np.zeros_like(b_), _zero_rhs_stats(f"{s}-block", atol, rtol)

    m, Mblk, B = _extract_mechanical_blocks(A_, gdof_sigma)
    D_inv = _diag_inv_stress_block(Mblk)
    S = _approximate_schur_spd(Mblk, B, D_inv)
    diagS = np.asarray(S.diagonal(), dtype=float)
    diagS = np.where(np.abs(diagS) > 1e-30, np.abs(diagS), 1.0)
    D_s_inv = 1.0 / diagS
    ilu_s = make_ilu_preconditioner(S, drop_tol=schur_drop_tol, fill_factor=schur_fill_factor)
    amg_s = make_amg_preconditioner(S)

    def solve_s(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        if amg_s is not None:
            return np.asarray(amg_s.matvec(x)).reshape(-1)
        if ilu_s is not None:
            return np.asarray(ilu_s.matvec(x)).reshape(-1)
        return x

    def precondition_gmres(r):
        r = np.asarray(r, dtype=float).reshape(-1)
        r0 = r[:m]
        r1 = r[m:]

        u0 = D_inv * r0
        rhs1 = r1 - B @ u0
        u1 = solve_s(rhs1)
        u0 = u0 - D_inv * (B.T @ u1)
        return np.concatenate([u0, u1])

    def precondition_minres(r):
        r = np.asarray(r, dtype=float).reshape(-1)
        r0 = r[:m]
        r1 = r[m:]
        # Symmetric block-diagonal preconditioner for MINRES.
        u0 = D_inv * r0
        u1 = D_s_inv * r1
        return np.concatenate([u0, u1])

    def scipy_minres_with_compat(Ax, bx, Mx, cb):
        """Handle SciPy minres signature differences across versions."""
        try:
            return minres(Ax, bx, M=Mx, rtol=rtol, atol=atol, maxiter=maxit, callback=cb)
        except TypeError:
            try:
                return minres(Ax, bx, M=Mx, rtol=rtol, maxiter=maxit, callback=cb)
            except TypeError:
                return minres(Ax, bx, M=Mx, tol=rtol, maxiter=maxit, callback=cb)

    residuals = []

    if s == "gmres":
        P = LinearOperator(A_.shape, matvec=precondition_gmres, dtype=A_.dtype)

        def callback(rk):
            residuals.append(float(rk))

        fgmres = _fealpy_krylov("gmres")
        if fgmres is not None:
            try:
                x, info = _call_fealpy_gmres(fgmres, A_, b_, M=P, restart=restart, maxit=maxit, atol=atol, rtol=rtol)
            except TypeError:
                fgmres = None
        if fgmres is None:
            x, info = gmres(
                A_,
                b_,
                M=P,
                restart=restart,
                maxiter=maxit,
                atol=atol,
                rtol=rtol,
                callback=callback,
                callback_type="pr_norm",
            )
        return x, _krylov_stats(residuals, "gmres-block", info, atol, rtol)

    if s == "minres":
        P = LinearOperator(A_.shape, matvec=precondition_minres, dtype=A_.dtype)

        def callback(xk):
            rk = b_ - A_ @ np.asarray(xk).reshape(-1)
            residuals.append(float(np.linalg.norm(rk)))

        fminres = _fealpy_krylov("minres")
        if fminres is not None:
            try:
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    x, info = fminres(A_, b_, M=P, rtol=rtol, maxit=maxit)
                x = np.asarray(x, dtype=float).reshape(-1)
                if not np.isfinite(x).all():
                    raise FloatingPointError("FEALPy minres produced non-finite values")
            except Exception:
                x, info = scipy_minres_with_compat(A_, b_, P, callback)
        else:
            x, info = scipy_minres_with_compat(A_, b_, P, callback)
        return x, _krylov_stats(residuals, "minres-block", info, atol, rtol)

    raise ValueError(f"Unknown solver '{solver}', expected 'gmres' or 'minres'.")


def _infer_displacement_scalar_order(vspace) -> int:
    """Polynomial degree of the scalar base space inside ``TensorFunctionSpace`` (displacement)."""
    ss = getattr(vspace, "scalar_space", None)
    if ss is None or not hasattr(ss, "p"):
        raise RuntimeError("Cannot infer order: vspace.scalar_space / .p is missing.")
    return int(ss.p)


def _resolve_fast_schur_precond(schur_precond: str, order_p: int) -> str:
    """
    Map user-facing ``schur_precond`` to an internal Schur strategy.

    Aligns with ``huzhang_fast_solver.HZFEMFastSolve`` variants:

    - ``gs_amg_gs`` (1): forward GS on ``S``, P1 coarse MG per displacement block, backward GS.
    - ``coarse_amg_halfgs`` (2): P1 coarse from full ``r1``, then ``+ 0.5 *`` one symmetric GS sweep (surrogate for vertex patch).
    - ``cheb_amg_cheb`` (3): Chebyshev / coarse / Chebyshev (needs ``diag(S)^{-1}`` spectrum estimate).

    ``auto``: use (1) when ``order_p < 6``, else (2).
    """
    s = str(schur_precond).strip().lower()
    if s in ("auto", ""):
        return "gs_amg_gs" if int(order_p) < 6 else "coarse_amg_halfgs"
    aliases = {
        "1": "gs_amg_gs",
        "gs": "gs_amg_gs",
        "gs_amg_gs": "gs_amg_gs",
        "2": "coarse_amg_halfgs",
        "coarse": "coarse_amg_halfgs",
        "coarse_amg_halfgs": "coarse_amg_halfgs",
        "pre_of_s0": "coarse_amg_halfgs",
        "3": "cheb_amg_cheb",
        "cheb": "cheb_amg_cheb",
        "cheb_amg_cheb": "cheb_amg_cheb",
    }
    if s not in aliases:
        raise ValueError(
            f"Unknown schur_precond={schur_precond!r}; use 'auto', 'gs_amg_gs', 'coarse_amg_halfgs', "
            f"'cheb_amg_cheb' (or aliases 1/2/3, gs, coarse, cheb)."
        )
    return aliases[s]


def _fast_cached_schur(mesh, m, schur_mode, form, Mblk, B, D_inv, interval):
    """Schur approximation S ≈ B diag(A)^-1 B^T, reused for ``interval`` solves.

    Inputs:
        mesh, m, schur_mode, form: cache key parts (per mesh / block size / mode).
        Mblk, B, D_inv: blocks of the CURRENT A (used only when (re)building).
        interval: rebuild every ``interval`` calls (1 = rebuild every call).
    Output:
        scipy CSR Schur matrix (possibly from an earlier, slightly stale d).
    """
    interval = max(int(interval), 1)
    key = (id(mesh), int(m), schur_mode, form, "S")
    c = _FAST_PRECOND_CACHE.get(key)
    if c is not None and int(c["interval"]) == interval and int(c["calls"]) < interval:
        c["calls"] = int(c["calls"]) + 1
        _auxspace_log(f"fast Schur reuse {c['calls']}/{interval}")
        return c["S"]
    S = _approximate_schur_spd(Mblk, B, D_inv)
    _FAST_PRECOND_CACHE[key] = {"S": S, "interval": interval, "calls": 1}
    return S


def _fast_cached_coarse_ml(mesh, q, form, weighted_aux, damage, state, d_fun, degradation_fun, interval):
    """P1 coarse-diffusion pyamg hierarchy, reused for ``interval`` solves.

    The weighted (``g(d)``) coarse operator is otherwise reassembled + re-setup on
    EVERY solve, which dominates fast-path cost. Reusing a slightly stale hierarchy is
    a valid preconditioner (affects niter only). Unweighted path stays on its own
    mesh-level cache (already d-independent).
    """
    interval = max(int(interval), 1)
    key = (id(mesh), int(q), form, int(bool(weighted_aux)), "ml")
    c = _FAST_PRECOND_CACHE.get(key)
    if c is not None and int(c["interval"]) == interval and int(c["calls"]) < interval:
        c["calls"] = int(c["calls"]) + 1
        _auxspace_log(f"fast coarse-ml reuse {c['calls']}/{interval}")
        return c["ml"]
    coef = _make_coarse_diffusion_coef(
        formulation=form,
        weighted_aux=weighted_aux,
        damage=damage,
        state=state,
        d_fun=d_fun,
        degradation_fun=degradation_fun,
    )
    if coef is not None:
        ml = _assemble_p1_diffusion_pyamg(mesh, int(q), coef=coef)
    else:
        ml = _get_or_build_p1_isotropic_pyamg(mesh, int(q))
    _FAST_PRECOND_CACHE[key] = {"ml": ml, "interval": interval, "calls": 1}
    return ml


def solve_huzhang_block_gmres_fast(
    A,
    b,
    *,
    gdof_sigma: int,
    vspace,
    return_preconditioner: bool = False,
    rtol: float = 1e-8,
    atol: float = 0.0,
    restart: int = 20,
    maxit: int = 200,
    q: int = 3,
    precond_rebuild_interval: int = 1,
    schur_precond: str = "auto",
    order_p: Optional[int] = None,
    gs_iterations: int = 2,
    second_smooth_weight: float = 0.5,
    cheb_degree: int = 2,
    cheb_lower_ratio: float = 0.1,
    cheb_power_iters: int = 10,
    epsilon_diag: float = 1e-30,
    verbose_setup: bool = False,
    elastic_formulation: str = "standard",
    weighted_aux: bool = True,
    damage=None,
    state=None,
    d_fun=None,
    degradation_fun=None,
):
    """
    GMRES for the Hu-Zhang saddle-point system with Schur preconditioners aligned with the
    three ``pre_of_S`` variants in ``huzhang_fast_solver.HZFEMFastSolve``.

    Uses the same block layout and Schur approximation as
    :func:`solve_huzhang_block_gmres_auxspace` (``S(d_h) \\approx B\\,\\mathrm{diag}(A)^{-1}B^\\top``).

    **Schur strategies** (``schur_precond``):

    - ``gs_amg_gs`` / ``1``: forward Gauss–Seidel on ``S``, P1 coarse MG per displacement block,
      backward GS (recommended for **low order**, default when ``auto`` and ``order_p < 6``).
    - ``coarse_amg_halfgs`` / ``2``: coarse MG from full ``r1``, then add ``second_smooth_weight``
      times one symmetric GS sweep (surrogate for the original ``+ 0.5 * vertex_pre``; for **high
      order**, default when ``auto`` and ``order_p >= 6``).
    - ``cheb_amg_cheb`` / ``3``: Chebyshev / coarse / Chebyshev using a power estimate of
      ``lambda_max(diag(S)^{-1} S)``.

    **``auto``** (default): ``gs_amg_gs`` if ``order_p < 6``, else ``coarse_amg_halfgs``.

    **``elastic_formulation``**: ``standard`` weights the P1 coarse Laplacian by ``g(d)`` when
    ``weighted_aux=True`` (matches ``1/g(d)`` on the stress block); ``effective_stress`` uses an
    unweighted coarse Laplacian (``g(d)`` on B only, via the Schur block from ``A``).

    **``order_p``**: FE degree used only for ``auto``. If ``None``, uses ``vspace.scalar_space.p``
    (displacement scalar order, i.e. ``HuZhangDiscretization.u_space_order``). To key off stress
    degree instead, pass e.g. ``order_p=discr.p``.

    On every call, ``M``, ``B``, ``D_inv``, and ``S`` are rebuilt from the current ``A`` (variable
    coefficients). P1 ``PI_s`` and pyamg hierarchy on the mesh Laplacian are cached.
    """
    if not FEALPY_AVAILABLE:
        raise RuntimeError(f"FEALPy required for fast Hu-Zhang solver (FEALPY_AVAILABLE={FEALPY_AVAILABLE}).")

    form = _normalize_elastic_formulation(elastic_formulation)
    order_p_val = int(order_p) if order_p is not None else _infer_displacement_scalar_order(vspace)
    schur_mode = _resolve_fast_schur_precond(schur_precond, order_p_val)
    if schur_mode in ("gs_amg_gs", "coarse_amg_halfgs") and not PYAMG_AVAILABLE:
        raise RuntimeError(
            "Schur modes gs_amg_gs and coarse_amg_halfgs require pyamg (Gauss–Seidel relaxation). "
            f"PYAMG_AVAILABLE={PYAMG_AVAILABLE}"
        )

    A_ = as_scipy_csr(A)
    b_ = np.asarray(b, dtype=float).reshape(-1)
    _auxspace_log(
        f"enter solve_huzhang_block_gmres_fast: n={A_.shape[0]}, nnz={getattr(A_, 'nnz', 'n/a')}, "
        f"gdof_sigma={gdof_sigma}, schur_mode={schur_mode}, order_p={order_p_val}, "
        f"elastic_formulation={form}"
    )
    if _is_zero_rhs(b_, atol=atol):
        tag = f"gmres-fast-{schur_mode}"
        return np.zeros_like(b_), _zero_rhs_stats(tag, atol, rtol)

    from fracturex.utilfuc.matfree_elastic import MatrixFreeElasticOperator

    if isinstance(A_, MatrixFreeElasticOperator):
        # Matrix-free: pull divergence block + (approx) stress diagonal from the
        # operator instead of slicing/.diagonal() an assembled matrix. Mblk stays
        # None — the Schur builder only needs its size (derived from D_inv).
        m = A_.gdof_sigma
        B = A_.B_div
        BT = A_.B_div_T
        D_inv = A_.diag_inv_sigma()
        Mblk = None
    else:
        m, Mblk, B = _extract_mechanical_blocks(A_, gdof_sigma)
        BT = B.T.tocsr()
        D_inv = _diag_inv_stress_block(Mblk)
    S = _fast_cached_schur(
        vspace.mesh, gdof_sigma, schur_mode, form, Mblk, B, D_inv, precond_rebuild_interval
    )

    S_lmax: Optional[float] = None
    S_dinv: Optional[np.ndarray] = None
    if schur_mode == "cheb_amg_cheb":
        d_s = np.asarray(S.diagonal(), dtype=np.float64)
        dmax = float(np.max(np.abs(d_s)))
        eps = max(float(epsilon_diag) * dmax, float(epsilon_diag))
        S_dinv = 1.0 / np.where(np.abs(d_s) > eps, d_s, 1.0)
        S_lmax = _estimate_lambda_max_dinv_s_numpy(S, S_dinv, iters=int(cheb_power_iters))
        if verbose_setup:
            print(
                f"[gmres-fast] Schur S shape={S.shape} nnz={S.nnz}, lambda_max(D^-1 S)~{S_lmax:.4g}",
                flush=True,
            )
    elif verbose_setup:
        print(f"[gmres-fast] Schur S shape={S.shape} nnz={S.nnz}, mode={schur_mode}", flush=True)

    mesh = vspace.mesh
    cached_pi = _get_or_build_auxspace_pi_operators(mesh, vspace, int(q))
    gdim = int(cached_pi["gdim"])
    sgdof = int(cached_pi["sgdof"])
    PI_s = cached_pi["PI_s"]
    PI_s_T = cached_pi["PI_s_T"]

    ml = _fast_cached_coarse_ml(
        mesh, int(q), form, weighted_aux, damage, state, d_fun, degradation_fun,
        precond_rebuild_interval,
    )

    nu = int(S.shape[0])
    if nu != gdim * sgdof:
        raise RuntimeError(
            f"Schur size mismatch: S has {nu} rows but gdim*sgdof = {gdim}*{sgdof} = {gdim * sgdof} "
            "(check gdof_sigma and vspace)."
        )

    n_gs = max(1, int(gs_iterations))

    def _coarse_add_residual(res: np.ndarray, e_out: np.ndarray) -> None:
        for i in range(gdim):
            i0 = i * sgdof
            i1 = (i + 1) * sgdof
            crm = PI_s_T @ res[i0:i1]
            X = ml.solve(np.asarray(crm, dtype=np.float64).reshape(-1), maxiter=1, cycle="V")
            e_out[i0:i1] = e_out[i0:i1] + (PI_s @ np.asarray(X, dtype=np.float64).reshape(-1))

    def _coarse_from_rhs(r1v: np.ndarray) -> np.ndarray:
        out = np.zeros_like(r1v)
        for i in range(gdim):
            i0 = i * sgdof
            i1 = (i + 1) * sgdof
            crm = PI_s_T @ r1v[i0:i1]
            X = ml.solve(np.asarray(crm, dtype=np.float64).reshape(-1), maxiter=1, cycle="V")
            out[i0:i1] = PI_s @ np.asarray(X, dtype=np.float64).reshape(-1)
        return out

    if schur_mode == "gs_amg_gs":

        def pre_of_S(r1: np.ndarray) -> np.ndarray:
            r1 = np.asarray(r1, dtype=np.float64).reshape(-1)
            e1 = np.zeros_like(r1)
            gauss_seidel(S, e1, r1, iterations=n_gs, sweep="forward")
            r2 = r1 - S @ e1
            _coarse_add_residual(r2, e1)
            gauss_seidel(S, e1, r1, iterations=n_gs, sweep="backward")
            return e1

    elif schur_mode == "coarse_amg_halfgs":

        def pre_of_S(r1: np.ndarray) -> np.ndarray:
            r1 = np.asarray(r1, dtype=np.float64).reshape(-1)
            e1 = _coarse_from_rhs(r1)
            tmp = np.zeros_like(r1)
            gauss_seidel(S, tmp, r1, iterations=1, sweep="forward")
            gauss_seidel(S, tmp, r1, iterations=1, sweep="backward")
            w = float(second_smooth_weight)
            return e1 + w * tmp

    else:

        def pre_of_S(r1: np.ndarray) -> np.ndarray:
            assert S_dinv is not None and S_lmax is not None
            r1 = np.asarray(r1, dtype=np.float64).reshape(-1)
            e1 = np.zeros_like(r1)
            e1 = _chebyshev_smooth_numpy(
                S,
                e1,
                r1,
                S_dinv,
                float(S_lmax),
                degree=int(cheb_degree),
                lower_ratio=float(cheb_lower_ratio),
            )
            r2 = r1 - S @ e1
            _coarse_add_residual(r2, e1)
            e1 = _chebyshev_smooth_numpy(
                S,
                e1,
                r1,
                S_dinv,
                float(S_lmax),
                degree=int(cheb_degree),
                lower_ratio=float(cheb_lower_ratio),
            )
            return e1

    def fast_preconditioner(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        r0 = r[:m]
        r1 = r[m:].copy()
        e0 = D_inv * r0
        r1 -= B @ e0
        e1 = pre_of_S(r1)
        e0 = e0 + (BT @ e1) * D_inv
        return np.concatenate([e0, -e1])

    P = LinearOperator(A_.shape, matvec=fast_preconditioner, dtype=A_.dtype)
    if return_preconditioner:
        # Expose the (right) block preconditioner operator P^{-1} itself (no GMRES run),
        # e.g. for spectral analysis of P^{-1}K (precond_spectrum.py). Returns (P, None).
        return P, None
    residuals: list = []
    cb_count = {"n": 0}
    bnorm = float(np.linalg.norm(b_))
    bnorm = max(bnorm, 1e-30)
    true_every = int(os.environ.get("FRACTUREX_FAST_TRUE_RES_EVERY", "20"))
    true_every = max(true_every, 1)
    stats_tag = f"gmres-fast-{schur_mode}"

    def _true_relres(xk) -> float:
        xk = np.asarray(xk, dtype=float).reshape(-1)
        rk = b_ - A_ @ xk
        return float(np.linalg.norm(rk) / bnorm)

    def callback_x(xk):
        cb_count["n"] += 1
        if cb_count["n"] % true_every == 0:
            _auxspace_log(f"gmres-fast x-callback #{cb_count['n']}: true_relres={_true_relres(xk):.3e}")

    def callback_pr(rk):
        residuals.append(float(rk))
        cb_count["n"] += 1
        every = int(os.environ.get("FRACTUREX_FAST_CBLOG_EVERY", "5"))
        every = max(every, 1)
        if cb_count["n"] % every == 0:
            _auxspace_log(f"gmres-fast pr_norm #{cb_count['n']}: {float(rk):.3e}")

    fgmres = _fealpy_krylov("gmres")
    use_scipy = fgmres is None
    if not use_scipy:
        try:
            x, info = _call_fealpy_gmres(
                fgmres,
                A_,
                b_,
                M=P,
                restart=int(restart),
                maxit=int(maxit),
                atol=float(atol),
                rtol=float(rtol),
            )
        except TypeError:
            use_scipy = True
    if use_scipy:
        try:
            x, info = gmres(
                A_,
                b_,
                M=P,
                restart=int(restart),
                maxiter=int(maxit),
                atol=float(atol),
                rtol=float(rtol),
                callback=callback_x,
                callback_type="x",
            )
        except TypeError:
            x, info = gmres(
                A_,
                b_,
                M=P,
                restart=int(restart),
                maxiter=int(maxit),
                atol=float(atol),
                rtol=float(rtol),
                callback=callback_pr,
                callback_type="pr_norm",
            )

    _auxspace_log(f"gmres-fast finished: info={info!r}")
    try:
        tr_final = _true_relres(x)
        _auxspace_log(f"gmres-fast final true_relres={tr_final:.3e}")
        conv = _extract_converged_from_info(info)
        rt = max(float(rtol), 1e-15)
        # SciPy may return non-zero `info` (e.g. maxiter) while the true relative residual is
        # already tiny; avoid noisy warnings in that case.
        if tr_final > max(rt * 100.0, 1e-5):
            print(
                "[gmres-fast warning] "
                f"large residual: info={info!r}, true_relres={tr_final:.3e}, rtol={rtol:.1e}, "
                f"restart={restart}, maxit={maxit}, schur_mode={schur_mode}",
                flush=True,
            )
        elif (not conv) and tr_final > rt * 10.0:
            print(
                "[gmres-fast warning] "
                f"non-converged (info={info!r}) and true_relres={tr_final:.3e} > 10*rtol={10*rt:.1e}; "
                f"restart={restart}, maxit={maxit}, schur_mode={schur_mode}",
                flush=True,
            )
        elif not conv:
            _auxspace_log(
                f"gmres-fast: solver info={info!r} (e.g. maxiter) but true_relres={tr_final:.3e} "
                f"<= 10*rtol (acceptable for engineering rtol={rtol:.1e})"
            )
    except Exception:
        pass
    return x, _krylov_stats(residuals, stats_tag, info, atol, rtol, cb_count_hint=cb_count["n"])


def solve_huzhang_block_gmres_auxspace(
    A,
    b,
    *,
    gdof_sigma: int,
    vspace,
    rtol: float = 1e-8,
    atol: float = 0.0,
    restart: int = 20,
    maxit: int = 200,
    sstep: int = 3,
    theta: float = 0.25,
    q: int = 3,
    smoother_steps: int = 2,
    schur_rebuild_interval: int = 1,
    coarse_rebuild_interval: int = 1,
    weighted_aux: bool = False,
    elastic_formulation: str = "standard",
    damage=None,
    state=None,
    d_fun=None,
    degradation_fun=None,
    schur_ilu_in_precond: bool = False,
):
    """
    Hu-Zhang mixed system solved by GMRES with an auxiliary-space Schur preconditioner.

    For a fixed phase-field iterate ``d_h``, the mechanical block system is

    .. math:: K_h(d_h) = \\begin{bmatrix} A(d_h) & B^\\top \\\\ B & 0 \\end{bmatrix},

    with ``A(d_h)`` the weighted stress operator and ``B`` the discrete divergence.
    The SPD Schur complement is ``S(d_h) = B\\,A(d_h)^{-1}B^\\top`` (sign absorbed).
    This module approximates ``S`` by ``B\\,\\mathrm{diag}(A)^{-1}B^\\top`` and builds

    - ``B_A(d_h) \\approx A(d_h)^{-1}`` (diagonal scaling on the stress block; auxiliary
      P1 GAMG on ``S_coarse`` refines ``B_A`` for the ``standard`` formulation);
    - ``B_S(d_h) \\approx S(d_h)^{-1}`` (GS sweeps on ``S_hat``, optional ILU).

    Block-triangular application (see ``gmres_preconditioner``):
    ``[e_sigma; e_u] = [B_A r_sigma + B_A B^T B_S r_u; -B_S r_u]`` (up to the diagonal ``B_A`` shortcut).

    **Degradation placement** (must match ``HuZhangElasticAssembler.formulation``):

    - ``standard``: g(d) on stress block M (in ``A``, integrator ``1/g``); optional coarse diffusion
      weighted by ``g(d)`` from ``damage.coef_bary`` when ``weighted_aux=True`` (Chen et
      al. 2017 §5: g-weighted auxiliary elasticity / vector Poisson on the Schur complement —
      same g(d) as the stress block, no floor).
    - ``effective_stress``: g(d) on coupling block B (in ``A``); coarse diffusion is always
      unweighted — damage enters only via the Schur block extracted from ``A``.

    Parameters
    ----------
    gdof_sigma:
        Number of sigma unknowns, i.e. the first block size.
    vspace:
        TensorFunctionSpace for displacement, used to access mesh and coarse dofs.
    smoother_steps:
        Number of Gauss-Seidel iterations in each forward/backward sweep (default 2 for speed).
    elastic_formulation:
        ``"standard"`` or ``"effective_stress"`` (same as elastic assembler).
    damage, state:
        Used with ``weighted_aux`` on the standard path for ``coef_bary``-consistent coarse weights.
    """
    if not FEALPY_AVAILABLE:
        raise RuntimeError(f"Aux-space preconditioner requires FEALPy core modules. FEALPy available: {FEALPY_AVAILABLE}")
    if not PYAMG_AVAILABLE:
        raise RuntimeError(
            "Aux-space preconditioner currently uses pyamg's Gauss-Seidel relaxation on the Schur block. "
            f"Please install pyamg (PYAMG available: {PYAMG_AVAILABLE})."
        )

    form = _normalize_elastic_formulation(elastic_formulation)
    A_ = as_scipy_csr(A)
    b_ = np.asarray(b, dtype=float).reshape(-1)
    stress_weighted_coarse = _coarse_diffusion_uses_stress_weight(form, weighted_aux)
    _auxspace_log(
        f"enter solve_huzhang_block_gmres_auxspace: n={A_.shape[0]}, nnz={getattr(A_, 'nnz', 'n/a')}"
        f", gdof_sigma={gdof_sigma}, restart={restart}, maxit={maxit}, weighted_aux={weighted_aux}"
        f", elastic_formulation={form}, stress_weighted_coarse={stress_weighted_coarse}"
        f", schur_ilu_in_precond={schur_ilu_in_precond}"
    )
    if _is_zero_rhs(b_, atol=atol):
        return np.zeros_like(b_), _zero_rhs_stats("gmres-auxspace", atol, rtol)

    m = int(gdof_sigma)
    # D^{-1} ≈ diag(A_sigma)^{-1}: read the stress-block diagonal directly off A_;
    # only the diagonal is needed for the Schur approximation, so no full M2 slice.
    diag = np.asarray(A_.diagonal(), dtype=float)[:m]
    diag = np.where(np.abs(diag) > 1e-30, diag, 1.0)
    D_inv = 1.0 / diag
    # B = B_div = A_[m:, :m]; cached per mesh for `standard` (d-independent).
    from fracturex.utilfuc.matfree_elastic import MatrixFreeElasticOperator

    if isinstance(A_, MatrixFreeElasticOperator):
        B, BT = A_.B_div, A_.B_div_T   # divergence block carried by the MF operator
    else:
        B, BT = _schur_div_blocks(A_, m, id(vspace.mesh), form)
    S_csr = _schur_spd_from_blocks(B, BT, D_inv)
    _auxspace_log(f"built Schur S ≈ B A^{-1} B^T: nnz={S_csr.nnz}, shape={S_csr.shape}")

    mesh = vspace.mesh
    mesh_id = id(mesh)
    cached = _get_or_build_auxspace_pi_operators(mesh, vspace, q)
    gdim = int(cached["gdim"])
    sgdof = int(cached["sgdof"])
    cspace = cached["cspace"]
    PI_s = cached["PI_s"]
    PI_s_T = cached["PI_s_T"]

    coef_aux = _make_coarse_diffusion_coef(
        formulation=form,
        weighted_aux=weighted_aux,
        damage=damage,
        state=state,
        d_fun=d_fun,
        degradation_fun=degradation_fun,
    )

    _auxspace_log("assembling auxiliary coarse diffusion S_coarse ...")
    bform = BilinearForm(cspace)
    if coef_aux is None:
        bform.add_integrator(ScalarDiffusionIntegrator(q=q))
    else:
        bform.add_integrator(ScalarDiffusionIntegrator(coef=coef_aux, q=q))
    S_coarse = bform.assembly()
    bc = DirichletBC(cspace, bm.zeros(S_coarse.shape[0], dtype=S_coarse.dtype))
    S_coarse, _ = bc.apply(S_coarse, bm.zeros(S_coarse.shape[0], dtype=S_coarse.dtype))
    S_coarse_scipy = as_scipy_csr(S_coarse)
    _auxspace_log(f"S_coarse scipy: nnz={S_coarse_scipy.nnz}, shape={S_coarse_scipy.shape}")

    coarse_interval = max(int(coarse_rebuild_interval), 1)
    coarse_key = (mesh_id, int(q), form, int(stress_weighted_coarse))
    coarse_cached = _AUXSPACE_COARSE_CACHE.get(coarse_key, None)
    coarse_reuse_ok = (
        coarse_cached is not None
        and int(coarse_cached.get("interval", 1)) == coarse_interval
        and int(coarse_cached.get("calls_since_build", 0)) < coarse_interval
    )
    if coarse_reuse_ok:
        P_coarse = coarse_cached["P_coarse"]
        calls = int(coarse_cached.get("calls_since_build", 0)) + 1
        coarse_cached["calls_since_build"] = calls
        _auxspace_log(f"coarse GAMG reuse: call {calls}/{coarse_interval}")
    else:
        if not PYAMG_AVAILABLE:
            raise RuntimeError(
                "Aux-space coarse correction requires pyamg (Gauss–Seidel + coarse AMG). "
                "Install with: pip install pyamg"
            )
        import pyamg

        _auxspace_log("building pyamg hierarchy on S_coarse (smoothed aggregation) ...")
        ml_coarse = pyamg.smoothed_aggregation_solver(S_coarse_scipy)
        n0 = int(S_coarse_scipy.shape[0])

        def _coarse_vcycle(r, _ml=ml_coarse):
            rr = np.asarray(r, dtype=float).reshape(-1)
            x = _ml.solve(rr, maxiter=1, cycle="V")
            return np.asarray(x, dtype=float).reshape(-1)

        P_coarse = LinearOperator((n0, n0), matvec=_coarse_vcycle, dtype=S_coarse_scipy.dtype)
        _AUXSPACE_COARSE_CACHE[coarse_key] = {
            "P_coarse": P_coarse,
            "interval": coarse_interval,
            "calls_since_build": 1,
        }
        _auxspace_log("pyamg coarse hierarchy setup done")
    # Temporarily disable the consistency augmentation term (+ theta*C_h):
    # use plain Schur approximation to reduce setup/assembly cost.
    S_hat = S_csr.tocsr()
    _auxspace_log(f"S_hat = S (C_h term disabled): nnz={S_hat.nnz}")

    # Optional Schur-side preconditioner reuse across consecutive solves.
    # This is an approximation when d/load changes; keep interval=1 for strict rebuild.
    rebuild_interval = max(int(schur_rebuild_interval), 1)
    schur_key = (mesh_id, int(m), float(theta), form, int(bool(schur_ilu_in_precond)))
    # The Schur-side ILU is only applied inside the preconditioner when
    # `schur_ilu_in_precond=True`. When disabled (the default), skip the spilu
    # factorization entirely — it can dominate setup cost for nothing.
    if not schur_ilu_in_precond:
        ilu_s = None
        _auxspace_log("Schur ILU skipped (schur_ilu_in_precond=False)")
    else:
        schur_cached = _AUXSPACE_SCHUR_CACHE.get(schur_key, None)
        reuse_ok = (
            schur_cached is not None
            and int(schur_cached.get("interval", 1)) == rebuild_interval
            and int(schur_cached.get("calls_since_build", 0)) < rebuild_interval
        )
        if reuse_ok:
            ilu_s = schur_cached.get("ilu_s", None)
            calls = int(schur_cached.get("calls_since_build", 0)) + 1
            schur_cached["calls_since_build"] = calls
            _auxspace_log(f"Schur ILU reuse: call {calls}/{rebuild_interval}")
        else:
            _auxspace_log("building Schur ILU preconditioner on S_hat ...")
            ilu_s = make_ilu_preconditioner(S_hat, drop_tol=1e-4, fill_factor=10.0)
            _auxspace_log("Schur ILU build done" if ilu_s is not None else "Schur ILU build failed (None)")
            _AUXSPACE_SCHUR_CACHE[schur_key] = {
                "ilu_s": ilu_s,
                "interval": rebuild_interval,
                "calls_since_build": 1,
            }

    p_apply_count = {"n": 0}

    def gmres_preconditioner(r):
        t0p = time.perf_counter()
        p_apply_count["n"] += 1
        r = np.asarray(r, dtype=float).reshape(-1)
        r1 = r[m:]

        # B_A(d_h) ≈ A(d_h)^{-1} on stress residual (diagonal part)
        u0 = r[:m] * D_inv
        u1 = np.zeros_like(r1)
        r1_local = r1 - B @ u0

        # B_S(d_h) ≈ S(d_h)^{-1}: GS + auxiliary coarse correction on S_hat ≈ S(d_h)
        gauss_seidel(S_hat, u1, r1_local, iterations=max(int(sstep), int(smoother_steps), 1), sweep="forward")

        # pyamg V-cycle on auxiliary P1 Laplacian (per displacement component)
        r2 = r1_local - S_hat @ u1
        for i in range(gdim):
            i0 = i * sgdof
            i1 = (i + 1) * sgdof
            crm = PI_s_T @ r2[i0:i1]
            # One FEALPy algebraic-MG V-cycle on the scalar coarse Laplacian (per displacement component).
            delta = P_coarse @ crm
            u1[i0:i1] += PI_s @ delta

        # Backward Gauss-Seidel sweep
        gauss_seidel(S_hat, u1, r1_local, iterations=max(int(sstep), int(smoother_steps), 1), sweep="backward")

        # Optional Schur correction with ILU inside the preconditioner.
        # Default OFF: it can dominate cost and may worsen spectral properties for GMRES.
        if schur_ilu_in_precond and ilu_s is not None:
            corr = np.asarray(ilu_s.matvec(r1_local - S_hat @ u1)).reshape(-1)
            u1 += corr

        # Assemble preconditioned residual: [M^{-1}*r1 + M^{-1}*B^T*u1; -u1]
        out = np.concatenate([u0 + (BT @ u1) * D_inv, -u1])
        dt = time.perf_counter() - t0p
        every = int(os.environ.get("FRACTUREX_AUXSPACE_PLOG_EVERY", "20"))
        every = max(every, 1)
        if p_apply_count["n"] % every == 0 or dt > 2.0:
            _auxspace_log(f"preconditioner apply #{p_apply_count['n']}: wall={dt:.3f}s")
        return out

    P = LinearOperator(A_.shape, matvec=gmres_preconditioner, dtype=A_.dtype)
    residuals = []
    cb_count = {"n": 0}
    bnorm = float(np.linalg.norm(b_))
    bnorm = max(bnorm, 1e-30)
    true_every = int(os.environ.get("FRACTUREX_AUXSPACE_TRUE_RES_EVERY", "20"))
    true_every = max(true_every, 1)

    def _true_relres(xk) -> float:
        xk = np.asarray(xk, dtype=float).reshape(-1)
        rk = b_ - A_ @ xk
        return float(np.linalg.norm(rk) / bnorm)

    def callback_pr(rk):
        # NOTE (SciPy>=1.13): callback_type 'legacy' and 'pr_norm' both pass a scalar here:
        #   rk = presid / ||b||
        residuals.append(float(rk))
        cb_count["n"] += 1
        every = int(os.environ.get("FRACTUREX_AUXSPACE_CBLOG_EVERY", "5"))
        every = max(every, 1)
        if cb_count["n"] % every == 0:
            _auxspace_log(f"gmres pr_norm callback #{cb_count['n']}: pr_norm={float(rk):.3e}")

    def callback_x(xk):
        # SciPy: callback_type 'x' passes the current iterate (ndarray) after a restart cycle.
        cb_count["n"] += 1
        if cb_count["n"] % true_every == 0:
            tr = _true_relres(xk)
            _auxspace_log(f"gmres x-callback #{cb_count['n']}: true_relres={tr:.3e}")

    fgmres = _fealpy_krylov("gmres")
    force_scipy = os.environ.get("FRACTUREX_AUXSPACE_USE_SCIPY_GMRES", "1").strip() in (
        "1",
        "true",
        "True",
        "yes",
        "YES",
    )
    use_scipy = fgmres is None or force_scipy
    if force_scipy and fgmres is not None:
        _auxspace_log("FRACTUREX_AUXSPACE_USE_SCIPY_GMRES=1: using scipy.sparse.linalg.gmres")
    if not use_scipy:
        _auxspace_log("starting fealpy.solver.gmres ...")
        try:
            x, info = _call_fealpy_gmres(
                fgmres,
                A_,
                b_,
                M=P,
                restart=restart,
                maxit=maxit,
                atol=atol,
                rtol=rtol,
            )
        except TypeError:
            # If FEALPy gmres doesn't accept a preconditioner in this build, fall back to SciPy.
            _auxspace_log("fealpy gmres rejected arguments; falling back to scipy.sparse.linalg.gmres")
            use_scipy = True

    if use_scipy:
        _auxspace_log("starting scipy.sparse.linalg.gmres ...")
        try:
            x, info = gmres(
                A_,
                b_,
                M=P,
                restart=restart,
                maxiter=maxit,
                atol=atol,
                rtol=rtol,
                callback=callback_x,
                callback_type="x",
            )
        except TypeError:
            _auxspace_log("gmres: callback_type='x' unsupported; falling back to pr_norm callbacks")
            x, info = gmres(
                A_,
                b_,
                M=P,
                restart=restart,
                maxiter=maxit,
                atol=atol,
                rtol=rtol,
                callback=callback_pr,
                callback_type="pr_norm",
            )
    _auxspace_log(f"gmres finished: info={info!r}, P_applies={p_apply_count['n']}, callbacks={cb_count['n']}")
    try:
        tr_final = _true_relres(x)
        _auxspace_log(f"gmres final true_relres={tr_final:.3e}")
        if (not _extract_converged_from_info(info)) or (tr_final > max(float(rtol), 1e-9) * 10.0):
            print(
                "[aux-gmres warning] "
                f"non-converged or loose residual: info={info!r}, true_relres={tr_final:.3e}, "
                f"rtol={rtol:.1e}, restart={restart}, maxit={maxit}",
                flush=True,
            )
    except Exception:
        pass
    return x, _krylov_stats(residuals, "gmres-auxspace", info, atol, rtol, bnorm=bnorm, cb_count_hint=cb_count["n"])

