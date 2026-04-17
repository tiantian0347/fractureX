from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.sparse import block_diag, csr_matrix, spdiags
from scipy.sparse.linalg import LinearOperator, gmres, lgmres, minres, spilu

try:
    from fealpy.backend import backend_manager as bm
    from fealpy.functionspace import LagrangeFESpace
    from fealpy.fem import BilinearForm, DirichletBC, ScalarDiffusionIntegrator
    from fealpy.solver.gamg_solver import GAMGSolver
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


@dataclass
class KrylovInfo:
    solver: str
    niter: int
    converged: bool
    residual_norm: float
    atol: float
    rtol: float


def as_scipy_csr(A):
    if hasattr(A, "to_scipy"):
        return A.to_scipy().tocsr()
    if hasattr(A, "tocsr"):
        return A.tocsr()
    return A


def make_ilu_preconditioner(A, *, drop_tol: float = 1e-4, fill_factor: float = 10.0):
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
    try:
        if isinstance(info, dict):
            for k in ("niter", "iter", "iterations", "num_iter"):
                if k in info:
                    return int(info[k])
        if hasattr(info, "niter"):
            return int(getattr(info, "niter"))
        if hasattr(info, "iterations"):
            return int(getattr(info, "iterations"))
    except Exception:
        pass
    return 0


def _extract_converged_from_info(info) -> bool:
    try:
        if isinstance(info, dict):
            for k in ("converged", "success"):
                if k in info:
                    return bool(info[k])
            if "info" in info:
                return int(info["info"]) == 0
        if isinstance(info, (int, np.integer)):
            return int(info) == 0
    except Exception:
        pass
    return False


def _krylov_stats(callback_residuals, solver: str, info, atol, rtol) -> KrylovInfo:
    niter = len(callback_residuals)
    if niter == 0:
        niter = _extract_niter_from_info(info)
    residual_norm = float(callback_residuals[-1]) if callback_residuals else float("nan")
    converged = _extract_converged_from_info(info)
    return KrylovInfo(
        solver=solver,
        niter=int(niter),
        converged=bool(converged),
        residual_norm=residual_norm,
        atol=float(atol),
        rtol=float(rtol),
    )


def _zero_rhs_stats(solver: str, atol: float, rtol: float) -> KrylovInfo:
    return KrylovInfo(
        solver=solver,
        niter=0,
        converged=True,
        residual_norm=0.0,
        atol=float(atol),
        rtol=float(rtol),
    )


def _is_zero_rhs(b, *, atol: float = 0.0) -> bool:
    b_ = np.asarray(b, dtype=float).reshape(-1)
    tol = max(float(atol), 1e-30)
    return float(np.linalg.norm(b_)) <= tol


def _fealpy_krylov(name: str):
    """Return FEALPy krylov callable if available, else None."""
    try:
        if name == "gmres":
            from fealpy.solver.gmres import gmres as fgmres  # type: ignore

            return fgmres
        if name == "minres":
            from fealpy.solver.minres import minres as fminres  # type: ignore

            return fminres
    except Exception:
        return None
    return None


def solve_lgmres_ilu(A, b, *, rtol: float = 1e-8, atol: float = 0.0, maxit: int = 200, drop_tol: float = 1e-4, fill_factor: float = 10.0):
    A_ = as_scipy_csr(A)
    b_ = np.asarray(b, dtype=float).reshape(-1)
    if _is_zero_rhs(b_, atol=atol):
        return np.zeros_like(b_), _zero_rhs_stats("lgmres", atol, rtol)
    M = make_ilu_preconditioner(A_, drop_tol=drop_tol, fill_factor=fill_factor)
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
    return x, _krylov_stats(residuals, "lgmres", info, atol, rtol)


def solve_gmres_ilu(A, b, *, rtol: float = 1e-8, atol: float = 0.0, restart: int = 50, maxit: int = 200, drop_tol: float = 1e-4, fill_factor: float = 10.0):
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

    m = int(gdof_sigma)
    M = A_[:m, :m].tocsr()
    B = A_[m:, :m].tocsr()
    diagM = np.asarray(M.diagonal(), dtype=float)
    diagM = np.where(np.abs(diagM) > 1e-30, diagM, 1.0)
    D_inv = 1.0 / diagM

    S = B @ spdiags(D_inv, 0, m, m).tocsr() @ B.T

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

    m = int(gdof_sigma)
    Mblk = A_[:m, :m].tocsr()
    B = A_[m:, :m].tocsr()
    diagM = np.asarray(Mblk.diagonal(), dtype=float)
    diagM = np.where(np.abs(diagM) > 1e-30, diagM, 1.0)
    D_inv = 1.0 / diagM

    S = B @ spdiags(D_inv, 0, m, m).tocsr() @ B.T
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
            x, info = fgmres(A_, b_, M=P, restart=restart, rtol=rtol, maxit=maxit)
        else:
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
):
    """
    Hu-Zhang mixed system solved by GMRES with an auxiliary-space Schur preconditioner.

    The preconditioner follows the FEALPy-style idea:
    - block L/U smoothing on the Schur complement approximation
    - AMG V-cycle on a scalar auxiliary P1 space
    - block triangular update back to the mixed variables

    Parameters
    ----------
    gdof_sigma:
        Number of sigma unknowns, i.e. the first block size.
    vspace:
        TensorFunctionSpace for displacement, used to access mesh and coarse dofs.
    smoother_steps:
        Number of Gauss-Seidel iterations in each forward/backward sweep (default 2 for speed).
    """
    if not FEALPY_AVAILABLE or not PYAMG_AVAILABLE:
        raise RuntimeError(
            "Aux-space preconditioner requires FEALPy and pyamg. "
            f"FEALPy available: {FEALPY_AVAILABLE}, pyamg available: {PYAMG_AVAILABLE}"
        )

    A_ = as_scipy_csr(A)
    b_ = np.asarray(b, dtype=float).reshape(-1)
    if _is_zero_rhs(b_, atol=atol):
        return np.zeros_like(b_), _zero_rhs_stats("gmres-auxspace", atol, rtol)

    m = int(gdof_sigma)
    M = A_[:m, :m].tocsr()
    B = A_[m:, :m].tocsr()
    diagM = np.asarray(M.diagonal(), dtype=float)
    diagM = np.where(np.abs(diagM) > 1e-30, diagM, 1.0)
    D_inv = 1.0 / diagM

    S = B @ spdiags(D_inv, 0, m, m).tocsr() @ B.T
    S_csr = as_scipy_csr(S)
    BT = B.T.tocsr()

    mesh = vspace.mesh
    mesh_id = id(mesh)
    cache_key = (mesh_id, int(q))
    cached = _AUXSPACE_STATIC_CACHE.get(cache_key, None)
    if cached is None:
        gdim = mesh.geo_dimension()

        # Build coarse P1 space and AMG solver once per mesh/q.
        cspace = LagrangeFESpace(mesh, 1)
        bform = BilinearForm(cspace)
        bform.add_integrator(ScalarDiffusionIntegrator(q=q))
        S_coarse = bform.assembly()
        bc = DirichletBC(cspace, bm.zeros(S_coarse.shape[0], dtype=S_coarse.dtype))
        S_coarse, _ = bc.apply(S_coarse, bm.zeros(S_coarse.shape[0], dtype=S_coarse.dtype))

        import pyamg

        S_coarse_scipy = as_scipy_csr(S_coarse)
        ml = pyamg.smoothed_aggregation_solver(S_coarse_scipy)

        NC = mesh.number_of_cells()
        # TensorFunctionSpace has no `.dof`; use scalar base space then lift interpolation to each component
        sspace = getattr(vspace, "scalar_space", None)
        if sspace is None or not hasattr(sspace, "dof"):
            raise RuntimeError("TensorFunctionSpace.scalar_space/dof is unavailable")

        bc_ref = sspace.dof.multiIndex / sspace.p  # (fldof_s, gdim+1)
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
        # Auxiliary-space consistency term on scalar space:
        # C_s = PI_s * A_coarse * PI_s^T, then lifted to vector block-diagonal.
        C_s = (PI_s @ S_coarse_scipy @ PI_s_T).tocsr()
        C_h = block_diag([C_s] * int(gdim), format="csr")
        cached = {
            "gdim": int(gdim),
            "sgdof": int(sgdof),
            "PI_s": PI_s.tocsr(),
            "PI_s_T": PI_s_T,
            "ml": ml,
            "C_h": C_h,
        }
        _AUXSPACE_STATIC_CACHE[cache_key] = cached

    gdim = int(cached["gdim"])
    sgdof = int(cached["sgdof"])
    PI_s = cached["PI_s"]
    PI_s_T = cached["PI_s_T"]
    ml = cached["ml"]
    C_h = cached["C_h"]

    # Schur approximation with auxiliary consistency term:
    #   S_{h,d} = B_h (M_h^d)^{-1} B_h^T + theta * C_h
    S_hat = (S_csr + float(theta) * C_h).tocsr()

    # Optional Schur-side preconditioner reuse across consecutive solves.
    # This is an approximation when d/load changes; keep interval=1 for strict rebuild.
    rebuild_interval = max(int(schur_rebuild_interval), 1)
    schur_key = (mesh_id, int(m), float(theta))
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
    else:
        ilu_s = make_ilu_preconditioner(S_hat, drop_tol=1e-4, fill_factor=10.0)
        _AUXSPACE_SCHUR_CACHE[schur_key] = {
            "ilu_s": ilu_s,
            "interval": rebuild_interval,
            "calls_since_build": 1,
        }

    def gmres_preconditioner(r):
        r = np.asarray(r, dtype=float).reshape(-1)
        r1 = r[m:]

        # block L solve: M^{-1} * r_sigma
        u0 = r[:m] * D_inv
        u1 = np.zeros_like(r1)
        r1_local = r1 - B @ u0

        # Forward Gauss-Seidel sweep on S_hat.
        gauss_seidel(S_hat, u1, r1_local, iterations=max(int(sstep), int(smoother_steps), 1), sweep="forward")

        # AMG V-cycle on auxiliary P1 space (per displacement component)
        r2 = r1_local - S_hat @ u1
        for i in range(gdim):
            i0 = i * sgdof
            i1 = (i + 1) * sgdof
            crm = PI_s_T @ r2[i0:i1]
            # Use pyamg's solve method directly (returns 1D array)
            delta = ml.solve(crm, tol=1e-6, maxiter=1, cycle="V")
            u1[i0:i1] += PI_s @ delta

        # Backward Gauss-Seidel sweep
        gauss_seidel(S_hat, u1, r1_local, iterations=max(int(sstep), int(smoother_steps), 1), sweep="backward")

        # Optional Schur correction with reused ILU; cheap and often stabilizes reuse.
        if ilu_s is not None:
            corr = np.asarray(ilu_s.matvec(r1_local - S_hat @ u1)).reshape(-1)
            u1 += corr

        # Assemble preconditioned residual: [M^{-1}*r1 + M^{-1}*B^T*u1; -u1]
        return np.concatenate([u0 + (BT @ u1) * D_inv, -u1])

    P = LinearOperator(A_.shape, matvec=gmres_preconditioner, dtype=A_.dtype)
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
    return x, _krylov_stats(residuals, "gmres-auxspace", info, atol, rtol)

