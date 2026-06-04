"""Spectrum estimation for preconditioned Hu-Zhang + phase-field elastic system.

For a frozen damage field ``d(x)``, builds the elastic block system

.. math:: K_h(d) = \\begin{bmatrix} A(d) & B^\\top \\\\ B & 0 \\end{bmatrix}

and estimates the leading / trailing eigenvalues of ``P^{-1} K_h(d)`` where
``P`` is one of the preconditioners in
``fracturex/utilfuc/linear_solvers.py``. The spectrum visualizes whether the
preconditioned operator has clustered eigenvalues (good) or a long tail (bad),
and is the primary figure for Paper §5.6.

Usage:
    python -m fracturex.tests.precond_spectrum \\
        --case model0 --algorithm aux_weighted --formulation standard \\
        --hmin 0.02 --l0 0.001 --eps-g 1e-6 --max-d 0.9 \\
        --k-large 20 --k-small 5 --npz-out results/paper_precond/spectrum.npz

Note: ``P^{-1} K`` is non-symmetric for block-triangular ``P``, so we use ARPACK
``eigs`` (general eigenproblem) rather than ``eigsh``. For mesh-independence
plots, run this for 3-4 different ``hmin`` values and overlay the spectra.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs, spilu

from fracturex.tests.precond_sweep import (
    CASE_BUILDERS,
    _set_synthetic_damage,
)
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization


# --------------------------------------------------------------------------- #
# Preconditioner factories: each returns a callable ``apply(x) -> P^{-1} x``.
# Kept independent of ``solve_huzhang_block_gmres_*`` so we get the operator
# action directly rather than a full Krylov solve.
# --------------------------------------------------------------------------- #


def _precond_identity(A_scipy, **_) -> LinearOperator:
    n = A_scipy.shape[0]
    return LinearOperator(shape=(n, n), matvec=lambda x: x, dtype=A_scipy.dtype)


def _precond_ilu(A_scipy, *, drop_tol: float = 1e-4, fill_factor: float = 10.0, **_) -> LinearOperator:
    ilu = spilu(A_scipy.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
    n = A_scipy.shape[0]
    return LinearOperator(shape=(n, n), matvec=ilu.solve, dtype=A_scipy.dtype)


def _precond_block_aux(
    A_scipy,
    *,
    discr,
    damage,
    formulation: str,
    weighted: bool,
    **_,
) -> LinearOperator:
    """Wrap one application of the aux-space block preconditioner.

    Implementation note: ``solve_huzhang_block_gmres_auxspace`` constructs the
    preconditioner internally and runs GMRES. To expose just the operator
    action we re-import its private builder when available; otherwise we
    approximate ``P^{-1}`` by running a single restart of GMRES (zero rtol,
    maxit=1) which yields one application of ``P^{-1}`` on the initial residual.

    TODO: refactor ``linear_solvers.py`` to expose ``build_auxspace_precond``
    as a public function so this wrapper becomes one-line.
    """
    from fracturex.utilfuc.linear_solvers import solve_huzhang_block_gmres_auxspace

    n = A_scipy.shape[0]

    def _apply(x: np.ndarray) -> np.ndarray:
        b = np.asarray(x, dtype=float).reshape(-1)
        # Single GMRES restart with rtol=1 effectively returns P^{-1} b for the
        # zero initial guess. We force ``restart=1, maxit=1`` and rtol very
        # loose so the iteration terminates after preconditioning the residual.
        y, _info = solve_huzhang_block_gmres_auxspace(
            A_scipy,
            b,
            gdof_sigma=discr.gdof_sigma,
            vspace=discr.space_u,
            atol=1e30,  # disable absolute tolerance
            rtol=1.0,   # accept after one application
            restart=1,
            maxit=1,
            sstep=3,
            theta=0.25,
            q=3,
            schur_rebuild_interval=1,
            coarse_rebuild_interval=1,
            weighted_aux=weighted,
            elastic_formulation=formulation,
            damage=damage,
            state=discr.state,
            schur_ilu_in_precond=False,
        )
        return np.asarray(y, dtype=float).reshape(-1)

    return LinearOperator(shape=(n, n), matvec=_apply, dtype=A_scipy.dtype)


def _precond_block_aux_fast(A_scipy, *, discr, damage, formulation: str, **_) -> LinearOperator:
    """One application of the `fast` (symmetric two-level V-cycle) aux preconditioner.

    论文主 aux 列（D12 §3.2）。同 `_precond_block_aux` 的做法：用 1 步 GMRES（rtol=1,
    restart=1, maxit=1）从零初值得到 P^{-1} b 的一次作用，暴露算子供 ARPACK 估谱。
    """
    from fracturex.utilfuc.linear_solvers import solve_huzhang_block_gmres_fast

    n = A_scipy.shape[0]
    # Expose the exact block preconditioner operator P^{-1} (no GMRES run). Pass a
    # nonzero dummy b to bypass the zero-rhs short-circuit; return_preconditioner=True
    # returns (P, None) right after P is built.
    P, _ = solve_huzhang_block_gmres_fast(
        A_scipy, np.ones(n, dtype=float),
        gdof_sigma=discr.gdof_sigma, vspace=discr.space_u,
        return_preconditioner=True,
        atol=0.0, rtol=1e-8, restart=60, maxit=1, q=3,
        precond_rebuild_interval=1, schur_precond="auto",
        weighted_aux=True, elastic_formulation=formulation,
        damage=damage, state=discr.state,
    )
    return P


PRECOND_FACTORIES = {
    "identity": _precond_identity,
    "ilu": _precond_ilu,
    "aux_weighted": lambda A, **kw: _precond_block_aux(A, weighted=True, **kw),
    "aux_unweighted": lambda A, **kw: _precond_block_aux(A, weighted=False, **kw),
    "aux_fast": _precond_block_aux_fast,
}


# --------------------------------------------------------------------------- #
# Spectrum estimation
# --------------------------------------------------------------------------- #


def estimate_spectrum(
    A_scipy,
    P_inv: LinearOperator,
    *,
    k_large: int = 20,
    k_small: int = 5,
    tol: float = 1e-6,
    maxiter: int = 2000,
) -> dict:
    """Return leading-magnitude and smallest-magnitude eigenvalues of ``P^{-1} A``.

    Uses ARPACK on the matvec ``x -> P^{-1} (A x)``. ``LM`` = largest magnitude,
    ``SM`` = smallest magnitude. The condition-number proxy is ``|lambda_max| /
    |lambda_min|`` (treating the operator as approximately normal — for
    block-triangular ``P^{-1}`` this is a heuristic, not a true bound).
    """
    n = A_scipy.shape[0]
    K = LinearOperator(
        shape=(n, n),
        matvec=lambda x: P_inv @ (A_scipy @ x),
        dtype=A_scipy.dtype,
    )

    eig_large, _ = eigs(K, k=int(k_large), which="LM", tol=tol, maxiter=maxiter)
    if int(k_small) <= 0:
        # LM-only: ARPACK SM is slow/unstable on this non-normal block-preconditioned
        # operator (kappa via SM is a heuristic anyway). Report the LM clustering range.
        eig_small = np.asarray([], dtype=eig_large.dtype)
        sm_abs = np.abs(eig_large)  # min over the computed LM set (lower bound proxy)
    else:
        try:
            eig_small, _ = eigs(K, k=int(k_small), which="SM", tol=tol, maxiter=maxiter)
        except Exception as exc:
            print(f"[precond_spectrum] SM eigs failed ({exc}); falling back to shift-invert")
            from scipy.sparse.linalg import eigs as _eigs

            eig_small, _ = _eigs(K, k=int(k_small), sigma=0.0, which="LM", tol=tol, maxiter=maxiter)
        sm_abs = np.abs(eig_small)

    lm_abs = np.abs(eig_large)
    kappa = float(lm_abs.max() / max(sm_abs.min(), 1e-30))

    return {
        "eig_large": eig_large,
        "eig_small": eig_small,
        "lambda_max_abs": float(lm_abs.max()),
        "lambda_min_abs": float(sm_abs.min()),
        "kappa_proxy": kappa,
    }


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--case", default="model0", choices=list(CASE_BUILDERS))
    p.add_argument(
        "--algorithm",
        default="aux_fast",  # 默认用最优算法（D12 §3.2）
        choices=list(PRECOND_FACTORIES),
    )
    p.add_argument("--formulation", default="standard", choices=["standard", "effective_stress"])
    p.add_argument("--hmin", type=float, default=0.02)
    p.add_argument("--l0", type=float, default=0.001)
    p.add_argument("--eps-g", type=float, default=1e-6)
    p.add_argument("--max-d", type=float, default=0.9)
    p.add_argument("--load", type=float, default=0.0)
    p.add_argument("--k-large", type=int, default=20)
    p.add_argument("--k-small", type=int, default=5)
    p.add_argument("--npz-out", default=None, help="if given, save eigenvalue arrays")
    args = p.parse_args(argv)

    builder = CASE_BUILDERS[args.case]
    case, mat = builder(args.hmin)
    mat.l0 = float(args.l0)

    mesh = case.make_mesh()
    discr = HuZhangDiscretization(case=case, p=3, damage_p=2, use_relaxation=True).build(mesh=mesh)

    damage = PhaseFieldDamageModel(
        density_type="AT2",
        degradation_type="quadratic",
        split="hybrid",
        eps_g=float(args.eps_g),
        debug=False,
    )
    damage.on_build(discr, discr.state, case)
    actual_max_d = _set_synthetic_damage(
        discr.state, mesh, max_d=args.max_d, l0=float(args.l0)
    )

    elastic = HuZhangElasticAssembler(
        discr, case, damage, formulation=args.formulation, assembly_parallel=False
    )
    elastic.begin_load_step(float(args.load))
    system = elastic.assemble(float(args.load))

    A_scipy = system.A.to_scipy().tocsr() if hasattr(system.A, "to_scipy") else system.A.tocsr()

    factory = PRECOND_FACTORIES[args.algorithm]
    t0 = time.perf_counter()
    P_inv = factory(
        A_scipy,
        discr=discr,
        damage=damage,
        formulation=args.formulation,
    )
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    spec = estimate_spectrum(
        A_scipy, P_inv, k_large=args.k_large, k_small=args.k_small
    )
    t_eig = time.perf_counter() - t0

    print(
        f"case={args.case} algo={args.algorithm} form={args.formulation} "
        f"hmin={args.hmin} l0={args.l0} eps_g={args.eps_g} max_d={actual_max_d:.4f} "
        f"n={A_scipy.shape[0]} "
        f"|lambda|_max={spec['lambda_max_abs']:.3e} "
        f"|lambda|_min={spec['lambda_min_abs']:.3e} "
        f"kappa_proxy={spec['kappa_proxy']:.3e} "
        f"t_build={t_build:.2f}s t_eig={t_eig:.2f}s"
    )

    if args.npz_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.npz_out)) or ".", exist_ok=True)
        np.savez(
            args.npz_out,
            eig_large=spec["eig_large"],
            eig_small=spec["eig_small"],
            case=args.case,
            algorithm=args.algorithm,
            formulation=args.formulation,
            hmin=args.hmin,
            l0=args.l0,
            eps_g=args.eps_g,
            max_d_actual=actual_max_d,
            kappa_proxy=spec["kappa_proxy"],
        )
        print(f"[precond_spectrum] wrote {args.npz_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
