"""Parameter sweep for Hu-Zhang + phase-field block preconditioners.

Runs a single elastic block-system solve at a frozen damage field and reports
metrics to stdout (one CSV line) or appends to a CSV file. Intended to be driven
from a shell script (see ``scripts/paper_precond/run_sweep.sh``) that loops over
the full parameter matrix described in
``docs/preconditioner/D12_PRECONDITIONER_PAPER_PLAN.md`` §4.

Usage (single run):
    python -m fracturex.tests.precond_sweep \\
        --case model0 --algorithm aux_weighted --formulation standard \\
        --hmin 0.02 --l0 0.001 --eps-g 1e-6 --max-d 0.9 \\
        --csv-out results/paper_precond/sweep.csv

The frozen damage field is synthetic (Gaussian peak at a prescribed center)
rather than from a real staggered run. This is the standard "frozen-d" recipe
used in preconditioner studies (Heister-Wick 2015 §6, Farrell-Maurini 2017 §5):
the linear-algebra question is independent of how d(x) was produced.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.sparse.linalg import spsolve as scipy_spsolve

from fealpy.backend import backend_manager as bm

from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.utilfuc.linear_solvers import (
    KrylovInfo,
    solve_huzhang_block_gmres,
    solve_huzhang_block_gmres_auxspace,
    solve_huzhang_block_gmres_fast,
    solve_lgmres_ilu,
)


# --------------------------------------------------------------------------- #
# Case registry: name -> (case_builder, material_builder)
# Extend here when adding new test geometries (model2, lshape, ...).
# --------------------------------------------------------------------------- #


def _build_model0(hmin: float):
    from fracturex.cases.model0_circular_notch import Model0CircularNotchCase

    from fracturex.tests.phasefield_model0_huzhang import Model0Material  # reuse

    mat = Model0Material()
    case = Model0CircularNotchCase(
        _model=mat, hmin=hmin, distmesh_maxit=100, debug_mesh=False
    )
    return case, mat


CASE_BUILDERS: Dict[str, Callable[[float], Tuple[Any, Any]]] = {
    "model0": _build_model0,
    # TODO: add "square_precrack", "model2_notch_shear", "lshape"
}


# --------------------------------------------------------------------------- #
# Synthetic damage field: Gaussian peak normalized to reach a target max(d).
# --------------------------------------------------------------------------- #


def _set_synthetic_damage(
    state, mesh, *, max_d: float, l0: float, center: Optional[Tuple[float, float]] = None
) -> float:
    """Set ``state.d`` to a Gaussian profile peaking at ``max_d``.

    Returns the actually-realized ``max(d)`` (clipped to ``[0, clamp_max]``).
    """
    node = bm.to_numpy(mesh.entity("node"))  # (NN, 2)
    if center is None:
        xc, yc = float(np.median(node[:, 0])), float(np.median(node[:, 1]))
    else:
        xc, yc = center
    r2 = (node[:, 0] - xc) ** 2 + (node[:, 1] - yc) ** 2
    sigma = max(2.0 * l0, 1e-6)
    profile = np.exp(-r2 / (sigma ** 2))
    profile *= float(max_d) / max(profile.max(), 1e-30)
    profile = np.clip(profile, 0.0, 0.999)
    # state.d may live in a higher-order Lagrange space; assume P1/P2 → use
    # interpolation by node lookup. For P2 we set nodal dofs only; quadrature
    # consumers re-evaluate. Good enough for preconditioner study.
    d_arr = bm.asarray(profile, dtype=bm.float64)
    if state.d.shape[0] != d_arr.shape[0]:
        # Damage space is P2 (or higher): pad / project via simple resize.
        # This is a placeholder — replace with a proper P1-to-Pk interpolation
        # once the sweep reaches §5.4 of the paper plan.
        new = np.zeros(state.d.shape[0], dtype=np.float64)
        n = min(new.shape[0], d_arr.shape[0])
        new[:n] = bm.to_numpy(d_arr)[:n]
        state.d[:] = bm.asarray(new)
    else:
        state.d[:] = d_arr
    return float(np.max(bm.to_numpy(state.d[:])))


# --------------------------------------------------------------------------- #
# Algorithm registry: each entry returns (x, niter, converged, relres).
# --------------------------------------------------------------------------- #


def _solve_direct(A, b, **_) -> Tuple[np.ndarray, int, bool, float]:
    A_ = A.to_scipy().tocsr() if hasattr(A, "to_scipy") else A
    b_ = np.asarray(b, dtype=float).reshape(-1)
    x = scipy_spsolve(A_, b_)
    res = float(np.linalg.norm(A_ @ x - b_) / max(np.linalg.norm(b_), 1e-30))
    return x, 0, True, res


def _solve_ilu_gmres(A, b, **_) -> Tuple[np.ndarray, int, bool, float]:
    x, info = solve_lgmres_ilu(A, b, rtol=1e-8, atol=0.0, maxit=400)
    return x, int(info.niter), bool(info.converged), float(info.residual_norm)


def _solve_aux(weighted: bool, formulation: str) -> Callable:
    def _impl(A, b, *, discr, damage, **_):
        x, info = solve_huzhang_block_gmres_auxspace(
            A,
            b,
            gdof_sigma=discr.gdof_sigma,
            vspace=discr.space_u,
            atol=1e-12,
            rtol=1e-8,
            restart=60,
            maxit=400,
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
        return x, int(info.niter), bool(info.converged), float(info.residual_norm)

    return _impl


def _solve_aux_fast(formulation: str) -> Callable:
    """论文主 aux 列：fast 对称两层 V-cycle 辅助空间预条件子（见 D12 §3.2）。"""
    def _impl(A, b, *, discr, damage, **_):
        x, info = solve_huzhang_block_gmres_fast(
            A, b,
            gdof_sigma=discr.gdof_sigma,
            vspace=discr.space_u,
            atol=1e-12, rtol=1e-8, restart=60, maxit=400, q=3,
            precond_rebuild_interval=1,       # 单解扫描：每解重建，干净
            schur_precond="auto",
            weighted_aux=True,
            elastic_formulation=formulation,
            damage=damage, state=discr.state,
        )
        return x, int(info.niter), bool(info.converged), float(info.residual_norm)
    return _impl


def _solve_block_gmres(A, b, *, discr, **_) -> Tuple[np.ndarray, int, bool, float]:
    x, info = solve_huzhang_block_gmres(
        A, b, gdof_sigma=discr.gdof_sigma, rtol=1e-8, atol=0.0, maxit=400
    )
    return x, int(info.niter), bool(info.converged), float(info.residual_norm)


# --------------------------------------------------------------------------- #
# Main sweep entry
# --------------------------------------------------------------------------- #


@dataclass
class SweepRecord:
    case: str
    algorithm: str
    formulation: str
    hmin: float
    l0: float
    eps_g: float
    max_d_target: float
    max_d_actual: float
    ndof_sigma: int
    ndof_u: int
    n_iter: int
    converged: bool
    rel_residual: float
    t_setup_s: float
    t_solve_s: float

    def csv_row(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__annotations__}


CSV_COLUMNS = list(SweepRecord.__annotations__.keys())


def run_once(args: argparse.Namespace) -> SweepRecord:
    builder = CASE_BUILDERS.get(args.case)
    if builder is None:
        raise SystemExit(f"unknown --case={args.case}; choices={list(CASE_BUILDERS)}")

    case, mat = builder(args.hmin)
    mat.l0 = float(args.l0)  # override l0 from CLI

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

    # Synthetic frozen damage field
    actual_max_d = _set_synthetic_damage(
        discr.state, mesh, max_d=args.max_d, l0=float(args.l0)
    )

    elastic = HuZhangElasticAssembler(
        discr, case, damage, formulation=args.formulation, assembly_parallel=False
    )
    elastic.begin_load_step(float(args.load))
    t0 = time.perf_counter()
    system = elastic.assemble(float(args.load))
    t_setup = time.perf_counter() - t0

    # Algorithm dispatch
    algo_map: Dict[str, Callable] = {
        "direct": _solve_direct,
        "ilu_gmres": _solve_ilu_gmres,
        "block_gmres": _solve_block_gmres,
        "aux_unweighted": _solve_aux(weighted=False, formulation=args.formulation),
        "aux_weighted": _solve_aux(weighted=True, formulation=args.formulation),
        "aux_fast": _solve_aux_fast(formulation=args.formulation),
    }
    if args.algorithm not in algo_map:
        raise SystemExit(f"unknown --algorithm={args.algorithm}; choices={list(algo_map)}")

    solver = algo_map[args.algorithm]
    t0 = time.perf_counter()
    _x, niter, conv, relres = solver(system.A, system.F, discr=discr, damage=damage)
    t_solve = time.perf_counter() - t0

    return SweepRecord(
        case=args.case,
        algorithm=args.algorithm,
        formulation=args.formulation,
        hmin=float(args.hmin),
        l0=float(args.l0),
        eps_g=float(args.eps_g),
        max_d_target=float(args.max_d),
        max_d_actual=float(actual_max_d),
        ndof_sigma=int(discr.gdof_sigma),
        ndof_u=int(discr.gdof_u),
        n_iter=int(niter),
        converged=bool(conv),
        rel_residual=float(relres),
        t_setup_s=float(t_setup),
        t_solve_s=float(t_solve),
    )


def _append_csv(path: str, record: SweepRecord) -> None:
    new_file = not os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(record.csv_row())


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--case", default="model0", choices=list(CASE_BUILDERS))
    p.add_argument(
        "--algorithm",
        default="aux_fast",  # 默认用最优算法：fast 对称两层 V-cycle 辅助空间预条件子
        choices=["direct", "ilu_gmres", "block_gmres", "aux_unweighted", "aux_weighted", "aux_fast"],
    )
    p.add_argument("--formulation", default="standard", choices=["standard", "effective_stress"])
    p.add_argument("--hmin", type=float, default=0.02)
    p.add_argument("--l0", type=float, default=0.001)
    p.add_argument("--eps-g", type=float, default=1e-6)
    p.add_argument("--max-d", type=float, default=0.9)
    p.add_argument("--load", type=float, default=0.0, help="scalar load passed to assembler")
    p.add_argument("--csv-out", default=None, help="if given, append one row to this CSV")
    args = p.parse_args(argv)

    record = run_once(args)

    # stdout: machine-readable single line for shell loops
    print(",".join(str(record.csv_row()[c]) for c in CSV_COLUMNS))

    if args.csv_out:
        _append_csv(args.csv_out, record)

    return 0 if record.converged else 1


if __name__ == "__main__":
    sys.exit(main())
