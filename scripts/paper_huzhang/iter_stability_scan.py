#!/usr/bin/env python3
"""迭代稳定性对照扫描（D12 C2/C3 核心图数据）。

对固定网格 + 均匀损伤 d 装配同一弹性鞍点系统 A，用多个预条件子各解一次，记 GMRES
niter（同 rtol/atol/restart/maxit）。论点：随 d→1（应力块被 1/g(d)≈1/eps_g 放大）与网格
细化，无预条件 / Jacobi / ILU 的 niter 爆炸或打满 maxit，而 aux-space（对称 V-cycle）
niter 平稳 → 验证 mesh- 与 parameter-independence。

输出 CSV: results/phasefield/_iter_stability/iter_stability.csv
用法: iter_stability_scan.py [h1 h2 h3]   (默认 h1 h2)
"""
from __future__ import annotations
import csv, os, sys, time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import gmres as scipy_gmres

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from fealpy.backend import backend_manager as bm
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.utilfuc.linear_solvers import (
    as_scipy_csr, make_ilu_preconditioner, solve_huzhang_block_gmres_fast,
)

HMIN = {"h1": 0.05, "h2": 0.025, "h3": 0.013}
DVALS = [0.0, 0.5, 0.9, 0.99, 0.999]
RTOL, ATOL, RESTART, MAXIT = 1e-8, 1e-12, 200, 300
LOAD = 0.09


class Mat:
    E = 200.0; nu = 0.2; Gc = 1.0; l0 = 0.02
    @property
    def mu(self): return self.E / (2 * (1 + self.nu))
    @property
    def lam(self): return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))


def _count_gmres(A_, b_, M, label):
    """scipy GMRES with a niter-counting callback; returns (niter, converged, relres)."""
    it = {"n": 0}
    def cb(_): it["n"] += 1
    t0 = time.perf_counter()
    try:
        x, info = scipy_gmres(A_, b_, M=M, restart=RESTART, maxiter=MAXIT,
                              rtol=RTOL, atol=ATOL, callback=cb, callback_type="pr_norm")
    except Exception as e:
        return {"niter": -1, "converged": False, "relres": float("nan"),
                "t_s": time.perf_counter() - t0, "note": f"err:{type(e).__name__}"}
    bn = max(float(np.linalg.norm(b_)), 1e-30)
    rr = float(np.linalg.norm(A_ @ x - b_) / bn)
    return {"niter": it["n"], "converged": bool(info == 0 and rr <= RTOL * 10),
            "relres": rr, "t_s": time.perf_counter() - t0, "note": ""}


def main():
    levels = sys.argv[1:] or ["h1", "h2"]
    outdir = _REPO / "results/phasefield/_iter_stability"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for lvl in levels:
        dmg = PhaseFieldDamageModel(density_type="AT2", degradation_type="quadratic",
                                    split="hybrid", eps_g=1e-6, debug=False)
        case = Model0CircularNotchCase(_model=Mat(), hmin=HMIN[lvl])
        mesh = case.make_mesh()
        discr = HuZhangDiscretization(case, p=3, use_relaxation=True).build(mesh=mesh)
        dmg.on_build(discr, discr.state, case)
        asm = HuZhangElasticAssembler(discr, case, dmg, formulation="standard",
                                      assembly_parallel=False)
        m = int(discr.gdof_sigma)
        for dval in DVALS:
            discr.state.d[:] = bm.asarray(np.full(discr.space_d.number_of_global_dofs(),
                                                  float(dval)))
            asm.begin_load_step(LOAD)
            sys_e = asm.assemble(LOAD)
            A_ = as_scipy_csr(sys_e.A); F = np.asarray(sys_e.F, float).reshape(-1)
            # preconditioners
            # 1) none
            r_none = _count_gmres(A_, F, None, "none")
            # 2) Jacobi (diag) on full saddle
            diag = np.asarray(A_.diagonal(), float)
            diag = np.where(np.abs(diag) > 1e-30, diag, 1.0)
            from scipy.sparse.linalg import LinearOperator
            Mjac = LinearOperator(A_.shape, matvec=lambda v, d=diag: v / d)
            r_jac = _count_gmres(A_, F, Mjac, "jacobi")
            # 3) ILU on full saddle
            try:
                Milu = make_ilu_preconditioner(A_)
                r_ilu = _count_gmres(A_, F, Milu, "ilu")
            except Exception as e:
                r_ilu = {"niter": -1, "converged": False, "relres": float("nan"),
                         "t_s": 0.0, "note": f"ilu_build_err:{type(e).__name__}"}
            # 4) aux-space fast (symmetric V-cycle)
            t0 = time.perf_counter()
            _, info_aux = solve_huzhang_block_gmres_fast(
                sys_e.A, F, gdof_sigma=m, vspace=discr.space_u,
                rtol=RTOL, atol=ATOL, restart=60, maxit=MAXIT, q=3,
                weighted_aux=True, elastic_formulation="standard",
                damage=dmg, state=discr.state)
            r_aux = {"niter": int(getattr(info_aux, "niter", -1)),
                     "converged": bool(getattr(info_aux, "converged", False)),
                     "relres": float("nan"), "t_s": time.perf_counter() - t0, "note": ""}
            for pc, r in (("none", r_none), ("jacobi", r_jac), ("ilu", r_ilu), ("aux_fast", r_aux)):
                rows.append({"level": lvl, "sigma": m, "d": dval, "precond": pc, **r})
                print(f"[{lvl} d={dval:5.3f}] {pc:9s} niter={r['niter']:>5} "
                      f"conv={r['converged']!s:5} relres={r['relres']:.1e} t={r['t_s']:.1f}s {r['note']}",
                      flush=True)
    csvp = outdir / "iter_stability.csv"
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {csvp}")


if __name__ == "__main__":
    main()
