#!/usr/bin/env python3
"""D3：model2（剪切/混合模式 x-拉伸）的迭代稳定性对照扫描。

同 `iter_stability_scan.py`（model0）的设置，但算例换 square：固定网格 + 均匀损伤 d 装配
同一弹性鞍点 A，用 none/Jacobi/ILU/aux_fast 各解一次记 GMRES niter（同 rtol）。验证 §5.2-5.4
的结论（aux 对 d→1 鲁棒、对手爆炸）在第二个算例上同样成立。

用法: iter_stability_model2.py [nx1 nx2 ...]   默认 80 120
输出: results/phasefield/_iter_stability/iter_stability_model2.csv
"""
from __future__ import annotations
import csv, sys, time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import gmres as scipy_gmres, LinearOperator

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from fealpy.backend import backend_manager as bm
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.cases.model2_notch_shear import Model2NotchXStretchCase
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.utilfuc.linear_solvers import (
    as_scipy_csr, make_ilu_preconditioner, solve_huzhang_block_gmres_fast,
)

DVALS = [0.0, 0.5, 0.9, 0.99, 0.999]
RTOL, ATOL, RESTART, MAXIT = 1e-8, 1e-12, 200, 300
LOAD = 1.2e-2  # mid x-stretch (model2)


class SquareMat:
    E = 210.0; nu = 0.3; Gc = 2.7e-3; l0 = 0.015; ft = 3.0
    @property
    def mu(self): return self.E / (2 * (1 + self.nu))
    @property
    def lam(self): return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))


def _count_gmres(A_, b_, M):
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
    nxs = [int(x) for x in sys.argv[1:]] or [80, 120]
    outdir = _REPO / "results/phasefield/_iter_stability"; outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for nx in nxs:
        dmg = PhaseFieldDamageModel(density_type="AT2", degradation_type="quadratic",
                                    split="hybrid", eps_g=1e-6, debug=False)
        case = Model2NotchXStretchCase(_model=SquareMat(), nx=nx, ny=nx, debug_mesh=False)
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
            r_none = _count_gmres(A_, F, None)
            diag = np.asarray(A_.diagonal(), float); diag = np.where(np.abs(diag) > 1e-30, diag, 1.0)
            r_jac = _count_gmres(A_, F, LinearOperator(A_.shape, matvec=lambda v, d=diag: v / d))
            try:
                r_ilu = _count_gmres(A_, F, make_ilu_preconditioner(A_))
            except Exception as e:
                r_ilu = {"niter": -1, "converged": False, "relres": float("nan"),
                         "t_s": 0.0, "note": f"ilu_err:{type(e).__name__}"}
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
                rows.append({"case": "model2", "nx": nx, "sigma": m, "d": dval, "precond": pc, **r})
                print(f"[m2 nx={nx} d={dval:5.3f}] {pc:9s} niter={r['niter']:>6} "
                      f"conv={r['converged']!s:5} t={r['t_s']:.1f}s {r['note']}", flush=True)
    csvp = outdir / "iter_stability_model2.csv"
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {csvp}")


if __name__ == "__main__":
    main()
