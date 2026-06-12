#!/usr/bin/env python3
"""B1 验证（真实场）：加载真实相场 run 局部化步的 checkpoint d 场，
比较 aux_fast 基线 vs interface_aware(α 扫) 的 GMRES niter。

这是真正有意义的测试：合成 band 不能复现 O(100) regime，但真实 checkpoint 的
d 场（maxd≈0.998，尖锐裂纹界面）能。若 B1 在此把 niter 从 O(100) 压下，则路线 B 成立。

用法: validate_interface_aware_realfield.py <checkpoint.npz> [hmin]
  例: ... checkpoints/step_015.npz 0.025
输出: 控制台表 + results/phasefield/_iter_stability/interface_aware_realfield.csv
"""
from __future__ import annotations
import csv, sys, time
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from fealpy.backend import backend_manager as bm
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.utilfuc.linear_solvers import solve_huzhang_block_gmres_fast

RTOL, ATOL, RESTART, MAXIT = 1e-8, 1e-12, 60, 300
LOAD = 0.092  # ~step15 load; only sets BC scale, d field is what drives conditioning
ALPHAS = [0.5, 1.0, 2.0, 4.0, 8.0]


class Mat:
    E = 200.0; nu = 0.2; Gc = 1.0; l0 = 0.02
    @property
    def mu(self): return self.E / (2 * (1 + self.nu))
    @property
    def lam(self): return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))


def main():
    ckpt = Path(sys.argv[1])
    hmin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.025
    z = np.load(ckpt)
    d_real = np.asarray(z["d"], float)
    print(f"checkpoint {ckpt.name}: d range [{d_real.min():.4f}, {d_real.max():.4f}]  ndof={d_real.size}")

    dmg = PhaseFieldDamageModel(density_type="AT2", degradation_type="quadratic",
                                split="hybrid", eps_g=1e-6, debug=False)
    case = Model0CircularNotchCase(_model=Mat(), hmin=hmin)
    mesh = case.make_mesh()
    discr = HuZhangDiscretization(case, p=3, damage_p=2, use_relaxation=True).build(mesh=mesh)
    dmg.on_build(discr, discr.state, case)
    asm = HuZhangElasticAssembler(discr, case, dmg, formulation="standard",
                                  assembly_parallel=False)
    m = int(discr.gdof_sigma)
    if d_real.size != discr.space_d.number_of_global_dofs():
        raise RuntimeError(f"d dof mismatch: ckpt {d_real.size} vs space {discr.space_d.number_of_global_dofs()} "
                           f"(wrong hmin? got {hmin})")
    discr.state.d[:] = bm.asarray(d_real)
    # also restore H history if present so the assembled operator matches the real step
    if "H" in z.files and discr.state.H is not None:
        try:
            discr.state.H[:] = bm.asarray(np.asarray(z["H"], float))
        except Exception as e:
            print(f"[warn] could not restore H: {e}")

    asm.begin_load_step(LOAD)
    sys_e = asm.assemble(LOAD)
    F = np.asarray(sys_e.F, float).reshape(-1)

    rows = []
    print(f"\nsigma-dof={m}\n{'mode':>22} {'alpha':>6} {'niter':>6} {'conv':>5} {'t_s':>8}")
    for aware in (False, True):
        alphas = [0.0] if not aware else ALPHAS
        for a in alphas:
            t0 = time.perf_counter()
            _, info = solve_huzhang_block_gmres_fast(
                sys_e.A, F, gdof_sigma=m, vspace=discr.space_u,
                rtol=RTOL, atol=ATOL, restart=RESTART, maxit=MAXIT, q=5,
                weighted_aux=True, elastic_formulation="standard",
                interface_aware=aware, interface_alpha=a,
                precond_rebuild_interval=1,
                damage=dmg, state=discr.state)
            dt = time.perf_counter() - t0
            ni = int(getattr(info, "niter", -1))
            cv = bool(getattr(info, "converged", False))
            label = "baseline (geom P1)" if not aware else "interface_aware"
            rows.append({"ckpt": ckpt.name, "sigma": m, "mode": label,
                         "alpha": a, "niter": ni, "converged": cv, "t_s": dt})
            print(f"{label:>22} {a:>6.1f} {ni:>6} {str(cv):>5} {dt:>8.1f}", flush=True)

    base = next(r["niter"] for r in rows if r["mode"].startswith("baseline"))
    cand = [r for r in rows if r["mode"] == "interface_aware" and r["niter"] > 0]
    if cand and base > 0:
        best = min(cand, key=lambda r: r["niter"])
        print(f"\nbaseline niter={base}  best interface_aware niter={best['niter']} "
              f"(alpha={best['alpha']})  ->  {100*(1-best['niter']/base):+.0f}% change")

    outp = _REPO / "results/phasefield/_iter_stability/interface_aware_realfield.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
