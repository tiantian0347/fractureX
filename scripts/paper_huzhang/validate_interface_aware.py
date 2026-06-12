#!/usr/bin/env python3
"""B1 验证：界面感知粗空间 (interface_aware) 对尖锐裂纹界面的 niter 改善。

关键点：均匀 d 下 grad d=0，B1 是恒等无操作；只有**尖锐界面** d 场才触发。
本脚本人工构造一条裂纹带 d 剖面（沿 y=0.5 的 tanh 过渡，宽度 ~l0），在固定网格上
装配同一弹性鞍点系统，比较 aux_fast 基线 vs interface_aware(α 扫) 的 GMRES niter。

用法: validate_interface_aware.py [h1|h2|h3] [band_halfwidth_factor]
输出: 控制台表 + results/phasefield/_iter_stability/interface_aware_validate.csv
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
from fracturex.utilfuc.linear_solvers import as_scipy_csr, solve_huzhang_block_gmres_fast

HMIN = {"h1": 0.05, "h2": 0.025, "h3": 0.013}
RTOL, ATOL, RESTART, MAXIT = 1e-8, 1e-12, 60, 300
LOAD = 0.09
ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0]


class Mat:
    E = 200.0; nu = 0.2; Gc = 1.0; l0 = 0.02
    @property
    def mu(self): return self.E / (2 * (1 + self.nu))
    @property
    def lam(self): return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))


def _localized_d(space_d, *, y0=0.5, halfwidth=0.04, dmax=0.9995):
    """Sharp crack-band damage: d = dmax * sech^2-like bump centered at y=y0.

    Returns a dof vector with d≈dmax on the band and d≈0 away — a sharp d≈1/d≈0
    interface like a fully localized crack.
    """
    ip = np.asarray(space_d.interpolation_points())  # (ndof, 2)
    y = ip[:, 1]
    # tanh-profile band: 1 inside |y-y0|<hw, decaying to 0 over ~hw
    d = dmax / np.cosh((y - y0) / halfwidth) ** 2
    d = np.clip(d, 0.0, dmax)
    return d


def main():
    lvl = sys.argv[1] if len(sys.argv) > 1 else "h1"
    hw_factor = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    halfwidth = hw_factor * Mat.l0  # band half-width relative to l0

    dmg = PhaseFieldDamageModel(density_type="AT2", degradation_type="quadratic",
                                split="hybrid", eps_g=1e-6, debug=False)
    case = Model0CircularNotchCase(_model=Mat(), hmin=HMIN[lvl])
    mesh = case.make_mesh()
    discr = HuZhangDiscretization(case, p=3, use_relaxation=True).build(mesh=mesh)
    dmg.on_build(discr, discr.state, case)
    asm = HuZhangElasticAssembler(discr, case, dmg, formulation="standard",
                                  assembly_parallel=False)
    m = int(discr.gdof_sigma)

    d_field = _localized_d(discr.space_d, halfwidth=halfwidth)
    discr.state.d[:] = bm.asarray(d_field)
    print(f"[{lvl}] sigma-dof={m}  band halfwidth={halfwidth:.3f} (={hw_factor}*l0)  "
          f"d range [{d_field.min():.3f}, {d_field.max():.3f}]")

    asm.begin_load_step(LOAD)
    sys_e = asm.assemble(LOAD)
    F = np.asarray(sys_e.F, float).reshape(-1)

    rows = []
    print(f"\n{'mode':>22} {'alpha':>6} {'niter':>6} {'conv':>5} {'t_s':>7}")
    for aware in (False, True):
        alphas = [0.0] if not aware else ALPHAS[1:]
        for a in alphas:
            t0 = time.perf_counter()
            _, info = solve_huzhang_block_gmres_fast(
                sys_e.A, F, gdof_sigma=m, vspace=discr.space_u,
                rtol=RTOL, atol=ATOL, restart=RESTART, maxit=MAXIT, q=3,
                weighted_aux=True, elastic_formulation="standard",
                interface_aware=aware, interface_alpha=a,
                precond_rebuild_interval=1,
                damage=dmg, state=discr.state)
            dt = time.perf_counter() - t0
            ni = int(getattr(info, "niter", -1))
            cv = bool(getattr(info, "converged", False))
            label = "baseline (geom P1)" if not aware else "interface_aware"
            rows.append({"level": lvl, "sigma": m, "halfwidth": halfwidth,
                         "mode": label, "alpha": a, "niter": ni,
                         "converged": cv, "t_s": dt})
            print(f"{label:>22} {a:>6.1f} {ni:>6} {str(cv):>5} {dt:>7.1f}", flush=True)

    base = next(r["niter"] for r in rows if r["mode"].startswith("baseline"))
    best = min((r for r in rows if r["mode"] == "interface_aware"),
               key=lambda r: r["niter"], default=None)
    if best and base > 0 and best["niter"] > 0:
        print(f"\nbaseline niter={base}  best interface_aware niter={best['niter']} "
              f"(alpha={best['alpha']})  ->  {100*(1-best['niter']/base):+.0f}% change")

    outp = _REPO / "results/phasefield/_iter_stability/interface_aware_validate.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
