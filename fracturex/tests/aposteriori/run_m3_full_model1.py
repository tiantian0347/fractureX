"""M3 full（完整版）runner：model1 真实 staggered + η_T 自适应加密，跑到裂纹贯通失效。

每个载荷步：solve_one_step（真实 Hu–Zhang + 相场交错）→ 记录指标/内存/VTU →
η_T 估计 → Dörfler 标记 + bisect 加密（d/r_hist 转移）→ 重建离散 + 新装配器 → 下一步。

停机：反力越过峰值并跌到峰值的 `DROP_FRAC` 以下（裂纹贯通），或达 `MAX_STEPS`。

配置（环境变量）：
  FRACTUREX_NX        初始网格 (默认 24)
  FRACTUREX_DU        每步 y-位移增量 (默认 2.5e-4)
  FRACTUREX_MAX_STEPS 最大载荷步 (默认 80)
  FRACTUREX_THETA     Dörfler bulk 比例 (默认 0.3)
  FRACTUREX_MAX_LEVELS 单元最大加密层数 (默认 4)
  FRACTUREX_DROP_FRAC 反力跌破峰值比例即停 (默认 0.4)
  FRACTUREX_KRES      残余刚度 (默认 1e-6)
  FRACTUREX_OUTDIR    输出目录 (默认 results/adaptive_m3_full_model1)
  FRACTUREX_SMOKE=1   冒烟档：少数步 + 低层数 + 不按失效停（覆盖上面若干默认）
  FRACTUREX_NO_VTU=1  跳过 VTU（更快）

运行: PYTHONPATH=$PWD python fracturex/tests/aposteriori/run_m3_full_model1.py
环境: py312 + PYTHONPATH（fealpy_env_py312）。计算走 bm；numpy 仅文件 I/O。
"""
from __future__ import annotations

import csv
import os
import resource
import time

import numpy as np

from fealpy.backend import backend_manager as bm
from fracturex.cases.square_tension_precrack import SquareTensionPreCrackCase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.drivers.huzhang_phasefield_staggered import HuZhangPhaseFieldStaggeredDriver

from fracturex.adaptivity.adaptive_staggered import (
    eta_from_state, make_assemblers, refine_and_rebuild, initial_hmin_floor,
)


class _Mat:
    """model1 材料（与 test_m3_real_staggered 一致）：钢类参数 + 相场长度尺度 l0。"""
    E, nu, Gc, l0, ft = 210.0, 0.3, 2.7e-3, 0.015, 3.0

    @property
    def mu(self):
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self):
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


def _env_f(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_i(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _rss_now_mb() -> float:
    """当前进程常驻内存 (MB)，读 /proc/self/statm（resident pages × page size）。"""
    try:
        with open("/proc/self/statm") as fh:
            resident_pages = int(fh.read().split()[1])
        return resident_pages * resource.getpagesize() / (1024.0 ** 2)
    except Exception:
        return float("nan")


def _rss_peak_mb() -> float:
    """进程峰值常驻内存 (MB)，resource.getrusage ru_maxrss（Linux 单位 KB）。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    bm.set_backend("numpy")
    smoke = os.environ.get("FRACTUREX_SMOKE", "0") == "1"

    nx = _env_i("FRACTUREX_NX", 24)
    du = _env_f("FRACTUREX_DU", 1.0e-3 if smoke else 2.5e-4)
    max_steps = _env_i("FRACTUREX_MAX_STEPS", 5 if smoke else 80)
    theta = _env_f("FRACTUREX_THETA", 0.3)
    max_levels = _env_i("FRACTUREX_MAX_LEVELS", 2 if smoke else 4)
    drop_frac = _env_f("FRACTUREX_DROP_FRAC", 0.4)
    k_res = _env_f("FRACTUREX_KRES", 1e-6)
    outdir = os.environ.get("FRACTUREX_OUTDIR", "results/adaptive_m3_full_model1")
    want_vtu = os.environ.get("FRACTUREX_NO_VTU", "0") != "1"
    use_failure_stop = not smoke
    # 均匀网格基线（关加密）：用于 (b) 等精度 DOF 效率对比。仍走同一 staggered + 失效停机。
    no_refine = os.environ.get("FRACTUREX_NO_REFINE", "0") == "1"

    os.makedirs(outdir, exist_ok=True)
    vtu_dir = os.path.join(outdir, "vtu")
    if want_vtu:
        os.makedirs(vtu_dir, exist_ok=True)
    csv_path = os.path.join(outdir, "history.csv")

    mat = _Mat()
    case = SquareTensionPreCrackCase(_model=mat, nx=nx, ny=nx,
                                     crack_y=0.5, crack_length=0.5)
    mesh = case.make_mesh()
    discr = HuZhangDiscretization(case=case, p=3, damage_p=1,
                                  use_relaxation=True).build(mesh=mesh)
    damage = PhaseFieldDamageModel(density_type="AT2", degradation_type="quadratic",
                                   split="hybrid", eps_g=k_res)
    el_asm, ph_asm = make_assemblers(discr, case, damage)

    driver = HuZhangPhaseFieldStaggeredDriver(
        case=case, discr=discr, damage=damage,
        elastic_assembler=el_asm, phase_assembler=ph_asm,
        tol=1e-4, maxit=200, d_relaxation=1.0,
        elastic_solver=HuZhangPhaseFieldStaggeredDriver._default_spsolve,
        compute_linear_residual=False, debug=False, timing=False,
        save_vtu_per_step=False, stagger_print_interval=0,
    )
    driver.initialize()

    hmin_floor = initial_hmin_floor(mesh, max_levels=max_levels)
    print(f"[cfg] nx={nx} du={du:.3e} max_steps={max_steps} theta={theta} "
          f"max_levels={max_levels} k_res={k_res:.1e} hmin_floor(area)={hmin_floor:.3e} "
          f"smoke={smoke} vtu={want_vtu} no_refine={no_refine}", flush=True)

    fields = ["step", "load", "nc", "dof_sigma", "dof_u", "dof_d", "eta",
              "max_d", "reaction", "iters", "converged", "refined",
              "nc_after", "n_marked", "band_frac", "t_step_s", "rss_now_mb",
              "rss_peak_mb"]
    fh = open(csv_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()

    t0_all = time.perf_counter()
    peak_R = 0.0
    rows = []
    for s in range(max_steps):
        load = float(s * du)
        t_s0 = time.perf_counter()
        info = driver.solve_one_step(step=s, load=load)
        t_step = time.perf_counter() - t_s0

        R = abs(float(info.meta.get("R", 0.0)))
        peak_R = max(peak_R, R)
        max_d = float(info.max_d)

        # η_T 估计（也用于自适应标记）
        est = eta_from_state(discr, lam=mat.lam, mu=mat.mu, k_res=k_res)
        eta, eta_T, cen = est["eta"], est["eta_T"], est["cen"]
        thr = float(bm.sort(eta_T)[int(0.8 * len(eta_T))])
        hi = cen[eta_T >= thr]
        band_frac = float(bm.sum(bm.abs(hi[:, 1] - 0.5) < 0.15)) / max(len(hi), 1)

        if want_vtu:
            discr.mesh.celldata["eta_T"] = bm.asarray(eta_T)
            fname = os.path.join(vtu_dir, f"step_{s:03d}.vtu")
            try:
                driver._save_vtkfile(fname, cell_mode="mean",
                                     q=discr.damage_p + 3,
                                     sigma_eval=driver._sigma_physical_eval)
            except Exception as exc:
                print(f"[vtu] step {s} write failed: {exc}", flush=True)

        # 自适应加密（在记录完本步状态后；下一载荷步用加密后的网格）
        ref = {"refined": False, "nc_after": int(discr.mesh.number_of_cells()),
               "n_marked": 0}
        if s >= 1 and not no_refine:  # step0 载荷为 0，不加密；no_refine=均匀基线
            ref = refine_and_rebuild(discr, damage, case, eta_T,
                                     theta=theta, hmin_floor=hmin_floor)
            if ref["refined"]:
                # 重建后用新空间重造装配器，丢弃旧缓存；driver 在内部复位预裂纹标志
                el_asm, ph_asm = make_assemblers(discr, case, damage)
                driver.elastic_assembler = el_asm
                driver.phase_assembler = ph_asm

        row = dict(
            step=s, load=load, nc=int(ref.get("nc_before", discr.mesh.number_of_cells())),
            dof_sigma=int(info.meta["gdof_sigma"]), dof_u=int(info.meta["gdof_u"]),
            dof_d=int(info.meta["gdof_d"]), eta=eta, max_d=max_d, reaction=R,
            iters=int(info.iters), converged=bool(info.converged),
            refined=bool(ref["refined"]), nc_after=int(ref["nc_after"]),
            n_marked=int(ref["n_marked"]), band_frac=band_frac, t_step_s=t_step,
            rss_now_mb=_rss_now_mb(), rss_peak_mb=_rss_peak_mb(),
        )
        # nc 记录为本步求解时的网格单元数（加密前）
        row["nc"] = int(ref.get("nc_before", row["nc_after"]))
        rows.append(row)
        writer.writerow(row)
        fh.flush()
        print(f"[M3full] step={s:02d} load={load:.3e} nc={row['nc']}→{row['nc_after']} "
              f"dofσ={row['dof_sigma']} η={eta:.3e} max_d={max_d:.3f} R={R:.3e} "
              f"iters={info.iters} band={100*band_frac:.0f}% "
              f"t={t_step:.1f}s rss={row['rss_now_mb']:.0f}/{row['rss_peak_mb']:.0f}MB",
              flush=True)

        # 失效停机：越过峰值并跌破 drop_frac
        if use_failure_stop and peak_R > 0 and s >= 2:
            if R < drop_frac * peak_R and max_d > 0.95:
                print(f"[M3full] 反力跌破峰值 {drop_frac:.0%}（R={R:.3e} < "
                      f"{drop_frac*peak_R:.3e}），裂纹贯通，停机。", flush=True)
                break

    fh.close()
    wall = time.perf_counter() - t0_all
    print(f"\n[M3full] DONE steps={len(rows)} wall={wall:.1f}s "
          f"peak_R={peak_R:.3e} rss_peak={_rss_peak_mb():.0f}MB", flush=True)
    print(f"[M3full] history -> {csv_path}")
    if want_vtu:
        print(f"[M3full] vtu -> {vtu_dir}/step_*.vtu")


if __name__ == "__main__":
    main()
