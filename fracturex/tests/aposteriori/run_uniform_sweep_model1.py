"""均匀网格扫描 runner（§(b) DOF 效率对比基线）：model1 单边预裂纹拉伸。

对一串**均匀**网格分辨率 nx 各跑一次真实 staggered（不加密）到裂纹贯通失效，
记录每个 nx 的**峰值反力 peak_R**、σ-DOF、h/l0、墙钟、峰值内存。配合自适应
predictor–corrector 运行（run_m3_pc_model1.py）做「等峰值精度下 DOF」对比，坐实
THEORY_marking_strategy.md §9 第 3 条：自适应 M-DF 在远少 DOF 下达到与细均匀网格
相当的峰值载荷（消除 +16% 欠分辨偏差，命题 4）。

★ 关键：求解配置与 run_m3_pc_model1.py **逐项一致**（p=3, damage_p=1, AT2/hybrid,
  k_res, l0, du, tol=1e-4, maxit=200, spsolve, crack_y/length, drop_frac）——唯一变量
  是「均匀 nx」vs「自适应」，否则 DOF 对比不公平。

峰值载荷的网格依赖（理论锚点）：均匀 h/l0 越小，峰值反力应单调降向 Γ-收敛极限
（Gerasimov–De Lorenzis；Miehe）。h/l0 = (1/nx)/l0（单位正方形域，h=1/nx）。
  nx= 24 → h/l0≈2.78    nx= 60 → 1.11    nx=100 → 0.67
  nx= 40 → h/l0≈1.67    nx= 80 → 0.83    nx=120 → 0.56（≈现有 nx120 参照）

配置（环境变量）：
  FRACTUREX_NX_LIST   逗号分隔 nx 列表（默认 "24,40,60,80"）。加 "100,120" 取细
                      锚点（很贵，spsolve 到失效可数小时）；亦可复用现有 nx120 参照。
  FRACTUREX_DU        每步 y-位移增量（默认 2.5e-4，与 PC run 一致）
  FRACTUREX_MAX_STEPS 每个 nx 最大载荷步（默认 80）
  FRACTUREX_DROP_FRAC 反力跌破峰值比例即停（默认 0.4）
  FRACTUREX_KRES      残余刚度（默认 1e-6）
  FRACTUREX_ELASTIC_SOLVER 弹性块线性求解器：spsolve(默认)/pardiso/mumps/lgmres。
                      细网格（nx≥120, σ-dof≳4.8e5）spsolve(SuperLU) 可能极慢/段错，
                      用 pardiso（pypardiso，已装）或 mumps（python-mumps，需另装）。
                      见 memory square_direct_needs_pardiso。
  FRACTUREX_OUTDIR    输出目录（默认 results/uniform_sweep_model1）
  FRACTUREX_NO_VTU=1  跳过 VTU（默认跳过；本扫描只要标量曲线）
  FRACTUREX_RESUME=1  断点续算：summary.csv 中已完成的 nx 跳过（默认开）

输出：
  <OUTDIR>/summary.csv          每个 nx 一行（见 SUMMARY_FIELDS）——主交付，喂给绘图
  <OUTDIR>/nx_<nx>/history.csv   该 nx 的逐载荷步历史（载荷–位移曲线用）

运行: PYTHONPATH=$PWD python fracturex/tests/aposteriori/run_uniform_sweep_model1.py
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

from fracturex.adaptivity.adaptive_staggered import make_assemblers


class _Mat:
    """model1 材料（与 run_m3_pc_model1 / run_m3_full_model1 逐项一致）。"""
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


def _rss_peak_mb() -> float:
    """进程峰值常驻内存 (MB)。ru_maxrss 在 Linux 上单位为 KB。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _make_elastic_solver(name):
    """构建弹性块线性求解器回调，启动即校验后端可用（fail-fast）。

    输入:
      name : str，spsolve/direct/pardiso/mumps/lgmres（见 driver.linear_solver）
    返回:
      (callback, name) —— callback 形如 (A, F) -> x，交给 driver.elastic_solver
    异常:
      pardiso/mumps 选中但对应包未安装时，立即 ImportError（含安装提示），
      避免跑到第一次 solve 才崩、白等装配时间。
    """
    m = (name or "spsolve").strip().lower()
    if m == "pardiso":
        try:
            import pypardiso  # noqa: F401
        except ImportError as exc:
            raise ImportError("FRACTUREX_ELASTIC_SOLVER=pardiso 需 pypardiso："
                              "conda install -c conda-forge pypardiso") from exc
    elif m == "mumps":
        try:
            import mumps  # noqa: F401
        except ImportError as exc:
            raise ImportError("FRACTUREX_ELASTIC_SOLVER=mumps 需 python-mumps："
                              "pip install python-mumps（import 名为 mumps）") from exc
    return HuZhangPhaseFieldStaggeredDriver.linear_solver(m), m


# summary.csv 字段（每个 nx 一行）。
SUMMARY_FIELDS = [
    "nx", "ncell", "dof_sigma", "h_over_l0", "peak_R", "peak_load",
    "max_d_at_peak", "n_steps", "n_dnf", "completed", "wall_s", "rss_peak_mb",
]


def run_uniform(nx, *, mat, du, max_steps, drop_frac, k_res, outdir, want_vtu,
                elastic_solver):
    """单个均匀网格 nx 的 staggered 失效模拟。

    输入:
      nx        : int，均匀网格每边单元数（单位正方形域 ⇒ h=1/nx，h/l0=(1/nx)/l0）
      mat       : _Mat，材料 + l0
      du        : float，每载荷步 y-位移增量
      max_steps : int，最大载荷步
      drop_frac : float，反力跌破峰值此比例且 max_d>0.95 即判贯通停机
      k_res     : float，残余刚度 g=(1-d)^2+k_res
      outdir    : str，本扫描总输出目录（本函数写 outdir/nx_<nx>/history.csv）
      want_vtu  : bool，是否每步写 VTU（扫描默认 False）
      elastic_solver : callable (A,F)->x，弹性块线性求解器（见 _make_elastic_solver）
    返回:
      dict，SUMMARY_FIELDS 的一行：含 peak_R/peak_load/dof_sigma/h_over_l0/wall 等
    """
    t0 = time.perf_counter()
    l0 = mat.l0
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
        elastic_solver=elastic_solver,
        compute_linear_residual=False, debug=False, timing=False,
        save_vtu_per_step=False, stagger_print_interval=0,
    )
    driver.initialize()

    nx_dir = os.path.join(outdir, f"nx_{nx}")
    os.makedirs(nx_dir, exist_ok=True)
    vtu_dir = os.path.join(nx_dir, "vtu")
    if want_vtu:
        os.makedirs(vtu_dir, exist_ok=True)

    h_over_l0 = (1.0 / nx) / l0
    ncell = int(mesh.number_of_cells())
    print(f"[uniform nx={nx}] ncell={ncell} h/l0={h_over_l0:.3f} "
          f"du={du:.2e} max_steps={max_steps} k_res={k_res:.1e}", flush=True)

    hist_fields = ["step", "load", "dof_sigma", "max_d", "reaction",
                   "iters", "converged", "t_step_s"]
    hist_path = os.path.join(nx_dir, "history.csv")
    hf = open(hist_path, "w", newline="")
    hw = csv.DictWriter(hf, fieldnames=hist_fields)
    hw.writeheader()

    peak_R = 0.0
    peak_load = 0.0
    max_d_at_peak = 0.0
    n_dnf = 0
    n_steps = 0
    dof_sigma = 0
    for s in range(max_steps):
        load = float(s * du)
        ts0 = time.perf_counter()
        info = driver.solve_one_step(step=s, load=load)
        t_step = time.perf_counter() - ts0

        R = abs(float(info.meta.get("R", 0.0)))
        max_d = float(info.max_d)
        dof_sigma = int(info.meta["gdof_sigma"])
        if not bool(info.converged):
            n_dnf += 1
        if R > peak_R:
            peak_R = R
            peak_load = load
            max_d_at_peak = max_d
        n_steps += 1

        if want_vtu:
            try:
                driver._save_vtkfile(os.path.join(vtu_dir, f"step_{s:03d}.vtu"),
                                     cell_mode="mean", q=discr.damage_p + 3,
                                     sigma_eval=driver._sigma_physical_eval)
            except Exception as exc:
                print(f"[vtu] nx={nx} step {s} failed: {exc}", flush=True)

        hw.writerow(dict(step=s, load=load, dof_sigma=dof_sigma, max_d=max_d,
                         reaction=R, iters=int(info.iters),
                         converged=bool(info.converged), t_step_s=t_step))
        hf.flush()
        print(f"  [nx={nx}] step={s:02d} load={load:.3e} dofσ={dof_sigma} "
              f"max_d={max_d:.3f} R={R:.3e} iters={info.iters} t={t_step:.1f}s",
              flush=True)

        # 失效停机：越过峰值并跌破 drop_frac（与 PC run 同判据）
        if peak_R > 0 and s >= 2 and R < drop_frac * peak_R and max_d > 0.95:
            print(f"  [nx={nx}] 反力跌破峰值 {drop_frac:.0%}（R={R:.3e}<"
                  f"{drop_frac*peak_R:.3e}），贯通停机。", flush=True)
            break

    hf.close()
    completed = bool(peak_R > 0 and s >= 2)  # 至少越过峰值才算这条曲线可用
    wall = time.perf_counter() - t0
    summary = dict(nx=nx, ncell=ncell, dof_sigma=dof_sigma,
                   h_over_l0=round(h_over_l0, 4), peak_R=peak_R,
                   peak_load=peak_load, max_d_at_peak=max_d_at_peak,
                   n_steps=n_steps, n_dnf=n_dnf, completed=completed,
                   wall_s=round(wall, 1), rss_peak_mb=round(_rss_peak_mb(), 1))
    print(f"[uniform nx={nx}] DONE peak_R={peak_R:.4e} @load={peak_load:.3e} "
          f"h/l0={h_over_l0:.3f} dofσ={dof_sigma} steps={n_steps} dnf={n_dnf} "
          f"wall={wall:.1f}s -> {hist_path}", flush=True)
    return summary


def _load_done_nx(summary_path):
    """读 summary.csv 已完成（completed=True）的 nx 集合，供断点续算跳过。"""
    done = set()
    if not os.path.exists(summary_path):
        return done
    try:
        with open(summary_path, newline="") as fh:
            for row in csv.DictReader(fh):
                if str(row.get("completed", "")).strip().lower() in ("true", "1"):
                    done.add(int(row["nx"]))
    except Exception:
        pass
    return done


def main():
    bm.set_backend("numpy")
    nx_list = [int(x) for x in
               os.environ.get("FRACTUREX_NX_LIST", "24,40,60,80").split(",") if x.strip()]
    du = _env_f("FRACTUREX_DU", 2.5e-4)
    max_steps = _env_i("FRACTUREX_MAX_STEPS", 80)
    drop_frac = _env_f("FRACTUREX_DROP_FRAC", 0.4)
    k_res = _env_f("FRACTUREX_KRES", 1e-6)
    outdir = os.environ.get("FRACTUREX_OUTDIR", "results/uniform_sweep_model1")
    want_vtu = os.environ.get("FRACTUREX_NO_VTU", "1") != "1"  # 默认不写 VTU
    resume = os.environ.get("FRACTUREX_RESUME", "1") == "1"
    # 弹性块线性求解器：spsolve(默认)/pardiso/mumps/lgmres。pardiso/mumps 缺包即 fail-fast。
    elastic_solver, solver_name = _make_elastic_solver(
        os.environ.get("FRACTUREX_ELASTIC_SOLVER", "spsolve"))

    os.makedirs(outdir, exist_ok=True)
    summary_path = os.path.join(outdir, "summary.csv")
    done = _load_done_nx(summary_path) if resume else set()

    mat = _Mat()
    print(f"[sweep-cfg] nx_list={nx_list} du={du:.2e} max_steps={max_steps} "
          f"k_res={k_res:.1e} solver={solver_name} resume={resume} "
          f"done={sorted(done)} l0={mat.l0} outdir={outdir}", flush=True)

    # 增量写 summary：保留已有完成行，重跑剩余 nx 后整体重写（按 nx 排序）。
    existing_rows = []
    if os.path.exists(summary_path):
        with open(summary_path, newline="") as fh:
            existing_rows = [r for r in csv.DictReader(fh)
                             if int(r["nx"]) in done]

    t0_all = time.perf_counter()
    new_rows = []
    for nx in nx_list:
        if nx in done:
            print(f"[sweep] nx={nx} 已完成，跳过（resume）。", flush=True)
            continue
        row = run_uniform(nx, mat=mat, du=du, max_steps=max_steps,
                          drop_frac=drop_frac, k_res=k_res, outdir=outdir,
                          want_vtu=want_vtu, elastic_solver=elastic_solver)
        new_rows.append(row)
        # 每完成一个 nx 立即落盘（合并已有 + 新行，按 nx 排序去重）
        merged = {}
        for r in existing_rows + new_rows:
            merged[int(r["nx"])] = r
        with open(summary_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
            w.writeheader()
            for k in sorted(merged):
                w.writerow(merged[k])

    wall = time.perf_counter() - t0_all
    print(f"\n[sweep] DONE new_nx={[r['nx'] for r in new_rows]} "
          f"wall={wall:.1f}s -> {summary_path}", flush=True)
    # 打印当前完整曲线（peak_R vs h/l0），便于一眼看收敛趋势
    if os.path.exists(summary_path):
        print("[sweep] peak_R(h/l0) 收敛曲线：", flush=True)
        with open(summary_path, newline="") as fh:
            for r in csv.DictReader(fh):
                print(f"    nx={r['nx']:>4} h/l0={r['h_over_l0']:>6} "
                      f"dofσ={r['dof_sigma']:>8} peak_R={float(r['peak_R']):.4f} "
                      f"@load={float(r['peak_load']):.3e} "
                      f"completed={r['completed']}", flush=True)


if __name__ == "__main__":
    main()
