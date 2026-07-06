"""M3 full v2（重设计）runner：Hu–Zhang 应力驱动预测型标记 (M-DF) + predictor–corrector。

修正 v1（per-step 一次加密、η_T-Dörfler 标记滞后 ⇒ 带 h/l0≈0.70 欠分辨、峰值反力 +16%）。
理论见 docs/adaptive/THEORY_marking_strategy.md：
  - 标记 M-DF：𝒟_τ=(2l0/Gc)·max_q H_q ≥ θ_D=β·𝒟_c=β/3，且 h_τ>l0/c_h（预测型，命题1）。
  - 加密 predictor–corrector（Heister–Wheeler–Wick）：步内「解→标记→加密+转移→回到步首重解」
    反复直到无标记（命题3 终止、命题4 接受态 h≤l0/2 分辨保证）。
  - checkpoint–restore：每个 corrector 从**步首损伤** d_ck 重解本载荷（避免同载荷重复解累积损伤）。

配置（环境变量）：
  FRACTUREX_NX(24) DU(2.5e-4) MAX_STEPS(80) BETA(0.6) CH(2.0) MAX_CORR(8)
  DROP_FRAC(0.4) KRES(1e-6) OUTDIR(results/adaptive_m3_pc_model1) SMOKE NO_VTU
  FRACTUREX_ELASTIC_SOLVER 弹性块线性求解器 spsolve(默认)/pardiso/mumps/lgmres
    （细网格用 pardiso；pardiso 缺包即 fail-fast。见 square_direct_needs_pardiso）
运行: PYTHONPATH=$PWD python fracturex/tests/aposteriori/run_m3_pc_model1.py
环境 py312。计算走 bm；numpy 仅文件 I/O。
"""
from __future__ import annotations

import csv
import os
import resource
import time

import numpy as np

from fealpy.backend import backend_manager as bm
from fracturex.cases.square_tension_precrack import SquareTensionPreCrackCase
from fracturex.cases.model2_notch_shear import Model2NotchXStretchCase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.drivers.huzhang_phasefield_staggered import HuZhangPhaseFieldStaggeredDriver

from fracturex.adaptivity.adaptive_staggered import (
    make_assemblers, refine_masked, driving_force_per_cell, mark_driving_force,
    eta_from_state, recovery_indicator_d, mark_recovery, mark_hybrid,
    mark_eta_T_indicator,
)
from fracturex.adaptivity.primal_resolve_real import solve_primal_real


class _Mat:
    E, nu, Gc, l0, ft = 210.0, 0.3, 2.7e-3, 0.015, 3.0
    @property
    def mu(self): return self.E / (2.0 * (1.0 + self.nu))
    @property
    def lam(self): return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


def _f(n, d):
    try: return float(os.environ.get(n, d))
    except (TypeError, ValueError): return float(d)
def _i(n, d):
    try: return int(os.environ.get(n, d))
    except (TypeError, ValueError): return int(d)


def _rss_now_mb():
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * resource.getpagesize() / 1024.0**2
    except Exception:
        return float("nan")
def _rss_peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _make_elastic_solver(name):
    """弹性块线性求解器回调，启动即校验 pardiso/mumps 后端可用（fail-fast）。

    输入: name str，spsolve(默认)/direct/pardiso/mumps/lgmres。
    返回: (callback (A,F)->x, 规范化名)。缺包时 ImportError 含安装提示。
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


def main():
    bm.set_backend("numpy")
    # Anderson 默认开（depth=5）：归因实测它把临界步 staggered 收敛、墙钟 6.98h→1.13h(6×)、
    # 且峰值更贴参照（v2 无 Anderson +2.8% → -1.5%）。显式 FRACTUREX_ANDERSON_DEPTH=0 可关。
    os.environ.setdefault("FRACTUREX_ANDERSON_DEPTH", "5")
    smoke = os.environ.get("FRACTUREX_SMOKE", "0") == "1"
    # 算例选择：square(model1 拉伸,默认) / model2(剪切 x-stretch)。默认行为与历史一致。
    case_name = os.environ.get("FRACTUREX_CASE", "square").lower()
    is_model2 = case_name in ("model2", "shear", "notch_shear")
    nx = _i("FRACTUREX_NX", 24)
    _du_default = (1.0e-3 if smoke else (1.0e-4 if is_model2 else 2.5e-4))
    du = _f("FRACTUREX_DU", _du_default)
    max_steps = _i("FRACTUREX_MAX_STEPS", 4 if smoke else 80)
    beta = _f("FRACTUREX_BETA", 0.6)            # θ_D = β·𝒟_c = β/3
    c_h = _f("FRACTUREX_CH", 2.0)               # h_τ ≤ l0/c_h 即停加密
    max_corr = _i("FRACTUREX_MAX_CORR", 8)      # 每载荷步 corrector 上限
    drop_frac = _f("FRACTUREX_DROP_FRAC", 0.4)
    k_res = _f("FRACTUREX_KRES", 1e-6)
    # 标记器：stress(σ 驱动 heuristic, 兼容默认) / eta_T(§3 Prager–Synge, 论文首推、有理论)
    #         recovery(tian2024) / hybrid(σ ∪ recovery, 需耦合可靠性分析) — 后二为诊断分支。
    marker = os.environ.get("FRACTUREX_MARKER", "stress").strip().lower()
    theta_rec = _f("FRACTUREX_THETA_REC", 0.5)      # recovery / eta_T 阈值参数
    d_cut = _f("FRACTUREX_D_CUT", 0.9)              # σ-mask 上限（hybrid 用）
    rec_method = os.environ.get("FRACTUREX_REC_METHOD", "area").strip().lower()
    # eta_T marker 策略：max(默认, tian2024 同款) 或 L2(Dörfler bulk，弹性阶段会过标记)
    eta_T_strategy = os.environ.get("FRACTUREX_ETA_T_STRATEGY", "max").strip().lower()
    # eta_T marker 停机：相对下降不足即停。Dörfler contraction (CKNS 2008) 保证 q<1 收缩；
    # 若一轮 corrector 后 η_new > eta_decrement · η_old，即视为 diminishing return，停止再加密。
    # eta_decrement=0.7 默认（宽松）；越小越紧、越晚停。
    eta_decrement = _f("FRACTUREX_ETA_DECREMENT", 0.7)
    # eta_T marker 上限：cell min d > d_hi 视为完全断裂胞，marker 前清零。
    # 默认 0.995 对 SENS 过严（seed 邻胞被过滤）；欠分辨案例用 0.999。
    d_hi = _f("FRACTUREX_D_HI", 0.995)
    # (ii) corrector 内中间网格用松 tol 定位标记（解会被加密丢弃，无需解准），接受态补紧 tol 终解。
    #   ⚠ 实测结论（2026-06-14 归因 run）：开 Anderson 后，corrector 的整解本就便宜
    #   （step3 2320s→63s 由 Anderson 贡献，非松 tol），松 tol 反而**微扰标记**致曲线漂移 ~1.5%
    #   且因多一次终解略慢。故**默认关**(tol_coarse=tol_fine)，仅留作可选；要开设 FRACTUREX_TOL_COARSE。
    tol_fine = _f("FRACTUREX_TOL_FINE", 1e-4)
    tol_coarse = _f("FRACTUREX_TOL_COARSE", tol_fine)
    # (a) 严格 Θ 认证：每 cert_every 个接受步做一次连续 primal 重解 + η_τ（0=关，默认关，贵）。
    cert_every = _i("FRACTUREX_CERTIFY_EVERY", 0)
    outdir = os.environ.get("FRACTUREX_OUTDIR",
                            "results/adaptive_m3_pc_model2" if is_model2
                            else "results/adaptive_m3_pc_model1")
    want_vtu = os.environ.get("FRACTUREX_NO_VTU", "0") != "1"
    use_failure_stop = not smoke

    os.makedirs(outdir, exist_ok=True)
    vtu_dir = os.path.join(outdir, "vtu")
    if want_vtu: os.makedirs(vtu_dir, exist_ok=True)
    csv_path = os.path.join(outdir, "history.csv")

    mat = _Mat()
    l0 = mat.l0
    CaseCls = Model2NotchXStretchCase if is_model2 else SquareTensionPreCrackCase
    case = CaseCls(_model=mat, nx=nx, ny=nx, crack_y=0.5, crack_length=0.5)
    print(f"[cfg-case] {'model2(shear x-stretch)' if is_model2 else 'square(model1 tension)'} "
          f"reaction_dir={case.reaction_direction()}", flush=True)
    mesh = case.make_mesh()
    discr = HuZhangDiscretization(case=case, p=3, damage_p=1,
                                  use_relaxation=True).build(mesh=mesh)
    damage = PhaseFieldDamageModel(density_type="AT2", degradation_type="quadratic",
                                   split="hybrid", eps_g=k_res)
    el_asm, ph_asm = make_assemblers(discr, case, damage)
    elastic_solver, solver_name = _make_elastic_solver(
        os.environ.get("FRACTUREX_ELASTIC_SOLVER", "spsolve"))
    driver = HuZhangPhaseFieldStaggeredDriver(
        case=case, discr=discr, damage=damage,
        elastic_assembler=el_asm, phase_assembler=ph_asm,
        tol=tol_fine, maxit=200, d_relaxation=1.0,
        elastic_solver=elastic_solver,
        compute_linear_residual=False, debug=False, timing=False,
        save_vtu_per_step=False, stagger_print_interval=0,
    )
    driver.initialize()

    area_floor = (l0 / c_h) ** 2 / 2.0
    a_depth = _i("FRACTUREX_ANDERSON_DEPTH", 0)   # driver 已从 env 读；此处仅回显
    print(f"[cfg-PC] nx={nx} du={du:.3e} max_steps={max_steps} beta={beta} "
          f"theta_D={beta/3:.3f} c_h={c_h} (h<=l0/{c_h:.0f}) area_floor={area_floor:.2e} "
          f"max_corr={max_corr} k_res={k_res:.1e} solver={solver_name} "
          f"tol_coarse={tol_coarse:.1e}/tol_fine={tol_fine:.1e} anderson_depth={a_depth} "
          f"marker={marker} theta_rec={theta_rec} d_cut={d_cut} rec_method={rec_method} "
          f"eta_decrement={eta_decrement} "
          f"smoke={smoke} vtu={want_vtu}", flush=True)

    fields = ["step", "load", "nc", "dof_sigma", "D_max", "max_d", "reaction",
              "iters", "converged", "n_corr", "refine_events", "n_marked_total",
              "t_step_s", "rss_now_mb", "rss_peak_mb", "eta_tau", "eta_dg"]
    fh = open(csv_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader()

    t0_all = time.perf_counter()
    peak_R = 0.0
    n_done = 0
    for s in range(max_steps):
        load = float(s * du)
        t_s0 = time.perf_counter()
        state = discr.state
        # checkpoint：步首损伤（上一步收敛态）
        d_ck = bm.copy(state.d[:]); r_ck = bm.copy(state.r_hist[:])
        n_corr = 0; refine_events = 0; n_marked_total = 0
        eta_prev = None  # eta_T marker 用于相对下降停机（每 load step 重置）
        info = None
        # (ii) corrector loop：中间网格松 tol 定位标记；接受态（无标记/触上限）再紧 tol 终解。
        while True:
            state = discr.state
            # 从步首损伤重解本载荷（H 由 solve 重建/累积）；中间解用松 tol。
            state.d[:] = d_ck; state.r_hist[:] = r_ck
            driver.tol = tol_coarse
            info = driver.solve_one_step(step=s, load=load)
            Dcell = driving_force_per_cell(discr, damage)
            D_max = float(bm.max(Dcell)) if len(Dcell) else 0.0
            accept = s == 0 or n_corr >= max_corr
            if not accept:
                if marker == "stress":
                    marked = mark_driving_force(discr, Dcell, l0=l0, beta=beta, c_h=c_h)
                elif marker == "eta_t" or marker == "eta_T".lower():
                    pr = solve_primal_real(discr, case, lam=mat.lam, mu=mat.mu,
                                           load=load, k_res=k_res)
                    r = eta_from_state(discr, lam=mat.lam, mu=mat.mu, k_res=k_res,
                                       u_override=pr["uh"])
                    eta_now = float(bm.sqrt(bm.maximum(r["eta"], 0.0)))
                    # 相对下降停机：η_new > eta_decrement · η_old ⇒ diminishing return，停 corrector
                    if eta_prev is not None and eta_now > float(eta_decrement) * eta_prev:
                        marked = bm.zeros(discr.mesh.number_of_cells(), dtype=bool)
                    else:
                        marked = mark_eta_T_indicator(discr, r["eta_T"],
                                                      l0=l0, c_h=c_h, theta=theta_rec,
                                                      d_hi=d_hi,
                                                      strategy=eta_T_strategy)
                    eta_prev = eta_now
                elif marker == "recovery":
                    eta_d = recovery_indicator_d(discr, method=rec_method)
                    marked = mark_recovery(discr, eta_d, l0=l0, c_h=c_h, theta=theta_rec)
                elif marker == "hybrid":
                    eta_d = recovery_indicator_d(discr, method=rec_method)
                    marked = mark_hybrid(discr, damage, Dcell, eta_d, l0=l0,
                                         beta=beta, c_h=c_h, d_cut=d_cut,
                                         theta_rec=theta_rec)
                else:
                    raise ValueError(f"unknown FRACTUREX_MARKER={marker!r}; "
                                     "expected stress/eta_T/recovery/hybrid")
                accept = int(bm.sum(marked)) == 0
            if accept:
                # 接受态：在最终网格上补一次紧 tol 终解（记录物理量）。从**松解收敛态
                # 暖启动**续到 tol_fine（同网格+同载荷的 staggered 不动点唯一，暖续即同解、更快）；
                # 不重置 d_ck——否则丢失 once-latch 的预裂纹（precrack 只在装配器内施加一次）。
                if tol_fine < tol_coarse:
                    driver.tol = tol_fine
                    info = driver.solve_one_step(step=s, load=load)
                    Dcell = driving_force_per_cell(discr, damage)
                    D_max = float(bm.max(Dcell)) if len(Dcell) else 0.0
                break
            # 回到步首损伤再加密（bisect 转移 d_ck，不转移已演化的 d）
            state.d[:] = d_ck; state.r_hist[:] = r_ck
            ref = refine_masked(discr, damage, case, marked)
            if not ref["refined"]:
                # 加密失败（已到尺寸下限）：接受态。当前 state 已被重置到 d_ck，须用松 tol
                # 在本网格上从 d_ck 解一次拿物理（precrack 此前已 latch，d_ck 含 precrack），
                # 再暖续到紧 tol。
                state = discr.state
                state.d[:] = d_ck; state.r_hist[:] = r_ck
                driver.tol = tol_coarse
                info = driver.solve_one_step(step=s, load=load)
                if tol_fine < tol_coarse:
                    driver.tol = tol_fine
                    info = driver.solve_one_step(step=s, load=load)
                Dcell = driving_force_per_cell(discr, damage)
                D_max = float(bm.max(Dcell)) if len(Dcell) else 0.0
                break
            state = discr.state                      # rebuild 后 state 是新对象
            d_ck = bm.copy(state.d[:]); r_ck = bm.copy(state.r_hist[:])
            el_asm, ph_asm = make_assemblers(discr, case, damage)
            driver.elastic_assembler = el_asm; driver.phase_assembler = ph_asm
            n_corr += 1; refine_events += 1; n_marked_total += ref["n_marked"]

        # 接受：state 为本载荷在最终网格上的解
        t_step = time.perf_counter() - t_s0
        R = abs(float(info.meta.get("R", 0.0))); peak_R = max(peak_R, R)
        max_d = float(info.max_d)
        nc = int(discr.mesh.number_of_cells())
        Dcell = driving_force_per_cell(discr, damage)
        D_max = float(bm.max(Dcell)) if len(Dcell) else 0.0

        # (a) 认证：接受态做连续 primal 重解 → 严格 η_τ（reconstruction-free, 常数=1）。
        #   同记 DG-u 版作对照（v2 诚实标注#1：DG-u 与 σ_h 同源 ⇒ 循环虚低）。贵 ⇒ 仅每 k 步。
        eta_tau = float("nan"); eta_dg = float("nan")
        if cert_every > 0 and (s % cert_every == 0):
            try:
                pr = solve_primal_real(discr, case, lam=mat.lam, mu=mat.mu,
                                       load=load, k_res=k_res)
                eta_tau = float(eta_from_state(discr, lam=mat.lam, mu=mat.mu,
                                               k_res=k_res, u_override=pr["uh"])["eta"])
                eta_dg = float(eta_from_state(discr, lam=mat.lam, mu=mat.mu,
                                              k_res=k_res)["eta"])
            except Exception as exc:
                print(f"[certify] step {s} failed: {exc}", flush=True)

        if want_vtu:
            discr.mesh.celldata["D_cell"] = bm.asarray(Dcell)
            try:
                driver._save_vtkfile(os.path.join(vtu_dir, f"step_{s:03d}.vtu"),
                                     cell_mode="mean", q=discr.damage_p + 3,
                                     sigma_eval=driver._sigma_physical_eval)
            except Exception as exc:
                print(f"[vtu] step {s} failed: {exc}", flush=True)

        row = dict(step=s, load=load, nc=nc, dof_sigma=int(info.meta["gdof_sigma"]),
                   D_max=D_max, max_d=max_d, reaction=R, iters=int(info.iters),
                   converged=bool(info.converged), n_corr=n_corr,
                   refine_events=refine_events, n_marked_total=n_marked_total,
                   t_step_s=t_step, rss_now_mb=_rss_now_mb(), rss_peak_mb=_rss_peak_mb(),
                   eta_tau=eta_tau, eta_dg=eta_dg)
        writer.writerow(row); fh.flush(); n_done += 1
        cert_str = f" η_τ={eta_tau:.3e}(DG {eta_dg:.2e})" if cert_every > 0 and (s % cert_every == 0) else ""
        print(f"[PC] step={s:02d} load={load:.3e} nc={nc} dofσ={row['dof_sigma']} "
              f"𝒟max={D_max:.2f} max_d={max_d:.3f} R={R:.3e} iters={info.iters} "
              f"corr={n_corr} refev={refine_events} t={t_step:.1f}s "
              f"rss={row['rss_now_mb']:.0f}/{row['rss_peak_mb']:.0f}MB{cert_str}", flush=True)

        if use_failure_stop and peak_R > 0 and s >= 2 and R < drop_frac * peak_R and max_d > 0.95:
            print(f"[PC] 反力跌破峰值 {drop_frac:.0%}（R={R:.3e}<{drop_frac*peak_R:.3e}），停机。", flush=True)
            break

    fh.close()
    print(f"\n[PC] DONE steps={n_done} wall={time.perf_counter()-t0_all:.1f}s "
          f"peak_R={peak_R:.3e} rss_peak={_rss_peak_mb():.0f}MB -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
