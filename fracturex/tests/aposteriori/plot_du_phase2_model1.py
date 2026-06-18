"""Phase 2 (du=1e-4) 出图：du 细化对峰值路径带的影响 + 严格 η_τ 轨迹。

读：
  results/adaptive_m3_pc_model1_du1e4/history.csv         （du=1e-4，本轮 + 每5步 η_τ）
  results/adaptive_m3_pc_model1_v3/history_anderson_canonical.csv （du=2.5e-4 规范版）
  results/phasefield/square_tension_precrack/paper_direct_full_nx120/.../history.csv（参照）

头条：细化 du(2.5e-4→1e-4) **不**让峰值收敛到参照（−1.5%→+3.6%，换了条 staggered
路径），坐实峰值是局部化**本质路径依赖**——论文报 ±4% 路径带而非单点。
副产：每5步严格 η_τ 轨迹 0.063→0.681 近完美线性，η_τ/DG-η ~16–35×（坐实 DG-u 循环虚低）。

约定：纯文件 I/O + 画图（允许 np）。
运行: PYTHONPATH=$PWD python fracturex/tests/aposteriori/plot_du_phase2_model1.py
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "docs/figures/adaptive"
DU1 = "results/adaptive_m3_pc_model1_du1e4/history.csv"                   # du=1e-4
V3 = "results/adaptive_m3_pc_model1_v3/history_anderson_canonical.csv"    # du=2.5e-4
REF = ("results/phasefield/square_tension_precrack/"
       "paper_direct_full_nx120/epsg_1e-06/history.csv")


def _read(path, dcol, rcol, cols=()):
    rows = list(csv.DictReader(open(path)))
    out = {c: [] for c in (dcol, rcol) + tuple(cols)}
    for r in rows:
        try:
            float(r[dcol]); float(r[rcol])
        except (KeyError, ValueError):
            continue
        out[dcol].append(float(r[dcol]))
        out[rcol].append(abs(float(r[rcol])))
        for c in cols:
            try:
                out[c].append(float(r[c]))
            except (KeyError, ValueError):
                out[c].append(np.nan)
    return {k: np.asarray(v) for k, v in out.items()}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    du1 = _read(DU1, "load", "reaction", cols=("eta_tau", "eta_dg", "step", "t_step_s"))
    v3 = _read(V3, "load", "reaction", cols=("step",))
    ref = _read(REF, "disp_y", "R")

    du1_peak = (du1["load"][np.argmax(du1["reaction"])], float(np.max(du1["reaction"])))
    v3_peak = (v3["load"][np.argmax(v3["reaction"])], float(np.max(v3["reaction"])))
    ref_peak = (ref["disp_y"][np.argmax(ref["R"])], float(np.max(ref["R"])))
    d1 = (du1_peak[1] - ref_peak[1]) / ref_peak[1] * 100
    d3 = (v3_peak[1] - ref_peak[1]) / ref_peak[1] * 100
    w1 = float(np.nansum(du1["t_step_s"])) / 3600

    # ---------- Fig 1: load–displacement（du=1e-4 vs du=2.5e-4 vs 参照）----------
    fig, axL = plt.subplots(figsize=(6.8, 4.9))
    # ±4% 路径带（围绕参照峰值）
    axL.axhspan(ref_peak[1] * 0.96, ref_peak[1] * 1.04, color="0.85", alpha=0.5,
                label="±4% path band around ref peak")
    axL.plot(ref["disp_y"], ref["R"], "-", color="0.35", lw=2,
             label=f"uniform nx=120 ref, peak {ref_peak[1]:.3f}")
    axL.plot(v3["load"], v3["reaction"], "s--", color="C7", ms=3, lw=1.3,
             label=f"du=2.5e-4 (v3), peak {v3_peak[1]:.3f} ({d3:+.1f}%)")
    axL.plot(du1["load"], du1["reaction"], "o-", color="C0", ms=3.5, lw=1.6,
             label=f"du=1e-4 ({w1:.1f}h), peak {du1_peak[1]:.3f} ({d1:+.1f}%)")
    axL.scatter([du1_peak[0]], [du1_peak[1]], color="C0", zorder=5, s=45)
    axL.scatter([v3_peak[0]], [v3_peak[1]], color="C7", zorder=5, s=45)
    axL.scatter([ref_peak[0]], [ref_peak[1]], color="0.35", zorder=5, s=45)
    axL.set_xlabel("prescribed displacement  u_y")
    axL.set_ylabel("reaction force  |R|")
    axL.set_title("model1: time-step refinement does NOT collapse peak to reference\n"
                  "(-1.5% -> +3.6%: peaks straddle ref, intrinsic path band)")
    axL.legend(loc="upper left", fontsize=8)
    axL.grid(alpha=0.3)
    fig.tight_layout()
    f1 = os.path.join(OUTDIR, "m3pc_du_phase2_model1_load_displacement.png")
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    # ---------- Fig 2: 严格 η_τ 轨迹 + η_τ/DG-η 比 ----------
    m = np.isfinite(du1["eta_tau"]) & (du1["eta_tau"] > 0)
    ld = du1["load"][m]; et = du1["eta_tau"][m]; ed = du1["eta_dg"][m]
    ratio = et / ed

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    a = ax[0]
    a.plot(ld, et, "o-", color="C3", ms=4, label=r"$\eta_\tau$ (continuous primal, strict)")
    a.plot(ld, ed, "s--", color="C7", ms=3, label=r"$\eta$ (DG-u, cycle-deflated)")
    # 线性参考线（过原点最小二乘）
    slope = float(np.sum(ld * et) / np.sum(ld * ld))
    a.plot(ld, slope * ld, ":", color="C3", lw=1, alpha=0.7,
           label=f"linear fit  slope={slope:.1f}")
    a.set_xlabel("prescribed displacement  u_y"); a.set_ylabel(r"$\eta$")
    a.set_title(r"strict $\eta_\tau$ grows linearly with load (0.063 -> 0.681)")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1]
    a.plot(ld, ratio, "D-", color="C0", ms=4)
    a.set_xlabel("prescribed displacement  u_y")
    a.set_ylabel(r"$\eta_\tau / \eta_{\mathrm{DG}}$")
    a.set_title(r"DG-u under-estimates by ~%d-%dx (cycle deflation, RESULTS caveat #1)"
                % (int(ratio.min()), int(ratio.max())))
    a.grid(alpha=0.3)

    fig.suptitle("Phase 2 (du=1e-4): strict eta_tau certification trajectory", y=1.0)
    fig.tight_layout()
    f2 = os.path.join(OUTDIR, "m3pc_du_phase2_model1_eta_tau.png")
    fig.savefig(f2, dpi=150)
    plt.close(fig)

    print(f"[plot] du=1e-4   peak={du1_peak[1]:.4f} ({d1:+.1f}%) @load={du1_peak[0]:.3e} wall={w1:.2f}h")
    print(f"[plot] du=2.5e-4 peak={v3_peak[1]:.4f} ({d3:+.1f}%) @load={v3_peak[0]:.3e}")
    print(f"[plot] ref       peak={ref_peak[1]:.4f} @load={ref_peak[0]:.3e}")
    print(f"[plot] eta_tau {et[0]:.3f}->{et[-1]:.3f} (linear slope {slope:.1f}); "
          f"eta_tau/eta_dg in [{ratio.min():.1f},{ratio.max():.1f}]")
    print(f"[plot] wrote {f1}")
    print(f"[plot] wrote {f2}")


if __name__ == "__main__":
    main()
