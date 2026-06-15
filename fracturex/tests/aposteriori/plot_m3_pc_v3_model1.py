"""M3 PC v3 (model1) 出图：Anderson 加速版 vs v2(无 Anderson) vs 均匀 nx120 参照
+ 提速/收敛诊断。

读：
  results/adaptive_m3_pc_model1_v3/history_anderson_canonical.csv（v3 = Anderson depth=5，规范版）
  results/adaptive_m3_pc_model1/history.csv（v2 = 无 Anderson/无松 tol）
  results/phasefield/square_tension_precrack/paper_direct_full_nx120/.../history.csv（参照）

头条：Anderson 把墙钟 6.98h→1.13h(6×)、消 v2 的 step22 DNF，且峰值更贴参照
（v2 +2.8% → Anderson −1.5%）——因 Anderson 收敛临界步，不再坐在 v2 的近-DNF 膨胀迭代上。

约定：纯文件 I/O + 画图（允许 np；见 fracturex_multibackend_convention）。
运行: PYTHONPATH=$PWD python fracturex/tests/aposteriori/plot_m3_pc_v3_model1.py
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "docs/figures/adaptive"
V3 = "results/adaptive_m3_pc_model1_v3/history_anderson_canonical.csv"   # Anderson 规范版
V2 = "results/adaptive_m3_pc_model1/history.csv"                          # 无 Anderson
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
    v3 = _read(V3, "load", "reaction",
               cols=("D_max", "nc", "iters", "t_step_s", "step", "rss_peak_mb"))
    v2 = _read(V2, "load", "reaction", cols=("iters", "t_step_s", "step"))
    ref = _read(REF, "disp_y", "R")

    v3_peak = (v3["load"][np.argmax(v3["reaction"])], float(np.max(v3["reaction"])))
    v2_peak = (v2["load"][np.argmax(v2["reaction"])], float(np.max(v2["reaction"])))
    ref_peak = (ref["disp_y"][np.argmax(ref["R"])], float(np.max(ref["R"])))
    d3 = (v3_peak[1] - ref_peak[1]) / ref_peak[1] * 100
    d2 = (v2_peak[1] - ref_peak[1]) / ref_peak[1] * 100
    w3 = float(np.nansum(v3["t_step_s"])) / 3600
    w2 = float(np.nansum(v2["t_step_s"])) / 3600

    # ---------- Fig 1: load–displacement（v3 Anderson vs v2 vs 参照）----------
    fig, axL = plt.subplots(figsize=(6.6, 4.8))
    axL.plot(ref["disp_y"], ref["R"], "-", color="0.35", lw=2,
             label=f"uniform nx=120 ref, peak {ref_peak[1]:.3f}")
    axL.plot(v2["load"], v2["reaction"], "s--", color="C7", ms=3, lw=1.3,
             label=f"v2 no-Anderson ({w2:.1f}h), peak {v2_peak[1]:.3f} ({d2:+.1f}%)")
    axL.plot(v3["load"], v3["reaction"], "o-", color="C0", ms=4, lw=1.8,
             label=f"v3 Anderson ({w3:.1f}h, 6x), peak {v3_peak[1]:.3f} ({d3:+.1f}%)")
    axL.scatter([v3_peak[0]], [v3_peak[1]], color="C0", zorder=5, s=45)
    axL.scatter([ref_peak[0]], [ref_peak[1]], color="0.35", zorder=5, s=45)
    axL.set_xlabel("prescribed displacement  u_y")
    axL.set_ylabel("reaction force  |R|")
    axL.set_title("model1 load-displacement: Anderson-accelerated (v3) vs v2 vs reference")
    axL.legend(loc="upper left", fontsize=8.5)
    axL.grid(alpha=0.3)
    fig.tight_layout()
    f1 = os.path.join(OUTDIR, "m3pc_v3_model1_load_displacement.png")
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    # ---------- Fig 2: 提速诊断 (per-step time v2 vs v3) + iters ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    a = ax[0]
    # 按 step 对齐
    s2 = {int(s): t for s, t in zip(v2["step"], v2["t_step_s"])}
    s3 = {int(s): t for s, t in zip(v3["step"], v3["t_step_s"])}
    steps = sorted(set(s2) & set(s3))
    a.semilogy(steps, [s2[s] for s in steps], "s--", color="C7", ms=3, label="v2 no-Anderson")
    a.semilogy(steps, [s3[s] for s in steps], "o-", color="C0", ms=3, label="v3 Anderson")
    a.set_xlabel("load step"); a.set_ylabel("wall time / step (s, log)")
    a.set_title(f"per-step speedup (step3: 2320s->63s); total {w2:.1f}h -> {w3:.1f}h")
    a.legend(fontsize=8); a.grid(alpha=0.3, which="both")

    a = ax[1]
    i2 = {int(s): n for s, n in zip(v2["step"], v2["iters"])}
    i3 = {int(s): n for s, n in zip(v3["step"], v3["iters"])}
    a.plot(steps, [i2[s] for s in steps], "s--", color="C7", ms=3, label="v2 no-Anderson")
    a.plot(steps, [i3[s] for s in steps], "o-", color="C0", ms=3, label="v3 Anderson")
    a.axhline(200, color="r", ls="--", lw=1, label="maxit=200 (v2 DNF here)")
    a.set_xlabel("load step"); a.set_ylabel("staggered iters")
    a.set_title("Anderson converges critical steps (v2 step22 DNF -> 96)")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    fig.suptitle("M3 PC v3 (model1) Anderson acceleration: speedup + convergence", y=1.0)
    fig.tight_layout()
    f2 = os.path.join(OUTDIR, "m3pc_v3_model1_speedup.png")
    fig.savefig(f2, dpi=150)
    plt.close(fig)

    print(f"[plot] v3 Anderson peak={v3_peak[1]:.4f} ({d3:+.1f}%) wall={w3:.2f}h")
    print(f"[plot] v2 no-Anderson peak={v2_peak[1]:.4f} ({d2:+.1f}%) wall={w2:.2f}h")
    print(f"[plot] ref peak={ref_peak[1]:.4f}")
    print(f"[plot] wrote {f1}")
    print(f"[plot] wrote {f2}")


if __name__ == "__main__":
    main()
