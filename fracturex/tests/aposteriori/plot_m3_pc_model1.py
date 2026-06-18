"""M3 PC v2 (model1) 结果出图：载荷曲线（v2 σ-driven vs v1 η-Dörfler vs 均匀 nx120 参照）
+ 𝒟/网格/迭代/内存诊断。

读：
  results/adaptive_m3_pc_model1/history.csv（v2，σ 驱动 M-DF + predictor–corrector）
  results/adaptive_m3_full_model1/history.csv（v1，η-Dörfler，作对照曲线）
  results/phasefield/square_tension_precrack/paper_direct_full_nx120/.../history.csv（均匀参照）
出图到 docs/figures/adaptive/m3pc_model1_*.png。

头条：v2 把 v1 峰值高估 +16%/起裂 +32% 压到 +2.8%/+2.9%，三曲线对账即此图。

约定：纯文件 I/O + 画图，用 numpy/matplotlib（非计算内核，允许 np；见 fracturex_multibackend_convention）。
运行: PYTHONPATH=$PWD python fracturex/tests/aposteriori/plot_m3_pc_model1.py
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "docs/figures/adaptive"
PC = "results/adaptive_m3_pc_model1/history.csv"        # v2
V1 = "results/adaptive_m3_full_model1/history.csv"      # v1（对照）
REF = ("results/phasefield/square_tension_precrack/"
       "paper_direct_full_nx120/epsg_1e-06/history.csv")  # 均匀参照


def _read(path, dcol, rcol, cols=()):
    """读 csv，dcol=位移列，rcol=反力列（取绝对值），cols=附加列。缺列/非数跳过。"""
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


def _stiffness(u, R, lo=1e-3, hi=3e-3):
    """弹性段割线刚度 dR/du，u∈[lo,hi]。"""
    m = (u >= lo) & (u <= hi)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(u[m], R[m], 1)[0])


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    pc = _read(PC, "load", "reaction",
               cols=("D_max", "nc", "dof_sigma", "iters", "max_d",
                     "rss_peak_mb", "step", "n_marked_total", "n_corr"))
    ref = _read(REF, "disp_y", "R")
    try:
        v1 = _read(V1, "load", "reaction")
    except Exception:
        v1 = None

    pc_peak = (pc["load"][np.argmax(pc["reaction"])], float(np.max(pc["reaction"])))
    ref_peak = (ref["disp_y"][np.argmax(ref["R"])], float(np.max(ref["R"])))
    pc_k = _stiffness(pc["load"], pc["reaction"])
    over_R = (pc_peak[1] - ref_peak[1]) / ref_peak[1] * 100
    over_u = (pc_peak[0] - ref_peak[0]) / ref_peak[0] * 100

    # ---------- Fig 1: load–displacement（v2 vs v1 vs nx120 参照）★头条 ----------
    fig, axL = plt.subplots(figsize=(6.6, 4.8))
    axL.plot(ref["disp_y"], ref["R"], "-", color="0.35", lw=2,
             label=f"uniform nx=120 (h/l₀≈0.56), peak {ref_peak[1]:.3f}")
    if v1 is not None:
        v1_peak = (v1["load"][np.argmax(v1["reaction"])], float(np.max(v1["reaction"])))
        axL.plot(v1["load"], v1["reaction"], "s--", color="C7", ms=3, lw=1.3,
                 label=f"v1 η-Dörfler (h/l₀≈0.70), peak {v1_peak[1]:.3f} (+16%)")
    axL.plot(pc["load"], pc["reaction"], "o-", color="C3", ms=4, lw=1.8,
             label=f"v2 σ-driven PC (h≤l₀/2), peak {pc_peak[1]:.3f} ({over_R:+.1f}%)")
    axL.scatter([pc_peak[0]], [pc_peak[1]], color="C3", zorder=5, s=45)
    axL.scatter([ref_peak[0]], [ref_peak[1]], color="0.35", zorder=5, s=45)
    axL.set_xlabel("prescribed displacement  u_y")
    axL.set_ylabel("reaction force  |R|")
    axL.set_title("model1 load–displacement: σ-driven PC (v2) vs η-Dörfler (v1) vs reference")
    axL.legend(loc="upper left", fontsize=8.5)
    axL.grid(alpha=0.3)
    fig.tight_layout()
    f1 = os.path.join(OUTDIR, "m3pc_model1_load_displacement.png")
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    # ---------- Fig 2: diagnostics 2x2 ----------
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    a = ax[0, 0]
    a.semilogy(pc["load"], np.maximum(pc["D_max"], 1e-3), "o-", color="C0", ms=3)
    a.set_xlabel("displacement u_y"); a.set_ylabel("D_max (driving force, log)")
    a.set_title("M-DF driving force D = (2 l0/Gc) max H  vs load")
    a.axvline(pc_peak[0], color="C3", ls=":", lw=1, label="peak load")
    a.legend(fontsize=8); a.grid(alpha=0.3, which="both")

    a = ax[0, 1]
    a.plot(pc["step"], pc["nc"], "s-", color="C2", ms=3, label="NC (cells)")
    a.set_xlabel("load step"); a.set_ylabel("NC (cells)")
    a.set_title("predictor–corrector mesh growth (front-loaded)")
    a2 = a.twinx()
    a2.bar(pc["step"], pc["n_corr"], color="C1", alpha=0.35, width=0.8)
    a2.set_ylabel("n_corr / step", color="C1")
    a.grid(alpha=0.3)

    a = ax[1, 0]
    a.semilogy(pc["load"], pc["iters"], "o-", color="C4", ms=3)
    a.set_xlabel("displacement u_y"); a.set_ylabel("staggered iters (log)")
    a.set_title("crack-onset iteration blow-up (step22–23 DNF)")
    a.axhline(200, color="r", ls="--", lw=1, label="maxit=200")
    a.axvline(pc_peak[0], color="C3", ls=":", lw=1, label="peak load")
    a.legend(fontsize=8); a.grid(alpha=0.3, which="both")

    a = ax[1, 1]
    a.plot(pc["step"], pc["rss_peak_mb"], "o-", color="C5", ms=3, label="running peak RSS")
    a.set_xlabel("load step"); a.set_ylabel("RSS (MB)")
    a.set_title(f"memory: peak RSS ≈ {np.nanmax(pc['rss_peak_mb'])/1024:.2f} GB")
    a.set_ylim(0, max(2200, float(np.nanmax(pc["rss_peak_mb"])) * 1.1))
    a.legend(fontsize=8); a.grid(alpha=0.3)

    fig.suptitle("M3 PC v2 (model1) σ-driven M-DF + predictor–corrector: diagnostics", y=1.0)
    fig.tight_layout()
    f2 = os.path.join(OUTDIR, "m3pc_model1_diagnostics.png")
    fig.savefig(f2, dpi=150)
    plt.close(fig)

    print(f"[plot] v2 peak |R|={pc_peak[1]:.4f} @ u_y={pc_peak[0]:.5f} "
          f"({over_R:+.1f}% R, {over_u:+.1f}% u vs ref); dR/du={pc_k:.1f}")
    print(f"[plot] nx120 ref peak |R|={ref_peak[1]:.4f} @ u_y={ref_peak[0]:.5f}")
    print(f"[plot] wrote {f1}")
    print(f"[plot] wrote {f2}")


if __name__ == "__main__":
    main()
