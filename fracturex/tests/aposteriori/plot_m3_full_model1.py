"""M3 full (model1) 结果出图：载荷曲线（对账 nx120 参照）+ η/网格/迭代/内存诊断。

读 results/adaptive_m3_full_model1/history.csv（自适应）与
results/phasefield/square_tension_precrack/paper_direct_full_nx120/.../history.csv（均匀 nx120 参照），
出图到 docs/figures/adaptive/。

约定：纯文件 I/O + 画图，用 numpy/matplotlib（非计算内核，允许 np；见 fracturex_multibackend_convention）。
运行: PYTHONPATH=$PWD python fracturex/tests/aposteriori/plot_m3_full_model1.py
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "docs/figures/adaptive"
AD = "results/adaptive_m3_full_model1/history.csv"
REF = ("results/phasefield/square_tension_precrack/"
       "paper_direct_full_nx120/epsg_1e-06/history.csv")
# 可选：其它均匀基线 (label, path, dcol, rcol)，由 (b) 生成后追加
EXTRA_UNIFORM = []  # e.g. [("uniform nx=24", "results/uniform_m3_model1_nx24/history.csv", "load", "reaction")]


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
    ad = _read(AD, "load", "reaction",
               cols=("eta", "nc", "nc_after", "dof_sigma", "iters",
                     "max_d", "rss_peak_mb", "step", "n_marked"))
    ref = _read(REF, "disp_y", "R")

    ad_peak = (ad["load"][np.argmax(ad["reaction"])],
               float(np.max(ad["reaction"])))
    ref_peak = (ref["disp_y"][np.argmax(ref["R"])],
                float(np.max(ref["R"])))

    # ---------- Fig 1: load–displacement (correctness vs nx120) ----------
    fig, axL = plt.subplots(figsize=(6.2, 4.6))
    axL.plot(ref["disp_y"], ref["R"], "-", color="0.35", lw=2,
             label=f"uniform nx=120 (h/l₀≈0.56), peak {ref_peak[1]:.3f}")
    for lab, path, dc, rc in EXTRA_UNIFORM:
        try:
            u = _read(path, dc, rc)
            axL.plot(u[dc], u[rc], "--", lw=1.6,
                     label=f"{lab}, peak {np.max(u[rc]):.3f}")
        except Exception:
            pass
    axL.plot(ad["load"], ad["reaction"], "o-", color="C3", ms=4, lw=1.6,
             label=f"adaptive nx=24+4lvl (h/l₀≈0.70), peak {ad_peak[1]:.3f}")
    axL.scatter([ad_peak[0]], [ad_peak[1]], color="C3", zorder=5, s=40)
    axL.scatter([ref_peak[0]], [ref_peak[1]], color="0.35", zorder=5, s=40)
    axL.set_xlabel("prescribed displacement  u_y")
    axL.set_ylabel("reaction force  |R|")
    axL.set_title("model1 load–displacement: adaptive vs uniform nx=120 reference")
    axL.legend(loc="upper left", fontsize=8.5)
    axL.grid(alpha=0.3)
    fig.tight_layout()
    f1 = os.path.join(OUTDIR, "m3full_model1_load_displacement.png")
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    # ---------- Fig 2: diagnostics 2x2 ----------
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    a = ax[0, 0]
    a.plot(ad["load"], ad["eta"], "o-", color="C0", ms=3)
    a.set_xlabel("displacement u_y"); a.set_ylabel("η (global estimator)")
    a.set_title("equilibrated estimator η vs load")
    a.axvline(ad_peak[0], color="C3", ls=":", lw=1, label="adaptive peak load")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[0, 1]
    a.plot(ad["step"], ad["nc"], "s-", color="C2", ms=3, label="NC (cells)")
    a.set_xlabel("load step"); a.set_ylabel("NC (cells)")
    a.set_title("adaptive mesh growth (tracks crack)")
    a2 = a.twinx()
    a2.bar(ad["step"], ad["n_marked"], color="C1", alpha=0.35, width=0.8)
    a2.set_ylabel("n_marked / step", color="C1")
    a.grid(alpha=0.3)

    a = ax[1, 0]
    a.semilogy(ad["load"], ad["iters"], "o-", color="C4", ms=3)
    a.set_xlabel("displacement u_y"); a.set_ylabel("staggered iters (log)")
    a.set_title("crack-onset iteration blow-up")
    a.axhline(200, color="r", ls="--", lw=1, label="maxit=200")
    a.axvline(ad_peak[0], color="C3", ls=":", lw=1, label="peak load")
    a.legend(fontsize=8); a.grid(alpha=0.3, which="both")

    a = ax[1, 1]
    a.plot(ad["step"], ad["rss_peak_mb"], "o-", color="C5", ms=3, label="running peak RSS")
    a.set_xlabel("load step"); a.set_ylabel("RSS (MB)")
    a.set_title("memory: peak RSS ≈ 1.09 GB (bounded)")
    a.set_ylim(0, max(1200, float(np.nanmax(ad["rss_peak_mb"])) * 1.1))
    a.legend(fontsize=8); a.grid(alpha=0.3)

    fig.suptitle("M3 full (model1) adaptive η-driven staggered: diagnostics", y=1.0)
    fig.tight_layout()
    f2 = os.path.join(OUTDIR, "m3full_model1_diagnostics.png")
    fig.savefig(f2, dpi=150)
    plt.close(fig)

    print(f"[plot] adaptive peak |R|={ad_peak[1]:.4f} @ u_y={ad_peak[0]:.5f}")
    print(f"[plot] nx120 ref peak |R|={ref_peak[1]:.4f} @ u_y={ref_peak[0]:.5f}")
    print(f"[plot] wrote {f1}")
    print(f"[plot] wrote {f2}")


if __name__ == "__main__":
    main()
