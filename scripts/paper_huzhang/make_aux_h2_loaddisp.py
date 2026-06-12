#!/usr/bin/env python3
"""aux_h2 完整加载步 load-displacement 曲线（含局部化 + 退化区诚实标注）。

读 model0 aux_h2 完整 history.csv（step0->30），画 |reaction_y| vs disp_y，三段配色：
  - 弹性加载 + 峰值（step<=13，maxd<=0.43）：实线实心
  - 软化 / 完全局部化（step14-16，maxd->1，弹性解仍收敛 O(100)）：实线高亮
  - 完全分离退化区（step17-30，弹性解 DNF lin_res~600、反力非物理翻正）：灰色虚线 + 阴影

判据：用 linear_converged_elastic 区分 bounded-convergence 区 vs DNF 区，
而非凭 step 号硬切——退化区由"弹性线性解未收敛"客观界定。

用法: make_aux_h2_loaddisp.py
输出: docs/figures/precond/aux_h2_loaddisp.{png,pdf}
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
HIST = _REPO / "results/phasefield/model0_circular_notch/paper_aux_h2/epsg_1e-06/history.csv"
OUTDIR = _REPO / "docs/figures/precond"
OUTDIR.mkdir(parents=True, exist_ok=True)


def load():
    rows = list(csv.DictReader(open(HIST)))
    d = {k: np.array([float(r[k]) for r in rows]) for k in
         ("disp_y", "reaction_y", "max_d", "linear_res_elastic", "step")}
    conv = np.array([str(r["linear_converged_elastic"]).strip().lower() == "true" for r in rows])
    return d, conv


def main():
    d, conv = load()
    disp = d["disp_y"]
    R = np.abs(d["reaction_y"])
    maxd = d["max_d"]
    step = d["step"].astype(int)

    # regime masks (objective: elastic solve converged or not)
    dnf = ~conv & (d["linear_res_elastic"] > 1e-3)      # fully-separated singular regime
    phys = ~dnf                                          # physically meaningful
    loc = phys & (maxd > 0.9)                            # localization band (bounded O(100))

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    # physical branch (elastic + softening), solid
    ax.plot(disp[phys], R[phys], "-o", color="#1f77b4", ms=4, lw=1.8,
            label="physical (elastic solve converged)", zorder=3)
    # localization markers
    if loc.any():
        ax.plot(disp[loc], R[loc], "o", mfc="#d62728", mec="#d62728", ms=7, zorder=4,
                label="localization (max_d→1, niter O(100), still converged)")
    # peak annotation
    ipk = int(np.argmax(R * phys))
    ax.annotate(f"peak |R|={R[ipk]:.1f}\n(step{step[ipk]}, max_d={maxd[ipk]:.2f})",
                xy=(disp[ipk], R[ipk]), xytext=(disp[ipk] - 0.035, R[ipk] + 8),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="#333", lw=0.8))

    # DNF / degenerate regime, grey dashed + shaded
    if dnf.any():
        ax.plot(disp[dnf], R[dnf], "--s", color="#999999", ms=4, lw=1.2, alpha=0.8,
                label="fully-separated degenerate (elastic DNF, R non-physical)", zorder=2)
        x0 = disp[dnf].min()
        ax.axvspan(x0, disp.max(), color="#cccccc", alpha=0.25, zorder=0)
        ax.text(x0 + 0.001, R.max() * 0.55,
                "crack fully separated\nsaddle singular → elastic DNF\nreaction not trustworthy",
                fontsize=7.5, color="#666", va="center")

    ax.set_xlabel("applied displacement  $u_y$")
    ax.set_ylabel("|reaction force|  $|R_y|$")
    ax.set_title("model0 aux_h2 (σ-DOF 48k): full load path with honest regime flags",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUTDIR / f"aux_h2_loaddisp.{ext}"
        fig.savefig(out, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"saved {out}")

    # console summary
    print("\nregime split:")
    print(f"  physical (incl. localization): steps {step[phys].min()}–{step[phys].max()}, "
          f"peak |R|={R[phys].max():.2f}")
    if dnf.any():
        print(f"  DNF/degenerate:                steps {step[dnf].min()}–{step[dnf].max()} "
              f"(elastic lin_res up to {d['linear_res_elastic'][dnf].max():.0f}, R flips sign)")


if __name__ == "__main__":
    main()
