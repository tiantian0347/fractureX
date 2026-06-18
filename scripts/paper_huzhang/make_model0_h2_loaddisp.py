#!/usr/bin/env python3
"""model0 h2 load-displacement: direct (完整物理曲线) + aux 一致性叠加 (step0-15)。

读 direct_h2（pardiso，完整 31 步物理正确）与 aux_h2（restart=200，物理收敛止于 step15）
的 history.csv，画 |reaction_y| vs disp_y：
  - direct：实线，完整物理路径（升→峰值 28.1→软化→完全分离趋零）。论文载荷曲线用这条。
  - aux 叠加：step0-15（max_d≤0.998）与 direct 重合（C1 一致性，反力差≤1.5e-3）。
  - step16（max_d→1.0 完全分离瞬间）标为 aux 迭代解的边界：该鞍点 direct(pardiso) 可解，
    aux-GMRES 即使 restart=200+maxit=400 仍 DNF（迭代法在完全分离奇异鞍点的真实边界）。

用法: make_model0_h2_loaddisp.py
输出: docs/figures/precond/model0_h2_loaddisp.{png,pdf}
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
CASE = _REPO / "results/phasefield/model0_circular_notch"
OUTDIR = _REPO / "docs/figures/precond"
OUTDIR.mkdir(parents=True, exist_ok=True)


def load(tag):
    p = CASE / tag / "epsg_1e-06/history.csv"
    rows = list(csv.DictReader(open(p)))
    disp = np.array([float(r["disp_y"]) for r in rows])
    R = np.abs(np.array([float(r["reaction_y"]) for r in rows]))
    maxd = np.array([float(r["max_d"]) for r in rows])
    step = np.array([int(r["step"]) for r in rows])
    return disp, R, maxd, step


def main():
    dd, dR, dmaxd, dstep = load("paper_direct_h2")
    ad, aR, amaxd, astep = load("paper_aux_h2")   # physical clean: step0-15

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    # direct = complete physical load curve
    ax.plot(dd, dR, "-", color="#1f77b4", lw=2.0, zorder=3,
            label="direct (pardiso): full physical path")
    ipk = int(np.argmax(dR))
    ax.annotate(f"peak |R|={dR[ipk]:.1f} (max_d={dmaxd[ipk]:.2f})",
                xy=(dd[ipk], dR[ipk]), xytext=(dd[ipk] - 0.042, dR[ipk] + 5),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="#333", lw=0.8))

    # aux overlay (step0-15, clean agreement with direct)
    ax.plot(ad, aR, "o", mfc="none", mec="#d62728", mew=1.4, ms=6, zorder=4,
            label="aux-space GMRES: agrees with direct (max_d≤0.998)")

    # mark aux DNF boundary at step16 (first un-resolvable step)
    xb = dd[dstep == 16]
    if len(xb):
        xb = float(xb[0])
        ax.axvline(xb, color="#888", ls="--", lw=1.0, zorder=1)
        ax.text(xb + 0.0015, dR.max() * 0.78,
                "step16: max_d→1.0\n(complete separation)\ndirect solves; aux-GMRES\nDNF even at restart=200",
                fontsize=7, color="#555", va="top")

    ax.set_xlabel("applied displacement  $u_y$")
    ax.set_ylabel("|reaction force|  $|R_y|$")
    ax.set_title("model0 h2 (σ-DOF 48k): load–displacement\n"
                 "direct = full physical curve; aux coincides through softening (max_d≤0.998)",
                 fontsize=9.5)
    ax.legend(fontsize=7.8, loc="upper right", framealpha=0.92)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUTDIR / f"model0_h2_loaddisp.{ext}"
        fig.savefig(out, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"saved {out}")

    print(f"\ndirect: {len(dd)} steps, peak |R|={dR.max():.2f}, final |R|={dR[-1]:.3f}")
    print(f"aux: physical clean overlay through step{astep.max()} (max_d={amaxd.max():.4f})")


if __name__ == "__main__":
    main()
