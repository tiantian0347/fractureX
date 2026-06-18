#!/usr/bin/env python3
"""model0 h3 load-displacement: direct (完整物理曲线) + aux 一致性叠加。

读 direct_h3（pardiso，完整 31 步物理正确）与 aux_h3（restart=60 Anderson run，
物理良态止于峰值 step13）的 history.csv，画 |reaction_y| vs disp_y。形式同
make_model0_h2_loaddisp.py：
  - direct：实线，完整物理路径（升→峰值 28.2→完全分离趋零）。论文载荷曲线用这条。
  - aux 叠加：step0-13（max_d≤0.42）与 direct 重合（C1 一致性，rel diff≤1.3e-5）。
  - step14（max_d 0.42→1.0 完全局部化瞬间）aux 已偏离（rel diff 3.4e-2），此后 DNF——
    与 h2 step16 同一迭代法边界（direct 可解、aux-GMRES 在完全分离奇异鞍点 DNF）。

注：h3 比 h2 更脆，step13→14 max_d 直接 0.42→1.0（无 h2 的 0.998 中间软化步），
故 aux 物理良态区干净止于峰值 step13。

用法: make_model0_h3_loaddisp.py
输出: docs/figures/precond/model0_h3_loaddisp.{png,pdf}
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
    rows = list(csv.DictReader(open(CASE / tag / "epsg_1e-06/history.csv")))
    return (np.array([float(r["disp_y"]) for r in rows]),
            np.array([abs(float(r["reaction_y"])) for r in rows]),
            np.array([float(r["max_d"]) for r in rows]),
            np.array([int(r["step"]) for r in rows]))


def main():
    dd, dR, dmd, ds = load("paper_direct_h3")
    ad, aR, amd, asx = load("paper_aux_h3")

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(dd, dR, "-", color="#1f77b4", lw=2.0, zorder=3,
            label="direct (pardiso): full physical path")
    ipk = int(np.argmax(dR))
    ax.annotate(f"peak |R|={dR[ipk]:.1f} (max_d={dmd[ipk]:.2f})",
                xy=(dd[ipk], dR[ipk]), xytext=(dd[ipk] - 0.042, dR[ipk] + 5),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="#333", lw=0.8))

    # aux clean region: step<=13 (max_d<=0.42, reldiff<=1.3e-5)
    clean = asx <= 13
    ax.plot(ad[clean], aR[clean], "o", mfc="none", mec="#d62728", mew=1.4, ms=6, zorder=4,
            label="aux-space GMRES: agrees with direct (max_d≤0.42, step≤13)")
    # step14: max_d->1.0, deviates then DNF
    dev = asx == 14
    if dev.any():
        ax.plot(ad[dev], aR[dev], "^", color="#ff7f0e", ms=8, zorder=4,
                label="aux step14: max_d→1.0 (deviates, then DNF)")
        xb = float(ad[dev][0])
        ax.axvline(xb, color="#888", ls="--", lw=1.0, zorder=1)
        ax.text(xb + 0.0015, dR.max() * 0.7,
                "step14: max_d→1.0\n(complete localization)\ndirect solves; aux-GMRES\nstalls beyond (DNF)",
                fontsize=7, color="#555", va="top")

    ax.set_xlabel("applied displacement  $u_y$")
    ax.set_ylabel("|reaction force|  $|R_y|$")
    ax.set_title("model0 h3 (σ-DOF 184k): load–displacement, direct vs aux\n"
                 "direct = full physical curve; aux coincides through peak (max_d≤0.42)",
                 fontsize=9.5)
    ax.legend(fontsize=7.8, loc="upper right", framealpha=0.92)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUTDIR / f"model0_h3_loaddisp.{ext}"
        fig.savefig(out, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"saved {out}")
    print(f"\ndirect: {len(dd)} steps, peak |R|={dR.max():.2f}, final |R|={dR[-1]:.3f}")
    print(f"aux: clean overlay steps≤13={int(clean.sum())}, deviate step14={int(dev.sum())}")


if __name__ == "__main__":
    main()
