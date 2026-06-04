#!/usr/bin/env python3
# make_precond_schematic.py
#
# Draw the construction schematic of the block preconditioner used in the paper
# (fig:precond_schematic): the block-triangular sweep over the Hu-Zhang elastic
# saddle system, with the stress block handled by its diagonal (matrix-free A
# action) and the Schur surrogate handled by a two-level auxiliary-space
# V-cycle.  Mirrors Section "Block preconditioners ..." and Algorithm 1.
#
# Output: Frac_huzhang/figures/precond_schematic.{png,pdf}

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("/home/gongshihua/tian/Frac_huzhang/figures")
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#1f3b73"
RED = "#c0392b"
GREEN = "#1e7a46"
GREY = "#444444"
LGREY = "#9aa0a6"


def box(ax, xy, w, h, text, fc="white", ec=BLUE, fs=10, tc="black", lw=1.4):
    """Rounded box centered at xy with multi-line text."""
    x, y = xy
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.012,rounding_size=0.02",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4)


def arrow(ax, p0, p1, color=GREY, lw=1.6, style="-|>", ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                        lw=lw, color=color, linestyle=ls,
                        shrinkA=2, shrinkB=2, zorder=2)
    ax.add_patch(a)


def main(use_tex=False):
    if use_tex:
        matplotlib.rcParams["text.usetex"] = True
    fig, ax = plt.subplots(figsize=(12.0, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- top: saddle system ----------------------------------------------
    box(ax, (0.5, 0.935), 0.54, 0.095,
        r"mechanical saddle system  "
        r"$\mathcal{K}_h(d_h)=[\,A(d_h)\ \ B^{\top}\,;\ \ B\ \ 0\,]$",
        fc="#eef2fb", ec=BLUE, fs=14, lw=1.8)
    ax.text(0.5, 0.862, r"right-preconditioned GMRES  $\bullet$  "
            r"block-triangular  $\mathcal{P}_{\mathrm{tri}}^{-1}(d_h)$",
            ha="center", va="center", fontsize=12, color=GREY, style="italic")
    arrow(ax, (0.5, 0.887), (0.5, 0.795), color=GREY, lw=2.0)

    # ---- middle: the block-triangular sweep (3 boxes) --------------------
    yb = 0.69
    box(ax, (0.165, yb), 0.255, 0.15,
        r"stress block" + "\n" +
        r"$e_\sigma \leftarrow D^{-1} r_\sigma$" + "\n" +
        r"$D=\mathrm{diag}\,A(d_h)$",
        ec=RED, fs=13, lw=1.7)
    box(ax, (0.5, yb), 0.255, 0.15,
        r"Schur solve" + "\n" +
        r"$e_u \leftarrow B_S\,(r_u - B\,e_\sigma)$",
        ec=GREEN, fs=13, lw=1.7)
    box(ax, (0.835, yb), 0.255, 0.15,
        r"back-correct" + "\n" +
        r"$e_\sigma \leftarrow e_\sigma + D^{-1} B^\top e_u$" + "\n" +
        r"return $(e_\sigma,\,-e_u)$",
        ec=RED, fs=13, lw=1.7)
    # long forward-sweep arrows across the gaps
    arrow(ax, (0.293, yb), (0.372, yb), color=GREY, lw=2.4)
    arrow(ax, (0.628, yb), (0.707, yb), color=GREY, lw=2.4)
    ax.text(0.3325, yb + 0.022, "1", ha="center", va="bottom", fontsize=10, color=GREY)
    ax.text(0.6675, yb + 0.022, "2", ha="center", va="bottom", fontsize=10, color=GREY)

    # annotation: matrix-free A action -- highlighted callout box
    arrow(ax, (0.165, yb - 0.075), (0.165, yb - 0.115), color=RED, lw=1.8)
    box(ax, (0.17, yb - 0.205), 0.30, 0.135,
        r"$A(d_h)$ applied " + r"$\mathbf{matrix\!-\!free}$" + "\n" +
        r"$A(d_h)\,\mathbf{s}=\sum_K I_K A_K(d_h) I_K^\top \mathbf{s}$" + "\n" +
        r"(element quadrature; $A$ never assembled)",
        fc="#fdecea", ec=RED, fs=11.5, tc=RED, lw=2.0)

    # ---- callout from Schur box to the V-cycle ---------------------------
    arrow(ax, (0.5, yb - 0.077), (0.5, 0.45), color=GREEN, ls="--", lw=1.8)
    ax.text(0.52, 0.51, r"$B_S \approx \widehat{S}^{-1}$:  two-level $V$-cycle",
            ha="left", va="center", fontsize=12.5, color=GREEN, style="italic")

    # ---- bottom: the two-level V-cycle -----------------------------------
    yf = 0.29          # fine level
    yc = 0.075         # coarse level
    xf1, xf2 = 0.265, 0.735
    box(ax, (xf1, yf), 0.32, 0.135,
        r"fine: $\widehat{S}=B\,D^{-1}B^\top$" + "\n" +
        r"pre-smooth  $\mathrm{GS}_{\mathrm{fwd}}$",
        ec=GREEN, fs=12)
    box(ax, (xf2, yf), 0.32, 0.135,
        r"fine: residual update" + "\n" +
        r"post-smooth  $\mathrm{GS}_{\mathrm{bwd}}$",
        ec=GREEN, fs=12)
    box(ax, (0.5, yc), 0.54, 0.125,
        r"coarse: continuous $P_1$ vector space" + "\n" +
        r"$g(d_h)$-weighted vector Poisson  $\widetilde{L}(d_h)$  (one AMG $V$-cycle)",
        fc="#eaf5ee", ec=GREEN, fs=12)

    # restriction (down) and prolongation (up)
    arrow(ax, (xf1, yf - 0.068), (0.305, yc + 0.063), color=GREY, lw=1.8)
    ax.text(0.235, 0.185, r"$\Pi^\top$" + "\n(restrict)", ha="right",
            va="center", fontsize=11, color=GREY)
    arrow(ax, (0.695, yc + 0.063), (xf2, yf - 0.068), color=GREY, lw=1.8)
    ax.text(0.765, 0.185, r"$\Pi$" + "\n(prolong)", ha="left",
            va="center", fontsize=11, color=GREY)
    arrow(ax, (xf1 + 0.16, yf), (xf2 - 0.16, yf), color=GREEN, lw=1.8)

    # dashed frame around the V-cycle group
    fr = FancyBboxPatch((0.035, 0.005), 0.93, 0.45,
                        boxstyle="round,pad=0.005,rounding_size=0.01",
                        linewidth=1.1, edgecolor=LGREY, facecolor="none",
                        linestyle="--", zorder=1)
    ax.add_patch(fr)
    ax.text(0.95, 0.43, "auxiliary-space Schur preconditioner",
            ha="right", va="center", fontsize=11, color=LGREY, style="italic")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"precond_schematic.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "precond_schematic.pdf")


if __name__ == "__main__":
    main()
