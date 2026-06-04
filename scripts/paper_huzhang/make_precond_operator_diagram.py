#!/usr/bin/env python3
# make_precond_operator_diagram.py
#
# Operator/space ("fictitious-space") diagram of the auxiliary-space block
# preconditioner (fig:precond_operator).  Complements the algorithmic flow
# (fig:precond_schematic) with the mathematical structure: the three spaces
# Sigma_h (stress), V_h (displacement / Schur), W_h (continuous P1 auxiliary),
# the transfer operators, the V-cycle error propagation, and the spectral-
# equivalence chain that yields an h- and damage-independent condition number.
#
# Output: Frac_huzhang/figures/precond_operator_diagram.{png,pdf}

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


def node(ax, xy, w, h, text, ec=BLUE, fc="white", fs=10):
    x, y = xy
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.012,rounding_size=0.03",
                       linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4)


def edge(ax, p0, p1, color=GREY, rad=0.0, style="-|>", lw=1.6, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                        lw=lw, color=color, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}",
                        shrinkA=3, shrinkB=3, zorder=2)
    ax.add_patch(a)


def selfloop(ax, xy, text, color, dy=0.135, fs=8.5):
    """A self-loop above a node, with a label."""
    x, y = xy
    a = FancyArrowPatch((x - 0.05, y + 0.05), (x + 0.05, y + 0.05),
                        arrowstyle="-|>", mutation_scale=12, lw=1.4,
                        color=color, connectionstyle="arc3,rad=-1.6",
                        shrinkA=2, shrinkB=2, zorder=2)
    ax.add_patch(a)
    ax.text(x, y + dy, text, ha="center", va="bottom", fontsize=fs, color=color)


def main(use_tex=False):
    if use_tex:
        matplotlib.rcParams["text.usetex"] = True
    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, "auxiliary-space block preconditioner: operators and spaces",
            ha="center", va="center", fontsize=11, color=GREY, style="italic")

    # space nodes
    S = (0.17, 0.66)   # Sigma_h
    V = (0.5, 0.66)    # V_h
    W = (0.84, 0.66)   # W_h
    node(ax, S, 0.26, 0.15,
         r"$\Sigma_h\subset H(\mathrm{div};\mathbb{S})$" + "\n" + r"stress (Hu--Zhang)",
         ec=RED)
    node(ax, V, 0.26, 0.15,
         r"$V_h\subset L^2$" + "\n" + r"displacement",
         ec=BLUE)
    node(ax, W, 0.28, 0.15,
         r"$W_h=(P_1\cap H^1)^n$" + "\n" + r"auxiliary (coarse)",
         ec=GREEN)

    # B : Sigma_h -> V_h  (top),  B^T : V_h -> Sigma_h (bottom)
    edge(ax, (S[0] + 0.13, V[1] + 0.045), (V[0] - 0.13, V[1] + 0.045),
         color=GREY, rad=0.32)
    ax.text(0.335, 0.80, r"$B$  (divergence)", ha="center", fontsize=9, color=GREY)
    edge(ax, (V[0] - 0.13, V[1] - 0.045), (S[0] + 0.13, V[1] - 0.045),
         color=GREY, rad=0.32)
    ax.text(0.335, 0.535, r"$B^\top$", ha="center", fontsize=9, color=GREY)

    # Pi^T : V_h -> W_h (restrict, top),  Pi : W_h -> V_h (prolong, bottom)
    edge(ax, (V[0] + 0.13, V[1] + 0.045), (W[0] - 0.14, V[1] + 0.045),
         color=GREEN, rad=0.32)
    ax.text(0.67, 0.80, r"$\Pi^\top$ (restrict)", ha="center", fontsize=9, color=GREEN)
    edge(ax, (W[0] - 0.14, V[1] - 0.045), (V[0] + 0.13, V[1] - 0.045),
         color=GREEN, rad=0.32)
    ax.text(0.67, 0.535, r"$\Pi$ (prolong)", ha="center", fontsize=9, color=GREEN)

    # self-loops: operators acting within each space
    selfloop(ax, S, r"$A(d_h)\approx D=\mathrm{diag}$,  $B_A=D^{-1}$", RED)
    selfloop(ax, V, r"$\widehat S=B\,D^{-1}B^\top$;  smoother $M$ (GS)", BLUE)
    selfloop(ax, W, r"$\widetilde L(d_h)$: $g(d_h)$-weighted" + "\n" +
             r"vector Poisson (AMG)", GREEN)

    # dashed grouping around W_h: fictitious / coarse space
    fr = FancyBboxPatch((0.685, 0.55), 0.305, 0.34,
                        boxstyle="round,pad=0.004,rounding_size=0.01",
                        linewidth=1.0, edgecolor=LGREY, facecolor="none",
                        linestyle="--", zorder=1)
    ax.add_patch(fr)

    # ---- formulas ---------------------------------------------------------
    ax.text(0.5, 0.40,
            r"$\mathcal{P}_{\mathrm{tri}}^{-1}="
            r"\left(\begin{smallmatrix}I&-D^{-1}B^\top\\0&I\end{smallmatrix}\right)"
            r"\left(\begin{smallmatrix}D^{-1}&0\\0&-B_S\end{smallmatrix}\right)"
            r"\left(\begin{smallmatrix}I&0\\-BD^{-1}&I\end{smallmatrix}\right)$"
            if use_tex else
            r"$\mathcal{P}_{\mathrm{tri}}^{-1}$: block-LU with "
            r"$A^{-1}\!\to D^{-1},\ \ S^{-1}\!\to B_S$",
            ha="center", va="center", fontsize=11, color="black")

    ax.text(0.5, 0.305,
            r"two-level $V$-cycle:  "
            r"$I-B_S\widehat S=(I-M^{-\top}\widehat S)\,"
            r"(I-\Pi\widetilde L^{-1}\Pi^\top\widehat S)\,(I-M^{-1}\widehat S)$",
            ha="center", va="center", fontsize=10.5, color=GREEN)

    # spectral-equivalence chain
    ax.text(0.5, 0.175,
            r"$c\,D\preceq A(d_h)\preceq C\,D"
            r"\ \Rightarrow\ c_1\widehat S\preceq S(d_h)\preceq c_2\widehat S"
            r"\ \Rightarrow\ \kappa(\mathcal{P}^{-1}\mathcal{K}_h(d_h))\leq \mathrm{const}$",
            ha="center", va="center", fontsize=10.5, color=BLUE)
    ax.text(0.5, 0.085,
            r"constants independent of $h$ and of the damage $d_h$",
            ha="center", va="center", fontsize=9, color=GREY, style="italic")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"precond_operator_diagram.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "precond_operator_diagram.pdf")


if __name__ == "__main__":
    main()
