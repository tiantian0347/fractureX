#!/usr/bin/env python3
# make_case_geometries.py
#
# Draw the schematic of the three phase-field benchmark geometries
# (fig:case_geometries) used in the paper's numerical-experiments section.
#
#   model-0  circular-notch tension : unit square with central hole r=0.2,
#            inner circle fully clamped, top edge pulled vertically (u_y).
#   model-1  SEN tension            : unit square, horizontal pre-crack
#            d=1 on {y=0.5, 0<=x<=0.5}; bottom clamped, top pulled u_y.
#   model-2  SEN shear              : same geometry/pre-crack, top sheared u_x.
#
# Output: Frac_huzhang/figures/case_geometries.{png,pdf}

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, FancyArrow

OUT = Path("/home/gongshihua/tian/Frac_huzhang/figures")
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#1f3b73"
RED = "#c0392b"
GREY = "#444444"


def clamp_hatch(ax, x0, x1, y, side="bottom", n=9):
    """Draw a fixed-support (ground) symbol: a baseline with short hatch ticks."""
    xs = np.linspace(x0, x1, n)
    ax.plot([x0, x1], [y, y], color=GREY, lw=2.0, zorder=5)
    dx, dy = 0.035, 0.05
    for x in xs[:-1]:
        if side == "bottom":
            ax.plot([x, x + dx], [y, y - dy], color=GREY, lw=1.0, zorder=5)
        else:  # top
            ax.plot([x, x + dx], [y, y + dy], color=GREY, lw=1.0, zorder=5)


def load_arrows(ax, y, direction, color, n=5, length=0.16):
    """Row of load arrows along the top edge; direction 'up' or 'right'."""
    xs = np.linspace(0.12, 0.88, n)
    for x in xs:
        if direction == "up":
            ax.add_patch(FancyArrow(x, y, 0, length, width=0.006,
                                    head_width=0.035, head_length=0.05,
                                    length_includes_head=True, color=color, zorder=6))
        else:  # right (shear)
            ax.add_patch(FancyArrow(x, y, length, 0, width=0.006,
                                    head_width=0.035, head_length=0.05,
                                    length_includes_head=True, color=color, zorder=6))


def base_square(ax):
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec="k", lw=1.6, zorder=2))
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.22, 1.28)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_model0(ax):
    base_square(ax)
    # central hole, clamped boundary
    ax.add_patch(Circle((0.5, 0.5), 0.2, fc="white", ec=GREY, lw=2.0,
                        hatch="////", zorder=3))
    ax.text(0.5, 0.5, r"$\mathbf{u}=0$" + "\n" + r"$d=0$", ha="center",
            va="center", fontsize=8, zorder=4,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    # vertical load on top edge
    load_arrows(ax, 1.0, "up", BLUE)
    ax.text(0.5, 1.24, r"$u_y=\bar u(t)$", ha="center", va="bottom",
            fontsize=10, color=BLUE)
    ax.text(0.5, -0.30, "(a) circular-notch tension",
            ha="center", va="top", fontsize=10)


def panel_model1(ax):
    base_square(ax)
    # horizontal pre-crack from left edge to centre
    ax.plot([0.0, 0.5], [0.5, 0.5], color=RED, lw=3.2, solid_capstyle="butt",
            zorder=4)
    ax.text(0.24, 0.55, r"pre-crack $d=1$", ha="center", va="bottom",
            fontsize=8, color=RED)
    # bottom clamped
    clamp_hatch(ax, 0.0, 1.0, 0.0, side="bottom")
    ax.text(0.5, -0.16, r"$\mathbf{u}=0$", ha="center", va="top",
            fontsize=9, color=GREY)
    # top vertical load
    load_arrows(ax, 1.0, "up", BLUE)
    ax.text(0.5, 1.24, r"$u_y=\bar u(t)$", ha="center", va="bottom",
            fontsize=10, color=BLUE)
    ax.text(0.5, -0.30, "(b) single-edge-notched tension",
            ha="center", va="top", fontsize=10)


def panel_model2(ax):
    base_square(ax)
    ax.plot([0.0, 0.5], [0.5, 0.5], color=RED, lw=3.2, solid_capstyle="butt",
            zorder=4)
    ax.text(0.24, 0.55, r"pre-crack $d=1$", ha="center", va="bottom",
            fontsize=8, color=RED)
    clamp_hatch(ax, 0.0, 1.0, 0.0, side="bottom")
    ax.text(0.5, -0.16, r"$\mathbf{u}=0$", ha="center", va="top",
            fontsize=9, color=GREY)
    # top horizontal (shear) load
    load_arrows(ax, 1.0, "right", RED, length=0.18)
    ax.text(0.5, 1.24, r"$u_x=\bar u(t),\ u_y=0$", ha="center", va="bottom",
            fontsize=10, color=RED)
    ax.text(0.5, -0.30, "(c) single-edge-notched shear",
            ha="center", va="top", fontsize=10)


def main(use_tex=False):
    if use_tex:
        matplotlib.rcParams["text.usetex"] = True
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    panel_model0(axes[0])
    panel_model1(axes[1])
    panel_model2(axes[2])
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"case_geometries.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "case_geometries.pdf")


if __name__ == "__main__":
    main()
