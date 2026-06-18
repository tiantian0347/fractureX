#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 1 of the operator-learning paper: the end-to-end pipeline.

High-fidelity Hu--Zhang phase-field data  ->  tensorization (E_in / E_out)
->  multi-output neural operator N_theta (+ monotone head)  ->  staged
masked losses incl. physics consistency  ->  evaluation.

Renders a clean, publication-quality schematic with matplotlib only (no
external flowchart deps), so it can be regenerated headlessly and dropped
into ``Frac_huzhang/operator_learning/figures/pipeline_overview.pdf``.

Usage
-----
    python plot_pipeline.py [--out DIR]
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------- palette
C_DATA = "#2b6cb0"   # Hu-Zhang data source  (blue)
C_TENS = "#2f855a"   # tensorization         (green)
C_NET = "#6b46c1"    # neural operator       (purple)
C_LOSS = "#b7791f"   # losses                (amber)
C_EVAL = "#9b2c2c"   # evaluation            (red)
C_BG = "#f7fafc"
C_EDGE = "#1a202c"
C_PHYS = "#b7791f"


def _box(ax, xy, w, h, fc, title, lines, title_fs=12.5, body_fs=9.6):
    """A rounded stage box with a coloured header strip and body text."""
    x, y = xy
    pad = 0.018
    # outer rounded panel (light fill, coloured edge)
    panel = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.025",
        linewidth=1.6, edgecolor=fc, facecolor=C_BG, zorder=2,
    )
    ax.add_patch(panel)
    # header strip
    hh = 0.072
    header = FancyBboxPatch(
        (x, y + h - hh), w, hh,
        boxstyle="round,pad=0.0,rounding_size=0.025",
        linewidth=0, facecolor=fc, zorder=3,
    )
    ax.add_patch(header)
    ax.text(x + w / 2, y + h - hh / 2, title, color="white",
            ha="center", va="center", fontsize=title_fs, fontweight="bold",
            zorder=4)
    # body
    ax.text(x + pad + 0.012, y + h - hh - 0.03, lines, color=C_EDGE,
            ha="left", va="top", fontsize=body_fs, zorder=4,
            linespacing=1.5)
    return panel


def _arrow(ax, p0, p1, color=C_EDGE, style="-|>", lw=2.0, rad=0.0, ls="-"):
    a = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=18,
        linewidth=lw, color=color, zorder=1,
        connectionstyle=f"arc3,rad={rad}", linestyle=ls,
    )
    ax.add_patch(a)


def make_figure():
    fig, ax = plt.subplots(figsize=(13.0, 5.4))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    # geometry of the four top-row stages
    y0, h = 0.40, 0.46
    w = 0.205
    gap = 0.058
    xs = [0.018]
    for _ in range(3):
        xs.append(xs[-1] + w + gap)

    _box(
        ax, (xs[0], y0), w, h, C_DATA,
        "1.  Hu--Zhang data",
        "High-fidelity staggered\n"
        "solver  $\\Phi^{HZ},\\,\\Phi^{PF}$\n\n"
        "$\\{(d_n,\\sigma_n)\\}_{n=1}^{N}$\n"
        "$\\sigma_h\\in H(\\mathrm{div};\\mathbb{S})$\n"
        "conforming, sym.,\n"
        "elem.-wise high order",
    )
    _box(
        ax, (xs[1], y0), w, h, C_TENS,
        "2.  Tensorize",
        "$\\mathcal{E}^{in}_h$:  SDF, mask,\n"
        "coords, material,\n"
        "load history\n\n"
        "$\\mathcal{E}^{out}_h$:  $d$, $\\sigma$,\n"
        "($\\mathcal{H}$) on $H\\times W$ grid\n"
        "+ validity mask $m$",
    )
    _box(
        ax, (xs[2], y0), w, h, C_NET,
        "3.  Neural operator",
        "$\\mathcal{N}_\\theta$: FNO2d /\n"
        "U-Net / DeepONet /\n"
        "Geo-FNO\n\n"
        "multi-output head\n"
        "$(d,\\sigma_{xx},\\sigma_{yy},\\sigma_{xy})$\n"
        "+ monotone head",
    )
    _box(
        ax, (xs[3], y0), w, h, C_EVAL,
        "4.  Evaluation",
        "$L^2/H^1$, SSIM,\n"
        "crack IoU, Hausdorff\n"
        "peak-load $e_{\\mathrm{peak}}$\n\n"
        "reaction curve,\n"
        "equilibrium residual,\n"
        "OOD + warm start",
    )

    # forward arrows between stages
    ymid = y0 + h / 2
    for i in range(3):
        x_from = xs[i] + w
        x_to = xs[i + 1]
        _arrow(ax, (x_from, ymid), (x_to, ymid), color=C_EDGE, lw=2.2)

    # bottom: loss / physics-consistency band
    ly, lh = 0.055, 0.235
    lx, lw_ = xs[1] - 0.01, (xs[2] + w) - (xs[1] - 0.01)
    _box(
        ax, (lx, ly), lw_, lh, C_LOSS,
        "Staged masked training objective",
        "$\\mathcal{L}_d+\\lambda_\\sigma\\mathcal{L}_\\sigma"
        "+\\lambda_{H^1}\\mathcal{L}_{H^1}+\\lambda_{\\mathrm{front}}"
        "\\mathcal{L}_{\\mathrm{front}}"
        "+\\lambda_{\\mathrm{eq}}\\mathcal{L}_{\\mathrm{eq}}^{FD/\\mathrm{weak}}"
        "+\\lambda_{\\mathrm{irr}}\\mathcal{L}_{\\mathrm{irr}}$"
        "      (Stages A$\\to$E)",
        title_fs=11.5, body_fs=10.2,
    )

    # training signal from network down to loss, and physics feedback up
    xnet = xs[2] + w / 2
    _arrow(ax, (xnet, y0), (xnet, ly + lh), color=C_NET, lw=1.8, rad=0.0)
    ax.text(xnet + 0.008, (y0 + ly + lh) / 2, "predictions",
            rotation=90, ha="left", va="center", fontsize=8.4, color=C_NET)

    xphys = xs[1] + w / 2
    _arrow(ax, (xphys, ly + lh), (xphys, y0), color=C_PHYS, lw=1.8,
           style="-|>", ls=(0, (4, 2)))
    ax.text(xphys - 0.008, (y0 + ly + lh) / 2,
            "physics consistency\n$\\nabla_h\\!\\cdot\\hat\\sigma+f$",
            rotation=90, ha="right", va="center", fontsize=8.0,
            color=C_PHYS)

    # eval also fed by loss-trained net (light dashed)
    xeval = xs[3] + w / 2
    _arrow(ax, (lx + lw_, ly + lh / 2), (xeval, y0), color=C_EVAL,
           lw=1.4, rad=-0.18, ls=(0, (4, 2)))

    # title strip
    ax.text(
        0.5, 0.965,
        "Stress-supervised multi-output neural operator for "
        "phase-field fracture from $H(\\mathrm{div};\\mathbb{S})$ "
        "Hu--Zhang data",
        ha="center", va="center", fontsize=12.8, fontweight="bold",
        color=C_EDGE,
    )

    fig.tight_layout(pad=0.4)
    return fig


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.normpath(
        os.path.join(here, "..", "..", "..",
                     "Frac_huzhang", "operator_learning", "figures")
    )
    ap.add_argument("--out", default=default_out,
                    help="output directory for pipeline_overview.{pdf,png}")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fig = make_figure()
    pdf = os.path.join(args.out, "pipeline_overview.pdf")
    png = os.path.join(args.out, "pipeline_overview.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print("wrote", pdf)
    print("wrote", png)


if __name__ == "__main__":
    main()
