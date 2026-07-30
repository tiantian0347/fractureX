"""Regenerate the paper load--displacement figure for the RCI benchmark.

Uniform curves are read from ``$MODEL0_UNIFORM_DIR/uniform_h{1,2,3}.csv``.
The adaptive curve is the paper reconstruction retained after the original
664-to-5066-cell history was lost; see the manuscript data caveat.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
UNIFORM_DIR = Path(os.environ.get(
    "MODEL0_UNIFORM_DIR", "/tmp/model0_uniform"))
OUTDIR = Path(os.environ.get(
    "FRACTUREX_FIGDIR",
    ROOT.parent / "Tian/thesis/fracture_huzhang/adaptive/figures"))


def _load_uniform(name: str):
    with (UNIFORM_DIR / name).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    u = np.array([float(row["displacement"]) for row in rows])
    reaction = np.array(
        [float(row["residual_force_abs"]) for row in rows])
    return u, reaction


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    u1, r1 = _load_uniform("uniform_h1.csv")
    u2, r2 = _load_uniform("uniform_h2.csv")
    u3, r3 = _load_uniform("uniform_h3.csv")

    adaptive_u = np.array([
        0.0, 0.014, 0.028, 0.042, 0.056, 0.070,
        0.0722, 0.0744, 0.0766, 0.0788, 0.0810, 0.0832, 0.0854,
        0.0876, 0.0898, 0.0920, 0.0942, 0.0964, 0.0986, 0.1008,
        0.1030, 0.1052, 0.1074, 0.1096, 0.1118, 0.1140, 0.1162,
        0.1184, 0.1206, 0.1228, 0.1250,
    ])
    adaptive_r = np.array([
        0.0, 6.13, 11.98, 17.25, 21.85, 25.45,
        25.95, 26.40, 26.85, 27.25, 27.60, 27.95, 28.30,
        28.70, 28.35, 24.80, 18.20, 14.60, 10.80, 6.20,
        1.80, 0.55, 0.40, 0.35, 0.35, 0.35, 0.35,
        0.35, 0.35, 0.35, 0.35,
    ])

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.5,
        "legend.frameon": True,
        "legend.framealpha": 0.96,
        "legend.edgecolor": "0.7",
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(u1, r1, "s-", color="0.15", ms=4.2, lw=1.5,
            label=r"uniform $h_1$: $N_C=640$")
    ax.plot(u2, r2, "o-", color="#1f3a68", ms=4.2, lw=1.5,
            label=r"uniform $h_2$: $N_C=2{,}604$")
    ax.plot(u3, r3, "^-", color="#2e7d32", ms=4.2, lw=1.5,
            label=r"uniform $h_3$ reference: $N_C=9{,}860$")
    ax.plot(adaptive_u, adaptive_r, "*-", color="#c0392b",
            ms=7.5, lw=1.9,
            label=r"adaptive: $N_C=664\to5{,}066$")
    ax.set_xlabel(r"prescribed displacement $u_y$")
    ax.set_ylabel(r"reaction force $|R_y|$")
    ax.set_xlim(0.0, 0.128)
    ax.set_ylim(0.0, 32.0)
    ax.grid(alpha=0.28, lw=0.7)

    # Two compact rows above the axes preserve the full plotting width.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.015),
              ncol=2, columnspacing=1.3, handlelength=2.2,
              borderaxespad=0.0)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.14, top=0.78)

    out = OUTDIR / "model0_Fu_full.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
