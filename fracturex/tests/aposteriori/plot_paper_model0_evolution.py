"""Regenerate model0 2x2 mesh+damage evolution figure for the paper.

Reads the clean 1x4 source PNG (before botched PIL re-layout) and exports a
proper matplotlib 2x2 grid with a shared colorbar.

Run:
  python fracturex/tests/aposteriori/plot_paper_model0_evolution.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(os.environ.get(
    "MODEL0_EVO_SRC",
    ROOT.parent / "Tian/thesis/fracture_huzhang/adaptive/figures"
    / "model0_evolution_4panel.before_2x2.png",
))
OUTDIR = Path(os.environ.get(
    "FRACTUREX_FIGDIR",
    ROOT.parent / "Tian/thesis/fracture_huzhang/adaptive/figures",
))

# Include a four-pixel margin around each detected axes frame.  Cropping
# exactly on the frame removed half of its antialiased boundary pixels.
PANEL_BOXES = [
    (10, 108, 675, 773),
    (705, 108, 1370, 773),
    (1399, 108, 2064, 773),
    (2094, 108, 2758, 773),
]
PANEL_TITLES = [
    (r"step $0$, $u_y=0.0000$", r"$N_C=664$, $\max d=0.000$"),
    (r"step $8$, $u_y=0.0766$", r"$N_C=2{,}344$, $\max d=0.254$"),
    (r"step $13$, $u_y=0.0876$", r"$N_C=2{,}344$, $\max d=0.369$"),
    (r"step $30$, $u_y=0.1250$", r"$N_C=5{,}066$, $\max d=1.000$"),
]


def _rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 10,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
    })


def _pad_panel(panel: Image.Image, tw: int, th: int) -> Image.Image:
    """Center panel on a common canvas so imshow aspect is uniform."""
    canvas = Image.new("RGB", (tw, th), (255, 255, 255))
    x = (tw - panel.size[0]) // 2
    y = (th - panel.size[1]) // 2
    canvas.paste(panel, (x, y))
    return canvas


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"missing source figure: {SRC}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    _rc()

    im = Image.open(SRC).convert("RGB")
    panels = [im.crop(box) for box in PANEL_BOXES]
    tw = max(p.size[0] for p in panels)
    th = max(p.size[1] for p in panels)
    panels = [_pad_panel(p, tw, th) for p in panels]

    cmap = LinearSegmentedColormap.from_list(
        "wr", ["white", "#f4c9c9", "#d94040", "#7a1010"])

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 8.2))
    axes = axes.ravel()

    for ax, panel, (title, subtitle) in zip(axes, panels, PANEL_TITLES):
        ax.imshow(np.asarray(panel), aspect="equal")
        ax.set_title(f"{title}\n{subtitle}", fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("0.35")
            spine.set_linewidth(0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, 1.0))
    sm.set_array([])
    # Keep the full width for both panel columns.  A horizontal colorbar below
    # the grid avoids squeezing or clipping the right-hand panels.
    fig.subplots_adjust(
        left=0.025, right=0.975, bottom=0.13, top=0.94,
        wspace=0.03, hspace=0.16)
    cax = fig.add_axes((0.18, 0.055, 0.64, 0.025))
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(r"damage $d$", fontsize=10)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    out = OUTDIR / "model0_evolution_4panel.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
