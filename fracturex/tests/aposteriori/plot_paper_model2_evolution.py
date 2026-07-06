"""SENS 4-panel mesh + damage evolution figure.

Outputs (default: Tian/thesis/fracture_huzhang/adaptive/figures/):
  paper_model2_evolution_4panel.png

Reads (under $FRACTUREX_RESULTS, default ~/repository/results):
  adaptive_m3_pc_model2_eta_T/vtu/step_{000,020,030,039}.vtu

Style matches model0_evolution_4panel.png: damage overlay white->deep red,
mesh in thin black lines, four panels sharing colorbar.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = os.environ.get("FRACTUREX_RESULTS",
                      os.path.expanduser("~/repository/results"))
OUTDIR = os.environ.get(
    "FRACTUREX_FIGDIR",
    os.path.expanduser(
        "~/repository/Tian/thesis/fracture_huzhang/adaptive/figures"))
VTU_DIR = os.path.join(ROOT, "adaptive_m3_pc_model2_eta_T/vtu")

# (step, u_x, label)
PANELS = [
    (0,  0.0000,     r"step $0$: initial state, $u_x{=}0$"),
    (20, 5.000e-3,   r"step $20$: elastic ascent, $u_x{=}5.0{\times}10^{-3}$"),
    (30, 7.500e-3,   r"step $30$: peak load, $u_x{=}7.5{\times}10^{-3}$"),
    (39, 9.750e-3,   r"step $39$: softening, $u_x{=}9.75{\times}10^{-3}$"),
]


def _read_vtu(path):
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(path)
    r.Update()
    m = r.GetOutput()
    pts = vtk_to_numpy(m.GetPoints().GetData())
    xy = pts[:, :2]
    nc = m.GetNumberOfCells()
    tris = np.empty((nc, 3), dtype=np.int64)
    for i in range(nc):
        c = m.GetCell(i)
        for j in range(3):
            tris[i, j] = c.GetPointId(j)
    damage = vtk_to_numpy(m.GetPointData().GetArray("damage"))
    return xy, tris, damage, nc


def _rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
    })


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    _rc()

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "wr", ["white", "#f4c9c9", "#d94040", "#7a1010"])

    fig, axes = plt.subplots(1, 4, figsize=(15.6, 4.0),
                             constrained_layout=True)
    im = None
    for ax, (step, ux, title) in zip(axes, PANELS):
        path = os.path.join(VTU_DIR, f"step_{step:03d}.vtu")
        xy, tris, damage, nc = _read_vtu(path)
        triang = mtri.Triangulation(xy[:, 0], xy[:, 1], tris)
        im = ax.tripcolor(triang, damage, cmap=cmap, vmin=0.0, vmax=1.0,
                          shading="gouraud")
        ax.triplot(triang, color="black", lw=0.15, alpha=0.55)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_title(f"{title}\n$N_C{{=}}{nc}$", fontsize=9.5)
        ax.set_xlabel(r"$x$")
        if ax is axes[0]:
            ax.set_ylabel(r"$y$")

    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.85, aspect=30,
                        location="right", pad=0.01)
    cbar.set_label(r"damage $d$", fontsize=10)

    out = os.path.join(OUTDIR, "paper_model2_evolution_4panel.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
