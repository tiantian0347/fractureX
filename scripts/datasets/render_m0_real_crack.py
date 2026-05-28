"""Render real crack-evolution figures from a model0 paper_aux_h1 run.

Reads three vtu frames (early / mid / late) from
results/phasefield/model0_circular_notch/paper_aux_h1/epsg_1e-06/vtk/
and plots the damage field as a triangulation contour + crack-front contour.

Run:
    PYTHONPATH=$PWD $FEALPY_PYTHON scripts/datasets/render_m0_real_crack.py

Output: docs/figures/m0/fig_real_crack_d.png
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import vtk
from vtk.util.numpy_support import vtk_to_numpy


VTK_DIR = Path(
    "results/phasefield/model0_circular_notch/paper_aux_h1/epsg_1e-06/vtk"
)
OUT = Path("docs/figures/m0/fig_real_crack_d.png")


def _read_vtu(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    ug = reader.GetOutput()
    pts = vtk_to_numpy(ug.GetPoints().GetData())[:, :2]

    n_cells = ug.GetNumberOfCells()
    tri_cells: list[list[int]] = []
    for i in range(n_cells):
        c = ug.GetCell(i)
        if c.GetCellType() == vtk.VTK_TRIANGLE:
            tri_cells.append([c.GetPointId(j) for j in range(3)])
        elif c.GetCellType() == vtk.VTK_LAGRANGE_TRIANGLE:
            # High-order Lagrange triangle: just use the three corner DOFs
            # (first three point ids by FEALPy's writer convention).
            tri_cells.append([c.GetPointId(0), c.GetPointId(1), c.GetPointId(2)])
    cells = np.asarray(tri_cells, dtype=np.int64)
    damage = vtk_to_numpy(ug.GetPointData().GetArray("damage"))
    return pts, cells, damage


def _frame_paths() -> list[tuple[str, Path]]:
    files = sorted(VTK_DIR.glob("step_*_load_*.vtu"))
    if not files:
        raise FileNotFoundError(f"no vtu under {VTK_DIR}")
    early = files[1]                    # step_0001
    mid = files[len(files) // 2]
    late = files[-1]
    return [
        ("early  step 1", early),
        (f"mid    step {len(files) // 2}", mid),
        (f"late   step {len(files) - 1}", late),
    ]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = _frame_paths()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    levels = np.linspace(0.0, 1.0, 21)
    for ax, (label, path) in zip(axes, frames):
        pts, cells, dmg = _read_vtu(path)
        # Triangulation in (x, y); cells reference node indices.
        # Note: corner-only triangles miss higher-order nodes, so the contour
        # uses the corner subset — the dataset is small (NN=372) so this is
        # fine for a sanity figure; for publication-quality we'd interpolate
        # all DOFs onto a refinement.
        tri = mtri.Triangulation(pts[:, 0], pts[:, 1], cells)
        cs = ax.tricontourf(tri, dmg, levels=levels, cmap="hot_r", vmin=0, vmax=1)
        ax.tricontour(
            tri, dmg, levels=[0.5], colors="cyan", linewidths=1.5
        )
        # outline the notch (circle r=0.2 at center 0.5,0.5 for model0)
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(0.5 + 0.2 * np.cos(theta), 0.5 + 0.2 * np.sin(theta),
                color="black", linewidth=0.8)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{label}\nmax d = {dmg.max():.3f}")

    cbar = fig.colorbar(cs, ax=axes, fraction=0.025, pad=0.02, shrink=0.85)
    cbar.set_label("damage  d")
    fig.suptitle(
        "Real model0 crack evolution from paper_aux_h1 (cyan = d=0.5 contour)",
        y=1.02,
    )
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
