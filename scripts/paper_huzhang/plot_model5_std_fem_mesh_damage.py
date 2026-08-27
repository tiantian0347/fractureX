#!/usr/bin/env python3
"""Plot model5 standard-FEM mesh + damage from the synced restart npz.

No VTU was saved for the h=0.015 lab run. This rebuilds the gmsh beam
(same ``mesh_size=0.015``) and colours nodes by ``d`` from
``model5_std_state.npz`` (end of u=0.03→0.06 continuation; last NR step
did not converge).

Run from fractureX:
  PYTHONPATH=. python scripts/paper_huzhang/plot_model5_std_fem_mesh_damage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from fracturex.cases.phase_field.model5_three_point_bending import Model5StandardFEM
from fracturex.postprocess.vtu_plot import VtuMesh, plot_mesh_scalar

NPZ = (
    REPO
    / "results/huzhang_fracture_result/phasefield/model5_standard_fem"
    / "std_bg_h015_smoke_a_cont/model5_std_state.npz"
)
OUT_DIR = ROOT / "docs/benchmarks/figures/phasefield"
MESH_SIZE = 0.015
CMAP = LinearSegmentedColormap.from_list(
    "wr", ["white", "#f4c9c9", "#d94040", "#7a1010"]
)


def _load_mesh_and_damage() -> VtuMesh:
    """Rebuild the TPB mesh and attach nodal damage from the restart file.

    Returns:
        ``VtuMesh`` with point field ``damage``. ``n_nodes`` must match ``d``.

    Raises:
        FileNotFoundError: npz missing.
        ValueError: rebuilt mesh node count ≠ ``d.size``.
    """
    if not NPZ.is_file():
        raise FileNotFoundError(NPZ)
    d = np.asarray(np.load(NPZ)["d"], dtype=np.float64)
    model = Model5StandardFEM(mesh_size=MESH_SIZE, with_geometric_notch=True)
    mesh = model.build_mesh()
    xy = np.asarray(mesh.entity("node"), dtype=np.float64)[:, :2]
    cell = np.asarray(mesh.entity("cell"), dtype=np.int64)
    nn = int(xy.shape[0])
    if d.shape != (nn,):
        raise ValueError(
            f"rebuilt mesh NN={nn} but npz d.shape={d.shape}; "
            "gmsh connectivity does not match the lab run"
        )
    return VtuMesh(
        xy=xy,
        triangles=cell,
        n_cells=int(cell.shape[0]),
        point_data={"damage": d},
        path=NPZ,
    )


def _panel(ax, mesh: VtuMesh, *, lw: float, title: str) -> object:
    """Draw damage + mesh on ``ax``. Returns the tripcolor mappable."""
    tri = mtri.Triangulation(mesh.xy[:, 0], mesh.xy[:, 1], mesh.triangles)
    tpc = ax.tripcolor(
        tri,
        mesh.scalar("damage"),
        cmap=CMAP,
        shading="gouraud",
        vmin=0.0,
        vmax=1.0,
    )
    ax.triplot(tri, color="0.25", lw=lw, alpha=0.35)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    return tpc


def main() -> None:
    mesh = _load_mesh_and_damage()
    d = mesh.scalar("damage")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dmax = float(d.max())
    title_full = (
        rf"TPB standard FEM $h={MESH_SIZE}$, $u_y=0.06$ mm"
        f"\nNC={mesh.n_cells}, $d_\\mathrm{{max}}$={dmax:.3f} (last step not conv.)"
    )

    plot_mesh_scalar(
        mesh,
        "damage",
        OUT_DIR / "model5_std_fem_h015_mesh_damage.png",
        title=title_full,
        cmap="YlOrRd",
        mesh_lw=0.06,
        figsize=(9.6, 3.0),
        dpi=220,
        colorbar_label=r"damage $d$",
    )
    plot_mesh_scalar(
        mesh,
        "damage",
        OUT_DIR / "model5_std_fem_h015_mesh_damage_notch.png",
        title=rf"notch zoom, $h={MESH_SIZE}$",
        cmap="YlOrRd",
        mesh_lw=0.18,
        figsize=(5.4, 4.4),
        xlim=(3.4, 4.6),
        ylim=(-0.02, 1.35),
        dpi=220,
        colorbar_label=r"damage $d$",
    )

    fig = plt.figure(figsize=(9.0, 6.6))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.0, 0.035], height_ratios=[1.05, 1.25],
        hspace=0.32, wspace=0.08,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[:, 1])
    tpc = _panel(ax0, mesh, lw=0.05, title=title_full)
    ax0.set_xlim(0.0, 8.0)
    ax0.set_ylim(0.0, 2.0)
    ax0.set_xlabel(r"$x$ (mm)")
    ax0.set_ylabel(r"$y$ (mm)")
    _panel(ax1, mesh, lw=0.16, title="notch zoom")
    ax1.set_xlim(3.4, 4.6)
    ax1.set_ylim(-0.02, 1.35)
    ax1.set_xlabel(r"$x$ (mm)")
    ax1.set_ylabel(r"$y$ (mm)")
    fig.colorbar(tpc, cax=cax, label=r"damage $d$")
    stacked = OUT_DIR / "model5_std_fem_h015_mesh_damage_stacked.png"
    fig.savefig(stacked, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(
        f"NN={mesh.xy.shape[0]} NC={mesh.n_cells} dmax={dmax:.4f} "
        f"wrote {OUT_DIR}"
    )


if __name__ == "__main__":
    main()
