#!/usr/bin/env python3
"""Plot F–u (and model2 damage) from lab results synced to huzhang_fracture_result.

Does not overwrite the existing h=0.1 model5 figures. Last model5 load step is
marked as non-converged (maxit=80). Model2 step 33 is excluded from F–u.

Run from fractureX:
  python scripts/paper_huzhang/plot_synced_completed_runs.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
FX = Path(__file__).resolve().parents[2]
HZ = REPO / "results/huzhang_fracture_result"
OUT_M5 = FX / "docs/benchmarks/figures/loaddisp"
OUT_M2 = FX / "docs/benchmarks/figures/loaddisp"
OUT_EVOL = FX / "docs/adaptive/figures"
AMBATI_CSV = OUT_M5 / "ambati_fig22_model5_tpb_digitized.csv"
FORMAL_TITLE = "Three-Point Bending of a Notched Beam"


def _load_model5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, skiprows=1)
    return np.abs(data[:, 0]), np.abs(data[:, 1])


def _plot_model5() -> None:
    u, r = _load_model5(
        HZ / "phasefield/model5_standard_fem/std_bg_h015_smoke_a_cont/model5_std_force_disp.txt"
    )
    ua, ra = np.loadtxt(AMBATI_CSV, delimiter=",", skiprows=1, usecols=(0, 1)).T
    i_fx = int(np.nanargmax(r))
    i_am = int(np.nanargmax(ra))
    OUT_M5.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 3.7))
    curve_color = "#1F6D8F"
    peak_color = "#C4513D"
    ax.plot(
        u[:-1],
        r[:-1],
        color=curve_color,
        linewidth=2.0,
        marker="o",
        markersize=2.8,
        markerfacecolor="white",
        markeredgewidth=0.8,
        solid_capstyle="round",
        label="standard finite element solution",
    )
    ax.plot(u[-2:], r[-2:], "--", lw=1.1, color=curve_color, alpha=0.7)
    ax.plot(
        u[-1],
        r[-1],
        "o",
        mfc="none",
        color="#7b1fa2",
        ms=6,
        label="endpoint (maxit=80, not converged)",
    )
    ax.scatter(
        u[i_fx],
        r[i_fx],
        s=34,
        color=peak_color,
        zorder=4,
        label=rf"peak $|R_y|={r[i_fx]:.3f}$ at $|u_y|={u[i_fx]:.3f}$ mm",
    )
    ax.axvline(u[i_fx], color=peak_color, linewidth=0.9, linestyle="--", alpha=0.65)
    ax.set_xlabel("prescribed displacement")
    ax.set_ylabel("reaction force")
    ax.grid(axis="y", color="#DDE3E8", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    ax.legend(frameon=False, fontsize=8.3, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_M5 / f"model5_std_fem_h015_loaddisp.{ext}", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(ua, ra, "o-", ms=4, lw=1.2, color="#555555", label="Ambati (2015), Fig. 22")
    ax.plot(u[:-1], r[:-1], "-", lw=1.5, color="#2e7d32", label="standard finite element solution ($h=0.015$)")
    ax.plot(u[-2:], r[-2:], "--", lw=1.2, color="#2e7d32", alpha=0.7)
    ax.plot(
        u[-1],
        r[-1],
        "o",
        mfc="none",
        color="#7b1fa2",
        ms=6,
        label="endpoint (maxit=80, not converged)",
    )
    ax.plot(u[i_fx], r[i_fx], "s", color="#1b5e20", ms=6,
            label=rf"finite element peak {r[i_fx]:.3f} kN @ {u[i_fx]:.3f} mm")
    ax.plot(ua[i_am], ra[i_am], "D", color="#c0392b", ms=5,
            label=rf"Ambati peak {ra[i_am]:.3f} kN @ {ua[i_am]:.3f} mm")
    ax.set_xlabel(r"prescribed displacement $|u_y|$ (mm)")
    ax.set_ylabel(r"reaction force $|R_y|$ (kN)")
    ax.set_xlim(0.0, max(float(u.max()), float(ua.max())) * 1.05)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_M5 / f"model5_std_fem_h015_vs_ambati_loaddisp.{ext}", dpi=220)
    plt.close(fig)
    print(f"model5 peak |R|={r[i_fx]:.4f} kN @ u={u[i_fx]:.4f} mm (Ambati {ra[i_am]:.4f} @ {ua[i_am]:.4f})")


def _load_effstress(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open()))
    # Drop the non-converged last step (step 33).
    valid = [r for r in rows if r["converged"].strip() == "True"]
    return {
        "load": np.array([float(r["load"]) for r in valid]),
        "R": np.array([abs(float(r["reaction"])) for r in valid]),
        "D": np.array([float(r["D_max"]) for r in valid]),
        "NC": np.array([int(r["nc"]) for r in valid]),
        "step": np.array([int(r["step"]) for r in valid]),
    }


def _plot_model2_fu() -> dict[str, np.ndarray]:
    d = _load_effstress(HZ / "results_model2/adaptive_m3_pc_model2_effstress/history.csv")
    i = int(np.argmax(d["R"]))
    OUT_M2.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(d["load"], d["R"], "s-", color="#c0392b", ms=3.5, lw=1.3,
            label=rf"$\mathcal{{D}}_{{\tau,T}}$ marker, peak $|R_x|={d['R'][i]:.3f}$ at $u_x={d['load'][i]:.2e}$")
    ax.scatter([d["load"][i]], [d["R"][i]], s=36, color="#c0392b", zorder=5)
    ax.axvline(d["load"][-1], color="0.5", ls=":", lw=0.8, label=rf"last valid step {int(d['step'][-1])}")
    ax.set_xlabel(r"prescribed shear displacement $u_x$")
    ax.set_ylabel(r"reaction $|R_x|$")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_M2 / "model2_effstress_loaddisp.png", dpi=300)
    fig.savefig(OUT_M2 / "model2_effstress_loaddisp.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(d["load"], d["NC"], "s-", color="#c0392b", ms=3.5, lw=1.3)
    ax.set_xlabel(r"prescribed shear displacement $u_x$")
    ax.set_ylabel(r"number of cells $\mathrm{NC}$")
    ax.set_xlim(left=0)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT_M2 / "model2_effstress_NC.png", dpi=300)
    plt.close(fig)

    print(f"effstress peak |R|={d['R'][i]:.4f} at u={d['load'][i]:.4e}, last valid step={int(d['step'][-1])}")
    return d


def _plot_model2_damage(hist: dict[str, np.ndarray]) -> None:
    import matplotlib.tri as mtri
    from matplotlib.colors import LinearSegmentedColormap
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    vtu_dir = HZ / "results_model2/adaptive_m3_pc_model2_effstress/vtu"
    i_peak = int(np.argmax(hist["R"]))
    panels = [
        (0, r"step 0: $u_x=0$"),
        (int(hist["step"][i_peak]), rf"peak $|R_x|$, $u_x={hist['load'][i_peak]:.2e}$"),
        (int(hist["step"][-1]), rf"last valid, $u_x={hist['load'][-1]:.2e}$"),
    ]

    def read_vtu(step: int):
        path = vtu_dir / f"step_{step:03d}.vtu"
        r = vtk.vtkXMLUnstructuredGridReader()
        r.SetFileName(str(path))
        r.Update()
        m = r.GetOutput()
        xy = vtk_to_numpy(m.GetPoints().GetData())[:, :2]
        nc = m.GetNumberOfCells()
        npts = np.array([m.GetCell(i).GetNumberOfPoints() for i in range(nc)])
        if not np.all(npts == 3):
            raise RuntimeError(f"{path.name}: expected triangles, got npts={set(npts.tolist())}")
        tris = np.empty((nc, 3), dtype=np.int64)
        for i in range(nc):
            c = m.GetCell(i)
            for j in range(3):
                tris[i, j] = c.GetPointId(j)
        arr = m.GetPointData().GetArray("damage")
        if arr is None:
            arr = m.GetPointData().GetArray(0)
        damage = vtk_to_numpy(arr)
        return xy, tris, damage, nc

    cmap = LinearSegmentedColormap.from_list("wr", ["white", "#f4c9c9", "#d94040", "#7a1010"])
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), constrained_layout=True)
    last_pc = None
    for ax, (step, title) in zip(axes, panels):
        xy, tris, damage, nc = read_vtu(step)
        tri = mtri.Triangulation(xy[:, 0], xy[:, 1], tris)
        last_pc = ax.tripcolor(tri, damage, cmap=cmap, shading="gouraud", vmin=0.0, vmax=1.0)
        ax.triplot(tri, color="0.25", lw=0.12, alpha=0.35)
        ax.set_aspect("equal")
        ax.set_title(title + f"\nNC={nc}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(last_pc, ax=axes.ravel().tolist(), shrink=0.72, label=r"damage $d$")
    OUT_EVOL.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_EVOL / "model2_effstress_evolution.png", dpi=220)
    plt.close(fig)
    print(f"wrote {OUT_EVOL / 'model2_effstress_evolution.png'}")


def main() -> None:
    _plot_model5()
    hist = _plot_model2_fu()
    _plot_model2_damage(hist)


if __name__ == "__main__":
    main()
