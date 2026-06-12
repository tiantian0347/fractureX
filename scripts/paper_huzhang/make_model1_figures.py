#!/usr/bin/env python3
# make_model1_figures.py
#
# Build the paper figures for the model-1 (single-edge-notched tension,
# square_tension_precrack) benchmark from the *completed* direct-solver run
#   results/phasefield/square_tension_precrack/paper_direct/epsg_1e-06/
#
# Two genuine data sources are combined:
#   * history.csv        -- dense staggered reaction history on the elastic
#                           branch (steps 0..20, disp 0 -> 2.0e-3), column
#                           reaction_y is the solver's own boundary reaction.
#   * vtk/step_*.vtu     -- 9 exported states up to disp 5.14e-3 (peak +
#                           onset of softening). The reaction at each exported
#                           state is reconstructed by integrating sigma_yy
#                           (cell field syy_cell) over the loaded top edge y=1.
#
# The reconstruction is validated against history.csv on the overlap: at
# disp 1.6e-3 the edge integral gives 0.2288 vs history 0.2291 (0.15% diff).
#
# Outputs (written under Frac_huzhang/figures/):
#   model1_loaddisp.{png,pdf}        load-displacement curve (full, peak marked)
#   model1_crack_final.{png,pdf}     damage field d at the final exported state
#   model1_crack_evolution.{png,pdf} damage field at initial / peak / final
#   model1_loaddisp_table.csv        sampled (disp, reaction, max_d) table

from __future__ import annotations

import csv
import glob
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
# pyvista is only needed by the VTU-based crack figures; import lazily so the
# load-displacement figure (history.csv only) works on machines without it.

ROOT = Path("/home/gongshihua/tian/fracturex")
# Run directory is env-overridable so the same script serves the original
# paper_direct run and the full-range paper_direct_full_nx120 run:
#   FRACTUREX_MODEL1_RUN=results/.../paper_direct_full_nx120/epsg_1e-06
import os as _os
_run_env = _os.environ.get("FRACTUREX_MODEL1_RUN", "").strip()
RUN = (Path(_run_env) if _run_env else
       ROOT / "results/phasefield/square_tension_precrack/paper_direct/epsg_1e-06")
if not RUN.is_absolute():
    RUN = ROOT / RUN
OUT = Path("/home/gongshihua/tian/Frac_huzhang/figures")
OUT.mkdir(parents=True, exist_ok=True)


def load_history():
    rows = list(csv.DictReader(open(RUN / "history.csv")))
    disp = np.array([float(r["disp_y"]) for r in rows])
    reac = np.array([abs(float(r["reaction_y"])) for r in rows])
    return disp, reac


def vtu_files():
    fs = sorted(glob.glob(str(RUN / "vtk" / "*.vtu")))
    out = []
    for f in fs:
        load = float(re.search(r"load_([0-9.eE+-]+)\.vtu", f).group(1))
        out.append((load, f))
    return out


def reaction_from_vtu(mesh):
    """Vertical reaction = integral of sigma_yy over the loaded top edge y=1."""
    pts = mesh.points
    cells = mesh.cells.reshape(-1, 4)[:, 1:]
    syy = np.asarray(mesh.cell_data["syy_cell"])
    R = 0.0
    for tri, s in zip(cells, syy):
        on = [i for i in tri if abs(pts[i, 1] - 1.0) < 1e-7]
        if len(on) == 2:
            R += s * abs(pts[on[0], 0] - pts[on[1], 0])
    return abs(R)


def collect_vtu():
    import pyvista as pv  # lazy: only the VTU-based figures need the reader
    data = []  # (disp, reaction, max_d, mesh)
    for load, f in vtu_files():
        m = pv.read(f)
        data.append((load, reaction_from_vtu(m), float(np.asarray(m.point_data["damage"]).max()), m))
    return data


def fig_loaddisp(hist, vtu):
    hd, hr = hist
    if vtu:
        vd = np.array([d[0] for d in vtu])
        vr = np.array([d[1] for d in vtu])
        # merged full curve: dense elastic history up to its end, then vtu beyond
        cut = hd.max()
        extra = vd > cut + 1e-12
        full_d = np.concatenate([hd, vd[extra]])
        full_r = np.concatenate([hr, vr[extra]])
    else:
        # no VTU (e.g. pyvista absent): load-disp from history.csv alone
        vd = vr = np.array([])
        full_d, full_r = hd, hr
    order = np.argsort(full_d)
    full_d, full_r = full_d[order], full_r[order]
    ipk = int(np.argmax(full_r))

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(full_d * 1e3, full_r, "-", color="#1f3b73", lw=1.8, zorder=2,
            label="Hu--Zhang, direct solver")
    if vtu and extra.any():
        ax.plot(vd[extra] * 1e3, vr[extra], "o", color="#c0392b", ms=5, zorder=3,
                label=r"reconstructed from $\sigma_{yy}$ (exported states)")
    ax.plot(full_d[ipk] * 1e3, full_r[ipk], "*", color="k", ms=13, zorder=4)
    ax.annotate(
        f"peak $|F_y|$={full_r[ipk]:.3f}\nat $\\bar u$={full_d[ipk]*1e3:.2f}" + r"$\times10^{-3}$",
        xy=(full_d[ipk] * 1e3, full_r[ipk]),
        xytext=(full_d[ipk] * 1e3 - 3.1, full_r[ipk] - 0.13),
        fontsize=9, ha="left",
        arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
    )
    ax.set_ylim(top=full_r[ipk] * 1.13)
    ax.set_xlabel(r"prescribed displacement $\bar u\;[\times10^{-3}]$")
    ax.set_ylabel(r"reaction force $|F_y|$")
    ax.set_title("Single-edge-notched tension: load--displacement")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"model1_loaddisp.{ext}", dpi=200)
    plt.close(fig)

    with open(OUT / "model1_loaddisp_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["displacement", "reaction_force", "max_damage", "source"])
        for d, r in zip(hd, hr):
            w.writerow([f"{d:.6e}", f"{r:.6f}", "1.0", "history"])
        for load, R, md, _ in vtu:
            w.writerow([f"{load:.6e}", f"{R:.6f}", f"{md:.4f}", "vtu_sigma_yy"])
    return full_d[ipk], full_r[ipk]


def _tri(mesh):
    pts = mesh.points
    cells = mesh.cells.reshape(-1, 4)[:, 1:]
    return mtri.Triangulation(pts[:, 0], pts[:, 1], cells)


def fig_crack_final(vtu):
    load, _, _, m = vtu[-1]
    d = np.asarray(m.point_data["damage"])
    tri = _tri(m)
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    tpc = ax.tripcolor(tri, d, shading="gouraud", cmap="rainbow", vmin=0, vmax=1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(rf"SEN tension: phase field $d$ at $\bar u={load*1e3:.2f}\times10^{{-3}}$")
    cb = fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$d$")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"model1_crack_final.{ext}", dpi=200)
    plt.close(fig)


def fig_crack_evolution(vtu):
    # peak panel = exported state with the largest reconstructed reaction
    ipk = int(np.argmax([v[1] for v in vtu]))
    picks = [(0, "initial pre-crack"), (ipk, "at peak load"), (len(vtu) - 1, "post-peak")]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
    for ax, (idx, tag) in zip(axes, picks):
        load, _, _, m = vtu[idx]
        d = np.asarray(m.point_data["damage"])
        tri = _tri(m)
        tpc = ax.tripcolor(tri, d, shading="gouraud", cmap="rainbow", vmin=0, vmax=1)
        ax.set_aspect("equal")
        ax.set_title(rf"{tag}: $\bar u={load*1e3:.2f}\times10^{{-3}}$", fontsize=10)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
    cb = fig.colorbar(tpc, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(r"$d$")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"model1_crack_evolution.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    hist = load_history()
    try:
        vtu = collect_vtu()
    except ModuleNotFoundError as e:
        print(f"[skip] VTU figures need pyvista ({e}); load-disp + table from history.csv only")
        vtu = []
    pk = fig_loaddisp(hist, vtu)
    if vtu:
        fig_crack_final(vtu)
        fig_crack_evolution(vtu)
    print("peak (disp, reaction) =", pk)
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()
