#!/usr/bin/env python3
# make_model5_figures.py
#
# Build figures for the Ambati / Miehe three-point bending benchmark
# (repo case model5 / Model5ThreePointBendingCase) from a completed
# Hu–Zhang + phase-field run:
#   results/phasefield/model5_three_point_bending/<run>/
#
# Data sources (no VTU required):
#   * history.csv              — load / reaction / max_d
#   * mesh.npz + checkpoints/  — nodal phase-field d (P2 vertex DOFs)
#
# Outputs
# -------
# docs/benchmarks/figures/loaddisp/
#   model5_fx_loaddisp.{png,pdf}
# docs/benchmarks/figures/phasefield/
#   three_point_bending_phasefield_evolution.{png,pdf}
#   three_point_bending_phasefield_final.{png,pdf}
#
# Titles use the formal benchmark name
#   "Three-Point Bending of a Notched Beam"
# (not the internal case id "model5").
#
# Override the run directory with:
#   FRACTUREX_MODEL5_RUN=/path/to/recorder_dir

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASE_NAME = "model5_three_point_bending"
FORMAL_TITLE = "Three-Point Bending of a Notched Beam"

_run_env = os.environ.get("FRACTUREX_MODEL5_RUN", "").strip()
_default_run = ROOT / "results/phasefield" / CASE_NAME / "huzhang_bg_h015_n80"
# Allow a checkout that keeps results/ beside the fractureX package root.
_sibling_run = ROOT.parent / "results/phasefield" / CASE_NAME / "huzhang_bg_h015_n80"
RUN = Path(_run_env) if _run_env else (
    _default_run if _default_run.is_dir() else _sibling_run
)
if not RUN.is_absolute():
    RUN = (ROOT / RUN).resolve()

OUT_LOADDISP = ROOT / "docs/benchmarks/figures/loaddisp"
OUT_PHASE = ROOT / "docs/benchmarks/figures/phasefield"
OUT_LOADDISP.mkdir(parents=True, exist_ok=True)
OUT_PHASE.mkdir(parents=True, exist_ok=True)


def _reaction_row(row: dict) -> float:
    for key in ("residual_force", "reaction_y", "R"):
        raw = row.get(key, "")
        if raw not in ("", None):
            return abs(float(raw))
    return float("nan")


def load_history():
    rows = list(csv.DictReader(open(RUN / "history.csv")))
    steps = np.asarray([int(r["step"]) for r in rows], dtype=int)
    loads = np.asarray([float(r["load"]) for r in rows], dtype=float)
    reac = np.asarray([_reaction_row(r) for r in rows], dtype=float)
    max_d = np.asarray([float(r["max_d"]) for r in rows], dtype=float)
    load_of = {int(r["step"]): float(r["load"]) for r in rows}
    reac_of = {int(r["step"]): _reaction_row(r) for r in rows}
    return steps, loads, reac, max_d, load_of, reac_of


def load_mesh():
    mesh = np.load(RUN / "mesh.npz")
    node = np.asarray(mesh["node"], dtype=float)
    cell = np.asarray(mesh["cell"], dtype=int)
    return node, cell


def list_checkpoints():
    out = []
    for path in sorted((RUN / "checkpoints").glob("step_*.npz")):
        m = re.match(r"step_(\d+)\.npz$", path.name)
        if m:
            out.append((int(m.group(1)), path))
    return out


def damage_nodal(path: Path, nn: int) -> np.ndarray:
    """Vertex values of the P2 Lagrange damage field (first NN DOFs)."""
    d = np.asarray(np.load(path)["d"], dtype=float).reshape(-1)
    return np.clip(d[:nn], 0.0, 1.0)


def nearest_checkpoint(avail, target: int):
    return min(avail, key=lambda t: abs(t[0] - target))


def fig_loaddisp(loads, reac):
    peak_i = int(np.nanargmax(reac))
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.plot(loads, reac, "-o", ms=2.5, lw=1.4, color="#1f4e79")
    ax.plot(
        loads[peak_i],
        reac[peak_i],
        "s",
        color="#c0392b",
        ms=6,
        label=rf"peak $|R|={reac[peak_i]:.3f}$ at $u={loads[peak_i]:.3f}$",
    )
    ax.set_xlabel(r"imposed displacement $u$")
    ax.set_ylabel(r"reaction force $|R|$")
    ax.set_title(rf"{FORMAL_TITLE}: load–displacement response")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_LOADDISP / f"model5_fx_loaddisp.{ext}", dpi=220)
    plt.close(fig)
    print(f"wrote {OUT_LOADDISP / 'model5_fx_loaddisp.png'}")


def fig_phasefield_evolution(node, cell, avail, load_of, reac_of, peak_step: int):
    nn = node.shape[0]
    tri = mtri.Triangulation(node[:, 0], node[:, 1], cell)
    want = [0, max(10, peak_step // 2), peak_step, avail[-1][0]]
    seen = set()
    panels = []
    for target in want:
        step, path = nearest_checkpoint(avail, target)
        if step in seen:
            continue
        seen.add(step)
        panels.append(
            (
                step,
                path,
                load_of.get(step, float("nan")),
                reac_of.get(step, float("nan")),
            )
        )

    n = len(panels)
    fig, axes = plt.subplots(
        1, n, figsize=(3.2 * n + 0.8, 2.8), constrained_layout=True
    )
    if n == 1:
        axes = [axes]
    tpc = None
    for ax, (step, path, load, reac) in zip(axes, panels):
        d = damage_nodal(path, nn)
        tpc = ax.tripcolor(
            tri, d, shading="gouraud", cmap="rainbow", vmin=0.0, vmax=1.0
        )
        ax.set_aspect("equal")
        ax.set_xlim(node[:, 0].min(), node[:, 0].max())
        ax.set_ylim(node[:, 1].min(), node[:, 1].max())
        ax.set_xlabel(r"$x$")
        if ax is axes[0]:
            ax.set_ylabel(r"$y$")
        ax.set_title(rf"$u={load:.3f}$, $R={reac:.3f}$", fontsize=10)
    cb = fig.colorbar(tpc, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label(r"phase field $d$")
    fig.suptitle(rf"{FORMAL_TITLE}: phase-field evolution", fontsize=12)
    stem = "three_point_bending_phasefield_evolution"
    for ext in ("png", "pdf"):
        fig.savefig(OUT_PHASE / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PHASE / (stem + '.png')}  (steps={[p[0] for p in panels]})")


def fig_phasefield_final(node, cell, avail, load_of):
    nn = node.shape[0]
    step, path = avail[-1]
    d = damage_nodal(path, nn)
    load = load_of.get(step, float("nan"))
    tri = mtri.Triangulation(node[:, 0], node[:, 1], cell)

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    tpc = ax.tripcolor(
        tri, d, shading="gouraud", cmap="rainbow", vmin=0.0, vmax=1.0
    )
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(rf"{FORMAL_TITLE}: phase field $d$ at $u={load:.3f}$")
    cb = fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$d$")
    fig.tight_layout()
    stem = "three_point_bending_phasefield_final"
    for ext in ("png", "pdf"):
        fig.savefig(OUT_PHASE / f"{stem}.{ext}", dpi=220)
    plt.close(fig)
    print(f"wrote {OUT_PHASE / (stem + '.png')}  (step={step})")


def main():
    if not (RUN / "history.csv").is_file():
        raise SystemExit(
            f"missing history.csv under RUN={RUN}\n"
            "Set FRACTUREX_MODEL5_RUN to the recorder directory."
        )
    if not (RUN / "mesh.npz").is_file():
        raise SystemExit(f"missing mesh.npz under RUN={RUN}")
    avail = list_checkpoints()
    if not avail:
        raise SystemExit(f"no checkpoints/step_*.npz under {RUN / 'checkpoints'}")

    steps, loads, reac, _max_d, load_of, reac_of = load_history()
    peak_step = int(steps[int(np.nanargmax(reac))])
    node, cell = load_mesh()

    print(f"RUN = {RUN}")
    print(f"peak step={peak_step}, |R|_max={float(np.nanmax(reac)):.4f}")
    fig_loaddisp(loads, reac)
    fig_phasefield_evolution(node, cell, avail, load_of, reac_of, peak_step)
    fig_phasefield_final(node, cell, avail, load_of)
    print("OUT_LOADDISP =", OUT_LOADDISP)
    print("OUT_PHASE    =", OUT_PHASE)


if __name__ == "__main__":
    main()
