#!/usr/bin/env python3
# make_model2_figures.py
#
# Build the load-displacement figure + sampled table for the model-2
# (notch x-stretch, model2_notch_x_stretch) benchmark from the authoritative
# direct-solver run
#   results/phasefield/model2_notch_x_stretch/paper_direct_full/epsg_1e-06/
#
# Data source: history.csv only (no VTU / pyvista needed). The driver records
# its own boundary reaction in column ``reaction_x`` against prescribed
# ``disp_x``; ``max_d`` / ``max_H`` track damage localization and the history
# field. Steps 0..137 (disp_x 0 -> 1.142e-2) are the trustworthy range: H field
# is correct on this NC=51200 mesh (cf. memory model2_paper_direct_frozen_H);
# the still-running tail (step 138+) is appended automatically as it lands.
#
# NOTE on the other two runs (do NOT use here):
#   * paper_direct (NC=93312, sigma-dof 1.95M): trivial u==0 artifact,
#     reaction_x == 0 / max_H == 0 across all 1701 steps -> bogus.
#   * paper_aux: valid but only reaches step 53 (pre-peak).
#
# Outputs (under Frac_huzhang/figures/):
#   model2_loaddisp.{png,pdf}     reaction_x vs disp_x, peak load marked
#   model2_crack_final.{png,pdf}  phase field d at the final exported state
#                                 (rainbow cmap, styled like model0/model1)
#   model2_loaddisp_table.csv     sampled (step, disp_x, reaction_x, max_d, max_H)

from __future__ import annotations

import csv
import glob
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

ROOT = Path("/home/gongshihua/tian/fracturex")
_run_env = os.environ.get("FRACTUREX_MODEL2_RUN", "").strip()
RUN = (Path(_run_env) if _run_env else
       ROOT / "results/phasefield/model2_notch_x_stretch/paper_direct_full/epsg_1e-06")
if not RUN.is_absolute():
    RUN = ROOT / RUN
OUT = Path("/home/gongshihua/tian/Frac_huzhang/figures")
OUT.mkdir(parents=True, exist_ok=True)


def load_history():
    """Read history.csv -> dict of float arrays (sorted by step, deduped).

    Output: dict with keys step, disp_x, reaction_x (abs), max_d, max_H as
    1-D np.float64 arrays. Rows with non-finite reaction are dropped.
    """
    rows = list(csv.DictReader(open(RUN / "history.csv")))
    by_step: dict[int, dict] = {}
    for r in rows:
        try:
            s = int(r["step"])
            rx = float(r["reaction_x"])
        except (KeyError, ValueError):
            continue
        if not np.isfinite(rx):
            continue
        by_step[s] = r  # last writer wins (a resumed step overwrites)
    steps = sorted(by_step)
    g = lambda k: np.array([float(by_step[s][k]) for s in steps])
    return {
        "step": np.array(steps, dtype=float),
        "disp_x": g("disp_x"),
        "reaction_x": np.abs(g("reaction_x")),
        "max_d": g("max_d"),
        "max_H": g("max_H"),
    }


def monotonic_frontier(h):
    """Monotonic loading-envelope mask over the history.

    A mid-run resume restarted this case on a *different* displacement schedule
    (per-step increment 1.0e-4 for steps 0..64, then 8.333e-5 for steps 65+),
    so the prescribed disp_x jumped backward at the step 64->65 boundary
    (0.006400 -> 0.005417). That injects a spurious unload-reload excursion
    (steps 65..76, all disp_x <= 0.0064) that shows up as a disconnected loop
    in the load-displacement curve. The two branches agree where they overlap
    (step 64 R=0.27582 vs step 77 R=0.27651 at disp~0.0064, 0.25%), so the
    physically meaningful monotonic-loading response is recovered by keeping
    only points that strictly advance the prescribed displacement.

    Input:  h -- dict from load_history().
    Output: boolean np.ndarray mask (True = on the monotonic loading front).
            history.csv itself is left untouched (raw measured data).
    """
    disp = h["disp_x"]
    mask = np.zeros(len(disp), dtype=bool)
    mx = -np.inf
    for i, d in enumerate(disp):
        if d > mx + 1e-12:
            mask[i] = True
            mx = d
    return mask


# Crack-pattern step: the SENS run is only trustworthy through step 137 (the H
# field develops spurious blips and the broken specimen accrues diffuse spurious
# damage in the post-peak tail 138..200, see the max_H-spike report in main()).
# By step 137 the kinked shear crack is fully formed but the background is still
# clean, so that is the state plotted as the "final" crack pattern. Override the
# step via FRACTUREX_MODEL2_CRACK_STEP (use -1 for the very last exported state).
CRACK_STEP = int(os.environ.get("FRACTUREX_MODEL2_CRACK_STEP", "137"))
# Colormap floor: clamp d below this to the bottom (purple) colour so the
# residual sub-threshold spurious spots/diffuse background do not distract from
# the crack. Set to 0.0 to recover the plain model0/model1 [0,1] colour range.
CRACK_VMIN = float(os.environ.get("FRACTUREX_MODEL2_CRACK_VMIN", "0.5"))
# Connectivity filter: the standard AT2 model nucleates isolated *spurious*
# damage spots away from the crack in the SENS shear field (e.g. the d->1 blob
# at (0.20,0.21) that appears at step 126, where the material was intact one
# coarse load step earlier, H=0.07 << crack-tip). These are a known model
# artifact, not physical crack. When CRACK_CONNECTED=1 (default) the damage
# field is masked to the connected component(s) reachable from the notch
# (the pre-crack ligament y=0.5, x<=0.5) along mesh edges through cells with
# d>CRACK_CONNECT_THR; isolated spurious damage is set to 0 for the figure
# only (raw VTU untouched). Set CRACK_CONNECTED=0 to plot the raw field.
CRACK_CONNECTED = os.environ.get("FRACTUREX_MODEL2_CRACK_CONNECTED", "1").strip() == "1"
CRACK_CONNECT_THR = float(os.environ.get("FRACTUREX_MODEL2_CRACK_CONNECT_THR", "0.5"))


def _crack_vtu():
    """(path, step, load) of the exported VTU used for the crack figure.

    Picks the file whose step == CRACK_STEP; falls back to the highest step
    when CRACK_STEP < 0 or that exact step was not exported.
    """
    fs = glob.glob(str(RUN / "vtk" / "step_*.vtu"))
    keyed = []
    for f in fs:
        m = re.search(r"step_(\d+)_load_([0-9.eE+-]+)\.vtu", os.path.basename(f))
        if m:
            keyed.append((int(m.group(1)), float(m.group(2)), f))
    if not keyed:
        return None, None, None
    if CRACK_STEP >= 0:
        exact = [t for t in keyed if t[0] == CRACK_STEP]
        if exact:
            step, load, f = exact[0]
            return f, step, load
    step, load, f = max(keyed, key=lambda t: t[0])
    return f, step, load


def _connected_damage_mask(pts, tris, d, thr):
    """Boolean node mask: damage connected to the notch ligament.

    Seeds from notch nodes (d>thr on y~0.5, x<=0.15) and floods over mesh edges
    through nodes with d>thr. Nodes not reached are isolated spurious damage.
    Returns an all-True mask if no seed is found (so the figure never blanks).
    """
    from collections import defaultdict
    x, y = pts[:, 0], pts[:, 1]
    adj = defaultdict(set)
    for a, b, c in tris:
        adj[a].update((b, c)); adj[b].update((a, c)); adj[c].update((a, b))
    seed = [i for i in range(len(d))
            if d[i] > thr and 0.47 < y[i] < 0.53 and x[i] <= 0.15]
    if not seed:
        return np.ones(len(d), dtype=bool)
    visited = set(seed); stack = list(seed)
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in visited and d[v] > thr:
                visited.add(v); stack.append(v)
    keep = np.zeros(len(d), dtype=bool)
    keep[list(visited)] = True
    return keep


def fig_crack_final():
    """Phase field d at the final (trustworthy) state (SENS, single-edge-notched
    shear). Styling matches model0_crack_final / model1_crack_final: rainbow
    colormap, gouraud-shaded triangulation, square aspect. The colour floor is
    raised to CRACK_VMIN so the post-peak diffuse spurious damage stays purple.

    Reads the VTU with the ``vtk`` package directly (pyvista is not required),
    so it runs on the same host as the solver.
    """
    f, step, load = _crack_vtu()
    if f is None:
        print("[skip] fig_crack_final: no VTU found under", RUN / "vtk")
        return
    import vtk  # lazy: only this figure needs the VTU reader
    from vtk.util.numpy_support import vtk_to_numpy

    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(f)
    r.Update()
    g = r.GetOutput()
    pts = vtk_to_numpy(g.GetPoints().GetData())
    d = vtk_to_numpy(g.GetPointData().GetArray("damage"))
    tris = np.array([[g.GetCell(i).GetPointIds().GetId(j) for j in range(3)]
                     for i in range(g.GetNumberOfCells())])
    n_removed = 0
    if CRACK_CONNECTED:
        keep = _connected_damage_mask(pts, tris, d, CRACK_CONNECT_THR)
        hi = d > CRACK_CONNECT_THR
        n_removed = int((hi & ~keep).sum())
        d = np.where(keep, d, 0.0)
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)

    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    tpc = ax.tripcolor(tri, d, shading="gouraud", cmap="rainbow",
                       vmin=CRACK_VMIN, vmax=1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(rf"SEN shear: phase field $d$ at $\bar u={load:.3e}$")
    cb = fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$d$")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"model2_crack_final.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {OUT/'model2_crack_final.png'}, .pdf  (state step={step} "
          f"u={load:.3e}, max_d={float(d.max()):.4f}, cmap vmin={CRACK_VMIN}, "
          f"connectivity_filter={'on' if CRACK_CONNECTED else 'off'}, "
          f"isolated_spurious_nodes_removed={n_removed})")


def _read_damage_vtu(f, apply_filter=True):
    """Read (pts, tris, d, n_removed) from a VTU; optionally connectivity-filter d."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(f)
    r.Update()
    g = r.GetOutput()
    pts = vtk_to_numpy(g.GetPoints().GetData())
    d = vtk_to_numpy(g.GetPointData().GetArray("damage")).copy()
    tris = np.array([[g.GetCell(i).GetPointIds().GetId(j) for j in range(3)]
                     for i in range(g.GetNumberOfCells())])
    n_removed = 0
    if apply_filter and CRACK_CONNECTED:
        keep = _connected_damage_mask(pts, tris, d, CRACK_CONNECT_THR)
        n_removed = int(((d > CRACK_CONNECT_THR) & ~keep).sum())
        d = np.where(keep, d, 0.0)
    return pts, tris, d, n_removed


def _vtu_for_step(target):
    """(path, step, load) of the exported VTU at step==target (or nearest <=)."""
    fs = glob.glob(str(RUN / "vtk" / "step_*.vtu"))
    keyed = []
    for f in fs:
        m = re.search(r"step_(\d+)_load_([0-9.eE+-]+)\.vtu", os.path.basename(f))
        if m:
            keyed.append((int(m.group(1)), float(m.group(2)), f))
    if not keyed:
        return None, None, None
    exact = [t for t in keyed if t[0] == target]
    if exact:
        s, l, f = exact[0]
        return f, s, l
    le = [t for t in keyed if t[0] <= target]
    s, l, f = (max(le, key=lambda t: t[0]) if le else min(keyed, key=lambda t: t[0]))
    return f, s, l


def fig_crack_evolution():
    """Three-panel damage evolution (initial / peak / final), connectivity-
    filtered, styled like model1_crack_evolution: rainbow [0,1], shared colorbar,
    xticks/yticks at 0/0.5/1. Panel steps overridable via
    FRACTUREX_MODEL2_EVOL_STEPS="s0,s1,s2" (default 0,124,200 = pre-crack, peak,
    final). vmin fixed at 0 here so the (now spurious-free) field reads cleanly.
    """
    raw = os.environ.get("FRACTUREX_MODEL2_EVOL_STEPS", "0,124,200").strip()
    try:
        want = [int(s) for s in raw.split(",")][:3]
    except ValueError:
        want = [0, 124, 200]
    tags = ["initial pre-crack", "at peak load", "final"]
    picks = []
    for t in want:
        f, s, l = _vtu_for_step(t)
        if f is not None:
            picks.append((f, s, l))
    if len(picks) < 3:
        print("[skip] fig_crack_evolution: need 3 exported states, got", len(picks))
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
    tpc = None
    total_removed = 0
    for ax, (f, s, l), tag in zip(axes, picks, tags):
        pts, tris, d, nrem = _read_damage_vtu(f)
        total_removed += nrem
        tri = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
        tpc = ax.tripcolor(tri, d, shading="gouraud", cmap="rainbow", vmin=0, vmax=1)
        ax.set_aspect("equal")
        ax.set_title(rf"{tag}: $\bar u={l:.3e}$", fontsize=10)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
    cb = fig.colorbar(tpc, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(r"$d$")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"model2_crack_evolution.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'model2_crack_evolution.png'}, .pdf  (steps={[p[1] for p in picks]}, "
          f"connectivity_filter={'on' if CRACK_CONNECTED else 'off'}, "
          f"isolated_spurious_nodes_removed={total_removed})")


def main():
    h = load_history()
    mask = monotonic_frontier(h)
    dropped = [int(s) for s, m in zip(h["step"], mask) if not m]
    # raw (for optional transparency overlay) vs cleaned monotonic envelope
    raw_disp, raw_reac = h["disp_x"], h["reaction_x"]
    h = {k: v[mask] for k, v in h.items()}
    disp, reac = h["disp_x"], h["reaction_x"]
    n = len(disp)
    ipk = int(np.argmax(reac))
    print(f"run = {RUN}")
    print(f"raw steps = {len(raw_disp)}; dropped {len(dropped)} resume-reload "
          f"steps {dropped if dropped else '(none)'}")
    print(f"monotonic-envelope steps = {n}  (step {int(h['step'][0])}.."
          f"{int(h['step'][-1])}), disp_x {disp[0]:.4e}..{disp[-1]:.4e}")
    print(f"peak |reaction_x| = {reac[ipk]:.6g} at step {int(h['step'][ipk])} "
          f"disp_x = {disp[ipk]:.6e}")
    assert np.all(np.diff(disp) > 0), "envelope not strictly monotonic!"

    # --- load-displacement curve -------------------------------------------
    # Styling matches the companion model1_loaddisp figure in the paper:
    # title, "Hu--Zhang, direct solver" legend, dark-blue curve, black star
    # peak marker with an arrowed "peak |F_x|=... at u=..." annotation.
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    if os.environ.get("FRACTUREX_MODEL2_SHOW_RAW", "0").strip() == "1":
        ax.plot(raw_disp, raw_reac, "-", color="#bbbbbb", lw=0.8, zorder=1,
                label="raw (with resume reload loop)")
    ax.plot(disp, reac, "-", color="#1f3b73", lw=1.8, zorder=2,
            label="Hu--Zhang, direct solver")
    ax.plot(disp[ipk], reac[ipk], "*", color="k", ms=13, zorder=4)
    ax.annotate(
        rf"peak $|F_x|={reac[ipk]:.3f}$ at $\bar u={disp[ipk]:.3e}$",
        xy=(disp[ipk], reac[ipk]),
        xytext=(disp[ipk] * 0.28, reac[ipk] * 0.78),
        fontsize=9, ha="left",
        arrowprops=dict(arrowstyle="->", color="k", lw=0.8),
    )
    ax.set_xlabel(r"prescribed displacement $\bar u$")
    ax.set_ylabel(r"reaction force $|F_x|$")
    ax.set_title("Single-edge-notched shear: load--displacement")
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"model2_loaddisp.{ext}", dpi=200)
    plt.close(fig)

    # --- sampled table ------------------------------------------------------
    idx = sorted(set(np.linspace(0, n - 1, min(n, 25)).astype(int)) | {ipk})
    tbl = OUT / "model2_loaddisp_table.csv"
    with open(tbl, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "disp_x", "reaction_x_abs", "max_d", "max_H", "peak"])
        for i in idx:
            w.writerow([int(h["step"][i]), f"{disp[i]:.6e}", f"{reac[i]:.6e}",
                        f"{h['max_d'][i]:.4f}", f"{h['max_H'][i]:.6e}",
                        "PEAK" if i == ipk else ""])
    print(f"wrote {OUT/'model2_loaddisp.png'}, .pdf")
    print(f"wrote {tbl}")

    # flag the step-133 max_H spike if present (numerical blip, see notes)
    mh = h["max_H"]
    if n > 2:
        med = np.median(mh[mh > 0]) if np.any(mh > 0) else 0.0
        spikes = [(int(h["step"][i]), mh[i]) for i in range(n)
                  if med > 0 and mh[i] > 100 * med]
        if spikes:
            print("max_H spikes (>100x median, suspected numerical blips):",
                  ", ".join(f"step {s}: {v:.3e}" for s, v in spikes))

    # final crack pattern (rainbow d field), styled like model0/model1
    fig_crack_final()
    # three-panel evolution, styled like model1_crack_evolution
    fig_crack_evolution()


if __name__ == "__main__":
    main()
