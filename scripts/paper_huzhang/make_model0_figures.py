#!/usr/bin/env python3
# make_model0_figures.py
#
# Build the paper figures for the model-0 (circular-notch tension) benchmark
# from the *completed* runs under
#   results/phasefield/model0_circular_notch/paper_{direct,aux}_h{1,2,3}/
#
# Two messages:
#   (1) mesh convergence of the load--displacement response across the three
#       completed tiers h1,h2,h3 (direct solver, full history.csv);
#   (2) aux-vs-direct consistency: at h1 the aux-space and direct curves are
#       identical (max rel. diff 4.8e-8 over the whole fracture path), the
#       quantitative form of Claim C1.
#
# Outputs under Frac_huzhang/figures/:
#   model0_loaddisp.{png,pdf}      load-displacement, 3 tiers + aux overlay
#   model0_crack_final.{png,pdf}   phase field d at the final step, finest tier
#   model0_consistency_table.csv   peak reaction direct vs aux per tier

from __future__ import annotations

import csv
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
# pyvista is only needed by fig_crack_final (VTU read); import lazily so the
# load-displacement figure can be generated on machines without pyvista.

ROOT = Path("/home/gongshihua/tian/fracturex")
CASE = ROOT / "results/phasefield/model0_circular_notch"
OUT = Path("/home/gongshihua/tian/Frac_huzhang/figures")
OUT.mkdir(parents=True, exist_ok=True)

TIERS = ["h1", "h2", "h3"]
COLORS = {"h1": "#1f77b4", "h2": "#2ca02c", "h3": "#d62728"}


def curve(tag):
    p = CASE / f"paper_{tag}/epsg_1e-06/history.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    d = np.array([float(r["disp_y"]) for r in rows])
    f = np.array([abs(float(r["reaction_y"])) for r in rows])
    return d, f


def fig_loaddisp():
    """Circular-notch tension load--displacement.

    Two messages in one clean panel:
      (1) mesh convergence: direct h1/h2/h3 full curves (linear rise -> converged
          peak -> brittle softening); a peak-zoom inset makes the convergence of
          the peak load explicit (h2~h3, h1 slightly lower).
      (2) solver consistency: aux-space results overlaid as open markers tracking
          the direct lines (h1 spans the full softening branch; h2/h3 cover the
          rise--peak range their runs reached). A single proxy legend entry keeps
          the legend to four lines instead of six.
    """
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    peaks = {}
    direct = {t: curve(f"direct_{t}") for t in TIERS}
    aux = {t: curve(f"aux_{t}") for t in TIERS}

    # --- main panel: direct full curves (mesh convergence) ------------------
    # All three direct tiers run the FULL fracture path (linear rise -> peak ->
    # brittle softening), so the curves are end-to-end comparable.
    for t in TIERS:
        if direct[t] is None:
            continue
        d, f = direct[t]
        ax.plot(d, f, "-", color=COLORS[t], lw=1.9, zorder=2,
                label=f"direct, {t} ($N_\\sigma$={_nsig(t)})")
        peaks[t] = (d[int(np.argmax(f))], f.max())

    # aux overlay: ONLY h1 -- the single tier whose aux run spans the WHOLE
    # fracture path, so the aux-vs-direct comparison is fair end-to-end (Claim
    # C1: curves identical to rel diff 4.8e-8 over the whole path). The h2/h3
    # aux runs stopped near peak (localization niter blowup) and are
    # deliberately NOT overlaid -- mixing a full curve with truncated ones
    # would misrepresent coverage. aux mesh-independence is shown by niter
    # (paper fig 1 / D12 sec5.2), not by this load-displacement panel.
    if aux["h1"] is not None:
        da, fa = aux["h1"]
        step = max(1, len(da) // 12)
        ax.plot(da[::step], fa[::step], "o", mfc="none", mec=COLORS["h1"],
                ms=5.5, mew=1.2, ls="none", zorder=4)

    ax.set_xlabel(r"prescribed displacement $\bar u$")
    ax.set_ylabel(r"reaction force $|F_y|$")
    ax.set_title("Circular-notch tension: load--displacement")
    ax.grid(True, ls=":", alpha=0.45)
    ax.set_xlim(-0.003, 0.128)
    ax.set_ylim(-1.0, 30.5)
    # legend: 3 direct lines + 1 proxy for the aux overlay (h1 open markers)
    aux_proxy = Line2D([], [], ls="none", marker="o", mfc="none",
                       mec=COLORS["h1"], ms=6, mew=1.2,
                       label="aux-space, h1 (overlay)")
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h + [aux_proxy], fontsize=8.5, loc="lower left",
              framealpha=0.92)

    # --- inset: zoom on the peak to show mesh convergence -------------------
    # upper-left empty triangle (small u, high F) keeps the inset clear of the
    # brittle softening drops clustered at u in [0.09, 0.12].
    axin = inset_axes(ax, width="40%", height="33%", loc="upper left",
                      borderpad=1.4)
    for t in TIERS:
        if direct[t] is not None:
            d, f = direct[t]
            axin.plot(d, f, "-", color=COLORS[t], lw=1.8)
    if aux["h1"] is not None:
        da, fa = aux["h1"]
        axin.plot(da, fa, "o", mfc="none", mec=COLORS["h1"], ms=5, mew=1.1,
                  ls="none")
    axin.set_xlim(0.078, 0.094)
    axin.set_ylim(26.6, 28.6)
    axin.set_title("peak (zoom)", fontsize=8)
    axin.tick_params(labelsize=7)
    axin.grid(True, ls=":", alpha=0.5)
    mark_inset(ax, axin, loc1=2, loc2=4, fc="none", ec="0.6", lw=0.8, ls="--")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"model0_loaddisp.{ext}", dpi=200)
    plt.close(fig)
    return peaks


_NSIG = {"h1": "10.9K", "h2": "48K", "h3": "184K"}


def _nsig(t):
    return _NSIG[t]


def fig_crack_final(tag="direct_h3"):
    import pyvista as pv  # lazy: only this figure needs the VTU reader
    fs = sorted(glob.glob(str(CASE / f"paper_{tag}/epsg_1e-06/vtk/*.vtu")))
    m = pv.read(fs[-1])
    load = float(fs[-1].split("load_")[1].split(".vtu")[0])
    pts = m.points
    cells = m.cells.reshape(-1, 4)[:, 1:]
    d = np.asarray(m.point_data["damage"])
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], cells)
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    tpc = ax.tripcolor(tri, d, shading="gouraud", cmap="rainbow", vmin=0, vmax=1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(rf"Circular-notch tension: phase field $d$ at $\bar u={load:.3f}$")
    cb = fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$d$")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"model0_crack_final.{ext}", dpi=200)
    plt.close(fig)


def consistency_table(peaks):
    with open(OUT / "model0_consistency_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tier", "Nsigma", "peak_direct", "peak_aux", "rel_diff_curve"])
        for t in TIERS:
            cd, ca = curve(f"direct_{t}"), curve(f"aux_{t}")
            pd_ = cd[1].max() if cd else float("nan")
            pa = ca[1].max() if ca else float("nan")
            # rel diff over overlapping loaded steps
            rel = float("nan")
            if cd and ca:
                n = min(len(cd[1]), len(ca[1]))
                den = cd[1][1:n]
                rel = float(np.max(np.abs(cd[1][1:n] - ca[1][1:n]) / (den + 1e-30)))
            w.writerow([t, _NSIG[t], f"{pd_:.4f}", f"{pa:.4f}", f"{rel:.2e}"])


def main():
    peaks = fig_loaddisp()
    try:
        fig_crack_final()
    except ModuleNotFoundError as e:
        print(f"[skip] fig_crack_final needs pyvista ({e}); load-disp + table still written")
    consistency_table(peaks)
    print("peaks:", {k: (round(v[0], 4), round(v[1], 3)) for k, v in peaks.items()})
    print("wrote model0 figures to", OUT)


if __name__ == "__main__":
    main()
