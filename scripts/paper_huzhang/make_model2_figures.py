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
#   model2_loaddisp_table.csv     sampled (step, disp_x, reaction_x, max_d, max_H)

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    # no title, $|F_x|$ vs $\bar u$ labels, star peak marker.
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    if os.environ.get("FRACTUREX_MODEL2_SHOW_RAW", "0").strip() == "1":
        ax.plot(raw_disp, raw_reac, "-", color="#bbbbbb", lw=0.8, zorder=1,
                label="raw (with resume reload loop)")
    ax.plot(disp, reac, "-", color="#1f4e79", lw=1.4)
    ax.plot(disp[ipk], reac[ipk], "*", color="#c00000", ms=13, zorder=5)
    ax.annotate(rf"$|F_x|_{{\max}}={reac[ipk]:.3f}$",
                xy=(disp[ipk], reac[ipk]),
                xytext=(disp[ipk] * 0.30, reac[ipk] * 0.96),
                fontsize=10, color="#c00000")
    ax.set_xlabel(r"prescribed displacement $\bar u$")
    ax.set_ylabel(r"reaction force $|F_x|$")
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
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


if __name__ == "__main__":
    main()
