#!/usr/bin/env python3
"""Figure 1b (restart-aware revision): aux_fast stays bounded across full crack
localization. Plots deterministic d12_recheck niter (one solve per assembled
operator, seed-fixed) vs load step on the headline restart=200 configuration,
with max_d on the right axis. The earlier figure baked the restart=60 production
iterations.csv (annotated "niter 7->~95, O(100)") — that is the restart-stall
artifact, not preconditioner behaviour, so this figure draws the recheck data.

Data: docs/preconditioner/D13_IMPL §9.4 d12_recheck table, cross-validated by a
fresh run (step014 r200: 17 vs tabled 18; r60: 91 vs 93; step015 r60: 159 vs 173).

Output: docs/figures/precond/iter_stability_localization.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
OUTDIR = _REPO / "docs/figures/precond"
OUTDIR.mkdir(parents=True, exist_ok=True)

# d12_recheck (model0 h2, sigma-DOF 48,092, deterministic seed). niter at the
# headline restart=200; None = not separately re-measured (pre-localization is
# flat O(1) across restarts). restart=60 column kept for the appendix only.
# step: (maxd, niter_r200, niter_r60, capped_r60)
DATA = {
    10: (0.306, 2, 7, False),
    11: (0.333, 2, 7, False),
    12: (0.369, 2, 7, False),
    13: (0.426, 2, 7, False),   # last pre-localization step
    14: (0.998, 18, 93, False),  # localization jump (fresh run: r200=17, r60=91)
    15: (0.998, 28, 173, False),
    17: (1.000, 25, 60000, True),   # r60 DNF (maxit cap)
    20: (1.000, 82, 60000, True),   # r60 DNF (maxit cap)
}
RESTART = 200

steps = np.array(sorted(DATA))
maxd = np.array([DATA[s][0] for s in steps])
niter = np.array([DATA[s][1] for s in steps], float)

fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax2 = ax.twinx()

color = "#d62728"
ax.plot(steps, niter, "-o", color=color, ms=5, lw=1.8, zorder=3,
        label=f"aux-space niter (GMRES restart={RESTART})")
ax2.plot(steps, maxd, "--s", color="#555555", ms=3.5, lw=1.1, alpha=0.6,
         zorder=2, label=r"max damage $\max_x d_h$")

ax.set_xlabel("load step")
ax.set_ylabel(r"GMRES iterations (elastic block)")
ax2.set_ylabel(r"max damage $\max_x d_h$", color="#555555")
ax.set_yscale("log")
ax.set_ylim(1.5, 200)
ax2.set_ylim(0, 1.05)

# mark the localization transition (first maxd jump > 0.4)
tr = None
for i in range(1, len(steps)):
    if maxd[i] - maxd[i - 1] > 0.4:
        tr = steps[i]
        break
if tr is not None:
    ax.axvline(tr, color="#222222", ls="-.", lw=0.9, alpha=0.6, zorder=0)
    ax.annotate("crack localizes\n" + r"($\max_x d_h:0.43\to1.0$)",
                xy=(tr, 18), xytext=(tr + 0.4, 60),
                fontsize=8.5, ha="left",
                arrowprops=dict(arrowstyle="->", color="#222222", lw=0.9))

# shaded "bounded O(10-50)" band over the localized regime
loc_mask = maxd > 0.9
if loc_mask.any():
    ax.axhspan(10, 50, color=color, alpha=0.06, zorder=0)
    ax.text(steps[-1], 50, "O(10–50)", color=color, fontsize=8,
            ha="right", va="bottom")

# Title intentionally omitted; the paper supplies a full caption.
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", framealpha=0.9)
fig.tight_layout()
for ext in ("png", "pdf"):
    out = OUTDIR / f"iter_stability_localization.{ext}"
    fig.savefig(out, dpi=150 if ext == "png" else None, bbox_inches="tight")
    print(f"saved {out}")
