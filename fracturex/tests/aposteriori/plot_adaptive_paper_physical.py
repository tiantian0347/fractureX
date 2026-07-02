"""Physical-validation figures for the equilibrated a-posteriori adaptive paper
(`Frac_huzhang/adaptive/equilibrated_aposteriori.tex`).

Generates, into Frac_huzhang/adaptive/figures/ :
  1. adaptive_load_displacement.png — adaptive (sigma-driven M-DF + predictor-
     corrector) load-displacement vs uniform nx=120 reference and a coarse
     uniform nx=24 mesh (under-resolution overestimate).
  2. adaptive_marker_comparison.png — marking-strategy comparison: eta-Dorfler
     (once-per-step, run `adaptive_m3_full_model1`) vs sigma-driven M-DF +
     predictor-corrector (`adaptive_m3_pc_model1_v3`), both vs the reference.
     NOTE (honest caveat, stated in the paper): the two runs differ not only in
     the marker but also in the refinement scheme (once-per-step vs
     predictor-corrector), so this is a strategy-level, not a pure-marker,
     ablation.
  3. adaptive_mesh_damage.png — adapted mesh + damage field at the peak step and
     at a post-peak (propagation) step, showing refinement tracking the band.

Pure file I/O + plotting (np allowed). Reads only existing results; no solves.
Run: PYTHONPATH=$PWD python fracturex/tests/aposteriori/plot_adaptive_paper_physical.py
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import meshio

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TIAN = os.path.abspath(os.path.join(REPO, ".."))
RES = os.path.join(REPO, "results")
OUT = os.path.join(TIAN, "Frac_huzhang", "adaptive", "figures")

V3 = os.path.join(RES, "adaptive_m3_pc_model1_v3", "history_anderson_canonical.csv")
V1 = os.path.join(RES, "adaptive_m3_full_model1", "history.csv")
NX24 = os.path.join(RES, "uniform_m3_model1_nx24", "history.csv")
REF = os.path.join(RES, "phasefield", "square_tension_precrack",
                   "paper_direct_full_nx120", "epsg_1e-06", "history.csv")
VTU = os.path.join(RES, "adaptive_m3_pc_model1_v3", "vtu")


def _read(path, dcol, rcol):
    """Return (disp, |reaction|) arrays sorted by disp, skipping non-numeric rows."""
    xs, ys = [], []
    for r in csv.DictReader(open(path)):
        try:
            x = float(r[dcol]); y = abs(float(r[rcol]))
        except (KeyError, ValueError):
            continue
        xs.append(x); ys.append(y)
    o = np.argsort(xs)
    return np.asarray(xs)[o], np.asarray(ys)[o]


def _peak(x, y):
    i = int(np.argmax(y))
    return x[i], float(y[i])


def fig_load_displacement(ref, v3, nx24):
    rx, ry = ref; vx, vy = v3; ux, uy = nx24
    rp, vp, up = _peak(*ref), _peak(*v3), _peak(*nx24)
    dv = (vp[1] - rp[1]) / rp[1] * 100
    du = (up[1] - rp[1]) / rp[1] * 100
    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    ax.plot(rx, ry, "-", color="0.30", lw=2.2,
            label=f"uniform nx=120 reference (peak {rp[1]:.3f})")
    ax.plot(ux, uy, "^--", color="C3", ms=3.5, lw=1.3,
            label=f"uniform nx=24 (peak {up[1]:.3f}, {du:+.0f}%)")
    ax.plot(vx, vy, "o-", color="C0", ms=4, lw=1.8,
            label=f"adaptive M-DF + PC (peak {vp[1]:.3f}, {dv:+.1f}%)")
    ax.scatter([up[0]], [up[1]], color="C3", zorder=5, s=45)
    ax.scatter([vp[0]], [vp[1]], color="C0", zorder=5, s=50)
    ax.scatter([rp[0]], [rp[1]], color="0.30", zorder=5, s=50)
    ax.set_xlabel(r"prescribed displacement $u_D$")
    ax.set_ylabel(r"reaction force $|R|$")
    ax.set_xlim(0, 1.02 * max(vx.max(), rx[ry > 0].max() if (ry > 0).any() else rx.max(), up[0]))
    ax.set_title("Single-edge tension: adaptivity recovers the reference peak;\n"
                 "a coarse uniform mesh overestimates it")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, "adaptive_load_displacement.png")
    fig.savefig(p, dpi=160); plt.close(fig)
    print(f"[fig1] ref={rp[1]:.4f} adaptive={vp[1]:.4f}({dv:+.1f}%) nx24={up[1]:.4f}({du:+.0f}%) -> {p}")


def fig_marker_comparison(ref, v3, v1):
    rx, ry = ref; vx, vy = v3; ex, ey = v1
    rp, vp, ep = _peak(*ref), _peak(*v3), _peak(*v1)
    dv = (vp[1] - rp[1]) / rp[1] * 100
    de = (ep[1] - rp[1]) / rp[1] * 100
    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    ax.plot(rx, ry, "-", color="0.30", lw=2.2,
            label=f"uniform nx=120 reference (peak {rp[1]:.3f})")
    ax.plot(ex, ey, "s--", color="C7", ms=3.5, lw=1.4,
            label=fr"$\eta$-Dorfler, once/step (peak {ep[1]:.3f}, {de:+.0f}%)")
    ax.plot(vx, vy, "o-", color="C0", ms=4, lw=1.8,
            label=fr"$\mathcal{{D}}_\tau$ M-DF + PC (peak {vp[1]:.3f}, {dv:+.1f}%)")
    ax.scatter([ep[0]], [ep[1]], color="C7", zorder=5, s=45)
    ax.scatter([vp[0]], [vp[1]], color="C0", zorder=5, s=50)
    ax.scatter([rp[0]], [rp[1]], color="0.30", zorder=5, s=50)
    ax.set_xlabel(r"prescribed displacement $u_D$")
    ax.set_ylabel(r"reaction force $|R|$")
    ax.set_title("Marking strategy: stress pre-damage marker (M-DF) reaches the\n"
                 r"reference peak; $\eta$-driven once-per-step marking overestimates")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, "adaptive_marker_comparison.png")
    fig.savefig(p, dpi=160); plt.close(fig)
    print(f"[fig2] ref={rp[1]:.4f} M-DF={vp[1]:.4f}({dv:+.1f}%) eta-Dorfler={ep[1]:.4f}({de:+.0f}%) -> {p}")


def _load_vtu(step):
    m = meshio.read(os.path.join(VTU, f"step_{step:03d}.vtu"))
    pts = m.points[:, :2]
    tris = np.vstack([c.data for c in m.cells if c.type == "triangle"])
    d = np.asarray(m.point_data["damage"]).ravel()
    return pts, tris, d


def fig_mesh_damage(peak_step, fail_step):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.0))
    for ax, step, tag in ((axes[0], peak_step, "peak"), (axes[1], fail_step, "propagation")):
        pts, tris, d = _load_vtu(step)
        tri = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
        tpc = ax.tripcolor(tri, d, shading="gouraud", cmap="inferno", vmin=0, vmax=1)
        ax.triplot(tri, color="white", lw=0.18, alpha=0.55)
        ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{tag} (step {step}): {len(tris)} elements")
        ax.set_xlabel("x"); ax.set_ylabel("y")
    cb = fig.colorbar(tpc, ax=axes, fraction=0.046, pad=0.03)
    cb.set_label("damage $d$")
    fig.suptitle("Adapted mesh and damage: refinement concentrates on and tracks the crack band", y=0.98)
    p = os.path.join(OUT, "adaptive_mesh_damage.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"[fig3] peak step {peak_step}, fail step {fail_step} -> {p}")


def main():
    os.makedirs(OUT, exist_ok=True)
    ref = _read(REF, "disp_y", "R")
    v3 = _read(V3, "load", "reaction")
    v1 = _read(V1, "load", "reaction")
    nx24 = _read(NX24, "load", "reaction")
    fig_load_displacement(ref, v3, nx24)
    fig_marker_comparison(ref, v3, v1)
    fig_mesh_damage(peak_step=20, fail_step=24)


if __name__ == "__main__":
    main()
