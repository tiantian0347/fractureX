"""Publication figures for the SENT (model1) load--displacement study.

Outputs (default: Tian/thesis/fracture_huzhang/adaptive/figures/):
  paper_model1_Fu_main.png     -- uniform reference vs adaptive (Δu_y = 2.5e-4)
                                  used as the main load--displacement figure.
  paper_model1_Fu_pathband.png -- + adaptive (Δu_y = 1e-4) + ±4% band
                                  used as the path-sensitivity figure.

Both figures use identical styling (serif, LaTeX math labels, no in-figure
narrative titles) so the caption in the paper carries the interpretation.

Reads (under $FRACTUREX_RESULTS, default ~/repository/results):
  adaptive_m3_pc_model1_v3/history_anderson_canonical.csv (Δu_y=2.5e-4)
  adaptive_m3_pc_model1_du1e4/history.csv                 (Δu_y=1e-4)
  phasefield/square_tension_precrack/paper_direct_full_nx120/
    epsg_1e-06/history.csv                                (reference)

Run:
  python fracturex/tests/aposteriori/plot_paper_model1_Fu.py
Override I/O with env vars FRACTUREX_RESULTS / FRACTUREX_FIGDIR.
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.environ.get("FRACTUREX_RESULTS",
                      os.path.expanduser("~/repository/results"))
OUTDIR = os.environ.get(
    "FRACTUREX_FIGDIR",
    os.path.expanduser(
        "~/repository/Tian/thesis/fracture_huzhang/adaptive/figures"))
DU25 = os.path.join(
    ROOT, "adaptive_m3_pc_model1_v3/history_anderson_canonical.csv")
DU10 = os.path.join(ROOT, "adaptive_m3_pc_model1_du1e4/history.csv")
REF = os.path.join(
    ROOT,
    "phasefield/square_tension_precrack/"
    "paper_direct_full_nx120/epsg_1e-06/history.csv")

COL_REF = "0.25"
COL_A25 = "#1f3a68"
COL_A10 = "#c0392b"
BAND_COL = "0.86"


def _read(path, dcol, rcol):
    rows = list(csv.DictReader(open(path)))
    d, r = [], []
    for row in rows:
        try:
            dv = float(row[dcol]); rv = float(row[rcol])
        except (KeyError, ValueError):
            continue
        d.append(dv); r.append(abs(rv))
    return np.asarray(d), np.asarray(r)


def _peak(d, r):
    i = int(np.argmax(r))
    return float(d[i]), float(r[i])


def _rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7",
        "legend.fancybox": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
    })


def _axes_common(ax):
    ax.set_xlabel(r"prescribed displacement $u_y$")
    ax.set_ylabel(r"reaction $|R_y|$")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    _rc()

    d_ref, r_ref = _read(REF, "disp_y", "R")
    d_25, r_25 = _read(DU25, "load", "reaction")
    d_10, r_10 = _read(DU10, "load", "reaction")

    p_ref = _peak(d_ref, r_ref)
    p_25 = _peak(d_25, r_25)
    p_10 = _peak(d_10, r_10)
    dev_25 = (p_25[1] - p_ref[1]) / p_ref[1] * 100
    dev_10 = (p_10[1] - p_ref[1]) / p_ref[1] * 100

    # ---------- Fig A: main load--displacement (ref + Δu=2.5e-4) ----------
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(d_ref, r_ref, "-", color=COL_REF, lw=1.8,
            label=(rf"uniform reference $n_x{{=}}120$, "
                   rf"peak $|R_y|_{{\max}}{{=}}{p_ref[1]:.3f}$"))
    ax.plot(d_25, r_25, "o-", color=COL_A25, ms=3.5, lw=1.4,
            label=(rf"adaptive $\Delta u_y {{=}} 2.5\times 10^{{-4}}$, "
                   rf"peak $={p_25[1]:.3f}$ (${dev_25:+.1f}\%$)"))
    ax.scatter([p_ref[0]], [p_ref[1]], s=30, color=COL_REF, zorder=5)
    ax.scatter([p_25[0]], [p_25[1]], s=30, color=COL_A25, zorder=5)
    _axes_common(ax)
    ax.legend(loc="lower right")
    fig.tight_layout()
    f_main = os.path.join(OUTDIR, "paper_model1_Fu_main.png")
    fig.savefig(f_main, dpi=300)
    plt.close(fig)

    # ---------- Fig B: path band (ref + Δu=2.5e-4 + Δu=1e-4 + ±4%) ----------
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.axhspan(p_ref[1] * 0.96, p_ref[1] * 1.04, color=BAND_COL,
               label=r"$\pm 4\%$ band around reference peak")
    ax.plot(d_ref, r_ref, "-", color=COL_REF, lw=1.8,
            label=(rf"uniform reference $n_x{{=}}120$, "
                   rf"peak $={p_ref[1]:.3f}$"))
    ax.plot(d_25, r_25, "s--", color=COL_A25, ms=3.2, lw=1.2,
            label=(rf"adaptive $\Delta u_y {{=}} 2.5\times 10^{{-4}}$, "
                   rf"peak $={p_25[1]:.3f}$ (${dev_25:+.1f}\%$)"))
    ax.plot(d_10, r_10, "o-", color=COL_A10, ms=3.2, lw=1.4,
            label=(rf"adaptive $\Delta u_y {{=}} 1.0\times 10^{{-4}}$, "
                   rf"peak $={p_10[1]:.3f}$ (${dev_10:+.1f}\%$)"))
    ax.scatter([p_ref[0]], [p_ref[1]], s=30, color=COL_REF, zorder=5)
    ax.scatter([p_25[0]], [p_25[1]], s=30, color=COL_A25, zorder=5)
    ax.scatter([p_10[0]], [p_10[1]], s=30, color=COL_A10, zorder=5)
    _axes_common(ax)
    ax.legend(loc="lower right")
    fig.tight_layout()
    f_band = os.path.join(OUTDIR, "paper_model1_Fu_pathband.png")
    fig.savefig(f_band, dpi=300)
    plt.close(fig)

    print(f"[plot] ref      peak={p_ref[1]:.4f} at u_y={p_ref[0]:.3e}")
    print(f"[plot] Δu=2.5e-4 peak={p_25[1]:.4f} ({dev_25:+.2f}%)")
    print(f"[plot] Δu=1.0e-4 peak={p_10[1]:.4f} ({dev_10:+.2f}%)")
    print(f"[plot] wrote {f_main}")
    print(f"[plot] wrote {f_band}")


if __name__ == "__main__":
    main()
