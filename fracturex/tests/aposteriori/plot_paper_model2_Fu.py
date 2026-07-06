"""Publication figures for the SENS (model2) benchmark.

Outputs (default: Tian/thesis/fracture_huzhang/adaptive/figures/):
  paper_model2_Fu_main.png       -- eta_T marker load--displacement curve
  paper_model2_marker_compare.png -- eta_T vs D_tau,T (M-DF) marker comparison
  paper_model2_NC_growth.png     -- cell-count growth over load steps
  paper_model2_Dmax_evolution.png -- D_max observable (log scale)

Reads (under $FRACTUREX_RESULTS, default ~/repository/results):
  adaptive_m3_pc_model2_eta_T/history.csv        (eta_T marker, 40 steps)
  adaptive_m3_pc_model2_effstress/history.csv    (D_tau,T marker, diverged)

Run:
  python fracturex/tests/aposteriori/plot_paper_model2_Fu.py
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
ETA_T = os.path.join(ROOT, "adaptive_m3_pc_model2_eta_T/history.csv")
D_TAU = os.path.join(ROOT, "adaptive_m3_pc_model2_effstress/history.csv")

COL_ETA = "#1f3a68"
COL_DTAU = "#c0392b"
COL_REF = "0.25"


def _load(path):
    rows = list(csv.DictReader(open(path)))
    d = np.array([float(r["load"]) for r in rows])
    R = np.array([abs(float(r["reaction"])) for r in rows])
    D = np.array([float(r["D_max"]) for r in rows])
    NC = np.array([int(r["nc"]) for r in rows])
    return dict(load=d, R=R, D=D, NC=NC)


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


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    _rc()

    eta = _load(ETA_T)
    dtau = _load(D_TAU)

    p_eta = _peak(eta["load"], eta["R"])
    p_dtau = _peak(dtau["load"], dtau["R"])

    # last step reached by D_tau marker before divergence
    fail_step_dtau = len(dtau["R"]) - 1
    fail_load_dtau = dtau["load"][-1]

    # ---------- Fig A: eta_T main F-u curve ----------
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(eta["load"], eta["R"], "o-", color=COL_ETA, ms=3.5, lw=1.4,
            label=(rf"adaptive $\eta_T$ marker, "
                   rf"peak $|R_x|_{{\max}}{{=}}{p_eta[1]:.3f}$ at "
                   rf"$u_x{{=}}{p_eta[0]:.2e}$"))
    ax.scatter([p_eta[0]], [p_eta[1]], s=30, color=COL_ETA, zorder=5)
    ax.set_xlabel(r"prescribed shear displacement $u_x$")
    ax.set_ylabel(r"reaction $|R_x|$")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    f_main = os.path.join(OUTDIR, "paper_model2_Fu_main.png")
    fig.savefig(f_main, dpi=300)
    plt.close(fig)

    # ---------- Fig B: marker comparison eta_T vs D_tau,T ----------
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(dtau["load"], dtau["R"], "s--", color=COL_DTAU, ms=3.2, lw=1.2,
            label=(rf"$\mathcal{{D}}_{{\tau,T}}$ marker "
                   rf"(diverges at step {fail_step_dtau}, "
                   rf"$u_x{{=}}{fail_load_dtau:.2e}$)"))
    ax.plot(eta["load"], eta["R"], "o-", color=COL_ETA, ms=3.5, lw=1.4,
            label=(rf"$\eta_T$ marker "
                   rf"(completes 40 steps, peak $={p_eta[1]:.3f}$)"))
    ax.axvline(x=fail_load_dtau, color=COL_DTAU, ls=":", lw=0.8, alpha=0.6)
    ax.scatter([p_dtau[0]], [p_dtau[1]], s=30, color=COL_DTAU, zorder=5,
               marker="s")
    ax.scatter([p_eta[0]], [p_eta[1]], s=30, color=COL_ETA, zorder=5)
    ax.set_xlabel(r"prescribed shear displacement $u_x$")
    ax.set_ylabel(r"reaction $|R_x|$")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    f_cmp = os.path.join(OUTDIR, "paper_model2_marker_compare.png")
    fig.savefig(f_cmp, dpi=300)
    plt.close(fig)

    # ---------- Fig C: NC growth ----------
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(dtau["load"], dtau["NC"], "s--", color=COL_DTAU, ms=3.2, lw=1.2,
            label=(rf"$\mathcal{{D}}_{{\tau,T}}$ marker "
                   rf"({dtau['NC'][0]}$\to${dtau['NC'][-1]} in "
                   rf"{fail_step_dtau} steps)"))
    ax.plot(eta["load"], eta["NC"], "o-", color=COL_ETA, ms=3.5, lw=1.4,
            label=(rf"$\eta_T$ marker "
                   rf"({eta['NC'][0]}$\to${eta['NC'][-1]} in 40 steps)"))
    ax.set_xlabel(r"prescribed shear displacement $u_x$")
    ax.set_ylabel(r"number of triangles $\mathrm{NC}$")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlim(left=0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    f_nc = os.path.join(OUTDIR, "paper_model2_NC_growth.png")
    fig.savefig(f_nc, dpi=300)
    plt.close(fig)

    # ---------- Fig D: D_max evolution (log scale) ----------
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.semilogy(dtau["load"], np.maximum(dtau["D"], 1e-3), "s--",
                color=COL_DTAU, ms=3.2, lw=1.2,
                label=(rf"$\mathcal{{D}}_{{\tau,T}}$ marker "
                       rf"(marker and observable coincide)"))
    ax.semilogy(eta["load"], np.maximum(eta["D"], 1e-3), "o-",
                color=COL_ETA, ms=3.5, lw=1.4,
                label=(rf"$\eta_T$ marker "
                       rf"($\mathcal{{D}}_{{\tau,T}}$ is observable only)"))
    ax.axhline(y=1.0 / 3.0, color="0.4", ls=":", lw=0.8,
               label=r"$\mathcal{D}_c = 1/3$ (AT2 softening threshold)")
    ax.set_xlabel(r"prescribed shear displacement $u_x$")
    ax.set_ylabel(r"$\mathcal{D}_{\max} := \max_T \mathcal{D}_{\tau,T}$")
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.set_xlim(left=0)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    f_dmax = os.path.join(OUTDIR, "paper_model2_Dmax_evolution.png")
    fig.savefig(f_dmax, dpi=300)
    plt.close(fig)

    print(f"[plot] eta_T   peak={p_eta[1]:.4f} at u_x={p_eta[0]:.3e}, "
          f"NC {eta['NC'][0]}->{eta['NC'][-1]}, 40 steps completed")
    print(f"[plot] D_tau,T peak={p_dtau[1]:.4f} at u_x={p_dtau[0]:.3e}, "
          f"diverged at step {fail_step_dtau}")
    print(f"[plot] wrote {f_main}")
    print(f"[plot] wrote {f_cmp}")
    print(f"[plot] wrote {f_nc}")
    print(f"[plot] wrote {f_dmax}")


if __name__ == "__main__":
    main()
