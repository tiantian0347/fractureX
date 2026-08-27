#!/usr/bin/env python3
"""Plot standard-FEM (MainSolve) model5 TPB vs Ambati Fig.22.

Reads:
  results/phasefield/model5_standard_fem/<run>/model5_std_force_disp.txt
  docs/benchmarks/figures/loaddisp/ambati_fig22_model5_tpb_digitized.csv

Outputs:
  docs/benchmarks/figures/loaddisp/model5_std_fem_loaddisp.{png,pdf}
  docs/benchmarks/figures/loaddisp/model5_std_fem_vs_ambati_loaddisp.{png,pdf}

Override run dir: FRACTUREX_MODEL5_STD_FEM=/path/to/run
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FORMAL_TITLE = "V-notch beam"

_run_env = os.environ.get("FRACTUREX_MODEL5_STD_FEM", "").strip()
_default = ROOT / "results/phasefield/model5_standard_fem/std_bg_h010_full"
RUN = Path(_run_env) if _run_env else _default
if not RUN.is_absolute():
    RUN = (ROOT / RUN).resolve()

OUT = ROOT / "docs/benchmarks/figures/loaddisp"
AMBati_CSV = OUT / "ambati_fig22_model5_tpb_digitized.csv"


def load_std_fem():
    path = RUN / "model5_std_force_disp.txt"
    data = np.loadtxt(path, skiprows=1)
    u = np.abs(data[:, 0])  # downward magnitude (mm)
    r = np.abs(data[:, 1])  # |reaction| (kN)
    return u, r


def load_ambati():
    data = np.loadtxt(AMBati_CSV, delimiter=",", skiprows=1, usecols=(0, 1))
    return data[:, 0], data[:, 1]


def fig_std_only(u, r):
    i = int(np.nanargmax(r))
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.plot(u, r, "-", lw=1.4, color="#2e7d32", label="standard finite element solution")
    ax.plot(
        u[i],
        r[i],
        "s",
        color="#c0392b",
        ms=6,
        label=rf"peak $|R|={r[i]:.3f}$ at $u={u[i]:.3f}$ mm",
    )
    ax.set_xlabel(r"prescribed displacement $|u_y|$ (mm)")
    ax.set_ylabel(r"reaction force $|R_y|$ (kN)")
    ax.set_title(rf"{FORMAL_TITLE}: standard FEM")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"model5_std_fem_loaddisp.{ext}", dpi=220)
    plt.close(fig)


def fig_vs_ambati(u, r, ua, ra):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(ua, ra, "o-", ms=4, lw=1.2, color="#555555", label="Ambati (2015), Fig. 22")
    ax.plot(u, r, "-", lw=1.5, color="#2e7d32", label="standard finite element solution")
    i_fx = int(np.nanargmax(r))
    i_am = int(np.nanargmax(ra))
    ax.plot(
        u[i_fx],
        r[i_fx],
        "s",
        color="#1b5e20",
        ms=6,
        label=rf"FX peak {r[i_fx]:.3f} kN @ {u[i_fx]:.3f} mm",
    )
    ax.plot(
        ua[i_am],
        ra[i_am],
        "D",
        color="#c0392b",
        ms=5,
        label=rf"Ambati peak {ra[i_am]:.3f} kN @ {ua[i_am]:.3f} mm",
    )
    ax.set_xlabel(r"prescribed displacement $|u_y|$ (mm)")
    ax.set_ylabel(r"reaction force $|R_y|$ (kN)")
    ax.set_title(rf"{FORMAL_TITLE}: standard FEM vs Ambati")
    ax.set_xlim(0.0, max(u.max(), ua.max()) * 1.05)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"model5_std_fem_vs_ambati_loaddisp.{ext}", dpi=220)
    plt.close(fig)


def main():
    u, r = load_std_fem()
    ua, ra = load_ambati()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"RUN = {RUN}")
    print(f"FX  peak |R|={r.max():.4f} kN @ u={u[np.argmax(r)]:.4f} mm")
    print(f"Amb peak |R|={ra.max():.4f} kN @ u={ua[np.argmax(ra)]:.4f} mm")
    fig_std_only(u, r)
    fig_vs_ambati(u, r, ua, ra)
    print(f"wrote {OUT / 'model5_std_fem_vs_ambati_loaddisp.png'}")


if __name__ == "__main__":
    main()
