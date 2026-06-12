#!/usr/bin/env python3
"""§5.2b 头条图：真实裂纹局部化下 aux_fast 的有界收敛。

读真实相场 run 的 history.csv + iterations.csv（model0 aux_h2[/h3]），绘双轴图：
  左轴  GMRES niter（per-step 中位 + 峰值带）vs load step；
  右轴  max_d vs load step（局部化指示量）。
标注 step13->14 的同步跃迁（maxd 0.43->0.998 与 niter 7->~95 同时发生），
触及 maxit 上限的步用空心标记区分。

用法: paper_make_iter_localization.py [run_dir ...]
  默认 run_dir = results/phasefield/model0_circular_notch/paper_aux_h2/epsg_1e-06
输出: docs/figures/precond/iter_stability_localization.{png,pdf}
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
OUTDIR = _REPO / "docs/figures/precond"
OUTDIR.mkdir(parents=True, exist_ok=True)

MAXIT = 200  # FRACTUREX_GMRES_MAXIT default; iter at this value w/ converged=False = capped

DEFAULT_RUNS = [
    _REPO / "results/phasefield/model0_circular_notch/paper_aux_h2/epsg_1e-06",
]
STYLE = {  # tag -> (label, color)
    "h2": ("aux-space, σ-DOF 48k (h₂)", "#d62728"),
    "h3": ("aux-space, σ-DOF 184k (h₃)", "#1f77b4"),
}


def _tag(run_dir: Path) -> str:
    name = run_dir.parent.name  # e.g. paper_aux_h2
    for t in ("h2", "h3", "h1", "h4", "h5"):
        if name.endswith(t):
            return t
    return name


def load_run(run_dir: Path):
    """Return per-step dict: step -> {load, maxd, niter_med, niter_max, capped}."""
    its = list(csv.DictReader(open(run_dir / "iterations.csv")))
    hist = list(csv.DictReader(open(run_dir / "history.csv")))
    maxd_by_step = {}
    load_by_step = {}
    for r in hist:
        s = int(r["step"])
        maxd_by_step[s] = float(r["max_d"])
        load_by_step[s] = float(r["load"])
    niters = defaultdict(list)
    capped = defaultdict(bool)
    for r in its:
        s = int(r["step"])
        ni = int(r["linear_niter_elastic"])
        if ni <= 0:
            continue
        niters[s].append(ni)
        if ni >= MAXIT and str(r["linear_converged_elastic"]).strip().lower() != "true":
            capped[s] = True
    out = {}
    for s in sorted(niters):
        v = sorted(niters[s])
        out[s] = {
            "load": load_by_step.get(s, np.nan),
            "maxd": maxd_by_step.get(s, np.nan),
            "niter_med": v[len(v) // 2],
            "niter_max": v[-1],
            "niter_min": v[0],
            "capped": capped[s],
        }
    return out


def main():
    args = [Path(a) for a in sys.argv[1:] if not a.startswith("-")]
    runs = args if args else DEFAULT_RUNS

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax2 = ax.twinx()

    transition = None
    for run_dir in runs:
        if not (run_dir / "iterations.csv").exists():
            print(f"[skip] no iterations.csv in {run_dir}")
            continue
        tag = _tag(run_dir)
        label, color = STYLE.get(tag, (tag, "#444444"))
        data = load_run(run_dir)
        steps = np.array(sorted(data))
        med = np.array([data[s]["niter_med"] for s in steps])
        lo = np.array([data[s]["niter_min"] for s in steps])
        hi = np.array([data[s]["niter_max"] for s in steps])
        maxd = np.array([data[s]["maxd"] for s in steps])
        capped = np.array([data[s]["capped"] for s in steps])

        # niter (left axis): median line + min/max band
        ax.plot(steps, med, "-o", color=color, ms=4, lw=1.6,
                label=f"{label}: niter (median)", zorder=3)
        ax.fill_between(steps, lo, hi, color=color, alpha=0.15, zorder=1)
        # hollow markers for capped (maxit reached, converged=False)
        if capped.any():
            ax.plot(steps[capped], hi[capped], "o", mfc="white", mec=color,
                    mew=1.4, ms=8, zorder=4,
                    label=f"{label}: hit maxit={MAXIT} (res~1e-7)")
        # max_d (right axis)
        ax2.plot(steps, maxd, "--s", color=color, ms=3, lw=1.1, alpha=0.55,
                 zorder=2, label=f"{label}: max_d")

        # detect transition step (first step with maxd jump > 0.4)
        if transition is None:
            for i in range(1, len(steps)):
                if maxd[i] - maxd[i - 1] > 0.4:
                    transition = steps[i]
                    break

    ax.set_xlabel("load step")
    ax.set_ylabel("GMRES iterations (elastic block)")
    ax2.set_ylabel("max damage  max_d", color="#555555")
    ax.set_yscale("log")
    ax.set_ylim(4, 400)
    ax2.set_ylim(0, 1.05)
    ax.axhline(MAXIT, color="#999999", ls=":", lw=0.9, zorder=0)
    ax.text(0.5, MAXIT * 1.02, f"maxit = {MAXIT}", color="#777777", fontsize=8)

    if transition is not None:
        ax.axvline(transition, color="#222222", ls="-.", lw=0.9, alpha=0.6, zorder=0)
        ax.annotate("crack fully localizes\nmax_d 0.43→0.998, niter 7→~95",
                    xy=(transition, 95), xytext=(transition - 7, 200),
                    fontsize=8, ha="left",
                    arrowprops=dict(arrowstyle="->", color="#222222", lw=0.9))

    ax.set_title("Real phase-field run: bounded O(100) convergence at localization\n"
                 "(synthetic uniform-d underestimates this — see §5.2)", fontsize=10)
    # merge legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center left", framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUTDIR / f"iter_stability_localization.{ext}"
        fig.savefig(out, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
