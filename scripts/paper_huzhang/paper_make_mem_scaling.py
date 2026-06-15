#!/usr/bin/env python3
"""B2 图：peak RSS vs σ-DOF（matrix-free vs direct 分解），D12 §5.x 内存支线。

读 `results/phasefield/_iter_stability/mem_scaling.csv`（mem_scaling.py 产），log-log 绘
mf 与 direct 两条曲线；direct 的 OOM/超时点用空心标记 + 注 "infeasible"。附 slope-1 参考线。

用法: paper_make_mem_scaling.py [csv]
输出: docs/figures/precond/mem_scaling.{png,pdf}
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    _REPO / "results/phasefield/_iter_stability/mem_scaling.csv"
OUTDIR = _REPO / "docs/figures/precond"; OUTDIR.mkdir(parents=True, exist_ok=True)

STYLE = {"mf": ("matrix-free (ours)", "#d62728", "o"),
         "direct": ("direct (pardiso)", "#1f77b4", "s")}


def load():
    rows = {"mf": [], "direct": []}
    for r in csv.DictReader(open(CSV)):
        try:
            sig = int(float(r["sigma"]))
        except (ValueError, KeyError):
            sig = None
        note = (r.get("note") or "").lower()
        feasible = not any(k in note for k in ("oom", "infeasible", "timeout", "err"))
        try:
            mb = float(r["peak_rss_mb"])
        except (ValueError, KeyError):
            mb = None
        rows.setdefault(r["mode"], []).append({"sigma": sig, "mb": mb, "feasible": feasible})
    for k in rows:
        rows[k] = sorted([x for x in rows[k] if x["sigma"]], key=lambda d: d["sigma"])
    return rows


def main():
    if not CSV.is_file():
        print(f"no csv {CSV}"); sys.exit(1)
    rows = load()
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    allmb = []
    for mode in ("direct", "mf"):
        rs = rows.get(mode, [])
        xs = [r["sigma"] for r in rs if r["mb"]]
        ys = [r["mb"] for r in rs if r["mb"]]
        fe = [r["feasible"] for r in rs if r["mb"]]
        if not xs:
            continue
        allmb += ys
        lab, col, mk = STYLE[mode]
        ax.plot(xs, ys, "-", color=col, lw=1.7, label=lab, zorder=2)
        xs, ys, fe = map(np.asarray, (xs, ys, fe))
        if fe.any():
            ax.scatter(xs[fe], ys[fe], c=col, marker=mk, s=48, edgecolors="k",
                       linewidths=0.4, zorder=3)
        if (~fe).any():
            ax.scatter(xs[~fe], ys[~fe], facecolors="none", edgecolors=col, marker="x",
                       s=70, linewidths=1.8, zorder=3)
            for xv, yv in zip(xs[~fe], ys[~fe]):
                ax.annotate("infeasible", (xv, yv), fontsize=7, color=col,
                            ha="center", va="bottom")
    # slope-1 reference through the smallest mf point
    mf = rows.get("mf", [])
    if mf:
        x0, y0 = mf[0]["sigma"], mf[0]["mb"]
        xr = np.array([x0, mf[-1]["sigma"]], float)
        ax.plot(xr, y0 * (xr / x0), "k:", lw=0.9, alpha=0.7, label="slope 1 (linear)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"stress degrees of freedom $N_\sigma$"); ax.set_ylabel("peak RSS (MB)")
    # Title intentionally omitted; the paper supplies a full caption.
    ax.legend(fontsize=8, framealpha=0.9); ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = OUTDIR / f"mem_scaling.{ext}"; fig.savefig(p, dpi=150); print(f"wrote {p}")


if __name__ == "__main__":
    main()
