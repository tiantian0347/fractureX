#!/usr/bin/env python3
"""D2/D3 推广算例（square / model2）的 niter-vs-d 对照图。

读 case-schema CSV（列 case,nx,sigma,d,precond,niter,converged,...，由
iter_stability_square.py / iter_stability_model2.py 产），取**最细 nx**绘半对数
niter-vs-d（none/Jacobi/ILU/aux_fast 四条），DNF/未收敛点空心标记。与主算例
paper_make_iter_stability.py 同风格，但适配 nx 列（非 level 列）。

用法: paper_make_iter_stability_case.py <csv> [out_stem]
  csv      : 例 results/phasefield/_iter_stability/iter_stability_model2.csv
  out_stem : 输出文件名干（默认取 csv 文件名去扩展名）
输出: docs/figures/precond/<out_stem>_vs_d.{png,pdf}
"""
from __future__ import annotations
import csv as _csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
OUTDIR = _REPO / "docs/figures/precond"; OUTDIR.mkdir(parents=True, exist_ok=True)
DNF = 60000

STYLE = {  # precond -> (label, color, marker)
    "none":     ("no precond",       "#888888", "v"),
    "jacobi":   ("Jacobi",           "#1f77b4", "s"),
    "ilu":      ("ILU",              "#ff7f0e", "^"),
    "aux_fast": ("aux-space (ours)", "#d62728", "o"),
}
ORDER = ["none", "jacobi", "ilu", "aux_fast"]
TITLE = {"square": "square (mode-I straight crack)",
         "model2": "model2 (mode-II notch shear)"}


def _plot(ax, xs, ys, conv, label, color, marker):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float); conv = np.asarray(conv, bool)
    o = np.argsort(xs); xs, ys, conv = xs[o], ys[o], conv[o]
    ax.plot(xs, ys, "-", color=color, label=label, lw=1.6, zorder=2)
    if conv.any():
        ax.scatter(xs[conv], ys[conv], c=color, marker=marker, s=46, zorder=3,
                   edgecolors="k", linewidths=0.4)
    if (~conv).any():
        ax.scatter(xs[~conv], ys[~conv], facecolors="none", edgecolors=color,
                   marker=marker, s=46, zorder=3, linewidths=1.4)


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    csvp = Path(sys.argv[1])
    stem = sys.argv[2] if len(sys.argv) > 2 else csvp.stem
    if not csvp.is_file():
        print(f"no csv {csvp}"); sys.exit(1)
    rows = list(_csv.DictReader(open(csvp)))
    case = rows[0].get("case", "")
    nxs = sorted({int(float(r["nx"])) for r in rows})
    nx = nxs[-1]  # 最细网格
    sub = [r for r in rows if int(float(r["nx"])) == nx]
    sig = int(float(sub[0]["sigma"]))
    ds = sorted({float(r["d"]) for r in sub})
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for pc in ORDER:
        xs, ys, cv = [], [], []
        for d in ds:
            m = [r for r in sub if r["precond"] == pc and abs(float(r["d"]) - d) < 1e-12]
            if m:
                xs.append(d); ys.append(int(float(m[0]["niter"])))
                cv.append(str(m[0]["converged"]).strip().lower() == "true")
        if xs:
            lab, col, mk = STYLE[pc]
            _plot(ax, xs, ys, cv, lab, col, mk)
    ax.set_yscale("log")
    ax.set_xlabel(r"uniform damage $d$  ($g=(1-d)^2+\epsilon_g$, $\epsilon_g$=1e-6)")
    ax.set_ylabel("GMRES iterations (rtol=1e-8)")
    ax.set_title(f"Iteration stability vs damage — {TITLE.get(case, case)} "
                 f"($\\sigma$-dof={sig:,})")
    ax.axhline(DNF, color="grey", ls=":", lw=0.8)
    ax.text(ds[0], DNF, " DNF (hit maxit)", va="bottom", ha="left", fontsize=7, color="grey")
    ax.legend(fontsize=8, framealpha=0.9); ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = OUTDIR / f"{stem}_vs_d.{ext}"; fig.savefig(p, dpi=150); print(f"wrote {p}")


if __name__ == "__main__":
    main()
