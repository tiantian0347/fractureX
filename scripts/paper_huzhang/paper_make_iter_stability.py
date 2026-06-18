#!/usr/bin/env python3
"""C2/C3 核心对比图：预条件子迭代稳定性（对照组 none/Jacobi/ILU/aux）。

读 `results/phasefield/_iter_stability/iter_stability.csv`（由 iter_stability_scan.py 产），
绘两张图：
  (C3) niter vs 损伤 d（半对数，固定网格）——aux 平稳、others 爆炸/erratic；
  (C2) niter vs σ-DOF（半对数，固定 d）——aux mesh-independent、others 随 N 涨。
打满 maxit / 未收敛(DNF) 的点用空心标记区分。

与已有 `paper_make_iter_vs_N.py`（从真实 phase-field run 的 iterations.csv 聚合 aux/direct
的 niter-vs-N）互补：本脚本是受控 uniform-d 的多预条件子对照。

用法: paper_make_iter_stability.py [csv_path]
输出: docs/figures/precond/iter_stability_vs_{d,N}.{png,pdf}
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
    else _REPO / "results/phasefield/_iter_stability/iter_stability.csv"
OUTDIR = _REPO / "docs/figures/precond"
OUTDIR.mkdir(parents=True, exist_ok=True)

DNF = 60000  # restart*maxit 上限：达到即视为未收敛

STYLE = {  # precond -> (label, color, marker)
    "none":     ("no precond",      "#888888", "v"),
    "jacobi":   ("Jacobi",          "#1f77b4", "s"),
    "ilu":      ("ILU",             "#ff7f0e", "^"),
    "aux_fast": ("aux-space (ours)", "#d62728", "o"),
}
ORDER = ["none", "jacobi", "ilu", "aux_fast"]


def load_rows(path: Path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "level": r["level"], "sigma": int(float(r["sigma"])),
                "d": float(r["d"]), "precond": r["precond"],
                "niter": int(float(r["niter"])),
                "converged": str(r["converged"]).strip().lower() == "true",
            })
    return rows


def _plot(ax, xs, ys, conv, label, color, marker):
    xs = np.asarray(xs); ys = np.asarray(ys, float); conv = np.asarray(conv, bool)
    order = np.argsort(xs); xs, ys, conv = xs[order], ys[order], conv[order]
    ax.plot(xs, ys, "-", color=color, label=label, lw=1.6, zorder=2)
    if conv.any():
        ax.scatter(xs[conv], ys[conv], c=color, marker=marker, s=46, zorder=3,
                   edgecolors="k", linewidths=0.4)
    if (~conv).any():  # DNF / 未收敛: 空心
        ax.scatter(xs[~conv], ys[~conv], facecolors="none", edgecolors=color,
                   marker=marker, s=46, zorder=3, linewidths=1.4)


def fig_vs_d(rows, level: str):
    sub = [r for r in rows if r["level"] == level]
    if not sub:
        return None
    ds = sorted({r["d"] for r in sub})
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for pc in ORDER:
        xs, ys, cv = [], [], []
        for d in ds:
            m = [r for r in sub if r["precond"] == pc and r["d"] == d]
            if m:
                xs.append(d); ys.append(m[0]["niter"]); cv.append(m[0]["converged"])
        if xs:
            lab, col, mk = STYLE[pc]
            _plot(ax, xs, ys, cv, lab, col, mk)
    sig = sub[0]["sigma"]
    ax.set_yscale("log")
    ax.set_xlabel(r"uniform damage $d$  ($g=(1-d)^2+\epsilon_g$, $\epsilon_g=10^{-6}$)")
    ax.set_ylabel(r"GMRES iterations ($r_{\mathrm{tol}}=10^{-8}$)")
    # Title intentionally omitted; the paper supplies a full caption.
    ax.axhline(DNF, color="grey", ls=":", lw=0.8)
    ax.text(ds[0], DNF, " DNF (hit maxit)", va="bottom", ha="left", fontsize=7, color="grey")
    ax.legend(fontsize=8, framealpha=0.9); ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return fig


def _load_aux_meshindep(d_fixed: float):
    """补充：aux_fast 的大规模网格无关数据（h₃–h₅，对照组在此 DNF/不可行故缺）。
    读 aux_mesh_indep_d{d}.csv（H1 脚本产，列 hmin,sigma,n_iter,converged,t_solve_s），
    返回 [{sigma, niter, converged}]，供 fig_vs_N 把 aux 线延到 187× DOF。无文件则空。"""
    p = CSV.parent / f"aux_mesh_indep_d{d_fixed:g}.csv"
    out = []
    if not p.is_file():
        return out
    with open(p) as f:
        for r in csv.DictReader(f):
            out.append({"sigma": int(float(r["sigma"])),
                        "niter": int(float(r["n_iter"])),
                        "converged": str(r["converged"]).strip().lower() == "true"})
    return out


def fig_vs_N(rows, d_fixed: float):
    sub = [r for r in rows if abs(r["d"] - d_fixed) < 1e-9]
    # 合并 aux_fast 的大规模补充档（h₃–h₅）：仅 aux 在此规模可行
    extra = _load_aux_meshindep(d_fixed)
    have_aux = {r["sigma"] for r in sub if r["precond"] == "aux_fast"}
    for e in extra:
        if e["sigma"] not in have_aux:
            sub.append({"level": "", "sigma": e["sigma"], "d": d_fixed,
                        "precond": "aux_fast", "niter": e["niter"],
                        "converged": e["converged"]})
    sizes = sorted({r["sigma"] for r in sub})
    if len(sizes) < 2:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for pc in ORDER:
        xs, ys, cv = [], [], []
        for N in sizes:
            m = [r for r in sub if r["precond"] == pc and r["sigma"] == N]
            if m:
                xs.append(N); ys.append(m[0]["niter"]); cv.append(m[0]["converged"])
        if xs:
            lab, col, mk = STYLE[pc]
            _plot(ax, xs, ys, cv, lab, col, mk)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"stress degrees of freedom $N_\sigma$ (mesh refinement $\rightarrow$)")
    ax.set_ylabel(r"GMRES iterations ($r_{\mathrm{tol}}=10^{-8}$)")
    # Title intentionally omitted; the paper supplies a full caption.
    ax.legend(fontsize=8, framealpha=0.9); ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return fig


def main():
    if not CSV.is_file():
        print(f"CSV not found: {CSV}\n(先跑 iter_stability_scan.py)"); sys.exit(1)
    rows = load_rows(CSV)
    # vs-d 图选数据最全的网格（distinct d 最多；并列取最细），避免部分数据画残图
    by_level = {}
    for r in rows:
        by_level.setdefault(r["level"], {"ds": set(), "sigma": r["sigma"]})
        by_level[r["level"]]["ds"].add(r["d"])
    level_vsd = max(by_level, key=lambda L: (len(by_level[L]["ds"]), by_level[L]["sigma"]))
    saved = []
    f1 = fig_vs_d(rows, level_vsd)
    if f1:
        for ext in ("png", "pdf"):
            p = OUTDIR / f"iter_stability_vs_d.{ext}"; f1.savefig(p, dpi=150); saved.append(p)
    ds = sorted({r["d"] for r in rows})
    d_fixed = 0.9 if 0.9 in ds else (0.0 if 0.0 in ds else ds[0])
    f2 = fig_vs_N(rows, d_fixed)
    if f2:
        for ext in ("png", "pdf"):
            p = OUTDIR / f"iter_stability_vs_N.{ext}"; f2.savefig(p, dpi=150); saved.append(p)
    else:
        print("(<2 网格, 跳过 vs-N 图; 等 h2/h3)")
    for p in saved:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
