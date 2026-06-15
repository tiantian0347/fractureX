#!/usr/bin/env python3
"""C1 谱图（D12 §5.6 图3 / Prop2-3 支撑）。

读 `precond_spectrum.py` 产的 `spec_auxfast_d*.npz`（含 eig_large/eig_small 复特征值、
max_d_actual、kappa_proxy），绘：
  (图3a) P^{-1}K 特征值复平面散点，按损伤 d 叠加 —— aux 谱聚集随 d→1 稳定；
  (图3b) kappa_proxy vs max_d —— 有界 → 参数无关性 (Prop3) 数值验证。

用法: paper_make_spectrum.py [glob]   默认 results/phasefield/_iter_stability/spec_auxfast_d*.npz
输出: docs/figures/precond/spectrum_{scatter,kappa_vs_d}.{png,pdf}
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
PAT = sys.argv[1] if len(sys.argv) > 1 else str(
    _REPO / "results/phasefield/_iter_stability/spec_auxfast_d*.npz")
OUTDIR = _REPO / "docs/figures/precond"; OUTDIR.mkdir(parents=True, exist_ok=True)


def load():
    items = []
    for p in sorted(glob.glob(PAT)):
        z = np.load(p, allow_pickle=True)
        items.append({
            "d": float(z["max_d_actual"]),
            "eig": np.concatenate([np.atleast_1d(z["eig_large"]),
                                   np.atleast_1d(z["eig_small"])]),
            "kappa": float(z["kappa_proxy"]),
        })
    items.sort(key=lambda r: r["d"])
    return items


def main():
    items = load()
    if not items:
        print(f"no npz matched {PAT}"); sys.exit(1)
    cmap = plt.cm.viridis
    saved = []

    # 图3a: 特征值复平面散点 vs d
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for i, it in enumerate(items):
        c = cmap(i / max(len(items) - 1, 1))
        e = it["eig"]
        ax.scatter(e.real, e.imag, s=30, color=c, alpha=0.8,
                   label=f"d={it['d']:.2f}", edgecolors="k", linewidths=0.3)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel(r"Re $\lambda(P^{-1}\mathcal{K}_h)$"); ax.set_ylabel(r"Im $\lambda$")
    # Title intentionally omitted; the paper supplies a full caption.
    ax.legend(fontsize=8, framealpha=0.9, ncol=2); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = OUTDIR / f"spectrum_scatter.{ext}"; fig.savefig(p, dpi=150); saved.append(p)

    # 图3b: kappa_proxy vs max_d (bounded → Prop3)
    fig2, ax2 = plt.subplots(figsize=(5.0, 3.6))
    ds = [it["d"] for it in items]; ks = [it["kappa"] for it in items]
    ax2.plot(ds, ks, "o-", color="#d62728", lw=1.6, mec="k", mew=0.4)
    ax2.set_xlabel("uniform/peak damage $d$"); ax2.set_ylabel(r"$\kappa_{\mathrm{proxy}}(P^{-1}\mathcal{K}_h)=|\lambda|_{\max}/|\lambda|_{\min}$")
    # Title intentionally omitted; the paper supplies a full caption.
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(ks) * 1.3)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        p = OUTDIR / f"spectrum_kappa_vs_d.{ext}"; fig2.savefig(p, dpi=150); saved.append(p)

    print("d, kappa_proxy:")
    for it in items:
        print(f"  d={it['d']:.3f}  kappa={it['kappa']:.3e}")
    for p in saved:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
