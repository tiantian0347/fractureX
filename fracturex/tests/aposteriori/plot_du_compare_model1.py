"""Phase 2 分析：载荷步 du 对峰值/曲线的影响（du=1e-4 vs du=2.5e-4 vs 均匀 nx120 参照）。

问题：M3 PC 峰值与参照的残差里，du 采样占多少？峰值的 solver-path 带是否随 du 收窄？
读：
  du=1e-4   : results/adaptive_m3_pc_model1_du1e4/history.csv
  du=2.5e-4 : results/adaptive_m3_pc_model1_v3/history_anderson_canonical.csv（规范 Anderson）
  参照      : .../paper_direct_full_nx120/.../history.csv（du=1e-4, nx120）
出图 docs/figures/adaptive/m3pc_du_compare_model1.png + 打印峰值表。

运行: PYTHONPATH=$PWD python fracturex/tests/aposteriori/plot_du_compare_model1.py
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "docs/figures/adaptive"
DU1 = "results/adaptive_m3_pc_model1_du1e4/history.csv"
DU25 = "results/adaptive_m3_pc_model1_v3/history_anderson_canonical.csv"
REF = ("results/phasefield/square_tension_precrack/"
       "paper_direct_full_nx120/epsg_1e-06/history.csv")


def _read(path, dcol, rcol):
    rows = list(csv.DictReader(open(path)))
    d, r = [], []
    for row in rows:
        try:
            dv, rv = float(row[dcol]), abs(float(row[rcol]))
        except (KeyError, ValueError):
            continue
        d.append(dv); r.append(rv)
    return np.asarray(d), np.asarray(r)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    series = []
    for nm, path, dc, rc, sty in [
        ("uniform nx120 ref (du=1e-4)", REF, "disp_y", "R", ("-", "0.35", 2.0)),
        ("PC du=2.5e-4 (Anderson)", DU25, "load", "reaction", ("s--", "C7", 1.4)),
        ("PC du=1e-4 (Anderson)", DU1, "load", "reaction", ("o-", "C0", 1.8)),
    ]:
        try:
            u, R = _read(path, dc, rc)
            pk_i = int(np.argmax(R))
            series.append((nm, u, R, (u[pk_i], R[pk_i]), sty))
        except FileNotFoundError:
            print(f"[skip] {nm}: {path} not found")

    ref_peak = series[0][3][1]
    print("=== peaks (ref = %.4f) ===" % ref_peak)
    for nm, u, R, pk, _ in series:
        print(f"{nm:30s} peak={pk[1]:.4f} @u={pk[0]:.4e}  vs ref {(pk[1]-ref_peak)/ref_peak*100:+.1f}%")

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for nm, u, R, pk, (ls, c, lw) in series:
        ax.plot(u, R, ls, color=c, lw=lw, ms=3,
                label=f"{nm}, peak {pk[1]:.3f}")
        ax.scatter([pk[0]], [pk[1]], color=c, s=35, zorder=5)
    ax.set_xlabel("prescribed displacement  u_y")
    ax.set_ylabel("reaction force  |R|")
    ax.set_title("model1 peak vs load-step du (Phase 2)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f = os.path.join(OUTDIR, "m3pc_du_compare_model1.png")
    fig.savefig(f, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {f}")


if __name__ == "__main__":
    main()
