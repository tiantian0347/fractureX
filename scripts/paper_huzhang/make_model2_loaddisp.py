"""model2 (notch x-stretch) 正式 reaction–displacement 曲线（最终版）。

数据：results/phasefield/model2_notch_x_stretch/paper_direct_full/epsg_1e-06/history.csv
（Hu–Zhang p=3 混合元 + AT2 相场，direct/pardiso，真实 staggered，跑满 step200）。

要点（见 memory model2_loadschedule_discontinuity）：
  - 续算换过位移步长，steps 65–76 回退-重载（disp_x 非单调）。作图取**单调位移前沿**
    （丢 disp_x ≤ running-max 的回退点），raw history 不动。
  - summary.json 是旧续算段(n_load_steps=63)的陈旧统计；峰值等结论全部由 history.csv 现算。
  - max_d 全程 =1（预裂纹自始存在），标注为"预裂纹+扩展"，非"max_d→1"。

输出：docs/figures/precond/model2_loaddisp.{png,pdf} 与 Frac_huzhang/figures/model2_loaddisp.{png,pdf}。
计算/IO 用 numpy+matplotlib（绘图脚本，非求解内核）。
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/gongshihua/tian/fracturex"
HIST = os.path.join(ROOT, "results/phasefield/model2_notch_x_stretch/"
                    "paper_direct_full/epsg_1e-06/history.csv")
OUTDIRS = [os.path.join(ROOT, "docs/figures/precond"),
           os.path.join(ROOT, "Frac_huzhang/figures")]


def monotonic_front(rows):
    """取单调位移前沿：丢弃 disp_x ≤ 已见最大位移的回退点。"""
    front = []
    mx = -np.inf
    for r in rows:
        d = float(r["disp_x"])
        if d > mx + 1e-15:
            front.append(r)
            mx = d
    return front


def main():
    rows = list(csv.DictReader(open(HIST)))
    front = monotonic_front(rows)
    disp = np.array([float(r["disp_x"]) for r in front])
    R = np.array([abs(float(r["R"])) for r in front])
    max_d = np.array([float(r["max_d"]) for r in front])

    ipk = int(np.argmax(R))
    d_pk, R_pk = disp[ipk], R[ipk]
    print(f"[model2] front {len(front)}/{len(rows)} pts; "
          f"peak |R|={R_pk:.4f} @ disp_x={d_pk:.5f} (step {front[ipk]['step']}); "
          f"final |R|={R[-1]:.4f} @ {disp[-1]:.5f}; max_d in [{max_d.min():.3f},{max_d.max():.3f}]")

    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    ax.plot(disp, R, "-", color="C0", lw=1.8, zorder=2)
    ax.plot(disp, R, "o", color="C0", ms=3, zorder=3)
    ax.scatter([d_pk], [R_pk], s=60, facecolor="none", edgecolor="C3",
               lw=1.8, zorder=4, label=f"peak |R|={R_pk:.3f} @ u={d_pk:.4f}")
    ax.annotate(f"crack propagation\n(|R| drops {R_pk:.3f}→{R[ipk+1]:.3f})",
                xy=(d_pk, R_pk), xytext=(d_pk + 0.0015, R_pk - 0.02),
                fontsize=8.5, color="C3",
                arrowprops=dict(arrowstyle="->", color="C3", lw=1))
    ax.set_xlabel("prescribed x-displacement  $u_x$")
    ax.set_ylabel("reaction force  $|R_x|$")
    ax.set_title("model2 (notch, x-stretch): Hu–Zhang p=3 + AT2 phase field\n"
                 "load–displacement (pre-crack present, $\\max d=1$ throughout)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()

    for od in OUTDIRS:
        os.makedirs(od, exist_ok=True)
        for ext in ("png", "pdf"):
            f = os.path.join(od, f"model2_loaddisp.{ext}")
            fig.savefig(f, dpi=150)
            print(f"[model2] wrote {f}")
    plt.close(fig)


if __name__ == "__main__":
    main()
