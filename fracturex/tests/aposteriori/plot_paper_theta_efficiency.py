"""自适应论文图（2026-06-21 会话新结果）：平衡型估计子 reliability+efficiency。

产两张论文级图（数据来自本会话已验证的确定性诊断 run，见 RESULTS §Θ<1 根因诊断 / §M3 效率）：
  fig1 adaptive_theta_convergence.png —— 左:err vs nref(BC bug 修复前发散 / 修复后收敛);
                                          右:Θ=η/err vs nref 单调→1⁺（reliability+紧界双坐实）。
  fig2 adaptive_efficiency_dof.png    —— 等精度 σ-DOF 对照柱状（自适应省 93% DOF；
                                          均匀粗网格峰值虚高 +25/37% 论证需自适应）。

数据（生产配置 nx=24/nstep=20/k_res=1e-6 真实预裂纹，fine 连续-g truth；脚本
diag_theta_breakdown.py / m3_efficiency_table.py 的确定性输出，恒可复现）。
约定：纯画图（允许 np）。运行: PYTHONPATH=$PWD python fracturex/tests/aposteriori/plot_paper_theta_efficiency.py
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "docs/figures/adaptive"
os.makedirs(OUTDIR, exist_ok=True)

# --- Θ 诊断数据（k_res=1e-6 真实预裂纹；fine 连续-g truth）---
NREF = np.array([1, 2, 3, 4])
NCF = np.array([4608, 18432, 73728, 294912])
ETA = 0.04571368                          # η_τ（接受态，固定）
ERR_FIXED = np.array([0.02565893, 0.03510631, 0.04065697, 0.04380244])   # 修复后
ERR_BUGGY = np.array([0.4059668, 0.7031572, 1.074092, 1.572311])         # 修复前(BC bug)
THETA_FIXED = ETA / ERR_FIXED            # 1.78,1.30,1.12,1.044

# --- 等精度 DOF 效率（参照 nx=120, R_ref=0.6306）---
EFF = [  # (label, sigma_dof@peak, peak_dev_%)
    ("Uniform\nnx=24", 19347, +36.9),
    ("Uniform\nnx=48", 76707, +25.3),
    ("Adaptive\nPC v3", 31406, -1.5),
    ("Uniform\nnx=120\n(ref)", 476883, 0.0),
]


def fig_theta():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.0))
    # left: err vs nref, before (divergent) vs after (convergent)
    axL.plot(NREF, ERR_BUGGY, "o--", color="crimson", label="before fix (BC bug, divergent)")
    axL.plot(NREF, ERR_FIXED, "s-", color="steelblue", label="after fix (convergent, ~0.044)")
    axL.set_yscale("log")
    axL.set_xlabel("nested refinement level  nref")
    axL.set_ylabel(r"energy error $\|e\|_{C_d}$ (log)")
    axL.set_xticks(NREF)
    axL.set_title("(a) reference energy error: before / after BC-lifting fix")
    axL.legend(fontsize=9); axL.grid(alpha=0.3, which="both")
    # right: Theta vs nref -> 1
    axR.axhline(1.0, color="gray", ls=":", lw=1.2, label=r"$\Theta=1$ (reliability)")
    axR.plot(NREF, THETA_FIXED, "s-", color="steelblue")
    for x, y in zip(NREF, THETA_FIXED):
        axR.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=9)
    axR.set_xlabel("nested refinement level  nref")
    axR.set_ylabel(r"effectivity $\Theta=\eta_\tau/\|e\|$")
    axR.set_xticks(NREF); axR.set_ylim(0.9, 1.95)
    axR.set_title(r"(b) $\Theta\to 1^+$: $\eta_\tau$ exceeds true error by only 4.4%")
    axR.legend(fontsize=9); axR.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUTDIR, "adaptive_theta_convergence.png")
    fig.savefig(p, dpi=160); plt.close(fig)
    return p


def fig_efficiency():
    labels = [e[0] for e in EFF]
    dofs = np.array([e[1] for e in EFF])
    devs = [e[2] for e in EFF]
    colors = ["#d99", "#d99", "steelblue", "#888"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bars = ax.bar(range(len(EFF)), dofs, color=colors, edgecolor="k", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\sigma$-DOF @ peak step (log)")
    ax.set_xticks(range(len(EFF))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("equal-accuracy DOF efficiency: adaptive saves 93% DOF")
    for b, dev, dof in zip(bars, devs, dofs):
        tag = f"{dof}\npeak {dev:+.1f}%"
        ax.annotate(tag, (b.get_x() + b.get_width() / 2, dof), ha="center",
                    va="bottom", fontsize=8.5,
                    color=("crimson" if abs(dev) > 5 else "steelblue"))
    ax.annotate("", xy=(2, 31406), xytext=(3, 476883),
                arrowprops=dict(arrowstyle="<->", color="green", lw=1.3))
    ax.text(2.5, 1.2e5, "93% fewer DOF\n(same -1.5% accuracy)", color="green",
            ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(1e4, 1.2e6)
    fig.tight_layout()
    p = os.path.join(OUTDIR, "adaptive_efficiency_dof.png")
    fig.savefig(p, dpi=160); plt.close(fig)
    return p


def main():
    # 中文字体（落到默认时不致报错）
    try:
        plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    p1 = fig_theta(); p2 = fig_efficiency()
    print("[fig] wrote:", p1); print("[fig] wrote:", p2)


if __name__ == "__main__":
    main()
