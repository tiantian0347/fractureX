"""Publication figures for the equilibrated a posteriori paper.

  fig1 adaptive_theta_convergence.png
    Left: nested-reference energy error (increases toward E_h).
    Right: P1 diagnostic reference effectivity upper bound.
  fig2 adaptive_efficiency_dof.png
    Equal-accuracy stress-DOF comparison.

Run: PYTHONPATH=$PWD python fracturex/tests/aposteriori/plot_paper_theta_efficiency.py
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = os.environ.get("FRACTUREX_FIGDIR", "docs/figures/adaptive")
os.makedirs(OUTDIR, exist_ok=True)

# Nested-reference P1 diagnostic: one continuous prolonged P1 damage
# coefficient on every level.
NREF = np.array([1, 2, 3, 4])
ETA = 0.0607245943
ERR_REF = np.array([0.0222725088, 0.0320995060, 0.0385444494, 0.0424783408])
THETA_REF = ETA / ERR_REF  # 2.726, 1.892, 1.575, 1.430

# Equal-accuracy DOF efficiency (reference nx=120, R_ref=0.6306).
EFF = [  # (label, sigma_dof@peak, peak_dev_%)
    ("Uniform\nnx=24", 19347, +36.9),
    ("Uniform\nnx=48", 76707, +25.3),
    ("Adaptive\n" r"$(\mathcal{D}_{\tau,T}+\mathrm{PC})$", 31406, -1.5),
    ("Uniform\nnx=120\n(ref)", 476883, 0.0),
]


def fig_theta():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.0))
    axL.plot(NREF, ERR_REF, "s-", color="steelblue",
             label=r"$\mathrm{err}_{\mathrm{ref},\ell}$")
    axL.set_yscale("log")
    axL.set_xlabel("nested refinement level  nref")
    axL.set_ylabel(r"nested-reference energy error (log)")
    axL.set_xticks(NREF)
    axL.set_title(r"(a) $P_1$ surrogate nested-reference error")
    for x, y in zip(NREF, ERR_REF):
        axL.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), fontsize=9, ha="center")
    axL.legend(fontsize=9)
    axL.grid(alpha=0.3, which="both")

    axR.axhline(1.0, color="gray", ls=":", lw=1.2, label=r"$\Theta=1$ (reliability)")
    axR.plot(NREF, THETA_REF, "s-", color="steelblue")
    for x, y in zip(NREF, THETA_REF):
        axR.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), fontsize=9, ha="center")
    axR.set_xlabel("nested refinement level  nref")
    axR.set_ylabel(r"effectivity $\Theta_{\mathrm{ref}}=\eta/\mathrm{err}_{\mathrm{ref}}$")
    axR.set_xticks(NREF)
    axR.set_ylim(0.9, 2.95)
    axR.set_title(r"(b) $P_1$ diagnostic: discrete nested-reference ratio")
    axR.legend(fontsize=9)
    axR.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUTDIR, "adaptive_theta_convergence.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
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
    ax.text(2.5, 1.2e5,
            r"$93\%$ fewer stress DOFs" "\n" r"at $-1.5\%$ peak error",
            color="green", ha="center", fontsize=9, fontweight="bold")
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
