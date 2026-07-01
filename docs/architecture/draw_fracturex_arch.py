"""
FractureX architecture diagram, v6 -- minimal & clean.

Render command:
    python3 fractureX/docs/architecture/draw_fracturex_arch.py

Generates fracturex_arch.png at the same directory, and a copy is
shipped to Tian_silde/figures/ (slide deck).

Layout:
- Modules grouped by function layer, NO inter-module arrows.
- One single arrow only: fracturex -> FEALPy backend.
- discretization shows two sub-components: Hu--Zhang sigma element +
  phase-field d discretization; damage/phasefield card clarifies that
  local damage uses a closed-form d while the phase-field model
  discretizes d explicitly.
- Flow layer now includes adaptivity to match the actual package
  structure (fractureX/fracturex/adaptivity/).
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm
from pathlib import Path

def find_cn_font():
    for c in ["PingFang SC", "Heiti SC", "STHeiti", "Songti SC",
              "Arial Unicode MS", "Hiragino Sans GB"]:
        if c in {f.name for f in fm.fontManager.ttflist}:
            return c
    return None

cn = find_cn_font()
if cn:
    plt.rcParams["font.family"] = cn
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(13.5, 7.0), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 70)
ax.axis("off")

PRIMARY = "#005A9C"
AIGREEN = "#008C5A"
WARN = "#C85A00"
GRAY = "#666666"
TEXT_DARK = "#222"


def card(x, y, w, h, name, desc, *, ec=PRIMARY, fc="white",
        name_color=None, name_fs=14, desc_fs=11, lw=1.8):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.5",
        linewidth=lw, edgecolor=ec, facecolor=fc))
    nc = name_color or ec
    ax.text(x + w / 2, y + h * 0.72, name,
            ha="center", va="center", fontsize=name_fs,
            fontweight="bold", color=nc)
    ax.text(x + w / 2, y + h * 0.30, desc,
            ha="center", va="center", fontsize=desc_fs, color=GRAY,
            linespacing=1.3)


def layer_label(y, label):
    ax.text(2, y, label, ha="left", va="center",
            fontsize=11, color="#999", style="italic")


# title
ax.text(50, 67, "FractureX 程序架构",
        ha="center", va="center", fontsize=22,
        fontweight="bold", color=TEXT_DARK)
ax.text(50, 63.5, "按功能分层 · 模块解耦 · 基于 FEALPy",
        ha="center", va="center", fontsize=12, color=GRAY, style="italic")

# Outer fracturex container
outer = FancyBboxPatch((6, 14), 88, 45,
                       boxstyle="round,pad=0.02,rounding_size=0.8",
                       linewidth=1.4, edgecolor=PRIMARY,
                       facecolor="#F7FAFC", linestyle="--")
ax.add_patch(outer)
ax.text(8, 57, "fracturex",
        ha="left", va="center", fontsize=14,
        fontweight="bold", color=PRIMARY, style="italic")

# ====== Layer 1: 算例层 (cases) ======
y1 = 47
layer_label(y1 + 3.5, "算例层")
card(20, y1, 60, 7,
     "cases", "物理场景：网格 · 材料 · 边界 · 加载",
     ec=PRIMARY, name_fs=15)

# ====== Layer 2: 建模层 (discretization + damage/phasefield) ======
y2 = 36
layer_label(y2 + 3.5, "建模层")
card(10, y2, 40, 7,
     "discretization",
     "Hu--Zhang 应力元 (σ)\n相场场量 d ∈ [0,1]   ·   历史变量 H",
     ec=PRIMARY, name_fs=14)
card(52, y2, 40, 7,
     "damage / phasefield",
     "局部损伤：闭式 d   ·   相场：离散 d   ·   g(d)",
     ec=WARN, name_color=WARN, name_fs=14)

# ====== Layer 3: 算法层 (assemblers + utilfuc) ======
y3 = 25
layer_label(y3 + 3.5, "算法层")
card(10, y3, 40, 7,
     "assemblers", "应力块 · 相场块 · 三套模型支持",
     ec=PRIMARY, name_fs=14)
card(52, y3, 40, 7,
     "utilfuc  (linear_solvers)",
     "直接法 / Krylov / 辅助空间预条件",
     ec=AIGREEN, name_color=AIGREEN, name_fs=14)

# ====== Layer 4: 流程层 (drivers + adaptivity + postprocess) ======
y4 = 15
layer_label(y4 + 3.0, "流程层")
card(6, y4, 28, 6.5,
     "drivers", "加载步 · 交错迭代 · 收敛判据",
     ec=PRIMARY, name_fs=13)
card(36, y4, 28, 6.5,
     "adaptivity", "σ 驱动标记 · bisect 加密",
     ec=AIGREEN, name_color=AIGREEN, name_fs=13)
card(66, y4, 28, 6.5,
     "postprocess", "反力 · 历史 · VTK 导出",
     ec=PRIMARY, name_fs=13)

# ====== FEALPy backend ======
fy = 2
fh = 9
ax.add_patch(FancyBboxPatch(
    (6, fy), 88, fh,
    boxstyle="round,pad=0.02,rounding_size=0.7",
    linewidth=2.0, edgecolor="#444", facecolor="#EEEEEE"))
ax.text(50, fy + fh / 2 + 1.3, "FEALPy 多后端",
        ha="center", va="center", fontsize=15,
        fontweight="bold", color=TEXT_DARK)
ax.text(50, fy + fh / 2 - 1.7,
        "NumPy · PyTorch · JAX   |   网格 · 函数空间 · 积分 · 稀疏线性代数",
        ha="center", va="center", fontsize=11, color=GRAY)

# Single arrow: fracturex -> FEALPy
arrow = FancyArrowPatch((50, 14), (50, 11.2),
                        arrowstyle="-|>", mutation_scale=24,
                        color="#555", linewidth=2.4,
                        shrinkA=0, shrinkB=0)
ax.add_patch(arrow)
ax.text(52, 12.6, "depends on", ha="left", va="center",
        fontsize=10, color="#555", style="italic")

plt.tight_layout()
out_local = Path(__file__).parent / "fracturex_arch.png"
plt.savefig(out_local, dpi=200, bbox_inches="tight",
            facecolor="white", pad_inches=0.15)

# Also copy to slide-deck figures (if the path exists)
slide_out = Path(
    "/Users/tian00/repository/Tian/ppt/Tian_silde/figures/fracturex_arch.png")
if slide_out.parent.exists():
    plt.savefig(slide_out, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
plt.close(fig)
print(f"Saved {out_local}")
if slide_out.parent.exists():
    print(f"Saved {slide_out}")
