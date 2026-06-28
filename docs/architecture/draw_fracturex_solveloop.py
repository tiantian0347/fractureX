"""
FractureX solve-loop flow diagram.

Render command:
    python3 fractureX/docs/architecture/draw_fracturex_solveloop.py

Generates fracturex_solveloop.png at the same directory, with a copy
shipped to Tian_silde/figures/ (slide deck).

Loop structure:
- Outer loop: load steps t_n = n*dt
- Inner loop (staggered): elastic step  -> damage / phase-field step
                          -> convergence check (||sigma_{k+1}-sigma_k||<tol?)
- On convergence -> Postprocess (record reaction, history, optional VTK)
- Outer termination: t_n >= T
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
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

fig, ax = plt.subplots(figsize=(13.5, 8.2), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

PRIMARY = "#005A9C"
AIGREEN = "#008C5A"
WARN = "#C85A00"
PURPLE = "#7030A0"
GRAY = "#666666"
TEXT_DARK = "#222"


def box(x, y, w, h, title, body=None, *, fc="white", ec=PRIMARY,
        title_fs=13, body_fs=10, title_color=None, lw=1.6, rounding=0.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=lw, edgecolor=ec, facecolor=fc))
    tc = title_color or ec
    if body:
        ax.text(x + w / 2, y + h * 0.68, title,
                ha="center", va="center", fontsize=title_fs,
                fontweight="bold", color=tc)
        ax.text(x + w / 2, y + h * 0.30, body,
                ha="center", va="center", fontsize=body_fs, color=GRAY)
    else:
        ax.text(x + w / 2, y + h / 2, title,
                ha="center", va="center", fontsize=title_fs,
                fontweight="bold", color=tc)


def diamond(x_center, y_center, w, h, text, *, fc="#FFF6E0", ec=WARN,
            text_color=None, fs=11, lw=1.8):
    pts = [(x_center, y_center + h / 2),
           (x_center + w / 2, y_center),
           (x_center, y_center - h / 2),
           (x_center - w / 2, y_center)]
    poly = Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(poly)
    ax.text(x_center, y_center, text,
            ha="center", va="center", fontsize=fs,
            fontweight="bold", color=text_color or ec)


def varrow(x, y_top, y_bot, color=PRIMARY, lw=2.2, label=None,
           lx=None, ly=None, label_fs=10, label_color=None):
    a = FancyArrowPatch((x, y_top), (x, y_bot),
                        arrowstyle="-|>", mutation_scale=22,
                        color=color, linewidth=lw, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if label:
        ax.text(lx if lx is not None else x + 1,
                ly if ly is not None else (y_top + y_bot) / 2,
                label, fontsize=label_fs, color=label_color or color,
                ha="left", va="center")


# title
ax.text(50, 96, "FractureX 求解逻辑",
        ha="center", va="center", fontsize=22,
        fontweight="bold", color=TEXT_DARK)
ax.text(50, 92.6, "外循环：加载步推进  ·  内循环：交错迭代（弹性 ↔ 损伤/相场）",
        ha="center", va="center", fontsize=12, color=GRAY, style="italic")

# Outer loop band
outer_box = FancyBboxPatch((4, 8), 92, 80,
                           boxstyle="round,pad=0.02,rounding_size=0.8",
                           linewidth=1.6, edgecolor=PRIMARY,
                           facecolor="#FBFAF5", linestyle="-")
ax.add_patch(outer_box)
ax.text(8, 86, "外循环  ·  加载步  $t_n = n \\cdot \\Delta t,\\ n=0,1,\\ldots,N$",
        ha="left", va="center", fontsize=13, fontweight="bold",
        color=PRIMARY)

# Step 0: initialize
box(8, 78, 84, 5.0,
    "初始化", body=r"读取 Case · 构建空间与初值 · $\sigma_0 = 0,\ d_0 = 0,\ H_0 = 0$",
    fc="#EAF2FA", ec=PRIMARY, title_fs=13, body_fs=11)

# Step 1: load step BC
box(8, 70, 84, 5.0,
    "施加加载步 $t_n$",
    body=r"更新位移/牵引边界条件 $\bar{u}(t_n)$",
    fc="#EAF2FA", ec=PRIMARY, title_fs=13, body_fs=11)

# Inner loop band
inner_box = FancyBboxPatch((10, 18), 80, 50,
                           boxstyle="round,pad=0.02,rounding_size=0.6",
                           linewidth=1.4, edgecolor=AIGREEN,
                           facecolor="#F1F8F4", linestyle="--")
ax.add_patch(inner_box)
ax.text(14, 66, "内循环  ·  交错迭代  $k = 0, 1, 2, \\ldots$",
        ha="left", va="center", fontsize=13, fontweight="bold",
        color=AIGREEN)

# Elastic step
box(14, 54, 72, 6.0,
    "弹性步（HZ 混合元）",
    body=r"给定 $d_k$，装配 $A(d_k), B(d_k)$ + 求解 $\to (\sigma_{k+1}, u_{k+1})$",
    fc="white", ec=PRIMARY, title_fs=13, body_fs=11, lw=1.8)

# Damage / phase-field step
box(14, 44, 72, 6.0,
    "损伤 / 相场步",
    body=r"更新历史 $H = \max\{H, \Psi^+(\sigma_{k+1})\}$ + 求解 $\to d_{k+1}$",
    fc="white", ec=WARN, title_fs=13, body_fs=11, title_color=WARN, lw=1.8)

# Convergence diamond
diamond(50, 36, 36, 7,
        text=r"$\|\sigma_{k+1}-\sigma_k\| < \mathrm{tol}\,?$",
        fc="#FFF6E0", ec=WARN, fs=12)

# arrows top -> bottom
varrow(50, 70, 60.0, color=PRIMARY)  # load step BC -> elastic
varrow(50, 54, 50.0, color=AIGREEN)  # elastic -> damage
varrow(50, 44, 39.5, color=WARN)     # damage -> diamond

# NO branch -> back to elastic step
a_no = FancyArrowPatch((32, 36), (10, 36),
                       arrowstyle="-", color=AIGREEN, linewidth=1.8,
                       connectionstyle="arc3,rad=0")
ax.add_patch(a_no)
a_no2 = FancyArrowPatch((10, 36), (10, 57),
                        arrowstyle="-", color=AIGREEN, linewidth=1.8)
ax.add_patch(a_no2)
a_no3 = FancyArrowPatch((10, 57), (14, 57),
                        arrowstyle="-|>", mutation_scale=18,
                        color=AIGREEN, linewidth=1.8)
ax.add_patch(a_no3)
ax.text(7, 46, "NO\n$k\\leftarrow k+1$", fontsize=10, color=AIGREEN,
        ha="center", va="center", fontweight="bold")

ax.text(70, 36, "YES", fontsize=11, color=WARN,
        ha="left", va="center", fontweight="bold")

# Inner loop YES out -> down to record
varrow(50, 32.5, 21, color=PRIMARY, lw=2.2)

# Postprocess
box(8, 13, 84, 5.5,
    "Postprocess",
    body=r"记录反力 $F_x$ · 加载--位移历史 · 状态 $(\sigma, u, d, H)$ · 可选 VTK",
    fc="#EAF2FA", ec=PRIMARY, title_fs=13, body_fs=11)

# Outer loop back to top
a_outer = FancyArrowPatch((92, 16), (97, 16),
                          arrowstyle="-", color=PRIMARY, linewidth=2.0)
ax.add_patch(a_outer)
a_outer2 = FancyArrowPatch((97, 16), (97, 72.5),
                           arrowstyle="-", color=PRIMARY, linewidth=2.0)
ax.add_patch(a_outer2)
a_outer3 = FancyArrowPatch((97, 72.5), (92, 72.5),
                           arrowstyle="-|>", mutation_scale=20,
                           color=PRIMARY, linewidth=2.0)
ax.add_patch(a_outer3)
ax.text(99, 44, "$n \\leftarrow n+1$",
        fontsize=11, color=PRIMARY, ha="left", va="center",
        rotation=90, fontweight="bold")

ax.text(50, 5.0,
        "外循环结束条件： $t_n \\geq T$  $\\Rightarrow$  输出最终结果",
        ha="center", va="center", fontsize=12,
        color=PRIMARY, style="italic")

plt.tight_layout()
out_local = Path(__file__).parent / "fracturex_solveloop.png"
plt.savefig(out_local, dpi=200, bbox_inches="tight",
            facecolor="white", pad_inches=0.15)
slide_out = Path(
    "/Users/tian00/repository/Tian/ppt/Tian_silde/figures/fracturex_solveloop.png")
if slide_out.parent.exists():
    plt.savefig(slide_out, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
plt.close(fig)
print(f"Saved {out_local}")
if slide_out.parent.exists():
    print(f"Saved {slide_out}")
