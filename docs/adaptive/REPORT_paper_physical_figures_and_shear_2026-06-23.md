# 验证报告：adaptive 论文物理配图 + 标记对比 + shear 可行性（2026-06-23）

> 对应论文 `Frac_huzhang/adaptive/equilibrated_aposteriori.tex`。
> 本轮处理三件 TODO：P4 baseline marker 对比、补 load–disp/mesh/damage 图、P1 shear pending。
> 配套实验日志见 [RESULTS_aposteriori.md](RESULTS_aposteriori.md)。环境 py312。

## 1. 算例 / 数据来源

全部基于**已有**结果，无新解算（除尝试性 shear run，见 §4）：

| 用途 | 数据 | schema |
|------|------|--------|
| 参照 load–disp | `results/phasefield/square_tension_precrack/paper_direct_full_nx120/epsg_1e-06/history.csv` | `disp_y,R`，峰值 0.6306 @ σ-DOF 476883 |
| 自适应 M-DF+PC | `results/adaptive_m3_pc_model1_v3/history_anderson_canonical.csv` | `load,reaction`，峰值 0.6214（−1.5%）@ step20 |
| η-Dörfler once/step | `results/adaptive_m3_full_model1/history.csv` | `load,reaction`，峰值 0.7336（+16%）@ step27 |
| 粗均匀 nx24 | `results/uniform_m3_model1_nx24/history.csv` | `load,reaction`，峰值 0.8633（+37%） |
| 网格+损伤 | `results/adaptive_m3_pc_model1_v3/vtu/step_{020,024}.vtu` | PointData `damage`，1918→3400 三角形 |

## 2. 算法 / 出图

脚本 `fracturex/tests/aposteriori/plot_adaptive_paper_physical.py`（纯文件 I/O + matplotlib；vtu 用 meshio 读），
输出到 `Frac_huzhang/adaptive/figures/`：

- `adaptive_load_displacement.png`：自适应 vs nx120 参照 vs 粗均匀 nx24（峰值标记）。
- `adaptive_marker_comparison.png`：η-Dörfler(once/step) vs σ-driven M-DF+PC vs 参照（P4）。
- `adaptive_mesh_damage.png`：峰值步 / 扩展步的 adapted mesh（triplot）+ damage（tripcolor）。

## 3. 输出 / 关键数字（与 RESULTS 既有记录逐项一致）

```
[fig1] ref=0.6306  adaptive=0.6214(-1.5%)  nx24=0.8633(+37%)
[fig2] ref=0.6306  M-DF=0.6214(-1.5%)      eta-Dorfler=0.7336(+16%)
[fig3] peak step 20 (1918 elem), propagation step 24 (3400 elem)
```

## 4. P1 shear（model2）可行性结论：**不可完成，保持 future work**

- **尝试**：`FRACTUREX_CASE=model2 FRACTUREX_DU=2.5e-4 FRACTUREX_ELASTIC_SOLVER=pardiso`
  重跑（输出 `results/adaptive_m3_pc_model2_v2/`）。
- **硬证据**：上一轮 model2 进程（PID 3748342）自 2026-06-22 起 **连续运行 1 天 7 小时、累计 CPU 20.5 h**，
  `history.csv` 自 06-22 12:39 起**再未推进过 step 12**——卡在 step 12 之后的第一个局部化步上（DNF 黑洞）。
- **数据形态（model2 既有 13 步）**：反力 step 1–11 **完美线性弹性**（dR/du≈42.7，每步 +0.01068），
  𝒟max 到 step11 仅 0.19（恰低于标记阈值 θ_D=0.20，损伤尚未演化）；step12 𝒟max 暴涨到 22805（局部化起始）、
  网格 1152→1688、反力 0.1175→0.0856。即：**只有弹性段 + 1 个非线性点，之后不收敛**。
- **判读**：无峰值平台、无软化分支 ⇒ **无可用的 load–disp/峰值/σ-DOF@peak**。瓶颈是 staggered 在 mode-II
  局部化的**串行不收敛**（算法问题，176 核/2TB 无法加速）。与 model1 不同：model1 峰值在 DNF **之前**（step20<step22 DNF）
  故仍拿得到峰值；model2 峰值落在 DNF 步上，拿不到。
- **处置**：进程已 kill（旧 zombie + 本轮重跑均停）。论文维持 shear 在 future work（与当前主稿一致），
  不报假数据。本结论入档供后续（如换 Anderson 更深 / 弧长 / mode-II 专用稳定化时复核）。

## 5. 论文改动（`equilibrated_aposteriori.tex`）

§Numerical experiments 在 fig:eff 后新增三段 + 三图：
- `\paragraph{Load--displacement response.}` + Fig `fig:loaddisp`。
- `\paragraph{Predictive versus error-driven marking.}` + Fig `fig:marker`，
  **含诚实标注**：v1/v3 同时改了 marker 与加密节奏（once/step vs predictor–corrector），
  是 strategy-level 对比而非单变量 marker ablation，单变量 ablation 留 future work。
- `\paragraph{Adapted mesh and damage.}` + Fig `fig:meshdamage`。

> ⚠ 本机无 LaTeX 引擎（pdflatex/xelatex/tectonic 均无），**未本地编译**；新图块照搬既有 figure 环境写法、
> 三个 png 均在 `figures/`。需在 Overleaf / 有 TeX 的机器重编译核验。

## 6. 复现

```bash
cd /home/gongshihua/tian/fracturex
conda activate py312
PYTHONPATH=$PWD python fracturex/tests/aposteriori/plot_adaptive_paper_physical.py
# 三图写入 ../Frac_huzhang/adaptive/figures/
```
shear 复现（会卡 step12 之后，勿久等）：
```bash
FRACTUREX_CASE=model2 FRACTUREX_DU=2.5e-4 FRACTUREX_ELASTIC_SOLVER=pardiso \
FRACTUREX_OUTDIR=results/adaptive_m3_pc_model2_v2 \
PYTHONPATH=$PWD python fracturex/tests/aposteriori/run_m3_pc_model1.py
```
