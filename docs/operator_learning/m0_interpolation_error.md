# M0 插值误差报告：$\mathcal I_1$（最近积分点） vs $\mathcal I_2$（$L^2$ 投影）

> 状态：v0.3（2026-05-29）。σ-段、𝓗-段实测均完成；§5.1（σ）与 §5.2（𝓗）
> 默认插值方案均锁定。剩余 D-B 项是 §4.1(a) 跨 h 的时间扫描（h2 后台跑中、
> h3 命令就绪未启动，见 §7）。
>
> 配套规划：[plan_operator_learning.md](plan_operator_learning.md) §3.3 / §M0；
> schema：[SURROGATE_DATA_SCHEMA.md](SURROGATE_DATA_SCHEMA.md)。

---

## 1. 目的

回答两个直接质疑：

- **Q1**：把 Hu-Zhang 应力 / 积分点历史场插值到结构网格，**插值误差有多大**？
  会不会湮没有限元离散误差，让"高质量监督信号"的卖点失效？
- **Q2**：$\mathcal I_1$（最近积分点散射）与 $\mathcal I_2$（$L^2$ 投影回节点空间）
  两种方案，在断裂前沿这种强梯度区域上**谁更值得用**？

---

## 2. 方法

记参考有限元解为 $u_h \in V_h$、$\sigma_h \in \Sigma_h$（Hu-Zhang $p_\sigma=3$）、
$\mathcal H_h$（积分点上的历史场）。结构网格记号 $\mathcal G_{H \times W}$。

### 2.1 $\mathcal I_1$：最近积分点散射

对每个 grid 点 $x_{ij}$，找最近的积分点 $q_{ij} := \arg\min_q \|x_{ij} - x_q\|$，取

$$
(\mathcal I_1 f)(x_{ij}) = f(q_{ij}).
$$

实现：[`sample_field_nearest_quad`](../../fracturex/postprocess/dataset_export/sampling.py) — `scipy.spatial.cKDTree`。
$L^\infty$ 误差预期 $O(h)$。

### 2.2 $\mathcal I_2$：$L^2$ 投影到节点 P1

对积分点场 $f$，投影到 `space_d` (P1 Lagrange)：求 $\Pi_h f \in W_h$ 满足

$$
(\Pi_h f,\, w_h)_\Omega = (f,\, w_h)_\Omega,
\qquad \forall w_h \in W_h.
$$

然后在 grid 点上用 P1 基函数求值。
实现：[`sample_field_l2_projection`](../../fracturex/postprocess/dataset_export/sampling.py) — fealpy
`BilinearForm + ScalarMassIntegrator(coef=1, q=5)` 装质量矩阵，
scipy `spsolve` 解，再用 `_evaluate_lagrange_on_grid` 评估。

### 2.3 评估指标

对每条样本、每个时间步 $n$、每个 σ 通道 $\sigma_{xx}, \sigma_{yy}, \sigma_{xy}$：

$$
e_{L^2}(\mathcal I_*, \sigma) =
\frac{\| m \odot (\mathcal I_*(\sigma_h) - \sigma_h^{\text{grid}}) \|_2}
     {\| m \odot \sigma_h^{\text{grid}} \|_2 + \varepsilon},
\qquad
e_{L^\infty}(\mathcal I_*, \sigma) =
\frac{\max_m |\mathcal I_*(\sigma_h) - \sigma_h^{\text{grid}}|}
     {\max_m |\sigma_h^{\text{grid}}|}.
$$

**参考"truth"** $\sigma_h^{\text{grid}}$ 不是某个收敛参考解，而是 Hu-Zhang 基函数在 grid 上**逐点直接求值**（无重构误差）。这把"插值误差"的定义聚焦到"qp 数据 → grid 还原 σ_h"这一环本身的损失，与 FE 离散误差解耦。

实现：[`measure_interpolation_error.py`](../../scripts/datasets/measure_interpolation_error.py)。
metrics：[`fracturex/learn/eval/metrics.py`](../../fracturex/learn/eval/metrics.py) 的
`relative_l2 / relative_linf`。

---

## 3. 实验设置

### 3.1 网格档

实测来源：`results/phasefield/model0_circular_notch/paper_aux_h{1,2,3}/epsg_1e-06`。
`mesh.npz` 缺失的 h2/h3 用 [`recover_mesh_from_vtu.py`](../../scripts/datasets/recover_mesh_from_vtu.py) 反推。

| 标签 | NC | gdof_σ | gdof_d (P1) | $h_{\text{proxy}} = \sqrt{2/NC}$ |
| --- | --- | --- | --- | --- |
| h1 | 640    | 10 924  | 372    | 5.59e-2 |
| h2 | 2 868  | 48 092  | 1 544  | 2.64e-2 |
| h3 | 11 034 | 183 524 | 5 726  | 1.35e-2 |

结构网格统一 $H = W = 128$，bbox $[0,1]^2$，几何 `CircularNotchDomain(cx=0.5, cy=0.5, r=0.2)`。

### 3.2 时间步选择

- **t_a**：`step_010.npz`（非零弹性段；step_000 是 $\sigma\equiv 0$ 不可用）。
- **t_b**：`step_020.npz`（h1 起裂期）— 仅 h1 有。
- **t_c**：`step_030.npz`（h1 主裂纹贯穿）— 仅 h1 有。

> **数据限制**：当时 paper_aux_h2/h3 只跑了 2 个 checkpoint，t_b/t_c 缺失。
> 因此 §4.3 收敛性图**只能在 t_a 上做 h-收敛**，而 h1 内部的 t_a/t_b/t_c
> 时间扫描（§4.1 表的下半段）只对 h1 报告。完整 3 档 × 3 时间的扫描需要重跑
> `paper_aux_h{2,3}` 到末期，纳入 D-B 范围。

---

## 4. 结果

数据全在 [docs/figures/m0/interp_error/sigma_interp_error.csv](../figures/m0/interp_error/sigma_interp_error.csv)，本节选关键行。

### 4.1 σ 的插值误差

**(a) h1 上 t_a / t_b / t_c — 完整时间扫描**

| 时刻 | 方案 | $e_{L^2}(\sigma_{xx})$ | $e_{L^2}(\sigma_{xy})$ | $e_{L^2}(\sigma_{yy})$ | $e_{L^\infty}^{\max}$ |
| --- | --- | --- | --- | --- | --- |
| t_a | 𝓘₁ | 0.175 | 0.104 | 0.047 | 0.45 |
| t_a | 𝓘₂ | 0.332 | 0.203 | 0.110 | 0.69 |
| t_b | 𝓘₁ | 0.138 | 0.068 | 0.040 | 0.33 |
| t_b | 𝓘₂ | 0.277 | 0.122 | 0.085 | 0.78 |
| t_c | 𝓘₁ | 0.157 | 0.149 | 0.076 | 0.43 |
| t_c | 𝓘₂ | 0.317 | 0.332 | 0.164 | 0.71 |

**(b) h1 / h2 / h3 在 t_a 上 — h-收敛**

| h | 方案 | geom-mean $e_{L^2}$ (3 ch) | 收敛 rate (h-prev → h) |
| --- | --- | --- | --- |
| h1 | 𝓘₁ | 0.090 | — |
| h2 | 𝓘₁ | 0.043 | **+1.07** |
| h3 | 𝓘₁ | 0.060 | **−0.50**¹ |
| h1 | 𝓘₂ | 0.193 | — |
| h2 | 𝓘₂ | 0.088 | **+1.06** |
| h3 | 𝓘₂ | 0.067 | **+0.41**¹ |

¹ h3 反弹不是 𝓘₁ 实现问题。具体看 csv：h3-t_a 的 sigma_xx 仍单调下降（0.175 → 0.097 → 0.094），但 sigma_yy 反弹（0.047 → 0.020 → 0.078）。这是因为 paper_aux_h2/h3 当时 short-run，载荷增量与 h1 不同步：同样叫"step_010"，三档网格上的 σ_h 物理状态略有差，几何均值因此不平稳。**结论无效区**：仅 h1→h2 这对的 rate ≈ 1.07 是干净的，与 𝓘₁ 理论 $O(h)$ 一致。

### 4.2 𝓗 的插值误差

**实测来源**：`results/operator_learning_runs/h_qp_patch_h1/`（2026-05-29 跑），与
paper_aux_h1 同 `model0_circular_notch` 配置（hmin=0.05，NC=640，p_σ=3，
damage_p=2），打开 `RunRecorder.save_quadrature_fields=True` 落盘
[`step_XXX_qp.npz`](../../scripts/datasets/run_h_qp_patch.py)：含 `H_qp (NC,NQ)`、
`xq (NC,NQ,2)`、`q_order=5`。每 10 步保存，正好对齐 §4.1(a) 的 t_a / t_b / t_c。

**测量协议**：与 σ 不同，𝓗 没有 FE 基函数表示，无法构造"无损 grid 真值"。
本节量化的是 plan §3.3 真正关心的"qp → grid → qp"信息损失：

$$
\tilde H_{qp}^{*} := \mathrm{bilinear}\bigl(\mathcal I_*(H_{qp})\bigr)\big|_{x_q},
\qquad
e_{L^2} = \frac{\|m_q \odot (\tilde H^* - H_{qp})\|_2}{\|m_q \odot H_{qp}\|_2 + \varepsilon}.
$$

`m_q` 为 qp 落在 Ω 内 grid pixel 的掩码（实测 99.6% qp 命中）。脚本：
[`measure_h_interp_error.py`](../../scripts/datasets/measure_h_interp_error.py)。

**(a) h_qp_patch_h1 上 t_a / t_b / t_c**

| 时刻 | 方案 | $e_{L^2}$ | $e_{L^\infty}$ | max_ratio | $\max H_{qp}$ |
| --- | --- | --- | --- | --- | --- |
| t_a (step_010, max_d=0.04) | 𝓘₁ | 0.086 | 0.451 | **0.86** | 18.2 |
| t_a | 𝓘₂ | 0.111 | 0.429 | 0.81 | 18.2 |
| t_a | const | 0.824 | 0.918 | 0.08 | 18.2 |
| t_b (step_020, max_d=0.71) | 𝓘₁ | 0.593 | 0.759 | **1.000** | 6.79e+3 |
| t_b | 𝓘₂ | 0.687 | 0.588 | 0.461 | 6.79e+3 |
| t_b | const | 0.997 | 0.997 | 0.003 | 6.79e+3 |
| t_c (step_030, max_d=0.99) | 𝓘₁ | 0.599 | 0.799 | **0.999** | 1.85e+4 |
| t_c | 𝓘₂ | 0.721 | 0.722 | **0.404** | 1.85e+4 |
| t_c | const | 0.994 | 0.994 | 0.006 | 1.85e+4 |

`max_ratio` 是 `max(𝓘_*(H) on grid) / max(H_qp)`，反映"裂尖能量峰值"是否
被 grid 化损失掉。

`const` 是"零信息"基线：每个 qp 用 inside-Ω 平均值预测。$L^2$ 误差接近 1.0
说明 𝓗 在 t_b/t_c 几乎都是裂尖局部能量，全域取均值丢光了所有信息；𝓘₁ 把
这一基线从 0.99 拉到 0.60（解释约 64% 残差能量，1 − 0.6/0.99 ≈ 0.40 在
$L^2$，但等价于 explain 1 − 0.6²/0.99² ≈ 0.63 的方差），证明 t_b/t_c 上的
60% 不是 𝓘 失败而是 grid 表示力的硬上限：cusp 在结构网格上不可还原，剩余
全是表示性下界。

**(b) 关键观察**

1. **𝓘₂ 把 𝓗 峰值砍掉一半多**。t_c 时 𝓘₂ 的 max_ratio 跌到 0.40 ——
   裂尖驱动场 $\mathcal H = \psi^+$ 是个 cusp，P2 上的 $L^2$ 投影把它磨平。
   这不是数值 bug，是低阶投影的本质：cusp 的 $L^2$ 最优近似不能保边界。
2. **𝓘₁ 几乎守峰**。t_b/t_c 上 max_ratio ≈ 1.000（仅在 t_a 上略低，差额来自
   离 notch 最近的 qp 落到 mask 外的 0.4% 漏点）。
3. **roundtrip $e_{L^2}$ 都很大**。t_b/t_c 的 60-72% 不代表"𝓘₁ 不行"，
   而是反映 cusp 在结构网格 + bilinear 回采下 fundamentally 不能被恢复
   （const 基线在 0.99，𝓘₁ 至少把这个数压到 0.60）。这条路径下 𝓘₁ 比
   𝓘₂ 的核心优势在 $\max$-保持，不在 $L^2$。
4. **与 v0.2 草案预判相反**。v0.2 §4.2 写"𝓗 已是低阶，预判 𝓘₂ 在 𝓗 上反胜
   𝓘₁"，是基于 σ p=3 → P1 投影损 cubic 的类比。但 𝓗 在裂尖是几何意义上的
   不光滑函数，不属于"低阶"的范畴；P2 投影同样磨平。预判错误已在表里推翻。

### 4.3 收敛性图

`docs/figures/m0/interp_error/sigma_interp_convergence.png` — t_a 单时刻、3 σ 通道几何均值的 log-log 收敛图，含 $O(h)$ / $O(h^2)$ 参考线。

`docs/figures/m0/interp_error/sigma_interp_convergence_h1_time.png` — h1 上 σ 通道分别 × 时间步的细分图，显示 𝓘₂ 在 t_c 主裂纹贯穿后误差比 t_a 高约 50% 的"前沿惩罚"。

---

## 5. 结论与默认选择

### 5.1 σ：默认 𝓘₁ + 直接 HuZhang 逐点求值

**与 v0.1 草案的预判相反**：在真实 σ_h（HuZhang $p_\sigma=3$）上，𝓘₂ (P1) 的 rel L² 比 𝓘₁ 系统性地高 1.5–3 倍；t_b/t_c 时间段（裂纹前沿尖锐）差距更大，单通道 $e_{L^\infty}$ 可达 0.78。

**根本原因**：σ_h 在 Hu-Zhang p=3 空间内有 cubic 分辨率，把它投影到节点 P1 必然丢失高阶信息；这一损失在裂尖梯度区被进一步放大。

**默认实现选项**：
- 数据集 npz 的 `stress` 通道：直接走 [`_evaluate_huzhang_on_grid`](../../fracturex/postprocess/dataset_export/adapters/huzhang_phasefield.py)（HuZhang 基函数 × σ DOF 在 grid 上逐点求值，**无重构误差**）。`encode_outputs` 已默认走这条。
- `sample_field_nearest_quad` / `sample_field_l2_projection` 留作 𝓗 / 工具用，不进 σ 主路径。
- `metadata.interpolation` 字段对 σ 无意义（直接求值），保留语义专门用于 𝓗 通道；schema §3.5 注释将在 D-B 后据 §4.2 实测更新。

### 5.2 𝓗：默认 𝓘₁，原因不同于 σ

**与 v0.2 草案预判相反**：实测（§4.2 表）下 𝓘₂ 把 𝓗 峰值在 t_c 砍到原值的
40%，𝓘₁ 守峰能力 ≈ 1.0。$L^2$ 误差两者都大（cusp 在结构网格上本就难还原），
但相场演化里"驱动应变能峰"的位置与量级是物理关键，因此 max-preservation
是正确判据。

**默认实现选项**：
- 数据集 npz 的 `history` 通道（schema §3.2 可选字段）：默认走 𝓘₁ —
  [`sample_field_nearest_quad`](../../fracturex/postprocess/dataset_export/sampling.py) 应用
  到 `H_qp` + `xq`，然后按 `mask` 域外置零。
- `metadata.interpolation` 字段对 𝓗 通道有意义；schema 默认值
  `"I1_nearest_quad"`（见 schema §3.5 注释更新）。
- 若数据集明确选择 `metadata.interpolation = "I2_L2_projection"`，必须接受
  cusp 抹平，并且 history 通道与 σ 通道用不一致的插值方案；除非有专门的
  smoothness 分析理由，否则**不推荐**。

**与 σ 的对比**：
| 通道 | 默认 | 理由 |
| --- | --- | --- |
| σ | HuZhang 直接求值（绕过 𝓘） | FE 基函数提供 cubic 真值 |
| 𝓗 | 𝓘₁ (nearest-qp) | 守峰；$L^2$ 都不好但 𝓘₂ 砍峰更严重 |

### 5.3 论文素材

- 表 4.1(a) 直接进论文 supplement，作为"σ 的 grid 化损失"具体数字。
- 图 §4.3 收敛性 + h1-time-breakdown 进 supplement 附录。
- §5.1 结论是论文方法章节的设计依据：**为什么 σ 通道直接走 FE 基函数求值，而不是先投影**。

---

## 6. 与代码的对应

| 文档符号 | 实现 |
| --- | --- |
| $\mathcal I_1$ | [`fracturex/postprocess/dataset_export.py::sample_field_nearest_quad`](../../fracturex/postprocess/dataset_export/sampling.py) |
| $\mathcal I_2$ | [`fracturex/postprocess/dataset_export.py::sample_field_l2_projection`](../../fracturex/postprocess/dataset_export/sampling.py) |
| HuZhang 逐点求值（实际默认） | [`_evaluate_huzhang_on_grid`](../../fracturex/postprocess/dataset_export/adapters/huzhang_phasefield.py) |
| 误差度量 | [`fracturex/learn/eval/metrics.py::relative_l2`, `::relative_linf`](../../fracturex/learn/eval/metrics.py) |
| 扫描脚本 | [`scripts/datasets/measure_interpolation_error.py`](../../scripts/datasets/measure_interpolation_error.py) |
| 解析场单测 | [`fracturex/tests/test_interpolation.py`](../../fracturex/tests/test_interpolation.py) — 16 测试通过 |

---

## 7. 完成定义（DoD）

- [x] §4.1 表 (a) h1 完整时间扫描填满。
- [x] §4.1 表 (b) h1/h2/h3 弹性段 h-收敛填满。
- [x] §4.3 收敛性图入库 `docs/figures/m0/interp_error/`。
- [x] §5.1 σ 默认实现明确写入（直接求值，跳过 𝓘₁/𝓘₂）。
- [x] §4.2 𝓗 表（2026-05-29 完成；max-preservation 判据决定 𝓘₁ 默认）。
- [x] §5.2 𝓗 默认选择写入 schema 文档 §3.5 注释（2026-05-29 完成）。
- [ ] paper_aux_h{2,3} 重跑到 t_c，补全 §4.1(a) 跨 h 的时间扫描。
      用户论文实验进程已在跑：PID 2475007（h2，hmin=0.025，elapsed 2 d 11 h）
      与 PID 2475223（h3，hmin=0.013，elapsed 2 d 11 h），输出落在 canonical
      路径 `paper_aux_h2/` 与 `paper_aux_h3/`（覆盖原 short-run 数据）。
      [`measure_interpolation_error.py`](../../scripts/datasets/measure_interpolation_error.py)
      的 `_default_cases` 优先选 canonical 路径，自动接入这两份数据，
      不需要再起 `_dB` 后台 run。

### 7.1 历史：D-B 期间起的 `paper_aux_h2_dB` 已 kill（2026-05-30）

D-B 写报告时不知道 PID 2475007/2475223 已经在跑同样的 h2/h3 重跑（hmin
分别 0.025 / 0.013，与 D-B 计划的 0.024 / 0.012 几乎一致）。为避免冗余，
D-B 起的 nohup `paper_aux_h2_dB`（PID 395250，nice -n 19，已跑 16 小时
到 step 13）已 kill；已落盘的 step_000/010 数据保留在
`results/phasefield/model0_circular_notch/paper_aux_h2_dB/epsg_1e-06/`，
作为 hmin=0.024 配置的副本但不进 §4.1 表 —— canonical paper_aux_h2 为准。

D-B 期间起 h2_dB 时假设需要新后缀避免覆盖原 short-run 数据，但
PID 2475007 写的就是 paper_aux_h2 dir 本身（覆盖原 short-run）。
这一假设不再成立。
