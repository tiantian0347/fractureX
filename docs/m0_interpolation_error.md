# M0 插值误差报告：$\mathcal I_1$（最近积分点） vs $\mathcal I_2$（$L^2$ 投影）

> 状态：v0.1 骨架（2026-05-27），数值结果待 M0 数据生成完毕后填入。
>
> 配套规划：[plan_operator_learning.md](plan_operator_learning.md) §3.3 / §M0
> 硬性交付物 2；schema：[SURROGATE_DATA_SCHEMA.md](SURROGATE_DATA_SCHEMA.md)。

---

## 1. 目的

回答审稿人最直接的两个质疑：

- **Q1**：把 Hu-Zhang 应力 / 积分点历史场插值到结构网格，**插值误差有多大**？
  会不会湮没有限元离散误差，让"高质量监督信号"的卖点失效？
- **Q2**：$\mathcal I_1$（最近积分点 散射）与 $\mathcal I_2$（$L^2$ 投影回节点空间）
  两种方案，在断裂前沿这种强梯度区域上**谁更值得用**？

这里把上述问题量化为可在 M0 数据生成阶段一次性测得的实验。

---

## 2. 方法

记参考有限元解为 $u_h \in V_h$、$\sigma_h \in \Sigma_h$、$\mathcal H_h$
（积分点上的历史场）。结构网格记号 $\mathcal G_{H \times W}$。

### 2.1 $\mathcal I_1$：最近积分点散射

对每个 grid 点 $x_{ij}$，找最近的积分点 $q_{ij} := \arg\min_q \|x_{ij} - x_q\|$，取

$$
(\mathcal I_1 \mathcal H_h)(x_{ij}) = \mathcal H_h(q_{ij}).
$$

特点：$O(NH^2)$ 一次最近邻搜索，便宜；$L^\infty$ 误差 $\sim O(h)$；
对 $\sigma_h$ 同样可做（用 Hu-Zhang 自由度逐点求值更准，但 $\mathcal I_1$ 对历史场用得更多）。

### 2.2 $\mathcal I_2$：$L^2$ 投影回节点空间

对历史场 $\mathcal H_h$（定义在积分点），先解一个小型质量矩阵问题，
投影到节点 Lagrange 空间 $W_h$：求 $\Pi_h \mathcal H \in W_h$ 满足

$$
(\Pi_h \mathcal H,\, w_h)_\Omega = (\mathcal H_h,\, w_h)_\Omega,
\qquad \forall w_h \in W_h.
$$

然后在 grid 点上用 Lagrange 基函数精确求值。

特点：保持 $L^2$ 阶 $O(h^{p+1})$；需要装 + 解一个 P1 / P2 mass matrix；
对前沿梯度更友好。

### 2.3 评估指标

对每条样本、每个时间步 $n$：

$$
e_{L^2}(\mathcal I_*, \sigma) =
\frac{\|\mathcal R_h \mathcal I_*(\sigma_h) - \sigma_h\|_{L^2(\Omega)}}
     {\|\sigma_h\|_{L^2(\Omega)}},
\qquad
e_{L^\infty}(\mathcal I_*, \sigma) =
\frac{\|\mathcal R_h \mathcal I_*(\sigma_h) - \sigma_h\|_{L^\infty(\Omega)}}
     {\|\sigma_h\|_{L^\infty(\Omega)}}.
$$

$\mathcal R_h$ 是 grid → 连续函数空间的分片线性 / 多项式重构（见 plan §3.3）。
同样的指标对 $\mathcal H$ 测一次。

**关键比较**：插值误差 $e_{L^2}$ 与 FE 离散误差 $\|\sigma_h - \sigma_{\text{ref}}\|_{L^2} / \|\sigma_{\text{ref}}\|_{L^2}$
（参考解从最细网格外插）。**目标：插值误差 $\leq 0.1 \times$ FE 离散误差**，
此时 grid 化不构成主导误差源。

---

## 3. 实验设置

### 3.1 测试网格

| 标签 | h | grid 分辨率 | 适用 |
| --- | --- | --- | --- |
| h0 | 较粗（参考前的初值） | $64 \times 64$ | S 档训练集 |
| h1 | 中等 | $128 \times 128$ | M 档训练集 |
| h2 | 较细 | $256 \times 256$ | 收敛性参考 |
| h3 | 最细（参考解） | $512 \times 512$ | 不归一化误差时的 "truth" |

> 网格尺寸 $h$ 与 FE mesh 的 $h_{\max}$ 同步缩放；具体数值待 model0 配置定。

### 3.2 测试时间步

- $t_a$：弹性段（无裂纹 / 微弱损伤）— 评估"平滑场"的插值误差。
- $t_b$：起裂期 — 前沿出现，梯度大。
- $t_c$：贯穿后 — 主裂纹形成，前沿尖锐。

每个时间步对每种插值方案，每个字段（$\sigma_{xx}, \sigma_{yy}, \sigma_{xy}, \mathcal H, d$）
分别报告 $e_{L^2}$ 与 $e_{L^\infty}$。

### 3.3 测试 case

至少覆盖：
- `model0_circular_notch`（光滑边界，notch 应力集中）；
- `square_tension_precrack`（precrack 端点处更尖锐）。

---

## 4. 结果（待 M0 数据生成完毕后填入）

### 4.1 $\sigma$ 的插值误差

| case | $h$ | 方案 | $t$ | $e_{L^2}(\sigma_{xx})$ | $e_{L^2}(\sigma_{yy})$ | $e_{L^2}(\sigma_{xy})$ | $e_{L^\infty}(\sigma_{xx})$ | FE 离散误差比 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model0 | h1 | $\mathcal I_1$ (Hu-Zhang 逐点) | $t_a$ | — | — | — | — | — |
| model0 | h1 | $\mathcal I_1$ | $t_b$ | — | — | — | — | — |
| model0 | h1 | $\mathcal I_1$ | $t_c$ | — | — | — | — | — |
| model0 | h1 | $\mathcal I_2$ | $t_a$ | — | — | — | — | — |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

### 4.2 $\mathcal H$ 的插值误差

| case | $h$ | 方案 | $t$ | $e_{L^2}(\mathcal H)$ | $e_{L^\infty}(\mathcal H)$ | FE 离散误差比 |
| --- | --- | --- | --- | --- | --- | --- |
| model0 | h1 | $\mathcal I_1$ | $t_a$ | — | — | — |
| model0 | h1 | $\mathcal I_2$ | $t_a$ | — | — | — |
| ... | ... | ... | ... | ... | ... | ... |

### 4.3 收敛性曲线

预期：$\mathcal I_2$ 在 $L^2$ 下表现 $O(h^{p+1})$，$\mathcal I_1$ 表现 $O(h)$。
图：log-log $h$ vs $e_{L^2}$，分两条曲线。**图未生成**。

---

## 5. 结论与默认选择（占位）

待数值结果出来后写入。预判：

- **默认选 $\mathcal I_2$**：投影成本一次性付出，换来更准的前沿；
- **大数据集**：若 $\mathcal I_2$ 的 mass solve 成本压不下来，对 $\sigma$ 用 Hu-Zhang
  逐点求值（无重构误差），对 $\mathcal H$ 用 $\mathcal I_2$；
- **审稿人答辩素材**：本节的 §4.1 / §4.2 表 + §4.3 图直接进论文 supplement。

---

## 6. 与代码的对应

| 文档符号 | 实现 |
| --- | --- |
| $\mathcal I_1$ | `fracturex/postprocess/dataset_export.py::sample_field_nearest_quad` |
| $\mathcal I_2$ | `fracturex/postprocess/dataset_export.py::sample_field_l2_projection` |
| $\mathcal R_h$ | `fracturex/learn/eval/reconstruct.py::piecewise_polynomial_reconstruct` |
| 误差度量 | `fracturex/learn/eval/metrics.py::relative_l2`, `::relative_linf` |
| 收敛性扫描脚本 | `scripts/datasets/measure_interpolation_error.py`（待写） |

---

## 7. 完成定义（DoD）

- [ ] §4.1 表 model0 全部填满（$\mathcal I_1$ / $\mathcal I_2$ × 3 个 $t$ × 4 个网格档）。
- [ ] §4.2 表 model0 + square_tension_precrack 至少一份。
- [ ] §4.3 收敛性图入库 `docs/figures/m0_interp_convergence.png`。
- [ ] §5 默认选择确定并写入 schema 文档 §3.5 注释。
- [ ] `tests/test_dataset_roundtrip.py` 的 §6 第 5 条不变量阈值由本报告设置。
