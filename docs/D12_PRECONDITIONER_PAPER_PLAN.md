# D12：Hu–Zhang + 相场鞍点系统的块预条件分析（论文实验设计 + 骨架）

> 目标：以现有 [`fracturex/utilfuc/linear_solvers.py`](../fracturex/utilfuc/linear_solvers.py) 中已实现的块 ILU-GMRES / Schur 近似 / 辅助空间 GAMG 预条件器为对象，在 Hu–Zhang 应力-位移混合元 + 相场损伤的耦合系统上做**系统性谱分析与参数无关性实验**，3–5 个月内产出 1 篇方法-数值线性代数交叉的论文（目标：SISC / CMAME / NLAA）。

---

## 1. 研究问题（一句话）

> 给定 Hu–Zhang 混合元离散下的相场断裂耦合系统，弹性子问题在每个交错步呈现 **σ–u 鞍点结构 + 损伤依赖系数 g(d)**；本工作问：**哪种块预条件构造能保证 GMRES 迭代次数对 (h, l₀, load step, max d) 均匀（mesh- and parameter-independent）？**

具体可量化的子问题：

- Q1：`standard` 公式下 `weighted_aux` P1 GAMG 是否在 d→1 时保持 robust？
- Q2：`effective_stress` 公式下不加权的辅助空间 Schur 近似是否同样 robust？
- Q3：σ–u 鞍点的 Schur 补 `S(d_h) = B A(d_h)⁻¹ B^T` 的对角近似 `S_hat = B diag(A)⁻¹ B^T` 误差随 d 变化的谱估计；
- Q4：与块对角 ILU、块上三角 Schur 近似、纯 AMG 等"对照组"的工程对比。

---

## 2. 现状盘点（已有 / 待补）

### 已有（直接复用，无需重写）

| 函数 | 位置 | 角色 |
|------|------|------|
| `solve_huzhang_block_gmres` | [linear_solvers.py:668](../fracturex/utilfuc/linear_solvers.py#L668) | 块 ILU-GMRES baseline |
| `solve_huzhang_block_krylov` | [linear_solvers.py:737](../fracturex/utilfuc/linear_solvers.py#L737) | FEALPy gmres/minres 接口 |
| `solve_huzhang_block_gmres_fast` | [linear_solvers.py:907](../fracturex/utilfuc/linear_solvers.py#L907) | 优化版块预条件 |
| `solve_huzhang_block_gmres_auxspace` | [linear_solvers.py:1216](../fracturex/utilfuc/linear_solvers.py#L1216) | **核心**：aux-space + Schur 近似 |
| `_approximate_schur_spd` | [linear_solvers.py:305](../fracturex/utilfuc/linear_solvers.py#L305) | `S_hat = B diag(A)⁻¹ B^T` |
| `_make_coarse_diffusion_coef` | [linear_solvers.py:241](../fracturex/utilfuc/linear_solvers.py#L241) | g(d) 加权粗空间扩散 |
| `_estimate_lambda_max_dinv_s_numpy` | [linear_solvers.py:99](../fracturex/utilfuc/linear_solvers.py#L99) | 谱半径估计（已内建） |

跑论文实验的基础设施：
- 算例：[phasefield_model0_huzhang.py](../fracturex/tests/phasefield_model0_huzhang.py)、[phasefield_model2_notch_shear_huzhang.py](../fracturex/tests/phasefield_model2_notch_shear_huzhang.py)、[phasefield_square_tension.py](../fracturex/tests/phasefield_square_tension.py)
- 直接对比驱动：[scripts/paper_huzhang/run_aux_model0.sh](../scripts/paper_huzhang/run_aux_model0.sh)（aux）vs `run_direct.sh`（pardiso/mumps）
- 计时与记录：[`RunRecorder`](../fracturex/postprocess/recorder.py)、`StepInfo.meta` 已写入 `lin_iters`、`lin_resid`

### 需要新增（约 1–2 周工程量）

- `fracturex/tests/precond_spectrum.py`：对一次性弹性子问题导出**前 N 个广义特征值**（用 `scipy.sparse.linalg.eigsh` 在 `P⁻¹ K_h` 上跑 Lanczos）；
- `fracturex/tests/precond_sweep.py`：参数扫描入口（输入：算例、h、l₀、目标 d-max；输出：CSV 一行 = 一组参数 + 迭代次数 + 收敛标志）；
- `scripts/paper_precond/`：批跑脚本目录（仿 `paper_huzhang` 风格），含 `run_sweep.sh`、`collect_tables.py`、`make_figures.py`；
- 一个 **"frozen d-snapshot"** 工具：从已跑过的 staggered 仿真取若干 `(step, d-field)` 快照，反复求解同一弹性子问题以排除非线性外循环噪声。

---

## 3. 数学设置（论文 §2 写作用）

每个交错迭代外层固定 `d_h`，求解：

$$
K_h(d_h)\begin{bmatrix}\sigma\\ u\end{bmatrix} = \begin{bmatrix}f_\sigma\\ f_u\end{bmatrix},\quad K_h=\begin{bmatrix}A(d_h) & B^\top \\ B & 0\end{bmatrix}
$$

**两种 g(d) 放置**（与 `HuZhangElasticAssembler.formulation` 严格对应）：

- **standard**：`A(d) = ∫ (1/g(d)) σ:τ`（在应力块上）；`B` 与 d 无关
- **effective_stress**：`A` 与 d 无关；`B(d) = ∫ g(d) τ:ε(v)`（在耦合块上）

Schur 补与近似：

$$
S(d_h) = B A(d_h)^{-1} B^\top,\quad \widehat S = B\,\mathrm{diag}(A)^{-1} B^\top
$$

辅助空间路径（Chen et al. 2017 §5）：把 `S` 由 H¹ 上的（加权）向量 Poisson 近似，用 P1 GAMG 处理。
- `standard` + `weighted_aux=True`：P1 扩散系数 = `max(g(d), eps_g)`；
- `effective_stress`：P1 扩散系数 = 1（d 仅通过 Schur 块进入）。

**块上三角应用**（代码中 `gmres_preconditioner`）：

$$
\mathcal P^{-1}\begin{bmatrix}r_\sigma\\ r_u\end{bmatrix} = \begin{bmatrix}B_A(r_\sigma + B^\top B_S r_u)\\ -B_S r_u\end{bmatrix},\quad B_A\approx A^{-1},\ B_S\approx S^{-1}
$$

---

## 4. 实验矩阵

### 4.1 算法对照组（列）

| 标签 | 说明 | 代码入口 |
|------|------|---------|
| `direct` | SciPy `spsolve` | baseline 收敛标准 |
| `pardiso` | Intel MKL PARDISO | 时间对比基准 |
| `ilu_gmres` | 块 ILU-GMRES | `solve_huzhang_block_gmres` |
| `aux_unweighted` | aux-space，coarse 系数=1 | `solve_huzhang_block_gmres_auxspace(weighted_aux=False)` |
| `aux_weighted` | aux-space + g(d) 加权粗 | `weighted_aux=True`（仅 standard） |
| `aux_schur_ilu` | aux-space + Schur ILU | `schur_ilu_in_precond=True` |
| `minres_diag` | MINRES 对角预条件 | `solve_minres_diag`（理论对照） |

### 4.2 算例（行）

| 算例 | 物理 | 临破坏行为 |
|------|------|----------|
| Model0：圆孔板拉伸 | 拉伸主导 | 单一裂纹贯穿 |
| Square + precrack：单边切口拉伸 | I 型 | 直裂纹 |
| Model2：单边切口剪切 | II 型 + 局部拉 | 弯曲裂纹 |
| L-shape（可选）| 混合模式 | 多裂纹分支 |

### 4.3 参数扫描（每个 cell 一组实验）

- **网格 h**：4 级 ≈ {1/32, 1/64, 1/128, 1/256}（取 `h/l₀ ∈ {0.5, 0.25, 0.125, 0.0625}`，文献最常用比例）；
- **正则化 l₀**：3 级，覆盖典型范围（如 `2e-3, 1e-3, 5e-4`）；
- **退化下界 eps_g**：`{1e-3, 1e-6, 1e-9}`（探测 d→1 退化的鲁棒性）；
- **d-snapshot**：`max(d) ∈ {0.1, 0.5, 0.9, 0.99, 0.999}`（从 frozen-d 仿真取）；
- **公式**：standard / effective_stress 双路径。

总组合：`4 算法 × 3 算例 × 4 h × 3 l₀ × 3 eps_g × 5 d × 2 公式 ≈ 4300` 个数据点。
每点 30s–5min（取决于规模），总机时约 **2–4 天** 单机；可走 `paper_huzhang/background_batch.sh` 框架分布式跑。

### 4.4 度量指标

| 指标 | 提取方式 |
|------|---------|
| GMRES 迭代次数 `n_it` | `KrylovInfo.iters`（已有） |
| 是否收敛 `converged` | `KrylovInfo.converged` |
| 残差下降率 | `callback_residuals`（已有） |
| 预条件 setup 时间 | 包 `time.perf_counter` |
| 单次 GMRES 求解时间 | 同上 |
| **谱估计**：`P⁻¹ K_h` 前 20 个特征值（Lanczos）、`κ(P⁻¹ K_h)` | 新工具 `precond_spectrum.py` |
| 内存峰值（可选）| `tracemalloc` |

---

## 5. 理论部分（论文 §3 写作用）

### 5.1 必须给出的命题（按难度排序）

**Prop 1（一致性，容易）**：在 `standard` 公式下 `eps_g→0` 极限，`A(d)⁻¹` 在裂纹支撑集上无界；`weighted_aux` 的 P1 GAMG 通过 g(d) 加权使粗空间扩散在裂纹处退化，**与连续问题的退化方向一致**。给出形式化的等价性陈述。

**Prop 2（Schur 近似谱界，中等）**：证明
$$
c_1\,\widehat S \preceq S(d_h) \preceq c_2\,\widehat S
$$
其中 `c_1, c_2` 仅依赖于 Hu–Zhang 元的 inf-sup 常数与 g(d) 的最大/最小值之比 `r_g := max(g)/max(eps_g, min(g))`。**核心**：把 r_g 与 max d 关联，得到 `κ(P⁻¹ K_h) = O(r_g^α)` 的上界（α 待定，数值上拟合）。

**Prop 3（参数无关性，难——这是论文核心卖点）**：在 `weighted_aux` + 适当 `eps_g` 下，存在常数 C 使
$$
\kappa(\mathcal P^{-1} K_h(d_h)) \le C,\quad \forall h, l_0, d_h\in[0, 1-\delta]
$$
（δ 为退化截止）。给不出严格证明的话，**至少给出数值验证 + 启发式 argument**（CMAME 接受这一档；SISC 倾向需严格证明）。

### 5.2 与文献的差异化

- Chen et al. 2017（aux-space for elasticity）**没有相场退化**；
- Heister et al. 2015 / Farrell & Maurini 2017（monolithic phase-field 预条件）**用位移-Lagrange 元**，不处理 σ 主变量；
- 本工作把 (Hu–Zhang 元) × (相场 g(d)) × (aux-space) 三者合起来做的完整谱分析在文献中**没有直接对照**。

---

## 6. 论文骨架（目标 12–14 页双栏）

| 节 | 标题 | 内容 | 字数 / 图表 |
|----|------|------|------------|
| §1 | Introduction | 相场断裂 + 混合元背景；现有预条件文献综述；本文贡献 | 1.5 页 |
| §2 | Hu–Zhang 相场离散与耦合系统 | 引现有 [architecture doc](HUZHANG_PHASEFIELD_ARCHITECTURE.en.md) §2–3；公式 (1)–(8) | 2 页 |
| §3 | 块预条件构造 | Schur 近似、aux-space P1 GAMG、加权 vs 不加权两支 | 2 页，1 图（构造示意） |
| §4 | 谱分析 | Prop 1–3，定理与证明 | 2 页 |
| §5 | 数值实验 | §5.1 配置；§5.2 网格无关性（表 1）；§5.3 l₀ 无关性（表 2）；§5.4 d→1 鲁棒性（图 2：迭代数 vs max d）；§5.5 时间对比 vs PARDISO（表 3）；§5.6 谱图验证（图 3：特征值散点） | 4 页，3 图 4 表 |
| §6 | Discussion | 失败模式；扩展到 3D / monolithic 的指向；与 [A2 路线](#) 的衔接 | 1 页 |
| §7 | Conclusion | | 0.5 页 |
| App. A | 完整谱数据 | 补充材料 | — |

**关键图/表**（reviewer 看的就是这几个）：

- **表 1**：固定 l₀=1e-3、max d=0.9，4 种网格 × 4 个算法，列迭代次数。**期望**：`aux_weighted` 那一列基本不变。
- **表 2**：固定 h，3 个 l₀ × 4 算法。**期望**：l₀ 减小时 `ilu_gmres` 显著恶化，`aux_weighted` 仍平稳。
- **图 2**：x 轴 = max d ∈ [0, 1)，y 轴 = GMRES 迭代次数。3 条曲线（standard-aux_weighted、effective_stress-aux_unweighted、ilu_gmres）。**期望**：aux 系平、ilu 系翘起。
- **图 3**：`P⁻¹ K_h` 的前 20 个特征值散点（Lanczos 估计），不同 max d 下叠加。**期望**：aux 路径下谱聚集稳定。
- **表 3**：与 PARDISO 在中等规模（dof ≈ 1M）下的 wall-time 对比，给出**何时 iterative 胜出**的 dof 阈值。

---

## 7. 时间线（5 个月，按 2026 年 5 月起算）

| 月 | 里程碑 | 交付物 |
|----|--------|--------|
| **M1（5/26–6/22）** | 工具补齐 | `precond_spectrum.py`、`precond_sweep.py`、`scripts/paper_precond/` 跑通 Model0 一组小扫描 |
| **M2（6/23–7/20）** | 全矩阵扫描完成 | 所有 4300 组数据点 CSV、`collect_tables.py` 自动出表 1–3 |
| **M3（7/21–8/17）** | 谱分析 + 理论 | Prop 1–2 完整证明初稿；Prop 3 数值验证（若无证明） |
| **M4（8/18–9/14）** | 论文初稿 | §1–§7 全文 + 全部图表；内部审阅 |
| **M5（9/15–10/12）** | 修改 + 投稿 | 处理共同作者意见，arXiv preprint + 投 CMAME |

**buffer**：预留 4 周给审稿轮回。

---

## 8. 风险与对策

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| `aux_weighted` 实测**不**参数无关 | 中 | 高 | 这本身是论文卖点（"在 effective_stress 下需另设计"），转 framing 即可 |
| Prop 2/3 严格证明卡住 | 高 | 中 | 退到 CMAME（接受数值验证）；不冲 SISC |
| 谱估计 Lanczos 不收敛（K_h 非对称）| 中 | 中 | 改用 `P^{-1} K K^T P^{-T}` 估计或直接报 GMRES 残差曲线 |
| 实验机时不足 | 低 | 中 | `paper_huzhang/background_batch.sh` 已支持后台分批；或缩减扫描到 2 个算例 |
| Reviewer 质疑"为什么不直接 monolithic"| 高 | 低 | §6 中明确指出本文聚焦交错求解的**内层线性子问题**，monolithic 留 A2 后续工作 |

---

## 9. 与后续 A2 路线的衔接

D12 的产出（鲁棒 σ–u 块预条件器）**直接是 A2 Monolithic 求解器的零阶子模块**：
- A2 的 3 场 Jacobian `[[A_σσ, B^T, A_σd], [B, 0, 0], [A_dσ, 0, A_dd]]` 的 σ–u 块 = D12 研究对象；
- 论文 §6 写一段"how this preconditioner extends to monolithic"，为 A2 论文埋伏笔；
- 两篇互引（D12 论文先发，A2 论文晚 6 个月）。

---

## 10. 立即下一步（本周可启动）

1. 在 [fracturex/tests/](../fracturex/tests/) 新增 `precond_spectrum.py`，对单个 Model0 弹性子系统输出 `eigsh` 前 20 个特征值（基线工具）；
2. 写 `precond_sweep.py`：CLI 参数 `(case, h, l0, eps_g, max_d_target, formulation, algorithm)`，输出 CSV 单行；
3. 在 [scripts/](../scripts/) 新建 `paper_precond/run_sweep.sh`，仿 `paper_huzhang/run_all.sh` 的结构；
4. 用 Model0 + 4 网格 × 1 l₀ × 1 算法跑一次小扫描（约 1 小时），验证管线 + 出第一张 mesh-independence 草图。

完成 1–4 即可作为论文 §5.2 的第一张表的雏形。

---

## 11. 维护说明

- 本文件与代码同步：当 [linear_solvers.py](../fracturex/utilfuc/linear_solvers.py) 新增算法或重命名函数时，请更新 §2 表格；
- 实验脚本目录 `scripts/paper_precond/` 与 `scripts/paper_huzhang/` 共享 `env.sh` 解释器解析逻辑，复用 `_case_id.sh`；
- 论文成稿后，把投稿版 PDF 链接补到本文件首部。
