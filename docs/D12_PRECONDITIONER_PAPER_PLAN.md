# D12：Hu–Zhang + 相场鞍点系统的块预条件分析（论文实验设计 + 骨架）

> 目标：以现有 [`fracturex/utilfuc/linear_solvers.py`](../fracturex/utilfuc/linear_solvers.py) 中已实现的块 ILU-GMRES / Schur 近似 / 辅助空间 GAMG 预条件器为对象，在 Hu–Zhang 应力-位移混合元 + 相场损伤的耦合系统上做**系统性谱分析与参数无关性实验**，3–5 个月内产出 1 篇方法-数值线性代数交叉的论文（目标：SISC / CMAME / NLAA）。
>
> 配套活稿：`~/tian/Frac_huzhang/phasefield_huzhang.tex`（首选 CMAME / 备选 JCP、IJNME）。本文件是该论文的唯一权威规划文档：§1–§11 为设计/理论层，§12 为 Claim↔图表映射，§13 为易变运行状态追踪。所有实验跑完都来更新 §13 的"状态"列。
> （原 `docs/experiment_matrix.md` 已并入本文件并删除。）

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
- `standard` + `weighted_aux=True`：P1 扩散系数 = `g(d)`（取自 `damage.coef_bary`，与应力块 `1/g` 同源、**不另叠 `max(·, eps_g)` floor**；eps_g 仅在 `damage.coef_bary` 内部作下界，见 [`_make_coarse_diffusion_coef`](../fracturex/utilfuc/linear_solvers.py#L235) 及 §4.3 分层说明）；
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
| Square + precrack：单边切口拉伸（model1） | I 型 | 直裂纹 |
| Model2：单边切口剪切 | II 型 + 局部拉 | 弯曲裂纹 |
| L-shape（可选）| 混合模式 | 多裂纹分支 |

**物理参数速查**（改前先核对，避免误改；几何与文献对照见末列）：

| 参数 | model0 | model1 (square) | model2 |
|------|--------|-----------------|--------|
| E | 200 | 210 | 210 |
| ν | 0.2 | 0.3 | 0.3 |
| Gc | 1.0 | 2.7e-3 | 2.7e-3 |
| l₀ | 0.02 | 0.015 | 0.0133 |
| h_target = l₀/2 | 0.010 | 0.0075 | 0.0067 |
| 加载步数（默认）| 31 | 161 | 由 `case.default_loads()` 决定 |
| 几何 | distmesh 圆缺口（Miehe-type radial pre-notch）| box + 预裂纹（Miehe 2010 tension）| box + 缺口剪切（Miehe 2010 shear）|

### 4.3 参数扫描（每个 cell 一组实验）

> **eps_g 分层框架（刻意为之，勿合并两层措辞 — 防 drift）**：主结果（C1–C5，对应活稿 `phasefield_huzhang.tex`）**固定** eps_g，且 eps_g **不进入论文措辞**——它只作为 `damage.coef_bary` 的内部数值下界，应力块 `1/g` 与 P1 粗扩散 `g` 两侧同源（见 [`_make_coarse_diffusion_coef`](../fracturex/utilfuc/linear_solvers.py#L235)），不在辅助子里另叠 `max(g, eps_g)` floor。**只有 §5 的谱分析 / Prop 1–2** 才把 eps_g 当扫描轴，用于建立 `r_g → κ` 上界。两层数据隔离、不互相污染。

**主结果固定参数**（不在论文里乱扫描）：

- `formulation = "standard"`、`p = 3`（应力空间次数）、`damage_p = 2`
- `AT2 + quadratic degradation + hybrid split`
- eps_g：仅 `damage.coef_bary` 内部下界，固定、不进入论文措辞

**网格档定义**（model0 作为 N-scaling 主案例；distmesh，`hmin` 为控制参数）：

| 档位 | hmin | 预估 h_max | σ DOF（≈）| u DOF（≈）| d DOF（≈）| 用途 |
|------|------|-----------|-----------|-----------|-----------|------|
| h₁ | 0.05 | 7.8e-2 | 10K | 7K | 1.4K | 小算例热身 + 调流程 |
| h₂ | 0.025 | 4e-2 | 40K | 30K | 5K | 中等，direct 可承受 |
| h₃ | 0.013 | 2e-2 | 160K | 120K | 20K | direct 仍可，aux 显优势 |
| h₄ | 0.0065 | 1e-2 | 640K | 480K | 80K | direct 临界，aux 必须 |
| h₅ | 0.004 | 5e-3 | 1.9M | 1.4M | 230K | 招牌图压轴 |

**分辨率判据**：每档要求 `h_max < l₀/2 = 0.01`，从 **h₃** 起才严格满足相场分辨率；h₁、h₂ 用于显示求解器对 under-resolved 的鲁棒性。

**§5 谱分析消融轴**（仅用于谱界 / Prop 验证，与上面物理主结果隔离）：

- **正则化 l₀**：3 级（如 `2e-3, 1e-3, 5e-4`），用于 Prop 3 的 l₀-无关性验证（主结果按 §4.2 每算例固定 l₀，此处仅 model0 上扫）；
- **退化下界 eps_g**：`{1e-3, 1e-6, 1e-9}`（探测 d→1 退化鲁棒性；`r_g := max(g)/max(eps_g, min g)`）；
- **d-snapshot**：`max(d) ∈ {0.1, 0.5, 0.9, 0.99, 0.999}`（从 frozen-d 仿真取）；
- **公式**：standard / effective_stress 双路径（effective_stress 在主结果只 model0 补一次消融表）。

总组合（谱分析全矩阵）：`4 算法 × 3 算例 × 5 h × 3 l₀ × 3 eps_g × 5 d × 2 公式 ≈ 5400` 个数据点。
每点 30s–5min（取决于规模），总机时约 **2–4 天** 单机；可走 `paper_huzhang/background_batch.sh` 框架分布式跑。

### 4.4 度量指标

| 指标 | 提取方式 | 是否已记录 |
|------|---------|-----------|
| GMRES 迭代次数 `n_it` | `KrylovInfo.iters` | ✓ |
| 是否收敛 `converged` | `KrylovInfo.converged` | ✓ |
| 残差下降率 | `callback_residuals` | ✓ |
| 预条件 setup 时间 | 包 `time.perf_counter` | ✓ |
| 单次 GMRES 求解时间 | 同上 | ✓ |
| **谱估计**：`P⁻¹ K_h` 前 20 特征值（Lanczos）、`κ(P⁻¹ K_h)` | `precond_spectrum.py` | ✓（新工具） |
| 内存峰值 `peak_rss_mb`（per load step）| `psutil`，**C4 表依赖** | ❌ **需加 ~5 行 psutil** |
| `wall_time_elastic_solve_per_iter` | driver，per staggered iter | ❓ 需验证 |
| `wall_time_phase_solve_per_iter` | driver，per staggered iter | ❓ 需验证 |
| `wall_time_assembly_per_iter` | assembler，per staggered iter | ❓ 需验证 |
| `n_gmres_iter_elastic` / `n_gmres_iter_phase` | solver，per staggered iter | ❓ 需验证 |
| `max_d` / `err_u` / `err_d` / `reaction_force` | state / driver / postprocess | ✓ |

**P1 内必须把 `peak_rss_mb` 接入**（~5 行 psutil），否则 C4 效率表（§12）画不出。per-iter wall_time 拆分与 `n_gmres_iter_*` 上线前先验证 `RunRecorder` 是否已落盘。

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

> M1–M5 为主时间线；括号内并入活稿的 P1–P5 阶段关键节点（同一工作的细化粒度）。

| 月 | 里程碑 | 交付物 |
|----|--------|--------|
| **M1（5/26–6/22）/ P1 止血** | 工具补齐 + 接入 RSS | `precond_spectrum.py`、`precond_sweep.py`、`scripts/paper_precond/` 跑通 Model0 小扫描；完成 model0_aux（PID 3441015）；启动 square direct+aux；接入 `peak_rss_mb`；§13 状态表初版 |
| **M2（6/23–7/20）/ P2 招牌图** | 全矩阵扫描完成 | model0 h₁–h₅ 全部 direct+aux；C2 招牌图（iter-vs-N）初版；`collect_tables.py` 自动出表 1–3；草稿 intro + algorithm 章节 |
| **M3（7/21–8/17）/ P3 物理+C5** | 谱分析 + 理论 + C5 | Prop 1–2 完整证明初稿；Prop 3 数值验证；model1+model2 aux 完成；Lagrange 对照（model0/model1）跑完；C5 V2+V5 图；草稿 70% |
| **M4（8/18–9/14）/ P4 完稿** | 论文初稿 | C4 内存采样补完；§1–§7 全文 + 全部图表；内部 2 轮审阅 |
| **M5（9/15–10/12）/ P5 投稿** | 修改 + 投稿 | 处理共同作者意见，arXiv preprint + 投 CMAME |

**buffer**：预留 4 周给审稿轮回。

---

## 8. 风险与对策

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| `aux_weighted` 实测**不**参数无关（C2 iter-vs-N 曲线不平坦）| 中 | 高 | 这本身是论文卖点（"在 effective_stress 下需另设计"），转 framing 即可；降级写"实用预条件"而非"N-independent"，并补 condition number 估计实验 |
| Prop 2/3 严格证明卡住 | 高 | 中 | 退到 CMAME（接受数值验证）；不冲 SISC |
| 谱估计 Lanczos 不收敛（K_h 非对称）| 中 | 中 | 改用 `P^{-1} K K^T P^{-T}` 估计或直接报 GMRES 残差曲线 |
| 实验机时不足 / 6 个月内 P3 来不及 | 低-中 | 中 | `paper_huzhang/background_batch.sh` 已支持后台分批；或缩减到 2 算例；必要时砍 C5 V2 保留 V5，HuZhang vs Lagrange 移到 Discussion 定性论述 |
| Reviewer 质疑"为什么不直接 monolithic"| 高 | 低 | §6 中明确指出本文聚焦交错求解的**内层线性子问题**，monolithic 留 A2 后续工作 |
| direct（spsolve/pardiso）在 h₄/h₅ OOM 无法对比 | 高 | 中 | C4 图改成"direct OOM 临界 + aux 仍 work"对比，反而是亮点 |
| square（model1）direct 重启后仍不收敛 | 中 | 中 | 先加密载荷步（dt 0.5e-3→0.25e-3），再调 staggered tol 1e-5→1e-4 |
| Lagrange 对照（`MainSolve`）接口与 HuZhang 不对齐（C5 依赖）| 中 | 中 | P1 内做接口适配检查 |

---

## 9. 与后续 A2 路线的衔接

D12 的产出（鲁棒 σ–u 块预条件器）**直接是 A2 Monolithic 求解器的零阶子模块**：
- A2 的 3 场 Jacobian `[[A_σσ, B^T, A_σd], [B, 0, 0], [A_dσ, 0, A_dd]]` 的 σ–u 块 = D12 研究对象；
- 论文 §6 写一段"how this preconditioner extends to monolithic"，为 A2 论文埋伏笔；
- 两篇互引（D12 论文先发，A2 论文晚 6 个月）。

---

## 10. 立即下一步（本周可启动）

> 工具已落地：`precond_spectrum.py`、`precond_sweep.py`、`scripts/paper_precond/run_sweep.sh` 均已存在（见 §2）。本周重点转向跑通首个小扫描 + 接入 RSS。

1. `precond_spectrum.py`：对单个 Model0 弹性子系统输出 `eigsh` 前 20 个特征值（基线工具，已就位）；
2. `precond_sweep.py`：CLI `(case, h, l0, eps_g, max_d_target, formulation, algorithm)` → CSV 单行（已就位）；
3. `scripts/paper_precond/run_sweep.sh`：仿 `paper_huzhang/run_all.sh`（已就位）；
4. 用 Model0 + 5 网格档（h₁–h₅，见 §4.3）× 1 l₀ × 1 算法跑一次小扫描（约 1 小时），验证管线 + 出第一张 mesh-independence 草图；同步接入 `peak_rss_mb`（§4.4）。

完成 1–4 即可作为论文 §5.2 的第一张表（C2 招牌图，见 §12）的雏形。

---

## 11. 维护说明

- 本文件与代码同步：当 [linear_solvers.py](../fracturex/utilfuc/linear_solvers.py) 新增算法或重命名函数时，请更新 §2 表格；
- 实验脚本目录 `scripts/paper_precond/` 与 `scripts/paper_huzhang/` 共享 `env.sh` 解释器解析逻辑，复用 `_case_id.sh`；
- **运行状态只更新 §13**（易变层）；§1–§11 为设计/理论层，其章节号被 `D13_LEARNED_PRECONDITIONER_PAPER_PLAN.md`、`scripts/paper_precond/run_sweep.sh`、`fracturex/tests/precond_sweep.py`、`docs/readme.md` 按号引用，**重排前先 grep 这些引用**；
- 论文成稿后，把投稿版 PDF 链接补到本文件首部。

---

## 12. 论文核心 Claim 与图表映射

> 对应活稿 `~/tian/Frac_huzhang/phasefield_huzhang.tex`。所有主结果按 §4.3「主结果固定参数」跑（eps_g 固定、不进入论文措辞）。

| ID | Claim | 怎么证明 | 主要图表 |
|----|-------|----------|----------|
| **C1** | aux-precond 解与 direct 法物理一致、与文献吻合 | 三 case load-displacement 曲线重合；裂纹路径定性一致；model1 对照 Miehe(2010)/Borden(2012) | Fig. load-disp ×3；Fig. crack pattern ×3 |
| **C2** | aux-space 预条件子让 GMRES 迭代数**几乎与 N 无关**（核心招牌图）| model0 五档网格（h₁–h₅）扫描下 GMRES iter vs N 平坦曲线 | **Fig. iter-vs-N（论文核心图）** |
| **C3** | 整个损伤演化过程预条件子稳定（弹性段→起裂→软化→贯穿）| 三 case 各自 GMRES iter 随载荷步曲线（含 staggered iter 热力图）| Fig. iter-vs-step ×3 |
| **C4** | 相比 direct（spsolve/pardiso）有实用时间/内存优势；direct 大 N OOM 时 aux 仍可用 | model0 五档下 wall-time 与 peak RSS 对比表 + scaling 曲线 | Table efficiency；Fig. time-vs-N |
| **C5** | Hu-Zhang 元相对位移型 Lagrange 元在断裂问题上有结构性优势 | **V2**：弹性段 σ 的 h-收敛阶高于 Lagrange；**V5**：裂纹尖端 σ 探针对比 | Fig. σ-conv（V2）；Fig. tip-stress（V5）|

**C5 当前 `.tex` 初稿完全没有体现**，需新增数值实验小节（HuZhang vs Lagrange 对照，Lagrange 走 `MainSolve`，接口对齐见 §8 风险行）。

> 注：§5 的谱图（Prop 1–3 验证、图 3 特征值散点、κ 估计）属理论层，用 §4.3 的 eps_g/frozen-d 消融轴，**不**计入上面五个面向 `.tex` 的主 Claim。

---

## 13. 运行状态追踪（易变层 — 跑完即更新）

> 本节是唯一允许频繁改动的层；§1–§11 设计层勿动。状态图例：✅ done / 🟡 in progress / 🔴 受阻 / ⬜ todo。

### 13.1 实验主表

| 案例 | 模式 | 网格档 | 对应 Claim | 优先级 | 状态 |
|------|------|-------|-----------|--------|------|
| **model0**（circular notch）| baseline | 极细，1.9M σ DOF | C1 / C5 V2 | P0 | ✅ done (`paper_baseline/`) |
| model0 | direct | h₁ ~7.8e-2 | C1+C2+C4 | P0 | ✅ done (`paper_direct/`) |
| model0 | aux | h₁ ~7.8e-2 | C1+C2+C3+C4 | P0 | 🟡 in progress (PID 3441015) |
| model0 | direct | h₂ ~4e-2 | C2+C4 | P0 | ⬜ todo |
| model0 | aux | h₂ ~4e-2 | C2+C4 | P0 | ⬜ todo |
| model0 | direct | h₃ ~2e-2 | C2+C4 | P0 | ⬜ todo |
| model0 | aux | h₃ ~2e-2 | C2+C4 | P0 | ⬜ todo |
| model0 | direct | h₄ ~1e-2 | C2+C4（direct 临界）| P1 | ⬜ todo |
| model0 | aux | h₄ ~1e-2 | C2+C4 | P0 | ⬜ todo |
| model0 | aux | h₅ ~5e-3 | C2（aux 高 N 验证）| P0 | ⬜ todo |
| model0 | direct | h₅ ~5e-3 | C4（看是否 OOM）| P2 | ⬜ todo |
| **model0 Lagrange 对照** | — | h₂ | C5 V2（σ 收敛阶）| P1 | ⬜ todo |
| **model0 Lagrange 对照** | — | h₃ | C5 V2 | P1 | ⬜ todo |
| **model1 / square** | direct | h_target≈l₀/2 | C1（含 Miehe 对照）| P0 | 🔴 之前没收敛，重启 maxit=500 |
| model1 / square | aux | 同上 | C1+C3 | P0 | ⬜ todo |
| model1 Lagrange 对照 | — | 同上 | C5 V5（尖端 σ 探针）| P1 | ⬜ todo |
| **model2**（notch shear）| direct | h_target≈l₀/2 | C1+C3（剪切场景）| P0 | ✅ done |
| model2 | aux | 同上 | C3（剪切场景预条件子稳定性）| P0 | ⬜ todo |

**优先级**：P0 论文必须有；P1 应该有，缺了会被审稿人质疑；P2 补充材料/future work。

### 13.2 数据后处理脚本清单

| 脚本 | 用途 | 状态 |
|------|------|------|
| `collect_paper_bundle.py` | 汇总三 case × 三模式 metadata 到 PAPER_INDEX | ✓ 已有 |
| `paper_make_load_disp.py` | C1 三 case load-displacement 曲线对比图 | ❌ 待编写 |
| `paper_make_iter_vs_N.py` | C2 核心图：GMRES iter vs N（aux/direct 两条曲线）| ❌ 待编写 |
| `paper_make_iter_heatmap.py` | C3 热力图：staggered iter × GMRES iter × load step | ❌ 待编写 |
| `paper_make_efficiency_table.py` | C4 效率表：wall-time、RSS、scaling | ❌ 待编写 |
| `paper_make_sigma_conv.py` | C5 V2：σ 收敛阶（HuZhang vs Lagrange）| ❌ 待编写 |
| `paper_make_tip_stress.py` | C5 V5：裂纹尖端 σ 探针对比 | ❌ 待编写 |
