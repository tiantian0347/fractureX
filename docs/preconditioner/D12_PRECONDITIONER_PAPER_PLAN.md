# D12：Hu–Zhang + 相场鞍点系统的块预条件分析（论文实验设计 + 骨架）

> 目标：以现有 [`fracturex/utilfuc/linear_solvers.py`](../../fracturex/utilfuc/linear_solvers.py) 中已实现的块 ILU-GMRES / Schur 近似 / 辅助空间 GAMG 预条件器为对象，在 Hu–Zhang 应力-位移混合元 + 相场损伤的耦合系统上做**系统性谱分析与参数无关性实验**，3–5 个月内产出 1 篇方法-数值线性代数交叉的论文（目标：SISC / CMAME / NLAA）。
>
> 配套活稿：`~/tian/Frac_huzhang/phasefield_huzhang.tex`（首选 CMAME / 备选 JCP、IJNME）。本文件是该论文的唯一权威规划文档：§1–§11 为设计/理论层，§12 为 Claim↔图表映射，§13 为易变运行状态追踪。所有实验跑完都来更新 §13 的"状态"列。
> （原 `docs/experiment_matrix.md` 已并入本文件并删除。）
>
> **📄 论文就绪结果见 [`D12_RESULTS.md`](D12_RESULTS.md)**：§5 全部表/图/结论按论文小节组织，可直接转写 TeX。本文件（计划/理论/状态）与之互补。

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
| `solve_huzhang_block_gmres` | [linear_solvers.py:668](../../fracturex/utilfuc/linear_solvers.py#L668) | 块 ILU-GMRES baseline |
| `solve_huzhang_block_krylov` | [linear_solvers.py:737](../../fracturex/utilfuc/linear_solvers.py#L737) | FEALPy gmres/minres 接口 |
| `solve_huzhang_block_gmres_fast` | [linear_solvers.py:907](../../fracturex/utilfuc/linear_solvers.py#L907) | 优化版块预条件 |
| `solve_huzhang_block_gmres_auxspace` | [linear_solvers.py:1216](../../fracturex/utilfuc/linear_solvers.py#L1216) | **核心**：aux-space + Schur 近似 |
| `_approximate_schur_spd` | [linear_solvers.py:305](../../fracturex/utilfuc/linear_solvers.py#L305) | `S_hat = B diag(A)⁻¹ B^T` |
| `_make_coarse_diffusion_coef` | [linear_solvers.py:241](../../fracturex/utilfuc/linear_solvers.py#L241) | g(d) 加权粗空间扩散 |
| `_estimate_lambda_max_dinv_s_numpy` | [linear_solvers.py:99](../../fracturex/utilfuc/linear_solvers.py#L99) | 谱半径估计（已内建） |

跑论文实验的基础设施：
- 算例：[phasefield_model0_huzhang.py](../../fracturex/tests/phasefield_model0_huzhang.py)、[phasefield_model2_notch_shear_huzhang.py](../../fracturex/tests/phasefield_model2_notch_shear_huzhang.py)、[phasefield_model1_square_tension.py](../../fracturex/tests/phasefield_model1_square_tension.py)
- 直接对比驱动：[scripts/paper_huzhang/run_aux_model0.sh](../../scripts/paper_huzhang/run_aux_model0.sh)（aux）vs `run_direct.sh`（pardiso/mumps）
- 计时与记录：[`RunRecorder`](../../fracturex/postprocess/recorder.py)、`StepInfo.meta` 已写入 `lin_iters`、`lin_resid`

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
- `standard` + `weighted_aux=True`：P1 扩散系数 = `g(d)`（取自 `damage.coef_bary`，与应力块 `1/g` 同源、**不另叠 `max(·, eps_g)` floor**；eps_g 仅在 `damage.coef_bary` 内部作下界，见 [`_make_coarse_diffusion_coef`](../../fracturex/utilfuc/linear_solvers.py#L235) 及 §4.3 分层说明）；
- `effective_stress`：P1 扩散系数 = 1（d 仅通过 Schur 块进入）。

**块上三角应用**（代码中 `gmres_preconditioner`）：

$$
\mathcal P^{-1}\begin{bmatrix}r_\sigma\\ r_u\end{bmatrix} = \begin{bmatrix}B_A(r_\sigma + B^\top B_S r_u)\\ -B_S r_u\end{bmatrix},\quad B_A\approx A^{-1},\ B_S\approx S^{-1}
$$

### 3.1 Schur 预条件子 $B_S\approx S^{-1}$ 的两种实现（auxspace 朴素圈 vs fast 对称 V-cycle）

$B_A$ 取 $\mathrm{diag}(A)^{-1}$（$A$ 为应力质量型，对角即良好近似）。$B_S$ 是性能关键。本工作两种实现 **数学骨架相同**（同一 $\widehat S=B\,\mathrm{diag}(A)^{-1}B^\top$、同一 P1 向量 Poisson 辅助空间），**只差"光滑 + 粗校正"的组合圈**：

**(a) `auxspace`（朴素圈）** — [`solve_huzhang_block_gmres_auxspace`](../../fracturex/utilfuc/linear_solvers.py)
前向 Gauss–Seidel 光滑 + 一次 P1 粗校正（+ 可选 Schur ILU）；单边、非对称、无回扫。作为对照 / ablation。

**(b) `fast`（对称两层 V-cycle）** — [`solve_huzhang_block_gmres_fast`](../../fracturex/utilfuc/linear_solvers.py)
$B_S$ 是 $\widehat S$ 上一个对称 V$(\nu_1,\nu_2)$-cycle：

1. $e \leftarrow \mathrm{GS}_{\text{fwd}}(\widehat S,\, r;\ \nu_1)$（前向光滑，吃高频误差）
2. $r_2 \leftarrow r - \widehat S\,e$（光滑后残差）
3. **粗校正**：$e \mathrel{+}= \Pi\,\big[\mathrm{AMG}_V\big]^{-1}\,\Pi^\top r_2$ —— 残差经插值 $\Pi^\top$ 限制到 P1 向量 Poisson 辅助空间，pyamg 一个 V-cycle 求解，$\Pi$ 延拓回（吃低频/光滑误差）
4. $e \leftarrow \mathrm{GS}_{\text{bwd}}(\widehat S,\, e,\, r;\ \nu_2)$（**后向**光滑）

前向GS → 粗校正 → 后向GS 构成**对称、近似 SPD** 的预条件子，谱更紧 → GMRES 迭代数显著下降。本质是 **(几何) Hu–Zhang 应力迹空间 →（代数）P1 Poisson 的两层辅助空间方法**（Hiptmair–Xu / Chen et al. 2017）。

**按位移阶 $p_u$ 自适应**（`schur_precond="auto"`）：
- $p_u<6$：`gs_amg_gs`（上式，默认低阶）；
- $p_u\ge6$：`coarse_amg_halfgs`（先粗校正 + 半步对称 GS，高阶更稳）；
- 可选 `cheb_amg_cheb`：幂法估 $\lambda_{\max}(\mathrm{diag}(\widehat S)^{-1}\widehat S)$ 后用 Chebyshev 多项式光滑子替代 GS。

每次调用按当前 $A$ 重建 $M,B,D^{-1},\widehat S$（变系数）；P1 插值 $\Pi$ 与 mesh Laplacian 的 pyamg 层级按网格缓存。

> **实证（2026-06-02 受控扫描，model0 起裂前弹性区）**：同一 aux 方法，`auxspace` 在 square 需 `niter_elastic≈29`，`fast` 仅 `≈6`（与历史 88→6 优化一致）；h₁ 上 `fast` 单解墙钟已与 pardiso 持平（FAST=0 当初慢 6–35×）。**结论：§7.3 的 aux 主列取 `fast`，`auxspace` 作 ablation。**（受控扫描脚本 `scripts/paper_huzhang/run_scan_level.sh`：统一 pardiso baseline + `fast` + 每步 checkpoint/VTU + 起裂前区。）

### 3.2 `fast` 预条件子的数学理论（论文 §3–§4 写作用；与代码同步）

> 本节给出 `fast`（对称两层辅助空间 V-cycle）作为 $B_S\approx \widehat S^{-1}$ 的算子级定义、谱等价界、以及由此得到的 GMRES 迭代次数无关性。理论符号与实现一一对应，**改代码须同步本节**（见 §3.2.5 映射表与 §11 维护规则）。

**3.2.1 块预条件子（与 `fast_preconditioner` 一致）**
弹性鞍点 $K=\begin{psmallmatrix}A & B^\top\\ B & 0\end{psmallmatrix}$，$A=A(d_h)\succ0$（应力质量型，系数 $1/g(d)$），$B$ 离散散度。取 $D=\mathrm{diag}(A)$，近似 Schur $\widehat S = B D^{-1} B^\top\succ0$。代码 `fast_preconditioner` 实现的是非精确块 LU 预条件子
$$\mathcal P^{-1}=\begin{psmallmatrix}I & -D^{-1}B^\top\\ 0 & I\end{psmallmatrix}\begin{psmallmatrix}D^{-1} & 0\\ 0 & B_S\end{psmallmatrix}\begin{psmallmatrix}I & 0\\ -B D^{-1} & I\end{psmallmatrix}\begin{psmallmatrix}I&0\\0&-I\end{psmallmatrix},$$
其中 $B_S\approx\widehat S^{-1}$ 是核心（$D^{-1}$ 廉价且谱等价 $A^{-1}$，见 3.2.3）。

**3.2.2 $B_S$ 的两层对称 V-cycle（与 `pre_of_S`=`gs_amg_gs` 一致）**
设 $V_h$ 为 $\widehat S$ 所在的应力迹空间，$M$ 为对称 Gauss–Seidel 光滑子（$\nu_1$ 前向 + $\nu_2=\nu_1$ 后向，对应 `gs_iterations`），辅助空间 $W_h=$ P1 向量空间、迁移算子 $\Pi:W_h\!\to\!V_h$（代码 `PI_s`），粗算子 $A_H=$ P1（$g(d)$ 加权）向量 Poisson（代码 `_make_coarse_diffusion_coef`+pyamg）。$B_S$ 的**误差传播算子**
$$I-B_S\widehat S=(I-M^{-\top}\widehat S)\,(I-\Pi A_H^{-1}\Pi^\top\widehat S)\,(I-M^{-1}\widehat S),$$
即前向光滑→粗校正→后向光滑，$B_S$ 对称半正定。（`coarse_amg_halfgs`、`cheb_amg_cheb` 只改 $M$ 的多项式形式，不改本结构；pyamg 的一次 V-cycle 是 $A_H^{-1}$ 的有界内近似，引入与 $h$ 无关的常数因子。）

**3.2.3 谱等价（辅助空间 / 虚拟空间引理，Hiptmair–Xu 2007）**
若 (i) 光滑性 $\|v\|_{\widehat S}^2\le\omega\langle Mv,v\rangle$；(ii) 迁移稳定 $\|\Pi w\|_{\widehat S}\le c_\Pi\|w\|_{A_H}$；(iii) 稳定分解 $\forall v\in V_h,\ \exists v=v_s+\Pi w$ s.t. $\|v_s\|_M^2+\|w\|_{A_H}^2\le c_0^2\|v\|_{\widehat S}^2$，则
$$\kappa(B_S\widehat S)\le C(\omega,c_\Pi,c_0),\quad\text{与 }h\text{ 无关}.$$
**相场关键**：粗算子 $A_H$ 用 $g(d)$ 加权（与应力块 $1/g(d)$ 同源），使 (iii) 的 $c_0$ 在 $d\!\to\!1$ 时仍有界（即 Prop 1 的退化方向一致性）→ 得 Prop 3 的参数无关性。$D^{-1}$：$A$ 为质量型，拟一致网格下 $\mathrm{diag}(A)$ 谱等价 $A$，常数仅依赖单元形状。

**3.2.4 全块预条件子的迭代界**
$B_S$ 对 $\widehat S$ 一致谱等价 + $D$ 对 $A$ 谱等价 + 离散 inf-sup（$\widehat S$ 谱等价真 Schur $S=BA^{-1}B^\top$，常数依赖 $r_g:=\max g/\max(\epsilon_g,\min g)$，见 Prop 2）⟹ $\mathcal P^{-1}K$ 的特征值聚集在与 $(h,l_0,d_h{\in}[0,1{-}\delta])$ 无关的区间 ⟹ **GMRES 迭代数有界**（C2/C3/图2/图3 的理论依据；实测 niter 6–10，§13.4）。

**3.2.5 matrix-free 实现与理论一致性（不改谱，只改实现）**
- $A(d)$ 的作用 $v\mapsto A(d)v$（及 model0 本质边界 $T A T+T_{bd}$）逐单元施加、不组装 $A$（`matfree_elastic.py`，数学 = `HuZhangStressIntegrator` 同一收缩）；$B,\widehat S,\Pi,A_H$ 仍组装（小）。**算子谱不变 ⟹ 3.2.1–3.2.4 全部不变**，仅峰值内存降 1.9×/7.8×（§13.4 B）。
- **预条件子近似的合法性**：$D=\mathrm{diag}(M_2)$ 取近似 $(\mathrm{TM}\!\circ\!\mathrm{TM})^\top\mathrm{diag}(M)$、$B_S$ 跨多个交错步复用（`precond_rebuild_interval`）——二者只进 $\mathcal P$，GMRES 仍解精确 $K$，**只影响收敛常数、不改解**（实测 niter 不变、解机器精度一致）。

**3.2.6 理论 ↔ 代码符号映射（改代码同步本表）**

| 理论 | 代码（`linear_solvers.py` / `matfree_elastic.py`） |
|---|---|
| $A(d)$ 作用 / $D=\mathrm{diag}(A)$ | `MatrixFreeElasticOperator._M_action` / `.diag_inv_sigma` |
| $\widehat S=BD^{-1}B^\top$ | `_fast_cached_schur` → `_approximate_schur_spd` |
| 光滑子 $M$（SGS, $\nu_1$） | `pre_of_S` 内 `gauss_seidel(...,iterations=gs_iterations)` |
| 迁移 $\Pi$ / 粗算子 $A_H$ | `PI_s` / `_make_coarse_diffusion_coef`+pyamg V-cycle |
| $\mathcal P^{-1}$ 块 LU | `fast_preconditioner` |
| 模式 `gs_amg_gs/coarse_amg_halfgs/cheb_amg_cheb` | `schur_precond`（按 $p_u$ 自适应）|
| $B_S$ 复用 / 近似 diag | `precond_rebuild_interval` / 3.2.5 |
| 暴露 $\mathcal P^{-1}$ 算子（谱分析用） | `solve_huzhang_block_gmres_fast(..., return_preconditioner=True)` → `(P, None)` |

---

## 4. 实验矩阵

### 4.1 算法对照组（列）

| 标签 | 说明 | 代码入口 |
|------|------|---------|
| `direct` | SciPy `spsolve` | baseline 收敛标准 |
| `pardiso` | Intel MKL PARDISO | 时间对比基准 |
| `ilu_gmres` | 块 ILU-GMRES | `solve_huzhang_block_gmres` |
| `aux_fast` | **aux-space + 对称 V-cycle Schur（论文主 aux 列，niter≈6）** | `solve_huzhang_block_gmres_fast`（FAST=1；见 §3.1） |
| `aux_unweighted` | aux-space 朴素圈，coarse 系数=1 | `solve_huzhang_block_gmres_auxspace(weighted_aux=False)` |
| `aux_weighted` | aux-space 朴素圈 + g(d) 加权粗 | `weighted_aux=True`（仅 standard） |
| `aux_schur_ilu` | aux-space 朴素圈 + Schur ILU | `schur_ilu_in_precond=True` |
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

> **eps_g 分层框架（刻意为之，勿合并两层措辞 — 防 drift）**：主结果（C1–C5，对应活稿 `phasefield_huzhang.tex`）**固定** eps_g，且 eps_g **不进入论文措辞**——它只作为 `damage.coef_bary` 的内部数值下界，应力块 `1/g` 与 P1 粗扩散 `g` 两侧同源（见 [`_make_coarse_diffusion_coef`](../../fracturex/utilfuc/linear_solvers.py#L235)），不在辅助子里另叠 `max(g, eps_g)` floor。**只有 §5 的谱分析 / Prop 1–2** 才把 eps_g 当扫描轴，用于建立 `r_g → κ` 上界。两层数据隔离、不互相污染。

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
| §2 | Hu–Zhang 相场离散与耦合系统 | 引现有 [architecture doc](../architecture/huzhang_phasefield_architecture.en.md) §2–3；公式 (1)–(8) | 2 页 |
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
| square（model1）direct 用 scipy SuperLU 段错误（exit 139, core dumped）| 高 | 低 | 已定位 2026-05-29：设 `FRACTUREX_ELASTIC_DIRECT_BACKEND=pardiso`（pypardiso 0.4.7 已装）；与内存无关，短规模也崩，仅 square 鞍点系统触发 |
| square（model1）direct 重启后仍不收敛（step 67 物理失效）| 中 | 中 | 先加密载荷步（dt 0.5e-3→0.25e-3），再调 staggered tol 1e-5→1e-4；pardiso 救不了物理不收敛 |
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

- 本文件与代码同步：当 [linear_solvers.py](../../fracturex/utilfuc/linear_solvers.py) 新增算法或重命名函数时，请更新 §2 表格；
- **理论与代码同步（硬规则）**：§3.2 是 `fast` 预条件子的权威数学理论，与 [`linear_solvers.py`](../../fracturex/utilfuc/linear_solvers.py)（`solve_huzhang_block_gmres_fast`/`pre_of_S`/`fast_preconditioner`/`_fast_cached_schur`）+ [`matfree_elastic.py`](../../fracturex/utilfuc/matfree_elastic.py) **一一对应**。**任何求解器优化**（换光滑子、改 Schur 近似、调 `precond_rebuild_interval`、matrix-free 变体、新增 `schur_precond` 模式）**必须同步更新 §3.2 的算子定义/谱论证 + §3.2.6 符号映射表**；若优化改变了 $M$/$\Pi$/$A_H$/$\widehat S$ 的数学形式，须复核 3.2.3 的谱等价前提 (i)-(iii) 是否仍成立。改完在 §13 记一行变更。
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
| model0 | aux | h₁ ~7.8e-2 | C1+C2+C3+C4 | P0 | ✅ done (`paper_aux_h1/`，32 行跑满；PID 3441015 早已退出) |
| model0 | direct | h₂ ~4e-2 | C2+C4 | P0 | ✅ done (`paper_direct_h2/`，32 行) |
| model0 | aux | h₂ ~4e-2 | C2+C4 | P0 | 🟡 in progress (PID 2475007；2026-05-30 在 step 16/31，**step16 交错振荡不收敛** error≈0.18–0.23) |
| model0 | direct | h₃ ~2e-2 | C2+C4 | P0 | ✅ done (`paper_direct_h3/`，32 行) |
| model0 | aux | h₃ ~2e-2 | C2+C4 | P0 | 🟡 in progress (PID 2475223；2026-05-30 仅到 step 14/31，**交错收敛极慢** 130+ iter/步) |
| model0 | direct | h₄ ~1e-2 | C2+C4（direct 临界）| P1 | ⬜ todo |
| model0 | aux | h₄ ~1e-2 | C2+C4 | P0 | ⬜ todo |
| model0 | aux | h₅ ~5e-3 | C2（aux 高 N 验证）| P0 | ⬜ todo |
| model0 | direct | h₅ ~5e-3 | C4（看是否 OOM）| P2 | ⬜ todo |
| **model0 Lagrange 对照** | — | h₂ | C5 V2（σ 收敛阶）| P1 | ⬜ todo |
| **model0 Lagrange 对照** | — | h₃ | C5 V2 | P1 | ⬜ todo |
| **model1 / square** | direct | h_target≈l₀/2 | C1（含 Miehe 对照）| P0 | 🟡 重启进行中（PID 564628，pardiso backend 绕过 SuperLU segfault）；2026-05-30 在 **step 22/161**，history.csv 已被重启覆盖重写、收敛正常（旧崩溃 run 的 step67 数据已弃）；约 3 天+ 跑完 |
| model1 / square | aux | 同上 | C1+C3 | P0 | ⬜ todo |
| model1 Lagrange 对照 | — | 同上 | C5 V5（尖端 σ 探针）| P1 | ⬜ todo |
| **model2**（notch shear）| direct | h_target≈l₀/2 | C1+C3（剪切场景）| P0 | 🔴 **BOGUS**：`paper_direct/` 1700 行是 u≡0 平凡解（reaction_x≡0、max_H≡0、max_d 自 step0=1.0），x-Dirichlet 未被 enforce；勿用于 C1/C3，需重做（见 §13.3） |
| model2 | aux | 同上 | C3（剪切场景预条件子稳定性）| P0 | ⬜ todo |

**优先级**：P0 论文必须有；P1 应该有，缺了会被审稿人质疑；P2 补充材料/future work。

### 13.2 数据后处理脚本清单

| 脚本 | 用途 | 状态 |
|------|------|------|
| `collect_paper_bundle.py` | 汇总三 case × 三模式 metadata 到 PAPER_INDEX | ✓ 已有 |
| `paper_make_load_disp.py` | C1 三 case load-displacement 曲线对比图 | ❌ 待编写 |
| `iter_stability_scan.py` | C2/C3 数据：固定网格×均匀d 扫 none/Jacobi/ILU/aux 的 niter | ✅ 已写（h₁ done, h₂ 跑中）|
| `paper_make_iter_stability.py` | **C3 核心图**：niter vs d（半对数，对照组爆炸 vs aux 平稳）+ C2 niter vs N | ✅ 已写并验（出 `docs/figures/precond/iter_stability_vs_d.*`，vs_N 待 h₂/h₃）|
| `paper_make_iter_vs_N.py` | C2 备用图：从真实 run iterations.csv 聚合 aux/direct niter-vs-N | ✅ 已有（472 行）|
| `precond_spectrum.py` | C1 数据：$P^{-1}K$ 特征值（含 `aux_fast` factory；`--k-small 0` 走 LM-only）| ✅ 已加 aux_fast + 暴露 P（`return_preconditioner`）|
| `paper_make_spectrum.py` | **C1 图3**：特征值复平面散点 vs d + kappa_vs_d | ✅ 已写并验（`docs/figures/precond/spectrum_*.{png,pdf}`）|
| `precond_sweep.py` | A3 数据：niter vs (l0/eps_g/d)（含 `aux_fast`，默认）| ✅ 已加 aux_fast；修 lgmres bug |
| `mem_scaling.py` | B2 数据：peak RSS vs N（mf vs direct，独立进程）| ✅ 已写 |
| `paper_make_mem_scaling.py` | **B2 图**：peak_rss-vs-σDOF log-log + OOM 标记 | ✅ 已写并验（`docs/figures/precond/mem_scaling.*`）|
| `paper_make_iter_heatmap.py` | C3 热力图：staggered iter × GMRES iter × load step | ❌ 待编写 |
| `paper_make_efficiency_table.py` | C4 效率表：wall-time、RSS、scaling | ❌ 待编写 |
| `paper_make_sigma_conv.py` | C5 V2：σ 收敛阶（HuZhang vs Lagrange）| ❌ 待编写 |
| `paper_make_tip_stress.py` | C5 V5：裂纹尖端 σ 探针对比 | ❌ 待编写 |

### 13.3 model2 direct BOGUS 复查（2026-05-30 开）

`results/phasefield/model2_notch_x_stretch/paper_direct/` 那份 1700 步产出经核为 **u≡0 平凡解**（`reaction_x≡0`、`max_H≡0`、`err_u≡0`、`max_d` 自 step0 恒 1.0），即顶边 x 向 Dirichlet 位移**未被 enforce 进 HuZhang 鞍点系统**。与 memory `model2_paper_direct_bogus` 一致。**不可用于 C1/C3**，§13.1 已标 🔴。

**正确物理设定**（用户 2026-05-30 提供的 FEALPy `AFEMPhaseFieldCrackHybridMixModel` 参照程序，作为重做基准）：

| 项 | 值 |
|----|----|
| E / ν / Gc / l₀ | 210 / 0.3 / 2.7e-3 / **0.015** |
| 几何 | 单位正方形 [0,1]²，中心 (0.5,0.5) **半径 0.2 圆孔**，内圆周 u=d=0 |
| 加载 | **顶边 y=1 沿 x 方向**位移，0 → **0.024**（参照 2401 步；fracturex 按 `case.default_loads()` 步长）|
| 固定 | y=0 与 y=1 两条边均 Dirichlet |
| 受载分量 | 顶边 **x 分量**（`is_disp_boundary` → `np.c_[isDNode, 0]`）|

**复查待办**：① 定位 fracturex model2 case 的 BC 装配，确认 x-Dirichlet 提升项是否真正进右端 / 消元；② 小网格短跑验证 `reaction_x≠0 且 u≠0`；③ 通过后重跑全程，替换 bogus 产出。

**✅ 已修并验证（2026-06-03）**：根因在 `model2_notch_shear.py` 的 `neumann_data`——旧版 `(_on_y1, gd0, "nt", "t")` 把顶边 y=1 的**切向反力置零**，而该边 x、y 位移全约束（全位移边）→ 整 traction 是未知反力，不该作本质应力 BC → 把承载 x-拉伸的反力清零 → trivial `u≡0`。**已改为** `[(is_nedge_free, gd0, "nt", None)]`（只对真正 traction-free 边加本质 BC，源码 line 74-87 含 NOTE）。短跑验证（nx=24, load=0.012, d=0.9）：`|F|=7.15e-4≠0`、aux niter=9 收敛、`|u|=0.589≠0` → **不再 bogus**。③ 全程重跑替换旧产出仍待做（E1/I3 物理曲线用）。

---

### 13.4 论文定位修订 + 迭代稳定性 / matrix-free 实测整理（2026-06-02）

> **定位（专家建议，2026-06-02）**：**不卖计算效率（墙钟），卖迭代稳定性 + 小内存。** 受控扫描已坐实 2D 中等规模 aux 墙钟输 pardiso（~6× 算法硬墙，见记忆 `aux_loses_to_pardiso_2d`），但这**不影响论文论点**——本论文（§1）本就问"GMRES 迭代次数是否对 (h, l₀, load, max d) 均匀"。**主线 = 辅助空间预条件子（迭代稳定）+ matrix-free（小内存）**；C4 效率表降级为附录/诚实标注；**GPU（④）暂缓，仅记录于 `docs/routes/plan_gpu_multibackend.md` §7.6，作为 future work**。

**A. 迭代稳定性实测**（model0, fast=aux-space 对称 V-cycle, `linear_niter_elastic` = 每次弹性解 GMRES 迭代数）：

| step | h₁ niter / max_d | h₂ niter / max_d | h₃ niter / max_d |
|---|---|---|---|
| 1 | 6 / 0.009 | 7 / 0.009 | 8 / 0.009 |
| 5 | 7 / 0.225 | 7 / 0.214 | 8 / 0.214 |
| 12 | 6 / 0.412 | 7 / 0.368 | 8 / 0.365 |
| 13 | **7 / 0.954**（起裂后）| 7 / 0.426 | 8 / 0.418 |
| 14–16 | 8–9 / 0.96–0.97 | — | — |
| **σ-dof** | **10,924** | **48,092** | **183,524** |

- **网格无关性**：niter 6→7→8，DOF 增 16× 仅 +2 迭代 ≈ 最优预条件子（C2 核心证据）。
- **损伤鲁棒性**：h₁ 跨整个损伤演化（max_d 0→**0.97**，含起裂跳变），niter 仅 6→9；即 d→1 时应力块被 ~1/eps_g=10⁶ 放大，预条件子仍稳（C1/Q1 核心证据）。
- 对照 pardiso direct niter≡1（直接解，但内存/fill-in 随规模涨）。

**A2. 对照组 niter 扫描**（model0 h₁, 固定网格, 均匀损伤 d 扫；同 rtol=1e-8；MAXIT=300 restart-cycle，restart=200，故"60000"=打满未收敛；脚本 `scripts/paper_huzhang/iter_stability_scan.py`，数据 `results/phasefield/_iter_stability/iter_stability.csv`）：

| d (max_d) | none(无预条件) | jacobi | ILU | **aux_fast** |
|---|---|---|---|---|
| 0.0 | 60000 (DNF) | 3787 | **发散** (relres 1e5) | **7** |
| 0.5 | 60000 (DNF) | 3072 | 159 | **6** |
| 0.9 | 27184 | 2195 | 7 | **9** |
| 0.99 | 3534 | 5712 | 3 | **10** |
| 0.999 | 42843 | 18923 | 253 | **10** |

- **aux_fast 跨 6 个量级条件数(d=0→0.999, g(d)=1→eps_g=1e-6)始终 6–10**，唯一对 d 鲁棒。
- 无预条件几乎不收敛（多数打满 60000）；Jacobi 2k–19k 且非单调；**ILU erratic**——d=0 发散、高 d 才小（d→1 时 1/g(d) 使应力块对角占优、ILU 才偶然奏效），不可作稳定预条件子。
- 这正是 C2/C3 核心对比图："别人爆炸/erratic，aux 平稳"。已绘 **C3 图 `docs/figures/precond/iter_stability_vs_d.{png,pdf}`**（`paper_make_iter_stability.py`，半对数：aux 平贴底 6–10，none 多数 DNF 打满、ILU erratic、Jacobi 数千）。
- **C2 vs-N 图已出** `iter_stability_vs_N`（d=0.9, h₁+h₂）：**aux 近平 9→15、ILU 爆炸 7→59802(8500×)、Jacobi 2195→4693、none DNF**——mesh-independence 招牌图。aux 物理 run 另证 6→7→8(h₁₂₃)。h₃ 对照可选(none/jacobi/ilu 各 ~1hr DNF)。

**B. matrix-free（内存支线，详见 GPU 路线文档 §7）**：σ 应力块 M2 做成 matrix-free（不存 M2/(ldof,ldof)Phi），正确性机器精度（matvec/解/反力 3e-16~4e-15，**niter 与装配版不变**），h₃ peak RSS 比精简装配省 **1.9×**、比生产默认省 **7.8×**；M2_const 跳过为免费降峰值。诚实标注：CPU 时间 ~18×（不卖墙钟，按上面定位）。

**C. 受控扫描数据位置**：`results/phasefield/model0_circular_notch/paper_{direct_scan_pardiso,aux_scan_auxfast}_{h1,h2,h3}/`（统一 pardiso baseline + fast + 每步 checkpoint/VTU + 单线程）；pardiso@1 判别数据在 `paper_direct_scan_pardiso1_{h1,h2}/`；对照组 niter 在 `_iter_stability/{iter_stability,l0_sweep}.csv`。

**C2. 默认算法已设为最优（2026-06-02）**：① `run_case.py` 的 `_use_elastic_fast` 默认改为 **True**——aux 模式默认走 `fast`（niter≈6），`FRACTUREX_ELASTIC_FAST=0` 仅留作 auxspace ablation；② `precond_sweep.py` 新增 `aux_fast` 算法且设为默认 `--algorithm`；③ 修复 `solve_lgmres_ilu` 的 scipy 不兼容 bug（`lgmres` 不接受 `callback_type`，回调收解向量）——ILU 对照列现可跑（但在大鞍点上极慢，佐证非 robust）。

**D. 下一步实验（支撑迭代稳定性论点）**：① ✅ 对照组 niter vs max_d（h₁）已完成（见 A2，d-snapshot 平台已验，aux 6–10 平稳）；🟡 h₂/h₃ 跨网格对照在跑；② 把 A2 表绘成 C2/C3 核心图（niter vs d 半对数 + niter vs N），写 `paper_make_iter_vs_N.py`；③ eps_g 轴扫（{1e-3,1e-6,1e-9}）验 aux 对退化下界鲁棒；④ h₂/h₃ 跑过起裂区补全真实损伤演化的 niter 曲线。

---

### 13.5 完成全文所需数值实验清单（按新定位盘点，2026-06-02）

> 定位：卖**迭代稳定性 + 小内存**（§13.4）。下表 = 完成投稿全文还差哪些数值结果。状态：✅done / 🟡partial-running / ⬜todo。

**A. 核心：迭代稳定性（论文卖点，对应 §5.2–5.4 / 图2 / 表1-2 / Prop3 数值验证）**

| # | 实验 | 论文位置 | 状态 | 还需 |
|---|------|---------|------|------|
| A1 | niter vs 损伤 d（none/Jacobi/ILU/aux，固定网格）| §5.4 图2(C3) | ✅ model0 h₁（`iter_stability_vs_d`，aux 6–10 平、others 爆炸）| 可加 1–2 个更细网格叠加 |
| A2 | niter vs 网格 N（4 算法跨 h₁,h₂）| §5.2 表1+图(C2) | ✅ 图 `iter_stability_vs_N`（d=0.9）| **aux 近平 9→15、ILU 爆炸 7→59802(8500×)、Jacobi 2195→4693、none DNF**；aux 物理 run 另证 6→7→8(h₁₂₃)。可选加 h₃ 对照(每个 none/jacobi/ilu ~1hr DNF) |
| A3 | niter vs 正则化 l₀（固定 h，高斯损伤宽~l0）| §5.3 表2 | ✅ aux 列（`l0_sweep.csv`）| **aux_fast l₀-无关 7/7/7**（l0=0.04/0.02/0.01，h2，max_d=0.9）、aux_weighted 7/7/6；ILU(lgmres) 在 80k 鞍点上极慢/不 robust（佐证）。待补：更细 l0 + 配套 h∝l0 resolved 版 |
| A4 | niter vs eps_g 退化下界（{1e-3,1e-6,1e-9}）| §5 消融 | ⬜（A2 已隐含 eps_g=1e-6 d→1）| 单独 eps_g 轴扫，验 aux 对 r_g 鲁棒 |

**B. 核心：小内存（matrix-free 支线，对应新增 §5.x / 表3 重构）**

| # | 实验 | 状态 | 还需 |
|---|------|------|------|
| B1 | matrix-free 正确性（matvec/解 机器精度，niter 不变）| ✅ model0 h₃ | — |
| B2 | peak_rss vs N：matrix-free vs direct（h₁–h₅）| ✅ `mem_scaling.csv`，图 `mem_scaling.{png,pdf}` | **mf 全程更省、比值随 N 改善 0.81→0.27×**；σ-dof 2.0M 处 **direct=11.4GB 且 pardiso 失败(singular)、mf=4.6GB 仍可行**。诚实标注：2D pardiso 嵌套剖分内存近线性，故 mf 赢的是**常数因子 2–4× + 大规模鲁棒性**；戏剧性 OOM(O(N²) vs O(N))是 **3D** 论点（future）。表：11k/48k/184k/504k/2.0M → mf 340/429/719/1374/4629MB，direct 421/768/2043/5185/11425(fail) |

**C. 谱验证（理论支撑，对应 §5.6 图3 / Prop2）**

| # | 实验 | 状态 | 还需 |
|---|------|------|------|
| C1 | $P^{-1}K_h$ 谱（LM 体谱）散点 vs max_d | ✅ aux_fast（`spec_auxfast_d*.npz`，图 `spectrum_{scatter,kappa_vs_d}`）| **体谱跨 d 不变**：20 个 LM 特征值全实数、聚集 [109,158]，d=0→0.999 各色几乎完全重叠，$\kappa_{LM}\approx1.44$ 恒定 → Prop3 数值验证。诚实标注：ARPACK **SM（近零特征值）在该非正规块预条件算子上不收敛/不可靠**（kappa via SM 是启发非真界，见 estimate_spectrum 注释），故只报 LM 体谱聚集；预条件子质量的主证据是 niter（A1-A3）。待补：多网格谱叠加验 mesh-independence |

**D. 广度：多算例 + 双公式（对应 §5.1 / C1 物理可信度，避免"只在 model0"被质疑）**

| # | 实验 | 状态 | 还需 |
|---|------|------|------|
| D1 | model0（圆孔拉伸）迭代稳定+内存 | ✅ 主力 | — |
| D2 | model1/square（I 型直裂纹）aux 迭代稳定 | ✅ niter-vs-d nx=20,30（`iter_stability_square.csv`，图 `iter_stability_square_vs_d`）| **aux 9–34 有界、nx=30 唯一收敛**；损伤鲁棒推广到 I 型成立。见 D12_RESULTS §5.7 |
| D3 | model2（II 型剪切）aux 迭代稳定 | ✅ **nx=24/32 双网格完成**：aux nx=24=9/9/8/12/11、nx=32=10/10/11/14/14（d=0→0.999），**O(10)（8–14）有界且网格稳定**；对照组全 DNF/爆炸（none/Jacobi 数千–6万、ILU 几乎全 DNF）。D12_RESULTS §5.8 表8a/8b（σ=19,347/34,243）+ 图7 `iter_stability_model2_vs_d` 已齐 | BOGUS 已修验证见 §13.3；脚本 `paper_make_iter_stability_case.py` |
| D4 | effective_stress 公式 aux_unweighted 鲁棒性（Q2）| ✅ 可行性已验（**附录 A**，niter=5 跨 d→0.999 同 standard）| **不进正文/Discussion**——留作未来论文维度（GPU/monolithic/3D）。Q2 在本文不答 |

**E. 支撑：物理正确性（轻量，让审稿人信任测试问题）**

| # | 实验 | 状态 | 还需 |
|---|------|------|------|
| E1 | 3 算例 load-displacement 曲线（含 Miehe 对照）| 🟡 square direct 跑中 | 各 case 一条到底的完整曲线（已有 checkpoint/VTU 机制）|

**F. 诚实标注（降级为附录，不再当卖点）**

| # | 实验 | 状态 | 处理 |
|---|------|------|------|
| F1 | wall-time aux vs pardiso（2D 中等规模）| ✅ 已测（aux ~6× 算法慢）| §5.5 改写"2D 迭代法不占墙钟优势、价值在稳定性+内存；direct 在大 N/3D OOM"——诚实标注，不卖速度 |

**最小可投稿集（MVP）**：A1✅ + A2(补完) + A3 + B2(N曲线) + C1 + D2 + E1 → 覆盖 §5 全部图表 + Prop3 数值验证。**优先级**：A2/A3/B2/C1（核心四项）> D2/D3（广度）> D4/E1（加分）。GPU(④) 仍为 future work，不进本文。

---

### 13.6 成稿后复盘：仍缺的数值实验（2026-06-03，按已写入 `phasefield_huzhang.tex` 复盘）

> §5 主体已落入 tex（理论 §5/§6 已与程序对齐：对角应力块 + Schur 辅助空间 V-cycle + matrix-free；数值 §7 已搬入 A1/A2/A3/B2/C1）。下表补充 §13.5 之外、**当前主线（迭代稳定 + 小内存）下成稿仍缺**的实验，含 tex 对应位置与 `\needexp` 标记。状态：✅done / 🟡partial / ⬜todo。

**G. 论文已声称但尚未数值验证（P0，补不上要改写论文措辞）**

| # | 实验 | tex 位置 | 状态 | 说明 |
|---|------|---------|------|------|
| G1 | ~~块对角 + MINRES 迭代数~~ → **改写措辞，不补实验**（2026-06-03 决定）| §6 "Resulting solvers" | ✅ 已处理 | 决定：block-tri/GMRES 为唯一工作求解器、报全部数值；block-diag/MINRES 仅作"同一谱等价下的对称理论对照"一句带过。tex §6 已加该句（"we adopt it as the working solver and report all numerical results for it, the symmetric block-diagonal/MINRES variant being its theoretical counterpart"）。**无需 MINRES 实验** |
| G2 | **C5 — V2** σ 的 L² h-收敛阶（制造解，HuZhang p3 vs Lagrange 后处理 p2）| §7.9 `tab:hz_vs_lag`/`fig:hz_vs_lag` | ✅ **已并入 tex**（2026-06-03）：HuZhang 4.01 阶 vs Lagrange 2.00 阶，N=48 误差小 **>3 量级**（3.5e-7 vs 8.7e-4；原记"5量级"实测约 3.4，已订正）。表7+图6 转写进 §7.9，preliminary 已撤 | — |
| ~~G3~~ | ~~C5 — V5 裂纹尖端 σ 切片~~ | — | ❌ **已删**（2026-06-03）| 决定不做：偏离主线（求解器迭代稳定+小内存）、且需昂贵 Lagrange 相场 SENT/SENS 同 DOF run + 接口对齐。tex §7.9/贡献(vi)/结论(vi)/outlook 均已去 V5、收成"V2 一项+文献动机" |

> 注：主线（辅助空间迭代稳定 + matrix-free 小内存）**不依赖 C5**；C5 仅支撑"为何选 Hu-Zhang"动机，已由 intro/Remark 的文献性质（最优 L² 应力阶/强对称/无锁）+ C1 物理一致 + 现在保留的 V2 一张收敛阶图承担。V5 砍掉，tex 相应贡献/章节已收口。

**H. 核心claim 的补强（P0–P1，让招牌图/谱图更硬）**

| # | 实验 | tex 位置 | 状态 | 说明 |
|---|------|---------|------|------|
| H1 | **C2 招牌图补点 h₃–h₅**：synthetic d=0.9 的 aux niter（+ 对照至少 h₃）| §7.4 表`tab:mesh_indep`/图`iter_stability_vs_N` | ✅ aux 跨满 **187×**：h₁–h₅ = **9/15/9/8/9**（10,924→2,045,540），全程 O(8–15) 有界 mesh-indep；对照组 h₂ 已 DNF/爆炸、h₃⁺ 不可行（表标"—"）| 数据 `aux_mesh_indep_d0.9.csv`；招牌图已重绘合并 aux 全 5 档（对照止于 h₂）；D12_RESULTS §5.2 表1+结论已更 |
| H2 | **谱的 mesh-independence** | §7.7 `tab:spectrum_mesh` | ✅ **已并入 tex**（2026-06-03）：d=0.9 扫网格 κ_proxy h₁/h₂/h₃ = **1.45/1.15/1.08**（跨 30× DOF 谱恒紧聚 O(100–180)、随细化更紧）→ 谱对 h+d 双向无关，Prop3 完整。新增表 `tab:spectrum_mesh` + 段落 + caveat 覆盖两轴 | — |
| ~~H3~~ | ~~eps_g 轴扫~~ | — | ❌ **不做**（2026-06-03：作用不大）| d→0.999 的损伤扫（A1/D2）在 eps_g=1e-6 下已把 1/g 推到 10⁶、r_g 鲁棒性已覆盖；单独轴扫边际收益低。tex §damage 红标已去 eps_g |

**I. 物理可信度 + 广度（P1，避免"只 model0/只 standard"）**

| # | 实验 | tex 位置 | 状态 | 说明 |
|---|------|---------|------|------|
| I1 | **CNT aux 跑过软化段**与 direct 全程重合 | §7.2 `\needexp` | 🟡 **在跑**（h₁ 已全程，h₂/h₃ 续算中）| 跑到底即坐实 C1 端到端一致 |
| I2 | **SENT 全分离刷新** load-disp + 裂纹路径（Miehe 对照）| §7.3 `\needexp` | 🟡 **在跑**（direct，现 ū=5.14e-3）| 跑到完全分离后刷新图表 |
| I3 | **剪切第三算例（model2，II 型）损伤 niter 扫** | §7.6 damage（待加）| 🟡 **数据 nx=24✅ / nx=32 跑中**（D12_RESULTS §5.8 表8a；aux 8–12 有界、对照全 DNF）| **tex 集成 DEFERRED**（2026-06-03 用户决定：等 nx=32 落盘 + 统一几何命名后再整理）。⚠️ 见下方几何对齐注 |
| I4 | **D2 model1/square**（I型）aux 迭代稳定 + 内存 | §7.6 damage 小节（`tab:precond_dscan_sent20/30`/`fig:iter_vs_d_sent`）| 🟡 **损伤鲁棒已做并入 tex**（nx=20/30 d-扫，aux 9–34 有界、nx=30 others 全 DNF；§5.7/图5）| 剩：square 的 mesh-indep(iter-vs-N) + 内存档；可选补全 |
| I5 | （并入 I3）剪切算例物理曲线 load-disp + 裂纹 | §7.2/§7.3 | 🔴 全程重跑替换旧 bogus 产出仍待做 | E1 物理曲线用；与 I2 同批 |
| ~~I6~~ | ~~D4 effective_stress 公式 aux 鲁棒~~ | §5 仅保留构造注 | ❌ **不做**（2026-06-03：主线只 standard，非必要）| tex §5 已有一句构造注（权重移到 B、粗算子改无权 Laplace），不承诺数值 ⇒ 无需实验。Q2 留 future work |

> ⚠️ **几何对齐注（2026-06-03，剪切算例集成前必须解决）**：实验跑的 **model2 = 中心圆孔(r=0.2)板 x-向剪切**；但当前 tex §7.1 把第三算例 `SENS` 定义为**预裂方板剪切**（几何不同，且 `SENS` 目前无数据）。统一整理剪切结果时需二选一：
> 1. **（倾向）** 把论文第三算例正式定为 model2：更新 §7.1 几何/载荷表(`tab:case_geometry`)、材料表、几何图(c) caption 为"中心圆孔 x-剪切"，并把 `SENS`(single-edge-notched shear) 重命名为如 `CNS`(circular-notch shear) 或统称"shear benchmark (mode II)"；
> 2. 或保留预裂方板 `SENS` 另跑、model2 不进主表。
> 因 `SENS` 现无数据，方案 1 无损失且与实验一致。**model2 数据（表8a，nx=24）+ 三算例横向（model0 8–18 / square 9–34 / model2 8–12，aux 均 O(10)）已在 D12_RESULTS §5.8 就位，待 nx=32 + 几何决定后一次性并入 §7.6。**

**J. 支撑/加分（P2，可投稿后补或进附录）**

| # | 实验 | tex 位置 | 状态 | 说明 |
|---|------|---------|------|------|
| J1 | **Anderson 外层加速量化**（staggered iter on/off + 反力差）| §7.10 末 `\needexp`（附录/可选）| 🟡 定性已在 tex（~270 vs ~50、降约一量级）+ 红标待补表 | 跑出 staggered iter on/off×载荷阶段 + 反力相对差一小表即可填 |
| J2 | **matrix-free niter 跨档不变** | §7.8 `\needexp`（可选）| 🟡 现 B1 只 h₃，红标已就位 | 轻量：每档一行 niter（装配 vs matrix-free）一致即可填 |
| J3 | **块预条件子构造示意图**（非实验，画图）| §6 `fig:precond_schematic` | ✅ **已画并并入**（2026-06-03）| Python `make_precond_schematic.py`（工程流程版：块三角 sweep + 两层 V-cycle，matplotlib，无 usetex 依赖）。已放大字号 + matrix-free 高亮红框 + 加长 forward-sweep 箭头(1/2 编号) + 修边界越界。另画过数学/算子版 `make_precond_operator_diagram.py`，按偏好**取工程版、弃数学版**（脚本留存） |

**更新后 MVP**：A1✅ A2🟡 A3✅ B2✅ C1✅ + G1✅（改写不补实验）+ **H1✅（C2 补点 h₃–h₅，跨 187×）** + G2✅（V2 已并入）+ D2✅/E1🟡。
G3（V5）已删。**优先级**：~~H1~~✅ > ~~I（广度 D2✅/D3✅）~~ > J（加分）。G1/G3/G2/H1 已分别以"改写"/"删除"/"已并入"/"已补满"解决；I 组三算例（model0/square/model2）全成稿。

---

### 13.7 §5 数值实验章重叠梳理 + tex 红标（`\needexp`）清单（2026-06-03）

> 对 `phasefield_huzhang.tex` 数值实验章（10 小节）做了重复/重叠排查，并把"需补充/修改"的表/图用红色 `\needexp` 标在对应 Table/Figure 旁，结果回来即可按标就位。

**章节结构**：Setup → Consistency(C1) → SENT model1(C1物理) → Mesh-indep(C2) → l₀-indep → Damage evolution(C1/C3) → Spectrum(Prop) → Memory(matrix-free) → Lagrange V2(C5) → Discussion(staggered)。

**重叠排查结论**：

| 严重度 | 位置 | 说明 | 处理 |
|---|---|---|---|
| **真重复** | `tab:mesh_indep`(d=0.9, h₁/h₂) ⊂ `tab:precond_dscan_h1/h2`(d=0.9 行) | 8 个数字逐字相同——同一受控扫描两种切片（mesh-indep 固定 d 扫网格；damage 固定网格扫 d）| 不删（双轴呈现合理）；tex 已加交叉引用句点明二者在 d=0.9 相交；并由 **H1**（扩 h₃–h₅）使 mesh-indep 跨真实范围而非只剩 2 共享格 |
| 轻微 | 真实 run niter 6/7/8（mesh 轴）vs 6→9（damage 轴）| 同一物理 run 两切片 | 保留，属不同轴 |
| 轻微 | matrix-free "round-off/niter 不变" 在 §matrix_free（方法/定性）与 §memory（数值/带 3e-16）各一次 | 性质 + 数值佐证 | 可接受，未动 |
| 标注差异(非重复) | "10⁸"（frozen-damage，ξ∈{1e-6,1e-8}，`tab:auxverify`）vs "10⁶"（迭代扫，ξ=1e-6, d→0.999）| 两个不同研究，各自诚实 | 各 caption 已自洽，无需改 |

→ **只有一处真重复**（mesh-indep ⊂ damage 的 d=0.9 切片），已用交叉引用 + H1 扩档处理。

**tex 当前红标 `\needexp` 全清单**（行号随编辑漂移，按 label 定位）：

| tex 锚点 | 红标内容 | 对应清单项 | 优先级 |
|---|---|---|---|
| `tab:mesh_indep`/`fig:iter_vs_N` | ✅ 已扩 h₃–h₅（aux 跨 187× = 9/15/9/8/9 有界；baseline 止于 h₂）| H1✅ | ~~P0~~ done |
| ~~`tab:spectrum`~~ | ✅ 谱 h₂/h₃ 网格无关已并入（新表 `tab:spectrum_mesh`，κ 1.45/1.15/1.08）| H2 done | — |
| §damage 末 | （已去 eps_g/eff-stress）剩 SENS 损伤扫待跑 + 可选 h₃ 叠加；**SENT/square 损伤鲁棒已并入** `tab:precond_dscan_sent20/30`/`fig:iter_vs_d_sent` | I3（SENS）| P1 |
| ~~§hz_vs_lag~~ | ✅ V2 收敛阶（`tab:hz_vs_lag`/`fig:hz_vs_lag`）已并入 | G2 done | — |
| §consistency (i) | CNT aux 跑过软化段（h₂,h₃）| I1 | P1 |
| §consistency 末 | SENS aux/direct load-disp + 裂纹 | I3 | P1(先修BOGUS) |
| §model1 | SENT 全分离刷新图表 | I2 | P1 |
| `fig:iter_vs_d`（optional）| 叠加 h₃ 证 damage 鲁棒也 mesh-indep | A1 加分 | P2 |
| `tab:l0_indep`（optional）| 加细 l₀ + h∝l₀ resolved 版 | A3 加分 | P2 |

> 说明：标 `(optional)` 的为加分项，不阻塞成稿；其余红标补齐即覆盖 §5 全部主线图表。全部红标的逐条填充清单见 §13.8。

---

### 13.8 tex 红标逐条填充清单（按 label 索引，跑完勾销，2026-06-03）

> tex 当前 11 处 `\needexp`（红色 TODO）。每条给：论文锚点（label / 小节）→ 对应清单项 → 优先级 → 跑/画完把结果填到哪。P0 在程序侧跑；P1 多在跑或一键可跑；P2 加分。
> 状态记号：`[ ]` 待办 / `[~]` 在跑 / `[x]` 完成（勾销时改记号 + 注 commit/数据路径）。

**P0（成稿核心，程序侧）**
- [x] **H1** ✅ — `tab:mesh_indep` / `fig:iter_vs_N`（§7.4）：iter-vs-N 扩到 **h₃–h₅** 完成。aux d=0.9 跨 187× DOF（10,924→2,045,540）= **9/15/9/8/9** 全程 O(8–15) 有界；对照组止于 h₂（h₃⁺ 不可行，表标"—"）。数据 `aux_mesh_indep_d0.9.csv`，表+招牌图 `iter_stability_vs_N` 已重绘。
- [x] **G2** — `tab:hz_vs_lag`/`fig:hz_vs_lag`（§7.9）✅ 已并入：4.01 vs 2.00 阶、N=48 误差差 >3 量级；表7+图6（`figures/hz_vs_lagrange_v2.pdf`）就位，preliminary 已撤。

**P1（物理可信度 + 补强；多在跑/一键可跑）**
- [~] **I1** — `tab:model0_consistency`（§7.2）：CNT aux 跑过软化段（h₂,h₃ 续算；h₁ 已全程）→ 刷新 `tab:model0_consistency` + `fig:model0_loaddisp`。
- [~] **I2** — `sec:numerics_model1_done`（§7.3）：SENT direct 跑到完全分离 → 刷新 `fig:model1_loaddisp`/`fig:model1_crack_evolution`/`tab:model1_summary`。
- [~] **I3（剪切第三算例 model2，II 型）** — §7.6 damage（待加 表8b+图7）：数据 nx=24✅（aux 8–12 有界）/ nx=32 跑中。**tex 集成 DEFERRED**（等 nx=32 + 几何命名统一，见 §13.6 几何对齐注）。
- [x] **H2** ✅ — §7.7 `tab:spectrum_mesh`：谱 mesh-independence 已并入（d=0.9 扫 h₁/h₂/h₃，κ_proxy 1.45/1.15/1.08，跨 30× DOF 谱恒紧聚、随细化更紧）；Prop3 的 h+d 双向无关完整。

**P2（加分/附录，全 optional）**
- [ ] **A1+** — `fig:iter_vs_d`（§7.6）：CNT 损伤扫叠加一档 h₃，证 damage 鲁棒本身 mesh-indep。
- [ ] **A3+** — `tab:l0_indep`（§7.5）：加细 l₀ + h∝l₀ resolved 版，分离 l₀-无关与网格效应。
- [ ] **J2** — `sec:numerics_efficiency`（§7.8）：matrix-free niter 跨档一致（现仅 h₃）；每档一行装配 vs matrix-free niter。
- [ ] **J1** — `sec:numerics_discussion`（§7.10，附录）：Anderson on/off × 载荷阶段 staggered iter + 反力相对差一小表（坐实 ~1.5–2× / <0.01%）。
- [x] **J3** ✅ — §6 `fig:precond_schematic`：构造示意图已画并并入 tex（Python `scripts/paper_huzhang/make_precond_schematic.py` → `Frac_huzhang/figures/precond_schematic.{png,pdf}`）。工程流程版：块三角 sweep + 两层 V-cycle；已大字号 + matrix-free 高亮红框 + 加长 sweep 箭头 + 修越界。数学/算子版 `make_precond_operator_diagram.py` 试做后按偏好弃用（脚本留存，未并入）。

> 对照：上面 11 条 = tex 全部 `\needexp`。**主动新增只有 P0 两项**；I1/I2 在跑、I3 一键可跑；P2 五项纯加分。砍掉的 H3（eps_g）、I6（effective_stress，可行性已于本节末验证、留 future）不在清单。

---

## 14. 交错外层的 Anderson 加速（数学过程）

> **定位**：本节描述的是**交错（staggered / 交替最小化 AM）外层不动点迭代**的加速，作用在损伤场 `d` 上；它与 §3–§5 的**内层 σ–u 块预条件器正交**——加速改变外层 `d` 迭代序列，预条件器只管每个外层步内那一次弹性线性解。两者可独立开关、增益基本叠加（实测内层 GMRES 迭代数几乎不受外层加速影响）。
>
> 本节为纯数学/算法层（不写软件细节）。实现与验证见末尾"实现指针"。

### 14.1 交错不动点与朴素 AM

固定一个载荷步。交替最小化在外层迭代 `k` 上：给定当前损伤 $d^{k}$，先解弹性子问题 $K_h(d^{k})(\sigma,u)$（§3），再解相场子问题，得到一次"原始"不动点像

$$
\tilde d^{k} \;=\; G(d^{k}),
$$

其中 $G:\mathbb R^{n_d}\to\mathbb R^{n_d}$ 是"一次完整交错扫描"算子。定义不动点残差

$$
f^{k} \;=\; G(d^{k}) - d^{k}.
$$

朴素 AM（带欠松弛 $\omega\in(0,1]$、不可逆约束、bound 投影）的更新为

$$
d^{k+1}_{\mathrm{plain}}
=\Pi_{[0,1]}\!\Big(\max\big(d^{k},\,\omega\,G(d^{k})+(1-\omega)\,d^{k}\big)\Big),
$$

其中 $\Pi_{[0,1]}$ 为逐分量 clip，$\max(d^{k},\cdot)$ 施加损伤不可逆。AM 能量单调、（子列意义）收敛到临界点 [Bourdin–Francfort–Marigo, JMPS 48:797, 2000；Bourdin, IFB 2007]，但在起裂/局部化段外层迭代数会飙升（教科书行为，非发散）。

### 14.2 投影 Anderson(m)

把一次交错扫描视为不动点映射 $x_{k+1}=G(x_k)$（$x\equiv d$），用窗口大小 $m$ 的 Anderson 外推后处理其增量 [Storvik et al., CMAME 381:113822, 2021；Farrell & Maurini, IJNME 109:648, 2017]。维护近 $m$ 步的迭代差与残差差矩阵

$$
\Delta X^{k}=\big[\,d^{k}-d^{k-1},\;\dots,\;d^{k-m+1}-d^{k-m}\,\big],\quad
\Delta F^{k}=\big[\,f^{k}-f^{k-1},\;\dots\,\big],
$$

解（Tikhonov 正则的）最小二乘

$$
\gamma^{k}=\arg\min_{\gamma}\big\|f^{k}-\Delta F^{k}\gamma\big\|_2,
\qquad
\big(\Delta F^{k\top}\Delta F^{k}+\lambda_k I\big)\gamma^{k}=\Delta F^{k\top}f^{k},
$$

$\lambda_k=\lambda\,\mathrm{tr}(\Delta F^{k\top}\Delta F^{k})/m$（数值稳健化）。加速候选

$$
x^{k+1}=\big(d^{k}-\Delta X^{k}\gamma^{k}\big)+\beta\big(f^{k}-\Delta F^{k}\gamma^{k}\big),
\qquad \beta\in(0,1]\ (\beta{=}1:\text{plain Anderson}).
$$

**投影 Anderson**：不可逆与 clip 在加速*之后*施加，

$$
d^{k+1}=\Pi_{[0,1]}\!\big(\max(d^{k},\,x^{k+1})\big).
$$

### 14.3 信赖域 + restart 安全机制 + 起裂 kick

裂纹突跳处 $\Delta F^{k}$ 近奇异，最小二乘外推会产生*有限大*过冲。两层保护：

**(i) 信赖域**：以朴素步长 $\|f^{k}\|$ 为尺度，限制单次加速步

$$
s=x^{k+1}-d^{k},\qquad
\text{若 } \|s\|>\tau\|f^{k}\|\ \text{则}\ x^{k+1}\leftarrow d^{k}+\frac{\tau\|f^{k}\|}{\|s\|}\,s,
$$

$\tau$ 为信赖域系数（trust-region factor）。

**(ii) restart（清空窗口）**，触发条件二选一：

$$
\text{(blow-up)}\ \|f^{k}\|>\mu\,\|f^{k-1}\|,
\qquad
\text{(stall)}\ \#\{\text{连续无改善步}\}\ge p,
$$

$\mu$=blow-up 倍数、$p$=停滞容忍步数。**关键**：blow-up 基准用*上一迭代*残差 $\|f^{k-1}\|$，**不**用步内全局最小残差——否则峰前收敛后全局最小趋零，起裂残差突跳必触发 restart，使加速在最需要处退化成朴素步（极限环）。

**(iii) 起裂 kick（窗口空时的重播种）**：窗口为空（载荷步首迭代或 restart 后第一步）时取

$$
x^{k+1}=d^{k}+\omega_{\mathrm{seed}}\,f^{k},
\qquad
\omega_{\mathrm{seed}}=
\begin{cases}
\omega_R\ (>1), & \text{紧随一次 restart（over-relax kick，破极限环）}\\[2pt]
\omega\ (\,=1\,), & \text{载荷步首迭代的自然播种（plain，不 over-relax）}
\end{cases}
$$

over-relax **只在 restart 后的那一步**施加，用于把迭代扰动出起裂极限环；**稳态绝不 over-relax**，以免在固定容差下把解推偏、损害反力精度。

### 14.4 收敛判据（防假收敛）

外层收敛测度用**未加速**的投影朴素增量

$$
\big\|\,d^{k+1}_{\mathrm{plain}}-d^{k}\,\big\|\Big/\,e_0\;<\;\mathrm{tol},
$$

（$e_0$ 为该载荷步首迭代增量的尺度归一）。即收敛反映真实不动点残差 $\|G(d)-d\|$，**而非可能很小的加速增量**——加速因此不会制造假收敛。

### 14.5 性质与定位（论文措辞）

- **理论锚点**：AM 能量单调、子列收敛到临界点（上引）；Anderson 为启发式效率手段，加速格式本身无收敛证明。
- **实测（model0 aux, 粗网格）**：峰前段外层迭代数约 **2×** 下降；起裂步可靠收敛（朴素 AM 在此磨数百步未收）；稳定段反力与朴素 AM 相对差 **≤0.007%**；外层加速对内层 GMRES 迭代数影响可忽略（外/内正交）。**现实总加速约 1.5–2×**（与 Farrell–Maurini 的 ~5–6× 同一量级、偏保守），论文按此口径写，不宣称数量级加速。
- **非唯一性提醒**：起裂后加速 ON/OFF 可能落到不同临界点（AM 在非凸局部化段的固有非唯一性，非加速缺陷）；若需唯一 post-peak 曲线需走耗散弧长路径跟踪（另案）。

> **实现指针**：`fracturex/drivers/anderson_acceleration.py`（`AndersonAccelerator`，参数 $m,\beta,\omega,\tau,\mu,p,\omega_R$）+ `huzhang_phasefield_staggered.py` 集成；commit `6dc62ee`（安全机制）/ `a779abe`（restart kick）。默认关闭，开启口径见 `scripts/paper_huzhang/run_aux_model{0,1}.sh` 注释。详见 memory `staggered_acceleration_refs`。

---

## 附录 A：effective_stress 公式可行性（D4 — 不进正文/不进 Discussion）

> **定位决定（2026-06-03）**：本论文聚焦 **standard 公式**（$1/g(d)$ 在应力块；正文 §5 全部数据 + 两算例 D1/D2 + 理论 §3.2 均此公式）。**effective_stress 公式（$g(d)$ 在耦合块 $B$，aux 用不加权粗空间）不进正文、不进 Discussion**——加进来会引入第二个 aux 变体 + 第二套谱论证，稀释主线。它作为**未来论文的一个维度**（折进 GPU/matrix-free ④ 或 monolithic A2 或 3D 某篇），不单独成篇。本附录仅留一次可行性记录。

**可行性测试**（model0, hmin=0.05, l₀=0.02, eps_g=1e-6, load=0.09, 均匀损伤 d-扫；`precond_sweep.py --formulation effective_stress --algorithm aux_unweighted`）：

| d | **effective_stress**（aux 不加权）niter | standard（aux 加权）niter |
|---|---|---|
| 0.0 | 5 (conv) | 5 |
| 0.5 | 5 (conv) | 5 |
| 0.9 | 5 (conv) | 5 |
| 0.99 | 5 (conv) | 5 |
| 0.999 | 5 (conv) | 5 |

**结论**：effective_stress + 不加权辅助空间在 model0 上 **niter=5 跨 d→0.999 完全平稳**，与 standard 加权完全一致 → **aux 框架对两种 $g(d)$ 放置方式同等可行、同等鲁棒**。可行性已明确，留待未来论文展开。代码入口：`HuZhangElasticAssembler(formulation="effective_stress")` + `solve_huzhang_block_gmres_auxspace(weighted_aux=False)`。

