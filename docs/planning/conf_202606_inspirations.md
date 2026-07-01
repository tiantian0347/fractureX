# 2026/06 港中深"科学计算与快速求解器"研讨会 — fracturex 可迁移方法整理

会议：翔龙鸣凤科学论坛之科学计算与快速求解器研讨会，2026/06/25–27，香港中文大学（深圳）。
原始材料：会议手册 `会议手册草稿（外部）V3.3 - 副本.docx`，3 天 slide。
对应 fracturex 现状：Hu–Zhang 混合元相场 + paper_aux baseline + 自适应/PC 实验（参见 `project_fracturex.md`）。

下面按 fracturex 当前的几条主线（线性求解器、自适应、约束/能量保持、混合元、数据驱动、复杂几何）归类，每条点出报告人主张 + 对 fracturex 的具体迁移路径。

---

## 1. 快速线性求解器 / 预处理子（fracturex 主要瓶颈）

fracturex 当前组合：弹性 pardiso 直接解（paper_aux）或 aux-space GMRES；相场 GMRES no-precond。3D / 自适应一旦放大，pardiso 内存/scaling 立刻吃紧，这条线相关报告最多。

- **罗力 — Data-driven preconditioning for linear and nonlinear PDE systems**
  - 核心：从 Krylov 残差历史用 PCA 抽取"慢/近不变子空间"得增强预条件子；非线性场景用 PCA + NN 学 residual→error 映射，作为更优初值/修正方向；并提出 stagnation-shortening inexact Newton。
  - 迁移：
    - 弹性步：把 GMRES 的残差历史在线 PCA，做 deflation / spectral preconditioner，针对 `d` 演化导致的 stiffness 慢慢变奇异这条特性应该收益大。可以接到 `solver/elastic_*` 现有 GMRES 框架后面。
    - 交替迭代：弹性–相场 staggered/Newton 的非线性残差也满足"低维慢子空间"假设，可以学一个轻量 NN 给下一步 Newton 提供初值，配合 paper_aux 的 du=1e-4 small-step 序列很自然。
    - 现成可对照：他在 Stokes / 高 Re cavity / 超弹性血管上都验证过，超弹性血管和 Hu–Zhang 弹性结构最近。

- **胡齐芽 — 两层杂交加权 Schwarz 预条件子（高频 Helmholtz/Maxwell）**
  - 核心：重叠区域分解 + 子问题加 impedance/Robin BC 保唯一可解 + **局部 GEP 自适应构造粗空间** + 加权调和扩展粗基；理论上一致收敛。
  - 迁移：相场严重退化区域 `g(d)` 让弹性刚度跨数量级跳变（"高对比系数"），正好对应 GDSW / 谱粗空间擅长的场景。可以替代 aux-space precond 做 elastic 步的两层 DD：
    - 子域用现有划分，子问题用 impedance 型 Robin 来稳化退化区。
    - 粗空间从 element-local GEP（基于 H–Z 的弹性张量）来构造，不需要先验。
  - 备注：他强调"粗空间维数小但基函数构造贵"，对相场而言可以仅在 d 大幅更新的子域重算粗空间。

- **夏建林 — 结构化快速稳定求解器（rank-structured, HSS/FMM 风格）**
  - 核心：在矩阵里挖低秩结构（HSS、HBS、hierarchical semiseparable），near-O(n) 复杂度，并补齐 FMM 等经典算法的稳定性分析。
  - 迁移：Hu–Zhang 中应力子块（局部 BDM-like）+ rigid-body-mode 耦合块本身具有 off-diagonal low-rank 行为。值得做的实验：
    - 把弹性 direct 求解器从 pardiso 替成 HSS 直接解器（如 STRUMPACK / SuperLU-DIST 的 HSS 选项），在 3D / 大 NC 上看是否能突破 pardiso 内存上限。
    - paper_aux 当前 2D nx=160 pardiso 比 aux 快 11–14x，但 3D 极大概率反转；HSS 是有望的桥梁。

- **谢和虎 — PASE 扩展子空间算法 + GCGE + OpenPFEM**
  - 核心：扩展子空间多水平校正算法，配套并行软件（adaptive multigrid 的 OpenPFEM、matrix-free / vector-free 的 GCGE、并行特征值 solver PASE）。
  - 迁移：
    - fracturex 的自适应模块（`tests/aposteriori`）目前还是单层 refine；可以借鉴 OpenPFEM 的 adaptive multilevel 思路把粗/细网格的层次显式管理起来，给弹性步直接用 V-cycle / 校正子空间做 precondioner。
    - 长远：临界载荷检测（看 Hessian 半正定）需要 GEP，PASE/GCGE 是现成 candidate。

- **安恒斌 — CRS：Chebyshev-RQI 子空间 GEP 算法**
  - 核心：在 gCD 基础上把 Chebyshev filter 和 inexact RQI 联合扩张子空间，仅依赖 mat-vec，前 20 个特征值上比 gCD 快 ~3.9x 迭代 / 2.5x 时间，更鲁棒。
  - 迁移：用于稳定性分析 / 后屈曲监测 — 在 paper_aux 240 步加载过程中插入轻量 CRS，监控弹性切线矩阵最小特征对，提早判定分支 / 失稳。也适合给 Hu–Zhang 配套的混合元 GEP（对应他的"振动"问题型）。

- **梁启刚 — 自适应局部多重水平 PJD（椭圆 / Maxwell 特征值）**
  - 核心：在 AFEM 网格上做局部 smoothing 的 PJD 校正方程；mesh level 与 dof 无关的一致收敛。
  - 迁移：fracturex 已经用 AFEM，缺一个"AFEM-aware"的迭代求解器；他的局部 multilevel + PJD 框架是直接可对接的，且不需要重组全局多重网格层级。

---

## 2. 自适应与后验误差（fracturex 已有 aposteriori 框架要升级的方向）

- **李雨文 — Smoothing-based a posteriori + superconvergence（升 ★★★，零代价大收益）**
  - 关键观察：FE 残差 r_h = f - A u_h 在 finer 辅助网格上做 1–2 步 Jacobi/GS smoothing 得 G·r_h，则
    - **G·r_h ≈ u - u_h（误差函数本身的近似，不是上/下界 estimator）**。
    - ‖G·r_h‖_K 既是 estimator，又给出 (u-u_h)|_K 的具体函数形式。
  - 同样 smoother 用在高阶 FE 系统上、以低阶 FE 解为初值 → 几步 GS 后得 **superconvergent postprocessed** 解。
  - 实现"非侵入式"：等价于跑一个两层迭代解器的前几步；无需重新组装、无需解辅助系统。
  - 验证范围：Poisson、Helmholtz、Maxwell（含不定 / 非协调），都稳定。
  - 迁移到 fracturex（比"备份 estimator"重要得多）：
    1. **拿到误差函数而非标量**：fracturex 当前 aposteriori 给单元 η_K 标量，李雨文方法给 (u-u_h)(x) 近似函数。可以**直接渲染误差场**，对相场断裂的 marking 策略给出比标量 η_K 多得多的方向信息（哪里是低估、哪里是高估）。
    2. **超收敛 d 用于等值面提取**：fracturex 用 p=3 解 d；把 d_h^{p=3} 当低阶解代到 p=4 系统的 1–2 步 GS 上，得超收敛 d̃。**这条与 §6 马利敏 unfitted direct FE 直接耦合**：马利敏的方法需要可靠的 Γ = {d = 0.9} 等值面，d̃ 比 d_h 提供更准的 Γ，下游 unfitted 离散精度提升。
    3. **AFEM marking 升级**：fracturex AFEM 当前用 residual-type η_K；改成 ‖G·r_h‖_K 是 1 行代码（同样的 r_h，多 1 步 GS）。理论上更紧、对相场退化区更鲁棒。
    4. **与 fracturex 现有 GMRES 一体化**：弹性 / 相场 GMRES 在跑的时候本身就在做 Jacobi-like smoothing；李雨文的 estimator 输入正好是 GMRES 中间残差，**几乎零额外计算**。
    5. **超收敛 σ recovery**：H–Z 张量元 σ_h 的 patch-recovery 当前用 ZZ 类，可以用李雨文的"高阶系统 + 低阶初值 + smoothing"路线替代，与 fracturex 现有迭代求解器无缝。
  - 优先实验：在 paper_aux 第 240 步的 d_h 上跑 1 步 GS（在 p=4 系统上），对比 d_h 与 d̃ 的等值面位置；目测能给出比当前 d_h 更光滑的裂纹边界。

- **汪艳秋 — 曲边界 Argyris 元的 boundary value correction**
  - 核心：体拟合曲边界三角网格 + 边界值修正，恢复 biharmonic 高阶最优收敛。
  - 迁移：相场 ε∇²d-like 项 + 曲边试样（含 SEN-T、CT）几乎是直接对位的；当前 fracturex 默认假设直边/多边形边界，遇到 ASTM 标准曲边试样会掉阶。可以为 model 3+ 的 CT-shape 实验单独引入 BVC，避免大量加密边界单元。

---

## 3. 约束保持 / 能量保持 / 多物理结构（相场 d∈[0,1] + 能量耗散）

- **王冀鲁 — Projection-free Gauss methods for harmonic maps（unit-length 约束）**
  - 核心：把 CN 推到任意高阶；不投影直接保持单位长度约束、无条件离散能量耗散；提供 fixed-point 线性化变体，对约束违反量阶数可控；可大时间步算稳态。
  - 迁移：相场 d∈[0,1] 的 box 约束在工程实现里靠 active-set / 罚 / 投影；他的"projection-free + iteration 到约束机器精度"哲学可以套到 d 上：
    - 构造一组 Gauss-type 时间格式，在 alternate minimization 内层用 fixed-point 迭代去满足 d∈[0,1]，不再每步硬投影 → 避免破坏 Hu–Zhang 应力的一致性。
    - 同时利用其"无条件离散能量耗散"思路证 paper_aux 全离散能量单调；fracturex 现在有 monotonicity 数值现象但缺干净的能量证明。

- **张倩 — Energy- and helicity-conserving enriched Galerkin for NS**
  - 核心：CG⊕RT enrichment（修正法向分量）做 inf-sup 稳定，旋转型对流离散 + CN/线性化，**精确**离散能量与螺度守恒。Bernardi–Raugel 同等自由度数。
  - 迁移（弱迁移）：结构保持哲学；相场系统对应的不是 helicity，而是裂纹能量与体能量之差的单调性。她证明的"每个 Picard 迭代都保持守恒"的技术对相场 staggered 内层是直接可借鉴的：让每个 Picard step 都保持能量耗散，而不仅是终态。

- **龚伟 — Penalty + 非局部 perimeter + projected gradient（拓扑优化）**
  - 核心：非局部凸近似 perimeter；Γ-收敛证连续/离散一致；projected gradient 严格单调下降目标函数；扩展到 pointwise stress 约束的 minimum compliance。
  - 迁移：**与相场断裂极相关**。AT2/Modica-Mortola 自身就是 Γ-收敛极限框架，他的"非局部 perimeter + projected GD with strict monotone descent"几乎可以平移到 d 上：
    - 把 fracturex 内层相场最小化改成 projected gradient，保证目标函数严格单调下降（当前 alternate min 难做形式化下降证明）。
    - pointwise stress constraint 的处理方式恰好可对应 Hu–Zhang 中 σ 的逐点约束（如限制 trace σ > 0 才驱动 d，这对 split 模型非常有用）。
  - 优先级：建议作为下一轮论文的方法骨架候选之一。

- **葛志昊 — 多物理 FEM for nonlinear thermo-poroelasticity**
  - 核心：把原热-多孔弹性模型改写成 thermo-fluid 耦合形式后做 PDE 分析，全离散稳定性 + 最优收敛。
  - 迁移：相场也是多物理（弹性 + 损伤 d）。fracturex 当前的 staggered 缺少严格 well-posedness/收敛 — 可以借他的"模型重构再分析"思路：把 (u, σ, d) 系统重写为某种 saddle-point 等价形式，做全离散一致的稳定性 + 误差估计，给方法发文带来理论补强。

---

## 4. 混合元 / 张量元（Hu–Zhang 自身的算法生态）

- **黄学海 — Superconvergent & divergence-free mixed FE for Stokes**
  - 核心：H(div) 速度 + 间断压强 → pointwise div-free；用 tangential-normal continuous traceless tensor 构造弱反对称梯度算子，避免 DG/VEM 的稳定化项；得超收敛 v、p 估计。
  - 迁移：他的"traceless tensor element + 弱算子"路子和 Hu–Zhang 张量元复形是亲戚。值得对照他的实现，把 fracturex 中 σ 元的 superconvergence postprocess（patch recovery 类）做成正式 estimator；以及借他的 unified perspective 把非协调 VEM 当作另一条 path 做对照实验。

- **马睿（第二条线）— 扩展 H-Z 张量元空间 + NVB 自适应处理 L-shape 角点奇异（★★★，fracturex L-shape 困境的直接答案）**

  **核心论文**：
  - **Hu & Ma (2020)**: "Partial Relaxation of C⁰ Vertex Continuity of Stresses of Conforming Mixed Finite Elements for the Elasticity Problem", *Computational Methods in Applied Mathematics*, 2020. arXiv:1807.08090. — **这是扩展空间的定义 + AFEM 最优收敛证明 + L-shape 数值实验**。
  - Hu & Ma (2021 ESAIM:M2AN): "Quasi-optimal adaptive hybridized mixed finite element methods for linear elasticity" — 平行工作，hybridized 版本。
  - Hu-Ma-Zhang (Math. Comp. accepted 2026): biharmonic 版本，马睿 slide 后半的主线。

  **背景**：fracturex 当前用 Hu-Zhang H(div, S) 张量元处理弹性 σ-u 系统，在 L-shape 类带凹角的几何上遇到困难。L 角处解奇异 u ~ r^α（α=0.544483...）、σ ~ r^(α-1)，**Hu-Zhang 元在内部顶点强加 σ 的全 C⁰ 连续性**，这是高次混合元能在自由度上"省"的来源，但同时也是**非嵌套性的根源**：T 上的 σ ∈ Σ(T) 在细化后 T̂ 上的新顶点 x_e 处不必连续，所以 Σ(T) ⊄ Σ(T̂)，AFEM 拟正交性失效。

  **Hu-Ma (2020) 的核心机制（论文§3）**：

  设 NVB 加密 T → T̂，新顶点 x_e 是 T 的边 e 的中点（单位切向 t_e、单位法向 n_e = t_e⊥）。Hu-Zhang 元 Σ(T̂) 在 x_e 处有三个基函数（k=3 时）：

      φ_{x_e}(x) n_e n_e^T,  φ_{x_e}(x) (n_e t_e^T + t_e n_e^T),  φ_{x_e}(x) t_e t_e^T

  其中 φ_{x_e} 是节点 x_e 处的标量 Lagrange 元基函数。**扩展空间** Σ̃(T̂) := Σ(T̂) + E(T̂)，其中 E(T̂) := span{τ⁺_{x_e}, τ⁻_{x_e}}：

      τ⁺_{x_e} := φ⁺_{x_e}(x) t_e t_e^T,    τ⁻_{x_e} := φ⁻_{x_e}(x) t_e t_e^T

  φ⁺_{x_e}(x) 是 φ_{x_e}(x) 在 ω⁺_{x_e}（沿 e 的 n_e 正侧 patch）上的限制，在 ω⁻_{x_e} 上取零；φ⁻_{x_e} 反之。

  **关键观察**：
  - 把"纯切向分量基函数 φ_{x_e} t_e t_e^T"沿 e 拆成 ω⁺ 和 ω⁻ 上的两个独立基；
  - 法向相关基（n_e n_e^T 和 n_e t_e^T + t_e n_e^T）**保留不动**；
  - 故 x_e 处的全局基从 3 个变成 **4 个**；
  - **t_e^T σ t_e（纯切向分量）允许在 e 两侧不同**，但 σ n_e（法向分量）仍连续 → **H(div) conforming 性质保持**。

  **关键定理**：
  1. **嵌套性（Thm 3.2）**：T → T̂ 是 NVB 加密 ⟹ Σ̃(T) ⊂ Σ̃(T̂)。证明：T 上的 σ 在新顶点 x_e 处其纯切向分量是不连续的，正好可以由 τ⁺_{x_e}, τ⁻_{x_e} 表示；法向连续性自动满足。
  2. **稳定性 + 最优阶（Thm 3.1）**：(3.3) 有唯一解，且 ‖σ - σ_h‖_{H(div)} + ‖u - u_h‖_0 ≤ C h^k (‖σ‖_{k+1} + ‖u‖_k)。
  3. **可靠性 + 高效性（Thm 3.3）**：residual-based estimator η²(T̂) := Σ_K [h_K^4‖curl curl(Aσ_h)‖² + h_K‖[Aσ_h t_e · t_e]_e‖² + h_K^3‖[curl(Aσ_h) · t_e]_e‖²] 同时是 reliable 和 efficient。
  4. **AFEM 最优收敛**（拟正交性 Thm 3.5 + 离散可靠性 + axioms of adaptivity）：标准 Dörfler 标记 + NVB 细化 → 最优代数收敛阶。

  **L-shape 数值结果（§5.2, §5.5）**：

  Rotated L-shape Ω, 解 u_r, u_φ in polar coordinates with α = 0.544483736782, ω = 3π/4, E = 10^5, ν = 0.499（接近不可压）。
  - **§5.2 uniform mesh + corner relaxation**: 表 5.1 显示，在 origin x_c 处 relax C⁰ 后，‖σ - σ_h‖_A 从 7.5e-3 降到 3.1e-3（首次 refinement），收敛阶从 ~0.5 略有提升；‖u - u_h‖_0 阶从 ~1.0 提到 ~1.2。
  - **§5.5 adaptive (extended element vs original)**: 图 5.5 显示 AFEM 下两个元（原始 Hu-Zhang 元 vs 扩展元）**收敛曲线非常接近**，**都达到最优代数阶 -1**（图上 reference line 斜率 1:1 vs #dofs 即 -1/2 vs h）；但**只有扩展元有理论保证**（拟正交性需要 nestedness）。

  **§4 角点处理（boundary corner）**：

  L-shape 真正的"凹角难题"还在于 ΓN 上若有 inconsistent traction（即 n_+^T g|e_+(x_c) ≠ n_-^T g|e_-(x_c)）时，原 Hu-Zhang 元在 x_c 的 3 个 DoF 不够表示。Hu-Ma §4.1 的做法：
  - 把 x_c 所在三角形 K 一分为二（K_+ ∪ K_-，公共边 e），
  - 沿公共边 e relax 纯切向分量连续性，
  - x_c 处变成 **4 个 DoF**（图 4.1d）：两个标准的 n_e n_e^T、(n_e t_e^T + t_e n_e^T) 型 + 两个分裂的 t_e^T τ t_e（在 K_+、K_-）。
  - 4 个新基函数的具体形式见论文 (4.2)，常数 c_i, d_i 由法向跨 e 的连续性条件解出。

  Cook's membrane 实测：corner relax 后 σ 误差在粗网格上**显著下降**（粗网格上 boundary error 主导）。

  **对 fracturex 的迁移（直接答案）**：

  1. **改动局部化**：H-Z 元的实现不动，只在两处加 DoF：
     - **AFEM 路径**：所有 NVB 新顶点 x_e ∈ V_0(T̂) \ V(T_0) 处，加 2 个基 τ⁺_{x_e}, τ⁻_{x_e}（纯切向，沿 coarse edge e 拆分）；
     - **L-shape 凹角**（如果在域内部，是 L 角；如果在 ΓN，是 traction inconsistency 角）：把角点所在三角形二分，沿公共边 relax 切向连续性，按 (4.2) 加 4 个 DoF。
  2. **DoF 计数**：dim Σ̃(T̂) = dim Σ(T̂) + |V_0(T̂) \ V(T_0)|，每个新内部顶点 +1（k 阶元的纯切向 Lagrange 基为标量，故只 +1，论文摘要确认）。
  3. **嵌套性保证**：AFEM 拟正交性证明可平移到 fracturex 的 H-Z 元（k=3 paper_aux 直接对应）。
  4. **对相场断裂的额外意义**：
     - 初始 V/L 形预切口处 σ ~ r^(-1/2) 比 L-shape 的 r^(-0.456) 更强；
     - 完全断裂区 d ≈ 1 + 未损伤区 d ≈ 0 的界面 ↔ 高对比 g(d) ↔ 与 §6 马利敏 unfitted direct FE 配合，扩展元在裂纹尖端 + unfitted 在界面是双重保险。

  **优先实验路线**：
  1. 在 fealpy 上先复现 Hu-Ma §5.2：rotated L-shape, α=0.544..., 不带相场。验证扩展空间 + NVB AFEM 收敛阶达到 1。
  2. 把 corner relaxation 实装到 H-Z 元装配模块（fealpy `fracturex/.../huzhang_*.py` 中新增 vertex DoF 分裂逻辑）。
  3. 嵌回 fracturex paper_aux，跑 model2 + L-shape 几何或 V-notch 几何，对照当前次最优收敛。
  4. 论文方向：把 Hu-Ma 扩展空间 + AFEM 推广到**含相场退化的 H-Z 元**——这是 Hu-Ma 没做过的，且与 fracturex 直接对位。

  **重要细节**：
  - 论文要求 k ≥ 3（paper_aux 用 p=3 正好满足）。
  - estimator (3.5) 用的是 curl-curl 型 residual，比标准 H(div) 椭圆问题 estimator 多了 curl-curl 项——这是 H-R 公式的特性，fracturex 当前 estimator 需要核对是否已用此型。
  - displacement 空间 V(T̂) 是 full discontinuous P_{k-1}（论文 (2.7)），fracturex 也应是这个。

---

- **马睿（第一条线） — Body-plate mixed/primal 耦合（与 fracturex 直接同源，重点）**
  - 核心：3D 弹性体用 Hellinger–Reissner（σ 作辅助变量，**正是 Hu–Zhang 张量元**，σ ∈ H(div, S)、u ∈ L²）；Kirchhoff 板用位移主元（w ∈ H²，C¹ 板元）。两套子问题在板中线面 Γ 上通过 **Lagrange multiplier λ 弱施加 σ·n 与板的弯矩 / 剪力对接**；网格允许在 Γ 上完全非匹配。
  - 离散方面给了两套：
    - conforming：H–Z σ 元 + Argyris / Bell 类板元 + 在 Γ 上的乘子空间，证 inf-sup 与最优阶。
    - nonconforming：板元退到 Morley 等低阶非协调元，同样给了 inf-sup，对工程更实用。
  - 接口耦合矩阵是 σ·n 与 λ 在 Γ 上的 ∫，**完全不依赖两侧网格匹配** —— 这是关键。
  - 迁移到 fracturex（这条强相关，建议升 ★★★）：
    - **核心观察**：fracturex 当前所有 u、σ、d 共用一张网格。但物理上 d 只在裂纹带（宽 ~l₀）需要细，绝大部分体单元 d≡0；u/σ 在远场也需要相对细以解析应力波。强行让 d 与 u 同细 → DoF 浪费在远场的 d 上；反过来让 u 与 d 同粗 → 应力解析掉阶。
    - **马睿的路径直接套用**：把 (σ, u) Hu–Zhang 子问题放在一套细网格 𝒯ₕ，把 d 子问题放在另一套粗网格 𝒯_H（或反之，按 active set 分），两套网格独立 refine，仅在过渡界面 Γ（取 d 某等值面 / active set 边界）用 Lagrange multiplier 耦合 σ·n 与 d 的 trace。
    - **稳定性可平移**：他的 inf-sup 证明里乘子空间的构造对 H–Z + 跨界面的对位条件成立，移植到 fracturex 只需把"板"替成"d 的子问题"；fracturex 中 d 是 H¹ scalar，比 Kirchhoff 板的 H² 还简单，inf-sup 反而更容易。
    - **AFEM 友好**：弹性单元和 d 单元可以独立加密，配 §2 的 a posteriori 各自给一个 estimator，不必互相牵制。这是把当前自适应模块的瓶颈解开的关键。
    - **paper_aux 量化对照**：当前 nx=160 全网格 NC=51200，d≪1 区域占比目测 > 80%。若 d 走粗网格（如 nx=40），d 子问题 DoF 直接降 16×，对 240 步加载下的相场 GMRES 是数量级收益。
  - 优先实验：先在 model2 上做一版"u/σ 细网格 + d 粗网格 + Γ 取固定矩形带"的简化耦合，量化 vs paper_aux 的总时间与裂纹路径误差。

- **徐岩 — 高阶 structure-preserving mimetic difference**
  - 核心：把高阶 mimetic 算子分解成 2 阶 mimetic + 简单结构算子的组合，配套 reduction operator，构造与连续 de Rham 复形交换的 mimetic de Rham 复形；天然保 curl-free / div-free。
  - 迁移：作为 Hu–Zhang 的"差分版竞品 / 比对"——对 model2 这种规则矩形/笛卡尔背景试样，mimetic FD 可以是更轻量的 baseline，方便验证 H-Z 数值结果。

- **陈刚 — Nonsymmetric Nitsche on convex polytopes 的统一估计**
  - 核心：penalty 全范围、regularity-dependent 估计；证 H^{3/2} 阈值是 sharp。
  - 迁移：fracturex 边界条件以 Dirichlet 为主，若移到 Nitsche 弱实施（避免 H-Z 上强加约束的复杂度），他的估计直接给阶数指导，避免选错 penalty。

---

## 5. 数据驱动 / 神经网络相关（fracturex 已有 operator_learning 子方向）

- **蔡智强 — Structure-guided Gauss-Newton (SgGN) + damped block Newton (dBN) for shallow ReLU NN（升 ★★★，与 fracturex 强相关）**
  - 关键观察：shallow ReLU NN 训练等价于带"分段线性 + 自由结点"的 free-knot spline 拟合。传统 Adam 在求解 PDE 类问题上对 1D-singular / 高对比解需要 2 万 + 步，且常陷入 local min（断点位置错）。
  - SgGN 的"structure-guided"是什么：
    - 把 shallow NN 的参数显式分为**线性外层 c** 与**非线性内层 r**（断点位置 + 方向），用 separable nonlinear LS 的 VarPro / 交替框架。
    - 质量矩阵 A(r)、Gauss-Newton 矩阵 G(r)、layer-GN 矩阵都是 **SPD/SPSD 但极度病态**；他在层级结构下显式构造各自的 inverse 近似，让 GN/Newton 在病态下仍稳定收敛。
    - LMA 退化为"非 GN 方向 + tiny damping"；SgGN 不需要 damping。
  - 数值结果：piecewise-constant / step function、ut + ux = 0 在 100 步 SgGN 收敛，对照 Adam 2 万步还差很多；1D 奇异解 u = x^{2/3}（H^{1+1/6}）也能精确捕捉断点。
  - 迁移到 fracturex（远不止"训练 trick"）：
    - **NN-表征 d 场的训练求解器换成 SgGN**：当前 operator_learning 路线如果走"NN 表示 d_h"路线（裂纹尖端 d 形状是奇异的，正是 free-knot 该解决的），shallow NN + SgGN 就是天然 fit，能用 100 步以内得到 d 的精确分段表示。
    - **相场子问题的内层非凸最小化也是 separable LS 结构**：alternate min 的 d 子问题在固定 u 下是凸的，但 staggered 全局问题非凸；可以借 SgGN 的"显式分块 + 病态 SPD 的结构化解法"作为内层求解器，比当前 GMRES no-precond 更稳。
    - **与 §1 罗力 data-driven 一致的哲学**：都是"分析结构 → 显式构造结构化 inverse"，而不是黑箱迭代。SgGN 给罗力的非线性 NN preconditioner 一个更可靠的训练后端。
    - **裂纹尖端奇异性**：他证 H^{1+α} 解 shallow NN + SgGN 收敛阶最优；fracturex 中 d 在尖端有 H^{1+α} 型奇异，理论上直接对应。

- **何俊材 — 线性/非线性 NN 逼近的 adaptive FEM 视角**
  - 核心：经典 FE 空间用 NN 表示；用 integral representation + Radon-BV 给 random feature 最优逼近率；含 divergence-free vector field 应用。
  - 迁移：背景理论支撑——给 fracturex 后续"NN 表示 d 场 / σ 场"做收敛率参考；他 divergence-free vector field 的 random feature 构造，可以直接试 σ 满足平衡方程 (div σ = -f) 的约束。

- **贺巧琳 — HPIHNN: Physics-informed holomorphic NN + 传统数值**
  - 核心：线性叠加分解（particular + homogeneous）；particular 用谱方法在简单扩展域上算（避开复杂边界 meshing），homogeneous 用 PIHNN 仅在边界上训练满足修正边界条件；维度降为边界维。在 Poisson / Stokes / biharmonic 上验证含角点奇异性问题。
  - 迁移：复杂裂纹几何的相场可以分裂：
    - particular = 在矩形 bounding box 上谱解平衡方程；
    - homogeneous = 用 PIHNN 在裂纹边界上做修正。
    - 对应 fracturex 3D 大试样 + 不规则初始缺陷的场景，有降维潜力，避免每步重 meshing。

- **徐一峰 — Computing the p-Laplacian by AFEM and dual variational NN**
  - 核心：AFEM 协调 + Crouzeix-Raviart 非协调收敛分析；DVNN 通过 Helmholtz 分解把原问题拆成线性 Poisson + 散度自由空间上的无约束极小化（curl 向量势），用 PINN / Deep Ritz 处理。
  - 迁移：相场 g(d)∇u 的退化扩散 与 p-Laplace 在退化结构上同源；他的 AFEM 收敛证 + DVNN 拆分两条线都给相场提供方法论参考。优先看 AFEM 那条 — 直接对应 fracturex 现有自适应模块。

- **明平兵 — DL for Schrödinger 谱**
  - 核心：损失 + 架构联合设计，sine-Barron 空间下分析泛化误差，能算前 30 个特征值。
  - 迁移：偏理论；若 fracturex 走"特征值监测分支判定"+ NN 加速这条路，可参考其 a-prior 结构嵌入网络的做法。

---

## 6. 复杂几何 / 界面 / 边界处理

- **马利敏 — Direct FE for elliptic interface problems on unfitted meshes（与 fracturex 强相关，重点）**
  - 核心思想：考虑 −∇·(β∇u) = f 含界面 Γ、β 跨 Γ 跳变。**经典做法**把两个界面条件 [u]=0 与 [β∂u/∂n]=0 当 jump 显式处理（Nitsche / XFEM / IFEM 各种罚项 / ghost penalty）。**马利敏的做法**：把这两个条件视作子域 1 与子域 2 之间的"对偶对"——
    - 子域 1：u₁ 用 conforming H¹ 元（primal）。
    - 子域 2：(σ₂, u₂) 用 conforming mixed FE，σ₂ = β₂∇u₂ ∈ H(div)（即 RT₀ 或更高阶）。
    - 界面 Γ 上自然出现 ∫_Γ u₁ (σ₂ · n) ds 这一耦合项，**不需要罚参数、不需要 ghost penalty、不需要稳定化**。
  - **unfitted**：网格不必与 Γ 对齐，Γ 任意切割单元。
  - **β-跨数量级鲁棒**：误差估计与 β₁/β₂ 比值无关，β₁/β₂ ∈ [10⁻⁶, 10⁶] 的数值实验稳定。
  - 最低阶实现：P1 + RT₀，代码量极小，几乎是把现有 conforming FE 加一段界面积分。
  - 迁移到 fracturex（强相关）：
    - **物理对应**：相场 g(d)∇u 退化扩散，未损伤区 g≈1 与完全损伤区 g≈ε 之间的过渡正是 **高对比椭圆界面问题**。取等值面 Γ = {x : d(x) = d*}（如 d*=0.9 或 0.95）作为离散"裂纹界面"。
    - **直接套用**：未损伤子域 Ω₁ 用 primal CG（保持当前 fracturex 弹性的 H–Z primal/dual 结构），损伤子域 Ω₂ 用 mixed (σ, u)；Γ 上自然项 ∫_Γ u₁(σ₂·n) 自动处理对位。
    - **关键好处**：
      1. **不需要 mesh 跟着裂纹走** —— 一直用规则背景网格（甚至笛卡尔），界面切过单元内部即可。
      2. **不需要罚参数** —— 比 Nitsche / 加 stab 的方案少一个调参，工程鲁棒。
      3. **β-鲁棒** —— g(d) 从 1 到 ε 跨 6 个数量级正是他验证过的工况；现有 fracturex 在 d→1 接近断裂瞬间出现的弹性 GMRES 病态，这条路有可能根除。
    - **与 §1 求解器互动**：unfitted direct FE 给出的离散系统块结构清晰（primal-mixed 两个块 + 界面对偶项），可与胡齐芽两层 Schwarz 的子域划分天然契合——子域 1（未损伤）做 P1-FE 块，子域 2（损伤）做混合元块，粗空间在界面附近自适应增维。
    - **与 §2 自适应互动**：界面 Γ 动态移动（载荷推进时 d 等值面前进），切割单元集合每步更新；可与 fracturex 现有 AFEM 配合，在 Γ 附近重新切割而不重新 remesh。
  - 优先实验：取 model1 单边切口拉伸，固定 Γ = {d > 0.9} 边界，用 P1 + RT₀ 的最低阶 direct FE 与 paper_aux baseline 对比裂纹路径与峰值载荷。若可行再扩到 model2。

- **应文俊 — Cartesian-grid BIM for acoustic scattering**
  - 核心：传统 BIM 思想 + 笛卡尔网格快速求解器，避开直接 boundary/volume integral。
  - 迁移：弱相关；若走 Cartesian-grid 相场（避免非结构 mesh 在 3D 的开销），这条路是参考。

- **杜宇 — Discrete PML for peridynamic scalar waves**
  - 核心：在 quadrature-based FD 上构造离散 PML，避开 nonlocal 算子带来的核函数难题。
  - 迁移：仅在 fracturex 走"动态 / 波致断裂"分支时相关；当前静态加载场景几乎无关。

- **鲁汪涛 — PML for step-like surface scattering**
  - 与 fracturex 无明显交集。

---

## 7. 优先级建议（按"对 fracturex 短期收益"排序）

| 优先级 | 报告 | 方向 | 投入 vs 收益 |
| --- | --- | --- | --- |
| ★★★ | 马睿（线 1） | H–Z 弹性细网格 + d 粗网格的 Lagrange-multiplier 非匹配耦合 | 中 — 直接用同源 H–Z 张量元，inf-sup 证明可平移，自由度数量级收益 |
| ★★★ | 马睿（线 2）/ Hu-Ma 2021 | 扩展 H-Z 空间 + NVB 处理 L-shape / 切口角点奇异（**fracturex 当前 L-shape 困境的直接答案**） | 中 — 不改 H-Z 元本身，只需 NVB + 在新顶点放松 t_e^T σ t_e 连续性 |
| ★★★ | 马利敏 | unfitted direct FE 处理 g(d) 高对比界面（损伤区 vs 未损伤区） | 中 — 不要罚参数、不要 remesh，β-鲁棒，与 paper_aux 一致网格仍可用 |
| ★★★ | 罗力 | data-driven preconditioner（弹性 GMRES + 非线性 Newton 初值） | 轻 — 接现成 GMRES 框架，可直接对 paper_aux 240 步做离线 PCA 实验 |
| ★★★ | 龚伟 | projected gradient + 非局部 perimeter（相场 d 求解器替换 + 单调下降证明） | 中 — 需要改 alternate min，但给论文一个"严格单调下降"的卖点 |
| ★★★ | 胡齐芽 | 两层杂交加权 Schwarz + 谱粗空间（弹性 step 在高对比 g(d) 上的 precond） | 中 — 已有 DD 子域划分，加 GEP-coarse 即可；可与马利敏 unfitted direct FE 子域划分天然契合 |
| ★★★ | 李雨文 | smoothing-based 误差函数 + 超收敛 d̃（给 AFEM marking、马利敏 Γ 提取、σ recovery 共同提供基础设施） | 轻 — 1 步 GS 接现有 GMRES 中间残差，几乎零成本 |
| ★★★ | 蔡智强 | SgGN / dBN —— shallow NN 求解 PDE 的结构化 Newton（NN 表征 d 或非线性内层求解器） | 中 — operator_learning 路线后端替换；裂纹尖端 H^{1+α} 奇异是 free-knot spline 天然 fit |
| ★★ | 王冀鲁 | projection-free Gauss for d∈[0,1] 约束（高阶 + 能量耗散） | 中 — 改时间步格式，需理论配套 |
| ★★ | 谢和虎 / 梁启刚 | AFEM 上的局部多重水平校正（替代或加速现有自适应求解链） | 中 — 需要梳理 fracturex 的层级数据结构 |
| ★ | 夏建林 | rank-structured 直接解器（替 pardiso，跑 3D） | 中-重 — 接 STRUMPACK 之类 |
| ★ | 葛志昊 | 多物理重构 + 全离散稳定性分析 | 重 — 偏理论补强 |
| ★ | 贺巧琳 | particular + homogeneous 分离 + spectral/NN | 重 — 适合后续 3D 论文 |
| ★ | 徐一峰 | p-Laplace AFEM 收敛证 | 轻-中 — 理论引用 |
| ★ | 汪艳秋 | 曲边界 BVC | 轻 — 仅在做 ASTM 曲边试样时启用 |

---

## 8. 落到后续论文 / 开发的具体动作

1. **下一篇方法论文骨架候选（短期 / 中期）**
   - **候选 A（求解器线）**：罗力 data-driven preconditioner + 龚伟 projected gradient with Γ-convergence，合并到 fracturex paper_aux baseline 上；量化 240 步加载的总时间与迭代数，并给相场子问题严格单调下降证明。卖点：data-driven precond 在工程相场断裂上的首次系统验证 + 单调下降离散性证明。
   - **候选 B（离散化线，更与 H–Z 同源）**：马睿"H–Z 弹性细网格 + d 粗网格 + Lagrange multiplier 非匹配耦合" + 马利敏"unfitted direct FE 处理 g(d) 高对比界面"。卖点：
     - 把当前 fracturex u/σ/d 共用一张网格的瓶颈打开（DoF 数量级下降）；
     - 不再依赖 g(d) 退化区域的人工 mesh refinement，直接 unfitted；
     - inf-sup + β-鲁棒误差估计两套都已现成，仅需平移到相场场景。
   - 候选 B 与 fracturex 当前路线（H–Z + 自适应）契合度更高，建议优先调研。

2. **求解器实验路线（中期）**
   - 在 `fracturex/tests/preconditioner/` 增 `data_driven_gmres` 子模块，离线收集 paper_aux 的 Krylov 残差做 PCA → 上线 deflation。
   - 评估 STRUMPACK HSS 替代 pardiso 在 3D model2 上的内存 / 时间（夏建林）。
   - 实现 GEP-coarse 两层 Schwarz 作为 elastic step 第三档 preconditioner（胡齐芽）。

3. **自适应模块升级（中期）**
   - 把 `tests/aposteriori` 的 marking 输入从 residual η_K 升级为李雨文的 ‖G·r_h‖_K（1 行：跑 1 步 GS）。同时把 G·r_h 作为误差函数缓存下来，给可视化与诊断用。
   - 用李雨文超收敛后处理输出 d̃ p=4，作为 §6 马利敏 unfitted direct FE 的 Γ = {d̃ = 0.9} 等值面提取输入；形成"AFEM → 误差函数 → 超收敛 d̃ → unfitted Γ → 下一步求解"的闭环。
   - 把当前单层 AFEM 改成 OpenPFEM 风格的多层管理（谢和虎），把 PJD 局部 multilevel 作为 elastic step 的 AFEM-aware preconditioner（梁启刚）。

4. **理论补强（长期 / 与合作者）**
   - 葛志昊式的"模型重构 + 全离散稳定性"——把 (u, σ, d) Hu–Zhang 相场系统重写并给全离散误差估计。
   - 龚伟式的 Γ-收敛 + projected gradient monotone descent 框架。

5. **3D / 复杂几何分支（长期）**
   - 贺巧琳 HPIHNN 思路：particular（谱）+ homogeneous（PIHNN）分离做大试样。
   - 马睿 H-R + primal 非匹配耦合 → u-细 / d-粗 网格。

---

附：会议手册原始摘要文件位置 `/Users/tian00/Desktop/gong办公资料/202606科学计算会议/old/会议手册草稿（外部）V3.3 - 副本.docx`；slide 目录 `slide/{6_25,6_26,6_27}ppt/`。
