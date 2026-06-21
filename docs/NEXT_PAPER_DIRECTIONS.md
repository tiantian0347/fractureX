# 下一步论文方向建议（基于 `Tian/paper/fracture/READING_NOTES.md` × `fractureX` 现状）

> 状态：建议稿 v0.1（2026-06-18）。
> 来源：综合 `Tian/paper/fracture/READING_NOTES.md`（21 篇 PDF 五段式摘要 + point 库）与 `fractureX/docs/` 已有四条路线
> （architecture、operator_learning、preconditioner、routes/adaptive）。
> 目标：在已有代码基础设施（Hu–Zhang + 相场交错、aux-space 预条件、算子学习数据管线、平衡型 a posteriori 估计子）之上，
> 给出 **方向选择 → 文献钩子 → 与现仓库的对接点 → Milestone 草案 → 可发表期刊** 的清单。
>
> 用法：每条候选方向独立可起，优先级排序见 §0；不打算并行铺所有线，只挑 1–2 条作为下一阶段主攻。

---

## 0. 一句话推荐

按「与现有代码的协同度 × 卖点稀缺度 × 6 个月可写成稿」排序，推荐顺序：

1. **A. 平衡型 a posteriori + σ-驱动自适应** —— 已经有 `docs/routes/plan_adaptive_aposteriori.md` 和 `docs/adaptive/THEORY_equilibrated_aposteriori.md`，理论与代码骨架最齐，**直接接着写**。
2. **B. 平衡保持算子学习（HZ-supervised neural operator）+ Stage D 平衡残差正则** —— `docs/operator_learning/paper_thesis.md` 已把主定理链 B/T2/T3 锁定，欠的是「Stage D 损失实现 + Hu-Zhang vs 恢复应力 σ 对照表」。
3. **C. 高阶（4 阶）+ 应变梯度 PFM 在 Hu–Zhang 应力路径上的扩展** —— READING_NOTES point 31 直接对接，是「为什么混合元值得付的代价」的第二条战线，但理论改造较多，作为 B 之后的延伸。
4. **D. 块预条件论文 D12 收尾 + 「难 regime 唯一收敛」头条** —— `docs/preconditioner/D12_PRECONDITIONER_PAPER_PLAN.md` 已锁定头条措辞，是「短平快补写一篇 NLA 向」的备选。
5. **E.（探索）软-硬复合 BTD + 可微 HZ 相场拓扑韧化** —— Fucheng Tian PNAS 2025 ([14][15]) × AuTO ([25])，是高远但非短期。

下文每条单独展开。

---

## A. 平衡型 a posteriori + σ-驱动自适应（**首推**）

### A.1 论点

> *Hu–Zhang 应力 σ_h ∈ H(div; 𝕊) 直接给出无常数、reconstruction-free、保证型超圆界 (Prager–Synge)；
> 用它驱动 Dörfler 标记 + h-加密，把 DOF 精确堆到裂纹带 / 裂尖，治 `surrogate_data_underresolved_hl0`
> 揭示的 h/l₀≈5 欠分辨痛点；并把「为什么值得付 Hu–Zhang 的代价」做成方法论卖点。*

### A.2 文献钩子（来自 READING_NOTES）

- **point 14**：Verfürth 残差估计子可靠 + 有效 / Braess–Schöberl equilibrated flux —— 标准对照轴。
- **point 24** + **[24] p4est**：自适应加密的并行基础设施；2D 写完后 3D 升级直接挂 p4est。
- **point 23 / [10] 郭雯 2024**：四叉树自适应 + 双相场，FRC 算例 CPU 省 56%、网格省 89% —— **直接 benchmark 对照**。
- **point 13 / [03] Ambati Hybrid**：staggered 收敛判据 → §M3 加载步内部收敛准则可直接引。
- **point 16 / [05] Cervera 2021**：PFM 2D 网格量 ~ mixed 的 100×，**这一倍数正是自适应能省回多少的"理论天花板"**，写动机最有说服力。
- **point 17 / [20][21][22] Cervera mixed FEM + OSGS**：「无 tracking 的 mesh-objective 失效带」与「自适应 + HZ」是同一价值主张的两端，可在 §1 motivation 里并列引用、强调本工作走 HZ 路径的差异（**变分严谨 + 平衡可证**）。

### A.3 与现仓库对接

| 模块 | 文件 | 用途 |
|---|---|---|
| 现成 | `docs/adaptive/THEORY_equilibrated_aposteriori.md` v0.2 | Theorem 1/2、$\eta_T$ 公式、$k$-依赖分析 |
| 现成 | `docs/adaptive/DECISION_sigma_driven_adaptivity.md` | σ-驱动 vs $\eta_T$-驱动的分工已写定 |
| 现成 | `docs/adaptive/THEORY_marking_strategy.md` | M-DF 主驱动标记策略 |
| 现成 | `fracturex/adaptivity/equilibrated_estimator.py` | $\eta_T$ 实现入口 |
| 现成 | `fracturex/adaptivity/adaptive_loop_equilibrated.py` | 自适应主循环 |
| 现成 | `fracturex/adaptivity/primal_elastic_solve.py` + `degraded_huzhang_solve.py` | 标准 FEM + Hu–Zhang 双解装配 |
| 待补 | hp-自适应（光滑远场 p↑、裂纹带 h↑） | READING_NOTES point 7（高阶 PFM mesh budget）+ 9（应变梯度）→ A 之延伸 |
| 待补 | p4est 并行接入（3D 章节） | 当前在 deal.II 生态，fealpy 侧需写 backend 桥；READING_NOTES point 24 直接给出 z-order/2:1 balance 抽象 |

### A.4 Milestones（接着 `plan_adaptive_aposteriori.md` 写）

- **M0–M3 已在原 plan**，**新增**：
  - **M4：与 [10] 郭雯 2024 四叉树双相场对照**。同算例（FRC 30/45/60/90°）跑「无加密 vs $\eta_T$-驱动加密 vs 启发式 d-梯度加密」三档，报 CPU 节省、网格节省、裂纹路径 Hausdorff；目标超过郭雯的 56.17% / 89.47%（**他没有 guaranteed estimator**，这是结构性差距）。
  - **M5：3D 起步（小规模）**。在 model0 / model2 的 3D 版本上证「2D 论点不退化」，**不**做大算例（留 D 篇）。

### A.5 期刊 / 篇幅定位

- 首选：**CMAME** 或 **SINUM**（保证型估计子 + 工程算例，两边都吃）。
- 备选：**IJNAMG**、**Comput. Mech.**。

### A.6 风险

- M0 门槛：split 情形 $\Theta$ 在合理 $k_{\rm res}$ 下是否有界。Plan 已经把这条作为 go/no-go。
- $\eta_T$ 需要额外一次标准 FEM 位移解 —— 用 Stenberg 后处理消掉这一步是潜在加分项，但也可能是审稿人攻击点（"既然要再解一次，为何不直接 Braess–Schöberl 重构"）。**对策**：在论文中明确列「直接拿 vs 重构」的 effectivity + 成本曲线（M3 已规划）。

---

## B. 平衡保持算子学习（HZ-supervised neural operator） + Stage D 平衡正则

> **注意：B ≠ D13（已封存）**。D13（[`docs/preconditioner/D13_LEARNED_PRECONDITIONER_PAPER_PLAN.md`](preconditioner/D13_LEARNED_PRECONDITIONER_PAPER_PLAN.md)）
> 学习的是**辅助空间粗空间的延拓算子** $[PI_s\mid\Phi_\theta]$，**嵌进 GMRES 内层**，目标是把局部化 regime 的迭代数 O(100)→O(10)；
> 因「让 solve 更快/更省」对 2D Hu–Zhang 价值耗尽（`aux_loses_to_pardiso_2d`，见
> [`plan_adaptive_aposteriori.md` §0](routes/plan_adaptive_aposteriori.md)），已封存。
>
> 本节 §B 学习的是**整套 PDE 解算子** $\mathcal G:(几何,材料,载荷)\mapsto (d,\sigma)$ 的网格代理 $\hat{\mathcal G}_\theta$（FNO / U-Net / DeepONet），
> **完全不进求解器**——一次前向直接替换整个 staggered 外环，性能口径是 wall-clock ~5000×（不是 niter）。Hu–Zhang 在这里是**监督源**
> （提供 $\sigma_h\in H(\mathrm{div};\mathbb S)$ 的高质量数据），不是被求解的对象。`paper_thesis.md` 主定理链 B/T2/T3 已锁定（2026-06-02），主线在跑。
>
> 两者除了「都用神经网络 + 都基于 Hu–Zhang」之外，没有结构性重叠，故 D13 封存**不**牵连 §B。

### B.1 论点

`docs/operator_learning/paper_thesis.md` 已锁定（2026-06-02），收紧版主定理：

> *在本文设定与数值假设下，Hu–Zhang 平衡监督配合平衡正则，是实现低平衡残差代理的**结构最优方案**；
> 任何基于位移 FEM 微分恢复应力 σ_h^rec 的监督，其平衡残差被数据固有缺陷 Θ(h^m) 不可约地下界锁死，
> 且平衡正则与数据拟合相冲突。诚实边界：逐点裂尖应力峰为不可稳定恢复的残余难量。*

### B.2 文献钩子

- **point 28 / [03]**：「PFM 与 gradient-enhanced damage 同源」—— 给「平衡残差 R(σ̂) 是物理一致性指标」找了 CDM 侧的支撑。
- **point 29 / [14][15][25]**：可微 PFM + Heaviside 失效 + JAX-AD 拓扑韧化 —— **B 的下一步可以是 E**。
- **point 20 / [25] AuTO**：JAX/AD 把伴随门槛降到普通用户，是 Stage D 损失的实现路径。
- **point 5 / [08]**：动态 PFM 的「模型依赖性」缺陷 —— 提示 σ 监督代理可作为动态 PFM 跨模型一致性的诊断器。
- **point 12**：staggered vs monolithic —— rollout 训练时如何处理 staggered 迭代内部不收敛步，可借鉴 Ambati 收敛判据。
- **point 22 / [14][15] SH-com**：Heaviside 失效 + 线弹性最小模型 —— Stage E（可微 PFM → 拓扑韧化）的最简数值原型，预测有限差分 + 掩码 + 求积的 R̃_h 与 SH-com 弹簧网络架构天然兼容。

### B.3 与现仓库对接

| 模块 | 文件 | 现状 |
|---|---|---|
| 主线锁定 | `docs/operator_learning/paper_thesis.md` | B/T2/T3 定理链已写完 |
| 数据 schema | `docs/operator_learning/SURROGATE_DATA_SCHEMA.md` v0.1 | σ 已按 `stress_scale` 归一 |
| 数据生成 | `fracturex/learn/datasets.py` | M1 pilot 27 样本 → M2 S 档 ~1152 样本 |
| 模型 | `fracturex/learn/models/` | multioutput_fno / multioutput_unet 已实现 |
| 训练 | `fracturex/learn/train.py` | Stage A/B/C 已通 |
| **待补** | **Stage D 平衡损失 `equilibrium_residual_l2`** | 当前 `metrics.py` 仍是 `NotImplementedError`（paper_thesis §"诚实清单" #2） |
| **待补** | **对照实验：σ_h 监督 vs σ_h^rec 监督** | paper_thesis §"诚实清单" #3，命题 B2 + T2 的实验证据 |
| **待补** | **数据重生成满足 h_FE ≤ l₀/2** | paper_thesis §"诚实清单" #1，[[surrogate_data_underresolved_hl0]] |
| 延伸 | `fracturex/utilfuc/vtk_lagrange_writer.py` | 已能高阶采样，可直接给 §I (Airy 势) 增强方案做可视化 |

### B.4 Milestones（接着 `plan_operator_learning.md` 走）

- **M3a：Stage D 实现 + 头牌指标 R̃_h 测量**（2 周）
  - 实现 `equilibrium_residual_l2(sigma_grid, mask, dx, dy, f=0.0, d=None, d_c=0.9)` （paper_thesis §C）
  - 在已有 M2 Stage B 检查点上**只评估**，先证「现有 σ̂ 的 R̃_h 离 σ_h 的 R̃_h 多远」
- **M3b：σ_h 监督 vs σ_h^rec 监督对照表**（2 周）
  - 同架构同数据量；预期 HZ 组 R̃_h(σ̂) → ε（随 Stage D / 数据量降）、rec 组停在 ~δ_rec=Θ(h^m) 平台
  - **这张「平台 vs 下降」曲线就是命题 B2 + T2 的实验证据**（paper_thesis §F.4）
- **M3c：数据重生成 + h_FE ≤ l₀/2 后峰幅值与 D1/T1 闭环**（3 周）
  - 验证 T1c 预言：修数据后峰幅值升、但仍因「裂尖定位不确定 Δ」饱和
- **M4（可选）：§I Airy 势硬约束实现**作为 ablation 头牌
  - 不是基础定理，但作为「实现层的最强配方」给一张图（R(σ̂) 严格 = 0 vs Stage D 软罚收敛曲线）

### B.5 期刊定位

- 首选：**JCP** 或 **CMAME**（计算数学 + AI4Science 交叉）。
- 备选：**SISC**、**Computer Methods in Applied Mechanics and Engineering**、**Engineering Fracture Mechanics**（如果工程量算例足够强）。

### B.6 风险

- M3a 的 R̃_h 在裂纹带差分放大已知（paper_thesis §C），需用 `d_c=0.9` 掩码 + 全域/起裂前区双口径报告。
- §I (Airy) 二阶微分放大噪声，可能与 §T1 正交（B2 修平衡、不修分辨率），需写清「平衡 + 富集」组合。

---

## C. 高阶（4 阶）PFM + 应变梯度 / Hu–Zhang 应力的耦合（延伸方向）

### C.1 论点

> *Borden 2014 / Ali 2024 ([06]) 的 4 阶 PFM 把 mesh budget 从 ℓ/h ≥ 4–10 放松到 ≥ 2；
> 而 Hu–Zhang 应力 + Aifantis 应变梯度弹性 ([09]) 又把尺寸效应纳入。把三者叠加在
> A 的自适应循环上，可同时拿到：mesh 放松 + 尺寸效应 + 保证型估计子 + 微纳尺度断裂能力。*

### C.2 文献钩子

- **point 7 / [06]**：4 阶 brittle PFM + 强形式 LRBFCM；引入中间变量 χ=(ℓ²/2)Δφ 把 4 阶降阶为两个 2 阶（**关键工程技巧**，fealpy 侧可复用）。
- **point 8 / [09]**：strain-gradient elasticity + brittle PFM 的耦合，方法一（应变 + 应变梯度同时谱分解）效果最好。
- **point 31**：READING_NOTES 已把「4 阶 + 应变梯度」标为 explicit gap。
- **point 6**：拉/压分解三派，本方向需要在 4 阶能量下重做（[06] 选择 no-split，但工程算例需要）。

### C.3 与现仓库对接

| 模块 | 现状 | 改造点 |
|---|---|---|
| 相场离散 | `fracturex/phasefield/main_solve.py`、`crack_surface_density_function.py` | 加 (Δφ)² 项、降阶为 (φ, χ) 两个 2 阶问题 |
| Hu–Zhang 装配 | `fracturex/assemblers/huzhang_elastic_assembler.py` | 应力空间无需改；ε(u) 替换为 ε(u) + ℓ²∇²ε(u) 需扩展 |
| 损伤 | `fracturex/damage/phasefield_damage.py` | 历史场 H 公式微调（4 阶情形见 [06] §3.2） |
| 自适应 | 复用 A 的 $\eta_T$ | 估计子需在「ε-gradient 加权能量」下重做有效性分析（M0 门槛重跑） |

### C.4 Milestone

- **M0：4 阶 PFM 单元测试**：复现 Ali 2024 单边裂纹板 v.s. 二阶 PFM 的 mesh 放松曲线（O(h^2.0) vs O(h^2.3)）。
- **M1：strain-gradient 元素插入**：先在均匀网格、小应变上验证 [09] 单边裂拉/剪基准；
- **M2：与 A 的 $\eta_T$ 拼接**：写「保证型估计子在高阶 PFM 上仍可靠」的扩展定理（实操可能要 $H^2$-协调位移空间，**这是难点**）。

### C.5 风险

- 4 阶 + Hu–Zhang 应力空间联合的 inf-sup 还没有现成结论，**M2 的理论改造工作量可能 ≥ A 全部**。建议作为 A 完成后的下一篇。

---

## D. 块预条件论文 D12（短平快收尾）

### D.1 头条已锁定

`docs/preconditioner/D12_PRECONDITIONER_PAPER_PLAN.md` §13.9（2026-06-09）：

> *在损伤完全局部化 + 直接法 OOM 的最难 regime 里，本工作的辅助空间预条件子是唯一仍给出有界收敛
> （O(100)）的求解器；对手（none/Jacobi/ILU）全 DNF、direct 在大 N OOM/singular。*

### D.2 与 READING_NOTES 的接点

- **point 17 [20][21][22] Cervera ε-u 混合元 + VMS-OSGS**：是「另一种鞍点离散 + 另一种稳定化」的对照轴，D12 论文 §Related Work 必引；可在 §"为什么不直接用 ε-u + OSGS" 里给一个对比段落。
- **point 24 [24] p4est**：3D 大规模 sweep 的并行后端来源（D12 §13 已经留了 3D 章节）。
- **point 14 / [10]**：四叉树自适应 + 双相场 —— 与 D12 的「难 regime」对照，可以用来证「即使加密，求解器选择仍重要」。

### D.3 建议

- D12 不是新写，而是 collect 现有 `D12_RESULTS.md` 写成稿。
- 把「头条」从「迭代数恒定」改成「难 regime 唯一收敛」需要把 `precond_sweep` 跑全（§13 状态表里仍有空白）。
- **优先级低于 A/B**，作为 A 篇出门前的备胎写作。

---

## E. 软-硬复合 BTD + 可微 HZ-PFM 拓扑韧化（远景）

### E.1 论点

> *Fucheng Tian PNAS 2025 ([14][15]) 的 SH-com 框架已证「线弹性最小模型 + Heaviside 失效」即可解释 Mullins
> 与 BTD；其结构与 fractureX 的 Hu–Zhang 应力 + (1−d)² 退化天然同构。
> 把这个最小模型 + AuTO/JAX ([25]) 的 reverse-mode AD 串起来，可写「可微相场断裂 → 韧化拓扑优化」一篇 ML4Science。*

### E.2 与现仓库对接

| 模块 | 现状 | 改造 |
|---|---|---|
| 后端 | `bm = backend_manager`，支持 numpy/jax/pytorch | **JAX 后端是路径** |
| 数据生成 | `fracturex/learn/` 已 vectorize 算例参数化 | 把参数从静态 sweep 变成可微变量 |
| AD | 待引入 | 在 JAX 后端上重写 `phasefield_assembler` 的 closure，使 σ_h → loss 全可微 |
| Mesh AD | 无 | E 的关键工程挑战；可绕过用 SIMP / 密度场代替几何变形 |

### E.3 建议

- 远景方向，建议在 B 主篇出版后另立项目。
- 短期可以先做「在 jax 后端上把现有 staggered driver 跑通」作为 §I (Airy 势) 实现的副产品。

---

## 1. 与 READING_NOTES point 库的总映射（速查）

| Point | 内容 | 本仓库去向 |
|---|---|---|
| 2 | PFM 优势：自动捕捉起裂/分叉 | A §1 / B §P3 motivation |
| 3 | PFM 两短板（cost + ℓ 张力） | A 全文中心 motivation；B §P3 |
| 4 | 不同方法 force-CMOD 趋同 | A §M3 对照表 |
| 6 | 拉/压分解三派 | A §"split 情形 $\Theta$"；C |
| 7 | 高阶 PFM mesh 放松 | C 核心 |
| 8 | strain-gradient PFM | C 核心 |
| 9 | 双相场 + 各向异性结构张量 | A §M4 对照 [10] |
| 10 | 软材料 + 大变形 + c̃_s | E |
| 11 | SH-com BTD 标度律 | E |
| 12 | staggered vs monolithic | A、B §Stage D 训练数据划分 |
| 13 | Hybrid formulation 省 90% CPU | A §3 motivation；D12 §Related |
| 14 | 自适应阈值 d_c≈0.1–0.5 + p4est | A §M2 / §M5 |
| 15 | 2:1 balance + ghost layer | A §3D 章节 / D12 §Related |
| 16 | 强形式 LRBFCM + PHS | C 备选离散 |
| 17 | VMS-OSGS ε-u | D12 §Related；A §1 motivation 并列 |
| 18 | 嵌入不连续 + 应力杂交 | A / D12 §Related |
| 19 | VEM 多边形切割 | A §"why not VEM" |
| 20 | AD/JAX 端到端 | B §I (Airy)；E |
| 21 | COMSOL holed plate benchmark | A §M3 verification；B Stage A pilot |
| 22 | COMSOL single edge crack J/G/K | B §H (J 积分 B1 定理) 数值证据 |
| 23 | Wedge/Arrea/Garcia-Alvarez | A §M3 |
| 24 | Pullout 2D/3D | A 备选 §M3 |
| 25 | 3D mixed-mode I+III | A §M5 候选；C 远景 |
| 26 | FRC 单层板 | A §M4 主对照 |
| 27 | SH-com 立方体 | E |
| 28 | PFM ↔ CDM 同源 | B §G.3；A §1 |
| 29 | 可微 PFM + 拓扑优化 | E |
| 30 | mixed FEM + PFM | **fractureX 整个仓库的中心立场**，A/B 都在用 |
| 31 | 高阶 PFM + 应变梯度 | C |
| 32 | p4est + 相场 brittle | A §M5；D12 §13 |
| 33 | 动态 PFM 模型依赖性修正 | B 延伸（动态 PFM 代理） |

---

## 2. 「不建议」的方向（避坑）

- ❌ **重写 XFEM/E-FEM 路线**：fractureX 是 mixed-FEM + PFM 仓库，重启 XFEM 既无差异化（[17] Stolarska 2001 已是经典），也偏离 `dont_modify_fealpy` 的工程约束。
- ❌ **以「Hu–Zhang 高次收敛阶」为带裂纹算例卖点**：`plan_high_order_huzhang.md` §0 已经明确否决，**裂尖奇异会把阶压回去**。
- ❌ **只做 PFM 收敛率分析**：与 [05] Cervera 2021、[16] 中文学位论文重叠度高，无差异化。
- ❌ **算子学习里只学 damage 不学 σ**：与 [03] Ambati 综述后的众多 phase-field + DeepONet 工作差异化不足（paper_thesis §0 已锁定 T3 多输出 rollout）。

---

## 3. 下一步实操建议（给作者本人）

按时间盒：

| 时间 | 主线 | 副线 |
|---|---|---|
| 6–7 月 | **A 的 M0–M1**：超圆 effectivity index 概念验证 + 双解装配 | **B 的 M3a**：实现 `equilibrium_residual_l2`（1–2 周可完成，paper_thesis 头牌指标） |
| 8 月 | **A 的 M2–M3**：自适应循环 + 与 [10] 郭雯 2024 对照 | **B 的 M3b**：σ_h vs σ_h^rec 对照实验（命题 B2 + T2 实验证据） |
| 9–10 月 | **A 论文初稿**（CMAME / SINUM） | **B 的 M3c**：数据重生成 + T1c 闭环 |
| 11–12 月 | A 投稿；启动 **B 论文初稿** | D12 收尾备稿 |

A 与 B 的共同祖先 = Hu–Zhang 应力 σ_h 的两大独家性质（H(div) 协调 + 离散平衡）：

- **A** 把 σ_h 用在「数值分析 / 自适应」侧 —— 一篇 numerical analysis 论文；
- **B** 把 σ_h 用在「AI4Science 监督源」侧 —— 一篇 SciML 论文；
- 两篇共享同一 §2 "Why Hu–Zhang" 段落（**这正是 `plan_adaptive_aposteriori.md` §0 的「为什么值得付混合元的代价」与 `paper_thesis.md` §P1 的「平衡监督是结构最优」的两面**）。

---

## 4. 已验证的代码锚点（保证 §A/§B 立刻可起）

下列路径均已 `ls` / `grep` 核实存在，建议方向不依赖未实现幻觉：

**A 路线（自适应 + a posteriori）**
- `fracturex/adaptivity/equilibrated_estimator.py` ✓
- `fracturex/adaptivity/adaptive_loop_equilibrated.py` ✓
- `fracturex/adaptivity/primal_elastic_solve.py` ✓ — 标准 FEM 位移解（超圆界的 $v$）
- `fracturex/adaptivity/degraded_huzhang_solve.py` ✓ — 退化 Hu–Zhang（超圆界的 $\sigma^*$）
- `fracturex/adaptivity/degraded_mms.py` ✓ — MMS 验证（M0 effectivity index 用）
- `fracturex/adaptivity/primal_resolve_real.py` ✓
- `docs/adaptive/RESULTS_aposteriori.md` ✓ — 已经有 PoC 结果

**B 路线（HZ-supervised 算子学习）**
- `fracturex/learn/eval/metrics.py:277` —— `equilibrium_residual_l2(sigma_pred, body_force, mask)`
  当前实为 `raise NotImplementedError("M2 Stage D: ‖m⊙(∇_h·σ̂ + f)‖_2")`，
  **B 的 M3a 第一笔代码就是把这个 stub 实现掉**（paper_thesis §C 给了完整公式）。
- `fracturex/learn/models/{multioutput_fno,multioutput_unet}.py` ✓
- `fracturex/learn/datasets.py` ✓ + `fracturex/learn/train.py` ✓
- `fracturex/learn/transforms.py` ✓ —— 已含 arcsinh 重尾压缩（可作为 σ 监督的一档 ablation）

**D 路线（D12 块预条件）**
- `fracturex/utilfuc/linear_solvers.py` 中 `solve_huzhang_block_gmres_auxspace`、
  `_approximate_schur_spd`、`_make_coarse_diffusion_coef` 均已就位（D12_PLAN §2 已索引到行号）。

> **结论**：A 与 B 不需要等任何外部库或新基础设施；从今天到投稿只欠实验跑数 + 写作。
> C 的 4 阶 + 应变梯度部分需要新写离散，工程量见 §C。E 需要 JAX 端到端，工程量最大。

---

## 附：本文与既有 docs 的位置

```
docs/
├── NEXT_PAPER_DIRECTIONS.md          ← 本文（方向选择 + READING_NOTES 映射）
├── routes/
│   ├── plan_adaptive_aposteriori.md  ← A 的详细规划
│   ├── plan_high_order_huzhang.md    ← C 的入口（高次定位）
│   └── plan_gpu_multibackend.md      ← 与 E 的 JAX 后端相关
├── operator_learning/
│   ├── plan_operator_learning.md     ← B 的详细规划
│   └── paper_thesis.md               ← B 的定理链 + 主定理（已锁定）
├── adaptive/
│   ├── THEORY_equilibrated_aposteriori.md  ← A 的理论核心
│   ├── THEORY_marking_strategy.md          ← A 的 M-DF 主驱动
│   └── DECISION_sigma_driven_adaptivity.md ← A 的分工决策
└── preconditioner/
    └── D12_PRECONDITIONER_PAPER_PLAN.md    ← D 的详细规划（头条已锁）
```

新增或调整本文引用的 plan 后，请同步 [readme.md](readme.md) 的索引行。
