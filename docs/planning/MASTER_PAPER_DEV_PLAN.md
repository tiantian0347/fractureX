# fracturex 论文 + 开发 主计划（2026-07-04 v0.2）

> **v0.2 更新（2026-07-04）**：D12 (`Tian/thesis/fracture_huzhang/phasefield_huzhang.tex`) 与 A (`.../adaptive/equilibrated_aposteriori.tex`) 两篇 tex 均已完成正文 + Conclusion + Outlook。基于两篇 Outlook 自认的欠账，重新校准 Phase 1/2 排期：短期先补两篇 gap 送审，中期改为 **T3 (B 线) + T5 (D+ 线) 并行**，T4 (A+ 线) 后置到 2027。


> **输入源**：① `docs/planning/p1_action_checklist.md`（P1 止血清单）② `docs/planning/conf_202606_inspirations.md`（港中深会议迁移路径）③ `docs/NEXT_PAPER_DIRECTIONS.md`（A/B/C/D/E 五方向）④ `docs/preconditioner/D12_PRECONDITIONER_PAPER_PLAN.md` + `PIPELINE_STATUS.md`（D12 头条与在跑流水线）⑤ `docs/adaptive/*`、`docs/routes/plan_adaptive_aposteriori.md`（A 线理论与主循环代码就位）⑥ `docs/operator_learning/paper_thesis.md`（B 线主定理锁定 2026-06-02）⑦ `/Users/tian00/Desktop/gong办公资料/TalksAndPapers/SUMMARY.md`（龚世华 2018 博士论文 + 2015 H(div,S) 手稿）。
>
> **目的**：把当前散在各 `docs/` 里的论文/开发线拉平成**一张顺序表**——按「短期可发表 × 已有代码/理论 × 卖点稀缺度」三维排序，标出每一步的依赖、状态与产出。所有条目在原始规划文件里都能追溯，非重复内容。

---

## 0. 一图看懂：龚世华 2018 博士论文 vs fracturex 五条论文线

**博士论文（SUMMARY.md）是 fracturex 方法学的祖谱**，其三部分几乎逐一对应 fracturex 的五条论文线：

| 博士论文章节 | 内容 | 对应 fracturex 线 | 状态 |
|---|---|---|---|
| 第 3 章 杂交混合有限元（放松顶点连续性 + 任意维奇异点代数定义） | Hu-Zhang 元扩展 | **conf §4 马睿线 2** ★★★（L-shape/切口角点） | **未开工**，是 A 线的天然升级项 |
| 第 4 章 内罚混合有限元（非协调面泡，任意阶） | 低阶 Hu-Zhang 替代 | fracturex 未采用（p≥3 已足） | 备胎，暂不投入 |
| 第 6 章 杂交系统的多水平求解器（非嵌套粗空间 + 区域分解磨光 + 顶点块局部估计） | 多水平预条件 | **conf §1 胡齐芽两层 Schwarz** ★★★ | 未开工，D 线**升级路径** |
| 第 7 章 H(div, S) 辅助空间预条件（离散弹性正合列 + 离散正则分解） | 块对角预条件的理论底座 | **D12 论文的理论根基** | ✅ 已在 fracturex 使用（`solve_huzhang_block_gmres_auxspace`），论文正在收尾 |
| 第 8–10 章 仿射不变 Lipschitz + NEPIN/ASPIN | 非线性预条件牛顿法 | fracturex staggered/Newton **理论补强候选** | **未开工**，可与 §葛志昊"全离散稳定性"合并做理论文 |
| 2015 H(div,S) 手稿（`thiese/`） | 第 7 章雏形 | = D12 前身 | — |

**结论**：D12（论文 §D）在**用**博士论文第 7 章；A 线自适应可**接**第 3 章；staggered 收敛的严格理论可**用**第 8–10 章的仿射不变框架。这三条是龚老师博士工作在 fracturex 上的直接延续，可以在论文里理直气壮地引自己。

---

## 1. 总优先级表（按"6 个月内可推进"排序）

| # | 类别 | 项目 | 状态 | 依赖 | 目标产出 | ETA |
|---|---|---|---|---|---|---|
| **T0** | 运维 | P1 遗留 + aux_h2/h3/model1 三段并行流水线 | 🟡 运行中（2026-06-04 起） | — | paper_aux 数据齐全 | 1–2 周 |
| **T1** | 论文 D | D12 tex 已成稿（`phasefield_huzhang.tex` 3148 行含 Conclusion + Appendix）→ 补 Outlook 自认欠账 + §13 sweep 表 + 收稿 | 🟡 tex 骨架 100%，欠 SENT shear 完整 aux-vs-direct + 局部化 mesh-independence + shear 局部化 iteration 一栏 | T0（h2/h3/model1 pipeline 出货） | **SISC/CMAME 短稿** | 2–3 月 |
| **T2** | 论文 A | A tex 已成稿（`equilibrated_aposteriori.tex` 907 行含 Conclusion）→ 补 SENT shear、equilibrating correction、spectral split majorant、marker efficiency 下界 | 🟡 tex 骨架 100%，SENT tension + CNT red-green 已交付 | T0 无关；代码全在手 | **CMAME/SINUM 主推** | 4–5 月 |
| **T3** | 论文 B | HZ-supervised 算子学习 Stage D + 对照表 + 数据重生 | 🟢 主定理锁定 2026-06-02，欠 `equilibrium_residual_l2` stub 实现 | T1/T2 无关，可并行 | **JCP/CMAME** | 5–6 月 |
| **T5** | 论文 D+ | 多水平/两层 Schwarz 预条件（博士论文第 6 章 + conf 胡齐芽） | 🟢 D12 Outlook 自己点名"contrast-adapted, interface-resolving coarse space"；paper_aux 数据可直接复用 | T1 tex 送审后启动 | D 论文续作 / SIMAX-NLAA | 8–12 月 |
| **T4** | 论文 A+ | Hu-Ma 扩展 H-Z 空间 + NVB 处理 L-shape（博士论文第 3 章 + conf 马睿线 2） | 🔵 fealpy 需新增顶点分裂逻辑；工程量最重 | T2 送审后接入 | A 论文附加章 / 独立短文 | 12–14 月 |
| **T6** | 论文 C | 4 阶 PFM + 应变梯度（Ali 2024 + Aifantis） | 🔵 需新写离散 | T2 完成 | Comput. Mech./IJNME | 14–18 月 |
| **T7** | 理论 | 仿射不变 Lipschitz + NEPIN 用到 fracturex staggered（博士论文第 8–10 章 + conf §3 葛志昊） | 🔵 未开工，理论文 | T2/T3 完成 | SINUM/M2AN | 15–20 月 |
| **T8** | 论文 E | 可微 HZ-PFM + AuTO 拓扑韧化 | 🔵 需 JAX 端到端 | T3 完成 + jax backend | ML4Science / PNAS 短篇 | 20–24 月 |

**颜色**：🟡 在跑/在写 🟢 立即可起 🔵 依赖前置

---

## 2. 详细排期（半年视角）

### Phase 0（2026-07 上）— 止血 & 数据齐全

**T0 剩余动作**（`p1_action_checklist.md` §"P1 完成判定"未打勾项）
1. Lagrange 路线 `MainSolve` 在 model0 上跑通（C5 对照必备）
2. EXPERIMENT_MATRIX 自动扫描脚本（P0 优先级不高，0.5 天可写）
3. 等 aux_h2/h3/model1 三段过完局部化区（h2 在 step 13/31 已滞、h3 在 step 13、model1 在 step 54/161）——**这些数据是 T1 D12 §13 sweep 表的原料**

**门槛条件（Go / No-Go）**
- T2/T3 不阻塞 T0 完成，可并行启动；
- T1 需要 model0/model1 至少一个跑到破断点后 direct/aux 对比数据齐。

---

### Phase 1（2026-07 中 → 2026-09）— **T1 (D) + T2 (A) 两篇 tex 补 gap 送审**

> **v0.2 起点**：两篇 tex 骨架已完成正文 + Conclusion + Outlook。以下"欠账"清单直接来自两篇 Conclusion 里作者自认的 open extension。

**T1 · D 线 · D12 tex 收稿**（`Tian/thesis/fracture_huzhang/phasefield_huzhang.tex` 3148 行）

Outlook 自认欠账（Conclusion §Future Work 明写）：
1. SENT shear 的 aux-vs-direct 端到端复现 + crack-path 场对比图（tension 已交付）
2. SENT shear 上的局部化 iteration 研究（tension 已在 §Bounded convergence 交付）
3. 局部化区 mesh-independence 一栏（uniform 已交付）
4. §13 sweep 表仍留空档 → 等 T0 pipeline (h2/h3/model1) 出货填补
5. Cervera VMS-OSGS 对照段落补进 §Related Work

**T2 · A 线 · A tex 收稿**（`Tian/thesis/fracture_huzhang/adaptive/equilibrated_aposteriori.tex` 907 行）

Outlook 自认欠账（Conclusion 尾段明写）：
1. SENT shear 验证（tension + CNT red-green 已交付）
2. 一般数据 $(f, t_N)$ 的显式 equilibrating correction，把 canonical setting 之外的 Prager–Synge 补齐
3. spectral tension–compression split 的 convex-duality majorant（当前只覆盖 isotropic $\mathbb C_d = g(d)\mathbb C$）
4. marker efficiency 下界 / 认证（predictor 循环之外的 marker certification）

**T3 · B 线 · 并行打底**（不阻塞 T1/T2 收稿）
- **M3a（2 周）**：`fracturex/learn/eval/metrics.py:277` 的 `equilibrium_residual_l2` NotImplementedError 实现掉（B 的第一笔代码；`paper_thesis.md` §C 给了完整公式）
- **M3b（2 周）**：σ_h 监督 vs σ_h^rec 监督对照表（命题 B2 + T2 实验证据）
- **M3c（3 周）**：数据重生成满足 h_FE ≤ ℓ₀/2（`surrogate_data_underresolved_hl0`），验证 T1c 峰幅值预言
- **M4 选做**：§I Airy 势硬约束 ablation

**T1 与 T2 共享**：§2 "Why Hu-Zhang" 段落（**同一祖父段落，两篇复用**——SUMMARY 博士论文摘要 "$H(\mathrm{div},\mathbb{S})$ 上的 inf-sup 稳定 + 逐元精确平衡" 两条独家性质）。

---

### Phase 2（2026-10 → 2027-03）— **T3 (B) + T5 (D+) 双线主推**，T4 后置

> **v0.2 调整**：D12 Conclusion §Future Work 自己点名 "contrast-adapted, interface-resolving coarse space — replacing the geometric $P_1$ correction where the damage interface is sharp"，这正是 T5 (D+) 的入口，直接从 D12 头条 "局部化区 aux niter 骤升 ~14×" 的口子接续。T4 (A+) 由于 fealpy 顶点分裂逻辑工程量最重，后置到 2027 春。

**T5 · D+ 线 · 多水平/两层 Schwarz 预条件**（D12 收稿即刻启动）
（博士论文第 6 章 + `conf_202606_inspirations.md` §1 胡齐芽/谢和虎/梁启刚）

**为何现在做**：
- D12 Outlook 自认 "residual gap between the few-tens count at full localization and the $\mathcal O(10)$ of the uniform regime" 需要 contrast-adapted coarse space 来关闭——**这是作者自己的 next-step**
- 博士论文第 6 章的**非嵌套粗空间 + 稳定提升算子 + 覆盖 DD 磨光**，正是对症方案
- 胡齐芽的**局部 GEP 自适应构造粗空间** = 龚博士非嵌套粗空间的现代版
- paper_aux 数据可**直接复用**做离线 GEP-coarse 实验，不需要新造 dataset

**里程碑**：
1. 在 paper_aux baseline 上离线做 GEP-coarse 实验
2. 与现有 aux-space Schwarz 对照 niter 曲线（重点看 max d → 0.99 区段）
3. 出成 D 线续作（SIMAX/NLAA）

**T4 · A+ 线 · Hu-Ma 扩展 H-Z 空间处理 L-shape/角点奇异**（A 送审后启动，2027 春）
（博士论文第 3 章 + `conf_202606_inspirations.md` §4 马睿线 2 ★★★）

**核心机制**：在 NVB 加密新顶点 x_e 处，把纯切向 φ_{x_e} t_e t_e^T 基沿边 e 拆成 ω⁺/ω⁻ 两侧的 τ⁺、τ⁻；法向分量（n_e n_e^T、n_e t_e^T + t_e n_e^T）不动。x_e 处 DoF 从 3 变 4，H(div) 协调性保持。

**为何后置**：
- fracturex 处理 L-shape 目前是 `corner_relaxation_PR.md` 里的临时补丁，缺理论——**A 论文的现有 CNT/SENT 场景未受此影响**，不阻塞 A 送审
- fealpy 需新增顶点分裂逻辑，工程量最重
- 与 A 自适应循环天然嵌套：AFEM 加密后的新顶点自动落进 Hu-Ma 扩展空间——A 送审后接入更自然

**里程碑**：
1. fealpy 上复现 Hu-Ma §5.2 rotated L-shape（α=0.544...，不带相场）
2. corner relaxation 装配模块升级（`architecture/huzhang_corner_relaxation_design.md` 已有草案）
3. 嵌回 fracturex model2 + L-shape 或 V-notch 几何
4. 写成 A 线论文附加章 or 独立短文

---

### Phase 3（2027-04 之后）— T6/T7/T8

**T6 · C 线 · 4 阶 PFM + 应变梯度**（`NEXT_PAPER_DIRECTIONS.md` §C）
- 依赖 T2 完成；4 阶 + Hu-Zhang inf-sup 是理论新工作，工程量 ≥ A 全部
- 卖点：mesh budget ℓ/h≥4 → ≥2 + 尺寸效应 + 保证型估计子

**T7 · 理论文 · 仿射不变 NEPIN + 全离散稳定性**（博士论文第 8–10 章 + `conf §3` 葛志昊）
- 用博士论文第 8 章的**仿射不变 Lipschitz 常数**理论，分析 fracturex staggered/Newton 收敛
- 补 fracturex 目前**只有单调性数值现象、没干净能量证明**的短板
- 输出：一篇偏理论的 SINUM/M2AN 文章

**T8 · E 线 · 可微 HZ-PFM + AuTO 拓扑韧化**（远景）
- 依赖 T3 完成 + fealpy jax backend 稳定
- SH-com 弹簧网络最小模型（Fucheng Tian PNAS 2025）+ JAX AD 端到端

---

## 3. 论文清单（按投稿时间排序）

| 顺序 | 代号 | 标题草案 | 期刊首选 | 关键 selling point | 预计投稿 |
|---|---|---|---|---|---|
| 1 | **D12** | Auxiliary-space preconditioning for Hu-Zhang mixed phase-field fracture: uniqueness of convergence in the fully-localized regime | SISC / CMAME | 难 regime 唯一收敛；引龚博论 Ch 7 为理论根基 | 2026-09 |
| 2 | **A** | Equilibrated a posteriori error estimation and σ-driven adaptivity for Hu-Zhang mixed phase-field fracture | CMAME / SINUM | 无常数超圆界；osc(f)=0 干净；超过郭雯 2024 的 CPU/mesh 节省 | 2026-11 |
| 3 | **B** | Balance-preserving neural operators for phase-field fracture via Hu-Zhang supervision | JCP / CMAME | 平衡监督 = 结构最优；R̃_h 平台 vs 下降曲线；诚实边界 T1/D1 | 2027-01 |
| 4 | **D+** | Non-nested coarse spaces and two-level Schwarz preconditioning for Hu-Zhang phase-field fracture in the localized regime | SIMAX / NLAA | 龚博论 Ch 6 多水平 + 胡齐芽 GEP-coarse；D12 自己 Outlook 点名的续作 | 2027-03 |
| 5 | **A+** | Extended Hu-Zhang element with vertex-tangent relaxation for adaptive elasticity at reentrant corners | Math. Comp. / M2AN | 合并 Hu-Ma 2020 + 龚博论 Ch 3 任意维奇异点代数定义 | 2027-06 |
| 6 | **T7** | Affine-invariant analysis of staggered Newton for Hu-Zhang phase-field fracture | SINUM / M2AN | 龚博论 Ch 8–10 NEPIN 框架用到相场 staggered，补严格能量下降证明 | 2027-09 |
| 7 | **C** | Fourth-order phase-field fracture with strain-gradient elasticity in Hu-Zhang mixed setting | Comput. Mech. / IJNME | mesh 放松 + 尺寸效应 + T2 估计子扩展 | 2027-12 |
| 8 | **E** | Differentiable Hu-Zhang phase-field for topology-toughening design | ML4Science / PNAS | JAX 端到端 + SH-com 最小模型 | 2028 |

---

## 4. 开发任务清单（按优先级）

### 4.1 P0 · 立即（Phase 0）
- [ ] T0.1 完成 Lagrange 路线 (`MainSolve`) 在 model0 上跑通（C5 对照必备）
- [ ] T0.2 EXPERIMENT_MATRIX 自动扫描脚本（0.5 天）
- [ ] T0.3 aux_h2/h3/model1 三段过完局部化，产出 D12 sweep 数据

### 4.2 P1 · Phase 1（Q3 2026）· D12 + A tex 补 gap 送审 + B 线并行打底
- [ ] T1.tex.1 SENT shear 完整 aux-vs-direct 端到端 + crack-path 场对比
- [ ] T1.tex.2 SENT shear 局部化 iteration 研究一栏
- [ ] T1.tex.3 局部化区 mesh-independence 一栏
- [ ] T1.tex.4 §13 sweep 表空档从 pipeline (h2/h3/model1) 出货后补齐
- [ ] T1.tex.5 Cervera VMS-OSGS 对照段落补进 §Related Work
- [ ] T2.tex.1 SENT shear 场景验证（tension + CNT red-green 已交付）
- [ ] T2.tex.2 一般数据 $(f, t_N)$ 的显式 equilibrating correction
- [ ] T2.tex.3 spectral tension–compression split 的 convex-duality majorant
- [ ] T2.tex.4 marker efficiency 下界 / marker certification
- [ ] T3.M3a `equilibrium_residual_l2` stub 实现（`fracturex/learn/eval/metrics.py:277`）
- [ ] T3.M3b σ_h vs σ_h^rec 监督对照表
- [ ] T3.M3c 数据重生成 h_FE ≤ ℓ₀/2

### 4.3 P2 · Phase 2（Q4 2026 – Q1 2027）· T3 完稿 + T5 D+ 主推，T4 后置

- [ ] T5.1 GEP-coarse 离线实验（paper_aux 数据复用）
- [ ] T5.2 两层 Schwarz vs 现 aux-space 对照（重点 max d → 0.99 区段）
- [ ] T5.3 D+ 论文成稿（SIMAX/NLAA，2027-03 投稿）
- [ ] T3.M4（选做）§I Airy 势硬约束 ablation
- [ ] T4.1 fealpy 复现 Hu-Ma §5.2 rotated L-shape（A 送审后启动）
- [ ] T4.2 corner relaxation 装配模块升级（对接 `architecture/huzhang_corner_relaxation_design.md`）
- [ ] T4.3 扩展 H-Z 空间嵌回 fracturex model2 + L-shape

### 4.4 P3 · Phase 3（Q2 2027+）
- [ ] T6 4 阶 PFM + 应变梯度离散
- [ ] T7 仿射不变 Lipschitz 用到 staggered 收敛证明
- [ ] T8 fealpy jax backend + AuTO 端到端

---

## 5. 与龚博论 SUMMARY 的直接对接点（论文写作时要引的段落）

| 论文 | 引龚博论 | 用途 |
|---|---|---|
| D12 | Ch 7 "H(div,S) 辅助空间预条件" §7.4–7.5（离散正则分解 + 辅助空间预条件子） | 直接作为 D12 §Theory 的引用，避免自证 |
| A | Ch 7 §7.2 "$H(\div,\mathbb{S})$ 空间的正则分解与弹性正合序列" | 支撑 A 论文的 σ_h ∈ H(div,S) 平衡性讨论 |
| A+ | Ch 3 §3.1 "Hu-Zhang 元及其推广" + §附录 A "奇异点相关结果的证明" | 任意维奇异点代数定义与 Hu-Ma 2020 合并 |
| D+ | Ch 6 全章 "杂交化问题的多水平求解器" | 非嵌套粗空间 + 顶点块局部估计的直接祖本 |
| T7 | Ch 8 "牛顿法与仿射不变性" + Ch 10 "非线性消去预条件牛顿法" | 仿射不变 Lipschitz 常数用到 staggered/Newton |

---

## 6. 明确不做的方向（避坑；`NEXT_PAPER_DIRECTIONS.md` §2）

- ❌ 重写 XFEM/E-FEM
- ❌ 以 Hu-Zhang 高次收敛阶为带裂纹算例卖点（裂尖奇异压回阶）
- ❌ 只做 PFM 收敛率分析
- ❌ 算子学习里只学 d 不学 σ
- ❌ D13 学习增广线（已封存，`d13_learn_coarse_space`）

---

## 7. 一句话总纲

> **短期（半年）** 把 D12 收稿、A 主推、B 打底三线并行推完；**中期（一年）** 用博士论文第 3 章升级 A（→A+）、用第 6 章升级 D（→D+）；**长期（一年半+）** 用第 8–10 章补 staggered 理论（T7）、用 JAX AD 打远景（T8）。**所有线都以 σ_h ∈ H(div,S) 的两条独家性质（inf-sup 稳定 + 逐元平衡）为共同支点**——这正是龚博论第 7 章、conf 马睿线、fracturex 三者最大的方法学交汇。

---

## 附：文件锚点索引

- 本文件：`docs/planning/MASTER_PAPER_DEV_PLAN.md`
- P1 清单：`docs/planning/p1_action_checklist.md`
- 会议迁移：`docs/planning/conf_202606_inspirations.md`
- 五方向：`docs/NEXT_PAPER_DIRECTIONS.md`
- D12 计划：`docs/preconditioner/D12_PRECONDITIONER_PAPER_PLAN.md`
- D12 结果：`docs/preconditioner/D12_RESULTS.md`
- Pipeline：`docs/preconditioner/PIPELINE_STATUS.md`
- A 线计划：`docs/routes/plan_adaptive_aposteriori.md`
- A 线理论：`docs/adaptive/THEORY_equilibrated_aposteriori.md`
- B 线论文：`docs/operator_learning/paper_thesis.md`
- B 线计划：`docs/operator_learning/plan_operator_learning.md`
- 龚博论 SUMMARY：`/Users/tian00/Desktop/gong办公资料/TalksAndPapers/SUMMARY.md`
