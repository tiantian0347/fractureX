# fracturex 论文 + 开发 主计划（2026-07-05 v0.9）

> **v0.9 更新（2026-07-05）**：**A 送审关键路径落到执行层 + 二投候选双路并行**——A 的瓶颈是 scp + 后处理一步，理论侧 A-2/3/4 从 remark 提到 §body（spectral majorant 拎成独立子节，避免 reviewer 觉得藏在 remark 里）；A 走 CMAME、arXiv 同日挂占坑。第二篇不押死 D12：**路径 α (D12)** 立刻起 shear aux **h1**（比 h2/h3 快，3–4 天过 peak 够 aux-vs-direct），**路径 β (B/T3)** M3b.4 服务器改 writer 立即起（不阻塞任何在跑 job）；两路平行 2 周，谁先齐谁投。§2 Phase 1 加执行层清单。

> **v0.8 更新（2026-07-05）**：**A 先于 D12 送审**——A tex 理论侧 A-2/3/4 已落地（v0.2），SENT shear 数据 7/3 已跑完（step 32 有效），只欠本地 scp + 图表处理（A-1），1–2 周可送审；D12 的 SENT shear aux 侧尚未起跑 + h2/h3 续算需 1–2 周，Conclusion 已明写 shear 是头条卖点，硬砍会伤 selling point。**动作**：§1 表把 T2 ETA 提到 **2026-07**、T1 保持 2026-09；§3 论文清单顺序调整为 **A → D12 → B → …**，D12 投稿从 2026-09 保留不动、A 从 2026-11 提前到 2026-07；§2 Phase 1 标题改为「T2 (A) 先送审，T1 (D12) 补 shear 后跟上」。

> **v0.7 更新（2026-07-05）**：**章节顺序调整**——把 §0（龚博论祖谱）与 §0.5（田博论祖谱）两段背景章从最前面移到最后面（现改称 **§8** / **§9**，位于 §7 一句话总纲之后、§附 A 之前）。目的：§1 总优先级表直接顶到读者眼前，背景性对接表下沉。**内容不变**，仅编号与位置变。历史 changelog 里"§0.5"字样保留原样以反映当时结构。

> **v0.6 更新（2026-07-05）**：**T6a tex 已有底稿**——`Tian/thesis/ip_fracture/ipfem_paper.tex` 1187 行，Intro/Model/离散/**完整误差分析章**（stability + coercivity + a priori $h^{p-1}$）/两算例（circular hole + notch）/Conclusion 骨架 100%；结果目录 `ipfem_fp_results/` 中 model0 p=2,3,4 × 多 h 全 √；六张图齐。**Conclusion 自认五条 open extension** 已识别。**核心 delta 是应变梯度耦合**（相对博论 IP-FEM 章的独家性）——**理论先行**：先补 §Model / §Discretization 里应变梯度变分形式与稳定性分析，再回 fealpy3 实现。**动作**：§2 T6a 预研任务从 3 条扩到 6 条（明确"先理论后代码"顺序），§4.3 加对应 checkbox；fealpy_old → fealpy3 迁移作 T6a.pre.6，是 T6b 的前置。附 §附 T6a 现状盘点表。
>
> **v0.5 更新（2026-07-04）**：**T6 (C 线) 拆两路径**——博论 IP-FEM 地基稳，把 C 线拆成 **T6a·标准/IP-FEM 4 阶 PFM**（博论直接延伸，理论风险低，safe win）和 **T6b·Hu-Zhang mixed 4 阶 PFM**（inf-sup 重证，novel）。投稿排开：**2027-06 T6a → 2027-09 T7 → 2027-12 T6b**，同时解决 v0.4 遗留的 T7/C 撞档 2027-09。**动作**：§1 表拆两行、§2 Phase 2 尾预研先做 T6a 数值扩展 / T6b inf-sup 草稿、§3 论文清单加一行 T6a、§4 拆预研与代码接入。
>
> **v0.4 更新（2026-07-04）**：**T6 (C 线 4 阶 PFM) 优先级提半档**——依据 §0.5 博论 IP-FEM 4 阶章已把可行性验证完，C 线现在做的是 "IP-FEM → Hu-Zhang mixed setting" 的**离散升级**而非从头造轮子；且 T6 卖点（mesh budget ℓ/h≥4）与 T5 D+ 卖点（求解器 niter）正交。**动作**：ETA 14–18 月 → 10–14 月；投稿 2027-12 → 2027-09；§2 Phase 2 尾新增"T6 预研段"（离散设计 + inf-sup 草稿），代码接入仍待 T2 送审后。**不改**：T2/T5 仍是 Phase 2 主推，T6 不抢资源。
>
> **v0.3 更新（2026-07-04）**：接入 **田甜博士论文（`ttthesis/thesis/tianPHD.tex`, 已完成）** 的方法学接续关系——博论四项创新（任意次 tensorized FEM / recovery-based AFEM + GPU / IP-FEM 4 阶 PFM / FractureX 平台）与主计划五+条论文线一一对上（新增 §0.5）。§5 引用表加入博论 A/C/B 三行；附索引加 ttthesis 锚点。
>
> **v0.2 更新（2026-07-04）**：D12 (`Tian/thesis/fracture_huzhang/phasefield_huzhang.tex`) 与 A (`.../adaptive/equilibrated_aposteriori.tex`) 两篇 tex 均已完成正文 + Conclusion + Outlook。基于两篇 Outlook 自认的欠账，重新校准 Phase 1/2 排期：短期先补两篇 gap 送审，中期改为 **T3 (B 线) + T5 (D+ 线) 并行**，T4 (A+ 线) 后置到 2027。

## 服务器 job 状态（2026-07-04 22:28 CST 核对 + 本轮改动）

| Job | 现状 | 数据到 | 下一步 |
|---|---|---|---|
| **A adaptive model2 effstress** | ✅ **已跑完** (7/3 17:31, `DONE steps=34, peak_R=0.234`) | step 33 fallback 崩停；有效数据到 step 32 (R=0.156)；history.csv 35 行 + vtu 11MB | 已给 scp 命令；下拉后处理成 SENT shear 图表补进 A tex §sec:num (A-1) |
| **D12 aux h2** | 🟢 **已重启在跑** (7/4 22:14, PID 267325, restart=400/maxit=800 + Anderson, from step_013.npz) | 之前 6/19 停在 step 59, u_y=0.0876（已过 peak -28.14） | 目标：u_y=0.125 (breakthrough)；每步 ~1–4h（局部化区更慢），估计 1 周 |
| **D12 aux h3** | 🟢 **已重启在跑** (7/4 22:14, PID 267327, 同上) | 之前 6/22 停在 step 106, u_y=0.0876（已过 peak -28.19） | 每步 ~8h（局部化区更慢），估计 2 周 |
| **D12 model2 direct pardiso_gmres** | 🟡 在跑 (PID 4142145, 5天+, step 8, u_y=6.67e-4) | 未到 peak (peak ~ u_y=1.03e-2) | 每步 ~15h，到 breakthrough 3–4 周；性价比低 |
| **D12 square/model0 direct** | 🟡 在跑 (PID 3782490, 16天+) | 用途待确认 | 保留 |

**本轮 tex 改动（2026-07-04）**：

- D12 (`phasefield_huzhang.tex`, 36pp)：**D-7 完成**——ref.bib 加 Cervera 2010 Part I/II + 2017 三条，intro 加 VMS-OSGS 对照段落
- A (`equilibrated_aposteriori.tex`, 18pp)：**A-2/3/4 完成**——`rem:gen-data` 扩成 Braess-Schöberl patch-local 校正 sketch；`rem:split` 加 Fenchel 对偶 majorant；`rem:marker-eff` 加 η_ω_z local lower bound；Conclusion 相应重写把三条 open extension 从"待做"改为"已 sketch + companion paper"

**剩余 gap 状态**：

- 🟢 D-1..D-5 (SENT/SENS aux 端到端 + 局部化 mesh-indep)：**等 h2/h3 续算出货**（1–2 周）
- 🟢 A-1 (SENT shear 场景)：**数据到手**，等本地 scp + 图表处理
- 🟡 D-3/D-4 (SENS aux)：model2 direct 太慢，作 "future work" 处理
- ✅ D-7, A-2/3/4：本轮全部落地
- ✅ **T3.M3a**：`equilibrium_residual_l2` stub 实现完成（`fracturex/learn/eval/metrics.py:277`），签名对齐 `paper_thesis.md §C`（`sigma_grid, mask, dx, dy, f=0.0, d=None, d_c=0.9, L=None`）；`fracturex/tests/test_learn_metrics_equilibrium.py` 8 项 pytest 全过（零应力、常应力 div=0、σ_xx=x 解析 R̃_h≈2.55、体积力抵消、批 shape、d>d_c 剔除、shape 校验、显式 L 覆盖）；`test_learn_m1_smoke.py` 11/11 无回归——B 线迈出第一步
- ✅ **T3.M3b.1**：Stage D 训练损失 `equilibrium_residual_fd` 落地（`fracturex/learn/losses.py:114`），torch 可微版对齐 `paper_thesis.md §C.219`（"训练用 R_h 可微版, 评估用 R̃_h"）；`fracturex/tests/test_learn_losses_equilibrium.py` 9 项 pytest 全过（含 autograd flows to sigma 验证 + 与 numpy 度量的一致性校验）
- ✅ **T3.M3b.2**：`fracturex/learn/stress_recovery.py` 新模块，`stress_recovered_from_displacement(u_grid, d_grid, C, dx, dy, kres, stress_scale)` 给出 σ_h^rec = g(d)·C·ε(u_h)，包含 plane-strain / plane-stress C 矩阵、Voigt 应变、AT2 退化、schema §3.2 stress_scale 归一；**用 fealpy `bm` 后端**（可切 GPU/torch，无 `import numpy as np`）；`test_learn_stress_recovery.py` 10 项 pytest 全过（含拉伸/纯剪解析解、退化标度、shape 校验）
- ✅ **T3.M3b.3**：`train.py` supervision-source 分派 + Stage D 平衡损失接线
  - `TrainConfig.supervision_source ∈ {sigma_h, sigma_h_rec}`（默认 sigma_h）
  - `DatasetConfig.include_stress_rec` + `target_stress_rec` + collate `stress_rec` 支持
  - `_compute_loss` 按 supervision_source 挑 batch["stress" | "stress_rec"]；`cfg.lambda_eq>0` 时加 λ·R_h² 项（在物理空间应用，先反变换 σ）
  - `_make_loader` 按 `supervision_source == 'sigma_h_rec'` 自动开 `include_stress_rec`
  - 全套 38 项 (stress_recovery 10 + losses 9 + metrics 8 + m1_smoke 11) 零回归——B 线对照实验的**训练侧基础设施**齐全，等 M3b.4 数据生成脚本吐出带 `stress_rec` 的数据集即可 A/B 训练

## T3.M3b 服务器待办（等待落地）

**M3b.4 · 数据管线：把 u_h 和 σ_h^rec 写进 npz 数据集**（服务器侧）

现状：数据集 npz 已存 `stress`=σ_h(HZ) + `damage` + `mask`，但**没有存 u_h**（位移场）。要跑 σ_h vs σ_h^rec 对照，必须先把 u_h 也存进去。

具体步骤：

1. **改数据生成器**（`fracturex/learn/datasets.py` 写侧，或 `scripts/data_generation/*.py` 里对应写 npz 的入口）：把 HZ 求解器输出的位移 u_h 采样到 (H, W) 网格，作为 `displacement` 键写入 npz，shape `(T, 2, H, W)`
2. **重生成 dataset**（服务器上跑）：以现有 dataset generation 脚本为基线（参考 `results/operator_learning_smoke/sample_paper_aux_h1.*` 附近的 pipeline），加 `displacement` 输出。paper_thesis 目标 `h_FE ≤ ℓ₀/2` — 如要重生也顺便把这个约束一起补上（对应 master plan 里的 T3.M3c）
3. **σ_h^rec 预计算脚本**（新写 `scripts/data_generation/add_stress_rec.py`）：对已生成 npz 批处理调 `fracturex.learn.stress_recovery.stress_recovered_from_displacement`，写回 `stress_rec` 键。这是**offline 数据变换**，本地或服务器都能跑，不涉及 FEM 求解

**M3b.5 · A/B 对照训练 + 图表**（服务器 GPU）

数据齐后：

1. **A 组**（HZ 监督基线）：`TrainConfig(..., supervision_source='sigma_h', lambda_eq=0.0)` 若干 epoch，记录 val R̃_h 曲线
2. **A' 组**（HZ + Stage D）：同 A 组但扫 `lambda_eq ∈ {0.01, 0.1, 1.0}`，观察 R̃_h 下降与 σ 拟合的 trade-off
3. **B 组**（σ_h^rec 监督）：`supervision_source='sigma_h_rec'`，同架构同 hyperparams
4. **B' 组**（σ_h^rec + Stage D）
5. **产图**（对应 `paper_thesis.md §F.3` "平台 vs 下降" 曲线）：R̃_h vs epoch，四条曲线；预期 HZ 组 → ε 可降，rec 组 → Θ(h^m) 平台锁死
6. **写进 B 论文**（`docs/operator_learning/paper_thesis.md` §5 / `docs/operator_learning/plan_operator_learning.md`）：λ·R_h² ablation + supervision-source 对照，作为命题 B2 + T2 的实验证据

**服务器 job 优先级**：M3b.4 先于 M3b.5；M3b.5 需要 GPU；两者与 D12 aux h2/h3 续算 (~1–2 周)、A adaptive model2 后处理相互独立，可并行。

**依赖表**：

| 步骤 | 阻塞条件 | 何时能起 |
|---|---|---|
| M3b.4 步 1（改 writer） | 需要读 HZ 求解器接口，找 u_h 在哪个字段 | 立即 |
| M3b.4 步 2（重生 dataset） | writer 改完 + 服务器时窗 | 1 天写代码 + 数天/一周跑数据 |
| M3b.4 步 3（add_stress_rec.py） | dataset 有 `displacement` 键 | offline，小时级 |
| M3b.5 全部 | 数据集有 `stress_rec` 键 + GPU 时窗 | 2–3 天训练 + 图表 |



> **输入源**：① `docs/planning/p1_action_checklist.md`（P1 止血清单）② `docs/planning/conf_202606_inspirations.md`（港中深会议迁移路径）③ `docs/NEXT_PAPER_DIRECTIONS.md`（A/B/C/D/E 五方向）④ `docs/preconditioner/D12_PRECONDITIONER_PAPER_PLAN.md` + `PIPELINE_STATUS.md`（D12 头条与在跑流水线）⑤ `docs/adaptive/*`、`docs/routes/plan_adaptive_aposteriori.md`（A 线理论与主循环代码就位）⑥ `docs/operator_learning/paper_thesis.md`（B 线主定理锁定 2026-06-02）⑦ `/Users/tian00/Desktop/gong办公资料/TalksAndPapers/SUMMARY.md`（龚世华 2018 博士论文 + 2015 H(div,S) 手稿）。
>
> **目的**：把当前散在各 `docs/` 里的论文/开发线拉平成**一张顺序表**——按「短期可发表 × 已有代码/理论 × 卖点稀缺度」三维排序，标出每一步的依赖、状态与产出。所有条目在原始规划文件里都能追溯，非重复内容。

---

## 1. 总优先级表（按"6 个月内可推进"排序）

| # | 类别 | 项目 | 状态 | 依赖 | 目标产出 | ETA |
|---|---|---|---|---|---|---|
| **T0** | 运维 | P1 遗留 + aux_h2/h3/model1 三段并行流水线 | 🟡 运行中（2026-06-04 起） | — | paper_aux 数据齐全 | 1–2 周 |
| **T2** | 论文 A | A tex 已成稿（`equilibrated_aposteriori.tex` 907 行含 Conclusion）→ 补 SENT shear 图表 + 已 sketch 的 equilibrating correction / spectral split majorant / marker efficiency 下界收尾 | 🟢 tex 骨架 100%，理论侧 A-2/3/4 已 sketch；SENT tension + CNT red-green + shear 数据齐 | T0 无关；只欠本地图表 | **CMAME/SINUM 主推** | **1–2 月（v0.8 提档）** |
| **T1** | 论文 D | D12 tex 已成稿（`phasefield_huzhang.tex` 3148 行含 Conclusion + Appendix）→ 补 Outlook 自认欠账 + §13 sweep 表 + 收稿 | 🟡 tex 骨架 100%，欠 SENT shear 完整 aux-vs-direct + 局部化 mesh-independence + shear 局部化 iteration 一栏 | T0（h2/h3/model1 pipeline 出货 + shear aux 起跑） | **SISC/CMAME 短稿** | 2–3 月 |
| **T3** | 论文 B | HZ-supervised 算子学习 Stage D + 对照表 + 数据重生 | 🟢 主定理锁定 2026-06-02，欠 `equilibrium_residual_l2` stub 实现 | T1/T2 无关，可并行 | **JCP/CMAME** | 5–6 月 |
| **T5** | 论文 D+ | 多水平/两层 Schwarz 预条件（博士论文第 6 章 + conf 胡齐芽） | 🟢 D12 Outlook 自己点名"contrast-adapted, interface-resolving coarse space"；paper_aux 数据可直接复用 | T1 tex 送审后启动 | D 论文续作 / SIMAX-NLAA | 8–12 月 |
| **T4** | 论文 A+ | Hu-Ma 扩展 H-Z 空间 + NVB 处理 L-shape（博士论文第 3 章 + conf 马睿线 2） | 🔵 fealpy 需新增顶点分裂逻辑；工程量最重 | T2 送审后接入 | A 论文附加章 / 独立短文 | 12–14 月 |
| **T6a** | 论文 C1 | 4 阶 PFM + 应变梯度 **on 标准/IP-FEM**（博论 IP-FEM 章直接延伸） | 🟢 博论已验可行；数值扩展 + 收敛率补齐即成稿 | T2 无关；预研阶段无阻塞 | Comput. Mech./IJNME | **8–12 月**（v0.5 新增，safe win） |
| **T6b** | 论文 C2 | 4 阶 PFM + 应变梯度 **on Hu-Zhang mixed**（离散升级 + inf-sup 重证） | 🟢 卖点与 T5 正交（mesh budget vs niter）；inf-sup 非现成推论 | T2 送审后接入代码；预研阶段无阻塞 | Comput. Mech./IJNME | **12–16 月**（v0.5 后置，novel） |
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

### Phase 1（2026-07 中 → 2026-09）— **T2 (A) 先送审，T1 (D12) 补 shear 后跟上**（v0.8）

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

### Phase 1 执行层清单（v0.9 · 追求"尽快投出两篇"）

**A 送审关键路径**（3 步 × 半天–2 天，1–2 周收官）

1. **今晚 scp**：`history.csv` + `step_032.vtu` (SENT shear model2 effstress job 目录) → 本地跑后处理出 force-disp + damage 场图 — 半天
2. **§num shear 骨架先摆**：图占位 + caption + 正文语气对齐 tension，scp 数据到直接塞 — 1 天，与 (1) 并行
3. **A-2/3/4 sketch 从 remark 提到 §body**：spectral split majorant 至少提成独立子小节（避免 reviewer 觉得藏在 remark 弱）— 1–2 天
4. **arXiv 同日挂 + CMAME**：CMAME 首审 2–3 月，arXiv 先占引用坑

**第二篇 · 双路并行 2 周，谁先齐谁投**

| 路径 | 阻塞 | 关键动作 | ETA |
|---|---|---|---|
| **α · D12** | shear aux pipeline **根本没起** | 立刻起 **shear aux h1**（比 h2/h3 快）→ 3–4 天过 peak 够 aux-vs-direct → 写作 2 周 | 8 月中 |
| **β · B/T3** | M3b.4 需要 dataset 加 `displacement` 键 | M3b.4 步 1（改 writer）立即起，不阻塞任何在跑 job → dataset 重生 ~1 周 → A/B 训练 2–3 天 | 8 月初 |

**期刊策略**

- A → **CMAME**（首审 2–3 月，比 SINUM 4–6 月快）
- D12 → **SISC**（短稿契合 letter 长度，审稿快过 CMAME）
- 同期投**避开同一 handling editor 池**，减少 reviewer overlap 风险

**明确不做**

- ❌ A 拆两投（CNT red-green 单独拎出）：削主论文卖点
- ❌ D12 v1 砍 shear：Conclusion 明写 shear 头条 selling point
- ❌ Phase 1 未清干净前起 T4/T6 预研占时窗

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

**T6 · C 线 · 4 阶 PFM 预研（v0.5 拆两路径：T6a std FEM safe win + T6b HZ mixed novel）**

**为何拆两条**：
- 博论 IP-FEM 4 阶章（`ttthesis/thesis/body/ipfem.tex`）**已把标准/IP-FEM 4 阶 PFM 的可行性验证完**——T6a 直接延伸即可成稿，理论风险最低，作 **safe win** 前置到 2027-06
- T6b 在 Hu-Zhang mixed setting 下做 4 阶：σ_h ∈ H(div,S) + ∇²d 耦合的 inf-sup 稳定性**要重证**（非 D12/A 现成结论的推论），novel 卖点更强，后置到 2027-12
- 两条共享 mesh budget ℓ/h≥4 → ≥2 + 尺寸效应的核心 selling point；T6a 提前占坑，T6b 补理论深度

**T6a 预研任务**（不阻塞 T2/T3/T5 主推；**理论先行，代码后置**——v0.6 依据 tex 现状盘点重排）：

**A. 理论 tex 侧**（`Tian/thesis/ip_fracture/ipfem_paper.tex` 上原地扩写，1187 → ~1600 行）：
1. **[核心 delta]** §Model 加应变梯度耦合项（Aifantis + Ali 2024），§Discretization 加对应 IP 处理与稳定性讨论——这是相对博论 IP-FEM 章的独家性来源
2. **[承接 Conclusion.③]** 同 mesh 同 p 下 2 阶 vs 4 阶直接对比章节：正文加半节 + 一组表
3. **[承接 Conclusion.②]** penalty 参数 γ 敏感性系统研究：加半节表（现有 tex 直接选 γ=5,10,20，未论证）
4. **[Conclusion.⑤ 桥接 T6b]** aposteriori-adaptive 展望段：明写待接 T2 (A 线) equilibrated estimator，作为 T6a → T6b 的桥梁

**B. 数值算例侧**（现有 model0/1 加算例，先在 fealpy_old 里跑，收官前再迁移）：
5. **[benchmark 补齐]** SENT tension（Miehe/Ambati 标准）新增一个算例，reviewer 通用；SENS/L-shape 视时间选做

**C. 代码实现侧**（作 T6b 前置依赖，v0.6 明确后置）：
6. **[fealpy3 迁移]** 现有 C⁰-IP + 4 阶 PFM 实现在 fealpy_old；在 fealpy3 里补齐实现——这是 T6b 的前置（T6b 的 HZ mixed 必须建在 fealpy3 上）

**优先级**：A.1 (应变梯度) > A.4 (aposteriori 展望) > A.2 (2 vs 4) > A.3 (penalty 敏感性) > B.5 (SENT) > C.6 (fealpy3 迁移)
**送审门槛（2027-06）**：A.1 必做（否则与博论重合），A.2/A.3 + B.5 建议做（rebuttal 常问），A.4 + C.6 可放 v2 或与 T6b 一起
**跳过**：Conclusion.① 参考解精细化（现有自参考够用）、Conclusion.④ 3D（工程量大，放 v2）

**T6b 预研任务**：
1. Hu-Zhang mixed 4 阶变分形式草稿：σ_h ∈ H(div,S) + ∇²d 耦合
2. inf-sup 稳定性草稿（重证，是 T6b 的核心理论工作量）
3. 与 T6a IP-FEM 结果对照的 baseline 表设计

**为何不进 Phase 2 主推**：
- T2 (A) 的 equilibrated estimator 是 T6b aposteriori 章要复用的家伙什，A 先送审稳
- T6a 代码接入可与 T2/T5 送审并行（因为不依赖 A/D 的产出）；T6b 代码接入等 T2 送审后

---

### Phase 3（2027-04 之后）— T6a 收稿 / T7 / T6b 收稿 / T8

**T6a · C1 线 · 标准/IP-FEM 4 阶 PFM 代码接入 + 论文成稿**（预研在 Phase 2 尾已启动）
- 把预研阶段 IP-FEM 数值扩展 + 应变梯度耦合落成 fracturex 代码
- 目标投稿 **2027-06**（v0.5 新增，safe win）

**T7 · 理论文 · 仿射不变 NEPIN + 全离散稳定性**（博士论文第 8–10 章 + `conf §3` 葛志昊）
- 用博士论文第 8 章的**仿射不变 Lipschitz 常数**理论，分析 fracturex staggered/Newton 收敛
- 补 fracturex 目前**只有单调性数值现象、没干净能量证明**的短板
- 输出：一篇偏理论的 SINUM/M2AN 文章
- 投稿 **2027-09**（独占档位，v0.5 T6 拆分后不再与 C 撞档）

**T6b · C2 线 · Hu-Zhang mixed 4 阶 PFM 代码接入 + 论文成稿**（预研在 Phase 2 尾已启动）
- 把 inf-sup 稳定性草稿落成严格证明；4 阶 mixed 变分形式接 T2 equilibrated estimator 做 aposteriori 章
- 与 T6a IP-FEM 结果做对照 baseline，突出 σ_h ∈ H(div,S) 逐元平衡的独家性
- 目标投稿 **2027-12**（v0.5 后置，novel 理论工作量）

**T8 · E 线 · 可微 HZ-PFM + AuTO 拓扑韧化**（远景）
- 依赖 T3 完成 + fealpy jax backend 稳定
- SH-com 弹簧网络最小模型（Fucheng Tian PNAS 2025）+ JAX AD 端到端

---

## 3. 论文清单（按投稿时间排序）

| 顺序 | 代号 | 标题草案 | 期刊首选 | 关键 selling point | 预计投稿 |
|---|---|---|---|---|---|
| 1 | **A** | Equilibrated a posteriori error estimation and σ-driven adaptivity for Hu-Zhang mixed phase-field fracture | CMAME / SINUM | 无常数超圆界；osc(f)=0 干净；超过郭雯 2024 的 CPU/mesh 节省 | **2026-07（v0.8 提档）** |
| 2 | **D12** | Auxiliary-space preconditioning for Hu-Zhang mixed phase-field fracture: uniqueness of convergence in the fully-localized regime | SISC / CMAME | 难 regime 唯一收敛；引龚博论 Ch 7 为理论根基 | 2026-09 |
| 3 | **B** | Balance-preserving neural operators for phase-field fracture via Hu-Zhang supervision | JCP / CMAME | 平衡监督 = 结构最优；R̃_h 平台 vs 下降曲线；诚实边界 T1/D1 | 2027-01 |
| 4 | **D+** | Non-nested coarse spaces and two-level Schwarz preconditioning for Hu-Zhang phase-field fracture in the localized regime | SIMAX / NLAA | 龚博论 Ch 6 多水平 + 胡齐芽 GEP-coarse；D12 自己 Outlook 点名的续作 | 2027-03 |
| 5 | **A+** | Extended Hu-Zhang element with vertex-tangent relaxation for adaptive elasticity at reentrant corners | Math. Comp. / M2AN | 合并 Hu-Ma 2020 + 龚博论 Ch 3 任意维奇异点代数定义 | 2027-08（v0.5 后移 1 档给 T6a 让路） |
| 6 | **T6a** | Fourth-order phase-field fracture with strain-gradient elasticity via interior-penalty FEM | Comput. Mech. / IJNME | 博论 IP-FEM 章直接延伸；mesh budget 放松 + 尺寸效应；safe win 前置占坑 | **2027-06**（v0.5 新增） |
| 7 | **T7** | Affine-invariant analysis of staggered Newton for Hu-Zhang phase-field fracture | SINUM / M2AN | 龚博论 Ch 8–10 NEPIN 框架用到相场 staggered，补严格能量下降证明 | 2027-09 |
| 8 | **T6b** | Fourth-order phase-field fracture in Hu-Zhang mixed setting: inf-sup stability and equilibrated aposteriori | Comput. Mech. / IJNME / M2AN | σ_h ∈ H(div,S) + ∇²d 耦合 inf-sup 重证；接 T2 estimator；对 T6a 的 novel 升级 | **2027-12**（v0.5 新增） |
| 9 | **E** | Differentiable Hu-Zhang phase-field for topology-toughening design | ML4Science / PNAS | JAX 端到端 + SH-com 最小模型 | 2028 |

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

### 4.3 P2 · Phase 2（Q4 2026 – Q1 2027）· T3 完稿 + T5 D+ 主推，T4/T6 预研

- [ ] T5.1 GEP-coarse 离线实验（paper_aux 数据复用）
- [ ] T5.2 两层 Schwarz vs 现 aux-space 对照（重点 max d → 0.99 区段）
- [ ] T5.3 D+ 论文成稿（SIMAX/NLAA，2027-03 投稿）
- [ ] T3.M4（选做）§I Airy 势硬约束 ablation
- [ ] T4.1 fealpy 复现 Hu-Ma §5.2 rotated L-shape（A 送审后启动）
- [ ] T4.2 corner relaxation 装配模块升级（对接 `architecture/huzhang_corner_relaxation_design.md`）
- [ ] T4.3 扩展 H-Z 空间嵌回 fracturex model2 + L-shape
- [ ] **T6a.pre.A1** [核心 delta] `ipfem_paper.tex` §Model 加应变梯度耦合项 + §Discretization 加 IP 处理与稳定性（Aifantis + Ali 2024，v0.6 重排）
- [ ] **T6a.pre.A2** `ipfem_paper.tex` 加"同 mesh 同 p 下 2 阶 vs 4 阶直接对比"章节 + 表（承接 Conclusion.③）
- [ ] **T6a.pre.A3** `ipfem_paper.tex` 加 penalty γ 敏感性系统研究章节 + 表（承接 Conclusion.②）
- [ ] **T6a.pre.A4** `ipfem_paper.tex` Conclusion 加 aposteriori-adaptive 展望段，桥接 T2 (A) equilibrated estimator（承接 Conclusion.⑤，T6a → T6b 桥梁）
- [ ] **T6a.pre.B5** SENT tension（Miehe/Ambati 标准）新增算例，先在 fealpy_old 跑
- [ ] **T6a.pre.C6** fealpy3 里补齐 C⁰-IP + 4 阶 PFM 实现（现在 fealpy_old）——T6b 前置依赖
- [ ] **T6b.pre.1** Hu-Zhang mixed 4 阶变分形式草稿：σ_h ∈ H(div,S) + ∇²d 耦合
- [ ] **T6b.pre.2** inf-sup 稳定性草稿（4 阶下重证，非 D12/A 推论）
- [ ] **T6b.pre.3** 与 T6a IP-FEM 结果对照的 baseline 表设计

### 4.4 P3 · Phase 3（Q2 2027+）
- [ ] T6a.1 IP-FEM 4 阶 PFM 代码接入 fracturex（预研落地）
- [ ] T6a.2 T6a 论文成稿（Comput. Mech./IJNME，**2027-06 投稿**，v0.5 safe win）
- [ ] T7 仿射不变 Lipschitz 用到 staggered 收敛证明（**2027-09 投稿**）
- [ ] T6b.1 HZ mixed 4 阶 PFM 代码接入 fracturex
- [ ] T6b.2 inf-sup 严格证明 + 接 T2 estimator 做 aposteriori 章
- [ ] T6b.3 T6b 论文成稿（Comput. Mech./IJNME/M2AN，**2027-12 投稿**，v0.5 novel）
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

**与田甜博士论文（`ttthesis/thesis/`）的直接对接**（"prior work by the author" 口径）：

| 论文 | 引田博论章节 | 用途 |
|---|---|---|
| A | `body/afem.tex` recovery-based AFEM 章节 | §Related Work / §Baseline 对照——从 recovery-based 升到 equilibrated 的理论 selling point |
| C1 (T6a) | `body/ipfem.tex` 内罚 4 阶 PFM 章节 | §Method——直接延伸博论 IP-FEM 的新算例 + 收敛率 + 应变梯度耦合 |
| C2 (T6b) | `body/ipfem.tex` 同上 | §Baseline / §Related Work——对照 IP-FEM 结果，突出 HZ mixed 升级的 inf-sup + 平衡性 selling point |
| B / E | `body/design.tex` + `design1.tex` FractureX 平台设计 | §Implementation / §Software——平台底座出处，NumPy/PyTorch backend + 模块化架构 |
| 所有 | `body/mfem.tex`（任意次 tensorized FEM）+ `body/phase_theory.tex`（能量泛函/本构/张量化） | model2 baseline 与相场理论表述的出处 |

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

## 8. 背景：龚世华 2018 博士论文 vs fracturex 五条论文线

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

## 9. 背景：田甜博士论文 vs fracturex 五条论文线（自己的祖谱）

**博士论文（`ttthesis/thesis/tianPHD.tex`, 已完成）是 fracturex 工程与部分方法学的直接前身**，四项创新与五+条论文线的接续关系：

| 博论创新 | 内容 | 对应 fracturex 线 | 接续方式 |
|---|---|---|---|
| **创新 1** 任意次 tensorized FEM（任意维/任意网格/任意阶） | 张量化实现 Lagrange 相场 baseline | **所有 model2 baseline 的底座**（T1/T2/T3 共享） | ✅ 已在 fracturex 内部使用；论文里引博论作 baseline 出处，避免重复推导 |
| **创新 2** Recovery-based AFEM + matrix-free + GPU | 基于重构型误差估计的自动加密，避免用户阈值 | **T2 (A 线) equilibrated aposteriori 的直接前身** | **理论升级**：recovery-based → equilibrated（无常数、保证型界）。A 论文 §Related Work 里把博论作 "prior work by the author"，明确升级路径 |
| **创新 3** IP-FEM 4 阶 PFM | 内罚有限元求解 4 阶相场，减少非物理振荡；p 提高时 mesh 依赖弱化 | **T6a (C1 线) 标准/IP-FEM 4 阶 PFM 的直接延伸** + **T6b (C2 线) Hu-Zhang mixed 4 阶 PFM 的对照 baseline** | **两路径**：T6a 数值扩展博论章节即成稿（safe win, 2027-06）；T6b 做 IP-FEM → Hu-Zhang mixed 的离散升级（inf-sup 重证, novel, 2027-12） |
| **创新 4** FractureX 平台（NumPy/PyTorch backend, 模块化, GPU/CPU 切换） | 网格/离散/后端解耦；VTK 可视化 | **T3 (B 线) / T8 (E 线) 的平台底座**；也是 T0 pipeline 的运行时 | ✅ 已在用；B 论文数据管线、E 论文 JAX backend 都在此平台上扩展 |

**结论**：博论四项创新与主计划的关系是「**baseline → 升级**」而非「**并列**」——
- **A 线**明确从 recovery-based 升到 equilibrated（理论 selling point）；
- **C 线**拆两路径：**C1/T6a** 直接延伸博论 IP-FEM 章（safe win, 2027-06）+ **C2/T6b** 升到 Hu-Zhang mixed（inf-sup 重证 novel, 2027-12）；
- **B/E 线**则**直接站在博论 FractureX 平台之上**扩展。

论文写作时的一致口径：**"prior work by the author (Tian PhD thesis)"** 作为对照或前身来引，突出主计划里各线的方法学升级点。

---

## 附 A：T6a tex 现状盘点（`Tian/thesis/ip_fracture/ipfem_paper.tex`，v0.6 新增）

**tex 骨架**（1187 行）：

| 章节 | 现状 | v0.6 gap |
|---|---|---|
| §1 Introduction | ✅ 完整 | — |
| §2 Mathematical model（4 阶 PFM + hybrid split） | ✅ 完整 | **A1** 缺应变梯度耦合项（核心 delta） |
| §3 Discretization and solution strategy（C⁰-IP + staggered） | ✅ 完整 | **A1** 缺应变梯度对应 IP 处理；**A3** penalty γ 直接选 5/10/20 未论证 |
| §4 Error analysis（stability + coercivity + a priori $h^{p-1}$） | ✅ 完整 | — |
| §5 Numerical experiments（circular hole + notch） | ✅ p × h 二维扫描全；六图齐 | **A2** 缺 2 阶 vs 4 阶直接对比章；**B5** 缺 SENT tension benchmark |
| §6 Conclusion | ✅ 骨架 | **A4** 缺 aposteriori-adaptive 展望桥接 T2 |

**Conclusion 自认五条 open extension 的处理**：

| 序号 | 内容 | v0.6 处理 |
|---|---|---|
| ① | 参考解精细化（更细 mesh 或 benchmark） | ⏭️ 跳过（现有自参考够用） |
| ② | penalty γ 敏感性 | ✅ **A3 补** |
| ③ | 同 mesh 同 p 下 2 阶 vs 4 阶 | ✅ **A2 补** |
| ④ | 3D + solver scalability | ⏭️ v2 或跳过（工程量大） |
| ⑤ | aposteriori-adaptive | ✅ **A4 桥接**——正好接 T2 送审后 estimator，作 T6a → T6b 桥梁 |

**结果目录**（`ipfem_fp_results/`）：
- `ipfem_fp_model0/` — circular hole，p=2,3,4 × 多 h，disp_node + results.txt 齐
- `ipfem_fp_model1_result/` — notch，p=2,3,4 × 多 n，force curve + zoom 图齐

**fealpy 版本状态**：**现有实现在 fealpy_old**；fealpy3 侧的 C⁰-IP + 4 阶 PFM 尚未落地（**T6a.pre.C6**）——**是 T6b 的前置依赖**（HZ mixed 4 阶必须建在 fealpy3 上，不能回退到 fealpy_old）。

**理论先行策略**（v0.6 明确）：
- 优先级 A（tex 理论） > B（新算例，先跑 fealpy_old）> C（fealpy3 迁移，T6b 之前必须）
- 2027-06 送审门槛只要 A.1 必做；A.2/A.3/B.5 建议做；A.4/C.6 可放 v2 或与 T6b 一起

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
- 田博论根：`ttthesis/thesis/tianPHD.tex`（已完成）
- 田博论摘要（四大创新）：`ttthesis/thesis/preface/abstract.tex`
- 田博论 recovery AFEM 章：`ttthesis/thesis/body/afem.tex`（A 线前身）
- 田博论 IP-FEM 4 阶 PFM 章：`ttthesis/thesis/body/ipfem.tex`（C 线前身）
- **T6a tex 底稿**：`Tian/thesis/ip_fracture/ipfem_paper.tex`（1187 行，v0.6 主战场）
- **T6a 结果目录**：`Tian/thesis/ip_fracture/ipfem_fp_results/`（model0/model1，fealpy_old 出）
- **T6a 图**：`Tian/thesis/ip_fracture/figures/`（model0/1 geometry + damage + force 共 6 张）
- 田博论 FractureX 平台设计章：`ttthesis/thesis/body/design.tex` + `design1.tex`（B/E 平台底座）
- 田博论任意次 tensorized FEM 章：`ttthesis/thesis/body/mfem.tex`（所有 model2 baseline）
- 田博论相场理论章：`ttthesis/thesis/body/phase_theory.tex`（能量泛函 / 张量化表达）
