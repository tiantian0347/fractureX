# fracturex 论文 + 开发 主计划（2026-08-25 v2.6）

> **v2.6**：完成论文第二阶段压缩。正文保留“诊断--定位--消元--验收”主线、核心理论、收益表、网格证据和 reference-free 验收；实现细节与审计参数收敛为简短说明，当前 PDF 为 24 页（含附录和参考文献）。


## 服务器 job 状态（2026-08-20 09:55 CST 核对）

| Job | 现状 | 数据到 | 下一步 |
|---|---|---|---|
| **A adaptive model2 effstress** | ✅ **已跑完 + 已同步本地** (7/3 17:31) | step 33 fallback 崩停；有效数据到 step 32 (R=0.156) | 本地 `huzhang_fracture_result/results_model2/adaptive_m3_pc_model2_effstress/`；后处理 SENT shear 图表补进 A tex §sec:num (A-1) |
| **D12 aux h2** | 🔴 **已停，无运行进程**；7/4 resume 只恢复 checkpoint，未产出 step 14 | history/checkpoint 均到 **step 13/31**，u_y=0.0876，max_d=0.4264 | 决定是否重启；若保留 D12 完整 mesh-independence 结论，需从 step_013.npz 续算 |
| **D12 aux h3** | 🔴 **已停，无运行进程**；7/4 resume 只恢复 checkpoint，未产出 step 14 | history/checkpoint 均到 **step 13/31**，u_y=0.0876，max_d=0.4193 | 同 h2；重启前先评估局部化区成本与论文最低数据需求 |
| **D12 model2 direct pardiso_gmres** | 🟡 **在跑**（PID 4142145，6/29 起） | 已完成 **step 171/240**，u_y=0.01425，max_d=1.0；step 171 用 482 次 staggered iteration 收敛 | 继续到目标步 240；已过 peak，但后局部化单步成本很高 |
| **D12 square/model0 direct** | 🟡 **在跑**（PID 3782490，6/18 起） | 已完成 **step 81**，u_y=0.00531，max_d=1.0；step 81 用 10 次 iteration 收敛 | 保留运行；明确该数据在 D12/SENS 对照中的用途 |
| **Model5 standard FEM h=0.015** | ✅ **已跑完 + 已同步本地** (8/19 23:40) | u=0→0.06 mm 两段 continuation；peak \|R\|≈0.0411 @ u≈0.047 mm（≈Ambati 0.042）；末步 maxit=80 未收敛 | 本地 `huzhang_fracture_result/phasefield/model5_standard_fem/std_bg_h015_smoke_a{_cont}/`；更新 `PHASEFIELD_BENCHMARKS.md` |
| **M3b A/A' sweep** | ✅ **已完成 + 已同步本地** (7/6) | λ_eq∈{0,0.01,0.1,1.0} 四组；σ_rel L2≈0.986–0.988，λ_eq 无明显改善 | 本地 `huzhang_fracture_result/results/learn/m3b_hz_sweep/`；见 `m3b_stageD_status.md` §5 |
| **paper_aux_h1** | ✅ **已同步本地** | VTU/checkpoint 到 step 30；history 仅到 step 16 | 本地 `huzhang_fracture_result/results_model0/paper_aux_h1/epsg_1e-06/` |
| **M3b hires dataset** | 🟡 **在跑**（PID 291306，7/5 起） | sample_000004 已完成 step 15；sample_000005 已完成 step 6，正在 step 7 | 继续生成；完成后做 stress_rec 与 B/B' sweep |
| **Model4 standard FEM h=1.0** | 🟡 **在跑** PID **3851518**（2026-08-20 11:36 CST） | `path`：gmres，`Δu=0.01` 到 2 mm / 200 步；NN=9605 NC=18689；step 0 已 4 NR 收敛 | `results/phasefield/model4_standard_fem/std_h1_path/`；log `results/logs/model4_std_fem_path.nohup` |
| **Model6 standard FEM h=0.15** | 🟡 **在跑** PID **3846352**（2026-08-20 11:32 CST） | `path`：gmres，u=0→0.25 mm / 250 步；NN=9435 NC=18442；step 1 起 3 NR/步 | `results/phasefield/model6_standard_fem/std_h015_path/`；log `results/logs/model6_std_fem_path.nohup` |

**本轮 tex 改动（2026-07-04）**：

- D12 (`phasefield_huzhang.tex`, 36pp)：**D-7 完成**——ref.bib 加 Cervera 2010 Part I/II + 2017 三条，intro 加 VMS-OSGS 对照段落
- A (`equilibrated_aposteriori.tex`, 18pp)：**A-2/3/4 完成**——`rem:gen-data` 扩成 Braess-Schöberl patch-local 校正 sketch；`rem:split` 加 Fenchel 对偶 majorant；`rem:marker-eff` 加 η_ω_z local lower bound；Conclusion 相应重写把三条 open extension 从"待做"改为"已 sketch + companion paper"

**剩余 gap 状态**：

- 🔴 D-1..D-5 (SENT/SENS aux 端到端 + 局部化 mesh-indep)：**h2/h3 均停在 step 13/31**，需明确重启或缩减论文结论范围
- 🟢 A-1 (SENT shear 场景)：**数据已 scp 至 `huzhang_fracture_result/results_model2/adaptive_m3_pc_model2_effstress/`**，待本地后处理图表
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

## T3.M3b 服务器进展（2026-08-20 更新）

**M3b.4 · 数据管线：把 σ_h^rec 写进 npz 数据集** ✅ 完成 (2026-07-05)

**关键决策**：**不用重生 dataset**。HZ solver 的 checkpoints (`runs/sample_XXX/checkpoints/step_XXX.npz` 的 `u` 键) 已经存了 P2-DG × 2 分量的位移 FE DOF 向量。offline 采样到网格即可。

- 新脚本 `scripts/datasets/add_stress_rec.py`（scripts/datasets/ 而不是 data_generation/，跟其他数据生成脚本目录一致）
- 关键实现：自写向量化 barycentric point-locate（`mesh.location` 是 fealpy 空占位）+ 硬编码 fealpy DG P2 local DOF 顺序 `[v0, mid(v0v1), mid(v0v2), v1, mid(v1v2), v2]`
- Round-trip 验证：已知位移 u=(x, 2y) 经 `tspace.interpolate` → 我的 sampler → 网格误差 = 4e-16（机器精度）
- σ_h^rec / p95 归一化至 O(1) 训练空间；非收敛步用 σ_h 覆盖避免裂尖数值污染
- **观测到裂尖 10⁴× 于 p95 的重尾**（p95=1.0 vs max=9780 in sample_000000）—— σ_h^rec ∉ H(div,S) 的法向跳跃 pathology，正是 paper_thesis §F.3 想暴露的对照点
- 详细状态：`docs/operator_learning/m3b_stageD_status.md`

**M3b.5 · A/B 对照训练** 🟡 A/A' sweep 已完成，B/B' 待 hires 数据

已完成（log `/tmp/m3b_hz_sweep.log`，summary 为 `results/learn/m3b_hz_sweep/sweep_summary.json`）：

- **A/A' sweep**（HZ σ_h supervision）：`run_m3b_lambda_eq_sweep.py` × λ_eq ∈ {0, 0.01, 0.1, 1.0}
- 数据集：m1_pilot（train=19/test=8，64×64）
- 模型：`multioutput_fno`, Stage B, 100 epochs
- 四组训练与评估均已结束；σ relative L2 约 0.986–0.988，未见 λ_eq 带来明显改善
- hires 数据生成仍在跑：sample_000004 已完成，sample_000005 正在生成

**待跑（hires 数据生成与 stress_rec 批处理完成后）**：

- **B/B' sweep**（σ_h^rec supervision）：同架构同 hyperparams，加 `--supervision sigma_h_rec --sigma-transform arcsinh`（压缩 10⁴× 重尾）
- **产图**（对应 `paper_thesis.md §F.3` "plateau vs descent"）：R̃_h vs epoch，四条曲线；预期 HZ 组 → ε 可降，rec 组 → Θ(h^m) plateau
- **写进 B 论文**（`docs/operator_learning/paper_thesis.md` §5 / `plan_operator_learning.md`）：λ·R_h² ablation + supervision-source 对照，作为命题 B2 + T2 的实验证据



> **输入源**：① `docs/planning/p1_action_checklist.md`（P1 止血清单）② `docs/planning/conf_202606_inspirations.md`（港中深会议迁移路径）③ `docs/NEXT_PAPER_DIRECTIONS.md`（A/B/C/D/E 五方向）④ `docs/preconditioner/D12_PRECONDITIONER_PAPER_PLAN.md` + `PIPELINE_STATUS.md`（D12 头条与在跑流水线）⑤ `docs/adaptive/*`、`docs/routes/plan_adaptive_aposteriori.md`（A 线理论与主循环代码就位）⑥ `docs/operator_learning/paper_thesis.md`（B 线主定理锁定 2026-06-02）⑦ `/Users/tian00/Desktop/gong办公资料/TalksAndPapers/SUMMARY.md`（龚世华 2018 博士论文 + 2015 H(div,S) 手稿）。
>
> **目的**：把当前散在各 `docs/` 里的论文/开发线拉平成**一张顺序表**——按「短期可发表 × 已有代码/理论 × 卖点稀缺度」三维排序，标出每一步的依赖、状态与产出。所有条目在原始规划文件里都能追溯，非重复内容。

---

## 1. 总优先级表（按"6 个月内可推进"排序）

| # | 类别 | 项目 | 状态 | 依赖 | 目标产出 | ETA |
|---|---|---|---|---|---|---|
| **T0** | 运维 | 服务器长任务收尾：model2/square direct + M3b hires；处理 aux h2/h3 停跑 | 🟡 direct/dataset 在跑；🔴 aux h2/h3 停在 step 13 | — | paper_aux 与 direct 对照数据齐全 | 先决策 aux 是否重启 |
| **T2** | 论文 A | A tex 已成稿（`equilibrated_aposteriori.tex` 907 行含 Conclusion）→ 补 SENT shear 图表 + 已 sketch 的 equilibrating correction / spectral split majorant / marker efficiency 下界收尾 | 🟢 tex 骨架 100%，理论侧 A-2/3/4 已 sketch；SENT tension + CNT red-green + shear 数据齐 | T0 无关；只欠本地图表 | **CMAME/SINUM 主推** | **1–2 月（v0.8 提档）** |
| **T1** | 论文 D | D12 tex 已成稿（`phasefield_huzhang.tex` 3148 行含 Conclusion + Appendix）→ 补 Outlook 自认欠账 + §13 sweep 表 + 收稿 | 🟡 tex 骨架 100%，欠 SENT shear 完整 aux-vs-direct + 局部化 mesh-independence + shear 局部化 iteration 一栏 | T0：h2/h3 已停在 step 13，须重启或缩减结论范围 | **SISC/CMAME 短稿** | 待 aux 决策后重估 |
| **T3** | 论文 B | HZ-supervised 算子学习 Stage D + 对照表 + hires 数据生成 | 🟡 M3a 与 HZ A/A' sweep 已完成；hires dataset 在跑，欠 σ_h^rec 的 B/B' 对照 | T1/T2 无关，可并行 | **JCP/CMAME** | hires 完成后进入 B/B' sweep |
| **T9** | 论文 F | FractureX 框架论文：一套解耦架构承载三种离散范式（标准 Lagrange / HZ 混合 / C⁰-IP 4 阶）+ 并行组装 + 双块最优求解器 | 🟢 代码/文档骨架齐；欠 scaling benchmark（本地快出，不卡服务器） | 引 D12/T6a/田博论方法；预研无阻塞 | **CiCP（与 FEALPy v3 同刊）** | 详见 `T9_FRAMEWORK_PAPER_PLAN.md`；≈2027 Q1（benchmark 后） |
| **T5** | 论文 D+ | 多水平/两层 Schwarz 预条件（博士论文第 6 章 + conf 胡齐芽） | 🟢 D12 Outlook 自己点名"contrast-adapted, interface-resolving coarse space"；paper_aux 数据可直接复用 | T1 tex 送审后启动 | D 论文续作 / SIMAX-NLAA | 8–12 月 |
| **T4** | 论文 A+ | Hu-Ma 扩展 H-Z 空间 + NVB 处理 L-shape（博士论文第 3 章 + conf 马睿线 2） | 🔵 fealpy 需新增顶点分裂逻辑；工程量最重 | T2 送审后接入 | A 论文附加章 / 独立短文 | 12–14 月 |
| **T6a** | 论文 C1 | 4 阶 PFM + 应变梯度 **on 标准/IP-FEM**（博论 IP-FEM 章直接延伸） | 🟢 **tex 2345 行成稿**，应变梯度/penalty/aposteriori 桥接/manufactured-solution 全落地；不卡算例 | 无（本地即可投） | Comput. Mech./IJNME | **立即投（2026-07，v1.0 插队队首）** |
| **T6b** | 论文 C2 | 4 阶 PFM + 应变梯度 **on Hu-Zhang mixed**（离散升级 + inf-sup 重证） | 🟢 卖点与 T5 正交（mesh budget vs niter）；inf-sup 非现成推论 | T2 送审后接入代码；预研阶段无阻塞 | Comput. Mech./IJNME | **12–16 月**（v0.5 后置，novel） |
| **T7** | 论文 Solver | Coupled slow subspace localization and local nonlinear elimination | 🟡 六状态启动准则已完成；待在线区域与网格验证 | T2/T3 不阻塞 | CMAME / SINUM / SISC | 主线 |
| **T8** | 论文 E | 可微 HZ-PFM + AuTO 拓扑韧化 | 🔵 需 JAX 端到端 | T3 完成 + jax backend | ML4Science / PNAS 短篇 | 20–24 月 |

**颜色**：🟡 在跑/在写 🟢 立即可起 🔵 依赖前置

---

## 2. 详细排期（半年视角）

### Phase 0（2026-07 上）— 止血 & 数据齐全

**T0 剩余动作**（`p1_action_checklist.md` §"P1 完成判定"未打勾项）
1. Lagrange 路线 `MainSolve` 在 model0 上跑通（C5 对照必备）
2. EXPERIMENT_MATRIX 自动扫描脚本（P0 优先级不高，0.5 天可写）
3. aux_h2/h3 已确认停在 step 13/31；model1/square direct 已到 step 81，model2 direct 已到 step 171/240。先决定 aux 是否重启；若不重启，T1 必须缩减局部化 mesh-independence 的结论范围

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

**T7 · Solver 线 · coupled slow subspace 与局部非线性消元**（博士论文第 8–10 章）
- H1--H6、五档预算和六状态启动准则已完成；
- 下一步：在线慢空间与网格稳健性；
- 历史场、动态、非对称投影和拓扑粗空间暂存为 \`T7_COUPLED_SLOW_SUBSPACE_PLAN.md\` 的后续分支；
- 目标输出：一篇以慢子空间 survival factor 为核心、面向可靠条件加速的求解器方法论文，投稿 **2027-09**。

**T6b · C2 线 · Hu-Zhang mixed 4 阶 PFM 代码接入 + 论文成稿**（预研在 Phase 2 尾已启动）
- 把 inf-sup 稳定性草稿落成严格证明；4 阶 mixed 变分形式接 T2 equilibrated estimator 做 aposteriori 章
- 与 T6a IP-FEM 结果做对照 baseline，突出 σ_h ∈ H(div,S) 逐元平衡的独家性
- 目标投稿 **2027-12**（v0.5 后置，novel 理论工作量）

**T8 · E 线 · 可微 HZ-PFM + AuTO 拓扑韧化**（远景）
- 依赖 T3 完成 + fealpy jax backend 稳定
- SH-com 弹簧网络最小模型（Fucheng Tian PNAS 2025）+ JAX AD 端到端

---

## 2.5 求解器论文方向（2026-08 决策版，索引）

求解器论文正文现在只保留一条主线，数学模型、算法和验证计划统一以
[T7_COUPLED_SLOW_SUBSPACE_PLAN.md](T7_COUPLED_SLOW_SUBSPACE_PLAN.md)
和 \`Tian/thesis/phasefield_solver/phasefield_solver.tex\` 为准：

- **唯一主问题**：如何依据 coupled slow subspace 的消除收益与局部代价选择强耦合区域，并保持原 KKT 离散解？
- **中心算子**：\(G\)、\(\mathcal V_r\)、\(Q_\omega G\)；
- **一般中心定理**：\(\|Q_\omega Gw\|_W/\|w\|_W=|\lambda|\chi_{\omega,W}(w)\)；
- **SPD 推论**：\(\chi_{\omega,J}(w)=\sqrt{1-c_\omega(w)}\)。

本节其余内容为历史候选记录，不再作为当前论文结构或执行入口。当前执行顺序为六点启动准则、在线增量慢空间和网格稳健性。

这一节是当前所有“下一篇论文 idea”的唯一入口。它把龚世华博士论文第 8–10 章的仿射不变量、非精确 Newton 和 NEPIN 思路，与 fractureX 当前的历史场、交替求解和路径敏感性问题接起来。

### 2.5.1 历史候选记录（冻结，不作为当前执行入口）

以下内容保留历史讨论和备选实现。当前论文主线、数学对象和实验顺序以
\`T7_COUPLED_SLOW_SUBSPACE_PLAN.md\`为准；本节不再扩展。

**题目草案：**

> Transactional History Fields and Affine-Invariant Process-Zone Nonlinear Elimination for Phase-Field Fracture

主线分为两个互相依赖的贡献：

1. **事务型历史场。** 一个物理载荷步开始时保存已提交历史场 \(H_n\)。非线性迭代只生成

   \[
   H_{\mathrm{trial}}^k
   =
   \max\{H_n,\psi^+(\varepsilon(u^k))\},
   \]

   只有载荷步通过统一真残量、不可逆性和能量判据后才 commit；失败或减小步长时 rollback。这样当前载荷步对应固定的 \(G_{H_n}\)，Anderson、固定点收缩率和 Newton 判据才作用于同一个数学对象。

2. **过程区联合 NEPIN。** 冻结 \(u,H\) 后，标准 AT2 相场子问题通常是线性的，因此“只对损伤场做局部 NEPIN”不够有力。真正需要消除的是裂纹过程区内联合的

   \[
   \begin{bmatrix}
   F_u(u,d)\\
   F_d(u,d;H_n)
   \end{bmatrix}_{\omega_{\mathrm{pz}}}=0.
   \]

   先用仿射不变量、\(u\)-\(d\) 耦合强度、损伤梯度和活动集变化率选出移动 patch，再在 patch 内联合求解，区域外保留便宜的全局迭代。

核心因果链是：

> 非线性试算写入历史场
> \(\rightarrow\) 载荷步映射不固定、路径产生响应带宽
> \(\rightarrow\) 事务型 commit/rollback 恢复固定问题
> \(\rightarrow\) 成核阶段慢模态局部化到过程区
> \(\rightarrow\) 局部 \(u\)-\(d\) 消元恢复快速收敛。

### 2.5.2 代码与结果依据

- phasefield/main_solve.py 在相场组装路径中调用历史场最大值；
- damage/phasefield_damage.py 的历史更新函数执行累计最大值；
- drivers/huzhang_phasefield_staggered.py 在外迭代中更新 \(H\)，Anderson 主要作用于 \(d\)；
- analysis/affine_invariant.py 已有诊断原型，但需区分真正 Newton 修正和 staggered 固定点增量；
- docs/adaptive/RESULTS_aposteriori.md 已记录 Anderson 从 6.98 h 降至 1.13 h，以及局部化后峰值载荷约 \(\pm4\%\) 的路径带宽；
- Tian/thesis/fracture_huzhang/phasefield_huzhang.tex 已区分收敛态历史场与算法累计历史场，并给出 \(H_{\mathrm{alg}}\ge H_{\mathrm{conv}}\) 的理论种子；
- docs/preconditioner/THEORY_nonlinear_elimination.md 需要按“过程区内联合 \(u\)-\(d\) 消元”重新收紧，而不是继续扩展 damage-only NEPIN。

### 2.5.3 文献边界与可用空白

已有工作已经覆盖：

| 已有方向 | 代表文献 | 本计划避开的弱贡献 |
|---|---|---|
| 交替求解与 Anderson | Farrell & Maurini 2017；Storvik et al. 2021 | 只调 Anderson 深度或松弛参数 |
| matrix-free multigrid | Jodlbauer et al. 2020 | 只更换 AMG/GMG |
| 全场 nonlinear field-split / SPIN | Kopaničáková et al. 2023 | 把全场 staggered sweep 改名为 NEPIN |
| 域分解与浮动子结构 | Rannou & Bovet 2024 | 只在固定几何子域上局部求解 |
| TNNMG 与不可逆约束 | Gräser et al. 2023 | 只更换 active-set 求解器 |
| 通用 NEPIN | Cai & Keyes 2002；Gong & Cai 2019；Liu et al. 2022 | 不揭示断裂中的慢模态来源 |
| 局部极小与分岔 | Terzi et al. 2025 | 不把“存在多解”本身作为新发现 |

本轮检索未发现与以下三点直接对应的相场断裂工作：

1. 带 commit/rollback 语义的事务型历史场，以及它对固定点映射、算法路径和断裂响应的系统影响；
2. 由仿射不变量和耦合指标自动选择移动过程区，并在其中联合消除 \(u\)-\(d\) 强耦合；
3. 由损伤拓扑自动生成裂纹片段近刚体模态粗空间。

### 2.5.4 两个必须先做的 go/no-go 实验

**实验 A：历史场事务对照。**
在同一已收敛 checkpoint 上比较 cumulative 与 transactional，组合 plain staggered / Anderson \(m=3,5,8\)、\(10^{-4},10^{-6},10^{-8}\) 容差和不同初值。记录历史场差值、峰值、耗散能、裂纹路径、真残量、失败/回滚次数。

- Go：跨算法/容差的峰值或耗散能离散降低至少 50%，或观察到可重复的 cumulative 历史场过冲；
- 合并入主线：差异稳定但小于约 0.2%，作为过程区 NEPIN 的问题定义部分；
- 停止独立论文：多个局部化模型和故意较差初值下均无可测差异。

**实验 B：staggered 慢模态定位。**
在峰前、成核和扩展三个 checkpoint 组装

\[
T_{\mathrm{stag}}=D^{-1}CA^{-1}B,
\]

用矩阵自由方法计算领先特征对，统计单元能量与过程区覆盖率，再做一次理想 patch 联合消元。

- Go：峰前 \(\rho(T_{\mathrm{stag}})<0.8\)，成核时 \(>0.95\)；不超过 20% 单元承载至少 70% 慢模态能量；理想 patch 能降低谱半径；
- 若慢模态不局部：转向 2.5.6 的裂纹拓扑近核粗空间，不把 patch 扩展成全场 SPIN。

### 2.5.5 过程区 NEPIN 的最小实现

定义指标

\[
\mathcal S_k=\{K:\omega_K>\theta_\omega
\ \text{or}\ \chi_K>\theta_\chi
\ \text{or}\ |\nabla d|_K>\theta_d\},
\]

并扩张一到两层单元形成 \(\omega_{\mathrm{pz}}^k\)。局部变换 \(T_{\mathcal S}\) 固定 patch 外自由度，在 patch 内联合求解两场方程与不可逆约束，外层求解

\[
\mathcal F(x)=F(T_{\mathcal S}(x))=0.
\]

局部问题采用自适应非精确容差

\[
\|F_{\mathcal S}(z^\ell)\|
\le \eta_k\|F_{\mathcal S}(z^0)\|,
\qquad
\eta_k=\min(\eta_{\max},c\|F(x_k)\|^\alpha),
\]

并配合 Newton–Krylov、线搜索和 patch 更新迟滞。理论先证明根一致性、patch 外慢模态能量对收缩因子的控制，以及 forcing term 下的局部快速收敛；再做完整大规模性能实验。

### 2.5.6 备选线：裂纹拓扑感知近核粗空间

若实验 B 证明慢模态不是局部的，转向纯线性求解器方向：

\[
K(d)=\int_\Omega g(d)B^\top\mathbb C B\,\mathrm dx.
\]

利用 \(g(d)\) 加权图识别损伤分裂后的连通片段，为每个片段注入二维 3 个、三维 6 个近刚体模态，或补充局部广义特征向量。它区别于固定子域浮动模态：粗空间由损伤拓扑事件触发，并随裂纹连通性更新。

直接复用 docs/preconditioner/D12_RESULTS.md 的冻结矩阵，在 \(\max d\approx0.43,0.998,1.0\) 和不同残余刚度 \(\kappa\) 下比较全局刚体模态、片段刚体模态、局部特征粗空间和直接法。

### 2.5.7 求解器线四周执行顺序

| 周次 | 任务 | 产出 | 决策 |
|---|---|---|---|
| 第 1 周 | 历史场 snapshot/trial/commit/rollback 原型；3 个 checkpoint 对照 | \(H\) 过冲、峰值/能量带宽、真残量图 | 决定 A 独立或并入主线 |
| 第 2 周 | 固定点收缩指标与 Newton 仿射不变量分开；计算 \(T_{\mathrm{stag}}\) 前几个特征对 | 慢模态能量分布、谱半径 | 决定 B 或拓扑粗空间 |
| 第 3 周 | 最小联合 \(u\)-\(d\) patch solve；外层 Newton–Krylov | plain/Anderson/NEPIN 收敛与时间 | 验证过程区消元 |
| 第 4 周 | 三网格、两参数、两几何消融；整理主图表 | 机制图、效率表、物理一致性表 | 锁定论文题目与投稿期刊 |

### 2.5.8 论文结构与下一步路径

推荐论文结构：

1. 问题：历史场被求解器试算路径改变，且成核阶段 staggered 变慢；
2. 机制：累计历史场的棘轮上界与固定映射破坏；慢模态的过程区局部化；
3. 方法：事务型状态机 + 仿射不变量 patch + 局部联合 NEPIN；
4. 理论：不可逆性、回滚一致性、根一致性、局部谱改善和非精确 Newton 收敛；
5. 实验：历史语义、慢模态、算法消融、网格/材料/几何扩展；
6. 结论：先恢复正确的载荷步问题，再以局部非线性工作量换取全局快速收敛。

### 2.5.9 动态断裂：T7 的条件扩展

动态断裂值得加入，但不单独新开论文线。fractureX 当前没有惯性项和独立时间积分器；已有工作已经覆盖动态相场的单体/交错积分、显式方法、时空自适应、域分解预条件和自适应 BDF，因此“加入 Newmark 和动态 benchmark”本身不足以形成贡献（Borden et al. 2012, doi:10.1016/j.cma.2012.01.008；Svolos et al. 2020, doi:10.1016/j.jcp.2020.109746；Rannou and Bovet 2024, doi:10.1002/nme.7544；Cassese et al. 2026, doi:10.1016/j.finel.2026.104622）。

可形成闭环的切口是 **时间步相关的 regime-adaptive nonlinear preconditioning**。隐式 Newmark 离散后，位移块近似为

\[
A_{\Delta t}(d)=\frac{1}{\beta\Delta t^2}M+K_{uu}(d),
\qquad
T_{\Delta t}=K_{dd}^{-1}K_{du}A_{\Delta t}^{-1}K_{ud}.
\]

小时间步下惯性项占优，预期 \(\rho(T_{\Delta t})=O(\Delta t^2)\)，普通 staggered 已足够；时间步增大或裂纹成核后，退化刚度与局部 \(u\)-\(d\) 耦合占优，再启用过程区 NEPIN。所有 \(u,v,a,d,H\) 采用 trial/commit/rollback，拒绝时间步不写入不可逆状态。

最快的 go/no-go 不需要先实现完整动态断裂：组装一致质量矩阵，在现有峰前/成核/扩展 checkpoint 上扫描 \(\Delta t\)，计算 \(\rho(T_{\Delta t})\)、慢模态局部性和理想 patch 消元后的谱半径。只有当该指标能预测 staggered 迭代突增并给出稳定切换阈值时，才实现 Newmark 平均加速度格式、弹性波能量 smoke test 和一个 2D Kalthoff benchmark。

**当前唯一下一步：** 在六个峰值邻域状态建立启动准则；随后构造在线增量慢空间，动态分支继续冻结。

---

## 3. 论文清单（按投稿时间排序）

| 顺序 | 代号 | 标题草案 | 期刊首选 | 关键 selling point | 预计投稿 |
|---|---|---|---|---|---|
| 1 | **A** | Equilibrated a posteriori error estimation and σ-driven adaptivity for Hu-Zhang mixed phase-field fracture | CMAME / SINUM | 无常数超圆界；osc(f)=0 干净；超过郭雯 2024 的 CPU/mesh 节省 | **2026-07（v0.8 提档）** |
| 2 | **D12** | Auxiliary-space preconditioning for Hu-Zhang mixed phase-field fracture: uniqueness of convergence in the fully-localized regime | SISC / CMAME | 难 regime 唯一收敛；引龚博论 Ch 7 为理论根基 | 2026-09 |
| 3 | **B** | Balance-preserving neural operators for phase-field fracture via Hu-Zhang supervision | JCP / CMAME | 平衡监督 = 结构最优；R̃_h 平台 vs 下降曲线；诚实边界 T1/D1 | 2027-01 |
| 3′ | **F (T9)** | FractureX: a decoupled framework for phase-field fracture with three discretization paradigms | **CiCP** / CAMWA | 首次存档 FractureX；三离散同底座（Lagrange / HZ / C⁰-IP 4 阶）；并行组装 + 双块最优求解器 | 并行推进，≈2027 Q1（scaling benchmark 后） |
| 4 | **D+** | Non-nested coarse spaces and two-level Schwarz preconditioning for Hu-Zhang phase-field fracture in the localized regime | SIMAX / NLAA | 龚博论 Ch 6 多水平 + 胡齐芽 GEP-coarse；D12 自己 Outlook 点名的续作 | 2027-03 |
| 5 | **A+** | Extended Hu-Zhang element with vertex-tangent relaxation for adaptive elasticity at reentrant corners | Math. Comp. / M2AN | 合并 Hu-Ma 2020 + 龚博论 Ch 3 任意维奇异点代数定义 | 2027-08（v0.5 后移 1 档给 T6a 让路） |
| 6 | **T6a** | Fourth-order phase-field fracture with strain-gradient elasticity via interior-penalty FEM | Comput. Mech. / IJNME | 博论 IP-FEM 章直接延伸；mesh budget 放松 + 尺寸效应；应变梯度 delta 已入正文 | **2026-07（v1.0 提前，稿已齐，插队队首）** |
| 7 | **T7** | Coupled slow subspace localization and slow-mode-targeted local nonlinear elimination | CMAME / SINUM / SISC | 以 \(G\)、\(\mathcal V_r\)、\(Q_\omega G\) 建立 survival factor 与收缩因子闭环；局部性失败时转拓扑粗空间 | 2027-09 |
| 8 | **T6b** | Fourth-order phase-field fracture in Hu-Zhang mixed setting: inf-sup stability and equilibrated aposteriori | Comput. Mech. / IJNME / M2AN | σ_h ∈ H(div,S) + ∇²d 耦合 inf-sup 重证；接 T2 estimator；对 T6a 的 novel 升级 | **2027-12**（v0.5 新增） |
| 9 | **E** | Differentiable Hu-Zhang phase-field for topology-toughening design | ML4Science / PNAS | JAX 端到端 + SH-com 最小模型 | 2028 |

---

## 4. 开发任务清单（按优先级）

### 4.1 P0 · 立即（Phase 0）
- [ ] T0.1 完成 Lagrange 路线 (`MainSolve`) 在 model0 上跑通（C5 对照必备）
- [ ] T0.2 EXPERIMENT_MATRIX 自动扫描脚本（0.5 天）
- [ ] T0.3 处理 aux_h2/h3 停在 step 13：重启续算或缩减 D12 局部化 mesh-independence 结论

### 4.2 P1 · Phase 1（Q3 2026）· D12 + A tex 补 gap 送审 + B 线并行打底
- [ ] T1.tex.1 SENT shear 完整 aux-vs-direct 端到端 + crack-path 场对比
- [ ] T1.tex.2 SENT shear 局部化 iteration 研究一栏
- [ ] T1.tex.3 局部化区 mesh-independence 一栏
- [ ] T1.tex.4 §13 sweep 表空档从 pipeline (h2/h3/model1) 出货后补齐
- [x] T1.tex.5 Cervera VMS-OSGS 对照段落补进 §Related Work（2026-07-04）
- [ ] T2.tex.1 SENT shear 场景验证（tension + CNT red-green 已交付）
- [x] T2.tex.2 一般数据 $(f, t_N)$ 的显式 equilibrating correction（sketch 已入正文）
- [x] T2.tex.3 spectral tension–compression split 的 convex-duality majorant（sketch 已入正文）
- [x] T2.tex.4 marker efficiency 下界 / marker certification（sketch 已入正文）
- [x] T3.M3a `equilibrium_residual_l2` stub 实现（`fracturex/learn/eval/metrics.py:277`，8 项测试通过）
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
- [x] **T6a.pre.A1** [核心 delta] `ipfem_paper.tex` §2.2 应变梯度耦合项 + §3.2/§4.6 离散/误差分析扩展（Aifantis $O(\ell_s^2)$ 分量式，✅ 2026-07-10 落地）
- [ ] **T6a.pre.A2** `ipfem_paper.tex` 加"同 mesh 同 p 下 2 阶 vs 4 阶直接对比"章节 + 表（承接 Conclusion.③）——⚠️ 未做，tex §2299 明写留 future work；rebuttal 储备，不阻塞送审
- [x] **T6a.pre.A3** `ipfem_paper.tex` §5.4 penalty γ 敏感性系统研究章节 + 表（承接 Conclusion.②，✅ 2026-07-10 落地）
- [x] **T6a.pre.A4** `ipfem_paper.tex` Conclusion 已加 aposteriori-adaptive 展望段桥接 T2 equilibrated estimator（✅ 2026-07-10 落地；另加计划外 §5.5 manufactured-solution 收敛验证）
- [ ] **T6a.pre.B5** SENT tension（Miehe/Ambati 标准）新增算例，先在 fealpy_old 跑——⚠️ 未做（现有算例：rigid inclusion + notch + size-effect + manufactured）；rebuttal 储备，不阻塞送审
- [ ] **T6a.pre.C6** fealpy3 里补齐 C⁰-IP + 4 阶 PFM 实现（现在 fealpy_old）——T6b 前置依赖
- [ ] **T6b.pre.1** Hu-Zhang mixed 4 阶变分形式草稿：σ_h ∈ H(div,S) + ∇²d 耦合
- [ ] **T6b.pre.2** inf-sup 稳定性草稿（4 阶下重证，非 D12/A 推论）
- [ ] **T6b.pre.3** 与 T6a IP-FEM 结果对照的 baseline 表设计

### 4.4 P3 · Phase 3（Q2 2027+）
- [ ] T6a.1 IP-FEM 4 阶 PFM 代码接入 fracturex（预研落地）
- [ ] T6a.2 T6a 论文成稿（Comput. Mech./IJNME，**2027-06 投稿**，v0.5 safe win）
- [x] T7.0 完成 H1--H4：完整 \(G\)、慢子空间低维性与 solver-aware 局部性
- [x] T7.1 固定 patch 验证 \(Q_\omega G\) 和中心衰减律
- [x] T7.2 峰值固定区域 H6：约化解一致性与总成本对照
- [x] T7.3 coupled \(J_{\omega\omega}\) 与五档 matched-coupled 扫描
- [x] T7.4 六状态收益--成本稳定性与启动准则
- [x] T7.5 在线增量慢空间与长度尺度可解析路径慢率扫描
- [x] T7.5b Reduced–NE `J_{ud}` 方向校验；小算例误差 (2.2\times10^{-11})，解析网格误差 (7.4\times10^{-7})
- [x] T7.5c 外层 block-LU 预条件器与局部 predictor；Krylov 285→131，保留严格验收结果
- [x] T7.5d reference-free 自适应 warmup、残差下降门控与外层 Armijo 回溯：warmup 记录全场/约化/局部残差、在线率和残差比；慢率触发还要求最新残差比不超过 0.8
- [x] T7.5e 固定历史参考根上的全局化闭环原型：新增线性 continuation（4 阶段）与 Schur 方向残差、步长、回溯次数记录；四阶段均收敛，最终投影残差 $5.785\times10^{-10}$，但等价工作量和时间高于基线
- [ ] T7.5f Reduced--NE 可靠性到性能闭环：作为后续性能工作，优化 continuation 阶段自适应、局部消元精度与外层 Krylov 工作
- [x] T7.7 论文整理第一阶段：标题、摘要、引言主线统一为“诊断--定位--消元--验收”；加入科学思想图、solver workflow 图和“适用范围与当前实现边界”小节，cost-aware 降为诊断层，TeX 编译通过
- [x] T7.8 论文整理第二阶段：压缩实现细节与审计表，保留慢模态理论、核心收益表、网格证据和 reference-free 验收；当前 PDF 为 24 页（含附录和参考文献），后续仅做版式级微调
- [ ] T7.5a 在 $\bar u=0.0986,0.1030$ 做矩阵自由谱核对；选择一个可靠状态完成 Reduced-NE 对照（**2027-09 投稿**）
- [ ] T7.6 仅在主线闭环后评估历史场、动态和拓扑粗空间分支
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
| T7 | Ch 8 "牛顿法与仿射不变性" + Ch 10 "非线性消去预条件牛顿法" | 事务型状态、过程区 patch 谱作用与非精确 Newton；不再只做泛化的 staggered 理论 |

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

> **短期（半年）** 把 D12 收稿、A 主推、B 打底三线并行推完；**中期（一年）** 用博士论文第 3 章升级 A（→A+）、用第 6 章升级 D（→D+）；**长期（一年半+）** 用第 8–10 章推进 T7 coupled slow subspace 求解器论文，用 JAX AD 打远景（T8）。**所有线都以 σ_h ∈ H(div,S) 的两条独家性质（inf-sup 稳定 + 逐元平衡）为共同支点**——这正是龚博论第 7 章、conf 马睿线、fracturex 三者最大的方法学交汇。

---

## 8. 背景：龚世华 2018 博士论文 vs fracturex 五条论文线

**博士论文（SUMMARY.md）是 fracturex 方法学的祖谱**，其三部分几乎逐一对应 fracturex 的五条论文线：

| 博士论文章节 | 内容 | 对应 fracturex 线 | 状态 |
|---|---|---|---|
| 第 3 章 杂交混合有限元（放松顶点连续性 + 任意维奇异点代数定义） | Hu-Zhang 元扩展 | **conf §4 马睿线 2** ★★★（L-shape/切口角点） | **未开工**，是 A 线的天然升级项 |
| 第 4 章 内罚混合有限元（非协调面泡，任意阶） | 低阶 Hu-Zhang 替代 | fracturex 未采用（p≥3 已足） | 备胎，暂不投入 |
| 第 6 章 杂交系统的多水平求解器（非嵌套粗空间 + 区域分解磨光 + 顶点块局部估计） | 多水平预条件 | **conf §1 胡齐芽两层 Schwarz** ★★★ | 未开工，D 线**升级路径** |
| 第 7 章 H(div, S) 辅助空间预条件（离散弹性正合列 + 离散正则分解） | 块对角预条件的理论底座 | **D12 论文的理论根基** | ✅ 已在 fracturex 使用（`solve_huzhang_block_gmres_auxspace`），论文正在收尾 |
| 第 8–10 章 仿射不变 Lipschitz + NEPIN/ASPIN | 非线性预条件牛顿法 | T7 慢子空间 survival factor、\(Q_\omega G\) 与局部联合消元 | **先做 H1--H4**；通过后进入固定 patch 与总成本验证，no-go 时转拓扑近核粗空间 |
| 2015 H(div,S) 手稿（`thiese/`） | 第 7 章雏形 | = D12 前身 | — |

**结论**：D12（论文 §D）在**用**博士论文第 7 章；A 线自适应可**接**第 3 章；T7 将第 8–10 章的仿射不变框架用于事务型状态、过程区慢模态和非精确 NEPIN。这三条是龚老师博士工作在 fracturex 上的直接延续，可以在论文里理直气壮地引自己。

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

## 附 A：T6a tex 现状盘点（`Tian/thesis/ip_fracture/ipfem_paper.tex`，v0.6 新增 / **v1.0 已更新为投稿态**）

**tex 骨架**（**2345 行**，v0.6 时 1187 行；v1.0 大部 gap 已闭合）：

| 章节 | 现状 | 剩余 gap（v1.0） |
|---|---|---|
| §1 Introduction | ✅ 完整 | — |
| §2 Mathematical model（4 阶 PFM + hybrid split + §2.2 length-scale） | ✅ **A1 应变梯度耦合已入正文** | — |
| §3 Discretization and solution strategy（C⁰-IP + staggered + §3.2 length-scale 扩展） | ✅ **A1 对应 IP 处理已补** | — |
| §4 Error analysis（stability + coercivity + a priori $h^{p-1}$ + §4.6 length-scale 扩展） | ✅ 完整 | — |
| §5 Numerical experiments（rigid inclusion + notch + §5.3 size-effect + §5.4 penalty + §5.5 manufactured-solution） | ✅ **A3 penalty 敏感性已补**；manufactured-solution 加分项 | ⚠️ **A2** 缺 2 vs 4 阶直接对比（§2299 明写 future work）；**B5** 缺 SENT tension |
| §6 Conclusion | ✅ **A4 aposteriori 展望桥接 T2 已加** | — |

**Conclusion 自认五条 open extension 的处理**：

| 序号 | 内容 | 状态（v1.0） |
|---|---|---|
| ① | 参考解精细化（更细 mesh 或 benchmark） | ⏭️ 跳过（现有自参考够用） |
| ② | penalty γ 敏感性 | ✅ **A3 已补（§5.4）** |
| ③ | 同 mesh 同 p 下 2 阶 vs 4 阶 | ⚠️ **A2 未做**——tex §2299 留 future work，rebuttal 储备 |
| ④ | 3D + solver scalability | ⏭️ v2 或跳过（工程量大） |
| ⑤ | aposteriori-adaptive | ✅ **A4 已桥接（Conclusion 提 equilibrated estimator → T2）** |

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
