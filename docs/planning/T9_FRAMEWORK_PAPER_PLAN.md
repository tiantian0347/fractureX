# T9 · FractureX 框架/软件论文 计划（2026-07-11 v0.1）

> **代号**：T9 / 论文 **F**（Framework）。主计划 `MASTER_PAPER_DEV_PLAN.md` 之外新增的第 5 个近期投稿候选。
> **一句话**：把 FractureX 平台**首次正式存档发表**为一篇软件/框架论文，两根工程支柱是 **HZ 混合系统的并行组装** 与 **双块最优求解器（力学 aux-space + 相场 SA-AMG）**。
> **起因**：D12（`phasefield_huzhang.tex`）里新加了相场 AMG 预条件、且已有 HZ 矩阵并行组装。评估结论是这两块**不并入 D12**（会和 D12 的 matrix-free 主线自相矛盾、并稀释 aux-space 头条），而是抽出来单独成一篇框架论文。

---

## 1. 已敲定的决定（2026-07-11 讨论）

| 讨论点 | 决定 | 备注 |
|---|---|---|
| **定位** | ✅ **软件/框架论文** | 不走"性能优化方法"路线（那要补 MPI/GPU，工作量翻几倍） |
| **范围** | ✅ **宽版 = 整个 FractureX 平台完整框架** | **脊梁 = 一套解耦架构承载三种离散范式**（标准 Lagrange / HZ 混合 / C⁰-IP 4 阶）；窄版（只并行组装+相场 AMG 两柱）浪费"首次存档"身位，两柱降级为性能亮点 |
| **期刊** | 🟢 **Advances in Engineering Software（AES）** | ~~CiCP~~ **已出局**（用户 2026-07-11 决定不投）。AES = 工程软件平台向、IF 稍高，平台论文对口；并列第一梯队 CAMWA；备选 CPC/FEAD，见 §5 |
| **与 ICCES 会议稿关系** | ✅ **独立新写** | ICCES2025 **只投了摘要 + 作报告，无正式论文稿**（DOI `10.32604/icces.2025.011175`）；照例引一句作 "prior presentation" 透明声明即可，无版权/重叠问题 |

**"宽版 + 首次存档"的核心 novelty 抓手**：FractureX 此前没有任何存档描述，这是**第一篇完整介绍该框架**的论文，novelty 不必只靠两根柱子撑。

### 1.1 中心论点（2026-07-11 补，三离散脊梁）

> FractureX 在**同一套 mesh / case / 后端（NumPy↔PyTorch）/ 求解器基础设施**上，统一承载相场断裂的**三种离散范式**：

| 离散范式 | 主变量 | 实现路径（已坐实成熟） | 方法出处（框架论文只引不 claim） |
|---|---|---|---|
| 标准 Lagrange | 位移 | `phasefield/main_solve.py`（`MainSolve`, 1218 行） | 田博论 model2 baseline |
| Hu-Zhang 混合 $H(\mathrm{div},\mathbb S)$ | 应力 | `drivers/huzhang_phasefield_staggered.py` | **D12** |
| C⁰-内罚（4 阶） | — | `interior_penalty/ipfem_phasefield_solver.py` | **T6a/ipfem** |

三者**共用** `fealpy.backend`（NumPy↔PyTorch 可切）+ 共用 `cases/`（model0/model2/square_tension…）网格与算例基座。**"三种离散跑在同一底座上"把"解耦架构"这个软件贡献从'嘴上说'变成'摆出来看'**——并行组装 + 双块求解器是其中的性能亮点，不是全部卖点。

#### 1.1.1 能力矩阵（2026-07-11 二次补，脊梁再加宽）

三离散只是第一根轴。用户核实平台**同时具备高次 + 多种自适应 + 2D/3D**，脊梁升级为**四维能力矩阵**，一张表把"同一底座撑起多少组合"摆出来：

| 轴 | 取值 | 代码坐实 |
|---|---|---|
| **离散范式** | 标准 Lagrange / HZ 混合 / C⁰-IP 4 阶 | `main_solve.py` / `huzhang_phasefield_staggered.py` / `ipfem_phasefield_solver.py` |
| **多项式次数 p** | 高次可调 | 标准 `p`（default 1）；IP `p_disp/p_phase`；HZ 元次可选 |
| **自适应** | RG（red-green）/ NVB（newest-vertex-bisection）+ 粗化 | `mesh/halfedge_mesh.py`（`refine_triangle_rg` 604 / `refine_triangle_nvb` 801 / coarsen） |
| **维数** | 2D（三离散全）/ 3D（标准） | `cases/phase_field/model3d.py`（标准 3D） |

**演示矩阵（挑代表格，不追求笛卡尔全填）**：标准×{model0/1/2, L-shape, 3D}、HZ×{model0/2, L-shape}、C⁰-IP×{L-shape, notch 4 阶}；高次 p 至少标准+IP 各出一档；自适应 RG/NVB 各一条 SENT/L-shape 收敛曲线。

> ⚠️ **自适应边界（硬红线）**：`adaptivity/equilibrated_estimator.py` = **A 线（T2）**的均衡估计子机器。框架论文里自适应**只能"演示能跑 + 网格图 + DOF-vs-error 曲线"**，**方法一律引 A/T2 + 田博论 recovery-based AFEM，绝不 claim 估计子理论/可靠性有效性证明**（那是 A 的头条）。RG/NVB 的 refine/coarsen 本身是平台设施，可正常讲实现。

**边界规则（3 篇方法论文）**：框架论文引三篇方法论文讲方法、自己只讲实现+性能+统一。标准→田博论；HZ→D12；IP→T6a；**自适应估计子→A/T2**。四条方法线管"方法+理论+数值"，T9 管"一个平台统一实现+跑得快"，零重叠。

---

## 2. 家底盘点（已在手，2026-07-11 核）

| 模块 | 状态 | 位置 |
|---|---|---|
| HZ 弹性块并行组装器 | ✅ 成熟，**1119 行** | `fracturex/assemblers/huzhang_elastic_assembler.py` |
| 相场并行组装器 | ✅ 成熟，**1091 行** | `fracturex/assemblers/phasefield_assembler.py` |
| HZ 特色组装优化 | ✅ 角点松弛变换 `M2=TMᵀ M TM` 列块并行 matmul（`_parallel_TMt_M_TM`）+ d-无关几何内核缓存 + standard/effective-stress 两本构分支各自缓存 | 同上 |
| 并行控制 | ✅ 环境变量 `FRACTUREX_ASSEMBLY_PARALLEL` / `_NPROC` | 同上 |
| 相场 AMG / ILU 对照 | ✅ `ilu_vs_amg.py` 脚本 + pyamg 集成 | `docs/preconditioner/scripts/`, `utilfuc/huzhang_fast_solver.py` |
| 力学 aux-space fast solver | ✅ Gauss-Seidel/Chebyshev 磨光 + P1 粗空间（pyamg） | `utilfuc/huzhang_fast_solver.py` |
| 架构文档（≈论文骨架） | ✅ 性能调优已成文 §5.5/§3.5/§7.1 | `docs/architecture/huzhang_phasefield_architecture.en.md` |
| 组装计时钩子 | ✅ `_install_assembly_timer` | `tests/phasefield_model0_huzhang.py` 等 |
| 软著 | ✅ 已登记（断裂力学仿真计算软件 v1.0） | `Tian/软件著作申请/fractureX软件著作/` |
| ICCES 报告 | ✅ 摘要 + 口头报告（无论文稿） | publication `2025-fracturex-icces.md` |

**结论**：代码 + 文档骨架基本都在，起点很好。

---

## 3. 决定成败的缺口：系统 scaling 研究（必须新跑）

计时钩子有，但**没有成体系的 benchmark**：组装墙钟 vs 线程数 / vs 网格规模 / 缓存命中加速比 —— 软件/性能论文的命脉曲线**现在是空的**。

- ✅ **好消息**：组装 benchmark 是**本地几分钟~几十分钟就能跑完**的短实验，不卡服务器，不阻塞 D12/A。这篇的实验反而是四篇里最快出的。
- 待补实验清单（草案，2026-07-11 定稿）：
  1. **strong scaling**：固定网格，组装墙钟 vs `FRACTUREX_ASSEMBLY_NPROC`（1→N 核）
  2. **weak scaling / 规模曲线**：组装墙钟 vs 网格 DOF（多档 h）
  3. **p-version 轴（新增，"高次"引入）**：组装墙钟 vs 多项式次数 p（标准 `p`、IP `p_disp/p_phase` 各扫一档），量高次单元的组装成本增长
  4. **缓存命中加速比**：d-无关几何内核缓存 命中 vs 冷启，跨 staggered 步复用收益
  5. **standard vs effective-stress** 两本构分支组装成本对照
  6. **求解器侧**：相场 SA-AMG 迭代数（D12 已有 12/15/18，可复用/引用）+ ILU-vs-AMG（`ilu_vs_amg.py` 已有）

- **算例矩阵（用户 2026-07-11 授权定死）**：
  - 标准 Lagrange：`model0` / `model1` / `model2` + **L-shape**（`phase_field/Lshape_cyclic.py`）+ **3D**（`phase_field/model3d.py`）
  - C⁰-IP：`model0/1/2` + **L-shape**（IP 侧）
  - HZ 混合：`model0` / `model2`（+ 可选 `damage_model/fracture_huzhang_Lshape_example.py`）
  - 3D 仅标准有限元跑；L-shape 仅标准 + IP 跑（HZ L-shape 视时间可选）

---

## 4. 诚实定位（避坑）

现并行是 **ThreadPoolExecutor（共享内存线程）+ BLAS 卸载 + 缓存**，**不是 MPI/分布式，也不是 GPU/numba**。

- ❌ 别包装成"新颖并行算法"投 Parallel Computing / IJHPCA —— reviewer 一眼看穿 ThreadPool+BLAS，会挨批。
- ✅ 正确调子：**软件/框架论文**，卖点是**工程整体性**（解耦架构 + 组装策略 + 双块最优求解器），并行组装与相场 AMG 是其中两根支柱，不是全部。

---

## 5. 期刊候选（2026-07-11 定：CiCP 出局 → 主投 AES）

按"框架 + 三离散 + 性能 + 算例 + 无 HPC"画像的契合度排序：

| 排名 | 期刊 | 体例 | 节奏 | IF/分量 | 适配 |
|---|---|---|---|---|---|
| **①** | **Advances in Engineering Software (AES)** | 工程软件平台向，全长 | 中 | 中高 | ✅ **主投**——明确"工程软件开发"向，平台论文对口，无 program-summary 负担 |
| ② | **Computers & Math. with Applications (CAMWA)** | 方法+软件+算例，全长 | 中 | 中高 | 并列第一梯队——计算数学组框架论文常见去处，改动最小 |
| ③ | **Computer Physics Communications (CPC)** | 长，要 program summary + 代码入库 | 慢 | 重、IF 高 | "首次存档"最契合，但体例最重 |
| ④ | **Finite Elements in Analysis & Design (FEAD)** | FE 方法+软件+算例 | 中 | 中 | 三离散 FE 底座对味 |
| ⑤ | ~~CiCP~~ | ~~方法+软件+物理~~ | — | — | ❌ **出局**（用户决定不投）；FEALPy v3 同刊背景不再作卖点 |
| — | SoftwareX / JOSS | 短、强制开源 | 快 | 轻 | 短体例浪费"首次存档"身位，不作主投 |

**决策**：✅ **主投 AES（Advances in Engineering Software）**——CiCP 已弃；AES 平台论文对口、无 Program Summary 负担。CAMWA 并列备胎，CPC 留给想要最强存档性时。


---

## 6. 与 D12 的边界（关键，别自相残杀）

- **力学 aux-space 求解器是 D12 头条** → 软件论文**只能"引 D12 的方法，这里只讲实现"**，绝不重复 claim 为自己的方法贡献。
- **相场 AMG** 两篇都可出现，但角度切开：
  - D12 = "迭代数结果"（数值，`tab:phase_precond` 12/15/18）
  - 软件论文 = "实现/集成"（工程），**不原样复制 D12 的表**
- 清爽切分：**D12 = 方法 + 理论 + 数值**；**T9/F = 框架 + 实现 + 并行组装 scaling**。

---

## 7. 宽版章节大纲（草案，三离散脊梁，待细化）

1. **Introduction** —— PFM 断裂仿真软件现状；HZ 混合方法为何需要专门平台；FractureX 定位与贡献（首次存档 + 三离散统一）
2. **Software architecture** —— 网格/离散/后端解耦；NumPy/PyTorch 后端切换（引田博论 `design.tex`）；模块化（cases / assemblers / drivers / interior_penalty / …）；**共享基座如何让三种离散即插即换**
3. **Three discretization paradigms on a shared base**（脊梁）—— 同一 case/mesh/backend 上：
   - 3.1 标准 Lagrange 位移 PFM（`MainSolve`，引田博论 model2）
   - 3.2 Hu-Zhang 混合 $H(\mathrm{div},\mathbb S)$（引 D12）
   - 3.3 C⁰-内罚 4 阶 PFM（引 T6a/ipfem）
4. **Assembly strategy**（柱一）—— formulation-split 缓存组装 + 角点变换 `TMᵀMTM` 块并行 + d-无关几何内核缓存
5. **Solver stack**（柱二）—— 力学 aux-space 块预条件（引 D12）+ 相场 SA-AMG；matrix-free 应力块
6. **Performance study** —— §3 的 scaling 曲线（strong/weak/缓存命中/本构分支）
7. **Application examples** —— 三离散各自的端到端算例（SENT tension/shear、circular-notch、notch 4 阶等），展示同一平台的通用性
8. **Conclusion** —— 开源、可复现、后续（jax backend 等接 T8/E 线）

---

## 8. 可行性结论

**可行，且是"多投一篇"里性价比最高的候选**：代码熟、文档骨架有、软著背书、实验本地快出、无会议稿版权顾虑。**唯一实打实工作量 = 补一轮 scaling benchmark（几天）+ 按软件论文体例重组架构文档。**

**排期建议**：不与 D12/A 抢本地写作带宽的前提下并行推进；scaling benchmark 可在等 D12/B 服务器算例的窗口期插空跑完。目标投稿窗口视期刊而定（SoftwareX 可争取与 B 同期或稍后）。

---

## 9. 待办 / 下一步

- [x] **期刊拍板**（§5）—— ✅ 首选 **CiCP**（与 FEALPy v3 同刊），待跟老师核实一句定稿
- [x] **scaling benchmark 脚本**（柱一 = HZ 组装器）—— ✅ 写成并本地冒烟通过：`fracturex/tests/hz_assembly_scaling_benchmark.py`，四轴 strong/size/pver/formulation 全跑通，孤立 `assemble()` 计时（不跑 staggered），出 JSON + 表。**下一步 = 换细网格档正式跑数**（小网格并行增益吃在线程开销里，见下）
  - ⚠️ 两个 API 硬约束（踩过坑）：**① HZ 元 p≥3**（u 空间次数 = p−1；p=2 触 fealpy `div_basis` 角点 dof bug）；**② 计时前必须 `damage.on_build(discr, view, case)`**，否则 `_gfun=None`（driver 的 `initialize()` 干这活，孤立组装要自己补）。
  - ⚠️ 诚实观察：strong-scaling 在 784-cell 小网格只有 ~1.1×——`TM'MTM` 列块并行 matmul 的收益要**细网格大 DOF** 才显现，正式图必须跑细档 `hmin`。
- [x] **三离散 p-version 横扫脚本**（标准 Lagrange / C⁰-IP）—— ✅ 写成并冒烟通过：`fracturex/tests/discretization_pversion_benchmark.py`。共享单位方结构网格，孤立组装计时三核：**standard-elastic**（`MainSolve` 位移刚度 vs p_disp）/ **ip-elastic**（IP 同款位移刚度 vs p_disp，对照）/ **ip-phase4th**（`IPFEMPhaseFieldSolver._assemble_phase_lhs`：biharmonic + 内罚 4 阶核 vs p_phase）。9 行 p=1-3 / 2-4 全跑通，4 阶核增长最快（warm 0.012→0.073s，p2→4）。
  - ⚠️ 踩坑：**① 标准要先 `ms._method='lfem'` 再 `initialize_settings(p)`**（`_method` 平时在 `solve()` 里才设）；**② IP `_assemble_phase_lhs` 前要 `solver.H=0.0`**（历史场，u=0 时本为 0，newton 里由 pfcm 先算）。位移刚度组装标准/IP 同款（`LinearElasticIntegrator voigt`），复刻求解器里那 3 行即忠实。
- [ ] 架构文档 `huzhang_phasefield_architecture.en.md` → 论文体例重组
- [ ] D12 Intro 是否补一句相场块说明（属 D12 收稿，见 MASTER §建议1，另议）
- [x] 把 T9/F 登记进 `MASTER_PAPER_DEV_PLAN.md` §1 总表 + §3 论文清单 + MEMORY 索引（✅ 2026-07-11）

---

## 附：文件锚点

- 本文件：`docs/planning/T9_FRAMEWORK_PAPER_PLAN.md`
- 主计划：`docs/planning/MASTER_PAPER_DEV_PLAN.md`
- D12 tex：`Tian/thesis/fracture_huzhang/phasefield_huzhang.tex`（相场 AMG 在 §2705，并行组装引用在 §1583 matrix-free）
- HZ 组装器：`fracturex/fracturex/assemblers/huzhang_elastic_assembler.py`
- 相场组装器：`fracturex/fracturex/assemblers/phasefield_assembler.py`
- aux-space fast solver：`fracturex/fracturex/utilfuc/huzhang_fast_solver.py`
- 架构文档：`fracturex/docs/architecture/huzhang_phasefield_architecture.en.md`
- **scaling benchmark 脚本（柱一，已成）**：`fracturex/fracturex/tests/hz_assembly_scaling_benchmark.py`（结果落 `results/benchmarks/hz_assembly/`）
- **三离散 p-version 脚本（已成）**：`fracturex/fracturex/tests/discretization_pversion_benchmark.py`（结果落 `results/benchmarks/pversion/`）
- 计时 harness 模板（脚本据此写）：`fracturex/fracturex/tests/phasefield_model0_huzhang.py`（`_install_assembly_timer`）
- ILU-vs-AMG：`fracturex/docs/preconditioner/scripts/ilu_vs_amg.py`
- ICCES 报告：`tiantian0347.github.io/content/publication/2025-fracturex-icces.md`
- **FEALPy v3 平台论文（底座引擎，同刊参照）**：Zheng, Wei* et al., *FEALPy v3: A Cross-platform Intelligent Numerical Simulation Engine*, Communications in Computational Physics, DOI `10.4208/cicp.OA-2025-0327`
