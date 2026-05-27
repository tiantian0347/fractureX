# FractureX 总体介绍：架构、技术路线与 Hu-Zhang 相场实现

本文在原有 Hu-Zhang 文档基础上，扩展为对整个 `fracturex` 仓库的统一介绍，面向对外技术说明与新成员上手。

FractureX 以 **FEALPy** 为核心数值后端，在其网格、函数空间、积分与线性代数能力之上，组织了面向断裂问题的领域层模块：**算例（case）**、**离散（discretization）**、**装配（assembler）**、**驱动（driver）**、**损伤模型（damage）** 与 **后处理（postprocess）**。

---

## 1. 项目定位与能力边界

- **目标问题**：材料损伤与断裂数值模拟，重点覆盖脆性断裂中的相场方法与混合有限元方法。
- **核心后端**：FEALPy（网格、空间、积分、装配容器、部分求解器），FractureX 提供领域层抽象与可复用工程化流程。
- **当前主战场**：二维三角网格场景，支持多个求解栈并行演化（标准位移相场、Hu-Zhang 混合元相场、Hu-Zhang 局部损伤）。
- **面向对象**：科研复现、算法对比、教学演示与新方法试验。

---

## 2. 仓库结构与分层职责（全局）

```
fracturex/cases            # 物理场景定义：网格、材料、边界、载荷
fracturex/discretization   # 空间与状态：mesh + FE spaces + state
fracturex/damage           # 损伤模型：相场或局部损伤演化
fracturex/assemblers       # 组装代数系统 A x = F
fracturex/drivers          # 编排加载步与交错/单向求解流程
fracturex/phasefield       # 标准位移-相场主类与配套物理函数
fracturex/postprocess      # 反力、运行记录、报告导出
fracturex/utilfuc          # 通用工具（线性求解器、网格工具等）
fracturex/tests            # 可执行算例脚本（也是最实用入口）
```

设计原则保持一致：

- **Case 只描述物理，不求解**。
- **Discretization 只管理离散对象与状态**。
- **Assembler 只负责系统构造**。
- **Driver 只负责流程编排与收敛控制**。
- **Postprocess 只负责结果组织与导出**。

---

## 3. 三条主要技术路线

### 3.1 标准位移型相场路线（`MainSolve`）

- 核心入口：`fracturex/phasefield/main_solve.py`
- 特点：位移与相场均采用 Lagrange 空间，流程偏经典相场实现。
- 典型参考：`fracturex/cases/phase_field/model0_example.py`

### 3.2 Hu-Zhang 混合元 + 相场路线（当前主文档重点）

该路线将应力与位移作为独立未知，和相场损伤进行交错耦合，模块关系是：

```
CaseBase
  -> HuZhangDiscretization (space_sigma, space_u, space_d, state)
  -> HuZhangElasticAssembler + PhaseFieldAssembler
  -> HuZhangPhaseFieldStaggeredDriver
  -> RunRecorder / reaction_from_sigma
```

关键特征：

- 支持 `standard` 与 `effective_stress` 两种弹性装配表述。
- 在加载步级别提供 `begin_load_step(load)` 缓存，减少交错迭代中的重复构造。
- 相场历史变量 `H` 以积分点布局存储于 `state.H`，与节点损伤 `d` 分离。
- 求解器可注入，便于直接法与 Krylov/预条件方法性能对比。
- 可选 **单元并行装配**、**standard 下 TMᵀMTM 分块乘**、**effective_stress 下 B2 双分支重叠**、**历史分块并行**、**历史 profiling**，以及 **`save_vtkfile_highorder` 默认并行采样**（纯可视化，不计入论文核心求解耗时表）；细节见 **§7.1**（英文 §3.5 / §5.5）。

### 3.3 Hu-Zhang 混合元 + 局部损伤路线

- 入口路径：`fracturex/drivers/huzhang_damage_staggered.py`
- 配套模型：`fracturex/damage/local_node_damage.py`
- 适用：需要与相场路线做方法学对照，或开展局部损伤演化实验时。

---

## 4. 核心抽象对象（跨路线通用）

- **Case (`fracturex/cases/base.py`)**  
  定义网格、材料、边界和加载，不直接参与线性求解。
- **Discretization (`fracturex/discretization/...`)**  
  负责构建 FE 空间与状态容器，管理解向量在函数空间中的表达。
- **Damage (`fracturex/damage/...`)**  
  负责损伤能量驱动、历史变量更新和不可逆约束相关逻辑。
- **Assembler (`fracturex/assemblers/...`)**  
  输入 `case + state`，输出代数系统 `A, F`。
- **Driver (`fracturex/drivers/...`)**  
  组织加载步、交错迭代、收敛判据、异常处理、记录回调。
- **Postprocess (`fracturex/postprocess/...`)**  
  标准化导出元数据、历史曲线、checkpoint 与结果摘要。

---

## 5. 通用求解数据流（以交错迭代为例）

1. 由 `case` 生成网格与边界定义，`discretization` 建立空间和初值状态。  
2. 对每个加载步，driver 触发装配并调用线性求解器。  
3. 若是交错方法，执行“力学步 -> 损伤步 -> 不可逆投影/截断 -> 收敛检查”。  
4. 在步骤末记录反力、能量、迭代次数、耗时等信息。  
5. 输出到 `RunRecorder` 指定目录（`meta.json`、`history.csv`、`summary.json`、可选 `npz` checkpoint）。  

---

## 6. 关键模块速览（面向维护和二次开发）

- `fracturex/cases/base.py`：统一算例协议，定义边界和载荷分段接口。  
- `fracturex/discretization/huzhang_discretization.py`：Hu-Zhang 路线的空间和状态管理。  
- `fracturex/assemblers/huzhang_elastic_assembler.py`：混合元弹性块系统组装。  
- `fracturex/assemblers/phasefield_assembler.py`：相场系统组装与历史驱动耦合。  
- `fracturex/damage/phasefield_damage.py`：AT1/AT2、退化函数、正应变能分裂、历史演化。  
- `fracturex/utilfuc/linear_solvers.py`：直接法、Krylov、aux-space 预条件等实验路径。  

---

## 7. 可配置能力（对外常见问题）

- **离散参数**：`p`、`damage_p`、`use_relaxation`、`u_space_order`。  
- **相场物理**：`AT1/AT2`、退化律、`split`、`eps_g`、`history_source`。  
- **装配策略**：`formulation="standard"` 或 `"effective_stress"`（`g(d)` 仅在 **M** 或 **B** 之一，互斥）。  
- **求解器注入**：`elastic_solver` 与 `phase_solver` 可替换为用户自定义实现。  
- **迭代预条件与 formulation 对齐**：`solve_huzhang_block_gmres_auxspace` / `solve_huzhang_block_gmres_fast` 须传入与 assembler 相同的 `elastic_formulation`；`standard` 可在粗 P1 扩散上用 `coef_bary`→**g**（Chen 等 2017 §5），`effective_stress` 粗空间不加 g、损伤仅经 Schur（**B**）进入。  
- **标准模型 Schur**：固定 \(d_h\) 时 \(\mathcal K_h=[[A(d_h),B^\top],[B,0]]\)，\(S(d_h)=B A(d_h)^{-1}B^\top\)；代码用 \(\widehat S=B\,\mathrm{diag}(A)^{-1}B^\top\) 近似，并分别构造 \(B_A\)、\(B_S\)。详见 `docs/huzhang_interface_test_manual.md` §3.0。  
- **运行记录**：`RunRecorder` 可控制记录粒度、checkpoint 频率与输出目录。  

### 7.1 环境变量：装配并行与历史更新 profiling

- **`FRACTUREX_ASSEMBLY_PARALLEL`**（`1` / `true` / `yes` / `on`）：开启后，`HuZhangElasticAssembler` 与 `PhaseFieldAssembler` 中已实现的 **单元分块并行装配**（含 standard 下弹性 `M(d)`、相场 `A_const` / `A_hist` / RHS，以及 `effective_stress` 下 B2 相关块）。也可在构造 assembler 时传 `assembly_parallel=True/False` 覆盖环境默认。  
- **`FRACTUREX_ASSEMBLY_NPROC`**：并行线程数上限（正整数）；不传则倾向使用可用 CPU 数。构造时 `assembly_nproc` 可覆盖。  
- **`FRACTUREX_PROFILE_HISTORY_UPDATE`**（`1` / `true` / `yes` / `on`）：在 `PhaseFieldAssembler.assemble` 内对 **`damage.update_history_on_quadrature`** 打墙钟计时，每次调用打印一行（含 `wall_s`、`NC`、`NQ`、`split`），便于对比不同网格与线程环境。  
- **`FRACTUREX_HISTORY_UPDATE_PARALLEL`**：设为 `0` / `false` / `off` 时，即使全局 **`assembly_parallel`** 开启，也强制历史更新走 **串行** 路径（便于对照）；默认随装配并行开启 **按单元分块线程更新**。  
- **`FRACTUREX_VTK_HIGHORDER_PARALLEL`**：控制 **`save_vtkfile_highorder`** 内按单元采样的分块并行；**默认为开启**（设为 `0` 等可关闭）。该耗时属 **纯可视化**，论文或核心求解效率对比中 **不应** 与交错迭代/装配时间混在同一表中。  
- **standard 弹性 `M2 = TMᵀ M TM`**：当 **`assembly_parallel`** 开启时，对 sufficiently large DOF 规模尝试 **按列分块的稀疏三重乘**（失败则自动回退串行）。  
- **effective_stress `B2`**：在 **`assembly_parallel`** 下，**g·div 块（含装配与 `TMᵀ·`）** 与 **∇g 修正块** 在两个线程上 **重叠执行**。  
- **历史更新与 BLAS**：并行 batch 与 BLAS/OpenMP 线程仍可能相互作用；若数值或伸缩异常，可用 `FRACTUREX_HISTORY_UPDATE_PARALLEL=0` 或调节线程环境做对照。

---

## 8. 运行入口与建议上手顺序

推荐入口脚本（由易到难）：

- `fracturex/tests/phasefield_square_tension.py`：较小规模的 Hu-Zhang + 相场示例。  
- `fracturex/tests/phasefield_model0_huzhang.py`：Model0 圆缺口完整流程，含记录与对比配置。  
- `fracturex/cases/phase_field/model0_example.py`：标准位移相场参考路径。  

如果目标是复现实验结果，建议优先阅读脚本中的：

- 载荷构造与分段边界设定；
- 求解器切换（直接法/迭代法）；
- 结果目录组织与 summary 导出逻辑。

---

## 9. 与 FEALPy 的关系（快速对照）

FractureX 不替代 FEALPy，而是建立在 FEALPy 之上的领域编排层。可简化理解为：

- FEALPy 提供网格、函数空间、积分与基础装配/求解能力；
- FractureX 定义断裂问题中的对象边界、流程组织和可复现实验框架；
- 因此同一 FEALPy 后端上，可并存多条 FractureX 技术路线并做可控对比。

---

## 10. 文档维护约定

- 文档索引：`docs/readme.md`
- 本文路径：`docs/huzhang_phasefield_architecture.md`
- 英文版：`docs/huzhang_phasefield_architecture.en.md`
- 关键路径校验：在仓库根运行 `python scripts/verify_huzhang_docs.py`
- 若源码路径变更，请同步更新中英文文档与校验脚本内的清单。
- **装配并行 / 历史 profiling 等环境变量**：修改实现或变量名时，请同步更新本文 **§7.1** 与英文 **§3.5（相场步骤说明）**、**§5.5（环境变量表）**。

---

---

### 附：Hu-Zhang + 相场路线的最小流程备忘

1. `case` 定义网格/边界/载荷；  
2. `discr.build()` 创建 `space_sigma/space_u/space_d` 与 `state`；  
3. `driver.initialize()` 调用损伤模型初始化；  
4. 每个加载步执行弹性与相场交错迭代并更新 `state`；  
5. `recorder` 输出历史曲线、元数据与 checkpoint。  

*具体类名与路径以仓库为准；本文需随开发人工维护。*
