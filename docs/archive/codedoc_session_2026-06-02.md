# 核心引擎 docstring 补全 + 两处 latent bug 修复（2026-06-02）

> 范围：fracturex 核心引擎源码（utilfuc/、assemblers/、phasefield/、damage/、drivers/、
> discretization/、boundarycondition/、adaptivity/）的注释补全，跳过 tests/ 与 cases/。
> 状态：已提交 `d7fbf26`（21 文件，+661/−30）；附带 3 处 bug 修复均经实跑验证；
> 不动求解器主链逻辑，与现有 P1/P2 论文实验完全兼容。

## 0. 起点 / 范围

需求：写 fracturex 代码时要加注释 + 函数输入/输出说明（已记入长期记忆
`fracturex_code_comment_style`）。本会话对**已有源码**做一次性补全。

风格基准：对齐 `utilfuc/recover_strain.py` 既有风格——Google 风格 docstring
（Args/Returns，含参数 dtype/shape 与返回值），关键步骤行内注释，中英按文件现状；
**已有的好 docstring 不重写**，只补缺口。

扫描口径：用 AST 统计每个文件"缺 docstring 的模块/顶层函数/类"。内部闭包
（callback、局部系数函数等实现细节）按 recover_strain 既有风格不强加 docstring。

## 1. 补全成果

核心引擎 30 个源文件，模块 + 每个顶层函数/类的 docstring 全覆盖。分 4 批：

| 批 | 文件 |
|---|---|
| utilfuc | linear_solvers, linear_elastic_pde, huzhang_fast_solver, recover_strain, phasefield_mesh, mesh_patch, sparse_direct_backends, linear_elastic_huzhang, utils, vtk_lagrange_writer |
| assemblers | huzhang_elastic_assembler, phasefield_assembler |
| phasefield | phase_fracture_material, main_solve, crack_surface_density_function, energy_degradation_function, vector_Dirichlet_bc |
| damage | phasefield_damage, base, local_node_damage |
| drivers | huzhang_phasefield_staggered, huzhang_fe_solve, huzhang_damage_solve, huzhang_damage_staggered |
| 其它 | huzhang_discretization, huzhang_boundary_condition, adaptive_refinement |

补全中顺手标注的两个文件性质（写进了模块 docstring）：
- `huzhang_fast_solver.py`：实验/原型脚本，依赖包外模块（orthogonal_*、smoother），
  无法独立 import，含死代码；生产版快速求解器在 `linear_solvers.py`。
- `linear_elastic_huzhang.py`：手动收敛验证脚本，依赖包外模块。

## 2. 附带修复的 3 处 latent bug

补注释时发现并修复（逐一经实跑验证）：

| # | 位置 | 问题 | 修复 | 验证 |
|---|---|---|---|---|
| 1 | `LinearElasticPDE.boundart_stress` | `t_y` 分量 n_x/n_y 写反（`σyy·n_x+σxy·n_y`） | 改为 `t=σ·n` 即 `σxy·n_x+σyy·n_y` | 单元测试 vs σ·n **误差 0**；旧式偏差 2.22 |
| 2 | `HuzhangStressBoundaryCondition.apply_essential_bc_to_system` | 用了 `spdiags` 但文件未 import | 补 `from scipy.sparse import spdiags` | 见 #3 同步验证 |
| 3 | 同上方法 line 451 | `bm.bool_`（fealpy numpy 后端无此属性） | 改为 `bm.bool` | 方法全程跑通：10 个 σ 边界 dof 正确约束（对角=1、RHS=规定值、非对角清零、解有限） |

**数据质量诚实标注**：
- bug #1 那个函数 `boundart_stress` 在**全库无任何调用方**，所以从未污染过结果，
  修复纯属未雨绸缪。它是功能性改动，若日后被用到 Neumann/应力边界算例需复核。
- bug #2/#3 所在的 `apply_essential_bc_to_system` 只被早期 driver
  (`huzhang_fe_solve.py`、`huzhang_damage_solve.py`) 调用；**现行 staggered+phasefield
  主链**走的是装配器自带的 `apply_sigma_essential_to_system`（spdiags 在方法内局部
  import，一直正常），不受这两个 bug 影响。

## 3. auxspace 测试"假阴性"澄清（重要）

验证主链时跑 `test_auxspace_precond_degraded_elastic`（aux-space GMRES vs 直接解），
intact 算例报 `rel_diff≈1.0`、pass=False。**逐步定位后确认是假象，非 bug**：

1. 用干净 HEAD worktree 对照：同样的 `rel_diff≈1.0`、0/2 fail → **与本次任何改动无关**。
2. 矛盾点：`rel_res_aux≈8e-9`（x_aux 满足 A x≈F）但 `rel_diff≈1.0` → 一度怀疑 A 奇异
   / trivial 解（呼应 [[model2_paper_direct_bogus]] 那类病）。
3. 把网格从我为快跑设的 `hmin=0.06` 细化到 `0.04` 重跑同一算例：

   | 量 | 值 |
   |---|---|
   | ‖F‖ | 5.97e-04 |
   | ‖x_ref‖ (spsolve) | 131.6，rel_res_direct=5.4e-12 |
   | ‖x_aux‖ (gmres) | 131.6，rel_res_aux=8.0e-9 |
   | rel_diff | **1.42e-11 → PASS** |

   两解范数都 131.6（非零、非 trivial），A 不奇异（spsolve 残差 5e-12）。

**结论**：`rel_diff≈1.0` 纯粹是我把 `FRACTUREX_HMIN` 压到 0.06（默认 0.02）的**粗网格
假象**——p=3 Hu-Zhang + 圆缺口 + 1/eps_g 放大在粗网格上极病态，GMRES 残差到 1e-9 也
不足以唯一化解。已记入长期记忆 `auxspace_test_mesh_sensitivity`：
**这类 aux-vs-direct 对比别在 `hmin≥0.06` 判 pass/fail**。

## 4. 提交与 WIP 隔离

会话开始时工作区已有大量**未提交的 WIP**（不是本会话产生）：`learn/` 整片、
`postprocess/dataset_export.py`（−768 行）、`cases/`、若干 `tests/`，以及
`linear_solvers.py` / `huzhang_elastic_assembler.py` / `phasefield_assembler.py` 等处的
aux/装配缓存重构。

为避免把 WIP 卷进 commit，用 **AST 剥离 docstring 后比对代码**的办法精确分离：

- **提交 `d7fbf26`（21 文件）**：20 个纯 docstring 文件 + linear_elastic_pde（traction
  修复）+ huzhang_boundary_condition（spdiags/bool 修复）。暂存区代码级改动经校验
  **精确等于那 2 处 bugfix**，零 WIP 混入。
- **排除并保留为未提交 WIP（5 文件）**：linear_solvers、huzhang_elastic_assembler、
  phasefield_assembler、phasefield_damage、huzhang_damage_staggered。它们的 docstring
  等与各自 WIP 一并提交即可（同文件、互不冲突）。
- `huzhang_phasefield_staggered.py` 的 docstring 是 2026-06-01 commit `6dc62ee` 已写好，
  本次无新增。
- `recover_strain.py` 捎带 1 行 doc 路径改名（docs 重组的一部分，正确无害），已并入。

提交信息按 `tian/CLAUDE.md` 约定，无 Claude 署名。

## 5. 遗留

- 5 个夹 WIP 文件的 docstring 待你提交 aux/装配缓存线时一并带上。
- auxspace 测试本身在默认 hmin=0.02 下是否全绿、`rel_diff≈1` 是否还有非网格因素，
  本会话未深挖（只确认 0.04 已 PASS）；如需可另起一刀核默认网格。
