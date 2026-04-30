# HuZhang 相场接口测试使用手册

本文档面向 `fracturex/tests/phasefield_model0_huzhang.py` 的日常接口测试，目标是让你可以快速切换关键模块并观察输出差异。

## 1. 测试入口

- 入口脚本：`fracturex/tests/phasefield_model0_huzhang.py`
- 推荐在仓库根目录执行：

```bash
python fracturex/tests/phasefield_model0_huzhang.py
```

## 2. 主要可测试接口（脚本内开关）

在 `main()` 中可直接修改以下参数：

- `performance_mode`
  - `True`：轻量保存，适合快速迭代
  - `False`：保存更完整（含更频繁数据/可选 vtk）
- `elastic_formulation`
  - `"standard"`：标准退化装配路径
  - `"effective_stress"`：有效应力路径（含耦合项）
- `use_direct_solver`
  - `True`：弹性子问题走默认直接解法
  - `False`：使用 driver 默认路径（按构造方式）
- `release_elastic_iterative_only`
  - `True`：弹性子问题切到 `solve_huzhang_block_gmres_auxspace`
  - `False`：不启用该专用迭代分支
- `eps_g`
  - 控制退化函数下界，常用于稳定性对比测试
- `hmin`
  - 控制网格尺寸，对称性和峰值响应较敏感
- `use_phase_gmres_no_precond`
  - `True`：相场线性子问题使用 driver 内置 `_default_lgmres`（无预条件）
  - `False`：相场线性子问题使用 `_default_spsolve`
  - 也可用环境变量 `FRACTUREX_PHASE_GMRES_NOPREC=1` 开启

## 3. 结果输出位置与文件说明

默认输出根目录由 `outdir` 决定（例如 `results_model0/standard_elastic_direct`），子目录按 `tag=epsg_xx` 区分。

每次运行常见产物：

- `meta.json`：运行配置、网格规模、材料参数
- `history.csv`：每个 load step 的主指标
- `iterations.csv`：每个非线性迭代的诊断信息
- `summary.json` / `summary.csv`：聚合统计
- `TEST_REPORT.md`：自动生成的测试报告
- `residual_force_vs_displacement.csv/.png`：残余力-位移曲线
- `checkpoints/*.npz`：可选中间状态快照（受 `performance_mode` / `save_npz` 影响）

## 4. 推荐接口测试矩阵

可按下表做最小对比集（每次只改 1-2 个变量，便于定位）：

1. **基线**
   - `elastic_formulation="standard"`
   - `use_direct_solver=True`
   - `release_elastic_iterative_only=False`
2. **有效应力接口**
   - 将 `elastic_formulation` 改为 `"effective_stress"`
3. **弹性迭代接口**
   - `use_direct_solver=True`
   - `release_elastic_iterative_only=True`
4. **退化下界敏感性**
   - 固定其他参数，测试 `eps_g=1e-6 / 1e-8 / 1e-10`
5. **网格敏感性**
   - 固定求解参数，测试 `hmin=0.01 / 0.008`（资源允许再细化）

建议每组运行后记录：

- `summary.json` 中 `reaction_force_final`、`damage_max_final`、`avg_nonlinear_iters`
- `history.csv` 的峰值附近响应变化
- 是否出现不收敛或曲线异常抖动

## 5. 快速排错清单

1. **先看终端每步打印**
   - 关注 `err_u`、`err_d`、`error` 是否单调下降或停滞
2. **看 `iterations.csv`**
   - 检查 `linear_converged_*` 与 `linear_niter_*` 是否异常偏大
3. **看 `summary.json`**
   - 对比 `step_convergence_rate`、`max_nonlinear_iters`
4. **检查边界与历史驱动**
   - 当前建议保持 `history_source="from_u"`
5. **用环境变量打开辅助日志**
   - 当使用 aux-space GMRES 分支时，可开启：

```bash
export FRACTUREX_AUXSPACE_DEBUG=1
python fracturex/tests/phasefield_model0_huzhang.py
```

## 6. 建议测试流程（可复用）

1. 复制一份基线参数（或提交前保存当前脚本）
2. 一次仅改一个核心接口参数
3. 运行并保留对应 `tag` 输出目录
4. 对比 `summary.json` + `residual_force_vs_displacement.csv`
5. 若性能或收敛异常，再下钻到 `iterations.csv`

## 7. 备注

- 本手册聚焦“接口切换测试”，不覆盖完整理论推导。
- 若你新增了新的 solver / assembler 开关，建议同步更新本手册第 2、4 节。
