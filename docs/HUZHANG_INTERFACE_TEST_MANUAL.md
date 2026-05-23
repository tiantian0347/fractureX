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
- 相场线性子问题默认统一使用 driver 内置 `_default_lgmres`（无预条件）。
- `use_elastic_fast`（环境变量 `FRACTUREX_ELASTIC_FAST=1`）
  - 弹性子问题走 `solve_huzhang_block_gmres_fast`（Schur + `gs_amg_gs` 等）

## 3. 弹性 formulation 与 GMRES 预条件（系数位置）

### 3.0 标准模型：鞍点块与 Schur 补（理论 ↔ 代码）

固定相场迭代 \(d_h\) 时，离散力学系统为鞍点形式

\[
\mathcal K_h(d_h)
=
\begin{bmatrix}
A(d_h) & B^\top\\
B & 0
\end{bmatrix},
\]

其中 \(A(d_h):\Sigma_h\to\Sigma_h'\) 为退化应力算子（`standard` 下由 \(a_{d_h}(\cdot,\cdot)\) 诱导），\(B:\Sigma_h\to V_h\) 为离散散度。在 \(g(d_h)\ge\xi>0\) 时 \(A(d_h)\) 对称正定，定义 **SPD Schur 补**（吸收常见写法中的负号）

\[
S(d_h) := B\,A(d_h)^{-1}B^\top.
\]

在离散 inf-sup 条件下 \(S(d_h)\) 在 \(V_h\) 上正定。高效求解 \(\mathcal K_h\) 归结为近似两类逆算子：

- \(B_A(d_h)\approx A(d_h)^{-1}\)（应力块，辅助空间预条件）；
- \(B_S(d_h)\approx S(d_h)^{-1}\)（位移块 / Schur 补）。

**与 `HuZhangElasticAssembler` 分块对应**（未知量顺序 `[σ; u]`）：

| 理论符号 | 装配矩阵块 | `linear_solvers` 中提取 |
|----------|------------|-------------------------|
| \(A(d_h)\) | `M2`（`standard` 下随 \(d\) 变） | `A_sigma = A[:m,:m]` |
| \(B\) | `B2.T`（`standard` 下常数） | `B_div = A[m:,:m]` |
| \(B^\top\) | `B2` | `B_div.T` |

代码中 Schur **近似**（每步由当前 `A` 重算）：

\[
\widehat S(d_h) = B\,\mathrm{diag}(A(d_h))^{-1}B^\top,
\]

预条件子对 Schur 残差用 GS / 粗网格 / 可选 ILU 实现 \(B_S\)；对角缩放 `D_inv` 实现 \(B_A\) 的 cheap 部分，`weighted_aux` 时在 P1 粗空间上进一步近似 \(B_A\)（仅 `standard`）。

`effective_stress` 下 \(A(d_h)\equiv A\) 常数、\(B=B(d_h)\) 随损伤变，仍用同一提取公式，但 \(\widehat S=B(d_h)\,\mathrm{diag}(A)^{-1}B(d_h)^\top\)，且 **不在** 粗 Laplacian 上加 \(g(d)\)（见下表）。

### 3.1 formulation 与退化系数位置（互斥）

装配与迭代求解器采用 **同一套互斥约定**：`g(d)` 只出现在 **一块** 上，不能同时在应力块 `M` 与耦合块 `B` 上重复加权。

| `elastic_formulation` | 装配中 `g(d)` 位置 | 线性系统 `A` | 预条件子中粗空间 / Schur |
|----------------------|-------------------|--------------|-------------------------|
| `"standard"` | 应力块 **M**（`HuZhangStressIntegrator`，系数 `1/g`） | `M(d)` 退化，`B` 常数 | Schur 由当前 `A` 分块得到（损伤经 **M** 进入 `D⁻¹`）；`weighted_aux=True` 时 P1 粗扩散用 **`damage.coef_bary` → g(d)²** |
| `"effective_stress"` | 耦合块 **B**（`HuZhangMixIntegrator`，系数 `g`） | `M` 常数，`B(d)` 退化 | Schur 由 **B(d)** 携带损伤；粗空间 **不加** `g(d)`（无权重 P1 Laplacian） |

实现位置：`fracturex/utilfuc/linear_solvers.py`（`solve_huzhang_block_gmres_auxspace`、`solve_huzhang_block_gmres_fast`）。

### 3.2 调用时必须与 assembler 一致

`HuZhangElasticAssembler(..., formulation=elastic_formulation)` 与 GMRES 求解器传入 **相同的** `elastic_formulation`，并建议同时传入 `damage` 与 `state=discr.state`（粗空间权重与 `coef_bary` 一致）：

```python
solve_huzhang_block_gmres_auxspace(
    A,
    F,
    gdof_sigma=discr.gdof_sigma,
    vspace=discr.space_u,
    weighted_aux=True,
    elastic_formulation=elastic_formulation,  # 与 HuZhangElasticAssembler 相同
    damage=damage,
    state=discr.state,
    ...
)
```

`phasefield_model0_huzhang.py` 中 aux / fast 分支已按上表传参。

### 3.3 `weighted_aux` 含义（按 formulation 解释）

- **`standard` + `weighted_aux=True`**：在辅助 P1 扩散上施加 **g(d)²**（应力侧粗空间，与 M 块退化一致）。
- **`effective_stress` + `weighted_aux=True`**：仍做辅助空间校正，但粗 Laplacian **不加 g**；损伤只通过 Schur（来自 **B(d)**）进入预条件。
- **`weighted_aux=False`**：两种 formulation 下粗扩散均为 **无系数** 各向同性 Laplacian；Schur 仍从当前 `A` 提取。

### 3.4 fast Schur 路径（`FRACTUREX_ELASTIC_FAST=1`）

`solve_huzhang_block_gmres_fast` 同样接受 `elastic_formulation`、`damage`、`state`、`weighted_aux`：

- `standard`：每步按当前 `d` 组装 **加权** P1 pyamg 粗校正；
- `effective_stress`：使用 **缓存的无权重** P1 Laplacian，损伤仅在 Schur 块 `S` 中。

### 3.5 接口测试注意

1. 切换 `elastic_formulation` 时，**同时**改 assembler 与 `elastic_solver` 内 GMRES 参数，不要只改一边。
2. 对比 direct vs aux vs fast 时，三组应使用 **同一** `elastic_formulation`。
3. 若 `effective_stress` 下迭代发散而 direct 正常，先确认未误用“在粗空间加 g²”的旧假设；再调 `maxit` / `restart` 或暂时用直接法对照。

## 4. 结果输出位置与文件说明

默认输出根目录由 `outdir` 决定（例如 `results_model0/standard_direct_cache_on`），子目录按 `tag=epsg_xx` 区分。

每次运行常见产物：

- `meta.json`：运行配置、网格规模、材料参数
- `history.csv`：每个 load step 的主指标
- `iterations.csv`：每个非线性迭代的诊断信息
- `summary.json` / `summary.csv`：聚合统计
- `TEST_REPORT.md`：自动生成的测试报告
- `residual_force_vs_displacement.csv/.png`：残余力-位移曲线
- `checkpoints/*.npz`：可选中间状态快照（受 `performance_mode` / `save_npz` 影响）

## 5. 推荐接口测试矩阵

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
   - 确认 GMRES 已传与 assembler 相同的 `elastic_formulation`（见 §3）
4. **退化下界敏感性（弹性子问题 / 辅助空间预条件）**
   - 固定其他参数，测试 `eps_g=1e-6 / 1e-8 / 1e-10`
   - 专用脚本（冻结相场、预设 `d=1` 区域、`M(d)` 系数 `1/g(d)`）：
     ```bash
     bash scripts/run_python.sh fracturex/tests/test_auxspace_precond_degraded_elastic.py
     ```
   - 报告：`results/tests/auxspace_degraded_elastic/TEST_REPORT.md`
5. **网格敏感性**
   - 固定求解参数，测试 `hmin=0.01 / 0.008`（资源允许再细化）

建议每组运行后记录：

- `summary.json` 中 `reaction_force_final`、`damage_max_final`、`avg_nonlinear_iters`
- `history.csv` 的峰值附近响应变化
- 是否出现不收敛或曲线异常抖动

## 6. 快速排错清单

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

## 7. 建议测试流程（可复用）

1. 复制一份基线参数（或提交前保存当前脚本）
2. 一次仅改一个核心接口参数
3. 运行并保留对应 `tag` 输出目录
4. 对比 `summary.json` + `residual_force_vs_displacement.csv`
5. 若性能或收敛异常，再下钻到 `iterations.csv`

## 8. 备注

- 本手册聚焦“接口切换测试”，不覆盖完整理论推导。
- 若你新增了新的 solver / assembler 开关，建议同步更新本手册第 2、3（尤其 §3.0–§3.1）、5 节。
