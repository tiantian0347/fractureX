# 标准有限元慢模态验证结果

## 验证入口

```bash
env MPLCONFIGDIR=/tmp/fracturex_mplconfig \
    XDG_CACHE_HOME=/tmp/fracturex_cache \
    FONTCONFIG_PATH=/tmp/fracturex_fontconfig \
    PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /Users/tian00/venv_fealpy3/bin/python \
    scripts/paper_solver/verify_slow_mode_fracturex.py \
    --nx 4 --output-dir results/phasefield_solver/slow_mode_smoke
```

脚本调用 `fracturex.phasefield.MainSolve` 的标准 Lagrange 有限元装配与
直接稀疏求解。历史场在每次映射调用前恢复为已提交快照，因而测试的是确定的
准静态交错映射，而不是累计中间试算状态。

## 首次烟雾算例

| 量 | 结果 |
|---|---:|
| 网格 | (4\times4)，32 个三角形单元 |
| 标量相场自由度 | 25 |
| 交错固定点迭代次数 | 13 |
| 固定点最终增量范数 | (1.20\times10^{-10}) |
| 传播算子谱半径 | (8.8016\times10^{-2}) |
| 主模态重放收敛比 | (8.8014\times10^{-2}) |
| 主模态速率误差 | (2.08\times10^{-6}) |
| 主模态能量覆盖阈值 | 70% |
| 入选单元比例 | 28.1% |
| 入选单元能量覆盖率 | 72.4% |
| 历史场回退误差 | 0 |
| 映射重复调用误差 | (2.40\times10^{-15}) |

完整机器可读结果写入：

```text
results/phasefield_solver/slow_mode_smoke/summary.json
results/phasefield_solver/slow_mode_smoke/meta.json
```

随后在 (6\times6) 网格上复核：72 个单元、49 个相场自由度，谱半径
(8.1673\times10^{-2})，主模态重放误差 (4.75\times10^{-7})，入选单元比例
26.4%，能量覆盖率 71.4%，全部检查通过。该组结果写入
`results/phasefield_solver/slow_mode_nx6/`。

## 耦合慢子空间检查点扫描

验证入口已扩展为完整交错映射 \((u,d)\mapsto(u^+,d^+)\) 的有限差分诊断。完整状态按“位移自由度、相场自由度”排列；相场子块与既有 \(T\) 的差在有限差分容差内，位移输入列为零。以固定的 90% 相对谱半径阈值选取实慢子空间，并用位移与相场切线对角构成的块对角权重计算单元迹指标。

在单调提交损伤与历史场的三检查点扫描中，所有代数与确定性检查通过：

| 网格 | 载荷 | 迭代数 | \(\rho(T)\) | 慢子空间维数 | 70% 迹覆盖所需单元比例 |
|---|---:|---:|---:|---:|---:|
| \(4\times4\) | 0.0125 | 45 | 0.6058 | 2 | 43.8% |
| \(4\times4\) | 0.0250 | 10 | 0.0876 | 1 | 18.8% |
| \(4\times4\) | 0.0500 | 7 | 0.0608 | 1 | 18.8% |
| \(6\times6\) | 0.0125 | 25 | 0.2764 | 2 | 36.1% |
| \(6\times6\) | 0.0250 | 8 | 0.0820 | 1 | 18.1% |
| \(6\times6\) | 0.0500 | 6 | 0.0779 | 1 | 18.1% |

在后两个检查点，等规模损伤区和 \(\|\nabla d\|\) 区分别只覆盖约 4% 和 14%--18% 的耦合慢子空间迹，而慢模态选区覆盖约 71%--76%。这说明完整耦合指标与常规损伤图存在实质差异；该结论目前适用于本预制损伤种子的快速验证算例。

完整结果及复现元数据位于工作区结果根：

```text
/Users/tian00/repository/results/phasefield_solver/coupled_slow_subspace_scan_nx4/
/Users/tian00/repository/results/phasefield_solver/coupled_slow_subspace_scan_nx6/
```

每个目录包括 `summary.json`、`checkpoints.csv`、`checkpoints.npz` 和 `meta.json`。
复现命令：

```bash
env MPLCONFIGDIR=/tmp/fracturex_mplconfig \
    PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /Users/tian00/venv_fealpy3/bin/python \
    scripts/paper_solver/verify_slow_mode_fracturex.py \
    --nx 6 --loads 0.0125,0.025,0.05 \
    --output-dir /Users/tian00/repository/results/phasefield_solver/coupled_slow_subspace_scan_nx6
```

该扫描建立了 H1--H4 的完整程序链路，但尚未模拟裂纹萌生至扩展过程。下一阶段应以带缺口的标准基准算例生成物理检查点，再复用同一诊断接口；在该证据具备之前，不进入局部联合消元的效率比较。

## Model-0 圆孔基准初测

在标准位移有限元 Model-0 圆孔算例上进行了快速四检查点扫描。使用 distmesh `hmin=0.1`，网格含 123 个三角形单元和 83 个相场自由度；内圆边界固定 \(u=0,d=0\)，顶部施加竖向位移。固定相场 Dirichlet 自由度从传播矩阵中显式排除。

| 载荷 | 最大损伤 | 迭代数 | \(\rho(T)\) | 实测衰减比 | 慢子空间维数 |
|---:|---:|---:|---:|---:|---:|
| 0.014 | 0.0107 | 6 | 0.0179 | 0.0181 | 1 |
| 0.070 | 0.2149 | 22 | 0.3802 | 0.3802 | 1 |
| 0.100 | 0.3421 | 55 | 0.6950 | 0.6950 | 1 |
| 0.125 | 0.5875 | 84 | 0.7697 | 0.7697 | 2 |

该结果给出清晰的 H1--H2 证据：损伤发展过程中，谱半径与实际交错衰减比同步增大，迭代数由 6 增至 84；90% 相对谱阈值下慢子空间维数保持在 1--2。70% 耦合迹覆盖需要约 12.2% 单元，在最后一个二维慢子空间检查点增至 23.6%。Model-0 中等规模损伤区与慢模态区高度重叠，而 \(\|\nabla d\|\) 区在中间检查点只覆盖约 18%--37% 的慢子空间迹。

完整结果位于：

```text
/Users/tian00/repository/results/phasefield_solver/model0_coupled_slow_scan_h010/
```

该组结果已具备完整映射、实际收敛率、低维性和局部性四项诊断，可作为后续加密复核及固定 patch 生存因子验证的首个基准。

## Model-0 加密网格与连续路径

为消除直接跳载带来的路径依赖，进一步固定 distmesh 随机种子 `seed=0`，使用最大载荷步
`0.0025` 从零载荷连续推进，仅在五个检查点计算完整传播算子。网格为 `hmin=0.05`，
含 640 个单元和 372 个相场自由度。

| 载荷 | 检查点迭代数 | \(\rho(T)\) | 慢空间维数 | 最大损伤 | 70% 迹覆盖单元比例 |
|---:|---:|---:|---:|---:|---:|
| 0.0700 | 27 | 0.5071 | 1 | 0.207 | 10.0% |
| 0.0850 | 51 | 0.7002 | 1 | 0.309 | 10.5% |
| 0.1000 | 77 | 0.7883 | 1 | 0.478 | 10.5% |
| 0.1125 | 241 | 0.9103 | 2 | 0.879 | 14.2% |
| 0.1250 | 113 | 0.8550 | 2 | 0.941 | 20.0% |

连续路径将慢化峰值定位在 \(u=0.1125\)：此处 \(\rho(T)=0.9103\)，实际衰减比为
0.9103，交错迭代达到 241 次；随后进入裂纹扩展后的新状态，谱半径回落但慢空间维数
保持为 2。该趋势在粗网格与加密网格上保持一致，说明 H1--H2 具有网格稳定性迹象。

峰值邻域的固定步长加密结果如下。三类 patch 使用相同单元数预算。

| 载荷 | 交错迭代 | \(\rho(T)\) | \(r\) | 70% 单元占比 | \(\chi_{\rm slow}\) | \(\chi_{\rm damage}\) | \(\chi_{\nabla d}\) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1050 | 73 | 0.7813 | 1 | 10.9% | 0.432 | 0.515 | 0.554 |
| 0.1075 | 108 | 0.8531 | 1 | 8.9% | 0.465 | 0.506 | 0.525 |
| 0.1100 | 173 | 0.8966 | 1 | 8.9% | 0.457 | 0.516 | 0.529 |
| 0.1125 | 241 | 0.9103 | 2 | 14.2% | 0.349 | 0.548 | 0.598 |
| 0.1150 | 130 | 0.8648 | 2 | 15.3% | 0.351 | 0.593 | 0.629 |
| 0.1175 | 113 | 0.8480 | 2 | 16.1% | 0.385 | 0.610 | 0.649 |

六个检查点均满足 \(\chi_{\rm slow}<\chi_{\rm damage}<\chi_{\nabla d}\)，说明区域排序
并非单个峰值状态的偶然现象。该表属于 SPD 对角权校准；真实 history-field
\(Q_\omega\) 的匹配预算结果见下一节。

机器可读结果位于：

```text
/Users/tian00/repository/results/phasefield_solver/model0_peak_scan_h005_uniform/
```

## 真实 history-field 区域比较

在峰值 \(u=0.1125\) 处组装物理联合残差的固定主动集有限差分 Jacobian。三个区域按
光滑自由度数匹配，结果如下。

| 区域 | 单元数 | 光滑自由度 | \(\chi(w_1)\) | \(\chi(\mathcal V_2)\) | \(\kappa(J_{\omega\omega})\) |
|---|---:|---:|---:|---:|---:|
| 慢子空间区 | 91 | 174 | 0.696 | 0.717 | \(2.78\times10^4\) |
| 损伤区 | 113 | 173 | 0.628 | 0.768 | \(1.47\times10^4\) |
| \(\nabla d\) 区 | 105 | 175 | 0.787 | 0.792 | \(1.21\times10^4\) |

损伤区对第一主模态的 survival factor 最小；慢子空间区对整个二维慢子空间的最坏
survival factor 最小。区域选择基于 \(\mathcal V_2\) 的迹指标，因此后者是与选择目标
一致的评价量。该结果说明 solver-aware 区域的优势体现在同时覆盖多个慢方向。

该 Jacobian 包含 982 个光滑自由度，相对非对称度为 \(1.05\times10^{-3}\)；基态残差
相对尺度为 \(5.36\times10^{-8}\)，三类局部消元方程误差均小于
\(7.6\times10^{-15}\)。机器可读结果位于：

```text
/Users/tian00/repository/results/phasefield_solver/model0_history_q_peak_h005_matched/
```

复现命令：

```bash
env MPLCONFIGDIR=/tmp/fracturex_mplconfig PYTHONPATH=. \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /Users/tian00/venv_fealpy3/bin/python \
    scripts/paper_solver/verify_slow_mode_fracturex.py \
    --case model0_circular_hole --mesh-size 0.05 --seed 0 \
    --loads 0.1,0.1125 --continuation-step 0.0025 --max-iterations 320 \
    --history-survival-loads 0.1125 \
    --output-dir /Users/tian00/repository/results/phasefield_solver/model0_history_q_peak_h005_matched
```

在同一组检查点上进行 SPD 对角权校准的坐标 patch 比较。以相同单元规模构造慢模态区、
损伤区和 \(\|\nabla d\|\) 区，慢模态 patch 的 survival factor 在峰值点为 0.349，低于
损伤区的 0.548 和梯度区的 0.598；对应预测复合衰减因子分别为 0.318、0.499 和 0.545。
直接计算 \(Q_\omega\bm w\) 与 \(|\lambda|\chi_\omega(\bm w)\) 的误差为
\(1.1\times10^{-16}\)，H5 的 SPD 代数闭环通过。

## 真实局部联合残差消元先导验证

在同一标准有限元接口上进一步实现固定 patch 的真实局部残差求解。局部问题采用
投影残差
\[
  r_\omega(x_\omega)=x_\omega-
  \Pi_{[l_\omega,u_\omega]}\bigl(x_\omega-F_\omega(x)\bigr)=0,
\]
以处理相场不可逆约束；局部 Jacobian 仅在 patch 自由度上有限差分。每次校正只有在
完整块尺度修正范数下降时才接受，否则回退到普通交错，从而排除局部复合映射的伪固定点。

Model--0 粗网格（\(h_{\min}=0.1\)）连续路径的先导结果如下。表中“接受数”是被残差
保护实际接受的局部校正次数；局部求解失败或未降低联合残差的试次均回退。

| 载荷 | 普通交错 | 慢模态 patch | 损伤 patch | 慢模态接受数 | 损伤接受数 | 最终块修正范数 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.100 | 52 | 52 | 52 | 8 | 8 | \(4.1\times10^{-12}\) |
| 0.125 | 46 | 46 | 46 | 0 | 0 | \(1.0\times10^{-11}\) |

如果只用相场增量作为停止判据，未加保护的 \(\mathcal E_\omega\circ\mathcal G\) 在
\(u=0.125\) 会给出表面上的 16 次迭代；但其完整块修正范数为
\(3.6\times10^{-3}\)，而普通交错固定点为 \(6.7\times10^{-12}\)，相场解与基准解的
\(L^2\) 差异为 \(3.8\times10^{-3}\)。这一对照确定了论文实现必须采用完整联合残差和残差下降保护。

当前局部 Jacobian 采用逐列有限差分，\(u=0.10\) 单个 patch 约需 20 s，计算成本高于
普通交错；\(u=0.125\) 的梯度 patch 在严格局部容差下出现主动集停滞并自动回退。下一步
采用严格的约化非线性残差：给定区域外变量先求解 \(F_\omega=0\)，再求解
\(\widehat F=F_{\bar\omega}(\Phi_\omega,\cdot)=0\)。局部 Jacobian 将由逐列有限差分改为
装配型 \(J_{\omega\omega}\)，随后在 \(u=0.105\)--\(0.1175\) 的峰值邻域比较完整残差、
离散解一致性和实际总时间。

机器可读结果位于：

```text
/Users/tian00/repository/results/phasefield_solver/model0_local_elimination_h010_safe/
```

复现命令：

```bash
env MPLCONFIGDIR=/tmp/fracturex-mpl \
    PYTHONPATH=/Users/tian00/repository/fractureX \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /Users/tian00/venv_fealpy3/bin/python \
    scripts/paper_solver/verify_slow_mode_fracturex.py \
    --case model0_circular_hole --mesh-size 0.1 --seed 0 \
    --loads 0.1,0.125 --continuation-step 0.005 --max-iterations 180 \
    --local-elimination --local-patches slow,damage \
    --output-dir /Users/tian00/repository/results/phasefield_solver/model0_local_elimination_h010_safe
```

结果目录：

```text
/Users/tian00/repository/results/phasefield_solver/model0_coupled_slow_scan_h005_path/
```

复现命令：

```bash
env MPLCONFIGDIR=/tmp/fracturex_mplconfig \
    PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /Users/tian00/venv_fealpy3/bin/python \
    scripts/paper_solver/verify_slow_mode_fracturex.py \
    --case model0_circular_hole --mesh-size 0.05 --seed 0 \
    --loads 0.07,0.085,0.1,0.1125,0.125 \
    --continuation-step 0.0025 --max-iterations 320 \
    --output-dir /Users/tian00/repository/results/phasefield_solver/model0_coupled_slow_scan_h005_path
```

## 长度尺度可解析路径上的在线慢率

Model--0 采用与 `model0_example.py` 一致的几何、材料、边界条件和报告载荷，
目标网格尺寸为 `hmin=0.0065`，实际
\(h_{\max}=0.0099296<\ell_0/2=0.01\)。物理路径保存每个已接受内部续接状态，
并从真实的相邻状态重放普通交错映射；用末五个完整状态增量和切线对角权计算在线慢率。

主要结果为：

| 载荷区间/状态 | \(\widehat\rho_{\rm on}\) | 结论 |
|---|---:|---|
| 0.0799--0.0854 | 0.718--0.770 | 峰前逐步变慢 |
| 0.0865--0.0876 | 0.800--0.870 | 接近峰值并进入灰区 |
| 0.0898--0.0986 | 0.949--0.981 | 反力突降后的持续强慢区 |
| 0.1008 | 0.514 | 普通交错在 22 次内恢复收缩 |
| 0.1030 | 1.004 | 最终失稳步的观察窗口内增量未衰减 |

在线增量子空间维数为 1--3。物理路径在 \(0.1030\) 使用 safeguarded Anderson 和
内部续接共 152 次迭代；相邻内部状态的路径一致重放在反力突降步给出
\(0.949\)--\(0.981\)，在 \(0.1008\) 的稳定支段降至 \(0.514\)，说明该低点是真实局部快速收缩，
而非报告点大步重放造成的伪影。

结果与复现入口：

```text
results/phasefield_solver/model0_resolved_internal_prefix_h0065/
results/phasefield_solver/model0_resolved_internal_postpeak_h0065/
results/phasefield_solver/model0_resolved_internal_online_rate_h0065/
scripts/paper_solver/scan_model0_internal_path_online_rate.py
scripts/paper_solver/plot_model0_resolved_online_rate.py
```

正式结果以 `results/phasefield_solver/` 下的目录为唯一来源。其中主目录保存
`online_rate_internal_path.csv`、逐内部步 trace、`meta.json` 以及
`physical_slowdown_bridge.{pdf,png,meta.json}`。论文目录只保留该 PDF 的排版副本
`model0_resolved_online_rate.pdf`。

内部路径诊断的定义是：每个重放从实际前一接受状态的
\((\bm z_j,H_j,d_j^{\rm lb})\) 出发，固定该状态的历史场和相场下界，
而不是跨报告点直接重放。所有 \(\widehat\rho_{\rm on}\) 均标记为五步有限窗口观察值；
普通交错未在 40 次内收敛的步仍保留在 CSV 中，并通过 `plain_replay_converged=false`
区分于已收敛的局部收缩观察。

## 解释

该结果验证了以下工程链路：真实标准有限元交错映射可重复调用；有限差分传播矩阵可由工程装配获得；主特征模态可预测沿该模态扰动的渐近衰减；历史场回退与单元能量分配满足预设代数不变量。工程脚本采用相场切线矩阵的正对角构造对角权矩阵，并按 FE 连接关系分配到单元；它是论文中一般 SPD 单元权的可复现实例。

该烟雾算例尚未验证局部非线性消元的总成本优势。后续应在同一工程接口下，对多个载荷状态和网格尺度比较普通交错、Anderson 加速、理想局部投影及联合局部消元。

## Reduced--NE 外层 Schur--Krylov 优化

解析网格 Model--0 的 Reduced--NE 采用相场 patch、矩阵自由精确 Schur 作用和
外层 block-LU 预条件器。`J_{ud}` 已按生产 FE 的 Dirichlet 行消元规则组装，并通过
中心差分验证：小算例最大相对误差 $2.2e-11$，解析网格为 $7.4e-7$。

| 配置 | 初始投影残差 | 最终投影残差 | 外层 Krylov | 局部 Newton |
|---|---:|---:|---:|---:|
| block-diagonal，warmup 3 | $1.275e-2$ | $1.250e-2$ | 285 | 155 |
| block-LU，warmup 3 | $1.275e-2$ | $1.250e-2$ | **131** | **118** |
| block-LU，warmup 10 | $1.233e-2$ | $1.182e-2$ | 125 | 96 |

结果目录：

```text
results/phasefield_solver/model0_resolved_reduced_ne_0898_reference_blocklu_jacobianfix_v1/
results/phasefield_solver/model0_resolved_reduced_ne_0898_reference_blocklu_warmup10_v1/
```

block-LU 显著降低线性迭代和局部消元工作；warmup 增加到 10 次进一步降低初始残差。
`patch_secant` 会把初始残差增大到 $1.477e-2$，因此保留为可选诊断模式。
严格 KKT 验收闭环仍需完成，下一步先验证 reference-free 自适应初始化与外层回溯。

## Reference-free 自适应初始化与外层回溯

Reduced--NE 的初始化现在由少量已接受的交错扫掠构造，不读取参考根。默认流程为：至少
3 次、最多 12 次热启动；每次记录完整耦合投影残差和相邻状态增量比率。当连续慢率条件满足
且最新直接残差比不超过 `0.8`

\[
\widehat\rho_{\rm warm}\ge 0.89
\]

或直接投影残差达到给定阈值时，停止热启动并进入约化 Newton；否则继续扫掠至上限。外层约化步保留 Armijo
回溯，每个接受态的 `projected_residual_norms` 写入结果 JSON，用于检查残差是否单调下降。

该流程对应的实现入口为
`_run_reference_free_warmup`，配置项包括 `reduced_warmup_mode`、
`reduced_warmup_min_sweeps`、`reduced_warmup_max_sweeps`、
`reduced_warmup_slow_rate` 和 `reduced_warmup_required_slow_steps`。
当前阶段先验证求解可靠性；严格 KKT 通过后再恢复外层性能优化。

Model--0 集成烟雾测试结果保存在
`results/phasefield_solver/model0_resolved_reduced_ne_0898_adaptive_warmup_smoke_v1/`。
该测试限制为一次局部/外层修正，仅用于验证自适应 warmup、残差记录和 Reduced--NE
调用链；正式 KKT 结果仍以固定历史耦合参考根实验为准。

完整固定历史 reference-root 测试保存在
`results/phasefield_solver/model0_resolved_reduced_ne_0898_reference_adaptive_warmup_full_v1/`。
自适应 warmup 执行 5 次扫掠，末两次在线率为 $0.9645$ 和 $0.9896$；block-LU 外层
接受 12 次 Armijo 修正，投影残差从 $1.259e-2$ 降至 $1.197e-2$，但状态修正
$1.65e-3$ 和条件数加权残差 $6.29e-1$ 均未通过 reference-free 验收。总时间为
$489.73$ s，等价工作量为 $308$，因此该记录用于可靠性边界分析，不作为加速结果。

## Schur 方向诊断与线性 continuation

外层求解器现在额外记录三类量：完整投影残差、patch 外约化投影残差和局部投影残差；每个外层修正还记录线性化 Schur 方向残差、接受步长和回溯次数。线性化方向残差用于区分 Schur--Krylov 方向误差与非线性全局化误差，相关字段写入每个 patch 的 `summary.json`。

自适应 warmup 的切换条件为：连续在线增量率达到慢率阈值，且最新直接耦合残差比不超过 `0.8`；否则继续扫掠，直至达到最大次数。该条件避免仅凭高慢率提前切换到尚未进入可靠 Newton 邻域的状态。

在长度尺度可解析的 Model--0 $\bar u=0.0898$ 状态运行四阶段线性 continuation：

```text
results/phasefield_solver/model0_resolved_reduced_ne_0898_reference_adaptive_warmup_continuation_v1/
```

四个阶段 $	heta=0.25,0.50,0.75,1.00$ 的外层修正次数分别为 $6,4,3,7$，阶段末投影残差分别为 $1.43\times10^{-10}$、$2.55\times10^{-10}$、$4.37\times10^{-10}$ 和 $5.79\times10^{-10}$。最终相对状态修正为 $2.63\times10^{-8}$，条件数加权残差为 $3.04\times10^{-8}$，完整 KKT 根状态差为 $4.93\times10^{-7}$，可靠性验收通过。该运行共使用 20 次外层修正、442 次 Krylov 迭代和 94 次等价残差评估，总时间 $142.52$ s；相对于 40 次、$42.77$ s 的 staggered 基线，当前 continuation 已解决外层可靠性问题，但仍未形成加速结果。
