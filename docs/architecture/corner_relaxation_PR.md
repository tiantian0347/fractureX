# PR: Hu-Zhang corner relaxation — fealpy 标架 bug 修复 + [HM18] §3.1 wrapper

## Summary

修复 fracturex `HuzhangStressBoundaryCondition` 在 σ·n ≠ 0 时因 fealpy `HuZhangFESpace2d`
节点/边 trace DOF **标架不一致** 而**不收敛**的 bug。同时实现 [HM18] §3.1 的角点松弛
扩展空间 wrapper 作为后续自适应/断裂研究的基础设施（数学正确性已通过与
[HM18] §3.1 (3.1) 手工构造 basis 的严格 span 比对验证）。

**关键结果**：

| 场景 | 修前 | 修后 |
|---|---|---|
| L 形 全 Dirichlet + 光滑 manufactured, p=3 | 4 阶 ✓ | **4 阶** ✓ 未变 |
| 方形 3 边 ΓN + sin·sin (σn≠0), p=3 | 发散 (7e-1, 6.6e-1) | **4 阶 (3.5e-3 → 2.1e-4 → 1.3e-5)** ✓ |
| L 形 NN 凹角 quartic (σn≠0), p=3 | 发散 (负阶) | 稳定有界 (0.32 稳定) |

## Changes

### 生产代码（3 个）
| 文件 | 类型 | 说明 |
|---|---|---|
| `fracturex/boundarycondition/huzhang_boundary_condition.py` | 修改 | 增 `set_essential_bc_v2(skip_nn_corner_nodes=True)`, 内部函数 `_detect_nn_corner_nodes` |
| `fracturex/discretization/huzhang_corner_relax.py` | 新增 | `HuZhangCornerRelax` wrapper: 任意 m 的 [HM18] §3.1 扩展空间实现 |
| `fracturex/assemblers/huzhang_unc_assembler.py` | 新增 | `assemble_M_unc`, `assemble_B_unc`, `project_to_rel` — unc 空间直接装配 |

### 测试（pytest + experiments 分离）
| 文件 | 类型 | 内容 |
|---|---|---|
| `fracturex/tests/test_huzhang_corner_relax.py` | 已有 | 5 case 结构 sanity（DOF/nullspace/c2d/TM）|
| `fracturex/tests/test_corner_relaxation.py` | 新增 | 3 case：4 阶收敛 + wrapper vs [HM18] §3.1 + v2 BC 修复 |
| `fracturex/tests/verify_wrapper_basis_hm18.py` | 新增 | wrapper vs [HM18] §3.1 数学等价性详细报告（诊断脚本）|
| `fracturex/tests/corner_relaxation/experiments/*` | 归档 | 7 个长跑实验（Williams、点力、自适应等），不入 CI |
| `fracturex/tests/corner_relaxation/archive/*` | 归档 | 3 个反例脚本（数学上不可能的场景）|
| `fracturex/tests/corner_relaxation/README.md` | 新增 | 目录说明和复现命令 |

### 文档（2 份）
| 文件 | 内容 |
|---|---|
| `docs/architecture/huzhang_corner_relaxation_design.md` | 5 章设计文档 + wrapper 决策表 |
| `docs/architecture/lshape_point_load_report.md` | 点力工程算例详细报告 |

## Test summary

在 `venv_fealpy3` 环境下：

```
$ pytest fracturex/tests/test_huzhang_corner_relax.py fracturex/tests/test_corner_relaxation.py -v
...
========================== 8 passed in 4.43s ==================================
```

- **`test_huzhang_corner_relax.py`（5 case, 0.5s）**：wrapper 结构 sanity。
- **`test_corner_relaxation.py`（3 case, 3.9s）**：
  - `test_lshape_smooth_convergence_p3` — 严格 4 阶 (σ) + 3 阶 (u) 收敛
  - `test_wrapper_math_equivalence_hm18_section3_1` — wrapper 与 [HM18] §3.1 数学等价
  - `test_v2_essential_bc_fixes_framework_bug` — v2 相对原 fealpy set_essential_bc 修 bug

## 数学正确性

wrapper 的 4 rel basis 在 unc 空间中张成的子空间**严格等于** [HM18] §3.1 (3.1) 手工
构造的 4 basis 张成的子空间：

- rank(T_wrapper) = 4, rank(T_HM18) = 4, rank([T_wrapper | T_HM18]) = **4**
- $T_{\text{wrapper}} = T_{\text{HM18}} \cdot A$ with $|A x - b| \sim 10^{-15}$, $\det(A) \neq 0$

## 已知边界

1. **只识别几何 NN 角点**（不共线的 ≥2 条 ΓN 边交点）。同一直边上的 traction 不连续
   点（[HM18] §5.1 分片常应力界面）不识别。
2. **两条 ΓN 边给不相容 σ 数据时无解**（数学事实，H(div) 约束）。KKT wrapper 在这种
   人为构造场景下矩阵奇异。反例见 `archive/hm18_inconsistent_traction_corner.py`。
3. **3D 未支持**（`huzhang_fe_space_3d.py` 与 wrapper 都仅 2D）。
4. **wrapper 对光滑 σ 数据无 rate 改善**（[HM18] 承诺阶不变、常数改善，实测常数
   改善微弱）。自适应场景下才能突破 Williams α 阶天花板。

## 生产推荐

**日常线弹性 mixed FEM**：直接开 `skip_nn_corner_nodes=True`，不启用 wrapper。

```python
HSBC = HuzhangStressBoundaryCondition(space=base_space)
uh_sig, isbd_sig = HSBC.set_essential_bc_v2(
    stress_gd, threshold=isN, coord='auto',
    skip_nn_corner_nodes=True,   # ← 唯一推荐参数
)
```

**遇 σ 奇异算例 + 需要 rate 提升**：接自适应循环 (`experiments/lshape_adaptive_wrapper.py`
或 `experiments/lshape_point_load_adaptive.py` 模板)。

**wrapper 主要用于**：后续自适应/相场断裂研究，需 [HM18] §4 松弛结构的场景。

## 参考

- **[HM18]** Hu-Ma, [Partial relaxation of C⁰ vertex continuity](https://doi.org/10.1515/cmam-2020-0003), CMAM 2020.
- 数学理论详细笔记：`Tian/paper/fracture/mixFEM/HuZhang_corner_theory.md`

## 复现

```bash
cd /Users/tian00/repository/fractureX
export PYTHONPATH=/Users/tian00/repository/fealpy:.

# 快速回归测试
/Users/tian00/venv_fealpy3/bin/python -m pytest \
    fracturex/tests/test_huzhang_corner_relax.py \
    fracturex/tests/test_corner_relaxation.py -v

# 详细的 wrapper vs HM18 数学等价性诊断
/Users/tian00/venv_fealpy3/bin/python \
    fracturex/tests/verify_wrapper_basis_hm18.py

# 长跑实验（各 5-30 min）
/Users/tian00/venv_fealpy3/bin/python \
    fracturex/tests/corner_relaxation/experiments/lshape_corner_relax_solve.py 3 4
/Users/tian00/venv_fealpy3/bin/python \
    fracturex/tests/corner_relaxation/experiments/hm18_williams_singular.py 3 4
/Users/tian00/venv_fealpy3/bin/python \
    fracturex/tests/corner_relaxation/experiments/lshape_point_load.py
```
