# Hu-Zhang 角点松弛（corner relaxation）测试脚本目录

本目录汇总关于 `HuZhangCornerRelax` wrapper、`set_essential_bc_v2` 修复、以及
[HM18] §3.1/§4 相关的所有测试脚本。

## 结构

```
corner_relaxation/
├── experiments/       长跑实验（不进 CI，供人工查看和复现）
├── archive/           反例、负结果、已被淘汰的对照
└── README.md          本文
```

## 快速入口（pytest CI）

- **`fracturex/tests/test_huzhang_corner_relax.py`**：wrapper 结构 sanity（DOF 计数、
  零空间维度、cell_to_dof 覆盖、TM 恒等块）。快跑，无副作用。
- **`fracturex/tests/test_corner_relaxation.py`**（本次 PR 新增）：4 阶收敛严格证据
  + [HM18] §3.1 数学等价性验证。适度慢（跑 4 档均匀网格），可作为回归测试。

## experiments/

| 脚本 | 目的 | 时间 |
|---|---|---|
| `lshape_corner_relax_solve.py` | L 形全 Dirichlet quartic 4 阶最优 + 凹角 NN wrapper 集成 | ~5 min |
| `lshape_corner_diagnose.py` | 诊断 fealpy `_filter_active_corners_by_support` 过滤 bug | 秒级 |
| `hm18_williams_singular.py` | [HM18] §5.2 Williams 奇异算例, 1 阶收敛验证 | ~15 min |
| `lshape_adaptive_wrapper.py` | oracle-based 自适应循环（Williams 算例上突破 α 天花板到 DOF^{-2}） | ~30 min |
| `lshape_point_load.py` | L 形短边固定 + 平行短边点力工程算例, Cauchy 阶率 | ~10 min |
| `lshape_point_load_adaptive.py` | 点力算例 + residual estimator (fluctuation + 本构残差) | ~15 min |
| `lshape_point_load_adaptive_convrate.py` | 尝试 fine-mesh reference 算真 L² 阶率（结果不可靠, 归档） | ~30 min |

## archive/

历史反例，说明 wrapper **不能**处理的场景（数学上就无解）——保留作为负结果记录，避免
后续误用：

| 脚本 | 反例内容 |
|---|---|
| `hm18_piecewise_const_stress.py` | [HM18] §5.1-like 分片常应力：fracturex 只检测**几何**角点，不识别"同一直边上 traction 不连续"角点 |
| `hm18_inconsistent_traction_corner.py` | 人为让两条 ΓN 边给出**不相容** σ 数据：任何 wrapper 都无解（违反 H(div)） |
| `lshape_corner_relaxation.py` | 对照 fealpy 自带 `use_relaxation=True`：`_filter_active_corners_by_support` 把所有 NN 角点过滤为 NCP=0，从未生效 |

## 复现命令

```bash
cd /Users/tian00/repository/fractureX
PYTHONPATH=/Users/tian00/repository/fealpy:. \
/Users/tian00/venv_fealpy3/bin/python \
    fracturex/tests/corner_relaxation/experiments/<script>.py
```
