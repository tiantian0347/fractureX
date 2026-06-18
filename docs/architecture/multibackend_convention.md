# FractureX 多后端编码规范（统一约定）

> 状态：**统一规范**（2026-06-09 立）。适用于 `fracturex/` 下所有**新增或修改**的代码。
> 目的：让计算代码在 FEALPy 的 numpy / torch / jax 后端上**同一份源码可运行**，为
> GPU 路线（[routes/plan_gpu_multibackend](../routes/plan_gpu_multibackend.md)）与算子学习
> 可微化打底。关联记忆：`fracturex_multibackend_convention`、`dont_modify_fealpy`。

---

## 0. 一句话规范

> **计算用 `bm`（`from fealpy.backend import backend_manager as bm`），不用 `import numpy as np`；
> numpy 只允许出现在 I/O 与 scipy 边界，且必须显式用 `bm.to_numpy(...)` 跨过边界。**

---

## 1. 默认：计算走 `bm`

新增/修改的数值代码（特征提取、装配、退化律、后处理统计等）一律用后端管理器：

```python
from fealpy.backend import backend_manager as bm

x = bm.zeros(n, dtype=bm.float64)
y = bm.sqrt(bm.sum(g * g, axis=-1))
acc = bm.index_add(acc, idx, vals)        # 多后端 scatter-add（= np.add.at）
phi = bm.stack([c0, c1, c2], axis=1)      # 不要用 phi[:, i] = ... 原地赋值
```

**理由**：numpy 专属的原地操作（`np.add.at`、布尔掩码赋值 `a[mask]=...`、fancy 赋值
`a[:, i]=...`）在 jax 下不可用、在 torch 下语义不一致。用 `bm` 的函数式等价物可移植。

---

## 2. 禁用清单（在计算代码里）

| 禁用 | 替代 |
|------|------|
| `import numpy as np` 作计算 | `from fealpy.backend import backend_manager as bm` |
| `np.add.at(a, idx, v)` | `a = bm.index_add(a, idx, v)`（或 `bm.scatter_add`） |
| `a[mask] = b[mask]` | `a = bm.where(mask, b, a)` 或 `acc / bm.maximum(cnt, 1.0)` |
| `phi[:, i] = col`（逐列原地） | `bm.stack([...], axis=1)` |
| `np.asarray(mesh.entity(...))` 强转 | 直接用（已在活动后端）；需要时 `bm.asarray` / `bm.astype` |
| 类型注解 `np.ndarray` | `Any`（后端张量类型不定） |

---

## 3. 允许 numpy 的边界（明确豁免）

numpy/scipy **不是要消灭**，而是**约束在边界**。以下场景**应当**用 numpy，且跨边界处显式转换：

1. **scipy 稀疏 / Krylov 求解器**：`scipy.sparse.linalg.gmres/spilu`、`pyamg` 等只吃
   numpy/scipy 数据。`utilfuc/linear_solvers.py`、`sparse_direct_backends.py`、
   `matfree_elastic.py` 的求解主链**保持 numpy/scipy**——这是设计，不是欠债。
2. **文件 I/O**：`np.savez`/`np.load`/VTK 写出是 numpy 格式。落盘前用
   `bm.to_numpy(x).astype(np.float32)` 显式转换（torch/jax 张量没有 numpy 的 `.astype`）。
3. **第三方库**：matplotlib、pandas 等只认 numpy。

**边界写法**（唯一正确的跨界方式）：

```python
import numpy as np                      # 仅在 I/O / scipy 文件顶部
...
np.savez_compressed(out, phi=bm.to_numpy(cf.phi).astype(np.float32))
```

---

## 4. 不改 fealpy

bm 缺某个算子时，**不改 fealpy 源码**（见记忆 `dont_modify_fealpy`）。在 fracturex 内用
现有 bm 原语组合实现（如本规范用 `index_add`+`broadcast_to` 复刻 scatter-mean），需要时
向 fealpy 接口对接但不动其内核。

---

## 5. 落地策略（不搞大爆炸重写）

- **存量代码**（57 文件含 `import numpy as np`）：**不强制立即重写**。仅当你**修改某文件的
  计算逻辑**时，顺手把触及的部分迁到 bm；纯 I/O/scipy 文件按 §3 豁免保留。
- **新代码**：从第一行就按本规范写。
- **参考实现**：`fracturex/ml/coarse_features.py` 是本规范的范例（纯 bm 计算 + I/O 在
  `scripts/paper_precond/dump_features.py` 边界转 numpy）。

---

## 6. 验证

- 后端无关性测试：同一函数在 numpy 后端跑通且数值与改造前一致（回归基线）。
- 切换后端（`bm.set_backend("pytorch")`）后计算路径不报「numpy 专属操作」错误。
  （注：整链切后端还需 fealpy 侧配置；本规范保证 fracturex 侧无障碍。）
- 架构契约：计算模块 import 不泄漏与职责无关的重依赖（如特征模块不 import torch/solver）。

---

## 附录 A：存量代码扫描清单（2026-06-09，已逐处人工核验）

全量扫描 `fracturex/`（非 tests）的 numpy 专属原地操作（`np.add.at`、bm 张量 fancy/掩码赋值）。
分三类：**真违规**（计算路径、应随改随迁）、**豁免-scipy/IO**（设计如此，保留）。

### A.1 真违规（计算路径，待随改随迁）

| 文件:行 | 操作 | 说明 |
|---|---|---|
| `boundarycondition/huzhang_boundary_condition.py` | 多处 `gval[idx,:,:]=` / `wcoef[idx,:]=` / `phin[idx,...]=` / `F_new[mask]=` / `uh[dof]=`（约 10 处，L166/254/256/343/356/361/365/368/464/636） | 左侧是 `bm.zeros/ones` 张量，对其做 fancy/掩码原地赋值 —— numpy 后端可跑、jax 不支持。应改 `bm.set_at` / `bm.index_add` / `bm.where` |
| `assemblers/phasefield_assembler.py` L744/748/761 | `rhs=np.zeros`+`np.add.at` | 相场 RHS 的 **numpy 并行装配快路径**（bm 路径是上方 `lform.assembly()`）。是有意的 numpy-only 优化，非 portable；迁移需保并行性能，优先级中 |

### A.2 豁免（设计如此，**不迁**）

| 文件 | 操作 | 豁免类别 |
|---|---|---|
| `utilfuc/matfree_elastic.py` L186/198 | `np.add.at` | scipy：整类是 `scipy.sparse.linalg.LinearOperator` 的 matvec，喂 scipy GMRES；其 docstring 已注明 GPU 化时才换 numpy 内核 |
| `utilfuc/linear_solvers.py` 全文 | numpy/scipy | scipy：gmres/lgmres/minres/spilu/pyamg 求解器包装 |
| `utilfuc/sparse_direct_backends.py` | numpy 格式转换 | scipy：PARDISO/MUMPS/SuperLU 直接求解 |
| `postprocess/dataset_export/sampling.py` L263 | `np.add.at`+`spsolve` | IO+scipy：L² 投影导出工具，scatter 后即 `scipy.spsolve` |
| `utilfuc/recover_strain.py`、`phasefield_mesh.py`、`linear_elastic_pde.py` | numpy 计算/sympy/统计 | 边界：无专属原地操作或 sympy 制造解/网格统计 |
| `drivers/*`、`postprocess/dataset_export/*`（其余） | numpy | IO/可视化/scipy 诊断（残差范数、VTK 重采样、npz 导出） |

### A.3 已合规（参考）

`utilfuc/huzhang_fast_solver.py`、`assemblers/huzhang_elastic_assembler.py`、`damage/*`、
`phasefield/*`、`discretization/*`、`ml/coarse_features.py` 计算路径已全 bm（`bm.add.at`/
`bm.set_at` 等后端 API）。

> **总计**：真违规 2 文件（bc ~10 处 + phasefield_assembler 1 处 numpy 快路径）；其余皆豁免或已合规。
> 按规范 §5，这些不大爆炸重写，仅在下次修改对应文件的计算逻辑时顺手迁移。
