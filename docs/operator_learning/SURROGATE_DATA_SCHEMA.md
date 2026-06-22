# Operator-Learning Surrogate Data Schema (v0.1)

> 状态：v0.1 草案（2026-05-27）。此文件是 `fracturex` 仿真侧与
> `fracturex/learn/` 训练侧之间的**外部协议**：训练代码只依赖本文档
> 定义的字段、形状、单位、归一化与 mask 语义，不读求解器内部状态。
>
> 配套规划：[plan_operator_learning.md](plan_operator_learning.md) §3.8 / §M0。

---

## 1. 适用范围与设计目标

- **范围**：本 schema 定义"一次相场断裂仿真 → 一个或多个训练样本"的落盘格式。
  对应 `plan_operator_learning.md` §2 中 T1 / T2 / T3 三类任务的共同输入输出。
- **设计目标**：
  1. 训练侧（PyTorch / JAX）只读 `.npz` + `.json`，与 FEALPy / fracturex 解耦；
  2. 全部字段可在 `RunRecorder` 输出目录上由 `dataset_export.py` 离线计算，
     不修改求解主链；
  3. 单条样本自洽（含 mask、归一化因子、metadata），可以脱离 manifest 单独评估；
  4. 跨 commit 可复现：metadata 记录 git commit + config hash。

---

## 2. 目录布局

一次"数据集"对应 `data/<dataset_name>/` 下的下列文件结构：

```text
data/<dataset_name>/
├── dataset_manifest.json          # 所有样本索引、配置 hash、commit
├── README.md                      # 数据集描述（生成时间、参数空间、统计）
├── samples/
│   ├── sample_000000.npz
│   ├── sample_000000.meta.json
│   ├── sample_000001.npz
│   ├── sample_000001.meta.json
│   └── ...
└── plots/                         # 可选：sanity check 可视化
    └── sample_000000_overview.png
```

**单条样本**由配对的 `sample_XXXXXX.npz` + `sample_XXXXXX.meta.json` 构成。
两者必须同时存在，缺一视为数据集损坏。

---

## 3. NPZ 字段规范

记 $H$、$W$ 为结构网格分辨率（默认 `H=W`，但允许长方形），$T$ 为加载步数，
$k$ 为材料参数维度（默认 5：$\lambda,\mu,G_c,\ell_0,\eta$）。

所有 float 默认 `float32`；mask / 整型字段标注。

### 3.1 输入字段（geometry / material / load）

| 字段名 | 形状 | dtype | 单位 | 说明 |
| --- | --- | --- | --- | --- |
| `sdf` | `(1, H, W)` | float32 | length | 到 $\partial\Omega$ 的有向距离；$\Omega$ 内为正，$\Omega$ 外为负，notch 边界处为 0。 |
| `mask` | `(1, H, W)` | uint8 | — | 1 = $\Omega$ 内（含边界），0 = $\Omega$ 外。**与 `valid_mask` 等价，冗余保留以避免训练时学 SDF → mask 的子任务**。 |
| `coords` | `(2, H, W)` | float32 | — | 归一化坐标 $(x, y) \in [0, 1]^2$，仅在 grid bbox 内归一化（不是 $\Omega$ 内）。 |
| `material` | `(k,)` | float32 | SI | 标量材料参数向量，顺序固定见 §3.5。 |
| `material_field` | `(k, H, W)` | float32 | SI | **可选**。异质材料时使用；为均质则不写入或全为常量广播。 |
| `load_history` | `(T, q)` | float32 | length | 加载历史；$q=1$ 单轴位移加载，$q>1$ 多自由度。**单位与 `material`/`stress_scale` 协调**（见 §3.6）。 |
| `time` | `(T,)` | float32 | — | 归一化时间 $\in [0, 1]$；通常 `time[n] = n / (T-1)`。 |
| `boundary_code` | `(nb, H, W)` | uint8 | — | **可选**。每个边界条件类型一个通道；`nb` 由 `metadata.boundary_codes` 解释。 |

### 3.2 输出字段（damage / stress / 反力等）

| 字段名 | 形状 | dtype | 单位 | 说明 |
| --- | --- | --- | --- | --- |
| `damage` | `(T, 1, H, W)` | float32 | — | 相场 $d \in [0, 1]$，**不归一化**。 |
| `stress` | `(T, 3, H, W)` | float32 | stress / `stress_scale` | 通道顺序 $(\sigma_{xx}, \sigma_{yy}, \sigma_{xy})$。**已按 `metadata.scaling.stress_scale` 归一化**；恢复物理值需乘回。 |
| `history` | `(T, 1, H, W)` | float32 | energy density / `stress_scale` | **可选**。历史场 $\mathcal H$。是否纳入由生成 config 决定，见 plan §3.5 (a)/(b) 方案。 |
| `reaction` | `(T, r)` | float32 | force | 反力曲线；`r` 为加载面数量，与 `load_history` 对应。 |
| `energy` | `(T, e)` | float32 | energy | **可选**。能量分解，列顺序 `(Ψ_elastic, Ψ_crack)`，$e=2$。 |
| `step_iters` | `(T,)` | int32 | — | staggered 迭代次数（debug / 训练样本权重用）。 |
| `step_converged` | `(T,)` | uint8 | — | 0/1，求解是否收敛。**未收敛步默认排除训练，但保留以便人工核查**。 |

### 3.3 Mask 字段

| 字段名 | 形状 | dtype | 说明 |
| --- | --- | --- | --- |
| `valid_mask` | `(1, H, W)` | uint8 | 与 `inputs.mask` 等价的拷贝（便于损失代码不依赖 inputs.mask 命名）。 |
| `boundary_mask` | `(nb, H, W)` | uint8 | **可选**。各类边界条件位置；与 `boundary_code` 配对。 |

### 3.4 域外点（$\Omega$ 外）填充规则

- **输入** `sdf`、`coords`、`material_field`：填实际值（SDF 为负、coords 是 grid 坐标）。
- **输出** `damage`、`stress`、`history`：填 0。
- **所有损失代码必须乘 `valid_mask`**；评估指标按 mask 加权。

### 3.5 `material` 向量顺序

`material[:k]` 的固定顺序（$k=5$）：

| index | 符号 | SI 单位 | 描述 |
| --- | --- | --- | --- |
| 0 | $\lambda$ | Pa | Lamé 第一参数 |
| 1 | $\mu$ | Pa | 剪切模量 |
| 2 | $G_c$ | J/m² | 断裂能 |
| 3 | $\ell_0$ | m | 正则化长度尺度 |
| 4 | $\eta$ | — | 退化函数小数残值，$g(d) = (1-d)^2 + \eta$ |

扩展字段（如 anisotropy）从 index 5 起，须在 `metadata.material_order` 中说明。

> **`metadata.interpolation` 通道适用范围（2026-05-29 据 m0_interpolation_error.md §5 实测确定）**：
> - **σ 通道**：`stress` 走 HuZhang 基函数直接逐点求值，不走 $\mathcal I_1$ /
>   $\mathcal I_2$。`metadata.interpolation` 对 σ 通道无意义，保留语义专门用于
>   $\mathcal H$。
> - **$\mathcal H$ 通道**（schema §3.2 可选字段 `history`）：默认值
>   `"I1_nearest_quad"`，因为裂尖 cusp 在 P2 上的 $L^2$ 投影把峰值抹掉一半多
>   （t_c 上 max_ratio ≈ 0.40，见 m0_interpolation_error.md §4.2 表）；
>   $\mathcal I_1$ 几乎守峰（max_ratio ≈ 1.0），是相场驱动场的正确选择。
> - 数据集 `metadata.interpolation = "I2_L2_projection"` 仍合法，但生成端必须
>   显式承担 cusp 抹平的物理后果；除非有专门 smoothness 分析理由，否则不推荐。

### 3.6 归一化约定

记 `metadata.scaling`（详见 §4）。

- **位移** `load_history`：以 `u_scale` 归一化到无量纲。
- **应力** `stress`：除以 `stress_scale`。反归一化时 `σ_phys = stress * stress_scale`。
- **时间** `time`：恒按加载步数归一化到 `[0, 1]`。
- **损伤** `damage`：恒不归一化（$d \in [0, 1]$ 天然）。
- **历史场** `history`：除以 `stress_scale`（能量密度 $\sim$ 应力量级）。

---

## 4. `<sample>.meta.json` 字段规范

每条样本配套一个 JSON，结构如下：

```json
{
  "schema_version": "0.1",
  "sample_id": "sample_000042",
  "geometry_params": {
    "case": "model0_circular_notch",
    "notch_radius": 0.05,
    "notch_center": [0.0, 0.0],
    "domain_bbox": [[-0.5, 0.5], [-0.5, 0.5]]
  },
  "material_params": {
    "lambda": 121.15e3,
    "mu": 80.77e3,
    "Gc": 2.7,
    "l0": 0.015,
    "eta": 1e-9
  },
  "material_order": ["lambda", "mu", "Gc", "l0", "eta"],
  "formulation": "standard",
  "interpolation": "I2_L2_projection",
  "mesh_info": {
    "NC": 12345,
    "NN": 6789,
    "h_min": 0.001,
    "h_max": 0.01,
    "p_sigma": 2,
    "p_d": 1,
    "p_u": 1
  },
  "grid": {
    "H": 128,
    "W": 128,
    "domain_bbox": [[-0.5, 0.5], [-0.5, 0.5]]
  },
  "load": {
    "kind": "monotone_uy",
    "u_max": 0.01,
    "N_steps": 100,
    "load_surfaces": ["top"]
  },
  "scaling": {
    "stress_scale": 1.0e3,
    "u_scale": 0.01,
    "length_scale": 1.0,
    "time_scale": 1.0
  },
  "boundary_codes": {
    "0": "free",
    "1": "dirichlet_uy",
    "2": "neumann_traction"
  },
  "solver_config": {
    "staggered_tol": 1e-4,
    "staggered_maxit": 500,
    "elastic_solver": "gmres-auxspace",
    "phase_solver": "gmres-phase",
    "elastic_tol": 1e-6,
    "phase_tol": 1e-10
  },
  "git_commit": "e118cc9",
  "config_hash": "sha256:...",
  "run_paths": {
    "recorder_dir": "results/phasefield/model0_circular_notch/paper_aux_h1/epsg_1e-06"
  },
  "stats": {
    "max_damage": 0.987,
    "peak_reaction": 152.3,
    "converged_step_ratio": 1.0,
    "n_valid_steps": 100
  }
}
```

**强制字段**：`schema_version`、`sample_id`、`grid`、`material_params`、
`material_order`、`scaling`、`git_commit`、`config_hash`、`formulation`、
`interpolation`、`solver_config`、`stats`。

`formulation` 必须从 `{"standard", "effective_stress"}` 中取值；同一数据集
内必须一致（不允许混用）。

`interpolation` 必须从 `{"I1_nearest_quad", "I2_L2_projection"}` 中取值，
对应 plan §3.3 中的 $\mathcal I_1$ / $\mathcal I_2$ 两种历史场采样方案。

---

## 5. `dataset_manifest.json`

```json
{
  "schema_version": "0.1",
  "dataset_name": "m0_small",
  "created_utc": "2026-05-27T12:00:00Z",
  "git_commit": "e118cc9",
  "config_hash": "sha256:...",
  "n_samples": 200,
  "samples": [
    {"id": "sample_000000", "npz": "samples/sample_000000.npz",
     "meta": "samples/sample_000000.meta.json",
     "geometry_params": {...}, "material_params": {...}, "ok": true},
    ...
  ],
  "splits": {
    "train": ["sample_000000", ...],
    "val":   ["sample_000150", ...],
    "test":  ["sample_000180", ...]
  },
  "global_stats": {
    "stress_scale_mean": 1.0e3,
    "stress_scale_std": 2.1e2,
    "damage_max_p95": 0.98
  },
  "notes": "Generated from scripts/datasets/configs/m0_small.yaml"
}
```

`splits` 字段：**任何按 sample id 划分的 train/val/test 都必须写入 manifest**，
保证训练复现性。

---

## 6. 不变量与 sanity check

`tests/test_dataset_roundtrip.py` 应验证以下不变量：

1. **形状一致性**：`damage[t].shape == (1, H, W)`、`stress[t].shape == (3, H, W)`、
   `mask.shape == (1, H, W)`；所有时间维度长度等于 `metadata.load.N_steps`。
2. **mask 一致性**：`(inputs.mask == valid_mask).all()`；
   `mask.sum()` 与 `metadata.stats.n_inside_pixels` 一致。
3. **范围**：`damage ∈ [0, 1]`（数值 ε 容忍 1e-6）；`damage` 沿时间轴**单调不降**
   （`(damage[t+1] >= damage[t] - 1e-6).all()`）。
4. **域外置零**：`(damage * (1 - mask)).max() < 1e-6`；`stress` 在 $\Omega$ 外
   通道范数同样小于 1e-6。
5. **归一化往返**：从 npz 读 `stress`，乘 `stress_scale` 后与 recorder 原始
   $\sigma_h$ 在采样点上误差小于插值容差（由 `metadata.interpolation` 决定）。
6. **schema 版本**：`metadata.schema_version` 必须等于代码常量
   `fracturex.postprocess.dataset_export.SCHEMA_VERSION`。

---

## 7. 版本与兼容性

- **schema_version 语义版本**：major.minor。同 major 内可向后兼容地新增**可选**字段；
  major 升级表示破坏性变更（字段重命名、形状改变、强制字段增减）。
- 加载代码必须显式检查 `schema_version`，不兼容时**报错而非默默跳过**。
- 新增字段时同步更新本文档与 `tests/test_dataset_roundtrip.py`。

---

## 8. 参考

- 完整数学定义：[plan_operator_learning.md](plan_operator_learning.md) §3
- 插值误差量化：[m0_interpolation_error.md](m0_interpolation_error.md)
- **接入新模型指南（SolverAdapter 接口）**：[surrogate_porting_guide.md](surrogate_porting_guide.md)
- 实现入口：`fracturex/postprocess/dataset_export/`（包；`adapter.py` 为接口，`adapters/huzhang_phasefield.py` 为参考实现）
- Roundtrip 测试：`tests/test_dataset_roundtrip.py`
