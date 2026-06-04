# 算子学习管线接入指南：把新模型接到 SolverAdapter

> 面向：想把"仿真 → 训练数据 → 算子学习"这套管线复用到**另一个物理模型**
> （非 Hu-Zhang 相场）的开发者。
>
> 配套：数据协议 [SURROGATE_DATA_SCHEMA.md](SURROGATE_DATA_SCHEMA.md)、
> 规划 [plan_operator_learning.md](plan_operator_learning.md)。

---

## 1. 一句话原则

**接入新模型 = 在 `fracturex/postprocess/dataset_export/adapters/` 写一个
`SolverAdapter` 实现。** 导出引擎（`core` / `grid` / `sampling` / `meta`）和整个
训练侧 `fracturex/learn/` 都不需要改。

```
 仿真侧（你的模型）            协议            训练侧（通用，无需改）
 ┌────────────────┐   .npz + .meta.json   ┌──────────────────┐
 │  SolverAdapter │ ───────────────────▶ │ fracturex/learn/ │
 │  (你写这一个)   │   SURROGATE_DATA_     │ datasets/models/ │
 └────────────────┘     SCHEMA v0.1       │ losses/train     │
        ▲  delegate                        └──────────────────┘
 ┌────────────────┐
 │  export core   │  ← 通用引擎，照常调用
 └────────────────┘
```

---

## 2. 三个接口

全部在 [fracturex/postprocess/dataset_export/](../../fracturex/postprocess/dataset_export/)。

### 2.1 `SolverAdapter`（`adapter.py`）— 唯一必须实现的接缝

```python
class SolverAdapter(Protocol):
    schema_version: str                          # 须等于 core 的 SCHEMA_VERSION（当前 "0.1"）
    material_order: tuple[str, ...]              # 材料向量的字段顺序，长度 = k
    output_field_specs: tuple[FieldSpec, ...]    # 声明产出哪些输出场

    def load_discretization(self, recorder_dir: Path) -> Any: ...
    def mesh(self, discr) -> Any: ...
    def list_checkpoints(self, recorder_dir: Path) -> list[Path]: ...
    def material_vector(self, recorder_meta: dict, overrides: dict | None = None) -> np.ndarray: ...
    def evaluate_outputs(self, discr, checkpoint: dict, locator, grid) -> dict[str, np.ndarray]: ...
    def geometry_meta(self, recorder_dir: Path, recorder_meta: dict, cfg) -> dict: ...
```

| 成员 | 你要返回什么 |
| --- | --- |
| `schema_version` | 字符串，须与 `dataset_export.SCHEMA_VERSION` 一致，否则训练侧加载会报版本不符。 |
| `material_order` | 材料标量参数的固定顺序，如 `("lambda","mu","Gc","l0","eta")`。长度 `k` 自由——训练侧从数据读 `k`，不写死。 |
| `output_field_specs` | `FieldSpec` 元组，声明每个输出场的名字 / 通道数 / 归一化策略（见 §2.2）。 |
| `load_discretization(recorder_dir)` | 从一次仿真的输出目录重建"离散化句柄"`discr`（不透明对象，后面几个方法会原样传回）。通常含网格 + 有限元空间。 |
| `mesh(discr)` | 返回 `discr` 的三角网格（core 用它建像素定位器 `locator`）。须暴露 `entity("node")`/`entity("cell")`，cell 为 (NC,3)。 |
| `list_checkpoints(recorder_dir)` | 按时间顺序返回逐步 checkpoint 文件路径列表（长度 = T）。 |
| `material_vector(meta, overrides)` | 按 `material_order` 从 recorder meta 装出 `(k,)` float 向量。 |
| `evaluate_outputs(discr, checkpoint, locator, grid)` | **核心**：把**单帧** checkpoint 的各输出场求值到结构网格，返回 `{spec.name: (C, H, W)}`，且已是 **schema 通道顺序**。域外置零和归一化由 core 之后统一做。 |
| `geometry_meta(recorder_dir, meta, cfg)` | 返回写进 `<sample>.meta.json::geometry_params` 的几何描述 dict（如 `{"case":..., "domain_bbox":...}`）。 |

### 2.2 `FieldSpec`（`adapter.py`）— 声明一个输出场

```python
FieldSpec(name="damage", channels=1, scaling="none")
FieldSpec(name="stress", channels=3, scaling="stress_scale")
```

- `name`：落盘到 npz 的键，须与 schema §3.2 对齐（`damage`/`stress`/…）。
- `channels`：该场 `(C,H,W)` 的 C。
- `scaling`：
  - `"none"` — 原样存（如 d∈[0,1]）；
  - `"stress_scale"` — core 除以统一的 `stress_scale`（cfg 指定，或自动取**第一个** stress-scaled 场最后一帧、域内 95 分位），并写进 `meta.scaling.stress_scale`。

### 2.3 `Geometry`（`geometry.py`）— 描述域 Ω

只需一个有向距离函数，**Ω 内为正**：

```python
class Geometry(Protocol):
    def signed_distance(self, points: np.ndarray) -> np.ndarray: ...  # (...,2) -> (...)
```

可直接传一个 `signed_distance(points)` 的裸 callable，或用现成的
`CircularNotchDomain`，或照抄它写自己的形状。core 用它生成 `sdf`/`mask`/`coords`
三个输入通道。

---

## 3. 接入步骤

### 步骤 1：写适配器

在 `adapters/` 下新建 `my_model.py`。下面是一个**最小可照抄**示例：一个产出
单个标量场 `damage`（来自连续 Lagrange 空间）的模型。复用 `sampling.py` 里的
通用 FE 采样工具，自己只写"模型怎么重建、checkpoint 里有什么键"。

```python
# fracturex/postprocess/dataset_export/adapters/my_model.py
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..adapter import FieldSpec
from ..sampling import evaluate_lagrange_on_grid   # 通用：任意 Lagrange 标量场 → 网格


@dataclass(frozen=True)
class MyModelAdapter:
    schema_version: str = "0.1"
    material_order: tuple[str, ...] = ("kappa", "source")   # 你的材料参数
    output_field_specs: tuple[FieldSpec, ...] = (
        FieldSpec("damage", 1, "none"),                     # 单标量输出场
    )

    def load_discretization(self, recorder_dir: Path):
        # 从 recorder 重建网格 + 空间，返回任意句柄（这里假设有个 build 函数）
        from my_pkg.discretization import rebuild_from_dir
        return rebuild_from_dir(Path(recorder_dir))

    def mesh(self, discr):
        return discr.mesh                                    # 须支持 entity("node"/"cell")

    def list_checkpoints(self, recorder_dir: Path) -> list[Path]:
        return sorted((Path(recorder_dir) / "checkpoints").glob("step_*.npz"))

    def material_vector(self, recorder_meta: dict, overrides=None) -> np.ndarray:
        m = dict(recorder_meta.get("material") or {})
        if overrides:
            m.update(overrides)
        return np.array([float(m[k]) for k in self.material_order], dtype=np.float64)

    def evaluate_outputs(self, discr, checkpoint: dict, locator, grid) -> dict:
        d_dofs = np.asarray(checkpoint["d"])                 # 你的 checkpoint 键
        damage = evaluate_lagrange_on_grid(discr.space_d, d_dofs, locator)  # (1,H,W)
        return {"damage": damage}                            # 名字须与 FieldSpec 一致

    def geometry_meta(self, recorder_dir, recorder_meta, cfg) -> dict:
        return {"case": recorder_meta.get("case", "my_model"),
                "domain_bbox": [list(cfg.grid.bbox[0]), list(cfg.grid.bbox[1])]}
```

**多通道 / 张量场**（如应力）：把它加进 `output_field_specs`，在
`evaluate_outputs` 里返回 `(C,H,W)` 并**自己负责通道顺序**对齐 schema。如果你的
基函数不是 Lagrange（如 Hu-Zhang、RT、BDM），就在适配器里写对应的逐点求值
（参考 `adapters/huzhang_phasefield.py::evaluate_huzhang_on_grid`）。

> 可复用的通用工具（都在 `sampling.py`，与模型无关）：
> - `build_pixel_locator(mesh, grid)` / `group_pixels_by_cell(locator)`
> - `evaluate_lagrange_on_grid(space, dofs, locator)` — 任意 Lagrange 标量场
> - `sample_field_nearest_quad(...)` / `sample_field_l2_projection(...)` — 𝓘₁ / 𝓘₂ 采样
> - `grid.compute_sdf / compute_valid_mask / compute_coords` — 输入通道

### 步骤 2：导出样本

```python
from fracturex.postprocess.dataset_export.core import (
    export_recorder_to_sample, ExportConfig,
)
from fracturex.postprocess.dataset_export import GridSpec, CircularNotchDomain
from fracturex.postprocess.dataset_export.adapters.my_model import MyModelAdapter

cfg = ExportConfig(grid=GridSpec(H=64, W=64, bbox=((0,1),(0,1))))
geom = CircularNotchDomain(box=(0,1,0,1), cx=0.5, cy=0.5, r=0.2)  # 或你自己的 Geometry

export_recorder_to_sample(
    recorder_dir, out_npz, out_meta, cfg, geom,
    adapter=MyModelAdapter(),      # ← 只有这一处指明了你的模型
    sample_id="sample_000000",
)
```

> 兼容提示：包顶层 `from fracturex.postprocess.dataset_export import export_recorder_to_sample`
> 保留的是**旧位置参数签名** `(recorder_dir, out_npz, out_meta, cfg, discr, geometry, ...)`
> （discr 在前），给历史调用用。**新模型请用上面 `core.export_recorder_to_sample`
> 的签名**（geometry 在前，`adapter=`/`discr=` 走关键字），更清晰。`discr` 不传时
> 由 `adapter.load_discretization(recorder_dir)` 自动重建。

### 步骤 3：训练侧——通常零改动

`fracturex/learn/` 只读 schema，自动适配：
- `PhaseFieldOperatorDataset` 读 `dataset_manifest.json` 的 `splits` + 各样本 npz/meta；
- 输入通道由 `assemble_input_channels` 拼装：`sdf(1)+mask(1)+coords(2)+material(k)`，
  **k 从数据读，不写死**；
- 三 baseline 走 `models.build_model(name, in_ch, T, grid_hw=...)` 统一接口，
  `forward(x): (B,C_in,H,W) → (B,T,H,W)`。

只有当你**新增输出场监督**（如 M2 引入 σ）时才动训练侧：
1. 在 `models/__init__.py::build_model` 注册新模型（这是扩展点）；
2. 若要把新场喂进损失，在 `datasets.py` 扩展通道拼装 / 在 `losses.py` 加项；
3. 评估指标加在 `eval/metrics.py`。

---

## 4. 必须满足的协议不变量（schema §6）

`evaluate_outputs` 产出会被 core 统一做域外置零 + 归一化，但你仍要保证：

1. **形状**：每帧每场 `(C, H, W)`，C 与 `FieldSpec.channels` 一致；T = `len(list_checkpoints)`。
2. **通道顺序**：`evaluate_outputs` 返回的通道已是 schema 顺序（如 stress = `[xx, yy, xy]`）。
3. **damage 单调不降**（若是相场类）、范围 `[0,1]`。
4. **往返归一化**：`stress * stress_scale ≈ 原始 σ_h`（core 自动记录 `stress_scale`）。
5. **mask**：core 用 `geometry` 生成 `mask`/`valid_mask`，两者等价；域外输出置零由 core 做。
6. **schema_version** 与 core 常量一致。

---

## 5. 验证（照抄 roundtrip 模式）

参考 [fracturex/tests/test_dataset_roundtrip.py](../../fracturex/tests/test_dataset_roundtrip.py)：
造一个迷你 `discr` + 几个伪 checkpoint，跑 `export_recorder_to_sample(..., adapter=YourAdapter())`，
断言 §4 的形状 / dtype / mask / 单调性 / 域外置零 / 必需 meta 字段。

训练侧端到端可参考 [fracturex/tests/test_learn_m1_smoke.py](../../fracturex/tests/test_learn_m1_smoke.py)：
直接造 schema npz（不跑求解器）→ 1 epoch 训练 → 断言产物齐全。

自检"接缝有效"：

```bash
# 训练侧对求解器零依赖
grep -rn -E "import fealpy|from fealpy|import fracturex|from fracturex" fracturex/learn/   # 应为空
# 模型特化只在你的 adapter 里
grep -rln -i "your_model_keyword" fracturex/postprocess/dataset_export/                    # 仅命中 adapters/your_model.py
```

---

## 6. 参考实现

[adapters/huzhang_phasefield.py](../../fracturex/postprocess/dataset_export/adapters/huzhang_phasefield.py)
是完整的参考：Hu-Zhang 应力空间逐点求值、Voigt `[xx,xy,yy]→[xx,yy,xy]` 重排、
k=5 相场材料别名表、从 `mesh.npz` 重建离散化。照它的结构写你自己的模型即可。
