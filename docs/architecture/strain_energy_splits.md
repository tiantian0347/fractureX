# 标准相场断裂应变能分解与标准 FEM 集成

`fracturex.phasefield` 采用“无网格分解策略 + 有限元材料适配器”的两层结构，实现论文
`ttthesis/thesis/body/phase_theory.tex` 中的应变能分解。分解策略只接收应变张量，标准材料
类负责有限元位移/相场取值、退化函数和不可逆历史场。

实现的核心目标是模块解耦：分解公式独立于网格和有限元阶次，材料适配器独立管理退化、
历史场与 Voigt 映射，`MainSolve` 只通过统一材料接口完成标准位移有限元装配。

## 已实现模型

| 工厂名称 | 分解 | 平衡方程 | 历史场驱动力 |
| --- | --- | --- | --- |
| `IsotropicModel` / `Bourdin` | 各向同性 | 全部弹性能退化 | 全部弹性能 |
| `AnisotropicModel` / `Lancioni` | 偏差–体积分解 | 偏差部分退化，体积部分保留 | 偏差能 |
| `DeviatoricModel` / `Amor` | 体积–偏应变分解 | 正体积与偏差部分退化 | 正体积与偏差能 |
| `SpectralModel` / `Miehe` | 主应变谱分解 | 正谱部分退化 | 正谱能 |
| `HybridModel` / `Ambati` | 混合 | 各向同性退化 | 正谱能 |

体积模量按空间维数定义为 `kappa = lambda + 2*mu/GD`，偏差张量按
`dev(epsilon) = epsilon - tr(epsilon)*I/GD` 计算。因此纯张量策略适用于任意 `GD >= 2`。

## 统一接口

无网格策略位于 `fracturex.phasefield.strain_energy_split`，核心接口为：

```python
from fracturex.phasefield import StrainEnergySplitFactory

split = StrainEnergySplitFactory.create("miehe", lame_lambda, shear_modulus)
psi_plus, psi_minus = split.energy_density_decomposition(strain)
sigma_plus, sigma_minus = split.stress_decomposition(strain)
C_plus, C_minus = split.tangent_decomposition(strain)
```

其中 `strain.shape == (..., GD, GD)`，能量输出为 `(...)`，应力输出与应变同形，四阶切线
输出为 `(..., GD, GD, GD, GD)`。这些计算全部通过 FEALPy `backend_manager` 执行。

标准相场材料位于 `fracturex.phasefield.phase_fracture_material`：

```python
from fracturex.phasefield import PhaseFractureMaterialFactory
from fracturex.phasefield.energy_degradation_function import EnergyDegradationFunction

material = PhaseFractureMaterialFactory.create(
    "SpectralModel",
    {"E": 210e3, "nu": 0.3},
    EnergyDegradationFunction("quadratic"),
)

stress = material.stress_from_strain(strain, phase)
D = material.elastic_matrix_from_strain(strain, phase)
H = material.maximum_historical_field_from_strain(strain)
```

有限元求解时继续使用既有入口：先调用 `update_disp(uh)` 和 `update_phase(d)`，然后把材料
传给 `LinearElasticIntegrator(..., method="voigt")`。材料的 Voigt 顺序为：

- 2D：`(xx, yy, xy)`；
- 3D：`(xx, yy, zz, xy, yz, xz)`；
- 更高维：先列法向分量，再按索引距离列剪切分量。

剪应变采用工程剪应变，剪应力采用张量分量。

材料适配器提供两组语义明确的能量接口：

- `mechanical_energy_density_decomposition(strain)`：平衡方程采用的分解；
- `strain_energy_density_decomposition(strain)`：不可逆历史场采用的裂纹驱动力分解。

除 `HybridModel` 外，两者使用同一策略。`HybridModel` 的平衡方程采用各向同性退化，历史场
采用 Miehe 正谱能量，从而在不复制有限元代码的情况下表达 Ambati 混合模型。

## 标准 `MainSolve` 集成

`MainSolve` 在 `initialize_settings` 中通过 `PhaseFractureMaterialFactory` 创建材料，标准位移子步
继续使用 FEALPy 的 `LinearElasticIntegrator(material, method="voigt")`。选择不同模型只需修改
`model_type`，边界条件、相场子步和线性求解器接口保持一致：

```python
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

from fracturex.phasefield.main_solve import MainSolve

bm.set_backend("numpy")  # 也可选择当前 FEALPy 环境支持的 pytorch 后端
mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=8, ny=8)
params = {"E": 210e3, "nu": 0.3, "Gc": 2.7, "l0": 0.02}

top = lambda p: bm.abs(p[..., 1] - 1.0) < 1.0e-12
bottom = lambda p: bm.abs(p[..., 1]) < 1.0e-12

solver = MainSolve(mesh, params, model_type="SpectralModel")
solver.add_boundary_condition(
    "force", "Dirichlet", top, [0.0, 1.0e-5], "y"
)
solver.add_boundary_condition(
    "displacement", "Dirichlet", bottom, 0.0
)
solver.solve(
    method="lfem",
    p=2,
    q=4,
    maxit=20,
    linear_solver_options={"method": "direct"},
)
```

可传给 `model_type` 的标准名称与文献别名如下：

| 标准名称 | 可用别名 |
| --- | --- |
| `IsotropicModel` | `isotropic`, `Bourdin` |
| `AnisotropicModel` | `anisotropic`, `Lancioni` |
| `DeviatoricModel` / `VolumetricDeviatoricModel` | `deviatoric`, `Amor` |
| `SpectralModel` | `spectral`, `Miehe` |
| `HybridModel` | `hybrid`, `Ambati` |

载荷路径既可传 FEALPy 后端张量，也可直接传 Python 列表或元组；`MainSolve` 会在初始化边界
条件时转换为当前后端和位移场 dtype。载荷路径是一维序列，至少包含初始状态和一个加载步。

## 多后端、维数与有限元阶次

- 分解策略仅依赖 `fealpy.backend.backend_manager`，已验证 NumPy、PyTorch、JAX；
- 纯张量能量、应力和四阶切线已验证 `GD=2/3/4`，接口接受任意 `GD >= 2`；
- 材料的 Voigt 映射与应变矩阵由 `GD` 动态生成，没有写死二维/三维分支；
- 材料适配器不保存有限元次数，标准装配已覆盖二维 `p=1/2/3` 与三维 `p=2`；
- `MainSolve` 一步交错求解已覆盖 NumPy 和 PyTorch CPU 后端；JAX 已覆盖材料系数求值。

因此，新网格、新阶次或新后端可以复用同一材料接口；具体可运行范围由对应 FEALPy 网格、
函数空间、稀疏矩阵与线性求解器后端共同决定。

## 模块位置

| 模块 | 职责 |
| --- | --- |
| `fracturex/phasefield/strain_energy_split.py` | 无网格能量、应力、切线分解与 Voigt 工具 |
| `fracturex/phasefield/phase_fracture_material.py` | 退化、历史场、有限元字段和材料工厂 |
| `fracturex/phasefield/main_solve.py` | 标准位移 FEM 与相场交错求解入口 |
| `fracturex/tests/test_phasefield_strain_energy_split.py` | 分解公式、旋转不变性、有限差分与多后端单测 |
| `fracturex/tests/test_fracture_constitutive_model.py` | 任意维/阶材料系数与标准 FEM 装配测试 |
| `fracturex/tests/test_main_solver.py` | 当前 `MainSolve` API 的 NumPy/PyTorch smoke 测试 |

## 验证

```bash
pytest -q \
  fracturex/tests/test_main_solver.py \
  fracturex/tests/test_phasefield_history_source.py \
  fracturex/tests/test_phasefield_strain_energy_split.py \
  fracturex/tests/test_fracture_constitutive_model.py \
  fracturex/tests/test_model5_standard_fem_elastic.py
```

测试覆盖能量/应力/切线重组、纯压单边响应、谱分解旋转不变性、应力–切线有限差分
一致性、历史场不可逆性、NumPy/PyTorch/JAX 后端，以及 2D/3D、`p=1/2/3` 标准有限元
Voigt 组装和五种模型各一个标准 `MainSolve` 交错载荷步。2026-08-19 的聚焦回归结果为
`86 passed`。

测试模块不导入交互式调试器，开发测试依赖仅包含 `pytest` 与 `pytest-cov`，因此执行测试时
无需额外安装 `ipdb`。
