# FEALPy Agent Pack（可复制到任意新项目）

把本文件和 [​.github/copilot-instructions.md](.github/copilot-instructions.md) 一起复制到你的新工程根目录，可让智能体快速理解如何基于 FEALPy 搭程序。

## 你要先告诉智能体的最小上下文

- 本项目依赖 `fealpy`，核心流程是：网格 `mesh` → 空间 `functionspace` → 组装 `fem` → 求解 `solver`。
- 代码应优先使用 `from fealpy.backend import backend_manager as bm`，避免直接绑定 NumPy。
- 如需跨后端一致性，测试至少覆盖 `numpy/pytorch/jax` 中可用后端。

## 推荐让智能体执行的任务顺序

1. 先生成项目骨架：`src/`、`test/`、`examples/`。
2. 先写最小可跑通案例：`TriangleMesh` + `LagrangeFESpace` + `BilinearForm` + `DirichletBC`。
3. 再拆分为：PDE 定义、离散组装、边界处理、求解器配置、后处理。
4. 最后补参数化测试与误差验证。

## FEALPy 风格约定（高优先级）

- 命名：`NN/NE/NF/NC`、`GD/TD`、实体名 `node/edge/face/cell`。
- 张量创建保持上下文：优先 `bm.context(...)` + `bm.zeros/bm.arange/bm.tensor`。
- 更新数组优先 `bm.set_at`、`bm.index_add`，少用原地写法。

## 让智能体直接开工的提示词模板

请基于 FEALPy 为我搭建一个【问题类型】求解程序，要求：

- 输入：几何与边界条件参数化；
- 离散：`TriangleMesh` + `LagrangeFESpace(p=1/2)`；
- 组装：`BilinearForm`/`LinearForm`，包含【扩散/质量/对流】项；
- 边界：`DirichletBC`（必要时加 Neumann/Robin）；
- 求解：先给直接法，再给迭代法可切换；
- 输出：误差、收敛阶、VTK 导出；
- 测试：最少 1 个可重复单测。
并按“先最小可运行、再模块化重构”的方式提交。

## 参考入口

- 主说明： [README.md](README.md)
- Hu-Zhang 混合元 + 相场架构（与纯 FEALPy 对照、中英双文）： [docs/architecture/huzhang_phasefield_architecture.md](docs/architecture/huzhang_phasefield_architecture.md) / [docs/architecture/huzhang_phasefield_architecture.en.md](docs/architecture/huzhang_phasefield_architecture.en.md)（结构变更后跑 `python3 scripts/verify_huzhang_docs.py`）
- AI 指南： [​.github/copilot-instructions.md](.github/copilot-instructions.md)
- 命名规范： [Develop_note.md](Develop_note.md)
- 典型测试： [test/mesh/test_triangle_mesh.py](test/mesh/test_triangle_mesh.py)
