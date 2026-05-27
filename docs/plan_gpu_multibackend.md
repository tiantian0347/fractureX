# 规划：GPU 加速 + 多后端相场仿真（工程论文路线）

> 状态：草案 v0.1（2026-05-27）。本文件用于指导后续 1–3 个月的工程化推进，目标刊物：
> **Advances in Engineering Software** / **SoftwareX** / **Computer Physics Communications**。

## 0. 一句话定位

把 `fracturex` 现有的 NumPy + Hu-Zhang 相场代码，系统地补齐 **JAX / PyTorch / GPU** 后端，建立一套
**可复现的相场 benchmark 套件**，对外发布 speedup 曲线与可重跑脚本。卖点：**工程价值高、审稿友好、可
被外部团队直接拿来评测**。

---

## 1. 现状盘点（决定我们从哪里改起）

下面只列与本路线强相关的现状，避免重复 `docs/huzhang_phasefield_architecture.md` 的内容。

- **后端抽象已就位但只走 NumPy**
  - 所有装配与状态都通过 `from fealpy.backend import backend_manager as bm` 取数（见
    `fracturex/assemblers/huzhang_elastic_assembler.py:13`、
    `fracturex/assemblers/phasefield_assembler.py:13`）。
  - 但 hot path 上仍然大量 `bm.to_numpy(...).astype(np.float64)` 把张量拉回 NumPy 再喂
    `scipy.sparse.coo_matrix`（例如 `phasefield_assembler.py:322–326`、
    `huzhang_elastic_assembler.py:109–111`）。这是当前**后端无关性的最大泄漏点**。
- **CPU 端单元并行装配已落地**
  - `FRACTUREX_ASSEMBLY_PARALLEL`、`FRACTUREX_ASSEMBLY_NPROC`、`FRACTUREX_PROFILE_HISTORY_UPDATE`、
    `FRACTUREX_HISTORY_UPDATE_PARALLEL` 已实现（见 `huzhang_phasefield_architecture.md` §7.1）。
  - 实现方式是 ThreadPool + 单元分块 einsum。GPU 友好度不高，因为：
    - 任务粒度仍是"按 Python 线程喂 chunk"；
    - 多处显式做了 `to_numpy → coo_matrix`，无法保留在 device 内存。
- **线性求解器层耦合 SciPy 较深**
  - `fracturex/utilfuc/linear_solvers.py`、`fracturex/utilfuc/sparse_direct_backends.py`、以及测试脚本
    `fracturex/tests/phasefield_model0_huzhang.py:48–80` 都直接调 `scipy.sparse.linalg.gmres / spsolve`。
  - 多后端论文里，求解器侧暂时不强求统一，但需要**显式声明"装配 GPU + 求解 CPU"是当前栈的边界**。
- **可执行 benchmark 已经有雏形**
  - `fracturex/tests/summarize_phasefield_timing.py`（已有汇总脚本）
  - `fracturex/tests/phasefield_model0_huzhang.py`（Model0 圆缺口完整流程，含 `RunRecorder` 输出）
  - `fracturex/tests/phasefield_square_tension.py` / `phasefield_model2_notch_shear_huzhang.py`
  - 结果目录 `results/PAPER_INDEX.{json,md}` 已经做了"论文索引"骨架。

> **结论**：我们不缺主流程，缺的是把 NumPy 隐式假设清理干净 + 给 JAX 端一条 jit/vmap 的真正路径 + 把
> benchmark 套件标准化。

---

## 2. Milestone 拆解

每个 M 都设计成"可写一段实验章节"的粒度，里程碑之间互相独立、可中断。

### M0：基础设施清理（1–2 周）

**目标**：让任何一个 hot path 的代码可以在 `bm.set_backend("jax")` 下跑完，不抛异常、不偷偷退回 NumPy。

- 在 `fracturex/utilfuc/utils.py` 新增 `backend_context.py`（或扩展现有 `utils.py`），统一封装：
  - `set_backend(name, device=None)` —— 包一层 `bm.set_backend` + 记录当前 device；
  - `to_host(x)` / `to_device(x, device)` —— 收口所有"装配完要交给 SciPy"的位置；
  - `assert_no_host_roundtrip(...)` —— debug 开关，扫描函数体内是否出现 `bm.to_numpy` 调用栈。
- 在 `assemblers/phasefield_assembler.py` 与 `assemblers/huzhang_elastic_assembler.py` 中
  把 `to_numpy + coo_matrix` 的入口集中成一个 `_finalize_local_to_global(I, J, V)` 函数：
  - NumPy / Torch CPU 后端 → 直接 `coo_matrix` 然后 `.tocsr()`；
  - JAX 后端 → 暂时也走 host 路径，但**留接口**给 `jax.experimental.sparse` 或自写的 CSR builder。
- 写 `fracturex/tests/test_backend_smoke.py`：参数化 `["numpy", "jax", "pytorch"]`，对一个 3×3 网格
  跑一步 Hu-Zhang + 相场，断言结果数值与 NumPy 一致（容差 1e-10）。
- 修补 `bm.einsum` 在不同后端下的小坑（typical：`bm.broadcast_to` 在 JAX 下返回不可变数组，触发
  下游 `reshape(-1)` 时无问题，但 `bm.set_at` 等就地写法需要替换）。

**交付物**：CI 多后端 smoke 通过 + 一份 `docs/BACKEND_BOUNDARY.md` 描述目前哪些点仍是 NumPy 专属。

### M1：JAX 端 jit + vmap 装配（2–3 周）

**目标**：把相场 `A_const / A_hist / RHS` 与 Hu-Zhang `M(d) / B / B2` 的核心 einsum 改成 jit 友好版本，
benchmark 上拿到对 NumPy 的实质加速。

- 在 `fracturex/assemblers/_kernels/` 下新建 `kernels_jax.py`，把现有 chunk 函数翻成纯函数：
  - 来源参考：`phasefield_assembler.py` 中 `_assemble_phase_mass_chunk` / `_assemble_phase_diff_chunk` /
    历史驱动相关函数；
  - Hu-Zhang 侧参考：`huzhang_elastic_assembler.py` 中 `_b2_gradg_K_chunk`、`_md_chunk_mul` 等；
- 关键技巧：
  - 用 `jax.vmap` 替掉单元循环，再外层 `jax.jit` 静态化形状；
  - 历史变量 `H` 的 max 投影用 `jax.lax.scan` 还是 `jnp.maximum` 直接做要做一次对比（前者对 batch 大
    步长更稳）；
  - 把 cell→DOF 的 scatter 用 `jax.ops.segment_sum` 替掉 `coo_matrix` 路径，构造 device 端 CSR；
- 在 `assemblers/_dispatch.py` 引入"按当前后端选 kernel"的轻量分发表，避免在 hot path 里写
  `if bm.backend_name() == "jax"`。
- 装配出来的稀疏矩阵传给 CPU 求解器之前再 `to_host`，并把这次拷贝**计入 benchmark 表的一列**，让
  审稿人能看出 device→host 的代价占比。

**交付物**：在 `fracturex/tests/bench_assembly.py` 中给出 NumPy / Torch-CPU / Torch-CUDA / JAX-CPU /
JAX-CUDA 的装配墙钟对比，至少覆盖 2 个网格规模（约 5e4 / 5e5 DOF）。

### M2：相场 GPU benchmark 套件（2 周）

**目标**：把已经有雏形的算例（Model0、Square Tension、Notch Shear）整理成"零配置可复现"的 benchmark。

- 在 `fracturex/tests/bench_phasefield/` 下新建：
  - `cases.yaml`：算例 ID → mesh / 材料 / 加载 / 容差 / 期望反力曲线 hash。
  - `runner.py`：读取 yaml，按 `--backend {numpy,jax,torch}` + `--device {cpu,cuda}` 跑完整加载历史，
    复用 `fracturex/postprocess/recorder.py` 输出 `meta.json + history.csv`。
  - `plot.py`：吃多个 `RunRecorder` 输出目录，画 (a) 反力-位移曲线对齐图、(b) 装配/求解墙钟 stacked bar、
    (c) GPU memory peak 折线。
- 选三个数据规模做 scaling 曲线：~2e4、~1e5、~5e5 DOF（与 §M1 重合一组用于交叉验证）。
- 在 `results/PAPER_INDEX.md` 中新增"§GPU 多后端 benchmark"小节，登记每次跑出来的 commit / 设备 /
  耗时摘要，避免论文写作时翻不回 raw data。
- 起步时复用：`fracturex/tests/summarize_phasefield_timing.py` 已经有"扫描结果目录→汇总耗时"的逻辑，
  可作为 `plot.py` 的骨架。

**交付物**：一条命令 `python -m fracturex.tests.bench_phasefield.runner --suite all` 完成全部跑批；
一个 `bench_phasefield/REPORT.md` 自动生成。

### M3：scalability 与论文实验（2–3 周）

**目标**：拿到能写进论文 Section "Performance" 的图表。

- **strong scaling**：固定 5e5 DOF，扫 CPU 线程数 {1,2,4,8,16,32}；GPU 上扫 batch chunk 大小
  {32,64,128,...}。
- **weak scaling**：DOF/线程 比例固定，画规模随线程数变化的总耗时。
- **后端对比表**：NumPy / JAX-CPU / JAX-CUDA / Torch-CPU / Torch-CUDA，每个 cell 给"装配 / 历史更新 /
  线性求解 / 端到端"四列。
- **数值正确性表**：所有后端在所有算例上的反力-位移误差 ≤ 1e-8（与 NumPy reference 对照）。
- 选一个反例：把"完全不写 GPU 友好"的 baseline 留一组（直接关掉 §M1 的 jit），让审稿人看到优化贡献。
- 论文实验脚本统一放在 `scripts/paper_gpu_multibackend/`（与已有的 `scripts/paper_huzhang/` 风格一致）。

**交付物**：论文用图表 PDF + 原始 CSV + 复现脚本，全部进 `results/paper_gpu_multibackend/`。

### M4：论文写作与开源发布（2 周）

- 论文骨架（建议）：
  1. Introduction — 相场仿真的算力痛点，已有开源软件的后端覆盖现状。
  2. Architecture — `fracturex` 三条技术路线、后端抽象、Hu-Zhang 块系统。
  3. GPU-aware assembly — jit/vmap 改造细节、与 segment_sum 的对接。
  4. Benchmark suite — 算例描述与可复现协议。
  5. Performance — §M3 全部图表。
  6. Limitations — 求解器仍在 CPU；MPI 分布式不在本文范围。
- 发布：
  - 给 benchmark suite 一个独立的 git tag，比如 `bench-v0.1`；
  - 在 README 中加入"如何在自己的机器上复现 §5 图表"的 3 步说明；
  - 若投 SoftwareX，准备 metadata 文件与 "Software impact" 段落。

---

## 3. 起步可以直接动的文件清单

下表按"先改这里、再扩展那里"的顺序排，每一项后面写为什么动它。

| 文件 | 改什么 | 为什么 |
| --- | --- | --- |
| `fracturex/utilfuc/utils.py` | 新增 `set_backend / to_host / to_device` | 收口所有 backend 切换 |
| `fracturex/assemblers/phasefield_assembler.py:55–326` 区域 | 抽出 `_finalize_local_to_global` | 这里 `to_numpy + coo_matrix` 重复出现 |
| `fracturex/assemblers/huzhang_elastic_assembler.py:101–222` | 同上 | Hu-Zhang 块装配的 host roundtrip |
| `fracturex/assemblers/_kernels/kernels_jax.py`（新增） | 装 jit/vmap 版 einsum 核 | §M1 的实际加速来源 |
| `fracturex/assemblers/_dispatch.py`（新增） | 后端 → kernel 路由 | 避免在 hot path 写 if |
| `fracturex/tests/bench_phasefield/`（新增目录） | 标准 benchmark suite | §M2 的对外交付 |
| `fracturex/tests/test_backend_smoke.py`（新增） | 多后端一致性 | §M0 的安全网 |
| `fracturex/tests/summarize_phasefield_timing.py` | 扩展支持 backend 维度 | 重用现有汇总能力 |
| `fracturex/postprocess/recorder.py` | `meta.json` 加 `backend / device / commit` 字段 | benchmark 可追溯 |
| `scripts/paper_gpu_multibackend/`（新增） | 跑批与画图脚本 | §M3 / §M4 论文素材 |
| `docs/BACKEND_BOUNDARY.md`（新增） | 标注当前仍为 NumPy 专属的位置 | 防止后续静默回退 |

不动的部分（避免无谓 churn）：

- `fracturex/utilfuc/linear_solvers.py` 与 `sparse_direct_backends.py`：本文范围内**只把求解器作为
  CPU 侧黑盒**，不重写。
- `fracturex/drivers/*`：流程编排无需大改，只需要在 driver 里读 `backend / device` 元信息以便
  recorder 记录。

---

## 4. 文献清单（按写作章节分组）

**相场断裂方法（Background）**
- Bourdin, Francfort, Marigo (2000). *Numerical experiments in revisited brittle fracture*. JMPS.
- Miehe, Hofacker, Welschinger (2010). *A phase field model for rate-independent crack propagation*. CMAME.
- Miehe, Welschinger, Hofacker (2010). *Thermodynamically consistent phase-field models of fracture*. IJNME.
- Ambrosio, Tortorelli (1990/1992). *Approximation of functionals depending on jumps by elliptic functionals*. CPAM.
- Wu, Nguyen, Nguyen-Xuan, Sutula, Sinaie, Bordas (2020). *Phase-field modeling of fracture* (Advances in Applied Mechanics review).

**Hu-Zhang 混合元 / 应力杂交（Method foundations）**
- Hu, Zhang (2014/2015). 关于对称应力张量混合有限元的系列论文（SIAM J. Numer. Anal.）。
- Chen, Hu, Huang (2017). 关于 Hu-Zhang 元的辅助空间预条件（你已经在 `linear_solvers.py` 中实现了
  `auxspace` 路径，需要给出引用）。

**GPU / 多后端有限元（核心对照组）**
- Bradbury et al. (2018). *JAX: composable transformations of Python+NumPy programs*. 官方白皮书 / 文档。
- Paszke et al. (2019). *PyTorch: An imperative style, high-performance deep learning library*. NeurIPS.
- Kjolstad, Kamil, Chou, Lugato, Amarasinghe (2017). *The Tensor Algebra Compiler*. OOPSLA.（GPU einsum/
  scatter 优化的常被引基线）
- Markidis et al. (2018). *NVIDIA Tensor Core programmability, performance & precision*.（如果会用到 fp32/fp16 对比）
- Häfner, Vicentini (2021). *Veros: A python-based ocean simulator*.（多后端架构对照案例）
- Bezgin, Buhendwa, Adams (2023). *JAX-Fluids*. CPC.（同等 JAX-加速 PDE solver 的工程文章范式，对
  SoftwareX/CPC 投稿模板特别值得对照）
- Häfner et al. (2024+). 任何 JAX-FEM / FEniCS-JAX 类型的最新工作（投稿前一周扫一遍 arXiv 更新）。

**Benchmark / 软件论文范式**
- Logg, Mardal, Wells (eds.) (2012). *Automated Solution of Differential Equations by the Finite Element Method*.（FEniCS 书，benchmark 写法可借）
- Ahrens, Geveci, Law (2005). *ParaView: an end-user tool for large-data visualization*.
- 任何近 3 年 SoftwareX 的相场或 FEM 文章 1–2 篇，用来对照 metadata / 复现说明结构。

**审稿人会问的"周边"**
- 关于 `segment_sum` / `scatter` 在 GPU 上的 race condition：JAX docs `jax.ops.segment_sum` 章节 +
  CUDA atomicAdd 相关 NVIDIA 技术博客。
- 关于"GPU 装配 + CPU 求解"的代价分摊：找 1–2 篇 PETSc/Trilinos 团队的近年工作做对照。

---

## 5. 风险与对策

- **风险：JAX 后端在 FEALPy 里覆盖不全** → M0 阶段先列出未实现操作清单（典型：`bm.set_at` / 复杂索引），
  必要时给 FEALPy 提 PR 或在 fracturex 侧做 thin shim。
- **风险：稀疏组装在 GPU 上反而比 CPU 慢** → 提前在 M1 跑一组只装配不求解的 micro-benchmark，如果
  segment_sum 路径不达预期，回退到"GPU 端只算 local stiffness，host 端 scatter"的混合方案，并在论文
  中如实展示对比。
- **风险：benchmark suite 维护成本高** → §M2 的 yaml 设计要保证"新增一个算例 = 加一段 yaml"，禁止
  benchmark 脚本里写 if-case。
- **风险：与 §HuZhang 论文路线抢工** → 本路线刻意只动 assembler 与 utilfuc，不碰 driver/discretization，
  保持与 Hu-Zhang 论文的 PR 平行可合并。

---

## 6. 立即可执行的下一步（如果今天就要动手）

1. `git checkout -b feat/multibackend-cleanup`
2. 在 `fracturex/utilfuc/utils.py` 加 `set_backend / to_host / to_device` 三个函数（含 docstring）。
3. 把 `phasefield_assembler.py:319–326` 与 `huzhang_elastic_assembler.py:105–111` 的 host roundtrip
   抽成 `_finalize_local_to_global`，先保持行为不变。
4. 写 `fracturex/tests/test_backend_smoke.py`，跑 numpy + jax + torch 三个后端的 3×3 网格一致性。
5. CI（或本地 `pytest -q`）通过后即可开 PR，作为后续 §M1 工作的基础。
