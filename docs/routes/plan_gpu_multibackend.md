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

---

## 7. Matrix-free 弹性应力块（③：内存 + GPU 前置）— 2026-06-02

> 背景：受控扫描定论 2D 中等规模 aux 在时间(11–14×)与内存(≈持平)都输 pardiso，且 14× = ~2×线程 + ~6×算法硬墙（见记忆 `aux_loses_to_pardiso_2d`）。辅助空间预条件子的真正主场是 **3D / GPU / matrix-free**。本节实现 ③ 的第一步：把 Hu–Zhang 弹性系统中内存大头**应力块 M2 做成 matrix-free**（不显式存 M2 与 (ldof,ldof) 元核 Phi），后端无关，作为上 GPU（④）的前置。

### 7.1 数学理论

每个交错弹性子步求解鞍点系统（standard 公式，σ∈Hu–Zhang H(div,sym)，p=3）：

$$A\begin{bmatrix}\sigma\\u\end{bmatrix}=\begin{bmatrix}M_2 & B_2\\ B_2^\top & 0\end{bmatrix}\begin{bmatrix}\sigma\\u\end{bmatrix},\quad M_2=\mathrm{TM}^\top M(d)\,\mathrm{TM},\ \ B_2=\mathrm{TM}^\top B$$

其中 TM 为角点松弛变换、$B$ 为离散散度、$M(d)$ 为退化加权应力质量阵（系数 $1/g(d)$）。**只有 $M(d)$ 随 $d$ 变**；$B_2,\mathrm{TM}$ 与 $d$ 无关。

**关键观察**：迭代解（GMRES）只需 $A$ 的 **matvec**，不需显式矩阵；预条件子需要的是 $B_2$（小）、Schur $S=B_2\,\mathrm{diag}(M_2)^{-1}B_2^\top$（u 空间，小）与 $\mathrm{diag}(M_2)$。故只把 $M_2$ 的作用 matrix-free，$B_2/\mathrm{TM}/S$ 仍装配。

$M_2$ 作用 $y=\mathrm{TM}^\top M(d)\,(\mathrm{TM}\,x)$，其中 $M(d)$ 的单元作用**不形成任何 $(ldof,ldof)$ 块**，直接与输入向量收缩。设单元基 $\phi_{cqld}$（$d$ 为对称分量，nsym=3）、迹 $\mathrm{tr}\phi_{cql}$、权 $W_{cq}=|K_c|\,w_q$、系数 $\kappa_{cq}=1/g(d)$、对称重数 $\nu_d$、本构常数 $c_0=1/2\mu,\ c_1=\lambda/[2\mu(2\mu+n\lambda)]$：

$$s_{cqd}=\sum_m\phi_{cqmd}x^{loc}_{cm},\quad t_{cq}=\sum_m\mathrm{tr}\phi_{cqm}x^{loc}_{cm}$$
$$y^{loc}_{cl}=\sum_q W_{cq}\kappa_{cq}\Big(c_0\sum_d\nu_d\,\phi_{cqld}\,s_{cqd}-c_1\,\mathrm{tr}\phi_{cql}\,t_{cq}\Big)$$

再按 `cell2dof` 散射累加。这与 `HuZhangStressIntegrator` 装配的元贡献是同一收缩，只是与向量缩并而非组装全局阵。

- **diag(M2) 近似**：$\mathrm{diag}(M)$ 精确元素级算出（$Ke[c,l,l]$ 散射），再 $\mathrm{diag}(M_2)\approx(\mathrm{TM}\!\circ\!\mathrm{TM})^\top\mathrm{diag}(M)$。仅喂预条件子 $D^{-1}$ → 误差只影响 niter，不影响解。**实测在 model0 角点松弛下该近似几乎精确**（rel 3e-16）。
- **本质边界（model0 有 σ 本质 BC）**：装配路做 $A\!\leftarrow\!T A T+T_{bd},\ F\!\leftarrow\!F-A u_{bd}$。matrix-free 等价实现为算子内 mask：$A_{\text{eff}}z=T(A(Tz))+T_{bd}z$，RHS 用一次 unmasked $A\,u_{bd}$ 生成（与装配路逐位一致）。

### 7.2 实现

- 新模块 [`fracturex/utilfuc/matfree_elastic.py`](../../fracturex/utilfuc/matfree_elastic.py)：`MatrixFreeElasticOperator(scipy LinearOperator)`，携带 $B_2/\mathrm{TM}$、元核（W/coef/num/cell2dof + phi 或 space+bcs）、近似 diag、可选本质 mask；暴露 `.diagonal()`、`.B_div/.B_div_T`、`.diag_inv_sigma()`。
- 两种内存模式：**cache**（存 $\phi(NC,NQ,ldof,3)$，比装配路的 $(ldof,ldof)$ Phi+M2 省，但 $\phi$ 仍不小）；**recompute（分块）**（不存 $\phi$，每 matvec 按 cell-chunk 经 `space.basis(bcs,index=chunk)` 重算，瞬态内存有界——真省内存，CPU 更慢，GPU 兑现速度）。
- env 旋钮：`FRACTUREX_ELASTIC_MATFREE=1`、`FRACTUREX_ELASTIC_MATFREE_RECOMPUTE=1`、`FRACTUREX_MATFREE_CHUNK`。仅 standard + iterative(fast/aux) 路径生效；direct/baseline 仍装配。
- 求解器接缝（[`linear_solvers.py`](../../fracturex/utilfuc/linear_solvers.py)）duck-type：`as_scipy_csr` 早返回算子；`fast`/`auxspace` 用算子提供的 $B$/diag 替代切片，下游 Schur/GS 不变（装配路零改动）。
- 顺带优化：`_prepare_constant_blocks` 的 `M2_const` 只在 effective_stress 用，standard 下白建（含三重积大瞬态）→ 改为按需建，**所有路径免费降峰值**。

### 7.3 测试过程

1. **单元 matvec 正确性**：同 state 下装配 $A_{ref}$ 与 matrix-free $A_{mf}$，对随机向量比 $\|A_{ref}x-A_{mf}x\|/\|A_{ref}x\|$ 与 $\|F_{ref}-F_{mf}\|$，覆盖 intact / half-cracked 两种损伤、走完整 `assemble()`（触发本质边界 mask），cache 与 recompute（chunk=256/4096）各测。
2. **端到端**：model0 h1 fast，MATFREE=0 vs 1（cache 与 recompute），逐步比 max_d/反力/niter。
3. **峰值内存**：h3（184k σ-dof）独立进程测 `ru_maxrss`，对比 assembled(默认缓存Phi) / assembled(关Phi缓存) / mf_cache / mf_recompute。

### 7.4 测试结果

**正确性**（机器精度，含本质边界）：

| 项 | cache | recompute |
|---|---|---|
| $\|A_{ref}x-A_{mf}x\|$rel | 3.4e-16 | 3.4e-16（chunk 无关）|
| $\|F\|$rel | 0 | 0 |
| diag 近似 rel | 3e-16 | 3e-16 |
| 端到端 max_d 逐步 Δ | ≤4e-15 | ≤4e-15 |
| niter_elastic vs 装配 | 不变(6–7) | 不变(6–7) |

**峰值内存**（h3, 184k σ-dof）：

| 模式 | peak RSS | vs 精简装配 | vs 生产默认 |
|---|---|---|---|
| assembled（缓存Phi，生产默认，与扫描 5563MB 吻合）| 5558 MB | — | 1× |
| assembled（关Phi缓存，精简）| 1374 MB | 1× | 4.0× |
| mf_cache | 819 MB | 1.7× | 6.8× |
| **mf_recompute** | **717 MB** | **1.9×** | **7.8×** |

`_prepare_constant_blocks` 单独峰值 719MB——**mf_recompute(717MB) 几乎贴着常量块装配底**，即在常量块之上几乎不加内存。

**速度**：CPU 上 matrix-free 单次弹性解约慢 ~18×（h1 step5: ~57s vs 装配 ~3s），因每次 GMRES 迭代重算元收缩；**这是预期代价，速度收益在 GPU 兑现**。

### 7.5 内存底分析（2026-06-02 追加）

为判断"再降内存"是否值得，逐步测 h3 峰值贡献：

| 步骤 | peak RSS | current |
|---|---|---|
| mesh + discr.build | 327 | 326 |
| **B 的 BilinearForm 装配** | **719**（瞬态 +392MB）| 478 |
| B→csr / B2=TMᵀB | 719 | 479 |

**结论：MF 的 717MB 峰值底完全由 `_prepare_constant_blocks` 里 B 的一次性 BilinearForm 装配瞬态(+392MB)设定**，与 MF 算子内部存储无关。推论：
- $B_2/\mathrm{TM}$ 单副本只省**持久** ~40MB，**不动 717 峰值**（峰值是装配那一刻的瞬态）。
- $B_2$ **不能完全 matrix-free**：Schur $S=B_2\,\mathrm{diag}^{-1}B_2^\top$ 与其上的 GS 都需 $B_2$ 显式。
- 要降 717 须**分块装配 B**，但仓库现有 `_assemble_huzhang_mix_coupled_chunk` 调用的 `assembly_cell_matrix` 已随 fealpy API 失效（死代码）；重写版本兼容的 chunked mix 装配为中等工作量、且 fealpy 版本敏感，收益仅 CPU、1.9×→~2.7×。
- MF **真正的内存赢已落袋**：它避开 M2/A 构建（lean-assembled 的 1374MB 峰值正是 M2/A），这就是 1.9×/7.8× 的来源；剩下的 B 瞬态是装配路也付的公共底。

**判定：CPU 上的内存 floor-lowering 性价比低且被失效 helper 挡住，不再深挖。**

### 7.6 下一步与定位（2026-06-02 修订）

> **论文定位（专家建议）**：不卖计算效率（墙钟），卖**迭代稳定性 + 小内存**。matrix-free 在本论文中作为**内存支线**（与辅助空间预条件子的"迭代稳定"主线互补）；详见 `docs/preconditioner/D12_PRECONDITIONER_PAPER_PLAN.md` §13.4。**GPU 端口暂缓，本节作为 future work 记录**——CPU 上 matrix-free 的内存收益(1.9×/7.8×)与 niter 不变性已足以支撑内存支线论点，无需 GPU 即可成文。

- ✅ ③ 在 CPU 上已达有用极限：matrix-free 算子正确（机器精度）、内存 1.9×/7.8×、niter 与装配版不变、M2_const 跳过为免费收益。
- ⏸ **GPU 端口（④）= future work（暂缓）**：上 GPU 后元基重算廉价、带宽足，B 装配/元收缩留 device，可消解 CPU 时间(~18×)劣势与 B 装配瞬态——但**这是"双赢加速"的增量，不是当前论点所必需**。GPU 集群可用，待主线（迭代稳定 + 内存）成文后再推。
- 后续：扩 square/model2 与 effective_stress；CPU 降底（chunked B 装配）性价比低，暂不做。

> 交叉引用：aux-vs-direct 定位见 `docs/preconditioner/D12_PRECONDITIONER_PAPER_PLAN.md`；2D 失利定论见会话记忆 `aux_loses_to_pardiso_2d`。
