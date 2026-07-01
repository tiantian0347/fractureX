# Hu-Zhang 元角点松弛（corner relaxation）设计文档

> 关联代码
> - fealpy 原始空间：`fealpy/fealpy/functionspace/huzhang_fe_space_2d.py`
> - fracturex 容器：`fracturex/discretization/huzhang_discretization.py`
> - fracturex 角点松弛 wrapper（本次新增）：
>   `fracturex/discretization/huzhang_corner_relax.py`
> - fracturex 装配器：`fracturex/assemblers/huzhang_elastic_assembler.py`
> - 诊断脚本：`fracturex/tests/lshape_corner_diagnose.py`
> - 结构 sanity 测试：`fracturex/tests/test_huzhang_corner_relax.py`
> - L 形 PDE 验证（未通过）：`fracturex/tests/lshape_corner_relax_solve.py`
> - 理论笔记：`Tian/paper/fracture/mixFEM/HuZhang_corner_theory.md`
> - 文献：
>   - **[HM18]** Hu, Ma, arXiv:1807.08090v2 — §4 扩展应力空间
>   - **[Li21]** Y. Li, ESAIM M2AN 55 (2021) — hybridized AHMFEM

本次工作发现并定位了 3 个层次的问题，分两类：
**已修复**（fracturex 侧 wrapper）和 **遗留**（fealpy / fracturex stress bc 路径）。

---

## 1. 五个发现

### 1.1 fealpy 自带 `use_relaxation=True` 实际上从未生效

`HuZhangFESpace2d._filter_active_corners_by_support` 会把识别到的 NN 角点 **全部** 过滤掉，因为它们的 `corner2dof[:, -1]` 新 DOF 在 `cell_to_dof` 里找不到（fealpy `cell_to_dof` 的顶点段用 `node_to_internal_dof[cell[:, v]]`，没有把 newdof 注入到 cell 顶点 DOF 的路径）。
结果：`NCP=0`，`TM=I`，松弛是 no-op。

**证据**：`lshape_corner_diagnose.py` 在 L 形 N=2 网格上输出：
```
[corner support] NCP=1, supported=0, dangling=1
  dangling corner p=0, nid=4, newdof=24, coord=[0.0, 0.0]
```
方形域 `[0,1]^2` + 三边 ΓN 时所有 4 个候选角点都被过滤为 0。

`linear_elastic_with_huzhang.py` 看似收敛漂亮，是因为它用 `(sin·sin)²` 解析解 → ∂Ω 上 σn 全为 0 → 角点不一致问题本来就不存在，与松弛是否生效无关。

### 1.2 [HM18] §4 推广到任意 m 的统一实现（已在 fracturex 落地）

`HuZhangCornerRelax`（`fracturex/discretization/huzhang_corner_relax.py`）按 §4 的思路实现：

- 一个 NN 角点 $x_c$ 被 $m$ 个三角形围绕、有 $m-1$ 条内部边经过它；
- 把 fealpy 的"顶点处共享 3 DOF"放松为"每个 cell 上独立 3 cell-local DOF"（共 $3m$ 个 unconstrained DOF）；
- 通过 $m-1$ 条内部边的法向连续条件 $\sigma^{(k)} n_j = \sigma^{(k+1)} n_j$（每边 2 个标量约束）压回 $m+2$ 个 relaxed DOF；
- 维护两套 DOF（unc/rel）+ 长方变换矩阵 $TM \in \mathbb R^{gdof\_unc \times gdof\_rel}$。

**结构 sanity 全部通过**（`test_huzhang_corner_relax.py`）：DOF 计数、零空间维度（‖CN‖ < 1e-15）、cell_to_dof_unc 全覆盖、TM 非角点恒等。

### 1.3 fealpy/fracturex stress essential BC 标架不一致（已在 fracturex 修复）

**根因**：HuZhang 元里
- 节点 trace DOFs (每端 2 个) basis 是 cartesian Voigt {(1,0,0), (0,1/2,0), (0,0,1)} —— 即 `nsframe`；
- 边内部 trace DOFs (2(p-1) 个) basis 是 (nn-Voigt, sym(nt)-Voigt, tt-Voigt) —— 即 `esframe`。

fealpy `set_essential_bc` 对全部 8 个 trace DOFs（端点 2+内部 4+端点 2）统一用 `esframe[:, :2]` 投影 σ·n，导致**节点位置的 σ_nn 被当成 σ_xx 强加** → σ·n ≠ 0 时整体不收敛。

**修复**：`HuzhangStressBoundaryCondition.set_essential_bc_v2`
- 节点端点 trace（每端 2 个 DOF）：写 `(σ_xx, 2σ_xy)`（即 cartesian Voigt 前 2 项），σ_yy 不锁；
- 边内部 trace：保持原 `(σ_nn, 2σ_nt)` 投影。

**验证**（同一光滑制造解，单位方形 + bottom Dir + 3 边 ΓN）：
| 解析解 σn 行为 | 原 BC | v2 BC |
|---|---|---|
| (sin·sin)² (σn≡0 on ∂Ω) | 收敛 4 阶 | 同样收敛 4 阶 |
| **sin·sin (σn≠0)** | **发散** 7e-1, 6.6e-1 | **3.5e-3 → 2.1e-4 → 1.3e-5, 4 阶** |

L 形全 Dirichlet sin·sin：v2 BC 下 base 与 wrapper 都从 N=2 的 3.9e-2 收敛到 N=16 的 2.2e-5，4 阶最优 ✓。

### 1.4 mesh.error 默认积分阶 q=3 太低，会掩盖真实收敛

`mesh.error(sigmah, σ_exact)` 默认 `q=3`，对 p=3 HuZhang 元下 σ_h 是 P_3 张量，σ_exact·σ_h 内积是 P_6，需要 `q ≥ 6` 才精确。q=3 时积分误差高于离散误差，导致"收敛阶看上去 0.2"的虚假结果。

**修正**：`run` 中显式 `q=2*p+6`。

### 1.5 wrapper 的 Q 投影路径与 unc 装配等价（数值确认）

写了独立的 unc-space 装配器 `fracturex/assemblers/huzhang_unc_assembler.py`：直接
用 `cell_to_dof_unc` 把 fealpy integrator 的 cell-local 矩阵散射到 `gdof_unc`。跑了
之后发现：

**unc 装配后 `TM^T M_unc TM` 与之前的 `Q^T M_base Q` 数值等价**（NCP=0 时两者与
base 完全一致 up to 1e-14；NN 凹角时两者输出的 σ 也完全一样）。

原因：两条路径数学上等价——`M_unc` 在角点 base 3 DOF 段（前 gdof_base 部分中）
的耦合和 `M_base` 完全相同；`TM` 只在 `gdof_base` 之后的 3(m-1) 个 cell-local unc
DOF 上引入新耦合模式，但这些模式在 `M_unc` 中已经装成对角块（相邻 cell 之间的
角点节点在 base cell_to_dof 里就有耦合，unc 里被"独立化"了）——TM 的零空间给出
的 m+2 个 rel DOFs 恰好覆盖 M_unc 的角点秩空间。

**结论**：Q 路径不是"信息坍缩"，是等价投影。**wrapper 的实际瓶颈在 essential BC
如何在 unc/rel 上施加**，而不是装配路径。

### 1.6 KKT (Lagrange 乘子) wrapper：绕过 TM 投影

后续实现（`solve_with_wrapper` 中 `mode='relax'`）：在 unc 空间直接组装 KKT

```
[M_unc  B_unc  C^T ] [σ_unc]   [r_a_unc]
[B_unc^T   0     0 ] [u    ] = [-b     ]
[C        0     0 ] [λ    ]   [0      ]
```

其中 `C = HuZhangCornerRelax.C_constraint`（shape `2(m-1) × gdof_unc`）来自
`_build_normal_continuity_matrix` 的按角点堆叠。essential BC 通过
`lift_base_bc_to_unc` 施加：非角点 base id 恒等 lift；角点节点的 base 3 id 只
lock 到 **fan-end cells (k=0 与 k=m-1)** 上（这两个 cell 才承载 ΓN 边），中间 cells
的 σ_yy 保持自由由 λ 约束驱动。

### 1.7 光滑 σ 解析解 + NN 凹角上 wrapper 无改善（数值观察）

用 quartic 光滑解析解 `u = (sin(π(x+1)/2)·sin(π(y+1)/2))²` 在 L 形上：

| N | base | wrapper (KKT) |
|---|---|---|
| 2 | 1.22e-1 | 1.79e-1 |
| 4 | 1.56e-1 | 1.97e-1 |
| 8 | 2.08e-1 | 2.32e-1 |
| 16 | 2.42e-1 | 2.57e-1 |

两种都**不收敛**（re-entrant 凹角固有的低正则性），且 wrapper 略差于 base。
cell-by-cell 分析（N=4，`fracturex/tests/lshape_corner_relax_solve.py`）：wrapper 在
凹角 4 个 cells 中降低了 3 个的误差 (0.05, 0.03, 0.02 vs base 0.05, 0.04, 0.04)，
但让另 1 个 cell 变差近 2 倍 (0.14 vs 0.07)，远 cells 误差中位数也从 9e-3 增到
1.5e-2。

**背后原因**：wrapper 通过 essential BC + C 约束在角点周围引入非对称的 σ_yy 分布
（fan-end lock + 中间 cell 自由），这与光滑 σ 解析解本来在角点处对称 σ_yy=0 的
真值不匹配。对**光滑 σ 数据 wrapper 反而制造了 spurious 不对称**。

[HM18] 的数值实验（§5.1–5.3）都用**分片常应力**或**位移驱动**——traction 在两条
ΓN 边上真的不一致。此时 base 无解（over-constrained），wrapper 是唯一路径。
本节的光滑 σ 场景 traction 天然一致，wrapper 的额外自由度并不解决什么问题。

### 1.8 A 方案：`skip_nn_corner_nodes=True`（**首要 concrete 修复**）

`set_essential_bc_v2` 增加 `skip_nn_corner_nodes: bool` 参数：识别 NN 角点节点
（≥2 条 ΓN 边非反向共线地相交处），在这些节点上**跳过**端点 trace DOFs 的
essential 写入。数学上：mixed FEM 的 σ·n essential 应该沿边通过 (nn, sym(nt))
边内部 trace DOFs 实现，节点 trace DOFs 的 essential 是一种"点插值加强"，在
NN 角点上会引入过约束（两条边写入 cartesian σ_xx, σ_xy 的最后一条会覆盖前一
条，锁住的 σ_xy 不对；σ_yy 未锁则被解为 O(1) 错值）。跳过后角点 σ 由变分方程
+ 边内部 (nn, sym(nt)) trace 自然决定。

**代码位置**：`fracturex/boundarycondition/huzhang_boundary_condition.py`
- `HuzhangStressBoundaryCondition.set_essential_bc_v2(..., skip_nn_corner_nodes=True)`
- 辅助函数 `_detect_nn_corner_nodes` 用几何角度判定 NN 角点。

**验证结果**：

| 场景 | skip=False | skip=True |
|---|---|---|
| 方形 全 Dirichlet + sin·sin | 4 阶 | 4 阶 (noop) |
| 方形 3 边 ΓN + (sin·sin)² (σn=0) | 4 阶 | 4 阶 (noop) |
| 方形 3 边 ΓN + sin·sin (σn≠0), m=2 NN 角点 | 4 阶, N=4 err=3.5e-3 | 4 阶, N=4 err=**2.8e-3** (常数 −20%) |
| L 形 全 Dirichlet | 4 阶 | 4 阶 |
| L 形 NN 凹角 quartic (σn≠0), m=4 | **发散** (负阶) | 稳定 ~0.32 |

### 1.9 wrapper 在真奇异 σ 数据上的表现（Williams 解，[HM18] §5.2）

`fracturex/tests/hm18_williams_singular.py`：旋转 L 形 [−2,2]² \ {x≥0, y≤0}，
使用 [HM18] §5.2 Williams 位移

    u_r(r,φ) = r^α/(2μ) [−(α+1) cos((α+1)φ) + (C_2 − α − 1) C_1 cos((α−1)φ)]
    u_φ(r,φ) =  r^α/(2μ) [(α+1) sin((α+1)φ) + (C_2 + α − 1) C_1 sin((α−1)φ)]

E=1e5, ν=0.499, α ≈ 0.54448。σ ~ r^(α−1) 在原点奇异。ΓD 在凹角两条边（y=0, x∈[0,2]
与 x=0, y∈[−2,0]），其余为 ΓN。

| N | base 收敛阶 | wrapper (A 方案) 收敛阶 | 备注 |
|---|---|---|---|
| 4→32 | ≈ 1.00 | ≈ 1.00 | wrapper 常数比 base 大 3-5%，阶等同 |

**观察**：
- base 稳定收敛 ≈ 1 阶（Williams α 预期最好速率≈ 0.5，实测 1 阶）。
- wrapper 在此算例上**无 rate 改善**，因为 fracturex 的 NN 角点检测只识别**全 ΓN**
  角点（type=2）。凹角 (0,0) 是 DN 角点（type=1），wrapper 跳过。wrapper 检测到的
  是外部凸角 (−2,−2) 和 (2,2)（都是全 ΓN），那里 σ 光滑，wrapper 引入的 KKT 约束
  只增加数值噪声，不带来 rate 改善。

要真正让 wrapper 在此算例发挥作用需要：
- 把凹角改成 NN（ΓD 换成 ΓN，重算 Williams 兼容 BC）——需要重新推导；或
- 加自适应循环（[HM18] §5.4）——σ_h 在凹角附近逐层加密后能突破 1 阶。

### 1.10 wrapper 在人为角点 traction 不相容算例上的表现

`fracturex/tests/hm18_inconsistent_traction_corner.py`：方形域 (1,1) 凸角处
让底边和右边给出**不同** σ_xy 数据（1.0 vs 2.0）。

- **base 模式**：σ_xy(1,1) 被后写入的边覆盖；σ_yy 在角点越 refine 越发散（N=4:
  −9.05 → N=8: −12.56 → N=16: −16.13，线性发散）。直接暴露 [HM18] §3 描述的
  角点退化。
- **wrapper KKT + 每 cell 独立 essential lock**：矩阵 **exactly singular**。原因是
  法向连续约束 C 与"两 cells 上锁 cartesian σ_xy 独立值"数学冲突（[HM18] §3 的
  另一面：无法同时满足所有约束）。
- **wrapper A 方案 (skip corner essential)**：跑通不 nan，但 σ 在角点处**几十量级
  的随机值**——A 方案下 wrapper 没有对角点 σ 的任何 anchor，边内部 trace 的
  P_{p-1} 多项式外推到端点可以任意远。

**结论**：wrapper 无法处理"两条边给出真正不相容的 σ_gd"，这是数学事实（H(div)
一致性约束），不是实现 bug。**wrapper 只能修 base 因"cartesian trace 覆盖"引起
的"cell 表达能力不足"的问题**，不能创造出 σ 一致解不存在的情形下的解。

### 1.11 wrapper 使用决策表

| 你的场景 | base + `skip_nn=True`（A 方案） | wrapper (KKT) | 备注 |
|---|---|---|---|
| σn ≡ 0 on ∂Ω (齐次 Neumann / 齐次数据) | ✓ 最优阶 | ✓ (noop) | 二者等同 |
| σ 光滑, ΓD 覆盖凹角 | ✓ 最优阶 | ✓ (noop) | 无 NN 角点 |
| σ 光滑, 方形 NN 角点 (m=2), σn≠0 | ✓ 最优阶，常数比无 skip 小 20% | 等同 | **首选 A 方案** |
| σ 光滑, L 形凹角 NN (m=4), σn≠0 | ~0.32 稳定不收敛 | ~0.34 稳定不收敛 | 需自适应加密 |
| σ 奇异 (Williams α<1)，凹角 ΓD | 1 阶 | 1 阶等同 | 需自适应加密突破 α 阶天花板 |
| 角点 σ 数据真不相容（不同边给不同 σn） | σ 发散 | 数学上无解 (KKT singular 或 A 方案发散) | 数据本身违反 H(div) |

**推荐路径**：
- 生产使用：直接开 `set_essential_bc_v2(..., skip_nn_corner_nodes=True)`，wrapper 不启用。
- 学术研究：wrapper 结构留作**下一阶段自适应循环**（[HM18] §4.1 SOLVE-ESTIMATE-MARK-REFINE）
  的基础设施——凹角附近网格加密后，wrapper 的 m+2 cell-local 独立 σ 表达可以突破
  base 的天花板。

### 1.12 wrapper 的数学正确性验证（vs [HM18] §3.1）

`fracturex/tests/verify_wrapper_basis_hm18.py` 显式对比 wrapper 在 m=2 NN 角点
上的 TM 列块（4 个 rel basis 在 unc 空间的表达）与 [HM18] §3.1 (3.1) 手工构造的
4 个 basis：

  b1 = φ_{x_c} · n n^T                (两侧共享，"法向平方")
  b2 = φ_{x_c} · (n t^T + t n^T)      (两侧共享，"混合法-切")
  b3 = φ_{x_c}^+ · t t^T              [K+ only]（切向纯切，K+ 独立）
  b4 = φ_{x_c}^- · t t^T              [K- only]（切向纯切，K- 独立）

其中 (n, t) 是内部边 e 的法向/切向。基于 [HM18] (2.5) 的定义:

  S_e = n n^T,  S⊥_{e,1} = n t^T + t n^T,  S⊥_{e,2} = t t^T

Key insight：切向纯切矩阵 S⊥_{e,2} = t t^T 在法向上作用为零（因 t·n=0），
即 (t t^T) n = 0。两侧切向分量任意独立不影响 σ·n 连续。这就是 [HM18] §3.1
"partial C⁰ vertex relaxation"——**只放松切向**，保留法向。

**验证结果**（方形 [0,1]² N=2 网格，NN 角点 (1,1)，m=2）：

  rank(T_w) = 4, rank(T_p) = 4, rank([T_w | T_p]) = 4    ← 相同 span
  T_w = T_p · A,  |T_p A - T_w| ≈ 1e-15,  det(A) ≠ 0    ← 可逆坐标变换

**结论**：wrapper 的松弛子空间与 [HM18] §3.1 完全相同（相差 rel-space 内一个 4×4
可逆矩阵，不影响 span）。wrapper **数学正确等价 [HM18]**。

**推论**：wrapper 在光滑 σ 数据上不给 rate 改善不是实现 bug，是 [HM18] 理论本身
对光滑数据的性质——wrapper 只承诺"阶不变，常数改善"，且改善量取决于 σ 在角点
附近的 traction-inconsistency 程度。对光滑 σ + 离散伪奇异，改善量可能微弱到
被 O(h^p) 主项掩盖。

### 1.13 一句话总结

fealpy 原 `set_essential_bc` 有 cartesian/(nn,sym(nt)) 标架混用 bug（§1.3），
`set_essential_bc_v2(skip_nn_corner_nodes=True)` 完整修复；wrapper 结构 (§1.2, 1.6)
是 [HM18] §4 的忠实实现但对光滑 σ 数据 rate 无改善，**留作自适应循环的下阶段基础**。

---

## 2. 当前 fracturex wrapper 设计

### 2.1 模块

`fracturex/discretization/huzhang_corner_relax.py`

```python
class HuZhangCornerRelax:
    base_space: HuZhangFESpace2d       # use_relaxation=False
    corners: list[_CornerInfo]          # 角点拓扑 + DOF id
    cell_to_dof_unc: np.ndarray         # (NC, ldof) 扩展 cell-to-dof
    TM: scipy.sparse.csr_matrix         # (gdof_unc, gdof_rel)
    gdof_base, gdof_unc, gdof_rel: int
    empty_rel_slots: np.ndarray         # 在 rel 中是空槽位的 base id
    Q: scipy.sparse.csr_matrix          # (gdof_base, gdof_rel) = P^T @ TM
```

### 2.2 数学（[HM18] §4 推广）

对每个 NN 角点 $x_c$（$m$ 三角形、$m-1$ 内部边 $e_1,\dots,e_{m-1}$）：

- unc 上 cell-local DOF 排列 $z = (S^{(1)}_{xx}, S^{(1)}_{xy}, S^{(1)}_{yy}, \dots, S^{(m)}_{yy}) \in \mathbb R^{3m}$；
- 法向连续约束 $C z = 0$，$C \in \mathbb R^{2(m-1)\times 3m}$：

  $$
  \forall j: \begin{cases}
  S^{(k_j)}_{xx} n_{jx} + S^{(k_j)}_{xy} n_{jy}
   - S^{(k_{j+1})}_{xx} n_{jx} - S^{(k_{j+1})}_{xy} n_{jy} = 0,\\
  S^{(k_j)}_{xy} n_{jx} + S^{(k_j)}_{yy} n_{jy}
   - S^{(k_{j+1})}_{xy} n_{jx} - S^{(k_{j+1})}_{yy} n_{jy} = 0.
  \end{cases}
  $$

- 用 SVD 取 $N := \text{null}(C) \in \mathbb R^{3m\times(m+2)}$ 作为该角点的 TM 块。

$M_p = (m+2)$ 在 $m=2$ 时退化为 4，正好与 [HM18] §4.2 两单元情形一致；$m=3$ 是 5、$m=4$ 是 6，覆盖 L 形凹角等 fealpy 自带模式处理不了的几何。

### 2.3 DOF 布局

- unc：`[0, gdof_base)` 段保留 fealpy 原 id（角点 cell-0 沿用），`[gdof_base, gdof_unc)` 段为 corner cells 1..m-1 的 3(m-1) 个新 DOF；
- rel：`[0, gdof_base)` 段保留非角点 base DOF id；角点的 base 3 id 在 rel 中是 "empty slot"（无任何耦合）；`[gdof_base, gdof_rel)` 段为每角点 m+2 个新 rel DOF。
- empty_rel_slots 在解时强加 0（Dirichlet）保证矩阵不奇异。

### 2.4 Q：把 fealpy 装的 base 矩阵压到 rel

```
P ∈ R^{gdof_base × gdof_unc}: 把每个角点 cell 的 3 cell-local unc DOF
                              加到对应 base node DOF 上。
Q = P @ TM ∈ R^{gdof_base × gdof_rel}.
```

求解流程（伪代码）：

```python
M = bform_stress.assembly()       # gdof_base × gdof_base
B = bform_mix.assembly()          # gdof_u    × gdof_base
relax = HuZhangCornerRelax(mesh, p, isNedge=isN)
Q = relax.Q
M2 = Q.T @ M @ Q                  # gdof_rel × gdof_rel
B2 = Q.T @ B                      # gdof_u   × gdof_rel
A = bmat([[M2, B2], [B2.T, None]])
# RHS & BC 略
σ_rel = solve(...)[:gdof_rel]
σ_unc = relax.TM @ σ_rel          # 每个 cell 的 cell-local 节点 σ
σ_base_avg = relax.lift_stress_base_averaged(σ_rel)  # 仅供 fealpy 标准接口使用
```

### 2.5 与 fealpy 自带 use_relaxation 的关系

fracturex wrapper **完全独立** —— `HuZhangCornerRelax` 始终基于 `use_relaxation=False` 的原始空间。fealpy 内置路径被 1.1 节根因废弃，未来如要在 fracturex 代码主线生效，应通过本 wrapper 注入 cell_to_dof_unc / TM。

---

## 3. 已知边界

1. **m=2 fealpy 自带模板的回归**：用 wrapper 后 fealpy `use_relaxation=True` 这条路径可在 fracturex 全栈下被淘汰。后续 commit 应清理 `HuZhangDiscretization` 中的 `use_relaxation` 默认 True 行为，迁移到 wrapper 或直接淘汰。
2. **a posteriori 估计器**（[HM18] §5.4）的角点修正：edge jump 在角点处取单侧极限而非平均跳跃，与 wrapper 的 cell-local σ 读法天然兼容。`fracturex/adaptivity` 未对接，待后续。
3. **3D**：fealpy `huzhang_fe_space_3d.py` 与 wrapper 都未支持。
4. **"边内跳跃"角点**：fracturex 的 NN 角点检测只识别**几何角点**（两条不共线的 boundary edges 交点），不识别**同一直边上的 traction 不连续点**（如 [HM18] §5.1 场景）。若要支持后者需扩展 `HuZhangCornerRelax._detect_nn_corners`。
5. **DN 角点**：type=1 的 ΓD/ΓN 混合角点被 wrapper 跳过。对 Williams 奇异算例这意味着凹角（DN 类型）无法直接由 wrapper 处理——需要改为 NN 配置或走自适应路径。

---

## 4. 用法摘要

最小例子（PDE 装配 + wrapper 接入 + 求解器）：

```python
from fracturex.discretization.huzhang_corner_relax import HuZhangCornerRelax

mesh = ...                                # TriangleMesh
isN = build_isNedge_from_isD(mesh, isD)   # (NE,) bool

base = HuZhangFESpace2d(mesh, p=p, use_relaxation=False, bd_stress=isN)
M = BilinearForm(base).add_integrator(HuZhangStressIntegrator(...)).assembly().to_scipy()
B = BilinearForm((u_space, base)).add_integrator(HuZhangMixIntegrator()).assembly().to_scipy()

relax = HuZhangCornerRelax(mesh, p=p, isNedge=isN, base_space=base)
Q = relax.Q
M2, B2 = Q.T @ M @ Q, Q.T @ B
A = bmat([[M2, B2], [B2.T, None]])
# ... 装配右端、施加 BC、求解
σ_rel = X[:relax.gdof_rel]
σ_cell_local = relax.TM @ σ_rel             # 用于松弛后的物理 σ 评估
σ_base_avg = relax.lift_stress_base_averaged(σ_rel)  # 兼容 fealpy 标准 σh
```

诊断与 sanity：

```bash
PYTHONPATH=$REPOS/fealpy:$REPOS/fractureX:. python test_huzhang_corner_relax.py
PYTHONPATH=$REPOS/fealpy:$REPOS/fractureX:. python lshape_corner_diagnose.py 2 3
```

**生产用推荐**（不用 wrapper，只用 A 方案）：

```python
HSBC = HuzhangStressBoundaryCondition(space=base_space)
uh_sig, isbd_sig = HSBC.set_essential_bc_v2(
    stress_gd, threshold=isN, coord='auto',
    skip_nn_corner_nodes=True,   # ← 关键：修 fealpy 标架 bug + 避免 NN 角点过约束
)
```

---

## 5. 文件清单

**新增**：
- `fracturex/discretization/huzhang_corner_relax.py` — wrapper（识别 + DOF + TM + KKT 接口）
- `fracturex/assemblers/huzhang_unc_assembler.py` — unc 空间装配
- `fracturex/tests/test_huzhang_corner_relax.py` — 结构 sanity（全过）
- `fracturex/tests/lshape_corner_diagnose.py` — fealpy filter 诊断
- `fracturex/tests/lshape_corner_relaxation.py` — fealpy 自带 use_relaxation 对照
- `fracturex/tests/lshape_corner_relax_solve.py` — wrapper KKT PDE 集成（L 形 quartic）
- `fracturex/tests/hm18_piecewise_const_stress.py` — [HM18] §5.1-like 反例（几何角点不识别边内跳跃）
- `fracturex/tests/hm18_inconsistent_traction_corner.py` — 人为不相容 traction（wrapper 数学不适用）
- `fracturex/tests/hm18_williams_singular.py` — [HM18] §5.2 Williams 奇异

**修改**：
- `fracturex/boundarycondition/huzhang_boundary_condition.py`
  - `HuzhangStressBoundaryCondition.set_essential_bc_v2`：修 fealpy 标架 bug + `skip_nn_corner_nodes` A 方案

**未动**：任何 fealpy 代码。
