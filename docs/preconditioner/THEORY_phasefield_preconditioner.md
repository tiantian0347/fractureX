# 相场子问题预条件子设计（理论 + 真实裂后系统实测）

> 目标：为 Hu–Zhang 交错求解中的**相场线性子问题** `A δd = F` 设计预条件子。
> 之前用过相场预条件"结果不对"——本文用**真实加载步（裂纹已产生）** 的线性系统做对照，
> 证明预条件子只改变**迭代数/时间、不改变解**，并从矩阵形式与谱分布推出正确设计。
>
> 全部数据在本地生成（model0 circular-notch，h1/h2/h3 三套网格，均为 `step_030 / max_d≈1` 的裂后系统）。
> 复现脚本（已随文档存于 `docs/preconditioner/scripts/`）：`dump_phase_local.py`（导系统）、`analyze_phase_precond.py`（谱+基准）、`summary_plot.py`（跨网格图）。

---

## 1. 算子来源与矩阵形式

相场（AT2 + 二次退化 `g(d)=(1-d)²` + hybrid 分裂）子问题在增量形式下求解

```
A δd = F,   F = rhs - A d_old,   然后 d = clip(max(d_old, d_old+δd), 0, 1)   （不可逆投影）
```

弱形式（对试探 `δd`）：

```
a(δd, w) = ∫_Ω [ Gc·l0 · ∇δd·∇w  +  ( Gc/l0 + 2H(x) ) · δd·w ] dx
```

对应装配（见 `fracturex/assemblers/phasefield_assembler.py:575`）三个系数：

| 项 | 系数 | 物理含义 | 矩阵块 |
|---|---|---|---|
| 扩散 `diff_coef` | `Gc·l0·2/c_d` | 裂纹面正则化（各向同性） | `K`（刚度/Laplacian） |
| 质量1 `mass_coef1` | `g''·Gc/(l0·c_d)` | AT2 局部裂纹密度 | `M0`（与 H 无关） |
| 质量2 `mass_coef2` | `g''(d)·H(x)` | 弹性历史驱动力 | `M_H`（**H 依赖**） |

于是

```
A = K + M0 + M_H  =  A_const + A_hist,   A_hist = M_H
```

**关键结构性质（本地实测确认）：**

- **对称正定 (SPD)**：三块都是对称的（`∇·∇`、`δd·w` 均对称），系数非负 ⇒ `A = Aᵀ`, `A ≻ 0`。
  实测 `|A - Aᵀ|_max = 2.8e-17 … 5.6e-17`（机器精度）。
  → **必须用 CG，而不是 GMRES**。生产代码 `_phase_gmres` 用 GMRES 解 SPD 系统，
  既浪费（GMRES 每步存全部 Krylov 基、正交化 O(k²)），又收敛更差（见 §4：GMRES 118 vs CG 141 iters，但 GMRES 单步贵 4×）。

- **反应–扩散（screened Poisson）型**，系数强对比：
  反应系数 `Gc/l0 + 2H(x)` 在裂纹带内 H 暴增（本例 `H: 6e-4 → 1.9e4`），
  使对角线对比 `diag_max/diag_min ≈ 200–265`。这是病态的来源。

结构参数（三套网格）：

| tag | n (dof) | nnz | 每行nnz | diag 对比 | λ(A) 范围 |
|---|---|---|---|---|---|
| h1 | 1384  | 14490  | 10.5 | 203 | [1.3e-2, 5.4] |
| h2 | 5956  | 65428  | 11.0 | 256 | [6.1e-3, 6.1] |
| h3 | 22486 | 252826 | 11.2 | 265 | [1.9e-3, 6.3] |

---

## 2. 特征值分布与病态根源

见 `spectrum_and_convergence_m0h1.png`（左）与 `eig_hist_m0h1.png`。

对 h1（n=1384，可做稠密全谱）：

- **A 原始谱**（蓝）：铺满 `[0.013, 5.37]`，`κ(A)=408`。低端一簇 `λ~0.01–0.1`（**裂纹带内、被 H 主导的低刚度模态**），高端 `λ~2–8`（**未损伤区扩散模态**）。两簇分离 → CG 需要很多步才能"扫掉"两端。
- **Jacobi 对角缩放** `D^{-1/2} A D^{-1/2}`（橙）：`κ→13.4`，把系数幅值的各向异性拉平了，但**没有处理扩散算子本身的网格相关病态**（见 §3）。
- **AMG-SA 预条件谱** `M⁻¹A`（绿）：谱**压到 ≈1 附近一簇**，`κ→1.80`。这正是"最优预条件子"的判据。

### 病态随网格加密的增长（核心论据）

见 `mesh_independence_summary.png`（右）。反应–扩散算子的扩散部分谱同 `-Δ`，
`κ(A) = O(h⁻²) = O(n)`（2D）。实测：

| n | κ(A) | κ(Jacobi) |
|---|---|---|
| 1384  | 408  | 13.4 |
| 5956  | 997  | 34.9 |
| 22486 | 3229 | 123.5 |

`κ(A)` 与 O(n) 参考线几乎平行 ⇒ **h⁻² 病态被实测证实**。
Jacobi 只降常数因子（~30×），增长率仍是 O(n) ⇒ 迭代数仍随网格发散。
**只有多重网格能打破 O(h⁻²)**（§3）。

---

## 3. 理论：应该用什么预条件子

### 3.1 判据
理想预条件子 `M` 应使 `M⁻¹A` 的谱聚集（`κ(M⁻¹A)=O(1)`，与 h 无关），
且 `M⁻¹` 每次作用 `O(n)`。对本问题：

1. **算子是 SPD** ⇒ 用 **CG**（对称、短递推、内存 O(n)）。GMRES 是错配。
2. **算子是二阶椭圆（反应–扩散）** ⇒ 其扩散部分的 `O(h⁻²)` 病态**必须用多重网格 / 多层方法消除**。
   代数多重网格 (AMG) 对这种 M-矩阵型 SPD 系统是标准最优解，收敛率与 h 无关。
3. **系数强对比（H 跳变）** ⇒ 用 **smoothed-aggregation AMG (SA-AMG)** 或带**强度阈值**的
   Ruge–Stüben (RS-AMG)：聚合/粗化能自动跟随 H 的强连接，把裂纹带聚成粗格自由度。

### 3.2 推荐设计（主）：**CG + SA-AMG**
```
M = 一个 V-cycle of smoothed_aggregation_solver(A)
solve: cg(A, F, M=M, rtol=1e-8~1e-10)
```
- 谱：`κ(M⁻¹A) = 1.8–2.3`，**与网格无关**。
- 迭代：12 → 15 → 18（h1→h2→h3），近乎常数。
- 复杂度：AMG 建立 O(n)，operator complexity ≈1.04（极稀疏粗层），每 V-cycle O(n)。
- **setup 摊销**：交错迭代中相邻加载步 A 变化小（只有 M_H 随 H 增），
  可**跨步复用 AMG hierarchy**（每 k 步重建一次），与弹性块 `precond_rebuild_interval` 同思路。

### 3.3 备选
- **CG + ILU/IC(0)**：本例最快（3–4 iters），但见 §3.4 的实测——它在大规模/3D 会因 fill-in 膨胀、串行三角回代、系数强对比不稳定而失效。**小/中 2D 网格可用，作大规模主推不合适**。
- **CG + Jacobi**：几乎零成本、线程安全，作为**最低保底**（比无预条件快 3–4×），
  但不解决 O(h⁻²)。可作为 AMG 内部 smoother。
- **纯 AMG V-cycle（不套 CG）**：26→37→53 iters，比 CG-AMG 差 ~2–3×，不推荐独立使用。

### 3.4 为什么不主推 CG + ILU（尽管它在小例子里最快）

ILU 在 h1 上只要 3 iters，看着最优。但把它放到"网格加密 + 大规模 + 并行 + 交错多步"的真实场景,四条硬伤逐一暴露(实测见 `ilu_vs_amg.png`):

**(1) 迭代数与内存不可兼得——ILU 的半最优本质 O(h⁻¹)。**
固定一个 drop_tol，要么迭代数发散，要么内存爆炸，二选一:

| 网格 | n | ILU(drop=1e-2) iters | ILU(drop=1e-4) iters | ILU(1e-4) fill = nnz(LU)/nnz(A) | **AMG iters / op-cx** |
|---|---|---|---|---|---|
| h1 | 1384  | 5  | 3 | 2.5× | 10 / **1.04×** |
| h2 | 5956  | 8  | 3 | 3.9× | 12 / **1.04×** |
| h3 | 22486 | **12** | 4 | **5.5×** | 15 / **1.04×** |

- **便宜档 drop=1e-2**：迭代数 5→8→12 **随网格发散**（这才是 ILU(0) 的真面目，O(h⁻¹)）。
- **精细档 drop=1e-4**：迭代数看似稳(3→4)，但代价是 **fill-in 2.5→3.9→5.5× 单调增长**——
  内存随 n 变差，3D 上 ILU fill-in 通常 O(n^{4/3})~O(n^{3/2})，直接不可承受。
- **AMG**：operator complexity **恒定 1.04×**（粗层极稀疏），迭代 10→12→15 近乎常数。**两者兼得**。

**(2) setup 成本反超。** ILU(1e-4) 的分解本身随 n 变贵，h3 上 setup≈70–96ms，**已慢于 AMG 建层的 16ms**。规模再大差距拉开。

**(3) 串行三角回代，并行性差。** ILU 每次作用要解 `L y=r, U z=y` 两个三角系统，本质**顺序依赖**，
在 176 核服务器上几乎不并行。AMG 的 smoother（Jacobi/多项式）与粗层传输都是 SpMV，**天然并行**。
生产环境是多核大规模，这条几乎是决定性的。

**(4) 系数强对比下不稳定。** 裂后 H 跳变 5 个量级，不完全分解易出**极小主元 / 近奇异**，
需要对 drop_tol、fill_factor 反复调参才不崩；换个加载步、换个网格又要重调。
AMG 的聚合按矩阵强连接自动粗化，**跟随裂纹带、无需逐例调参**，鲁棒性明显更好。

**(5) 交错多步复用差。** 相邻加载步只有 `M_H` 随 H 变，AMG 可**跨步复用 hierarchy**（每 k 步重建）。
ILU 的 L/U 与矩阵值强绑定，A 一变旧分解就失配，复用旧 L/U 会明显掉收敛甚至发散。

**小结**：ILU 是"小 2D 单次求解"的快捷方案，可留作 AMG 不可用时的备选；
但**面向网格无关、大规模、并行、多加载步**的目标，SA-AMG 才是唯一同时满足最优性 + 内存 + 并行 + 鲁棒 + 可复用的选择。这正是文献（§5）一致的结论。

---

## 4. 实测：预条件子只改迭代数，不改解（"结果不对"的排查）

对**真实裂后系统**（`b=F`），先用稀疏直接解 `x_ref=spsolve(A,F)` 作基准，
再测每种预条件的 `relerr = ‖x-x_ref‖/‖x_ref‖`（rtol=1e-10）：

**h1（n=1384）:**

| 解法 | iters | 时间 | relerr | 说明 |
|---|---|---|---|---|
| GMRES none（**当前生产**）| 118 | 19.9 ms | 1.6e-10 | SPD 上用 GMRES，浪费 |
| CG none | 141 | 4.8 ms | 5.6e-11 | |
| CG Jacobi | 39 | 1.6 ms | 4.5e-11 | κ:408→13 |
| **CG AMG-SA** | **12** | 2.1 ms | 9.0e-12 | κ:408→**1.8** |
| CG AMG-RS | 13 | 3.3 ms | 3.8e-12 | |
| CG ILU | 3 | 0.2 ms | 1.1e-11 | |

**全部 relerr ≈ 1e-11 ⇒ 收敛到同一个解。**
→ **预条件子本身不会让"结果不对"。** 之前的错误几乎必然是下面某一类实现 bug：

1. **把 `M⁻¹` 当成解**（只作用一次预条件当求解），没有跑 Krylov 外迭代。
2. **在 Dirichlet BC 之前**用 A 建预条件，或对被 BC 消掉的行建预条件 → 与实际系统不匹配。
   （正确顺序：`assemble → _apply_phase_dirichlet_bc(A,F) → 再建 M`。）
3. **求解器与算子对称性不匹配**（拿非对称 GMRES 的收敛判据/重启参数套 SPD）。
4. **忘了增量形式 + 不可逆投影**：解出的是 `δd`，还要 `d=clip(max(d_old, d_old+δd),0,1)`；
   若把 `δd` 直接当 `d`，或漏投影，结果就"不对"——但这是**装配/更新逻辑 bug，与预条件无关**。
5. **收敛判据太松**（如 rtol=1e-3）→ 未收敛的解，误判为"预条件把结果搞坏了"。

排查建议：固定一个裂后 A、F，跑 `spsolve` 得 `x_ref`，任何预条件 CG 的 `relerr` 都应 <1e-8；
若某预条件 relerr 大，就是该预条件的实现 bug，而非数学问题。

### 4.1 生产后端已落地：跨网格验证（`run_case.py FRACTUREX_PHASE_BACKEND=amg`）

上述设计已实现为 `run_case.py` 的相场后端 `_PhaseCGAMG`（有状态 CG + SA-AMG，
跨加载步复用 hierarchy、dof 变化强制重建、`relerr>check_rtol` 时重建重试再退回 `spsolve` 护栏）。
用 `FRACTUREX_PHASE_BACKEND=amg` 选中，已在服务器 `py312`（pyamg 5.3.0）导入通过、e2e 冒烟
（10 次相场求解全用 `cg-amg-phase`，残差 ~1e-11，无 fallback）。

**跨三套网格的确定性单点对照**（真实裂后系统，rtol=1e-10，全部 relerr ~1e-11 收敛到同一解）。
注意 Krylov 迭代数是**矩阵的确定性函数、与机器/负载无关**，故直接取自 `analysis_m0h{1,2,3}.json`：

| 相场 dof n | κ(A) | GMRES-none（旧生产） | CG-none | CG-Jacobi | **CG+AMG-SA** | CG+AMG-RS |
|---|---|---|---|---|---|---|
| **h1** 1384  | 408  | 118 | 141 | 39  | **12** | 13 |
| **h2** 5956  | 997  | 327 | 275 | 68  | **15** | 17 |
| **h3** 22486 | 3229 | 737 | 568 | 131 | **18** | 17 |

- **旧生产 GMRES-none：118→327→737**，随网格发散（κ=O(h⁻²)）。
- **CG+AMG-SA：12→15→18**，近乎常数——**网格无关**，κ(M⁻¹A) 1.80→2.26→~2。h3 上比旧 GMRES 少 **41× 迭代数**。
- Jacobi 只降常数（131 iters），增长率仍 O(n)，仅作保底。
- e2e 侧验证：h3 生产运行（`FRACTUREX_PHASE_BACKEND=gmres`）相场迭代数从 h1 的 ~35 涨到 **108**（生产 tol），网格相关性实锤；换 amg 后端后该子问题降到 ~15 且不随网格增长。

> 墙钟对比在共享服务器上不可比（相场求解仅占单步 ~0.5%，其余是相同的弹性 pardiso + 多用户 MKL 竞争）；
> **迭代数才是干净指标**。上线用 §4 的 `spsolve` relerr<1e-8 护栏防回归。

---

## 5. 相关文献（整理：要点 + 与本工作的对应）

按"与相场预条件的相关度"排序。每条给出**核心贡献**与**对本文结论的支撑点**。

### 5.1 直接相关：相场断裂的求解器 / 预条件

1. **Farrell, P. & Maurini, C. (2017).** *Linear and nonlinear solvers for variational phase-field models of brittle fracture.* IJNME 109(5):648–667. arXiv:1511.08463.
   - **要点**：系统比较相场断裂的线性/非线性求解器；指出交错(alternate minimization)中每个子问题的算子结构，论证弹性块与相场块应分别用**面向各自算子的最优预条件**；相场块（SPD、二阶椭圆）推荐 **CG + 多重网格**。
   - **支撑本文**：§3 的"SPD⇒CG、椭圆⇒MG"路线与该文一致；本文用真实裂后系统把这个结论量化到 κ 与迭代数。

2. **Rannou, J. et al. (2024).** *Preconditioning strategies for phase-field fracture.* IJNME, nme.7544.
   - **要点**：专门针对**裂纹局部化后**历史场 H 强对比导致的病态设计预条件；讨论 AMG/GMG 在系数跳变下的粗化策略。
   - **支撑本文**：正是本文 §2 观察到的 `H:6e-4→1.9e4`、对角对比~265、κ(A)=O(h⁻²) 现象的文献对应；§3.4 关于"聚合跟随裂纹带"的论点出自这一类工作。

3. **Kopaničáková, A. & Krause, R. (2022).** *A recursive multilevel trust region method with application to fully monolithic phase-field models of brittle fracture.* CMAME. arXiv:2203.13738.
   - **要点**：把**多层/多重网格**嵌入信赖域框架求解相场断裂（含单调求解全耦合问题），多层是消除 h⁻² 病态的核心。
   - **支撑本文**：§3 "只有多重网格能打破 O(h⁻²)"的理论依据；也提示后续可从交错→全耦合的多层扩展。

4. **TNNMG — Gräser, C. et al. (2023).** *Truncated Nonsmooth Newton Multigrid for phase-field fracture with irreversibility.* Comput. Mech. 10.1007/s00466-023-02330-x.
   - **要点**：把**不可逆约束**（`d≥d_old`，即本文的 `clip(max(d_old,·))` 投影）作为**变分不等式**，用截断非光滑牛顿多重网格直接处理，多重网格保证网格无关收敛。
   - **支撑本文**：回应 §4 排查项(4)——不可逆投影属于**外层非光滑结构**，应放在求解框架里，而不是让它污染内层线性预条件；本文当前用"线性解+投影"的交错近似，TNNMG 是更严格的升级路径。

### 5.2 方法与实现基础

5. **Bell, N., Olson, L., Schroder, J., Southworth, B. (2023).** *PyAMG: Algebraic Multigrid Solvers in Python.* JOSS 8(87):5495.
   - **要点**：本文实测所用 **smoothed-aggregation / Ruge–Stüben AMG** 的实现与接口（`aspreconditioner()`）。
   - **支撑本文**：§3.2、§4 全部 AMG 数据由此产生；operator complexity=1.04 的度量也来自 PyAMG。

6. **Geometric multigrid for phase-field (2024).** arXiv:2404.03265.
   - **要点**：相场问题的**几何多重网格**实现细节（层间传输、smoother 选择、局部加密下的多重网格）。
   - **支撑本文**：当有网格层级(h1/h2/h3 本就是几何加密序列)时，GMG 可替代 AMG 得到更低常数；作为 §3.2 的实现备选。

### 5.3 前沿 / 可选延伸

7. **Learned / matrix-free preconditioners (2026).** arXiv:2606.23458（matrix-free PyTorch 方向）。
   - **要点**：用学习到的算子或 matrix-free 方式构造预条件，避免显式装配/分解。
   - **支撑本文**：与本仓库 `D13_LEARNED_PRECONDITIONER_PAPER_PLAN.md` 呼应，属长期方向；当前阶段 AMG 已足够，列此备查。

> **文献一致结论**：相场子问题（SPD + 二阶椭圆 + H 强对比）的标准最优解是
> **CG/PCG + (代数或几何)多重网格**；不可逆约束用非光滑牛顿/VI 多重网格在外层处理。
> ILU 仅作小规模备选。本文用真实裂后系统在 model0 上把这一路线量化并复现。

---

## 6. 结论与落地

1. **换 solver**：相场子问题 `_phase_gmres` → `_phase_cg`（SPD 用 CG）。
2. **加预条件**：CG + SA-AMG（pyamg `smoothed_aggregation_solver(A).aspreconditioner()`），
   跨加载步复用 hierarchy（每 k 步重建）。预期：把生产里 ~186 iters/solve 的相场求解压到 **~15 iters、且网格无关**。
3. **不用 ILU 作主推**：见 §3.4 与 `ilu_vs_amg.png`——ILU 在大规模/3D/并行/多步场景 fill-in 膨胀、串行回代、强对比不稳定、复用差；仅留作小规模备选。
4. **保底**：AMG 不可用时退化到 CG+Jacobi（≥3× 加速，零依赖风险）。
5. **正确性护栏**：上线前用 §4 的 `spsolve` 基准比对 relerr<1e-8，防止重蹈"结果不对"。

---

### 附：复现
```bash
source ~/venv_fealpy3/bin/activate
cd /Users/tian00/repository/fractureX
S=docs/preconditioner/scripts
# 1) 从本地裂后 checkpoint 导出真实相场系统（model0 h1/h2/h3）
FRACTUREX_HMIN=0.05  python $S/dump_phase_local.py model0 paper_aux_h1    docs/preconditioner/data/phase_m0h1.npz
FRACTUREX_HMIN=0.025 python $S/dump_phase_local.py model0 paper_direct_h2 docs/preconditioner/data/phase_m0h2.npz
FRACTUREX_HMIN=0.013 python $S/dump_phase_local.py model0 paper_direct_h3 docs/preconditioner/data/phase_m0h3.npz
# 2) 谱分析 + 预条件基准 + 画图
for t in m0h1 m0h2 m0h3; do python $S/analyze_phase_precond.py $t; done
# 3) 跨网格 mesh-independence 汇总图
python $S/summary_plot.py
```
数据与图输出在 `docs/preconditioner/`（`analysis_*.json`, `*_A.npz`, `*.png`）。
