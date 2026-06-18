# D13 实现文档：学习式增广粗空间（coarse_space_enrich）

> **统一实现文档**：D13 学习粗空间路线的设计 / 接缝 / SPD 安全推导 / 测试 / 结果**全部汇总在本文件**，随实现持续追加。规划与理论权威见
> [D13_LEARNED_PRECONDITIONER_PAPER_PLAN.md](D13_LEARNED_PRECONDITIONER_PAPER_PLAN.md)（§4 理论、§13.1 模块）；
> 数据管线 L1 与本文上游见 [archive 会话记录](../archive/d13_learned_precond_session_2026-06-09.md)。
> 规范：遵守 [多后端编码规范](../architecture/multibackend_convention.md)；求解器边界保持 numpy/scipy（豁免）。

> # ⛔ 状态：D13 学习增广线**封存确认**（2026-06-09，h2-vs-h3 干净判决，见 §10.3d）。
> **判决数据**（同 d12_recheck、同 maxit、确定性 seed）：局部化 niter — h2(48k): r60=173/r300=**13**；
> h3(184k): r60=**113**/r300=**12**。**关键：restart=300 下 niter 既有界(O(13)) 又 dof-无关(13≈12，3.8×dof)，
> 且 niter 小→实际 Krylov 基用不满 restart→内存也小(h3 仅 0.81GB)。** §10.0 担心的"大规模 restart 内存爆"被否掉
> (niter~13 根本用不满 restart=300 的基)。**三者(有界/dof-无关/小内存)足够 restart 免费全拿 ⇒ 学粗空间无价值空间。**
> 价值并入 D12(restart 再诊断 + pyamg-RNG 方法学 + §5.2b 修订)；学习机制(命题0/4 + deflation + 35 测试)封存为干净负结果。
>
> 历史状态（封存前）：机制层已实现 + 验证。机制经谱实验从加性 Galerkin **改为 deflation**（加性不能单调降 κ，§6.2）。
> **已验证（成立）**：解不变性 1.65e-10（命题 4）、deflation 杀对比度依赖 κ（命题 6 谱证据）、worst-mode 降合成 κ。
> **⚠️ 已撤回（§6.4/§6.5）**：§6.3 的「真实算子 −11%」是 pyamg-RNG 噪声（同算子同进程 baseline 150/178）。
> 噪声根因=pyamg SA 消费全局 RNG，已修（测量脚本每次 solve 前 seed）。**控噪后真实结论（§6.5）**：
> seed 后 baseline 确定性 173/173；k=1 增广 niter 仅 −1.7%（worst-mode）/−0.6%（启发式）——**单模态增益可忽略**。
> **关键负结果**：「学习幅度×单模板」本质 rank-1，无法修复 ~149 维界面坏子空间 → **须 k≫1 多模态**（命题 6 GenEO 方向）。
> 机制（deflation/SPD/解不变 1.65e-10）全部已验证正确。
> **★★ 头条结论（§9，2026-06-09）**：用户目标"求解慢、迭代多"的**主因是 GMRES restart 太小**，不是预条件子。
> 真实局部化算子 niter：restart=60(生产默认)→**173**；restart=300→**13**（13× 少迭代，2× 墙钟）。
> **"局部化 O(100)" 主要是 GMRES(60) 在非正规鞍点上的 restart 停滞人工产物**——预条件子本身只需 ~13 iter。
> **直接修复：`FRACTUREX_GMRES_RESTART=300`。** 这把 D13 整条线攻的"O(100)"用一个 trivial 旋钮解决了，
> 也意味着 **D12 §5.2b 头条"localized O(100)"是 restart=60 人工产物，需大修（待 D12 配置复核）**。
>
> **D13 增广（次要）**：deflation 在 restart=200 上仅 28→20；修正 k-scan（§6.9）已判 deflation = constant-factor。
> **已坐实硬结果**：命题4 SPD安全(解不变1.65e-10)、deflation 杀合成对比度κ、命题0障碍、pyamg-RNG 噪声方法学、
> **restart 主因发现**（最有用的实用产出）。

---

## 0. 目标与范围

把辅助粗校正的几何纯 P1 延拓 \(PI_s\) 增广为 \(R_\theta=[PI_s\mid\Phi_\theta]\)，使裂纹局部化
regime 的 GMRES niter 从 O(100) 降回 O(10)（regime 改变），同时**无条件**保持 SPD / 正确性
（计划命题 4 推广）。本阶段（L2）只做**接缝 + SPD 安全的 Galerkin 增广机制**，Φ_θ 先用固定
跳变模板 × 标量幅度（学习幅度留到 L2 后半）。

不做：端到端可微求解器（A2）；effective_stress 分支（§6 备选）；B2 完整界面自适应延拓。

---

## 1. 关键事实（已核验，决定设计）

| 事实 | 值 / 出处 | 设计含义 |
|---|---|---|
| 损伤空间 | P2（`damage_p=2`，`space_d=LagrangeFESpace(mesh,1or2,ctype="C")`，model0 用 P2） | 特征按 FE 求值取节点值（已在 coarse_features 处理） |
| 位移标量空间 | \(p-1\) 阶、`ctype="D"`（DG），p=3→**P2 DG** | 位移细自由度 sgdof ≠ NN |
| P1 粗空间 | `LagrangeFESpace(mesh,1)`，cgdof = NN（顶点） | **Φ_θ 的列也住在这个 P1 粗空间**（NN 维），与逐节点特征同维 |
| PI_s | `(sgdof, cgdof)`，重心坐标插值，`_get_or_build_auxspace_pi_operators` 缓存 | 增广在**粗端**（cgdof 维）拼列，PI_s 不动 |
| 粗校正作用方式 | 每位移分量 i：`crm = PI_s_T @ r[i-block]; X = ml.solve(crm,V); e[i-block] += PI_s @ X` | 增广在 ml.solve 这一层旁加 Galerkin 校正（见 §3） |
| Schur \(\widehat S\) | `S_hat`（auxspace）/ `S`（fast），scipy CSR，SPD | Galerkin 粗矩阵 \(\Phi^\top \widehat S_{blk}\,\Phi\) 在每位移分量的 Schur 块上构造 |
| 两个插入点 | fast: `_coarse_add_residual`/`_coarse_from_rhs`（L1336-1352）；auxspace: 粗校正循环（L1740-1748） | 二者都按 gdim 分量循环，结构一致，可共用一个增广 helper |

> 维度澄清：Schur \(\widehat S\) 作用在**位移**空间（nu = gdim·sgdof）。现有粗校正把每个位移**分量块**
> （sgdof 维）经 \(PI_s^\top\) 限制到 P1 粗空间（cgdof=NN 维）解 Laplacian 再 \(PI_s\) 延拓回去。
> 因此 Φ_θ 自然定义在 **P1 粗空间（NN 维）**，与逐节点特征 \(\phi\) 一一对应；延拓回细空间用同一个 \(PI_s\)。

---

## 2. SPD 安全的 Galerkin 增广（数学，对应计划命题 4 推广）

### 2.1 现有两层粗校正（单个位移分量）

记单分量 Schur 块 \(\widehat S_b\)（sgdof×sgdof，SPD），残差 \(r_b\)。现有校正：

\[
e_b \mathrel{+}= PI_s\, M_c^{-1}\, PI_s^\top r_b,
\qquad M_c = \text{P1 (加权) Laplacian (cgdof×cgdof)},
\]

\(M_c^{-1}\) 由 pyamg V-cycle 近似（`ml.solve(...,maxiter=1,cycle="V")`）。这是 fictitious-space
两层校正，几何纯 P1（命题 0 的 \(\eta(V_H)\) 由此封顶）。

### 2.2 增广：把 Φ 拼进粗空间做**联合** Galerkin 求解（修订，2026-06-09）

> **⚠️ 重大修正（L2-β 谱实验发现）**：首版（L2-α）把增广做成**加性**校正
> \(e_b \mathrel{+}= W S_\Phi^+ W^\top r_b\)，与几何 P1 校正并列叠加。谱实验证明这**不能单调降 κ**
> ——加性两个 Galerkin 投影会**双计数重叠模态、过冲**。实测（高对比度链 + runtime 的乘性 GS-coarse-GS
> cycle）：baseline κ=1.606；加性增广 κ=1.730（**变差**）；而把 Φ **拼进粗空间** \(R=[PI_s\mid\Phi]\)
> 做**一次**联合 Galerkin 求解 κ=1.494（**变好**），随机列 κ=2.230（无效）。故改回计划原版
> \(R_\theta=[PI_s\mid\Phi_\theta]\) 的**联合粗空间**构造，而非加性旁路。

设学习增广列 \(\Phi\in\mathbb R^{N\times k}\)（N=NN，k 小，1–4），延拓到位移分量细空间，与几何 P1
延拓**拼成增广延拓** \(R := [\,PI_s \mid PI_s\Phi\,]\)（等价地在粗端拼 \([\,I_{NN}\mid\Phi\,]\)）。
**增广粗矩阵**（联合 Galerkin）

\[
\boxed{\,S_R := R^\top \widehat S_b\, R\,},\qquad
e_b \mathrel{+}= R\, S_R^{+}\, R^\top\, r_b \quad(\text{替换 §2.1 的纯 }PI_s\text{ 粗校正}).
\]

即增广是**扩大粗空间**、做单次粗求解，不是在几何粗校正之外再加一项。κ 的度量必须用 runtime 的
**乘性 GS-coarse-GS** cycle（加性 Jacobi+coarse 度量会误判：见 §6.2 谱实验记录）。

### 2.3 SPD 安全性（命题 4 推广，本文要点）

1. \(\widehat S_b\succ0\Rightarrow S_\Phi=W^\top\widehat S_b W\succeq0\)（congruence）；列满秩时严格 SPD。
2. \(W\) 列与 \(PI_s\) 几何列可能数值相关 ⇒ 用 **pseudo-inverse \(S_\Phi^{+}\)**（小 k×k，`scipy.linalg.pinvh`）
   或 \(+\epsilon I\) 正则，使校正 well-defined 且半正定。
3. 加性叠加两个半正定校正仍半正定；外层块三角 \(\mathcal P\) 非奇异不变。
4. **右预条件 GMRES 解集不变** ⇒ 正确性与 \(\Phi\)（含任意学习参数）**无关**；最坏情况 Φ 无用，退回 §2.1。

> 工程上 k 极小（≤4），\(S_\Phi^{+}\) 是 k×k 稠密求逆，可忽略；\(W=PI_s\Phi\) 每分量一次稀疏乘。
> 推理（生成 Φ）只在 setup 一次（计划 §3.3），不进 GMRES 内循环。

---

## 3. 程序接缝设计

### 3.1 新模块 `fracturex/ml/coarse_space_enrich.py`

纯计算 + bm 合规，**不 import 求解器**（被求解器 import，不反向）。核心对象：

```
EnrichmentOperator                      # 持有 Φ (NN×k) + 每分量预算好的 Galerkin 校正
  .from_modes(Phi, PI_s, schur_block_apply, *, gdim, sgdof, reg=...)
                                        # 预计算 W=PI_s@Φ, S_Φ=Wᵀ S_b W, S_Φ⁺；缓存
  .apply_add(r_block_i, e_block_i, comp)# e += W S_Φ⁺ Wᵀ r  （单位移分量）
build_jump_template_modes(features, ...) # 固定跳变模板：沿 |∇g| 大的界面节点造单侧 bump 列
scale_modes(template, amplitude)        # Φ = template ⊙ 学习/固定幅度（逐节点标量）
```

- `schur_block_apply`: 一个 `matvec`（\(\widehat S_b @ v\)），由求解器传入（fast 传 `S`，auxspace 传 `S_hat`）；
  避免模块依赖具体 Schur 构造。
- Φ、W、S_Φ⁺ 在 setup 期预算一次，存进 `EnrichmentOperator`；GMRES 每次 apply 只做 W/Wᵀ 乘 + k×k。

### 3.2 求解器注入点（唯一侵入，与 `interface_aware` 平行）

`solve_huzhang_block_gmres_fast` 与 `_auxspace` 各加一个可选参数：

```
learned_coarse_provider=None     # callable() -> EnrichmentOperator | None
```

- provider 非空 ⇒ 在粗校正阶段，几何 P1 V-cycle 之后**加** `enr.apply_add(...)`（§2.2 加性）。
  - fast：在 `_coarse_add_residual` / `_coarse_from_rhs` 的分量循环里追加。
  - auxspace：在 L1740-1748 分量循环里追加。
- provider 为空 ⇒ 现有代码路径**逐字节不变**（零回归）。
- 缓存：EnrichmentOperator 随 d 变（每 staggered 步重算 Φ），用独立缓存键（含 model 版本 + step），
  不污染现有 `_AUXSPACE_*` / `_FAST_*` 几何缓存（PI_s、Schur 几何块照常复用）。

### 3.3 provider 工厂（连接 ml ↔ solver，但 solver 不 import torch）

`learned_coarse_provider` 由调用方（脚本/driver）构造并传入；它内部可含 torch 模型，但**只在 setup
前向一次**生成 Φ（numpy/bm），打包成 `EnrichmentOperator`。求解器只见 `EnrichmentOperator`（纯 bm/numpy），
满足「热路径零 torch」+「solver 不 import torch」。

---

## 4. 推进顺序（决定 B：先机制后学习）

### 阶段 L2-α：接缝 + Galerkin 机制（先做）
1. `coarse_space_enrich.py`：`EnrichmentOperator` + `build_jump_template_modes`（固定模板，无学习）。
2. 求解器加 `learned_coarse_provider` 注入点（provider 空时零回归）。
3. 用**固定/合成 Φ** 验证：
   - **正确性**：provider 开/关解一致到机器精度（命题 4：解集不变）——硬测试。
   - **SPD/数值**：S_Φ⁺ 不产生 NaN；加性校正不破坏 GMRES 收敛。
   - **机制有效性**：在真实局部化 checkpoint（step_015）上，喂一个「理想」界面模态（直接取
     \(\widehat S\) 低能模态的近似），看 niter 是否下降——证明**接缝本身能传递增益**（与学习质量解耦）。

### 阶段 L2-β：学习幅度（后做）
4. `coarse_weight_model.py` 复用：小 MLP 从节点特征 \(\phi\)（datasets 标准化）输出逐节点幅度，
   乘固定跳变模板 ⇒ Φ_θ。
5. `spectral_labels.py` + `train_coarse_space.py`：目标 A1（最小化增广后 \(\kappa(\,\cdot\,)\) / 谱分散）。
6. `precond_learned_sweep.py`：在 §5.3 留出协议上对比 baseline / 固定模板 / 学习。

---

## 5. 测试计划（全部 pytest，随实现填结果于 §6）

| 测试 | 阶段 | 断言 |
|---|---|---|
| `test_enrich_spd_safety` | α | 任意（含病态/相关列）Φ，S_Φ⁺ 有限、半正定；apply_add 不产生 NaN |
| `test_enrich_solution_invariance` | α | provider 开/关，GMRES 解一致到机器精度（命题 4 解集不变）——头号安全测试 |
| `test_enrich_zero_modes_noop` | α | Φ=0 或 k=0 时增广完全等价于现有粗校正 |
| `test_enrich_isolation` | α | import `coarse_space_enrich` 不泄漏 torch；solver 不 import torch |
| `test_enrich_mechanism_gain`（真实 ckpt） | α | 喂近似低能模态，step_015 niter 较纯 P1 下降（机制能传增益） |
| `test_amplitude_model_bounded` | β | 学习幅度有界、Φ 列范数受控；anchor 初始化 ≈ baseline |
| 留出泛化（sweep，非 pytest） | β | cross_mesh/damage niter 不退化 + 优于固定模板 |

---

## 6. 结果

### 6.1 L2-α 落地（2026-06-09，已实现 + 测试通过）

**交付物**：
- `fracturex/ml/coarse_space_enrich.py`：`EnrichmentOperator`（Galerkin 增广，pinvh+εI 保 SPD）
  + `build_jump_template_modes`（固定界面跳变模板）+ `scale_modes`（幅度调制）。纯 bm/numpy 边界，
  不 import 求解器/torch。
- `fracturex/utilfuc/linear_solvers.py`：`solve_huzhang_block_gmres_fast` 与 `_auxspace` 各加
  `learned_coarse_provider=None`（与 `interface_aware` 平行）；模块级 helper `_build_coarse_enrichment`
  把 provider 产出的 Φ(NN×k) 绑定到当前 Schur 的逐分量块。**provider=None 时现有路径不变**。
  - provider 契约：`callable() -> Phi (NN, k) | None`（不返回 EnrichmentOperator，绑定在求解器侧做，
    因为 Schur 块是 per-solve 的）。

**单元测试**（`test_coarse_space_enrich.py`，7 例全过，0.2s）：
- `spd_safety_wellconditioned` / `rank_deficient_columns`：任意（含秩亏/重复列）Φ，pinvh 有限、对称 PSD、
  apply_add 无 NaN。
- `galerkin_correction_is_projection_like`：单模态下 \(w^\top S e = w^\top r\)（粗残差被消去，reg=0 验证一致性 rel 1e-8）。
- `zero_modes_is_none`：Φ=0 或 k=0 → None（no-op）。
- `template_and_scale_helpers` / `template_no_band_returns_zero`：模板只在 gradd 大的界面带非零、按 d 中位数定侧、幅度可缩放。
- `isolation_no_torch_no_solver`：import 不泄漏 torch/solver。

**集成测试**（`test_coarse_space_enrich_integration.py`，真实局部化算子 model0 step_015，maxd=0.998，
σ-dof 48k / total-dof 82k，2 例全过，19m15s）：

| 测试 | 结果 |
|---|---|
| **解不变性（头号安全测试）** | enrichment ON vs OFF：`|x_on−x_off|/|x_off| = 1.78e-10`（机器精度）；两者均收敛（relres off 9.85e-9 / on 5.93e-9）。**命题 4 推广成立：接缝绝不改解。** |
| niter（off / on） | 170 / 165（固定模板的小幅增益；学习 Φ 留 L2-β） |
| 不破坏收敛 | enrichment ON 仍收敛、niter≤400 有界 |

**结论（L2-α）**：SPD 安全的 Galerkin 增广接缝**正确性已坐实**（解不变到 1.78e-10）、**SPD 数值稳健**
（秩亏列不炸）、**零回归**（provider=None 路径不变）。固定模板只给 constant-factor 小增益（170→165），
符合预期——L2-α 的目标是**证明接缝能安全传递增益**，而非模板质量。把 O(100)→O(10) 的真正增益交给
L2-β 的学习 Φ（学习幅度 × 模板，目标 A1 谱优化）。

> 诚实标注：本轮固定模板增益微弱（−3%），**不能**据此判断学习路线成败；模板是占位、解耦测试用。
> 机制有效性的强证据是「解不变性 1.78e-10 + 收敛不破坏 + SPD 不炸」，即接缝本身正确。

**全 ml 单元测试**（coarse_features 8 + datasets 9 + enrich 7）= **24 例全过**（0.68s）。

### 6.2 L2-β：谱实验纠错 + deflation 改造（2026-06-09）

**关键发现（谱实验，改变了机制设计）**：L2-α 的**加性** Galerkin 增广被证明**不能单调降 κ**。

- 谱实验（高对比度 1D 链 + runtime 的乘性 GS-coarse-GS cycle，`test_coarse_learn_l2beta.py`）：
  - baseline（纯 P1 粗校正）κ = 1.606；
  - **加性**增广（L2-α，`e += W S_Φ⁺ Wᵀ r` 并列叠加）κ = **1.730（变差）**——两个 Galerkin 投影双计数重叠模态、过冲；
  - **联合粗空间** `[PI_s|Φ]` 单次 Galerkin κ = 1.494（变好），随机列 κ = 2.230（无效）；
  - **deflation**（投影：`Q r + (I−QS)base⁻¹(I−SQ)r`）κ = 1.508（worst-mode），随机列 2.316（无效）；
  - **deflation 杀对比度依赖**：contrast 1e3→1e5，baseline κ 1.607 几乎不变、deflation κ 1.509 几乎不变且更低
    ——即**对比度无关**，正是计划命题 6 的目标图像。
- 另一处纠错：κ 度量原用**加性 Jacobi+coarse**，与 runtime 不符；改为 runtime 的**乘性 GS-coarse-GS** cycle 才不误判。

**决定（用户拍板）**：增广机制从加性 Galerkin 改为 **deflation**（投影），对齐 GenEO/deflation 文献、更鲁棒
（wraps 任意 base 预条件子）。`EnrichmentOperator` 重写为 deflation：缓存 `W=PI_s·Φ`、`SW=S_b·W`、
`(WᵀS_bW)⁺`，提供 `apply_deflated`（单分量）与 `apply_deflated_full`（包整段 Schur 解）。

**求解器接缝改造**：
- fast：把 `pre_of_S` 用 `enr.apply_deflated_full(r1, _pre_of_S_base, sgdof)` 包起来；
- auxspace：把内联 Schur 解抽成 `_schur_solve_base(rhs)` 闭包，再 deflation 包裹。
- provider=None 时两者均零回归（已验）。

**交付物（L2-β）**：
- `spectral_labels.py`：`ideal_interface_amplitude`（监督标签，高对比度签名）+ `make_two_level_minv`（deflation 版 κ 度量）
  + `two_level_kappa_dense` / `power_lambda_max`。
- `coarse_weight_model.py`：bounded MLP（`w_min+(w_max−w_min)·sigmoid`，标准化烘进 buffer，torch 仅此 + 训练脚本）。
- `train_coarse_space.py`：监督回归到 ideal amplitude（A1 稳定变体，不反传特征值分解）。

**测试**：
- `test_coarse_learn_l2beta.py`（7 例全过）：ideal amplitude 范围/局部化、**worst-mode deflation 降 κ**、
  **随机列不降**（增益来自瞄准坏方向而非加列）、模型有界/存取、训练 smoke loss 下降、隔离。
- `test_coarse_space_enrich.py`（7 例，deflation API）：SPD 安全（含秩亏列）、**deflation 消粗残差**
  （`Wᵀ(r−S_b e)=0`）、零模态 no-op、模板、子进程隔离。
- 全 ml 单元 = **31 例全过**（2.73s）。

**集成测试（真实局部化算子 model0 step_015，deflation，15m55s，2 例全过）**：

| | 加性（L2-α） | **deflation（L2-β）** |
|---|---|---|
| 解不变性 \|x_on−x_off\|/\|x_off\| | 1.78e-10 | **1.65e-10** ✓ |
| niter off→on（同一固定模板 Φ） | 170→165（−3%） | **173→153（−12%）** |

- **结论**：同一个 crude 固定模板，deflation 取得 4× 的增益（−12% vs −3%）且解不变性保持机器精度
  ——证明 deflation 改造正确。真正的 O(100)→O(10) 增益待学习幅度瞄准 worst-mode（训练管线已端到端跑通：
  真实特征→split→训练 loss 0.267→0.002→provider 出合法 Φ）。

> 诚实标注：−12% 仍是 constant-factor（固定模板表达力有限）；deflation **杀对比度依赖**的谱证据（1e3/1e5 同 κ）
> 是命题 6 的强支撑，但真实算子上压回 O(10) 需 (1) 学习幅度 + (2) 可能 k>1 多模态/界面自适应模板（L2-β 续 / B2）。

**端到端训练 smoke（真实 dump 特征）**：train(maxd=0.31)→test(maxd=0.998)，loss 0.267→0.002，
amplitude∈[0,0.935]，Φ 合法。L1+L2-α+L2-β 管线全闭合。

### 6.3 L2-γ：worst-mode 谱标签 + 真实算子 niter 验证（2026-06-09）

**动机**：L2-β 的标签 `ideal_interface_amplitude` 是**纯特征启发式**，不知道真实谱 worst-mode。新增
`worst_mode_amplitude(S_b, PI_s)`：对 baseline 两层**误差传播算子** \(E=I-M_0^{-1}S_b\) 做幂迭代（仅算子作用、
不稠密化）取主特征向量（最慢收敛模态 = 增广该消的方向，命题 0/6），\(PI_s^\top\) 限制到 P1 粗空间、
逐节点幅度归一化为标签。

**谱代理度量的教训（诚实记录）**：试图用离线 two-level κ 在真实 17208×17208 Schur 块上比较标签，
但 `eigsh` 在**非对称** \(M^{-1}S_b\) 上算最小特征值失败（lmin 触 1e-30 floor）——与 D12 §5.6 的
「ARPACK SM 不可靠」一致。**结论：κ 代理在真实非正规算子上不可信，决定性度量是真实 GMRES niter**
（与论文措辞 §4 开头一致）。合成链上 κ 代理也时好时坏：worst-mode 只在 baseline 真卡时才有增益
（contrast=1e3 有空间 κ 1.606→1.508；本测试链已被几何粗+光滑子处理好，无增广空间）——再次印证须上真实算子。

**真实局部化算子 niter（model0 step_015，maxd=0.998，σ-dof 48092，total-dof 82508，q=5，rtol=1e-8）**：

| 策略 | niter | 收敛 | t_solve(s) |
|---|---|---|---|
| baseline（无增广） | 175 | ✓ | 485 |
| 启发式模板（heuristic×template） | 170（−3%） | ✓ | 465 |
| **worst-mode 谱标签** | **156（−11%）** | ✓ | **441** |

- **结论**：worst-mode 谱标签 niter 增益（−11%）是启发式（−3%）的 **~4×**，全部收敛——**坐实学习方向**：
  标签必须连真实谱 worst-mode，而非特征启发式。且这是 **k=1 单模态**，k>1 多模态应进一步压低。
- 脚本 `scripts/paper_precond/compare_enrichment_niter.py`（真实算子 niter 对照，~22min 三解）。
- 单元测试 `test_worst_mode_amplitude_label_shape_and_range`（标签形状/范围）。全 ml 单元 = **32 例全过**。

> 诚实标注：−11% 仍是 constant-factor（单模态 k=1，固定模板调制）。压回 O(10) 的路径：
> (1) 学习模型回归 worst-mode 标签；(2) k>1 多界面模态；(3) 可能 B2 界面自适应延拓。

> **⚠️ 重大修正（§6.4，2026-06-09）**：§6.3 的 −11% 是**单次运行、未控噪声**，**不成立**。见 §6.4。

### 6.4 niter 噪声地板 — §6.3 结论被推翻（2026-06-09，诚实纠错）

为验证 §6.3 的 −11%，做了控噪实验，结果**推翻了 §6.3**：

| 运行 | 配置 | baseline | enrich(worst) | 备注 |
|---|---|---|---|---|
| compare（§6.3）| OMP=2 | 175 | 156（−11%）| 单次 |
| validate（学习幅度）| OMP=2 | 160 | direct 168 / learned 169（**+5%**）| 单次 |
| **noise-floor** | **OMP=1** | **150（run1）/ 178（run2）** | direct 166 / 178 | **同算子同进程背靠背** |

- **决定性发现**：noise-floor 测试里 **baseline 同一确定性算子、OMP=1、同进程、背靠背两次 = 150 vs 178
  （差 ~19%）**。这**不是** OMP 线程问题（已 OMP=1），是**求解器在 maxd=0.998 局部化算子上本质 niter 噪声**
  ——该算子贴着收敛边缘（D12_RESULTS §5.2b 早记录 niter 95–200、偶触 maxit），niter 对极微扰动（cache 态、
  restart 周期边界）超敏感。
- **结论**：**§6.3 的 −11% 低于噪声地板（±~20，~15%），不可信。** 单一最难 checkpoint 上单次 niter
  无法分辨增广增益。L2-γ「worst-mode 4× 优于启发式」**撤回**。
- **两个待修问题**：
  1. **测量方法**：必须控噪——换稍低 maxd 的稳定工作点 / 多 checkpoint 平均 / 多次重复取统计 / 找确定性求解配置。
  2. **label 对象错配**：`worst_mode_amplitude` 算的是**加性** `make_two_level_minv` 代理的 worst mode，
     但 runtime 是**乘性 GS-coarse-GS**——应让 label 对齐乘性 cycle 的误差传播算子。
- 这是诚实纠错：把"看起来 work"的噪声当结论是 D13 最该避免的（参考 memory 多条"别当数据"教训）。
  机制层证据（解不变性 1.65e-10、deflation 杀对比度 κ、worst-mode 降合成 κ）仍成立；**真实算子上的 niter 增益尚未被可信测得**。

**噪声根因（已确认，可控）**：`pyamg 5.3.0` 的 `smoothed_aggregation_solver` **消费全局 numpy RNG**——实测
`np.random.seed` 不同 → SA 层级不同 → 解不同。背靠背两次 solve 间全局 RNG 状态前移（pyamg 自身抽 + GMRES），
故 baseline_1≠baseline_2。**修法（仅在 D13 测量脚本，不动生产求解器以免扰动 D12）**：每次 solve 前
`np.random.seed(SEED)` + 清 pyamg 缓存，使 baseline/heuristic/worst-mode **共享同一 SA 层级实现**、只差增广
→ 公平且确定性对比。脚本 `scripts/paper_precond/compare_enrichment_seeded.py`。

### 6.5 控噪后的真实结论 — k=1 单模态增益微乎其微（2026-06-09）

确定性对比（seed=12345、清 pyamg 缓存、OMP=1，`compare_enrichment_seeded.py`）：

| 策略 | niter | vs baseline |
|---|---|---|
| baseline（重复两次） | **173 / 173** | — （**确定性已坐实**）|
| 启发式模板 | 172 | −0.6% |
| worst-mode 谱标签 | 170 | **−1.7%** |

- **确定性达成**：seed 后 baseline 两次都 173（之前 150/178/160/175 全是 pyamg-RNG 噪声）。测量基础现在可信。
- **真实结论**：k=1 单模态增广（无论启发式还是 worst-mode）在真实局部化算子上 niter 增益**微乎其微**
  （−1.7% / −0.6%）。**§6.3 的 −11% 彻底是噪声。** worst-mode 比启发式略好（170 vs 172）方向正确，但量级可忽略。
- **根因分析（重要，决定下一步）**：裂纹界面跨 ~149 个 dof（§6.1），其低能/坏模态子空间是**高维的**
  （维数 ~ 界面节点数）。而「学习幅度 × 单模板」本质 **rank-1**（一个模板列按节点缩放仍是一列）
  → 只能提供**一个**增广方向 → 无法修复 ~149 维坏子空间。这解释了 −1.7%，且与命题 6（η→1 须张成坏子空间）一致。
- **结论**：**单模态 k=1 路线（首版设计 A）不足以压回 O(10)，需 k≫1 多模态增广。** 机制（deflation/SPD/解不变）
  全部正确且已验证，缺的是**增广基的维数与构造**：需 k 个不同的界面模态（如沿裂纹分段的局部 bump，
  或直接 top-k worst modes / GenEO 谱粗空间），而非单模板缩放。k 个模态的 deflation 成本仍低
  （(k·gdim)² 稠密解 + k 次稀疏 matvec，k~150 完全可行）。
- 诚实定位：本轮把"看起来 −11%"证伪为噪声、定位到 rank-1 这一**结构性瓶颈**，是 D13 的关键负结果——
  它把路线从"调模板/学幅度"导向"多模态/谱粗空间"（更接近命题 6 上档的 GenEO 正解）。

### 6.6 多模态 k-scan（GenEO）— 仅 constant-factor，未达 regime change（2026-06-09，关键负结果）

`top_k_worst_modes`：对误差传播算子 \(E=I-M_0^{-1}S_b\) 做 block 子空间迭代取 top-k 特征向量，作为**细空间
（sgdof）** deflation 基（genuine out-of-V_H，`EnrichmentOperator` 自动识别细模态不做 PI_s 延拓）。确定性
k-scan（seed=12345、清缓存、真实 step_015，`kscan_worst_modes.py`）：

| k | niter | vs baseline |
|---|---|---|
| 0（baseline） | 173 | — |
| 1 | 160 | −7.5% |
| 4 | 155 | −10.4% |
| 16 | **144** | **−16.8%（最佳）** |
| 32 | 171 | −1.2% |

- **结论（关键负结果）**：多模态 deflation **未把 O(100) 压回 O(10)**。最佳 k=16 仅 **−17%（173→144）**，
  且 **k=32 反弹回 173**（疑因：32 维子空间迭代 40 步对尾部模态未收敛 → 模态 17–32 是噪声污染 deflation；
  或加性 M0 代理的尾部 worst mode 偏离真实乘性 cycle）。即 **constant-factor，非 regime change。**
- **与 D12 B1 横比（重要）**：B1 界面感知**权重** −28%（170→123），本 GenEO **多模态 deflation** −17%
  ——**deflation 多模态并不优于 B1 权重**，两者同为 constant-factor。注：B1 的 −28% 当时也未控 pyamg-RNG 噪声，
  真实差距待同口径重测；但量级一致——**「学粗空间」目前并未跑赢「学权重」**。
- **诚实总结（D13 现状）**：机制层全部正确且验证（解不变 1.65e-10、SPD 安全、deflation 杀合成对比度 κ）；
  但在**真实完全局部化算子**上，无论 k=1 还是 k≤32 多模态，**只拿到 constant-factor（≤−17%），命题 6 的
  η→1 / O(100)→O(10) 在真实算子上尚未兑现**。瓶颈疑在：(a) 模态须对齐**乘性** cycle 而非加性代理；
  (b) 子空间迭代对大 k 需更多步；(c) 真实坏子空间可能须 k≫32（接近界面 dof 数 ~149）且 deflation 之外
  还需 B2 界面自适应延拓。**这是决定 D13 论文定位的节点**（见 §8）。

---

### 6.7 诊断：谱标签建在**发散代理**上（关键根因，2026-06-09）

离线诊断（`/tmp/ritz_diag.py`，无 GMRES，真实 step_015 Schur 块）：误差传播算子 \(E=I-M_0^{-1}S_b\) 的
top-48 Ritz 值（子空间迭代 40 步与 120 步**一致**，已收敛）**全部 ≈ 2.0–2.3（≫1）**。

- **含义**：收敛迭代要求 \(\rho(E)<1\)；此处 \(|\lambda(E)|\approx2.3\) ⇒ **加性 `make_two_level_minv`（ω=1 Jacobi+coarse）
  在该 Schur 块上是发散迭代**。（一致：早测 \(\lambda(M_0^{-1}S)\in(0,3.34]\) ⇒ \(\lambda(E)=1-\lambda(M^{-1}S)\in[-2.34,1)\)。）
- **致命后果**：`worst_mode_amplitude` / `top_k_worst_modes` 取的是**发散算子 E 的主特征向量**——它们是
  \(\lambda(M^{-1}S)\approx3.3\) 的**过校正**方向，**不是** runtime **乘性 GS-coarse-GS**（收敛、~173 iter）的真实坏模态。
  **谱标签机器建在错的（发散的）代理算子上。**
- **§6.6 的 −17% 重新解读**：不是"多模态 GenEO 只值 −17%"，而是"**用错模态（来自发散代理）**只值 −17%"。
  真实 GenEO（用乘性 cycle 的收敛误差传播算子的坏模态）**尚未测过**。
- **修法（明确、可执行）**：M0 必须收敛——(a) 加性需阻尼 \(\omega<1/\lambda_{\max}\)；或更对的 (b) 直接用
  runtime **乘性 GS-coarse-GS** 的误差传播 \(E_{\text{mult}}=I-B_S S\)（\(B_S\)=一次 pre_of_S）做子空间迭代取坏模态。
  \(E_{\text{mult}}\) 的 \(|\lambda|<1\)，其主模态才是该 deflation 的真实目标。
- **结论**：D13 **未被证伪**——谱标签的代理算子发散是个 bug，不是路线死刑。route B（对齐乘性 cycle）从
  "可选优化"升级为"**必做的根因修复**"。

### 6.8 修复（omega 阻尼）+ 修正 k-scan 状态（2026-06-09）

- **修复已落地**：`worst_mode_amplitude` / `top_k_worst_modes` 默认 `smoother_omega=0.5`。离线验证（真实 step_015）：
  omega=1.0 时 \(|\lambda(E)|=2.31\)（发散，24/24 模态 >1）；omega=0.5 时 \(|\lambda(E)|=0.996<1\)（收敛，0 模态 >1）
  → worst modes 现在是收敛代理的真实坏模态。单元测试加 `test_top_k_worst_modes_orthonormal_and_convergent_M0`
  （断言正交 + 收敛性 \(\rho(E)<1\)）。全 ml 单元 **35 例全过**。
- 性能优化：`make_two_level_minv` 保持 `PI_s` 稀疏（去掉 212MB 稠密化 + 内存带宽争用），mode-building 在真实
  17208 块上 **k=64 仅 10s / 0.24GB**（之前内存带宽受限）。数值不变（11 测试通过）。

### 6.9 修正 k-scan 决定性结果 — 仍 constant-factor，无 regime change（2026-06-09，最终判定）

修复后确定性 k-scan（omega=0.5 收敛模态、seed=12345、nice、真实 step_015，`kscan_worst_modes.py 0,1,16,64`）：

| k | niter | vs baseline | 对比 buggy(ω=1) |
|---|---|---|---|
| 0（baseline） | 173 | — | 173 |
| 1 | 155 | −10.4% | 160 |
| 16 | **142** | **−17.9%（最佳）** | 144 |
| 64 | 146 | −15.6% | 171（曾反弹）|

- **修复确有效**：修正模态比 buggy 略好（k=1：155 vs 160），且**消除了 k=32/64 反弹病态**（146 稳定，非 171）。
- **但根本结论不变且现已 robust**：即便用**收敛代理的正确 worst modes**，deflation 增广在真实完全局部化算子上
  **仅 constant-factor（最佳 −18%）且 k=16 后饱和**（142→146，加模态无further增益）。**未达 O(100)→O(10) regime change。**
- **饱和的可能根因**：(a) omega=0.5 加性代理虽收敛，仍**不是 runtime 乘性 GS-coarse-GS 的真实 worst modes**
  （§6.7 残留 proxy-vs-multiplicative gap）；(b) 更可能——局部化难度**不是低维坏子空间**，而是几何纯 P1 延拓
  **结构性**无法表示界面（命题 0 的本质），deflation（只加列、不改 PI_s）治标不治本，须 B2 改延拓本身。
- **横比 D12 B1 权重 −28%**：deflation 多模态 −18% **仍未跑赢 B1 权重**（注：两者口径下噪声待统一，但量级一致）。
- **最终判定（诚实）**：**「学粗空间 / deflation 增广」与「学权重 / B1」在真实局部化算子上同为 constant-factor
  （−18% vs −28%），都未兑现 regime change。** 命题 6 的 η→1 / O(10) 在真实算子上**未达成**。机制层（命题 4 解不变、
  deflation 杀合成对比度 κ、命题 0 障碍）全部正确——这是 D13 可写论文的硬核，但**真实加速是 constant-factor**。

---

## 8. D13 现状评估与路线决策点（2026-06-09，已由修正 k-scan 判定）

经 L2-α/β/γ + 控噪 + 多模态 k-scan，**诚实盘点**：

**已坐实（可写进论文的硬结果）**：
1. 命题 4（SPD 安全 / 解不变）——真实算子 \(|x_{on}-x_{off}|/|x|=1.65\text{e-}10\)，无条件成立。
2. deflation 机制正确：合成高对比度上**杀对比度依赖**（κ 在 contrast 1e3↔1e5 几乎不变）——命题 6 的**合成**谱证据。
3. 命题 0（障碍/权重天花板）：加性增广不能降 κ、必须 out-of-V_H 模态——谱实验佐证。
4. 方法学副产物：pyamg-SA 全局 RNG 致 niter 噪声的发现 + 确定性测量协议（对 D12 重测也有用）。

**未兑现（卡点，已由修正 k-scan 定论）**：真实完全局部化算子上 **niter 仅 constant-factor（最佳 k=16 −18%）
且 k≥16 饱和（142→146），未达 O(10)**；**未跑赢 B1 权重（−28%）**。修正模态（ω=0.5 收敛）只消除了 k=32 反弹病态、
小幅改善，**未改变 constant-factor 结论**——即 deflation 路线的真实加速天花板已基本探明。

**可选路线评估**（修正 k-scan 后，推荐倾向更新）：
- **A 改写论文定位（推荐）**：D13 不卖"压回 O(10)"，改卖"**可证明安全的学习增广框架 + 命题 0/4/6 理论 +
  deflation 杀对比度的合成谱验证 + 诚实的 constant-factor 真实加速 + pyamg-RNG 噪声方法学**"。
  定位 methodology/framework（CMAME 数值验证档可接受）。卖点是**理论+安全性+诚实**，不是 SOTA 加速。
- **B 攻乘性对齐 + 大 k（性价比低）**：修正 k-scan 已显示 k=16 后饱和（142→146），**proxy-vs-multiplicative
  gap 即便修了也大概率只小幅改善**，不太可能从 −18% 跳到 O(10)。除非有强理由，否则不建议投入。
- **C 转 B2 改延拓本身（唯一可能 regime change，但成本高一量级）**：§6.9 根因分析指向"几何纯 P1 延拓结构性
  无法表示界面"，deflation 治标不治本。B2（界面自适应 P2 延拓 / 改 PI_s）是唯一可能真正 O(10) 的路径，
  但破坏几何缓存、改双求解器，D12 已判 future work，工作量大。
- **D 并入 D12（最稳）**：D12（块预条件 robust + 谱框架 + matrix-free）本身已是完整论文；D13 的机制+理论
  作为 D12 的一节"learned/spectral 增广（constant-factor，future work 指向 B2）"。最低风险、最快收口。

**专家判断**：修正 k-scan 把"D13 能否 regime change"这个核心不确定性**关闭了——deflation 路线不能**。
真正的 regime change 只剩 C（B2 改延拓），其余路线都是 constant-factor。是否值得为 B2 的高成本投入，
是论文 ROI 的战略决策（A/D 立即可收口成中等论文；C 高风险博头条）。

---

## 9. ★ 重大发现：局部化 O(100) niter 主因是 GMRES restart 太小（2026-06-09）

用户目标是"真正解决求解慢、迭代多"。系统扫**现有求解器旋钮**（并行，装配一次存盘，`_assemble_save.py`
+ `_solve_knob.py`，确定性 seed），结果颠覆性：

**真实局部化算子 step_015（maxd=0.998，σ-dof 48092），扫 GMRES restart（gs=2, schur=auto, 无增广）：**

| restart | niter | wall | 备注 |
|---|---|---|---|
| **60（D12/生产默认）** | **173** | 497s | 基线 |
| 100 | 56 | 277s | |
| 150 | 35 | 285s | |
| 200 | 28 | 312s | |
| **300** | **13** | **249s** | **最佳，13× 少迭代** |

**结论（决定性）**：**"局部化 O(100) niter" 主要是 GMRES(60) restart 太小的人工产物**，不是预条件子缺陷。
真实预条件子在该算子上只需 **~13 iter**（restart=300）；restart=60 在收敛进入超线性段前就重启、丢掉 Krylov
子空间，对**非正规**鞍点系统造成 GMRES(m) 停滞，把 13 撑到 173。restart 60→300 单调：173/56/35/28/13。

**其他旋钮（次要）**：
- gs_iterations（光滑子）：restart=60 下 gs=2/4/8 → 173/110/89，迭代降但**每迭代变贵**（gs=8 wall 929s ＞ 基线 497s）→ 不划算。
- deflation：restart=60+defl16 → 151（−13%，即 §6.x 的 D13 增益）；restart=200+defl32 → 20（vs restart=200 单独 28，−29%）。**有效但远小于 restart**。
- schur=coarse_amg_halfgs → 400 DNF（该变体在此 regime 差）。

**墙钟分解**（497=C+173p, 249=C+13p ⇒ **setup≈229s, 每迭代≈1.55s**）：wall 在 niter 小后**被 setup 主导**
（Schur 三重积 + pyamg SA + PI_s ≈229s）。故 (1) restart 把 niter 173→13 给 2× 墙钟（497→249s）；
(2) 再快需降 setup——`precond_rebuild_interval>1` 跨 staggered 步复用昂贵 setup（已是现成功能）。

### 9.1 对用户目标的直接答案

> **求解慢、迭代多 = 两个独立修复，都是现成旋钮、零新方法：**
> 1. **迭代多** → `FRACTUREX_GMRES_RESTART=300`（或 200）：niter 173→13–28（**13×**）。
> 2. **求解慢** → 上面给 2× 墙钟；再加 `precond_rebuild_interval>1` 复用 ~229s setup 跨 staggered 步。

生产 `run_case.py` 当前 `_aux_gmres_settings` 默认 `restart=60`（env `FRACTUREX_GMRES_RESTART` 可覆盖）。

### 9.2 对 D12 / D13 的冲击（诚实，重要）

- **D12 §5.2b 头条"局部化 O(100) niter"是 restart=60 人工产物 —— ✅ 已用 D12 复核坐实（见 §9.4）**。
- **D13 的前提几乎消失**：D13 整条线（学增广粗空间压回 O(100)→O(10)）攻的"O(100)"在 restart 够时本就不存在。
  deflation 在 restart=200 上仅 28→20。**D13 的卖点（解决 localization niter 爆炸）被 restart 这个 trivial 修复抢走。**
- **方法学教训（写入 memory）**：上昂贵学习预条件前，必须先扫基础 Krylov 旋钮（restart/maxit）。restart 的 13×
  完全盖过 deflation 的 −18%。这是 D13 调研最该早做、却最晚做的实验。

### 9.4 ★ D12 复核结果 — O(100) 头条坐实为 restart 假象（2026-06-09）

在 D12 §5.2b **精确配置**（model0 h₂ σ-dof 48092，真实 staggered checkpoint）上，跨 maxd × restart 扫
（`d12_recheck.py`，确定性 seed，niter/conv/wall）：

| step | maxd | restart=60(D12默认) | restart=200 | restart=300 |
|---|---|---|---|---|
| 013 | 0.426（局部化前） | 7 / Y / 20s | 2 / Y / 19s | 2 / Y / 18s |
| **014** | **0.998（局部化跃迁）** | **93** / Y / 301s | 18 / Y / 225s | **9** / Y / 175s |
| 015 | 0.998 | **173** / Y / 555s | 28 / Y / 342s | **13** / Y / 293s |
| 017 | 1.000（完全局部化） | **400 / DNF** / 1350s | 25 / Y / 343s | **14** / Y / 307s |
| 020 | 1.000（完全局部化） | **400 / DNF** / 1302s | 82 / Y / 1113s | **49** / Y / 1070s |

- **★ 决定性**：step_014 正是 D12 §5.2b 的头条 checkpoint（step13→14 局部化跃迁，D12 报 niter **7→93（O(100)）**）。
  **复核：7→93 仅在 restart=60；restart=300 下是 7→9。** 即 **D12 头条的"localization blowup to O(100)" ≈ restart=60 人工产物。**
- **maxd=1.0 更强**：完全局部化（017/020）下 **restart=60 直接 DNF（打满 maxit=400 不收敛）**，而 restart=300 收敛于 14/49。
  即 D12 若报这些状态为"DNF/不可行"，**也是 restart 假象**——预条件子收敛，只是 GMRES(60) 重启停滞致不收敛。
- **正确解读（对 D12 是好消息）**：预条件子跨完全局部化（maxd 0.998→1.0）**始终有界收敛 O(10–50)**（restart 够时），
  比 D12 宣称更鲁棒。O(100)/DNF 全是 GMRES(60) 重启在**非正规鞍点**上停滞。
- **诚实标注**：step_020（maxd=1.0 最难态）restart=300 仍需 49 iter（非 O(10)），且 restart=200→82——**最难局部化态
  对 restart 更敏感、收敛更慢**。故主张应是"**O(10–50) 有界、随 restart 单调改善**"，不是一律 O(10)。但与"O(100)/DNF"相比，
  量级与"是否收敛"都是质变。
- **D12 §5.2b 须改写**：从"局部化致 niter 爆炸 O(100)/DNF、aux 唯一有界"改为"**aux 跨完全局部化保持有界收敛
  O(10–50)，前提 restart≥200–300；restart=60 时 GMRES 重启停滞虚高到 O(100) 乃至 DNF**"。更强更干净的鲁棒性主张。
- 待补：跨算例（square/model2）复核（见 §9.7）；D12 图 1b 重做（restart-aware）。

### 9.7 跨算例 restart 复核（square / model2，进行中）

**目的**：确认 §9.4 的"局部化 O(100)=restart 假象"不是 model0 专属，而是非正规鞍点的普遍现象（跨 I 型 square、
II 型 model2 剪切）。脚本 `restart_recheck_xcase.py`（确定性 seed）。

**机器现实约束（诚实）**：square/model2 的**真实局部化 checkpoint 只有生产分辨率**（square nx=216 σ-dof **1.5M**、
model2 nx=160 σ-dof **847k**，分别是 model0 48k 的 31×/18×）。在本机 load~120 下，单个 restart=60 局部化解
（停滞 = 大量迭代 × 大 matvec）极慢，square 1.5M 装配就吃 20GB。**D12 §5.7/§5.8 的 square/model2 表用的是
合成均匀-d（restart=60 也 O(10)，无尖界面）+ 小 nx（30/32）**，故无现成的小尺度真实局部化态可复用。

**结构性论证（强，先于数值）**：restart 停滞的根因是**块三角预条件作用在非正规不定鞍点上**——这是 Hu-Zhang
混合元 + 相场退化的**通用结构**，与裂纹几何/加载模式无关（I 型/II 型同一套 `solve_huzhang_block_gmres_fast`、
同一 \(\widehat S\) 近似、同一 GS-coarse-GS）。model0 的 restart 60→300 给 93/173/DNF→9/13/14 是该结构的体现，
**square/model2 同结构 ⇒ 预期同形**。数值复核是确认而非发现。

**进行中/受阻**：model2 step_030（maxd=1.0，847k σ-dof，1.46M total）装配 32s 完成，但 restart=60 局部化解在本机
load~120 下极慢（停滞解 = 大量迭代 × 1.46M matvec），单解 >15min 未完。square 1.5M 更重已搁置。
**结论：跨算例数值复核在本共享机（高负载 + 仅生产分辨率 checkpoint）不实际**——非阻断性，因结构性论证已强。
日志 `/tmp/xcase_model2b.log`（model2 结果若跑完会落此）。**建议**：(a) 机器空闲时跑；或 (b) 生成 model0-scale
（小 nx）的 square/model2 真实局部化 checkpoint 专供此复核——但 §5.7/§5.8 现有小 nx 数据是合成均匀-d，需新跑短
staggered 到局部化。鉴于结构性论证 + model0 三 checkpoint（014/015/017/020）已充分，此复核**降级为 future confirmation**。

> **侧面佐证**：model2 847k 局部化算子 restart=60 解在本机跑 >22min 仍未收敛——这本身就是"restart=60 在局部化
> 大系统上停滞"的体现（与 model0 step_017/020 的 restart=60 DNF 同形）。即便没拿到干净 niter 数，restart=60
> 的病态在 model2 上**已定性显现**。

### 9.5 生产化验证（真实 staggered run，restart=60 vs 200，进行中）

端到端真实 staggered run（`run_case.py model0 aux`，FRACTUREX_HMIN=0.025、N_LOAD_STEPS=16 跑到局部化、
两 restart 并行），监控 `iterations.csv` 的逐步弹性 niter + 墙钟。**起裂前（maxd≤0.25）阶段性观察**：

| | restart=60 | restart=200 |
|---|---|---|
| 弹性 niter（每步峰） | 7–8 | **2** |
| 累计弹性解时间(step5) | 658s | 978s |

- **重要诚实标注（两个混淆因素）**：(1) **生产 run 不 seed pyamg**（§6.4 的全局-RNG 噪声在此活跃），故起裂前
  "7 vs 2"含 RNG 噪声成分，非纯 restart 效应；(2) **墙钟受机器负载严重混淆**（本机 load ~100，用户自跑作业 +
  本对照两 run + 他人 FreeFem 抢核），r200 累计时间更高**不能**归因于 restart——并行两 run 抢同批核。
- **故起裂前的墙钟对比不可信**；**唯一干净信号是局部化步（step~14）的 niter 大效应**（单步 7→O(100) vs 7→O(10)）。
- **方法学修正**：生产化的**正确**验证应 (a) 串行跑（不并行抢核）或独占机器；(b) 评估指标用 **niter**（restart 的
  确定性效应）而非受污染的墙钟；(c) 若要墙钟，须 seed pyamg + 控负载。本轮并行+高负载下，**结论只取局部化步 niter**。
- **墙钟在本共享机上不可测（已放弃，诚实定论）**：并行对照自抢核；改串行后**单 run 仍更慢**（step3 cum 349s vs
  并行 283s）——因本机 load≈100（用户论文作业 + 他人 FreeFem，不可控）。**串行/并行都无法在此机得到干净墙钟。**
  故生产化墙钟验证**留待独占机器或低负载窗口**；脚本与方法已就绪（串行 + seed pyamg + 控负载）。
- **★ 但科学结论不依赖墙钟、已坐实**：决定性指标是 **niter**，它**确定性、与机器负载无关**（§9.4 d12_recheck：
  restart 60→300 给 93→9 / 173→13 / 400-DNF→14）。niter 的 restart 效应是物理事实，墙钟只是它 × 每迭代成本 × 机器速度。
  **"迭代多"已被 niter 证据完全解决**；"求解慢"在干净机器上必随 niter 下降而改善（§9 墙钟分解：niter 173→13 给 2×，
  独占机实测过 497→249s）。
- **生产建议（落地）**：
  1. **maxd-自适应 restart**（推荐）：起裂前 maxd<0.9 用小 restart（niter O(2–8)，省 Krylov 基内存）；
     局部化 maxd≥0.9 切 restart=300。零侵入求解器核——在 `run_case.py` 的 `_aux_gmres_settings` 按当前 max_d 选 restart。
  2. **或全程 restart=200**（最简）：起裂前 niter 已 O(2)、restart 不触发无额外迭代成本；只多分配 Krylov 基内存
     （restart=200 × 82508 dof × 8B ≈ 132MB，2TB 机可忽略）。**最省事且稳，推荐先用这个。**
- **结论**：生产化"如何配 restart"已明确（全程 200 或 maxd-自适应）；唯一未做的是干净机器上的端到端墙钟数字，
  但那是**确认性**而非**判定性**实验——判定性的 niter 证据（§9.4）已完整。

### 9.6 生产化落地（已实现，2026-06-09）

改 `scripts/paper_huzhang/run_case.py`（零侵入求解器核，仅改 driver 层 restart 配置）：

1. **默认 restart 60→200、maxit 200→400**（`_aux_gmres_settings`）。理由 docstring 内写明（§9.4 d12_recheck）。
   起裂前 niter 已 O(2–8)、restart 不触发无额外迭代成本；只多 ~0.1GB Krylov 基内存（2TB 机可忽略）。
2. **新增 maxd-自适应 restart**（`_adaptive_restart`，opt-in `FRACTUREX_GMRES_ADAPTIVE_RESTART=1`）：
   maxd<0.9 用小 restart（默认 60，省内存）、maxd≥0.9 用大 restart（默认 300，避局部化停滞）。
   阈值/高低值均可 env 调（`FRACTUREX_GMRES_ADAPTIVE_{MAXD,HI,LO}`）。在 `elastic_solver` 闭包按 `state.d` 当前 max_d 选。
3. **验证**：单元测旋钮（默认 200/400；adaptive OFF→200 恒定；adaptive ON→maxd0.5→60/maxd0.99→300）；
   smoke run（model0 aux 新默认）step1 弹性 niter=2 收敛——launch 正常。
4. **向后兼容**：`FRACTUREX_GMRES_RESTART` 仍可覆盖（设 60 可复现旧行为）。D12 旧数据用 restart=60 生成、保持可复现。

**对用户原始诉求"求解慢、迭代多"的最终交付**：
- **迭代多**：默认 restart 200 直接把局部化 niter 从 O(100)/DNF 降到 O(10–50)（§9.4 实测 93/173/DNF→9/13/14）。
- **求解慢**：niter 降给净墙钟改善（独占机 497→249s 实测 2×）；进一步可开 `precond_rebuild_interval`（默认已 5）复用 setup。
- **零风险**：restart 只改 niter 不改解；默认值变更向后兼容（env 可还原）。

### 9.3 待验证（下一步）

1. ~~D12 复核~~ ✅ 见 §9.4（step_014 坐实）。补 015/017/020 + 跨算例。
2. **生产化**：把 `_aux_gmres_settings` 默认 restart 提到 ~200（或按 dof 自适应），核 staggered 全程 + 内存
   （restart=300 基向量 ~200MB 上限，但 niter 小则实际仅几十向量）。
3. **setup 加速**：`precond_rebuild_interval` 跨步复用在真实 staggered run 的净收益量化。
4. **D12 §5.2b 重写 + 图 1b 重做**（restart-aware 鲁棒性叙事）。

---

## 7. 风险与对策（本阶段）

| 风险 | 对策 |
|---|---|
| 固定跳变模板表达力不足，机制增益不明显 | L2-α 的「理想低能模态」测试与模板解耦：先证接缝能传增益，再谈模板/学习质量 |
| Φ 列与 PI_s 强相关致 S_Φ 病态 | pinvh + εI 正则（§2.3）；测试覆盖病态列 |
| 增广破坏几何缓存 | 独立缓存键，几何块（PI_s/Schur 几何）照常复用；provider 空零回归 |
| setup 推理吃掉增益 | k≤4、setup 一次；β 阶段净时间表量化（计划 §7.3） |

---

## 10. 封存判断的重大修正：restart 不免费,与 D12 小内存卖点冲突（2026-06-09 二次复核）

> **⚠️ 撤回前一版「终判封存」——封存的核心理由错了。** 重读 D13_PLAN 原始目标后发现:封存逻辑
> 「restart=200 解决了 O(100) → 学习线没价值」**只在小规模(model0 h2, 82k dof)成立**,在 D13/D12 真正瞄准的
> **大规模(2M+ dof)上反而站不住**。

### 10.0 致命遗漏:restart 大的 Krylov 内存代价

GMRES restart=R 的 Krylov 基内存 = **R × total_dof × 8B**。实算:

| 规模 | total-dof | restart=60 | restart=300 |
|---|---|---|---|
| model0 h2 | 82,508 | 0.04 GB | 0.20 GB ← §9 在此扫,内存差仅 0.16GB,**故没看出问题** |
| model0 h5 | ~3.4M | 1.65 GB | **8.27 GB** |
| square 生产 | 2.66M | 1.28 GB | 6.39 GB |

**D12 §5.5 卖点是 matrix-free 把 2M dof 从 direct 11.4GB 降到 4.6GB(省 6.8GB)。但 restart=300 在 3.4M dof 上
Krylov 基就要 8.2GB——直接吃掉省下的内存。** 即 **restart 大 与 D12「小内存」定位在大规模上直接冲突。**

### 10.1 正确的研究问题(restart 没有解决)

restart 大 和 学粗空间(命题0/GenEO)是**同一问题的两种解,各有内存代价**:
- **restart 大**:GMRES 用大 Krylov 子空间**隐式 deflate** 粗空间没抓住的界面坏模态 → 代价 O(restart·dof) 内存,大规模爆。
- **学粗空间**:把坏模态**显式**吸进 O(k·dof) 增广(k 小 1–4),setup 一次 → **大规模省内存**。

> **正确的 D13 研究问题(restart 未触及)**:**在大规模(2M+ dof)局部化态,学增广粗空间能否用 O(k·dof) 小内存,
> 达到 restart=300 用 O(300·dof) 大内存才能达到的 O(10) 迭代数?** 这正好落在 D12「小内存」卖点上,与 PLAN
> §4.7 命题6/GenEO 路径一致。**"restart 大能救"在机制上恰恰印证命题0(GMRES 在替粗空间补课记界面模态)。**

### 10.2 但这取决于一个未测的事实(封存→待验证)

决定 D13 大规模价值的关键事实**谁都还没测**:**真实局部化态下,固定 restart, niter 是否随 dof 涨?**

- 若 **涨**(h2→h3→h5 niter 单调升,restart 须随 dof 涨才压得住)→ 大规模 restart 内存爆 → **学粗空间真有价值,D13 复活**。
- 若 **不涨**(各档局部化 niter 都 ~O(100)@r60 / ~O(13)@r300,固定 restart=300 全规模够用)→ 学粗空间无内存优势 → 封存正确。

**现有线索矛盾/不足**:§5.2(合成均匀 d=0.9, restart=200)niter 跨 187× dof 恒 9–15(mesh-indep)——但均匀 d 无尖界面,
命题0 坏模态不存在,不能外推到局部化。§5.2b L76:h₃(184k)真实局部化 step14 峰值 **29** < h₂ 的 **93**——暗示真实
局部化 niter **可能不随 dof 单调涨**(甚至 h3 比 h2 好),但这是 restart=60 不完整数据,且与"mesh 越细界面越尖→越难"
的直觉相反,需正式测。

### 10.3 修正后的状态与下一步(封存撤回为"待一个实验判定")

- **封存撤回**:前一版「终判封存」基于"restart 解决了问题"——该理由**只在小规模成立**,在大规模(D13 真正目标)未验证甚至可能反向。
- **降级为"待关键实验"**:D13 的生死取决于 §10.2 那个未测事实。**判定实验**:真实局部化 checkpoint 跨 h2/h3/h5
  × 固定 restart(60 与 300)扫 niter,看是否随 dof 涨。
  - 数据:h2 已有(93@r60/13@r300);h3 checkpoint 存在(`paper_aux_h3/`,§5.2b 提及但数据续算中);h5 需生成。
  - 若 niter 随 dof 涨 → D13 复活,且卖点更准:**"小内存达成 restart 大才有的迭代数"**(契合 D12 §5.5)。
  - 若不涨 → 封存,且 D12 §5.2b 改写为"局部化 niter mesh-independent(restart≥200),小内存"——D13 确实多余。
- **deflation 的 constant-factor 问题(§6.9)仍在**:即便 niter 随 dof 涨,也要重新评估 deflation/学粗空间能否真在大规模
  把 niter 压到 restart=300 水平且省内存——但至少**研究问题重新成立**,不再是 solution-in-search-of-a-problem。

### 10.3b 判定实验方案（就绪待跑，2026-06-09）

**目标**：测真实局部化态 niter 是否随 dof 涨 → 判 D13 生死（§10.2）。

**实验设计**（确定性 seed，控 pyamg-RNG 噪声）：
- **dof 轴**：model0 同几何不同网格的**局部化** checkpoint：h2(σ48k，已有 step_015 maxd0.998)、
  h3(σ184k，需续算)、h5(σ2M，需续算)。
- **restart 轴**：固定 restart ∈ {60, 300} 各跑一遍。
- **测量**：每个 (mesh, restart) 的局部化算子 niter（脚本 `d12_recheck.py` 已支持，扩 mesh 参数即可）。
- **判据**：
  - niter@固定restart **随 dof 单调涨**（如 h2=93→h3>93→h5≫）→ **D13 复活**：restart 须随 dof 涨→大规模内存爆→学粗空间(O(k·dof))有内存优势。
  - niter@固定restart **dof 无关**（各档 ~O(100)@r60 / ~O(13)@r300）→ **封存**：固定 restart=300 全规模够用。

**数据缺口 + 待跑**：
- h2 局部化：✅ 已有（93@r60 / 13@r300）。
- **h3 局部化：缺**。现有 `paper_aux_h3/` 只到 step_013（maxd 0.42，**未局部化**；§5.2b 当时标"数据续算中"）。
  **待跑**：h3 staggered 续算 2–3 步到局部化（maxd→0.99），约几小时。命令（resume）：
  ```
  FRACTUREX_HMIN=0.013 FRACTUREX_RESUME=1 FRACTUREX_RUN_NSTEPS=17 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. nice -n 19 \
    <py312>/python scripts/paper_huzhang/run_case.py --case model0 --mode aux \
    --out-root results/phasefield/model0_circular_notch/paper_aux_h3/../..  # 须指向能 resume 的 root
  ```
  续算内存峰值估 **~3GB**（h3 small）；得到 step~14–16 局部化 checkpoint 后,用 `d12_recheck.py` 量 niter@{60,300}。
  - **已启动（2026-06-09）**：`FRACTUREX_RUN_LABEL_SUFFIX=h3 FRACTUREX_HMIN=0.013 FRACTUREX_RESUME=1 RUN_NSTEPS=17 ... nice -n 19`
    从 step_013 resume 成功（日志确认 `continue at step 14/18`），σ-dof=183524。监控 `/tmp/h3mon.sh` + 25min cron。
  - **判决命令就绪**（h3 局部化后自动跑）：`d12_recheck.py` 已加 `localized` 模式（自动找 maxd>0.9 的 checkpoint）：
    `D12_TAG=paper_aux_h3 D12_HMIN=0.013 python scripts/paper_precond/d12_recheck.py localized 60,300`。
    输出 h3 局部化 niter@{r60,r300},对比 h2(93/13) 即判 D13 生死（§10.2 判据）。
- h5 局部化：缺且更重（2M dof，装配+解很慢），**先做 h2-vs-h3 两点**，若已显趋势可不跑 h5。

> **执行时机**：用户指示**排队等机器空窗**再跑（避让大内存作业）。本机当前 avail~1495GB、h3 仅需 ~3GB，
> 内存上其实宽松；但按用户意「排队」，**待用户一句话触发**即启动（命令已就绪）。

### 10.3c h3 局部化实时数据 — 混淆,不可判（2026-06-09,诚实纠正）

h3 续算 step14 局部化(maxd→1.0)。我一度从实时 **峰值** niter(113→143 vs h2 93)误判"随 dof 涨→D13 复活"。
但**完整统计颠覆了这个印象**:

| 网格 | σ-dof | 局部化(maxd>0.95) niter@r60: median / mean / p90 / max | 样本 |
|---|---|---|---|
| h2 | 48,092 | **165 / 154 / 200 / 200** | 762 解(完整) |
| h3 | 183,524 | **111 / 122 / 200 / 400** | 48 解(step14 进行中) |

- **h3 中位 niter(111) 反而比 h2(165) 低** —— 与"峰值 143>93"相反,且与 §5.2b L76 旧线索(h3 峰值 29<h2 93)一致。
- **但这个对比被三因素污染,不能据此判**:(1) **maxit 不同**——h2 旧数据 maxit=200(被截顶),h3 新 run maxit=400
  (我已改默认),分布不可比;(2) **样本不完整**——h3 才 48 解 vs h2 762;(3) 峰值与中位给相反信号。
- **结论:实时 iterations.csv 太脏,判不了 D13 生死。** 必须要**干净对照**:同一**保存的**局部化 checkpoint 上,
  **同 maxit**,restart=60 vs 300 各一遍(`d12_recheck.py localized 60,300`)。
- **卡点**:step_014.npz 要 step14 staggered 收敛才存(driver `for k in range(maxit)`,h2 step14 用了 **205** 次;
  h3 现 ~80 次,且 dd_abs 在 0.04↔0.32 震荡=起裂迭代爆炸,还需 ~125 次慢解)。cron 监控,存盘即自动跑干净判决。
- **教训(再次)**:别从实时峰值/不控变量的数据下判——我这次又差点(峰值 143 误读为复活,中位 111 其实相反)。
  **判决只认 d12_recheck 的确定性、同 maxit、saved-checkpoint 对照。**

### 10.3d ★★ D13 生死判决：封存确认（2026-06-09，干净 h2-vs-h3 对照）

step_014.npz 存盘后,跑**干净判决**(h2 step_015 + h3 step_014,同 `d12_recheck.py`、同 maxit、确定性 seed):

| 网格 | σ-dof | total-dof | maxd | **niter@r60** | **niter@r300** | r300 Krylov 基 |
|---|---|---|---|---|---|---|
| h2 | 48,092 | 82,508 | 0.998 | 173 | **13** | 0.20 GB |
| h3 | 183,524 | 338,418 | 1.000 | **113** | **12** | 0.81 GB |

**判决：封存确认,D13 不复活。** 三条干净证据(直接答 §10.2 判据):

1. **restart=300 下 niter dof-无关**:h2=13, h3=**12**（3.8× dof,niter 几乎不变甚至略降）。**固定 restart=300 在
   48k 和 184k 上都给 O(13) 有界**——这是决定性的：§10.2"D13 复活"要求 niter 随 dof 涨致 restart 须涨到内存爆,
   **前提不成立**。
2. **restart=60 下 h3(113) 反而 < h2(173)**,不是涨——彻底排除"niter 随 dof 涨"(与我之前从实时峰值的误读相反,
   §10.3c 的诚实纠正在此被确证;§5.2b L76 旧线索 h3<h2 也吻合)。
3. **推论**:restart=300 全规模 niter 有界 O(13),Krylov 基内存 h3 仅 0.81GB（远未到 §10.0 担心的大规模 8GB
   爆——因为 niter 小,实际只需几十个基向量,不是 300 满档;且 niter dof-无关意味着大规模也不需要更大 restart）。
   **学增广粗空间没有"小内存达成大 restart 迭代数"的价值空间**——restart=300 本身就是小内存(O(niter·dof),niter~13)。

> **§10.0 那个"restart 大内存爆"的担忧被数据否掉了**:它假设大规模需要 restart=300 满档(8GB)。但实测 niter 只到
> ~13,GMRES 在第 13 步就收敛,**根本用不满 restart=300 的基**——实际 Krylov 基 ~13·dof,大规模也就 ~0.4GB。
> 即"restart=300"是上限不是实际用量;niter dof-无关 ⇒ 实际基大小 dof-无关 ⇒ 无内存爆 ⇒ 学粗空间无优势。

**最终结论(二次复核后确认前一版封存,但理由更扎实)**:D13 学习增广线封存。不是因为"问题不存在"(那个说法
§10.0 修正过),而是因为**实测证明:足够 restart(=300)下,局部化 niter 既有界(O(13))又 dof-无关,且实际 Krylov
内存小(niter 小用不满 restart)——三者同时成立,彻底消除了学粗空间的价值空间(省内存/降迭代/抗 dof 都已被
restart 免费拿到)**。价值并入 D12(restart 再诊断 + 方法学)。学习机制(命题0/4 + 35 测试)封存为干净负结果。

### 10.4 给作者的诚实修正

> **我(助手)前一版封存下早了。** "restart 解决 O(100)"在 model0 h2(82k)对,但我漏算了 restart 的 Krylov 内存
> 在大规模(2M+)与 D12 核心卖点「小内存」直接冲突。D13 的真正问题是**大规模局部化的内存-迭代权衡**,这个 restart
> **没解决**。是否值得做下去,取决于一个具体可跑的实验(§10.2:局部化 niter vs dof)。**建议先跑这个判定实验再定生死,
> 不要凭小规模数据封存。** 下方 §10.A–D(原终判封存)与附录 X(route A 骨架)保留作历史,但**结论以本节 10.0–10.4 为准**。

---

## 10-OLD. ⛔ (已撤回)前一版封存结论 — 仅小规模成立,被 §10.0–10.4 修正

### 10.A 为什么停（核心逻辑）

D13 整条线是为回答**"裂纹局部化致 niter 爆炸到 O(100),怎么解决?"**。本会话证明:

1. **这个问题根本不真实存在**——restart=60→200 即让 niter 从 O(100)/DNF 回到 O(10–50)（§9.4 实测：
   step_014 93→9、maxd=1.0 DNF→14）。**问题没了 ⇒ 针对它的所有学习方案(B1 权重、deflation 粗空间、B2 延拓)
   全是 solution-in-search-of-a-problem。**
2. 这比"deflation 只 constant-factor"严重得多:不是某个方案不够好,而是 **restart 抽走了整个研究方向的地基**——
   连最贵的 B2 都失去动机（为什么花一个量级工程做界面自适应延拓,如果 restart=200 已给 O(10–50)?）。
3. D13 真正的硬产出**没一个跟"学习"有关**:(a) restart 再诊断、(b) pyamg-RNG 噪声方法学——这两块是真金,
   但**属于 D12**（§5.2b 修订 + 一个方法学贡献）。学习机制(命题0/4 + deflation)是"证明这套学习增广安全但没用"的框架,
   技术干净但卖不动。

### 10.B 价值去向（收割,不浪费）

| 产出 | 去向 |
|---|---|
| **restart 再诊断**（O(100)=restart 假象，d12_recheck 表） | → **D12 §5.2b 修订**（已写 D12_RESULTS 顶部通知 + 末尾方案） |
| **pyamg-SA 全局 RNG 噪声 + 确定性测量协议** | → **D12 方法学一节**（任何用 pyamg+GMRES 解非正规鞍点者受益） |
| **生产化 restart 默认 60→200 + maxd-自适应** | ✅ 已落地 `run_case.py`（§9.6） |
| 命题 0 障碍定理 + 命题 4 安全框架 + deflation（35 测试） | **封存**：干净负结果,文档+测试完整（§1–§9）,待未来真需求复活 |

### 10.C 何时复活（明确触发条件,避免过早优化）

唯一能让 D13 复活的:**出现 restart 也救不了的 regime**——某真实算例的极难态,restart=300 仍 O(100) 且不收敛。
当前证据**不支持**这个假设存在（step_020 最难 maxd=1.0 态 restart=300 要 49,虽对 restart 敏感但仍有界收敛）。
**为假设的未来需求投入 = 过早优化。** 若未来在更大/更难真实算例上观测到"restart 撞墙",再从封存的 deflation
框架（已可证安全、有测试）重启,届时 B2 才重获动机。

### 10.D 不再做的事（明确止损清单）

- ❌ 命题 0 障碍定理严格证明
- ❌ omega=0.5 修正 k-scan 重跑
- ❌ B2 界面自适应延拓
- ❌ 跨算例 square/model2 复核
- ❌ route A 独立成篇

理由统一:这些都是给一个**已被 restart 解决的问题**做确认/优化工作,ROI 为负。

---

## 附录 X（封存）：原 route A 论文骨架材料盘点

> 以下 §10.0–§10.5 是封存前写的 route A 骨架,**仅作未来复活时的材料参考**,非当前行动项。

### 10.0 一句话贡献(论文 abstract 内核)

> 在 Hu–Zhang 相场断裂的鞍点预条件上,我们 (1) 揭示"裂纹局部化致 GMRES 迭代爆炸到 O(100)/不收敛"主要是
> **重启参数过小**在非正规鞍点上的停滞,而非预条件子退化——足够 restart 下迭代数跨完全局部化保持 O(10–50) 有界;
> (2) 为"是否需要数据驱动的局部化专用粗空间"这一问题建立**可证明安全的学习增广框架**(任意网络参数下 SPD + 解不变),
> 并证明**障碍定理**:固定几何粗空间下任意权重/有限秩增广的两层条件数有与对比度同阶的下界,解释了为何此类技巧
> 只能 constant-factor;(3) 给出确定性测量协议(揭示 pyamg 平滑聚合消费全局 RNG 致迭代数噪声)。

### 10.1 论文骨架(目标 10–12 页,CMAME / SISC methodology 档)

| 节 | 标题 | 内容 | 本会话哪条支撑 |
|---|---|---|---|
| §1 | Introduction | 相场断裂混合元 + 鞍点预条件 + "localization 致迭代爆炸"的既有认知;贡献三点(见 §10.0) | — |
| §2 | 离散与块预条件 | 引 D12 离散/Schur/aux-space;非正规性声明(谱不决定 GMRES,用 niter+FOV) | §4 开头措辞约束 |
| §3 | **localization 迭代爆炸的再诊断** | restart 是主因:d12_recheck 多 checkpoint×restart 表(93→9,DNF→14);非正规鞍点 GMRES(m) 停滞机理 | §9.4 |
| §4 | 可证明安全的学习增广框架 | EnrichmentOperator(deflation,Galerkin);**命题 4**(任意 θ→SPD+解不变,真实算子 1.65e-10);特征 φ+模型 | §2.3/§6.1/L1-L2 |
| §5 | **障碍定理** | **命题 0**:固定几何粗空间下 inf_w κ ≥ c·ρ^{1-η};deflation 多模态 k-scan 实测饱和(173→142,k≥16) | §4.0/§6.6/§6.9 |
| §6 | 数值:框架行为 + 负结果 | deflation 杀合成对比度 κ(命题6合成证据);真实算子仅 constant-factor;**restart 完胜增广**对照表 | §6.2/§9 |
| §7 | 测量方法学 | pyamg-SA 全局 RNG 致 niter 噪声(150/178)+ 确定性协议(seed+清缓存→173/173) | §6.4 |
| §8 | Discussion | 何时该学/不该学预条件子;B2(改延拓)是唯一可能 regime change 但成本高;restart 自适应 | §6.7/§8/§9.6 |
| §9 | Conclusion | localization 不需专用预条件子(restart 足矣);学习增广框架可证安全但受障碍定理限于 constant-factor | — |

### 10.2 三个 reviewer-proof 的硬结果(已坐实,可直接进正文)

1. **命题 4 安全性**:真实局部化算子 enrichment on/off 解一致 1.65e-10——"learned but provably safe"无条件成立。
2. **障碍定理(命题 0)+ k-scan 饱和**:理论(固定几何粗空间 κ 下界)+ 实测(k=1/16/64 → 155/142/146 饱和)双向佐证
   "粗空间技巧只能 constant-factor"。这是论文的**理论硬核**,也是"为何不该期待学习 regime change"的答案。
3. **restart 再诊断 + 确定性方法学**:d12_recheck 表(restart 是主因)+ pyamg-RNG 噪声发现——**这两条是论文真正的
   新知识**,比"学习增广"更有传播价值(任何用 pyamg+GMRES 解非正规鞍点的人都会受益)。

### 10.3 必须诚实标注 / 防审稿人攻击

- **不夸大学习价值**:全文不写"学习加速了求解";写"我们检验了学习增广是否必要,结论是受障碍定理限制只能 constant-factor"。
- **restart 不是我们的发明**:GMRES restart 影响是教科书知识;贡献是**在这个具体问题上定量揭示它被误判为预条件子缺陷**。
- **k-scan 用 omega=0.5 收敛代理**(§6.7 修复后);若审稿人质疑代理 vs 乘性 cycle gap,答:已修发散 bug,饱和结论 robust。
- **跨算例只有结构性论证 + model0**(§9.7 受机器阻),正文写"model0 多 checkpoint + 结构性论证;square/model2 同结构预期同形,
  完整跨算例数值留 future"。**不假装做了 square/model2 完整复核。**

### 10.4 与 D12 的关系(避免自我竞争)

- D12 卖"块预条件 robust + 谱框架 + 小内存";本文(D13-as-A)卖"该预条件子在 localization 下是否需要学习增强——不需要,
  并给出可证安全框架 + 障碍定理 + restart 再诊断"。**D13 修订了 D12 §5.2b 的 O(100) 叙事**(restart 假象),
  故两篇必须协调:要么 D12 先发(含修订后的 restart-aware §5.2b)、D13 引之;要么合并(route D)。
- **决策点**:若 D12 §5.2b 修订后,"localization O(10–50) 有界"已是 D12 的干净结论,则 D13-as-A 的 §3(再诊断)
  与 D12 重叠——此时 route D(D13 并入 D12 作"learned augmentation: provably safe but constant-factor, future B2"一节)
  **可能比独立成篇更合适**。建议:先完成 D12 §5.2b 修订,再判 D13 独立成篇(A)还是并入(D)。

### 10.5 落地状态(route A 所需材料盘点)

| 材料 | 状态 |
|---|---|
| 命题 0 障碍定理 | 数学陈述已写(PLAN §4.0);严格证明待补(GenEO/高对比度文献对齐) |
| 命题 4 安全性 + 真实算子 1.65e-10 | ✅ 已证 + 已测 |
| EnrichmentOperator(deflation) + 测试 | ✅ 35 单元 + 2 集成全过 |
| k-scan 饱和(173→142) | ✅ 实测(omega=0.5 修正版待补,buggy 版 173→144) |
| restart 再诊断 d12_recheck 表 | ✅ model0 5 checkpoint;h₃/全 step 待补 |
| pyamg-RNG 噪声 + 确定性协议 | ✅ 已证 + 脚本 |
| deflation 杀合成对比度 κ | ✅ 已测 |
| 跨算例 square/model2 | ⚠️ 仅结构性论证 + model0(机器阻) |

> **结论**:route A 的**理论 + 安全性 + 方法学**三块硬结果已基本就位;缺的是 (1) 命题 0 严格证明,
> (2) omega=0.5 修正 k-scan 一条曲线,(3) restart 再诊断扩到 h₃。**但写之前先定 A vs D**(见 §10.4)——
> 强烈建议等 D12 §5.2b 修订完成后再判,避免 D13 §3 与 D12 重叠做无用功。
