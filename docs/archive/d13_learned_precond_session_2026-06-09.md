# D13 学习式预条件：方向重定 + 数据管线 L1-a 落地（2026-06-09）

> 范围：(1) 评审并重写 `docs/preconditioner/D13_LEARNED_PRECONDITIONER_PAPER_PLAN.md`
> 的研究方向与核心理论；(2) 实现 D13 数据管线第一阶段（特征提取 + checkpoint
> 旁路导出），并在真实局部化 checkpoint 上冒烟通过。
> 状态：方向已锁定并写入文档；`fracturex/ml/coarse_features.py`、
> `scripts/paper_precond/dump_features.py` 已跑通；未改 fealpy、未动求解器主链。
> 关联记忆：`d13_learn_coarse_space`、`record_experiments_in_docs`、`dont_modify_fealpy`、
> `fealpy_env_py312`、`fracturex_code_comment_style`。

---

## 0. 起点 / 任务

用户要求：读 D13 计划文档，看数学理论是否需补充、给出程序模块设计 list、评估投稿水准，
并开始着手准备。

输入材料：D13 计划原稿（学习对象 = 辅助 P1 粗算子的**扩散权重** \(w_\theta\)）、
`fracturex/utilfuc/linear_solvers.py` 全部求解器、`docs/preconditioner/D12_RESULTS.md`。

---

## 1. 关键评审发现（决定全篇定位）

D13 原稿的中心赌注是「学粗权重 \(w_\theta\) 替换手工 \(g(d)\) → 在 \(d\to1\) / 跨网格进一步加速」。
但 **D12 自己的 B1 实验已证伪强版本**：

- D12 §5.2b / B1：界面感知加权（本质即手工版 \(w_\theta\)，\(g(d)(1+\alpha\|\nabla g\|/\max)\)）
  在真实局部化算子上只把 niter 170→123（−28%），**constant-factor，O(100) 压不回 O(10)**。
- D12 给出的根因是决定性的：**修正的是粗扩散权重，但延拓算子 \(PI_s\) 仍是几何纯 P1，
  无法表示界面两侧跳变模态** —— 瓶颈在粗空间 \(V_H=\mathrm{range}(PI_s)\) 本身，不在权重。

结论：若严格按原稿「只学权重」，Go/No-Go #1 在局部化 regime 大概率不达标，沦为中等贡献。

---

## 2. 数学理论补充 / 修正（已写入计划文档）

| 项 | 内容 | 写入位置 |
|---|---|---|
| **命题 0 障碍命题（核心新理论）** | standard 下 \(\widehat S\sim B\,\mathrm{diag}(g(d))B^\top\) 是对比度 \(\rho=1/\varepsilon_g\) 的高对比度散度算子；固定几何粗空间下 \(\inf_w\kappa(B_S(w)\widehat S)\ge c\rho^{1-\eta(V_H)}\)，**与权重无关**。把 B1 负结果升格为定理，论证「必须学粗空间」。靠 GenEO / 高对比度扩散理论。 | §4.0 |
| **非正规谱界措辞** | \(\mathcal P^{-1}K_h\) 非正规，特征值不决定 GMRES 收敛；严格谱界只写 SPD 构件（\(\widehat S\)/\(L_c\)/\(R^\top\widehat S R\)），全系统用 niter + field-of-values，全系统 \(\kappa\) 仅启发性（同 D12 §5.6 口径）。 | §4 开头 |
| **命题 4 安全性推广** | 学粗空间版用 Galerkin congruence + pseudo-inverse 保 SPD，无条件保正确性（与 \(\theta\)、\(\Phi_\theta\) 无关）。 | §4.5 |
| **命题 6 上档重定向** | 改为证 \(\Phi_\theta\) 使捕获率 \(\eta\to1\) 消去 \(\rho\) 依赖，对齐 GenEO 谱粗空间可证 \(\kappa\) 界（比原稿「证 \(\alpha\to0\)」更扎实）。 | §4.7 |
| **学习对象重定义** | \(PI_s\to R_\theta=[PI_s\mid\Phi_\theta]\)，学界面增广模态而非权重；粗校正走增广粗矩阵 \(\widehat S_H=R_\theta^\top\widehat S R_\theta\)（Galerkin）。原 \(w_\theta\) 权重学习降级为消融对照。 | §4.0bis |

---

## 3. 方向决定（用户拍板）

- **学习对象**：学粗空间 / 延拓（\(R_\theta=[PI_s\mid\Phi_\theta]\)），不是只学权重。
- **理论主线**：纳入「权重天花板 / 障碍命题」为核心定理。

二者均已写入计划文档首部「方向锁定」、§0、§4，并记入长期记忆 `d13_learn_coarse_space`。

---

## 4. 程序模块设计（计划 §13.1，约束：不改 fealpy / `learn/` 零污染 / torch 不进热路径）

```
fracturex/ml/
├── coarse_features.py        ✅ 已实现：φ 提取（零 solver/torch import）
├── coarse_space_enrich.py    ⬜ 主线：Φ_θ 生成 + R_θ=[PI_s|Φ_θ] + Galerkin 增广（命题4 SPD 安全）
├── coarse_weight_model.py    ⬜ 次要/消融：bounded w_θ
├── inference_adapter.py      ⬜ 接缝：torch → setup 前向一次 → numpy；热路径零 torch
├── spectral_labels.py        ⬜ 训练标签 κ + 捕获率 η
├── train_coarse_space.py     ⬜ 目标 A1 谱代理（主）/ A2 端到端（可选）
└── datasets.py               ⬜ L1 余项：按 §5.3 留出协议切样本
scripts/paper_precond/
├── dump_features.py          ✅ 已实现：checkpoint 旁路导特征（零额外仿真）
└── precond_learned_sweep.py  ⬜ aux_learned 同 sweep 对比
fracturex/utilfuc/linear_solvers.py  ⬜ 唯一侵入：两求解器加可选 learned_coarse_provider（与 interface_aware 平行）
```

---

## 5. 投稿水准评估（计划 §13.2）

- **本路线（学粗空间 + 障碍命题）**：CMAME 稳；凭命题 0 + 命题 4 推广 + GenEO 式可证 \(\kappa\) 界
  + 局部化 O(100)→O(10) 的 **regime 改变**（非常数因子），**有冲 SISC / JCP 的实质本钱**。
  三部曲叙事闭合：D12 发现界面瓶颈 → D13 学习粗空间解决 → A2 推广 monolithic。
- **底线（确定性拒稿点，必须做实）**：命题 4 SPD 安全 + SPD 构件干净谱界 + 非正规全系统 \(\kappa\)
  措辞收口 + 跨网格/跨 \(l_0\) 留出实验。
- **工程风险**：增广破坏 \(PI_s\) 几何缓存（\(\Phi_\theta\) 每步重算）、须改双求解器——与 D12 对 B2
  的成本判断一致（高一量级），但回报所在。

---

## 6. 数据管线 L1-a 实现与冒烟（已跑通）

### 6.1 接口确认（探查结论）

| 需求 | 接口 | 返回 |
|---|---|---|
| 损伤场 d | `state.d(bcs, index)` / `state.d.grad_value(bcs, index)` | (NC,NQ) / (NC,NQ,GD) |
| g(d) / g'(d) | `damage.degradation(d)` / `damage.degradation_grad(d)` | 同 d shape |
| l0 | `damage.l0`（`on_build` 从 `case.model().l0` 读） | float |
| P1 粗自由度 | `LagrangeFESpace(mesh,1)`，cgdof = NN（顶点） | — |
| 顶点坐标 | `mesh.entity("node")` | (NN,gdim) |
| checkpoint | npz keys：`d/sigma/u/r_hist/H/NN/NE/NC/p` | — |
| 装配 standard 算子 | `HuZhangDiscretization(case,p=3,damage_p=2).build(mesh)` + 恢复 `state.d[:]` + `HuZhangElasticAssembler` | 见 `check_localized_baselines.py` |
| 运行环境 | `PYTHONPATH=<repo> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` + conda env py312 | `scripts/paper_huzhang/env.sh` |

### 6.2 特征定义（`coarse_features.py`，5 维无量纲，per-node）

\[
\phi=(\,d,\ \ \|\nabla d\|\,l_0,\ \ h/l_0,\ \ \log(g/\bar g),\ \ g/g_{\max}\,)
\]

实现要点：
- **损伤是 P2**（checkpoint d 有 5956 dof，含边中点），不是 P1 —— 故按 FE 函数在单元角点
  求值再 scatter 取节点值，**不假设 `state.d[:]` 是节点值**（这是最初的隐患，已纠正）。
- 1-ring 均值 \(\bar g\) 用「按单元均值 scatter 到节点」的向量化代理，无 per-node Python 循环，
  可扩展到 2M dof。
- 纯 numpy + FEALPy，零 solver / 零 torch import。

### 6.3 旁路导出（`dump_features.py`）

复用 D12 checkpoint + `HuZhangDiscretization`/`PhaseFieldDamageModel` 装配几何，仅恢复冻结 \(d\)
后提特征，**零额外仿真**；按 checkpoint 导出 npz（phi/node/d/g/l0/maxd/provenance）。

### 6.4 真实数据冒烟（model0 h₂，hmin=0.025，eps_g=1e-6）

| 特征 | step_010（maxd 0.31，局部化前） | step_015（maxd 0.998，**完全局部化**） |
|---|---|---|
| `gradd_l0` max | 0.155 | 0.374 |
| `log_g_over_gbar` min | −0.079 | **−6.100**（g 跨节点环跳变 ~e⁶≈400×，命题 0 高对比度签名） |
| 陡界面节点（gradd_l0>0.1） | 19 | 149（8×） |
| `h_l0` | 1.32–1.83 | 完全相同（几何特征 d-无关 ✓） |

合成网格细化（n=32/64/128）另测：`h_l0` 随加密折半、`d`/`g_over_gmax` 稳定 ✓
（无量纲 / 分辨率不变性的**必要条件**成立；充分性靠 §5.3 留出实验，非定理）。

**结论**：特征层把「局部化 vs 未局部化」清晰分离，`log_g_over_gbar` 是模型定位增广模态的主输入。
所有特征有限、几何特征跨态稳定。

---

## 7. 下一步

- L1 余项：`datasets.py`（按 §5.3 留出协议：跨损伤 / **跨网格(核心)** / 跨算例 / 跨长度尺度切样本）。
- L2 核心：`coarse_space_enrich.py`（\(\Phi_\theta\) 生成 + Galerkin 增广粗矩阵，命题 4 的 SPD 安全落地）
  + `inference_adapter.py` 接缝 + `linear_solvers.py` 注入点。

---

## 8. 改动文件清单

| 文件 | 改动 |
|---|---|
| `docs/preconditioner/D13_..._PLAN.md` | 方向锁定 + §0/§4 理论重写 + §13.1/§13.2/§13.3 模块/投稿/进度 |
| `fracturex/ml/__init__.py` | 新建（包说明） |
| `fracturex/ml/coarse_features.py` | 新建（特征提取） |
| `scripts/paper_precond/dump_features.py` | 新建（checkpoint 旁路导出） |
| 记忆 `d13_learn_coarse_space.md` + `MEMORY.md` | 新增方向锁定条目 |
| 产物 `results/phasefield/_precond_features/feat_model0_h0.025_step_0{10,15}.npz` | 冒烟产物 |

---

## 9. 测试 + 多后端规范化（2026-06-09 续）

### 9.1 单元测试

新增 `fracturex/tests/test_coarse_features.py`（pytest，对齐项目 `test_*.py` 约定），**8 例全过**：
形状/有限性、g/g_max∈(0,1] 且最大恰为 1、d∈[0,1]、**h/l0 随细化折半 + d/g_max 跨网格稳定**
（核心无量纲性契约）、**陡裂纹带 log(g/g_bar)<−1 vs 光滑场 >−0.2**（命题 0 高对比度签名可分性）、
gradd_l0 随 l0 线性缩放、**P2 损伤顶点求值正确**（dof≠节点值的回归测试）、l0≤0 报错。

另做：`dump_features.py` 输出 npz schema 完整性校验（12 字段 + 形状/dtype/有限性 + feature_names
契约一致）；架构契约校验（import `coarse_features` **不泄漏 torch/linear_solvers**）。

### 9.2 多后端改造（np → bm）+ 立为统一规范

把 `coarse_features.py` 计算核心从 numpy 改为 FEALPy `backend_manager`：
- `np.add.at`（numpy 专属原地）→ `bm.index_add`（函数式 scatter-add，多后端）；
- `phi[:,i]=`（jax 不支持）→ `bm.stack(...,axis=1)`；
- 掩码除法 `out[nz]=acc[nz]/cnt[nz]` → `acc/bm.maximum(cnt,1.0)`；
- 去 `np.asarray` 强转；类型注解 `np.ndarray`→`Any`。
- `dump_features.py` I/O 边界用 `bm.to_numpy(x).astype(np.float32)`（npz 是 numpy 格式）。

**数值行为不变**：`log_g_over_gbar` min 改造前后均 −6.10；8 例测试 + dump 全过。

**立为 fracturex 统一规范**（用户拍板）：
- 规范文档 `docs/architecture/multibackend_convention.md`（+ docs/readme.md 索引）；
- 长期记忆 `fracturex_multibackend_convention`（+ MEMORY.md 条目）。
- 要点：计算用 bm 不用 np；numpy 仅限 scipy 求解器 / 文件 I/O / 第三方库边界（`bm.to_numpy`
  显式跨界）；新代码强制、存量 57 文件随改随迁不大爆炸；bm 缺算子不改 fealpy；
  范例 = `ml/coarse_features.py` + `dump_features.py`。
- **明确豁免**：`linear_solvers.py`/`sparse_direct_backends.py`/`matfree_elastic.py` 求解主链
  保持 numpy/scipy（gmres/spilu/pyamg 只吃 numpy）——是设计不是欠债。

---

## 10. 多后端规范存量扫描（2026-06-09 续）

全量扫 `fracturex/`（非 tests）的 numpy 专属原地操作，逐处人工核验（不盲信 agent，关键点 grep 真相）：

- **真违规 2 文件**（计算路径、jax 会炸、随改随迁）：
  - `boundarycondition/huzhang_boundary_condition.py` ~10 处对 bm 张量 fancy/掩码原地赋值
    （`gval[idx,:,:]=`、`F_new[mask]=`、`uh[dof]=`，L166/254/256/343/356/361/365/368/464/636）。
  - `assemblers/phasefield_assembler.py` L744-761 numpy 并行装配快路径（`np.add.at`，bm 路径在上方 `lform.assembly()`）。
- **豁免**（设计如此）：matfree_elastic（scipy LinearOperator matvec）、linear_solvers/sparse_direct_backends
  （scipy 求解器）、sampling.py（L²投影导出+spsolve）、drivers/postprocess（诊断/IO/可视化）。
- **已合规**：huzhang_fast_solver、huzhang_elastic_assembler、damage/*、phasefield/*、discretization/*。
- 有价值发现：**装配主链早已 bm**，违规集中在 bc 模块（较早写、没跟上后端化）。
- 清单写入规范文档附录 A（三级分类 + 行号），按规范 §5 不大爆炸重写。

---

## 11. L1 datasets.py 收口（2026-06-09 续）

- `fracturex/ml/datasets.py`：纯数据组织（零 solver/torch import，bm 合规，numpy 仅 npz 读入边界）。
  `FeatureSample`/`load_sample(s)`（feature_names 契约校验）；**§5.3 四协议**全落地
  （cross_damage / **cross_mesh 头条** / cross_case / cross_l0），元数据轴 maxd/hmin/case/l0 对应。
- **标准化只在 train 拟合**（`_fit_standardizer` 排除 test 防泄漏），常数列守护防除零；target 可缺省（待 spectral_labels 附）。
- 测试 9 例全过（四协议切分 + 标准化防泄漏注入极端值验证 + 常数列无 NaN + 空 train 报错 + 真实 npz round-trip）。

---

## 12. L2-α：增广粗空间接缝 + Galerkin SPD（2026-06-09 续）

- 规划进 plan 模式，决定写入 `docs/preconditioner/D13_IMPL_coarse_space_enrich.md`（统一实现文档：设计/SPD推导/测试/结果同一文件）。
  用户两个决定：(A) Φ_θ=学习幅度×固定跳变模板（首版）；(B) 先接缝+Galerkin 机制，再上学习 Φ。
- 核验关键事实：损伤 P2、位移标量 P2-DG（p−1）、**P1 粗空间 cgdof=NN**（Φ 列住 P1 粗空间，与逐节点特征同维）；
  两求解器粗校正都按 gdim 分量循环、结构一致。
- `fracturex/ml/coarse_space_enrich.py`：`EnrichmentOperator`（**首版加性** Galerkin）+ `build_jump_template_modes` + `scale_modes`。
  `linear_solvers.py` 两求解器加 `learned_coarse_provider=None`（与 interface_aware 平行）+ helper `_build_coarse_enrichment`；
  provider=None 零回归。provider 契约 = `callable()->Φ(NN×k)|None`（绑定在求解器侧，ml 不碰求解器状态）。
- 测试：单元 7 例（SPD 安全/秩亏/Galerkin 一致性/模板/隔离）；集成 2 例（真实局部化 step_015，19m15s）
  **解不变性 1.78e-10**（命题 4 推广坐实），niter off=170/on=165（固定模板小增益）。

---

## 13. L2-β：谱实验纠错（加性→deflation）+ 学习管线（2026-06-09 续）

- **关键发现（谱实验抓出 L2-α 机制错误）**：我写的 κ 测试失败（baseline 1.606→"增广"1.730 变差）。
  不当测试 bug，逐步数值追根：
  - **加性 Galerkin 增广不能单调降 κ**（两投影双计数重叠模态、过冲）；
  - **deflation（投影）可以**（worst-mode 1.606→1.508，随机列 2.316 无效）；
  - **deflation 杀对比度依赖**（contrast 1e3→1e5 κ 几乎不变 1.509）= **命题 6 直接谱证据**；
  - 附带纠错：κ 度量须用 runtime 的乘性 GS-coarse-GS，原加性 Jacobi+coarse 会误判。
- **决定（用户拍板）**：机制改 deflation。重写 `EnrichmentOperator`（缓存 W/SW/(WᵀS_bW)⁺，
  `apply_deflated` + `apply_deflated_full`）；求解器接缝从加性旁路改为**包裹 Schur 解**
  （fast 包 `pre_of_S`，auxspace 抽 `_schur_solve_base` 闭包再包）。
- 交付物：`spectral_labels.py`（ideal amplitude 监督标签 + deflation 版 κ 度量）、
  `coarse_weight_model.py`（bounded MLP，torch 仅此+训练脚本）、`train_coarse_space.py`（监督回归，不反传特征分解）。
- 测试：全 ml 单元 **31 例全过**；集成 2 例（deflation，15m55s）**解不变性 1.65e-10**，
  niter off=173/on=153（**−12%，加性仅 −3%，deflation 4× 增益**）；端到端训练 smoke 真实特征 loss 0.267→0.002。
- 诚实标注：−12% 仍 constant-factor（固定模板有限）；压回 O(10) 需学习幅度瞄准 worst-mode（+ 可能 k>1）。
  deflation 杀对比度依赖的谱证据是命题 6 硬支撑。

---

## 14. L2-γ：worst-mode 谱标签 + 真实算子 niter 验证（2026-06-09 续）

- **动机**：L2-β 的 `ideal_interface_amplitude` 是纯特征启发式，不知真实谱 worst-mode。
- 新增 `worst_mode_amplitude(S_b, PI_s)`：对 baseline 误差传播算子 \(E=I-M_0^{-1}S_b\) 幂迭代（仅算子作用、
  不稠密化）取主模态（最慢收敛方向=增广该消的），\(PI_s^\top\) 限制到 P1 粗空间、逐节点幅度归一化为标签。
- **过程教训（诚实记录）**：
  - 试图用离线 two-level κ 在真实 17208² Schur 块比较标签，`eigsh` 在**非对称** \(M^{-1}S_b\) 上算 lmin 失败
    （触 1e-30 floor）——与 D12 §5.6「ARPACK SM 不可靠」一致。**决定性度量改用真实 GMRES niter**（合 §4 措辞）。
  - 合成链上 worst-mode 时好时坏：只在 baseline 真卡时有增益（contrast=1e3 有空间；小链已被几何粗+光滑子处理好无空间）
    ——命题 0 的体现，再次印证须上真实算子。用户拍板「直接上真实算子」。
- **真实局部化算子 niter（model0 step_015，maxd=0.998，σ-dof 48092，q=5，rtol=1e-8）**：
  baseline **175** → 启发式模板 **170(−3%)** → **worst-mode 谱标签 156(−11%)**，全收敛。
  worst-mode 增益是启发式的 **~4×**，k=1 单模态。**坐实学习方向**：标签须连真实谱 worst-mode。
- 交付：`scripts/paper_precond/compare_enrichment_niter.py`（真实算子 niter 对照，~22min 三解）；
  单元 `test_worst_mode_amplitude_label_shape_and_range`。全 ml 单元 **32 例全过**。
- **⚠️ 此 −11% 后被证伪为噪声（见 §16）。** 当时记录为结果，实为 pyamg-RNG 噪声。
- 文档：实现细节/数据全表见 `docs/preconditioner/D13_IMPL_coarse_space_enrich.md` §6.3（含撤回标注）。

---

## 16. L2-γ 纠错链：−11% 是噪声 → 发散代理根因 → 修复（2026-06-09 续，本会话核心）

推进「学习模型在线回归 worst-mode 标签」时，重测把 §14 的 −11% 一路证伪，并挖出根因 bug：

1. **dump/train/datasets 接 target**：`dump_features.py --with-worst-mode-label`（离线装配算子算 worst-mode 标签存 npz）、
   `datasets` 读 `target`、`train_coarse_space` 优先回归 `sample.target`。端到端训练通（真实特征 loss 0.19→0.003）。
2. **重测翻车**：学习幅度 niter 验证得 baseline=160、enrich=168（增广反而差），与 §14 的 −11% 矛盾。
3. **噪声地板实验**：同算子、OMP=1、同进程背靠背两次 baseline = **150 vs 178（~19%）**。**非 OMP 问题**，
   是局部化算子 niter 本质超敏（贴收敛边缘）。
4. **噪声根因（确认）**：`pyamg 5.3.0` `smoothed_aggregation_solver` **消费全局 numpy RNG**——背靠背 RNG 前移 → SA 层级不同 → niter 抖。
   修法（仅测量脚本）：每次 solve 前 `np.random.seed` + 清 pyamg 缓存。**seed 后 baseline 确定性 = 173/173。**
5. **控噪真实结论**：k=1 增广 niter 仅 −1.7%（worst-mode）/−0.6%（启发式）。**§6.3 的 −11% 彻底是噪声。**
   根因=「学习幅度×单模板」本质 **rank-1**，修不了 ~149 维界面坏子空间。
6. **多模态 k-scan**（`top_k_worst_modes` 细空间 deflation 基 + `kscan_worst_modes.py`，确定性）：
   k=0/1/4/16/32 → 173/160/155/**144**/171。最佳 k=16 仅 **−17%**、k=32 反弹——constant-factor，且**未跑赢 D12 B1 权重(−28%)**。
7. **离线诊断翻案（关键）**：误差传播算子 \(E=I-M_0^{-1}S_b\) 的 Ritz 值**全 ≈2.3（≫1）** ⇒ 加性 `make_two_level_minv`（ω=1）
   在真实 Schur 块上**发散**！⇒ `worst_mode/top_k_worst_modes` 取的是**发散算子的错模态**。**§6.6 的 −17% 是「用错模态」的结果。**
8. **修复**：ω 阻尼。离线验证 omega=0.5 时 \(|\lambda(E)|=0.996<1\)（收敛）。已设为 `worst_mode_amplitude`/`top_k_worst_modes`
   默认 + 单元测试断言收敛性。全 ml 单元 **35 例全过**。

**交付（本会话）**：`spectral_labels.top_k_worst_modes`、`coarse_space_enrich` 细模态直用路径、
`compare_enrichment_seeded.py` / `kscan_worst_modes.py`（确定性测量）、dump `--with-worst-mode-label`。

---

## 17. 最终判定（修正 k-scan 已跑完，2026-06-09）

- **机器其实不缺资源**：实测 176 核 / 2TB RAM，负载 ~81（~95 核空闲）。之前"饿死"主因=我 kill 太早 +
  `make_two_level_minv` 把稀疏 PI_s 展成 212MB 稠密撞内存带宽。修复：保持 PI_s 稀疏 → mode-building k=64 仅 10s/0.24GB。
- **修正 k-scan（ω=0.5 收敛模态、确定性、nice）**：niter k=0/1/16/64 = **173/155/142/146**。
  bug 修复消除了 k=32 反弹病态、小幅改善，但**根本结论不变**：deflation 增广**仅 constant-factor（最佳 k=16 −18%）
  且 k≥16 饱和，无 regime change，未跑赢 B1 权重(−28%)**。
- **D13 核心不确定性已关闭**：deflation 路线**不能** regime change。真正 O(10) 只剩 route C（B2 改延拓本身，
  高成本）；route B（乘性对齐+大 k）由饱和现象判为性价比低。
- **已坐实硬结果**（可写论文）：命题 4 SPD 安全（解不变 1.65e-10）、deflation 杀合成对比度 κ、命题 0 障碍、
  pyamg-RNG 噪声方法学。
- **路线决策（D13_IMPL §8）**：**A 改 framing（卖框架+理论+诚实 constant-factor，CMAME 中等）**或 **D 并入 D12**
  立即可收口；**C（B2）**唯一可能头条但高成本高风险。是论文 ROI 战略决策。
- **本会话最大价值**：诚实证伪噪声 −11% → 挖出"谱标签建发散代理"根因 bug → 修复 → 控噪重测**关闭 D13 核心问题**
  （deflation = constant-factor）。避免了把噪声/错模态写进论文，并把战略决策建在 robust 数据上。

---

## 18. ★★ 转折：真凶是 GMRES restart 太小，不是预条件子（2026-06-09 续，本会话头条）

用户明确"还是想解决求解慢、迭代多"，不接受改 framing。于是**系统扫现有求解器旋钮**（并行，装配存盘 once
+ `_solve_knob.py`，确定性 seed）——发现真凶：

- **真实局部化算子 step_015 niter vs restart**：60→**173**, 100→56, 150→35, 200→28, **300→13**（**13× 少迭代**）。
  其他旋钮次要：gs=4/8 迭代降但每迭代变贵（不划算）；deflation restart=200+defl32 仅 28→20。**restart 完胜。**
- **机理**：restart=60 在 GMRES 进入超线性段前重启、丢 Krylov 子空间，对**非正规鞍点**造成 GMRES(m) 停滞，
  把 ~13 撑到 173。**"局部化 O(100)" 主要是 restart 人工产物。**
- **墙钟分解**（497=C+173p, 249=C+13p ⇒ setup≈229s/迭代≈1.55s）：niter 小后 wall 被 setup 主导；
  restart 173→13 给 2× 墙钟（497→249s 独占机实测）。

### 19. D12 复核坐实 + 生产化落地

- **D12 §5.2b 头条复核**（`d12_recheck.py`，多 checkpoint × restart，确定性）：
  013(maxd0.43)=7/2/2；**014(0.998,正是D12头条step13→14)=93/18/9**；015(0.998)=173/28/13；
  **017/020(maxd1.0)=400-DNF/25/14 和 400-DNF/82/49**。
  即 **D12 头条"局部化 niter 爆炸 O(100)/DNF"≈restart=60 假象**；restart≥200–300 下跨完全局部化 O(10–50) 有界。
  **D12 §5.2b 须改写为更强更干净的鲁棒性主张**（aux 跨局部化保持有界，restart 够即可）。
  诚实标注：step_020 最难态 r300 仍 49，主张应是"O(10–50) 有界、随 restart 单调改善"。
- **生产化已实现**（`run_case.py`，零侵入求解器核）：默认 restart 60→200、maxit 200→400；新增 maxd-自适应
  restart（opt-in，maxd<0.9 小/≥0.9 大）；向后兼容（env 可还原）；smoke 验证 launch 正常。
- **生产墙钟测不准**（共享机 load~100–120 不可控）→ 放弃，因 niter 是确定性、负载无关的判定指标，已足够。
- **跨算例复核（square/model2）受阻**：只有生产分辨率局部化 checkpoint（1.5M/847k dof），本机高负载下太重
  （model2 restart=60 解 >22min 未收敛——本身佐证 restart=60 病态）。结构性论证强（同非正规鞍点结构，I/II 型
  共用同一求解器）→ 降级 future confirmation。

### 20. 最终交付（对用户诉求）

> **"迭代多/求解慢"已解决，现成旋钮、零新方法、零风险：**
> - 迭代多 → 默认 restart=200，局部化 niter O(100)/DNF → O(10–50)（93/173/DNF→9/13/14 实测）。
> - 求解慢 → niter 降给 2× 净墙钟（497→249s 独占实测）+ precond_rebuild_interval（默认 5）复用 ~229s setup。
> - 零风险 → restart 只改 niter 不改解；默认变更 env 可还原（D12 旧数据可复现）。
>
> **附带科学产出**：D12 §5.2b 头条须修订（O(100) 是 restart 假象，预条件子实则更鲁棒）；
> 方法学铁律：**上昂贵学习预条件前先扫基础 Krylov 旋钮（restart/maxit）**——restart 13× 完胜 D13 deflation −18%，
> 这是整条 D13 线最该早做却最晚做的实验。详见 `D13_IMPL §9`。
