# 龚博论可吸收清单（2018 Ch. 3–10 → fracturex）

> 来源：`/Users/tian00/Desktop/gong办公资料/TalksAndPapers/thesisCHN/` 及 SUMMARY
> `/Users/tian00/Desktop/gong办公资料/TalksAndPapers/SUMMARY.md`。
> 目的：明确"哪些是马上可用的、哪些留到后续论文、哪些不用抄"，避免下次重复评估。
> 关联：本清单在 [`MASTER_PAPER_DEV_PLAN.md`](MASTER_PAPER_DEV_PLAN.md) §5 已作短表引用。

---

## 0. 快速判断

| 优先级 | 章节 | 可吸收内容 | 立即用途 | 代价 |
|---|---|---|---|---|
| 🥇 立即 | Ch 8 | 仿射不变性 + 三种 Lipschitz 常数 ω | D12 附录 / A 论文 effectivity 分析 | 0 代码，2–3 天写作 |
| 🥈 立即 | Ch 10 | NEPIN（精确 / 非精确非线性消去） | D12 头条加分 spike 实验 | ~1–2 周实验 |
| 🥉 立即 | Ch 3 §附录 A | 任意维奇异顶点代数定义 | A+ 论文 corner_relaxation 严格化 | 0 代码，1 天整理 |
| 4 | Ch 7 §7.4–7.5 | 离散正则分解 + 辅助空间 fictitious space lemma | D12 §Theory 引用（避免自证） | 0 代码，写作随手引 |
| 5 | Ch 6 §6.3 | 顶点块局部估计（非嵌套粗空间的局部收敛） | D12 §3.2 加权粗空间在局部化区的替代证明 | 0 代码，理论素材搬运 |

---

## 1. 🥇 Ch 8 仿射不变性 —— 纯理论工具

### 现存痛点
- fracturex 现有 staggered / outer Newton 只报"数值单调性"，缺严格收敛率语言。
- D12 头条锁死"难 regime 唯一收敛"，但**为什么 aux-space 预条件在 outer Newton 层依旧鲁棒**没有理论刻画。
- A 论文 effectivity index 的 k-依赖（split 情形）还没有干净的 Lipschitz 语言。

### 龚博论给了什么
§8.2–8.4 把 Newton 收敛拆成三种**仿射不变**刻画：

- **仿射协变（affine covariant）**：
  $$\|F'(x)^{-1}\bigl(F'(y)-F'(x)\bigr)(y-x)\| \le \omega_{\mathrm{cov}}\|y-x\|^{2}$$
- **仿射逆变（affine contravariant）**：以残差范数刻画
- **仿射共轭（affine conjugate）**：在能量范数下刻画，天然对应鞍点/自伴问题

三种 ω 都是**在坐标变换下不变的常数**，因此 fracturex 里 $A(d)=g(d)^{-1}\mathbb C^{-1}$ 引入 d-加权柔度这一"坐标变换"后，ω 不会因 g(d) 大幅退化而失控 —— 这正是 D12 头条需要的理论根基。

### 立即怎么用（本任务的 Ch 8 系列文档）
- **`THEORY_affine_invariant_newton.md`**：把 §8.2–8.4 三个 ω 的定义与收敛定理复述为 fracturex 相场版本，加一节"退化柔度下的 ω 有界性"引理
- **D12 论文附录**：把上一步的引理放进去作为"outer Newton 收敛性"支撑
- **A 论文 §M3**：$\eta_T$ 的 effectivity index 在 split 情形下的 k-依赖，用 ω_{cov} 估计走一遍

### 与多后端框架的关系
ω 是**后处理量**（从 Jacobian 矩阵-向量积估计），代码路径只涉及：
- `bm.linalg.solve` 或 GMRES 内层结果的读取
- 范数计算 `bm.linalg.norm`
- 已存在的 `linear_solvers.py` 数据流

因此**天然多后端友好**，不需要新写任何 numpy-only 路径。

---

## 2. 🥈 Ch 10 NEPIN —— 直接对准 D12 头条痛点的实验

### 现存痛点
`docs/preconditioner/PIPELINE_STATUS.md`：
> aux_fast niter 在 maxd ≤ 0.82 时恒 = 7，局部化处**骤升到 ~95–121**（约 14×，单步弹性解从 13 s → 几万秒）。

结构性根因：g(d) 在裂纹带内跨 6 个数量级，弹性子问题的非线性（透过 d 的耦合）在局部化步变得极端。

### 龚博论给了什么
§10.1（精确非线性消去 EN）+ §10.2（非精确 NE）：
- 在牛顿迭代内层**自适应识别"强非线性子集"**（fracturex 场景 = d≈1 的单元集合）
- 先对该子集做局部非线性求解（局部化 → 消去强非线性影响）
- 再对全局线性化
- 龚老师在动脉粥样硬化超弹性上（跨数量级 g(d)C）验证过，与 fracturex 的 d→1 局部化**结构同构**

### §10.3 数值证据
- IN 算法 vs NEPIN 算法性能对比表：NEPIN 在强非线性区域**迭代数不再爆炸**
- 关于网格尺寸的一致收敛性

### 立即怎么用
- 短期：在 h₂ / h₃ 局部化步（`results/phasefield/model0_circular_notch/paper_aux/`）做一个 spike 实验
- 中期：写进 D12 §"Adaptive strategy for the fully-localized regime" 一节
- 长期：单独成篇 T7 "NEPIN for phase-field fracture with affine-invariant analysis"

### 与多后端框架的关系
NEPIN 主循环是 Python 层调度（判断哪个子集需要局部化 → 局部装配 → 局部 solve → 全局 update），装配调用 fealpy 现有 `bm` 装配路径，solve 走 `linear_solvers.py`。所以：
- 主循环用 `bm`（新代码从第一行就多后端）
- 局部 solve 复用现有 scipy/pardiso 边界

---

## 3. 🥉 Ch 3 附录 A —— corner_relaxation 严格化的现成素材

### 现存痛点
- `docs/architecture/corner_relaxation_PR.md` + `huzhang_corner_relaxation_design.md` 已有草案，但**缺严格 dim 计数 + inf-sup 证明**
- L-shape 处理目前是补丁式，未来 A+ 论文要**理论 + 代码一起严格**

### 龚博论给了什么
附录 A "奇异点相关结果的证明"：
- **任意维奇异顶点的代数定义**（Hu-Ma 2020 只给了 k=2/3 二维情形）
- 情况 1：$F \in \mathcal F_h^i$ 上的内部拉格朗日节点
- 情况 2：$\mathcal T_h$ 的顶点

配合 §3.3 杂交化：
- Schur 补系统的变分刻画
- 范数估计

### 立即怎么用
- 把附录 A 逐字整理进 `corner_relaxation_PR.md` 的"理论根据"章节
- **不写代码**，先把理论墙立起来
- 未来 A+ 论文的第 2 章现成素材（写论文时省 1–2 周）

---

## 4. Ch 7 §7.4–7.5 —— D12 §Theory 现成引用

### 内容
- §7.2 $H(\div, \mathbb S)$ 空间的正则分解 + 弹性正合序列
- §7.3 离散弹性正合序列 + 交换算子
- §7.4 离散正则分解定理：任 $\bsigma_h \in \Sigma_h$ 可分为
  $$\bsigma_h = \tilde{\bsigma}_h + \Pi \psi_h + \bairy q_h$$
  且各项在 $h^{-1}$ 加权下稳定
- §7.5 辅助空间预条件子构造（fictitious space lemma）

### 现状
`fracturex/utilfuc/linear_solvers.py` 中：
- `solve_huzhang_block_gmres_auxspace`（§7.5 的实现）
- `_make_coarse_diffusion_coef`（对应加权正则分解）

### 立即怎么用
D12 §3 "Auxiliary space theory" 直接引 Thm 7.4.1 / 7.5.1，标 "Following Gong, Hu, Xu (2018, Ch. 7)"。补一段"扩展到 d-加权柔度 A(d)"过渡即可。**省一大段自证工作**。

---

## 5. Ch 6 §6.3 —— 顶点块局部估计（治 D12 加权粗空间钝化）

### 现存痛点
D12 §3.2 加权 P1 粗空间在 d→1 时"骤变难"。本质：标准 element-wise 局部估计对高对比系数失效。

### 龚博论给了什么
§6.3 稳定分解：改成**以任一顶点为中心的块（patch）上做局部估计**，绕开奇异点干扰。这是龚老师在杂交化多水平求解器的核心技术。

### 立即怎么用
- 短期：把 §6.3 估计移植到 D12 §Theory，作为"加权粗空间在局部化区仍有界"的替代证明
- 中期：T5（D+ 论文，多水平预条件）的核心技术
- 不改代码，仅证明技巧搬运

---

## 6. 明确"不该抄"

- ❌ **Ch 4 内罚混合有限元**：只在 k ≤ n-1 时有价值；fracturex 用 p=3（k=3 ≥ n=2），用不着
- ❌ **Ch 1 动脉粥样硬化 / 腹主动脉瘤模型**：医学背景无关
- ❌ **Ch 5 §5.1–5.2 迭代法与 Krylov 基础**：Xu 1992 经典总结，D12 已有对应段落
- ❌ **Ch 9 ASPIN**：NEPIN 已足够；ASPIN 是并行扩展方向，当前非瓶颈

---

## 7. 本轮执行清单

按"这周就动手做的两件事"锁定 **Ch 8** 作为切入点（Ch 10 NEPIN 排在 Ch 8 之后作为下一批）：

1. `THEORY_affine_invariant_newton.md` — 理论文档
2. `DESIGN_affine_invariant_diagnostics.md` — 设计文档
3. `fracturex/analysis/affine_invariant.py` — 代码（多后端 bm）
4. `tests/analysis/test_affine_invariant.py` — 测试

**边写边测**，产出直接进 D12 论文附录草稿。
