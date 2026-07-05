# 决策：σ 驱动自适应的最优选择（M-DF 主驱动 + η_τ 认证）

> **✱ 2026-07-05 supersession（SENT-only）**: 本决策的 "M-DF 主驱动 + η_τ 认证" 组合**仅在
> tension 主导案例（SENT/model1）上仍然有效**。在 Mode-II 剪切（SENS/model2）上 M-DF 因 $g^{-2}$
> 权重放大 $\sigma_h$ 数值噪声、$d_{\mathrm{cut}}=0.9$ 过滤不足，**失效**并致求解器发散。当前论文
> 主线改用**$\eta_T$ 直接作 marker**（Prager–Synge 估计子既认证又标记）+ CKNS 相对下降停机——
> 见 [DECISION_marker_theory_for_model2.md](DECISION_marker_theory_for_model2.md)。SENT 数值验证：
> $\eta_T$ marker peak vs 参考 nx=120 差 −1.6%（M-DF 是 −1.5%），DOF 更少。本文档保留为
> **σ-driven 分工的历史决策**，涉及 M-DF 的部分被 $\eta_T$ 一体化路径取代。

> 状态：决策稿 v0.1（2026-06-14）。本文是 `docs/adaptive` 的**选择性总纲**，
> 在 [THEORY_marking_strategy.md](THEORY_marking_strategy.md)（预测型标记）与
> [THEORY_equilibrated_aposteriori.md](THEORY_equilibrated_aposteriori.md)（保证型误差估计）
> 之上裁定二者的**分工与优先级**。配合 [DESIGN_program_and_tests.md](DESIGN_program_and_tests.md)、
> [RESULTS_aposteriori.md](RESULTS_aposteriori.md) 与 [../routes/plan_adaptive_aposteriori.md](../routes/plan_adaptive_aposteriori.md)。
> 触发：用户拟「用 σ 展开自适应」，需在两篇理论稿间给出**依据理论的最优选择**。

---

## 0. 一句话结论

两篇理论稿**都不作废**，它们是**同一个 Hu–Zhang 平衡应力 $\sigma_h$ 的两种用途**，不是互斥方案。
对「用 $\sigma$ 驱动自适应」这一目标，最优组合是：

$$\boxed{\ \textbf{主驱动}=\text{M-DF}（\mathcal D(\sigma_h)\text{ 驱动力标记}）\ ;\quad
  \textbf{认证层}=\eta_\tau（\text{平衡型保证误差界}）.\ }$$

- **加密由 $\mathcal D(\sigma_h)$ 驱动**（[THEORY_marking_strategy](THEORY_marking_strategy.md)）：纯 $\sigma$ 驱动、
  预测型、捕起裂、有 1D AT2 理论锚点（$\mathcal D_c=\tfrac13$）。
- **误差由 $\eta_\tau$ 认证**（[THEORY_equilibrated](THEORY_equilibrated_aposteriori.md)）：可靠性常数严格 $=1$、
  reconstruction-free，是论文的**理论头条**；在**接受态网格**上经一次 primal 重解给认证报告。

> 论文叙事的最强点正是这条统一：**同一块平衡 $\sigma_h$ 同时给「预测型标记」和「保证型误差界」**。
> 这条统一的数学纽带见 §7；两篇 THEORY 文档**是否合并**的裁定见 §8（结论：不合并，保持三层结构��。

---

## 1. 二者不是竞品：同一个 $\sigma_h$ 的两种用途

| | $\eta_\tau$（[THEORY_equilibrated](THEORY_equilibrated_aposteriori.md)） | $\mathcal D$ / M-DF（[THEORY_marking_strategy](THEORY_marking_strategy.md)） |
|---|---|---|
| 量 | $\eta_\tau=\|\mathbb C_d\varepsilon(u_h)-\sigma_h\|_{\mathbb A(d)}$ | $\mathcal D=\tfrac{2l_0}{G_c}\,\psi^+(\sigma_h)$ |
| 身份 | **保证型误差界**（Prager–Synge，常数 $=1$，reconstruction-free） | **预测型分辨/物理标记**（领先裂尖、捕起裂） |
| 输入 | $\sigma_h$ **且** $u_h$（须额外一次标准 FEM 解） | **仅 $\sigma_h$**（求解器免费给出） |
| 理论锚 | Thm 1 / Thm 2（保证、T6 验 $k_{\mathrm{res}}$ 鲁棒） | $\mathcal D_c=\tfrac13$、$\sigma_c$、剖面 $e^{-|x|/l_0}$、命题 1–4（sympy 核验） |
| 短板 | 需额外 primal 解；量的是**弹性离散误差**，非 $l_0$ 分辨要求 | **非**保证误差界（无 $\eta\ge\|\text{err}\|$） |

---

## 2. 为什么主驱动选 M-DF（三条理论依据）

**(D1) 纯 $\sigma$ 驱动、零额外解。** $\eta_\tau$ 是 $\sigma_h$ 与 $u_h$ 的**差**，每步要多解一次连续 FEM
$u_h$ 才能成形 $r=\mathbb C_d\varepsilon(u_h)-\sigma_h$；$\mathcal D$ 只用 $\sigma_h$，而 $\sigma_h$ 是 Hu–Zhang
求解器本就产出的量（速度优先，memory `code_goal_fast_and_accurate`）。

**(D2) 起裂——决定性一条。** 起裂时刻 $d\equiv0,\ \nabla d\equiv0$ 于全域，故一切 $d$-型 / ZZ 标记
**恒零**（[THEORY_marking §2 推论](THEORY_marking_strategy.md)）。$\eta_\tau$ 在起裂前量的是**光滑弹性场**的
离散误差——小、且不必在未来裂纹处尖锐峰起，**不能可靠标出起裂位置**；而 $\mathcal D=\tfrac{2l_0}{G_c}\psi^+(\sigma)$
按定义在「将开裂处」峰起、**先**抵 $\mathcal D_c=\tfrac13$。**唯 $\mathcal D$ 型能标记起裂**（命题 1 推论）。

**(D3) 直击实测缺陷。** M3-full 实测峰值载荷 $+16\%$ 高估（[RESULTS](RESULTS_aposteriori.md) §正确性对账）
其根因是**裂纹带分辨率不足**（带停在 $h/l_0\approx0.70$），不是 $\eta$ 算错。$\mathcal D$-标记 + 尺寸下限
$h\le l_0/2$ 直接编码 $\Gamma$-收敛分辨要求，命题 4 给**分辨保证**；$\eta_\tau$ 编码的是弹性精度——另一回事。

> 关键澄清：[THEORY_marking §5](THEORY_marking_strategy.md) 称 $\eta_\tau$ 也「预测」，是指**裂纹已存在后**
> 应力集中使 $r$ 在裂尖偏大。但**起裂前**（纯弹性、$u_h$ 与 $\sigma_h$ 均光滑）这一预测性弱于 $\mathcal D$。
> 故作**驱动**用 $\mathcal D$，作**认证**用 $\eta_\tau$。

**(D4，选 $\sigma_h$ 而非 $\mathbb C_d\varepsilon(u_h)$ 算 $\mathcal D$)** 命题 2：误标带宽 $\propto$ 相对应力误差，
Hu–Zhang $O(h^{p+1})$（平衡、高一阶）vs 标准 FEM 原始应力 $O(h^{p_u})$ 且在裂尖最差。
这正是「用 $\sigma$（而非位移导出应力）展开自适应」的定量理论依据。

---

## 3. 为什么 $\eta_\tau$ 留作认证（不是丢弃）

$\eta_\tau$ 是二者中**理论更强**的对象：Prager–Synge 给可靠性常数严格 $=1$、断裂基准（$f=0$、裂面
traction-free）下**无数据振荡**、reconstruction-free（标准 FEM 须靠 Braess–Schöberl 局部问题去**造**
$\Sigma_f$ 成员，本法直接拿到）。这是**计算数学论文的理论头条**，不可弃。

定位：$\eta_\tau$ **不做每步驱动**（避开每步额外 primal 解），而是在 **predictor–corrector 接受态网格**上
经一次 primal 重解（路线 (a)）算 $\eta_\tau$ 与有效性 $\Theta$，作**保证型误差认证报告**
（[THEORY_marking §5](THEORY_marking_strategy.md) 的「认证伴随 M-EQ」）。

---

## 4. 推荐落地（与既有实现一致）

主循环 = [THEORY_marking §7](THEORY_marking_strategy.md) 的 predictor–corrector：
```
每载荷步：
  repeat (corrector):
    Hu–Zhang staggered 解出 σ_h, u_h, d
    M-DF：𝒟_τ = (2l₀/G_c) max_q H_q；标记 {𝒟_τ ≥ β·𝒟_c ∧ h_τ > l₀/2}
    M = ∅ ? break : bisect+迁移(d,H) 重解
  接受 → （可选）primal 重解算 η_τ/Θ 作认证 → 进下一载荷步
```
- 该路径在项目里**已落地且通过**：`mark_driving_force`/`refine_masked`（T `test_marking_driving_force.py` PASS）、
  predictor–corrector 冒烟 `run_m3_pc_model1.py` PASS（步内反复加密终止、裂尖到 $h\le l_0/2$）。
- $\beta\in[0.3,0.9]$（$\theta_D=\beta\mathcal D_c=\beta/3$），由 §(b) 标到峰值载荷收敛。
- 认证为**可选 / 出图用**：不必每步，按需在关键载荷步开 primal 重解。

---

## 5. 诚实边界 / 适用域

- **M-DF 是分辨/物理判据，不是误差界**：保证型只来自 $\eta_\tau$。论文须分工写清——$\mathcal D$ 定网格、
  $\eta_\tau$ 认证误差（[THEORY_marking §8](THEORY_marking_strategy.md)）。
- **限单调准静态加载**：$\mathcal D$ 经历史场 $H$，非单调/卸载下 $H$ 冻结不回落，$\mathcal D$ 不再是良定领先量。
- **$\mathcal D_c=1/3$ 由 1D 无分裂 $k_{\mathrm{res}}\to0$ 标定**；含拉压分裂时临界值 $O(1)$ 漂移，故用相对标定
  $\theta_D=\beta\mathcal D_c$ 比绝对数稳健。
- **谱分裂**：$\psi^+$（用于 $H$）与 $\eta_\tau$ 的 majorant 均以 Amor 闭式最干净，谱分裂留 future work
  （[THEORY_equilibrated §6.4](THEORY_equilibrated_aposteriori.md)）。

---

## 6. 对两篇 THEORY 文档的定位（裁定）

| 文档 | 角色 | 在本方案中的地位 |
|---|---|---|
| [THEORY_marking_strategy.md](THEORY_marking_strategy.md) | 预测型标记（M-DF）+ predictor–corrector | **主驱动**，自适应的运行核心 |
| [THEORY_equilibrated_aposteriori.md](THEORY_equilibrated_aposteriori.md) | 保证型 a posteriori（$\eta_\tau$，Thm 1/2） | **认证层 + 论文理论头条**，接受态上报告 |

二者**均有效、互补**；不存在「弃其一」。若未来出现「$\mathcal D$ 标记不足、须保证型驱动」的 regime，
再把 $\eta_\tau$ 提为 Dörfler 主驱动（M-EQ）——当前实测不需要。

---

## 7. 两者结合的统一理论（一个 $\sigma_h$，两个泛函，一条纽带）

### 7.1 统一原理：同住「应力侧 / 互补能量」世界

Hu–Zhang 给出**唯一**的平衡应力 $\sigma_h\in\Sigma_f$。标记与认证都是它的**互补能量（应力侧）泛函**：
- **标记** $\mathcal D(\sigma_h)=\tfrac{2l_0}{G_c}\psi^+(\sigma_h)$ —— $\sigma_h$ **自身**的拉伸互补能量密度
  （经柔度 $\varepsilon=\mathbb A(d)\sigma_h$ 取拉伸部分）。
- **认证** $\eta_\tau=\|\,\underbrace{\mathbb C_d\varepsilon(u_h)-\sigma_h}_{=:r}\,\|_{\mathbb A(d)}$ —— 残差 $r$ 的互补能量范数。

二者同住「应力侧 / 互补能量」世界——这正是 Hu–Zhang（**应力**元）原生给出、标准位移 FEM 要靠
Braess–Schöberl 重构才有的世界。这是「为什么值得付 Hu–Zhang 代价」的统一答案：**一块平衡 $\sigma_h$
被两次复用**。

### 7.2 数学纽带：$\eta_\tau$ 是**双边**估计子，认证的正是 $\mathcal D$ 的输入场

Prager–Synge 等式（[THEORY_equilibrated (7)](THEORY_equilibrated_aposteriori.md)）是**两项**的：
$$\eta_\tau^2=\underbrace{\|\varepsilon(u_h)-\varepsilon(u)\|_{\mathbb C_d}^2}_{\text{primal/位移误差}}
  +\underbrace{\|\sigma_h-\sigma\|_{\mathbb A(d)}^2}_{\text{dual/应力误差}}. \tag{7}$$
两个非负项均 $\le\eta_\tau^2$。**取右项**：
$$\boxed{\ \|\sigma_h-\sigma\|_{\mathbb A(d)}\ \le\ \eta_\tau.\ }\tag{L1}$$
即 $\eta_\tau$ 不只界定位移能量误差，**同时是 $\sigma_h$ 自身应力误差的可算、无常数上界**（two-sided 估计子）。
而 $\sigma_h$ **正是 $\mathcal D$ 的输入**。[THEORY_marking 命题 2](THEORY_marking_strategy.md) 给标记可靠性
$\tfrac{|\widetilde{\mathcal D}-\mathcal D|}{\mathcal D}\le 2\tfrac{\|\sigma_h-\sigma\|}{\|\sigma\|}+O(\cdot^2)$。
两式串联：

> **命题（结合，非正式）。** 同一个 $\eta_\tau$ 既以常数 $=1$ 认证弹性离散误差，又（经 (L1)）在互补能量范数下
> **上界预测型标记 $\mathcal D$ 所依赖的应力场精度**。故「$\mathcal D$ 驱动加密」与「$\eta_\tau$ 认证误差」
> 作用于**同一个场**，其精度由**同一个可算量** $\eta_\tau$ 控制。

这把 §0 的修辞统一（「一块 $\sigma_h$ 双重职责」）升为**定量纽带**：认证层不仅独立成立，还**反向背书**了驱动层。

### 7.3 互补的误差覆盖（为什么两个都要，而非冗余）

staggered 一个载荷步的总误差可分两个**正交**来源：
$$\text{总误差}\ \approx\ \underbrace{\text{分辨/}\Gamma\text{-收敛误差}}_{\text{受 }h/l_0\text{ 控}}\ \oplus\
  \underbrace{\text{冻结-}d\text{ 弹性子问题的离散误差}}_{\text{受 }\eta_\tau\text{ 认证}}.$$
$\mathcal D$ + 尺寸下限 $h\le l_0/2$ 控**前者**（[THEORY_marking §3](THEORY_marking_strategy.md) 命题 4）；
$\eta_\tau$ 认证**后者**（[THEORY_equilibrated Thm 1](THEORY_equilibrated_aposteriori.md)）。
**互补覆盖、非冗余**——这是「两个工具都保留」的误差论依据。

### 7.4 诚实边界（纽带的技术缺口）

- (L1) 是**全局** $\mathbb A(d)$ 范数（带 $g^{-1}$ 权）；命题 2 的相对误差是**逐点/局部**、且 $\psi^+$ 只取
  **拉伸投影**。故 7.2 的纽带是**结构性**的，**不是开箱即用的逐元界**。严格逐元版需：局部 Prager–Synge
  应力误差界 + $\psi^+$ 在 $\sigma_h$ 邻域的 Lipschitz 常数。**列为 future tightening，不声称已证。**
- $\psi^+$ 的正部投影在相应内积下 1-Lipschitz，配 $\psi^+$ 在 $\sigma$ 中二次的结构给出相对界——这一步可写实。
- 含**拉压能量分裂**时 $\eta_\tau^2$ 换成 majorant $\mathcal M$（[THEORY_equilibrated Thm 2](THEORY_equilibrated_aposteriori.md)），
  (L1) 的角色经凸对偶 gap $\mathcal M(u,\tau)$ 走，**同结构、更技术**。

---

## 8. 是否合并两篇 THEORY 文档：裁定**不合并**（保持三层结构）

§7 的统一理论**不**意味着该把两篇并成一篇。理由：

1. **不同数学机器。** equilibrated = Prager–Synge / 凸对偶 / functional majorant（连续 a posteriori 分析）；
   marking = 1D AT2 ODE 标定 / $\Gamma$-收敛分辨 / predictor–corrector 终止性。**两套引理、两套工具**，
   合并即把两个 framework 混进一处，稀释各自 Thm 1/2、命题 1–4 的自洽证明链。
2. **不同角色。** 认证 vs 驱动。分开使「驱动—认证」分工**一目了然**（教学/审稿价值）。
3. **体量。** 22 KB + 19 KB，合并成 ~40 KB monolith，反而更难读、难维护。
4. **维护与引用。** RESULTS / DESIGN / memory / 代码已按文件名交叉引用（`d12_theory_code_sync` 式同步规则），
   合并会扰动既有链接。
5. **综合层已存在。** 本 DECISION 文档即综合层——§7 已承载「结合理论」，无须把两篇自洽模块溶进一处。

**采用三层结构：**

| 层 | 文档 | 内容 |
|---|---|---|
| 驱动理论 | [THEORY_marking_strategy.md](THEORY_marking_strategy.md) | M-DF / 𝒟、predictor–corrector（自洽模块） |
| 认证理论 | [THEORY_equilibrated_aposteriori.md](THEORY_equilibrated_aposteriori.md) | $\eta_\tau$、Thm 1/2（自洽模块） |
| **综合层** | [DECISION_sigma_driven_adaptivity.md](DECISION_sigma_driven_adaptivity.md)（本文） | 角色裁定（§0–6）+ 结合理论（§7）+ 合并裁定（§8） |

> **何时该合并？** 仅在**投稿稿**里——论文 method section 需单一线性叙事时，把 §7 作主线把两者编织成一节
> （论文 $\ne$ 设计文档）。但 `docs/` 的设计/理论层保持**模块化**更利维护与独立复核。当前不合并。
