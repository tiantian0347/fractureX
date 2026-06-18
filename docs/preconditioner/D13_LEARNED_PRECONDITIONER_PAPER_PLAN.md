# D13：学习式辅助粗空间——在 Hu–Zhang 相场块预条件上突破局部化瓶颈的可证明加速

> **⚠️ 实现状态横幅（2026-06-09，必读）**：本文是**论文/理论规划**，正文目标「O(100)→O(10) regime change」
> **尚未在真实算子上兑现**。实现进度与诚实结果见 [D13_IMPL_coarse_space_enrich.md](D13_IMPL_coarse_space_enrich.md)。
> 当前实证状态：机制层全部正确（命题 4 解不变 1.65e-10、deflation 杀合成对比度 κ、命题 0 障碍）；但真实
> 完全局部化算子上 deflation 增广**仅 constant-factor（且唯一测量用了 buggy 发散模态，修正模态 niter 待测）**。
> **路线决策点见 IMPL §8**（A 改 framing / B 攻乘性+大k / C 转 B2 / D 并入 D12）。本规划的 O(10) 主张是**目标**非已达成。

> **方向锁定（2026-06-09，重大修订）**：本文档原稿的学习对象是辅助 P1 粗算子的**扩散权重场** \(w_\theta\)。但 D12 的 B1 实验（[D12_RESULTS §5.2b / B1](D12_RESULTS.md)）已经证明：在裂纹**完全局部化** regime，只学权重（界面感知加权 = 
手工版 \(w_\theta\)）仅给 constant-factor 改善（niter 170→123，−28%），**O(100) 压不回 O(10)**，根因是**几何纯 P1 延拓算子 \(PI_s\) 无法表示界面两侧的跳变模态**——瓶颈在粗空间 \(V_H=\mathrm{range}(PI_s)\) 本身，不在它上面的权重。故 D13 学习对象**改为粗空间/延拓本身**：往 \(V_H\) 注入数据驱动的界面模态（\(PI_s\to[PI_s\mid\Phi_\theta]\)），权重退为次要旋钮。

> 目标：在 [D12](D12_PRECONDITIONER_PAPER_PLAN.md) 已建立的 **robust 块预条件 + 谱框架 + 参数 sweep** 之上，把辅助空间粗校正的**延拓算子**用一个**带 SPD-安全约束的学习增广** \([PI_s\mid\Phi_\theta]\) 替换，在**同一套 sweep**上证明「局部化 regime 的迭代数从 O(100) 降回 O(10)（regime 改变，非常数因子），且不破坏参数无关性、不破坏正确性」。定位：计算数学 × 机器学习交叉，目标 CMAME（数值验证档），凭**障碍命题 + GenEO 式可证 \(\kappa\) 界**冲 SISC / JCP。

> 关系：本工作**寄生**于 D12。D12 先发，建立 baseline 与谱工具；D13 复用其离散（§2）、Schur / aux-space 构造（§3）、sweep 基建（§5）、谱脚本，正文只新增「学习模块 + 安全性命题 + 泛化实验」。两篇互引。

> 实现说明：本文档**不修改任何代码**。其中涉及的 `coef_g`（[`linear_solvers.py:265`](../../fracturex/utilfuc/linear_solvers.py#L265)）由作者后续自行从 `max(g(d), eps_g)` 调整为 `g(d)`，本计划的数学与代码引用以「权重场 = \(g(d)\)（带正性下界）」为准。

---

## 0. 一句话研究问题

> 辅助空间预条件的粗校正用几何纯 P1 延拓 \(PI_s\)；D12 证明它在裂纹局部化时无法表示界面跳变模态，致 niter O(10)→O(100)，且任意权重修正都救不回（B1 负结果）。**能否用一个仅依赖无量纲局部特征的小网络生成界面模态 \(\Phi_\theta\)，把粗空间增广为 \([PI_s\mid\Phi_\theta]\)，使局部化 regime 的 GMRES 迭代数从 O(100) 降回 O(10)（regime 改变），同时保证粗矩阵仍 SPD、预条件始终非奇异、收敛到精确解（即学习只在可证明的安全结构内优化，不改变正确性保证）？**

> **理论上的对称命题（本文核心新理论）**：在**固定**几何粗空间下，**任意**正权重 \(w\) 都无法消去高对比度依赖（权重天花板/障碍命题，§4.0），这把 B1 的经验负结果升格为定理，并严格论证「必须学粗空间而非权重」。

---

## 1. 数学设置：鞍点系统与两种退化放置

### 1.1 交错步内的弹性子问题

相场断裂交错求解在每个外层迭代中**固定损伤场** \(d_h\)，求解 Hu–Zhang 应力-位移混合元离散下的鞍点系统（未知量排序 \([\sigma; u]\)）：

\[
K_h(d_h)\begin{bmatrix}\sigma\\ u\end{bmatrix}
=\begin{bmatrix}f_\sigma\\ f_u\end{bmatrix},
\qquad
K_h(d_h)=\begin{bmatrix}A(d_h) & B^\top\\ B & 0\end{bmatrix},
\]

其中 \(A(d_h)\in\mathbb{R}^{m\times m}\) 为（可能退化的）应力块，\(B\in\mathbb{R}^{n\times m}\) 为离散散度算子（代码中 `A_sigma = A[:m,:m]`、`B_div = A[m:,:m]`，见 [`_extract_mechanical_blocks`](../../fracturex/utilfuc/linear_solvers.py#L281)）。

### 1.2 退化律 \(g(d)\)

退化函数取自 [`energy_degradation_function.py`](../../fracturex/phasefield/energy_degradation_function.py)：

\[
g(d)=(1-d)^2+\varepsilon \quad(\text{quadratic}),
\qquad
g(d)=3(1-d)^2-2(1-d)^3+\varepsilon \quad(\text{thrice}),
\]

其中 \(\varepsilon=10^{-10}\) 为模型层下界。注意区分两个下界：
- \(\varepsilon\)：退化函数自身的数值下界（本构层，固定）；
- \(\varepsilon_g\)：预条件中粗空间权重的截止下界（求解层，D12 §4.3 谱分析消融轴 \(\{10^{-3},10^{-6},10^{-9}\}\)；仅作 `damage.coef_bary` 内部下界，见 [`_make_coarse_diffusion_coef`](../../fracturex/utilfuc/linear_solvers.py#L235)）。

定义退化比（贯穿全文，是所有谱界的核心量）：

\[
\boxed{\;r_g \;:=\; \frac{\max_x g(d_h(x))}{\max\big(\varepsilon_g,\ \min_x g(d_h(x))\big)}\;}
\]

当 \(\max d \to 1\)，\(\min g \to \varepsilon\)，故 \(r_g \to \mathcal{O}(\varepsilon_g^{-1})\)，这是预条件恶化的根源。

### 1.3 两种公式（与代码 `formulation` 严格对应）

见 [`_coarse_diffusion_uses_stress_weight`](../../fracturex/utilfuc/linear_solvers.py#L175)：

- **standard**：\(A(d_h)=\int (1/g(d_h))\,\sigma:\tau\)，退化在应力块上；\(B\) 与 \(d\) 无关。粗空间用 **\(g(d)\) 加权**向量 Poisson。
- **effective_stress**：\(A\) 与 \(d\) 无关；\(B(d_h)=\int g(d_h)\,\tau:\varepsilon(v)\)，退化在耦合块上。粗空间用**不加权** Laplacian，\(d\) 仅通过 Schur 块进入。

D13 的学习模块**主攻 standard 公式**（粗权重显式可学），effective_stress 作为对照（学习作用于 Schur 对角缩放，见 §6 备选）。

---

## 2. 块预条件与辅助空间构造（复用 D12 §3）

### 2.1 Schur 补与对角近似

精确 Schur 补与代码使用的 SPD 近似（[`_approximate_schur_spd`](../../fracturex/utilfuc/linear_solvers.py#L305)）：

\[
S(d_h)=B\,A(d_h)^{-1}B^\top,
\qquad
\widehat S = B\,\operatorname{diag}(A(d_h))^{-1}B^\top
= B\,D^{-1}B^\top,
\]

其中 \(D=\operatorname{diag}(A(d_h))\)，\(D^{-1}\) 见 [`_diag_inv_stress_block`](../../fracturex/utilfuc/linear_solvers.py#L298)。在离散 inf-sup 条件下 \(S\succ 0\)；符号约定已吸收 \(-BA^{-1}B^\top\) 中的负号使 \(S\) 正定。

### 2.2 块上三角应用

预条件以块上三角方式作用（代码 `gmres_preconditioner`）：

\[
\mathcal P^{-1}\begin{bmatrix}r_\sigma\\ r_u\end{bmatrix}
=\begin{bmatrix}B_A\big(r_\sigma+B^\top B_S r_u\big)\\ -\,B_S r_u\end{bmatrix},
\qquad B_A\approx A^{-1},\quad B_S\approx S^{-1}.
\]

### 2.3 辅助空间近似 \(B_S\approx S^{-1}\)

按 Chen et al. (2017) §5，把 \(S\) 用 \(H^1\) 上的（加权）向量 Poisson 算子近似，并以 P1 smoothed-aggregation GAMG 处理。standard 公式下，**粗算子的逐点扩散系数**就是本工作的学习对象。当前实现（[`_make_coarse_diffusion_coef`](../../fracturex/utilfuc/linear_solvers.py#L241)）：

\[
c_{\text{coarse}}(x) \;=\; \max\big(g(d_h(x)),\ \varepsilon_g\big)
\quad\xrightarrow{\text{作者将改为}}\quad
c_{\text{coarse}}(x) \;=\; g(d_h(x)).
\]

D13 把它替换为：

\[
\boxed{\;c_{\text{coarse}}(x) \;=\; w_\theta\big(\phi(x)\big)\;}
\qquad
w_\theta:\ \mathbb{R}^p \to [\,w_{\min},\,w_{\max}\,],\ \ 0<w_{\min}\le w_{\max}.
\]

P1 加权扩散的装配见 [`_assemble_p1_diffusion_pyamg`](../../fracturex/utilfuc/linear_solvers.py#L185)（`ScalarDiffusionIntegrator(coef=...)`）。

---

## 3. 学习模块的定义（候选 A）

### 3.1 无量纲局部特征 \(\phi\)

特征必须**无量纲、局部、与网格分辨率无关**，否则跨网格泛化失效。每个粗自由度（或单元/积分点）取：

\[
\phi = \Big(\,d,\ \ \|\nabla d\|\,l_0,\ \ \tfrac{h}{l_0},\ \ \log\!\tfrac{g(d)}{\bar g},\ \ \tfrac{g(d)}{g_{\max}}\,\Big),
\]

其中 \(l_0\) 为相场长度尺度，\(h\) 局部网格尺寸，\(\bar g\) 单元邻域平均。**禁用**绝对坐标、绝对 \(h\)、绝对 dof 编号。

### 3.2 网络形态与正性约束

逐单元小 MLP（先不上图网络）：

\[
w_\theta(\phi) \;=\; w_{\min} + (w_{\max}-w_{\min})\cdot \operatorname{sigmoid}\!\big(\mathrm{MLP}_\theta(\phi)\big),
\]

或 \(w_\theta=w_{\min}+\operatorname{softplus}(\mathrm{MLP}_\theta(\phi))\) 再截断到 \(w_{\max}\)。

**关键不变量**：输出恒在 \([w_{\min},w_{\max}]\)。建议取 \(w_{\min}=\varepsilon_g\)、\(w_{\max}=g(0)=1+\varepsilon\)，使学习权重的取值范围**恰好等于手工权重的物理范围** —— 这保证 §4 的谱界对学习版与 baseline 同时成立，学习只在该界内优化常数。

### 3.3 推理位置（防止净时间反噬）

\(w_\theta\) **只在每个交错步的预条件 setup 阶段前向一次**，输出整个粗权重场后照常装配 P1 GAMG；**绝不进入 GMRES 内循环**。否则每次 matvec 都推理，墙钟必输。这一点在 §7 净时间表中必须量化。

---

## 4. 数学理论（论文 §3–§4 主体，含完整推导）

本节给出四组结果：**(0) 权重天花板/障碍命题（核心新理论）**、**(I) Schur 谱等价界**、**(II) 学习粗空间下的安全性与谱界**、**(III) 参数无关性陈述**。核心思想：把所有 SPD-构件谱量化归到退化比 \(r_g\)、对比度 \(\rho:=1/\varepsilon_g\) 与粗空间捕获率 \(\eta(V_H)\)。

> **⚠️ 非正规性的措辞约束（全文统一，与 D12 §5.6 口径一致）**：全系统预条件算子 \(\mathcal P^{-1}K_h\) 作用在不定鞍点上、**非对称非正规**，其特征值/条件数**不决定 GMRES 收敛**（D12 已诚实标注 ARPACK SM 不收敛、\(\kappa\) via SM 是启发非真界）。因此本节所有**严格谱界只陈述在 SPD 构件上**（\(\widehat S\)、\(L_c\)、增广粗矩阵 \(R^\top\widehat S R\)），全系统收敛主张以 **niter 实测 + SPD 构件谱界 + field-of-values 上界**（Elman–Silvester–Wathen / Loghin–Wathen 数值域型）为准；全系统 \(\kappa(\mathcal P^{-1}K_h)\) 仅作启发性描述符，不作收敛证明的依据。

### 4.0 命题 0（权重天花板 / 障碍命题，核心新理论）

**背景化简**：standard 公式下 \(A_\sigma=(1/g)\,M^{(0)}\)，故 \(D^{-1}=\mathrm{diag}(A_\sigma)^{-1}\sim g(d)\,\mathrm{diag}(M^{(0)})^{-1}\)，于是对角 Schur 近似

\[
\widehat S = B\,D^{-1}B^\top \ \sim\ B\,\operatorname{diag}\!\big(g(d_h)\big)\,B^\top
\]

**本身就是一个系数从 \(1\) 跳到 \(\varepsilon_g\) 的高对比度加权散度算子**，对比度 \(\rho:=g(0)/\varepsilon_g=\mathcal O(\varepsilon_g^{-1})\)。这把问题精确归为高对比度扩散的两层预条件经典情形（Pechstein–Scheichl；Spillane–Dolean–Hauret–Nataf–Pechstein–Scheichl 2014, GenEO）。

**陈述（障碍）**：固定几何 P1 粗空间 \(V_H=\mathrm{range}(PI_s)\)，记其对 \(\widehat S\) 低能跳变模态的捕获率为 \(\eta(V_H)\in[0,1]\)（精确定义：\(\eta=1-\sup\{(\widehat S v,v)/(\,\text{尺度}\,)\ :\ v\perp_{\widehat S}V_H,\ v\ \text{为界面跳变模态}\}\) 的归一化）。则对**任意**正权重场 \(w\in[w_{\min},w_{\max}]\) 加权的粗算子 \(L_c(w)\)，两层条件数有**与 \(w\) 无关的下界**

\[
\boxed{\;\inf_{w>0}\ \kappa\big(B_S(w)\,\widehat S\big)\ \ge\ c\,\rho^{\,1-\eta(V_H)}\;}
\]

当界面跳变模态 \(\notin V_H\)（局部化时 \(\eta\to0\)），下界退化为 \(c\,\rho\)：**无论权重怎么学，两层条件数都被对比度 \(\rho\sim\varepsilon_g^{-1}\) 卡死**。

**证明要点**：取一个集中在界面单元、在 \(V_H\) 的 \(\widehat S\)-正交补里的试探模态 \(v_\star\)（\(d\) 从 0 跳到 1 的台阶在粗 P1 基底下不可表示）。两层预条件子的 Rayleigh 商下界由 \(v_\star\) 上的能量与其粗校正残量之比给出（标准 fictitious-space / 两层框架，Nepomnyaschikh；Toselli–Widlund Ch.2）。权重 \(w\) 只整体缩放 \(L_c\) 的能量，不改变 \(v_\star\perp_{\widehat S}V_H\) 这一**结构性**事实，故对 \(\inf_w\) 取下确界后 \(w\) 被消去，残留 \(\rho^{1-\eta}\)。∎

> **意义（论文逻辑枢纽）**：(1) 把 D12 的 B1 经验负结果（−28% constant-factor）变成**可证、可引用的定理**；(2) 严格论证「**只学权重必然受限、必须学粗空间**」——直接堵住审稿人「为何不只调权重」的质疑，把它变成本文的卖点而非软肋；(3) 给出 D13 的正确目标：学习 \(\Phi_\theta\) 使 \(\eta(V_H\cup\mathrm{range}\Phi_\theta)\to1\)，从而消去 \(\rho\) 依赖（命题 6 上档的真正可证路径，对齐 GenEO 谱粗空间的 \(\kappa\) 界）。

### 4.0bis 学习对象的重定义（替换原 §3 的 \(w_\theta\) 主线）

把原 §3 的「学权重 \(w_\theta\)」改为「学界面增广模态 \(\Phi_\theta\)」：

\[
PI_s \ \longrightarrow\ R_\theta := [\,PI_s \mid \Phi_\theta(\phi)\,],
\qquad
\Phi_\theta:\ \mathbb R^{N_c}\times\mathbb R^p \to \mathbb R^{N_c\times k},
\]

其中 \(\Phi_\theta\) 由 §3.1 的无量纲局部特征 \(\phi\) 生成 \(k\)（小，如 1–4）个数据驱动的界面跳变基向量，与几何 P1 基底拼成增广粗空间。粗校正改为在增广粗矩阵 \(\widehat S_H:=R_\theta^\top\widehat S\,R_\theta\) 上做（Galerkin 投影）。**原 \(w_\theta\) 权重学习降级为次要旋钮 / 消融对照（§6）**，不再是主结果。

### 4.1 预备：加权范数与对角缩放

记 \(A=A(d_h)\)，\(D=\operatorname{diag}(A)\)。对 SPD 矩阵 \(M,N\)，记谱等价 \(M\simeq N\)（常数 \(c_1,c_2\)）为

\[
c_1\,(Nx,x)\le (Mx,x)\le c_2\,(Nx,x),\quad\forall x,
\]

等价于广义特征值 \(\lambda(M,N)\in[c_1,c_2]\)。条件数 \(\kappa(N^{-1}M)\le c_2/c_1\)。

### 4.2 命题 1（应力块的对角等价，standard 公式）

**陈述**：在 standard 公式下，\(A(d_h)=\sum_K (1/g(d_h|_K))\,A_K^{(0)}\)，其中 \(A_K^{(0)}\) 为单元 \(K\) 上与 \(d\) 无关的 Hu–Zhang 质量型块。则存在仅依赖参考单元形状正则性的常数 \(0<\underline\gamma\le\overline\gamma\)，使

\[
\underline\gamma\,D \preceq A(d_h) \preceq \overline\gamma\,D,
\qquad
\kappa(D^{-1}A)\le \overline\gamma/\underline\gamma =: \kappa_0,
\]

且 \(\kappa_0\) **与 \(d_h\) 无关**（退化因子 \(1/g\) 同时进入 \(A\) 与其对角 \(D\)，在每个单元上抵消）。

**证明要点**：逐单元 \(A_K=(1/g_K)A_K^{(0)}\)，\(\operatorname{diag}(A_K)=(1/g_K)\operatorname{diag}(A_K^{(0)})\)。标量 \(1/g_K>0\) 在 Rayleigh 商中约去，故 \(\lambda(A_K,\operatorname{diag}A_K)=\lambda(A_K^{(0)},\operatorname{diag}A_K^{(0)})\)，后者只由参考单元与形状正则性界定。装配（PSD 求和 + Dirichlet）保持下/上界。∎

> 意义：standard 公式下，**\(\operatorname{diag}(A)^{-1}\) 是与损伤无关地好的 \(A^{-1}\) 近似**。这正是 D12 选择对角 Schur 近似在 standard 下 robust 的根因，也是 D13 把学习放到**粗空间权重**而非 Schur 对角的理由（standard 下 Schur 对角已足够好，杠杆在粗算子）。

### 4.3 命题 2（Schur 补的谱等价界）

**陈述**：设命题 1 成立。则精确 Schur 补 \(S=BA^{-1}B^\top\) 与对角近似 \(\widehat S=BD^{-1}B^\top\) 谱等价：

\[
\kappa_0^{-1}\,\widehat S \preceq S \preceq \kappa_0\,\widehat S,
\qquad
\kappa(\widehat S^{-1}S)\le \kappa_0^2 .
\]

**证明**：由命题 1，\(\underline\gamma D\preceq A\preceq\overline\gamma D\) 蕴含 \(\overline\gamma^{-1}D^{-1}\preceq A^{-1}\preceq\underline\gamma^{-1}D^{-1}\)（SPD 求逆反序）。左右夹乘 \(B(\cdot)B^\top\) 保持半正定序：

\[
\overline\gamma^{-1}BD^{-1}B^\top \preceq BA^{-1}B^\top \preceq \underline\gamma^{-1}BD^{-1}B^\top,
\]

即 \(\overline\gamma^{-1}\widehat S\preceq S\preceq\underline\gamma^{-1}\widehat S\)。取 \(\kappa_0=\overline\gamma/\underline\gamma\) 得结论。∎

> 关键结论：standard 公式下 Schur 对角近似的质量 **与 \(\max d\) 无关**，瓶颈不在 \(\widehat S\) 本身，而在用 P1 GAMG 求解 \(\widehat S^{-1}\) 时**粗算子的扩散权重是否匹配 \(\widehat S\) 在裂纹处的各向异性退化**。这把问题精确地推到了 §4.4 的学习对象上。

### 4.4 命题 3（辅助空间粗算子与 Schur 的等价，权重依赖）

设 \(L_c(c_{\text{coarse}})\) 为以逐点系数 \(c_{\text{coarse}}(x)\) 装配的 P1 加权（向量）Poisson 粗算子。Chen et al. (2017) 的辅助空间框架给出：存在网格无关常数 \(\beta_1,\beta_2>0\)（依赖 inf-sup 常数与正则性，**与 \(h\) 无关**）使

\[
\beta_1\,\big(L_c(g(d_h))\,v,v\big)\ \le\ (\widehat S\,v,v)\ \le\ \beta_2\,r_g^{\,\alpha}\,\big(L_c(g(d_h))\,v,v\big),
\]

其中 \(\alpha\in(0,1]\) 刻画裂纹处权重退化与散度算子作用的耦合强度（数值拟合，典型 \(\alpha\approx\) 待定）。于是

\[
\kappa\big(L_c(g(d_h))^{-1}\widehat S\big)\ \le\ (\beta_2/\beta_1)\,r_g^{\,\alpha}.
\]

> 这说明：**用 \(g(d)\) 当粗权重时，残余的损伤依赖性是 \(r_g^{\alpha}\)**。手工 \(\max(g,\varepsilon_g)\) 通过 \(\varepsilon_g\) 截止把 \(r_g\) 封顶为 \(\mathcal O(\varepsilon_g^{-1})\)，但 \(\varepsilon_g\) 是全局常数、无法逐点适配裂纹几何 —— 这正是学习的空间。

### 4.5 命题 4（学习权重的安全性：SPD 保持，无条件）

**陈述**：对任意网络参数 \(\theta\) 与任意输入，只要 \(w_\theta\in[w_{\min},w_{\max}]\) 且 \(w_{\min}>0\)，则：
1. 粗算子 \(L_c(w_\theta)\) 对称正定（Dirichlet 处理后）；
2. 整体预条件 \(\mathcal P\) 非奇异，GMRES 适定且**收敛到与精确解一致的解**（预条件只改收敛速度，不改不动点）；
3. 故**最坏情况下学习模块不会破坏正确性**，仅可能不加速。

**证明**：加权刚度 \((L_c(w)v,v)=\sum_K w_K\,(\nabla v)^\top G_K(\nabla v)\ge w_{\min}\sum_K(\nabla v)^\top G_K(\nabla v)\)，\(G_K\succeq0\)；Dirichlet 约束去核后严格正定。\(\mathcal P\) 块上三角、对角块 \(B_A,B_S\) 非奇异故可逆。右预条件 GMRES 求解的是同一线性系统 \(K_h x=f\)，解集不变。∎

**推广（学粗空间版，本文主线所需）**：当学习对象为增广延拓 \(R_\theta=[PI_s\mid\Phi_\theta]\) 时，安全性同样**无条件**成立——只要把粗校正定义为 \(R_\theta(R_\theta^\top\widehat S R_\theta)^{+}R_\theta^\top\)（Galerkin），其中：
1. 增广粗矩阵 \(\widehat S_H=R_\theta^\top\widehat S R_\theta\succeq0\) 自动半正定（\(\widehat S\succ0\) 的 congruence），对线性无关列严格 SPD；若 \(\Phi_\theta\) 的列与 \(PI_s\) 数值相关，用 pseudo-inverse \((\cdot)^+\) 或 \(+\,\epsilon I\) 正则，仍 well-defined；
2. 故粗校正是 SPD 投影，叠加 GS 光滑子后 \(B_S\) 仍 SPD，\(\mathcal P\) 非奇异；
3. 右预条件 GMRES 解集不变，**与 \(\theta\) 和 \(\Phi_\theta\) 的具体值无关**——最坏情况只是 \(\Phi_\theta\) 无用（退回 \(\eta(V_H)\)），绝不破坏正确性。∎

> 这是论文的**安全垫**，必须在 §3 显式陈述：审稿人无法以「黑盒不可控」拒稿，因为正确性与可逆性是**无条件**的，与 \(\theta\) 无关。学粗空间版的关键是用 **Galerkin 投影**保 SPD（congruence + pseudo-inverse），这把「learned coarse space but provably safe」做成与「learned weight but safe」同级的硬保证。

### 4.6 命题 5（谱界的继承：学习在可证明区间内优化常数）

**陈述**：取 \(w_{\min}=\varepsilon_g,\ w_{\max}=g(0)\)。则对任意 \(\theta\)，学习预条件的条件数被与 baseline **同阶**的界控制：

\[
\kappa\big(\mathcal P_\theta^{-1}K_h(d_h)\big)\ \le\ C\,(\beta_2/\beta_1)\,\rho_w^{\,\alpha},
\qquad \rho_w=\frac{w_{\max}}{w_{\min}}=\frac{g(0)}{\varepsilon_g},
\]

\(C\) 仅依赖块上三角应用与 GAMG V-cycle 的常数。由于 \(\rho_w\) 与手工版的 \(r_g\) 封顶值同阶，**学习版与 baseline 共享同一条谱上界**；学习的收益体现在把实际谱**聚集到界内更优位置**（逐点适配裂纹），而非改变界的阶。

**证明**：由命题 2（\(\kappa(\widehat S^{-1}S)\le\kappa_0^2\)，与 \(d\) 无关）与命题 3、命题 4 复合，并用 \(w_\theta\in[w_{\min},w_{\max}]\) 把 \(r_g\) 替换为 \(\rho_w\)。块上三角预条件鞍点系统的标准谱估计（Benzi–Golub–Liesen 2005, Thm 10.x 型）给出整体常数 \(C\)。∎

> 论文核心论点（一句话）：**学习只在 \([\,w_{\min},w_{\max}]\) 这一可证明安全区间内优化谱常数；结构性保证（SPD、谱界阶数、网格无关性来源）由理论给定，与网络无关。** 这是计算数学接受「learned preconditioner」的标准范式。

### 4.7 命题 6（参数无关性，论文卖点；证明分档）

**目标陈述**：存在常数 \(C^\star\)（与 \(h,l_0,d_h\) 无关）使

\[
\kappa\big(\mathcal P_\theta^{-1}K_h(d_h)\big)\le C^\star,
\qquad\forall\, h,\ l_0,\ d_h\in[0,\,1-\delta],
\]

\(\delta\) 为退化截止。**证明分档**（与 D12 一致）：
- **下档（CMAME 可接受）**：命题 0–5 给出 SPD 构件上 \(\mathcal O(\rho^{1-\eta})\) 上界 + 留出集上 niter 的数值平台验证（局部化 regime O(100)→O(10)）；
- **上档（SISC 需要，路径已对齐 GenEO）**：证明学习增广模态 \(\Phi_\theta\) 使捕获率 \(\eta(V_H\cup\mathrm{range}\,\Phi_\theta)\to 1\)，从而命题 0 的下界 \(\rho^{1-\eta}\to\rho^0=\mathcal O(1)\)，**对比度依赖被消去**。这正是 GenEO 谱粗空间的可证 \(\kappa\) 界范式：若 \(\Phi_\theta\) 张成 \(\widehat S\) 在阈值 \(\tau\) 以下的局部特征模态（数据驱动地近似），两层 \(\kappa\le C(1+1/\tau)\) 与对比度无关。难点：证明学习模态确实覆盖低能空间（需 \(\Phi_\theta\) 的逼近性引理），作为 stretch goal，但比原稿「证 \(\alpha\to0\)」方向更扎实、有现成理论靠山。

---

## 5. 训练：目标函数与协议

### 5.1 数据来源（复用 D12 sweep，零额外仿真）

D12 的扫描矩阵即数据集：每个 `(case, h, l0, eps_g, max_d, formulation)` × 每个 frozen-\(d\) 快照 = 一个样本。新增的只是在 sweep 里加 `--dump-features` 旁路，导出每点的 \((\phi, A_\sigma, B_{\text{div}})\) 或其谱诊断量。**不引入新仿真**。

### 5.2 目标函数（按可控性，先易后难）

**目标 A1（代理谱目标，主结果）**：最小化粗预条件后 Schur 的谱分散。用已有幂迭代 [`_estimate_lambda_max_dinv_s_numpy`](../../fracturex/utilfuc/linear_solvers.py#L99) 估 \(\lambda_{\max}\)，配 Lanczos 估 \(\lambda_{\min}\)：

\[
\mathcal L_{\text{spec}}(\theta)
=\mathbb E_{\text{samples}}\Big[\log\kappa\big(L_c(w_\theta)^{-1}\widehat S\big)\Big].
\]

无需可微求解器，纯离线，最稳，作为论文主结果。

**目标 A2（迭代数目标，加分）**：以实际 GMRES 迭代数 \(n_{\text{it}}\) 为 reward：

\[
\mathcal L_{\text{iter}}(\theta)=\mathbb E\big[n_{\text{it}}(\theta)\big]
+\lambda\,\|\theta\|^2,
\]

因 \(n_{\text{it}}\) 对 \(\theta\) 不可微，用 (i) 有限差分 / SPSA、(ii) 以 A1 训练的 surrogate 预测 \(n_{\text{it}}\)、或 (iii) 进化策略优化。噪声大，作为「end-to-end 也 work」的补充。

**正则**：加 \(w_\theta\to g(d)\) 的锚定项 \(\lambda_g\|w_\theta-g(d)\|^2\)，使学习从手工 baseline 平滑出发（保证不劣于 baseline 的初始化）。

### 5.3 泛化协议（论文最值钱部分，严格留出）

| 协议 | 训练集 | 测试集 | 验证的命题 |
|------|--------|--------|-----------|
| 跨损伤 | \(\max d\in\{0.1,0.5,0.9\}\) | \(\{0.99,0.999\}\) | \(d\to1\) 退化鲁棒性（图 2） |
| **跨网格（核心）** | \(h\in\{1/32,1/64\}\) | \(\{1/128,1/256\}\) | learned mesh-independence |
| 跨算例 | Model0 + Square | Model2（II 型弯裂纹） | 几何泛化 |
| 跨长度尺度 | \(l_0\in\{2e\text{-}3,1e\text{-}3\}\) | \(\{5e\text{-}4\}\) | \(l_0\) 无关性 |

每条画 \(\kappa\) / \(n_{\text{it}}\) vs 参数，叠加「手工 \(\varepsilon_g\) 最优值」曲线对比。

---

## 6. 备选学习环节（effective_stress 公式 / 消融用）

effective_stress 下粗 Laplacian 不加权（命题 1 的对角等价不再直接给出 robust Schur），此时学习对象改为 **Schur 对角缩放** \(s_\theta>0\)：\(D^{-1}\leftarrow s_\theta\odot\operatorname{diag}(A)^{-1}\)，目标使 \(\widehat S_\theta=B(s_\theta\odot D^{-1})B^\top\) 逼近 \(S\)。正性约束 \(s_\theta>0\) 同样保 SPD（命题 4 类比）。此分支作为 §5 消融与「方法可推广」论据，不作主结果。

---

## 7. 实验矩阵与度量（在 D12 表上加行）

### 7.1 算法对照组（在 D12 §4.1 基础上加 `learned_*`）

| 标签 | 说明 | 角色 |
|------|------|------|
| `ilu_gmres` | 块 ILU-GMRES | 弱 baseline |
| `aux_weighted` | aux-space + 手工 \(g(d)\) 加权 | **D12 强 baseline** |
| `aux_learned` | aux-space + \(w_\theta\) 粗权重（候选 A） | 主方法 |
| `schur_learned` | Schur 对角缩放（effective_stress） | 备选 / 消融 |
| `aux_learned+warm` | 叠加学习 GMRES 初值 | 加分消融 |

### 7.2 度量（复用 D12 §4.4）

GMRES 迭代数 `KrylovInfo.niter`、是否收敛、残差曲线、预条件 setup 时间（**含网络推理**）、单次 GMRES 时间、\(\kappa(\mathcal P^{-1}K_h)\) 与前 20 特征值、内存峰值。

### 7.3 必做消融（防致命反驳）

1. **学习 vs 手工 \(\varepsilon_g\) 最优网格搜索**：证明收益不是简单调参可达；
2. **特征消融**：去掉 \(h/l_0\) → 跨网格泛化崩，反证特征设计必要性；
3. **网络规模 vs 推理开销**：证明 \(w_\theta\) 小到 setup 推理可忽略；
4. **净 wall-time 表**：含推理总时间 vs PARDISO vs `aux_weighted`，给出学习版净胜的 dof 阈值。

---

## 8. 论文骨架（目标 12–14 页双栏）

| 节 | 标题 | 内容 | 图表 |
|----|------|------|------|
| §1 | Introduction | 相场断裂 + 混合元 + 学习预条件文献；贡献：可证明安全的 learned 粗权重 | — |
| §2 | 离散与鞍点系统 | 引 D12 / [architecture doc](../architecture/huzhang_phasefield_architecture.en.md)；公式 §1 | — |
| §3 | 预条件与学习模块 | §2–§3 构造；\(w_\theta\) 定义、正性约束、推理位置；**命题 4（安全性）** | 1 图（构造示意） |
| §4 | 谱理论 | 命题 1–3（谱等价）、命题 5（谱界继承）、命题 6（参数无关，分档） | — |
| §5 | 训练与泛化协议 | §5 目标函数、留出协议 | 1 图（特征/网络示意） |
| §6 | 数值实验 | 网格无关（表 1）、\(l_0\) 无关（表 2）、\(d\to1\)（图 2）、谱图（图 3）、净时间 vs PARDISO（表 3）、跨网格泛化（表 4） | 3 图 4 表 |
| §7 | Discussion | effective_stress 分支；3D / monolithic 指向；与 D12 / A2 衔接 | — |
| §8 | Conclusion | | — |

**reviewer 必看的图表**：
- **表 1**：固定 \(l_0,\max d\)，4 网格 × {ilu, aux_weighted, aux_learned}，列迭代数。期望 `aux_learned` ≤ `aux_weighted` 且不随 \(h\) 涨。
- **图 2**：x=\(\max d\in[0,1)\)，y=迭代数。期望 `aux_learned` 在 \(d\to1\) 最平。
- **图 3**：\(\mathcal P^{-1}K_h\) 前 20 特征值散点，不同 \(\max d\) 叠加，展示学习版谱聚集。
- **表 4**：跨网格留出（train 粗 / test 细），证明 learned mesh-independence。
- **表 3**：净 wall-time（含推理）vs PARDISO，给 dof 阈值。

---

## 9. 时间线（建议 D12 投稿后启动，4 个月）

| 月 | 里程碑 | 交付物 |
|----|--------|--------|
| L1 | 数据管线 | sweep `--dump-features` 旁路；特征 \(\phi\) 提取；frozen-\(d\) 数据集 |
| L2 | 训练 + 主结果 | 目标 A1 训练通；`aux_learned` 跑通同一 sweep；表 1 / 图 2 草图 |
| L3 | 泛化 + 理论 | 三条留出协议；命题 1–5 完整证明；命题 6 数值验证 |
| L4 | 成稿 + 投稿 | §1–§8 全文 + 图表；arXiv + 投 CMAME（或冲 SISC） |

---

## 10. Go / No-Go 判据（任一不满足即停下重设计）

1. 目标 A1 下，`aux_learned` 的 \(\kappa\) **稳定优于**手工 \(\varepsilon_g\) 最优值；
2. 跨网格留出（表 4）**不退化**（迭代数不随 \(h\) 增长）；
3. 含推理净时间（表 3）在中等 dof **不输** `aux_weighted`；
4. 命题 4（SPD 安全性）与命题 5（谱界继承）证明成立（这两条是定位计算数学期刊的底线）。

---

## 11. 风险与对策

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| 学习收益不显著（≈ 手工 \(g(d)\)）| 中 | 高 | 改 framing 为「自动消除 \(\varepsilon_g\) 调参 + 跨网格鲁棒」，价值在自动化与泛化而非纯加速 |
| 网络推理吃掉迭代收益 | 中 | 高 | 限定逐单元小 MLP、仅 setup 前向一次；§7.3 净时间表量化 |
| 命题 6 上档证明卡住 | 高 | 中 | 退 CMAME（数值验证档）；不冲 SISC 严格证明 |
| 跨网格泛化失败 | 中 | 高 | 强化无量纲特征设计；若仍失败，缩为「跨损伤 + 跨载荷」泛化，仍可成文 |
| reviewer 质疑「为何不端到端学求解器」| 高 | 低 | §1 明确：本文聚焦**可证明安全**的单环节替换，端到端缺乏谱保证 |

---

## 12. 与 D12 / A2 的衔接

- **D12（先发）**：提供 baseline、谱工具、sweep、`aux_weighted` 强对照；D13 §2/§3/§5 基建全部复用。
- **D13（本篇）**：在 D12 框架内加一层可证明安全的学习粗权重，主打「自动化 + 跨网格泛化 + 谱界继承」。
- **A2（Monolithic，后续）**：D13 的学习粗权重可直接迁移到 3 场 Jacobian 的 σ–u 块，§7 埋伏笔。
- 三篇互引，叙事线：**解得稳（D12）→ 学得更快且可证明安全（D13）→ 推广到 monolithic（A2）**。

---

## 13. 维护说明

- 本文件与代码同步：当 [`linear_solvers.py`](../../fracturex/utilfuc/linear_solvers.py) 的 `coef_g` / Schur / aux-space 接口变化时更新 §2–§3 引用；
- 待作者把 `coef_g`（[L265](../../fracturex/utilfuc/linear_solvers.py#L265)）由 `max(g(d), eps_g)` 调整为 `g(d)` 后，§2.3 与命题 3 的「权重 = \(g(d)\)」即与实现一致；
- 成稿后把 arXiv / 投稿链接补到本文件首部。

### 13.1 程序设计模块 list（2026-06-09 锁定：学粗空间路线）

约束：不改 fealpy；内核在 fracturex 复刻 + 接 fealpy 接口；`learn/` 算子学习线零污染（本工作走独立 `fracturex/ml/`）；schema/接缝驱动；复用 D12 sweep 零额外仿真；torch 不进 GMRES 热路径，推理只在每 staggered 步 setup 前向一次。

```
fracturex/ml/                          # 新建（与算子学习 learn/ 分开）
├── coarse_features.py                 # φ 提取：(mesh, state.d, l0, damage) → per-coarse-dof 无量纲特征；零 solver/torch import
├── coarse_space_enrich.py            # 【主线】Φ_θ 生成界面增广模态；R_θ=[PI_s|Φ_θ]；Galerkin 增广粗矩阵 R^T Ŝ R（命题4 SPD 安全）
├── coarse_weight_model.py            # 【次要/消融】bounded w_θ：w_min+(w_max-w_min)·sigmoid(MLP)；anchor-to-g(d) 初始化
├── inference_adapter.py              # 关键接缝：torch 模型 → setup 阶段前向一次 → numpy 增广算子；热路径零 torch
├── spectral_labels.py                # 训练标签 κ(R^T Ŝ R 上的两层) + η(V_H) 捕获率；复用 _estimate_lambda_max_dinv_s_numpy + eigsh 求 λ_min
├── train_coarse_space.py            # 目标 A1（谱代理，主结果）；可选 A2（SPSA/进化，end-to-end niter）
└── datasets.py                       # frozen-d / 真实 checkpoint 快照 → (φ, Ŝ, B, D_inv) 样本；按 §5.3 留出协议切分

fracturex/utilfuc/linear_solvers.py   # 仅加可选注入点（不改既有逻辑），与 interface_aware 旋钮平行
└── solve_huzhang_block_gmres_{fast,auxspace}(..., learned_coarse_provider=None)
                                      #   provider 非空 → 用 R_θ 增广粗校正替换纯 PI_s V-cycle；否则走原路径
                                      #   缓存键含 learned 标志 + 模型版本（局部化每步 d 变 → Φ_θ 每步重算，PI_s 几何部分仍缓存）

scripts/paper_precond/
├── dump_features.py                  # D12 sweep --dump-features 旁路（零额外仿真）
└── precond_learned_sweep.py          # aux_learned 跑同一 sweep + 真实局部化 checkpoint，产表1b/图1b 对比

fracturex/tests/
└── precond_learned_sweep.py          # 命题4 等价性（SPD/解不变，机器精度）+ niter 回归 + 跨网格/跨l0 留出
```

接缝要点：(1) **唯一侵入**是两个求解器各加一个可选 `learned_coarse_provider`，与 `interface_aware` 平行；其余块三角 sweep、Schur 近似、谱诊断全复用。(2) 局部化下 \(d\) 每步变，\(\Phi_\theta\) 须每步重算，但 \(PI_s\) 几何部分与 Schur 几何块仍走现有 `_AUXSPACE_*_CACHE`（注意：增广破坏部分缓存假设，须新增 \(\Phi_\theta\) 专用缓存键，这是与 D12 B2 同级的工程成本，已知）。(3) torch/numpy 边界严格落在 `inference_adapter`，满足 §3.3 防净时间反噬。

### 13.2 投稿水准定位（专家判定）

- **本路线（学粗空间 + 障碍命题）**：CMAME 稳，凭命题 0（障碍）+ 命题 4 推广（Galerkin SPD 安全）+ GenEO 式可证 \(\kappa\) 界 + 局部化 O(100)→O(10) 的 **regime 改变**（非常数因子），**有冲 SISC / JCP 的实质本钱**。三部曲叙事闭合：D12 发现界面瓶颈 → D13 学习粗空间解决 → A2 推广 monolithic。
- **底线（任一路线都必须做实，否则确定性拒稿点）**：命题 4 SPD 安全 + SPD 构件上干净谱界 + 非正规全系统 \(\kappa\) 措辞按 §4 开头收口（不作收敛证明依据）+ 跨网格/跨 \(l_0\) 留出实验。
- **主要工程风险**：增广粗空间破坏 \(PI_s\) 几何缓存（每步 \(\Phi_\theta\) 重算）、须改 fast/auxspace 双求解器——与 D12 对 B2 的成本判断一致（高一个量级），但这正是回报所在。

### 13.3 实现进度日志

**L1-a 数据管线第一阶段（2026-06-09，已跑通）**：
- `fracturex/ml/__init__.py` + `fracturex/ml/coarse_features.py`：per-coarse-dof（=P1 顶点）无量纲特征 \(\phi=(d,\ \|\nabla d\|l_0,\ h/l_0,\ \log(g/\bar g),\ g/g_{\max})\)，5 维，纯 numpy+FEALPy 零 solver/torch import。损伤场按 FE 函数在单元角点求值再 scatter（兼容 P2 损伤，不假设 `d[:]` 是节点值）；1-ring 均值改为按单元均值 scatter 的向量化代理（可扩展到 2M dof）。
- `scripts/paper_precond/dump_features.py`：复用 D12 checkpoint + `HuZhangDiscretization`/`PhaseFieldDamageModel` 装配几何，仅恢复冻结 \(d\) 后提特征，**零额外仿真**；按 checkpoint 导出 `.npz`（phi/node/d/g/l0/maxd/provenance）。
- **真实数据冒烟（model0 h₂ hmin=0.025, eps_g=1e-6）**：
  - step_010（maxd=0.31，局部化前）：`gradd_l0` max 0.155、`log_g_over_gbar` min −0.079、陡界面节点 19；
  - step_015（maxd=0.998，**完全局部化**）：`gradd_l0` max 0.374、`log_g_over_gbar` min **−6.100**（\(g\) 跨节点环跳变 ~e⁶≈400×，正是命题 0 的高对比度签名）、陡界面节点 149（8×）；
  - `h_l0` 两态完全一致（几何特征 d-无关）✓；合成网格细化测得 `h_l0` 随加密折半、`d`/`g_over_gmax` 稳定 ✓（无量纲/分辨率不变性的必要条件成立）。
- **结论**：特征层把「局部化 vs 未局部化」清晰分离，`log_g_over_gbar` 为模型定位增广模态的主输入。L1 余项：`datasets.py`（按 §5.3 留出协议切样本）。下一步进入 L2：`coarse_space_enrich.py`（\(\Phi_\theta\) 生成 + Galerkin 增广，命题 4 SPD 安全）。
- 运行：`PYTHONPATH=<repo> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 <conda py312>/python scripts/paper_precond/dump_features.py ...`。
