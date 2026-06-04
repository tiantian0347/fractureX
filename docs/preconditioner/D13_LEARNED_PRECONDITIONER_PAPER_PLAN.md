# D13：学习式辅助空间粗权重——在 Hu–Zhang 相场块预条件上的可证明加速

> 目标：在 [D12](D12_PRECONDITIONER_PAPER_PLAN.md) 已建立的 **robust 块预条件 + 谱框架 + 参数 sweep** 之上，只把辅助空间 P1 粗算子的**扩散权重场**替换为一个**带正性约束的学习模块** \(w_\theta\)，在**同一套 sweep**上证明「迭代数 / 墙钟进一步下降，且不破坏参数无关性」。定位：计算数学 × 机器学习交叉，目标 CMAME（数值验证档）或 SISC / JCP（若跨网格泛化 + 谱界都做实）。

> 关系：本工作**寄生**于 D12。D12 先发，建立 baseline 与谱工具；D13 复用其离散（§2）、Schur / aux-space 构造（§3）、sweep 基建（§5）、谱脚本，正文只新增「学习模块 + 安全性命题 + 泛化实验」。两篇互引。

> 实现说明：本文档**不修改任何代码**。其中涉及的 `coef_g`（[`linear_solvers.py:265`](../../fracturex/utilfuc/linear_solvers.py#L265)）由作者后续自行从 `max(g(d), eps_g)` 调整为 `g(d)`，本计划的数学与代码引用以「权重场 = \(g(d)\)（带正性下界）」为准。

---

## 0. 一句话研究问题

> 辅助空间预条件中的粗算子扩散权重当前由手工退化律 \(g(d)\)（加下界 \(\varepsilon_g\)）给定。**能否用一个仅依赖无量纲局部特征的小网络 \(w_\theta\) 替换该权重，使 GMRES 迭代数 / 求解墙钟在 \(d\to 1\) 与跨网格场景下进一步下降，同时保证预条件始终对称正定、谱界仍由 inf-sup 常数与权重范围可控（即学习只在可证明的安全区间内优化常数，不改变结构性保证）？**

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

本节给出三组结果：**(I) Schur 谱等价界**、**(II) 学习权重下的安全性与谱界继承**、**(III) 参数无关性陈述**。核心思想：把所有谱量化归到退化比 \(r_g\) 与权重范围比 \(\rho_w:=w_{\max}/w_{\min}\)。

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

> 这是论文的**安全垫**，必须在 §3 显式陈述：审稿人无法以「黑盒不可控」拒稿，因为正确性与可逆性是**无条件**的，与 \(\theta\) 无关。

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
- **下档（CMAME 可接受）**：命题 1–5 给出 \(\mathcal O(\rho_w^\alpha)\) 上界 + 留出集上 \(\kappa\) 的数值平台验证；
- **上档（SISC 需要）**：证明学习权重选择使 \(\alpha\to 0\)（即 \(r_g\) 依赖被消去）—— 这需要对 \(L_c(w_\theta)\) 与 \(\widehat S\) 的 Fortin 算子构造给出 \(w_\theta\) 相关的 inf-sup 下界，难度高，作为 stretch goal。

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
- 学习模块代码落地时，建议新增 `fracturex/ml/learned_coarse_weight.py` 与 `fracturex/tests/precond_learned_sweep.py`，复用 `scripts/paper_precond/` 批跑框架；
- 成稿后把 arXiv / 投稿链接补到本文件首部。
