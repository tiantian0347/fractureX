# 理论：Hu–Zhang 平衡应力的保证型 a posteriori 误差估计

> 状态：推导稿 v0.2（2026-06-13，补全严格性）。配合 [../routes/plan_adaptive_aposteriori.md](../routes/plan_adaptive_aposteriori.md)。
> v0.2 相对 v0.1 的实质修订：①迹/边界项的 $H^{\pm1/2}$ 对偶严格化（§3）；
> ②补 $\sigma_h\in\Sigma_f$ 的**第二个**条件——牵引振荡 $\mathrm{osc}(t_N)$（§2、§4）；
> ③§5 有效性重写为「精确 / 全局朴素界（含 $k_{\mathrm{res}}^{-1}$，会爆）/ 局部对比度界」三层；
> ④§6 Theorem 2 给完整证明 + Amor 对偶势闭式（§6.3）。
> 目的：把「Hu–Zhang 平衡应力 ⇒ 无常数、reconstruction-free 的保证型误差界」推扎实，
> 并诚实标注退化系数 $g(d)$ 带来的技术难点。
> 记号遵循 `paper_writing_style_compmath`：先定义量与判据，再给界。

---

## 1. 记号与设定

域 $\Omega\subset\mathbb R^d$（$d=2$ 主），位移 $u:\Omega\to\mathbb R^d$，
对称应变 $\varepsilon(u)=\tfrac12(\nabla u+\nabla u^\top)$。各向同性弹性张量 $\mathbb C$
（Lamé $\lambda,\mu$），其逆（柔度）$\mathbb C^{-1}$。对称张量内积 $\tau:\rho=\sum_{ij}\tau_{ij}\rho_{ij}$。

**相场退化。** 相场 $d\in[0,1]$，退化函数
$$g(d)=(1-d)^2+k_{\mathrm{res}},\qquad 0<k_{\mathrm{res}}\ll 1.$$
残余刚度 $k_{\mathrm{res}}$ 保证 $g(d)\ge k_{\mathrm{res}}>0$，**全文严格要求 $g$ 不退化到 0**
（否则下面所有加权范数失去等价性）。退化后的弹性/柔度算子：
$$\mathbb C_d:=g(d)\,\mathbb C,\qquad \mathbb A(d):=\mathbb C_d^{-1}=g(d)^{-1}\,\mathbb C^{-1}.$$

**连续问题（先不带能量分裂，§5 再放开）。** 给定体力 $f$，边界 $\partial\Omega=\Gamma_D\cup\Gamma_N$，
位移满足
$$-\operatorname{div}\big(\mathbb C_d\,\varepsilon(u)\big)=f\ \text{in }\Omega,\quad
  u=u_D\ \text{on }\Gamma_D,\quad \mathbb C_d\varepsilon(u)\,n=t_N\ \text{on }\Gamma_N. \tag{1}$$
真应力 $\sigma:=\mathbb C_d\,\varepsilon(u)$。把 $d$ 视为本子步**已知冻结**的系数（staggered 框架，
弹性子步内 $d$ 不变；这是相场断裂的标准设定，也是本理论的前提）。

**两种范数。** 退化能量范数与其对偶（应力侧）：
$$\|\tau\|_{\mathbb A(d)}^2:=\int_\Omega \tau:\mathbb A(d)\,\tau
   =\int_\Omega g(d)^{-1}\,\tau:\mathbb C^{-1}\tau,\qquad
  \|\varepsilon\|_{\mathbb C_d}^2:=\int_\Omega g(d)\,\varepsilon:\mathbb C\,\varepsilon. \tag{2}$$
注意 $\sigma=\mathbb C_d\varepsilon(u)$ 时 $\|\sigma\|_{\mathbb A(d)}=\|\varepsilon(u)\|_{\mathbb C_d}$
（同一能量的应力/应变两种写法）。

**空间。**
- 运动学容许位移集 $V_g:=\{v\in H^1(\Omega)^d:\ v=u_D\ \text{on }\Gamma_D\}$（$H^1$-协调，连续）。
- 静力容许（平衡）应力集
  $$\Sigma_f:=\big\{\tau\in H(\operatorname{div};\mathbb S):\ -\operatorname{div}\tau=f\ \text{in }\Omega,\ \tau n=t_N\ \text{on }\Gamma_N\big\}.$$
  $\mathbb S$ = 对称矩阵场。真解 $\sigma\in\Sigma_f$，$u\in V_g$。

---

## 2. 关键观察：Hu–Zhang 给的是 $\Sigma_f$ 的成员

Hu–Zhang 元的定义性质（也是它比标准位移 FEM 贵的原因）：

1. **应力逐点对称**：$\sigma_h\in\mathbb S$ 点点成立（不是弱对称）。
2. **$H(\operatorname{div})$-协调**：法向分量 $\sigma_h n$ 跨单元边连续。
3. **逐元离散平衡**：混合格式的第二个方程
   $\int_\Omega \operatorname{div}\sigma_h\cdot v_h = -\int_\Omega f\cdot v_h\ \forall v_h\in V_h^{\mathrm{disp}}$，
   其中 $V_h^{\mathrm{disp}}$ 是**不连续**（DG）位移空间。

由 3 + $V_h^{\mathrm{disp}}\supseteq$ 逐元常向量 ⇒
$$\operatorname{div}\sigma_h=-P_h f\quad\text{逐元成立},\tag{3}$$
$P_h$ = 到位移空间的 $L^2$ 投影。即 **Hu–Zhang 应力精确满足「投影后的」平衡**。
若 $f$ 逐元落在 $V_h^{\mathrm{disp}}$ 中（典型：$f=0$ 或分片多项式），则 $P_h f=f$，内部平衡精确。

**$\sigma_h\in\Sigma_f$ 需要两个条件，不止 (3)。** $\Sigma_f$ 的定义含**牵引边界**
$\tau n=t_N$ 于 $\Gamma_N$。在 Hellinger–Reissner 混合格式里牵引是**本质（essential）边界条件**，
只能在离散迹空间里施加 ⇒ 实际成立的是
$$\sigma_h n=\Pi_h t_N\quad\text{于 }\Gamma_N,\tag{3'}$$
$\Pi_h$ = 到边界迹空间的投影。于是 $\sigma_h\in\Sigma_f$ 精确 $\iff$ **同时** $P_h f=f$（内部）
**且** $\Pi_h t_N=t_N$（牵引）。两者各自的残余分别生成 §4 的 $\mathrm{osc}(f)$ 与 $\mathrm{osc}(t_N)$。

> **断裂基准多为 $f=0$、$\Gamma_N$ 上无牵引（$t_N=0$，裂面 traction-free）、Dirichlet 位移加载**。
> 此时 (3) 与 (3') 同时精确，$\sigma_h\in\Sigma_f$ 精确，下面 Theorem 1 的界**无任何数据振荡**
> （$\mathrm{osc}(f)=\mathrm{osc}(t_N)=0$）。这是本方法相对标准重构法的结构性优势：
> 标准 FEM 要靠 Braess–Schöberl 局部问题去**造**一个 $\Sigma_f$ 成员，我们直接拿到。

**对照（标准 FEM 不行）。** 标准连续位移 FEM 的应力 $\mathbb C_d\varepsilon(u_h)$ 法向不连续、
不在 $H(\operatorname{div})$ 中、不平衡。这正是它要重构的原因；也是为什么超圆里
「平衡应力」那一边历史上是难点。

---

## 3. Theorem 1（无分裂：尖锐超圆 / Prager–Synge 界）

**命题。** 设真解 $(u,\sigma)$ 满足 (1)，$\sigma=\mathbb C_d\varepsilon(u)$。
取**任一**运动学容许位移 $v\in V_g$ 与**任一**静力容许应力 $\tau\in\Sigma_f$。则
$$\|\mathbb C_d\varepsilon(v)-\sigma\|_{\mathbb A(d)}^2
  +\|\tau-\sigma\|_{\mathbb A(d)}^2
  =\|\mathbb C_d\varepsilon(v)-\tau\|_{\mathbb A(d)}^2. \tag{Prager–Synge}$$
特别地，丢掉左边第二个非负项：
$$\boxed{\ \|\varepsilon(v)-\varepsilon(u)\|_{\mathbb C_d}
  =\|\mathbb C_d\varepsilon(v)-\sigma\|_{\mathbb A(d)}
  \ \le\ \|\mathbb C_d\varepsilon(v)-\tau\|_{\mathbb A(d)}=:\eta(v,\tau).\ }\tag{4}$$
右端 $\eta(v,\tau)$ **完全可计算**（不含真解、不含未知常数）。

**证明。** 记 $e_\sigma:=\mathbb C_d\varepsilon(v)-\tau$（可算量），分解
$$e_\sigma=\underbrace{(\mathbb C_d\varepsilon(v)-\sigma)}_{=:a}
          +\underbrace{(\sigma-\tau)}_{=:b}.$$
超圆恒等式 $\|e_\sigma\|_{\mathbb A}^2=\|a\|_{\mathbb A}^2+\|b\|_{\mathbb A}^2$ 当且仅当
$\langle a,b\rangle_{\mathbb A(d)}=\int_\Omega a:\mathbb A(d)\,b=0$。计算这个交叉项：
$$\langle a,b\rangle_{\mathbb A}
 =\int_\Omega \big(\mathbb C_d\varepsilon(v)-\sigma\big):\mathbb A(d)\,(\sigma-\tau)
 =\int_\Omega \big(\varepsilon(v)-\varepsilon(u)\big):(\sigma-\tau),$$
用了 $\mathbb A(d)\mathbb C_d=\mathrm{Id}$ 与 $\mathbb A(d)\sigma=\varepsilon(u)$。
令 $w:=v-u$。由 $v,u\in V_g$ ⇒ $w\in V_0$（$\Gamma_D$ 上为零的 $H^1$ 位移）。
$\varepsilon(v)-\varepsilon(u)=\varepsilon(w)$，于是交叉项
$=\int_\Omega \varepsilon(w):(\sigma-\tau)$。对**对称**张量 $\sigma-\tau$ 有
$\varepsilon(w):(\sigma-\tau)=\nabla w:(\sigma-\tau)$，分部积分（$\sigma-\tau\in H(\operatorname{div};\mathbb S)$、
$w\in H^1$，迹与法向迹在 $H^{1/2}(\partial\Omega)^d$ 与 $H^{-1/2}(\partial\Omega)^d$ 中对偶）：
$$\int_\Omega \nabla w:(\sigma-\tau)
 =-\int_\Omega w\cdot\operatorname{div}(\sigma-\tau)
  +\big\langle (\sigma-\tau)n,\ w\big\rangle_{\partial\Omega}.$$
体积项：$\sigma,\tau\in\Sigma_f$ ⇒ $\operatorname{div}(\sigma-\tau)=-f-(-f)=0$。
边界对偶配对 $\langle\cdot,\cdot\rangle_{\partial\Omega}$ 分两段：$\Gamma_D$ 上 $w=0$（迹为零）；
$\Gamma_N$ 上 $(\sigma-\tau)n=0$ 于 $H^{-1/2}(\Gamma_N)^d$（因 $\sigma n=\tau n=t_N$）。
两段配对均为 0。故交叉项 $=0$，恒等式成立，(4) 随之得证。$\qquad\blacksquare$

**离散取值。** 取 $v=u_h$（标准连续 FEM 位移，$\in V_g$）、$\tau=\sigma_h$（Hu–Zhang，当 $P_h f=f$ 时 $\in\Sigma_f$）：
$$\|\varepsilon(u_h)-\varepsilon(u)\|_{\mathbb C_d}\ \le\
  \eta:=\Big(\int_\Omega g(d)^{-1}\,(\mathbb C_d\varepsilon(u_h)-\sigma_h):\mathbb C^{-1}(\mathbb C_d\varepsilon(u_h)-\sigma_h)\Big)^{1/2}. \tag{5}$$
**逐元局部化**给加密指示子 $\eta_T:=\big(\int_T g(d)^{-1}(\cdots):\mathbb C^{-1}(\cdots)\big)^{1/2}$，
$\eta^2=\sum_T\eta_T^2$。Dörfler 用 $\{\eta_T\}$ 标记。

> 注意 (4)/(5) 是**上界且无常数**（可靠性常数 $=1$）。这是平衡型估计子相对残差型
> （含未知插值常数 $C$）的根本优势，也是「保证型 guaranteed」一词的来源。

---

## 4. 数据振荡余项（$P_h f\ne f$ 或 $\Pi_h t_N\ne t_N$ 时）

$\sigma_h\notin\Sigma_f$ 有两个来源（§2 的 (3) 与 (3')），各补一项。

**(a) 内部残余 $f-P_h f$。** 标准做法（Braess–Schöberl）用单元 Poincaré 不等式：
$$\mathrm{osc}_T(f):=\tfrac{h_T}{\pi}\,\|g(d)^{-1/2}\,(f-P_hf)\|_{0,T},\qquad
  \mathrm{osc}(f):=\big(\textstyle\sum_T\mathrm{osc}_T(f)^2\big)^{1/2}.$$
$h_T/\pi$ 来自单元上向量场的 Poincaré 常数（凸单元）。

**(b) 牵引残余 $t_N-\Pi_h t_N$。** 边界迹只满足 (3')，差出一项。用边迹 Poincaré/迹不等式：
$$\mathrm{osc}_E(t_N):=C_E\,h_E^{1/2}\,\|g(d)^{-1/2}\,(t_N-\Pi_h t_N)\|_{0,E},\quad E\subset\Gamma_N,$$
$C_E=O(1)$ 形状因子。汇总 $\mathrm{osc}(t_N):=\big(\sum_{E\subset\Gamma_N}\mathrm{osc}_E(t_N)^2\big)^{1/2}$。

**完整界。**
$$\|\varepsilon(u_h)-\varepsilon(u)\|_{\mathbb C_d}\ \le\
  \eta+\mathrm{osc}(f)+\mathrm{osc}(t_N). \tag{6}$$
两个 osc 项**均随网格高阶消失、均不破坏可靠性**（仍是上界）。

> **关键诚实标注**：断裂基准 $f=0$ 且 $\Gamma_N$ 上 $t_N=0$（裂面 traction-free、Dirichlet 加载）时，
> $\mathrm{osc}(f)=\mathrm{osc}(t_N)=0$，界回到 (5) 的纯 $\eta$，**无常数无振荡**。
> 论文正文必须明确：guaranteed bound 在此基准下严格成立；一般数据下多两个可算且高阶小的 osc 项。
> **不要把一般数据情形说成「无常数」**——那是错的。$t_N\ne0$ 的混合加载算例须显式报 $\mathrm{osc}(t_N)$。

---

## 5. 退化系数下的 effectivity 分析（审稿核心）

定义 **effectivity index** $\Theta:=\eta/\|\varepsilon(u_h)-\varepsilon(u)\|_{\mathbb C_d}$。
可靠性（$\Theta\ge1$）由 (4) 白给。问题在**有效性（efficiency）**：$\Theta$ 是否有上界、
该上界如何依赖 $k_{\mathrm{res}}$。下面分三层，**严格区分「精确表达式 / 朴素全局界（会爆）/ 局部对比度界（真起作用）」**。

**第 1 层：精确表达式。** 由 (Prager–Synge) 等式（注意左边第一项 $=$ 位移能量误差）：
$$\eta^2=\|\varepsilon(u_h)-\varepsilon(u)\|_{\mathbb C_d}^2+\|\sigma_h-\sigma\|_{\mathbb A(d)}^2
  \ \Longrightarrow\
  \Theta^2=1+\frac{\|\sigma_h-\sigma\|_{\mathbb A(d)}^2}{\|\mathbb C_d\varepsilon(u_h)-\sigma\|_{\mathbb A(d)}^2}. \tag{7}$$
**充要条件**：$\Theta\to1\iff$ Hu–Zhang 应力误差 $\|\sigma_h-\sigma\|_{\mathbb A}$ 收敛**不慢于**标准 FEM
的原始应力误差 $\|\mathbb C_d\varepsilon(u_h)-\sigma\|_{\mathbb A}$。两者来自**两个独立离散**，
须选次数使阶相容：取 Hu–Zhang 阶 $\ge$ 标准 FEM 应力阶，则比值 $\to0$、$\Theta\to1$。
（光滑解、Hu–Zhang 应力通常更准 ⇒ 这是常态。）

**第 2 层：朴素全局界（诚实的坏消息）。** 把 (7) 的加权比值解耦放缩到无权比值，
$g^{-1}\in[1,k_{\mathrm{res}}^{-1}]$ ⇒
$$\Theta^2-1\ \le\ k_{\mathrm{res}}^{-1}\cdot
  \frac{\|\sigma_h-\sigma\|_{\mathbb C^{-1}}^2}{\|\mathbb C_d\varepsilon(u_h)-\sigma\|_{\mathbb A(d)}^2}. \tag{8}$$
**这个界确实含 $k_{\mathrm{res}}^{-1}$，最坏情况会爆。** v0.1 直接说「$\kappa_T=O(1)$ 所以有界」
是把结论当前提——不成立。(8) 是悲观的、可被构造的反例触发的界，论文不能停在这里。

**第 3 层：局部对比度界（真正起作用的，也是要数值坐实的）。** 平衡型估计子的局部效率常数
（Braess–Pillwein–Schöberl 型分析）依赖的是 patch 上的**对比度**而非全局 $k_{\mathrm{res}}^{-1}$：
$$\eta_T\ \le\ C_{\mathrm{eff}}(\kappa_{\omega_T})\,\big(\|\varepsilon(u_h)-\varepsilon(u)\|_{\mathbb C_d,\,\omega_T}
  +\mathrm{osc}_{\omega_T}\big),\qquad
  \kappa_{\omega_T}:=\frac{\sup_{\omega_T}g}{\inf_{\omega_T}g}. \tag{9}$$
关键在 $d$ 由相场方程**连续**解出 ⇒ $g$ 在 patch 尺度上光滑 ⇒ $\kappa_{\omega_T}=1+O(h_T\,\|\nabla g\|_\infty/\inf_{\omega_T}g)=O(1)$，
**退化不在单元内剧变**。于是 $C_{\mathrm{eff}}$ 实际有界，与全局 $k_{\mathrm{res}}$ **解耦**。

> **审稿核心命题（M0 必须用数值坐实）**：实测 $\Theta$ 跟随**局部对比度 $\kappa_{\omega_T}$**，
> **不**跟随 (8) 的全局朴素界 $k_{\mathrm{res}}^{-1/2}$。操作：扫 $k_{\mathrm{res}}\in\{10^{-3},10^{-5},10^{-7}\}$
> 量 $\Theta$；预期 $\Theta$ 随 $k_{\mathrm{res}}\to0$ **平稳**（被 (9) 控制），而非按 (8) 发散。
> **若数值发散 ⇒ 触发 plan §4 M0 门槛**，退回无分裂版本或限定 $k_{\mathrm{res}}$ 范围声明适用域。
> 这一对照（实测 vs 朴素界）本身就是论文里有效性分析的核心图。

**实践校正。** 加权 $g^{-1}$ 把估计子质量堆到裂纹带——这恰是「该加密的地方」。
即使 $\Theta$ 在带内偏大，**标记方向仍正确**（指示子在裂纹带高 ⇒ 加密裂纹带）。
所以即便有效性常数略差，**自适应行为依然对**。这点要在论文里讲清，把潜在弱点转成中性。

---

## 6. Theorem 2（拉压能量分裂：凸对偶 majorant）

断裂相场常用拉压分裂（Amor 体积/偏量、Miehe 谱分裂），只退化「拉伸」部分：
$$\psi(\varepsilon,d)=g(d)\,\psi^+(\varepsilon)+\psi^-(\varepsilon),\qquad
  \sigma=g(d)\,\partial_\varepsilon\psi^+ +\partial_\varepsilon\psi^-. \tag{7'}$$
此时 (1) 是**非线性**（$\sigma$ 在 $\varepsilon$ 中分片线性、但非单一 $\mathbb C_d$），
Prager–Synge 的二次恒等式不再直接套用。出路：**Repin 型 functional a posteriori majorant**。

### 6.1 设定与凸性前提

记总能量泛函（$\ell(v):=\int_\Omega f\cdot v+\int_{\Gamma_N}t_N\cdot v$）
$$J(v):=\int_\Omega\psi(\varepsilon(v),d)\,dx-\ell(v),\qquad v\in V_g,$$
真解 $u=\arg\min_{V_g}J$。**前提**：Amor（体积/偏量）与 Miehe（谱）分裂的
$\psi^+,\psi^-$ 均为 $\varepsilon$ 的**凸**函数（谱分裂凸性见 Amor 2009 / Pham 2011），
故 $\psi(\cdot,d)$ 对每个固定 $d$ 凸，$J$ 凸、$u$ 唯一。

为定量，进一步设 $\psi(\cdot,d)$ 关于度量 $\|\cdot\|_{\mathbb C_d}$ **一致凸**，模 $\alpha(d)>0$：
$$\psi(\varepsilon_2,d)\ge\psi(\varepsilon_1,d)+\partial_\varepsilon\psi(\varepsilon_1,d):(\varepsilon_2-\varepsilon_1)
  +\tfrac{\alpha(d)}{2}\|\varepsilon_2-\varepsilon_1\|_{\mathbb C}^2. \tag{10}$$
退化使**拉伸方向** $\alpha\sim k_{\mathrm{res}}$（弱凸），压缩方向 $\alpha\sim1$——这是分裂情形
有效性 $k$-依赖的根源（§6.4 诚实标注）。

### 6.2 Theorem 2（functional majorant，完整证明）

**命题。** 设 $J$ 凸、(10) 成立。$\psi(\cdot,d)$ 的 Legendre–Fenchel 共轭
$\psi^\*(\tau,d):=\sup_{\rho\in\mathbb S}\big(\tau:\rho-\psi(\rho,d)\big)$。
定义 majorant
$$\mathcal M(v,\tau):=\int_\Omega\big[\psi(\varepsilon(v),d)+\psi^\*(\tau,d)-\varepsilon(v):\tau\big]\,dx,
  \qquad v\in V_g,\ \tau\in\Sigma_f. \tag{11}$$
则
$$\tfrac{\alpha_{\min}}{2}\,\|\varepsilon(v)-\varepsilon(u)\|_{\mathbb C}^2
  \ \le\ J(v)-J(u)\ \le\ \mathcal M(v,\tau),\qquad
  \alpha_{\min}:=\operatorname*{ess\,inf}_\Omega\alpha(d). \tag{12}$$
取 $v=u_h$、$\tau=\sigma_h$（Hu–Zhang 平衡应力，$\Sigma_f$ 约束自动满足）给**保证型上界**。

**证明（三步）。**

*第一步：被积函数逐点非负。* Fenchel–Young 不等式对凸 $\psi$ 恒成立：
$$\psi(\rho,d)+\psi^\*(\tau,d)\ge\rho:\tau\quad\forall\rho,\tau,\tag{FY}$$
取 $\rho=\varepsilon(v)$ ⇒ (11) 被积函数 $\ge0$ ⇒ $\mathcal M(v,\tau)\ge0$，且 $=0$ 逐点
$\iff\tau=\partial_\varepsilon\psi(\varepsilon(v),d)$（FY 取等条件）。

*第二步：$\mathcal M$ 控制能量差。* 关键恒等式——对**任意** $\tau\in\Sigma_f$，
$$\mathcal M(v,\tau)-\big(J(v)-J(u)\big)=\mathcal M(u,\tau)\ \ge\ 0. \tag{13}$$
证 (13)：把 $\mathcal M(v,\tau)-\mathcal M(u,\tau)$ 展开，
$$=\int_\Omega\!\big[\psi(\varepsilon(v),d)-\psi(\varepsilon(u),d)-(\varepsilon(v)-\varepsilon(u)):\tau\big].$$
而 $J(v)-J(u)=\int_\Omega[\psi(\varepsilon(v),d)-\psi(\varepsilon(u),d)]-\ell(v-u)$。两式相减：
$$\big[\mathcal M(v,\tau)-\mathcal M(u,\tau)\big]-\big[J(v)-J(u)\big]
  =\ell(v-u)-\int_\Omega(\varepsilon(v)-\varepsilon(u)):\tau.$$
令 $w=v-u\in V_0$。右端 $\int_\Omega\varepsilon(w):\tau$ 分部积分（$\tau\in\Sigma_f$，对称）：
$$\int_\Omega\varepsilon(w):\tau=-\int_\Omega w\cdot\operatorname{div}\tau+\langle\tau n,w\rangle_{\partial\Omega}
  =\int_\Omega f\cdot w+\int_{\Gamma_N}t_N\cdot w=\ell(w),$$
（用 $-\operatorname{div}\tau=f$、$\tau n=t_N$ 于 $\Gamma_N$、$w=0$ 于 $\Gamma_D$）。故右端
$\ell(w)-\ell(w)=0$，即 $\mathcal M(v,\tau)-\mathcal M(u,\tau)=J(v)-J(u)$，移项得 (13)。
由第一步 $\mathcal M(u,\tau)\ge0$ ⇒ $J(v)-J(u)\le\mathcal M(v,\tau)$，(12) 右半得证。

*第三步：能量差控制误差（左半）。* $u$ 是极小 ⇒ $J'(u)[w]=0$。一致凸 (10) 取
$\varepsilon_1=\varepsilon(u),\varepsilon_2=\varepsilon(v)$，并用一阶最优性消去线性项：
$$J(v)-J(u)\ge\tfrac{\alpha_{\min}}{2}\|\varepsilon(v)-\varepsilon(u)\|_{\mathbb C}^2.$$
合并三步得 (12)。$\qquad\blacksquare$

**要点。**
- (12) 推广 (Prager–Synge)：$\psi$ 纯二次（无分裂、$\alpha=g$）时 $\mathcal M(u_h,\sigma_h)=\tfrac12\eta^2$，
  退化回 (4) 的平方。
- 证明**只需一个平衡应力 $\tau\in\Sigma_f$** ⇒ Hu–Zhang 角色不变、优势不变。
- (13) 是关键：majorant 与能量差的**间隙恰为 $\mathcal M(u,\tau)$**，即 $\tau$ 偏离真应力
  $\sigma=\partial_\varepsilon\psi(\varepsilon(u),d)$ 的程度 ⇒ $\tau=\sigma_h$ 越准界越紧。

### 6.3 Amor 分裂的对偶势 $\psi^\*$ 闭式

Amor 体积/偏量分裂（$K$ = 体积模量，**2D 平面应变 $K=\lambda+\mu$**；$\mu$ = 剪切模量，
$\operatorname{tr}/\operatorname{dev}$ 为迹/偏量）：
$$\psi^\pm:\quad \psi^+=\tfrac{K}{2}\langle\operatorname{tr}\varepsilon\rangle_+^2+\mu\,|\operatorname{dev}\varepsilon|^2,\quad
  \psi^-=\tfrac{K}{2}\langle\operatorname{tr}\varepsilon\rangle_-^2,$$
$\langle x\rangle_\pm=\tfrac12(x\pm|x|)$。退化后
$\psi(\varepsilon,d)=g\big(\tfrac{K}{2}\langle\operatorname{tr}\varepsilon\rangle_+^2+\mu|\operatorname{dev}\varepsilon|^2\big)
  +\tfrac{K}{2}\langle\operatorname{tr}\varepsilon\rangle_-^2$。
体积/偏量正交 ⇒ 共轭可分。**关键**：应力–应变配对按
$\tau:\varepsilon=\tfrac12\operatorname{tr}\tau\,\operatorname{tr}\varepsilon+\operatorname{dev}\tau:\operatorname{dev}\varepsilon$，
迹通道的对偶变量是 $\tfrac12\operatorname{tr}\tau$（**不是 $\operatorname{tr}\tau$**），Legendre 共轭带出 $1/4$ 因子。
逐块 Legendre 给**闭式**：
$$\boxed{\ \psi^\*(\tau,d)=\frac{1}{8K}\!\left[\frac{\langle p\rangle_+^2}{g}+\langle p\rangle_-^2\right]
  +\frac{|\operatorname{dev}\tau|^2}{4g\mu},\qquad p:=\operatorname{tr}\tau.\ }\tag{14}$$
（推导：迹通道是 $s\mapsto\tfrac{gK}{2}\langle s\rangle_+^2+\tfrac{K}{2}\langle s\rangle_-^2$，$s=\operatorname{tr}\varepsilon$，
其共轭在对偶变量 $q=\tfrac12\operatorname{tr}\tau=p/2$ 处为 $\tfrac{\langle q\rangle_+^2}{gK}+\tfrac{\langle q\rangle_-^2}{K}$，
代 $q=p/2$ 得 $\tfrac{\langle p\rangle_+^2}{4gK}+\tfrac{\langle p\rangle_-^2}{4K}$，即 (14) 迹项；
$\operatorname{dev}$ 通道 $2g\mu|\operatorname{dev}\varepsilon|^2$ 的共轭为 $|\operatorname{dev}\tau|^2/(4g\mu)$。）
**(14) 已用 brute-force Legendre 数值验到机器精度**（T5，扫 $g\in\{1,0.37,10^{-3}\}$，max rel 8.5e-16）。
$f=0$ 基准下 (14) 全显式可算，估计子 $\mathcal M(u_h,\sigma_h)$ 逐元积分即得。

### 6.4 诚实边界

- **谱分裂**：$\psi^\*$ 无闭式（特征投影非线性）⇒ 需逐点小凸优化或近似。**建议 M0/M3 先做 Amor**
  （(14) 闭式、最干净）坐实 Theorem 2，谱分裂留 §Discussion / future work。
- **有效性的 $k$-依赖**：(12) 左半含 $\alpha_{\min}\sim k_{\mathrm{res}}$（拉伸弱凸）⇒ majorant 与真误差
  之比上界 $\sim\alpha_{\min}^{-1}$，**比无分裂更脆**。这是 plan §5 的 split-effectivity 风险，
  与 §5 第 2/3 层同源：朴素界含 $k_{\mathrm{res}}^{-1}$，实际有效性靠 $g$ 局部光滑 ⇒ M0 数值门槛覆盖。
  **可靠性（上界 (12)）不受影响**——对任意 $\alpha_{\min}>0$ 严格成立。

---

## 7. 小结：论文能严格断言什么

| 情形 | 可靠性（上界，常数=1） | 有效性（$\Theta$ 有界） | 数据振荡 |
|---|---|---|---|
| 无分裂，$f=0$ | ✅ 严格（Thm 1） | ✅ 光滑 $\Theta\to1$；退化区依赖 $g$ 局部光滑性（M0 验） | 无 |
| 无分裂，$f\ne0$ | ✅ 严格（Thm 1+osc） | ✅ 同上 | $+\mathrm{osc}(f)$，高阶小 |
| Amor 分裂 | ✅（Thm 2 majorant） | ⚠️ 需强凸，M0 门槛验 | 无（$f=0$） |
| 谱分裂 | ✅（Thm 2，对偶需数值） | ⚠️ future work | 无 |

**最稳的头条**（先确保这条无懈可击）：
*无能量分裂、$f=0$ 的退化弹性子问题上，Hu–Zhang 平衡应力给出可靠性常数严格等于 1、
无数据振荡、reconstruction-free 的 a posteriori 误差上界，驱动自适应加密。*
分裂情形（Thm 2）作为推广，有效性以 M0 数值坐实后写入。

## 8. 待验证清单（喂给 M0 概念验证）

1. 光滑 MMS（$d\equiv0$，$f=0$）：$\Theta\to1$，机器精度对账 (Prager–Synge) 等式两端。
2. 固定解析 $d$ 场（如 $d=\exp(-(\text{dist to crack})^2/\ell^2)$）：扫 $h$ 看 $\Theta$ 平稳。
3. 扫 $k_{\mathrm{res}}\in\{10^{-3},10^{-5},10^{-7}\}$：$\Theta$ 不随 $k_{\mathrm{res}}\to0$ 发散（核心断言）。
4. Amor 分裂下重复 1–3，量 $\mathcal M$-effectivity，定 split 版本是否进正文。
5. $f\ne0$ 一例：验 osc 项随 $h$ 高阶衰减、界仍是上界。
