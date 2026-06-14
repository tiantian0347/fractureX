# 理论：Hu–Zhang 应力驱动的预测型（anticipatory）自适应标记

> 状态：推导稿 v0.1（2026-06-13）。配合
> [THEORY_equilibrated_aposteriori.md](THEORY_equilibrated_aposteriori.md)（误差估计子）与
> [../routes/plan_adaptive_aposteriori.md](../routes/plan_adaptive_aposteriori.md)（路线）。
> 触发：M3-full（model1）实测峰值载荷较 nx=120 参照**高估 +16%**（见 RESULTS §M3 full），
> 根因是**每载荷步只加密一次、且标记量滞后裂尖**，裂纹带停在 $h/l_0\approx0.70$ 未达 $h\le l_0/2$。
> 本文给出修正的两条理论支柱：**(i) 用 Hu–Zhang 精确应力构造的预测型标记量**（领先裂尖、
> 能捕捉起裂），**(ii) 步内 predictor–corrector 反复加密直到网格分辨 $l_0$**。
> 记号承 [THEORY_equilibrated_aposteriori.md](THEORY_equilibrated_aposteriori.md) §1。

---

## 1. 记号与相场模型（AT2）

承上：退化 $g(d)=(1-d)^2+k_{\mathrm{res}}$，退化应力 $\sigma=\mathbb C_d\varepsilon(u)$，
柔度 $\mathbb A(d)=g^{-1}\mathbb C^{-1}$。拉压分裂时拉伸驱动能 $\psi^+(\varepsilon)$（Amor/谱）。
**历史场**（不可逆性）$H:=\max_{[0,t]}\psi^+(\varepsilon)$。AT2 裂纹密度 $\gamma_\ell(d)=\frac{1}{2l_0}(d^2+l_0^2|\nabla d|^2)$。

相场子问题（$u$ 冻结、对 $d$ 极小，AT2、谱/混合历史驱动）的一阶条件：
$$
-2(1-d)\,H+\frac{G_c}{l_0}\big(d-l_0^2\,\Delta d\big)=0\quad\text{in }\Omega,
\qquad \partial_n d=0\ \text{on }\partial\Omega. \tag{1}
$$

**定义（无量纲驱动力）。**
$$
\boxed{\ \mathcal D:=\frac{2 l_0}{G_c}\,H\ \ge 0.\ }\tag{2}
$$
$\mathcal D$ 把「拉伸弹性能 vs 断裂韧性」无量纲化，是判定某处是否**将要**开裂的自然标尺。

---

## 2. 1D AT2 标定：临界驱动力、临界应力、最优剖面

为给标记阈值与网格尺寸下限**定量的理论锚点**（而非经验数），先在 1D、无能量分裂、
$k_{\mathrm{res}}\to0$ 下解析标定 AT2。所有结果已 sympy 符号核验。

### 2.1 均匀响应与临界驱动力 $\mathcal D_c=\tfrac13$

均匀杆（$\nabla d=0$）能量密度 $W(\varepsilon,d)=\tfrac12 g(d)E\varepsilon^2+\tfrac{G_c}{2l_0}d^2$，$g=(1-d)^2$。
对 $d$ 稳态 $\partial_d W=0$：
$$-(1-d)E\varepsilon^2+\tfrac{G_c}{l_0}d=0
\ \Longrightarrow\
d^\*(\varepsilon)=\frac{\mathcal D}{1+\mathcal D},\qquad
\mathcal D=\frac{l_0 E\varepsilon^2}{G_c}=\frac{2l_0}{G_c}H,\ \ H=\tfrac12 E\varepsilon^2. \tag{3}$$
均匀应力 $\sigma=g(d^\*)E\varepsilon=\dfrac{E\varepsilon}{(1+\mathcal D)^2}$。沿单调加载 $\varepsilon\uparrow$ 求 $\sigma$ 极值
$\dfrac{d\sigma}{d\varepsilon}=E\dfrac{1-3\mathcal D}{(1+\mathcal D)^3}=0$：
$$\boxed{\ \mathcal D_c=\tfrac13,\qquad d_c=\tfrac14,\qquad
\sigma_c=\frac{3\sqrt3}{16}\sqrt{\frac{EG_c}{l_0}}=\frac{9}{16}\sqrt{\frac{EG_c}{3l_0}}.\ }\tag{4}$$
$\sigma_c$ 即 Pham–Marigo–Maurini 经典 AT2 临界应力。**判读**：$\mathcal D<\tfrac13$ 时 $\sigma$ 随载荷
上升（稳定、损伤弥散弱增长）；$\mathcal D=\tfrac13$ 起 $\sigma$ 转降（软化/局部化触发）。
故 **$\mathcal D_c=1/3$ 是「即将开裂」的精确分界，且触发时 $d_c=1/4$ 仍小**——软化在 $d$ 远未到 1 时已启动。

### 2.2 全局裂纹最优剖面与带宽 $l_0$

完全开裂（$d(0)=1,\ d(\pm\infty)=0$）时耗散能 $\int\tfrac{G_c}{2l_0}(d^2+l_0^2 d'^2)$ 的极小子由
Euler–Lagrange $d-l_0^2 d''=0$ 给出：
$$d(x)=e^{-|x|/l_0},\qquad \int_{\mathbb R}\tfrac{1}{2l_0}\big(d^2+l_0^2 d'^2\big)\,dx=1\ (\times G_c). \tag{5}$$
（表面能恰 $=G_c$，符号核验。）**裂纹带 e-折叠宽度 $=l_0$**，离散须在带内放足够单元（§3）。

### 2.3 命题 1（领先指标性质，严格版）

定义沿加载历史的**软化前沿** $\Gamma_c(t):=\{x:\ \mathcal D(x,t)=\tfrac13\}$（局部化触发面，(4)），
预测型标记集 $\mathcal M_{\mathrm{DF}}:=\{x:\ \mathcal D(x,t)\ge\theta_D\}$，
$d$-型标记集 $\mathcal M_d:=\{x:\ d(x,t)\ge\theta_d\}$。

**命题 1.** 取阈值 $0<\theta_D<\tfrac13$ 与 $\theta_d>\tfrac14$。则对单调准静态加载：
(i) $\mathcal M_{\mathrm{DF}}\supseteq\{\mathcal D\ge\tfrac13\}$，并含 $\Gamma_c$ **前方** $\mathcal D\in[\theta_D,\tfrac13)$
的尚未软化邻域；
(ii) $\mathcal M_d=\{d\ge\theta_d\}=\{\mathcal D\ge\tfrac{\theta_d}{1-\theta_d}\}$ 且 $\tfrac{\theta_d}{1-\theta_d}>\tfrac13$，
即 $\mathcal M_d$ 落在软化前沿**内侧**。
故 $\mathcal M_d\subsetneq\mathcal M_{\mathrm{DF}}$，两集合之差恰为裂尖前方「将软化未软化」带——
**$\mathcal M_{\mathrm{DF}}$ 领先 $d$-型标记**。

**证明.** (i) 由定义 $\mathcal M_{\mathrm{DF}}$ 含一切 $\mathcal D\ge\theta_D$；$\theta_D<\tfrac13$ 故含 $\{\mathcal D\ge\tfrac13\}$
及其前方 $[\theta_D,\tfrac13)$ 邻域。(ii) (3) 的 $d^\*(\mathcal D)=\mathcal D/(1+\mathcal D)$ 严格增 ⇒
$d\ge\theta_d\iff\mathcal D\ge\theta_d/(1-\theta_d)$；$\theta_d>\tfrac14\Rightarrow\theta_d/(1-\theta_d)>\tfrac13$。
二集合相对前沿 $\{\mathcal D=\tfrac13\}$ 一前一后，包含严格。$\qquad\blacksquare$

**推论（起裂）.** 起裂时刻 $d\equiv0,\ \nabla d\equiv0$ 于全域 ⇒ 任何基于 $d$ 或 $\nabla d$ 的标记量
**恒零、无信号**；而应力集中点 $\mathcal D$ 随载荷**先**抵 $\tfrac13$，$\mathcal M_{\mathrm{DF}}$ 在该处先触发 ⇒
唯驱动力型能**标记起裂位置**。这是「damage-AMR 无法捕捉起裂」的数学根源
（[CMAME 2022](https://www.sciencedirect.com/science/article/abs/pii/S0045782522004339)）。

### 2.4 与用户既有标准 FEM 标记量的关系

用户的相场梯度重构（ZZ）指示子 $\eta_\tau^{\mathrm{rec}}=\|R_h d_h-\nabla d_h\|_{0,\tau}$
（[arXiv:2410.01177](https://arxiv.org/pdf/2410.01177)）支撑在 $\{\nabla d\ne0\}$=现有裂纹带，
属 $\mathcal M_d$ 类（命题 1(ii)，落软化前沿内侧）⇒ **滞后**、起裂时恒零（推论）。
裂纹已存在时跟带加密有效（故"还不错"），但不预测裂尖、不捕起裂。
$\mathcal D$-型是其**预测型替代/补充**。

---

## 3. 网格尺寸下限 $h\le l_0/2$（由 (5) 最优剖面定）

由 (5) 裂纹带是 e-折叠宽 $l_0$ 的 $e^{-|x|/l_0}$ 剖面。线性/二次 FE 要分辨该指数剖面、
正确逼近表面能到 $G_c\mathcal H^{1}(\text{crack})$（$\Gamma$-收敛，Ambrosio–Tortorelli；
Bourdin–Francfort–Marigo），须每 e-折叠长放 $\gtrsim 2$–4 单元：
$$\boxed{\ h_\tau\le \tfrac12\,l_0\quad\text{（保守：峰值载荷网格无关的最低要求；更准用 }l_0/4\text{）}.}\tag{6}$$
欠分辨（$h\gtrsim l_0$）系统性**高估有效 $G_c\Rightarrow$ 高估峰值载荷、推迟起裂**
（Gerasimov–De Lorenzis; Miehe et al.）——M3-full $+16\%$ 峰值高估即此（带 $h/l_0\approx0.70$）。
$\Gamma$-收敛只要求在**裂纹路径上**分辨 $l_0$ ⇒ 自适应只把 (6) 施于 $\mathcal D$ 大处
（命题 1 的前沿邻域），均匀全域 $h\le l_0/2$ 是 DOF 浪费。**该加密处 = $\mathcal D$ 大处**——
$\mathcal D$-型标记 (§2) + 尺寸下限 (6) 把 DOF 精确投到刀刃。

---

## 4. 为什么用 Hu–Zhang 的应力来算 $\mathcal D$（模型特定的理论优势）

$\mathcal D$（经 $H=\max\psi^+$）由**应力/应变**算出。两种来源精度天差地别：

- **标准位移 FEM**：应力 $\mathbb C_d\varepsilon(u_h)$ 逐元不连续、法向不连续、**不平衡**，
  收敛阶 $O(h^{p_u})$；在裂尖（应力高梯度/奇异）**欠分辨最严重**——恰好是 $\mathcal D$ 最该准的地方。
  用它算的预测型标记**在刀刃上被污染**。
- **Hu–Zhang 混合元**：应力 $\sigma_h$ 逐点对称、$H(\operatorname{div})$-协调、逐元平衡
  （[THEORY_equilibrated §2](THEORY_equilibrated_aposteriori.md)），阶 $O(h^{p+1})$（高一阶且平衡）。
  $\psi^+(\sigma_h)$（经柔度 $\varepsilon=\mathbb A\sigma_h$ 取拉伸部分）在裂尖**精确**。

**命题 2（标记可靠性，定量敏感度）。** 标记判据 $\mathcal D_\tau\ge\theta_D$，$\mathcal D=\tfrac{2l_0}{G_c}\psi^+(\sigma)$
（经柔度由应力算）。设近似应力 $\tilde\sigma$（$\tilde\sigma=\sigma_h$ 或 $\mathbb C_d\varepsilon(u_h)$）误差
$\delta:=\tilde\sigma-\sigma$。因 $\psi^+$ 在 $\sigma$ 中二次，
$$\frac{|\widetilde{\mathcal D}-\mathcal D|}{\mathcal D}\ \le\ 2\,\frac{\|\delta\|}{\|\sigma\|}+O\!\Big(\tfrac{\|\delta\|^2}{\|\sigma\|^2}\Big). \tag{*}$$
设 $\mathcal D$ 在前沿邻域横向梯度 $|\nabla\mathcal D|>0$（应力集中单调过渡），则误标只发生在
$\{|\mathcal D-\theta_D|\lesssim |\widetilde{\mathcal D}-\mathcal D|\}$ 的薄带，其宽度 $\sim
\tfrac{2\theta_D}{|\nabla\mathcal D|}\tfrac{\|\delta\|}{\|\sigma\|}$，**正比于相对应力误差**。
代入两种来源：Hu–Zhang $\|\delta\|/\|\sigma\|=O(h^{p+1})$（平衡、高一阶）⇒ 误标带 $O(h^{p+1})$；
标准 FEM 原始应力 $O(h^{p_u})$ 且在裂尖（高应力梯度、奇异）相对误差**最大** ⇒ 误标带最宽、
恰在最关键处。故用 $\sigma_h$ 算 $\mathcal D$ 的标记集合更干净，差距在裂尖尤甚。$\qquad\blacksquare$

> 注：(*) 把「Hu–Zhang 应力精度高」从定性优势变成**标记误差的定量界**——这是本模型选 $\sigma_h$
> 而非位移导出应力做标记的理论依据。

> 这把「Hu–Zhang 贵在哪、值在哪」从误差估计（[THEORY_equilibrated](THEORY_equilibrated_aposteriori.md)）
> **延伸到标记**：精确平衡应力既给保证型误差界，又给可靠的预测型裂尖标记。

---

## 5. 认证链接：平衡估计子 $\eta_\tau$ 本身就是「应力型 + 预测型 + 保证型」

[THEORY_equilibrated (5)](THEORY_equilibrated_aposteriori.md) 的逐元估计子
$$
\eta_\tau=\Big(\int_\tau g^{-1}\,r:\mathbb C^{-1}r\Big)^{1/2},\qquad
r:=\mathbb C_d\varepsilon(u_h)-\sigma_h, \tag{7}
$$
有三重身份：

1. **应力型**：$r$ 是「位移导出应力 − 平衡应力」之差，纯应力量。
2. **预测型**：裂尖处应力集中 $\Rightarrow$ 标准 FEM 的 $\mathbb C_d\varepsilon(u_h)$ 与平衡 $\sigma_h$
   差距 $r$ 最大 $\Rightarrow\eta_\tau$ 在裂尖（含**起裂前**应力集中点）高 $\Rightarrow$ 领先 $d$。
3. **保证型**：Prager–Synge 给 $\|\varepsilon(u_h)-\varepsilon(u)\|_{\mathbb C_d}\le\eta$，可靠性常数 $=1$。

**关系。** $\mathcal D_\tau$（物理驱动力，捕捉起裂/路径，无需 $u_h$）与 $\eta_\tau$（认证弹性误差，
需协调 $u_h$）**互补**：二者都在裂尖峰值。工程上用 $\mathcal D_\tau$ 驱动加密（鲁棒、含起裂），
用 $\eta_\tau$ 给论文的**保证型误差认证**。代价：$\eta_\tau$ 需一次标准连续 FEM 解 $u_h$
（即路线 (a) primal 重解；Hu–Zhang 已免费给 $\sigma_h$）。

---

## 6. 标记策略（三选一 + 推荐组合）

记单元尺寸 $h_\tau$、单元驱动力 $\mathcal D_\tau:=\frac{2l_0}{G_c}\max_{q\in\tau}H_q$。

| 记号 | 标记集合 $\mathcal M$ | 性质 | 理论依据 | 需 $u_h$? |
|---|---|---|---|---|
| **M-DF** 驱动力 | $\{\tau:\ \mathcal D_\tau\ge\theta_D\ \wedge\ h_\tau>\tfrac{l_0}{2}\}$，$\theta_D=\beta\,\mathcal D_c=\tfrac{\beta}{3}$ | 预测、含起裂 | §2 命题1+(4) + §3 (6) | 否 |
| **M-EQ** 平衡误差 | Dörfler$(\{\eta_\tau\},\theta)\ \wedge\ h_\tau>\tfrac{l_0}{2}$ | 保证、预测 | §5 + [THEORY_equilibrated Thm1] | 是 |
| **M-REC** 应力重构 | $\{\tau:\ \|\sigma_h-G_h\sigma_h\|_\tau\ \text{大}\}$ | ZZ 渐近 | $\sigma$ 上的 ZZ（用户 $d$-重构的应力类比） | 否 |

- **M-DF**（推荐主标记）：**绝对阈值** $\theta_D=\beta\,\mathcal D_c=\beta/3$，$\beta\in(0,1)$——由 (4) 的临界
  驱动力 $\mathcal D_c=\tfrac13$ 直接定标：$\beta$ 越小越提前（裂尖前方更远即加密，更预测但更多 DOF），
  $\beta\to1$ 退到「软化触发处才加密」。建议 $\beta\in[0.3,0.9]$（即 $\theta_D\in[0.1,0.3]$）。
  $h_\tau>l_0/2$ 下限保证 (6) 且**自动终止**加密（有限层）。这是
  [CMAME 2022 effective-driving-energy](https://www.sciencedirect.com/science/article/abs/pii/S0045782522004339)、
  [stress-based AMR (FEniCS)](https://www.sciencedirect.com/science/article/pii/S0168874X24002051) 的预测型路线，
  但**用 Hu–Zhang 精确 $\sigma_h$ 算 $\mathcal D$**（命题 2）。
- **M-EQ**（认证伴随）：保留论文「保证型」头条；需 primal 重解。
- **M-REC**（备选）：与用户既有 $d$-重构最平行，但移到精确 $\sigma_h$ 上；无保证常数（ZZ 仅渐近精确）。

**推荐组合（mark–unmark 多级）**：主用 **M-DF** 驱动加密（鲁棒、捕起裂、定路径），
**M-EQ 的 $\eta_\tau$ 作认证报告**。可叠加 $d$-带细化（M-REC/用户 $d$-重构）作裂纹芯精修——
即 [CMAME 2022] 的「驱动能标记 + 损伤标记」二级方案。

---

## 7. Predictor–corrector 步内反复加密（修正"加密次数太少"）

**算法（单载荷步 $n$，由 $\mathcal M$ 触发）。**
```
给定上一步网格/状态 (T^{n-1}, d^{n-1}, H^{n-1}):
  T ← T^{n-1};  迁移状态到 T
  repeat (corrector k=0,1,...):
    在 T 上 staggered 求解本步 (σ_h, u_h, d) 至收敛
    评估标记量（M-DF 用 H/σ_h；M-EQ 用 η_τ）→ 标记集合 M
    if M = ∅:  break                      # 网格已分辨 l0，接受
    T ← bisect(T, M);  迁移 (d, r_hist) 到细网格（σ/u 下轮重解）
  accept (T^n, ·) ← (T, ·);  推进载荷
```
这是 [Heister–Wheeler–Wick 2015, CMAME 290:466](https://www.sciencedirect.com/science/article/abs/pii/S0045782515001115)
的 predictor–corrector：**步内反复**「解→标记→加密→重解」直到网格不再变，再进下一载荷步。
相比"每步只加密一次"，单步可加密多次，裂尖在本步内即被分辨到 $h\le l_0/2$。

**命题 3（终止性）。** $\mathcal M$ 的尺寸下限 $h_\tau>l_0/2$ 使任一单元至多被加密
$\lceil\log_2(h_0/(l_0/2))\rceil$ 次（$h_0$=初始尺寸）；故 corrector 循环在有限步内 $\mathcal M=\varnothing$ 终止。$\qquad\square$

**命题 4（接受态的分辨保证）。** 终止时 $\mathcal M=\varnothing\Rightarrow$ 所有 $\mathcal D_\tau\ge\theta_D\max\mathcal D$
的单元满足 $h_\tau\le l_0/2$ $\Rightarrow$ 本步裂纹路径处分辨 (6) 成立 $\Rightarrow$ 由 §3，
峰值载荷的网格依赖系统性偏差被消除（趋向 $\Gamma$-收敛极限）。$\qquad\square$

> 命题 4 是对 M3-full $+16\%$ 高估的**理论修复保证**：predictor–corrector + M-DF + 下限 (6)
> $\Rightarrow$ 接受态网格处处分辨裂纹路径 $\Rightarrow$ 峰值载荷应收敛向 nx=120 参照（待 §(b) 数值坐实）。

---

## 8. 诚实边界

- **M-DF 是分辨率/物理判据，非认证误差界**：其依据是 $\Gamma$-收敛分辨要求（§3）+ 领先指标
  性质（§2），**不**提供 $\eta\ge\|\text{err}\|$ 式保证。保证型只来自 **M-EQ 的 $\eta_\tau$**（§5、Thm 1）。
  论文应：用 M-DF 定网格（鲁棒、含起裂），用 $\eta_\tau$ 认证误差——分工写清。
- **$\mathcal D$ 经 $H$ 含历史最大**：单调准静态加载下 $H$ 单调，$\mathcal D$ 是良定领先量；
  非单调/卸载需谨慎（$H$ 冻结历史，$\mathcal D$ 不回落）。本文限单调加载。
- **$h\le l_0/2$：带宽 $l_0$ 严格（(5) 剖面 e-折叠长），但每 e-折叠的「单元数」是实践选择**——
  $l_0/2$（约 2 单元/折叠）是「峰值载荷网格无关」的最低保守要求，部分文献用 $l_0/4$、$l_0/5$
  求更准峰值。若 §(b) 数值显示 $l_0/2$ 仍偏高，收紧到 $l_0/4$（命题 4 的分辨阈相应改）。
- **$\mathcal D_c=1/3$ 由 1D、无分裂、$k_{\mathrm{res}}\to0$ 标定**：含拉压分裂时临界值随分裂模型与应力态
  漂移（同量级 $O(1)$）；故 M-DF 用 $\theta_D=\beta\mathcal D_c$ 的**相对标定** $\beta$ 比绝对数稳健，
  $\beta$ 由 §(b) 标定到峰值载荷收敛。
- **谱分裂的 $\psi^+$** 用于 $H$ 时与 [THEORY_equilibrated §6.4] 同：Amor 闭式最干净，谱分裂数值。

---

## 9. 待数值坐实（喂给 model1 重跑 + §(b)）

1. **峰值载荷收敛**：M-DF + predictor–corrector（$h\le l_0/2$）重跑 model1，验峰值载荷由 $+16\%$
   收敛向 nx=120 参照（命题 4 的数值证据）；扫下限 $l_0/2\to l_0/4$ 看是否进一步收敛。
2. **预测性**：可视化起裂前 $\mathcal D_\tau$（或 $\eta_\tau$）与 $d$，验 $\mathcal D$ 标记**领先**裂尖
   （命题 1）；对照 $d$-重构标记的滞后。
3. **DOF 效率**：等峰值精度下，自适应（M-DF）DOF vs 均匀 nx 扫描（§(b)）；预期远省。
4. **认证**：开 primal 重解算 $\eta_\tau$、$\Theta$，报真实裂纹上的保证型有效性（路线 (a)）。
5. **标记可靠性（命题 2）**：对比用 $\sigma_h$ vs 用 $\mathbb C_d\varepsilon(u_h)$ 算 $\mathcal D$ 的标记集合差异，
   坐实 Hu–Zhang 应力在裂尖标记更干净。

---

## 参考文献

- T. Heister, M. F. Wheeler, T. Wick. *A primal-dual active set method and predictor–corrector
  mesh adaptivity for computing fracture propagation using a phase-field approach.* CMAME 290 (2015) 466–495.
  [link](https://www.sciencedirect.com/science/article/abs/pii/S0045782515001115)
- *An adaptive mesh refinement algorithm for phase-field fracture models: brittle, cohesive, and
  dynamic fracture*（effective crack driving energy mark–unmark）. CMAME (2022).
  [link](https://www.sciencedirect.com/science/article/abs/pii/S0045782522004339)
- *Adaptive FEM for phase field fracture based on recovery error estimates*（$\eta=\|R_h d_h-\nabla d_h\|$，
  用户既用方法）. J. Comput. Appl. Math. (2025); [arXiv:2410.01177](https://arxiv.org/pdf/2410.01177).
- *An adaptive mesh refinement algorithm for stress-based phase field fracture models for
  heterogeneous media*（crack driving force as DG0）. Comput. Struct./FEniCS.
  [link](https://www.sciencedirect.com/science/article/pii/S0168874X24002051)
- K. Mang, M. Walloth, T. Wick, W. Wollner. *Mesh adaptivity for quasi-static phase-field fractures
  based on a residual-type a posteriori error estimator.* GAMM-Mitt. 43(1) (2020) e202000003;
  [arXiv:1906.04657](https://arxiv.org/abs/1906.04657).
- *Adaptive numerical simulation of a phase-field fracture model in mixed form*（residual estimator
  robust in $\varepsilon$, L-shape）. [arXiv:2003.09459](https://arxiv.org/pdf/2003.09459).
- *A posteriori estimator for adaptive solution of quasi-static fracture phase-field with
  irreversibility constraints.* [arXiv:2106.09469](https://arxiv.org/pdf/2106.09469).
- 平衡应力重构（保证型）背景：Braess–Schöberl equilibrated flux；对称 $H(\operatorname{div})$
  Arnold–Winther；goal-oriented locally equilibrated stress recovery [arXiv:1308.1941](https://arxiv.org/pdf/1308.1941).
- $\Gamma$-收敛 / $h$ vs $l_0$：Ambrosio–Tortorelli；Bourdin–Francfort–Marigo (2008);
  Miehe et al. (2010); Gerasimov–De Lorenzis（网格分辨 $l_0$ 与峰值载荷）。
