# 开放问题：contrast-uniform 有效性定理（Hu–Zhang 平衡应力）

> 状态：想法记录 v0.1（2026-07-13）。来源：与导师讨论——「这是价值最高的一条」。
> 关联：[THEORY_equilibrated_aposteriori.md](THEORY_equilibrated_aposteriori.md) §5–§7；
> `Tian/thesis/fracture_huzhang/adaptive/equilibrated_aposteriori.tex` 的 Prop（asymptotic effectivity, `eq:theta`）。
> 记号沿用 tex：$E:=\|\eps(u_h^{\mathrm c}-u)\|_{\Cd}$（位移能量误差，加权），
> $S:=\|\sigma_h-\sigma\|_{\Ad}$（互补应力误差，加权），$\Theta=\sqrt{1+(S/E)^2}$，
> $g(d)=(1-\kres)(1-d)^2+\kres$，$\Cd=g(d)\CC$，$\kres$ 残余刚度地板。

---

## 1. 现状：只能说 observed

论文已有的严格结论（Prop asymptotic effectivity）是**恒等式**层面的：
$$\Theta^2 = 1 + (S/E)^2,$$
配套的判据是「$\Theta\to1 \iff S/E\to0$」，以及充分条件「$S\le c\,E$ uniformly $\Rightarrow \Theta\le\sqrt{1+c^2}$」。

但 **$S\le C E$ 本身没有证明**。数值上（扫 $\kres$ 五个数量级、$d$ 直到 1）观察到
比值 $S/E$ 只以 $\approx\kres^{0.26}$ 温和增长、有效性稳在 $1.011$ 内——这只是
**observed / empirical evidence**，不是 theorem。论文目前只能诚实标注为
「contrast-robust efficiency 的经验证据」。

---

## 2. 目标：把 observed 升级成 theorem

### 2.1 一级目标（contrast-uniform bound）
证明存在常数 $C$，**与 $\kres$ 无关**，使得
$$S \le C\,E.$$
立即得到
$$\Theta \le \sqrt{1+C^2},$$
即有效性指数被一个**与退化对比度无关**的常数一致封顶。这一步就足以把
「observed」改写成正文定理，论文层次立刻上一个台阶（Numerische 口味）。

### 2.2 二级目标（asymptotic exactness with rate）
若进一步能证
$$S = O(h\,E),$$
则由 $\Theta=\sqrt{1+(S/E)^2}$ 与 $\Theta-1\le\tfrac12(S/E)^2$ 直接得到
$$\Theta = 1 + O(h^2).$$
这是 Numerische 非常喜欢的那种 asymptotic-exactness 定理。**导师判断这一条更难**。

---

## 3. 为什么难

- $S=\|\sigma_h-\sigma\|_{\Ad}$ 是 Hu–Zhang 应力在**加权（退化）compliance 范数** $\Ad$
  下的误差。**据我们所知，Hu–Zhang stress 的 weighted-norm 误差估计目前没人证明过。**
- 朴素路线（把 $g(d)$ 换成全局下确界 $\kres$）会把常数做成 $\sim\kres^{-1/2}$ 甚至
  $\kres^{-1}$，在 $\kres\to0$ 爆掉——这正是 THEORY §5 第 2 层「全局朴素界会爆」的病根。
- 关键观察（THEORY §5 第 3 层、tex 引言）：真正决定有效性的不是**全局**对比度，而是
  **局部 vertex-patch 对比度**；数值显示局部对比度与全局对比度**同步**以 $\kres^{-1}$ 增长，
  却没拖垮 $\Theta$。要害是把 $S/E$ 的界建立在**局部 $g$ 的光滑性 / patch 内 $g$ 的振荡**上，
  让 $\kres$ 因子在比值 $S/E$ 里相消，而不是各自发散。

导师估计：**这一条至少值一篇独立论文**。本仓库先记录，等有时间再推。

---

## 4. 可能的推导入口（备忘，未验证）

1. **加权 Prager–Synge / hypercircle**：$S,E$ 已由 `eq:theta` 的恒等式耦合；核心是找一个
   **与 $\kres$ 无关**的应力逼近误差界 $S\le C\,\inf_{\tau\in\Sigma_f}\|\tau-\sigma\|_{\Ad}$
   （Hu–Zhang 的最佳逼近性质在加权范数下的版本）。
2. **局部对比度而非全局**：把 $\Ad$-范数拆成 patch 局部，用 $g|_{\omega_z}$ 的振荡
   $\mathrm{osc}_z(g)=\sup_{\omega_z}g/\inf_{\omega_z}g$ 替代全局 $1/\kres$；证明相邻 patch 间
   $g$ 变化受网格分辨的裂纹层厚度 $\ell$ 控制 ⇒ $h\lesssim\ell$ 时局部对比度 $O(1)$。
3. **$S=O(hE)$ 路线**：需要 Hu–Zhang 应力的加权 $H(\mathrm{div})$ 逼近阶 + 位移误差的
   加权 Aubin–Nitsche 对偶，两者的 $\kres$ 因子在比值里抵消。这是二级目标，风险最高。

---

## 5. 交付形态

- 一级目标成功 ⇒ tex 里把 Prop asymptotic effectivity 后的「$S\le c E$ 是充分条件」升级为
  **带证明的 contrast-uniform 定理**，删掉「observed」措辞。
- 二级目标成功 ⇒ 单独一节 asymptotic-exactness $\Theta=1+O(h^2)$，很可能拆成独立论文。
- 在此之前：正文继续以 §2 的数值（$\kres^{0.26}$、$\Theta\le1.011$）作为经验证据，
  措辞保持诚实（「empirical / observed」）。
