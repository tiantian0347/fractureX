# 决策：面向 model2（SENS）的 marker 理论升级 —— 从 heuristic $\mathcal D_{\tau,T}$ 到 Prager–Synge $\eta_T$

> 状态：决策稿 v0.1（2026-07-05）。触发：`results/adaptive_m3_pc_model2_effstress`（SENS, 34 步 100 h wall）
> 在 step 25 峰值 $|R_x|=0.234$ 之后触发 M-DF 加密，$\mathcal D_{\max}$ 由 0.64 跳到 $10^8$-$10^{51}$，
> step 33 求解器崩溃。根因是 [THEORY_marking_strategy](THEORY_marking_strategy.md) 里的 M-DF 标记量
> $\mathcal D_{\tau,T}$ 在 seed 预裂缝 $d\equiv 1$ 上有 $g^{-2}\sim k_{\mathrm{res}}^{-2}=10^{12}$ 的
> 权重放大 $\sigma_h$ 数值噪声；$d_{\mathrm{cut}}$ 过滤在 Mode-II（$x$-stretch 剪切）加载下不足以隔离。
> 用户要求：**升级方案必须有理论支撑**（$\mathcal D_{\tau,T}$ 是 heuristic，`equilibrated_aposteriori.tex`
> Remark "certified efficiency of the accepted marker" 已诚实标注 "not claimed to satisfy a local
> efficiency inequality"），且必须**解决 model2 的病态**（不是绕开）。
>
> 本文在 [THEORY_equilibrated_aposteriori.md](THEORY_equilibrated_aposteriori.md)（$\eta_T$ 理论）与
> [THEORY_marking_strategy.md](THEORY_marking_strategy.md)（M-DF 预测型标记）之上，裁定 marker 层的
> 升级路径。与 [DECISION_sigma_driven_adaptivity.md](DECISION_sigma_driven_adaptivity.md) 的关系：
> 那份决策针对 model1（SENT）分工 "M-DF 主驱动 + $\eta_\tau$ 认证"；本文针对 model2 的
> $g^{-2}$ 病态，**修订该分工**，见 §5。

---

## 0. 一句话结论

$$\boxed{\ \textbf{marker} := \eta_T\ (\text{Prager–Synge, }\S3\text{ Cor.5.3 已给 reliability}=1)\ ;\quad
  \textbf{认证} := \eta_T\ (\text{同一个量、同一次评估}).\ }$$

- $\mathcal D_{\tau,T}$ 从 marker 位子上退役（在 model2 上失效、且本就是 heuristic）；保留于代码里作
  pre-damage 快速诊断，但不再进论文。
- Remark 5.6（$\eta_{\omega_z}$ 的 patch-contrast 局部下界）从 "companion paper 的待办" 提升为主文的
  **正式引理**，作为 marker 的效率论证。
- Recovery 型 $\eta_\tau^d$（tian2024）作诊断/备胎，不进主线（并入需要额外的耦合可靠性分析，见 §3）。

---

## 1. model2 病态复盘：为什么 $\mathcal D_{\tau,T}$ 在这里坏

设定：单位正方形，seed 水平预裂缝 $d\equiv 1$ on $\{y=0.5,\ x\in[0,0.5]\}$，底边固定，
顶边 $u_x=t,\ u_y=0$（$x$-stretch，Mode-II 主导）。材料 $E=210,\nu=0.3,G_c=2.7\times 10^{-3},l_0=0.0133$。
初始网格 $n_x=24$，$k_{\mathrm{res}}=10^{-6}$。运行结果 `results/adaptive_m3_pc_model2_effstress/history.csv`：

| step | load $u_x$ | $\lvert R_x\rvert$ | $\mathcal D_{\max}$ | nc |
|---:|---:|---:|---:|---:|
| 24 | 6.00e-3 | **0.2341** (峰) | 0.638 | 1152 |
| 25 | 6.25e-3 | 0.1764 | **1.685e+8** | 1494 |
| 26 | 6.50e-3 | 0.1605 | 5.31e+8 | 1874 |
| 33 | 8.25e-3 | 4.99e-13 | 2.48e+51 | 3292 → 崩溃 |

$\mathcal D_{\tau,T}$ 的定义（[THEORY_marking_strategy](THEORY_marking_strategy.md) §2 (4)）：
$$
\mathcal H_{\tau,q}=\frac{1}{2\,g(d_q)^2}\,\sigma_h(q):\mathbb C^{-1}\sigma_h(q),\qquad
\mathcal D_{\tau,T}=\frac{2 l_0}{G_c}\max_q\mathcal H_{\tau,q}. \tag{1}
$$
物理上（无噪声时）$g(d)^{-2}\sigma_h:\mathbb C^{-1}\sigma_h = g(d)^{-2}\cdot g(d)^2\,\varepsilon:\mathbb C\varepsilon=\varepsilon:\mathbb C\varepsilon$
是**未衰减弹性能**（AT2 的临界比对量）。但 $\sigma_h$ 是数值解，含 $O(\epsilon_{\mathrm{lin}})$ 的
方程残差噪声；在 $d\equiv 1$ seed 胞上 $g^{-2}=k_{\mathrm{res}}^{-2}=10^{12}$，噪声被 12 个量级放大。
`equilibrated_aposteriori.tex` §4 已写 "$d_{\mathrm{cut}}$ restricts marker to active pre-damage cells"，
但在 Mode-II 载荷下：
1. seed 相邻胞的 $d$ 从 $d_{\mathrm{cut}}=0.9$ 到 1 有过渡带；此带上 $g^{-2}$ 仍 $\sim 10^4$；
2. 加密后新生胞的插值 $d$ 落入过渡带更多；
3. Mode-II 让裂尖沿非纵向路径外推，扩大过渡带面积。

结果：$d_{\mathrm{cut}}=0.9$ 不足以隔离，$\mathcal D_{\max}$ 爆炸，随后 corrector 不断在噪声主导的
胞上加密、$H$ 场传染、$u$ 解发散。**这是 heuristic marker 的固有失败模式**，无法通过参数调优根治。

---

## 2. 候选升级方案的理论支撑对照

| | $\eta_T$（Prager–Synge, §3） | $\eta_\tau^d$（recovery on $d$, tian2024） | $\mathcal D_{\tau,T}$（M-DF, §4） |
|---|---|---|---|
| 估计对象 | 冻结损伤下**弹性子问题**误差 $\|\varepsilon(u_h^{\mathrm c}-u)\|_{\mathbb C_d}$ | 冻结位移下**相场子问题**梯度误差 $\|\nabla(d-d_h)\|_0$ | 未衰减弹性能与 $\mathcal D_c=1/3$ 的比较 |
| 全局可靠性 | 常数 $=1$（Cor.5.3，`equilibrated_aposteriori.tex` 已证） | 渐近精确，依 $\mathcal R_h$ 超收敛（Zhang 2005, Huang–Wei–Yang–Yi 2011） | **无**（heuristic，Remark 已诚实标注） |
| 局部下界 | patch contrast $\kappa_{\omega_z}$（Remark 5.6） | Zhang 2005 局部下界，需 $d\in H^{p+1}$ | **无声明** |
| $g^{-2}$ 病态 | **无**：$\int g^{-1}(\mathbb C_d\varepsilon-\sigma_h)^2$，seed 胞上 mismatch $O(g)$ ⇒ 净 $O(g)$ | **无**：只涉 $\nabla d$，与 $g$ 完全解耦；seed 内部 $\nabla d\equiv 0$ ⇒ $\eta^d\equiv 0$ | **有**：$g^{-2}\sim 10^{12}$ 放大 $\sigma_h$ 噪声 |
| 起裂前指示 | 有（弹性应力集中带上 $\eta_T$ 大） | **无**（$d\equiv 0$ ⇒ $\eta^d\equiv 0$） | 有（M-DF 的原初设计目标） |
| 额外代价 | 每次 mark 前解一次 conforming primal $u_h^{\mathrm c}$ | 一次节点平均（廉价） | 零（$\sigma_h,\mathcal H$ 已有） |

**$\eta_T$ 无 $g^{-2}$ 病态的展开**：
$$
\eta_T^2 = \int_T g^{-1}\,\big(g\,\mathbb C\varepsilon(u_h^{\mathrm c}) - \sigma_h\big):\mathbb C^{-1}\big(g\,\mathbb C\varepsilon(u_h^{\mathrm c}) - \sigma_h\big).
$$
seed 完全断裂胞上：$\sigma_h\approx g\,\mathbb C\varepsilon(u_h)$（物理应力随 $g$ 衰减），
$g\,\mathbb C\varepsilon(u_h^{\mathrm c})\approx g\,\mathbb C\varepsilon(u_h^{\mathrm c})$；两者都是 $O(g)$；
mismatch $O(g)$；$g^{-1}\cdot O(g^2)=O(g)$，天然小。**与 $\mathcal D_{\tau,T}$ 的 $g^{-2}$ 结构相反**。

---

## 3. "$\eta_T + \eta_\tau^d$" 组合是否 sound？

理论上 sound，但**不是零成本拿来主义**：需要额外的耦合可靠性引理，形如
$$
\underbrace{\|\varepsilon(u_h^{\mathrm c}-u)\|_{\mathbb C_d}^2 + c\,\|\nabla(d-d_h)\|_0^2}_{\mathcal E^2}
\ \le\ \eta_T^2 + c\,C_{\mathrm{sc}}(\eta_\tau^d)^2 + (\text{cross terms}). \tag{2}
$$
交叉项来自 $\mathbb C_d$ 的 $d$-依赖：$\eta_T$ 是"冻结 $d_h$"下的弹性误差上界，而真值 $u,d$ 用的是精确 $d$。
把 $\mathbb C_d$ 替换为 $\mathbb C_{d_h}$ 引入的偏差是 $\mathrm{Lip}(g)\|d-d_h\|_0\|\varepsilon(u)\|_\infty$，
用 Lipschitz + $\eta_\tau^d$ 吸收进第二项。**这个引理 tian2024 未做、当前论文也未做**，
需一节独立分析（约半到一页篇幅，属 companion paper 的量级）。

结论：若追求单一论文的最小 diff，用**单道 $\eta_T$**；若要发一篇 "coupled certified adaptivity"，
再上组合，那是新工作量。

---

## 4. 决策：单道 $\eta_T$

### 4.1 定义（`adaptive_staggered.mark_eta_T_indicator`）

在当前 discr 状态下，解 conforming primal $u_h^{\mathrm c}\in V_h^{\mathrm c}\cap V_g$，
用 `eta_from_state(u_override=u_h^{\mathrm c})` 逐元评估 $\eta_T^2$，**最大值准则**标记 $\theta_{\max}=0.9$：
$$
\eta_T^2 = \int_T g^{-1}(\mathbb C_d\varepsilon(u_h^{\mathrm c})-\sigma_h):\mathbb C^{-1}(\mathbb C_d\varepsilon(u_h^{\mathrm c})-\sigma_h),\qquad
\mathcal M = \big\{T:\eta_T^2(T)\ge \theta_{\max}\cdot\max_{T'}\eta_{T'}^2,\ h_T>l_0/c_h,\ \min_v d_v(T)\le d_{\mathrm{hi}}\big\}. \tag{3}
$$
- 尺寸下限沿用 M-DF（$c_h=2$）；
- 完全断裂胞过滤用 **cell min $d$**（而非 max），保留过渡带（seed 邻胞恰是 $\eta_T$ 最大处）；
- **准则与参数选择**：`FRACTUREX_ETA_T_STRATEGY=max`（默认），$\theta_{\max}=0.9$。**SENT smoke 4 步**
  实测：Dörfler L² θ=0.5 在弹性阶段一步标 76% 胞（NC 1152→17269）；max θ=0.5 也标 70%；只有 max θ=0.9
  保守收敛（4 步 NC 1152→2054，与 stress marker 相当），反力与 stress marker 差 <0.3%。
  与 tian2024 [adaptive_paper.tex L669](../../../../ttthesis/paper/adaptive_paper/adaptive_paper.tex) 同准则。
- 可选切换：`FRACTUREX_ETA_T_STRATEGY=L2` 回到 Dörfler bulk（θ≈0.1 保守）——保留仅作诊断。

### 4.2 理论主张（提升 Remark 5.6 为引理）

> **引理（$\eta_T$-marker 局部下界）**：设 $\omega_z$ 为顶点 $z$ 的一环 patch，$\kappa_{\omega_z}=\sup g/\inf g$
> patch 内有界（网格分辨 $l_0$ + patch 不横跨从完全 intact 到完全断裂的整条转变），则
> $$\eta_{\omega_z}^2 \le C(\kappa_{\omega_z})\,\big(\|\varepsilon(u_h^{\mathrm c}-u)\|_{\mathbb C_d,\omega_z}^2 + \|\sigma_h-\sigma\|_{\mathbb A_d,\omega_z}^2\big),$$
> 其中 $C(\kappa_{\omega_z})$ 依赖 patch contrast **而非全局 $k_{\mathrm{res}}^{-1}$**。

配合 Cor.5.3（reliability $=1$），$\eta_T$ 同时具备 marker 需要的**上、下界**——比 $\mathcal D_{\tau,T}$
的定位（"heuristic, no efficiency claim"）严格加强。**这是主文核心贡献的自然延伸**，不新增假设。

### 4.3 与 model2 病态的关系

$\eta_T$ 在 seed 上 $O(g)$ 天然小 ⇒ Dörfler 不会把 seed 挑进标记集；$\min d$ 过滤是廉价保险，
不是理论必需。加密不会引入 $g^{-2}$ 传染源，因此不预期 model2 的 $\mathcal D_{\max}$ 爆炸重现。

### 4.4 与 DECISION_sigma_driven_adaptivity 的差异

那份决策的两层结构（"$\mathcal D$ 主驱动 + $\eta_\tau$ 认证"）在 model1（SENT）上工作良好——peak $-1.5\%$，
$93\%$ DOF 节省。**保留** SENT 的既有结论。在 model2 里，那两层结构崩，因此**model2 走单层 $\eta_T$**。
论文叙事上：
- §3-5 主线不变：Prager–Synge + Hu–Zhang 联合的 reliability=1 上界；
- §4 marker 章节修订：主标记器换为 $\eta_T$；$\mathcal D_{\tau,T}$ 移至 Remark 作为快速 predictor 变体，
  在允许 heuristic 的场景（SENT）可用，在需要 marker 效率证明的场景（SENS）不用；
- Remark 5.6 由 "future work" 变正式引理。

---

## 5. 代价与实现

- **每步 conforming primal 重解**：`solve_primal_real` 已实现（用于 cert_every 路径）。marker 分支
  同一入口，只是从 "每 cert_every 步" 提升到 "每 corrector 内 mark 前"。
- 优化空间：（a）从上一 corrector 的 $u_h^{\mathrm c}$ 温启动 → 迭代数减半；（b）中间 corrector
  用 tol_coarse，接受态用 tol_fine（与现有 M-DF PC 循环一致）。
- 峰值成本上界：SENT 一步 $\sim$ 5 s primal → 全 40 步 200 s；SENS 一步长（100 h wall / 34 步 $\approx$ 3 h/步），
  额外 5 s primal 占比 $<0.05\%$，忽略。

---

## 6. 验证计划

1. **SENT smoke（4 步）**：验 `FRACTUREX_MARKER=eta_T` plumbing——primal 解得出、$\eta_T$ 逐元
   非零、Dörfler 挑到有意义的胞。**不比对峰值**（只 4 步）。
2. **SENT 完整（40 步）**：与既有 stress-marker 基线 `adaptive_m3_pc_model1_v3` 对比 peak $|R_y|$；
   目标 $\eta_T$ marker 达到 $\pm4\%$ 路径带内（即不比 $\mathcal D_{\tau,T}$ 差）。
3. **SENS 完整（40–60 步）**：`results/adaptive_m3_pc_model2_effstress` 的复现改成 `MARKER=eta_T`；
   验收标准：
   - $\mathcal D_{\max}$（观察量，不作为 marker）保持有限，不出现 $10^8+$ 爆炸；
   - 求解器不发散、跑完整个软化段；
   - 峰值 $|R_x|$ 与 stress-marker 版一致或更接近合理值。

---

## 7. 保留分支：$\eta_\tau^d$、hybrid 的定位

`adaptive_staggered.py` 保留 `recovery_indicator_d`, `mark_recovery`, `mark_hybrid` 三个函数：
- 用途：**代码原型 + 未来 companion paper 的基础**（组合可靠性引理若真做出来，直接可用）；
- 不进 `equilibrated_aposteriori.tex` 主文；
- 若未来发现 $\eta_T$ 每步 primal 成本不可接受，则以 recovery 作 corrector 内的**便宜 predictor**，
  接受态仍用 $\eta_T$——这时把 recovery 诚实定位为 "predictor of $\eta_T$" 而非独立估计器。

---

## 8. TODO（挂起项）

- [ ] 主文 §4 marker 章节改写为 $\eta_T$ 版；$\mathcal D_{\tau,T}$ 降级为 remark。
- [ ] Remark 5.6 升级为正式引理，含证明。
- [ ] SENT/SENS 数值验证跑通后，更新 [RESULTS_aposteriori.md](RESULTS_aposteriori.md)。
- [ ] `equilibrated_aposteriori.tex` §5.4 SENS 新章节等 SENS 结果收敛后再补图和文。
