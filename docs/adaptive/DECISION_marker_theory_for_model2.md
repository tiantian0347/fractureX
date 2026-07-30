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

- [x] 主文 §4 marker 章节改写为 $\eta_T$ 版；$\mathcal D_{\tau,T}$ 降级为 remark（§sec:sigma-marker + Remark rem:marker-variant）。
- [x] Remark 5.6 升级为正式引理（Lemma lem:local-eff）+ 证明。
- [x] `equilibrated_aposteriori.tex` §4 title 改为 "Equilibrated-marker adaptivity"；Algorithm 1 使用 $\eta_T$ + CKNS 相对下降停机；参数表加 $\theta_{\max},q,d_{\mathrm{hi}}$。
- [x] `refs.bib` 加 Dörfler 1996, Cascón--Kreuzer--Nochetto--Siebert 2008, Feischl--Führer--Praetorius 2014, Ainsworth--Oden 2000, Verfürth 2013, Zienkiewicz--Zhu 1992, Tian--Chen--He--Wei 2024。
- [x] SENS 数值验证（2026-07-06 完成）：跑完 40 步，peak $\lvert R_x\rvert=0.150$，NC $1152\to 1587$；见 §11。
- [x] `equilibrated_aposteriori.tex` §5.4 SENS 新章节（2026-07-06 完成）：正文 + 3 张图（`paper_model2_marker_compare.png`, `paper_model2_Dmax_evolution.png`, `paper_model2_NC_growth.png`）+ Abstract/P1/Conclusion 同步更新。绘图脚本 `plot_paper_model2_Fu.py`。
- [ ] SENS mesh evolution 4-panel 图（模仿 `model0_evolution_4panel.png`）——挂起项。

## 9. SENT 数值验证（2026-07-05）

配置：`FRACTUREX_MARKER=eta_T, ETA_T_STRATEGY=max, THETA_REC=0.9, ETA_DECREMENT=0.7, du=2.5e-4, nx=24, spsolve`。

| 指标 | $\eta_T$ marker (本次) | $\mathcal D_{\tau,T}$ v3 canonical | 参考 nx=120 |
|---|---:|---:|---:|
| Peak $\lvert R_y\rvert$ | **0.6206** | 0.6211 | 0.631 |
| 起 peak 位移 $u_y$ | 5.00e-3 | 5.25e-3 | 5.10e-3 |
| 相对参考偏差 | **−1.6%** | −1.5% | ref |
| Peak 处 NC | 1716 | ~1500 | 14400 |
| Peak 处 $\mathrm{dof}_\sigma$ | 28667 | 31406 | 476883 |
| 全跑 wall (25 步到 stop) | 378 s | 小时级（Anderson canonical run） | — |
| 每步 corr 平均 | ~1 | 3–8 | — |

**结论**：$\eta_T$-marker + CKNS 相对下降停机在 SENT 上 peak accuracy 与 $\mathcal D_{\tau,T}$ 持平，
DOF 略少，wall 快，每步只 1 轮 corrector（stopping 生效）。**理论升级不掉性能**。

SENT 实测让我们（1）确认了 max 准则 + $\theta_{\max}=0.9$ 是正确的候选：$\theta_{\max}=0.5$ 在
弹性阶段一步标 70%+ 胞，$\theta_{\max}=0.9$ 只标 top 1-2%；（2）确认了 CKNS 相对下降 $q=0.7$ 停机
能让 corrector 自然收敛，替代 "$\mathcal D_\tau\ge\theta_D$" 的阈值型自然停机。

## 10. SENS 数值验证（2026-07-06，nx=24——**已被 §10.1 nx=48 取代**）

> ⚠ **2026-07-10 复盘：本节 run 欠分辨**。nx=24 + c_h=2 + d_hi=0.995 三因素叠加：
> h_min=0.0074 > ℓ0/2=0.0066 永远够不到；d_hi=0.995 把 seed 邻胞过滤掉 59/100；
> marker 近乎失活。峰值 0.150 未收敛（nx=48 得 0.196，高 30%），损伤带弥散、
> 无清晰斜向下裂纹。**"消除 g⁻² 病态"的定性结论仍成立**；定量数字以 §10.1 为准。

配置：`FRACTUREX_MARKER=eta_T, ETA_T_STRATEGY=max, THETA_REC=0.9, ETA_DECREMENT=0.7, D_HI=0.995, du=2.5e-4, nx=24, pardiso`（lab 服务器 `~/tian/fracturex/results/adaptive_m3_pc_model2_eta_T/`；40 步 wall 81326.5s $\approx$ 22.6h）。

| 指标 | $\eta_T$ marker (本次) | $\mathcal D_{\tau,T}$ marker (effstress buggy 复现) |
|---|---:|---:|
| 完成步数 | **40 / 40** | 33（step 33 求解器崩溃）|
| Peak $\lvert R_x\rvert$ | **0.150** at $u_x=7.5\times 10^{-3}$ | 0.234 at $u_x=6.0\times 10^{-3}$（噪声污染的假峰）|
| $\mathcal D_{\max}$ 范围 | $\mathcal O(10^0$–$10^4)$，3 个孤立尖峰 $10^6$–$10^{13}$（观察量，不驱动 marker）| $\mathcal O(10^0)\to 2.5\times 10^{51}$ |
| NC 增长 | 1152 → 1587（+38%）| 1152 → 3292（+186%，噪声胞加密）|
| Corrector 每步 | 1 轮（相对下降停机生效）| 1–2 轮 |
| 软化段完成 | **是**（$u_x=7.5\times 10^{-3}$ 后进入下降段）| 否 |

**核心结论**：
1. **$\eta_T$ marker 消除了 $g^{-2}$ 病态**——不再有 $\mathcal D_{\max}$ 爆炸导致的加密失控；
2. **旧 $\mathcal D_{\tau,T}$ 的 0.234 峰是假高峰**：出现在 step 24（$u_x=6.0\times 10^{-3}$），但从 $\mathcal D_{\max}$ 演化图看，step 22 起 $\mathcal D$ 已进入 $10^3$ 量级（噪声开始污染），peak 后的 R 陡降是求解器崩溃的前兆，而非真物理软化；
3. **$\eta_T$ marker 得到的 0.150 峰 + 单调进入软化段**是物理合理的 Mode-II 承载路径；
4. **诚实边界**：本次没有 $n_x=120$ 的 Mode-II 参照（SENT 有），因此本 benchmark 无法做 SENT 式的 $\Theta$ 认证；claim 是"定性但决定性"——同 predictor-corrector 循环下，替换 $\mathcal D_{\tau,T}\to\eta_T$ 消除了 divergence pathology。

**tex 集成（`equilibrated_aposteriori.tex`）**：
- §5.4（`sec:num-model2`）完整重写：4 段正文 + 3 张图。
- Abstract、Contribution list (P1)、Conclusion 同步更新："deferred to a follow-up" → "eliminates divergence and completes the softening branch"。
- 4 张 SENS 图入 `Tian/thesis/fracture_huzhang/adaptive/figures/`：`paper_model2_Fu_main.png`（未用）、`paper_model2_marker_compare.png`、`paper_model2_Dmax_evolution.png`、`paper_model2_NC_growth.png`。
- 绘图脚本：`fracturex/tests/aposteriori/plot_paper_model2_Fu.py`。
- Build: 24 页, BUILD_OK, 0 undefined refs/cites/overfull。
- **2026-07-07/09 用户复审后回撤**：§5.4 改为 diagnostic 定位（只报 $\mathcal D_{\tau,T}$ 病态 + Dmax 图），
  完整 Mode-II 定量研究 "will be reported separately"——因 nx=24 数据欠分辨（见 §10.1）。

## 10.1 SENS 数值验证 v2（2026-07-12，nx=48 分辨达标——**当前有效数据**）

触发：用户指出 nx=24 结果"网格明显不够密"、"视觉上不是斜向下的裂纹"。修正配置：
`nx=48, C_H=4.0（h≤ℓ0/4）, D_HI=0.999, THETA_REC=0.9, ETA_DECREMENT=0.7, du=2.5e-4, pardiso`
（lab `results/adaptive_m3_pc_model2_eta_T_nx48/`；40 步 wall 319205.6s ≈ 88.7h）。

| 指标 | nx=48（本次） | nx=24（§10，欠分辨） |
|---|---:|---:|
| Peak $\lvert R_x\rvert$ | **0.1957** at $u_x=7.0\times 10^{-3}$ | 0.150 at $7.5\times 10^{-3}$（低 23%）|
| 裂纹路径 | **斜向下**：seed 尖端 (0.5,0.5) → (0.65,0.30)，符合 Miehe Mode-II 形态 | 弥散带，无清晰路径 |
| 裂缝带网格 | median h=0.0052 ≤ ℓ0/2；min h=0.0037 ≈ ℓ0/4 | h_min=0.0074 > ℓ0/2 |
| NC 增长 | 4608 → 5567（+21%）| 1152 → 1587（+38%）|
| $\mathcal D_{\max}$ | ≤ 3.3×10⁴ 有界、无爆炸 | 孤立尖峰至 10¹³ |
| Corrector 每步 | 1 轮（CKNS 停机生效）| 1 轮 |
| 软化段 | **仅早期**：峰后 R 在 0.177–0.195 平台震荡，40 步时裂纹未贯穿（y 至 0.30）| 假"软化"（欠分辨伪像）|

**结论**：g⁻² 病态消除的定性结论在分辨达标网格上继续成立；峰值 0.196 + 斜向下路径物理合理。
**软化段未完**：40 步只到裂纹半程。已启动**断点续跑**至 60 步
（`results/adaptive_m3_pc_model2_eta_T_nx48_cont/`，restart 机制见下）。

**断点续跑机制（2026-07-12 新增）**：`run_m3_pc_model1.py` 支持
`FRACTUREX_RESTART_NPZ/RESTART_STEP/PEAK_R0`；npz 由 `vtu_to_restart_npz.py` 从 vtu 转换
（node/cell/d）。原理：与 in-run 加密后流程同构——恢复 (mesh, d)，H=None 由 solve 重建；
r_hist 相场路径不读、置 0。smoke 验证（model2 nx=8 4 步，step2 重启）：R/NC/dofσ/iters
与整跑逐位一致。续跑首步重解 step 39 作连续性检查（应复现 R≈0.1914）。

## 12. 复审裁定（2026-07-10）：η_T-only marker 在 SENS 上**失效**——事后型钉扎

> 触发：用户复审 "eta_T 的结果看起来不太对，DT 的结果看起来反而更对一些"。
> 对照 **paper_direct_full 参照**（`results/phasefield/model2_notch_x_stretch/paper_direct_full/epsg_1e-06/reaction_curve.csv`，
> nx=160 均匀 NC=51200, p=3, du=1e-4, 200 步，用户已验证）后确认：**用户判断正确**。

### 12.1 参照曲线裁定（去重 restart 重叠段后）

参照真峰值 $|R_x|=0.421$ at $u_x=1.033\times 10^{-2}$，之后陡降至 ~0.23 平台。对比：

| $u_x$ | ref (nx=160) | $\mathcal D_{\tau,T}$ nx=24 | $\eta_T$ nx=48 |
|---:|---:|---:|---:|
| 4e-3 | 0.176 | 0.159 (−10%) | 0.125 (**−29%**) |
| 6e-3 | 0.259 | 0.234 (−10%) | 0.178 (−31%) |
| 7e-3 | 0.300 | （崩溃前兆） | 0.196 (−35%，§10.1 的"峰"是伪像) |
| 9.75e-3 | 0.404 | — | 0.191 (**−53%**) |

- $\mathcal D_{\tau,T}$ 整个上升段贴参照 −10%（此段它尚未加密，即均匀 nx=24），直到 g⁻² 崩溃；
  其 0.234 也非真峰，是崩溃起点。
- $\eta_T$ nx=48 从 $u\approx 10^{-3}$ 起系统性偏软 ~30%；§10.1 的 0.196 "峰" 与后续"软化平台"
  **整段是钉扎伪像**——参照在该处仍在近直线上升，真峰在 1.033e-2。§10.1 的定量结论全部作废，
  仅"无 $\mathcal D_{\max}$ 爆炸"仍成立。

### 12.2 根因：η_T 是事后型（reactive），裂尖前方饥饿

$\eta_T$ 度量**当前**损伤组态下的弹性误差（集中在已损伤团），max 准则 θ=0.9 + CKNS 每步 1 轮
把标记预算全耗在团上。实测（nx=48 run @ u=7.5e-3）：裂尖前方扇区 **0% 单元达到 h≤ℓ0/2**
（全部 h=0.0208≈1.4ℓ0）；对照 DT run 前方 h=0.0074≈ℓ0/2。前锋在欠分辨网格上弥散、钉扎、
人为增韧，损伤团向上分叉（与 Miehe/Ambati 文献的斜向下清晰路径不符）。
这不是分辨率或调参问题，是 marker 的**结构性缺陷**：认证型指示子 ≠ 预测型标记器。

### 12.3 g⁻² 病态的正确理解（修正 §2 表格的一处误导）

`state.H` 一直由 `from_u` 计算（`update_history_on_quadrature`: ψ⁺(ε(u_h))，无显式 g⁻² 除法），
但裂纹带内 ε~g⁻¹σ ⇒ ψ⁺(ε)~g⁻²·½σ:ℂ⁻¹σ——**g⁻² 隐含在应变里，from_u 与 effstress 公式等价**。
"换 H 来源"不解决病态；唯一结构性隔离是**限制 D 标记于低损伤区**：cell max d ≤ d_cap=0.5
⇒ g≥0.25，ψ⁺ 放大上界 16×，爆炸结构性不可能（d_cap=0.9 时 g=10⁻² ⇒ 放大 10⁴，Mode-II 过渡带失守）。

### 12.4 修复决策（用户已确认）：marker := η_T ∪ D-低损伤预测（`eta_T_df`）

$$\mathcal M = \underbrace{\{\eta_T^2 \ge \theta_{\max}\max\eta_T^2,\ \min_v d\le d_{\mathrm{hi}}\}}_{\text{认证型，CKNS 停机}}\ \cup\ \underbrace{\{\mathcal D_\tau\ge\beta/3,\ \max_v d\le d_{\mathrm{cap}}{=}0.5\}}_{\text{预测型，自限（h≤ℓ0/c_h 停）}}\quad(\text{均含 } h_\tau>l_0/c_h)$$

- 实现：`adaptive_staggered.mark_df_lowdamage` + runner `FRACTUREX_MARKER=eta_T_df`，`FRACTUREX_D_CAP`(0.5)。
- D-子掩码不受 CKNS 相对下降停机约束（阈值型 + 尺寸下限自限）；η_T 子掩码沿用 §4.1。
- 论文叙事：η_T 保留认证/停机角色（Cor.5.3 + Lemma 局部下界不动）；预测子掩码作为
  "damage-front safeguard" 进 §4，坦白 η_T 单独作 marker 在移动裂纹前锋上的事后性局限。
- 处置：lab 续跑（步 39→59，延续错误轨迹）已终止；§10.1 数据降级为反例。
- 验证：SENS smoke → SENT 40 步（对 §9 不退化）→ SENS nx=48 重跑对 paper_direct_full 参照
  （目标：上升段贴参照 ≤10%，真峰 ~0.42@1.03e-2 量级复现，斜向下裂纹路径）。

## 12.5 复审裁定（2026-07-14）：`eta_T_df` 对参照**同样失效**——+D 低损伤子掩码不足以喂饱前锋

> 触发：执行 §12.4 验证计划的 SENS nx=48 跑（`results/adaptive_m3_pc_model2_eta_T_df_nx48`，
> marker=eta_t_df, du=2.5e-4, pardiso, 33 步）。**跑完不发散**（内部一致）——但对 paper_direct_full
> 参照比对**从 u≈3e-3 起崩到 −40%、u=8e-3 时 −77%**，且 R 在 u=6e-3 见顶 0.131 后**回落**
> （假软化），NC 仅 4608→5798（**+26%**，未跟踪前锋）。与 §12.2 η_T-only 的前锋饥饿是**同一病**。

| $u_x$ | R (eta_t_df) | R (ref nx=160) | dev |
|---:|---:|---:|---:|
| 1e-3 | 0.0425 | 0.0445 | −4% |
| 3e-3 | 0.0811 | 0.1327 | **−39%** |
| 6e-3 | 0.131 | 0.259 | **−49%**（此后 R 回落=假软化）|
| 8e-3 | 0.077 | 0.349 | **−77%** |

**裁定**：§12.4 的 `eta_T_df` 决策**作废**。"D 低损伤预测子掩码"（d_cap=0.5, 𝒟≥β/3）在
Mode-II 前锋**触发太晚/太少**：裂尖前方 d 仍低、σ 集中带尚未把 𝒟 顶过 β/3，等 𝒟 够时前锋已过、
d 已越 d_cap 被过滤——两头够不着，NC 只 +26%。**η_T ∪ D-lowdamage 二者都缺前锋预视**。

**根因再确认（§12.2 结论加强）**：SENS 需要的是**移动前锋预加密**（process-zone lookahead），
而 η_T（事后型，度量当前损伤组态弹性误差）与 D-lowdamage（阈值型，等 σ 集中）都**不预视前锋**。
唯一天然预视的是 **recovery 型 η_τ^d = ‖R_h∇d − ∇d‖**：∇d 在推进的裂尖前方大 ⇒ 提前加密
（tian2024 已验证）。其唯一缺陷是在**静态 seed**（固定 d=1→0 跳变，recovery 误差不随加密下降）上
反复触发 Dörfler（nx=8 本地实测：corr 每轮 marked 单调涨 221→340→423，max_d≡1、𝒟max≈0，
纯 seed 抖动，永不收敛）。seed 几何预加密不解决——只把抖动移到粗细过渡环上。

### 12.6 新修复决策（2026-07-14）：recovery marker + **静态 seed 排除掩码**

$$\mathcal M = \{\,\eta_{\tau}^{d,2} \ge \theta\max_\tau \eta_\tau^{d,2}\ \text{(Dörfler-L²)},\ \ h_\tau>l_0/c_h,\ \ \min_v d\le d_{\mathrm{hi}},\ \ \tau\notin\mathcal S_{\mathrm{seed}}\,\}$$

其中 $\mathcal S_{\mathrm{seed}}$ = 跨越已知 seed 线（$y=\text{crack\_y},\ x\in[0,\text{crack\_length}]$）
的单元集（几何判据，与解无关），由 seed 几何预加密一次性解析、此后永久排除出 recovery 标记集。
- 动机：recovery 是唯一有前锋预视的 driver（§12.5）；其唯一病是静态 seed 抖动；seed 是**已知几何**，
  用几何掩码结构性剔除，比任何 d 阈值过滤都干净（seed 边胞 min_d=0 躲过 d_hi 过滤，正是抖动源）。
- 实现：`adaptive_staggered.mark_recovery` 加 `seed_exclude`(NC bool) 参数；runner 从
  case.crack_y/crack_length 构造 straddle 掩码传入；配合 `FRACTUREX_SEED_PREREFINE=1` 先解析 seed 带。
- 验证：nx=8 本地 smoke（秒级解，验 corrector 收敛、不抖 seed）→ nx=48 lab 对参照
  （目标：上升段 ≤10%、真峰 ~0.42@1.03e-2、NC 显著增长跟踪斜向下前锋）。

### 12.7 seed 排除掩码**证伪**（2026-07-14）——抖动源不是 seed，是网格梯度过渡环

nx=8 本地实测 + 直接 plumbing 测试推翻 §12.6 假设：

- **plumbing（合成均匀 η_τ^d，theta=0.5）**：`mark_recovery` 加/不加 seed_exclude 标记数
  **完全相同**（589 vs 589），尽管 seed 掩码覆盖 3144/4324（73%）胞。⇒ 被标记的胞**根本不在
  seed 带内**（seed 带胞已被 size-floor `cm>area_floor` 过滤，因预加密后它们最小）。
- **corrector 抖动实测（recovery，含/不含 seed_exclude 逐位一致）**：step1 每轮 marked
  221→340→…单调涨，全程 max_d≡1.000、𝒟max≈0（无物理演化），NC 4324→4660→5282…不收敛。

**真根因（probe 实测修正，2026-07-14）**：被标记的 221 胞**不是** d≡0 远场过渡环，而是
**静态 seed 的 AT2 损伤 halo**——max_d 分布 0.10–0.34、min_d 0.008–0.12，163/221 落在
|y−0.5|<2·l0 内、仅 2 个在裂尖前方(x>0.5)。u=1e-4 时**尚无移动前锋**，recovery 信号几乎全是
静态 seed 周围 d 从 1 衰减到 0 的 halo（宽 ~2·l0）。seed 预加密（halfwidth 0.75·l0）只解析了
d≳0.5 的核，halo 尾巴(d=0.01–0.34)留在细带外的粗胞上 ⇒ recovery 正确地要加密它们 ⇒
细/粗界面沿 halo 外爬 ⇒ "抖动"。**d_lo 阈值无法分离**：抖动胞(d=0.01–0.34)与真实低损伤前锋
占同一 d 区间，d_lo=0.2 只把 221→159；d_lo=0.5 全清空。加宽预加密带到 2.5·l0（NC 4324→12584）
也只把 marked 221→170——halo 更宽、过渡带更多，治标不治本。

**nx=8 是病态 proxy**：base 胞 0.125，解析 l0=0.015 需 ~11 级 bisect ⇒ 极端 graded mesh、
巨大过渡区、halo 严重欠分辨。nx=48（base 胞 0.021 ≈ 1.4 l0）只需 2–3 级达 l0/4，grading 温和、
静态 halo 近乎被 base+轻预加密解析 ⇒ recovery 在其上误差小、会聚焦于真正欠分辨的**演化前锋**。
**⇒ 停止在 nx=8 调参（最坏情形），process-zone recovery 的真实检验必须在 nx=48 上跑。**
另注：max_corr=8 封顶 ⇒ 抖动不会真挂死（eta_t_df nx=48 跑完 33 步为证），是**质量**问题
（标了噪声胞）非 hang。nx=48 温和 grading 下过渡环抖动可能小到 max_corr 内给出合理网格。

**当前裁定**：已连续证伪 4 个 marker——η_T-only（前锋饥饿 −53%）、eta_t_df（前锋饥饿 −77%）、
recovery-alone（过渡环抖动，不收敛）、recovery+seed排除（同抖动）。**问题分两类**：
(A) 认证/阈值型（η_T, D）→ 前锋预视不足、太软；(B) recovery 型 → graded-mesh 过渡环抖动。
下一步**不再试第 5 个 marker 变体**，需与用户确认方向（见会话）。候选：
(a) recovery 限定于**过程区**（0<max_d<1 的演化带，排除 d≡0 远场与 d≡1 断裂区）——但会牺牲
   d≡0 的前锋预视，需权衡；
(b) 回到 M-DF 应力驱动（model1 SENT 已验证有前锋预视），用 from_u 的 H（§12.3 已证与
   effstress 等价、g⁻² 隐含但 d_cap 可控），plain beta 触发；
(c) 直接放弃 SENS 的严格参照复现，改用中等**全局**加密（nx≥96）作 anchor（贵但无 marker 病）。

### §12.8 process-zone recovery（d_lo 下界）在 nx=48 证伪（第 5 个 marker 作废）

> 触发：用户经 AskUserQuestion 选定"recovery 限定过程区"方向后（§12.7 (a)），
> 在 lab 跑 `results/rec_pz_nx48_nopre`（marker=recovery, nx=48, du=2.5e-4, c_h=4,
> theta_rec=0.5, **d_lo=0.05**, 无 seed 预加密, pardiso）。

**结果：FAIL（不收敛，与 nx=8 同病）。** 决定性信号出现在 **step 1**（load=2.5e-4,
𝒟max=0.04 —— 纯弹性、裂纹尚未起裂）：
- marked 逐 corrector **上升**：corr0=41 → corr1=52（应 →0）；
- 同一 NC（4608→4690）re-solve 代价爆炸 **226s → 844s**，staggered iters 2 → 12。

**根因（nx=8→nx=48 一致，最终确认）**：整个弹性上升段没有移动前锋可追。域内唯一非零 ∇d
是**静态 seed 的 AT2 损伤晕**（d 沿 ~2·l0 从 1 衰减到 0，环绕预裂纹）。该晕的 d 值落在
[0.05, 1]，故 `d_lo` **排不掉它**——按此判据晕本身就是"过程区"。Dörfler bulk 每轮固定抓
误差质量的一个比例 ⇒ 反复重标同一处欠解析 seed 晕；加密缓慢减小该误差 ⇒ 抖动，
每 corrector 烧 ~800s 在非裂纹区。

**recovery 作 driver 的结构性缺陷**：它无法区分**静态** seed 晕（由 l0 固定、只需解析一次）
与**移动**前锋（需持续追踪）。任何 recovery 变体在弹性段都会去追 seed 晕。唯一能让 recovery
可用的架构：**预先把 seed 晕解析一次**（seed 预加密），使 recovery 在载荷步中于此处看到零误差、
只标真正的移动前锋。§12.6 曾因 hw=1.5·l0 预加密把 NC 吹到 8924（662s/solve）搁置，但更紧的带
（hw=0.75·l0、仅到 l0/c_h）可能以可接受 NC 解析该晕——待验。

**当前裁定（§12.7 加强）**：已连续证伪 **5 个** marker——η_T-only、eta_t_df、recovery-alone、
recovery+seed排除、**process-zone recovery(d_lo)**。前四类根因见 §12.7；第 5 个证明 d_lo 判据
无法把静态 seed 晕排出"过程区"。不再盲试第 6 个变体；候选收敛为两条：
(a′) **seed 紧带预加密 + recovery**（把静态晕解析掉，让 recovery 只见移动前锋）——recovery
    唯一可行架构，成本待验（hw=0.75·l0 是否 << §12.6 的 NC 8924）；
(c) 放弃严格参照复现，改用中等**全局**加密（nx≥96）作 anchor（贵但无 marker 病）。

## 13. Bibliography（进入 `equilibrated_aposteriori.tex`）

标准 AFEM 收敛与停机理论的引用一律通过 [refs.bib](../../Tian/thesis/fracture_huzhang/adaptive/refs.bib) 提供：

- `Doerfler1996`: Dörfler 收缩型标记 [Dörfler 1996, SIAM JNA 33(3)].
- `CasconKreuzerNochettoSiebert2008`: 拟最优 AFEM 收敛率 [SIAM JNA 46(5)].
- `FeischlFuehrerPraetorius2014`: 一般非对称/非线性问题的 AFEM 最优收敛 [SIAM JNA 52(2)].
- `AinsworthOden2000`: A Posteriori Error Estimation in FEA (book).
- `Verfuerth2013`: A Posteriori Error Estimation Techniques for FEM (book).
- `ZienkiewiczZhu1992a`: 超收敛 patch recovery [IJNME 33(7)].
- `TianChenHeWei2024`: 基于 recovery 型后验估计的相场自适应有限元 [JCAM 472].
