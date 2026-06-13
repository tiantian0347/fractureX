# 规划：Hu–Zhang 平衡应力驱动的保证型 a posteriori 误差估计 + 自适应相场断裂

> 状态：草案 v0.1（2026-06-13）。本文件把 [plan_high_order_huzhang.md §4](plan_high_order_huzhang.md)
> 标记的「非光滑高次的正确出口」展开为一篇独立文章的规划。
> 理论核见 [../adaptive/THEORY_equilibrated_aposteriori.md](../adaptive/THEORY_equilibrated_aposteriori.md)。
> 触发背景：D13 学习增广线封存（见 memory `d13_learn_coarse_space`），
> 「让 solve 更快/更省」对 2D Hu–Zhang 的价值已耗尽（`aux_loses_to_pardiso_2d`）。
> 本路线换一个论点：**为什么值得付 Hu–Zhang 的代价**。

## 0. 一句话定位

Hu–Zhang 混合元天然产出一个对称、$H(\mathrm{div};\mathbb S)$-协调、逐元满足平衡的应力场
$\sigma_h$。这正是 **平衡型（equilibrated）a posteriori 误差估计** 所需、而标准位移 FEM 必须
花代价**重构**（Braess–Schöberl 局部 patch 问题）才能得到的对象。

> **中心论点**：*HuZhang equilibrated stress yields a guaranteed, constant-free,
> reconstruction-free a posteriori error estimator that certifies a standard-FEM
> displacement solve and drives adaptive mesh refinement for phase-field fracture.*

「解得快」的论点已经失败；本文论点是「**解得可信 + 加密加在对的地方**」。

## 1. 为什么这个论点站得住（与「解得快」的区别）

1. **把 Hu–Zhang 的代价直接变成独家收益。** 超圆/Prager–Synge 误差界要一个平衡应力场，
   标准 FEM 拿不到、必须重构；Hu–Zhang 免费吐出。这是「为什么用混合元」最干净的答案。
2. **治真痛点。** `surrogate_data_underresolved_hl0` 里 $h/l_0\approx 5$ 欠分辨、$\sigma$ 峰不可信，
   根因是裂纹带/裂尖 DOF 不够。a posteriori 驱动加密把这个 bug 升级成方法论卖点。
3. **与既有定位一致。** `huzhang_high_order_paper_positioning`：非光滑高次走自适应 + a posteriori。
4. **标准 FEM 是公平对照轴，也是架构的一半。** 见 §3。

## 2. 核心理论结论（细节见 THEORY 文档）

设退化系数 $g(d)=(1-d)^2+k$（残余刚度 $k>0$，**绝不允许 $g\to 0$**），退化应力
$\sigma=g(d)\,\mathbb C\,\varepsilon(u)$，柔度 $\mathbb A(d)=g(d)^{-1}\mathbb C^{-1}$。

- **Theorem 1（无能量分裂，尖锐超圆界）**：对标准 FEM 连续位移 $u_h$ 与任一平衡应力
  $\sigma^\*$（$-\operatorname{div}\sigma^\*=f$），
  $$\|\sigma-\mathbb C_d\varepsilon(u_h)\|_{\mathbb A(d)}\le
    \underbrace{\|\mathbb C_d\varepsilon(u_h)-\sigma^\*\|_{\mathbb A(d)}}_{=:\,\eta\ \text{（全可计算）}}.$$
  取 $\sigma^\*=\sigma_h$（Hu–Zhang）即得无常数上界。
- **数据振荡项**：Hu–Zhang 满足 $\operatorname{div}\sigma_h=-P_h f$，界含 $\mathrm{osc}(f)$；
  **断裂基准多为 $f=0$（位移/牵引驱动）⇒ 振荡项恒为 0**，界完全干净。这是强卖点。
- **退化的代价**：$g(d)^{-1}$ 权在裂纹内放大到 $1/k$（有界），且权重恰好堆在裂纹带——
  与「该加密的地方」对齐。effectivity index 的 $k$-依赖是技术核心，须分析。
- **Theorem 2（拉压分裂，凸对偶 majorant）**：分裂能量在 $\varepsilon$ 中凸 ⇒ 用 Repin 型
  functional a posteriori 误差 majorant 推广超圆。这是分裂情形的严格出口。

## 3. 系统架构（标准 FEM 入手）

超圆需要 **运动学容许（$H^1$-协调）位移**；Hu–Zhang 混合位移是 DG、不可直接用。故：

```
每个 staggered 弹性子步：
  (a) 标准连续 FEM        →  u_h   （H^1-协调位移，超圆的 v）
  (b) Hu–Zhang 混合/局部平衡  →  σ_h   （平衡应力，超圆的 σ*）
  (c) 超圆合成 η_T         →  逐元误差指示子（有保证 + 无常数）
  (d) Dörfler 标记 + h-加密 →  细化裂纹带/裂尖，重解
  (e) 相场子步 d 更新（沿用现有无预条件子 GMRES）
```

- 设计权衡：(a)+(b) 同步要解两个问题。对照标准做法 = 标准 FEM + Braess–Schöberl 重构
  （也要解局部问题），论证「直接拿 vs 重构」的 effectivity 与成本优势。
- 备选：从 Hu–Zhang 混合位移做 **Stenberg 后处理重构**一个协调位移，省掉 (a)。
  作为 §5 概念验证里要二选一的开放点。

## 4. Milestones（每个 M 可写一段实验章节，互相独立可中断）

### M0：理论前提概念验证（1 周，先做，决定全文成立与否）
- **固定一个已知 $d$ 场**（解析或冻结的相场快照），在退化弹性上算 effectivity index
  $\Theta=\eta/\|\text{true error}\|$，看是否 $\to 1$（无分裂）/有界（分裂）。
- 扫 $k\in\{10^{-3},10^{-5},10^{-7}\}$，量 $\Theta$ 对 $k$ 的退化曲线。**这是审稿人盯的点。**
- 交付：`docs/adaptive/poc_effectivity.md` + 一张 $\Theta$-vs-$h$、$\Theta$-vs-$k$ 图。
- **门槛**：若 split 情形 $\Theta$ 在合理 $k$ 下无界 ⇒ 退回只做无分裂版本或换方向。

### M1：标准 FEM + Hu–Zhang 双解装配（2 周）
- 复用 `fracturex/assemblers/huzhang_elastic_assembler.py` 拿 $\sigma_h$。
- 加一个标准连续 Lagrange 弹性装配（若仓库无现成，新增 `assemblers/primal_elastic_assembler.py`，
  接 fealpy 接口，**不动 fealpy**，见 `dont_modify_fealpy`）。
- 实现超圆指示子 $\eta_T$（element-wise，按 §2 公式 + $g(d)^{-1}$ 权）。
- 单元测试：光滑 MMS（$d\equiv 0$）下 $\Theta\to 1$，机器精度对账。

### M2：自适应循环（2–3 周）
- Dörfler 标记 + h-加密（fealpy 网格加密接口），与相场 staggered 主循环耦合。
- 标记量 = $\eta_T$；与损伤指示量 $d$ 的耦合策略（纯 $\eta_T$ vs $\eta_T+$ d-梯度）做一次对比。
- 断点续算 + 完整 VTU（沿用 `FRACTUREX_RESUME=1`，见 `huzhang_run_checkpoint_vtu_requirement`）。

### M3：基准与对照（2 周）
- 算例：单边缺口拉伸 / 剪切（model1/model2，$f=0$，振荡项天然为 0）。
- 对照轴：
  1. 均匀网格 vs 自适应：等精度下 DOF / 墙钟。
  2. Hu–Zhang 直接平衡应力 vs 标准 FEM + Braess–Schöberl 重构：effectivity + 重构成本。
  3. 加密后 $\sigma$ 峰、peak-load 与文献对账（治 `surrogate_data_underresolved_hl0` 的痛点）。
- 报结果同时给精度和速度（`code_goal_fast_and_accurate`）。

## 5. 待定/风险点

- **(a)+(b) 双解成本**：是否用 Stenberg 后处理省掉标准 FEM？M5 概念验证二选一。
- **split 情形 effectivity 的 $k$-依赖**：M0 门槛。
- **加密后 Hu–Zhang 重装配成本**：自适应每步重装配，需确认不抵消 DOF 节省。
- **3D 延伸**：本文先 2D 落稳；3D 是后续（也能复活 aux 内存论点，见 `aux_loses_to_pardiso_2d`）。

## 6. 目标刊物

计算数学定位（`paper_writing_style_compmath`、`paper_no_software_details`）：
*SIAM J. Sci. Comput.* / *CMAME* / *IJNME* / *Comput. Mech.*。
论点是误差估计 + 自适应的**数学保证**，正文不写软件名/库/进程机制。

## 7. 与 D12 的关系

D12（aux 预条件子）是「怎么解」；本路线是「解得对不对、DOF 加在哪」。两者正交，
可在同一 staggered 框架下共存：D12 加速每次 solve，本文决定网格。不冲突、不重叠论点。
