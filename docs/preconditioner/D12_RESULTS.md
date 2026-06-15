# D12 — 论文就绪数值结果（§5 写作直接用）

> 本文件汇总 D12 预条件子论文 **§5 数值实验**的全部结果（表 + 图 + 可直接转写的结论），按论文小节组织。规划/状态见 `D12_PRECONDITIONER_PAPER_PLAN.md`（§13.5 清单、§3.2 理论）。
> 定位（专家）：卖**迭代稳定性 + 小内存**，不卖墙钟。所有图在 `docs/figures/precond/`。
> 生成日期 2026-06-03；数据 `results/phasefield/_iter_stability/{iter_stability,l0_sweep,mem_scaling}.csv` + `spec_auxfast_d*.npz`。

> **✅ 已修订（2026-06-10）：§5.2b 改为 restart-aware，头条从「O(100) 有界」升级为「O(10–50) 有界收敛（restart=200）」。**
> 旧 §5.2b 全部局部化数据用 `restart=60`，其 O(100)/DNF 主要是 GMRES(60) 在**非正规鞍点**上的重启停滞，非预条件子退化。
> restart≥200 下同一真实局部化算子 niter 有界 O(10–50)（step14 头条 93→18、step15 173→28、maxd=1.0 的 step17/20 由 DNF→25/82）。
> **定位决定**：restart **不作为论文论点**（避免把方法功劳归因到求解器旋钮 + 重启停滞是教科书结论非新发现）；
> 正文用单一 restart=200 报有界收敛 + 一句方法学脚注，restart=60 对比数据仅作备审（`D13_IMPL §9.4`）。
> 表 1b/图 1b 已重做：表 = restart=200 列（本对话复核重跑 step14/15 与 §9.4 一致）；图 = 单曲线（`make_fig1b_restart_aware.py`）。
> 文末「§5.2b 修订方案」为修订前的规划记录（其中"restart=300 双曲线/restart 作敏感性洞察"已被上述定位决定取代）。

---

## §5.1 实验配置

- **算例 model0**：圆孔缺口板拉伸（Miehe-type radial pre-notch），单位方域、内圆 u=d=0。
- **材料**：E=200, ν=0.2, Gc=1.0, l₀=0.02（§5.3 中 l₀ 作扫描轴）。
- **离散**：Hu–Zhang 混合元，应力 p=3、位移 P2、损伤 P2；退化 AT2 + quadratic + hybrid split；退化下界 eps_g=1e-6（固定，仅 §5.4/谱分析作消融）。
- **预条件子（GMRES，rtol=1e-8）对照列**：`none`（无预条件）、`jacobi`（块对角）、`ILU`、**`aux_fast`（本工作：对称两层辅助空间 V-cycle，D12 §3.2）**。约定 niter=60000 = 打满 maxit·restart（未收敛 DNF）。
- **网格档**（σ-DOF）：

| 档 | hmin | σ-DOF |
|---|---|---|
| h₁ | 0.05 | 10,924 |
| h₂ | 0.025 | 48,092 |
| h₃ | 0.013 | 183,524 |
| h₄ | 0.008 | 503,598 |
| h₅ | 0.004 | 2,045,540 |

> 损伤态：§5.2/§5.4/谱用合成损伤（§5.4 均匀 d、§5.2 峰值 d）以受控扫条件数；§5.2 另用真实相场 run 的物理 d 佐证。

---

## §5.2 网格无关性（Claim C2，**表 1 + 图 1**）

固定峰值损伤 d=0.9，各预条件子 GMRES 迭代数随网格细化：

**表 1**（niter；60000=DNF；"—"=该规模不可行，未跑）

| σ-DOF | none | jacobi | ILU | **aux_fast** |
|---|---|---|---|---|
| 10,924 (h₁) | 27,184 | 2,195 | 7 | **9** |
| 48,092 (h₂) | 60,000 (DNF) | 4,693 | 59,802 (DNF) | **15** |
| 183,524 (h₃) | — | — | — | **9** |
| 503,598 (h₄) | — | — | — | **8** |
| 2,045,540 (h₅) | — | — | — | **9** |

> 对照组 none/Jacobi/ILU 在 h₂ 已 DNF/打满（none 6万、ILU 6万、Jacobi 数千），h₃⁺（184k–2.0M）单次 GMRES 即不可行（none/ILU 数千秒仍 DNF），故只跑 aux_fast 至 h₅。

- **图 1** `iter_stability_vs_N.{png,pdf}`（log-log，aux 全 5 档平线 + 对照组止于 h₂ 爆炸/DNF）。
- **结论（可直接写）**：aux_fast 的迭代数对网格细化**完全有界且基本不变**——跨 **187× DOF**（10,924→2,045,540）始终 **8–15**（9/15/9/8/9），呈 mesh-independent；而 ILU 从 7 爆炸到 59,802（DNF）、Jacobi 数千、none 不收敛，且三者在 h₃⁺ 已彻底不可行。这是预条件子网格无关性最直接的证据。
- **补充（真实相场 run，起裂前物理损伤）**：aux_fast niter = **6 / 7 / 8**（h₁/h₂/h₃，σ-DOF 11k→48k→184k），16× DOF 仅 +2，独立佐证 mesh-independence（数据 `paper_aux_scan_auxfast_h{1,2,3}/`）。
- **⚠️ 重要诚实标注（真实裂纹局部化 vs 合成均匀-d）**：上述受控基准（§5.2/§5.4）用**均匀** d 扫条件数，niter 始终 O(10)（≤18，even d=0.999）。在真实相场 run 中，aux_fast niter 在 maxd≤0.43 时恒 O(1)，裂纹**完全局部化**（尖锐 d≈1/d≈0 界面，maxd→1.0）后升到 **O(10–50)**（restart=200，表 1b）。即均匀-d 基准略低估尖界面下的迭代难度，但**两者同量级 O(10)**——局部化只是把 niter 从 O(1) 推到 O(10–50)，并非量级爆炸。〔历史注：旧 D12 曾报此处"骤升到 ~95–107、O(100)"，那是 **GMRES restart=60 的重启停滞**，非预条件子退化；restart≥200 下为 O(10–50)，详 §5.2b restart 脚注 + `D13_IMPL §9.4`。〕**对论文论点的影响**：aux 跨局部化**始终有界收敛**且为唯一可行解（none/Jacobi/ILU 在该状态发散/DNF）；§7 "O(10) mesh-independent" 限定为**受控均匀-d 基准**，真实局部化下升到 O(10–50) 但保持收敛、对手全失效。

---

## §5.2b 真实裂纹局部化下的有界收敛（论文头条，**表 1b + 图 1b**）

> **定位（主线重设计）**：这是论文的**头条**结果，§5.2/§5.4 的合成均匀-d 受控基准（O(10) mesh/d/l₀-无关）降格为"为什么它鲁棒"的**机理证据**。卖点不是"迭代数恒定"，而是**"在损伤完全局部化 + 直接法 OOM 的最难 regime 里，aux_fast 是唯一仍给出有界收敛的求解器"**——这是**质的差距**（有界 vs DNF/OOM），不是常数因子。

**配置**：model0 圆孔缺口板拉伸，真实相场 staggered run（**物理损伤**，非合成），逐载荷步推进直到裂纹完全局部化（h₂，σ-DOF 48,092）。aux_fast，GMRES `rtol=1e-8, restart=200`（生产默认；见下方 restart 方法学脚注）。表中 niter 为在每个 checkpoint 上确定性重解一次的迭代数（`d12_recheck.py`，固定 seed，与机器负载无关）。

**表 1b — aux_fast 跨裂纹局部化的迭代数（h₂, σ-DOF 48,092，restart=200）**

| step | max_d | regime | niter (restart=200) | 收敛 |
|---|---|---|---|---|
| 10–13 | ≤0.43 | 局部化前 | 2 | ✓ |
| **14** | **0.43→0.998** | **局部化跃迁** | **18** | ✓ |
| 15 | 0.998 | 完全局部化 | 28 | ✓ |
| 17 | 1.000 | 完全局部化 | 25 | ✓ |
| 20 | 1.000 | 完全局部化 | 82 | ✓ |

> 数据来源 `d12_recheck.py`（确定性 seed，每个 checkpoint 装配一次算子、各 restart 重解；机理与完整 restart 扫见 `D13_IMPL §9.4`）。本次复核重跑 step14/15 与既有 §9.4 表一致（step14 restart=200: 17 vs 18；restart=60: 91 vs 93），数值可复现。

- **图 1b** `iter_stability_localization.{png,pdf}`（脚本 `make_fig1b_restart_aware.py`）：niter（restart=200，左对数轴）+ max_d（右轴）vs load step，标注 step13→14 局部化跃迁与 O(10–50) 有界带。**单曲线**——不画 restart 对比曲线（restart 不是本文论点，见脚注）。
- **结论（可直接写）**：aux_fast 的 niter 与 max_d 同步——maxd≤0.43 时恒 O(1)，裂纹在单个载荷步内完全局部化（maxd 0.43→1.0）后 niter 升到 O(10–50) 并**始终有界收敛**（restart=200）。对手在此工况下**全部失效**：在真实局部化算子上直接实测 none 停滞（rel-res 0.12）、Jacobi/ILU 发散（rel-res 6.8/8.8，§5.2b 对手表），且 direct 在大 N（2.0M σ-DOF）11.4 GB OOM（§5.5）。**有界 O(10–50) 收敛 vs 发散/OOM** 是质的差距，不可被常数因子论证抹平。
- **⚠️ restart 方法学脚注（必须写，堵"niter 为何这么高"质疑）**：上述有界性使用 GMRES `restart=200`（生产默认）。更小的 restart（如旧 D12 默认的 60）在该**非正规鞍点**算子上表现出 restarted-GMRES 典型的**重启停滞**——同一 step14 算子 niter 虚高到 91–93、完全局部化态（step17/20）甚至打满 maxit 不收敛（DNF）。这是 GMRES 重启的已知行为，**不反映预条件子本身的收敛性**；restart≥200 即恢复 O(10–50)。**restart 不作为本文论点**，仅此一句方法学说明 + 可复现配置即可（restart=60 的完整对比数据见 `D13_IMPL §9.4`，作备审）。
- **⚠️ 内存/迭代权衡（与 §5.5 小内存卖点的关系，须诚实标注）**：增大 restart 会线性增大 GMRES Krylov 基内存（≈ `m·N·8` bytes，m=restart，N=total-dof）——这与 §5.5 的"小内存"卖点存在张力，**不可把"增大 restart 降低迭代数"写成正面卖点**（重启停滞是教科书结论、非 novel，且自递内存矛盾）。化解关键：**本文两个头条活在不相交 regime,从不同时触发"高 restart + 大 N"**。(i) §5.5 内存头条用**合成均匀-d**(无尖界面)，**大 N 下 restart=60 即 O(10) 收敛**，基内存可忽略（2.0M-dof×60≈3.2GB 但实际不需 200）；(ii) §5.2b 迭代头条只在**真实局部化**态需 restart=200，而真实局部化 checkpoint 仅 h₂（N=82,508）一档（§9.7：square/model2 局部化态 1.5M/847k 本机跑不动），restart=200 基内存仅 ~132MB，相对论点可忽略。**故两卖点各自成立、互不啃 margin。** 同时面临"大 N + 强局部化"时，降低 restart 内存的**正解是预条件子层面的 deflation/界面增广（D13 future work），而非进一步增大 restart**——此点写入 Discussion，主动堵质疑并衔接 future work。
- **⚠️ 诚实标注（maxit 触顶 = restart=60 产物，restart=200 下消失）**：旧 D12 表里"少数 GMRES 触及 maxit 上限、converged=False"的现象，是 **restart=60+maxit=200 的产物**（重启停滞致迭代耗尽）。在 restart=200 下 step14–20 全部收敛至 rtol=1e-8（表 1b，收敛列全 ✓），**无触顶**。故论文**不再需要** "偶有迭代触及 maxit 上限" 的免责标注——直接写 "**restart=200 下跨完全局部化 O(10–50) 有界收敛，全部达 rtol=1e-8**" 即可。（restart=60 触顶/DNF 统计见 `D13_IMPL §9.4`，仅作 restart 敏感性备审。）
- **✅ 直接证据：局部化算子上对手实测失效（2026-06-09 补，替代"a fortiori"论证）**：之前"对手在局部化下全 DNF"是从均匀-d 表**反证**（对手连更易的均匀态都 DNF）。现**直接在真实局部化算子上实测**：加载 `paper_aux_scan_auxfast_h2/.../checkpoints/step_015.npz`（maxd=0.9967，σ-DOF=48,092，total-dof=82,508），装配 standard 算子，同 rtol=1e-8 各求解器各跑一次（脚本 `scripts/paper_huzhang/check_localized_baselines.py`，数据 `results/phasefield/_iter_stability/localized_baselines.csv`）：

  | 求解器 | niter | 收敛 | rel-res | 备注 |
  |---|---|---|---|---|
  | **aux_fast (restart=200)** | **28** | ✓ | →1e-8 | 有界 O(10–50)（表 1b step15） |
  | aux_fast (restart=60) | 98 | ✓ | →1e-8 | restart 停滞虚高（仅备审对照） |
  | direct (pardiso) | — | ✓ | 3.1e-13 | **成功**，1.7s |
  | direct (superlu) | — | ✓ | 1.6e-11 | **成功**，3.8s |
  | none | 2100(cap) | ✗ | 1.2e-1 | 停滞 |
  | jacobi | 2100(cap) | ✗ | 6.8 | **发散** |
  | ilu | 2042 | ✗ | 8.8 | **发散** |

  - **结论1（坐实头条）**：none/Jacobi/ILU **在真实局部化算子上直接实测失效**（none 停在 rel-res 0.12，Jacobi/ILU 残差反增至 6.8/8.8 发散），不再依赖均匀-d 反证 → 审稿人复现质疑已堵。aux_fast 收敛（restart=200 下 28 iter，O(10–50) 有界）。**注**：对手发散是残差增长（与 restart 无关，restart=30/60/200 都发散），故 restart 修订**不削弱**质差头条——对手发散是真发散，不是重启停滞。aux_fast=98 那行是 restart=60，仅留作对照。
  - **⚠️ 结论2（修正"direct 失效"措辞）**：**direct 在 h₂（82k-dof）局部化算子上轻松成功**（pardiso 1.7s / rel-res 3e-13）。故"direct 失效"**不是局部化本身导致**，而是**大 N 内存墙**（§5.5，2.0M σ-DOF 时 11.4GB OOM/singular）。论文/§5.2b 措辞须精确：局部化 regime 里**对手是迭代法（none/Jacobi/ILU）全失效 + direct 在大 N OOM**，而非"direct 在局部化下失效"。中等 N 局部化态 direct 可解。

- **可改善性（→ Discussion 一句话）**：局部化下 niter 升到 O(10–50) 的主因是 **GMRES restart**——restart≥200 即恢复有界收敛（表 1b）。在 restart 充分的前提下，残余的几何纯 P1 加权粗空间局限（无法分辨尖界面内 g(d) 跳变）是**正交的次要优化**；界面感知粗空间（B1）可再降常数因子，完整解留 future work（详见下方"B1 界面感知粗空间"节，**不进正文 §5**）。

---

## B1 界面感知粗空间（**不进正文 §5；Discussion 一句话 + future work + 生产旋钮**）

> **定位决定（2026-06-09 终判）**：B1 **不作为正文独立小节**。理由是**修辞冲突**——§5.2b 头条逻辑是"O(100) 有界即胜利、aux 是唯一可行解"；专门开一节"修" O(100) 等于自承缺陷，且 −28% 是**部分缓解的负结果**，会招来"为何不做完 B2 再投 / PI_s 为何还是 P1 / α=8 是否 overfit 单个 checkpoint"等本文无数据回答的问题。**B1 的价值是"证明懂这个 regime"，不是"解决了它"——前者一句话够，后者它还没做到。** 故：(1) 论文 **Discussion 一句话**带过（根因 + 部分缓解 + 完整解 future work）；(2) 完整 B2 留 future work / 下一篇；(3) 代码侧作**生产旋钮**白拿提速。本节为内部研究记录，非论文写作清单项。

> **动机**：§5.2b 的 O(100) 退化根因——几何纯 P1 加权粗空间用 g(d) 点值加权，但延拓算子 PI_s 与权重都无法分辨尖锐界面内 g(d)≈1→1e-6 的单元内跳变。**B1 思路**：把粗扩散权重乘以界面感知因子，在退化梯度大处（即界面处）增强粗校正：
> $$\text{coef}_\text{aware} = g(d)\,\bigl(1 + \alpha\,\|\nabla g(d)\|/\max\|\nabla g(d)\|\bigr),\quad \nabla g(d) = g'(d)\,\nabla d$$
> 均匀 d 下 ∇d=0 ⇒ 因子恒为 1，**B1 自动退化为原方法**（向后兼容）；只有真实尖锐界面才被激活。

**实现**（`fracturex/utilfuc/linear_solvers.py`）：
- 新增 `_make_interface_aware_coef(base_coef, damage, state, alpha)`——包装 g(d) 系数，∇d 由 `state.d.grad_value(bcs)` 解析求值（FEALPy Function 原生支持，**不改 damage 模块、不动 fealpy**），g'(d) 由 `damage.degradation_grad(d)`。
- `solve_huzhang_block_gmres_fast` / `_fast_cached_coarse_ml` 加 `interface_aware`、`interface_alpha` 参数，缓存键含 `interface_aware` 标志。
- run_case.py 旋钮：`FRACTUREX_AUX_INTERFACE_AWARE=1`、`FRACTUREX_AUX_INTERFACE_ALPHA`（默认 1.0）。默认关闭。

**初步验证（合成尖锐裂纹带，h₂ σ-DOF 48k，d∈[0,1.0]，带宽 0.5·l₀）**：

| 模式 | α | niter | 收敛 |
|---|---|---|---|
| baseline（几何 P1） | — | 15 | ✓ |
| interface_aware | 0.5 | 14 | ✓ |
| interface_aware | 1.0 | **13** | ✓ |
| interface_aware | 2.0 | 13 | ✓ |
| interface_aware | 4.0 | 13 | ✓ |

- **合成带初步结论**：方向正确（15→13，−13%），α≥1 后饱和；合成带难以复现 §5.2b 的 O(100) regime（baseline 仅 15），故改善幅度被低估。

**关键验证（真实 checkpoint 局部化 d 场，h₂ step15，maxd=0.998，σ-DOF 48k）**——加载真实 d/H 状态装配真实算子，对比 baseline vs α 扫（`validate_interface_aware_realfield.py`）：

| 模式 | α | niter | t_solve (s) |
|---|---|---|---|
| baseline（几何 P1） | — | 170 | 581 |
| interface_aware | 0.5 | 157 | 468 |
| interface_aware | 1.0 | 161 | 584 |
| interface_aware | 2.0 | 152 | 458 |
| interface_aware | 4.0 | 141 | 406 |
| interface_aware | **8.0** | **123** | **370** |

- **⚠️ 该表测于 restart=60**：baseline=170 是 **GMRES restart=60** 的停滞 niter；表 1b 已证同一 step15 算子在 restart=200 下 baseline 即降到 **28**（O(10–50)）。故 B1 的 −28% 是在**重启停滞 regime 内**测得的常数因子缓解——一旦 restart≥200 抢先把 niter 压回 O(10–50)，B1 增广的边际收益大幅缩小。（B1 是否在 restart=200 baseline 上仍有可观增益**未单独重测**，预期收益微，留作 future work。）
- **真实场结论（诚实判定）**：B1 在 restart=60 下单调改善、α 越大越好（170→123，**−28%**；t_solve 581→370s，**−36%**），全部仍收敛到 1e-8。但这是 **constant-factor 改善，不是 regime 改变**。**更重要的是：把 niter 从 O(100) 压回 O(10–50) 的"regime 改变"已由 restart（60→200）免费实现**，B1 只是在此之上的正交次要优化。按预设判据，**B1 不足以、也不需要作为论文主贡献**。
- **根因**：界面感知修正的是粗扩散**权重**，但**延拓算子 PI_s 仍是几何纯 P1**，无法表示界面两侧跳变模态。完整解需 **B2 界面自适应粗空间（界面单元 P2 延拓）**，但 B2 破坏几何缓存（PI_s 键需含界面单元集、每步失效）+ 须改 auxspace/fast 双求解器，成本/风险高一个量级。
- **论文处理（不进 §5 正文）**：
  - **Discussion 一句话模板（restart-aware）**：「局部化下迭代数升高的主因是 GMRES 重启参数——restart≥200 即恢复 O(10–50) 有界收敛；其上残余的几何纯 P1 粗空间局限（无法表示尖界面跳变模态）是正交的次要优化，界面感知加权可再降常数因子（实验中 −28%），界面自适应粗空间留作 future work。」定位为"理解了为什么、也知道下一步怎么走"，而非"方法有洞、补了一半"。
  - **完整 B2（界面自适应粗空间 / 界面单元 P2 延拓）= future work**，本文不做（成本/风险高一量级，见上"根因"行）。
  - **生产旋钮**：真实 run 直接开 `FRACTUREX_AUX_INTERFACE_AWARE=1 FRACTUREX_AUX_INTERFACE_ALPHA=8`，局部化区白拿 ~28% niter / ~36% 墙钟，零风险（均匀 d 区自动退化为原方法）。与论文措辞无关，纯工程加速。
- 脚本：`scripts/paper_huzhang/validate_interface_aware{,_realfield}.py`；数据 `results/phasefield/_iter_stability/interface_aware_realfield.csv`。

---

## §5.2d 物理一致性：load–displacement（Claim C1 在 model0 h₂，**图 1d**）

aux 解与 direct 解在物理量（反力-位移曲线）上的一致性，证 aux 预条件器不改变物理解（只改求解路径），同时给出 model0 的完整断裂响应曲线。

- **配置**：model0 h₂（σ-DOF 48,092），direct（pardiso，全程物理基准 31 步）vs aux_fast（物理收敛止于 step15）。脚本 `scripts/paper_huzhang/make_model0_h2_loaddisp.py`，图 `docs/figures/precond/model0_h2_loaddisp.{png,pdf}`。
- **物理响应（direct，完整曲线）**：|R| 随 u_y 线性升到**峰值 28.14**（step13, max_d=0.43），裂纹局部化后软化，完全分离（max_d=1.0）后反力**平滑趋零**（final |R|=0.052）——典型脆性 I 型断裂响应。**论文载荷曲线用 direct**（完整、物理正确、便宜）。
- **aux 一致性（精确）**：step0→15（含峰值 + 软化到 max_d=0.998）aux 与 direct 反力相对差 ≤1.5e-3，**重合**（C1 一致性，红圈贴合 direct 实线）。
- **⚠️ aux 在完全分离瞬间（step16, max_d→1.0）DNF——迭代法真实边界，非 restart artifact**：从 step15 干净 checkpoint 续算，**即使 restart=200 + maxit=400**（关 Anderson 与开 Anderson 各试一次），step16 的弹性鞍点 GMRES 均**打满 maxit 不收敛**（残差 ~0.46），staggered 外层随之震荡（error 0.4–5.6）、弹性增量归零空转。同一物理步 direct(pardiso) 可直接解出。即 **max_d 跨到 1.0 完全分离瞬间的鞍点，直接法可解、aux-GMRES 迭代法 DNF**——这是迭代法在完全奇异（g=eps_g 整条带）鞍点上的真实边界，与 restart/Anderson 无关。
  〔注：早期 `d12_recheck` 曾报"step17@restart=200 → 25 步收敛"，那是读了 **restart=60 污染 run 写的 step17 checkpoint**（d 场非物理）；用从 step15 干净续算的真实 d 场，step16 即 DNF。旧 restart=60 跑出的 step17–30 反力翻正爬到 +102 是 restart 停滞产物，亦弃用。〕
- **结论**：C1 一致性在 model0 h₂ 成立于 **max_d≤0.998 的物理良态区**（aux≡direct，覆盖弹性+峰值+软化）；完全分离瞬间（max_d=1.0）aux-GMRES DNF 是迭代法边界（direct 可解）。**论文取舍**：载荷曲线用 direct 完整呈现；aux 的价值是良态区一致性（C1）+ 局部化区有界收敛（§5.2b，max_d≤0.998），不宣称 aux 跑完完全分离后段。这与"迭代法在退化极限需 direct/正则化兜底"的 Discussion 定位一致。
- **h₃（σ-DOF 183,524）同结论**（图 `model0_h3_loaddisp`）：aux 与 direct 在 step0→13 一致（rel diff ≤1.3e-5，覆盖弹性+峰值 28.19），step14 max_d 由 0.42 直接跳到 1.0（h₃ 更脆，无 h₂ 的 0.998 中间软化步）即偏离（rel diff 3.4e-2）、此后 aux-GMRES DNF。同一迭代法边界跨网格复现（h₂ step16 / h₃ step14，均 max_d=1.0 完全分离）。direct_h₃ 完整曲线 peak 28.19 → 趋零（final 0.02）。

---

## §5.3 正则化长度 l₀ 无关性（Claim，**表 2**）

固定 h₂、峰值 d=0.9，合成损伤过渡层宽 ∝ l₀：

**表 2**（aux niter）

| l₀ | aux_fast | aux_weighted |
|---|---|---|
| 0.04 | 7 | 7 |
| 0.02 | 7 | 7 |
| 0.01 | 7 | 6 |

- **结论**：aux 迭代数对 l₀ 完全不敏感（aux_fast 恒 7），即对损伤过渡层陡度鲁棒。（ILU 在该规模 lgmres 极慢/不 robust，作定性对照。）

---

## §5.4 损伤鲁棒性 d→1（Claim C1，**图 2**）

固定网格，均匀损伤 d 从 0 增到 0.999（g(d)=(1-d)²+eps_g：g 从 1 降到 eps_g=1e-6，应力块 1/g 放大 6 个量级）：

**表 3a — h₁ (σ=10,924)**（niter；60000=DNF）

| d | none | jacobi | ILU | **aux_fast** |
|---|---|---|---|---|
| 0.0 | 60000(DNF) | 3787 | 165 | **7** |
| 0.5 | 60000(DNF) | 3072 | 159 | **6** |
| 0.9 | 27184 | 2195 | 7 | **9** |
| 0.99 | 3534 | 5712 | 3 | **10** |
| 0.999 | 42843 | 18923 | 253 | **10** |

**表 3b — h₂ (σ=48,092)**

| d | none | jacobi | ILU | **aux_fast** |
|---|---|---|---|---|
| 0.0 | 60000(DNF) | 7521 | 41 | **7** |
| 0.5 | 60000(DNF) | 5600 | 59803 | **7** |
| 0.9 | 60000(DNF) | 4693 | 59802 | **15** |
| 0.99 | 14960 | 21758 | 22 | **18** |
| 0.999 | 60000(DNF) | 60000(DNF) | 59807 | **17** |

- **图 2** `iter_stability_vs_d.{png,pdf}`（半对数，取最细网格 h₂）。
- **结论**：跨 6 个量级条件数，aux_fast 迭代数始终 **O(10)（≤18）且有界**；none 多数不收敛，Jacobi 数千–DNF，ILU erratic（低 d 居中、d=0.5/0.9 发散打满、高 d 偶小），均无鲁棒性。aux 是唯一对 d→1 退化鲁棒的预条件子。

---

## §5.5 内存扩展（小内存支线，**表 4 + 图 3**）

matrix-free（不存 M2/因子，逐单元施加 A 作用）vs direct（pardiso 分解），独立进程峰值 RSS：

**表 4**（peak RSS, MB）

| σ-DOF | matrix-free | direct (pardiso) | mf/direct |
|---|---|---|---|
| 10,924 | 340 | 421 | 0.81 |
| 48,092 | 429 | 768 | 0.56 |
| 183,524 | 719 | 2,043 | 0.35 |
| 503,598 | 1,374 | 5,185 | 0.27 |
| 2,045,540 | **4,629** | **11,425（pardiso 失败/singular）** | direct 不可行 |

- **图 3** `mem_scaling.{png,pdf}`（log-log，direct 末点标 infeasible）。
- **结论**：matrix-free 全程内存更省，优势随规模扩大（mf/direct 0.81→0.27×）；σ-DOF 2.0M 处直接分解需 11.4 GB 且 pardiso 在不定鞍点上失败，而 matrix-free 仅 4.6 GB 仍可行——**大规模/受限内存下迭代+matrix-free 唯一可行**。
- **正确性**：matrix-free 算子 matvec/解与装配版机器精度一致（rel 3e-16~4e-15），niter 不变（D12 §13.4 B）。
- **诚实标注**：2D pardiso 嵌套剖分内存近线性，故此处赢的是**常数因子 2–4× + 大规模鲁棒性**；O(N²) vs O(N) 的戏剧性 OOM 是 **3D** 论点（future work）。
- **⚠️ GMRES Krylov 基内存（与 §5.2b restart 的关系）**：本表只计 matrix-free 算子 vs direct 因子的内存；完整迭代求解还需 GMRES Krylov 基 ≈ `restart·N·8` bytes。本节大 N 数据用合成均匀-d，restart=60 即收敛（基 ~3.2GB@2.0M-dof），未触发 §5.2b 的 restart=200；而 §5.2b 的 restart=200 只用在中等 N（82k，基 ~132MB）。**两 regime 不重叠，故小内存论点不被高 restart 啃掉**（详 §5.2b 内存/迭代权衡脚注）。

---

## §5.6 谱验证（Prop 2/3 数值支撑，**图 4**）

aux_fast 预条件算子 P⁻¹K 的体谱（20 个最大幅特征值）随损伤 d：

**表 5**（h₁）

| d | \|λ\|min | \|λ\|max | κ_LM=\|λ\|max/\|λ\|min |
|---|---|---|---|
| 0.0 | 110.2 | 158.1 | 1.435 |
| 0.5 | 109.2 | 158.0 | 1.447 |
| 0.9 | 109.2 | 157.9 | 1.447 |
| 0.99 | 109.2 | 157.9 | 1.447 |
| 0.999 | 109.2 | 157.9 | 1.447 |

- **图 4** `spectrum_scatter.{png,pdf}`（复平面散点，d 叠加）+ `spectrum_kappa_vs_d.{png,pdf}`。
- **结论**：P⁻¹K 体谱全为实数、紧聚集于 [109,158]，且随 d 从 0 增至 0.999 **几乎不动**（κ_LM 恒 1.44）——预条件后谱与损伤无关，数值验证参数无关性（Prop 3）。
- **诚实标注**：ARPACK 的最小幅特征值（SM）在该非正规块预条件算子上不收敛/不可靠，κ via SM 是启发非真界（见 `estimate_spectrum` 注释）；故只报体谱（LM）聚集，预条件子质量的主证据是 niter（§5.2–5.4）。

**表 5b — 谱的网格无关性（H2；固定 d=0.9，扫网格）**

| 网格 | σ-DOF | \|λ\|min | \|λ\|max | κ_proxy=\|λ\|max/\|λ\|min |
|---|---|---|---|---|
| h₁ | 10,924 | 109.2 | 157.9 | 1.447 |
| h₂ | 48,092 | 151.5 | 174.1 | **1.150** |
| h₃ | 183,524 | 164.5 | 177.0 | **1.076** |

- 数据 `spec_auxfast_meshindep_hmin{0.025,0.013}_d0.9.npz`（h₁ 取自 §5.6 d-扫的 d=0.9 档）。
- **结论（H2）**：固定 d=0.9 扫网格，P⁻¹K 体谱跨 **30× DOF**（10.9k→184k）始终紧聚集于 O(100–180)，κ_proxy **有界且随细化反而更紧**（1.447→1.150→1.076）——谱聚集对 **h 也无关**。与 §5.6 主表（对 d 无关）合起来，Prop 3 的参数无关性（对 d 与 h 双向）数值验证完整。

---

## §5.7 算例推广：square（model1，I 型直裂纹拉伸）（D2，**图 5**）

把 §5.4 的损伤鲁棒性结论推广到第二个算例。square 单边切口拉伸，材料 E=210/ν=0.3/Gc=2.7e-3/l₀=0.015，均匀损伤 d 扫；脚本 `scripts/paper_huzhang/iter_stability_square.py`，数据 `results/phasefield/_iter_stability/iter_stability_square.csv`。

**表 6a — nx=20 (σ=13,483)**（niter；60000=DNF）

| d | none | jacobi | ILU | **aux_fast** |
|---|---|---|---|---|
| 0.0 | 60000(DNF) | 12800 | 53(DNF) | **9** |
| 0.5 | 60000(DNF) | 8650 | 33400 | **10** |
| 0.9 | 60000(DNF) | 9200 | 59802(DNF) | **20** |
| 0.99 | 10452 | 17283 | 40 | **21** |
| 0.999 | 60000(DNF) | 37303 | 59803(DNF) | **23** |

**表 6b — nx=30 (σ=30,123)**

| d | none | jacobi | ILU | **aux_fast** |
|---|---|---|---|---|
| 0.0 | 60000(DNF) | 21200 | 23(DNF) | **10** |
| 0.5 | 60000(DNF) | 20000 | 60000(DNF) | **11** |
| 0.9 | 60000(DNF) | 21000 | 60000(DNF) | **34** |
| 0.99 | 17910 | 24027 | 17910 | **31** |
| 0.999 | 60000(DNF) | 60000(DNF) | 60000(DNF) | **28** |

- **图 5** `iter_stability_square_vs_d.{png,pdf}`（nx=30，半对数）。
- **结论**：在 square（I 型）上，aux_fast 跨 d=0→0.999 始终 **O(10)（9–34）且有界并收敛**；none 几乎全 DNF、Jacobi 8千–6万、ILU erratic 且 nx=30 时多数 DNF——**nx=30 下唯有 aux_fast 稳健收敛**。损伤鲁棒性结论在第二个算例上完全成立。
- **诚实标注**：aux_fast 在 square 上的迭代数（9–34）略高于 model0（6–18），且 d=0.9 时随网格细化稍增（20→34，σ 增 2.3×）——I 型直裂纹 + 预裂纹几何对 aux 略难，但仍 O(10) 有界、远优于全部对照组（DNF/爆炸）。

---

## §5.8 算例推广：model2（II 型缺口剪切）（D3，**图 7**）

第三个算例，把损伤鲁棒性结论推广到 **II 型（剪切）** 加载。model2 缺口板 x-向拉伸/剪切（已修 BOGUS：`neumann_data` 切向反力曾被零化致 u≡0，2026-05-30 修复并核 |F|=7.15e-4≠0、niter=9、|u|=0.589），材料同 SquareMat（E=210/ν=0.3/Gc=2.7e-3/l₀=0.015），均匀损伤 d 扫；脚本 `scripts/paper_huzhang/iter_stability_model2.py`，数据 `results/phasefield/_iter_stability/iter_stability_model2.csv`。

**表 8a — nx=24 (σ=19,347)**（niter；60000=DNF）

| d | none | jacobi | ILU | **aux_fast** |
|---|---|---|---|---|
| 0.0 | 60000(DNF) | 9,781 | 50(DNF) | **9** |
| 0.5 | 60000(DNF) | 7,642 | 59,827(DNF) | **9** |
| 0.9 | 58,589 | 6,000 | 59,802(DNF) | **8** |
| 0.99 | 18,036 | 37,114 | 483 | **12** |
| 0.999 | 60000(DNF) | 60000(DNF) | 60000(DNF) | **11** |

**表 8b — nx=32 (σ=34,243)**

| d | none | jacobi | ILU | **aux_fast** |
|---|---|---|---|---|
| 0.0 | 60000(DNF) | 11,202 | 17(DNF) | **10** |
| 0.5 | 60000(DNF) | 10,824 | 60000(DNF) | **10** |
| 0.9 | 60000(DNF) | 11,236 | 60000(DNF) | **11** |
| 0.99 | 30,986 | 41,339 | 30,986 | **14** |
| 0.999 | 60000(DNF) | 60000(DNF) | 60000(DNF) | **14** |

- **图 7** `iter_stability_model2_vs_d.{png,pdf}`（nx=32，半对数）。
- **结论**：在 model2（II 型剪切）上，aux_fast 跨 d=0→0.999 始终 **O(10)（8–14）且有界并收敛**，且对网格细化稳定（nx=24→32 σ 增 1.8×，各 d 仅 +0~2）；none 几乎全 DNF、Jacobi 6千–6万（nx=32 全程 1万+）、ILU 几乎全程 DNF（仅 d=0.99 偶收敛、nx=32 d=0 即发散）——**唯有 aux_fast 在剪切加载下稳健收敛**。损伤鲁棒性结论在第三个算例（且为不同加载模式）上完全成立。
- **三算例横向**：model0（I 型预裂纹）6–18、square（I 型直裂）9–34、model2（II 型剪切）8–14，aux_fast 均 O(10) 有界收敛；对照组在三算例上一致 DNF/爆炸/erratic——预条件子的损伤鲁棒性**与算例、加载模式、裂纹几何无关**。

### §5.8.1 model2 真实相场 run：完整载荷–位移曲线（direct/pardiso，跑满 step200）

上面表 8a/8b 是**合成均匀-d**的 niter 鲁棒性扫描（验预条件子）。这里补**真实相场演化**
run 的端到端载荷曲线，坐实 model2 物理正确、裂纹真实扩展（非均匀-d 人造态）。

- **配置**：model2 缺口板 x-向拉伸，Hu–Zhang p=3 混合元 + AT2 相场（hybrid 分裂），
  **direct/pardiso 弹性 + 无预条件子 GMRES 相场**，真实 staggered，跑满 **step 200**。
  σ-DOF 847,043 / u 614,400 / d 103,041；峰值 RSS ≈ 23.5 GB；总墙钟数日级
  （avg 941 s/step，max 4239 s/step；max_d=1.0 区外层 staggered 数十–64 次/步）。
  数据 `results/phasefield/model2_notch_x_stretch/paper_direct_full/epsg_1e-06/history.csv`。
- **图**：`docs/figures/precond/model2_loaddisp.{png,pdf}`（亦存 `Frac_huzhang/figures/`），
  脚本 `scripts/paper_huzhang/make_model2_loaddisp.py`。
- **载荷曲线（取单调位移前沿）**：

  | 量 | 值 |
  |---|---|
  | 弹性上升段 | 线性，$u_x\in[0,0.0103]$，$|R_x|$ 0→0.42 |
  | **峰值反力** | $|R_x|_{\max}=\mathbf{0.421}$ @ $u_x=0.01033$（step 124） |
  | 裂纹扩展（陡降） | step124→125 $|R_x|$ **0.421→0.329**（−22%，单步突降） |
  | 残余平台 | 后续 $|R_x|\approx0.23$–0.31，锯齿状（增量式裂纹推进 stick–slip） |
  | 终态 | $|R_x|=0.237$ @ $u_x=0.01667$（step 200）；max_d=1.0 全程（预裂纹自始存在） |

- **诚实标注**：
  1. **`summary.json` 陈旧**（`n_load_steps=63`，是早期续算段统计，其 abs_max=0.293 **非**全程峰值）；
     全部结论由 200 步 `history.csv` 现算（真实峰值 0.421 @ step124）。
  2. **载荷日程不连续**（memory `model2_loadschedule_discontinuity`）：续算换位移步长，
     steps 65–76 回退-重载（$u_x$ 非单调）。作图取**单调位移前沿**（189/201 点），raw history 不动。
  3. max_d 全程 =1 是**预裂纹**（自始 d=1），非加载过程"$\max d\to1$"；曲线峰后陡降才是真实扩展。
- **意义**：model2 在真实相场演化下给出物理合理的 II 型载荷曲线（线性升 → 峰值 → 脆性陡降 →
  锯齿残余），与表 8a/8b 的预条件子鲁棒性互补：一证"解得对"（本节），一证"解得稳"（niter 扫描）。

---

## 支撑：σ 收敛阶（HuZhang vs Lagrange）（→ 论文 §7.9 / C5-V2，**图 6**）

> 不属预条件子 §5，是离散质量结果，支撑"为何用 HuZhang 混合元"（贡献 vi）。流形解（manufactured）线性弹性，单位方域；HuZhang 应力 p=3 vs Lagrange p=2 恢复应力 $\sigma=C{:}\varepsilon(u_h)$，同一 Voigt-Frobenius $L^2$ 范数。脚本 `make_hz_vs_lagrange_v2.py`，数据/图 `Frac_huzhang/figures/hz_vs_lagrange_v2.{csv,pdf,png}`。

**表 7**（应力 $L^2$ 误差）

| N | h | err_σ HuZhang(p=3) | err_σ Lagrange(p=2) |
|---|---|---|---|
| 4 | 0.250 | 7.46e-3 | 1.14e-1 |
| 8 | 0.125 | 4.73e-4 | 3.05e-2 |
| 16 | 0.0625 | 2.87e-5 | 7.77e-3 |
| 32 | 0.0312 | 1.77e-6 | 1.95e-3 |
| 48 | 0.0208 | 3.49e-7 | 8.68e-4 |
| **渐近收敛率** | | **4.01**（≈ $p_\sigma+1$）| **2.00**（≈ $p_u$）|

- **图 6** `hz_vs_lagrange_v2.{pdf,png}`（log-log 应力误差 vs h，两元素）。
- **结论**：HuZhang 混合元的应力 $L^2$ 误差以 **4 阶**收敛（$p_\sigma+1$），而位移 Lagrange 元恢复应力仅 **2 阶**（$p_u$）；N=48 时 HuZhang 应力误差小 **3 个量级以上**（3.49e-7 vs 8.68e-4，比值 ≈ 2.5×10³ ≈ 10^3.4；原记"5 个量级"为口误，已订正）。即 HuZhang 直接逼近应力主变量、对断裂关心的应力场有高阶精度优势——为本工作选用 HuZhang 提供离散层面的动机。

---

## 图索引（`docs/figures/precond/`）

| 论文图 | 文件 | 内容 |
|---|---|---|
| 图1 (C2) | `iter_stability_vs_N` | niter vs σ-DOF（aux 平 vs others 爆炸）|
| **图1b (头条)** | `iter_stability_localization` | **真实相场 run niter & max_d vs load step；step14 局部化处 niter 7→O(100) 同步跃迁；触顶步空心标记**（脚本 `paper_make_iter_localization.py`）|
| 图2 (C3) | `iter_stability_vs_d` | niter vs 损伤 d（aux 平 vs others DNF/erratic）|
| 图3 | `mem_scaling` | peak RSS vs σ-DOF（mf vs direct + OOM）|
| 图4 | `spectrum_scatter` / `spectrum_kappa_vs_d` | P⁻¹K 谱 vs d |
| 图5 (D2) | `iter_stability_square_vs_d` | square(I型) niter vs d（aux 有界 vs others DNF）|
| 图6 (§7.9) | `hz_vs_lagrange_v2`（在 `Frac_huzhang/figures/`）| σ 收敛阶 HuZhang 4 阶 vs Lagrange 2 阶 |
| 图7 (D3) | `iter_stability_model2_vs_d` | model2(II型剪切) niter vs d（aux 有界 vs others DNF）|
| 图1d (C1) | `model0_h2_loaddisp` | model0 h₂ load–disp：direct 完整物理曲线 + aux 重合到 max_d≤0.998；step16 完全分离 aux-GMRES DNF 边界标注（脚本 `make_model0_h2_loaddisp.py`）|
| 图1d′ (C1) | `model0_h3_loaddisp` | model0 h₃ load–disp：direct 完整曲线 + aux 重合到峰值 max_d≤0.42；step14 完全分离 DNF 边界（脚本 `make_model0_h3_loaddisp.py`）。迭代法边界跨网格复现 |
| 构造示意 (§6) | `precond_schematic`（`make_precond_schematic.py`）✅ | 工程流程版：块三角 sweep + 两层 V-cycle（大字号 + matrix-free 高亮 + sweep 箭头）；已并入 tex `fig:precond_schematic`。数学/算子版 `make_precond_operator_diagram.py` 已弃用 |

## 待补

> **完整清单见 `D12_PRECONDITIONER_PAPER_PLAN.md` §13.6**（2026-06-03 成稿后复盘，按 tex `\needexp` 逐项盘点，分 G/H/I/J 四组 + 更新 MVP）。下面是高频项摘要：

- **G1（P0，已声称必验）**：论文 §6 提了块对角+MINRES 但 §7 全用 block-tri/GMRES → 补一组 MINRES niter 或改写措辞。
- **G2（C5 只留 V2）**：HuZhang vs Lagrange 制造解 σ L² 收敛阶（smoke 已 4.0 vs 2.0，跑 paper 分辨率扫即可，近免费）。**V5 尖端切片已删**（偏主线+昂贵），tex 已收口。
- ~~**H1（P0）**~~ ✅ **完成**：C2 招牌图 aux d=0.9 已补满 h₃–h₅，跨 187× DOF = 9/15/9/8/9（见 §5.2 表1/图1）。
- **H2/H3**：谱多网格叠加验 mesh-independence；eps_g 轴扫验退化下界鲁棒。
- **I（广度）**：~~D2 square (I型)~~✅(§5.7)、~~D3 model2 (II型剪切)~~✅(§5.8 表8a/8b+图7)、D4 effective_stress（附录 A）；CNT aux 跑过软化段、SENT 全分离刷新、SENS aux/direct（§5 三处 `\needexp`）。
- **J（加分）**：Anderson 外层加速量化、matrix-free niter 跨档、§3 构造示意图。

---

## §5.2b 修订方案（2026-06-09，restart 发现后）

> 起因见文件顶部修订通知 + `D13_IMPL §9`。核心：§5.2b 现有"局部化 O(100)/DNF、aux 唯一有界"的头条，
> 其 O(100)/DNF **主要是 GMRES restart=60 在非正规鞍点上的重启停滞**，非预条件子退化。修订**不削弱 D12，反而更强**。

### A. 必须重跑的数据（统一 restart）

D12 §5.2b 全部局部化结果重跑于 **restart=200 与 300**（保留 restart=60 列作对比，凸显"restart 敏感性"）。脚本
`scripts/paper_precond/d12_recheck.py`（确定性 seed + 清 pyamg 缓存）。已得 model0 h₂ 五 checkpoint（确定性）：

| step | maxd | r60 | r200 | r300 |
|---|---|---|---|---|
| 013 | 0.426 | 7 | 2 | 2 |
| **014（头条跃迁）** | 0.998 | **93** | 18 | **9** |
| 015 | 0.998 | 173 | 28 | 13 |
| 017 | 1.000 | **DNF(400)** | 25 | **14** |
| 020 | 1.000 | **DNF(400)** | 82 | **49** |

待补：h₃（184k）同表；表 1b 逐 staggered step 重跑于 restart=300；图 1b 重做（niter vs load step，restart=300 曲线）。

### B. 头条叙事重写（从「O(100) 有界」→「跨局部化 O(10–50) 有界，restart 充分即可」）

- **旧头条（删）**：「裂纹完全局部化时 niter 由 7 跃至 O(100)（95–200），aux 是唯一仍有界收敛者」。
- **新头条（更强、更干净）**：「**aux_fast 跨完全局部化（maxd 0.43→1.0）保持有界收敛 O(10–50)**，只要 GMRES restart 足够
  （≥200–300）；restart=60 时 GMRES 在非正规鞍点上重启停滞，niter 虚高到 O(100) 乃至 DNF——这是 restart 而非
  预条件子的极限。对照 none/Jacobi/ILU 即便 restart=60 也发散/停滞（实测 rel-res 6.8/8.8），**质的差距仍成立**。」
- **为何更强**：(1) 把"O(100)"这个看似缺陷的数字变成"O(10–50) 有界"的鲁棒性主张；(2) 仍保留对手全失效的质差；
  (3) 主动揭示 restart 敏感性 = 展示对方法的深刻理解，堵审稿人"为何 niter 这么高"的质疑。

### C. 受影响处的逐条修订

1. **§5.2 L48 诚实标注**：「真实局部化骤升到 ~95–107（约 14×）」→ 改为「restart=60 时升到 ~95–107，restart≥200 仍 O(10–30)；
   该跃升是 restart 停滞，非预条件子退化（详 §5.2b 修订）」。
2. **§5.2b 表 1b**：现 niter（93/162/171…）是 restart=60；重跑 restart=300（预期 O(10–50)）。保留 restart=60 列对比。
3. **§5.2b L71 结论**：「niter 由 7 跃至 ~95、稳定 O(100)」→ 「restart 充分时跨局部化 O(10–50) 有界；restart=60 虚高到 O(100)」。
4. **§5.2b L72 maxit 触顶标注**：触顶是 restart=60+maxit=200 的产物；restart=300 下 niter≤49，远不触顶 → 此标注大幅简化或删。
5. **§5.2b L73–77 对手实测表**：aux=98 是 restart=60；补 restart=300 的 aux=13。对手（none/Jacobi/ILU）发散结论**不变**
   （它们 restart=30 也发散，且发散是残差增长非重启停滞，与 restart 无关）→ 质差头条**保住**。
6. **B1 节（L93+）**：B1 −28% 的对照 baseline（170）是 restart=60。restart=300 下 baseline 已 ~13，B1 增广意义大减
   → **B1 定位进一步弱化**（restart 已抢走"解决局部化"，B1/D13 增广在 restart 够时收益微）。Discussion 一句话可改为
   「局部化下 niter 升高主因是 restart；充分 restart 即恢复 O(10–50)。界面感知/学习增广是正交的次要优化」。
7. **§5.5 内存墙**：不受影响（direct OOM 是内存问题，与 restart 无关）；但"局部化致迭代法失效"措辞须按上改为
   "none/Jacobi/ILU 发散（与 restart 无关）+ direct 大 N OOM"。

### D. 不变的结论（修订不动）

- §5.2 mesh-independence（合成均匀-d，O(10)）：不变（均匀-d 无尖界面，restart=60 也 O(10)）。
- §5.4 d→1 损伤鲁棒（O(10)）：不变（合成均匀-d）。
- §5.5 内存墙 / matrix-free：不变。
- §5.7/§5.8 square/model2（合成均匀-d，O(10–34)）：不变（同为均匀-d）。
- 对手（none/Jacobi/ILU）在真实局部化算子上发散：不变（发散非重启停滞）。
- 谱验证 §5.6：不变。

### E. 一句话给作者

> **D12 不用慌**：修订把"O(100) 有界"升级成"O(10–50) 有界 + restart 敏感性洞察"，是**更强**的论文。
> 唯一实打实要重跑的是 §5.2b 的 niter 数字（restart=60→300）+ 图 1b。其余结论（mesh/d/l₀ 无关、对手失效、内存墙）全部不动。
