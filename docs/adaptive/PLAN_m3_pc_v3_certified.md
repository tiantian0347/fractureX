# 实施计划：M3 PC v3 — 提速 + 曲线修复 + 严格 Θ 认证 (a)(b)+

> 状态：计划稿 v0.1（2026-06-14）。承接 [RESULTS_aposteriori.md](RESULTS_aposteriori.md) §M3 PC v2 的
> 三个 next（(a) primal 重解补 Θ、(b) Anderson 消 DNF、曲线峰值差）。
> 用户已定：曲线先廉价复核 du；认证本轮一起做。

## 0. 现状诊断（已读代码确认）

**v2 的浪费（`run_m3_pc_model1.py:155` 内层 corrector loop）**：每个载荷步内反复
「整步 staggered 解(tol 1e-4, ≤200 iter) → 算 𝒟 → 标记 → 加密 → 回步首重解」，**中间网格的
整解全被丢弃**。step3 花 2320s = 8 个 corrector × 整步求解，前 7 个解完即弃。

**v2 曲线剩余 +2.8% 峰值差的三个独立来源**（拆开）：
| 来源 | v2 | nx=120 参照 | 判读 |
|---|---|---|---|
| **载荷步 du** | 2.5e-4 | **1e-4（2.5× 细）** | ⚠️ 主嫌：脆性软化下粗 du 系统性高估峰值（采样偏差，非网格）|
| 带分辨 h/l₀ | ≈0.5 | ≈0.56 | v2 已略优；非主因 |
| 离散格式 | Hu–Zhang p=3 | standard 位移 FEM | σ 表示不同 |

**已有可复用接缝**：
- `driver` Anderson 全 env 接好（`FRACTUREX_ANDERSON_DEPTH=5` 等），model2 已验 15–40×。
- `eta_from_state`（用 DG-u，**非严格 Θ**，= v2 诚实标注 #1 的根）、`refine_and_rebuild`(θ-Dörfler)。
- `solve_primal_degraded`（**仅 MMS**：吃 `pde.damage()`/`pde.source()`，不能直接喂真实算例）。

⇒ 严格 Θ 须**新写**非-MMS 连续标准 FEM primal 重解（吃离散 d 场 + 真实 Dirichlet 载荷）。

---

## Phase 1 — 提速 (b) + (i)(ii)：消 DNF + 消浪费

目标：7h → 目标 <2h，且 step22–23 DNF 消失。**纯性能，不动精度语义。**

**1.1 (b) Anderson 接入 corrector loop**
- runner 已用 env 驱动 driver；只需跑时设 `FRACTUREX_ANDERSON_DEPTH=5`（model2 同配置）。
- 但 corrector loop 每次 `solve_one_step` 都从步首 d_ck 重解 → Anderson 状态须每步内重置
  （driver `solve_one_step:392` 已每次新建 `AndersonAccelerator`，天然兼容）。**零代码改动，仅跑时开。**
- 验证：step22–23 iters 由 200(DNF) 降到收敛（model2 经验 ~10–25）。

**1.2 (ii) 中间网格松 tol，仅接受态紧 tol**
- corrector loop 内的「会被丢弃」的整解没必要 1e-4。给 driver 加 `solve_one_step(..., tol_override=)`
  或 runner 临时改 `driver.tol`：corrector 内用 `tol_coarse`（默认 1e-2），**最后一次（无标记，接受态）
  用 `tol=1e-4`**。
- 实现：runner 内层 loop——预判「这次解后大概率还要加密」时用松 tol。最简洁做法：
  corrector loop 全程松 tol 定位标记，**接受后在终网格上补一次紧 tol 终解**再记录。
- 改动点：`run_m3_pc_model1.py` 内层 loop + 加 `FRACTUREX_TOL_COARSE`(1e-2) env；driver 加
  per-call tol 覆盖（`solve_one_step` 读 `self.tol`，加可选参数）。

**1.3 (i) 真 predictor（外推 𝒟 先加密再解）**——**可选，本 Phase 末评估**
- 当前是 corrector-only。真 predictor：用**上一接受步的 𝒟 场**外推/直接复用，载荷步**开头先**对
  `𝒟_prev ≥ β𝒟_c` 单元加密一轮，再进 corrector。把「8 次整解」降到「1 预解雏形 + 1–2 终解」。
- 风险：外推过加密（DOF 浪费）；先用 1.1+1.2 拿到大头提速，**i 视 Phase1 实测墙钟决定是否做**。

**Phase 1 验收**：重跑 `adaptive_m3_pc_model1`（同 nx=24/du=2.5e-4/β=0.6），断言
(a) 无 DNF；(b) 墙钟显著降；(c) 峰值反力/曲线与 v2 一致（提速不改物理）。

---

## Phase 2 — 曲线修复 (iii)：先复核 du，再决定自适应载荷步

**2.1 廉价复核（用户选 A）**
- 跑一次 `du=1e-4`（与参照同）+ Phase1 提速后的 PC，其余不变。
- **判据**：峰值反力是否从 0.648 降向参照 0.631？
  - 若 +2.8% → ≲+1% ⇒ **证实主因是 du**，曲线差解释完毕，可只记录、不必造自适应载荷步。
  - 若仍 +2.8% ⇒ 主因是格式/带分辨，转 2.2 或加严 c_h（h≤l₀/4）。

**2.2 自适应载荷步（条件触发，仅当 2.1 证实 du 是主因且要生产特性）**
- 弹性段 du 放粗（2.5e-4），峰值/软化段自动砍细（如 1e-4 甚至 5e-5）。
- 触发判据：用 𝒟_max 增速或反力曲率——𝒟_max 跨步跳变大 ⇒ 临近起裂 ⇒ 砍 du。
- 改动点：`run_m3_pc_model1.py` 载荷推进改为 `while load<load_max` + 动态 `du`；加
  `FRACTUREX_DU_MIN`/`DU_MAX` env。**与 checkpoint-restore 兼容**（步首 d_ck 逻辑不变）。

**Phase 2 验收**：load-displacement 三曲线图 v3 峰值落进参照 ±1%，更新 RESULTS。

---

## Phase 3 — 严格 Θ 认证 (a) + (iv)(v) 理论升级（用户选 A：本轮做）

把 (a) 从「被动 η 报告」做成「主动最优 AFEM 认证」。

**3.1 非-MMS 连续标准 FEM primal 重解**（新模块/函数）
- 新增 `adaptivity/primal_resolve_real.py::solve_primal_real(discr, case, damage, *, k_res, q)`：
  在**当前接受态网格**上,用**离散 d 场**（`discr.state.d`，逐元重心取 g(d_T)）+ **真实算例
  Dirichlet 边界**（`case` 的 BC，非 MMS 解析 uD）解标准位移 FEM u_h^{cont}。
- 复用 `primal_elastic_solve.py::DegradedElasticMaterial`(g_cell 逐元常数) + `LinearElasticIntegrator`；
  **差异**：源项 f=0（断裂无体力）、边界取 case 的真实 Dirichlet/预裂纹，而非 MMS source/uD。
- 输出 `u_h^{cont}`（连续 P2），喂回 `eta_from_state` 的 `grad_uh` 通道**替换** DG-u
  ⇒ 这才是严格 Θ（残差 r=C_d ε(u_h^cont)−σ_h 用真连续 H¹ primal，消 v2 诚实标注 #1）。

**3.2 接受态认证报告**
- runner 在每个**接受态**（或关键载荷步，env `FRACTUREX_CERTIFY_EVERY=k`）调 3.1 + `eta_from_state`，
  记 η_τ、‖σ_h−σ‖、Θ=η/err 进 history（新列 `eta_tau`,`Theta`）。**可选、不每步**（DECISION §3）。

**3.3 (iv)(v) η_τ-Dörfler 起裂后驱动 + 联合标记**——**理论升级，置于 Discussion/附录**
- (v) 联合标记：`refine if 𝒟-marked OR η_τ-marked`——𝒟 控 Γ-收敛分辨误差、η_τ 控弹性离散误差
  （DECISION §7.3 互补覆盖）。改 runner 标记掩码为两者并集。
- (iv) 起裂后切 η_τ-Dörfler：d>0 区 η_τ 有 rate-optimality（CKNS AFEM 理论），起裂前仍用 𝒟
  （η_τ 标不了起裂，DECISION §2 D2）。**作对照实验**：纯 𝒟 vs 𝒟∨η_τ vs 起裂后 η_τ-Dörfler，
  比等精度 DOF 效率。
- 风险：η_τ 每步要 primal 重解（贵）⇒ 仅在认证步开；(iv)(v) 优先级低于 (a) 报告本身。

**Phase 3 验收**：接受态 Θ≈1（严格版，对账 T6 的 Θ→1 在真实裂纹上是否仍成立）；
认证报告进 RESULTS §M3-strict；(iv)(v) 作 Discussion 数据。

---

## 执行顺序与依赖

```
Phase 1 (提速 b/i/ii) ──┐
                        ├─→ Phase 2 (du 复核 → 自适应载荷步)  [依赖 P1 的快]
Phase 1 ────────────────┘
Phase 3 (严格 Θ 认证) ── 可与 P2 并行（独立模块 primal_resolve_real）
```

- **先 Phase 1**：提速是其余一切的前提（7h/run 迭代太慢）。
- **Phase 2 紧随**：一次 du=1e-4 复核即可能闭合曲线差（最便宜的科学问题先问）。
- **Phase 3 并行**：`primal_resolve_real.py` 是独立新模块，不阻塞 P1/P2。

## 改动文件清单

| 文件 | 改动 | Phase |
|---|---|---|
| `run_m3_pc_model1.py` | corrector 松 tol + 接受态紧终解；（条件）自适应 du；调认证 | 1,2,3 |
| `drivers/huzhang_phasefield_staggered.py` | `solve_one_step` 加可选 `tol`/`maxit` 覆盖 | 1 |
| `adaptivity/primal_resolve_real.py`（新） | 非-MMS 真实算例连续 primal 重解 | 3 |
| `adaptivity/adaptive_staggered.py` | （3.3）联合标记 `mark_union(𝒟, η_τ)` | 3 |
| `tests/aposteriori/plot_m3_pc_model1.py` | 加 Θ/η_τ 认证子图；v3 曲线 | 2,3 |
| `tests/aposteriori/test_*.py`（新） | 严格 Θ 单测、联合标记单测 | 3 |
| `RESULTS_aposteriori.md` | M3 PC v3 + M3-strict 记录 | 全 |

## 诚实边界（须保留）
- Anderson 在**局部化极限环**外加速显著，临界点仍可能需更深 depth（model2 经验）；DNF 若残留，
  记录而非隐藏（[[aux_niter_localization_degradation]] 措辞精确）。
- 严格 Θ 用真实算例后，T6 的 Θ→1「偏容易」（诚实标注 #1：σ_h 远准于 u_h）可能让真实裂纹上
  Θ 仍≈1——若如此，需如 T6 加固改报 ratio=‖σ_h−σ‖/err 灵敏探针。
- (iv) η_τ-Dörfler 的 rate-optimality 是连续 AFEM 理论，**相场+staggered 下不直接套**——只作
  数值对照，不声称已证最优性。
