# 程序设计与测试方案：平衡型 a posteriori 估计 + 自适应

> 状态：设计稿 v0.1（2026-06-13）。配合
> [plan_adaptive_aposteriori.md](../routes/plan_adaptive_aposteriori.md)（路线）与
> [../adaptive/THEORY_equilibrated_aposteriori.md](../adaptive/THEORY_equilibrated_aposteriori.md)（理论）。
> 原则：**每个模块都配测试，每个测试都进文档**（见 memory `record_experiments_in_docs`、
> `fracturex_code_comment_style`、`fracturex_multibackend_convention`、`dont_modify_fealpy`）。

## 0. 复用与新增盘点（先摸清，避免重造）

**已有可复用**
- `fracturex/assemblers/huzhang_elastic_assembler.py::HuZhangElasticAssembler` → 出平衡应力 $\sigma_h$。
- `fracturex/utilfuc/recover_strain.py::recover_strain_from_sigma` → 退化柔度 $\mathbb A(d)$ 应用（算 $\eta$ 用）。
- `fracturex/adaptivity/adaptive_refinement.py::AdaptiveRefinement` → 已有 Dörfler `Mesh.mark(eta,theta)`
  + bisect 加密骨架；`residual` 策略是 **stub**，正好挂我们的 `equilibrated`。
- `fracturex/tests/linear_elastic_with_huzhang.py` → 已有 MMS（sympy 造解）+ σ 的 L² 误差，M0 直接借。
- `fracturex/phasefield/main_solve.py` → 标准 FEM 弹性路径（取 $u_h$）。

**需新增（接 fealpy 接口，不动 fealpy）**
1. `fracturex/adaptivity/equilibrated_estimator.py` — 核心：$\eta_T$ 计算。
2. `fracturex/adaptivity/primal_elastic_solve.py` — 标准连续 Lagrange 弹性解 $u_h$（若 main_solve 不能直接复用）。
3. `AdaptiveRefinement` 增 `marking_strategy='equilibrated'` 分支。
4. M0 概念验证驱动脚本 `fracturex/tests/aposteriori/poc_effectivity.py`。

---

## 1. 模块设计

### 1.1 `equilibrated_estimator.py`（核心，最先写）

**职责**：给定 $(u_h,\sigma_h,d,\mathbb C)$ 算逐元 $\eta_T$ 与全局 $\eta$、以及（验证用）真误差。

```
def equilibrated_indicator(mesh, uh, sigmah, d_field, lam, mu, *, k_res, q=...) -> dict:
    """逐元平衡型误差指示子 η_T（THEORY (5)）。
    输入:
      mesh    : 三角网格
      uh      : 标准 FEM 位移 (连续 Lagrange FE function)
      sigmah  : Hu-Zhang 平衡应力 (FE function, 逐点对称, 出自 HuZhangElasticAssembler)
      d_field : 相场/损伤, qp 上可求值 (冻结系数)
      lam,mu  : Lamé 参数
      k_res   : 残余刚度 (g=(1-d)^2+k_res, 严格 >0)
      q       : 积分阶 (默认高于 uh,sigmah 阶以免欠积分污染 η)
    返回 dict:
      eta_T   : (NC,) 逐元指示子 (np/bm array)
      eta     : 标量 全局 = sqrt(sum eta_T^2)
    数学: η_T^2 = ∫_T g^{-1} (C_d ε(uh) - σ_h) : C^{-1} (C_d ε(uh) - σ_h)
          其中 C_d ε(uh) = g·(λ tr(ε)I + 2μ ε), g=(1-d)^2+k_res
    """
```

- **被积量分解**（数值稳定）：令 $r:=\mathbb C_d\varepsilon(u_h)-\sigma_h$，
  $\eta_T^2=\int_T g^{-1}\,r:\mathbb C^{-1}r$，$\mathbb C^{-1}$ 用 `recover_strain.py::_compliance_apply` 同款公式。
- **qp 求值**：$\varepsilon(u_h)$ 由 `uh.grad_value`；$\sigma_h$ 由 Hu–Zhang 空间在同一 bcs 求值；$d$ 在 qp 插值。
- **多后端**：算子用 `bm`（`bm.einsum`/`bm.index_add`），numpy 仅限文件 I/O（遵 `fracturex_multibackend_convention`）。

附 **effectivity 工具**（仅 M0 验证用，不进生产）：
```
def effectivity_index(eta, energy_err) -> float:   # Θ = η / ||ε(uh)-ε(u)||_{C_d}
def prager_synge_residual(eta, energy_err, stress_err) -> float:  # 验 (7) 等式两端机器精度
```

### 1.2 `primal_elastic_solve.py`（标准 FEM 位移）

- 连续 Lagrange 向量空间 + `LinearElasticIntegrator`（fealpy），系数 $g(d)\mathbb C$。
- 接口与 Hu–Zhang 装配同源边界条件，保证两个解在**同一 BC** 下可比。
- **备选（§概念验证二选一）**：Stenberg 后处理从 Hu–Zhang 混合位移重构协调 $u_h$，省掉独立解。
  M0 阶段先做独立标准 FEM（最直接），后处理留 M1 对比。

> **实现进度 2026-06-13**：`solve_primal` 已写完并验收敛阶——MMS（p=2，u=sinπx·sinπy）
> 在 N=4,8,16,32 上 L2 收敛率 2.94→2.99→2.99 = $O(h^{p+1})$ 最优阶 ✅。常系数 g 路径就绪；
> 变 g(d) 注入留 T6。

> **σ_h 来源决策（重要）**：Hu–Zhang 空间的 `interpolate`/`project` 是**空 stub（未实现）**，
> 不能靠典范插值直接拿平衡应力。**改用 `tests/linear_elastic_with_huzhang.py::solve` 的完整混合
> 系统求解**（已验 σ 收敛阶、含 HuzhangBoundaryCondition/HuzhangStressBoundaryCondition 边界处理），
> 它直接 return `(sigmah, uh)`。T6 把它包装成可调接口，不自己实现 L2 投影（避免引入未验证的投影误差）。

### 1.3 `AdaptiveRefinement` 扩展

```
marking_strategy='equilibrated':
    eta_T = equilibrated_indicator(...)['eta_T']
    marked = Mesh.mark(eta=eta_T, theta=self.theta)   # 复用已有 Dörfler
```
- 加密后**重装配** Hu–Zhang + 标准 FEM，进下一轮（M2）。
- 与损伤 $d$ 耦合标记（纯 $\eta_T$ vs $\eta_T$+d-梯度）做一次对比开关。

---

## 2. 测试矩阵（每个模块对应测试，全部进文档）

测试目录统一 `fracturex/tests/aposteriori/`。运行环境 py312 + PYTHONPATH（memory `fealpy_env_py312`）。

| ID | 测试文件 | 测什么 | 通过判据 | 对应理论 |
|----|---------|--------|---------|---------|
| T0 | `test_compliance_consistency.py` | $\mathbb C^{-1}$ 应用 + $g^{-1}$ 权正确 | $\mathbb A(d)\mathbb C_d=\mathrm{Id}$ 机器精度 | §1 (2) |
| T1 | `test_estimator_smooth_mms.py` | 光滑 MMS（$d\equiv0$,$f=0$）$\eta_T$ 正确 | Prager–Synge 等式 (7) 两端 rel<1e-10；$\Theta\to1$ | Thm 1, §5 第1层 |
| T2 | `test_estimator_convergence.py` | $\eta$ 随 $h$ 的收敛阶 | $\eta=O(h^{p+1})$，与真误差同阶 | §5 充要条件 |
| T3 | `test_reliability_upper_bound.py` | 可靠性：$\eta\ge$ 真误差 | 多网格/多 load 恒 $\eta\ge\|err\|$（常数=1） | Thm 1 (4) |
| T4 | `test_osc_terms.py` | $f\ne0$/$t_N\ne0$ 时 osc 项 | osc 随 $h$ 高阶衰减；界仍上界 | §4 (6) |
| T5 | `test_amor_dual_closed_form.py` | Amor 对偶势 (14) | $\psi+\psi^\*-\varepsilon:\tau\ge0$ 逐点；取等于真应力 | Thm 2 §6.3 |
| T6 | `test_degradation_effectivity.py` | **扫 $k_{\mathrm{res}}$ 的 $\Theta$**（核心门槛） | $\Theta$ 不随 $k_{\mathrm{res}}\to0$ 发散 | §5 第3层 |
| T7 | `test_marking_refines_crack_band.py` | $\eta_T$ 标记位置 | 高 $\eta_T$ 落在裂纹带/高梯度区 | §5 实践校正 |
| T8 | `test_adaptive_loop_dof_efficiency.py` | 自适应 vs 均匀 | 等精度下自适应 DOF/墙钟更优 | M2/M3 |

**测试编写规范**
- 每个测试**自带解析参照或机器精度对账**，不靠跑出来的数当真值（避免 memory `model2_paper_direct_bogus` 式假数据）。
- T1/T6 用固定解析 $d$ 场（如 $d(x)=\exp(-\mathrm{dist}(x,\text{crack})^2/\ell^2)$），不依赖相场子问题收敛。
- 跑相场耦合的（T7/T8）先过 sanity check（`phasefield_sanity_check`：`max_H>0`+非零反力）。
- 粗网格陷阱：`auxspace_test_mesh_sensitivity` 提醒——effectivity 别在 hmin≥0.06 判定，默认 hmin=0.02。

---

## 3. 实验记录规范（强制）

每跑一个 T*/M*，在 `docs/adaptive/RESULTS_aposteriori.md` 追加一节：
- 配置（网格 hmin、$p$、$k_{\mathrm{res}}$、$\theta$、积分阶 $q$）。
- 指标表（$\eta$、真误差、$\Theta$、DOF、墙钟）——同时给精度和速度（memory `code_goal_fast_and_accurate`）。
- 诚实标注（数据质量问题、未收敛、欠分辨）。
- 图存 `docs/figures/adaptive/`。
- 关键负结果同样记录（如某 $k_{\mathrm{res}}$ 下 $\Theta$ 发散 → 触发门槛的证据）。

---

## 4. 实施顺序（与 plan 的 M0–M3 对齐）

1. **T0 → 1.1 estimator → T1 → T3**（M0 地基：估计子对 + 可靠性）。
2. **1.2 primal solve → T2**（两解阶相容）。
3. **T6**（核心门槛；过不了就触发 plan §4 退路）。
4. **T5 + Amor majorant**（分裂情形）。
5. **1.3 + T7 → T8**（自适应循环，M2/M3）。

> 每步完成即更新 `RESULTS_aposteriori.md` 与本文件「实施进度」表，并在 memory
> `paper_adaptive_aposteriori_direction` 留一行指针。

## 5. 实施进度（滚动更新）

| 日期 | 模块/测试 | 状态 | 结果摘要 | 记录位置 |
|------|----------|------|---------|---------|
| 2026-06-13 | 设计稿 | ✅ | 本文件 v0.1 | — |
| 2026-06-13 | `equilibrated_estimator.py` | ✅ 写完 | 核心 η_T + Voigt 辅助 + effectivity 工具 | 模块就位 |
| 2026-06-13 | T0 一致性 | ✅ PASS | A(d)C_d=Id 机器精度；零残差 η=0 | RESULTS §T0 |
| 2026-06-13 | T1a Prager–Synge 核 | ✅ PASS | 交叉项正交+超圆恒等式机器精度（解析场） | RESULTS §T1a |
| 2026-06-13 | Amor 分裂函数 | ✅ 写完 | `amor_energy/amor_stress/amor_dual_energy` | 模块就位 |
| 2026-06-13 | T5 Amor 对偶势 | ✅ PASS | 闭式验机器精度；**订正 (14) 系数 1/2K→1/8K** | RESULTS §T5 |
| 2026-06-13 | `primal_elastic_solve.py` | ✅ 写完+验 | MMS p=2 收敛率→3.0 最优阶 | DESIGN §1.2 |
| 2026-06-13 | `degraded_mms.py` | ✅ 写完+验 | 退化 MMS+裂纹带 d(x)，自洽 2e-16 | T6 积木 |
| 2026-06-13 | `degraded_huzhang_solve.py` | ✅ 写完+验 | g⁻¹ 注入柔度块出平衡 σ_h (p=3) | T6 积木 |
| 2026-06-13 | `solve_primal_degraded` | ✅ 写完 | 逐元常数 g(d_T) 退化标准 FEM | T6 积木 |
| 2026-06-13 | T2/T3/T6 门槛 | ✅ **GO** | Θ≈1 不随 k_res→0 发散；可靠性=1 | RESULTS §T2/T3/T6 |
| 2026-06-13 | interp 四种场转移 | ✅ PASS | H/d/u可转移(u走IM)，σ须重解；M2只转移H/d | RESULTS §interp |
| 2026-06-13 | M2 自适应循环 + T8 | ✅ PASS | η单调降、Θ≈1；等精度省70% DOF | RESULTS §M2/T8 |
| 2026-06-13 | M3a 冻结真实裂纹带 | ✅ PASS | η_T 标记100%集中真实带(model2几何) | RESULTS §M3a |
| 2026-06-13 | T4 osc 衰减 | ✅ PASS | osc(f) rate≈4 高阶小量；osc(t_N) 留 M3 混合加载 | RESULTS §T4 |
| 2026-06-13 | **M3 full (model1)** | ✅ PASS | η_T 自适应加密耦合真实 staggered，跑到裂纹贯通失效(峰值反力0.734后陡降)；网格跟尖 NC 1152→1703；内存峰值 1.09 GB | RESULTS §M3 full |
| 2026-06-13 | **正确性对账(model1)** | ⚠️ 部分 | vs 均匀 nx=120 参照：弹性刚度匹配5%，但峰值反力高估+16%(带 h/l₀≈0.70 欠分辨)；触发标记策略重设计 | RESULTS §正确性对账 |
| 2026-06-13 | **标记策略理论(重设计)** | ✅ 推导完 | 应力驱动预测型标记+predictor-corrector；1D AT2 标定 𝒟_c=1/3/σ_c/剖面e^{-\|x\|/l₀}(sympy核验)；命题1领先性/命题2标记可靠/命题3-4终止+分辨保证 | THEORY_marking_strategy.md |
| 2026-06-13 | **M-DF模块+单元测试** | ✅ PASS | driving_force_per_cell/mark_driving_force/refine_masked；𝒟公式机器精度+𝒟_c=1/3标定+阈值/尺寸下限掩码逐元对账 | test_marking_driving_force.py |
| 2026-06-13 | **predictor-corrector冒烟** | ✅ PASS | run_m3_pc_model1.py；步内反复加密终止(corr 7→1)，裂尖加密到 h≤l₀/2(nc1152→1724/step1)，checkpoint-restore正确，反力对齐参照 | run_m3_pc_model1.py |
