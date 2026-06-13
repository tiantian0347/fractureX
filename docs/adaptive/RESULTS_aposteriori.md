# 实验记录：平衡型 a posteriori 估计 + 自适应

> 配合 [DESIGN_program_and_tests.md](DESIGN_program_and_tests.md) §3 记录规范。
> 每个 T*/M* 跑完即追加一节：配置 + 指标表（精度+速度）+ 诚实标注 + 图路径。
> 负结果同样记录。环境：py312 + PYTHONPATH（memory `fealpy_env_py312`）。

## 索引

| 实验 | 日期 | 状态 | 一句话结论 |
|------|------|------|-----------|
| T0   | 2026-06-13 | ✅ PASS | 柔度核 A(d)C_d=Id round-trip 机器精度；零残差 ⇒ η=0 |
| T1a  | 2026-06-13 | ✅ PASS | Prager–Synge 交叉项正交 + 超圆恒等式机器精度（解析场，真实网格） |
| T5   | 2026-06-13 | ✅ PASS | Amor 对偶势闭式 (14) 验机器精度；**订正了 v0.2 推导的系数错误** |

---

## T5: Amor 分裂对偶势闭式

- **日期**：2026-06-13
- **目的**：坐实 Theorem 2 的 Amor 对偶势 (14)（THEORY §6.3）。
- **⚠️ 关键修正**：T5 跑出 v0.1/v0.2 的 (14) 式**系数错误**——迹项写成 1/(2K)，
  brute-force Legendre 对不上（rel 0.1~0.5）。根因：应力–应变配对
  τ:ε=½ tr τ·tr ε + dev τ:dev ε，迹通道对偶变量是 **½ tr τ 而非 tr τ**，
  Legendre 共轭带出 1/4 因子 ⇒ 迹项应为 **1/(8K)**。订正后机器精度匹配。
  THEORY §6.3 已同步更正（含推导）。
- **配置**：后端 numpy；K=λ+μ=201.92（2D 平面应变）；扫 g∈{1,0.37,1e-3}。
- **结果**：

  | 子项 | 判据 | 实测 |
  |------|------|------|
  | ψ* 闭式 vs brute-force Legendre | rel<1e-9 | 1.0e-9（受限于 Nelder-Mead 优化精度，非闭式误差） |
  | Fenchel–Young ψ+ψ*-τ:ε ≥ 0 逐点 | gap≥-1e-9 | ✅ |
  | τ=∂_ε ψ ⇒ gap=0 | \|gap\|<1e-7 | ✅ |

- **诚实标注**：(a) 的 1e-9 是 brute-force 优化器精度极限；闭式本身（独立脚本直接代数核对）
  达 8.5e-16。effectivity 的强凸 k 依赖（majorant 偏松）未在此测——属 T6 范畴。
- **文件**：`fracturex/tests/aposteriori/test_amor_dual_closed_form.py`；
  模块新增 `amor_energy / amor_stress / amor_dual_energy`。
- **结论**：Theorem 2 的对偶势闭式订正并验证；分裂情形 majorant 可逐元算。
  **教训**：理论推导的闭式必须数值反验（brute-force）才能写进论文——本次正是数值抓出了系数错。

---

## T1a: Prager–Synge 恒等式核心正交性

- **日期**：2026-06-13
- **目的**：坐实 Thm 1 的分析心脏——交叉项 ∫ε(w):δ=0（w|∂Ω=0, div δ=0）⇒ 超圆恒等式（THEORY (7)）。
- **方法**：解析构造运动学场 w=(sin πx sin πy,0)（边界为零）+ Airy 自平衡应力 δ（div δ=0），
  d≡0、纯 Dirichlet（Γ_N=∅）；真实三角网格 8×8 上高阶求积 q=10。**不解 PDE**。
- **结果**：

  | 子项 | 判据 | 实测 |
  |------|------|------|
  | ∫ ε(w):δ = 0 | <1e-10 | 0.000 ✅ |
  | ‖a‖²_A+‖b‖²_A = ‖a-b‖²_A | rel<1e-10 | 0.00 ✅ |

- **诚实标注**：用解析场，未涉离散误差（那是 T2 的事）。但 (b) 非平凡：验证了
  ⟨Cε(w),C⁻¹δ⟩=∫ε(w):δ，即 estimator 的 Voigt 内积与柔度算子在加权内积下自洽——
  这是 η 计算正确的前提。
- **文件**：`fracturex/tests/aposteriori/test_prager_synge_identity.py`
- **结论**：超圆恒等式核在真实网格上成立；估计子代数与定理一致。下一步 T2 需标准 FEM 求解器
  （`primal_elastic_solve.py`）造真离散误差，量 Θ→1 收敛阶。

---

## T0: 柔度 / 退化权一致性

- **日期**：2026-06-13
- **目的**：验 `equilibrated_estimator` 代数核（THEORY (2)）；不依赖 FE 空间。
- **配置**：后端 numpy；Lamé λ=121.15, μ=80.77（平面应变）；扫 d∈{0,.3,.7,.99,1}, k_res∈{1e-3,1e-7}。
- **结果**：

  | 子项 | 判据 | 实测 |
  |------|------|------|
  | C⁻¹C = Id | rel<1e-12 | 1.34e-16 ✅ |
  | A(d)C_d = Id | rel<1e-12 | <1e-12 全通 ✅ |
  | σ_h=C_d ε(u_h) ⇒ η=0 | η<1e-12 | 0.00 ✅ |
  | g>0 & k_res≤0 抛错 | ValueError | ✅ |

- **诚实标注**：纯代数层，未涉 FE 离散/积分误差；FE 求值正确性留 T1。
- **文件**：`fracturex/tests/aposteriori/test_compliance_consistency.py`
- **结论**：估计子代数核正确，可作为 T1（MMS）地基。

---

<!-- 模板：复制下面一节填写
## T?: <标题>

- **日期**：YYYY-MM-DD
- **目的**：（对应理论/DESIGN 哪一条）
- **配置**：hmin=…, p=…, k_res=…, theta=…, q=…, 后端=…
- **结果**：

  | 量 | 值 | 备注 |
  |----|----|----|
  | η | | |
  | 真误差 ‖ε(uh)-ε(u)‖_{C_d} | | |
  | Θ | | |
  | DOF | | |
  | 墙钟(s) | | |

- **诚实标注**：（数据质量/未收敛/欠分辨/负结果）
- **图**：docs/figures/adaptive/…
- **结论 / 是否触发门槛**：
-->
