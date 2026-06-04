# D12 — 论文就绪数值结果（§5 写作直接用）

> 本文件汇总 D12 预条件子论文 **§5 数值实验**的全部结果（表 + 图 + 可直接转写的结论），按论文小节组织。规划/状态见 `D12_PRECONDITIONER_PAPER_PLAN.md`（§13.5 清单、§3.2 理论）。
> 定位（专家）：卖**迭代稳定性 + 小内存**，不卖墙钟。所有图在 `docs/figures/precond/`。
> 生成日期 2026-06-03；数据 `results/phasefield/_iter_stability/{iter_stability,l0_sweep,mem_scaling}.csv` + `spec_auxfast_d*.npz`。

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
- **⚠️ 重要诚实标注（真实裂纹局部化 vs 合成均匀-d）**：上述受控基准（§5.2/§5.4）用**均匀** d 扫条件数，niter 始终 O(10)（≤18，even d=0.999）。但在真实相场 run 中观测到：aux_fast niter 在 **maxd≤0.82 时恒 =7**，一旦裂纹**完全局部化**（尖锐 d≈1/d≈0 界面形成，maxd→0.997）**骤升到 ~95–107**（约 14×，t_solve 13s→150–540s；h₂ 实测，`paper_aux_scan_auxfast_h2/iterations.csv`）。即**均匀-d 基准系统性低估了尖锐裂纹界面下的迭代难度**——合成基准的 P1 加权粗空间对均匀退化良好，但对局部化尖界面变难。**对论文论点的影响**：aux 仍**有界收敛（O(100)）且仍是唯一可行**（none/Jacobi/ILU 在该状态全 DNF），但 §7 "O(10) mesh-independent" 须明确限定为**受控均匀-d 基准**，并补一句真实局部化下升到 O(100) 但保持收敛、对手全失效。**不可只报合成 O(10) 而不提真实裂纹的 O(100)。**

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
| 图2 (C3) | `iter_stability_vs_d` | niter vs 损伤 d（aux 平 vs others DNF/erratic）|
| 图3 | `mem_scaling` | peak RSS vs σ-DOF（mf vs direct + OOM）|
| 图4 | `spectrum_scatter` / `spectrum_kappa_vs_d` | P⁻¹K 谱 vs d |
| 图5 (D2) | `iter_stability_square_vs_d` | square(I型) niter vs d（aux 有界 vs others DNF）|
| 图6 (§7.9) | `hz_vs_lagrange_v2`（在 `Frac_huzhang/figures/`）| σ 收敛阶 HuZhang 4 阶 vs Lagrange 2 阶 |
| 图7 (D3) | `iter_stability_model2_vs_d` | model2(II型剪切) niter vs d（aux 有界 vs others DNF）|
| 构造示意 (§6) | `precond_schematic`（`make_precond_schematic.py`）✅ | 工程流程版：块三角 sweep + 两层 V-cycle（大字号 + matrix-free 高亮 + sweep 箭头）；已并入 tex `fig:precond_schematic`。数学/算子版 `make_precond_operator_diagram.py` 已弃用 |

## 待补

> **完整清单见 `D12_PRECONDITIONER_PAPER_PLAN.md` §13.6**（2026-06-03 成稿后复盘，按 tex `\needexp` 逐项盘点，分 G/H/I/J 四组 + 更新 MVP）。下面是高频项摘要：

- **G1（P0，已声称必验）**：论文 §6 提了块对角+MINRES 但 §7 全用 block-tri/GMRES → 补一组 MINRES niter 或改写措辞。
- **G2（C5 只留 V2）**：HuZhang vs Lagrange 制造解 σ L² 收敛阶（smoke 已 4.0 vs 2.0，跑 paper 分辨率扫即可，近免费）。**V5 尖端切片已删**（偏主线+昂贵），tex 已收口。
- ~~**H1（P0）**~~ ✅ **完成**：C2 招牌图 aux d=0.9 已补满 h₃–h₅，跨 187× DOF = 9/15/9/8/9（见 §5.2 表1/图1）。
- **H2/H3**：谱多网格叠加验 mesh-independence；eps_g 轴扫验退化下界鲁棒。
- **I（广度）**：~~D2 square (I型)~~✅(§5.7)、~~D3 model2 (II型剪切)~~✅(§5.8 表8a/8b+图7)、D4 effective_stress（附录 A）；CNT aux 跑过软化段、SENT 全分离刷新、SENS aux/direct（§5 三处 `\needexp`）。
- **J（加分）**：Anderson 外层加速量化、matrix-free niter 跨档、§3 构造示意图。
