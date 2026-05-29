# 实验矩阵：Hu-Zhang 元 + 辅助空间预条件子相场断裂论文

> 配套论文：`~/tian/Frac_huzhang/phasefield_huzhang.tex`
> 目标期刊：**CMAME**（首选）/ JCP / IJNME（备选）
> 目标投稿时间：6 个月内出初稿
> 文档维护：所有实验跑完都来更新这张表的"状态"列

---

## 1. 论文核心 Claim

| ID | Claim | 怎么证明 | 主要图表 |
|----|-------|----------|----------|
| **C1** | aux-precond 求解的物理解与 direct 法一致；与文献吻合 | 三 case 的 load-displacement 曲线重合；裂纹路径定性一致；model1 与 Miehe (2010) / Borden (2012) 对照 | Fig. load-disp × 3；Fig. crack pattern × 3 |
| **C2** | aux-space 预条件子让 GMRES 迭代数**几乎与 N 无关**（核心招牌图） | model0 五档网格扫描下 GMRES iter vs N 平坦曲线 | **Fig. iter-vs-N（论文核心图）** |
| **C3** | 整个损伤演化过程中预条件子稳定（弹性段→起裂→软化→贯穿） | 三 case 各自 GMRES iter 随载荷步的曲线（含 staggered iter 热力图） | Fig. iter-vs-step × 3 |
| **C4** | 相比 direct 法（spsolve / pardiso）有实用的时间 / 内存优势；direct 在大 N 下 OOM 时 aux 仍可用 | model0 五档网格下 wall-time 与 peak RSS 对比表 + scaling 曲线 | Table efficiency；Fig. time-vs-N |
| **C5** | Hu-Zhang 元相对位移型 Lagrange 元在断裂问题上有结构性优势 | **V2**：弹性段 σ 的 h-收敛阶高于 Lagrange；**V5**：裂纹尖端 σ 探针对比 | Fig. σ-conv（V2）；Fig. tip-stress（V5） |

C5 当前在 tex 初稿里**完全没有体现**，需要新增数值实验小节。

---

## 2. 实验矩阵主表

| 案例 | 模式 | 网格档 | 用途（对应 Claim） | 优先级 | 状态 |
|------|------|-------|----------|--------|------|
| **model0**（circular notch） | baseline | 极细，1.9M σ DOF | reference 解；C1 / C5 V2 | P0 | ✅ done (`paper_baseline/`) |
| model0 | direct | h₁ ~ 7.8e-2（粗） | C1 + C2 + C4 | P0 | ✅ done (`paper_direct/`) |
| model0 | aux | h₁ ~ 7.8e-2（粗） | C1 + C2 + C3 + C4 | P0 | 🟡 in progress (PID 3441015) |
| model0 | direct | h₂ ~ 4e-2 | C2 + C4 | P0 | ⬜ todo |
| model0 | aux | h₂ ~ 4e-2 | C2 + C4 | P0 | ⬜ todo |
| model0 | direct | h₃ ~ 2e-2 | C2 + C4 | P0 | ⬜ todo |
| model0 | aux | h₃ ~ 2e-2 | C2 + C4 | P0 | ⬜ todo |
| model0 | direct | h₄ ~ 1e-2 | C2 + C4（direct 临界） | P1 | ⬜ todo |
| model0 | aux | h₄ ~ 1e-2 | C2 + C4 | P0 | ⬜ todo |
| model0 | aux | h₅ ~ 5e-3 | C2（aux 高 N 验证） | P0 | ⬜ todo |
| model0 | direct | h₅ ~ 5e-3 | C4（看是否 OOM） | P2 | ⬜ todo |
| **model0 Lagrange 对照** | — | h₂ | C5 V2（σ 收敛阶） | P1 | ⬜ todo |
| **model0 Lagrange 对照** | — | h₃ | C5 V2 | P1 | ⬜ todo |
| **model1 / square** | direct | h_target≈l₀/2 | C1（含 Miehe 文献对照） | P0 | 🔴 之前没收敛，重启 maxit=500 |
| model1 / square | aux | 同上 | C1 + C3 | P0 | ⬜ todo |
| model1 Lagrange 对照 | — | 同上 | C5 V5（尖端 σ 探针） | P1 | ⬜ todo |
| **model2**（notch X stretch） | direct | h_target≈l₀/2 | C1 + C3（剪切场景） | P0 | ✅ done |
| model2 | aux | 同上 | C3（剪切场景预条件子稳定性） | P0 | ⬜ todo |

**优先级说明**：
- **P0**：论文必须有；不做或做不出来，论文站不住。
- **P1**：论文应该有；缺了会被审稿人质疑，但不致命。
- **P2**：补充材料 / future work；做到不做都可发。

---

## 3. 网格档定义（model0 作为 N-scaling 主案例）

model0 用 distmesh，`hmin` 是控制参数。从已有的 `paper_baseline` 看，hmin≈0.004 → h_max≈0.0082 → ~1.9M σ DOF。粗端从 `paper_main` 看，hmin≈0.05 → ~10K σ DOF。

| 档位 | hmin | 预估 h_max | σ DOF（≈） | u DOF（≈） | d DOF（≈） | 用途 |
|------|------|-----------|------------|------------|-----------|------|
| h₁ | 0.05 | 7.8e-2 | 10K | 7K | 1.4K | 小算例热身 + 调流程 |
| h₂ | 0.025 | 4e-2 | 40K | 30K | 5K | 中等，direct 可承受 |
| h₃ | 0.013 | 2e-2 | 160K | 120K | 20K | direct 仍可，aux 显示优势 |
| h₄ | 0.0065 | 1e-2 | 640K | 480K | 80K | direct 临界，aux 必须 |
| h₅ | 0.004 | 5e-3 | 1.9M | 1.4M | 230K | 招牌图压轴 |

**判据**：在每一档上 `h_max < l₀ / 2 = 0.01`，从 h₃ 开始才严格满足相场分辨率要求。h₁、h₂ 用于显示求解器对 under-resolved 的鲁棒性。

---

## 4. 每次实验必须记录的指标

凡是新跑的实验，`RunRecorder` 输出**至少**要包含：

| 字段 | 类型 | 来源 | 是否已记录 |
|------|------|------|----------|
| `wall_time_elastic_solve_per_iter` | per staggered iter | driver | ❓ 需验证 |
| `wall_time_phase_solve_per_iter` | per staggered iter | driver | ❓ 需验证 |
| `wall_time_assembly_per_iter` | per staggered iter | assembler | ❓ 需验证 |
| `n_gmres_iter_elastic` | per staggered iter | solver | ❓ 需验证 |
| `n_gmres_iter_phase` | per staggered iter | solver | ❓ 需验证 |
| `gmres_residual_history` | per staggered iter | solver | ❓ 可选 |
| `peak_rss_mb` | per load step | psutil | ❌ **没有，需要加** |
| `max_d` | per staggered iter | state | ✓ 已有（log 可见） |
| `error / err_u / err_d` | per staggered iter | driver | ✓ 已有 |
| `reaction_force` | per load step | postprocess | ✓ 已有 |

**P1 内必须把 `peak_rss_mb` 接入**（5 行 psutil 代码），否则 C4 表没法画。

---

## 5. 数据后处理脚本清单

| 脚本 | 用途 | 状态 |
|------|------|------|
| `collect_paper_bundle.py` | 汇总三 case × 三模式 metadata 到 PAPER_INDEX | ✓ 已有 |
| `paper_make_load_disp.py` | 生成三 case 的 load-displacement 曲线对比图 | ❌ 待编写 |
| `paper_make_iter_vs_N.py` | 生成 C2 核心图：GMRES iter vs N（aux/direct 两条曲线） | ❌ 待编写 |
| `paper_make_iter_heatmap.py` | C3 热力图：staggered iter × GMRES iter × load step | ❌ 待编写 |
| `paper_make_efficiency_table.py` | C4 效率表：wall-time、RSS、scaling | ❌ 待编写 |
| `paper_make_sigma_conv.py` | C5 V2：σ 收敛阶（HuZhang vs Lagrange） | ❌ 待编写 |
| `paper_make_tip_stress.py` | C5 V5：裂纹尖端 σ 探针对比 | ❌ 待编写 |

---

## 6. 案例物理参数速查（避免改了不知道）

| 参数 | model0 | model1 (square) | model2 |
|------|--------|------------|--------|
| E | 200 | 210 | 210 |
| ν | 0.2 | 0.3 | 0.3 |
| Gc | 1.0 | 2.7e-3 | 2.7e-3 |
| l₀ | 0.02 | 0.015 | 0.0133 |
| h_target = l₀/2 | 0.010 | 0.0075 | 0.0067 |
| 加载步数（默认） | 31 | 161 | 由 `case.default_loads()` 决定 |
| 几何 | distmesh 圆缺口 | box + 预裂纹（Miehe） | box + 缺口剪切 |
| 几何来源 | Miehe-type radial pre-notch | Miehe (2010) tension | Miehe (2010) shear |

**固定离散参数**（不要在论文里乱扫描）：
- `p = 3` （应力空间次数，Hu-Zhang）
- `damage_p = 2`
- `AT2 + quadratic degradation + hybrid split`
- 退化系数基准：`g(d)` 直接进入应力块 `1/g` 与 P1 粗扩散 `g`（两侧同源，不在辅助子里再叠 `max(g, ε_g)` floor；`ε_g` 仅作为 `damage.coef_bary` 内部数值下界，不进入论文措辞）
- `formulation = "standard"`

允许扫描的（消融实验区）：
- 网格档（h₁–h₅）
- `formulation = effective_stress`（只在 model0 上扫一次，C3 补充表）

---

## 7. 时间表（180 天，对齐 P1–P5）

| 阶段 | 周次 | 关键节点 |
|------|------|----------|
| P1 止血 | Week 1–2 | 完成 model0_aux；启动 square direct + aux；接入 RSS；写本表初版 |
| P2 招牌图 | Week 3–8 | 跑完 model0 h₁–h₅ 全部 direct + aux；生成 C2 图初版；草稿 intro + algorithm 章节 |
| P3 物理 + C5 | Week 9–14 | model1 + model2 aux 全部完成；Lagrange 对照在 model0/model1 上跑完；C5 V2+V5 图完成；草稿 70% |
| P4 完稿 | Week 15–22 | C4 内存采样补完；完整草稿；内部 2 轮 review；定稿 |
| P5 投稿 | Week 23–26 | arXiv 挂稿；CMAME 投稿 |

---

## 8. 风险登记

| 风险 | 概率 | 应对 |
|------|------|------|
| C2 的 iter-vs-N 曲线不平坦（aux-precond 不工作） | 中 | 降级写"实用预条件"而非"N-independent"；增加 condition number 估计的实验测量 |
| direct 法在 h₄/h₅ 上 OOM 没法对比 | 高 | C4 图改成 "direct OOM 临界" + "aux 仍 work" 的对比，反而是亮点 |
| square direct 重启后仍不收敛 | 中 | 先做载荷步加密（dt = 0.5e-3 → 0.25e-3），再调 staggered tol 1e-5 → 1e-4 |
| Lagrange 路线（`MainSolve`）接口与 HuZhang 不对齐 | 中 | P1 内做接口适配检查 |
| 6 个月内 P3 来不及 | 中 | 砍 C5 V2 保留 V5；HuZhang vs Lagrange 在 Discussion 部分定性论述 |
