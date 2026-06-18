# D12 — 迭代稳定性测试改用「局部化 d」方案

> 起因（2026-06-09 复盘）：论文 §7 全部迭代数表（mesh-independence / l₀ / damage-robustness / SENT / SENS）
> 的损伤态都用 **空间均匀 d**（`discr.state.d[:] = np.full(n, dval)`，dval∈{0,0.5,0.9,0.99,0.999}）。
> **问题**：真实裂纹是 d≈1 的 O(ℓ₀) 窄带 + 其余区域 d≈0；难点是 **尖锐 d≈1/d≈0 界面**（1/g 跨几个单元跳 6 个量级），
> 而均匀 d 只是把整条弹性算子按常数缩放——是个**简单**问题。故现有 O(10) 结论对真实裂纹**不代表性、偏乐观**。

---

## 1. 证据（已有真实 run 数据，model0 圆孔缺口，aux_fast，elastic GMRES niter）

数据：`results/phasefield/model0_circular_notch/paper_aux_scan_auxfast_h{1,2,3}/epsg_1e-06/iterations.csv`

| 网格 | 起裂前 d<0.95 niter | 局部化 d≥0.99 niter (min–max, 中位) | overall 中位 |
|---|---|---|---|
| h₁ | 0–8 | **未达完全局部化**（run 未跑到尖界面） | 6 |
| h₂ | 0–7 | **56–109**，中位 101 | 97 |
| h₃ | 0–11 | **14–200**，中位 76 | 8 |

关键事实：
- 均匀-d 扫描：niter 始终 O(10)（≤18，even d=0.999）。
- 真实局部化：maxd≤0.82 时恒 ≈7；一旦尖界面形成（maxd→0.997）**骤升到 O(100)**（h₂ 95–109）。
- **真实局部化下的 mesh-independence 尚未坐实**：h₁ 没跑到局部化、h₃ 跨度大（14–200），三档不可比。

---

## 2. 定性结论仍成立（不要丢）

真实局部化态下 aux **仍有界收敛（O(100)）**，而 none/Jacobi/ILU 在该态**全部 DNF**（打满 60000）。
所以「aux 是真实裂纹上唯一可行的预条件子」成立；**只有「O(10) 有界」这个量化招牌不成立**。

---

## 3. 需要的新数据（程序层面，按优先级）

> 已有现成的局部化-d 构造器：`fracturex/tests/test_auxspace_precond_degraded_elastic.py`
> 里的 `half_cracked` / `band_cracked` / `patch_cracked`（d=1 on band，d=0 elsewhere），
> 但**只接了一致性检验（rel_diff/rel_res），没接 niter 扫描**。把它接进 `iter_stability_scan.py` 即可。

### P0 — flagship：mesh-independence on localized d（§7.4 招牌图 fig:iter_vs_N）
- 在 h₁..h₅ 五档上，用**同一条尖界面裂纹 d 场**（band_cracked，带宽 ∝ ℓ₀，与网格无关的物理带宽）扫 none/Jacobi/ILU/aux_fast 的 niter。
- 目的：回答「真实裂纹界面下 aux niter 是否对 h 有界」。预期 O(100) 但**跨档基本不变**才算 mesh-independent；若随 h 增长则要如实报增长率。
- 产出：替换/并列 §7.4 表 1 + fig:iter_vs_N。

### P1 — damage-robustness on localized d（§7.6 fig:iter_vs_d）
- 固定网格（h₂），把"均匀 d 从 0→0.999"换成"**裂纹带峰值 d 从 0→0.999、带外 d≈0**"，扫 niter。
- 这才是论文要卖的"d→1 鲁棒性"在真实形态下的版本。
- 产出：§7.6 表 3a/3b + fig:iter_vs_d 重做（或并列两版：controlled uniform / realistic localized）。

### P2 — 真实 run 的 niter-vs-maxd 曲线（直接用已有数据，近免费）
- 用 `paper_aux_scan_auxfast_h2`（maxd vs linear_niter_elastic）画一条真实加载历史的 niter 曲线，
  展示 niter 在 maxd→1 时从 7 跳到 ~100。作为"局部化才是难点"的直接证据图。
- **h₁ 需补跑到完全局部化**（当前没跑到尖界面），h₃ 需复核（14–200 跨度异常，可能含未收敛/早停点）。

### P3 — l₀ 无关性 on localized d（§7.5 tab:l0_indep）
- 现有 l₀ 表也是均匀 d=0.9。改成裂纹带 + 带宽随 ℓ₀，确认 niter 对带陡度不敏感仍成立。

### P4 — SENT/SENS 局部化版（§7.7/§7.8 表 6/8）
- 同样把均匀 d 换成各算例的物理裂纹带，确认三算例横向结论在真实形态下仍成立。

---

## 4. 写作层面已做（本次，tex 端）

- §7.6「Through the loading history」已加 dual-report 段：均匀-d 是 controlled benchmark，真实局部化 niter 升到 O(100) 但有界且唯一可行。
- 待新数据回来后，把每处"均匀 d"的表/图按上面 P0–P4 替换或并列，并把 abstract/intro 的"bounded, at most 18"招牌改为"O(10) on controlled uniform-d benchmark / O(100) on a fully localized crack, vs. all baselines DNF"。
- tex 中所有**等新数据**的位置已用 `\needexp{...}`（红色）标出，搜索 "localized" 可定位。

---

## 5. 一句话给程序端

把 `test_auxspace_precond_degraded_elastic.py` 的 `band_cracked`（d=1 窄带 + 其余 d≈0，带宽 ∝ ℓ₀）
接进 `scripts/paper_huzhang/iter_stability_scan.py`（及 square/model2 版），
对 none/Jacobi/ILU/aux_fast 扫 niter，五档网格（P0）+ 单档峰值-d 扫（P1）。其余沿用现有 GMRES 计数逻辑。
