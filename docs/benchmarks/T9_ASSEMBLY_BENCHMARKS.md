# T9 / FractureX 框架论文 —— 组装 scaling & p-version benchmark 结果

> 归属：论文 **F（T9）**，柱一（HZ 并行组装）+ 三离散 p-version 性能研究。
> 计划见 `docs/planning/T9_FRAMEWORK_PAPER_PLAN.md` §3。
> 本文档汇总两个 benchmark 脚本的产出，供论文 §6 Performance study 取数、画图。
> **数据档位：细网格正式档（2026-07-11，10 核机）。**

---

## 0. 脚本与产出位置

| 脚本 | 内容 | 结果 JSON |
|---|---|---|
| `fracturex/tests/hz_assembly_scaling_benchmark.py` | 柱一：HZ 弹性组装器孤立计时（不跑 staggered），四轴 strong/size/pver/formulation + 缓存冷热 | `results/benchmarks/hz_assembly/hz_assembly_scaling.json` |
| `fracturex/tests/discretization_pversion_benchmark.py` | 三离散 p-version：标准 Lagrange + C⁰-IP，共享单位方结构网格，孤立组装计时三核 | `results/benchmarks/pversion/discretization_pversion.json` |

两脚本互不重叠：HZ 的 p 归柱一脚本（HZ 元 p≥3 + 独立 assembler），标准/IP 归第二脚本。

**计时口径**：每档一次 **cold**（首调，建 d-无关几何内核缓存）+ 若干次 **warm**（取中位数，复用缓存）。`cache_speedup = cold/warm`。墙钟是**纯组装**，不含线性求解。日志在 `results/benchmarks/logs/`。

---

## 1. 复现命令（venv_fealpy3）

```bash
source ~/venv_fealpy3/bin/activate
# 正式细网格档（本次数据的完整驱动脚本）：
bash results/benchmarks/run_fine.sh          # 串行跑 HZ 四轴 + p-version，日志分开
# 单独：
python -m fracturex.tests.hz_assembly_scaling_benchmark
python -m fracturex.tests.discretization_pversion_benchmark
```

env 覆盖：HZ `FRACTUREX_BENCH_AXES/_HMIN/_HMIN_LIST/_NPROC_LIST/_P/_P_LIST`；p-version `FRACTUREX_PVER_NX/_PDISP/_PPHASE/_KERNELS`。

---

## 2. 柱一 · HZ 组装 scaling（`model0_circular_notch`，p=3）

### 2.1 strong scaling（固定网格 130759 dof / 4555 cells，vs 线程数）

| nproc | warm_s | cold_s | 缓存加速 | vs 串行 |
|---:|---:|---:|---:|---:|
| 1 | 0.251 | 2.791 | 11.1× | 1.00× |
| 2 | 0.269 | 2.228 | 8.3× | 0.93× |
| 4 | 0.267 | 2.096 | 7.9× | 0.94× |
| 8 | 0.252 | 2.360 | 9.4× | 1.00× |
| 10 | 0.250 | 2.198 | 8.8× | 1.00× |

> **⭐ 核心诚实发现（改变卖点重心）**：即使到 **130k dof**，warm 组装墙钟随线程数**基本持平**（0.25s，加速 0.93–1.0×）——**ThreadPool 并行对 warm 组装几乎无收益**（memory-bound + BLAS 已用满）。**真正的赢面是 d-无关几何内核缓存：cold→warm 8–11×。** 论文柱一的头条应是**缓存复用**，并行组装降级为"cold 首装的适度加速"（cold 2.79→2.10s，~1.3×）。这条数据必须如实写进 §4 诚实定位，别把并行吹成主卖点。

### 2.2 size scaling（wall vs DOF，nproc=10）

| hmin | gdof | ncells | warm_s | cold_s | 缓存加速 |
|---:|---:|---:|---:|---:|---:|
| 0.04 | 29240 | 1010 | 0.061 | 0.480 | 7.9× |
| 0.03 | 55008 | 1908 | 0.108 | 0.920 | 8.5× |
| 0.02 | 130759 | 4555 | 0.259 | 2.213 | 8.5× |
| 0.015 | 234226 | 8174 | 0.484 | 5.133 | 10.6× |

> warm 墙钟近似线性于 DOF（29k→234k：0.061→0.484s，~8× DOF / ~8× 时间）。缓存加速稳定 8–11×，与规模无关。

### 2.3 p-version（HZ 元，固定 hmin=0.02）

| p | gdof | warm_s | cold_s | 缓存加速 |
|---:|---:|---:|---:|---:|
| 3 | 130759 | 0.242 | 1.684 | 7.0× |
| 4 | 222128 | 0.549 | 7.710 | 14.1× |

> p=3→4：warm 2.3×、cold 4.6×，缓存加速升到 14×（高次单元几何内核更贵，缓存收益更大）。

### 2.4 formulation（standard vs effective_stress，p=3, 130759 dof）

| formulation | warm_s | cold_s | 缓存加速 |
|---|---:|---:|---:|
| standard | 0.243 | 1.570 | 6.5× |
| effective_stress | 0.506 | 1.841 | 3.6× |

> effective_stress warm ~2× 于 standard（g(d) 落在 B 上、带 ∇g 链式项，每步重装耦合块）。
> ⚠️ 已知警告：effective_stress 的 `B2_gradg` 并行组装 einsum 触 fallback 转串行（`Size of label 'c'...` 标签不匹配），不影响结果正确性但吃掉该分支的并行；若论文要展示 effective_stress 并行，需修这个 einsum（记 TODO）。

---

## 3. 三离散 p-version 组装成本（共享单位方结构网格 nx=48，2304 cells）

材料 E=200/ν=0.2/Gc=1/ℓ₀=0.02。

| kernel | p | gdof | warm_s | cold_s | 缓存加速 |
|---|---:|---:|---:|---:|---:|
| standard-elastic | 1 | 4802 | 0.035 | 0.038 | 1.09× |
| standard-elastic | 2 | 18818 | 0.130 | 0.137 | 1.05× |
| standard-elastic | 3 | 42050 | 0.395 | 0.412 | 1.04× |
| ip-elastic | 1 | 4802 | 0.062 | 0.063 | 1.00× |
| ip-elastic | 2 | 18818 | 0.175 | 0.172 | 0.98× |
| ip-elastic | 3 | 42050 | 0.582 | 0.599 | 1.03× |
| ip-phase4th | 2 | 9409 | 0.142 | 0.169 | 1.19× |
| ip-phase4th | 3 | 21025 | 0.367 | 0.357 | 0.97× |
| ip-phase4th | 4 | 37249 | 1.143 | 1.263 | 1.10× |

**读数**：
- **IP 弹性 ≈ 1.5× 标准弹性**（同 p、同 dof）：`ip-elastic` 0.062/0.175/0.582 vs `standard-elastic` 0.035/0.130/0.395。内罚空间构建 + 更高积分次数 `q=2·max(p)+3` 的开销。
- **ip-phase4th 4 阶核最贵、增长最陡**：p2→p4 warm 0.142→1.143s（~8×），量化 C⁰-IP 4 阶（biharmonic + 内罚）离散的组装代价。
- 结构网格上标准/IP 弹性核缓存 ≈1.0×（每次重建 `BilinearForm`，无 HZ 那种 d-无关几何内核缓存）——**缓存复用是 HZ 组装器的专属优化**，正好反衬柱一卖点。

> 同 p 的 standard/ip-elastic gdof 相同（4802/18818/42050），可直接画 wall-vs-p、wall-vs-DOF、standard-vs-IP 对照。

---

## 4. 论文取图建议

| 图 | 数据源 | 说明 |
|---|---|---|
| wall-vs-DOF（组装线性 scaling） | §2.2 | 柱一主图，log-log 近线性 |
| **缓存冷热加速比（柱状/vs DOF）** | §2.1–2.4 | **柱一真头条**，8–14× 稳定 |
| wall-vs-nproc（诚实：并行 warm 近持平） | §2.1 | 配 §4 诚实定位，别夸大 |
| wall-vs-p（HZ p3/p4 + 三离散） | §2.3 + §3 | 高次代价 |
| standard vs IP 弹性 + IP 4 阶核 | §3 | 三离散同底座对照 |

---

## 5. 遗留 TODO

- [ ] effective_stress `B2_gradg` 并行 einsum 标签 bug（§2.4）：修好才能展示该分支并行
- [ ] 若要更长 nproc 尾巴看拐点：warm 已证并行无收益，不必再跑；如需 cold 并行曲线可加档
- [ ] HZ pver 若要 p≥5：先确认 fealpy HZ 空间支持

---

## 附：两脚本踩过的 API 坑（复现/扩展时必看）

1. **HZ 元 p≥3** —— u 空间次数 = p−1；p=2 触 fealpy `huzhang_fe_space_2d.div_basis` 角点 dof `IndexError`。
2. **HZ 孤立组装计时前必须 `damage.on_build(discr, DamageStateView(...), case)`** —— 否则 `_gfun=None`（driver 的 `initialize()` 平时干这活）。
3. **标准 `MainSolve` 要先 `ms._method='lfem'` 再 `initialize_settings(p)`** —— `_method` 平时在 `solve()` 里才设。
4. **IP `_assemble_phase_lhs()` 前要 `solver.H=0.0`** —— 历史场 u=0 时本为 0（newton 里由 `pfcm.maximum_historical_field` 先算），不设则 `mass_coef` 里 `2*None` TypeError。
