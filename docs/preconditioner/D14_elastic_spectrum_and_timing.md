# D14 — 弹性方程求解提速:开裂后矩阵谱分析 + 每步计时拆解 + 并行组装

> 日期 2026-07-11。目的:诊断相场断裂弹性(Hu–Zhang 混合)求解慢在哪,能否提速。
> 数据来源:`results/phasefield/_precond_knob/{A.npz,meta.npz}`(真实开裂态装配矩阵)、
> `results/phasefield/_iter_stability/{iter_stability.csv, spec_auxfast_d*.npz}`、
> `results/adaptive_m3_pc_model2_eta_T/{history.csv, ..._run.log}`(eta_T 40 步跑,**存疑**,见 §4)。
> 与 `D12_RESULTS.md`(§5 迭代稳定性/谱)一脉相承;本文补"raw 算子谱 + setup-vs-solve 计时 + 生产路径计时落盘"。

---

## §1 开裂后弹性矩阵的特征值分布

分析对象 `A.npz`:model0 圆孔缺口板、h2 档、step_015 checkpoint(来自 `paper_aux_h2`),开裂已发生。

- 尺寸 `82508×82508`,`nnz=3.74M`,**完全对称**(‖A−Aᵀ‖=0)。
- Hu–Zhang 鞍点结构:应力块 σ `48092` + 位移块 u `34416`。
- 损伤场 `d∈[0, 0.998]`,`frac(d>0.9)=1.3%`(裂纹是细窄带)。

**谱(shift-invert @0 求近零端):**

| 量 | 值 |
|---|---|
| λ_max | 1.0 |
| λ_min | **3.0e-08**,且**全为正**(近零端前 8 个 3.0–3.3e-8) |
| κ(raw) ≈ λmax/|λmin| | **≈ 3.3e7** |
| σ 块对角动态范围 max/min | **6.4e7**(1.0 → 1.56e-8) |

**结论(反直觉):开裂后的病态几乎全部来自"损伤退化",不是鞍点不定性。**
- 最小特征值是**正**的 ~3e-8,鞍点负特征值没有主导谱底 → 该装配算子在此状态实际近(半)正定。
- σ 块对角从 1.0 掉到 1.56e-8,对应退化系数 `g(d)=(1−d)²+ε_g`:`d→0.998` 处 `(1−d)²≈4e-6`,再被 `ε_g=1e-6` 兜底。**λ_min 与 κ 基本由退化下限 `ε_g` 决定**;裂纹带越窄、d 越接近 1,κ 越大。

---

## §2 各预条件子随开裂的表现(`iter_stability.csv`, h2/σ=48092)

| d | none | jacobi | ilu | **aux_fast** |
|---|---|---|---|---|
| 0.0 | 60000✗ | 7521 | 41✗ | **7** |
| 0.5 | 60000✗ | 5600 | ✗ | **7** |
| 0.9 | 60000✗ | 4693 | ✗ | **15** |
| 0.99 | 14960 | 21758 | 22 | **18** |
| 0.999 | 60000✗ | 60000✗ | ✗ | **17** |

- 裸算子 / Jacobi / ILU 在 d→1 时**全线崩溃**(ILU 因退化块近奇异发散)。
- **aux_fast(辅助空间预条件)是唯一 damage-robust 的**:迭代数 7→18 不随 d 爆炸;`spec_auxfast_d*.npz` 的 `kappa_proxy` 在 d=0→0.999 之间**恒为 1.44**。
- 呼应 D12 §5.2 网格无关性:aux_fast 对损伤退化也稳。

---

## §3 计时拆解:慢在哪?

### §3.1 eta_T 每步时间去向(40 步,总 22.58h;从 `_run.log` 的 `[PC]` 行解析)

| 指标 | 值 |
|---|---|
| **t/iter 中位数**(单次外层 staggered 迭代) | **187s** |
| t/iter p90 / max | 417s / 600s |
| iters=200(未收敛封顶)步 | 4 步(27/34/37/39),共 19942s = **仅 25%** |
| 其余 75% | 落在 iters≤30 的"正常收敛"步 |

典型行:step2 只迭代 2 次却花 566s;step28 迭代 4 次花 2398s。**瓶颈不是线性迭代数,而是"每一次外层迭代的固定成本"≈100–600s**(装配 HZ 应力算子 + 预条件子 setup + 弹性解 + 相场解 + H 历史更新)。

### §3.2 setup vs solve 分解(在 `A.npz` 上直接量)

用求解器自身 helper(即 `schur_rebuild_interval=1` 每次重建的对象):

| setup 分项(每次解都重建) | 耗时 |
|---|---|
| extract blocks | 16 ms |
| diag-inv stress block | 4 ms |
| Schur 三重积 B·D⁻¹·Bᵀ(nnz=5.06M) | 54 ms |
| pyamg SA build(Schur 上) | 372 ms |
| **可摊销 setup 合计** | **≈ 0.45 s** |

对照:aux 单次完整解 **14–47 s**(iter_stability,同 h2/48092)。

**结论:setup 仅占 ~1–3%,复用 setup 不是提速杠杆。** 谱既 damage-invariant(κ_proxy≡1.44)、setup 又便宜。成本在**每次 GMRES 迭代的预条件应用**(Chebyshev sweeps + 粗空间 GAMG 解),更在**外层迭代里非弹性解的部分**。

### §3.3 提速优先级(基于上面证据,非猜测)

1. **组装默认并行**(见 §5)——每次外层迭代都重装 82508² / 3.7M nnz 的 p=3 HZ 算子,原 `assembly_parallel=False` 串行,最可疑的非解开销。已改默认。
2. **修 4 个 blow-up 步(占 25% 墙钟)**:step32 `D_max=1e13` 等,外层非线性/能量发散问题,加 line-search / 能量单调守卫。与线代无关。
3. 之后再微调弹性解;`schur_rebuild_interval=k` 是免费 ~1%,可设 5 但非重点。

---

## §4 eta_T run 健康检查(判**存疑**,thesis 不可用)

`adaptive_m3_pc_model2_eta_T/history.csv`(40 步)红旗:

| 检查 | 结果 |
|---|---|
| `converged` 全 True | ✗ steps **27,34,37,39** 不收敛 |
| `eta_tau`/`eta_dg` | **全 40 行 nan** → estimator 无数据 |
| `max_d` 首步==1.0 | ✗ 首步即饱和(d-饱和隐患) |
| `D_max` 有界 | ✗ 爆到 **1.0e13**(step32)、4e6(step36) |

按 pitfalls:命中 d-饱和 + 不收敛,需先查 resume 时 H 是否被丢、d-饱和 marker 是否缺,再怀疑数值。**本文只用其 `t_step`/`iters` 做 §3.1 计时结构分析(结构性结论不依赖 estimator);estimator 结果与 thesis 数值实验不采用此 run。**

---

## §5 代码改动(最小 diff)

> **更新 2026-07-11(见 §7):§5 的"组装默认翻为并行"已被 §7 的 crossover 实测推翻并回退为串行。**
> 下方为历史记录,新默认与依据以 §7 为准。

**组装默认翻为并行**(仅改 `None`-回落分支,`FRACTUREX_ASSEMBLY_PARALLEL=0` 仍可强制串行):
- `fracturex/assemblers/huzhang_elastic_assembler.py:360` — `_env_flag(..., False)` → `..., True)`
- `fracturex/assemblers/phasefield_assembler.py:460` — `os.getenv(..., "")` → `os.getenv(..., "1")`
- `fracturex/adaptivity/adaptive_staggered.py:71` — `make_assemblers(parallel: bool=False)` → `Optional[bool]=None` 透传(eta_T/自适应路径经此;8 处调用均不带 `parallel=`,自动继承并行默认)
- `fracturex/tests/case_runners/{model0,model2}_runner.py` — 删除硬编码 `assembly_parallel=False`,回落新默认

**eta_T 分项计时落盘 + 打印**(`fracturex/tests/aposteriori/run_m3_pc_model1.py`):
- `history.csv` `fields` 增 4 列 `t_{elastic_assemble,elastic_solve,phase_assemble,phase_solve}_s`;`row` 从 `info.meta.get(...)` 取值。
- `[PC]` 行增 `tEA= tES= tPA= tPS=`。
- 注:driver 的 `timing=True` 只启 fealpy `timer` 生成器且从不 `send(None)` → **不打印**,是无效项;分项耗时本就用 `perf_counter` 无条件采集在 `info.meta`,故走 meta 落盘而非该 flag。

**验证**:6 文件 `py_compile` 全过;`make_assemblers` 8 调用点均无 `parallel=`;`info.meta` 确含那 4 键。

---

## §6 下一步

- 重跑 eta_T(或 model2 PC):新 `[PC]`/`history.csv` 直接给每步四阶段秒数 → 量到 (a) 并行组装把 `tEA` 降多少;(b) §3.1 "75% 非弹性解 vs 弹性解本身"用真实分项数替代 log 反推。
- 若 `tES`(弹性解)仍偏高,再看 restart/GMRES 应用成本(D12 §5.2b restart-aware 结论)。
- blow-up 步单独治(能量守卫),与本文线代结论解耦。

---

## §7 弹性解提速实测:direct 求解器选型 + 组装 crossover + MKL 过订阅(2026-07-11)

> 起因:eta_T run 里 `tES` 异常(pardiso 单解 ~140s、34k dof),追它到底。
> 全部 probe:model2、nx=24/48/64、`FRACTUREX_MAX_STEPS=2`、mumps 后端、机器空闲(其余为
> `run_m3b_lambda_eq_sweep` 训练,176 核占用 ~4)。warm=step1(step0 含 JIT/首触偏高)。
> **无红旗**:全部 `converged=True`、`R` 一致(nx24 4.285e-3 / nx48 4.409e-3 / nx64 4.421e-3)、
> 无 nan、解不变(mumps 换 pardiso 同解)。

### §7.1 direct 求解器台架(单次解同一 A,n=33171,后裂 HZ p=3,`_solver_bench.py`)

| 求解器 / ordering | 时间 | rel_resid |
|---|---|---|
| pardiso (mtype=11) | 112.0s | 2.8e-13 |
| mumps **amd** | **59.4s** | 3.4e-10 |
| mumps auto | 62.7s | 3.4e-10 |
| mumps metis | 69.9s | 3.4e-10 |
| mumps pord | 113.6s | 3.4e-10 |
| mumps scotch | 154.3s | 3.4e-10 |
| reuse_analysis | analyze(metis)=6.4s · factor#1=58.6s · factor#2(reuse)=**52.9s** | |

- **mumps-amd 比 pardiso 快 ~1.9×**;但 ordering 主导:`amd`/`auto`≈60s,`metis`70s,`pord`/`scotch` 与 pardiso 一样慢甚至更慢。
- **旧 `solve_direct_mumps` 硬编码 `ordering="pord"`(最差档)** → 潜在 bug,换 `amd` 直接砍半。
- **符号复用不是杠杆**:analyze 仅占 ~10%,factor(reuse) 仍 52.9s;数值分解主导,p=3 HZ 填充压不掉 → 不做跨 staggered 持久求解器。

### §7.2 生产路径头对头(runner,model2 nx=24,dofσ=19347,MKL=8,串行组装)

| 求解器 | tES step0 | tES step1 | 2 步墙钟 |
|---|---|---|---|
| pardiso | 38.6 | 51.0 | 115.2s |
| **mumps-amd** | 29.9 | 28.1 | **85.6s** |

同解(R=4.285e-3、iters=2),mumps-amd tES 快 ~1.6×、更稳(pardiso step1 抖到 51s)。

### §7.3 组装 crossover(serial vs parallel,MKL=8,warm tEA/tPA)

**(a) 全 staggered run(runner,2 步,warm=step1;tEA/tPA 为每步累积):**

| nx | dofσ | tEA 串行 | tEA 并行 | 比 | tPA 串行 | tPA 并行 | 比 |
|---|---|---|---|---|---|---|---|
| 24 | 19347 | 5.6 | 22.3 | 4.0× | 0.4 | 8.8 | 22× |
| 48 | 76707 | 35.0 | 83.6 | 2.4× | 2.3 | 22.2 | 9.6× |
| 64 | 136067 | 68.1 | 128.7 | 1.9× | 5.8 | 18.3 | 3.2× |

**(b) 组装-only micro-bench(`asm_bench.py`:单次 warm+单次计时,无线性解;含生产分辨率 nx=160):**

| nx | nc | tEA 串行 | tEA 并行 | 比 | tPA 串行 | tPA 并行 |
|---|---|---|---|---|---|---|
| 64 | 8192 | 23.6 | 47.9 | 2.0× | 1.2 | 5.7 |
| **160** | **51200** | **248.1** | **389.7** | **1.57×** | 15.5 | 16.3 |

micro-bench(b)复现(a)的 tEA 串:并比(nx64 2.0× vs run 1.9×),故为可信代理;瓶颈是 `_assemble_M_block_cached` 里的 `bmm_einsum`(RSS 9.2GB @nx160,基本单线程),并行按胞 chunk 该 einsum 但线程开销/缓存争用反噬。

- **并行组装��实测全区间(nx=24/48/64/160,直到 51200 胞 / ~850k dof)每档都更慢**;相场 p=1 尤甚(tPA 比 3–22×,大网格趋平)。
- 惩罚随规模递减(tEA 4.0→1.57×)但**始终不 crossover**;**nx=160 即 paper_aux 生产分辨率,串行仍胜 1.57×** → **§5 的并行默认回退为串行已在生产尺度证实**。`FRACTUREX_ASSEMBLY_PARALLEL=1` 仍可强开。

### §7.4 MKL 过订阅(tES 独立退化项)

- 并行组装 uncapped-MKL(当前旧默认)vs 串行 MKL=8:2 步墙钟 **275s → 85.6s(~3.2×)**。拆分:
  并行组装吃 tEA/tPA(§7.3);**MKL 不设上限吃 tES**:同为并行组装,tES 32s(MKL=8)→ 91s(uncapped,176 线程在 19k 系统上互抢)。
- 线程必须在 Python 启动前设定 → 只能"文档 + 启动期告警"。已在 runner 加 `_warn_mkl_oversubscription()`:
  `MKL_NUM_THREADS` 未设或 >16 时打印告警。**启动配方**:`MKL_NUM_THREADS=8 OMP_NUM_THREADS=8`。

### §7.5 代码改动(最小 diff)+ 验证

- `fracturex/utilfuc/sparse_direct_backends.py:solve_direct_mumps` — `ordering` 由硬编码 `"pord"` 改为 `os.getenv("FRACTUREX_MUMPS_ORDERING","amd")`。
- `fracturex/assemblers/huzhang_elastic_assembler.py` — `_env_flag(...,True)` → `False`(串行默认)。
- `fracturex/assemblers/phasefield_assembler.py` — `os.getenv(...,"1")` → `"0"`(串行默认)。
- `fracturex/tests/aposteriori/run_m3_pc_model1.py` — 新增 `_warn_mkl_oversubscription()`,`main()` 起始调用。
- **验证**:4 文件 `py_compile` 全过;默认 run(不设任何 flag)实测 tEA warm 4.0s / tPA 0.6s(串行档已生效)、`[warn]` 正常触发;mumps-amd 同解 iters=2 R=4.285e-3。

### §7.6 净结论

弹性解慢 = 三个可分离旋钮,均已量化并修:(1) mumps ordering 用 `amd` 不用 `pord`(~1.9×);
(2) 组装默认回串行(tEA/tPA 至 136k dof 均胜);(3) MKL 设 8 防过订阅(tES ~2.8×)。
最优配置(串行 + MKL=8 + mumps-amd)= 2 步 85.6s,对当前并行+uncapped 默认 ~3.2×。
**pardiso 并非天生慢;是 ordering/线程/组装三处配置叠加。**
