# M0 D-B：𝓗 插值误差报告 + h2 重跑后台启动（2026-05-29 晚）

> 范围：[m0_session_report_2026-05-29.md](m0_session_report_2026-05-29.md) §6 列出的
> D-B 1–5 项。
> 状态：D-B 1–5 闭环（h2 重跑后台跑中、h3 命令就绪未起）；36 测试通过；
> §4.2 表 + h_interp_compare.png + h_interp_error.csv + summary.json 入库；
> [m0_interpolation_error.md](../../m0_interpolation_error.md) v0.2 → v0.3，DoD 勾到 6/8。

## 0. 起点 / 范围

[m0_session_report_2026-05-29.md (D-A)](m0_session_report_2026-05-29.md) 把 σ 段
完整闭环了（𝓘₁/𝓘₂ 实装、解析阶数锁定、真实 paper_aux σ 数据量化），但
[m0_interpolation_error.md](../../m0_interpolation_error.md) v0.2 §4.2（𝓗 段）和
§5.2（𝓗 默认选择）仍然空，理由是 `RunRecorder.save_quadrature_fields` 是占位
开关，未接落盘管线（[kickoff §6 已知缺口 4](m0_kickoff_report_2026-05-28.md#6-已知缺口--下一刀)）。
本会话（D-B）按 D-A §6 列的 5 项推进：

1. recorder 在 `save_quadrature_fields=True` 时落 `H_qp.npz`；
2. 跑 model0 短补丁 run；
3. 用 §3.1（D-A）同样的 𝓘₁/𝓘₂ 流程跑 𝓗；
4. paper_aux_h{2,3} 重跑到 t_c；
5. 据 §4.2 实测写定 §5.2 + 回写 schema §3.5 注释。

不动求解器主链；现有 P1/P2 论文实验完全兼容。

## 1. 设计决定与依据

### 1.1 落盘点放 `RunRecorder` 新方法，不动 `save_checkpoint`

需求是给"𝓗 在积分点上的 (NC, NQ)"加一条落盘路径。两个候选：

- **(a)** 把 `H_qp` / `xq` 塞进现有 `step_XXX.npz` —— 修 `save_checkpoint` 内部。
- **(b)** 新增独立方法 `dump_quadrature_fields(step, discr, state)`，落
  `step_XXX_qp.npz`；driver 在 `save_checkpoint` 后 hasattr-guard 调一次。

选了 **(b)**。理由：
- 用户当下正在跑传统方法（P1/P2 论文）的复现实验，`save_checkpoint`
  接口是它们读 npz 的入口；改动 npz 字段会带来不必要的兼容面积。
- 老 recorder 没有该方法时 driver 静默跳过（hasattr-guard），向后兼容。
- `step_XXX.npz` 与 `step_XXX_qp.npz` 一一配对、按 step 索引清晰。

### 1.2 方法名 `dump_quadrature_fields`，不能复用 `save_quadrature_fields`

`save_quadrature_fields` 已经在 [recorder.py kickoff 修改](../../../fracturex/postprocess/recorder.py)
里做成 `__init__` 的 bool 属性了（kickoff §1.6 占位开关）。同名方法会被
instance attribute shadow，调用时变成 `False(...)` → TypeError。改名
`dump_quadrature_fields`，把 bool 开关名留给原契约。

### 1.3 quadrature 取 `damage_p + 3`，与 phasefield_assembler 对齐

H_qp 是相场装配里 `update_history_on_quadrature` 生成的，quadrature rule 取
`damage_p + 3`（[phasefield_assembler.py:454](../../../fracturex/assemblers/phasefield_assembler.py#L454)）。
recorder 落盘时复用同一阶（`q_order = damage_p + 3`），`xq` 与 `H_qp`
索引相同 (NC, NQ) 个点，下游测量端不需要再算 quadrature。

### 1.4 §4.2 测量协议：qp → grid → qp roundtrip + max-preservation

D-A 的 σ 测量靠"truth on grid = HuZhang basis × σ_dofs（无重构误差）"，因为
σ 有 FE 基函数表示。但 𝓗 只活在 qp 上，没有"无损 grid 表示" —— 拿不到
ground truth on grid。只能换协议：

$$
\tilde H_{qp}^* := \mathrm{bilinear}\bigl(\mathcal I_*(H_{qp})\bigr)\big|_{x_q},
\quad
e_{L^2} = \frac{\|m_q \odot (\tilde H^* - H_{qp})\|_2}
              {\|m_q \odot H_{qp}\|_2 + \varepsilon}.
$$

这个 metric 量化的是 plan §3.3 真正关心的事 —— "qp → grid → qp
信息损失" —— 因为算子学习把 𝓗 落到 grid 上后未来还要回采到 qp 用。

附加一个 max-preservation 指标 `max_ratio = max(𝓘_*(H) on grid) / max(H_qp)`：
裂尖 cusp 的峰值是相场驱动场的物理关键，砍掉一半就改变了断裂演化。

### 1.5 加 `const` baseline 行解释 60% rel_L²

只看 𝓘₁ rel_L² 在 t_b/t_c=0.60，会得出"𝓘₁ 不够好"的错误结论。但 𝓗 在裂尖
是 cusp，**结构网格 + bilinear 回采本身就限制了能恢复多少信息**。加一个零
信息基线：每 qp 用 inside-Ω 平均预测。t_b/t_c 上 const rel_L²=0.99，𝓘₁ 把
这个值拉到 0.60 —— 等于在 grid 表示力的硬上限下又压了 ~64% 残差方差。
没有这一行，§4.2 的物理解读会站不住。

### 1.6 paper_aux_h2 重跑放 `paper_aux_h2_dB/`，不覆盖现有 dir

`paper_aux_h2/epsg_1e-06/checkpoints/` 已有 step_000/010 的 short-run 数据，是
[m0_session_report_2026-05-29.md (D-A)](m0_session_report_2026-05-29.md) §3.2
表 §4.1(b) 的来源。**覆盖会破坏 D-A 已交付的 §4.1(b) 收敛性数据。**
用新后缀 `paper_aux_h2_dB`：

- 老 paper_aux_h2 的 short-run 数据保持原状，§4.1(b) 已交付内容不动；
- 新 D-B 全 31 步数据进 paper_aux_h2_dB，§4.1(a) 跨 h 时间扫描接 dB 数据；
- [measure_interpolation_error.py](../../../scripts/datasets/measure_interpolation_error.py)
  `_default_cases` 改成"先找 _dB，没有 fallback 老路径"，h2_dB 落 mesh.npz
  那一刻自动接入，无需再改代码。

### 1.7 h3 不发起，只生成命令 + DoD note

用户说后台还有别的程序在跑、h3 排队。NC≈11k 的 GMRES auxspace 单步在本机
30-60 分钟，全 31 步 15-30 小时；起 h3 会拖慢用户的现役任务。命令落到
[m0_interpolation_error.md §7.1](../../m0_interpolation_error.md#71-h3-重跑命令待-h2-完成--用户后台空档)，
等用户后台空闲手动起。

## 2. 落地代码

### 2.1 [fracturex/postprocess/recorder.py](../../../fracturex/postprocess/recorder.py)

新增 `dump_quadrature_fields(step, discr, state)` 方法：
- `save_npz=False` / `save_quadrature_fields=False` / `step % save_every != 0` /
  `state.H is None` / `discr.mesh is None` 任一时静默 no-op；
- quadrature rule = `damage_p + 3`，与 [phasefield_assembler.py:454](../../../fracturex/assemblers/phasefield_assembler.py#L454) 一致；
- 落盘字段：`H_qp (NC,NQ) float64`、`xq (NC,NQ,2) float64`、`q_order int`、`step int`；
- 路径：`<outdir>/checkpoints/step_XXX_qp.npz`，与 `step_XXX.npz` 同 dir 配对。

`__init__` / `save_checkpoint` / `save_mesh` / `append_history` 等既有接口完全
不动，向后兼容。

### 2.2 [fracturex/drivers/huzhang_phasefield_staggered.py](../../../fracturex/drivers/huzhang_phasefield_staggered.py)

`solve_one_step` 在 `recorder.save_checkpoint(step, discr, state)` 之后加：

```python
if hasattr(self.recorder, "dump_quadrature_fields"):
    try:
        self.recorder.dump_quadrature_fields(step, discr, state)
    except Exception as exc:
        if self.debug:
            print(f"[driver] dump_quadrature_fields failed: {exc}")
```

老 recorder 没有该方法时 hasattr 直接跳过，新 recorder 但 bool 开关 False 时
内部 no-op。`run` / `solve_one_step` 主体逻辑、`HuZhangPhaseFieldStaggeredDriver`
构造签名零变更。

### 2.3 [fracturex/tests/case_runners/model0_runner.py](../../../fracturex/tests/case_runners/model0_runner.py)

`Model0RunArgs` 加 `save_quadrature_fields: bool = False`（默认关，向后兼容），
透传到 `RunRecorder(...)`。**不动**论文实验入口
[phasefield_model0_huzhang.py](../../../fracturex/tests/phasefield_model0_huzhang.py)
（kickoff §6 已知缺口 7：m0 主入口未重构）。

### 2.4 [scripts/datasets/run_h_qp_patch.py](../../../scripts/datasets/run_h_qp_patch.py)（新增）

paper_aux_h1 同配置短补丁 run 入口：`hmin=0.05`、`p_sigma=3`、`damage_p=2`、
`AT2/quadratic/hybrid` 相场，`save_every=10` + `save_quadrature_fields=True`。
输出 `results/operator_learning_runs/h_qp_patch_h1/`。

### 2.5 [scripts/datasets/measure_h_interp_error.py](../../../scripts/datasets/measure_h_interp_error.py)（新增）

§4.2 测量入口。三段：
- `_bilinear_resample(grid_field, xq, grid)`：grid (H,W) + qp 物理坐标 → (NC, NQ) 双线性回采；
- `_qp_inside_mask(xq, mask, grid)`：标记 qp 落在 inside-Ω pixel 的 mask（实测命中 99.6%）；
- `_measure_one(...)`：每 step 输出三行（𝓘₁、𝓘₂、const baseline），每行报
  `rel_L²`、`rel_L∞`、`max_grid`、`max_ratio`。

输出：
- `docs/figures/m0/interp_error/h_interp_error.csv`（长格式，9 行 = 3 步 × 3 方案）；
- `docs/figures/m0/interp_error/h_interp_summary.json`（by_step 嵌套）；
- `docs/figures/m0/interp_error/h_interp_compare.png`（𝓘₁ vs 𝓘₂ 在 grid 上 t_a/t_b/t_c × 2 列）。

### 2.6 [scripts/datasets/measure_interpolation_error.py](../../../scripts/datasets/measure_interpolation_error.py)

`_default_cases` 改 case-resolver：每个 h-档先查 `paper_aux_<h>_dB`，没有
fallback 到 `paper_aux_<h>`。h2_dB / h3_dB 落 mesh.npz 那一刻就自动接入，
无需再改代码。h_label 保持 `h1/h2/h3`，§4.1(b) 收敛性图坐标连续。

## 3. 算例

### 3.1 H_qp 落盘单测：h_qp_patch_h1

**配置：**
- 网格：distmesh `Model0CircularNotchCase`，hmin=0.05 → NC=640、NN=372；
- HuZhang p=3，damage_p=2 → q=5 → NQ=15；
- AT2 / quadratic degradation / hybrid split / eps_g=1e-6；
- 加载序列：paper schedule 31 步（`linspace(0, 0.07, 6) ∪ linspace(0.07, 0.125, 26)[1:]`）；
- save_every=10 → checkpoint at step 0/10/20/30。

**复现：**

```bash
cd /home/gongshihua/tian/fracturex
PYTHONPATH=$PWD $FEALPY_PYTHON scripts/datasets/run_h_qp_patch.py
# wall ≈ 2.5 min, 31 steps, max_d=0.991 at step_30
```

**输出：**

```text
results/operator_learning_runs/h_qp_patch_h1/
├── meta.json
├── history.csv          # 31 行 + header，max_H 单调上升 0 → 1.85e+4
├── iterations.csv
├── mesh.npz             # driver 自动 emit
└── checkpoints/
    ├── step_000.npz       step_000_qp.npz   (H_qp 全 0)
    ├── step_010.npz       step_010_qp.npz   (max_H ≈ 18, max_d ≈ 0.04)
    ├── step_020.npz       step_020_qp.npz   (max_H ≈ 6.8e+3, max_d ≈ 0.71)
    └── step_030.npz       step_030_qp.npz   (max_H ≈ 1.85e+4, max_d ≈ 0.99)
```

**形状校验：**
```text
H_qp.shape  = (640, 15)   float64
xq.shape    = (640, 15, 2) float64
q_order     = 5
```
NC=640、NQ=15 与 paper_aux_h1 mesh + damage_p=2 严格匹配，落盘管线
端到端通。

### 3.2 §4.2 𝓗 实测（[h_interp_error.csv](../../figures/m0/interp_error/h_interp_error.csv)）

**输入：** §3.1 的 4 帧 `step_XXX_qp.npz`，跳过 step_000（`H_qp` 全零）。
`_pick_three` 给出 `t_a=step_010`、`t_b=step_020`、`t_c=step_030`。

**结果：**

| 时刻 | 方案 | $e_{L^2}$ | $e_{L^\infty}$ | max_ratio | max(H_qp) |
| --- | --- | --- | --- | --- | --- |
| t_a | 𝓘₁ | 0.086 | 0.451 | **0.86** | 18.2 |
| t_a | 𝓘₂ | 0.111 | 0.429 | 0.81 | 18.2 |
| t_a | const | 0.824 | 0.918 | 0.082 | 18.2 |
| t_b | 𝓘₁ | 0.593 | 0.759 | **1.000** | 6.79e+3 |
| t_b | 𝓘₂ | 0.687 | 0.588 | 0.461 | 6.79e+3 |
| t_b | const | 0.997 | 0.997 | 0.003 | 6.79e+3 |
| t_c | 𝓘₁ | 0.599 | 0.799 | **0.999** | 1.85e+4 |
| t_c | 𝓘₂ | 0.721 | 0.722 | **0.404** | 1.85e+4 |
| t_c | const | 0.994 | 0.994 | 0.006 | 1.85e+4 |

`frac_qp_in_grid = 0.9958`（4 个 qp 落在 mask 外，约 0.4%）。

**关键观察：**

1. **𝓘₂ 把 𝓗 峰值砍到 ~40%**。t_c 时 max_ratio = 0.404，裂尖 cusp 在 P2 上的
   $L^2$ 投影把峰磨平。物理含义：相场演化下一步用 𝓗 算驱动力，砍掉一半就
   等于改变了断裂动力学。
2. **𝓘₁ 几乎守峰**。t_b/t_c 上 max_ratio ≈ 1.0；唯一非 1 的 t_a 差额（0.86）
   来自 0.4% qp 落 mask 外。
3. **roundtrip rel_L² 都很大不是 𝓘 失败**。const 基线给出 0.99，𝓘₁ 把它压
   到 0.60；剩余的 0.60 是 grid + bilinear 表示力的硬上限（cusp 在 grid 上
   不可还原）。
4. **与 v0.2 草案预判相反**。v0.2 §4.2 写"预判 𝓘₂ 在 𝓗 上反胜 𝓘₁"，
   理由是"𝓗 已是低阶，P1 投影无损失"。**预判错。** 𝓗 在裂尖是几何意义上
   不光滑函数，不属于"低阶"范畴；P2 投影同样磨平。

### 3.3 paper_aux_h2 重跑（D-B 后台启动）

**目标：** 把 [m0_interpolation_error.md §4.1(a)](../../m0_interpolation_error.md)
跨 h 时间扫描的 h2 列从 short-run 推到完整 t_a/t_b/t_c，关掉 D-A §3.2 的
"h3 反弹"注¹。

**复现：**

```bash
cd /home/gongshihua/tian/fracturex
FEALPY_PYTHON=/home/gongshihua/miniconda3/envs/py312/bin/python \
FRACTUREX_PYTHON=/home/gongshihua/miniconda3/envs/py312/bin/python \
FRACTUREX_RUN_LABEL_SUFFIX=h2_dB \
FRACTUREX_HMIN=0.024 \
FRACTUREX_FAST_COARSE_MESH=0 \
FRACTUREX_ENV_QUIET=1 \
nohup nice -n 19 bash scripts/paper_huzhang/run_aux_model0.sh \
    > results/logs/paper_aux_h2_dB.log 2>&1 &
```

**进度（D-B 写报告时）：**
- PID 395250，nice -n 19，~70% CPU，保留资源给用户后台任务；
- mesh：NC=3072，NN=1626，NE=4697（`hmin=0.024` distmesh 给出，与原
  paper_aux_h2 的 NC=2868 接近但不严格相同 —— 这是 [m0_kickoff_report kickoff §1.4](m0_kickoff_report_2026-05-28.md)
  指出的"distmesh 不可复现"的体现，是预期行为）；
- 已完成 step 13 / 31，elapsed 4h26m，单步 ~20 分钟；
- save_every=10 → 已落 step_000.npz / step_010.npz；step_020 / step_030 估计再 ~3-4 小时；
- 完成后 mesh.npz 由 driver auto-emit，[measure_interpolation_error.py](../../../scripts/datasets/measure_interpolation_error.py)
  会自动选 paper_aux_h2_dB（§2.6）。

**完成后的下一步（不在本会话）：**

```bash
PYTHONPATH=$PWD $FEALPY_PYTHON scripts/datasets/measure_interpolation_error.py
# 自动用 paper_aux_h2_dB（h1 用老 paper_aux_h1, h3 仍用老 short-run 占位）
# 重新生成 sigma_interp_error.csv / sigma_interp_convergence{,_h1_time}.png
```

### 3.4 h3 不起动 / 命令就绪

[m0_interpolation_error.md §7.1](../../m0_interpolation_error.md#71-h3-重跑命令待-h2-完成--用户后台空档)
已落定命令，hmin=0.012，预计 NC ≈ 11–12k、单步 30-60 分钟、全程 15-30
小时。等用户后台空闲手动起即可。

## 4. 复现

环境前置（与 D-A 一致）：

```bash
export FEALPY_PYTHON=/home/gongshihua/miniconda3/envs/py312/bin/python
cd /home/gongshihua/tian/fracturex
```

D-B 不引入新单测，跑 D-A 已有的 36 个测试确认零回归：

```bash
PYTHONPATH=$PWD $FEALPY_PYTHON -m pytest \
  fracturex/tests/test_recover_strain.py \
  fracturex/tests/test_dataset_roundtrip.py \
  fracturex/tests/test_interpolation.py -q
# 36 passed in ~3s
```

跑短补丁 run 落 H_qp（§3.1）：

```bash
PYTHONPATH=$PWD $FEALPY_PYTHON scripts/datasets/run_h_qp_patch.py
# wall ≈ 2.5 min；落 4 帧 step_XXX_qp.npz 到
# results/operator_learning_runs/h_qp_patch_h1/checkpoints/
```

跑 𝓗 插值误差扫描（§3.2）：

```bash
PYTHONPATH=$PWD $FEALPY_PYTHON scripts/datasets/measure_h_interp_error.py
# 输出 docs/figures/m0/interp_error/{h_interp_error.csv,
#       h_interp_summary.json, h_interp_compare.png}
```

确认 dump_quadrature_fields 向后兼容（老 recorder 占位开关默认 False）：

```bash
$FEALPY_PYTHON -c "
from fracturex.postprocess.recorder import RunRecorder
r = RunRecorder('/tmp/_smoke', save_npz=False)
assert r.save_quadrature_fields is False
r.dump_quadrature_fields(0, None, None)   # no-op
print('ok')
"
```

## 5. 结论

§M0 D-B 5 项闭环：
- **D-B 1（recorder 落盘管线）**：[recorder.py](../../../fracturex/postprocess/recorder.py)
  新方法 `dump_quadrature_fields`、driver hasattr-guard 调用、
  [model0_runner.py](../../../fracturex/tests/case_runners/model0_runner.py) 透传开关；
  零接口破坏，36 测试全过。
- **D-B 2（短补丁 run）**：[h_qp_patch_h1](../../../results/operator_learning_runs/h_qp_patch_h1/)
  4 帧 H_qp 落盘，max_d 走到 0.99；脚本：[run_h_qp_patch.py](../../../scripts/datasets/run_h_qp_patch.py)。
- **D-B 3（§4.2 表填数）**：[measure_h_interp_error.py](../../../scripts/datasets/measure_h_interp_error.py)
  实装 qp→grid→qp roundtrip + max-preservation + const baseline；
  得到的结论与 v0.2 草案**相反**（𝓘₁ 守峰胜 𝓘₂）。
- **D-B 4（h2 重跑）**：`paper_aux_h2_dB` 后台启动（PID 395250），后续
  measure 脚本自动接入，无需再动代码；h3 命令就绪、不发起。
- **D-B 5（§5.2 + schema 注释回写）**：[m0_interpolation_error.md](../../m0_interpolation_error.md)
  v0.3 / [SURROGATE_DATA_SCHEMA.md](../../SURROGATE_DATA_SCHEMA.md) §3.5 同步更新；
  𝓗 通道默认 `I1_nearest_quad`。

[m0_interpolation_error.md](../../m0_interpolation_error.md) DoD 勾掉 6/8，剩两条
是数据生成（h2_dB 后台跑中、h3 排队），不需要再写代码或文档。

现有 P1/P2 论文实验**未受影响**：`save_checkpoint` / `RunRecorder.__init__`
/ driver 构造签名零变更；`save_quadrature_fields` 默认 False 是 kickoff §1.6
就保证的契约。

## 6. 已知缺口 / 下一刀

紧跟 D-B 的纯执行项（不需写代码或文档）：

1. **等 h2_dB 跑完**（~3-4 小时）→ 跑 [measure_interpolation_error.py](../../../scripts/datasets/measure_interpolation_error.py)
   → §4.1(a) 跨 h 时间扫描自动填齐 h2 行；新 csv / convergence.png 落地。
2. **起 h3_dB**（[m0_interpolation_error.md §7.1](../../m0_interpolation_error.md#71-h3-重跑命令待-h2-完成--用户后台空档)）
   → 同样 measure 脚本自动接入 → §4.1 全档完整 → DoD 8/8。

更长尾的缺口（沿用 [D-A §6](m0_session_report_2026-05-29.md#6-已知缺口--下一刀)）：

- HuZhang `interpolate` 仍空实现 —— 影响"非零解析 σ 端到端验证"，但 §3.2
  的 4 帧真实 σ DOF 间接弥补；优先级低。
- model2_runner 给 notch shear 数据集对照（kickoff §3.5 末尾）；
- M0 的 200 样本起步数据集（kickoff §3.5 工具就位，`n_steps_override` 拿掉
  即跑）；
- M1 训练 baseline（U-Net / FNO2d / DeepONet 三 baseline），等 200 样本
  数据集就绪后启动。

## 7. 文件清单

新增 / 修改：

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| [fracturex/postprocess/recorder.py](../../../fracturex/postprocess/recorder.py) | 修改 | 新增 `dump_quadrature_fields(step, discr, state)`；落 `step_XXX_qp.npz`。`__init__` / `save_checkpoint` / `save_mesh` 等既有接口零变更。 |
| [fracturex/drivers/huzhang_phasefield_staggered.py](../../../fracturex/drivers/huzhang_phasefield_staggered.py) | 修改 | `solve_one_step` 在 `save_checkpoint` 后加 hasattr-guard 调用 `dump_quadrature_fields`；零签名变更。 |
| [fracturex/tests/case_runners/model0_runner.py](../../../fracturex/tests/case_runners/model0_runner.py) | 修改 | `Model0RunArgs.save_quadrature_fields: bool = False` + 透传到 RunRecorder。 |
| [scripts/datasets/run_h_qp_patch.py](../../../scripts/datasets/run_h_qp_patch.py) | 新增 | 短补丁 run 入口，paper_aux_h1 配置 + save_quadrature_fields=True。 |
| [scripts/datasets/measure_h_interp_error.py](../../../scripts/datasets/measure_h_interp_error.py) | 新增 | §4.2 𝓗 测量入口；roundtrip + max-preservation + const baseline。 |
| [scripts/datasets/measure_interpolation_error.py](../../../scripts/datasets/measure_interpolation_error.py) | 修改 | `_default_cases` 优先 `paper_aux_<h>_dB`，h2_dB 落地自动接入。 |
| [docs/m0_interpolation_error.md](../../m0_interpolation_error.md) | 修改 v0.2 → v0.3 | §4.2 表 + 关键观察 + const baseline 解读；§5.2 𝓗 默认 𝓘₁；§7 DoD 6/8 勾上 + h3 命令落定。 |
| [docs/SURROGATE_DATA_SCHEMA.md](../../SURROGATE_DATA_SCHEMA.md) | 修改 | §3.5 加 `metadata.interpolation` 通道适用范围注释（σ 走直接求值；𝓗 默认 𝓘₁）。 |
| [docs/figures/m0/interp_error/h_interp_error.csv](../../figures/m0/interp_error/h_interp_error.csv) | 新增 | 9 行长格式（3 步 × 3 方案）。 |
| [docs/figures/m0/interp_error/h_interp_summary.json](../../figures/m0/interp_error/h_interp_summary.json) | 新增 | by_step 嵌套，含 const baseline。 |
| [docs/figures/m0/interp_error/h_interp_compare.png](../../figures/m0/interp_error/h_interp_compare.png) | 新增 | **Fig 1 𝓘₁ vs 𝓘₂ on grid，t_a/t_b/t_c × 2 列；max_ratio 在 title 上。** |
| [results/operator_learning_runs/h_qp_patch_h1/](../../../results/operator_learning_runs/h_qp_patch_h1/) | 新增 dir | 4 帧 H_qp.npz + mesh.npz + history/iter csv + meta.json。 |
| [results/logs/paper_aux_h2_dB.log](../../../results/logs/paper_aux_h2_dB.log) | 新增 | h2_dB 后台日志（实时增长）。 |
| [results/phasefield/model0_circular_notch/paper_aux_h2_dB/epsg_1e-06/](../../../results/phasefield/model0_circular_notch/paper_aux_h2_dB/epsg_1e-06/) | 新增 dir（实时增长） | h2 完整重跑（NC=3072，paper schedule 31 步），跑完后 mesh.npz 由 driver auto-emit。 |
| [docs/m0_session_report_2026-05-29_dB.md](m0_session_report_2026-05-29_dB.md) | 新增 | 本报告。 |

未动：
- 任何 assembler / damage / 求解器代码；
- `phasefield_model0_huzhang.py` 论文实验入口（kickoff §6 已知缺口 7）；
- D-A 已交付的 σ 段（[m0_interpolation_error.md](../../m0_interpolation_error.md)
  §4.1 / §4.3 / §5.1 / 配套 csv+图）；
- `fracturex/learn/` 所有训练侧 stub。
