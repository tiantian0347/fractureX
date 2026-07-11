# M3b Stage D 进度：平衡残差正则 + σ_h vs σ_h^rec 对照

> 状态：**已上线, sweep running**（2026-07-05）。规划见
> [paper_thesis.md §F.3 / §G](paper_thesis.md)、
> [plan_operator_learning.md](plan_operator_learning.md) §Stage D。
>
> 本页只记 **代码/数据管线** 与 **首轮 sweep 触发点**；训练指标结果落到
> `results/learn/m3b_hz_sweep/*/eval_report.md`，跑完后回填到 §5。

## 1. 代码落地（本地 + 服务器同步完成）

新增/修改（fracturex/learn/）：

| 文件 | 内容 | 备注 |
| --- | --- | --- |
| `stress_recovery.py` (新) | σ_h^rec = g(d)·C·ε(u_h) 恢复算子；plane_strain / plane_stress C；central-diff ε | **backend-agnostic**：`fealpy.backend.backend_manager as bm`，可 numpy/torch/jax 切换 |
| `datasets.py` | `DatasetConfig.include_stress_rec` 开关 + `target_stress_rec` extractor + `collate_masked` 增补 `stress_rec` batch key | 兼容旧 npz（无 `stress_rec` 时静默降级） |
| `train.py` | `TrainConfig.supervision_source ∈ {'sigma_h', 'sigma_h_rec'}` + `lambda_eq` Stage D 平衡项；`_compute_loss` 按 supervision 分发 target；`_make_loader` 自动带 include_stress_rec | Stage D 项：`total += λ · r_h²`（作用于物理 σ，先反 σ_forward） |
| `losses.py` | `equilibrium_residual_fd(sigma_pred, mask, dx, dy, d, d_c)` — torch autograd 版 R_h | 中心差分 divergence；边界与 `d > d_c` 排除 |
| `eval/metrics.py` | `equilibrium_residual_l2` — numpy 版 R̃_h（无量纲，scale-free） | 与 `equilibrium_residual_fd` 走同一 divergence 定义 |

三个测试文件 27 tests 全过（服务器 py312, 15.29 s）：

- `tests/test_learn_stress_recovery.py` (10) — bm 版本，含 plane-strain λ+μ 检验、纯剪应变、退化 g(d) 缩放、stress_scale
- `tests/test_learn_losses_equilibrium.py` (9)  — autograd 检验、mask 对齐 rank-mismatch
- `tests/test_learn_metrics_equilibrium.py` (8) — 常应力 R̃_h≈0、bounding box L 推断

## 2. 数据管线（M3b.4 完成 ✅）

**已有**：m1_pilot（27 样本，64×64，train=19/test=8）已经带 `stress` = σ_h（HZ 平衡应力）。
✅ **A / A' 组可以直接跑**（用现有 `stress` key）。

**M3b.4 完成 (2026-07-05)**：`scripts/datasets/add_stress_rec.py` 已上线，为每个样本
从 FE displacement 生成 σ_h^rec 并追加到 `sample_XXXXXX.npz` 的 `stress_rec` key（+ `stress_rec_scale` 归一化常数）。

**关键实现细节**：
- u_h 位置：`results/datasets/m1_pilot/runs/sample_XXX/checkpoints/step_XXX.npz` 的 `u` 键
- FE 空间：`TensorFunctionSpace(LagrangeFESpace(mesh, p=p_sigma-1, ctype='D'), shape=(-1, 2))`
  即 **P2-DG × 2 分量**（p_sigma=3 时 gdof = 6·NC·2 = 2460）
- DOF 布局：`u_dof.reshape(NC, 6, 2)`（cell → local_ldof → component 顺序，
  经 fealpy `tspace.interpolate` 已知位移的 round-trip 验证，误差 4e-16）
- fealpy DG P2 basis 局部顺序：`[v0, mid(v0v1), mid(v0v2), v1, mid(v1v2), v2]`（不是 vertex-first）
- 网格点定位：自写向量化 barycentric point-locate（`mesh.location` 是 fealpy 空占位）
- 归一化：σ_h^rec / p95(|σ_h^rec|)_收敛步 → O(1) 训练空间；`stress_rec_scale` 存 npz
- 非收敛步护栏：`step_converged=0` 的步 stress_rec 用 σ_h（`stress` key）覆盖，
  避免 P2-DG u_h 在裂尖的巨大跳跃（step 5 时 u_grid ~10⁴）污染 loss

**观测到的现象** (m1_pilot sample_000000):
- 收敛步（7/16）: σ_h p95=0.86, σ_h^rec p95=1.0（都 O(1) 训练空间）
- 但 σ_h^rec **max=9780**（裂尖单像素峰值 10⁴× 于 p95！）— 这就是 σ_h^rec ∉ H(div,S) 的
  **法向跳跃 pathology**，正是 paper_thesis §F.3 想暴露的对照点
- 训练时用 `--sigma-transform arcsinh` 可压缩这种重尾

## 3. 服务器首轮 sweep（A / A' 组）

已启动 nohup（服务器 lab 端，2026-07-05 10:57），跑完约需数小时（CPU 上让路 D12 aux h2/h3 + model2 direct）。

```bash
# 复现命令
source ~/miniconda3/etc/profile.d/conda.sh && conda activate py312
cd ~/tian/fracturex
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$PWD nice -n 10 \
  python -u scripts/datasets/run_m3b_lambda_eq_sweep.py \
    --dataset-dir results/datasets/m1_pilot \
    --out-dir     results/learn/m3b_hz_sweep \
    --epochs 100 --batch-size 4 \
    --lambda-eq 0 0.01 0.1 1.0 \
  > /tmp/m3b_hz_sweep.log 2>&1 &
```

配置（每 sweep 点）：`multioutput_fno`, Stage B, HZ 监督 (`supervision_source='sigma_h'`),
`lambda_sigma=1.0`, epochs=100, batch=4, lr=1e-3, seed=0, CPU。

产物：
```
results/learn/m3b_hz_sweep/
├── sigma_h_leq0/           # A 组 baseline (λ_eq=0)
│   ├── config.json
│   ├── metrics.csv
│   ├── eval_report.md
│   └── checkpoints/model_final.pt
├── sigma_h_leq0p01/        # A' λ_eq=0.01
├── sigma_h_leq0p1/         # A' λ_eq=0.1
├── sigma_h_leq1/           # A' λ_eq=1.0
└── sweep_summary.json      # 4-row 汇总
```

## 4. 观测口子

sweep 跑完后从 `eval_report.md` 抓这些指标做对比表：

- `sigma_relative_l2`（物理空间 σ 误差；反 arcsinh 之后）
- `sigma_peak_relative_l2`（q=95% 峰值处 σ 误差）
- `principal_stress_l2`（主应力误差）
- `peak_load_error`（反力峰值误差）
- `relative_l2` (d)、`crack_set_iou`、`crack_front_hausdorff`（damage 副产品）

期望：λ_eq 增大 → sigma_relative_l2 略微上升（正则拉离数据拟合），但
**equilibrium residual 应该单调下降**（Stage D 的物理约束在起作用）。若 d 或 σ 指标崩了，
说明 λ_eq 尺度太大 / 需要 log-space。

## 5. 结果

> **两轮**：§5.0–§5.2 是 **v1 首轮（2026-07-06，整体不可用）**；§5.3 是
> **v3 修复重跑（2026-07-08，可用）**——按 §5.2 postmortem 修了 baseline 配置
> + 无量纲 λ_eq 重标尺，结论看 **§5.3**。

### 5.0 v1 首轮（2026-07-06，**整体不可用，需重跑**）

配置：m1_pilot（19 训练 / 8 测试）、`multioutput_fno`、100 epoch、batch=4、lr=1e-3。

**A/A' (sigma_h supervision, no arcsinh)**：

| λ_eq | rel_l2(d) | crack IoU | Hausdorff | σ rel_l2 | σ_peak rel_l2 | peak_load | σ_train |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0    | 1.05 | 0.018 | 51.9 | 0.986 | 0.989 | 0.445 | 0.986 |
| 0.01 | 1.18 | 0.024 | 51.1 | 0.988 | 0.991 | 0.443 | 0.988 |
| 0.1  | **2.49** | 0.005 | 30.0 | 0.988 | 0.991 | 0.443 | 0.988 |
| 1.0  | 1.41 | **0.00** | nan  | 0.988 | 0.991 | 0.449 | 0.988 |

**B/B' (sigma_h_rec supervision, arcsinh)**：

| λ_eq | rel_l2(d) | crack IoU | Hausdorff | σ rel_l2(物理) | σ_peak rel_l2 | peak_load | σ_train |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0    | 1.11 | 0.032 | 45.8 | 0.99 | 0.99 | 0.45 | 0.920 |
| 0.01 | 1.22 | 0.00 | nan | **99.05** | 17.05 | **1724** | **3.54** |
| 0.1  | 1.22 | 0.00 | nan | **99.08** | 17.05 | **1727** | **3.58** |
| 1.0  | 1.23 | 0.00 | nan | **99.09** | 17.05 | **1727** | **3.59** |

### 5.1 判读（诚实、含否定结论）

1. **λ_eq=0 baseline 就已经废了，Stage D 消融失去参考**。A/A' 的 rel_l2(d)=1.05 / IoU=0.018 对比
   [m1_results §3](m1_results.md) 里 fno Stage A 的 rel_l2=0.645 / IoU=0.148、
   [m2_stageB §2](m2_stageB_results.md) 里 `multioutput_fno` 的 rel_l2=0.655 / IoU=0.177 —— 同数据集、
   同 backbone，**damage baseline 差了一倍**。主要嫌疑：epochs 100 vs pilot 用的 300；
   其次是本轮 `train.py` / `datasets.py` 的 Stage D 改动可能引入了对旧路径的 regression。
   在 λ_eq=0 对不上历史 baseline 之前，λ_eq 消融读数无效。
2. **A/A' λ_eq↑ → damage 单调恶化**（1.05→2.49→1.41），σ train ≈ test ≈ 0.987，
   说明 σ head 根本没在学，λ_eq 的梯度只是在搅乱 damage head。与 §4 预期"σ 略升、
   R̃_h 单调下降"完全相反。
3. **B/B' λ_eq=0.01 就炸**：`sigma_relative_l2_train=3.54`——**训练集就已经跑飞**，不是过拟合，
   是优化到坏 basin。且 0.01 / 0.1 / 1.0 三档几乎同一个坏解（σ rel_l2 都 ~99），
   说明 0.01 已经完全主导，λ_eq 粒度太粗。§4 末尾"若 σ 崩了 → λ_eq 尺度太大 / 需 log-space"
   的预警实测坐实。
4. B/B' λ_eq=0 baseline 也差（rel_l2 1.11 / IoU 0.032），说明 rec 组的问题不只是 λ_eq——
   `stress_rec` supervision + fno 在 19 样本上本身就不 work，需要单独 σ_h vs σ_h^rec
   backbone 消融，不能混进 Stage D 一起看。

### 5.2 postmortem（下一轮要修的三件事）

- **修 baseline 再谈消融**：λ_eq=0 那点必须复现 m2_stageB §2 的数字。先把 epochs 100→300，
  或换回 pilot §3.3 的 arcsinh + `multioutput_unet` 组合，确认本轮 train.py 改动没引 regression。
- **λ_eq 需要无量纲重标尺**：当前 `L_eq = ||R_h||²` 走物理 σ，量纲远大于 `L_d + λ_σ L_σ`。
  改成 `λ_eq · ||R_h||² / (||σ_h||² + ε)` 无量纲相对残差，或从 λ_eq ∈ {1e-6, 1e-4, 1e-2}
  重扫；本轮 {0.01, 0.1, 1.0} 三档都在饱和区，读不出趋势。
- **rec 组残差在 arcsinh 空间算**：不要反 arcsinh 再算 R_h，直接在 arcsinh 空间做，
  与 σ supervision loss 同空间，避开 §2 里 10⁴× 裂尖 pathology 通过 `sinh` 引爆。

产物路径：`results/learn/m3b_hz_sweep/sweep_summary.json`、
`results/learn/m3b_rec_sweep/sweep_summary.json`，每点 `eval_report.md` 齐。

### 5.3 v3 修复重跑（2026-07-08，**可用**）

按 §5.2 三条 postmortem 落地后重跑 A/A'（B/B' 待后续）：

- **修 baseline**：`multioutput_unet` + `--sigma-transform arcsinh` + **300 epoch**
  （对齐 [m2_stageB §3.3](m2_stageB_results.md) pilot 最优配置；`run_m3b_lambda_eq_sweep.py`
  默认已改为 unet/300/arcsinh）。
- **无量纲 λ_eq**：`train.py::_compute_loss` 的 Stage D 项改成
  `total += λ_eq · (R_h / (σ_ref/L))²`，σ_ref = HZ 目标 σ 的 masked RMS（detached 常数）。
  见 [train.py](../../fracturex/learn/train.py) §Stage D balance regularization。
- **R̃_h 落 metrics.csv**：每 epoch 记 `train_l_eq`, `train_l_eq_norm`, `train_sigma_ref`。

**A/A' (sigma_h supervision, arcsinh, 300ep)**，held-out test=8：

| λ_eq | rel_l2(d) | crack IoU | Hausdorff | σ rel_l2 | σ_peak rel_l2 | peak_load | σ_train |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **0** | **0.515** | **0.382** | **19.8** | 0.981 | 0.985 | 0.393 | 0.866 |
| 0.01 | 0.603 | 0.224 | 21.4 | 0.985 | 0.989 | 0.440 | 0.888 |
| 0.1  | 0.801 | 0.003 | 54.9 | 0.988 | 0.991 | 0.525 | 0.906 |
| 1.0  | 0.808 | 0.000 | nan  | 0.989 | 0.992 | 0.536 | 0.910 |

**R̃_h 收敛（`train_l_eq_norm`，epoch 0 → 299）**，图见 `rh_descent.png`：

| λ_eq | R̃_h 起 | R̃_h 终 |
| --- | --- | --- |
| 0.01 | 21.6 | **1.22** |
| 0.1  | 19.6 | **0.295** |
| 1.0  | 20.1 | **0.171** |

**判读（诚实，含否定结论）**：

1. **两条修复都生效**：(a) baseline λ_eq=0 复现了 pilot（rel_l2(d)=0.515 vs m2_stageB §3.3
   的 0.509，IoU 0.382 vs 0.429），消融**有了参考**；(b) 无量纲重标尺后 σ_train 全程
   0.87→0.91，**不再有 v1 那种 3.5/99 爆飞**——§5.2 第二条"量纲"诊断坐实并解决。
2. **R̃_h 单调下降、λ_eq 越大终值越低**（1.22 / 0.295 / 0.171）——Stage D 物理约束
   **确实在压平衡残差**，σ_h ∈ H(div,S) 可平衡，符合 §6"descent 分支"预期。这是 §F.3 的正面料。
3. **但 19 样本下 λ_eq 净有害**：damage 随 λ_eq **单调恶化**（IoU 0.382→0.224→0.003→0），
   σ 指标几乎不动（head 没在学，σ rel_l2 恒 ~0.98）。即 **R̃_h 的梯度只是在搅乱 damage head，
   平衡增益 < 数据拟合代价**。与 §4"σ 略升、R̃_h 下降"的预期在 R̃_h 侧吻合、在 σ 侧不吻合
   （σ 在 pilot 数据量下压根没学出，见 m2_stageB §3.2）。
4. **结论**：Stage D 正则**机制正确**（R̃_h descent 可复现），但要看到"物理约束换来 σ/泛化提升"
   的净收益，得先有一个 **σ 真的学出来的 baseline**（m2_stageB §6：S 档 ~1k 样本 σ rel_l2
   0.98→0.33）。在 19 样本 pilot 上做 λ_eq 消融只能得到"正则伤 damage"的否定结论。

**下一步**：把这套修好的配置（unet/arcsinh/300ep/无量纲 λ_eq）搬到 **S 档 ~1152 样本**上
重扫 λ_eq，才是能读出"平衡正则净收益"的场景；pilot 只用于确认机制 + 出 R̃_h descent 图。

产物路径：`results/learn/m3b_hz_sweep_v3_full/{sweep_summary.json, rh_descent.png}`，
每点 `{config.json, metrics.csv, eval_report.md, checkpoints/model_final.pt}`。

### 5.4 B/B' rec 组（2026-07-11，**完成**）— 补齐 §F.3 4×4 矩阵

同 §5.3 修好的配置（`multioutput_unet` + arcsinh + 300ep + 无量纲 λ_eq），
但监督换成 **σ_h^rec**（`supervision_source='sigma_h_rec'`，残差在 arcsinh 空间算，见 §5.2 第三条），
m1_pilot（19 训练 / 8 测试）：

**B/B' (sigma_h_rec supervision, arcsinh, 300ep)**，held-out test=8：

| λ_eq | rel_l2(d) | crack IoU | Hausdorff | σ rel_l2 | σ_peak rel_l2 | peak_load | σ_train |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **0** | 0.559 | 0.352 | 21.0 | 0.987 | 0.990 | 0.519 | 0.900 |
| 0.01 | 0.588 | 0.253 | 20.6 | 0.990 | 0.993 | 0.543 | 0.921 |
| 0.1  | 0.796 | 0.016 | 53.5 | 0.991 | 0.993 | 0.539 | 0.925 |
| 1.0  | 0.810 | 0.000 | nan  | 0.996 | 0.994 | 0.722 | 0.968 |

**§F.3 4×4 矩阵（{HZ, rec} × {λ_eq}）— R̃_h 终值（`train_l_eq_norm` @ epoch 299）**：

| supervision | λ_eq=0.01 | λ_eq=0.1 | λ_eq=1.0 |
| --- | --- | --- | --- |
| **HZ (σ_h)**    | 1.22 | 0.295 | 0.171 |
| **rec (σ_h^rec)** | 0.679 | 0.264 | 0.154 |

**判读（诚实，含否定结论）**：

1. **"plateau vs descent" 假设被证伪（在 pilot 尺度）**：§6 预期 rec 组 R̃_h 应 plateau 在
   Θ(h^m)（法向跳跃阻止 → 0），HZ 组降到噪声。实测 **rec R̃_h 和 HZ 一样单调降到 O(0.1)，
   甚至更低**（0.679/0.264/0.154 vs HZ 1.22/0.295/0.171）——**没有 plateau**。
2. **原因**：R_h 是在 **网络预测的 σ** 上算 divergence，不是在真 σ_h^rec 上算。网络在 19 样本下
   把 σ 压平（§5.3 判读 3：σ head 根本没学出，σ rel_l2 恒 ~0.99），**平滑预测的 divergence
   天然小** → R̃_h 能降。σ_h^rec 的 10⁴× 裂尖法向跳跃 pathology **压根没进到网络输出里**，
   所以 §F.3 想暴露的"rec 不可平衡"对照 **在 σ 没学出来的 pilot 上看不到**。
3. **damage 侧与 HZ 同构**：λ_eq↑ → damage 单调恶化（IoU 0.352→0.253→0.016→0），
   rec baseline（λ_eq=0）也和 HZ baseline 一个量级（rel_l2 0.559 vs 0.515，IoU 0.352 vs 0.382）——
   即 rec vs HZ 监督在 pilot 上 **damage 没有可区分差异**，σ 也都没学出。
4. **结论**：4×4 矩阵在 pilot 上填满了，但**只能给否定结论**——plateau 对照要成立，
   前提同 §5.3 结论 4：得先有 **σ 真学出来的 baseline**（S 档），
   让 σ_h^rec 的裂尖跳跃真进到网络输出，R_h 才可能 plateau。**pilot 尺度下 HZ 与 rec 无法区分**。

产物路径：`results/learn/m3b_rec_sweep/sigma_h_rec_leq{0,0p01,0p1,1}/`，
每点 `{config.json, metrics.csv, eval_report.md, checkpoints/model_final.pt}`。
（注：per-point 并行启动导致 `sweep_summary.json` 被互相覆盖，只剩 leq1 一行；4 点读数取自各
`eval_report.md`。）

## 6. 下一步

### M3b.4 已完成 ✅

`scripts/datasets/add_stress_rec.py` — server 上后台跑 27 样本批量生成
（2026-07-05 13:37，nohup，log `/tmp/add_stress_rec.log`）。

### M3b.5 B/B' 组（待 A/A' 完 + 批量完 之后跑）

```bash
# 单独 B 组（σ_h^rec supervision，λ_eq=0 baseline）
python scripts/datasets/run_m3b_lambda_eq_sweep.py \\
  --dataset-dir results/datasets/m1_pilot \\
  --out-dir     results/learn/m3b_rec_sweep \\
  --supervision sigma_h_rec \\
  --sigma-transform arcsinh \\   # 压缩裂尖 10⁴× 重尾
  --epochs 100 --lambda-eq 0 0.01 0.1 1.0
```

### 后续实验设计（paper §F.3 / §G）

"plateau vs descent" 图（R̃_h vs epoch）：
- HZ (A/A') 组 R̃_h 应下降到训练噪声（σ_h ∈ H(div,S) 可平衡）
- σ_h^rec (B/B') 组 R̃_h 应 plateau 在 Θ(h^m)（法向跳跃阻止 R̃_h → 0）

指标产出：
- 每个 sweep 点 `sweep_summary.json` 里的 `sigma_relative_l2`, `sigma_peak_relative_l2`
- 4×4 矩阵（{HZ, rec} × {λ_eq=0, 0.01, 0.1, 1.0}）
- R̃_h 收敛曲线对比图（附录 F.3 主图）
