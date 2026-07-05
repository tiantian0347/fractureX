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

## 5. 结果（跑完后回填）

（待 sweep 完成填充；模板见 [m2_stageB_results.md §2](m2_stageB_results.md)）

| λ_eq | rel_l2(d) | crack IoU | Hausdorff | σ rel_l2(物理) | σ peak rel_l2 | peak_load_error |
| --- | --- | --- | --- | --- | --- | --- |
| 0    | – | – | – | – | – | – |
| 0.01 | – | – | – | – | – | – |
| 0.1  | – | – | – | – | – | – |
| 1.0  | – | – | – | – | – | – |

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
