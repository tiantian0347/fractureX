# M1 结果：damage-only 三 baseline（pilot）

> 状态：**pilot 小规模实验**（2026-06-01）。目的是把 M1 管线（数据生成 → 三 baseline
> 训练 → 对比表/曲线）真实跑通并固化交付物格式，**不是** plan §M2 的 S 档（~1k 样本）。
> 规划见 [plan_operator_learning.md](plan_operator_learning.md) §M1；协议见
> [SURROGATE_DATA_SCHEMA.md](SURROGATE_DATA_SCHEMA.md)。

## 1. 数据集 `m1_pilot`

- 配置：`scripts/datasets/configs/m1_pilot.json`；几何 = Model0 圆缺口。
- 参数网格（笛卡尔积，27 组合）：
  `circle_r ∈ {0.15, 0.20, 0.25}` × `Gc ∈ {0.5, 1.0, 2.0}` × `l0 ∈ {0.015, 0.02, 0.025}`。
- 固定：`hmin=0.08`、`p_sigma=3`、`damage_p=2`、`elastic_mode=direct`。
- 加载：16 步裂纹 schedule `linspace(0,0.07,4) ∪ linspace(0.08,0.125,12)`（T=16）。
- 导出：64×64 结构网格，bbox `[0,1]²`，schema v0.1。
- 结果：**27/27 成功**，墙钟 ≈ 1743 s（~29 min，单核串行）。
- 统计：`max_damage` 跨样本 0.336–1.268；裂纹均能起裂传播。

> ⚠️ **数据质量标注（诚实）**：27 个样本中 **12 个 `max_damage > 1`**。这是末尾 1–2 个
> 加载步交错迭代未收敛（`staggered_maxit=50` 撞顶，交错起裂爆炸）导致的损伤过冲，
> 物理上 d 应 ∈ [0,1]。处理：
> - 非收敛步由 npz 的 `step_converged=0` 标记，保留以便人工核查；
> - 训练目标在消费端 `learn/datasets.py::target_damage` 统一 **clamp 到 [0,1]**，不让越界值进损失。

## 2. 训练设置

- 任务：Stage A（仅预测 damage），损失 `L_d` = masked relative L²。
- 方案 A（一次性预测全部 T 步为输出通道）：输入 `(4+k, 64, 64)`，输出 `(T=16, 64, 64)`。
- 优化：Adam，lr `1e-3`，batch 4，**300 epoch**，CPU（torch 2.12 CPU）。
- 划分：`train=19 / test=8`（`test_frac=0.3`, `seed=0`，已写入
  `dataset_manifest.json::splits`，可复现）。指标在**留出 test** 上评。
- 驱动：`scripts/datasets/run_m1_experiment.py`。

## 3. 对比表（held-out test，8 样本）

| model | relative_l2 ↓ | relative_h1 ↓ | crack_set_iou ↑ | crack_front_hausdorff ↓ | ssim ↑ |
| --- | --- | --- | --- | --- | --- |
| **U-Net** | **0.564** | **0.584** | **0.334** | **44.3** | **0.688** |
| FNO2d | 0.645 | 0.663 | 0.148 | 45.9 | 0.589 |
| DeepONet | 0.862 | 0.874 | 0 | nan | 0.172 |

训练曲线：`results/learn/m1_pilot/training_curves.png`（train/test loss，log 轴）。
U-Net train loss 4.33→0.347、test 2.59→0.458。

## 4. 观察

- **排序符合预期**（plan §3.9）：**U-Net > FNO2d > DeepONet**。U-Net 在局部 sharp front 上
  最强；DeepONet 作为弱理论 baseline，`crack_set_iou=0`（预测损伤几乎不过 0.5 阈值，
  故裂纹前沿空集、Hausdorff = nan）。
- **绝对精度偏低**（rel-L² ≈ 0.56）属正常：仅 19 训练样本、64×64、Stage A，是最低线对照，
  不是论文最终数。扩到 S 档（~1k 样本）应显著改善。

## 5. 已知局限 / 待办

- pilot 规模（27 样本）≪ plan S 档（~1k）。扩规模只需改 `m1_pilot.json` 的 grid 并重跑。
- `peak_load_error` / `equilibrium_residual` 未报：需 reaction / σ 字段，进 M2 Stage B/D。
- 驱动每 epoch 存一个 checkpoint（300/模型）——后续可改为只存 best/last。
- 末尾步非收敛（见 §1）：根因是交错迭代起裂爆炸，属求解器侧问题（Anderson 加速路线 B
  在跟进），与算子学习管线无关。

## 6. 复现

```bash
# 环境：conda env py312 + PYTHONPATH=$PWD（base 无 fealpy；torch CPU 已装）
# 1) 生成数据集（~29 min）
python scripts/datasets/generate_phasefield_dataset.py \
  --config scripts/datasets/configs/m1_pilot.json \
  --dataset-dir results/datasets/m1_pilot --skip-existing
# 2) 跑三 baseline + 出表/曲线（~12 min）
python scripts/datasets/run_m1_experiment.py \
  --dataset-dir results/datasets/m1_pilot \
  --out-dir results/learn/m1_pilot --epochs 300 --test-frac 0.3 --seed 0
```

产物：`results/learn/m1_pilot/{comparison_table.md, results.json, training_curves.png,
<model>/eval_report.md, <model>/metrics.csv}`。
