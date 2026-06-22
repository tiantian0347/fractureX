# M2 Stage B 结果：damage + Hu-Zhang σ 监督（pilot）

> 状态：**pilot 小规模实验**（2026-06-01），同 [m1_results.md](m1_results.md) 的
> `m1_pilot` 数据集（27 样本，train=19/test=8）。Stage B 加入应力监督
> `L = L_d + λ_σ L_σ`（λ_σ=1），多输出模型每步预测 4 通道 `(d, σxx, σyy, σxy)`。
> 规划见 [plan_operator_learning.md](plan_operator_learning.md) §M2 Stage B。
>
> **更新**：§1–§5 是 27 样本 pilot（σ 没学出来）；**§6 S 档 ~1152 样本把 σ rel-L²
> 0.98→0.33（数据量是关键）**；§7 损失消融 + ~5000× 速度；**§8 分辨率消融（64 vs 128）+
> peak-load**。当前结论看 **§8**：裂尖逐点 σ 峰值对损失与分辨率都鲁棒（已知局限），但
> **peak-load ~8% 已可用**。

## 1. 设置
- 模型：`multioutput_fno`、`multioutput_unet`（FNO/U-Net backbone，输出 (T,4,H,W)）。
- 损失：masked relative-L² 的 `L_d + λ_σ L_σ`，λ_σ=1。
- 训练：300 epoch，batch 4，lr 1e-3，CPU；留出 test=8。
- σ 目标：npz 内已按**每样本** `stress_scale`（末帧 95 分位）归一化，直接监督。
- 驱动：`scripts/datasets/run_m1_experiment.py --stage B`。

## 2. 对比表（held-out test）

| model | rel_l2(d) ↓ | rel_h1(d) ↓ | crack IoU ↑ | Hausdorff ↓ | SSIM ↑ | σ rel_l2 ↓ | σ₁(主应力) rel_l2 ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multioutput_unet | **0.537** | **0.564** | **0.373** | **20.67** | **0.744** | 0.983 | 0.983 |
| multioutput_fno | 0.655 | 0.668 | 0.177 | 45.63 | 0.580 | 0.985 | 0.984 |

## 3. 关键发现

### 3.1 σ 监督**改善了 damage**（多任务正则）
对比 M1 Stage A（仅 d）→ M2 Stage B（d+σ），U-Net 的 damage 指标全面变好：

| U-Net | rel_l2(d) | crack IoU | Hausdorff | SSIM |
| --- | --- | --- | --- | --- |
| M1 (Stage A) | 0.564 | 0.334 | 44.3 | 0.688 |
| **M2 (Stage B)** | **0.537** | **0.373** | **20.7** | **0.744** |

Hausdorff 几乎减半、IoU/SSIM 提升——应力监督给了 damage 额外的物理结构信号。
这正是用 Hu-Zhang 高质量 σ 数据的卖点雏形。

### 3.2 σ 本身：**结构学到了，峰值没学到**（诚实标注）
σ rel-L² ≈ 0.98 表面像"没学会"，但诊断（test 集，multioutput_fno）揭示更细的图景：

- **corr(pred σ, tgt σ) = 0.84** —— 应力**空间分布**学得不错；
- **‖pred σ‖ / ‖tgt σ‖ = 0.06** —— 但**幅值严重低估**，预测几乎被压平；
- 目标 σ **重尾**：归一化后 `|σ|` median=0.11、p95=14.3、**max=344**——裂尖应力奇异性
  使 masked relative-L² 被极少数尖峰像素主导，平滑网络在 19 样本下无法拟合该峰值。
- test 8 样本 `step_converged=100%`，**排除**了末尾非收敛 σ 噪声的影响（见 m1_results §1）。

**结论**：Stage B 管线打通且 σ 的空间结构可学；当前 σ 精度低是**幅值/归一化问题**，
不是结构问题。

### 3.3 arcsinh 重尾压缩实验：**没解决物理 σ 精度**（诚实标注）
按 §3.2 的诊断，加了 `arcsinh` σ 变换（训练在 `asinh(σ)` 空间，评估 `sinh` 反变换回物理量；
`fracturex/learn/transforms.py`，`--sigma-transform arcsinh`），重跑 Stage B：

| σ 处理 | 模型 | rel_l2(d) | crack IoU | Hausdorff | SSIM | **σ rel_l2(物理)** | σ rel_l2(arcsinh空间) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | multioutput_unet | 0.537 | 0.373 | 20.7 | 0.744 | 0.983 | — |
| **arcsinh** | multioutput_unet | 0.509 | 0.429 | 18.7 | 0.766 | **0.983** | 0.871 |
| none | multioutput_fno | 0.655 | 0.177 | 45.6 | 0.580 | 0.985 | — |
| **arcsinh** | multioutput_fno | 0.807 | 0.156 | 20.7 | 0.477 | **0.985** | 0.887 |

诊断（arcsinh, multioutput_unet, 反变换回物理量）：**corr=0.88**（结构略升，0.84→0.88）、
**‖pred σ‖/‖tgt σ‖=0.064**（幅值仍被压平）。

**判读**：
- arcsinh 让损失**不再被裂尖奇异主导**（arcsinh 空间 rel-L²≈0.87，可优化），结构相关性略好；
- 但 `sinh` 反变换**重新放大**裂尖处的预测误差 → **物理 σ rel-L² 仍≈0.98，没改善**；
- damage 端略有提升（unet：IoU 0.37→0.43、Hausdorff 20.7→18.7），但 8 样本 test 集波动大，
  不宜过度解读。

**真正的瓶颈是数据量（19 训练样本），不是损失/归一化工程**：网络拟合不出裂尖应力奇异的
**幅值**，再怎么换变换，物理量 L² 都被那几个奇异像素支配。

## 4. 下一步（Stage B 精化，按优先级）
1. **扩样本量**（pilot 19 → S 档 ~1k）——预期是 σ 幅值能否学出的主因；
2. 峰值感知 / 逐像素相对的 σ 度量与损失（当前全局 L² 被奇异点支配，掩盖了 88% 的良好结构）；
3. σ 报告补反力曲线 / 积分边界反力（需导出 reaction 字段，Stage B 评估项之一）；
4. arcsinh 作为**可选项保留**（结构相关性略好、训练更稳），但不再期待它单独修复物理精度。

> 工程修正（本轮）：训练默认只存 `model_final.pt`（原先每 epoch 存 ~64 MB，300×多模型
> 一度撑爆磁盘）。曲线在 `metrics.csv`。

## 5. 复现
```bash
# 环境 py312 + PYTHONPATH=$PWD；数据集见 m1_results.md §6 第 1 步
python scripts/datasets/run_m1_experiment.py \
  --dataset-dir results/datasets/m1_pilot \
  --out-dir results/learn/m2_stageB --stage B --lambda-sigma 1.0 \
  --epochs 300 --test-frac 0.3 --seed 0
# arcsinh σ 变换版（§3.3）
python scripts/datasets/run_m1_experiment.py \
  --dataset-dir results/datasets/m1_pilot \
  --out-dir results/learn/m2_stageB_arcsinh --stage B --lambda-sigma 1.0 \
  --sigma-transform arcsinh --epochs 300 --test-frac 0.3 --seed 0
```
产物：`results/learn/{m2_stageB,m2_stageB_arcsinh}/{comparison_table.md, results.json,
training_curves.png, <model>/eval_report.md}`。

---

## 6. S 档（~1152 样本）重跑：**数据量是关键**

把 pilot 的 27 样本扩到 **S 档 1152 样本**（`scripts/datasets/configs/m2_S.json`：
circle_r×Gc×l0×cx×cy×E 笛卡尔积，几何+材料同扫；64×64，16 步 schedule），
32 shard 并行生成（每 shard 36，`--cleanup-runs` 控盘，~37 s/样本，**1152/1152 成功**），
train=806 / test=346。Stage B 训练 120 epoch、batch 16、CPU。

### 6.1 结果（held-out test=346）

| σ 处理 | 模型 | rel_l2(d) | crack IoU | Hausdorff | SSIM | **σ rel_l2** | σ₁ rel_l2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arcsinh | multioutput_unet | **0.276** | **0.563** | **7.95** | **0.931** | **0.334** | 0.322 |
| arcsinh | multioutput_fno | 0.298 | 0.545 | 8.21 | 0.930 | 0.341 | 0.328 |
| none | multioutput_unet | 0.335 | 0.416 | 10.9 | 0.878 | 0.340 | 0.327 |

### 6.2 pilot → S 档（multioutput_unet, arcsinh）

| 指标 | pilot (19 train) | **S 档 (806 train)** |
| --- | --- | --- |
| σ rel_l2（物理） | 0.983 | **0.334** |
| σ corr(pred,tgt) | 0.88 | **0.981** |
| ‖pred σ‖/‖tgt σ‖ | 0.064 | **0.491** |
| damage rel_l2 | 0.509 | **0.276** |
| crack IoU | 0.429 | **0.563** |
| Hausdorff | 18.7 | **7.95** |
| SSIM | 0.766 | **0.931** |

### 6.3 判读
- **σ 学出来了**：rel-L² 0.98→**0.33**，结构相关 0.88→**0.98**，幅值比 0.06→**0.49**。
  §3.2 的诊断成立——**数据量是真瓶颈**，不是损失/归一化工程。
- 裂尖峰值仍**约欠预测 2×**（幅值比 0.49），σ rel-L² 0.33 已是可用代理；进一步压低
  需更多数据 / 更高分辨率 / 峰值感知损失。
- **arcsinh 在大数据下更优**：damage rel-L² 0.276 vs none 0.335、IoU 0.563 vs 0.416、
  Hausdorff 7.95 vs 10.9；σ 两者接近（0.334 vs 0.340）。arcsinh 设为 Stage B 默认推荐。
- damage 全面跃升（SSIM 0.93、Hausdorff~8 px），σ 监督 + 数据量共同作用。

### 6.4 复现
```bash
# 并行生成 S 档（32 shard，控盘）
N=32
for k in $(seq 0 $((N-1))); do
  OMP_NUM_THREADS=2 PYTHONPATH=$PWD $FEALPY_PYTHON \
    scripts/datasets/generate_phasefield_dataset.py \
    --config scripts/datasets/configs/m2_S.json --dataset-dir results/datasets/m2_S \
    --num-shards $N --shard $k --cleanup-runs --skip-existing &
done; wait
python scripts/datasets/merge_shard_manifests.py --dataset-dir results/datasets/m2_S
# Stage B（arcsinh）
OMP_NUM_THREADS=32 PYTHONPATH=$PWD $FEALPY_PYTHON \
  scripts/datasets/run_m1_experiment.py --dataset-dir results/datasets/m2_S \
  --out-dir results/learn/m2_S_arcsinh --stage B --sigma-transform arcsinh \
  --models multioutput_unet multioutput_fno --epochs 120 --batch-size 16
```
产物：`results/learn/m2_S_arcsinh/`、`results/learn/m2_S_none/`。

---

## 7. σ 损失消融 + 速度：峰值卡在**分辨率**，不在损失

S 档 `multioutput_unet`，三臂消融（`losses.py::peak_weighted_relative_l2` 按 ‖σ‖ 给裂尖
加权 (1+α)×，α=4；新增**峰值区指标** `sigma_peak_relative_l2` = 仅 top-5% |σ| 像素的 rel-L²）：

| arm | σ rel_l2 | **σ_peak rel_l2** | σ₁ rel_l2 | rel_l2(d) | IoU | Hausdorff | SSIM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none (rel_l2) | 0.340 | 0.345 | 0.327 | 0.335 | 0.416 | 10.9 | 0.878 |
| arcsinh (rel_l2) | 0.334 | 0.350 | 0.322 | **0.276** | **0.563** | **7.95** | **0.931** |
| none (peak_weighted) | 0.340 | **0.335** | 0.329 | 0.314 | 0.488 | 9.30 | 0.894 |

**判读（决定性）**：
- **峰值感知损失只把峰值区 σ 误差从 0.345 压到 0.335（~3%）**，全局 σ rel-L² 三臂几乎不动
  （0.334–0.340）。把裂尖像素加权到 5× 仍推不动——**说明瓶颈不是损失**。
- 结合 §6.3 的幅值比 0.49（峰值欠预测 ~2×）：**64×64 网格表示不出亚像素的应力奇异峰值**，
  这是**分辨率/表达能力**的天花板，不是优化目标的问题。
- ~~所以"压峰值"的有效杠杆可能是 **B：提分辨率 64→128**~~ —— **§8 实测推翻：128 同预算
  幅值比仍 0.55、全场反而退化，分辨率也不是杠杆**。损失与分辨率都不修峰值。
- 副带：`peak_weighted` 比 none 的 damage 略好（IoU 0.42→0.49、Hausdorff 10.9→9.3），但
  **arcsinh 仍是 damage 最优**（rel-L² 0.276、IoU 0.563、Hausdorff 7.95）。综合默认仍取 arcsinh。

### 7.1 速度（"算得快"那一半）
代理一次前向即出**全 16 步 d+σ 序列**：

| | 墙钟 / 样本 | 说明 |
| --- | --- | --- |
| **代理推理**（unet, 64×64, CPU 8 线程） | **7.3 ms** | 一次 forward = 16 步 d+σ |
| FE 生成（同数据集） | ~37 s | Hu-Zhang 相场交错迭代 |
| **加速比** | **~5000×** | |

精度（σ rel-L² 0.33 / damage SSIM 0.93）+ 速度（~5000×）共同支撑代理的价值主张。

### 7.2 下一步
1. **128×128 分辨率**（lever B）——直接攻峰值欠预测，预期 σ_peak / 幅值比明显改善；
2. 裂尖局部加密网格 / 多尺度 head（若 128 仍不够）；
3. 导出 reaction 字段，补反力曲线 / 峰值载荷误差（Stage B 评估项）。

---

## 8. 分辨率消融（64 vs 128）+ peak-load：**分辨率也不是峰值的杠杆**

按 §7.2 的指向，把数据重生成到 **128×128**（`m2_S_128`，1152/1152，同时导出 reaction），
unet/arcsinh **同预算重跑**（120 epoch、batch 16、lr 1e-3，对齐 64 基线）：

| 指标 | 64×64 | 128×128 |
| --- | --- | --- |
| σ rel_l2 | **0.334** | 0.371 |
| σ_peak rel_l2 (top5%) | **0.350** | 0.402 |
| **‖pred σ‖/‖tgt σ‖** | **0.548** | 0.550 |
| corr(pred,tgt) | 0.982 | 0.971 |
| **peak_load_error** | **0.078** | 0.087 |
| rel_l2(d) | **0.276** | 0.363 |
| crack IoU | **0.563** | 0.392 |
| Hausdorff | **7.95** | 23.66 |
| SSIM | **0.931** | 0.869 |

**判读（诚实、含否定结论）**：
- **峰值幅值比 64↔128 几乎不变（0.548 vs 0.550）**，σ_peak 还略升——**提分辨率没有压下 σ 峰值欠预测**。
- 同预算下 128 **全面回退**（damage/σ/IoU/Hausdorff 都变差）：4× 像素 + 同样 120 epoch/lr ⇒
  **欠训练**（arcsinh 空间训练误差 0.272→0.308 略升，但泛化指标退得更多）。要公平评 128 需更多
  epoch + 调 lr，而这与"算得快"冲突（128 推理 ~4× 慢）。
- **合并 §7+§8 的结论**：残余的裂尖 σ 峰值欠预测（幅值比 ~0.5、σ rel-L² ~0.33）对**损失**和
  **分辨率**都鲁棒——不是这两者能修的，更像算子网络在奇异场上的**固有平滑/泛化**特性。要再压
  需换路子（更多数据 / 裂尖增强特征 / 奇异自适应基），不是调损失或加分辨率。

### 8.1 好消息：工程量 **peak-load 已可用（~8%）**
裂尖**逐点** σ 峰值欠预测 ~2×，但**积分边界力**（力-位移曲线的峰值载荷，最有工程说服力的单一量）
**两档都 ~8% 误差**（64: 0.078、128: 0.087）。即：

> 代理已能把**峰值载荷**预到 ~8%、damage 场 SSIM 0.93、σ 全场 rel-L² 0.33，**一次前向 7.3 ms
> vs FE 37 s（~5000×）**——力-位移预测层面**又快又准、即可用**；唯**逐点裂尖应力奇异峰**仍是已知短板。

### 8.2 结论与建议
- **留在 64×64**（同预算更优、推理 4× 便宜）；128 无收益、且未收敛。
- reaction 字段已导出（FE 精确力，`reaction (T,1)`，schema §3.2），peak-load 误差进 Stage B 评估。
- 裂尖逐点峰值列为**已知局限**；后续若要攻，走数据量 / 裂尖增强，不再碰损失与分辨率。
