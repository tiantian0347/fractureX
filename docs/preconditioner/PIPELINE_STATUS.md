# 无人值守流水线状态 — aux_h2 ‖ aux_h3 ‖ model1（并行）

> 脚本 `scripts/paper_huzhang/pipeline_aux_h2h3_model1.sh`。**2026-06-04 22:42 改为三段并行**
> (旧版串行+100GB哨兵是针对小机器的过度保守;本机 2TB RAM / 176 核,三者峰值 RSS 合计
> 仅 ~25GB,无 OOM 风险,串行只白白浪费几十小时墙钟)。各段从自己的 checkpoint 续算,
> 末尾 monitor 每 30min 把进度写进本文件,某段完成时自动重绘对应 load-disp 图。
> 日志:各 run 的 `epsg_1e-06/resume.log` + `/tmp/pipeline.log`。

## 关键修复:SAVE_EVERY 1→ 每步存盘

旧版默认 `FRACTUREX_SAVE_EVERY=10`(每 10 步才存 checkpoint)。后果:
- **h2** history 已跑到 step15(maxd=0.997,**已穿过局部化墙**,step14/15 各烧 11h/32h),
  但 checkpoint 只到 step10。VTU 不含不可逆 H 场,无法从 step15 续 →
  **本次 resume 从 step10 重跑 11–15,认烧 ~43h**(已确认接受,无其他办法)。
- **model1** history 到 step54,checkpoint 到 step50 → 重跑 51–54,约 ~10h 重算。

新脚本统一 `FRACTUREX_SAVE_EVERY=1`,**以后绝不再丢局部化步**。

## 三段配置(均已离线验证 NC 匹配 checkpoint,且已成功 resume 开跑)

| run | case/mode | 网格 | σ-dof | resume@ | 续到 | 求解器 | loads |
|---|---|---|---|---|---|---|---|
| A | model0 / aux | hmin=0.025 NC=2868 | 48,092 | step10 | step11/31 | fast 两层 V-cycle | 31 |
| B | model0 / aux | hmin=0.013 NC=11034 | 183,524 | step10 | step11/31 | fast 两层 V-cycle | 31 |
| C | square(model1) / direct | nx=120 NC=28800 | 476,883 | step50 | step51/161 | pardiso(MKL 多线程) | 161 → u=6.1e-3 |

## 已知算法硬墙(诚实标注)

aux_h2/h3 续算要穿过**裂纹完全局部化区**(maxd→0.997,尖锐 d≈1/d≈0 界面):
aux_fast niter 在 maxd≤0.82 时恒 =7,局部化处**骤升到 ~95–121**(约 14×,单步弹性解
从 13s→几万秒)。这是 §3.2 加权 P1 粗空间对尖界面变难,合成均匀-d 基准没有此界面、
系统性低估(记忆 `aux_niter_localization_degradation`)。**重启/swap 不能解,只能等或
放宽 rtol。** 论文须双报:受控均匀-d 基准 O(10) + 真实局部化 O(100) 但仍有界收敛、
对手(none/Jacobi/ILU)在该状态全 DNF。

## 状态日志

- [2026-06-04 22:42:21] === 并行流水线启动 (aux_h2 ‖ aux_h3 ‖ model1, SAVE_EVERY=1) ===
- [2026-06-04 22:42:21] aux_h2 启动 (hmin=0.025, SAVE_EVERY=1, resume@checkpoint)
- [2026-06-04 22:42:21] aux_h3 启动 (hmin=0.013, SAVE_EVERY=1, resume@checkpoint)
- [2026-06-04 22:42:21] model1 nx=120 启动 (pardiso, SAVE_EVERY=1, resume@step50)
- [2026-06-04 22:42:21] PIDs: h2=337913 h3=337916 model1=337919
- [2026-06-04 23:12:21] aux_h2 进行中 [pid 337913] rows=13 step=12 maxd=0.3685 niter_e=7 load=0.0854
- [2026-06-04 23:12:21] aux_h3 进行中 [pid 337916] rows=11 step=10 maxd=0.3047 niter_e=7 load=0.081
- [2026-06-04 23:12:21] model1 进行中 [pid 337919] rows=52 step=51 maxd=1.0000 niter_e=1 load=0.00501
- [2026-06-04 23:42:21] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-04 23:42:21] aux_h3 进行中 [pid 337916] rows=12 step=11 maxd=0.3311 niter_e=8 load=0.08320000000000001
- [2026-06-04 23:42:21] model1 进行中 [pid 337919] rows=53 step=52 maxd=1.0000 niter_e=1 load=0.00502
- [2026-06-05 00:12:22] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 00:12:22] aux_h3 进行中 [pid 337916] rows=12 step=11 maxd=0.3311 niter_e=8 load=0.08320000000000001
- [2026-06-05 00:12:22] model1 进行中 [pid 337919] rows=53 step=52 maxd=1.0000 niter_e=1 load=0.00502
- [2026-06-05 00:42:22] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 00:42:22] aux_h3 进行中 [pid 337916] rows=12 step=11 maxd=0.3311 niter_e=8 load=0.08320000000000001
- [2026-06-05 00:42:22] model1 进行中 [pid 337919] rows=53 step=52 maxd=1.0000 niter_e=1 load=0.00502
- [2026-06-05 01:12:22] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 01:12:22] aux_h3 进行中 [pid 337916] rows=13 step=12 maxd=0.3652 niter_e=8 load=0.0854
- [2026-06-05 01:12:22] model1 进行中 [pid 337919] rows=53 step=52 maxd=1.0000 niter_e=1 load=0.00502
- [2026-06-05 01:42:22] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 01:42:22] aux_h3 进行中 [pid 337916] rows=13 step=12 maxd=0.3652 niter_e=8 load=0.0854
- [2026-06-05 01:42:22] model1 进行中 [pid 337919] rows=54 step=53 maxd=1.0000 niter_e=1 load=0.00503
- [2026-06-05 02:12:22] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 02:12:22] aux_h3 进行中 [pid 337916] rows=13 step=12 maxd=0.3652 niter_e=8 load=0.0854
- [2026-06-05 02:12:22] model1 进行中 [pid 337919] rows=54 step=53 maxd=1.0000 niter_e=1 load=0.00503
- [2026-06-05 02:42:22] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 02:42:22] aux_h3 进行中 [pid 337916] rows=13 step=12 maxd=0.3652 niter_e=8 load=0.0854
- [2026-06-05 02:42:22] model1 进行中 [pid 337919] rows=54 step=53 maxd=1.0000 niter_e=1 load=0.00503
- [2026-06-05 03:12:22] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 03:12:23] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 03:12:23] model1 进行中 [pid 337919] rows=54 step=53 maxd=1.0000 niter_e=1 load=0.00503
- [2026-06-05 03:42:23] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 03:42:23] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 03:42:23] model1 进行中 [pid 337919] rows=55 step=54 maxd=1.0000 niter_e=1 load=0.00504
- [2026-06-05 04:12:23] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 04:12:23] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 04:12:23] model1 进行中 [pid 337919] rows=55 step=54 maxd=1.0000 niter_e=1 load=0.00504
- [2026-06-05 04:42:23] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 04:42:23] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 04:42:23] model1 进行中 [pid 337919] rows=55 step=54 maxd=1.0000 niter_e=1 load=0.00504
- [2026-06-05 05:12:23] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 05:12:23] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 05:12:23] model1 进行中 [pid 337919] rows=56 step=55 maxd=1.0000 niter_e=1 load=0.00505
- [2026-06-05 05:42:23] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 05:42:23] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 05:42:23] model1 进行中 [pid 337919] rows=56 step=55 maxd=1.0000 niter_e=1 load=0.00505
- [2026-06-05 06:12:23] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 06:12:23] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 06:12:23] model1 进行中 [pid 337919] rows=56 step=55 maxd=1.0000 niter_e=1 load=0.00505
- [2026-06-05 06:42:23] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 06:42:23] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 06:42:23] model1 进行中 [pid 337919] rows=57 step=56 maxd=1.0000 niter_e=1 load=0.00506
- [2026-06-05 07:12:23] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 07:12:23] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 07:12:23] model1 进行中 [pid 337919] rows=57 step=56 maxd=1.0000 niter_e=1 load=0.00506
- [2026-06-05 07:42:23] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 07:42:23] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 07:42:24] model1 进行中 [pid 337919] rows=58 step=57 maxd=1.0000 niter_e=1 load=0.00507
- [2026-06-05 08:12:24] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 08:12:24] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 08:12:24] model1 进行中 [pid 337919] rows=58 step=57 maxd=1.0000 niter_e=1 load=0.00507
- [2026-06-05 08:42:24] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 08:42:24] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 08:42:24] model1 进行中 [pid 337919] rows=58 step=57 maxd=1.0000 niter_e=1 load=0.00507
- [2026-06-05 09:12:24] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 09:12:24] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 09:12:24] model1 进行中 [pid 337919] rows=58 step=57 maxd=1.0000 niter_e=1 load=0.00507
- [2026-06-05 09:42:24] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 09:42:24] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 09:42:24] model1 进行中 [pid 337919] rows=59 step=58 maxd=1.0000 niter_e=1 load=0.00508
- [2026-06-05 10:12:24] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 10:12:24] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 10:12:24] model1 进行中 [pid 337919] rows=59 step=58 maxd=1.0000 niter_e=1 load=0.00508
- [2026-06-05 10:42:24] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 10:42:24] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 10:42:24] model1 进行中 [pid 337919] rows=59 step=58 maxd=1.0000 niter_e=1 load=0.00508
- [2026-06-05 11:12:24] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 11:12:24] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 11:12:24] model1 进行中 [pid 337919] rows=59 step=58 maxd=1.0000 niter_e=1 load=0.00508
- [2026-06-05 11:42:25] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 11:42:25] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 11:42:25] model1 进行中 [pid 337919] rows=59 step=58 maxd=1.0000 niter_e=1 load=0.00508
- [2026-06-05 12:12:25] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 12:12:25] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 12:12:25] model1 进行中 [pid 337919] rows=60 step=59 maxd=1.0000 niter_e=1 load=0.00509
- [2026-06-05 12:42:25] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 12:42:25] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 12:42:25] model1 进行中 [pid 337919] rows=60 step=59 maxd=1.0000 niter_e=1 load=0.00509
- [2026-06-05 13:12:25] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 13:12:25] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 13:12:25] model1 进行中 [pid 337919] rows=60 step=59 maxd=1.0000 niter_e=1 load=0.00509
- [2026-06-05 13:42:25] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 13:42:25] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 13:42:25] model1 进行中 [pid 337919] rows=60 step=59 maxd=1.0000 niter_e=1 load=0.00509
- [2026-06-05 14:12:25] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 14:12:25] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 14:12:25] model1 进行中 [pid 337919] rows=60 step=59 maxd=1.0000 niter_e=1 load=0.00509
- [2026-06-05 14:42:25] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 14:42:25] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 14:42:25] model1 进行中 [pid 337919] rows=60 step=59 maxd=1.0000 niter_e=1 load=0.00509
- [2026-06-05 15:12:25] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 15:12:25] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 15:12:25] model1 进行中 [pid 337919] rows=60 step=59 maxd=1.0000 niter_e=1 load=0.00509
- [2026-06-05 15:42:25] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 15:42:25] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 15:42:25] model1 进行中 [pid 337919] rows=61 step=60 maxd=1.0000 niter_e=1 load=0.0051
- [2026-06-05 16:12:26] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 16:12:26] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 16:12:26] model1 进行中 [pid 337919] rows=61 step=60 maxd=1.0000 niter_e=1 load=0.0051
- [2026-06-05 16:42:26] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 16:42:26] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 16:42:26] model1 进行中 [pid 337919] rows=61 step=60 maxd=1.0000 niter_e=1 load=0.0051
- [2026-06-05 17:12:26] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 17:12:26] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 17:12:26] model1 进行中 [pid 337919] rows=61 step=60 maxd=1.0000 niter_e=1 load=0.0051
- [2026-06-05 17:42:26] aux_h2 进行中 [pid 337913] rows=14 step=13 maxd=0.4258 niter_e=7 load=0.08760000000000001
- [2026-06-05 17:42:26] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 17:42:26] model1 进行中 [pid 337919] rows=61 step=60 maxd=1.0000 niter_e=1 load=0.0051
- [2026-06-05 18:12:26] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-05 18:12:26] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 18:12:26] model1 进行中 [pid 337919] rows=61 step=60 maxd=1.0000 niter_e=1 load=0.0051
- [2026-06-05 18:42:26] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-05 18:42:26] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 18:42:26] model1 进行中 [pid 337919] rows=61 step=60 maxd=1.0000 niter_e=1 load=0.0051
- [2026-06-05 19:12:26] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-05 19:12:26] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 19:12:26] model1 进行中 [pid 337919] rows=61 step=60 maxd=1.0000 niter_e=1 load=0.0051
- [2026-06-05 19:42:26] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-05 19:42:27] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 19:42:27] model1 进行中 [pid 337919] rows=61 step=60 maxd=1.0000 niter_e=1 load=0.0051
- [2026-06-05 20:12:27] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-05 20:12:27] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 20:12:27] model1 进行中 [pid 337919] rows=61 step=60 maxd=1.0000 niter_e=1 load=0.0051
- [2026-06-05 20:42:27] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-05 20:42:27] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 20:42:27] model1 进行中 [pid 337919] rows=61 step=60 maxd=1.0000 niter_e=1 load=0.0051
- [2026-06-05 21:12:27] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-05 21:12:27] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 21:12:27] model1 进行中 [pid 337919] rows=61 step=60 maxd=1.0000 niter_e=1 load=0.0051
- [2026-06-05 21:42:27] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-05 21:42:27] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 21:42:27] model1 进行中 [pid 337919] rows=61 step=60 maxd=1.0000 niter_e=1 load=0.0051
- [2026-06-05 22:12:27] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-05 22:12:27] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 22:12:27] model1 进行中 [pid 337919] rows=62 step=61 maxd=1.0000 niter_e=1 load=0.00511
- [2026-06-05 22:42:27] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-05 22:42:27] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 22:42:27] model1 进行中 [pid 337919] rows=62 step=61 maxd=1.0000 niter_e=1 load=0.00511
- [2026-06-05 23:12:27] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-05 23:12:27] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 23:12:27] model1 进行中 [pid 337919] rows=62 step=61 maxd=1.0000 niter_e=1 load=0.00511
- [2026-06-05 23:42:27] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-05 23:42:27] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-05 23:42:27] model1 进行中 [pid 337919] rows=62 step=61 maxd=1.0000 niter_e=1 load=0.00511
- [2026-06-06 00:12:27] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 00:12:27] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 00:12:27] model1 进行中 [pid 337919] rows=62 step=61 maxd=1.0000 niter_e=1 load=0.00511
- [2026-06-06 00:42:28] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 00:42:28] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 00:42:28] model1 进行中 [pid 337919] rows=62 step=61 maxd=1.0000 niter_e=1 load=0.00511
- [2026-06-06 01:12:28] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 01:12:28] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 01:12:28] model1 进行中 [pid 337919] rows=62 step=61 maxd=1.0000 niter_e=1 load=0.00511
- [2026-06-06 01:42:28] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 01:42:28] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 01:42:28] model1 进行中 [pid 337919] rows=62 step=61 maxd=1.0000 niter_e=1 load=0.00511
- [2026-06-06 02:12:28] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 02:12:28] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 02:12:28] model1 进行中 [pid 337919] rows=62 step=61 maxd=1.0000 niter_e=1 load=0.00511
- [2026-06-06 02:42:28] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 02:42:28] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 02:42:28] model1 进行中 [pid 337919] rows=62 step=61 maxd=1.0000 niter_e=1 load=0.00511
- [2026-06-06 03:12:28] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 03:12:28] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 03:12:28] model1 进行中 [pid 337919] rows=62 step=61 maxd=1.0000 niter_e=1 load=0.00511
- [2026-06-06 03:42:28] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 03:42:28] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 03:42:28] model1 进行中 [pid 337919] rows=62 step=61 maxd=1.0000 niter_e=1 load=0.00511
- [2026-06-06 04:12:28] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 04:12:28] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 04:12:28] model1 进行中 [pid 337919] rows=63 step=62 maxd=1.0000 niter_e=1 load=0.00512
- [2026-06-06 04:42:28] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 04:42:28] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 04:42:28] model1 进行中 [pid 337919] rows=63 step=62 maxd=1.0000 niter_e=1 load=0.00512
- [2026-06-06 05:12:28] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 05:12:29] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 05:12:29] model1 进行中 [pid 337919] rows=63 step=62 maxd=1.0000 niter_e=1 load=0.00512
- [2026-06-06 05:42:29] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 05:42:29] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 05:42:29] model1 进行中 [pid 337919] rows=63 step=62 maxd=1.0000 niter_e=1 load=0.00512
- [2026-06-06 06:12:29] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 06:12:29] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 06:12:29] model1 进行中 [pid 337919] rows=63 step=62 maxd=1.0000 niter_e=1 load=0.00512
- [2026-06-06 06:42:29] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 06:42:29] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 06:42:29] model1 进行中 [pid 337919] rows=63 step=62 maxd=1.0000 niter_e=1 load=0.00512
- [2026-06-06 07:12:29] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 07:12:29] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 07:12:29] model1 进行中 [pid 337919] rows=63 step=62 maxd=1.0000 niter_e=1 load=0.00512
- [2026-06-06 07:42:29] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 07:42:29] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 07:42:29] model1 进行中 [pid 337919] rows=63 step=62 maxd=1.0000 niter_e=1 load=0.00512
- [2026-06-06 08:12:29] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 08:12:29] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 08:12:29] model1 进行中 [pid 337919] rows=63 step=62 maxd=1.0000 niter_e=1 load=0.00512
- [2026-06-06 08:42:29] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 08:42:29] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 08:42:29] model1 进行中 [pid 337919] rows=63 step=62 maxd=1.0000 niter_e=1 load=0.00512
- [2026-06-06 09:12:29] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 09:12:29] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 09:12:29] model1 进行中 [pid 337919] rows=63 step=62 maxd=1.0000 niter_e=1 load=0.00512
- [2026-06-06 09:42:29] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 09:42:29] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 09:42:29] model1 进行中 [pid 337919] rows=63 step=62 maxd=1.0000 niter_e=1 load=0.00512
- [2026-06-06 10:12:30] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 10:12:30] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 10:12:30] model1 进行中 [pid 337919] rows=63 step=62 maxd=1.0000 niter_e=1 load=0.00512
- [2026-06-06 10:42:30] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 10:42:30] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 10:42:30] model1 进行中 [pid 337919] rows=64 step=63 maxd=1.0000 niter_e=1 load=0.00513
- [2026-06-06 11:12:30] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 11:12:30] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 11:12:30] model1 进行中 [pid 337919] rows=64 step=63 maxd=1.0000 niter_e=1 load=0.00513
- [2026-06-06 11:42:30] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 11:42:30] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 11:42:30] model1 进行中 [pid 337919] rows=64 step=63 maxd=1.0000 niter_e=1 load=0.00513
- [2026-06-06 12:12:30] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 12:12:30] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 12:12:30] model1 进行中 [pid 337919] rows=64 step=63 maxd=1.0000 niter_e=1 load=0.00513
- [2026-06-06 12:42:30] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 12:42:30] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 12:42:30] model1 进行中 [pid 337919] rows=64 step=63 maxd=1.0000 niter_e=1 load=0.00513
- [2026-06-06 13:12:30] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 13:12:30] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 13:12:30] model1 进行中 [pid 337919] rows=64 step=63 maxd=1.0000 niter_e=1 load=0.00513
- [2026-06-06 13:42:30] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 13:42:30] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 13:42:30] model1 进行中 [pid 337919] rows=64 step=63 maxd=1.0000 niter_e=1 load=0.00513
- [2026-06-06 14:12:30] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 14:12:30] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 14:12:30] model1 进行中 [pid 337919] rows=64 step=63 maxd=1.0000 niter_e=1 load=0.00513
- [2026-06-06 14:42:30] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 14:42:30] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 14:42:30] model1 进行中 [pid 337919] rows=64 step=63 maxd=1.0000 niter_e=1 load=0.00513
- [2026-06-06 15:12:30] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 15:12:30] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 15:12:30] model1 进行中 [pid 337919] rows=64 step=63 maxd=1.0000 niter_e=1 load=0.00513
- [2026-06-06 15:42:31] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 15:42:31] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 15:42:31] model1 进行中 [pid 337919] rows=64 step=63 maxd=1.0000 niter_e=1 load=0.00513
- [2026-06-06 16:12:31] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 16:12:31] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 16:12:31] model1 进行中 [pid 337919] rows=64 step=63 maxd=1.0000 niter_e=1 load=0.00513
- [2026-06-06 16:42:31] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 16:42:31] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 16:42:31] model1 进行中 [pid 337919] rows=65 step=64 maxd=1.0000 niter_e=1 load=0.0051400000000000005
- [2026-06-06 17:12:31] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 17:12:31] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 17:12:31] model1 进行中 [pid 337919] rows=65 step=64 maxd=1.0000 niter_e=1 load=0.0051400000000000005
- [2026-06-06 17:42:31] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 17:42:31] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 17:42:31] model1 进行中 [pid 337919] rows=65 step=64 maxd=1.0000 niter_e=1 load=0.0051400000000000005
- [2026-06-06 18:12:31] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 18:12:31] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 18:12:31] model1 进行中 [pid 337919] rows=65 step=64 maxd=1.0000 niter_e=1 load=0.0051400000000000005
- [2026-06-06 18:42:31] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 18:42:31] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 18:42:31] model1 进行中 [pid 337919] rows=65 step=64 maxd=1.0000 niter_e=1 load=0.0051400000000000005
- [2026-06-06 19:12:31] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 19:12:31] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 19:12:31] model1 进行中 [pid 337919] rows=65 step=64 maxd=1.0000 niter_e=1 load=0.0051400000000000005
- [2026-06-06 19:42:31] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 19:42:31] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 19:42:31] model1 进行中 [pid 337919] rows=65 step=64 maxd=1.0000 niter_e=1 load=0.0051400000000000005
- [2026-06-06 20:12:31] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 20:12:31] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 20:12:31] model1 进行中 [pid 337919] rows=65 step=64 maxd=1.0000 niter_e=1 load=0.0051400000000000005
- [2026-06-06 20:42:31] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 20:42:32] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 20:42:32] model1 进行中 [pid 337919] rows=65 step=64 maxd=1.0000 niter_e=1 load=0.0051400000000000005
- [2026-06-06 21:12:32] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 21:12:32] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 21:12:32] model1 进行中 [pid 337919] rows=65 step=64 maxd=1.0000 niter_e=1 load=0.0051400000000000005
- [2026-06-06 21:42:32] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 21:42:32] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 21:42:32] model1 进行中 [pid 337919] rows=65 step=64 maxd=1.0000 niter_e=1 load=0.0051400000000000005
- [2026-06-06 22:12:32] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 22:12:32] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 22:12:32] model1 进行中 [pid 337919] rows=65 step=64 maxd=1.0000 niter_e=1 load=0.0051400000000000005
- [2026-06-06 22:42:32] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 22:42:32] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 22:42:32] model1 进行中 [pid 337919] rows=65 step=64 maxd=1.0000 niter_e=1 load=0.0051400000000000005
- [2026-06-06 22:58:20] model2 direct_full 首次续算尝试中止: 默认 nx/load 步在原 run 后被改过
  (nx 160→216, n_load_steps 240→2400),checkpoint NC 不匹配。须显式 pin FRACTUREX_NX=160
  + FRACTUREX_N_LOAD_STEPS=240 复现原网格/载荷,见下条重启。
- [2026-06-06 23:10:02] model2 direct_full 续算启动 (pardiso, nx=160, 240步, SAVE_EVERY=1, resume@latest checkpoint, 目标 step240)
- [2026-06-06 23:10:02] model2 direct_full PID=22799
- [2026-06-06 23:12:32] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 23:12:32] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 23:12:32] model1 进行中 [pid 337919] rows=66 step=65 maxd=1.0000 niter_e=1 load=0.00515
- [2026-06-06 23:40:02] model2 direct_full 进行中 [pid 22799] rows=52 step=51 maxd=1.0000 Rx=-0.2234071160531252 dispx=0.0051 niter_e=1
- [2026-06-06 23:42:32] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-06 23:42:32] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-06 23:42:32] model1 进行中 [pid 337919] rows=66 step=65 maxd=1.0000 niter_e=1 load=0.00515
- [2026-06-07 00:10:02] model2 direct_full 进行中 [pid 22799] rows=52 step=51 maxd=1.0000 Rx=-0.2234071160531252 dispx=0.0051 niter_e=1
- [2026-06-07 00:12:32] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 00:12:32] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 00:12:32] model1 进行中 [pid 337919] rows=66 step=65 maxd=1.0000 niter_e=1 load=0.00515
- [2026-06-07 00:40:02] model2 direct_full 进行中 [pid 22799] rows=52 step=51 maxd=1.0000 Rx=-0.2234071160531252 dispx=0.0051 niter_e=1
- [2026-06-07 00:42:32] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 00:42:32] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 00:42:32] model1 进行中 [pid 337919] rows=66 step=65 maxd=1.0000 niter_e=1 load=0.00515
- [2026-06-07 01:10:02] model2 direct_full 进行中 [pid 22799] rows=52 step=51 maxd=1.0000 Rx=-0.2234071160531252 dispx=0.0051 niter_e=1
- [2026-06-07 01:12:32] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 01:12:33] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 01:12:33] model1 进行中 [pid 337919] rows=66 step=65 maxd=1.0000 niter_e=1 load=0.00515
- [2026-06-07 01:40:02] model2 direct_full 进行中 [pid 22799] rows=52 step=51 maxd=1.0000 Rx=-0.2234071160531252 dispx=0.0051 niter_e=1
- [2026-06-07 01:42:33] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 01:42:33] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 01:42:33] model1 进行中 [pid 337919] rows=66 step=65 maxd=1.0000 niter_e=1 load=0.00515
- [2026-06-07 02:10:02] model2 direct_full 进行中 [pid 22799] rows=53 step=52 maxd=1.0000 Rx=-0.22744834227653368 dispx=0.005200000000000001 niter_e=1
- [2026-06-07 02:12:33] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 02:12:33] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 02:12:33] model1 进行中 [pid 337919] rows=66 step=65 maxd=1.0000 niter_e=1 load=0.00515
- [2026-06-07 02:40:02] model2 direct_full 进行中 [pid 22799] rows=53 step=52 maxd=1.0000 Rx=-0.22744834227653368 dispx=0.005200000000000001 niter_e=1
- [2026-06-07 02:42:33] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 02:42:33] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 02:42:33] model1 进行中 [pid 337919] rows=66 step=65 maxd=1.0000 niter_e=1 load=0.00515
- [2026-06-07 03:10:02] model2 direct_full 进行中 [pid 22799] rows=54 step=53 maxd=1.0000 Rx=-0.2316552255698471 dispx=0.0053 niter_e=1
- [2026-06-07 03:12:33] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 03:12:33] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 03:12:33] model1 进行中 [pid 337919] rows=70 step=69 maxd=1.0000 niter_e=1 load=0.00519
- [2026-06-07 03:40:02] model2 direct_full 进行中 [pid 22799] rows=54 step=53 maxd=1.0000 Rx=-0.2316552255698471 dispx=0.0053 niter_e=1
- [2026-06-07 03:42:33] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 03:42:33] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 03:42:33] model1 进行中 [pid 337919] rows=74 step=73 maxd=1.0000 niter_e=1 load=0.00523
- [2026-06-07 04:10:02] model2 direct_full 进行中 [pid 22799] rows=55 step=54 maxd=1.0000 Rx=-0.23587549239054245 dispx=0.0054 niter_e=1
- [2026-06-07 04:12:33] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 04:12:33] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 04:12:33] model1 进行中 [pid 337919] rows=77 step=76 maxd=1.0000 niter_e=1 load=0.00526
- [2026-06-07 04:40:02] model2 direct_full 进行中 [pid 22799] rows=55 step=54 maxd=1.0000 Rx=-0.23587549239054245 dispx=0.0054 niter_e=1
- [2026-06-07 04:42:33] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 04:42:33] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 04:42:33] model1 进行中 [pid 337919] rows=80 step=79 maxd=1.0000 niter_e=1 load=0.00529
- [2026-06-07 05:10:02] model2 direct_full 进行中 [pid 22799] rows=55 step=54 maxd=1.0000 Rx=-0.23587549239054245 dispx=0.0054 niter_e=1
- [2026-06-07 05:12:33] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 05:12:33] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 05:12:33] model1 进行中 [pid 337919] rows=83 step=82 maxd=1.0000 niter_e=1 load=0.00532
- [2026-06-07 05:40:02] model2 direct_full 进行中 [pid 22799] rows=55 step=54 maxd=1.0000 Rx=-0.23587549239054245 dispx=0.0054 niter_e=1
- [2026-06-07 05:42:33] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 05:42:33] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 05:42:34] model1 进行中 [pid 337919] rows=86 step=85 maxd=1.0000 niter_e=1 load=0.005350000000000001
- [2026-06-07 06:10:02] model2 direct_full 进行中 [pid 22799] rows=55 step=54 maxd=1.0000 Rx=-0.23587549239054245 dispx=0.0054 niter_e=1
- [2026-06-07 06:12:34] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 06:12:34] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 06:12:34] model1 进行中 [pid 337919] rows=88 step=87 maxd=1.0000 niter_e=1 load=0.00537
- [2026-06-07 06:40:02] model2 direct_full 进行中 [pid 22799] rows=55 step=54 maxd=1.0000 Rx=-0.23587549239054245 dispx=0.0054 niter_e=1
- [2026-06-07 06:42:34] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 06:42:34] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 06:42:34] model1 进行中 [pid 337919] rows=91 step=90 maxd=1.0000 niter_e=1 load=0.0054
- [2026-06-07 07:10:02] model2 direct_full 进行中 [pid 22799] rows=55 step=54 maxd=1.0000 Rx=-0.23587549239054245 dispx=0.0054 niter_e=1
- [2026-06-07 07:12:34] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 07:12:34] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 07:12:34] model1 进行中 [pid 337919] rows=94 step=93 maxd=1.0000 niter_e=1 load=0.00543
- [2026-06-07 07:40:02] model2 direct_full 进行中 [pid 22799] rows=55 step=54 maxd=1.0000 Rx=-0.23587549239054245 dispx=0.0054 niter_e=1
- [2026-06-07 07:42:34] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 07:42:34] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 07:42:34] model1 进行中 [pid 337919] rows=95 step=94 maxd=1.0000 niter_e=1 load=0.00544
- [2026-06-07 08:10:03] model2 direct_full 进行中 [pid 22799] rows=56 step=55 maxd=1.0000 Rx=-0.2400335354041149 dispx=0.0055000000000000005 niter_e=1
- [2026-06-07 08:12:34] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 08:12:34] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 08:12:34] model1 进行中 [pid 337919] rows=96 step=95 maxd=1.0000 niter_e=1 load=0.00545
- [2026-06-07 08:40:03] model2 direct_full 进行中 [pid 22799] rows=57 step=56 maxd=1.0000 Rx=-0.2442374981885097 dispx=0.0056 niter_e=1
- [2026-06-07 08:42:34] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 08:42:34] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 08:42:34] model1 进行中 [pid 337919] rows=98 step=97 maxd=1.0000 niter_e=1 load=0.00547
- [2026-06-07 09:10:03] model2 direct_full 进行中 [pid 22799] rows=58 step=57 maxd=1.0000 Rx=-0.248427522012369 dispx=0.0057 niter_e=1
- [2026-06-07 09:12:34] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 09:12:34] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 09:12:34] model1 进行中 [pid 337919] rows=99 step=98 maxd=1.0000 niter_e=1 load=0.0054800000000000005
- [2026-06-07 09:40:03] model2 direct_full 进行中 [pid 22799] rows=58 step=57 maxd=1.0000 Rx=-0.248427522012369 dispx=0.0057 niter_e=1
- [2026-06-07 09:42:34] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 09:42:34] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 09:42:34] model1 进行中 [pid 337919] rows=101 step=100 maxd=1.0000 niter_e=1 load=0.0055000000000000005
- [2026-06-07 10:10:03] model2 direct_full 进行中 [pid 22799] rows=58 step=57 maxd=1.0000 Rx=-0.248427522012369 dispx=0.0057 niter_e=1
- [2026-06-07 10:12:34] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 10:12:34] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 10:12:35] model1 进行中 [pid 337919] rows=102 step=101 maxd=1.0000 niter_e=1 load=0.00551
- [2026-06-07 10:40:03] model2 direct_full 进行中 [pid 22799] rows=58 step=57 maxd=1.0000 Rx=-0.248427522012369 dispx=0.0057 niter_e=1
- [2026-06-07 10:42:35] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 10:42:35] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 10:42:35] model1 进行中 [pid 337919] rows=103 step=102 maxd=1.0000 niter_e=1 load=0.005520000000000001
- [2026-06-07 11:10:03] model2 direct_full 进行中 [pid 22799] rows=58 step=57 maxd=1.0000 Rx=-0.248427522012369 dispx=0.0057 niter_e=1
- [2026-06-07 11:12:35] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 11:12:35] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 11:12:35] model1 进行中 [pid 337919] rows=103 step=102 maxd=1.0000 niter_e=1 load=0.005520000000000001
- [2026-06-07 11:40:03] model2 direct_full 进行中 [pid 22799] rows=58 step=57 maxd=1.0000 Rx=-0.248427522012369 dispx=0.0057 niter_e=1
- [2026-06-07 11:42:35] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 11:42:35] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 11:42:35] model1 进行中 [pid 337919] rows=104 step=103 maxd=1.0000 niter_e=1 load=0.00553
- [2026-06-07 12:10:03] model2 direct_full 进行中 [pid 22799] rows=58 step=57 maxd=1.0000 Rx=-0.248427522012369 dispx=0.0057 niter_e=1
- [2026-06-07 12:12:35] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 12:12:35] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 12:12:35] model1 进行中 [pid 337919] rows=106 step=105 maxd=1.0000 niter_e=1 load=0.00555
- [2026-06-07 12:40:03] model2 direct_full 进行中 [pid 22799] rows=59 step=58 maxd=1.0000 Rx=-0.2525615804092394 dispx=0.0058000000000000005 niter_e=1
- [2026-06-07 12:42:35] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 12:42:35] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 12:42:35] model1 进行中 [pid 337919] rows=108 step=107 maxd=1.0000 niter_e=1 load=0.00557
- [2026-06-07 13:10:03] model2 direct_full 进行中 [pid 22799] rows=59 step=58 maxd=1.0000 Rx=-0.2525615804092394 dispx=0.0058000000000000005 niter_e=1
- [2026-06-07 13:12:35] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 13:12:35] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 13:12:35] model1 进行中 [pid 337919] rows=110 step=109 maxd=1.0000 niter_e=1 load=0.00559
- [2026-06-07 13:40:03] model2 direct_full 进行中 [pid 22799] rows=59 step=58 maxd=1.0000 Rx=-0.2525615804092394 dispx=0.0058000000000000005 niter_e=1
- [2026-06-07 13:42:35] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 13:42:35] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 13:42:35] model1 进行中 [pid 337919] rows=112 step=111 maxd=1.0000 niter_e=1 load=0.00561
- [2026-06-07 14:10:03] model2 direct_full 进行中 [pid 22799] rows=59 step=58 maxd=1.0000 Rx=-0.2525615804092394 dispx=0.0058000000000000005 niter_e=1
- [2026-06-07 14:12:35] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 14:12:35] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 14:12:35] model1 进行中 [pid 337919] rows=112 step=111 maxd=1.0000 niter_e=1 load=0.00561
- [2026-06-07 14:40:03] model2 direct_full 进行中 [pid 22799] rows=59 step=58 maxd=1.0000 Rx=-0.2525615804092394 dispx=0.0058000000000000005 niter_e=1
- [2026-06-07 14:42:35] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 14:42:35] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 14:42:36] model1 进行中 [pid 337919] rows=113 step=112 maxd=1.0000 niter_e=1 load=0.00562
- [2026-06-07 15:10:03] model2 direct_full 进行中 [pid 22799] rows=60 step=59 maxd=1.0000 Rx=-0.25671803522748987 dispx=0.0059 niter_e=1
- [2026-06-07 15:12:36] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 15:12:36] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 15:12:36] model1 进行中 [pid 337919] rows=114 step=113 maxd=1.0000 niter_e=1 load=0.0056300000000000005
- [2026-06-07 15:40:03] model2 direct_full 进行中 [pid 22799] rows=60 step=59 maxd=1.0000 Rx=-0.25671803522748987 dispx=0.0059 niter_e=1
- [2026-06-07 15:42:36] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 15:42:36] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 15:42:36] model1 进行中 [pid 337919] rows=115 step=114 maxd=1.0000 niter_e=1 load=0.00564
- [2026-06-07 16:10:03] model2 direct_full 进行中 [pid 22799] rows=60 step=59 maxd=1.0000 Rx=-0.25671803522748987 dispx=0.0059 niter_e=1
- [2026-06-07 16:12:36] aux_h2 进行中 [pid 337913] rows=15 step=14 maxd=0.9981 niter_e=95 load=0.0898
- [2026-06-07 16:12:36] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 16:12:36] model1 进行中 [pid 337919] rows=115 step=114 maxd=1.0000 niter_e=1 load=0.00564
- [2026-06-07 16:40:03] model2 direct_full 进行中 [pid 22799] rows=61 step=60 maxd=1.0000 Rx=-0.260883453639981 dispx=0.006 niter_e=1
- [2026-06-07 16:42:36] aux_h2 进行中 [pid 337913] rows=16 step=15 maxd=0.9981 niter_e=170 load=0.092
- [2026-06-07 16:42:36] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 16:42:36] model1 进行中 [pid 337919] rows=116 step=115 maxd=1.0000 niter_e=1 load=0.0056500000000000005
- [2026-06-07 17:10:03] model2 direct_full 进行中 [pid 22799] rows=61 step=60 maxd=1.0000 Rx=-0.260883453639981 dispx=0.006 niter_e=1
- [2026-06-07 17:12:36] aux_h2 进行中 [pid 337913] rows=16 step=15 maxd=0.9981 niter_e=170 load=0.092
- [2026-06-07 17:12:36] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 17:12:36] model1 进行中 [pid 337919] rows=118 step=117 maxd=1.0000 niter_e=1 load=0.0056700000000000006
- [2026-06-07 17:40:03] model2 direct_full 进行中 [pid 22799] rows=62 step=61 maxd=1.0000 Rx=-0.26503803164069517 dispx=0.0061 niter_e=1
- [2026-06-07 17:42:36] aux_h2 进行中 [pid 337913] rows=16 step=15 maxd=0.9981 niter_e=170 load=0.092
- [2026-06-07 17:42:36] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 17:42:36] model1 进行中 [pid 337919] rows=119 step=118 maxd=1.0000 niter_e=1 load=0.00568
- [2026-06-07 18:10:03] model2 direct_full 进行中 [pid 22799] rows=63 step=62 maxd=1.0000 Rx=-0.26757890857463723 dispx=0.006200000000000001 niter_e=1
- [2026-06-07 18:12:36] aux_h2 进行中 [pid 337913] rows=16 step=15 maxd=0.9981 niter_e=170 load=0.092
- [2026-06-07 18:12:36] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 18:12:36] model1 进行中 [pid 337919] rows=121 step=120 maxd=1.0000 niter_e=1 load=0.0057
- [2026-06-07 18:40:03] model2 direct_full 进行中 [pid 22799] rows=63 step=62 maxd=1.0000 Rx=-0.26757890857463723 dispx=0.006200000000000001 niter_e=1
- [2026-06-07 18:42:36] aux_h2 进行中 [pid 337913] rows=16 step=15 maxd=0.9981 niter_e=170 load=0.092
- [2026-06-07 18:42:37] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 18:42:37] model1 进行中 [pid 337919] rows=122 step=121 maxd=1.0000 niter_e=1 load=0.00571
- [2026-06-07 19:10:04] model2 direct_full 进行中 [pid 22799] rows=63 step=62 maxd=1.0000 Rx=-0.26757890857463723 dispx=0.006200000000000001 niter_e=1
- [2026-06-07 19:12:37] aux_h2 进行中 [pid 337913] rows=16 step=15 maxd=0.9981 niter_e=170 load=0.092
- [2026-06-07 19:12:37] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 19:12:37] model1 进行中 [pid 337919] rows=123 step=122 maxd=1.0000 niter_e=1 load=0.00572
- [2026-06-07 19:40:04] model2 direct_full 进行中 [pid 22799] rows=63 step=62 maxd=1.0000 Rx=-0.26757890857463723 dispx=0.006200000000000001 niter_e=1
- [2026-06-07 19:42:37] aux_h2 进行中 [pid 337913] rows=16 step=15 maxd=0.9981 niter_e=170 load=0.092
- [2026-06-07 19:42:37] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 19:42:37] model1 进行中 [pid 337919] rows=124 step=123 maxd=1.0000 niter_e=1 load=0.005730000000000001
- [2026-06-07 20:10:04] model2 direct_full 进行中 [pid 22799] rows=64 step=63 maxd=1.0000 Rx=-0.2717194356726941 dispx=0.0063 niter_e=1
- [2026-06-07 20:12:37] aux_h2 进行中 [pid 337913] rows=16 step=15 maxd=0.9981 niter_e=170 load=0.092
- [2026-06-07 20:12:37] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 20:12:37] model1 进行中 [pid 337919] rows=125 step=124 maxd=1.0000 niter_e=1 load=0.00574
- [2026-06-07 20:40:04] model2 direct_full 进行中 [pid 22799] rows=64 step=63 maxd=1.0000 Rx=-0.2717194356726941 dispx=0.0063 niter_e=1
- [2026-06-07 20:42:37] aux_h2 进行中 [pid 337913] rows=16 step=15 maxd=0.9981 niter_e=170 load=0.092
- [2026-06-07 20:42:37] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 20:42:37] model1 进行中 [pid 337919] rows=126 step=125 maxd=1.0000 niter_e=1 load=0.00575
- [2026-06-07 21:10:04] model2 direct_full 进行中 [pid 22799] rows=65 step=64 maxd=1.0000 Rx=-0.2758230691048293 dispx=0.0064 niter_e=1
- [2026-06-07 21:12:37] aux_h2 进行中 [pid 337913] rows=16 step=15 maxd=0.9981 niter_e=170 load=0.092
- [2026-06-07 21:12:37] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 21:12:37] model1 进行中 [pid 337919] rows=127 step=126 maxd=1.0000 niter_e=1 load=0.00576
- [2026-06-07 21:40:04] model2 direct_full 进行中 [pid 22799] rows=65 step=64 maxd=1.0000 Rx=-0.2758230691048293 dispx=0.0064 niter_e=1
- [2026-06-07 21:42:37] aux_h2 进行中 [pid 337913] rows=16 step=15 maxd=0.9981 niter_e=170 load=0.092
- [2026-06-07 21:42:37] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 21:42:37] model1 进行中 [pid 337919] rows=128 step=127 maxd=1.0000 niter_e=1 load=0.00577
- [2026-06-07 22:10:04] model2 direct_full 进行中 [pid 22799] rows=65 step=64 maxd=1.0000 Rx=-0.2758230691048293 dispx=0.0064 niter_e=1
- [2026-06-07 22:12:37] aux_h2 进行中 [pid 337913] rows=16 step=15 maxd=0.9981 niter_e=170 load=0.092
- [2026-06-07 22:12:37] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 22:12:37] model1 进行中 [pid 337919] rows=132 step=131 maxd=1.0000 niter_e=1 load=0.00581
- [2026-06-07 22:40:04] model2 direct_full 进行中 [pid 22799] rows=65 step=64 maxd=1.0000 Rx=-0.2758230691048293 dispx=0.0064 niter_e=1
- [2026-06-07 22:42:37] aux_h2 进行中 [pid 337913] rows=16 step=15 maxd=0.9981 niter_e=170 load=0.092
- [2026-06-07 22:42:37] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 22:42:38] model1 进行中 [pid 337919] rows=135 step=134 maxd=1.0000 niter_e=1 load=0.005840000000000001
- [2026-06-07 23:10:04] model2 direct_full 进行中 [pid 22799] rows=65 step=64 maxd=1.0000 Rx=-0.2758230691048293 dispx=0.0064 niter_e=1
- [2026-06-07 23:12:38] aux_h2 进行中 [pid 337913] rows=16 step=15 maxd=0.9981 niter_e=170 load=0.092
- [2026-06-07 23:12:38] aux_h3 进行中 [pid 337916] rows=14 step=13 maxd=0.4188 niter_e=8 load=0.08760000000000001
- [2026-06-07 23:12:38] model1 进行中 [pid 337919] rows=136 step=135 maxd=1.0000 niter_e=1 load=0.00585
- [2026-06-08 09:19:00] === 四路续跑重启 (aux_h2 ‖ aux_h3 ‖ model1 ‖ model2→step200, SAVE_EVERY=1) ===
- [2026-06-08 09:19:00] model2 目标改为 step200 (保留 N_LOAD_STEPS=240 载荷表, FRACTUREX_RUN_NSTEPS=200 截断)
- [2026-06-08 09:22:00] aux_h2 PID=3665650 | aux_h3 首次启动 cwd 错误已 kill | model1 PID=3671595 | model2 PID=3671596
- [2026-06-08 09:23:00] aux_h3 修正重启 PID=3674958 (resume@step13→31, cwd=fracturex)
- [2026-06-08 09:19:04] === 四路续跑启动 (aux_h2 ‖ aux_h3 ‖ model1 ‖ model2→step200, SAVE_EVERY=1) ===
- [2026-06-08 09:21:45] model1 续跑 PID=3671595 (resume@step135→161)
- [2026-06-08 09:21:45] model2 direct_full 续跑 PID=3671596 (resume@step64→200)
- [2026-06-08 09:22:33] aux_h3 续跑重启 PID=3674958 (修正 cwd, resume@step13→31)
- [2026-06-09 10:31:21] === 续跑 aux_h2 ‖ aux_h3 → step30 (model1 已完成排除, SAVE_EVERY=1) ===
- [2026-06-09 10:31:21] PIDs: h2=915376 h3=915378
- [2026-06-09 16:11:21] === 续跑 aux_h2 ‖ aux_h3 → step30，开 Anderson(depth=5,omega=1,tr=20,restart_omega=1.6) 破起裂发散 ===
- [2026-06-09 16:11:21] PIDs(anderson): h2=1065680 h3=1065682 (resume h2@step16 h3@step13)

## [2026-06-09] 起裂发散 → 开 Anderson 重启 aux_h2/h3

- 无加速续算在起裂处 staggered 外层**发散**：h3 step14 振荡 262 次（error 0.14→1.68→极限环 1.4–1.6），h2 step17 卡死 5.5h 无日志。本次续算（PID 915376/915378）未开 Anderson。
- **处理**：kill 两路（model2 PID 3671596 保留不动），按 D12 §14 / 路线 B 开 Anderson 重启：
  `FRACTUREX_ANDERSON_DEPTH=5 OMEGA=1.0 TR_FACTOR=20 RESTART_OMEGA=1.6`（run_aux_model0.sh 验证口径）。
  脚本 `scripts/paper_huzhang/resume_aux_h2h3_anderson.sh`。新 PID: h2=1065680 h3=1065682。
  各自从最新 checkpoint 续：h2@step16, h3@step13（发散的 step14/17 未存盘）。
- **即时效果**：h3 step14 restart kick 破极限环成功——error 从峰值 1.679(iter35) 单调下降到 0.97(iter49) 并继续降，不再振荡。坐实 staggered_acceleration_refs 的起裂加速论点。
- [2026-06-11 15:54:28] === 续跑 aux_h2 从 step15 → step30，restart=200 + Anderson（修 restart=60 分离区垃圾反力）===
- [2026-06-11 15:54:28] aux_h2 r200 PID=3875122 (resume@step15, restart=200, Anderson on)

## [2026-06-11] aux_h2 restart=60 分离区垃圾 → 从 step15 restart=200 重算

- **发现**：aux_h2(Anderson 续算 PID 1065680，**旧内存 restart=60** 版)跑完 step0→30，但
  step17–30(maxd=1.0 完全分离)反力**翻正爬到 +102 非物理**(lin-res~600、niter 顶 maxit)。
  step16(maxd=0.9993)起已偏离 direct(aux -11.08 vs direct -7.07)。
- **诊断(d12_recheck 证实)**：是 restart=60 在分离后奇异非正规鞍点上的重启停滞，**非 aux 失效**。
  同 step17 算子：restart=60→400/DNF(true_relres=574)；**restart=200→25 步收敛**。
  与 §5.2b 表1b step17=25 一致。run_case `_aux_gmres_settings` 默认现已是 restart=200，
  但那两个 aux 进程是 Jun9 改默认前启动的旧内存版(restart=60)。
- **干净分界**：step15(maxd=0.998)是最后一个 aux≡direct 的 checkpoint(反力差4e-4)。
- **处理**：step16–30 旧(r60垃圾)checkpoint 移到 /tmp/aux_h2_r60_ckpt_bak/；从 step15 续，
  显式 restart=200 + Anderson(脚本 resume_aux_h2_r200.sh，PID 3875122)。resume 自动把
  history/iterations 截到 step15 再续写。即时验证 step16 Anderson 外层正常收敛(error→3e-5,
  不发散)。重算 step16→30 拿物理正确反力。
- **h3 冲突暂不动**(用户决定)：两个 aux_h3 进程(1065682 + 2912972)仍写同一目录、都是 r60 旧版。
- **教训**：跑要物理反力的 run 必须确保 restart=200(代码默认已是，但别用改默认前启动的旧进程)。
- [2026-06-12 08:51:57] === aux_h2 从 step15 续，restart=200，**关 Anderson**（诊断 step16 DNF 是否 Anderson 致病态）===
- [2026-06-12 08:51:57] aux_h2 noAnderson PID=1236405 (resume@step15, restart=200, ANDERSON_DEPTH=0)

## [2026-06-12] aux_h2 step16 完全分离 DNF = 迭代法真实边界（结案）

- 从 step15 干净 checkpoint 续算，**restart=200+maxit=400，关 Anderson(DEPTH=0) 与开 Anderson 各试一次**：
  step16(max_d→1.0 完全分离瞬间)弹性鞍点 GMRES 均**打满 maxit 不收敛**(残差~0.46)，staggered
  外层震荡(error 0.4-5.6)、弹性增量归零空转。**关 Anderson 也卡同一点 → DNF 非 Anderson/restart 所致**。
- 同物理步 direct(pardiso) 可直接解。**结论：max_d=1.0 完全分离瞬间鞍点 = aux-GMRES 迭代法真实边界
  (direct 可解、aux DNF)**。早期 d12_recheck "step17@r200=25步收敛" 是读了 r60 污染 checkpoint 的非物理 d 场，弃用。
- **交付定稿**：载荷曲线用 **direct h2**(完整 31 步、物理正确、反力趋零)；aux 物理一致区 = step0–15
  (max_d≤0.998，反力差≤1.5e-3，C1 成立)。图 `model0_h2_loaddisp` 已重绘(direct 完整曲线+aux step0-15
  叠加+step16 DNF 边界标注)。D12_RESULTS §5.2d + memory 已更正。
- aux_h2 进程已全部 kill；h2 目录 history/checkpoint 停在 step15(物理干净)。step16-30 r60 垃圾
  checkpoint 在 /tmp/aux_h2_r60_ckpt_bak/。
- **仍在跑**：model2 direct(PID 3671596, step126+) + 两个冲突 aux_h3(1065682/2912972, r60旧版, 用户暂缓)。
- [2026-06-13 19:57:37] model2 direct_full 续算启动 (pardiso, nx=160, RUN_NSTEPS=200, SAVE_EVERY=1, resume@latest checkpoint)
- [2026-06-13 19:57:38] model2 direct_full PID=76721
- [2026-06-13 20:02:39] model2 看门狗启动 (每 600s 检查；死则从最新 checkpoint 重启，summary.json 出现即退出)
