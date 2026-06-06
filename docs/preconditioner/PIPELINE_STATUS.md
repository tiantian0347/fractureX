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
