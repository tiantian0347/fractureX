# FractureX 文档索引

文档按主题分目录，每条给出一句话定位。

## architecture/ — 架构与接口
- [整体架构与技术路线（中文）](architecture/huzhang_phasefield_architecture.md)
- [Hu–Zhang + 相场架构（English, focused）](architecture/huzhang_phasefield_architecture.en.md)
- [HuZhang 相场接口测试手册（中文）](architecture/huzhang_interface_test_manual.md)
- [多后端编码规范（统一约定）](architecture/multibackend_convention.md) — 计算用 `bm` 不用 `np`；numpy 仅限 I/O/scipy 边界（`bm.to_numpy` 跨界）；新代码强制、存量随改随迁

## operator_learning/ — 算子学习代理（论文路线 + 协议）
- [路线规划 plan_operator_learning](operator_learning/plan_operator_learning.md) — 任务定义、数学、Milestone（M0→M3）
- [数据协议 SURROGATE_DATA_SCHEMA v0.1](operator_learning/SURROGATE_DATA_SCHEMA.md) — 仿真侧↔训练侧的外部协议
- [接入新模型指南 surrogate_porting_guide](operator_learning/surrogate_porting_guide.md) — `SolverAdapter` 接口使用文档
- [插值误差量化 m0_interpolation_error](operator_learning/m0_interpolation_error.md) — 𝓘₁ vs 𝓘₂、σ/𝓗 通道取舍
- [M1 结果 m1_results](operator_learning/m1_results.md) — damage-only 三 baseline pilot 对比（U-Net > FNO > DeepONet）
- [论文中心论点(锁定) paper_thesis](operator_learning/paper_thesis.md) — 平衡保持+Hu-Zhang监督;中心命题=代理平衡缺陷下界=监督目标平衡缺陷(故 Hu-Zhang 必要)
- [M2 Stage B 结果 m2_stageB_results](operator_learning/m2_stageB_results.md) — §6 S档 σ 0.98→0.33（数据量关键）；§7~§8 裂尖峰值对损失/分辨率均鲁棒（已知局限）；peak-load ~8% 可用 + ~5000× 加速

## preconditioner/ — 预条件论文路线
- [D12 块预条件论文计划（中文）](preconditioner/D12_PRECONDITIONER_PAPER_PLAN.md) — 谱分析 + 参数无关性
- [D13 学习型预条件论文计划（中文）](preconditioner/D13_LEARNED_PRECONDITIONER_PAPER_PLAN.md)

## routes/ — 其他研究路线规划
- [plan_gpu_multibackend](routes/plan_gpu_multibackend.md) — GPU + 多后端
- [plan_high_order_huzhang](routes/plan_high_order_huzhang.md) — Hu–Zhang 高次元定位 + 自适应延伸

## planning/ — 行动清单 / 实验收尾
- [P1 行动清单（Week 1–2）](planning/p1_action_checklist.md)

## archive/ — 会话记录归档
- [archive/m0_sessions/](archive/m0_sessions/) — M0 kickoff / session 报告

---

新增或移动这些文档引用的模块后，请同步两个语言版本并从仓库根运行：

```bash
python scripts/verify_huzhang_docs.py
```

该脚本只检查列出的路径是否仍存在，不改写正文。
