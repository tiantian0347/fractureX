# 辅助空间预条件子 — 退化弹性块（1/g(d)）验证报告

- 生成时间：2026-05-25T11:09:48
- 输出目录：`results/tests/auxspace_degraded_elastic`
- 总体结论：**通过**（24/24 组）

## 测试目的

在 **standard** Hu-Zhang 弹性装配下，应力块 `M(d)` 的积分离散系数为 **`1/g(d)`**（见
`HuZhangElasticAssembler._assemble_M_block_serial`）。当部分节点/单元 `d=1` 时，
`g(d)=eps_g`（如 `1e-6` 或 `1e-8`），对应局部 **`1/g(d) ~ 1e6` 或 `1e8`**，
这是相场裂纹形成后的典型病态情形。

本测试在 **冻结相场** 的前提下，对预设损伤场装配 `A`，用稀疏直接法 `spsolve` 作参考解，
检验 `solve_huzhang_block_gmres_auxspace`（`weighted_aux=True`）是否与直接法一致。

粗空间扩散权重为 **`max(g(d), eps_g)`**（Chen 等 2017 §5：Schur 辅助算子用 g 加权向量 Poisson / λ=0 弹性）。

## 配置

- 弹性形式：`standard`
- weighted_aux：`True`
- GMRES rtol：`1e-08`
- 网格 hmin：`0.02`（NN=2412, NC=4555）
- eps_g 列表：`[1e-06, 1e-08]`
- 损伤模式：intact, half_cracked, band_cracked, patch_cracked

## 损伤场模式

| 模式 | 说明 |
|---|---|
| `intact` | d=0 everywhere (g≈1, baseline) |
| `half_cracked` | d=1 on x>=0.5, d=0 elsewhere (mixed 1/eps_g scaling) |
| `band_cracked` | d=1 on |x-0.5|<0.05 band, d=0 elsewhere |
| `patch_cracked` | d=1 on disk center (0.75,0.5) r<0.12, d=0 elsewhere |

## 结果汇总

| 模式 | eps_g | 载荷 | d=1 占比 | g_min | max(1/g) | rel_diff | rel_res_aux | GMRES 迭代 | 通过 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| intact | 1e-06 | 0.0010 | 0.000 | 1.00e+00 | 1.00e+00 | 5.37e-12 | 2.61e-09 | 0 | 是 |
| intact | 1e-06 | 0.0050 | 0.000 | 1.00e+00 | 1.00e+00 | 4.91e-12 | 2.38e-09 | 0 | 是 |
| intact | 1e-06 | 0.0100 | 0.000 | 1.00e+00 | 1.00e+00 | 4.92e-12 | 2.38e-09 | 0 | 是 |
| half_cracked | 1e-06 | 0.0010 | 0.501 | 1.00e-06 | 1.00e+06 | 2.85e-10 | 6.70e-09 | 0 | 是 |
| half_cracked | 1e-06 | 0.0050 | 0.501 | 1.00e-06 | 1.00e+06 | 2.25e-10 | 6.51e-09 | 0 | 是 |
| half_cracked | 1e-06 | 0.0100 | 0.501 | 1.00e-06 | 1.00e+06 | 2.25e-10 | 6.53e-09 | 0 | 是 |
| band_cracked | 1e-06 | 0.0010 | 0.071 | 1.00e-06 | 1.00e+06 | 2.72e-11 | 3.72e-09 | 0 | 是 |
| band_cracked | 1e-06 | 0.0050 | 0.071 | 1.00e-06 | 1.00e+06 | 4.01e-11 | 3.98e-09 | 0 | 是 |
| band_cracked | 1e-06 | 0.0100 | 0.071 | 1.00e-06 | 1.00e+06 | 4.06e-11 | 4.03e-09 | 0 | 是 |
| patch_cracked | 1e-06 | 0.0010 | 0.043 | 1.00e-06 | 1.00e+06 | 4.58e-11 | 1.08e-08 | 0 | 是 |
| patch_cracked | 1e-06 | 0.0050 | 0.043 | 1.00e-06 | 1.00e+06 | 3.98e-11 | 8.10e-09 | 0 | 是 |
| patch_cracked | 1e-06 | 0.0100 | 0.043 | 1.00e-06 | 1.00e+06 | 3.97e-11 | 8.09e-09 | 0 | 是 |
| intact | 1e-08 | 0.0010 | 0.000 | 1.00e+00 | 1.00e+00 | 5.37e-12 | 2.61e-09 | 0 | 是 |
| intact | 1e-08 | 0.0050 | 0.000 | 1.00e+00 | 1.00e+00 | 4.91e-12 | 2.38e-09 | 0 | 是 |
| intact | 1e-08 | 0.0100 | 0.000 | 1.00e+00 | 1.00e+00 | 4.92e-12 | 2.38e-09 | 0 | 是 |
| half_cracked | 1e-08 | 0.0010 | 0.501 | 1.00e-08 | 1.00e+08 | 2.85e-10 | 7.61e-09 | 0 | 是 |
| half_cracked | 1e-08 | 0.0050 | 0.501 | 1.00e-08 | 1.00e+08 | 2.13e-10 | 5.70e-09 | 0 | 是 |
| half_cracked | 1e-08 | 0.0100 | 0.501 | 1.00e-08 | 1.00e+08 | 2.15e-10 | 6.34e-09 | 0 | 是 |
| band_cracked | 1e-08 | 0.0010 | 0.071 | 1.00e-08 | 1.00e+08 | 2.72e-11 | 3.72e-09 | 0 | 是 |
| band_cracked | 1e-08 | 0.0050 | 0.071 | 1.00e-08 | 1.00e+08 | 4.03e-11 | 4.00e-09 | 0 | 是 |
| band_cracked | 1e-08 | 0.0100 | 0.071 | 1.00e-08 | 1.00e+08 | 4.07e-11 | 4.06e-09 | 0 | 是 |
| patch_cracked | 1e-08 | 0.0010 | 0.043 | 1.00e-08 | 1.00e+08 | 4.58e-11 | 1.08e-08 | 0 | 是 |
| patch_cracked | 1e-08 | 0.0050 | 0.043 | 1.00e-08 | 1.00e+08 | 3.98e-11 | 8.11e-09 | 0 | 是 |
| patch_cracked | 1e-08 | 0.0100 | 0.043 | 1.00e-08 | 1.00e+08 | 3.98e-11 | 8.10e-09 | 0 | 是 |

## 验收标准

- `rel_diff = ||x_aux - x_direct|| / ||x_direct|| < 1e-5`
- `rel_res_aux = ||A x_aux - b|| / ||b|| < 1e-7`
- GMRES 收敛（`converged=True`）

## 运行方式

```bash
bash scripts/run_python.sh fracturex/tests/test_auxspace_precond_degraded_elastic.py
```

环境变量：`FRACTUREX_HMIN`、`FRACTUREX_EPS_G_LIST`、`FRACTUREX_RUN_SHORT=1`（快速子集）。
