# Auxiliary-Space Preconditioner — Degraded Elastic Block Test

- Generated at: 2026-05-25T11:09:48
- Output directory: `results/tests/auxspace_degraded_elastic`
- Overall: **PASS** (24/24 cases)

## Purpose

Verify `solve_huzhang_block_gmres_auxspace` on the Hu-Zhang mixed elastic system
with **standard** formulation: stress block `M(d)` uses integrator coefficient `1/g(d)`.
Nodes/cells with `d=1` yield `g(d)=eps_g`, hence local `1/g(d) ~ 1/eps_g` (e.g. `1e6` or `1e8`).
Coarse auxiliary diffusion uses `max(g(d), eps_g)` when `weighted_aux=True`.
Reference solution: sparse direct `spsolve`. Phase-field and damage evolution are **frozen**.

## Configuration

- elastic_formulation: `standard`
- weighted_aux: `True`
- GMRES rtol: `1e-08`
- hmin: `0.02`
- mesh: NN=2412, NC=4555
- eps_g values: `[1e-06, 1e-08]`
- damage patterns: intact, half_cracked, band_cracked, patch_cracked

## Damage patterns

| Pattern | Description |
|---|---|
| `intact` | d=0 everywhere (g≈1, baseline) |
| `half_cracked` | d=1 on x>=0.5, d=0 elsewhere (mixed 1/eps_g scaling) |
| `band_cracked` | d=1 on |x-0.5|<0.05 band, d=0 elsewhere |
| `patch_cracked` | d=1 on disk center (0.75,0.5) r<0.12, d=0 elsewhere |

## Results (per load / pattern / eps_g)

| pattern | eps_g | load | frac d=1 | g_min | inv_g_max | rel_diff | rel_res_aux | gmres_iters | PASS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| intact | 1e-06 | 0.0010 | 0.000 | 1.00e+00 | 1.00e+00 | 5.37e-12 | 2.61e-09 | 0 | yes |
| intact | 1e-06 | 0.0050 | 0.000 | 1.00e+00 | 1.00e+00 | 4.91e-12 | 2.38e-09 | 0 | yes |
| intact | 1e-06 | 0.0100 | 0.000 | 1.00e+00 | 1.00e+00 | 4.92e-12 | 2.38e-09 | 0 | yes |
| half_cracked | 1e-06 | 0.0010 | 0.501 | 1.00e-06 | 1.00e+06 | 2.85e-10 | 6.70e-09 | 0 | yes |
| half_cracked | 1e-06 | 0.0050 | 0.501 | 1.00e-06 | 1.00e+06 | 2.25e-10 | 6.51e-09 | 0 | yes |
| half_cracked | 1e-06 | 0.0100 | 0.501 | 1.00e-06 | 1.00e+06 | 2.25e-10 | 6.53e-09 | 0 | yes |
| band_cracked | 1e-06 | 0.0010 | 0.071 | 1.00e-06 | 1.00e+06 | 2.72e-11 | 3.72e-09 | 0 | yes |
| band_cracked | 1e-06 | 0.0050 | 0.071 | 1.00e-06 | 1.00e+06 | 4.01e-11 | 3.98e-09 | 0 | yes |
| band_cracked | 1e-06 | 0.0100 | 0.071 | 1.00e-06 | 1.00e+06 | 4.06e-11 | 4.03e-09 | 0 | yes |
| patch_cracked | 1e-06 | 0.0010 | 0.043 | 1.00e-06 | 1.00e+06 | 4.58e-11 | 1.08e-08 | 0 | yes |
| patch_cracked | 1e-06 | 0.0050 | 0.043 | 1.00e-06 | 1.00e+06 | 3.98e-11 | 8.10e-09 | 0 | yes |
| patch_cracked | 1e-06 | 0.0100 | 0.043 | 1.00e-06 | 1.00e+06 | 3.97e-11 | 8.09e-09 | 0 | yes |
| intact | 1e-08 | 0.0010 | 0.000 | 1.00e+00 | 1.00e+00 | 5.37e-12 | 2.61e-09 | 0 | yes |
| intact | 1e-08 | 0.0050 | 0.000 | 1.00e+00 | 1.00e+00 | 4.91e-12 | 2.38e-09 | 0 | yes |
| intact | 1e-08 | 0.0100 | 0.000 | 1.00e+00 | 1.00e+00 | 4.92e-12 | 2.38e-09 | 0 | yes |
| half_cracked | 1e-08 | 0.0010 | 0.501 | 1.00e-08 | 1.00e+08 | 2.85e-10 | 7.61e-09 | 0 | yes |
| half_cracked | 1e-08 | 0.0050 | 0.501 | 1.00e-08 | 1.00e+08 | 2.13e-10 | 5.70e-09 | 0 | yes |
| half_cracked | 1e-08 | 0.0100 | 0.501 | 1.00e-08 | 1.00e+08 | 2.15e-10 | 6.34e-09 | 0 | yes |
| band_cracked | 1e-08 | 0.0010 | 0.071 | 1.00e-08 | 1.00e+08 | 2.72e-11 | 3.72e-09 | 0 | yes |
| band_cracked | 1e-08 | 0.0050 | 0.071 | 1.00e-08 | 1.00e+08 | 4.03e-11 | 4.00e-09 | 0 | yes |
| band_cracked | 1e-08 | 0.0100 | 0.071 | 1.00e-08 | 1.00e+08 | 4.07e-11 | 4.06e-09 | 0 | yes |
| patch_cracked | 1e-08 | 0.0010 | 0.043 | 1.00e-08 | 1.00e+08 | 4.58e-11 | 1.08e-08 | 0 | yes |
| patch_cracked | 1e-08 | 0.0050 | 0.043 | 1.00e-08 | 1.00e+08 | 3.98e-11 | 8.11e-09 | 0 | yes |
| patch_cracked | 1e-08 | 0.0100 | 0.043 | 1.00e-08 | 1.00e+08 | 3.98e-11 | 8.10e-09 | 0 | yes |

## Acceptance criteria

- `rel_diff = ||x_aux - x_direct|| / ||x_direct|| < 1e-5`
- `rel_res_aux = ||A x_aux - b|| / ||b|| < 1e-7`
- GMRES reports converged (`converged=True`)

## Artifacts

- `results_detail.csv`: all numeric rows
- `summary.json`: aggregate pass/fail and metadata
- `comparison_by_epsg.json`: grouped metrics

> Auto-generated; re-run the test script to refresh.
