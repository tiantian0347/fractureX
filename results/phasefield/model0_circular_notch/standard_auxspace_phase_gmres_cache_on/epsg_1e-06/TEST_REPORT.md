# Model0 Hu-Zhang + Phase-Field Test Report

- Generated at: 2026-05-25T11:19:00
- Tag: `epsg_1e-06`
- Output directory: `results/phasefield/model0_circular_notch/standard_auxspace_phase_gmres_cache_on/epsg_1e-06`

## Run Configuration

- Solver mode: `standard:elastic_single_aux/phase_gmres_no_precond`
- Elastic formulation: `standard`
- history_source: `from_u`
- eps_g: `1.0e-06`
- hmin: `0.05`
- p: `3`
- use_relaxation: `True`

## Mesh and Material

- Mesh: NN=372, NE=1012, NC=640
- Material: E=200.0, nu=0.2, Gc=1.0, l0=0.02

## Key Metrics

| Metric | Value |
|---|---:|
| n_load_steps | 3 |
| n_converged_steps | 3 |
| step_convergence_rate | 1.0 |
| total_wall_s | 56.13507555425167 |
| avg_step_s | 18.710219606757164 |
| max_step_s | 29.078811705112457 |
| avg_nonlinear_iters | 3.3333333333333335 |
| max_nonlinear_iters | 5 |
| reaction_force_final (signed) | -11.745745174373162 |
| reaction_force_abs_final | 11.745745174373162 |
| damage_max_final | 0.03660841863418985 |

## Timing Breakdown

| Phase | Total (s) | Avg per step (s) |
|---|---:|---:|
| Elastic Assemble | 19.662841454148293 | 6.554280484716098 |
| Elastic Solve | 27.613552287220955 | 9.204517429073652 |
| Phase Assemble | 8.480876103043556 | 2.8269587010145187 |
| Phase Solve | 0.04762980341911316 | 0.015876601139704388 |

## Artifacts

- `summary.json`: machine-readable aggregate metrics
- `summary.csv`: flat metrics table
- `history.csv`: per-load-step history
- `meta.json`: run metadata
- `residual_force_vs_displacement.csv`: per-step displacement and |residual_force|
- `residual_force_vs_displacement.png`: residual-force/displacement curve

> This file is auto-generated and overwritten on each run.
