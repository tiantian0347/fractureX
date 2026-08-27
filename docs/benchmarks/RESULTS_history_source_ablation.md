# RESULT: `from_u` vs `from_sigma` history source (model0 coarse ablation)

## Purpose and scope

This experiment isolates the history-field source in the coupled Hu--Zhang
phase-field solve. It is a coarse diagnostic, not a mesh-converged benchmark:
the model0 mesh has 640 cells, 372 nodes, and `h_max=0.0782`, which is larger
than the paper target for `l0=0.02`. Both runs use the same mesh, 31 prescribed
displacements, standard mixed formulation, direct elastic solve, GMRES phase
solve, staggered tolerance `1e-5`, and no damage relaxation.

The stress-driven variant recovers the total strain at phase quadrature points
as `epsilon_h = C^{-1} sigma_h / g(d_h)` before applying the same positive-energy
split and irreversible maximum used by `from_u`.

## Results

| Metric | `from_u` | `from_sigma` |
|---|---:|---:|
| Peak reaction magnitude | 27.1181 | 27.2669 |
| Peak displacement | 0.0788 | 0.0810 |
| Final reaction magnitude | 0.7530 | 2.7188 |
| Final `max(d)` | 0.989724 | 1.0 |
| Final `max(H)` | 1.8936e4 | 6.0535e8 |
| Sum of staggered iterations | 2091 | 1837 |
| Maximum iterations in one step | 331 | 282 |

Cross-run metrics:

- reaction-curve relative L2 difference: `0.3274`;
- final damage-vector relative L2 difference: `0.2422`;
- maximum stepwise damage-vector relative L2 difference: `0.5907`;
- final damage correlation: `0.9628`;
- final crack-set Jaccard overlap: `0.7755` for `d>=0.5`, `0.5532` for
  `d>=0.95`.

Both sources produce the same qualitative topology: a horizontal crack band
above the circular hole. They do not produce the same coupled path. The
post-peak force response, propagation timing, and final crack-band intensity
differ substantially. The `from_sigma` history also grows by four additional
orders of magnitude after full localization. This is consistent with discrete
stress contamination being amplified by `1/g(d_h)` in nearly fully damaged
cells; it should not be presented as a harmless alternative history definition.

## Artifacts

- `from_u`: `/Users/tian00/repository/results/phasefield/model0_circular_notch/paper_direct_history_ablation_h005_from_u_gmres/epsg_1e-06`
- `from_sigma`: `/Users/tian00/repository/results/phasefield/model0_circular_notch/paper_direct_history_ablation_h005_from_sigma_gmres/epsg_1e-06`
- comparison: `/Users/tian00/repository/results/phasefield/model0_circular_notch/history_source_ablation_h005_comparison`
- analysis script: `/Users/tian00/repository/Tian/thesis/fracture_huzhang/scripts/analyze_history_source_ablation.py`

The comparison directory contains `comparison.json`, `comparison_by_step.csv`,
and `history_source_comparison.{png,pdf}`. The JSON records the input paths,
mesh size, thresholds, and creation timestamp.

## Interpretation limit

This one coarse, symmetric benchmark establishes that the choice affects the
coupled solution. It does not determine which source is mesh-consistent. Before
changing the paper's production model, repeat the comparison on one resolved
mesh and inspect whether the `from_sigma` history amplification decreases or
persists under refinement and/or an effective-stress formulation.
