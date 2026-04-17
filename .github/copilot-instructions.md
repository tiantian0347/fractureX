# Copilot Instructions for fractureX

## Project architecture (read this first)
- `fracturex` is FEALPy-based fracture simulation code with two main solver tracks:
  1. **Hu-Zhang + local damage** (`fracturex/drivers/huzhang_damage_staggered.py`)
  2. **Hu-Zhang + phase-field** (`fracturex/drivers/huzhang_phasefield_staggered.py`)
- Core runtime flow is: **Case -> Discretization -> DamageModel -> Assembler(s) -> Driver -> Postprocess**.
- `fracturex/cases/*.py` defines geometry, material access, Dirichlet/Neumann partitioning (`isD_bd`, `dirichlet_pieces`, `neumann_data`).
- `fracturex/discretization/huzhang_discretization.py` owns mesh/spaces/state only; it should not contain solver logic.
- `HuZhangState` stores mutable FE fields (`sigma`, `u`, `d`, `r_hist`, `H`); drivers/assemblers exchange this through `DamageStateView` (`fracturex/damage/base.py`).

## Critical domain conventions
- Material objects are duck-typed. For Lamé constants, code accepts any of:
  - `lam` + `mu`, or
  - `lambda0` + `lambda1`, or
  - `E` + `nu`
  (see `_material_lame_from_model` in damage models and `_lame` in elastic assembler).
- **Boundary classification drives Hu-Zhang corner relaxation**: `isNedge` is derived from `isD_bd` via `build_isNedge_from_isD(...)`.
- For fractured meshes, crack-edge metadata is injected by `augment_boundary_edges_inplace(...)` during discretization build.

## Phase-field-specific gotchas
- In `PhaseFieldDamageModel`, `state.H` is **quadrature history** (`(NC, NQ)` array), not a nodal FE function.
- `PhaseFieldAssembler` must update and consume `H` on the **same quadrature rule** (same `q`) before assembling `A`/`F`.
- Damage irreversibility is enforced in driver (`d <- max(d_old, d_trial)` and clipping to `[0,1]`).

## Local damage-specific gotchas
- `LocalNodeDamage.update_after_elastic(...)` computes strain from `u.grad_value`, reconstructs equivalent stress, then scatters cell values back to nodes.
- Local damage history uses nodal `r_hist` and irreversible updates (`max` on both `r_hist` and `d`).

## Developer workflows (current repo reality)
- `requirements.txt` is empty; dependencies are not auto-pinned. Install manually (at least `fealpy`, `numpy`, `scipy`; plotting/tests also use `matplotlib`, `pytest`, `ipdb`).
- Common run scripts are under `fracturex/tests/` but several are script-style examples, not clean unit tests.
- Useful entry scripts:
  - `python3 fracturex/tests/phasefield_square_tension.py`
  - `python3 fracturex/tests/smoke_run_square_tension.py`
- If you see `ModuleNotFoundError: fealpy.backend`, your FEALPy installation/version is incompatible or missing.

## Editing guidance for agents
- Keep solver separation: discretization/state classes should stay assembly/iteration-free.
- When changing boundary logic, update both case methods and any Hu-Zhang BC adapter usage (`fracturex/boundarycondition/huzhang_boundary_condition.py`).
- Preserve `bm` (`fealpy.backend.backend_manager`) tensor ops in solver paths; avoid silently replacing with raw NumPy operations on FE fields.
- Preserve output conventions: VTK under `results/`; optional run logs/checkpoints via `RunRecorder` (`fracturex/postprocess/recorder.py`).
- Prefer extending existing case/driver patterns instead of introducing parallel APIs.
