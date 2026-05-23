# FractureX: Hu–Zhang mixed element + phase-field damage — architecture and configuration

This document is intended for **external technical communication** and **onboarding**. It summarizes how `fracturex` is structured for **Hu–Zhang stress–displacement mixed finite elements** and **phase-field damage**: layering, data flow, and configuration. The stack builds on **FEALPy** for mesh, function spaces, integration, and sparse linear algebra; this repository adds **cases, discretization, assembly, drivers, and postprocessing**.

**Chinese version (同构维护)**: [HUZHANG_PHASEFIELD_ARCHITECTURE.md](HUZHANG_PHASEFIELD_ARCHITECTURE.md)

---

## 1. Problem and scope

- **Mechanics**: 2D linear elasticity with a **Hu–Zhang type mixed element** (stress \(\tilde\sigma\) and displacement \(u\) as primary variables), with **corner relaxation** (via `TM` and Neumann stress edges `bd_stress` together with FEALPy’s `HuZhangFESpace2d`).
- **Damage / fracture**: **Phase-field** scalar damage \(d\) (continuous Lagrange space) with **staggered** iteration with elasticity: at fixed current \(d\), solve elasticity → update history and phase-field equation → update \(d\) (with irreversibility).
- **History**: **quadrature-point history** \(H\) driving the phase field is stored in `state.H`, with the same cell–quadrature layout; it is separate from nodal \(d\), which matches AT-type crack surface density and spectral energy drivers.

The current discretization is restricted to **2D triangular meshes** (`GD==2` checked in `HuZhangDiscretization`).

---

## 2. Layered architecture (logical view)

```
CaseBase (physics: mesh, material, BCs, optional phase-field BC/initial data)
        │
        ▼
HuZhangDiscretization (mesh + FE spaces + HuZhangState)
        │
        ├── DamageModelBase (e.g. PhaseFieldDamageModel: AT density, degradation, split, H update)
        │
        ├── HuZhangElasticAssembler  ──► ElasticSystem(A, F, decode, meta)
        ├── PhaseFieldAssembler      ──► PhaseFieldSystem(A, F, decode, meta)
        │
        ▼
HuZhangPhaseFieldStaggeredDriver (load steps + staggered iteration + optional record/timing)
        │
        ├── elastic_solver(A, F)   # default SciPy spsolve; override with GMRES/FEALPy etc.
        ├── phase_solver(A, F)     # default no-preconditioner LGMRES
        │
        ▼
RunRecorder / reaction (CSV, NPZ, reactions)
```

**Principles**: the **Case** only describes physics and boundaries; **Discretization** only manages mesh and DOFs; **Assemblers** only build algebraic systems from `state` + `case`; the **Driver** sequences load steps and staggered iteration and calls solvers.

**Performance tuning**: optional **cell parallel assembly**, **`TMᵀ M TM`** block products, overlapped **effective-stress `B2`** branches, batched **history** updates, **`FRACTUREX_PROFILE_HISTORY_UPDATE`**, and **VTK high-order sampling** defaults are documented in **§5.5** / **§3.5** (Chinese **§7.1**). Treat **`save_vtkfile_highorder`** sampling time as **post-processing only**, not core solve benchmarks.

---

## 3. Core modules

### 3.1 Case layer: `fracturex/cases/base.py` — `CaseBase`

Typical methods to implement or override:

| API | Role |
|-----|------|
| `make_mesh(...)` | Create or connect a `TriangleMesh` |
| `model()` | Material object (at least `Gc`, `l0`; elastic moduli from `lam`/`mu` or `E`/`nu`, etc.) |
| `isD_bd(points)` | At boundary points or edge barycenters, whether the edge is **Dirichlet** (for Neumann stress edge sets) |
| `dirichlet_pieces(load)` | **Piecewise** displacement Dirichlet: list of `DirichletPiece` (`threshold`, `value`, `direction`, `tag`) |
| `neumann_data(load)` | Optional: stress-essential / traction data for `HuzhangStressBoundaryCondition` |
| `body_force(points)` | Optional body force |
| `phasefield_dirichlet_data(load)` | Optional Dirichlet data for damage \(d\) |
| `phasefield_initial_damage_data(load)` | Optional **one-shot** initial damage (e.g. crack line with \(d=1\)) before the staggered loop |
| `crack_edge_mask(mesh)` | Optional edges for boundary enhancement, etc. |

Load segment: `load_dirichlet_piece(load)` prefers the piece with `tag=="load"`, otherwise the last in the list.

### 3.2 Discretization: `fracturex/discretization/huzhang_discretization.py`

- **`HuZhangDiscretization`**: `build()` creates
  - `space_sigma`: `HuZhangFESpace2d` (order `p`, `use_relaxation`, `bd_stress=isNedge`)
  - `space_u`: `TensorFunctionSpace(Lagrange(p_u), shape=(2,-1))`, default `p_u = p - 1`
  - `space_d`: `LagrangeFESpace` with order `damage_p` (phase field / local history often **P1**)
- **`HuZhangState`**: `sigma`, `u`, `d`, `r_hist`, and phase-field **`H`** (may be `None` until damage `on_build` or first history update).

`rebuild_on_new_mesh` supports adaptive remeshing with optional `transfer`.

### 3.3 Damage model: `fracturex/damage/phasefield_damage.py` — `PhaseFieldDamageModel`

**Dataclass fields** (constructor) include:

- `density_type`: crack surface density, e.g. `"AT2"` / `"AT1"` (`CrackSurfaceDensityFunction`)
- `degradation_type`: elastic degradation, e.g. `"quadratic"` (`EnergyDegradationFunction`)
- `split`: history energy positive part: `"hybrid"` / `"spectral"` / `"isotropic"`
- `history_source`: how history is built; the implementation currently supports **only `"from_u"`** (strain from `u`’s gradient, then \(\psi^+\) drives \(H\))
- `eps_g`: lower bound in degradation, for stability
- `clamp_max`: upper clamp for damage
- `debug`: debug output

Lifecycle: in `on_build(discr, state, case)`, read `Gc`, `l0`, Lamé constants from `case.model()`, initialize `_gfun`, `_hfun`, set `state.H` to `None` until first quadrature history fill.

### 3.4 Elastic assembly: `fracturex/assemblers/huzhang_elastic_assembler.py`

- **Standard formulation**: block system `[M(d), B; B^T, 0]` with degraded `M(d)`; `B` and corner-related `B2`, `M2` with `TM`, etc.
- **Effective-stress formulation**: constant stiffness blocks with damage-coupled `B2(d)` (see source branches and debug output).

**Performance**:

- `_prepare_constant_blocks(load)`: caches `TM`, `B2_const`, body load vector, `M2_const` (effective branch) independent of current \(d\); keys depend on mesh and DOF size.
- **`begin_load_step(load)`** (per load step): before inner staggered iterations, reuse **piecewise Dirichlet list**, standard formulation’s **`r_dir`**, and **Neumann/essential stress** `set_essential_bc` for the step; under `effective_stress`, `r_dir` may still be updated every stagger iteration. If not called, `assemble` behavior matches the legacy path.

### 3.5 Phase-field assembly: `fracturex/assemblers/phasefield_assembler.py`

Conceptual steps:

1. Fix quadrature (order `q` tied to `damage_p`);
2. `damage.update_history_on_quadrature(...)` updates **`state.H`**;
3. Assemble `A` (diffusion + mass; AT1/AT2 terms independent of \(H\) may be cached) and `F` (incremental `A dd = rhs - A d_old`);
4. `DirichletBC` for phase-field boundaries.

**`begin_load_step(load)`**: caches `phasefield_dirichlet_data(load)` and warms quadrature data to avoid repeated case queries in the same load step.

**History update performance**: Step 2 is dominated by large tensor work (`grad_u`, spectral split, `maximum`, etc.). When **`assembly_parallel`** is enabled on `PhaseFieldAssembler`, Step 2 uses **batched cell-parallel** updates by default (override with **`FRACTUREX_HISTORY_UPDATE_PARALLEL=0`**). Compare with BLAS/OpenMP thread settings if scalability is unclear; optional wall-clock line per call: **`FRACTUREX_PROFILE_HISTORY_UPDATE=1`** (§5.5).

### 3.6 Boundary conditions: `fracturex/boundarycondition/huzhang_boundary_condition.py`

- From `CaseBase.isD_bd` and the mesh, build **Neumann stress edge** sets (`build_isNedge_from_isD`) for the Hu–Zhang space.
- `HuzhangBoundaryCondition`: displacement Dirichlet contribution to the \(\sigma\) equation, etc.
- `HuzhangStressBoundaryCondition`: stress BC elimination into the global system.

### 3.7 Driver: `fracturex/drivers/huzhang_phasefield_staggered.py` — `HuZhangPhaseFieldStaggeredDriver`

**Responsibilities**:

- `initialize()`: one-shot init including `damage.on_build`.
- `run(loads)`: for each scalar load, `solve_one_step`.
- **`solve_one_step(step, load)`**:
  1. Optional: initial damage from case;
  2. **`elastic_assembler.begin_load_step(load)`** and **`phase_assembler.begin_load_step(load)`**;
  3. Staggered loop: assemble & solve elastic → assemble & solve phase field → irreversibility `d <- max(d_old, d_trial)` → convergence (normalized increment vs first iteration, etc.);
  4. Reaction, metadata, `StepInfo`.

**Constructor options**:

| Parameter | Meaning |
|-----------|---------|
| `tol`, `maxit` | Staggered outer tolerance and max iterations |
| `elastic_solver`, `phase_solver` | Injectable `(A, F) -> x` or `(x, info)`; defaults: elastic `spsolve`, phase no-preconditioner `lgmres` |
| `compute_linear_residual` | Whether to record linear residual of the last solve |
| `debug`, `timing` | Debug and FEALPy `timer` tags |
| `recorder` | `RunRecorder` (below) |
| `adapt_hook` | Optional per–load-step remesh and `discr` / assembler sync |

**Other Hu–Zhang drivers** (not the main phase-field path): `huzhang_damage_staggered.py` (local damage), `huzhang_fe_solve.py` / `huzhang_damage_solve.py` (demos/legacy).

### 3.8 Linear algebra: `fracturex/utilfuc/linear_solvers.py`

Block **ILU-preconditioned GMRES**, **block Krylov** (optional FEALPy `gmres`/`minres`), and **auxiliary-space preconditioned GMRES** (coarse diffusion + FEALPy `GAMGSolver` + Schur approximation, etc.). Tests can inject a custom `elastic_solver`; production/paper runs often compare with **direct `spsolve`**.

### 3.9 Postprocessing: `fracturex/postprocess/`

- **`RunRecorder`**: `meta.json`, `history.csv` (per load step), `iterations.csv` (per inner iteration if the driver writes), optional `checkpoints/step_XXX.npz` (`save_npz` + `save_every`).
- **`reaction_from_sigma`**: scalar reaction from \(\sigma\) and load boundary thresholds, for `StepInfo.meta`.

---

## 4. Typical data flow (one load step)

1. Given scalar **`load`**, `begin_load_step(load)` warms per-step BC and phase-field Dirichlet descriptions.
2. Staggered iteration \(k\):
   - `HuZhangElasticAssembler.assemble(load)` → `A`, `F`;
   - `elastic_solver(A, F)` → `state.sigma`, `state.u`;
   - `PhaseFieldAssembler.assemble(load)` (updates `H` internally) → phase-field system;
   - `phase_solver` → `d_trial`, then **irreversible `max`** and \([0,1]\) clamp;
   - If normalized \(\|\Delta u\|\), \(\|\Delta d\|\) are below `tol`, converge.
3. Reactions, timing, linear solver info go to `StepInfo` and optional `RunRecorder`.

---

## 5. Configuration and extension (what you can set)

### 5.1 Discretization and mesh

- Hu–Zhang stress order **`p`**, **`use_relaxation`**, phase-field order **`damage_p`**, displacement order **`u_space_order`** (default `p-1`).
- Mesh from **`case.make_mesh`** or `discr.build(mesh=...)`.

### 5.2 Phase-field physics

- AT1/AT2, degradation, spectral split, `eps_g`, `history_source` (currently only `from_u`).
- **Case** extensions via `CaseBase.phasefield_*` for BCs and initial cracks.

### 5.3 Elastic formulation in assembly

- **`formulation="standard"`** vs **`"effective_stress"`** (see `HuZhangElasticAssembler` docstring and code).
- **Mutually exclusive degradation placement**: `standard` puts **g(d)** on the stress block **M**; `effective_stress` puts **g(d)** on the coupling block **B** — never both at once.

### 5.4 Solvers

- **Default**: elastic uses SciPy **`spsolve`**; phase-field uses no-preconditioner SciPy **`lgmres`**.
- **Extensions**: pass custom **`elastic_solver` / `phase_solver`** to `HuZhangPhaseFieldStaggeredDriver` (e.g. FEALPy Krylov, `solve_huzhang_block_gmres_auxspace` from this repo).
- **GMRES preconditioners** (`linear_solvers.py`): pass the same **`elastic_formulation`** as the assembler, plus **`damage`** and **`state`** when using weighted coarse diffusion. **`weighted_aux=True`** applies **g(d)²** on the P1 coarse Laplacian only for **`standard`**; for **`effective_stress`** the coarse Laplacian stays unweighted and damage enters via the Schur block from **B(d)**. See `docs/HUZHANG_INTERFACE_TEST_MANUAL.md` §3 (Chinese).
- **Standard Schur**: for fixed \(d_h\), \(\mathcal K_h=[[A(d_h),B^\top],[B,0]]\), \(S(d_h)=B A(d_h)^{-1}B^\top\); code builds \(\widehat S=B\,\mathrm{diag}(A)^{-1}B^\top\) and approximates \(B_A\), \(B_S\) (manual §3.0).

### 5.5 Environment variables (aux-space, assembly parallel, history profiling)

| Variable | Role |
|----------|------|
| **`FRACTUREX_ASSEMBLY_PARALLEL`** | When `1` / `true` / `yes` / `on`, enables **cell-chunk parallel assembly** where implemented: `HuZhangElasticAssembler` (e.g. standard `M(d)` block), `PhaseFieldAssembler` (`A_const`, `A_hist`, RHS), and **`effective_stress`** B2-related blocks. Assembler kwargs `assembly_parallel` / `assembly_nproc` override the env defaults when set. |
| **`FRACTUREX_ASSEMBLY_NPROC`** | Cap on worker threads (positive integer); defaults toward available CPUs if unset. |
| **`FRACTUREX_PROFILE_HISTORY_UPDATE`** | When enabled, `PhaseFieldAssembler.assemble` prints one line per **`update_history_on_quadrature`** call (`wall_s`, `NC`, `NQ`, `split`) for quick comparisons across meshes and thread settings. |
| **`FRACTUREX_HISTORY_UPDATE_PARALLEL`** | If `0` / `false` / `off`, forces a **serial** quadrature history update even when **`assembly_parallel`** is on (A/B testing). Otherwise history uses **batched cell-parallel** threads when assembly parallel is enabled (via `PhaseFieldAssembler` kwargs). |
| **`FRACTUREX_VTK_HIGHORDER_PARALLEL`** | Chunk-parallel FE sampling inside **`save_vtkfile_highorder`**; **default on** (`1`). Set `0` to disable. This cost is **visualization-only** and should **not** be folded into core staggered-solve / assembly timings in papers. |
| **`FRACTUREX_AUXSPACE_DEBUG`** (and related) | Iterative solvers / aux-space preconditioners may emit detailed logs (see `linear_solvers._auxspace_log` and related comments). |

**Also when `assembly_parallel` is on**: **standard** elasticity builds **`M2 = TMᵀ M TM`** with **column-block parallel sparse matmul** (two stages, fallback on failure); **effective-stress** **`B2`** overlaps **g·div assembly + `TMᵀ·`** with the **∇g correction** branch on **two threads**.

For reproducibility and tuning: combine **switchable Krylov + preconditioning**, the env flags above, and optional **history profiling**; compare batched history updates vs. BLAS-only parallelism if needed.

---

## 6. Reference entry script

- **`fracturex/tests/phasefield_model0_huzhang.py`**: notched disk Model0, Hu–Zhang + staggered phase field, load sequence, `RunRecorder` output and paper-oriented summary export; includes **direct vs iterative elastic** toggles, useful as a public demo template.

Smaller cases: `phasefield_square_tension.py`, `test_domain.py`, etc.

---

## 7. Evolution directions (brief)

- **Staged assembly**: `begin_load_step(load)` is in place; further split of `assemble` into `update_history` / `assemble_matrix` could help parallel/element-wise work.
- **Linear system object**: bind preconditioner lifetime to a discrete instance for repeated solves.
- **I/O**: buffer iteration-level CSV for large staggered runs.

---

## 8. Mapping to “plain” FEALPy

FractureX does **not** replace FEALPy; it adds a **domain layer** on top. Rough mapping:

| Typical FEALPy role | In this path |
|---------------------|-------------|
| `TriangleMesh` | `CaseBase.make_mesh` / `HuZhangDiscretization.build` |
| `HuZhangFESpace2d` | `discr.space_sigma` (`fealpy.functionspace.huzhang_fe_space_2d`) |
| `LagrangeFESpace` + `TensorFunctionSpace` | `discr.space_u`, `discr.space_d` |
| `HuZhangStressIntegrator` / `HuZhangMixIntegrator` | Inside `HuZhangElasticAssembler` via `BilinearForm` |
| `BilinearForm` / `LinearForm` / `DirichletBC` | Elastic and phase-field assembly and BCs |
| `backend_manager as bm` | Tensor/mesh entity conventions match FEALPy |
| `spsolve`, `lgmres`, or `fealpy.solver.*` | Injected `elastic_solver` / `phase_solver` in `HuZhangPhaseFieldStaggeredDriver`; this repo’s `fracturex/utilfuc/linear_solvers.py` adds block Krylov, aux-space, etc. |

**Relation to the non–Hu–Zhang phase-field main line**: `fracturex/phasefield/main_solve.py` targets **standard displacement–Lagrange phase field**, while **`HuZhangPhaseFieldStaggeredDriver` + `HuZhangDiscretization`** form a **mixed + phase-field** stack. They can be read side by side but are not interchangeable as one `main` class.

**FEALPy version note**: `HuZhangFESpace2d`, `GAMGSolver`, `solver.gmres`, etc. may change slightly across FEALPy versions; this repo mitigates common differences in `linear_solvers`. After upgrading FEALPy, use tests and `python scripts/verify_huzhang_docs.py` as a gate.

---

## 9. English/Chinese sync and maintenance

- **English (this file)** and **Chinese** [HUZHANG_PHASEFIELD_ARCHITECTURE.md](HUZHANG_PHASEFIELD_ARCHITECTURE.md) are maintained **in sync on substance** (section numbering differs; Chinese **§7.1** ↔ English **§3.5** history notes + **§5.5** env table).
- **Automated check (not auto-generated prose)**: from the repo root run  
  `python scripts/verify_huzhang_docs.py`  
  to verify that **key source paths** listed in the script still exist. When you **add or move modules**, update both `.md` files and the manifest at the top of the script.
- **Index**: [docs/README.md](README.md)

---

## 10. Repository convention

- Agent/collaboration notes: **`AGENT.md`** at the repo root (if present).
- This file: **`docs/HUZHANG_PHASEFIELD_ARCHITECTURE.en.md`**. For significant API or default-behavior changes, **update the Chinese file in sync** and run `python scripts/verify_huzhang_docs.py`.

---

*Class names and paths are authoritative in the repository; narrative text is not auto-generated and must be updated by developers.*
