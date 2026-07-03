# Design: NEPIN Damage-Only Preconditioner for the Staggered Driver

> Status: v0.1 (2026-07-03). Design-only; theory in
> [`THEORY_nonlinear_elimination.md`](THEORY_nonlinear_elimination.md);
> code in `fracturex/analysis/nonlinear_elimination.py` (to be created);
> tests in `fracturex/tests/analysis/test_nonlinear_elimination.py`.
>
> Related: [`GONG_THESIS_ABSORPTION.md`](../planning/GONG_THESIS_ABSORPTION.md) §2,
> [`DESIGN_affine_invariant_diagnostics.md`](DESIGN_affine_invariant_diagnostics.md),
> multi-backend convention
> [`multibackend_convention.md`](../architecture/multibackend_convention.md).

---

## 0. Scope

**This design covers only the "damage-only" NEPIN variant** (Theory §2.2):
eliminate strong nonlinearity on the damage field `d` over the localized
subset $\Omega_s = \{d > d_c\}$; leave the mechanical block untouched.

**Explicitly out of scope**:
- Full σ-u-d NEPIN (would require touching the Hu–Zhang mixed
  assembler and its scipy/pardiso boundary).
- Adaptive $d_c$ selection.
- Driver integration (a separate follow-up task after unit tests green).
- Distributed / MPI variants.

The unit under design is a **library module** callable from a driver
hook or from an offline spike script; it does not modify the driver.

---

## 1. Design goals

1. **Sidecar, not driver rewrite.** The staggered driver
   `huzhang_phasefield_staggered.py` is 2000+ LOC and stable. NEPIN is
   invoked as an *optional pre-step* inside the outer staggered loop, or
   run offline on stored checkpoints in the spike experiment.
2. **Multi-backend `bm` from line 1.** All compute uses
   `fealpy.backend.backend_manager`; `numpy` and scipy live only at the
   linear-solver / I/O boundary. Matches the pattern of
   `analysis/affine_invariant.py` and `ml/coarse_features.py`.
3. **Small blast radius.** No changes outside `fracturex/analysis/`
   and its tests. Driver hook is a **single new argument**
   (`nepin_precond=None`), disabled by default; failure of the hook
   must not affect the plain path.
4. **Testable without fracturex assembly.** Synthetic high-contrast
   nonlinear systems (see §5) reproduce the L_S/L_{S^c} ratio; unit
   tests do not need mesh, spaces, or the driver.
5. **Cheap local solve.** Local damage subproblem factors into a
   scipy-direct call at the I/O boundary; expected $|S| \sim 10^4$
   makes this negligible against the global GMRES.

---

## 2. Module layout

```
fracturex/
  analysis/
    __init__.py                     # add exports
    affine_invariant.py             # (existing, Ch. 8)
    nonlinear_elimination.py        # NEW — this task
  tests/
    analysis/
      __init__.py                   # (existing)
      test_affine_invariant.py      # (existing)
      test_nonlinear_elimination.py # NEW — this task
```

### 2.1 Public API sketch

```python
# fracturex/analysis/nonlinear_elimination.py
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence
from fealpy.backend import backend_manager as bm


# ---- Protocols (backend-neutral) -------------------------------------

class DamageResidual(Protocol):
    """Callable: given (d_full, x_frozen) -> full R_d vector.

    ``d_full``   : bm array of shape (n_d,), current damage dof vector
    ``x_frozen`` : bag of everything the residual needs that isn't
                   perturbed by the local solve (u, history, load, mesh
                   masks). Opaque to us.
    Returns: bm array (n_d,) — the damage-block residual R_d(u, d).
    """
    def __call__(self, d_full: Any, x_frozen: Any) -> Any: ...


class DamageJacobian(Protocol):
    """Callable: given (d_full, x_frozen, subset_dofs) -> local Jacobian.

    Returns a *dense* bm matrix of shape (|S|, |S|), the sub-Jacobian
    ``J_SS = d R_d / d d`` restricted to ``subset_dofs``. Only used for
    small |S| (~ 1e4). For larger |S| we'd swap in a sparse variant.
    """
    def __call__(
        self, d_full: Any, x_frozen: Any, subset_dofs: Any,
    ) -> Any: ...


# ---- Config + records -----------------------------------------------

@dataclass(frozen=True)
class NEPINConfig:
    """Configuration for one NEPIN elimination call.

    Parameters
    ----------
    d_c : float
        Threshold defining Omega_s = {d > d_c}.  Default 0.82 matches
        the paper_aux localization hard-wall.
    max_local_iter : int
        Cap on inner Newton iterations for the local damage solve.
    local_tol : float
        Inner stopping criterion: ||F_S(y)|| <= local_tol * ||F_S(0)||.
        Cai-Keyes recommend 1e-2 (deliberately loose).
    damping : float
        Newton step damping in [0, 1]; 1.0 = full step. Reduced only
        if a step increases ||F_S||.
    include_interface : bool
        If True, use max-vertex indicator (elements adjacent to Omega_s
        count too); if False, use element-mean indicator.
    """
    d_c: float = 0.82
    max_local_iter: int = 5
    local_tol: float = 1e-2
    damping: float = 1.0
    include_interface: bool = True


@dataclass(frozen=True)
class NEPINResult:
    """Report of one NEPIN elimination call.

    Fields
    ------
    d_corrected : Any
        bm array (n_d,) — the corrected damage after T_S.
    subset_size : int
        |S| — number of eliminated dofs.
    local_iters : int
        Actual inner Newton iterations used.
    local_res_reduction : float
        ||F_S(y_final)|| / ||F_S(0)||; sanity check.
    converged : bool
        True iff reduction met local_tol.
    wall_time : float
        seconds spent in the elimination call.
    """
    d_corrected: Any
    subset_size: int
    local_iters: int
    local_res_reduction: float
    converged: bool
    wall_time: float


# ---- Subset identifier ----------------------------------------------

def identify_subset(
    d_full: Any, *, d_c: float, include_interface: bool = True,
) -> Any:
    """Return a boolean bm mask of shape (n_d,) marking dofs in Omega_s.

    * ``include_interface=True`` : mark dof i if any element sharing
      dof i has max(d) > d_c (over-inclusive; catches interface).
    * ``include_interface=False`` : mark dof i if its scalar value
      d[i] > d_c (strict node-wise).

    The first variant needs mesh connectivity; to keep this module
    mesh-free, we expose a signature that accepts a precomputed
    element-to-dof incidence and defer connectivity to the caller.
    For the unit tests we use ``include_interface=False`` (no mesh).
    """
```

### 2.2 Core class

```python
class NEPINEliminator:
    """Damage-only NEPIN nonlinear-elimination preconditioner.

    Given callbacks (DamageResidual, DamageJacobian) that describe the
    coupled system in a backend-neutral way, ``eliminate(state, x_frozen)``
    performs one local Newton on Omega_s and returns a corrected state.
    """

    def __init__(
        self,
        residual: DamageResidual,
        jacobian: DamageJacobian,
        config: Optional[NEPINConfig] = None,
    ):
        self._F = residual
        self._J = jacobian
        self.config = config or NEPINConfig()

    def eliminate(
        self, d_full: Any, x_frozen: Any, *, subset_mask: Optional[Any] = None,
    ) -> NEPINResult:
        """Perform one local Newton on Omega_s(d_full) and return the
        corrected damage.

        If ``subset_mask`` is provided (bm bool array shape (n_d,)), use
        it; otherwise compute it from ``d_full`` and ``config``.
        """
        ...
```

### 2.3 Local solver internals

```python
    def _local_solve(self, F0: Any, J0: Any) -> Any:
        """Dense LU on the local Jacobian J0 of shape (|S|, |S|).

        Boundary: bm -> numpy for scipy factor, numpy -> bm on return.
        Per multi-backend convention §3, scipy path stays numpy.
        """
        import numpy as np
        from scipy.linalg import lu_factor, lu_solve
        F0_np = bm.to_numpy(F0)
        J0_np = bm.to_numpy(J0)
        lu, piv = lu_factor(J0_np, check_finite=False)
        y_np = lu_solve((lu, piv), -F0_np, check_finite=False)
        return bm.asarray(y_np)
```

### 2.4 Multi-backend contract

| Op | Backend path |
|---|---|
| ``d_full``, ``F_S``, subset mask | bm |
| Boolean indexing / where | ``bm.where`` (no ``a[mask] = ...``) |
| Norms, dot products | ``bm.linalg.norm``, ``bm.sum`` |
| Local LU | numpy/scipy at boundary (§3 exemption) |
| Return value | bm |

No `np.add.at`, no in-place fancy assignment; the reference is
`ml/coarse_features.py`.

---

## 3. Interaction with existing modules

### 3.1 With `analysis/affine_invariant.py`

Independent modules, but the spike experiment couples them: run NEPIN
during staggered iteration, record `dd_abs` and `du_abs` as usual, feed
the resulting `iterations.csv` through `AffineInvariantMonitor`. The
ω̂ curve is the empirical proof of NEPIN's Prop. of Theory §1.2.

**No API coupling** — each module works standalone.

### 3.2 With the staggered driver

For the spike experiment, we do **not** modify the driver in this task.
Instead, we run a *standalone* script `scripts/nepin_spike.py` that:

1. Loads a paper_aux checkpoint (state at `max d ≈ 0.82`).
2. Reconstructs the residual + Jacobian callbacks by wrapping the
   existing `phase_assembler` (which already knows how to build R_d
   and J_dd; we bind `u` and `H` frozen).
3. Runs `NEPINEliminator.eliminate` once.
4. Reports `NEPINResult`, then hands the corrected `d` back to a
   standard staggered iteration and measures the outer iter count.

**Driver integration** is a M4 milestone (Theory §5) after M3 tests green.

### 3.3 With `fracturex/utilfuc/linear_solvers.py`

None. The local solve is dense LU via scipy; the global inner GMRES
(elastic block) is untouched. NEPIN reduces the *outer* Newton work.

---

## 4. Callback wrappers (bind at spike time)

The two callbacks are constructed once from the driver's
`phase_assembler`:

```python
# scripts/nepin_spike.py (sketch)
def build_damage_residual(phase_assembler, u_frozen, H_frozen, load):
    def R_d(d_full, x_frozen):
        # temporarily set state.d = d_full, evaluate R_d = K_d d - f_d
        # (matches phase_assembler.assemble(load) internal residual)
        ...
    return R_d

def build_damage_jacobian(phase_assembler, u_frozen, H_frozen, load):
    def J_dd(d_full, x_frozen, subset_dofs):
        # K_d(H_frozen)[subset, subset] — subset of the assembled matrix
        ...
    return J_dd
```

These wrappers stay **outside** the analysis module; the module accepts
callables and doesn't know about fracturex assemblers. This keeps unit
tests mesh-free (§5).

---

## 5. Test plan

`tests/analysis/test_nonlinear_elimination.py`:

### 5.1 Unit tests (synthetic, mesh-free)

**T1: two-region high-contrast quadratic residual.**
Construct

$$F(x) = \begin{pmatrix}
   L_S \cdot x_S \odot x_S - b_S \\
   L_{S^c} \cdot x_{S^c} \odot x_{S^c} - b_{S^c}
\end{pmatrix}
\quad \text{with } L_S = 10^6,\ L_{S^c} = 1$$

on $\mathbb R^{10}$, $S = \{0, 1, 2\}$. Expected: standard Newton on
$F$ needs ~10 iters from any reasonable start; NEPIN eliminates $S$ in
1 local Newton, then global Newton on $\mathcal F$ needs ~3 iters
(elastic-bulk rate). Assert `local_iters <= 3` and check corrected $x_S$
is within `1e-4` of the analytic root.

**T2: subset identifier via threshold.**
Given `d_full = [0.1, 0.5, 0.85, 0.9, 0.3]`, `d_c = 0.82`, expect
`mask = [F, F, T, T, F]`.

**T3: local solve exact on linear residual.**
$F(x) = A x - b$ with $A$ SPD; one NEPIN local Newton should give the
exact restriction $x_S = A_{SS}^{-1}(b_S - A_{S,S^c} x_{S^c})$ within
`1e-10`.

**T4: damping guards against overshoot.**
Construct an $F_S$ with strong quadratic that overshoots at full step;
set `damping=0.5` in config; assert `converged=True` and
`local_res_reduction < 1`.

**T5: max_local_iter cap.**
Set `max_local_iter=1` on a problem needing 3; assert
`converged=False` (returned honestly, not raised).

**T6: subset_size accounting.**
Verify `NEPINResult.subset_size == mask.sum()`.

**T7: eps guards.**
Feed all-zero residual, expect `local_iters=0`,
`local_res_reduction=1.0` (nothing to do), no NaN/inf.

### 5.2 Multi-backend regression

**T8: numpy vs pytorch agreement.**
Same as `test_affine_invariant.py::T6`: skip if pytorch backend absent,
otherwise run T1 under both backends, assert corrected states agree
within `1e-10`.

### 5.3 Property test

**T9: contraction on ω̂ under NEPIN.**
Build a synthetic Newton sequence with and without NEPIN elimination on
a fixed high-contrast problem; measure ω̂ via
`AffineInvariantMonitor`; assert `omega_hat_nepin < omega_hat_plain`
by at least a factor of 2. This is the numerical version of Prop. §1.2.

### 5.4 Non-goal: no fracturex-mesh integration test in this task

Integration tests using the actual `phase_assembler` are deferred to
milestone M5 (spike script). The unit tests here are enough to certify
the module's core arithmetic.

---

## 6. Milestones (this task)

- **M1**: Module skeleton + T1–T3 green
- **M2**: T4–T7 (edge cases) green
- **M3**: T8–T9 (multi-backend + property) green
- **M4**: Update `analysis/__init__.py` exports; verify no regressions
  in the existing `test_affine_invariant.py` suite.

M5–M6 (driver integration + spike experiment) are follow-ups covered by
Theory §5 M4–M6, executed after this task closes.

---

## 6.5 M4–M6 landing (2026-07-03)

### M4 — Driver hook

`fracturex.drivers.huzhang_phasefield_staggered.HuZhangPhaseFieldStaggeredDriver`
now carries an opt-in NEPIN sweep on the damage sub-problem. Env vars
(all default to off / kernel defaults):

- `FRACTUREX_NEPIN=1`        — master switch
- `FRACTUREX_NEPIN_D_C=0.82` — $d_c$ threshold ($\Omega_s = \{d > d_c\}$)
- `FRACTUREX_NEPIN_MAX_LOCAL_ITER=5`
- `FRACTUREX_NEPIN_LOCAL_TOL=1e-2`

Wiring: between phase assemble and phase solve, `_maybe_nepin_precondition`
inspects `max(state.d)`; on the guard passing it builds the callbacks via
`fracturex.analysis.nepin_hook.build_nepin_callbacks`, runs one
`NEPINEliminator.eliminate`, writes the clipped `d_corrected` back into
`state.d`, and forces a phase re-assembly so that `sys_d.decode(dd)`
still has the correct anchor. When the flag is off, the driver path is
byte-identical to the pre-hook version (imports are deferred to the
inner branch).

New iter_row columns (all zero when the sweep skips):
`nepin_applied, nepin_subset_size, nepin_local_iters,
nepin_local_res_reduction, nepin_wall_time_s`. These are additive; the
`AffineInvariantMonitor._ALIASES` map is unchanged and reads existing
CSVs unmodified.

### M5 — Spike script

`scripts/paper_precond/nepin_spike.py` accepts a recorder root
(`--checkpoint-dir`), auto-picks the earliest step whose `max_d > d_c +
0.02` (or accepts `--step`), rebuilds the discretization via
`load_discr_from_dir`, loads state from `checkpoints/step_XXX.npz`,
assembles the phase system, and reports:
- $|S|$, local Newton iters, $\|F_S(y_\star)\|/\|F_S(0)\|$;
- $\|\Delta d\|$ from the outer LGMRES step before vs. after NEPIN;
- surrogate $\hat\omega \approx 2\|\Delta d_{\text{after}}\|/\|\Delta d_{\text{before}}\|^2$
  (single-point Deuflhard witness);
- a decision line: outer-increment reduction $\ge 5\%$ triggers the
  paper paragraph, otherwise the finding lives here only.

The script does not mutate the checkpoint; the eliminated state is
restored to `d_before` on exit.

### M6 — Paper paragraph (deferred)

Runs behind M5; the LaTeX edit at `phasefield_huzhang.tex` §
`sec:numerics_localization` (line 2646 anchor, before
`\paragraph{Restart}`) is gated on the spike output. If the contraction
is neutral, the paragraph is intentionally not written — see the "论文
诚实报告" rule in the plan
(`.claude/plans/radiant-squishing-shamir.md`).

### Test surface

`fracturex/tests/analysis/test_nepin_hook.py` adds four hook-level
regressions on synthetic scipy-sparse `A/F` (residual freezing,
sub-block extraction, end-to-end one-step convergence, CSRTensor lift).
Analysis suite: **26 passed, 2 skipped** (was 22+2 pre-Ch10-hook).

---

## 7. Open questions

- **Dense vs sparse local Jacobian.** Current design uses dense LU
  (fine for $|S| \le 10^4$). If the paper_aux h3 mesh gives $|S| \sim 5 \times 10^4$,
  we swap to sparse (`scipy.sparse.linalg.splu`) — the API is the same
  boundary shape. Decide at spike time.
- **Line search vs trust region.** Cai–Keyes 2002 uses a fixed-step
  local Newton with damping fallback; we follow. Trust-region variants
  (Liu et al. 2022) improve robustness but add complexity — deferred.
- **Interface indicator.** The `include_interface=True` branch needs
  mesh connectivity. In the module we expose the raw mask parameter;
  the caller (spike script) supplies the mesh-aware mask. This keeps
  the module mesh-free.
