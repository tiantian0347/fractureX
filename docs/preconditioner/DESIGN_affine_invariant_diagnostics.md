# Design: Affine-Invariant Newton Diagnostics for the Staggered Driver

> Status: v0.1 (2026-07-02). Design-only; theory in
> [`THEORY_affine_invariant_newton.md`](THEORY_affine_invariant_newton.md);
> code in `fracturex/analysis/affine_invariant.py` (to be created);
> tests in `fracturex/tests/analysis/test_affine_invariant.py`.
>
> Related: [`GONG_THESIS_ABSORPTION.md`](../planning/GONG_THESIS_ABSORPTION.md) §1,
> multibackend convention [`multibackend_convention.md`](../architecture/multibackend_convention.md).

---

## 0. Design goals

1. **Zero touch to the staggered driver.** The current driver
   `fracturex/drivers/huzhang_phasefield_staggered.py` (2000+ LOC) is stable and
   well-tested; we must not add computation inside its hot loop. The ω estimator
   is a **post-processing / online sidecar**.
2. **Consume existing artifacts.** Every increment we need
   ($\|\Delta d^k\|$, $\|\Delta u^k\|$, $\max d$, `iter`) is already written to
   `iterations.csv` (line 581–592 of the driver). We add a reader/estimator, not
   new instrumentation.
3. **Multi-backend from line 1.** Following
   [`multibackend_convention.md`](../architecture/multibackend_convention.md):
   `bm` for compute, `numpy` only at scipy/IO boundaries. This module has **no**
   scipy dependency, so it's pure `bm`.
4. **Testable in isolation.** Deterministic MMS Newton sequence → known ω →
   estimator recovers it within tolerance. No pipeline data needed to run the
   tests.
5. **Fits naturally into the D12 paper appendix.** Output includes a
   plot-ready summary table (ω̂ vs iter, ω̂ vs max_d) that maps 1:1 onto the
   figure planned in
   [`THEORY_affine_invariant_newton.md`](THEORY_affine_invariant_newton.md) §3.

---

## 1. What we're measuring

### 1.1 Estimator formula (from Theory §2)

$$\hat\omega_k := 2 \cdot \frac{\|\Delta x^{k+1}\|}{\|\Delta x^k\|^2}, \qquad
\Delta x^k \text{ = staggered Newton correction at iter } k.$$

### 1.2 Which increment is $\Delta x^k$?

For the fracturex staggered outer Newton, $x = d$ (damage) is the natural
outer variable — the mechanical block is linear in $(\sigma, u)$ for frozen
$d$, and the staggered fixed point is a map $d \mapsto \mathcal G(d)$. So

$$\Delta d^k := d^{k+1} - d^k \equiv d_{\text{plain}} - d_{\text{old}}
                \quad \text{(exactly the `dd_abs` in the driver, line 563).}$$

**Norm choice**: we use the same $\ell^2$-norm on dof vectors as the driver
(`bm.linalg.norm`), which corresponds to a mass-lumped $L^2(\Omega)$ norm on
the P1 damage space. Weighted variants ($H^1$-norm, $L^\infty$-norm) are
implemented as options for sensitivity analysis (Theorem B has a $H^1$-flavored
$\|\varepsilon(u)\|_{L^\infty}$ factor).

### 1.3 Auxiliary diagnostics

Beyond $\hat\omega_k$, the module produces:

- **Contraction ratio** $r_k := \|\Delta d^{k+1}\| / \|\Delta d^k\|$ — should
  decrease geometrically if we're in the linear convergence regime, and
  quadratically if in the Newton regime.
- **Residual contraction** $r_k^R := \|\Delta d^{k+1}\| / \|\Delta d^k\|$ but
  measured on the linear solver residuals (`lin_cb_res_e`, `lin_cb_res_d` in
  iter_row) — Deuflhard's affine contravariant flavor (Thm B in Theory).
- **Max damage curve** $\max d$ vs iter — the localization signature.

---

## 2. Module layout

```
fracturex/
  analysis/               # NEW subpackage
    __init__.py
    affine_invariant.py   # Core estimator + I/O
  tests/
    analysis/             # NEW subpackage
      __init__.py
      test_affine_invariant.py
docs/preconditioner/
  THEORY_affine_invariant_newton.md   # Already written (Task 1)
  DESIGN_affine_invariant_diagnostics.md   # This file
```

### 2.1 Why a new `analysis/` subpackage

Existing candidates:
- `fracturex/postprocess/` — runtime step-level postprocess (`RunRecorder`);
  our estimator consumes its output, so it's downstream, not sibling.
- `fracturex/ml/` — reserved for feature extraction; irrelevant.
- `fracturex/adaptivity/` — a-posteriori error estimators for spatial
  adaptivity; different concept (a-posteriori Newton convergence vs a-posteriori
  discretization error).

`analysis/` cleanly separates **algorithmic diagnostics** from runtime
computation and adaptivity. This is where Ch 10 NEPIN infrastructure will also
land eventually (as `analysis/nonlinear_elimination.py`).

### 2.2 Public API sketch

```python
# fracturex/analysis/affine_invariant.py
from fealpy.backend import backend_manager as bm
from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class NewtonStepRecord:
    """A single staggered Newton iteration, as post-processed for ω estimation.

    Fields mirror the subset of `iter_row` (driver line 582) that the estimator
    needs. Keep it minimal: the estimator must be reproducible from CSV.
    """
    iter: int
    load: float
    step: int
    delta_d_norm: float   # = dd_abs in the driver
    delta_u_norm: float   # = du_abs
    max_d: float
    lin_res_e: float      # inner GMRES residual for elastic block
    lin_res_d: float      # inner solver residual for damage block


@dataclass(frozen=True)
class AffineInvariantSummary:
    """Per-load-step summary of affine-invariant diagnostics."""
    step: int
    load: float
    max_d_end: float
    omega_hat: Any        # bm array, shape (n_iter - 1,)
    contraction_ratio: Any  # r_k, shape (n_iter - 1,)
    residual_contraction: Any  # r_k^R, shape (n_iter - 1,)
    n_iter: int
    within_quadratic_radius: bool   # heuristic: max(omega_hat) < THRESHOLD


class AffineInvariantMonitor:
    """Post-processing estimator for the outer Newton ω_cov.

    Consumes a stream of `NewtonStepRecord` (either in-memory or read from
    `iterations.csv`) and produces `AffineInvariantSummary` per load step.

    Usage (offline):
        records = read_iterations_csv(path)      # helper below
        mon = AffineInvariantMonitor()
        summaries = mon.process(records)
        mon.write_summary_csv(summaries, "affine_invariant.csv")

    Usage (online, if driver ever wants it):
        mon = AffineInvariantMonitor()
        for iter_row in driver_iterations:
            mon.push(NewtonStepRecord(**iter_row))
        summary = mon.finalize_step(step_index)
    """

    def __init__(self, *, eps: float = 1e-30, quadratic_thresh: float = 10.0):
        self.eps = float(eps)
        self.quadratic_thresh = float(quadratic_thresh)

    def process(self, records: Sequence[NewtonStepRecord]) -> List[AffineInvariantSummary]:
        """Group records by step and compute summaries.  Multi-backend, pure bm.
        """
        ...


def read_iterations_csv(path: str) -> List[NewtonStepRecord]:
    """Reader boundary: reads CSV via numpy/pandas (IO boundary per §3 of the
    multi-backend convention), returns pure-Python records for the estimator.
    """
    ...


def write_summary_csv(summaries: Sequence[AffineInvariantSummary], path: str) -> None:
    """Writer boundary: numpy for CSV write.  Compute inside stays bm."""
    ...
```

### 2.3 Multi-backend contract

- Public API accepts `bm` arrays via `NewtonStepRecord` (dataclass fields are
  floats — inherently backend-neutral for scalars).
- Internal compute (norms, ratios, index-add) uses `bm.*`.
- Only `read_iterations_csv` and `write_summary_csv` touch numpy, at the
  boundary (§3 of the convention).

Confirmed backend-friendly ops used:
- `bm.linalg.norm` ✓
- `bm.asarray`, `bm.stack` ✓
- `bm.where` (for eps-guarded division) ✓
- `bm.log10` (for growth-rate diagnostics) ✓
- No `np.add.at`, no fancy assignment, no `.astype` on non-numpy tensors.

---

## 3. Integration with existing artifacts

### 3.1 `iterations.csv` schema (already produced by driver)

Fields present today (verified against `huzhang_phasefield_staggered.py` L582–L620):

| Field | Type | Notes |
|---|---|---|
| `step` | int | load step index |
| `load` | float | current load value |
| `iter` | int | staggered iteration index (1-based) |
| `du_abs`, `dd_abs` | float | our $\|\Delta u^k\|$, $\|\Delta d^k\|$ ✓ |
| `err_u`, `err_d` | float | normalized to first-iter increment |
| `max_d` | float | current $\max d$ ✓ |
| `linear_solver_elastic/phase` | str | solver name |
| `linear_niter_elastic/phase` | int | inner GMRES iter count |
| `lin_cb_res_e/d` | float | inner solver relative residual ✓ |
| `gdof_sigma/u/d` | int | dof counts |
| `wall_*` | float | timings |

**Conclusion**: all required fields are already there. The estimator is purely
downstream of the CSV.

### 3.2 Output artifact

We produce `affine_invariant.csv` per run, alongside `iterations.csv`, with:

| Field | Notes |
|---|---|
| `step`, `load` | matches run schema |
| `max_d_end` | ω-vs-max_d plot axis |
| `n_iter` | staggered iters at this step |
| `omega_hat_max` | $\max_k \hat\omega_k$ at this step |
| `omega_hat_mean` | mean $\hat\omega_k$ |
| `contraction_max`, `contraction_mean` | $r_k$ statistics |
| `within_quadratic_radius` | boolean, `omega_hat_max < quadratic_thresh` |

Per-iteration detail is written to `affine_invariant_iterations.csv` for the
detailed plots.

### 3.3 D12 paper artifact wiring

The `analysis/affine_invariant.py` module is called offline from a script
`scripts/paper_precond/build_affine_invariant.py` (to be added) that:

1. Scans `results/phasefield/*/paper_aux/epsg_*/`
2. For each run, loads `iterations.csv`, runs `AffineInvariantMonitor.process`
3. Writes `affine_invariant.csv` and `affine_invariant_iterations.csv`
4. Emits a summary plot `affine_invariant_omega_vs_maxd.pdf` for §D12 appendix

This is exactly analogous to how `postprocess/recorder.py` is consumed by
downstream figure scripts.

---

## 4. Numerical care

### 4.1 Division-by-zero guard

$\hat\omega_k = 2 \|\Delta^{k+1}\| / \|\Delta^k\|^2$ blows up when
$\|\Delta^k\| \to 0$ (converged step). Use

```python
denom = bm.maximum(delta_norms[:-1] ** 2, self.eps)
omega_hat = 2.0 * delta_norms[1:] / denom
```

with `self.eps = 1e-30` by default. Setting `eps` too large (say $10^{-6}$)
would clip legitimate values in near-converged runs; setting it too small
risks IEEE inf which then poisons downstream stats. `1e-30` is the same value
`main_solve.py` uses for its `e0_u`, `e0_d` normalization (driver L571, L574).

### 4.2 Single-iteration runs

If a load step converges in 1 staggered iter (below threshold), there is no
$\hat\omega_k$ to estimate (needs at least 2 iters). Report `n_iter=1`,
`omega_hat=[]`, `within_quadratic_radius=True` (trivially).

### 4.3 Anderson-accelerated runs

Driver has optional Anderson acceleration (`self._anderson`, L544–550). When
Anderson is on, the reported `dd_abs` is $\|\Delta d_{\text{plain}}\|$
(un-accelerated projected image) — see driver L563 comment:
> True staggered fixed-point residual on d uses the UN-accelerated
> projected image d_plain, so convergence reflects ||G(d)-d|| rather
> than the (possibly small) accelerated increment.

This is **exactly** what we want for ω_cov measurement: the un-accelerated
increment is the correct Newton correction. So the driver's design happens to
be already ω-monitor friendly. Document this in the estimator docstring.

### 4.4 Near-linear regime (Thm A)

When the system is in the linear regime ($d \equiv 0$ or small), Thm A predicts
$\omega_{\mathrm{cov}} \approx 0$. Numerically the estimator will produce values
near machine epsilon divided by round-off in the norm; clip and report `<eps`
in the summary. This is documented behavior, not a bug.

---

## 5. Test plan

`fracturex/tests/analysis/test_affine_invariant.py` covers:

### 5.1 Unit tests (deterministic, seconds)

1. **T1: linear sequence** — Feed a Newton sequence with known
   $\|\Delta^k\| = c \cdot 2^{-k}$ (geometric contraction). Expected
   $\hat\omega_k = c \cdot 2^{-(k+1)} / (c^2 \cdot 4^{-k}) = 2/(c \cdot 2^{k-1}) \to 0$.
   Estimator should reproduce within 1e-12.
2. **T2: quadratic sequence** — Feed $\|\Delta^k\| = c^{2^k}$ with $c < 1$.
   Expected $\hat\omega_k \to 2/c \cdot c^{2^{k+1}-2 \cdot 2^k} = 2/c$ i.e. constant.
   Estimator should reproduce the constant.
3. **T3: division-by-zero guard** — Feed a step that ends with $\|\Delta^k\| = 0$;
   estimator returns `n_iter-1` finite values plus one guarded value, no NaN/inf.
4. **T4: single-iter step** — Feed a step with only 1 iter; empty `omega_hat`,
   `within_quadratic_radius=True`.
5. **T5: multi-step grouping** — Feed 3 concatenated steps of varying length;
   `process()` returns 3 `AffineInvariantSummary` with correct step tags.

### 5.2 Multi-backend tests

6. **T6: backend switching** — Run T2 under `numpy` and `pytorch` backends
   (skip pytorch if fealpy pytorch backend not installed), verify outputs
   agree within 1e-10 relative.

### 5.3 Integration test with realistic CSV

7. **T7: CSV round-trip** — Take a small synthetic `iterations.csv` (built by
   the test), read via `read_iterations_csv`, run monitor, write summary CSV,
   read it back, verify fields.
8. **T8: pipeline CSV compatibility** — Take a real `iterations.csv` from
   `results/` (mocked to a fixture file in `tests/analysis/data/`), verify the
   estimator runs end-to-end without error and produces plausible ranges
   (0 ≤ omega_hat < 1000).

### 5.4 Property tests (nice-to-have)

9. **T9: affine covariance** — Feed a sequence $x^k$ and its rescaling
   $A x^k$ for random invertible $A$; ω̂ should agree (this is the definition
   of affine covariance). Skip if we don't have $A$-transformed norms.

---

## 6. Non-goals (explicitly out of scope)

- ❌ **Modifying the staggered driver**. If the driver later wants to push
  online ω to the recorder, that's a *separate* patch, not this task.
- ❌ **Estimating $\omega_{\mathrm{cot}}$ or $\omega_{\mathrm{cnj}}$** in this
  first pass. Thm B requires access to the Jacobian action, which is not in
  the CSV; only $\omega_{\mathrm{cov}}$ is CSV-recoverable. The other two are
  future work when we plumb Jacobian introspection.
- ❌ **NEPIN (Ch 10)**. That's a separate module (`analysis/nonlinear_elimination.py`)
  and a separate spike experiment. Ch 8 lands the infrastructure; Ch 10 uses it.
- ❌ **GPU perf tuning**. bm makes this run on any backend; the workload is
  post-processing at CSV scale (thousands of iters), so GPU is overkill and
  unnecessary.

---

## 7. Milestones

- **M1 (this task)**: Module skeleton + unit tests T1–T5 green
- **M2**: Multi-backend test T6 + CSV round-trip T7
- **M3**: Pipeline CSV test T8 with a fixture harvested from `results/`
- **M4**: `scripts/paper_precond/build_affine_invariant.py` producing
  the D12 appendix figure
- **M5 (future / T7 paper)**: NEPIN spike consuming this module's ω̂ trace

Only M1–M3 are in the current task list. M4 is a follow-up (script only,
no new code in the package). M5 is Ch 10 / T7 territory.

---

## 8. Open questions

- Should the estimator also compute a **CI / confidence band** for
  $\hat\omega_k$ using bootstrap over iterations? Deuflhard doesn't; Gong
  doesn't either. We skip for now; add if reviewers ask.
- Should we support **online** mode (driver pushes iter rows into the monitor
  during the run)? Not needed for the D12 paper; keep as opt-in for future
  users. The API sketch above already allows it via `push()` + `finalize_step()`
  but we don't wire it into the driver in this task.
