"""Affine-invariant Newton diagnostics for the staggered driver.

Post-processing estimator for the outer-Newton affine-covariant Lipschitz
constant ``omega_cov`` per Gong (2018) Ch. 8 §8.2, computed from staggered
iteration increments already recorded by ``huzhang_phasefield_staggered.py``
into ``iterations.csv``.

The estimator is a **sidecar**: it never runs inside the driver's hot loop.
It reads either an in-memory sequence of ``NewtonStepRecord`` objects (online
use, e.g. from a driver hook) or the on-disk ``iterations.csv`` (offline use,
e.g. for the D12 appendix figure).

Theory
------
See ``docs/preconditioner/THEORY_affine_invariant_newton.md``:

    omega_hat_k := 2 * ||Delta d^{k+1}|| / ||Delta d^k||^2

Bounded ``omega_hat`` across staggered iterations certifies that the Newton
sequence lies inside the quadratic convergence radius r_cov = 2 / omega_cov;
growth of ``omega_hat`` toward the theoretical bound (~ C_B / k where k is
the residual stiffness) is the quantitative signature of localization.

Design
------
See ``docs/preconditioner/DESIGN_affine_invariant_diagnostics.md`` for
non-goals, integration with ``iterations.csv``, and the test plan.

Multi-backend
-------------
Compute uses ``fealpy.backend.backend_manager`` (``bm``); numpy is confined to
the CSV I/O boundary. No numpy-only ops (``np.add.at``, fancy assignment,
in-place broadcast) are used inside compute; see
``docs/architecture/multibackend_convention.md``.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np  # I/O boundary only (§3 of multi-backend convention)
from fealpy.backend import backend_manager as bm


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NewtonStepRecord:
    """One staggered Newton iteration, minimal fields for omega estimation.

    Mirrors a subset of ``iter_row`` written by the driver at line ~582 of
    ``huzhang_phasefield_staggered.py``. Extra fields there
    (``linear_solver_*``, ``wall_*``) are not needed for omega and are
    intentionally omitted so the CSV reader can accept incomplete fixtures.

    Fields
    ------
    iter : int
        Staggered iteration index within a load step, 1-based (matches driver).
    step : int
        Load step index, 0-based (matches driver).
    load : float
        Applied load at this step.
    delta_d_norm : float
        ``||d^{k+1} - d^k||_2`` on dof vectors (== ``dd_abs`` in the driver).
    delta_u_norm : float
        ``||u^{k+1} - u^k||_2`` (== ``du_abs``).
    max_d : float
        ``max d`` at end of this iter; localization diagnostic.
    lin_res_e : float
        Inner-solver relative residual on the elastic block (== ``lin_cb_res_e``).
    lin_res_d : float
        Inner-solver relative residual on the damage block (== ``lin_cb_res_d``).
    """
    iter: int
    step: int
    load: float
    delta_d_norm: float
    delta_u_norm: float
    max_d: float
    lin_res_e: float = float("nan")
    lin_res_d: float = float("nan")


@dataclass(frozen=True)
class AffineInvariantSummary:
    """Per-load-step summary of affine-invariant diagnostics.

    Fields
    ------
    step : int
        Load step index.
    load : float
        Applied load.
    max_d_end : float
        Final ``max d`` at end of the staggered loop for this step.
    n_iter : int
        Total staggered iterations at this step.
    omega_hat : Any
        ``bm`` array of shape ``(n_iter - 1,)``: pointwise
        ``omega_hat_k = 2 * ||Delta^{k+1}|| / ||Delta^k||^2`` on ``delta_d``.
        Empty if ``n_iter <= 1``.
    contraction_ratio : Any
        ``bm`` array of shape ``(n_iter - 1,)``: ``||Delta^{k+1}|| / ||Delta^k||``.
    residual_contraction : Any
        ``bm`` array of shape ``(n_iter - 1,)``: ``lin_res_d^{k+1} / lin_res_d^k``.
    within_quadratic_radius : bool
        True iff ``max omega_hat < quadratic_thresh`` (default 10.0); coarse
        heuristic for "we are in the quadratic regime".
    omega_hat_max : float
        ``max omega_hat`` (0.0 if empty).
    omega_hat_mean : float
        Mean over defined entries (0.0 if empty).
    """
    step: int
    load: float
    max_d_end: float
    n_iter: int
    omega_hat: Any
    contraction_ratio: Any
    residual_contraction: Any
    within_quadratic_radius: bool
    omega_hat_max: float
    omega_hat_mean: float


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class AffineInvariantMonitor:
    """Post-processing estimator for the outer-Newton omega_cov.

    Usage (offline, from CSV)::

        records = read_iterations_csv("iterations.csv")
        mon = AffineInvariantMonitor()
        summaries = mon.process(records)
        write_summary_csv(summaries, "affine_invariant.csv")

    Usage (online, if a driver hook wants to push per-iter)::

        mon = AffineInvariantMonitor()
        for iter_row in driver_iterations:
            mon.push(NewtonStepRecord(...))
        summary_step_5 = mon.finalize_step(5)

    Parameters
    ----------
    eps : float
        Denominator guard for the ratio (avoids IEEE inf when the increment
        goes to machine zero). Default ``1e-30`` matches the driver's
        normalization guard (``e0_u``, ``e0_d`` in the driver).
    quadratic_thresh : float
        Above this value ``omega_hat`` is deemed "outside the quadratic
        regime". Default ``10.0`` is a heuristic — Deuflhard suggests any
        finite bound is fine for the qualitative statement.
    """

    def __init__(
        self,
        *,
        eps: float = 1e-30,
        quadratic_thresh: float = 10.0,
    ):
        self.eps = float(eps)
        self.quadratic_thresh = float(quadratic_thresh)
        # Online buffer: step -> list of records
        self._buffer: Dict[int, List[NewtonStepRecord]] = {}

    # -- offline -----------------------------------------------------------

    def process(
        self, records: Sequence[NewtonStepRecord]
    ) -> List[AffineInvariantSummary]:
        """Group ``records`` by step and summarize each.

        Order within a step is preserved; caller is responsible for supplying
        records sorted by ``(step, iter)``. If steps are interleaved the
        grouping is still correct.
        """
        by_step: Dict[int, List[NewtonStepRecord]] = {}
        for r in records:
            by_step.setdefault(int(r.step), []).append(r)

        out: List[AffineInvariantSummary] = []
        for step in sorted(by_step.keys()):
            step_records = sorted(by_step[step], key=lambda r: r.iter)
            out.append(self._summarize_one_step(step, step_records))
        return out

    # -- online ------------------------------------------------------------

    def push(self, record: NewtonStepRecord) -> None:
        """Push a single Newton iter record into the online buffer.

        Caller is expected to call ``finalize_step(step)`` after the last
        iter at that step. If the same step is pushed multiple times the
        records are concatenated in call order.
        """
        self._buffer.setdefault(int(record.step), []).append(record)

    def finalize_step(self, step: int) -> AffineInvariantSummary:
        """Compute the summary for one step and evict it from the buffer.
        """
        records = self._buffer.pop(int(step), [])
        if not records:
            raise KeyError(f"no records buffered for step {step}")
        records = sorted(records, key=lambda r: r.iter)
        return self._summarize_one_step(step, records)

    # -- core --------------------------------------------------------------

    def _summarize_one_step(
        self, step: int, records: Sequence[NewtonStepRecord]
    ) -> AffineInvariantSummary:
        n_iter = len(records)
        # Extract the delta-d and delta-u sequences into bm arrays.
        # Scalars go through bm.asarray -> stacked -> length-n vector.
        delta_d = bm.asarray(
            [float(r.delta_d_norm) for r in records], dtype=bm.float64
        )
        delta_u = bm.asarray(  # noqa: F841 - reserved for future ω on u
            [float(r.delta_u_norm) for r in records], dtype=bm.float64
        )
        lin_res_d = bm.asarray(
            [float(r.lin_res_d) for r in records], dtype=bm.float64
        )
        load = float(records[-1].load)
        max_d_end = float(records[-1].max_d)

        if n_iter <= 1:
            empty = bm.asarray([], dtype=bm.float64)
            return AffineInvariantSummary(
                step=int(step),
                load=load,
                max_d_end=max_d_end,
                n_iter=n_iter,
                omega_hat=empty,
                contraction_ratio=empty,
                residual_contraction=empty,
                within_quadratic_radius=True,
                omega_hat_max=0.0,
                omega_hat_mean=0.0,
            )

        # omega_hat_k = 2 * delta_{k+1} / max(delta_k^2, eps)
        prev = delta_d[:-1]
        nxt = delta_d[1:]
        denom = bm.maximum(prev * prev, bm.asarray(self.eps, dtype=bm.float64))
        omega_hat = 2.0 * nxt / denom

        # r_k = delta_{k+1} / max(delta_k, eps)
        denom_r = bm.maximum(prev, bm.asarray(self.eps, dtype=bm.float64))
        contraction_ratio = nxt / denom_r

        # residual contraction on damage block
        prev_r = lin_res_d[:-1]
        nxt_r = lin_res_d[1:]
        denom_res = bm.maximum(prev_r, bm.asarray(self.eps, dtype=bm.float64))
        residual_contraction = nxt_r / denom_res

        omega_max = float(bm.max(omega_hat))
        omega_mean = float(bm.mean(omega_hat))
        within_q = bool(omega_max < self.quadratic_thresh)

        return AffineInvariantSummary(
            step=int(step),
            load=load,
            max_d_end=max_d_end,
            n_iter=n_iter,
            omega_hat=omega_hat,
            contraction_ratio=contraction_ratio,
            residual_contraction=residual_contraction,
            within_quadratic_radius=within_q,
            omega_hat_max=omega_max,
            omega_hat_mean=omega_mean,
        )


# ---------------------------------------------------------------------------
# I/O boundary (numpy allowed)
# ---------------------------------------------------------------------------

# CSV field aliases: the driver has written slightly different names across
# revisions; be lenient on input.
_ALIASES = {
    "iter": ("iter", "iteration"),
    "step": ("step", "load_step"),
    "load": ("load", "u_bar", "load_value"),
    "delta_d_norm": ("dd_abs", "delta_d_norm", "d_increment"),
    "delta_u_norm": ("du_abs", "delta_u_norm", "u_increment"),
    "max_d": ("max_d", "d_max"),
    "lin_res_e": ("lin_cb_res_e", "lin_res_e", "linear_residual_elastic"),
    "lin_res_d": ("lin_cb_res_d", "lin_res_d", "linear_residual_phase"),
}


def _pick(row: Dict[str, str], key: str, *, default: float = float("nan")) -> float:
    for name in _ALIASES[key]:
        if name in row and row[name] != "":
            try:
                return float(row[name])
            except (TypeError, ValueError):
                continue
    return default


def _pick_int(row: Dict[str, str], key: str, *, default: int = 0) -> int:
    for name in _ALIASES[key]:
        if name in row and row[name] != "":
            try:
                return int(float(row[name]))
            except (TypeError, ValueError):
                continue
    return default


def read_iterations_csv(path: str) -> List[NewtonStepRecord]:
    """Load ``iterations.csv`` into a list of ``NewtonStepRecord``.

    Field name aliases handle historical driver revisions. Rows lacking one
    of the required fields (``iter``, ``step``, ``dd_abs``) are skipped with
    a warning-free contract: partial CSVs are common in resumed runs.

    Multi-backend: uses ``numpy`` / ``csv`` at the I/O boundary only; returns
    plain ``NewtonStepRecord`` (Python floats/ints) so downstream compute is
    backend-agnostic.
    """
    out: List[NewtonStepRecord] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            it = _pick_int(row, "iter", default=-1)
            st = _pick_int(row, "step", default=-1)
            dd = _pick(row, "delta_d_norm")
            if it < 0 or st < 0 or not np.isfinite(dd):
                continue
            out.append(
                NewtonStepRecord(
                    iter=it,
                    step=st,
                    load=_pick(row, "load"),
                    delta_d_norm=dd,
                    delta_u_norm=_pick(row, "delta_u_norm", default=0.0),
                    max_d=_pick(row, "max_d", default=0.0),
                    lin_res_e=_pick(row, "lin_res_e"),
                    lin_res_d=_pick(row, "lin_res_d"),
                )
            )
    return out


def write_summary_csv(
    summaries: Sequence[AffineInvariantSummary], path: str
) -> None:
    """Write per-load-step summaries to CSV.

    Per-iteration detail (``omega_hat`` arrays) goes to a sibling file
    ``<path>_iterations.csv`` via ``write_iteration_detail_csv``.
    """
    fieldnames = [
        "step",
        "load",
        "max_d_end",
        "n_iter",
        "omega_hat_max",
        "omega_hat_mean",
        "within_quadratic_radius",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow(
                dict(
                    step=s.step,
                    load=f"{s.load:.12g}",
                    max_d_end=f"{s.max_d_end:.12g}",
                    n_iter=s.n_iter,
                    omega_hat_max=f"{s.omega_hat_max:.12g}",
                    omega_hat_mean=f"{s.omega_hat_mean:.12g}",
                    within_quadratic_radius=int(s.within_quadratic_radius),
                )
            )


def write_iteration_detail_csv(
    summaries: Sequence[AffineInvariantSummary], path: str
) -> None:
    """Write per-iteration ``omega_hat``, ratios to CSV for plotting."""
    fieldnames = [
        "step",
        "load",
        "iter_k",  # index into omega_hat (k = 0 corresponds to iter 2/1 pair)
        "omega_hat",
        "contraction_ratio",
        "residual_contraction",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            omega_np = bm.to_numpy(s.omega_hat).astype(np.float64)
            cr_np = bm.to_numpy(s.contraction_ratio).astype(np.float64)
            rc_np = bm.to_numpy(s.residual_contraction).astype(np.float64)
            for k in range(omega_np.size):
                writer.writerow(
                    dict(
                        step=s.step,
                        load=f"{s.load:.12g}",
                        iter_k=k,
                        omega_hat=f"{omega_np[k]:.12g}",
                        contraction_ratio=f"{cr_np[k]:.12g}",
                        residual_contraction=f"{rc_np[k]:.12g}",
                    )
                )


__all__ = [
    "NewtonStepRecord",
    "AffineInvariantSummary",
    "AffineInvariantMonitor",
    "read_iterations_csv",
    "write_summary_csv",
    "write_iteration_detail_csv",
]
