"""Reduced nonlinear elimination with an assembled local Jacobian.

This module solves a coupled nonlinear residual ``F(x)=0`` after eliminating a
selected coordinate patch.  The local Newton step uses a caller-supplied
assembled ``J_ww``; the outer Newton--Krylov step applies the Schur complement
matrix-free and never forms a dense global Jacobian.

The implementation supports box constraints through the natural projected
residual.  It is a reusable numerical kernel: finite-element assembly, patch
selection, and experiment I/O remain in caller modules and scripts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Optional

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, gmres


ResidualCallback = Callable[[np.ndarray], np.ndarray]
LocalJacobianCallback = Callable[[np.ndarray, np.ndarray], object]
JacobianVectorCallback = Callable[[np.ndarray, np.ndarray], np.ndarray]
ReducedPreconditionerBuilder = Callable[
    [np.ndarray, np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]
]
StateDiagnosticCallback = Callable[
    [int, np.ndarray, np.ndarray, np.ndarray], dict[str, object]
]


@dataclass(frozen=True)
class ReducedNewtonConfig:
    """Numerical controls for reduced nonlinear Newton--Krylov.

    Parameters are dimensionless.  Local and outer convergence use
    ``atol + rtol*max(1, initial_norm)``.  ``fd_step`` is used only when no
    Jacobian-vector callback is supplied.  ``minimum_outer_iterations`` can
    force accepted outer corrections before convergence is reported; this
    supports reference-free a posteriori state checks.
    """

    local_atol: float = 1.0e-10
    local_rtol: float = 1.0e-8
    outer_atol: float = 1.0e-10
    outer_rtol: float = 1.0e-8
    max_local_iterations: int = 12
    max_outer_iterations: int = 20
    krylov_rtol: float = 1.0e-6
    krylov_atol: float = 0.0
    krylov_max_iterations: int = 200
    fd_step: float = 1.0e-7
    finite_difference_scheme: str = "forward"
    minimum_step_length: float = 1.0e-6
    armijo_slope: float = 1.0e-4
    minimum_outer_iterations: int = 0
    use_local_predictor: bool = False


@dataclass(frozen=True)
class ReducedNewtonResult:
    """Result and work counters for one reduced nonlinear solve.

    ``state`` is a new ``float64`` vector in the caller's full-state ordering.
    Residual counts include line-search and matrix-free Jv evaluations.
    ``local_linear_solves`` counts both nonlinear elimination solves and the
    ``J_ww`` solves used inside Schur-complement products. The corresponding
    wall time includes local factorizations and triangular solves.
    local_predictor_applications counts implicit local-map predictor solves;
    these solves are included in local_linear_solves.
    """

    state: np.ndarray
    converged: bool
    termination_reason: str
    outer_iterations: int
    local_newton_iterations: int
    local_linear_solves: int
    local_linear_solve_wall_time_seconds: float
    krylov_iterations: int
    preconditioner_applications: int
    jvp_evaluations: int
    residual_evaluations: int
    krylov_residual_norms: np.ndarray
    projected_residual_norms: np.ndarray
    local_projected_residual_norm: float
    state_diagnostic_history: tuple[dict[str, object], ...]
    state_diagnostic_wall_time_seconds: float
    wall_time_seconds: float
    local_predictor_applications: int = 0
    schur_direction_residual_norms: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    outer_step_lengths: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    outer_backtracking_counts: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )


def _as_vector(value: object, *, name: str) -> np.ndarray:
    """Return a finite copied ``float64`` vector or raise ``ValueError``."""
    vector = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return vector


def _dense_local_matrix(matrix: object, size: int) -> np.ndarray:
    """Convert an assembled local matrix to finite dense ``float64`` form."""
    dense = matrix.toarray() if issparse(matrix) else np.asarray(matrix)
    dense = np.asarray(dense, dtype=np.float64)
    if dense.shape != (size, size) or not np.isfinite(dense).all():
        raise ValueError(
            f"local_jacobian must return a finite ({size}, {size}) matrix"
        )
    return dense.copy()


def _projected_residual(
    state: np.ndarray,
    residual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    fixed_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return natural box residual and the projection-interior row mask."""
    argument = state - residual
    projected = state - np.minimum(np.maximum(argument, lower), upper)
    projected[fixed_mask] = 0.0
    interior = (argument > lower) & (argument < upper) & ~fixed_mask
    interior[np.isneginf(lower) & np.isposinf(upper) & ~fixed_mask] = True
    return projected, interior


def _projected_local_jacobian(
    physical_local_jacobian: np.ndarray,
    local_interior: np.ndarray,
) -> np.ndarray:
    """Replace active projected-residual rows by coordinate identity rows."""
    matrix = physical_local_jacobian.copy()
    active_rows = np.flatnonzero(~local_interior)
    if active_rows.size:
        matrix[active_rows, :] = 0.0
        matrix[active_rows, active_rows] = 1.0
    return matrix


def solve_reduced_nonlinear_system(
    residual: ResidualCallback,
    local_jacobian: LocalJacobianCallback,
    initial_state: object,
    patch_dofs: object,
    *,
    lower_bound: Optional[object] = None,
    upper_bound: Optional[object] = None,
    fixed_mask: Optional[object] = None,
    jacobian_vector_product: Optional[JacobianVectorCallback] = None,
    reduced_preconditioner: Optional[ReducedPreconditionerBuilder] = None,
    state_diagnostic_callback: Optional[StateDiagnosticCallback] = None,
    config: Optional[ReducedNewtonConfig] = None,
) -> ReducedNewtonResult:
    """Solve a box-constrained residual by nonlinear patch elimination.

    Parameters
    ----------
    residual : callable
        Physical residual ``F(x)``. Input and output have shape ``(n_state,)``.
        The callback may update an external assembler.
    local_jacobian : callable
        ``local_jacobian(x, patch_dofs) -> J_ww``. It is called immediately
        after ``residual(x)`` and may reuse matrices captured by that assembly.
        The returned dense or sparse matrix has shape ``(n_patch, n_patch)``.
    initial_state : array-like, shape (n_state,)
        Finite starting state. It is not modified.
    patch_dofs : array-like, shape (n_patch,), dtype integer
        Unique non-Dirichlet coordinates eliminated by the inner solve.
    lower_bound, upper_bound : array-like, shape (n_state,), optional
        Feasible box. Infinite endpoints are accepted.
    fixed_mask : array-like, shape (n_state,), dtype bool, optional
        Coordinates excluded from both local and reduced unknown spaces.
    jacobian_vector_product : callable, optional
        Physical ``J(x) v`` action. If omitted, a forward or centered
        directional difference of ``residual`` is used according to
        ``config.finite_difference_scheme``. The infinitesimal perturbation may
        leave the feasible box because the derivative is taken for the smooth
        physical residual before projection.
    reduced_preconditioner : callable, optional
        Builder called once per outer Newton step as
        ``builder(x, outside_dofs, outside_interior)``. It returns a linear
        action approximating the inverse reduced Jacobian. This permits reuse
        of assembled block factors without exposing them to the generic
        kernel.
    state_diagnostic_callback : callable, optional
        Optional audit callback invoked at the initial locally eliminated state
        and after every accepted outer state. It receives
        ``(outer_iteration, state, projected_residual, interior_mask)`` and
        returns a diagnostic mapping. The callback is observational only; its
        work is excluded from solver counters.
    config : ReducedNewtonConfig, optional
        Local Newton, outer Newton, Krylov, globalization, optional
        minimum-iteration controls, and the local predictor switch.

    Returns
    -------
    ReducedNewtonResult
        Full state, convergence status, residual history, and work counters.

    Notes
    -----
    At every accepted outer state, the inner iteration first enforces the
    projected local equation.  The outer linear operator applies

    ``J_cc v - J_cw J_ww^{-1} J_wc v``

    using two physical Jv products and one assembled local solve.  Hence a
    converged result satisfies the original projected residual, not a modified
    post-correction fixed-point equation.
    When use_local_predictor is enabled, each outer Newton step initializes
    the local trial with delta_z_w = -J_ww^{-1} J_wc delta_z_c before local
    nonlinear elimination. The exact local solve and all acceptance checks
    remain unchanged.
    """
    cfg = config or ReducedNewtonConfig()
    scalar_controls = (
        cfg.local_atol,
        cfg.local_rtol,
        cfg.outer_atol,
        cfg.outer_rtol,
        cfg.krylov_rtol,
        cfg.krylov_atol,
        cfg.fd_step,
        cfg.minimum_step_length,
        cfg.armijo_slope,
    )
    if not np.isfinite(scalar_controls).all() or any(
        value < 0.0 for value in scalar_controls
    ):
        raise ValueError(
            "solver tolerances and step controls must be finite and nonnegative"
        )
    if cfg.fd_step <= 0.0 or not 0.0 < cfg.minimum_step_length <= 1.0:
        raise ValueError("fd_step must be positive and minimum_step_length in (0, 1]")
    if not 0.0 < cfg.armijo_slope < 1.0:
        raise ValueError("armijo_slope must lie in (0, 1)")
    if cfg.finite_difference_scheme not in {"forward", "centered"}:
        raise ValueError(
            "finite_difference_scheme must be 'forward' or 'centered'"
        )
    if cfg.minimum_outer_iterations < 0:
        raise ValueError("minimum_outer_iterations must be nonnegative")
    if min(
        cfg.max_local_iterations,
        cfg.max_outer_iterations,
        cfg.krylov_max_iterations,
    ) <= 0:
        raise ValueError("iteration limits must be positive")

    state = _as_vector(initial_state, name="initial_state")
    n_state = state.size
    patch = np.asarray(patch_dofs, dtype=np.int64).reshape(-1).copy()
    if patch.size == 0 or np.unique(patch).size != patch.size:
        raise ValueError("patch_dofs must contain unique coordinates")
    if np.any(patch < 0) or np.any(patch >= n_state):
        raise ValueError("patch_dofs contains an out-of-range coordinate")
    lower = (
        np.full(n_state, -np.inf, dtype=np.float64)
        if lower_bound is None
        else np.asarray(lower_bound, dtype=np.float64).reshape(-1).copy()
    )
    upper = (
        np.full(n_state, np.inf, dtype=np.float64)
        if upper_bound is None
        else np.asarray(upper_bound, dtype=np.float64).reshape(-1).copy()
    )
    fixed = (
        np.zeros(n_state, dtype=bool)
        if fixed_mask is None
        else np.asarray(fixed_mask, dtype=bool).reshape(-1).copy()
    )
    if lower.shape != state.shape or upper.shape != state.shape or fixed.shape != state.shape:
        raise ValueError("bounds and fixed_mask must match initial_state")
    if np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper):
        raise ValueError("bounds must be ordered and must not contain NaN")
    if np.any(state < lower) or np.any(state > upper):
        raise ValueError("initial_state lies outside the feasible box")
    if np.any(fixed[patch]):
        raise ValueError("patch_dofs must not contain fixed coordinates")

    free_mask = ~fixed
    patch_mask = np.zeros(n_state, dtype=bool)
    patch_mask[patch] = True
    outside_pool = np.flatnonzero(free_mask & ~patch_mask)
    start_time = perf_counter()
    residual_evaluations = 0
    jvp_evaluations = 0
    local_newton_iterations = 0
    local_linear_solves = 0
    local_linear_solve_wall_time_seconds = 0.0
    local_predictor_applications = 0
    krylov_iterations = 0
    preconditioner_applications = 0
    krylov_residual_norms: list[float] = []
    schur_direction_residual_norms: list[float] = []
    outer_step_lengths: list[float] = []
    outer_backtracking_counts: list[int] = []
    state_diagnostic_history: list[dict[str, object]] = []
    state_diagnostic_wall_time_seconds = 0.0

    def record_state_diagnostic(
        outer_iteration: int,
        candidate: np.ndarray,
        candidate_projected: np.ndarray,
        candidate_interior: np.ndarray,
    ) -> None:
        """Record one accepted-state audit row without affecting convergence."""
        if state_diagnostic_callback is None:
            return
        diagnostic_start = perf_counter()
        diagnostic = state_diagnostic_callback(
            int(outer_iteration),
            candidate.copy(),
            candidate_projected.copy(),
            candidate_interior.copy(),
        )
        nonlocal state_diagnostic_wall_time_seconds
        state_diagnostic_wall_time_seconds += perf_counter() - diagnostic_start
        if not isinstance(diagnostic, dict):
            raise ValueError("state_diagnostic_callback must return a dict")
        state_diagnostic_history.append(dict(diagnostic))

    def evaluate(candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal residual_evaluations
        residual_evaluations += 1
        physical = np.asarray(residual(candidate.copy()), dtype=np.float64).reshape(-1)
        if physical.shape != candidate.shape or not np.isfinite(physical).all():
            raise ValueError("residual must return a finite vector matching the state")
        projected, interior = _projected_residual(
            candidate, physical, lower, upper, fixed
        )
        return physical, projected, interior

    def eliminate_local(
        candidate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        nonlocal local_newton_iterations, local_linear_solves
        nonlocal local_linear_solve_wall_time_seconds
        current = candidate.copy()
        physical, projected, interior = evaluate(current)
        initial_norm = float(np.linalg.norm(projected[patch]))
        tolerance = cfg.local_atol + cfg.local_rtol * max(1.0, initial_norm)
        if initial_norm <= tolerance:
            return current, physical, projected, interior, True

        for _ in range(cfg.max_local_iterations):
            local_newton_iterations += 1
            assembled = _dense_local_matrix(
                local_jacobian(current.copy(), patch.copy()), patch.size
            )
            projected_jacobian = _projected_local_jacobian(
                assembled, interior[patch]
            )
            try:
                local_solve_start = perf_counter()
                step = np.linalg.solve(projected_jacobian, -projected[patch])
            except np.linalg.LinAlgError as exc:
                raise RuntimeError("assembled local Jacobian is singular") from exc
            local_linear_solve_wall_time_seconds += (
                perf_counter() - local_solve_start
            )
            local_linear_solves += 1

            base_norm = float(np.linalg.norm(projected[patch]))
            step_length = 1.0
            accepted = False
            while step_length >= cfg.minimum_step_length:
                trial = current.copy()
                trial[patch] = np.minimum(
                    np.maximum(current[patch] + step_length * step, lower[patch]),
                    upper[patch],
                )
                trial_physical, trial_projected, trial_interior = evaluate(trial)
                trial_norm = float(np.linalg.norm(trial_projected[patch]))
                if trial_norm <= (1.0 - cfg.armijo_slope * step_length) * base_norm:
                    current = trial
                    physical = trial_physical
                    projected = trial_projected
                    interior = trial_interior
                    accepted = True
                    break
                step_length *= 0.5
            if not accepted:
                return current, physical, projected, interior, False
            if float(np.linalg.norm(projected[patch])) <= tolerance:
                return current, physical, projected, interior, True
        return current, physical, projected, interior, False

    state, physical, projected, interior, local_converged = eliminate_local(state)
    initial_full_norm = float(np.linalg.norm(projected[free_mask]))
    outer_tolerance = cfg.outer_atol + cfg.outer_rtol * max(1.0, initial_full_norm)
    norm_history = [initial_full_norm]
    record_state_diagnostic(0, state, projected, interior)
    if not local_converged:
        return ReducedNewtonResult(
            state=state,
            converged=False,
            termination_reason="initial local elimination did not converge",
            outer_iterations=0,
            local_newton_iterations=local_newton_iterations,
            local_linear_solves=local_linear_solves,
            local_predictor_applications=local_predictor_applications,
            local_linear_solve_wall_time_seconds=(
                local_linear_solve_wall_time_seconds
            ),
            krylov_iterations=0,
            preconditioner_applications=0,
            jvp_evaluations=0,
            residual_evaluations=residual_evaluations,
            krylov_residual_norms=np.asarray(krylov_residual_norms),
            projected_residual_norms=np.asarray(norm_history),
            local_projected_residual_norm=float(np.linalg.norm(projected[patch])),
            state_diagnostic_history=tuple(state_diagnostic_history),
            state_diagnostic_wall_time_seconds=state_diagnostic_wall_time_seconds,
            wall_time_seconds=(
                perf_counter() - start_time - state_diagnostic_wall_time_seconds
            ),
        )
    if outside_pool.size == 0 or (
        initial_full_norm <= outer_tolerance
        and cfg.minimum_outer_iterations == 0
    ):
        return ReducedNewtonResult(
            state=state,
            converged=initial_full_norm <= outer_tolerance,
            termination_reason=(
                "projected residual converged"
                if initial_full_norm <= outer_tolerance
                else "no reduced coordinates remain"
            ),
            outer_iterations=0,
            local_newton_iterations=local_newton_iterations,
            local_linear_solves=local_linear_solves,
            local_predictor_applications=local_predictor_applications,
            local_linear_solve_wall_time_seconds=(
                local_linear_solve_wall_time_seconds
            ),
            krylov_iterations=0,
            preconditioner_applications=0,
            jvp_evaluations=0,
            residual_evaluations=residual_evaluations,
            krylov_residual_norms=np.asarray(krylov_residual_norms),
            projected_residual_norms=np.asarray(norm_history),
            local_projected_residual_norm=float(np.linalg.norm(projected[patch])),
            state_diagnostic_history=tuple(state_diagnostic_history),
            state_diagnostic_wall_time_seconds=state_diagnostic_wall_time_seconds,
            wall_time_seconds=(
                perf_counter() - start_time - state_diagnostic_wall_time_seconds
            ),
        )

    termination_reason = "maximum outer iterations reached"
    completed_outer_iterations = 0
    for outer_iteration in range(1, cfg.max_outer_iterations + 1):
        completed_outer_iterations = outer_iteration
        outside = outside_pool[
            interior[outside_pool]
            | (np.abs(projected[outside_pool]) > outer_tolerance)
        ]
        if outside.size == 0:
            termination_reason = "no unconverged reduced coordinates remain"
            break
        assembled = _dense_local_matrix(
            local_jacobian(state.copy(), patch.copy()), patch.size
        )
        projected_local_jacobian = _projected_local_jacobian(
            assembled, interior[patch]
        )
        try:
            local_factor_start = perf_counter()
            projected_local_factor = lu_factor(projected_local_jacobian)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise RuntimeError("assembled local Jacobian factorization failed") from exc
        local_linear_solve_wall_time_seconds += (
            perf_counter() - local_factor_start
        )
        base_physical = physical.copy()

        def apply_physical_jvp(direction: np.ndarray) -> np.ndarray:
            nonlocal jvp_evaluations
            jvp_evaluations += 1
            full_direction = np.asarray(direction, dtype=np.float64).reshape(-1)
            if jacobian_vector_product is not None:
                product = np.asarray(
                    jacobian_vector_product(state.copy(), full_direction.copy()),
                    dtype=np.float64,
                ).reshape(-1)
                if product.shape != state.shape or not np.isfinite(product).all():
                    raise ValueError("jacobian_vector_product returned an invalid vector")
                return product

            direction_norm = float(np.linalg.norm(full_direction))
            if direction_norm <= np.finfo(np.float64).tiny:
                return np.zeros_like(state)
            nominal = cfg.fd_step * max(1.0, float(np.linalg.norm(state))) / direction_norm

            step = nominal
            positive_physical, _, _ = evaluate(state + step * full_direction)
            if cfg.finite_difference_scheme == "forward":
                return (positive_physical - base_physical) / step
            negative_physical, _, _ = evaluate(state - step * full_direction)
            return (positive_physical - negative_physical) / (2.0 * step)

        local_interior = interior[patch]
        outside_interior = interior[outside]

        def apply_reduced_jacobian(reduced_direction: np.ndarray) -> np.ndarray:
            nonlocal local_linear_solves, local_linear_solve_wall_time_seconds
            full_direction = np.zeros(n_state, dtype=np.float64)
            full_direction[outside] = reduced_direction
            first_product = apply_physical_jvp(full_direction)
            local_rhs = first_product[patch].copy()
            local_rhs[~local_interior] = 0.0
            try:
                local_solve_start = perf_counter()
                local_response = lu_solve(projected_local_factor, local_rhs)
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise RuntimeError("assembled local Jacobian is singular in Schur action") from exc
            local_linear_solve_wall_time_seconds += (
                perf_counter() - local_solve_start
            )
            local_linear_solves += 1
            local_direction = np.zeros(n_state, dtype=np.float64)
            local_direction[patch] = local_response
            second_product = apply_physical_jvp(local_direction)
            reduced_product = first_product[outside] - second_product[outside]
            reduced_product[~outside_interior] = reduced_direction[~outside_interior]
            return reduced_product

        def apply_local_predictor(external_direction: np.ndarray) -> np.ndarray:
            """Predict the local response from the implicit local map derivative.

            The response solves J_ww * delta_z_w = J_wc * delta_z_c using
            the factorization assembled at the current outer state. Active
            local projected rows are suppressed because their coordinates stay
            at the current feasible bound until the local nonlinear solve
            updates the active set.
            """
            nonlocal local_linear_solves, local_linear_solve_wall_time_seconds
            nonlocal local_predictor_applications
            full_direction = np.zeros(n_state, dtype=np.float64)
            full_direction[outside] = external_direction
            coupling_product = apply_physical_jvp(full_direction)
            local_rhs = coupling_product[patch].copy()
            local_rhs[~local_interior] = 0.0
            try:
                local_solve_start = perf_counter()
                response = lu_solve(projected_local_factor, local_rhs)
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise RuntimeError(
                    "assembled local Jacobian is singular in local predictor"
                ) from exc
            local_linear_solve_wall_time_seconds += (
                perf_counter() - local_solve_start
            )
            local_linear_solves += 1
            local_predictor_applications += 1
            return np.asarray(response, dtype=np.float64)

        linear_operator = LinearOperator(
            (outside.size, outside.size),
            matvec=apply_reduced_jacobian,
            dtype=np.float64,
        )
        preconditioner_operator = None
        if reduced_preconditioner is not None:
            preconditioner_action = reduced_preconditioner(
                state.copy(), outside.copy(), outside_interior.copy()
            )
            if not callable(preconditioner_action):
                raise ValueError("reduced_preconditioner must return a callable")

            def apply_preconditioner(vector: np.ndarray) -> np.ndarray:
                nonlocal preconditioner_applications
                preconditioner_applications += 1
                output = np.asarray(
                    preconditioner_action(
                        np.asarray(vector, dtype=np.float64).reshape(-1).copy()
                    ),
                    dtype=np.float64,
                ).reshape(-1)
                if output.shape != (outside.size,) or not np.isfinite(output).all():
                    raise ValueError("reduced preconditioner returned an invalid vector")
                return output

            preconditioner_operator = LinearOperator(
                (outside.size, outside.size),
                matvec=apply_preconditioner,
                dtype=np.float64,
            )

        def count_krylov_iteration(_value: object) -> None:
            nonlocal krylov_iterations
            krylov_iterations += 1
            krylov_residual_norms.append(float(np.asarray(_value)))

        reduced_step, info = gmres(
            linear_operator,
            -projected[outside],
            M=preconditioner_operator,
            rtol=cfg.krylov_rtol,
            atol=cfg.krylov_atol,
            restart=min(outside.size, cfg.krylov_max_iterations),
            maxiter=cfg.krylov_max_iterations,
            callback=count_krylov_iteration,
            callback_type="legacy",
        )
        if info != 0 or not np.isfinite(reduced_step).all():
            termination_reason = f"reduced GMRES did not converge (info={info})"
            break

        # Record the linearized reduced residual before globalization.  This
        # distinguishes an inaccurate Schur direction from a step that merely
        # needs damping or continuation.
        linearized_reduced_residual = apply_reduced_jacobian(reduced_step)
        schur_direction_residual_norms.append(
            float(
                np.linalg.norm(
                    projected[outside] + linearized_reduced_residual
                )
            )
        )

        local_predictor_response = (
            apply_local_predictor(reduced_step)
            if cfg.use_local_predictor
            else None
        )
        base_norm = norm_history[-1]
        step_length = 1.0
        backtracking_count = 0
        accepted = False
        while step_length >= cfg.minimum_step_length:
            trial = state.copy()
            trial[outside] = np.minimum(
                np.maximum(
                    state[outside] + step_length * reduced_step,
                    lower[outside],
                ),
                upper[outside],
            )
            if local_predictor_response is not None:
                predicted_patch = state[patch].copy()
                predicted_patch[local_interior] -= (
                    step_length * local_predictor_response[local_interior]
                )
                trial[patch] = np.minimum(
                    np.maximum(predicted_patch, lower[patch]),
                    upper[patch],
                )
            (
                trial,
                trial_physical,
                trial_projected,
                trial_interior,
                trial_local_converged,
            ) = eliminate_local(trial)
            trial_norm = float(np.linalg.norm(trial_projected[free_mask]))
            if trial_local_converged and trial_norm <= (
                1.0 - cfg.armijo_slope * step_length
            ) * base_norm:
                state = trial
                physical = trial_physical
                projected = trial_projected
                interior = trial_interior
                norm_history.append(trial_norm)
                outer_step_lengths.append(float(step_length))
                outer_backtracking_counts.append(int(backtracking_count))
                accepted = True
                break
            step_length *= 0.5
            backtracking_count += 1
        if not accepted:
            outer_step_lengths.append(0.0)
            outer_backtracking_counts.append(int(backtracking_count))
            termination_reason = "outer reduced line search failed"
            break
        record_state_diagnostic(completed_outer_iterations, state, projected, interior)
        if (
            norm_history[-1] <= outer_tolerance
            and completed_outer_iterations >= cfg.minimum_outer_iterations
        ):
            termination_reason = "projected residual converged"
            return ReducedNewtonResult(
                state=state.copy(),
                converged=True,
                termination_reason=termination_reason,
                outer_iterations=completed_outer_iterations,
                local_newton_iterations=local_newton_iterations,
                local_linear_solves=local_linear_solves,
                local_predictor_applications=local_predictor_applications,
                local_linear_solve_wall_time_seconds=(
                    local_linear_solve_wall_time_seconds
                ),
                krylov_iterations=krylov_iterations,
                preconditioner_applications=preconditioner_applications,
                jvp_evaluations=jvp_evaluations,
                residual_evaluations=residual_evaluations,
                krylov_residual_norms=np.asarray(krylov_residual_norms),
                projected_residual_norms=np.asarray(norm_history),
                local_projected_residual_norm=float(np.linalg.norm(projected[patch])),
                state_diagnostic_history=tuple(state_diagnostic_history),
                state_diagnostic_wall_time_seconds=state_diagnostic_wall_time_seconds,
                wall_time_seconds=(
                    perf_counter() - start_time - state_diagnostic_wall_time_seconds
                ),
                schur_direction_residual_norms=np.asarray(
                    schur_direction_residual_norms
                ),
                outer_step_lengths=np.asarray(outer_step_lengths),
                outer_backtracking_counts=np.asarray(
                    outer_backtracking_counts, dtype=np.int64
                ),
            )

    return ReducedNewtonResult(
        state=state.copy(),
        converged=False,
        termination_reason=termination_reason,
        outer_iterations=completed_outer_iterations,
        local_newton_iterations=local_newton_iterations,
        local_linear_solves=local_linear_solves,
        local_predictor_applications=local_predictor_applications,
        local_linear_solve_wall_time_seconds=(
            local_linear_solve_wall_time_seconds
        ),
        krylov_iterations=krylov_iterations,
        preconditioner_applications=preconditioner_applications,
        jvp_evaluations=jvp_evaluations,
        residual_evaluations=residual_evaluations,
        krylov_residual_norms=np.asarray(krylov_residual_norms),
        projected_residual_norms=np.asarray(norm_history),
        local_projected_residual_norm=float(np.linalg.norm(projected[patch])),
        state_diagnostic_history=tuple(state_diagnostic_history),
        state_diagnostic_wall_time_seconds=state_diagnostic_wall_time_seconds,
        wall_time_seconds=(
            perf_counter() - start_time - state_diagnostic_wall_time_seconds
        ),
        schur_direction_residual_norms=np.asarray(
            schur_direction_residual_norms
        ),
        outer_step_lengths=np.asarray(outer_step_lengths),
        outer_backtracking_counts=np.asarray(
            outer_backtracking_counts, dtype=np.int64
        ),
    )


__all__ = [
    "ReducedNewtonConfig",
    "ReducedNewtonResult",
    "solve_reduced_nonlinear_system",
]
