"""Numerical diagnostics for a deterministic staggered fixed-point map.

This module computes fixed-point traces, finite-difference propagation
matrices, dominant modes, and additive cell energies.  It is independent of
any particular fracture assembler: callers must supply a deterministic map
``G: R^n -> R^n`` and are responsible for freezing committed state.

The finite-difference routine supports box constraints.  At a bound it uses
the available one-sided perturbation; a variable fixed by equal lower and
upper bounds receives a zero column.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


ArrayMap = Callable[[np.ndarray], np.ndarray]


def coupled_mode_lift(
    matrix_a: object,
    matrix_b: object,
    damage_mode: object,
    eigenvalue: complex,
) -> np.ndarray:
    """Lift a phase-field eigenmode to the full staggered state.

    Parameters
    ----------
    matrix_a : array-like, shape (n_u, n_u)
        Displacement Jacobian block. It must be nonsingular.
    matrix_b : array-like, shape (n_u, n_d)
        Damage-to-displacement Jacobian block.
    damage_mode : array-like, shape (n_d,)
        Eigenvector of D^{-1} C A^{-1} B.
    eigenvalue : complex
        Corresponding nonzero eigenvalue.

    Returns
    -------
    ndarray, shape (n_u+n_d,), dtype complex128
        Full eigenmode (-lambda^{-1} A^{-1} B v, v).
    """
    a = np.asarray(matrix_a, dtype=np.float64)
    b = np.asarray(matrix_b, dtype=np.float64)
    v = np.asarray(damage_mode, dtype=np.complex128).reshape(-1)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix_a must be square")
    if b.ndim != 2 or b.shape[0] != a.shape[0] or b.shape[1] != v.size:
        raise ValueError("matrix_b has incompatible shape")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Jacobian blocks must be finite")
    if abs(eigenvalue) <= np.finfo(float).eps:
        raise ValueError("eigenvalue must be nonzero")
    displacement_mode = -np.linalg.solve(a, b @ v) / eigenvalue
    return np.concatenate((displacement_mode, v))


def local_elimination_projection(jacobian: object, patch_dofs: object) -> np.ndarray:
    """Return the linearized local nonlinear-elimination projection.

    For a full Jacobian J and index set S this returns
    Q = I - P (P.T J P)^(-1) P.T J.

    Parameters
    ----------
    jacobian : array-like, shape (n, n)
        Full coupled Jacobian.
    patch_dofs : array-like, shape (n_patch,), dtype integer
        Full-vector indices eliminated by the local solve.

    Returns
    -------
    ndarray, shape (n, n), dtype float64
        Newly allocated oblique (or weighted orthogonal) projection.
    """
    matrix = np.asarray(jacobian, dtype=np.float64)
    patch = np.asarray(patch_dofs, dtype=np.int64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("jacobian must be square")
    if patch.size == 0:
        raise ValueError("patch_dofs must be nonempty")
    if np.any(patch < 0) or np.any(patch >= matrix.shape[0]):
        raise ValueError("patch_dofs contains an out-of-range index")
    if np.unique(patch).size != patch.size:
        raise ValueError("patch_dofs must contain unique indices")
    selector = np.zeros((matrix.shape[0], patch.size), dtype=np.float64)
    selector[patch, np.arange(patch.size)] = 1.0
    local = selector.T @ matrix @ selector
    correction = selector @ np.linalg.solve(local, selector.T @ matrix)
    return np.eye(matrix.shape[0], dtype=np.float64) - correction


def apply_local_elimination_projection(
    jacobian: object,
    patch_dofs: object,
    vector: object,
) -> np.ndarray:
    """Apply the local-elimination derivative without forming a dense projector.

    Parameters
    ----------
    jacobian : array-like, shape (n, n)
        Full coupled Jacobian in the same state ordering as ``vector``.
    patch_dofs : array-like, shape (n_patch,), dtype integer
        Unique full-state indices eliminated by the local solve.
    vector : array-like, shape (n,)
        Real or complex perturbation vector.  It is not modified.

    Returns
    -------
    ndarray, shape (n,)
        Newly allocated ``Q vector`` for
        ``Q=I-P(P.T J P)^{-1}P.T J``.  The dtype preserves complex modes.
    """
    matrix = np.asarray(jacobian, dtype=np.float64)
    patch = np.asarray(patch_dofs, dtype=np.int64).reshape(-1)
    direction = np.asarray(vector).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("jacobian must be square")
    if direction.shape != (matrix.shape[0],):
        raise ValueError("vector has incompatible shape")
    if patch.size == 0 or np.unique(patch).size != patch.size:
        raise ValueError("patch_dofs must contain unique indices")
    if np.any(patch < 0) or np.any(patch >= matrix.shape[0]):
        raise ValueError("patch_dofs contains an out-of-range index")
    if not np.isfinite(matrix).all() or not np.isfinite(direction).all():
        raise ValueError("jacobian and vector must be finite")
    local_matrix = matrix[np.ix_(patch, patch)]
    local_rhs = (matrix @ direction)[patch]
    correction = np.linalg.solve(local_matrix, local_rhs)
    projected = direction.copy()
    projected[patch] -= correction
    return projected


def subspace_cell_trace_indicator(
    basis: object,
    local_weights: object,
    cell_to_dof: object,
) -> np.ndarray:
    """Compute a basis-invariant additive energy indicator for a subspace.

    Parameters
    ----------
    basis : array-like, shape (n_dof, n_modes)
        W-orthonormal basis of the coupled slow subspace.
    local_weights : array-like, shape (n_cell, n_local, n_local)
        Symmetric positive-semidefinite cell weights whose assembled action is
        the diagnostic weight W.
    cell_to_dof : array-like, shape (n_cell, n_local), dtype integer
        Coupled finite-element connectivity in the full-vector ordering.

    Returns
    -------
    ndarray, shape (n_cell,), dtype float64
        trace(Z.T W_K Z) for every cell. The result is invariant under
        orthogonal changes of basis inside the supplied subspace.
    """
    vectors = np.asarray(basis, dtype=np.float64)
    weights = np.asarray(local_weights, dtype=np.float64)
    connectivity = np.asarray(cell_to_dof, dtype=np.int64)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("basis must be a nonempty two-dimensional array")
    if weights.ndim != 3 or weights.shape[0] == 0:
        raise ValueError("local_weights must be a nonempty three-dimensional array")
    if connectivity.ndim != 2 or connectivity.shape[0] != weights.shape[0]:
        raise ValueError("cell_to_dof has incompatible shape")
    if weights.shape[1] != weights.shape[2] or weights.shape[1] != connectivity.shape[1]:
        raise ValueError("local_weights and cell_to_dof have incompatible local size")
    if np.any(connectivity < 0) or np.any(connectivity >= vectors.shape[0]):
        raise ValueError("cell_to_dof contains an out-of-range index")
    if not np.isfinite(vectors).all() or not np.isfinite(weights).all():
        raise ValueError("basis and local_weights must be finite")
    indicators = np.empty(weights.shape[0], dtype=np.float64)
    for cell, dofs in enumerate(connectivity):
        local_basis = vectors[dofs, :]
        indicators[cell] = float(
            np.trace(local_basis.T @ weights[cell] @ local_basis)
        )
    if np.any(indicators < -1.0e-12):
        raise ValueError("local weights produced a negative energy indicator")
    return np.maximum(indicators, 0.0)


def weighted_survival_factor(
    projection: object,
    mode: object,
    weight: object,
) -> float:
    """Return the weighted norm ratio after a local elimination projection."""
    q = np.asarray(projection, dtype=np.float64)
    vector = np.asarray(mode, dtype=np.float64).reshape(-1)
    w = np.asarray(weight, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != q.shape[1] or q.shape[0] != vector.size:
        raise ValueError("projection and mode have incompatible shapes")
    if w.shape != q.shape or not np.isfinite(q).all() or not np.isfinite(vector).all():
        raise ValueError("projection, mode, and weight must be finite and compatible")
    if (
        not np.isfinite(w).all()
        or not np.allclose(w, w.T, rtol=1.0e-12, atol=1.0e-14)
        or np.min(np.linalg.eigvalsh(w)) <= 0.0
    ):
        raise ValueError("weight must be symmetric positive definite")
    denominator = float(vector @ w @ vector)
    if denominator <= np.finfo(float).tiny:
        raise ValueError("mode has zero diagnostic norm")
    reduced = q @ vector
    numerator = float(reduced @ w @ reduced)
    return float(np.sqrt(max(0.0, numerator) / denominator))


def diagonal_patch_survival_factor(
    mode: object,
    weight_diagonal: object,
    patch_dofs: object,
) -> float:
    """Return the SPD coordinate-patch survival factor in a diagonal weight.

    Parameters
    ----------
    mode : array-like, shape (n,)
        Real or complex coupled mode.
    weight_diagonal : array-like, shape (n,)
        Strictly positive diagonal of the SPD calibration weight ``W``.
    patch_dofs : array-like, shape (n_patch,), dtype integer
        Full-state coordinates eliminated by the local calibration solve.

    Returns
    -------
    float
        ``||Q_patch mode||_W / ||mode||_W``, where the ``W``-orthogonal
        complement projection sets the selected coordinate patch to zero.
    """
    vector = np.asarray(mode).reshape(-1)
    diagonal = _as_finite_vector(weight_diagonal, name="weight_diagonal")
    patch = np.asarray(patch_dofs, dtype=np.int64).reshape(-1)
    if vector.shape != diagonal.shape or not np.isfinite(vector).all():
        raise ValueError("mode and weight_diagonal must be finite and compatible")
    if np.any(diagonal <= 0.0):
        raise ValueError("weight_diagonal must be strictly positive")
    if patch.size == 0 or np.unique(patch).size != patch.size:
        raise ValueError("patch_dofs must contain unique indices")
    if np.any(patch < 0) or np.any(patch >= vector.size):
        raise ValueError("patch_dofs contains an out-of-range index")
    energy = diagonal * np.abs(vector) ** 2
    total = float(np.sum(energy))
    if total <= np.finfo(float).tiny:
        raise ValueError("mode has zero diagnostic norm")
    remaining = total - float(np.sum(energy[patch]))
    return float(np.sqrt(max(0.0, remaining) / total))


def _as_finite_vector(values: object, *, name: str) -> np.ndarray:
    """Return ``values`` as a newly allocated finite float64 vector."""
    array = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one entry")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite entries")
    return array


@dataclass(frozen=True)
class FixedPointResult:
    """Result of a deterministic fixed-point iteration.

    Attributes
    ----------
    solution : ndarray, shape (n,), dtype float64
        Final iterate, newly allocated and dimensionless.
    increments : ndarray, shape (n_iterations, n), dtype float64
        Consecutive differences ``x_{k+1} - x_k``.
    increment_norms : ndarray, shape (n_iterations,), dtype float64
        Euclidean norms of ``increments``.
    converged : bool
        Whether the absolute-plus-relative stopping criterion was met.
    """

    solution: np.ndarray
    increments: np.ndarray
    increment_norms: np.ndarray
    converged: bool

    @property
    def iterations(self) -> int:
        """Return the number of fixed-point map evaluations."""
        return int(self.increment_norms.size)

    @property
    def asymptotic_ratio(self) -> float:
        """Return the last nondegenerate consecutive increment-norm ratio."""
        if self.increment_norms.size < 2:
            return float("nan")
        previous = self.increment_norms[-2]
        if previous <= np.finfo(np.float64).tiny:
            return float("nan")
        return float(self.increment_norms[-1] / previous)


@dataclass(frozen=True)
class LocalEliminationResult:
    """Result of a nonlinear residual solve on a fixed coordinate patch.

    Attributes
    ----------
    state : ndarray, shape (n_state,), dtype float64
        Full state after the local solve. Coordinates outside ``patch_dofs``
        equal the input state up to roundoff.
    residual_norms : ndarray, shape (1,) or (2,), dtype float64
        Initial and, when solved, final projected-residual norms.
    converged : bool
        Whether the local residual met the absolute-plus-relative criterion.
    iteration_count : int
        Trust-region Jacobian evaluations reported by the nonlinear solver.
    residual_evaluations : int
        Actual callback evaluations, including finite-difference columns.
    """

    state: np.ndarray
    residual_norms: np.ndarray
    converged: bool
    iteration_count: int = 0
    residual_evaluations: int = 0

    @property
    def iterations(self) -> int:
        """Return the number of local trust-region Jacobian evaluations."""
        return int(self.iteration_count)

    @property
    def initial_residual_norm(self) -> float:
        """Return the local residual norm before the first update."""
        return float(self.residual_norms[0])

    @property
    def final_residual_norm(self) -> float:
        """Return the local residual norm after the final update."""
        return float(self.residual_norms[-1])


def solve_local_nonlinear_residual(
    local_residual: Callable[[np.ndarray], np.ndarray],
    initial_state: object,
    patch_dofs: object,
    *,
    lower_bound: Optional[object] = None,
    upper_bound: Optional[object] = None,
    relative_step: float = 1.0e-6,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-8,
    max_iterations: int = 200,
) -> LocalEliminationResult:
    """Solve ``P.T F(x)=0`` while holding coordinates outside a patch fixed.

    Parameters
    ----------
    local_residual : callable
        Deterministic map from a full state vector of shape ``(n_state,)`` to
        the residual restricted to ``patch_dofs``, with shape ``(n_patch,)``.
        The callback may update an external FE assembler, but must evaluate the
        residual for the supplied full state.
    initial_state : array-like, shape (n_state,)
        Full state. It is not modified.
    patch_dofs : array-like, shape (n_patch,), dtype integer
        Unique full-state coordinates solved by Newton updates.
    lower_bound, upper_bound : array-like, shape (n_patch,), optional
        Feasible box for local coordinates. Infinite endpoints are allowed.
    relative_step : float
        Relative finite-difference step used by the trust-region Jacobian.
    atol, rtol : float
        Stop when the local residual norm is at most
        ``atol + rtol*max(1, initial_residual_norm)``.
    max_iterations : int
        Maximum residual evaluations reported by the trust-region solver.

    Returns
    -------
    LocalEliminationResult
        Full state and residual trace. A new array is returned; the input state
        and patch index arrays are never modified.

    Raises
    ------
    ValueError
        If shapes, bounds, or tolerances are invalid.
    RuntimeError
        If the local trust-region solver returns a non-finite state.

    Notes
    -----
    The Jacobian is finite-differenced only in patch coordinates, so the
    implementation is a genuine residual elimination and does not assemble a
    dense global Jacobian. A bound-constrained trust-region least-squares
    globalization handles nonsmooth history-field and active-set transitions.
    """
    if relative_step <= 0.0 or atol < 0.0 or rtol < 0.0:
        raise ValueError("relative_step must be positive and tolerances nonnegative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    state = _as_finite_vector(initial_state, name="initial_state")
    patch = np.asarray(patch_dofs, dtype=np.int64).reshape(-1)
    if patch.size == 0 or np.unique(patch).size != patch.size:
        raise ValueError("patch_dofs must contain unique indices")
    if np.any(patch < 0) or np.any(patch >= state.size):
        raise ValueError("patch_dofs contains an out-of-range index")
    local = state[patch].copy()
    lower = (
        np.full(local.size, -np.inf, dtype=np.float64)
        if lower_bound is None
        else np.asarray(lower_bound, dtype=np.float64).reshape(-1).copy()
    )
    upper = (
        np.full(local.size, np.inf, dtype=np.float64)
        if upper_bound is None
        else np.asarray(upper_bound, dtype=np.float64).reshape(-1).copy()
    )
    if lower.shape != local.shape or upper.shape != local.shape:
        raise ValueError("local bounds must have the same shape as patch_dofs")
    if np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper):
        raise ValueError("local bounds must be ordered and must not contain NaN")
    if np.any(local < lower) or np.any(local > upper):
        raise ValueError("initial local state lies outside the supplied bounds")

    residual_evaluations = 0

    def evaluate(local_state: np.ndarray) -> np.ndarray:
        nonlocal residual_evaluations
        residual_evaluations += 1
        full = state.copy()
        full[patch] = local_state
        residual = _as_finite_vector(local_residual(full), name="local residual")
        if residual.shape != local.shape:
            raise ValueError("local residual must have the same shape as patch_dofs")
        # Box-constrained phase coordinates satisfy a variational inequality,
        # not an unconstrained residual equation. The natural projected
        # residual has the same interior root and encodes the KKT signs at
        # active bounds. Unbounded coordinates reduce exactly to ``residual``.
        projected_state = np.minimum(
            np.maximum(local_state - residual, lower), upper
        )
        return local_state - projected_state

    residual = evaluate(local)
    initial_norm = float(np.linalg.norm(residual))
    tolerance = float(atol + rtol * max(1.0, initial_norm))
    if initial_norm <= tolerance:
        return LocalEliminationResult(
            state=state.copy(),
            residual_norms=np.asarray([initial_norm], dtype=np.float64),
            converged=True,
            iteration_count=0,
            residual_evaluations=residual_evaluations,
        )

    from scipy.optimize import least_squares

    solver_tolerance = max(
        1.0e-12,
        min(1.0e-8, 0.1 * tolerance),
    )
    optimized = least_squares(
        evaluate,
        local,
        jac="2-point",
        bounds=(lower, upper),
        diff_step=relative_step,
        ftol=solver_tolerance,
        xtol=solver_tolerance,
        gtol=solver_tolerance,
        x_scale="jac",
        max_nfev=max_iterations,
    )
    local = np.asarray(optimized.x, dtype=np.float64)
    residual = evaluate(local)
    final_norm = float(np.linalg.norm(residual))
    converged = final_norm <= tolerance
    if not np.isfinite(local).all() or not np.isfinite(final_norm):
        raise RuntimeError("local trust-region solve returned a non-finite state")

    result = state.copy()
    result[patch] = local
    return LocalEliminationResult(
        state=result,
        residual_norms=np.asarray([initial_norm, final_norm], dtype=np.float64),
        converged=converged,
        iteration_count=int(optimized.njev or 0),
        residual_evaluations=residual_evaluations,
    )


@dataclass(frozen=True)
class DominantModeResult:
    """Dominant eigenpair of a real square propagation matrix."""

    eigenvalue: complex
    spectral_radius: float
    mode: np.ndarray
    eigen_residual: float
    spectral_gap: float


@dataclass(frozen=True)
class SlowSubspaceResult:
    """Spectrally selected real invariant subspace of a real propagation map.

    Attributes
    ----------
    eigenvalues : ndarray, shape (n,), dtype complex128
        All eigenvalues, sorted in nonincreasing modulus.
    selected_eigenvalues : ndarray, shape (n_selected,), dtype complex128
        Eigenvalues whose modulus is at least ``relative_radius`` times the
        spectral radius.  A complex-conjugate pair is retained as its shared
        real invariant subspace.
    basis : ndarray, shape (n, r), dtype float64
        Euclidean-orthonormal basis of that real slow subspace.  It has one
        column for a real eigenvector and two columns for a genuinely complex
        conjugate pair.
    spectral_radius : float
        Largest eigenvalue modulus.
    cutoff : float
        Absolute spectral cutoff used to select the subspace.
    """

    eigenvalues: np.ndarray
    selected_eigenvalues: np.ndarray
    basis: np.ndarray
    spectral_radius: float
    cutoff: float


@dataclass(frozen=True)
class IncrementSubspaceResult:
    """Weighted slow-subspace estimate from recent fixed-point increments.

    Attributes
    ----------
    basis : ndarray, shape (n_state, r), dtype float64
        Basis satisfying ``basis.T @ W @ basis = I``.
    singular_values : ndarray, shape (n_window,), dtype float64
        Singular values of the individually W-normalized increment snapshot
        matrix, sorted in nonincreasing order.
    increment_norms : ndarray, shape (n_window,), dtype float64
        W-norms of the chronological increments used in the estimate.
    window_size : int
        Actual number of trailing increments used.
    contraction_estimate : float
        Median ratio of consecutive W-norms in the selected window.  It is
        NaN when fewer than two usable consecutive increments are available.
    """

    basis: np.ndarray
    singular_values: np.ndarray
    increment_norms: np.ndarray
    window_size: int
    contraction_estimate: float

    @property
    def dimension(self) -> int:
        """Return the retained online subspace dimension."""
        return int(self.basis.shape[1])


@dataclass(frozen=True)
class MemoryAugmentedSubspaceResult:
    """Current online subspace augmented by independent stored directions.

    Attributes
    ----------
    basis : ndarray, shape (n_state, r), dtype float64
        W-orthonormal augmented basis. Current directions precede retained
        memory directions.
    current_dimension : int
        Effective dimension of the supplied current basis.
    memory_candidate_dimension : int
        Number of supplied memory directions.
    retained_memory_dimension : int
        Number of memory directions that pass the independence test and the
        optional dimension cap.
    independence_ratios : ndarray, shape (n_memory,), dtype float64
        W-norm of each memory direction after projection divided by its
        original W-norm. Zero-norm directions receive ratio zero.
    """

    basis: np.ndarray
    current_dimension: int
    memory_candidate_dimension: int
    retained_memory_dimension: int
    independence_ratios: np.ndarray

    @property
    def dimension(self) -> int:
        """Return the augmented subspace dimension."""
        return int(self.basis.shape[1])


def iterate_fixed_point(
    apply_map: ArrayMap,
    initial: object,
    *,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-10,
    max_iterations: int = 100,
) -> FixedPointResult:
    """Iterate ``x_{k+1}=G(x_k)`` until an increment criterion is met.

    Parameters
    ----------
    apply_map : callable
        Deterministic map accepting and returning shape ``(n,)`` arrays.  The
        callback may mutate internal solver state, but repeated calls at the
        same input must return the same output.
    initial : array-like, shape (n,)
        Initial dimensionless iterate.  It is not modified.
    atol, rtol : float
        Stop when ``||dx||_2 <= atol + rtol*max(1, ||x_new||_2)``.
    max_iterations : int
        Positive cap on map evaluations.

    Returns
    -------
    FixedPointResult
        Newly allocated solution and iteration trace.
    """
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("atol and rtol must be nonnegative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    current = _as_finite_vector(initial, name="initial")
    increments: list[np.ndarray] = []
    norms: list[float] = []
    converged = False

    for _ in range(max_iterations):
        next_iterate = _as_finite_vector(apply_map(current.copy()), name="map output")
        if next_iterate.shape != current.shape:
            raise ValueError(
                f"map output shape {next_iterate.shape} does not match {current.shape}"
            )
        increment = next_iterate - current
        increment_norm = float(np.linalg.norm(increment))
        increments.append(increment)
        norms.append(increment_norm)
        current = next_iterate
        threshold = atol + rtol * max(1.0, float(np.linalg.norm(current)))
        if increment_norm <= threshold:
            converged = True
            break

    return FixedPointResult(
        solution=current.copy(),
        increments=np.asarray(increments, dtype=np.float64),
        increment_norms=np.asarray(norms, dtype=np.float64),
        converged=converged,
    )


def finite_difference_jacobian(
    apply_map: ArrayMap,
    fixed_point: object,
    *,
    relative_step: float = 1.0e-6,
    lower_bound: Optional[object] = None,
    upper_bound: Optional[object] = None,
) -> np.ndarray:
    """Approximate the Jacobian of a deterministic fixed-point map.

    Parameters
    ----------
    apply_map : callable
        Map ``G`` with a shape ``(n,)`` float input and output.
    fixed_point : array-like, shape (n,)
        Linearization point, dimensionless.
    relative_step : float
        Positive perturbation scale.  Coordinate ``j`` uses
        ``relative_step*max(1, abs(x_j))`` before applying bounds.
    lower_bound, upper_bound : array-like, shape (n,), optional
        Feasible box.  Bound-active coordinates use one-sided differences.

    Returns
    -------
    ndarray, shape (n, n), dtype float64
        Dense Jacobian whose columns follow the input degree-of-freedom order.
    """
    if relative_step <= 0.0:
        raise ValueError("relative_step must be positive")
    point = _as_finite_vector(fixed_point, name="fixed_point")
    n = point.size
    lower = (
        np.full(n, -np.inf, dtype=np.float64)
        if lower_bound is None
        else np.asarray(lower_bound, dtype=np.float64).reshape(-1).copy()
    )
    upper = (
        np.full(n, np.inf, dtype=np.float64)
        if upper_bound is None
        else np.asarray(upper_bound, dtype=np.float64).reshape(-1).copy()
    )
    if lower.shape != point.shape or upper.shape != point.shape:
        raise ValueError("bounds must have the same shape as fixed_point")
    if np.isnan(lower).any() or np.isnan(upper).any():
        raise ValueError("bounds must not contain NaN")
    if np.any(lower > upper):
        raise ValueError("lower_bound exceeds upper_bound")
    if np.any(point < lower) or np.any(point > upper):
        raise ValueError("fixed_point lies outside the supplied bounds")

    jacobian = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        step = relative_step * max(1.0, abs(point[j]))
        plus = point.copy()
        minus = point.copy()
        plus[j] = min(point[j] + step, upper[j])
        minus[j] = max(point[j] - step, lower[j])
        denominator = plus[j] - minus[j]
        if denominator <= 0.0:
            continue
        image_plus = _as_finite_vector(apply_map(plus), name="positive map output")
        image_minus = _as_finite_vector(apply_map(minus), name="negative map output")
        if image_plus.shape != point.shape or image_minus.shape != point.shape:
            raise ValueError("map output shape changed during finite differencing")
        jacobian[:, j] = (image_plus - image_minus) / denominator
    return jacobian


def finite_difference_jacobian_rectangular(
    apply_map: ArrayMap,
    fixed_point: object,
    *,
    relative_step: float = 1.0e-6,
    lower_bound: Optional[object] = None,
    upper_bound: Optional[object] = None,
) -> np.ndarray:
    """Approximate the Jacobian of a vector map with unequal input/output sizes.

    Parameters
    ----------
    apply_map : callable
        Deterministic map from shape ``(n_in,)`` to a fixed shape
        ``(n_out,)``.  It may mutate internal solver state, but repeated calls
        at the same input must return the same vector.
    fixed_point : array-like, shape (n_in,)
        Input-space linearization point.
    relative_step : float
        Positive coordinate perturbation scale.
    lower_bound, upper_bound : array-like, shape (n_in,), optional
        Feasible input box. Infinite endpoints are allowed; equal endpoints
        produce a zero column.

    Returns
    -------
    ndarray, shape (n_out, n_in), dtype float64
        Dense rectangular Jacobian. The routine evaluates only input
        directions, which is useful when analytically zero state blocks should
        not trigger expensive solver calls.
    """
    if relative_step <= 0.0:
        raise ValueError("relative_step must be positive")
    point = _as_finite_vector(fixed_point, name="fixed_point")
    n_input = point.size
    lower = (
        np.full(n_input, -np.inf, dtype=np.float64)
        if lower_bound is None
        else np.asarray(lower_bound, dtype=np.float64).reshape(-1).copy()
    )
    upper = (
        np.full(n_input, np.inf, dtype=np.float64)
        if upper_bound is None
        else np.asarray(upper_bound, dtype=np.float64).reshape(-1).copy()
    )
    if lower.shape != point.shape or upper.shape != point.shape:
        raise ValueError("bounds must have the same shape as fixed_point")
    if np.isnan(lower).any() or np.isnan(upper).any():
        raise ValueError("bounds must not contain NaN")
    if np.any(lower > upper) or np.any(point < lower) or np.any(point > upper):
        raise ValueError("fixed_point and bounds define an invalid feasible box")

    reference = _as_finite_vector(apply_map(point.copy()), name="reference map output")
    jacobian = np.zeros((reference.size, n_input), dtype=np.float64)
    for column in range(n_input):
        step = relative_step * max(1.0, abs(point[column]))
        plus = point.copy()
        minus = point.copy()
        plus[column] = min(point[column] + step, upper[column])
        minus[column] = max(point[column] - step, lower[column])
        denominator = plus[column] - minus[column]
        if denominator <= 0.0:
            continue
        image_plus = _as_finite_vector(apply_map(plus), name="positive map output")
        image_minus = _as_finite_vector(apply_map(minus), name="negative map output")
        if image_plus.shape != reference.shape or image_minus.shape != reference.shape:
            raise ValueError("map output shape changed during finite differencing")
        jacobian[:, column] = (image_plus - image_minus) / denominator
    return jacobian


def dominant_mode(matrix: object) -> DominantModeResult:
    """Return the eigenpair of largest modulus for a real square matrix."""
    operator = np.asarray(matrix, dtype=np.float64)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("matrix must be square")
    if operator.shape[0] == 0 or not np.isfinite(operator).all():
        raise ValueError("matrix must be nonempty and finite")

    eigenvalues, eigenvectors = np.linalg.eig(operator)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    index = int(order[0])
    value = complex(eigenvalues[index])
    mode = np.asarray(eigenvectors[:, index], dtype=np.complex128)
    mode_norm = float(np.linalg.norm(mode))
    if mode_norm == 0.0:
        raise RuntimeError("dominant eigenvector has zero norm")
    mode /= mode_norm
    residual = float(np.linalg.norm(operator @ mode - value * mode))
    radius = float(abs(value))
    second = float(abs(eigenvalues[order[1]])) if len(order) > 1 else 0.0
    return DominantModeResult(
        eigenvalue=value,
        spectral_radius=radius,
        mode=mode,
        eigen_residual=residual,
        spectral_gap=radius - second,
    )


def spectral_slow_subspace(
    matrix: object,
    *,
    relative_radius: float,
) -> SlowSubspaceResult:
    """Select the real slow subspace by a relative spectral cutoff.

    Parameters
    ----------
    matrix : array-like, shape (n, n)
        Real propagation matrix in a fixed state-vector ordering.
    relative_radius : float
        Dimensionless cutoff in ``(0, 1]``.  Retain every eigenvalue satisfying
        ``abs(lambda) >= relative_radius * rho(matrix)``.

    Returns
    -------
    SlowSubspaceResult
        Eigenvalue list and a newly allocated Euclidean-orthonormal real basis.
        The basis spans real and imaginary parts of the selected eigenvectors,
        so it is meaningful even when a real propagation matrix has complex
        conjugate slow eigenpairs.

    Raises
    ------
    ValueError
        If the matrix is invalid, the cutoff is outside ``(0, 1]``, or every
        eigenvalue vanishes.
    """
    if not 0.0 < relative_radius <= 1.0:
        raise ValueError("relative_radius must lie in (0, 1]")
    operator = np.asarray(matrix, dtype=np.float64)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("matrix must be square")
    if operator.shape[0] == 0 or not np.isfinite(operator).all():
        raise ValueError("matrix must be nonempty and finite")

    eigenvalues, eigenvectors = np.linalg.eig(operator)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.complex128)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.complex128)
    radius = float(abs(eigenvalues[0]))
    if radius <= np.finfo(np.float64).tiny:
        raise ValueError("matrix has no nonzero spectral mode")
    cutoff = float(relative_radius * radius)
    selected = np.abs(eigenvalues) >= cutoff
    selected_vectors = eigenvectors[:, selected]
    real_candidates = np.concatenate(
        (np.real(selected_vectors), np.imag(selected_vectors)), axis=1
    )
    candidate_norms = np.linalg.norm(real_candidates, axis=0)
    real_candidates = real_candidates[:, candidate_norms > 1.0e-13]
    if real_candidates.shape[1] == 0:
        raise RuntimeError("selected eigenvectors have no real invariant span")
    left_vectors, singular_values, _ = np.linalg.svd(
        real_candidates, full_matrices=False
    )
    rank_tolerance = max(real_candidates.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    if rank == 0:
        raise RuntimeError("selected eigenvectors are numerically rank deficient")
    return SlowSubspaceResult(
        eigenvalues=eigenvalues.copy(),
        selected_eigenvalues=eigenvalues[selected].copy(),
        basis=np.asarray(left_vectors[:, :rank], dtype=np.float64).copy(),
        spectral_radius=radius,
        cutoff=cutoff,
    )


def coupled_slow_subspace_from_sweep_column(
    damage_to_full: object,
    *,
    relative_radius: float,
) -> SlowSubspaceResult:
    """Lift the slow eigenspace of ``T`` through a full sweep block column.

    Parameters
    ----------
    damage_to_full : array-like, shape (n_u+n_d, n_d)
        Nonzero block column of the exact staggered derivative,
        ``[U; T]``, where ``U = d u^+ / d d`` and the last ``n_d`` rows are
        the phase propagation matrix ``T``.
    relative_radius : float
        Dimensionless cutoff in ``(0, 1]`` applied to eigenvalue moduli of
        ``T``.

    Returns
    -------
    SlowSubspaceResult
        Eigenvalues of ``T`` and an Euclidean-orthonormal real basis in the
        full ``(u,d)`` state ordering. For every selected eigenpair
        ``T v = lambda v``, the lifted vector is ``(U v / lambda, v)``.

    Notes
    -----
    This avoids assembling or diagonalizing the larger block-triangular
    matrix ``G = [[0,U],[0,T]]`` while producing the same nonzero invariant
    subspace.
    """
    column = np.asarray(damage_to_full, dtype=np.float64)
    if column.ndim != 2 or column.shape[1] == 0 or column.shape[0] <= column.shape[1]:
        raise ValueError("damage_to_full must have shape (n_u+n_d, n_d) with n_u>0")
    if not np.isfinite(column).all():
        raise ValueError("damage_to_full must be finite")
    n_damage = column.shape[1]
    displacement_block = column[:-n_damage, :]
    propagation = column[-n_damage:, :]
    damage_slow = spectral_slow_subspace(
        propagation, relative_radius=relative_radius
    )

    eigenvalues, eigenvectors = np.linalg.eig(propagation)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.complex128)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.complex128)
    selected = np.abs(eigenvalues) >= damage_slow.cutoff
    lifted_vectors: list[np.ndarray] = []
    for eigenvalue, damage_mode in zip(eigenvalues[selected], eigenvectors[:, selected].T):
        if abs(eigenvalue) <= np.finfo(float).tiny:
            continue
        displacement_mode = displacement_block @ damage_mode / eigenvalue
        lifted_vectors.append(np.concatenate((displacement_mode, damage_mode)))
    if not lifted_vectors:
        raise RuntimeError("selected slow eigenvalues cannot be lifted")
    complex_basis = np.column_stack(lifted_vectors)
    real_candidates = np.concatenate(
        (np.real(complex_basis), np.imag(complex_basis)), axis=1
    )
    candidate_norms = np.linalg.norm(real_candidates, axis=0)
    real_candidates = real_candidates[:, candidate_norms > 1.0e-13]
    left_vectors, singular_values, _ = np.linalg.svd(
        real_candidates, full_matrices=False
    )
    rank_tolerance = max(real_candidates.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    if rank == 0:
        raise RuntimeError("lifted slow eigenvectors are numerically rank deficient")
    return SlowSubspaceResult(
        eigenvalues=damage_slow.eigenvalues.copy(),
        selected_eigenvalues=damage_slow.selected_eigenvalues.copy(),
        basis=np.asarray(left_vectors[:, :rank], dtype=np.float64).copy(),
        spectral_radius=damage_slow.spectral_radius,
        cutoff=damage_slow.cutoff,
    )


def weighted_orthonormalize(
    basis: object,
    weight: object,
) -> np.ndarray:
    """Return a basis orthonormal in a supplied SPD diagnostic weight.

    Parameters
    ----------
    basis : array-like, shape (n, r)
        Real full-column-rank vectors in the target subspace.  The input is
        never modified.
    weight : array-like, shape (n,) or (n, n)
        Positive diagonal of an SPD weight, or its full symmetric positive
        definite matrix.  The weight uses the same full-vector ordering as
        ``basis``.

    Returns
    -------
    ndarray, shape (n, r_eff), dtype float64
        Newly allocated vectors ``Z`` satisfying ``Z.T @ W @ Z = I``.  Linear
        dependencies in ``basis`` are removed using a scale-aware tolerance.
    """
    vectors = np.asarray(basis, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("basis must be a nonempty two-dimensional array")
    if not np.isfinite(vectors).all():
        raise ValueError("basis must be finite")
    diagnostic_weight = np.asarray(weight, dtype=np.float64)
    if diagnostic_weight.ndim == 1:
        if diagnostic_weight.shape != (vectors.shape[0],):
            raise ValueError("diagonal weight has incompatible shape")
        if not np.isfinite(diagnostic_weight).all() or np.any(diagnostic_weight <= 0.0):
            raise ValueError("diagonal weight must be finite and strictly positive")
        gram = vectors.T @ (diagnostic_weight[:, None] * vectors)
    elif diagnostic_weight.ndim == 2:
        if diagnostic_weight.shape != (vectors.shape[0], vectors.shape[0]):
            raise ValueError("weight matrix has incompatible shape")
        if (
            not np.isfinite(diagnostic_weight).all()
            or not np.allclose(diagnostic_weight, diagnostic_weight.T, rtol=1.0e-12, atol=1.0e-14)
            or np.min(np.linalg.eigvalsh(diagnostic_weight)) <= 0.0
        ):
            raise ValueError("weight matrix must be symmetric positive definite")
        gram = vectors.T @ diagnostic_weight @ vectors
    else:
        raise ValueError("weight must be a diagonal vector or a square matrix")

    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    tolerance = max(gram.shape) * np.finfo(float).eps * max(float(eigenvalues[-1]), 1.0)
    retained = eigenvalues > tolerance
    if not np.any(retained):
        raise ValueError("basis has zero norm in the supplied weight")
    return np.asarray(
        vectors @ (eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])),
        dtype=np.float64,
    )


def online_increment_subspace(
    increments: object,
    weight: object,
    *,
    window_size: int = 5,
    relative_singular_value: float = 1.0e-2,
    max_dimension: Optional[int] = None,
) -> IncrementSubspaceResult:
    """Estimate a slow subspace from recent coupled fixed-point increments.

    Parameters
    ----------
    increments : array-like, shape (n_iterations, n_state)
        Chronological coupled increments ``z[k+1]-z[k]``.  Only the last
        ``window_size`` rows are used.  The input is not modified.
    weight : array-like, shape (n_state,) or (n_state, n_state)
        Strictly positive diagonal or full SPD diagnostic weight ``W``.
    window_size : int
        Positive number of trailing increments in the snapshot window.
    relative_singular_value : float
        Retain singular directions at least this fraction of the largest
        singular value.  It must lie in ``(0, 1]``.
    max_dimension : int, optional
        Optional positive cap on the retained dimension.

    Returns
    -------
    IncrementSubspaceResult
        Weighted-orthonormal online basis and scale-free snapshot diagnostics.

    Notes
    -----
    Each increment is normalized in the W-norm before the small snapshot
    eigenproblem is solved.  This prevents geometric decay of fixed-point
    increments from making the oldest vector dominate solely by magnitude.
    The operation costs ``O(n_state * window_size**2)`` for diagonal weights
    and requires no Jacobian action or extra finite-element sweep.
    """
    snapshots = np.asarray(increments, dtype=np.float64)
    if snapshots.ndim != 2 or snapshots.shape[0] == 0 or snapshots.shape[1] == 0:
        raise ValueError("increments must be a nonempty two-dimensional array")
    if not np.isfinite(snapshots).all():
        raise ValueError("increments must be finite")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not 0.0 < relative_singular_value <= 1.0:
        raise ValueError("relative_singular_value must lie in (0, 1]")
    if max_dimension is not None and max_dimension <= 0:
        raise ValueError("max_dimension must be positive when supplied")

    actual_window = min(int(window_size), snapshots.shape[0])
    columns = snapshots[-actual_window:, :].T.copy()
    diagnostic_weight = np.asarray(weight, dtype=np.float64)
    if diagnostic_weight.ndim == 1:
        if diagnostic_weight.shape != (columns.shape[0],):
            raise ValueError("diagonal weight has incompatible shape")
        if not np.isfinite(diagnostic_weight).all() or np.any(
            diagnostic_weight <= 0.0
        ):
            raise ValueError("diagonal weight must be finite and strictly positive")
        gram = columns.T @ (diagnostic_weight[:, None] * columns)
    elif diagnostic_weight.ndim == 2:
        if diagnostic_weight.shape != (columns.shape[0], columns.shape[0]):
            raise ValueError("weight matrix has incompatible shape")
        if (
            not np.isfinite(diagnostic_weight).all()
            or not np.allclose(
                diagnostic_weight,
                diagnostic_weight.T,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
            or np.min(np.linalg.eigvalsh(diagnostic_weight)) <= 0.0
        ):
            raise ValueError("weight matrix must be symmetric positive definite")
        gram = columns.T @ diagnostic_weight @ columns
    else:
        raise ValueError("weight must be a diagonal vector or a square matrix")

    increment_norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    norm_tolerance = (
        max(columns.shape)
        * np.finfo(float).eps
        * max(float(np.max(increment_norms)), 1.0)
    )
    nonzero = increment_norms > norm_tolerance
    if not np.any(nonzero):
        raise ValueError("selected increment window has zero diagnostic norm")
    normalized = columns[:, nonzero] / increment_norms[nonzero]
    if diagnostic_weight.ndim == 1:
        correlation = normalized.T @ (diagnostic_weight[:, None] * normalized)
    else:
        correlation = normalized.T @ diagnostic_weight @ normalized
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (correlation + correlation.T)
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    singular_values = np.sqrt(eigenvalues)
    numerical_tolerance = (
        max(normalized.shape)
        * np.finfo(float).eps
        * max(float(singular_values[0]), 1.0)
    )
    retained = np.flatnonzero(
        (singular_values >= relative_singular_value * singular_values[0])
        & (singular_values > numerical_tolerance)
    )
    if max_dimension is not None:
        retained = retained[: int(max_dimension)]
    if retained.size == 0:
        raise RuntimeError("increment snapshot matrix has no retained direction")
    basis = normalized @ (
        eigenvectors[:, retained] / singular_values[retained]
    )

    ratios = np.divide(
        increment_norms[1:],
        increment_norms[:-1],
        out=np.full(max(actual_window - 1, 0), np.nan, dtype=np.float64),
        where=increment_norms[:-1] > norm_tolerance,
    )
    finite_ratios = ratios[np.isfinite(ratios)]
    contraction = (
        float(np.median(finite_ratios))
        if finite_ratios.size
        else float("nan")
    )
    return IncrementSubspaceResult(
        basis=np.asarray(basis, dtype=np.float64).copy(),
        singular_values=np.asarray(singular_values, dtype=np.float64).copy(),
        increment_norms=increment_norms.copy(),
        window_size=actual_window,
        contraction_estimate=contraction,
    )


def weighted_principal_angles(
    reference_basis: object,
    candidate_basis: object,
    weight: object,
    *,
    include_dimension_gap: bool = True,
) -> np.ndarray:
    """Return principal angles between two subspaces in an SPD weight.

    Both inputs have shape ``(n_state, r)`` and may use arbitrary full-rank
    bases.  Angles are returned in nondecreasing order in radians.  When
    ``include_dimension_gap`` is true, missing dimensions are represented by
    right angles, so a one-dimensional candidate cannot appear to recover a
    two-dimensional reference subspace exactly.
    """
    reference = np.asarray(reference_basis, dtype=np.float64)
    candidate = np.asarray(candidate_basis, dtype=np.float64)
    if (
        reference.ndim != 2
        or candidate.ndim != 2
        or reference.shape[0] != candidate.shape[0]
        or reference.shape[1] == 0
        or candidate.shape[1] == 0
    ):
        raise ValueError("bases must be nonempty matrices with the same row count")
    reference = weighted_orthonormalize(reference, weight)
    candidate = weighted_orthonormalize(candidate, weight)
    diagnostic_weight = np.asarray(weight, dtype=np.float64)
    if diagnostic_weight.ndim == 1:
        overlap = reference.T @ (diagnostic_weight[:, None] * candidate)
    else:
        overlap = reference.T @ diagnostic_weight @ candidate
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    singular_values[
        np.abs(1.0 - singular_values) <= 64.0 * np.finfo(float).eps
    ] = 1.0
    angles = np.arccos(singular_values)
    if include_dimension_gap:
        gap = abs(reference.shape[1] - candidate.shape[1])
        if gap:
            angles = np.concatenate(
                (angles, np.full(gap, 0.5 * np.pi, dtype=np.float64))
            )
    return np.sort(np.asarray(angles, dtype=np.float64))


def augment_weighted_subspace_with_memory(
    current_basis: object,
    memory_basis: object,
    weight: object,
    *,
    relative_independence: float = 1.0e-2,
    max_dimension: Optional[int] = None,
) -> MemoryAugmentedSubspaceResult:
    """Append only W-independent memory directions to a current subspace.

    Parameters
    ----------
    current_basis : array-like, shape (n_state, r_current)
        Nonempty current online basis. Current directions have priority.
    memory_basis : array-like, shape (n_state, r_memory)
        Nonempty directions stored from earlier load steps. Inputs are not
        modified and need not be orthogonal in the current weight.
    weight : array-like, shape (n_state,) or (n_state, n_state)
        Strictly positive diagonal or full SPD weight at the current load and
        switching iteration.
    relative_independence : float
        Retain a memory direction when the W-norm of its component orthogonal
        to the accumulated basis is at least this fraction of its original
        W-norm. It must lie in ``(0, 1]``.
    max_dimension : int, optional
        Positive cap on the augmented dimension. It cannot be smaller than
        the effective current dimension.

    Returns
    -------
    MemoryAugmentedSubspaceResult
        Newly allocated W-orthonormal basis and independence diagnostics.

    Notes
    -----
    A two-pass weighted Gram--Schmidt projection is used because the memory
    basis was normalized under an earlier load-dependent weight. The function
    performs no finite-element operation and costs
    ``O(n_state * r_memory * max_dimension)`` for a diagonal weight.
    """
    if not 0.0 < relative_independence <= 1.0:
        raise ValueError("relative_independence must lie in (0, 1]")
    current = np.asarray(current_basis, dtype=np.float64)
    memory = np.asarray(memory_basis, dtype=np.float64)
    if (
        current.ndim != 2
        or memory.ndim != 2
        or current.shape[0] != memory.shape[0]
        or current.shape[1] == 0
        or memory.shape[1] == 0
    ):
        raise ValueError("bases must be nonempty matrices with the same row count")
    if not np.isfinite(current).all() or not np.isfinite(memory).all():
        raise ValueError("current_basis and memory_basis must be finite")

    accumulated = weighted_orthonormalize(current, weight)
    current_dimension = int(accumulated.shape[1])
    if max_dimension is not None:
        if max_dimension <= 0:
            raise ValueError("max_dimension must be positive when supplied")
        if max_dimension < current_dimension:
            raise ValueError("max_dimension is smaller than the current dimension")
    diagnostic_weight = np.asarray(weight, dtype=np.float64)

    def apply_weight(vectors: np.ndarray) -> np.ndarray:
        """Apply the validated diagonal or full diagnostic weight."""
        if diagnostic_weight.ndim == 1:
            return diagnostic_weight[:, None] * vectors
        return diagnostic_weight @ vectors

    ratios = np.zeros(memory.shape[1], dtype=np.float64)
    retained = 0
    for column in range(memory.shape[1]):
        direction = memory[:, column : column + 1].copy()
        original_energy = float((direction.T @ apply_weight(direction)).item())
        original_norm = float(np.sqrt(max(0.0, original_energy)))
        if original_norm <= np.finfo(float).tiny:
            continue
        residual = direction
        for _ in range(2):
            coefficients = accumulated.T @ apply_weight(residual)
            residual = residual - accumulated @ coefficients
        residual_energy = float((residual.T @ apply_weight(residual)).item())
        residual_norm = float(np.sqrt(max(0.0, residual_energy)))
        ratios[column] = residual_norm / original_norm
        has_capacity = (
            max_dimension is None or accumulated.shape[1] < max_dimension
        )
        if ratios[column] < relative_independence or not has_capacity:
            continue
        accumulated = np.column_stack((accumulated, residual / residual_norm))
        retained += 1

    return MemoryAugmentedSubspaceResult(
        basis=np.asarray(accumulated, dtype=np.float64).copy(),
        current_dimension=current_dimension,
        memory_candidate_dimension=int(memory.shape[1]),
        retained_memory_dimension=retained,
        independence_ratios=ratios,
    )


def diagonal_patch_subspace_survival_factor(
    basis: object,
    weight_diagonal: object,
    patch_dofs: object,
) -> float:
    """Return the worst coordinate-patch survival factor on a subspace.

    The returned value is
    ``max(v in V) ||Q_patch v||_W / ||v||_W`` for the supplied subspace ``V``.
    Here ``Q_patch`` sets the selected coordinates to zero and ``W`` is a
    strictly positive diagonal weight.
    """
    diagonal = _as_finite_vector(weight_diagonal, name="weight_diagonal")
    if np.any(diagonal <= 0.0):
        raise ValueError("weight_diagonal must be strictly positive")
    vectors = np.asarray(basis, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != diagonal.size or vectors.shape[1] == 0:
        raise ValueError("basis and weight_diagonal have incompatible shapes")
    patch = np.asarray(patch_dofs, dtype=np.int64).reshape(-1)
    if patch.size == 0 or np.unique(patch).size != patch.size:
        raise ValueError("patch_dofs must contain unique indices")
    if np.any(patch < 0) or np.any(patch >= diagonal.size):
        raise ValueError("patch_dofs contains an out-of-range index")
    orthonormal = weighted_orthonormalize(vectors, diagonal)
    projected = orthonormal.copy()
    projected[patch, :] = 0.0
    reduced_gram = projected.T @ (diagonal[:, None] * projected)
    largest = float(np.max(np.linalg.eigvalsh(0.5 * (reduced_gram + reduced_gram.T))))
    return float(np.sqrt(max(0.0, min(1.0, largest))))


def diagonal_cell_weights(
    weight_diagonal: object,
    cell_to_dof: object,
) -> np.ndarray:
    """Split a global diagonal diagnostic weight into additive cell weights.

    Parameters
    ----------
    weight_diagonal : array-like, shape (n_dof,)
        Nonnegative diagonal entries of a global diagnostic weight.
    cell_to_dof : array-like, shape (n_cell, n_local_dof), dtype integer
        Connectivity in the same full-vector ordering as ``weight_diagonal``.
        Shared DOF weights are divided by their connectivity multiplicity.

    Returns
    -------
    ndarray, shape (n_cell, n_local_dof, n_local_dof), dtype float64
        Diagonal positive-semidefinite local weights ``W_K`` satisfying
        ``sum_K R_K.T @ W_K @ R_K = diag(weight_diagonal)``.
    """
    diagonal = _as_finite_vector(weight_diagonal, name="weight_diagonal")
    connectivity = np.asarray(cell_to_dof, dtype=np.int64)
    if connectivity.ndim != 2 or connectivity.shape[0] == 0:
        raise ValueError("cell_to_dof must be a nonempty two-dimensional array")
    if np.any(diagonal < 0.0):
        raise ValueError("weight_diagonal must be nonnegative")
    if np.any(connectivity < 0) or np.any(connectivity >= diagonal.size):
        raise ValueError("cell_to_dof contains an out-of-range index")
    multiplicity = np.bincount(connectivity.reshape(-1), minlength=diagonal.size)
    if np.any(multiplicity == 0):
        raise ValueError("at least one degree of freedom is absent from cell_to_dof")
    local_diagonal = diagonal[connectivity] / multiplicity[connectivity]
    weights = np.zeros(
        (connectivity.shape[0], connectivity.shape[1], connectivity.shape[1]),
        dtype=np.float64,
    )
    local_indices = np.arange(connectivity.shape[1])
    weights[:, local_indices, local_indices] = local_diagonal
    return weights


def compute_cell_energy_from_diagonal_weight(
    mode: object,
    weight_diagonal: object,
    cell_to_dof: object,
) -> np.ndarray:
    """Distribute a diagonal mode energy additively over finite elements.

    Parameters
    ----------
    mode : array-like, shape (n_dof,)
        Real or complex mode in the scalar FE degree-of-freedom order.
    weight_diagonal : array-like, shape (n_dof,)
        Nonnegative diagonal of the chosen SPD weight matrix.  Units are those
        of the weight; the returned cell energies have matching units.
    cell_to_dof : array-like, shape (n_cell, n_local_dof), dtype integer
        Scalar FE connectivity.  Shared degree-of-freedom energy is divided by
        its cell multiplicity, so cell energies sum to the global diagonal norm.

    Returns
    -------
    ndarray, shape (n_cell,), dtype float64
        Newly allocated, nonnegative cell energies in mesh cell order.
    """
    vector = np.asarray(mode).reshape(-1)
    weights = _as_finite_vector(weight_diagonal, name="weight_diagonal")
    connectivity = np.asarray(cell_to_dof, dtype=np.int64)
    if connectivity.ndim != 2 or connectivity.shape[0] == 0:
        raise ValueError("cell_to_dof must be a nonempty two-dimensional array")
    if vector.shape != weights.shape:
        raise ValueError("mode and weight_diagonal must have the same shape")
    if np.any(weights < 0.0):
        raise ValueError("weight_diagonal must be nonnegative")
    if np.any(connectivity < 0) or np.any(connectivity >= vector.size):
        raise ValueError("cell_to_dof contains an out-of-range index")

    multiplicity = np.bincount(connectivity.reshape(-1), minlength=vector.size)
    if np.any(multiplicity == 0):
        raise ValueError("at least one degree of freedom is absent from cell_to_dof")
    dof_energy = weights * np.abs(vector) ** 2
    shared_energy = dof_energy / multiplicity
    return np.sum(shared_energy[connectivity], axis=1, dtype=np.float64)


def select_bulk_cells(cell_energy: object, theta: float) -> np.ndarray:
    """Return the smallest cell mask carrying at least fraction ``theta``.

    ``cell_energy`` must be a finite, nonnegative vector in mesh cell order;
    ``theta`` is dimensionless and lies in ``(0, 1]``.
    """
    energy = _as_finite_vector(cell_energy, name="cell_energy")
    if np.any(energy < 0.0):
        raise ValueError("cell_energy must be nonnegative")
    if not 0.0 < theta <= 1.0:
        raise ValueError("theta must lie in (0, 1]")
    total = float(np.sum(energy))
    if total <= 0.0:
        raise ValueError("cell_energy must have positive total energy")

    order = np.argsort(energy)[::-1]
    count = int(np.searchsorted(np.cumsum(energy[order]), theta * total) + 1)
    selected = np.zeros(energy.size, dtype=bool)
    selected[order[:count]] = True
    return selected


__all__ = [
    "DominantModeResult",
    "FixedPointResult",
    "IncrementSubspaceResult",
    "MemoryAugmentedSubspaceResult",
    "LocalEliminationResult",
    "SlowSubspaceResult",
    "augment_weighted_subspace_with_memory",
    "compute_cell_energy_from_diagonal_weight",
    "coupled_slow_subspace_from_sweep_column",
    "apply_local_elimination_projection",
    "diagonal_cell_weights",
    "diagonal_patch_survival_factor",
    "diagonal_patch_subspace_survival_factor",
    "dominant_mode",
    "finite_difference_jacobian",
    "finite_difference_jacobian_rectangular",
    "iterate_fixed_point",
    "online_increment_subspace",
    "solve_local_nonlinear_residual",
    "select_bulk_cells",
    "spectral_slow_subspace",
    "subspace_cell_trace_indicator",
    "weighted_orthonormalize",
    "weighted_principal_angles",
]
