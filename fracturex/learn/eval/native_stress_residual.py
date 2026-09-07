"""Native-triangle residuals for equilibrated-stress supervision audits.

This module evaluates the source-level residual used by the M3b paper:
cellwise divergence plus interior-edge normal-traction jumps. It also provides
an explicit Oswald reconstruction that averages the Hu--Zhang discontinuous
``P2`` displacement onto a continuous ``P2`` space before recovering stress.

The module does not train neural operators and does not identify the Oswald
field with an independently solved displacement-FEM solution. Its purpose is
to separate native-mesh source regularity from the structured-grid FD metric.
All stress value callables use Hu--Zhang Voigt order ``(xx, xy, yy)``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


StressValue = Callable[[np.ndarray, np.ndarray], np.ndarray]
StressDivergence = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class NativeResidualResult:
    """Components of a dimensionless broken-equilibrium residual.

    All squared norms have the dimensions of stress squared times area in 2-D.
    ``relative_*`` fields are divided by the native stress ``L2`` norm.
    """

    volume_squared: float
    jump_squared: float
    stress_l2_squared: float
    volume_norm: float
    jump_norm: float
    estimator_norm: float
    stress_l2_norm: float
    relative_volume: float
    relative_jump: float
    relative_estimator: float
    max_traction_jump: float
    n_cells: int
    n_interior_edges: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a newly allocated JSON-serializable representation."""
        return asdict(self)


def _as_numpy(value) -> np.ndarray:
    """Convert FEALPy/backend arrays to a NumPy ``float64`` array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _validate_stress_values(values: np.ndarray, n_cells: int, n_q: int) -> np.ndarray:
    """Validate native stress values shaped ``(n_cells, n_q, 3)``."""
    values = _as_numpy(values)
    expected = (n_cells, n_q, 3)
    if values.shape != expected:
        raise ValueError(f"stress value callable returned {values.shape}; expected {expected}")
    if not np.isfinite(values).all():
        raise ValueError("stress value callable returned non-finite values")
    return values


def _validate_divergence(values: np.ndarray, n_cells: int, n_q: int) -> np.ndarray:
    """Validate native stress divergence shaped ``(n_cells, n_q, 2)``."""
    values = _as_numpy(values)
    expected = (n_cells, n_q, 2)
    if values.shape != expected:
        raise ValueError(f"stress divergence callable returned {values.shape}; expected {expected}")
    if not np.isfinite(values).all():
        raise ValueError("stress divergence callable returned non-finite values")
    return values


def _evaluate_edge_traces(
    mesh,
    stress_value: StressValue,
    edge_indices: np.ndarray,
    edge_bcs: np.ndarray,
    *,
    side: int,
) -> np.ndarray:
    """Evaluate cell traces on one side of selected interior edges.

    Parameters
    ----------
    mesh
        FEALPy triangle mesh.
    stress_value
        Callable ``(cell_bcs, cell_indices) -> (n_cells, n_q, 3)`` in
        ``(xx, xy, yy)`` order.
    edge_indices : ndarray, shape (n_edges,), dtype int
        Global interior-edge indices.
    edge_bcs : ndarray, shape (n_q, 2), dtype float
        Reference-edge quadrature coordinates.
    side : int
        ``0`` for ``edge_to_cell[:, 0/2]`` and ``1`` for ``[:, 1/3]``.

    Returns
    -------
    ndarray, shape (n_edges, n_q, 3), dtype float64
        Newly allocated traces ordered like ``edge_indices``.
    """
    edge_to_cell = _as_numpy(mesh.edge_to_cell()).astype(np.int64)
    cell = _as_numpy(mesh.entity("cell")).astype(np.int64)
    edge = _as_numpy(mesh.entity("edge")).astype(np.int64)
    node = _as_numpy(mesh.entity("node"))
    mapped_forward = [_as_numpy(item) for item in mesh.update_bcs(edge_bcs, "cell")]
    mapped_reverse = [
        _as_numpy(item) for item in mesh.update_bcs(edge_bcs[:, ::-1], "cell")
    ]
    cell_column = 0 if side == 0 else 1
    local_edge_column = 2 if side == 0 else 3
    traces = np.empty((edge_indices.size, edge_bcs.shape[0], 3), dtype=np.float64)
    selected = edge_to_cell[edge_indices]
    for local_edge in range(3):
        local_positions = np.flatnonzero(selected[:, local_edge_column] == local_edge)
        if local_positions.size == 0:
            continue
        cells = selected[local_positions, cell_column]
        target_points = np.einsum(
            "qv,evd->eqd",
            edge_bcs,
            node[edge[edge_indices[local_positions]]],
        )
        forward_points = np.einsum(
            "qv,cvd->cqd", mapped_forward[local_edge], node[cell[cells]]
        )
        reverse_points = np.einsum(
            "qv,cvd->cqd", mapped_reverse[local_edge], node[cell[cells]]
        )
        forward_error = np.max(np.abs(forward_points - target_points), axis=(1, 2))
        reverse_error = np.max(np.abs(reverse_points - target_points), axis=(1, 2))
        is_reversed = reverse_error < forward_error
        best_error = np.minimum(forward_error, reverse_error)
        if np.any(best_error > 1.0e-10):
            raise ValueError("could not orient cell traces to global edge quadrature points")
        for reverse in (False, True):
            subgroup = np.flatnonzero(is_reversed == reverse)
            if subgroup.size == 0:
                continue
            positions = local_positions[subgroup]
            mapped_bcs = (
                mapped_reverse[local_edge] if reverse else mapped_forward[local_edge]
            )
            values = stress_value(mapped_bcs, selected[positions, cell_column])
            traces[positions] = _validate_stress_values(
                values, positions.size, edge_bcs.shape[0]
            )
    return traces


def compute_native_stress_residual(
    mesh,
    stress_value: StressValue,
    stress_divergence: StressDivergence,
    *,
    quadrature_order: int = 6,
    normalization_floor: float = 1.0e-30,
) -> NativeResidualResult:
    """Compute a native broken-divergence/normal-jump residual.

    Parameters
    ----------
    mesh
        Two-dimensional FEALPy triangle mesh. Cell/edge numbering defines all
        returned aggregate terms.
    stress_value
        Callable accepting barycentric points ``(n_q, 3)`` and cell indices
        ``(n_cells,)``; returns ``(n_cells, n_q, 3)`` in physical stress units
        and Voigt order ``(xx, xy, yy)``.
    stress_divergence
        Callable with the same inputs returning ``(n_cells, n_q, 2)`` in
        stress/length units. Body force is assumed zero for the M3b benchmark.
    quadrature_order
        Cell and edge quadrature order, at least 2.
    normalization_floor
        Positive squared-norm floor used only for division at the zero-load
        state. Dimensionally it matches stress squared times area.

    Returns
    -------
    NativeResidualResult
        Aggregate volume, jump, stress and dimensionless residual components.

    Raises
    ------
    ValueError
        If the mesh is not 2-D, the quadrature order/floor is invalid, or a
        callable violates its numerical shape/finite-value contract.

    Notes
    -----
    The estimator is
    ``sum_K h_K^2 ||div sigma||_K^2 + sum_e h_e ||[sigma n]||_e^2``.
    Here ``h_K`` is the longest edge of cell ``K`` and ``h_e`` is edge length.
    """
    if int(mesh.geo_dimension()) != 2:
        raise ValueError("native stress residual currently requires a 2-D triangle mesh")
    if quadrature_order < 2:
        raise ValueError("quadrature_order must be at least 2")
    if normalization_floor <= 0.0:
        raise ValueError("normalization_floor must be positive")

    n_cells = int(mesh.number_of_cells())
    cell_indices = np.arange(n_cells, dtype=np.int64)
    cell_qf = mesh.quadrature_formula(quadrature_order, "cell")
    cell_bcs_raw, cell_weights_raw = cell_qf.get_quadrature_points_and_weights()
    cell_bcs = _as_numpy(cell_bcs_raw)
    cell_weights = _as_numpy(cell_weights_raw)
    cell_area = _as_numpy(mesh.entity_measure("cell"))
    cell_to_edge = _as_numpy(mesh.cell_to_edge()).astype(np.int64)
    edge_length = _as_numpy(mesh.entity_measure("edge"))
    cell_diameter = edge_length[cell_to_edge].max(axis=1)

    sigma = _validate_stress_values(
        stress_value(cell_bcs, cell_indices), n_cells, cell_bcs.shape[0]
    )
    divergence = _validate_divergence(
        stress_divergence(cell_bcs, cell_indices), n_cells, cell_bcs.shape[0]
    )
    stress_frobenius_sq = sigma[..., 0] ** 2 + 2.0 * sigma[..., 1] ** 2 + sigma[..., 2] ** 2
    divergence_sq = np.sum(divergence * divergence, axis=-1)
    stress_cell_integral = cell_area * np.einsum("q,cq->c", cell_weights, stress_frobenius_sq)
    divergence_cell_integral = cell_area * np.einsum("q,cq->c", cell_weights, divergence_sq)
    stress_l2_squared = float(stress_cell_integral.sum())
    volume_squared = float(np.sum(cell_diameter * cell_diameter * divergence_cell_integral))

    edge_to_cell = _as_numpy(mesh.edge_to_cell()).astype(np.int64)
    interior_edges = np.flatnonzero(edge_to_cell[:, 0] != edge_to_cell[:, 1]).astype(np.int64)
    edge_qf = mesh.quadrature_formula(quadrature_order, "edge")
    edge_bcs_raw, edge_weights_raw = edge_qf.get_quadrature_points_and_weights()
    edge_bcs = _as_numpy(edge_bcs_raw)
    edge_weights = _as_numpy(edge_weights_raw)
    left = _evaluate_edge_traces(mesh, stress_value, interior_edges, edge_bcs, side=0)
    right = _evaluate_edge_traces(mesh, stress_value, interior_edges, edge_bcs, side=1)
    normal = _as_numpy(mesh.edge_unit_normal(index=interior_edges))

    def traction(values: np.ndarray) -> np.ndarray:
        tx = values[..., 0] * normal[:, None, 0] + values[..., 1] * normal[:, None, 1]
        ty = values[..., 1] * normal[:, None, 0] + values[..., 2] * normal[:, None, 1]
        return np.stack([tx, ty], axis=-1)

    traction_jump = traction(left) - traction(right)
    traction_jump_sq = np.sum(traction_jump * traction_jump, axis=-1)
    jump_edge_integral = edge_length[interior_edges] * np.einsum(
        "q,eq->e", edge_weights, traction_jump_sq
    )
    jump_squared = float(np.sum(edge_length[interior_edges] * jump_edge_integral))
    max_traction_jump = float(np.max(np.linalg.norm(traction_jump, axis=-1), initial=0.0))

    volume_norm = float(np.sqrt(max(volume_squared, 0.0)))
    jump_norm = float(np.sqrt(max(jump_squared, 0.0)))
    estimator_norm = float(np.sqrt(max(volume_squared + jump_squared, 0.0)))
    stress_l2_norm = float(np.sqrt(max(stress_l2_squared, 0.0)))
    denominator = max(stress_l2_norm, float(np.sqrt(normalization_floor)))
    return NativeResidualResult(
        volume_squared=volume_squared,
        jump_squared=jump_squared,
        stress_l2_squared=stress_l2_squared,
        volume_norm=volume_norm,
        jump_norm=jump_norm,
        estimator_norm=estimator_norm,
        stress_l2_norm=stress_l2_norm,
        relative_volume=volume_norm / denominator,
        relative_jump=jump_norm / denominator,
        relative_estimator=estimator_norm / denominator,
        max_traction_jump=max_traction_jump,
        n_cells=n_cells,
        n_interior_edges=int(interior_edges.size),
    )


def compute_native_relative_l2_difference(
    mesh,
    stress_value: StressValue,
    reference_value: StressValue,
    *,
    quadrature_order: int = 6,
    normalization_floor: float = 1.0e-30,
) -> float:
    """Compute ``||sigma-reference||_L2 / ||reference||_L2`` on the mesh.

    Both callables follow :func:`compute_native_stress_residual` and return
    physical stresses in ``(xx, xy, yy)`` order. Symmetric-tensor Frobenius
    weights ``(1, 2, 1)`` are applied. Inputs are not modified.
    """
    n_cells = int(mesh.number_of_cells())
    indices = np.arange(n_cells, dtype=np.int64)
    qf = mesh.quadrature_formula(quadrature_order, "cell")
    bcs_raw, weights_raw = qf.get_quadrature_points_and_weights()
    bcs = _as_numpy(bcs_raw)
    weights = _as_numpy(weights_raw)
    area = _as_numpy(mesh.entity_measure("cell"))
    value = _validate_stress_values(stress_value(bcs, indices), n_cells, bcs.shape[0])
    reference = _validate_stress_values(reference_value(bcs, indices), n_cells, bcs.shape[0])
    component_weights = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    difference_sq = np.sum((value - reference) ** 2 * component_weights, axis=-1)
    reference_sq = np.sum(reference ** 2 * component_weights, axis=-1)
    numerator = float(np.sum(area * np.einsum("q,cq->c", weights, difference_sq)))
    denominator = float(np.sum(area * np.einsum("q,cq->c", weights, reference_sq)))
    return float(np.sqrt(max(numerator, 0.0) / max(denominator, normalization_floor)))


def make_huzhang_stress_callables(discr) -> tuple[StressValue, StressDivergence]:
    """Create native value/divergence callables for ``discr.state.sigma``.

    The FEALPy Hu--Zhang ``div_value`` method in the pinned version dispatches
    through an unimplemented generic gradient. This wrapper contracts the
    implemented ``div_basis`` with cell-local DOFs directly.
    """
    space = discr.space_sigma
    dofs = _as_numpy(discr.state.sigma)
    cell_to_dof = _as_numpy(space.dof.cell_to_dof()).astype(np.int64)

    def value(bcs: np.ndarray, cell_indices: np.ndarray) -> np.ndarray:
        # Contract explicitly: the pinned FEALPy ``space.value`` path routes
        # through ``entity_to_dof`` which ignores a subset index for this HZ
        # dof manager, producing an all-cell/local-dof shape mismatch.
        basis = _as_numpy(space.basis(bcs, index=cell_indices))
        local_dofs = dofs[cell_to_dof[cell_indices]]
        return np.einsum("cqld,cl->cqd", basis, local_dofs)

    def divergence(bcs: np.ndarray, cell_indices: np.ndarray) -> np.ndarray:
        basis = _as_numpy(space.div_basis(bcs, index=cell_indices))
        local_dofs = dofs[cell_to_dof[cell_indices]]
        return np.einsum("cqld,cl->cqd", basis, local_dofs)

    return value, divergence


def _p2_hessian_basis(mesh) -> np.ndarray:
    """Return physical Hessians of triangle ``P2`` basis functions.

    Returns a new ``(n_cells, 6, 2, 2)`` float64 array in FEALPy's local order
    ``[v0, e01, e02, v1, e12, v2]``. Hessians are constant per affine cell.
    """
    grad_lambda = _as_numpy(mesh.grad_lambda())
    g0, g1, g2 = grad_lambda[:, 0], grad_lambda[:, 1], grad_lambda[:, 2]

    def outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.einsum("ci,cj->cij", a, b)

    hessian = np.empty((grad_lambda.shape[0], 6, 2, 2), dtype=np.float64)
    hessian[:, 0] = 4.0 * outer(g0, g0)
    hessian[:, 1] = 4.0 * (outer(g0, g1) + outer(g1, g0))
    hessian[:, 2] = 4.0 * (outer(g0, g2) + outer(g2, g0))
    hessian[:, 3] = 4.0 * outer(g1, g1)
    hessian[:, 4] = 4.0 * (outer(g1, g2) + outer(g2, g1))
    hessian[:, 5] = 4.0 * outer(g2, g2)
    return hessian


def oswald_average_dg_displacement(discr) -> np.ndarray:
    """Average a vector ``P2``-DG displacement onto continuous ``P2`` DOFs.

    Parameters
    ----------
    discr
        Built :class:`HuZhangDiscretization` with a loaded ``state.u``. The
        scalar displacement base must be triangle ``P2`` DG.

    Returns
    -------
    ndarray, shape (n_cg_dofs, 2), dtype float64
        Continuous interpolation-point values in FEALPy global ``P2`` order.
        A new array is returned; the loaded state is unchanged.

    Raises
    ------
    ValueError
        If the displacement order, vector size or tensor layout is unsupported.
    """
    from fealpy.functionspace import LagrangeFESpace

    dg_space = discr.space_u.scalar_space
    if int(dg_space.p) != 2:
        raise ValueError(f"Oswald recovery requires P2 displacement; got p={dg_space.p}")
    mesh = discr.mesh
    cg_space = LagrangeFESpace(mesh, p=2, ctype="C")
    dg_scalar_dofs = int(dg_space.number_of_global_dofs())
    flat = _as_numpy(discr.state.u).reshape(-1)
    if flat.size != 2 * dg_scalar_dofs:
        raise ValueError(
            f"displacement has {flat.size} values; expected {2 * dg_scalar_dofs}"
        )
    if bool(discr.space_u.dof_priority):
        dg_vector = flat.reshape(2, dg_scalar_dofs).T
    else:
        dg_vector = flat.reshape(dg_scalar_dofs, 2)
    dg_cell_to_dof = _as_numpy(dg_space.cell_to_dof()).astype(np.int64)
    cg_cell_to_dof = _as_numpy(cg_space.cell_to_dof()).astype(np.int64)
    local_values = dg_vector[dg_cell_to_dof]
    n_cg_dofs = int(cg_space.number_of_global_dofs())
    averaged = np.zeros((n_cg_dofs, 2), dtype=np.float64)
    counts = np.zeros(n_cg_dofs, dtype=np.float64)
    np.add.at(averaged, cg_cell_to_dof.reshape(-1), local_values.reshape(-1, 2))
    np.add.at(counts, cg_cell_to_dof.reshape(-1), 1.0)
    if np.any(counts == 0.0):
        raise ValueError("continuous P2 space contains interpolation points with no DG contributors")
    return averaged / counts[:, None]


class OswaldRecoveredStress:
    """Evaluate ``g(d) C eps(u)`` from an Oswald-averaged continuous ``P2`` field.

    The class owns newly allocated continuous displacement coefficients and
    read-only references to the loaded mesh/damage state. Stress values use
    physical units and Hu--Zhang Voigt order ``(xx, xy, yy)``.
    """

    def __init__(
        self,
        discr,
        *,
        young_modulus: float,
        poisson_ratio: float,
        plane: str = "strain",
        residual_stiffness: float = 1.0e-6,
    ) -> None:
        """Build the continuous recovery for one loaded checkpoint.

        ``young_modulus`` is stress-valued, ``poisson_ratio`` and
        ``residual_stiffness`` are dimensionless, and ``plane`` is ``strain``
        or ``stress``. The loaded FE state is not modified.
        """
        from fealpy.functionspace import LagrangeFESpace
        from fracturex.learn.stress_recovery import plane_strain_C, plane_stress_C

        if plane not in {"strain", "stress"}:
            raise ValueError("plane must be 'strain' or 'stress'")
        if not (-1.0 < poisson_ratio < 0.5):
            raise ValueError("poisson_ratio must lie in (-1, 0.5)")
        if not (0.0 < residual_stiffness <= 1.0):
            raise ValueError("residual_stiffness must lie in (0, 1]")
        self.discr = discr
        self.mesh = discr.mesh
        self.cg_space = LagrangeFESpace(self.mesh, p=2, ctype="C")
        self.cg_cell_to_dof = _as_numpy(self.cg_space.cell_to_dof()).astype(np.int64)
        self.displacement = oswald_average_dg_displacement(discr)
        matrix = plane_strain_C(young_modulus, poisson_ratio) if plane == "strain" else plane_stress_C(young_modulus, poisson_ratio)
        self.constitutive = _as_numpy(matrix)
        self.residual_stiffness = float(residual_stiffness)
        self.hessian_basis = _p2_hessian_basis(self.mesh)

    def _kinematics(
        self,
        bcs: np.ndarray,
        cell_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return base stress, base divergence, damage and damage gradient."""
        cell_indices = np.asarray(cell_indices, dtype=np.int64)
        grad_basis = _as_numpy(self.cg_space.grad_basis(bcs, index=cell_indices))
        local_u = self.displacement[self.cg_cell_to_dof[cell_indices]]
        grad_u = np.einsum("cqld,clv->cqvd", grad_basis, local_u)
        strain = np.stack(
            [grad_u[..., 0, 0], grad_u[..., 1, 1], grad_u[..., 0, 1] + grad_u[..., 1, 0]],
            axis=-1,
        )
        base_stress = np.einsum("ij,cqj->cqi", self.constitutive, strain)

        hessian_u = np.einsum(
            "clab,clv->cvab",
            self.hessian_basis[cell_indices],
            local_u,
        )
        strain_dx = np.stack(
            [
                hessian_u[:, 0, 0, 0],
                hessian_u[:, 1, 1, 0],
                hessian_u[:, 0, 1, 0] + hessian_u[:, 1, 0, 0],
            ],
            axis=-1,
        )
        strain_dy = np.stack(
            [
                hessian_u[:, 0, 0, 1],
                hessian_u[:, 1, 1, 1],
                hessian_u[:, 0, 1, 1] + hessian_u[:, 1, 0, 1],
            ],
            axis=-1,
        )
        stress_dx = np.einsum("ij,cj->ci", self.constitutive, strain_dx)
        stress_dy = np.einsum("ij,cj->ci", self.constitutive, strain_dy)
        base_divergence = np.stack(
            [stress_dx[:, 0] + stress_dy[:, 2], stress_dx[:, 2] + stress_dy[:, 1]],
            axis=-1,
        )

        damage_space = self.discr.space_d
        damage_dofs = _as_numpy(self.discr.state.d)
        damage_cell_to_dof = _as_numpy(damage_space.cell_to_dof()).astype(np.int64)
        damage_basis = _as_numpy(damage_space.basis(bcs, index=cell_indices))
        if damage_basis.shape[0] == 1 and cell_indices.size != 1:
            damage_basis = np.broadcast_to(
                damage_basis, (cell_indices.size,) + damage_basis.shape[1:]
            )
        local_damage = damage_dofs[damage_cell_to_dof[cell_indices]]
        damage = np.einsum("cql,cl->cq", damage_basis, local_damage)
        damage_grad_basis = _as_numpy(damage_space.grad_basis(bcs, index=cell_indices))
        damage_gradient = np.einsum("cqld,cl->cqd", damage_grad_basis, local_damage)
        return base_stress, base_divergence, damage, damage_gradient

    def value(self, bcs: np.ndarray, cell_indices: np.ndarray) -> np.ndarray:
        """Return stress ``(n_cells,n_q,3)`` in ``(xx,xy,yy)`` order."""
        base_stress, _, damage, _ = self._kinematics(bcs, cell_indices)
        k = self.residual_stiffness
        degradation = (1.0 - k) * (1.0 - damage) ** 2 + k
        stress = base_stress * degradation[..., None]
        return stress[..., [0, 2, 1]]

    def divergence(self, bcs: np.ndarray, cell_indices: np.ndarray) -> np.ndarray:
        """Return analytic broken divergence ``(n_cells,n_q,2)``."""
        base_stress, base_divergence, damage, damage_gradient = self._kinematics(
            bcs, cell_indices
        )
        k = self.residual_stiffness
        degradation = (1.0 - k) * (1.0 - damage) ** 2 + k
        degradation_gradient = (
            -2.0 * (1.0 - k) * (1.0 - damage)[..., None] * damage_gradient
        )
        sxx = base_stress[..., 0]
        syy = base_stress[..., 1]
        sxy = base_stress[..., 2]
        div_x = (
            degradation * base_divergence[:, None, 0]
            + degradation_gradient[..., 0] * sxx
            + degradation_gradient[..., 1] * sxy
        )
        div_y = (
            degradation * base_divergence[:, None, 1]
            + degradation_gradient[..., 0] * sxy
            + degradation_gradient[..., 1] * syy
        )
        return np.stack([div_x, div_y], axis=-1)
