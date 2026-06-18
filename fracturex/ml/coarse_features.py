"""Dimensionless, local, mesh-resolution-invariant features per coarse (P1) dof.

The auxiliary coarse space is P1 (``LagrangeFESpace(mesh, 1)``), whose global dofs
coincide with mesh vertices, so "per coarse dof" == "per node". For the learned
coarse-space enrichment (D13 §3.1) every node carries the feature vector

    phi = ( d,
            ||grad d|| * l0,          # dimensionless gradient
            h / l0,                   # dimensionless local mesh size
            log( g(d) / g_bar ),      # log degradation vs 1-ring mean
            g(d) / g_max )            # degradation vs global max

All entries are dimensionless and use only local / relative quantities; absolute
coordinates, absolute ``h`` and dof indices are deliberately excluded so the model
can generalize across meshes and length scales (validated, not proven — see plan
§5.3 and the scale lemma in §4.6F).

This module imports no solver and no torch: it is FEALPy multi-backend
(``backend_manager``) + mesh/function evaluation, callable both offline (feature
dump over D12 checkpoints) and online (once per staggered setup, inside the
inference adapter). All array ops go through ``bm`` so the same code runs on the
numpy / torch / jax backends; no in-place fancy assignment is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from fealpy.backend import backend_manager as bm

# Order is the contract between feature extraction, training and inference.
FEATURE_NAMES: Sequence[str] = (
    "d",                # damage at node, in [0, 1]
    "gradd_l0",         # ||grad d|| * l0
    "h_l0",             # local mesh size / l0
    "log_g_over_gbar",  # log( g(d) / 1-ring mean g )
    "g_over_gmax",      # g(d) / max_x g(d)
)
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class CoarseFeatures:
    """Per-node feature block plus the metadata needed to map back / debug.

    Attributes:
        phi: ``(NN, N_FEATURES)`` float64 feature matrix, columns per FEATURE_NAMES.
        node: ``(NN, gdim)`` node coordinates (NOT a feature; kept for plotting/debug).
        d: ``(NN,)`` raw node damage.
        g: ``(NN,)`` raw node degradation g(d) (with model floor applied).
        l0: phase-field length scale used for non-dimensionalization.
    """

    phi: Any
    node: Any
    d: Any
    g: Any
    l0: float

    @property
    def feature_names(self) -> Sequence[str]:
        return FEATURE_NAMES


def _scatter_node_sum_count(values_per_corner, cell, nnode: int):
    """Scatter per-(cell,corner) values onto nodes; return (sum, count) per node.

    Backend-agnostic via ``bm.index_add`` (functional scatter-add, the multi-backend
    equivalent of ``np.add.at``). ``values_per_corner`` is ``(NC, nv)``; corner ``a``
    of cell ``c`` contributes to node ``cell[c, a]``.
    """
    nv = cell.shape[1]
    acc = bm.zeros(nnode, dtype=bm.float64)
    cnt = bm.zeros(nnode, dtype=bm.float64)
    ones = bm.ones(cell.shape[0], dtype=bm.float64)
    for a in range(nv):
        idx = cell[:, a]
        acc = bm.index_add(acc, idx, values_per_corner[:, a])
        cnt = bm.index_add(cnt, idx, ones)
    return acc, cnt


def _scatter_cell_to_node_mean(cell_val, cell, nnode: int):
    """Average a per-cell scalar onto nodes (each node = mean of incident cells).

    Args:
        cell_val: ``(NC,)`` per-cell scalar.
        cell: ``(NC, nv)`` connectivity.
        nnode: number of nodes.
    Returns:
        ``(NN,)`` per-node average. Isolated nodes (no incident cell) get 0.
    """
    nv = cell.shape[1]
    per_corner = bm.broadcast_to(cell_val.reshape(-1, 1), (cell.shape[0], nv))
    acc, cnt = _scatter_node_sum_count(per_corner, cell, nnode)
    return acc / bm.maximum(cnt, 1.0)


def _cell_diameters(node, cell):
    """Per-cell diameter = longest edge length (works for simplex cells in 2D/3D).

    Args:
        node: ``(NN, gdim)`` coordinates.
        cell: ``(NC, nv)`` connectivity (nv = gdim+1 for simplices).
    Returns:
        ``(NC,)`` longest pairwise vertex distance per cell.
    """
    nv = cell.shape[1]
    coords = node[cell]  # (NC, nv, gdim)
    diam = bm.zeros(cell.shape[0], dtype=bm.float64)
    for a in range(nv):
        for b in range(a + 1, nv):
            e = coords[:, a, :] - coords[:, b, :]
            diam = bm.maximum(diam, bm.sqrt(bm.sum(e * e, axis=-1)))
    return diam


def _eval_node_values(mesh, fe_fun, gdim: int):
    """Exact nodal values of a continuous FE function (any order) via corner eval.

    Evaluating at the reference-simplex corners gives ``(NC, nv)`` cell-corner values;
    for a continuous function the value at a shared vertex agrees across incident
    cells, so scatter-averaging recovers the exact nodal field. This avoids assuming
    ``fe_fun[:]`` are node values (false for P2+ damage).

    Args:
        mesh: FEALPy mesh.
        fe_fun: callable ``fe_fun(bcs) -> (NC, nb)`` FE function (e.g. ``state.d``).
        gdim: geometric dimension.
    Returns:
        ``(NN,)`` nodal values.
    """
    cell = mesh.entity("cell")
    nnode = int(mesh.number_of_nodes())
    corners = bm.eye(gdim + 1, dtype=bm.float64)  # (nv, gdim+1)
    vals = bm.asarray(fe_fun(corners))            # (NC, nv)
    acc, cnt = _scatter_node_sum_count(vals, cell, nnode)
    return acc / bm.maximum(cnt, 1.0)


def _node_grad_d_magnitude(mesh, state, gdim: int):
    """``||grad d||`` per node, by evaluating the FE gradient at cell barycenters
    (P1 gradient is cell-constant) and averaging incident cells onto nodes.

    Args:
        mesh: FEALPy mesh (provides cells/nodes).
        state: carries ``state.d`` as an FE Function with ``grad_value(bcs, index)``.
        gdim: geometric dimension.
    Returns:
        ``(NN,)`` nodal ``||grad d||``.
    """
    cell = mesh.entity("cell")
    nnode = int(mesh.number_of_nodes())
    bary = bm.full((1, gdim + 1), 1.0 / (gdim + 1), dtype=bm.float64)
    grad = bm.asarray(state.d.grad_value(bary))  # (NC, 1, gdim)
    if grad.ndim == 3:
        grad = grad[:, 0, :]
    gmag_cell = bm.sqrt(bm.sum(grad * grad, axis=-1))  # (NC,)
    return _scatter_cell_to_node_mean(gmag_cell, cell, nnode)


def extract_coarse_features(
    mesh,
    damage,
    state,
    *,
    l0: Optional[float] = None,
    eps_g: float = 1e-10,
) -> CoarseFeatures:
    """Build the per-node dimensionless feature block for a frozen damage field.

    Args:
        mesh: FEALPy mesh; P1 coarse dofs == its nodes.
        damage: damage model exposing ``degradation(d) -> g(d)`` and ``l0``.
        state: carries ``state.d`` (P1 FE Function; ``state.d[:]`` are node values).
        l0: override length scale; defaults to ``damage.l0``.
        eps_g: numerical floor for g before taking logs (matches model eps).
    Returns:
        :class:`CoarseFeatures` with ``phi`` of shape ``(NN, N_FEATURES)``.
    """
    if l0 is None:
        l0 = float(getattr(damage, "l0"))
    l0 = float(l0)
    if not (l0 > 0.0):
        raise ValueError(f"l0 must be positive, got {l0!r}")

    gdim = int(mesh.geo_dimension())
    node = bm.astype(bm.asarray(mesh.entity("node")), bm.float64)
    cell = mesh.entity("cell")
    nnode = int(mesh.number_of_nodes())

    # Damage may be P1 or P2+; evaluate the FE function at vertices rather than
    # reading raw dofs (P2 dofs include edge midpoints, not just nodes).
    d_node = bm.clip(_eval_node_values(mesh, state.d, gdim), 0.0, 1.0)

    g_node = bm.reshape(bm.asarray(damage.degradation(d_node)), (-1,))
    g_node = bm.maximum(g_node, eps_g)
    g_max = max(float(bm.max(g_node)) if g_node.shape[0] else 1.0, eps_g)

    gradd_node = _node_grad_d_magnitude(mesh, state, gdim)

    diam_cell = _cell_diameters(node, cell)
    h_node = _scatter_cell_to_node_mean(diam_cell, cell, nnode)

    # Neighborhood mean of g for the log-ratio feature: average the per-cell mean g
    # onto nodes (vectorized 1-ring surrogate; no per-node Python loop so it scales
    # to millions of dofs).
    g_cell_mean = bm.mean(g_node[cell], axis=1)  # (NC,)
    g_bar = bm.maximum(_scatter_cell_to_node_mean(g_cell_mean, cell, nnode), eps_g)

    phi = bm.stack(
        [
            d_node,
            gradd_node * l0,
            h_node / l0,
            bm.log(g_node / g_bar),
            g_node / g_max,
        ],
        axis=1,
    )

    return CoarseFeatures(phi=phi, node=node, d=d_node, g=g_node, l0=l0)
