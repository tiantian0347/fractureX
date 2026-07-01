"""Red-green refinement bridge for fealpy3 TriangleMesh.

Wraps :class:`fracturex.mesh.AdaptiveHalfEdgeMesh2d` so a caller with a
fealpy3 ``TriangleMesh`` can do one red-green refine step and get back a
fealpy3 ``TriangleMesh`` plus P1 nodal field transfer.

Data transfer semantics
-----------------------
Red-green refinement inserts new nodes only at midpoints of existing edges
(unlike NVB/bisection, which does the same). For continuous P1 fields we
therefore average the two parent node values onto every new node. This is
exact for linear functions; smooth fields incur the usual O(h) interpolation
error.

Usage
-----

    from fracturex.adaptivity.rg_refine_bridge import refine_rg_p1

    new_mesh, new_fields = refine_rg_p1(mesh, isMarkedCell, fields={
        'd': d_array,          # (NN,)
        'r_hist': r_array,     # (NN,)
    })

``fields`` values must all be indexed by the mesh's node numbering. Extra
per-cell fields can be passed via ``cell_fields`` (children inherit parent).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from fealpy.mesh import TriangleMesh
from fracturex.mesh import AdaptiveHalfEdgeMesh2d


__all__ = ['refine_rg_p1', 'tri_to_halfedge', 'halfedge_to_tri']


def tri_to_halfedge(mesh: TriangleMesh) -> AdaptiveHalfEdgeMesh2d:
    """Build an AdaptiveHalfEdgeMesh2d from a fealpy TriangleMesh."""
    node = np.asarray(mesh.entity('node'), dtype=np.float64)
    cell = np.asarray(mesh.entity('cell'), dtype=np.int64)
    NC = cell.shape[0]
    NV = 3

    halfedge = np.zeros((NC * NV, 5), dtype=np.int64)
    for c in range(NC):
        for i in range(NV):
            he = c * NV + i
            halfedge[he, 0] = cell[c, (i + 1) % NV]
            halfedge[he, 1] = c
            halfedge[he, 2] = c * NV + (i + 1) % NV
            halfedge[he, 3] = c * NV + (i - 1) % NV
            halfedge[he, 4] = he

    edge_map: Dict[Tuple[int, int], int] = {}
    for he in range(NC * NV):
        v_to = int(halfedge[he, 0])
        v_from = int(halfedge[halfedge[he, 3], 0])
        rkey = (v_to, v_from)
        if rkey in edge_map:
            opp = edge_map.pop(rkey)
            halfedge[he, 4] = opp
            halfedge[opp, 4] = he
        else:
            edge_map[(v_from, v_to)] = he
    return AdaptiveHalfEdgeMesh2d(node, halfedge, NV=3)


def halfedge_to_tri(m: AdaptiveHalfEdgeMesh2d) -> TriangleMesh:
    """Extract a fealpy TriangleMesh from a triangle halfedge mesh."""
    node = np.asarray(m._node_view()).copy()
    cell = np.asarray(m.cell_to_node()).copy()
    return TriangleMesh(node, cell)


def refine_rg_p1(
    mesh: TriangleMesh,
    isMarkedCell,
    fields: Optional[Dict[str, np.ndarray]] = None,
    cell_fields: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[TriangleMesh, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """One red-green refine step with P1 nodal field transfer.

    Parameters
    ----------
    mesh
        Old fealpy TriangleMesh.
    isMarkedCell
        (NC,) bool mask of cells to refine.
    fields
        Mapping ``name -> (NN,)`` array of continuous P1 nodal values to
        carry over. New nodes get the midpoint average of the parent edge.
    cell_fields
        Mapping ``name -> (NC,) array`` of per-cell values. Any newly born
        cell inherits from its parent's value; if a cell is subdivided by
        rg propagation, all children inherit the parent's value.

    Returns
    -------
    new_mesh
        New fealpy TriangleMesh.
    new_fields
        Dict of transferred nodal fields (same keys as ``fields``).
    new_cell_fields
        Dict of transferred per-cell fields (same keys as ``cell_fields``).

    Notes
    -----
    Cell-parent tracking is approximate: it uses centroid-nearest matching
    from new to old cells. This is fine for smooth or piecewise-constant
    fields that don't change across parent-cell boundaries; for fields that
    depend on the split geometry you should recompute rather than transfer.
    """
    isMarkedCell = np.asarray(isMarkedCell, dtype=bool)
    fields = dict(fields or {})
    cell_fields = dict(cell_fields or {})

    m = tri_to_halfedge(mesh)
    NN_old = m.number_of_nodes()
    NC_old = m.number_of_cells()

    # Snapshot old edge -> (parent_node_0, parent_node_1) so we can look up
    # each new node's parents by geometric match.
    old_node = np.asarray(m._node_view()).copy()
    old_he = np.asarray(m._halfedge_view()).copy()
    old_hedge = np.asarray(m.hedge).copy()
    old_edge_nodes = np.stack(
        [old_he[old_hedge, 0], old_he[old_he[old_hedge, 3], 0]], axis=1)
    old_edge_mid = 0.5 * (
        old_node[old_edge_nodes[:, 0]] + old_node[old_edge_nodes[:, 1]])

    # Old cell centroids for cell-parent tracking
    old_cell_centroids = None
    if cell_fields:
        cell = np.asarray(mesh.entity('cell'))
        node = np.asarray(mesh.entity('node'))
        old_cell_centroids = node[cell].mean(axis=1)

    m.refine_triangle_rg(isMarkedCell.copy())

    node_new = np.asarray(m._node_view())
    NN_new = node_new.shape[0]

    # Transfer nodal fields via edge-midpoint match
    new_fields: Dict[str, np.ndarray] = {}
    if fields and NN_new > NN_old:
        from scipy.spatial import cKDTree

        tree = cKDTree(old_edge_mid)
        new_pts = node_new[NN_old:]
        dist, edge_idx = tree.query(new_pts, k=1)
        if float(dist.max()) > 1e-8:
            raise RuntimeError(
                f'rg parent-edge match failed (max dist {dist.max():.3e}); '
                'is the mesh actually refined via edge midpoints?')
        parents = old_edge_nodes[edge_idx]

        for name, arr in fields.items():
            arr = np.asarray(arr, dtype=np.float64)
            if arr.shape[0] != NN_old:
                raise ValueError(
                    f"field {name!r}: expected length {NN_old}, got {arr.shape[0]}")
            new_arr = np.zeros(NN_new, dtype=np.float64)
            new_arr[:NN_old] = arr
            new_arr[NN_old:] = 0.5 * (arr[parents[:, 0]] + arr[parents[:, 1]])
            new_fields[name] = new_arr
    else:
        for name, arr in fields.items():
            arr = np.asarray(arr, dtype=np.float64)
            if NN_new == NN_old:
                new_fields[name] = arr.copy()

    new_mesh = halfedge_to_tri(m)

    # Transfer per-cell fields via centroid-nearest match to old cells.
    new_cell_fields: Dict[str, np.ndarray] = {}
    if cell_fields:
        new_cell = np.asarray(new_mesh.entity('cell'))
        new_node = np.asarray(new_mesh.entity('node'))
        new_cent = new_node[new_cell].mean(axis=1)

        from scipy.spatial import cKDTree
        tree = cKDTree(old_cell_centroids)
        _, parent_cell = tree.query(new_cent, k=1)

        for name, arr in cell_fields.items():
            arr = np.asarray(arr)
            if arr.shape[0] != NC_old:
                raise ValueError(
                    f"cell field {name!r}: expected length {NC_old}, got {arr.shape[0]}")
            new_cell_fields[name] = arr[parent_cell]

    return new_mesh, new_fields, new_cell_fields
