"""Layered tests for quad red-blue refinement + parent->child interpolation.

Covers ``AdaptiveHalfEdgeMesh2d.refine_quad_rb`` and the inheritance
interpolators ``inherit_nodal_data`` / ``inherit_cell_data``.

Layers
------
unit      : uniform-refine entity counts (NN'=NN+NE+NC, NC'=4NC, NE'=2NE+4NC);
            interpolation exactness on linear/bilinear fields.
smoke     : array-length sync (color/level) after a partial refine.
regression: full conformity (opp involution, next/prev inverse, all leaf cells
            are quads, no hanging nodes) and total-area invariance for the
            partial-refine transition closure.
"""
import numpy as np
import pytest

from fracturex.mesh.halfedge_mesh import AdaptiveHalfEdgeMesh2d as M


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def quad_box(nx, ny):
    """Structured nx*ny quad mesh on the unit square as a halfedge mesh."""
    N = (nx + 1) * (ny + 1)
    X, Y = np.mgrid[0:1:complex(0, nx + 1), 0:1:complex(0, ny + 1)]
    node = np.zeros((N, 2)); node[:, 0] = X.flat; node[:, 1] = Y.flat
    idx = np.arange(N).reshape(nx + 1, ny + 1)
    NC = nx * ny; cell = np.zeros((NC, 4), dtype=int)
    cell[:, 0] = idx[:-1, :-1].flat; cell[:, 1] = idx[1:, :-1].flat
    cell[:, 2] = idx[1:, 1:].flat;  cell[:, 3] = idx[:-1, 1:].flat
    return M._from_quad_cells(node, cell)


def total_area(m):
    """Signed shoelace area summed over interior leaf cells."""
    he = m._halfedge_view(); node = m._node_view()
    e0 = node[he[he[:, 3], 0]]; e1 = node[he[:, 0]]
    interior = he[:, 1] >= m.cellstart
    cross = e0[interior, 0] * e1[interior, 1] - e0[interior, 1] * e1[interior, 0]
    return 0.5 * cross.sum()


def assert_conforming(m):
    he = m._halfedge_view(); NHE = len(he)
    idx = np.arange(NHE)
    # opposite is an involution
    assert (he[he[:, 4], 4] == idx).all()
    # next/prev are inverse
    assert (he[he[:, 2], 3] == idx).all()
    # every leaf cell is a quad
    NV = m.number_of_vertices_of_all_cells()
    assert (NV[m.cellstart:] == 4).all()
    # no hanging nodes on interior edges (boundary halfedges are self-opposite)
    node = m._node_view()
    interior = he[:, 4] != idx
    a_to = node[he[interior, 0]]
    a_fr = node[he[he[interior, 3], 0]]
    o = he[interior, 4]
    o_to = node[he[o, 0]]
    o_fr = node[he[he[o, 3], 0]]
    assert np.allclose(a_to, o_fr) and np.allclose(a_fr, o_to)


# --------------------------------------------------------------------------
# unit: uniform-refine entity counts
# --------------------------------------------------------------------------

@pytest.mark.parametrize("nx,ny", [(1, 1), (2, 2), (2, 3)])
def test_uniform_refine_counts(nx, ny):
    m = quad_box(nx, ny)
    NN, NE, NC = m.number_of_nodes(), m.number_of_edges(), m.number_of_cells()
    m.refine_quad_rb()
    assert m.number_of_cells() == 4 * NC
    assert m.number_of_nodes() == NN + NE + NC
    assert m.number_of_edges() == 2 * NE + 4 * NC
    assert_conforming(m)


# --------------------------------------------------------------------------
# smoke: array-length sync after a partial refine
# --------------------------------------------------------------------------

def test_array_sync_partial_refine():
    m = quad_box(4, 4)
    isM = np.zeros(m.number_of_cells(), bool); isM[0] = True
    m.refine_quad_rb(isM)
    NHE = m.number_of_halfedges()
    assert len(m.halfedgedata['color']) == NHE
    assert len(m.halfedgedata['colorlevel']) == NHE
    assert len(m.halfedgedata['level']) == NHE
    assert len(m.celldata['level']) == m.number_of_all_cells()


# --------------------------------------------------------------------------
# regression: conformity + area invariance under transition closure
# --------------------------------------------------------------------------

@pytest.mark.parametrize("marks", [[0], [0, 5, 10], "diag"])
def test_partial_refine_conforming(marks):
    m = quad_box(4, 4)
    A0 = total_area(m)
    isM = np.zeros(m.number_of_cells(), bool)
    if marks == "diag":
        c = m.cell_barycenter()
        isM[np.abs(c[:, 0] - c[:, 1]) < 0.13] = True
    else:
        isM[marks] = True
    m.refine_quad_rb(isM)
    assert_conforming(m)
    assert abs(total_area(m) - A0) < 1e-12


def test_nested_refine_stays_conforming():
    m = quad_box(4, 4)
    A0 = total_area(m)
    for _ in range(2):
        isM = np.zeros(m.number_of_cells(), bool); isM[0] = True
        m.refine_quad_rb(isM)
        assert_conforming(m)
    assert abs(total_area(m) - A0) < 1e-12


# --------------------------------------------------------------------------
# regression: coarsen is the inverse of refine (round-trip restores mesh)
# --------------------------------------------------------------------------

def _coarsen_all_finer(m):
    """Mark every previously-refined (clevel>0) cell and coarsen it."""
    clevel = np.asarray(m.celldata['level'][:])
    isC = np.zeros(m.number_of_cells(), bool)
    isC[:] = clevel[m.cellstart:] > 0
    m.coarsen_quad_rb(isC)


@pytest.mark.parametrize("nx,ny,marks", [
    (1, 1, "all"), (2, 2, "all"), (3, 3, "all"),
    (4, 4, [0]), (4, 4, [0, 5, 10]),
])
def test_refine_coarsen_roundtrip(nx, ny, marks):
    m = quad_box(nx, ny)
    c0 = (m.number_of_cells(), m.number_of_nodes(), m.number_of_edges())
    A0 = total_area(m)
    isM = np.zeros(m.number_of_cells(), bool)
    if marks == "all":
        isM[:] = True
    else:
        isM[marks] = True
    m.refine_quad_rb(isM)
    _coarsen_all_finer(m)
    assert_conforming(m)
    c1 = (m.number_of_cells(), m.number_of_nodes(), m.number_of_edges())
    assert c1 == c0
    assert abs(total_area(m) - A0) < 1e-12


def test_nested_refine_coarsen_roundtrip():
    m = quad_box(4, 4)
    c0 = (m.number_of_cells(), m.number_of_nodes(), m.number_of_edges())
    A0 = total_area(m)
    for _ in range(2):
        isM = np.zeros(m.number_of_cells(), bool); isM[0] = True
        m.refine_quad_rb(isM)
    for _ in range(2):
        clevel = np.asarray(m.celldata['level'][:])
        mx = clevel[m.cellstart:].max()
        isC = np.zeros(m.number_of_cells(), bool)
        isC[:] = clevel[m.cellstart:] == mx
        m.coarsen_quad_rb(isC)
        assert_conforming(m)
    c1 = (m.number_of_cells(), m.number_of_nodes(), m.number_of_edges())
    assert c1 == c0
    assert abs(total_area(m) - A0) < 1e-12


def test_coarsen_then_refine_stays_conforming():
    m = quad_box(2, 2)
    m.refine_quad_rb()
    m.coarsen_quad_rb(np.ones(m.number_of_cells(), bool))
    isM = np.zeros(m.number_of_cells(), bool); isM[0] = True
    m.refine_quad_rb(isM)
    assert_conforming(m)


# --------------------------------------------------------------------------
# unit: interpolation exactness (parent->child inheritance)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("nx,ny,marks", [
    (2, 2, "all"), (2, 2, [0]), (4, 4, [0]), (4, 4, [0, 5, 10]),
])
def test_inherit_nodal_exact(nx, ny, marks):
    m = quad_box(nx, ny)
    node_old = m._node_view().copy()
    cell_old = m.cell_to_node().copy()
    f_lin = 2.0 * node_old[:, 0] - 3.0 * node_old[:, 1] + 1.0
    f_bil = node_old[:, 0] * node_old[:, 1] + 0.5 * node_old[:, 0]

    isM = np.zeros(cell_old.shape[0], bool)
    if marks == "all":
        isM[:] = True
    else:
        isM[marks] = True
    m.refine_quad_rb(isM)

    out_lin = m.inherit_nodal_data(f_lin, node_old, cell_old)
    out_bil = m.inherit_nodal_data(f_bil, node_old, cell_old)
    pts = m._node_view()
    exact_lin = 2.0 * pts[:, 0] - 3.0 * pts[:, 1] + 1.0
    exact_bil = pts[:, 0] * pts[:, 1] + 0.5 * pts[:, 0]

    assert out_lin.shape == (m.number_of_nodes(),)
    assert np.abs(out_lin - exact_lin).max() < 1e-10
    assert np.abs(out_bil - exact_bil).max() < 1e-9
    # old node values preserved exactly (indices are stable)
    assert np.abs(out_lin[:node_old.shape[0]] - f_lin).max() < 1e-12


@pytest.mark.parametrize("nq", [1, 4])
def test_inherit_cell_data_copies_parents(nq):
    m = quad_box(4, 4)
    node_old = m._node_view().copy()
    cell_old = m.cell_to_node().copy()
    NC0 = cell_old.shape[0]
    cd = np.outer(np.arange(NC0) + 1.0, np.arange(1, nq + 1))  # (NC0, nq)

    isM = np.zeros(NC0, bool); isM[[0, 5, 10]] = True
    m.refine_quad_rb(isM)

    out = m.inherit_cell_data(cd, node_old, cell_old)
    assert out.shape == (m.number_of_cells(), nq)
    # every child column-0 value is one of the parent column-0 values
    assert np.isin(out[:, 0], cd[:, 0]).all()
    assert np.isfinite(out).all()
