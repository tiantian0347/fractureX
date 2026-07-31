"""
AdaptiveHalfEdgeMesh2d
======================

Ports the fealpy2.0 half-edge mesh adaptive refinement/coarsening to fealpy3.

Migrated features:
  Stage 1+2 (triangle adaptive):
    - mark_halfedge (poly / rg / nvb / quad)
    - refine_halfedge / coarsen_halfedge
    - _refine_tri_cell_ / _coarsen_triangle_cell_
    - refine_triangle_rg / coarsen_triangle_rg   (red-green)
    - refine_triangle_nvb / coarsen_triangle_nvb (newest vertex bisection)
    - tri_uniform_refine / uniform_refine
  Stage 3 (polygon adaptive + subdomain):
    - subdomain support (``self.subdomain`` marking each cell)
    - refine_poly / coarsen_poly
    - _refine_poly_cell_ / _coarsen_poly_cell_
    - cell_barycenter (proper polygon area-weighted centroid)
  Stage 4 (interface cutting):
    - cut_interface(phi)
    - from_interface_cut_box(phi, box, nx, ny)
  Stage 5 (quad red-blue adaptive + cell-data interpolation):
    - refine_quad_rb / coarsen_quad_rb   (fully conforming red-blue quads)
    - interpolation_cell_data            (piecewise-constant transfer from a
      background triangle mesh, e.g. carrying the H history field across
      remeshing); helpers bc_to_point / cell_area / boundary_node_flag /
      find_point_in_triangle_mesh

The class inherits from fealpy3 ``HalfEdgeMesh2d`` and wraps ``self.node`` and
``self.halfedge`` as ``DynamicArray`` so fealpy2 style
``increase_size / adjust_size / extend`` calls still work.

A thin ``self.ds`` shim is exposed so legacy code written against
``mesh.ds.halfedge`` / ``mesh.ds.subdomain`` etc. keeps working.

Quad red-blue notes
-------------------
``refine_quad_rb`` / ``coarsen_quad_rb`` are a faithful port of the fealpy2
``half_edge_mesh_2d.py`` ``refine_quad`` / ``coarsen_quad`` (the module actually
imported at runtime, which DOES contain the working ``quad_coordinateCell``
blue->red closure). Colors live in ``self.halfedgedata['color']`` and
``self.halfedgedata['colorlevel']`` (both length NHE), with the fealpy2 value
convention 0=red, 1=green (center spoke), 2=blue (transition, coarse side),
3=yellow (transition, fine side). They are lazily initialised on first
``refine_quad_rb`` call via a checkerboard flood-fill over the reference mesh.
"""

import numpy as np
from scipy.spatial import cKDTree

from fealpy.mesh import HalfEdgeMesh2d as _FealpyHalfEdgeMesh2d
from fealpy.common import DynamicArray


class _DSShim:
    """Bridge fealpy2 ``mesh.ds.*`` accesses to fealpy3's flat layout."""

    def __init__(self, parent: 'AdaptiveHalfEdgeMesh2d'):
        self._m = parent

    @property
    def halfedge(self):
        return self._m.halfedge

    @property
    def hcell(self):
        return self._m.hcell

    @property
    def hedge(self):
        return self._m.hedge

    @property
    def hnode(self):
        return self._m.hnode

    @property
    def NV(self):
        return self._m.NV

    @property
    def subdomain(self):
        return self._m.subdomain

    @property
    def cellstart(self):
        return self._m.cellstart

    def number_of_all_cells(self):
        return self._m.number_of_all_cells()

    def number_of_halfedges(self):
        return self._m.number_of_halfedges()

    def number_of_cells(self):
        return self._m.number_of_cells()

    def number_of_edges(self):
        return self._m.number_of_edges()

    def number_of_vertices_of_cells(self):
        return self._m.number_of_vertices_of_cells()

    def main_halfedge_flag(self):
        return self._m.main_halfedge_flag()

    def boundary_halfedge_flag(self):
        return self._m.boundary_halfedge_flag()

    def halfedge_to_edge(self, index=None):
        return self._m.halfedge_to_edge(index)

    def cell_to_node(self, return_sparse=False):
        return self._m.cell_to_node(return_sparse=return_sparse)

    def reinit(self):
        self._m._reinit_after_edit()


class AdaptiveHalfEdgeMesh2d(_FealpyHalfEdgeMesh2d):
    """fealpy3 half-edge mesh with fealpy2 adaptive refine/coarsen ported in."""

    def __init__(self, node, halfedge, subdomain=None, NC=None, NV=None,
                 nodedof=None, initlevel=True):
        """
        Parameters
        ----------
        node : (NN, 2) array
        halfedge : (2*NE, 5) int array
        subdomain : (NC_all,) int array, optional
            subdomain[c] == 0 : unbounded outer region (must be single, cellstart=1)
            subdomain[c] > 0  : interior region id
            subdomain[c] < 0  : hole id
            If ``None`` (default), the mesh has no subdomain marking
            (all cells treated as interior, cellstart=0).
        NV : int, optional
            Fixed vertex count per cell (3 for triangle, 4 for quad, None for polygon).
        """
        node = np.asarray(node)
        halfedge = np.asarray(halfedge)

        from fealpy.mesh.mesh_base import Mesh
        Mesh.__init__(self, TD=2, itype=halfedge.dtype, ftype=node.dtype)
        self.meshtype = 'halfedge2d'
        self.type = 'TRI' if NV == 3 else 'POLY'
        self.NV = NV

        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}
        self.meshdata = {}
        self.halfedgedata = {}
        self.facedata = self.edgedata

        self.node = DynamicArray(node, dtype=node.dtype)
        self.halfedge = DynamicArray(halfedge, dtype=halfedge.dtype)

        self._set_subdomain(subdomain)

        self.ds = _DSShim(self)
        self._reinit_after_edit()

        if nodedof is not None:
            self.nodedata['dof'] = nodedof

        if initlevel:
            self._init_level_info()

    def _set_subdomain(self, subdomain):
        """Configure ``self.subdomain`` and ``self.cellstart``.

        subdomain=None means: all halfedge cells are interior (cellstart=0).
        Otherwise subdomain is stored as a DynamicArray of length NC_all.
        """
        if subdomain is None:
            self.subdomain = None
            self.cellstart = 0
            return
        subdomain = np.asarray(subdomain).astype(self.itype)
        idx, = np.nonzero(subdomain == 0)
        if len(idx) == 0:
            self.cellstart = 0
        elif len(idx) == 1:
            self.cellstart = int(idx[0]) + 1
        else:
            raise ValueError('The number of unbounded outer domains is > 1')
        self.subdomain = DynamicArray(subdomain, dtype=self.itype)

    def number_of_all_cells(self):
        if self.subdomain is None:
            return self.number_of_cells()
        return self.subdomain.size

    def number_of_cells(self):
        halfedge = self._halfedge_view()
        if self.subdomain is None:
            return int(halfedge[:, 1].max()) + 1
        return self.subdomain.size - self.cellstart

    def _init_level_info(self):
        NHE = self.number_of_halfedges()
        # celldata['level'] is length NC_all so it can be indexed directly by
        # halfedge[:, 1]; entries for outer/hole cells (< cellstart) stay 0.
        NC_all = self.number_of_all_cells()
        hlevel = np.zeros(NHE, dtype=self.itype)
        clevel = np.zeros(NC_all, dtype=self.itype)

        node = self._node_view()
        halfedge = self._halfedge_view()
        v0 = node[halfedge[halfedge[:, 2], 0]] - node[halfedge[:, 0]]
        v1 = node[halfedge[halfedge[:, 3], 0]] - node[halfedge[:, 0]]
        with np.errstate(divide='ignore', invalid='ignore'):
            angle = np.sum(v0 * v1, axis=1) / np.sqrt(
                np.sum(v0 ** 2, axis=1) * np.sum(v1 ** 2, axis=1))
        hlevel[angle < -0.98] = 1

        self.halfedgedata['level'] = DynamicArray(hlevel, dtype=self.itype)
        self.celldata['level'] = DynamicArray(clevel, dtype=self.itype)

    def _reinit_after_edit(self):
        halfedge = self._halfedge_view()
        self.NHE = halfedge.shape[0]
        self.NBE = int(np.sum(halfedge[:, 4] == np.arange(self.NHE)))
        self.NE = (self.NHE + self.NBE) // 2
        self.NN = self.node.size
        if self.subdomain is not None:
            self.NC = self.subdomain.size - self.cellstart
            NC_all = self.subdomain.size
        else:
            NC_all = int(halfedge[:, 1].max()) + 1 if self.NHE else 0
            self.NC = NC_all

        self.hcell = np.arange(NC_all, dtype=self.itype)
        self.hcell[halfedge[:, 1]] = np.arange(self.NHE, dtype=self.itype)

        self.hnode = np.arange(self.NN, dtype=self.itype)
        self.hnode[halfedge[:, 0]] = np.arange(self.NHE, dtype=self.itype)

        flag = np.arange(self.NHE) - halfedge[:, 4] >= 0
        self.hedge = np.arange(self.NHE, dtype=self.itype)[flag]

    def number_of_nodes(self):
        return self.node.size

    def number_of_halfedges(self):
        return len(self.halfedge)

    def number_of_edges(self):
        he = self._halfedge_view()
        nhe = he.shape[0]
        nbe = int(np.sum(he[:, 4] == np.arange(nhe)))
        return (nhe + nbe) // 2

    def number_of_vertices_of_all_cells(self):
        halfedge = self._halfedge_view()
        NC = self.number_of_all_cells()
        NV = np.zeros(NC, dtype=self.itype)
        np.add.at(NV, halfedge[:, 1], 1)
        return NV

    def number_of_vertices_of_cells(self):
        NV_all = self.number_of_vertices_of_all_cells()
        return NV_all[self.cellstart:]

    def _halfedge_view(self):
        return self.halfedge.data[:self.halfedge.size]

    def _node_view(self):
        return self.node.data[:self.node.size]

    def entity(self, etype=2, index=None):
        if etype in {'cell', 2}:
            return self.halfedge
        if etype in {'edge', 'face', 1}:
            return self.edge_to_node()[index] if index is not None else self.edge_to_node()
        if etype == 'halfedge':
            return self.halfedge
        if etype in {'node', 0}:
            return self.node
        raise ValueError(f"unknown etype {etype!r}")

    def main_halfedge_flag(self):
        halfedge = self._halfedge_view()
        NHE = halfedge.shape[0]
        flag = np.zeros(NHE, dtype=np.bool_)
        flag[self.hedge] = True
        return flag

    def boundary_halfedge_flag(self):
        halfedge = self._halfedge_view()
        NHE = halfedge.shape[0]
        return halfedge[:, 4] == np.arange(NHE)

    def cell_barycenter(self, index=None, return_all=False):
        """Signed-area weighted centroid of each cell.

        Uses the shoelace formula so it is correct for arbitrary polygons.
        When ``return_all`` is True and subdomain is present, centroids for
        all cells including holes/outer region are returned.
        """
        node = self._node_view()
        halfedge = self._halfedge_view()
        if return_all or self.subdomain is None:
            NC = self.number_of_all_cells()
            hflag = np.ones(halfedge.shape[0], dtype=np.bool_)
        else:
            NC = self.number_of_cells()
            hflag = self.subdomain[halfedge[:, 1]] > 0

        e0 = halfedge[halfedge[:, 3], 0][hflag]
        e1 = halfedge[hflag, 0]
        w = np.array([[0, -1], [1, 0]], dtype=np.int_)
        v = (node[e1] - node[e0]) @ w
        val = np.sum(v * node[e0], axis=1)
        ec = val.reshape(-1, 1) * (node[e1] + node[e0]) / 2

        cell_idx = halfedge[hflag, 1]
        if not return_all and self.subdomain is not None:
            cell_idx = cell_idx - self.cellstart

        a = np.zeros(NC, dtype=self.ftype)
        c = np.zeros((NC, 2), dtype=self.ftype)
        np.add.at(a, cell_idx, val)
        np.add.at(c, cell_idx, ec)
        a *= 0.5
        with np.errstate(divide='ignore', invalid='ignore'):
            c = c / (3 * a.reshape(-1, 1))
        if index is None:
            return c
        return c[index]

    def delete_entity(self, isMarked, etype='node'):
        """
        Remove entities where ``isMarked`` is True and remap ids inside halfedge.
        """
        L = len(isMarked)
        l = L - int(isMarked.sum())
        idxmap = np.zeros(L, dtype=np.int_)
        idxmap[~isMarked] = np.arange(l)
        halfedge = self._halfedge_view()
        if etype == 'node':
            halfedge[:, 0] = idxmap[halfedge[:, 0]]
        elif etype == 'cell':
            halfedge[:, 1] = idxmap[halfedge[:, 1]]
        elif etype == 'halfedge':
            halfedge[:, 2:] = idxmap[halfedge[:, 2:]]

    def mark_halfedge(self, isMarkedCell, method='poly'):
        clevel = self.celldata['level']
        hlevel = self.halfedgedata['level']
        halfedge = self._halfedge_view()
        if method == 'poly':
            while True:
                opp = halfedge[:, 4]
                isMarked = (~isMarkedCell[halfedge[:, 1]]) & isMarkedCell[halfedge[opp, 1]]
                isMarked = isMarked & (clevel[halfedge[opp, 1]] > clevel[halfedge[:, 1]])
                isMarkedCell[halfedge[isMarked, 1]] = True
                if np.all(~isMarked):
                    break
            isMarkedHEdge = clevel[halfedge[:, 1]] >= clevel[halfedge[halfedge[:, 4], 1]]
            isMarkedHEdge = isMarkedHEdge & isMarkedCell[halfedge[:, 1]]
        elif method == 'quad':
            color = self.halfedgedata['color']
            isRedHEdge = color == 0
            isGreenHEdge = color == 1
            isOtherHEdge = (color == 2) | (color == 3)

            flag0 = (hlevel - clevel[halfedge[:, 1]]) <= 0
            pre = halfedge[:, 3]
            flag1 = (hlevel[pre] - clevel[halfedge[:, 1]]) <= 0
            isMarkedHEdge = isMarkedCell[halfedge[:, 1]] & flag0 & flag1
            flag = ~isMarkedHEdge & isMarkedHEdge[halfedge[:, 4]]
            isMarkedHEdge[flag] = True
            while True:
                flag0 = isGreenHEdge & ~isMarkedHEdge & isMarkedHEdge[halfedge[:, 3]]
                flag1 = isRedHEdge & ~isMarkedHEdge & isMarkedHEdge[halfedge[:, 2]]
                flag2 = isOtherHEdge & ~isMarkedHEdge & (
                    isMarkedHEdge[halfedge[:, 2]] | isMarkedHEdge[halfedge[:, 3]])
                flag3 = isMarkedHEdge[halfedge[:, 4]] & ~isMarkedHEdge
                flag = flag0 | flag1 | flag2 | flag3
                isMarkedHEdge[flag] = True
                if (~flag).all():
                    break
        elif method == 'rg':
            color = self.halfedgedata['color']
            isRedHEdge = color == 0
            isGreenHEdge = color == 1
            isOtherHEdge = (color == 2) | (color == 3)

            flag0 = (hlevel - clevel[halfedge[:, 1]]) <= 0
            pre = halfedge[:, 3]
            flag1 = (hlevel[pre] - clevel[halfedge[:, 1]]) <= 0
            isMarkedHEdge = isMarkedCell[halfedge[:, 1]] & flag0 & flag1 & (~isGreenHEdge)
            while True:
                flag = ~isMarkedCell[halfedge[:, 1]] & isMarkedHEdge & (~isRedHEdge)
                isMarkedCell[halfedge[flag, 1]] = True

                flag0 = isMarkedCell[halfedge[:, 1]] & (~isGreenHEdge) & ~isMarkedHEdge
                flag1 = isMarkedHEdge[halfedge[:, 2]] & isMarkedHEdge[halfedge[:, 3]] & isRedHEdge & ~isMarkedHEdge
                flag2 = (isMarkedHEdge[halfedge[:, 2]] | isMarkedHEdge[halfedge[:, 3]]) & isOtherHEdge & ~isMarkedHEdge
                flag3 = isMarkedHEdge[halfedge[:, 4]] & ~isMarkedHEdge
                flag = flag0 | flag1 | flag2 | flag3
                isMarkedHEdge[flag] = True
                if (~flag).all():
                    break
        elif method == 'nvb':
            color = self.halfedgedata['color']
            isRedHEdge = color == 1

            isMarkedHEdge = isMarkedCell[halfedge[:, 1]] & isRedHEdge
            while True:
                flag0 = isRedHEdge & (
                    isMarkedHEdge[halfedge[:, 2]] | isMarkedHEdge[halfedge[:, 3]]
                ) & ~isMarkedHEdge
                flag1 = isMarkedHEdge[halfedge[:, 4]] & ~isMarkedHEdge
                flag = flag0 | flag1
                isMarkedHEdge[flag] = True
                if (~flag).all():
                    break
        else:
            raise ValueError(f"unknown mark method {method!r}")
        return isMarkedHEdge

    def refine_halfedge(self, isMarkedHEdge, newnode=None):
        NN = self.number_of_nodes()
        NHE = self.number_of_halfedges()

        hlevel = self.halfedgedata['level']
        halfedge = self.halfedge
        node = self.node
        isMainHEdge = self.main_halfedge_flag()

        flag = isMarkedHEdge & isMainHEdge
        pre = halfedge[flag, 3]
        NE1 = int(flag.sum())
        newNode = node.increase_size(NE1)
        if newnode is None:
            newNode[:] = (node[halfedge[flag, 0]] + node[halfedge[pre, 0]]) / 2
        elif isinstance(newnode, np.ndarray):
            newNode[:] = newnode

        edge2NewNode = np.zeros(NHE, dtype=np.int_)
        edge2NewNode[flag] = np.arange(NE1) + NN
        edge2NewNode[halfedge[flag, 4]] = np.arange(NE1) + NN

        NHE1 = int(isMarkedHEdge.sum())
        current, = np.where(isMarkedHEdge)
        opp = halfedge[current, 4]
        pre = halfedge[current, 3]
        flag = current != opp

        halfedge[pre, 2] = np.arange(NHE, NHE + NHE1)
        halfedge[current, 3] = np.arange(NHE, NHE + NHE1)
        halfedge[current[flag], 4] = halfedge[opp[flag], 3]

        newHalfedge = halfedge.increase_size(NHE1)
        newHalfedge[:, 0] = edge2NewNode[current]
        newHalfedge[:, 1] = halfedge[:NHE][isMarkedHEdge, 1]
        newHalfedge[:, 2] = current
        newHalfedge[:, 3] = pre
        newHalfedge[:, 4] = np.arange(NHE, NHE + NHE1)
        newHalfedge[flag, 4] = opp[flag]

        hlevel[isMarkedHEdge] += 1
        hlevel.extend(np.zeros(NHE1, dtype=np.int_))

        self._reinit_after_edit()
        return NHE1

    def coarsen_halfedge(self, isMarkedHEdge):
        self._reinit_after_edit()
        NN = self.number_of_nodes()

        node = self.node
        halfedge = self.halfedge
        hlevel = self.halfedgedata['level']

        nex = halfedge[isMarkedHEdge, 2]
        pre = halfedge[isMarkedHEdge, 3]
        opp = halfedge[isMarkedHEdge, 4]
        flag = nex != halfedge[nex, 4]

        halfedge[pre, 2] = nex
        halfedge[nex, 3] = pre
        halfedge[nex[flag], 4] = opp[flag]

        isRNode = np.zeros(NN, dtype=np.bool_)
        isRNode[halfedge[isMarkedHEdge, 0]] = True

        node.adjust_size(isRNode)
        self.delete_entity(isRNode, etype='node')

        self.delete_entity(isMarkedHEdge, etype='halfedge')
        halfedge.adjust_size(isMarkedHEdge)

        hlevel[nex] -= 1
        hlevel.adjust_size(isMarkedHEdge)

        self._reinit_after_edit()

    def _refine_tri_cell_(self, isMarkedCell, isMarkedHEdge, options={}):
        NC = self.number_of_cells()
        NC1 = int(isMarkedHEdge.sum())
        NHE = self.number_of_halfedges()

        halfedge = self.halfedge
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        current, = np.where(isMarkedHEdge)
        pre = halfedge[current, 3]
        ppre = halfedge[pre, 3]
        nex = halfedge[current, 2]
        nnex = halfedge[nex, 2]

        cidx = halfedge[current, 1]
        halfedge[nnex, 3] = np.arange(NHE + NC1, NHE + NC1 * 2)
        halfedge[pre, 2] = np.arange(NHE + NC1, NHE + NC1 * 2)
        halfedge[current, 1] = np.arange(NC, NC + NC1)
        halfedge[current, 3] = np.arange(NHE, NHE + NC1)
        halfedge[nex, 1] = np.arange(NC, NC + NC1)
        halfedge[nex, 2] = np.arange(NHE, NHE + NC1)

        halfedgeNew = halfedge.increase_size(NC1 * 2)
        halfedgeNew[:NC1, 0] = halfedge[pre, 0]
        halfedgeNew[:NC1, 1] = np.arange(NC, NC + NC1)
        halfedgeNew[:NC1, 2] = current
        halfedgeNew[:NC1, 3] = nex
        halfedgeNew[:NC1, 4] = np.arange(NHE + NC1, NHE + NC1 * 2)

        halfedgeNew[NC1:, 0] = halfedge[nex, 0]
        halfedgeNew[NC1:, 1] = cidx
        halfedgeNew[NC1:, 2] = nnex
        halfedgeNew[NC1:, 3] = pre
        halfedgeNew[NC1:, 4] = np.arange(NHE, NHE + NC1)

        flag = isMarkedHEdge[halfedgeNew[NC1:, 2]]
        current = np.where(flag)[0] + NC1
        nexpre = halfedge[halfedgeNew[current, 2], 3]
        prenex = halfedge[halfedgeNew[current, 3], 2]
        halfedgeNew[current, 2] = halfedge[nexpre, 4]
        halfedgeNew[current, 3] = halfedge[prenex, 4]

        if ('numrefine' in options) and (NC1 > 0):
            num = options['numrefine']
            num = np.r_[num, np.zeros(NC1)]
            num[cidx] -= 1
            flag = np.zeros(len(num), dtype=np.bool_)
            flag[cidx] = True
            flag = flag & (num < 0)
            num[flag] = 0
            num[-NC1:] = num[cidx]
            options['numrefine'] = num

        clevelNew = clevel.increase_size(NC1)
        clevel[cidx] += 1
        clevelNew[:] = clevel[cidx]

        hlevel.extend(np.zeros(NC1 * 2, dtype=np.int_))

        self._reinit_after_edit()

    def _coarsen_triangle_cell_(self, isMarkedCell, isMarkedHEdge, options={}):
        NC = self.number_of_cells()

        halfedge = self.halfedge
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        flag = (halfedge[:, 1] < halfedge[halfedge[:, 4], 1]) & isMarkedHEdge
        isRCell = np.zeros(NC, dtype=np.bool_)
        isRCell[halfedge[flag, 1]] = True

        for _ in range(2):
            nex = halfedge[:, 2]
            pre = halfedge[:, 3]
            opp = halfedge[:, 4]
            flag1 = isMarkedHEdge[nex]
            halfedge[flag1, 2] = nex[opp[nex[flag1]]]
            flag2 = isMarkedHEdge[pre]
            halfedge[flag2, 3] = pre[opp[pre[flag2]]]

        flag = isRCell[halfedge[:, 1]] & isMarkedHEdge

        opp = halfedge[:, 4]
        cidxmap = np.arange(NC)
        cidxmap[halfedge[opp[flag], 1]] = cidxmap[halfedge[flag, 1]]
        halfedge[:, 1] = cidxmap[halfedge[:, 1]]

        cell0 = np.unique(halfedge[:, 1])
        cidxmap0 = np.arange(NC)
        cidxmap0[cell0] = np.arange(len(cell0))
        halfedge[:, 1] = cidxmap0[halfedge[:, 1]]

        if 'numrefine' in options:
            num = options['numrefine']
            np.maximum.at(num, cidxmap, num)
            num[halfedge[flag, 1]] += 1
            options['numrefine'] = num[cell0]

        cidxflag = cidxmap != np.arange(NC)
        clevel[isRCell] -= 1
        clevel.adjust_size(cidxflag)

        hlevel.adjust_size(isMarkedHEdge)

        self.delete_entity(isMarkedHEdge, etype='halfedge')

        halfedge.adjust_size(isMarkedHEdge)

        self._reinit_after_edit()
        return isMarkedHEdge

    def refine_triangle_rg(self, isMarkedCell=None, options={}):
        NC = self.number_of_cells()
        NHE = self.number_of_halfedges()

        color = self.halfedgedata.get('color')
        halfedge = self.halfedge
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        if isMarkedCell is None:
            isMarkedCell = np.ones(NC, dtype=np.bool_)

        if color is None:
            color = np.zeros(NHE, dtype=np.int_)
            self.halfedgedata['color'] = color

        isBlueHEdge = color == 3

        isMarkedHEdge0 = self.mark_halfedge(isMarkedCell.copy(), method='rg')
        isMarkedHEdge = isMarkedHEdge0 & ((color == 0) | (color == 1))

        NHE1 = self.refine_halfedge(isMarkedHEdge)
        isMainHEdge = self.main_halfedge_flag()

        color = np.r_[color, np.zeros(NHE1, dtype=np.int_)]

        flag = isBlueHEdge & isMarkedHEdge0
        NE1 = int(flag.sum())

        NHE = self.number_of_halfedges()
        halfedgeNew = halfedge.increase_size(NE1 * 4)
        isMainHEdgeNew = np.zeros(NE1 * 4, dtype=np.bool_)

        current, = np.where(flag)
        pre = halfedge[current, 3]
        ppre = halfedge[pre, 3]
        pppre = halfedge[ppre, 3]
        opp = halfedge[current, 4]

        halfedge[current, 3] = ppre
        halfedge[current, 4] = np.arange(NHE, NHE + NE1)
        halfedge[ppre, 2] = current
        halfedge[pre, 1] = np.arange(NC, NC + NE1)
        halfedge[pre, 2] = halfedge[opp, 2]
        halfedge[pre, 3] = np.arange(NHE + NE1 * 2, NHE + NE1 * 3)
        halfedgeNew[:NE1, 0] = halfedge[ppre, 0]
        halfedgeNew[:NE1, 1] = np.arange(NC + NE1, NC + NE1 * 2)
        halfedgeNew[:NE1, 2] = np.arange(NHE + NE1 * 3, NHE + NE1 * 4)
        halfedgeNew[:NE1, 3] = np.arange(NHE + NE1, NHE + NE1 * 2)
        halfedgeNew[:NE1, 4] = current
        halfedgeNew[NE1 * 2:NE1 * 3, 0] = halfedge[ppre, 0]
        halfedgeNew[NE1 * 2:NE1 * 3, 1] = np.arange(NC, NC + NE1)
        halfedgeNew[NE1 * 2:NE1 * 3, 2] = pre
        halfedgeNew[NE1 * 2:NE1 * 3, 3] = halfedge[opp, 2]
        halfedgeNew[NE1 * 2:NE1 * 3, 4] = np.arange(NHE + NE1 * 3, NHE + NE1 * 4)
        isMainHEdgeNew[:NE1] = ~isMainHEdge[current]
        isMarkedHEdge[pre] = False

        current = opp.copy()
        nex = halfedge[current, 2]
        nnex = halfedge[nex, 2]
        nnnex = halfedge[nnex, 2]
        opp = halfedge[current, 4]

        halfedge[current, 0] = halfedge[nex, 0]
        halfedge[current, 2] = nnex
        halfedge[current, 4] = np.arange(NHE + NE1, NHE + NE1 * 2)
        halfedge[nnex, 3] = current
        halfedge[nex, 1] = np.arange(NC, NC + NE1)
        halfedge[nex, 2] = np.arange(NHE + NE1 * 2, NHE + NE1 * 3)
        halfedge[nex, 3] = pre
        halfedgeNew[NE1:NE1 * 2, 0] = halfedge[opp, 0]
        halfedgeNew[NE1:NE1 * 2, 1] = np.arange(NC + NE1, NC + NE1 * 2)
        halfedgeNew[NE1:NE1 * 2, 2] = np.arange(NHE, NHE + NE1)
        halfedgeNew[NE1:NE1 * 2, 3] = np.arange(NHE + NE1 * 3, NHE + NE1 * 4)
        halfedgeNew[NE1:NE1 * 2, 4] = current
        halfedgeNew[NE1 * 3:NE1 * 4, 0] = halfedge[nex, 0]
        halfedgeNew[NE1 * 3:NE1 * 4, 1] = np.arange(NC + NE1, NC + NE1 * 2)
        halfedgeNew[NE1 * 3:NE1 * 4, 2] = np.arange(NHE + NE1, NHE + NE1 * 2)
        halfedgeNew[NE1 * 3:NE1 * 4, 3] = np.arange(NHE, NHE + NE1)
        halfedgeNew[NE1 * 3:NE1 * 4, 4] = np.arange(NHE + NE1 * 2, NHE + NE1 * 3)
        isMainHEdgeNew[NE1:NE1 * 2] = ~isMainHEdge[current]
        isMainHEdgeNew[NE1 * 3:NE1 * 4] = True
        isMarkedHEdge[nnex] = False

        cidxmap = np.arange(NC + NE1 * 2)
        cidxmap[halfedge[current, 1]] = np.arange(NC + NE1, NC + NE1 * 2)
        cidxmap[NC + NE1:NC + NE1 * 2] = halfedge[current, 1]
        halfedge[:, 1] = cidxmap[halfedge[:, 1]]

        hlevel.extend(np.zeros(NE1 * 4, dtype=np.int_))
        clevel.extend(clevel[halfedge[opp, 1]])
        clevel.extend(clevel[halfedge[opp, 1]])

        idx = np.r_[current, opp, nex, pre, ppre, nnex, pppre, nnnex]
        color = np.r_[color, np.zeros(NE1 * 4, dtype=np.int_)]
        color[idx] = 0

        self._reinit_after_edit()

        NV = self.number_of_vertices_of_cells()
        isBlueCell = NV == 4
        isNewCell = (NV == 4) | (NV == 6)

        NC += NE1 * 2
        isMarkedHEdge = np.r_[isMarkedHEdge, np.zeros(NHE1 + NE1 * 4, dtype=np.bool_)]

        flag = isMarkedHEdge & isBlueCell[halfedge[:, 1]]
        tmp = np.where(flag)[0]
        isMarkedHEdge = isMarkedHEdge & isNewCell[halfedge[:, 1]]
        NC1 = int(isMarkedHEdge.sum())
        self._refine_tri_cell_(isNewCell, isMarkedHEdge, options=options)

        color = np.r_[color, np.zeros(NC1 * 2, dtype=np.int_)]
        color[tmp] = 1
        color[halfedge[tmp, 3]] = 3
        color[halfedge[color == 3, 4]] = 2
        color[halfedge[color == 2, 3]] = 1
        self.halfedgedata['color'] = color

    def coarsen_triangle_rg(self, isMarkedCell, options={}):
        NC = self.number_of_cells()
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        color = self.halfedgedata['color']
        halfedge = self.halfedge

        flag = (clevel[halfedge[:, 1]] == clevel[halfedge[halfedge[:, 4], 1]])
        flag = (hlevel[:] == 0) & (clevel[halfedge[:, 1]] > 0) & flag & isMarkedCell[halfedge[:, 1]]
        flag = flag & flag[halfedge[:, 4]]

        isRCell = np.ones(NC, dtype=np.bool_)
        np.logical_and.at(isRCell, halfedge[:, 1], flag)

        isMarkedHEdge = isRCell[halfedge[:, 1]]
        isMarkedHEdge[halfedge[isMarkedHEdge, 4]] = True

        flag = ((color == 2) | (color == 3)) & isMarkedCell[halfedge[:, 1]]
        flag = flag & flag[halfedge[:, 4]]
        isMarkedHEdge[flag] = True

        isMarkedHEdge = self._coarsen_triangle_cell_(isMarkedCell, isMarkedHEdge, options=options)

        color = color[~isMarkedHEdge]
        color[color == 1] = 0
        color[halfedge[color == 3, 2]] = 1
        color[halfedge[color == 2, 3]] = 1

        isBDHEdge = self.boundary_halfedge_flag()
        flag = (halfedge[halfedge[halfedge[halfedge[:, 2], 4], 2], 4]
                == np.arange(len(halfedge))) & (hlevel[halfedge[:, 4]] != 0)
        flag = flag | (isBDHEdge & isBDHEdge[halfedge[:, 2]])
        flag = flag & (hlevel[:] == 0) & (hlevel[halfedge[:, 2]] > 0)

        while True:
            count = np.zeros(self.number_of_cells(), dtype=np.int_)
            np.add.at(count, halfedge[:, 1], flag)

            NV = self.number_of_vertices_of_cells()
            flag0 = (count == 1) & (NV == 6)
            flag[flag0[halfedge[:, 1]]] = False
            flag[flag0[halfedge[halfedge[:, 4], 1]]] = False
            if (~flag0).all():
                break

        color[halfedge[flag, 4]] = 0
        color = color[~flag]
        self.coarsen_halfedge(flag)

        NV = self.number_of_vertices_of_cells()
        isBlueCell = NV == 4
        isRedCell = NV == 6
        isNewCell = (NV == 4) | (NV == 6)

        isBDHEdge = self.boundary_halfedge_flag()
        flag0 = (hlevel[:] > 0) & (hlevel[halfedge[:, 3]] == 0)
        flag1 = flag0 & (hlevel[:] != hlevel[halfedge[:, 4]]) & (hlevel[halfedge[halfedge[:, 3], 4]] > 0)
        flag1 = (flag1 & (~isBDHEdge)) | (flag0 & isBDHEdge & isRedCell[halfedge[:, 1]])
        tmp = flag1 & isBlueCell[halfedge[:, 1]]
        tmp, = np.where(tmp)

        flag = (hlevel[:] == 0) & (hlevel[halfedge[:, 2]] > 0) & (
            hlevel[halfedge[:, 4]] > 0) & isBlueCell[halfedge[:, 1]]
        flag = flag | ((hlevel[:] == 0) & isRedCell[halfedge[:, 1]])
        flag = flag[halfedge[:, 3]]
        NC1 = int(flag.sum())

        self._refine_tri_cell_(isNewCell, flag, options=options)

        color = np.r_[color, np.zeros(NC1 * 2, dtype=np.int_)]
        color[tmp] = 1
        color[halfedge[tmp, 3]] = 3
        color[halfedge[color == 3, 4]] = 2
        color[halfedge[color == 2, 3]] = 1
        self.halfedgedata['color'] = color

    # ------------------------------------------------------------------
    # Quad red-blue refine / coarsen  (Stage 5)
    # ------------------------------------------------------------------

    def _init_quad_color(self):
        """Lazily build the checkerboard halfedge color for red-blue quads.

        Faithful port of fealpy2 ``refine_quad`` colour seed (half_edge_mesh_2d
        lines 1464-1473). Fills ``self.halfedgedata['color']`` and
        ``['colorlevel']`` (both length NHE, int64) using the value convention
        0=red, 1=green, 2=blue, 3=yellow. Must be called on an all-quad mesh
        before any adaptive step so every regular quad edge is red/green.

        Side effects
        ------------
        Writes ``self.halfedgedata['color']`` and ``['colorlevel']``.
        """
        NHE = self.number_of_halfedges()
        he = self._halfedge_view()
        color = 3 * np.ones(NHE, dtype=np.int_)
        color[1] = 1
        guard = 0
        while (color == 3).any():
            red = color == 1
            gre = color == 0
            color[he[red][:, [2, 3, 4]]] = 0
            color[he[gre][:, [2, 3, 4]]] = 1
            guard += 1
            if guard > NHE + 10:
                raise RuntimeError("quad colour flood-fill did not converge; "
                                   "mesh is not a checkerboard-2-colourable quad mesh")
        colorlevel = ((color == 1) | (color == 2)).astype(np.int_)
        # Store as plain ndarrays (not DynamicArray): mark_halfedge relies on
        # elementwise ``color == k``, which a DynamicArray reduces to a scalar.
        self.halfedgedata['color'] = color
        self.halfedgedata['colorlevel'] = colorlevel

    def refine_quad_rb(self, isMarkedCell=None, options={}):
        """Fully conforming red-blue (红蓝) refinement of a quad halfedge mesh.

        Faithful new-architecture port of fealpy2
        ``half_edge_mesh_2d.HalfEdgeMesh2d.refine_quad`` (lines 1457-1552),
        including the ``quad_coordinateCell`` blue->red transition closure. Each
        marked quad is split into 4 sub-quads; edge-neighbours of differing
        level are joined by conforming blue/yellow transition quads so the mesh
        never has hanging nodes.

        Parameters
        ----------
        isMarkedCell : ndarray of bool, optional
            Cells to refine. Length ``number_of_cells()`` (interior only) or
            ``number_of_all_cells()`` (outer/hole entries are forced False).
            ``None`` marks every interior cell (uniform refine).
        options : dict, optional
            Passed through to the cell-split engine (e.g. ``'numrefine'``).

        Side effects
        ------------
        Mutates node/halfedge/subdomain/celldata['level']/halfedgedata['level']
        in place, and maintains ``halfedgedata['color']`` / ``['colorlevel']``.
        Lazily initialises the colour arrays via ``_init_quad_color`` on the
        first call.
        """
        NC_all = self.number_of_all_cells()
        cstart = self.cellstart

        # normalise mark to NC_all length, outer/hole always False
        if isMarkedCell is None:
            isMarkedCell = np.ones(NC_all, dtype=np.bool_)
        else:
            isMarkedCell = np.asarray(isMarkedCell).astype(np.bool_)
            if isMarkedCell.size == self.number_of_cells():
                full = np.zeros(NC_all, dtype=np.bool_)
                full[cstart:] = isMarkedCell
                isMarkedCell = full
            else:
                isMarkedCell = isMarkedCell.copy()
        isMarkedCell[:cstart] = False

        if 'color' not in self.halfedgedata:
            self._init_quad_color()

        color = np.asarray(self.halfedgedata['color'][:]).copy()
        colorlevel = np.asarray(self.halfedgedata['colorlevel'][:]).copy()

        halfedge = self.halfedge
        node = self._node_view()
        NHE0 = self.number_of_halfedges()

        isBlueHEdge = color == 2
        isYellowHEdge = color == 3

        # --- Phase 1: quad centre of every cell (corner average, forced NV=4) ---
        hv = self._halfedge_view()
        start = self.hcell[cstart:]
        c0 = node[hv[start, 0]]
        h = hv[start, 2]; c1 = node[hv[h, 0]]
        h = hv[h, 2];     c2 = node[hv[h, 0]]
        h = hv[h, 2];     c3 = node[hv[h, 0]]
        bc = np.zeros((NC_all, 2), dtype=self.ftype)
        bc[cstart:] = (c0 + c1 + c2 + c3) / 4

        # shift transition-cell centres to the true parent-quad centre
        bcur, = np.where(isBlueHEdge)
        if bcur.size:
            nex = hv[bcur, 2]; nnex = hv[nex, 2]; pre = hv[bcur, 3]
            np.add.at(bc, hv[bcur, 1], (node[hv[nnex, 0]] - node[hv[pre, 0]]) / 8)
        ycur, = np.where(isYellowHEdge)
        if ycur.size:
            nex = hv[ycur, 2]; nnex = hv[nex, 2]
            np.add.at(bc, hv[ycur, 1], (node[hv[nex, 0]] - node[hv[ycur, 0]]) / 8)

        # --- Phase A: mark halfedges (quad) and bisect the red/green ones ---
        isMarkedHEdge0 = self.mark_halfedge(isMarkedCell.copy(), method='quad')
        isMarkedHEdge = isMarkedHEdge0 & ((color == 0) | (color == 1))
        isMarkedCell[halfedge[np.where(isMarkedHEdge)[0], 1]] = True
        isMarkedCell[:cstart] = False

        # Track the green-spoke starts geometrically (robust to the new-arch
        # refine_halfedge numbering, which differs from fealpy2). After bisection
        # a marked halfedge index keeps pointing midpoint->corner, so it is a
        # center-split start; pre-existing green edges that were NOT bisected
        # (transition spokes) are also starts. Verified start set == oracle.
        pre_green = color == 1                       # length NHE0
        current_orig = np.where(isMarkedHEdge)[0]    # indices preserved by bisect

        self.refine_halfedge(isMarkedHEdge)
        NHE_A = self.number_of_halfedges()
        dA = NHE_A - NHE0

        color = np.r_[color, np.zeros(dA, dtype=np.int_)]
        colorlevel = np.r_[colorlevel, np.zeros(dA, dtype=np.int_)]
        # A bisected edge stays monochrome: the new (back) half inherits the
        # colour of the front half it feeds into (halfedge[new, 2]). Using the
        # opposite (old fealpy2 idiom) mis-parities self-opposite boundary
        # halves under the new-arch refine_halfedge numbering.
        new_idx = np.arange(NHE0, NHE_A)
        color[new_idx] = color[halfedge[new_idx, 2]]
        colorlevel[new_idx] = colorlevel[halfedge[new_idx, 2]]

        # --- Phase B: blue transition cells marked for refine become red ---
        flag = isBlueHEdge & isMarkedHEdge0
        current, = np.where(flag)
        NE1 = current.size
        if NE1:
            NCB = self.number_of_all_cells()   # bisect adds no cells
            pre = halfedge[current, 3]
            ppre = halfedge[pre, 3]
            opp = halfedge[current, 4]

            halfedge[current, 3] = ppre
            halfedge[current, 4] = np.arange(NHE_A, NHE_A + NE1)
            halfedge[ppre, 2] = current
            halfedge[pre, 1] = np.arange(NCB, NCB + NE1)
            halfedge[pre, 2] = halfedge[opp, 2]
            halfedge[pre, 3] = np.arange(NHE_A, NHE_A + NE1)

            halfedgeNew = halfedge.increase_size(NE1 * 2)
            halfedgeNew[:NE1, 0] = halfedge[ppre, 0]
            halfedgeNew[:NE1, 1] = np.arange(NCB, NCB + NE1)
            halfedgeNew[:NE1, 2] = pre
            halfedgeNew[:NE1, 3] = np.arange(NHE_A + NE1, NHE_A + NE1 * 2)
            halfedgeNew[:NE1, 4] = current

            current2 = opp.copy()
            nex = halfedge[current2, 2]
            nnex = halfedge[nex, 2]
            opp2 = halfedge[current2, 4]

            halfedge[current2, 0] = halfedge[nex, 0]
            halfedge[current2, 2] = nnex
            halfedge[current2, 4] = np.arange(NHE_A + NE1, NHE_A + NE1 * 2)
            halfedge[nnex, 3] = current2
            halfedge[nex, 1] = np.arange(NCB, NCB + NE1)
            halfedge[nex, 2] = np.arange(NHE_A + NE1, NHE_A + NE1 * 2)
            halfedge[nex, 3] = pre
            halfedgeNew[NE1:, 0] = halfedge[opp2, 0]
            halfedgeNew[NE1:, 1] = np.arange(NCB, NCB + NE1)
            halfedgeNew[NE1:, 2] = np.arange(NHE_A, NHE_A + NE1)
            halfedgeNew[NE1:, 3] = nex
            halfedgeNew[NE1:, 4] = current2

            # grow per-cell arrays for the NE1 new (merged) cells
            merged_cell = halfedge[current2, 1]
            self.celldata['level'].extend(
                np.asarray(self.celldata['level'][:])[merged_cell])
            if self.subdomain is not None:
                self.subdomain.extend(
                    np.asarray(self.subdomain[:])[merged_cell])
            self.halfedgedata['level'].extend(np.zeros(NE1 * 2, dtype=np.int_))

            self._reinit_after_edit()

            NHE_B = self.number_of_halfedges()
            color = np.r_[color, np.zeros(NE1 * 2, dtype=np.int_)]
            colorlevel = np.r_[colorlevel, np.zeros(NE1 * 2, dtype=np.int_)]
            color[NHE_A:NHE_A + NE1] = 0
            color[NHE_A + NE1:NHE_A + NE1 * 2] = 1
            oppB = halfedge[np.arange(NHE_A, NHE_B), 4]
            color[oppB] = (color[NHE_A:NHE_B] + 1) % 2
            colorlevel[NHE_A + NE1:NHE_A + NE1 * 2] += 1
        else:
            NHE_B = NHE_A

        # --- Phase C: centre-split every red quad with bisected edges ---
        NV = self.number_of_vertices_of_all_cells()
        isNewCell = (NV == 6) | (NV == 8)
        bc = np.r_[bc, np.zeros((NE1, 2), dtype=self.ftype)]

        # start halfedges = (unbisected pre-existing green | bisected spokes),
        # restricted to the new (NV 6/8) cells (empirically == oracle start set)
        NHE_B = self.number_of_halfedges()
        unbis_green = np.zeros(NHE_B, dtype=np.bool_)
        unbis_green[:NHE0] = pre_green & ~isMarkedHEdge
        co = np.zeros(NHE_B, dtype=np.bool_)
        co[current_orig] = True
        incell = isNewCell[halfedge[np.arange(NHE_B), 1]]
        isStartHEdge = (unbis_green | co) & incell
        NC1 = int(isStartHEdge.sum())
        center = bc[isNewCell]

        # keep isMarkedHEdge aligned to current NHE for the Phase-D recolour
        isMarkedHEdge = np.r_[isMarkedHEdge, np.zeros(NHE_B - NHE0, dtype=np.bool_)]

        self._refine_poly_cell_(isNewCell, isStartHEdge, options=options,
                                center=center)

        # --- Phase D: recolour the freshly created transition edges ---
        NHE_C = self.number_of_halfedges()
        tmp = np.where(~isMarkedHEdge & isStartHEdge)[0]
        color = np.r_[color, np.zeros(NC1 * 2, dtype=np.int_)]
        colorlevel = np.r_[colorlevel, np.zeros(NC1 * 2, dtype=np.int_)]
        color[NHE_C - NC1 * 2:NHE_C - NC1] = 1
        color[halfedge[tmp, 3]] = 3
        color[halfedge[halfedge[tmp, 3], 4]] = 2
        colorlevel[NHE_C - NC1 * 2:NHE_C - NC1] += 1

        self.halfedgedata['color'] = color
        self.halfedgedata['colorlevel'] = colorlevel

    def coarsen_quad_rb(self, isMarkedCell, options={}):
        """Red-blue coarsening of a quad halfedge mesh (inverse of refine).

        Removes the four sub-quads of a refined parent, recovering the parent
        quad, then rebuilds the checkerboard halfedge colour. The topological
        surgery (centre-node removal + edge un-bisection, with the built-in
        "adjacent levels differ by <=1" mark propagation) reuses the validated
        ``coarsen_poly`` engine; this method adds quad-specific colour
        maintenance and guarantees the result stays all-quad.

        Unlike the fealpy2 ``coarsen_quad`` (which was non-functional — it
        crashed for any mesh with more than one cell), this is written and
        validated directly: refine->coarsen restores the original NC/NN/NE and
        total area, the result is conforming (no hanging nodes) and all-quad,
        and the recoloured mesh can be refined again while staying conforming.

        Parameters
        ----------
        isMarkedCell : ndarray of bool
            Cells to coarsen. Length ``number_of_cells()`` (interior only) or
            ``number_of_all_cells()``. Only cells with ``celldata['level'] > 0``
            (i.e. previously refined) can actually be removed; the underlying
            engine propagates marks so coarsening never breaks the 2:1 balance.
        options : dict, optional
            Passed through to ``coarsen_poly`` (e.g. ``'numrefine'``).

        Side effects
        ------------
        Mutates node/halfedge/subdomain/celldata['level']/halfedgedata['level']
        in place and rewrites ``halfedgedata['color']`` / ``['colorlevel']``.

        Raises
        ------
        RuntimeError
            If coarsening leaves a non-quad transition cell that cannot be
            recoloured as a checkerboard (asymmetric partial marks); mark whole
            refined groups to avoid this.
        """
        NC_all = self.number_of_all_cells()
        cstart = self.cellstart

        isMarkedCell = np.asarray(isMarkedCell).astype(np.bool_)
        if isMarkedCell.size == self.number_of_cells():
            full = np.zeros(NC_all, dtype=np.bool_)
            full[cstart:] = isMarkedCell
            isMarkedCell = full
        else:
            isMarkedCell = isMarkedCell.copy()
        isMarkedCell[:cstart] = False

        # topological inverse: remove centre nodes + un-bisect edges. The engine
        # also enforces the 2:1 level balance via its own mark propagation.
        self.coarsen_poly(isMarkedCell, options=options)

        # rebuild the checkerboard colour on the coarsened mesh. _init_quad_color
        # only makes sense on an all-quad mesh; a leftover transition polygon
        # (from asymmetric partial marks) is reported rather than mis-coloured.
        NV = self.number_of_vertices_of_all_cells()
        if not (NV[cstart:] == 4).all():
            bad = np.unique(NV[cstart:][NV[cstart:] != 4])
            raise RuntimeError(
                f"coarsen_quad_rb: coarsening left non-quad cell(s) with "
                f"NV={bad.tolist()}; mark complete refined groups so the mesh "
                "stays all-quad")
        self.halfedgedata.pop('color', None)
        self.halfedgedata.pop('colorlevel', None)
        self._init_quad_color()

    # ------------------------------------------------------------------
    # Parent -> child inheritance interpolation (nested quad refinement)
    # ------------------------------------------------------------------

    @staticmethod
    def _locate_in_quads(points, node_old, cell_old, tol=1e-9):
        """Locate each point in the quad that contains it.

        Point-in-convex-quad test via edge cross products, assuming each old
        quad is given counter-clockwise. Used for parent->child inheritance,
        where every query point (child centroid or new node) lies inside
        exactly one parent quad by construction.

        Parameters
        ----------
        points : (NP, 2) float64
            Query points.
        node_old : (NN_old, 2) float64
            Coordinates of the pre-refine mesh nodes.
        cell_old : (NC_old, 4) int
            Parent quad connectivity, counter-clockwise, indexing ``node_old``.
        tol : float
            Boundary tolerance so points on an edge/corner still match.

        Returns
        -------
        loc : (NP,) int
            Index (into ``cell_old``) of the containing quad, or ``-1`` if the
            point fell outside every quad (should not happen for nested refine).
        """
        NP = points.shape[0]
        NC = cell_old.shape[0]
        loc = np.full(NP, -1, dtype=np.int_)
        v = node_old[cell_old]                       # (NC, 4, 2)
        px = points[:, 0]; py = points[:, 1]
        for i in range(NC):
            q = v[i]
            inside = np.ones(NP, dtype=np.bool_)
            for k in range(4):
                a = q[k]; b = q[(k + 1) % 4]
                cross = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
                inside &= cross >= -tol
            newly = inside & (loc < 0)
            loc[newly] = i
        return loc

    @staticmethod
    def _bilinear_weights(points, quads, niter=30, tol=1e-13):
        """Bilinear shape-function weights of ``points`` inside their quads.

        Newton solve of the inverse bilinear map for each point. Quad corners
        ``quads[p] = [N0, N1, N2, N3]`` (counter-clockwise) map to reference
        coords via ``X(xi, eta) = (1-xi)(1-eta) N0 + xi(1-eta) N1 +
        xi*eta N2 + (1-xi)eta N3`` with ``xi, eta in [0, 1]``.

        Parameters
        ----------
        points : (NP, 2) float64
            Query points.
        quads : (NP, 4, 2) float64
            The containing quad's corner coordinates for each point.

        Returns
        -------
        w : (NP, 4) float64
            Bilinear weights ``[N0, N1, N2, N3]`` summing to 1; a linear/bilinear
            field ``f`` is reproduced exactly by ``w @ f_corners``.
        """
        NP = points.shape[0]
        Q0, Q1, Q2, Q3 = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
        xi = np.full(NP, 0.5); eta = np.full(NP, 0.5)
        for _ in range(niter):
            omx = 1 - xi; ome = 1 - eta
            N0 = omx * ome; N1 = xi * ome; N2 = xi * eta; N3 = omx * eta
            X = (N0[:, None] * Q0 + N1[:, None] * Q1
                 + N2[:, None] * Q2 + N3[:, None] * Q3)
            r = X - points                                    # residual (NP,2)
            dXi = ome[:, None] * (Q1 - Q0) + eta[:, None] * (Q2 - Q3)
            dEta = omx[:, None] * (Q3 - Q0) + xi[:, None] * (Q2 - Q1)
            det = dXi[:, 0] * dEta[:, 1] - dXi[:, 1] * dEta[:, 0]
            det = np.where(np.abs(det) < 1e-300, 1e-300, det)
            dxi = -(dEta[:, 1] * r[:, 0] - dEta[:, 0] * r[:, 1]) / det
            deta = -(-dXi[:, 1] * r[:, 0] + dXi[:, 0] * r[:, 1]) / det
            xi = np.clip(xi + dxi, 0.0, 1.0)
            eta = np.clip(eta + deta, 0.0, 1.0)
            if np.max(np.abs(r)) < tol:
                break
        omx = 1 - xi; ome = 1 - eta
        return np.stack([omx * ome, xi * ome, xi * eta, omx * eta], axis=1)

    def inherit_nodal_data(self, node_data_old, node_old, cell_old):
        """Inherit a nodal field from the pre-refine mesh onto the current mesh.

        Exact parent->child inheritance for red-blue quad refinement: old node
        indices are preserved and every new node (edge midpoint / cell centre)
        lies inside one parent quad, so its value is the parent quad's bilinear
        interpolation. This reproduces any bi-/linear field exactly.

        Parameters
        ----------
        node_data_old : (NN_old,) or (NN_old, d) float64
            Nodal values on the pre-refine mesh, indexed by ``node_old`` order.
        node_old : (NN_old, 2) float64
            Pre-refine node coordinates (snapshot taken before ``refine_quad_rb``).
        cell_old : (NC_old, 4) int
            Pre-refine quad connectivity (from ``cell_to_node()``), CCW.

        Returns
        -------
        out : (NN_new,) or (NN_new, d) float64
            Values on all current nodes, indexed by the current node order.

        Raises
        ------
        RuntimeError
            If a current node cannot be located in any parent quad.
        """
        node_data_old = np.asarray(node_data_old)
        pts = self._node_view()
        loc = self._locate_in_quads(pts, node_old, cell_old)
        if (loc < 0).any():
            raise RuntimeError(
                f"inherit_nodal_data: {(loc < 0).sum()} node(s) outside every "
                "parent quad; snapshot node_old/cell_old must be the pre-refine mesh")
        quads = node_old[cell_old[loc]]                       # (NN_new, 4, 2)
        w = self._bilinear_weights(pts, quads)                # (NN_new, 4)
        corner_vals = node_data_old[cell_old[loc]]            # (NN_new, 4[, d])
        if node_data_old.ndim == 1:
            return np.einsum('nk,nk->n', w, corner_vals)
        return np.einsum('nk,nkd->nd', w, corner_vals)

    def inherit_cell_data(self, cell_data_old, node_old, cell_old):
        """Inherit piecewise-constant cell data from parents onto children.

        Each current (child) cell nests inside exactly one parent quad; its
        constant value is copied from that parent. Supports ``(NC_old,)`` and
        ``(NC_old, NQ)`` layouts (e.g. an ``(NC, NQ)`` quadrature-point history
        field), returning ``(NC_new, ...)``.

        Parameters
        ----------
        cell_data_old : (NC_old,) or (NC_old, NQ) float64
            Piecewise-constant data on the pre-refine mesh, indexed by the
            interior-cell order (``cell_old`` order).
        node_old : (NN_old, 2) float64
            Pre-refine node coordinates.
        cell_old : (NC_old, 4) int
            Pre-refine quad connectivity (CCW), indexing ``node_old``.

        Returns
        -------
        out : (NC_new,) or (NC_new, NQ) float64
            Piecewise-constant data on the current cells, interior-cell order.

        Raises
        ------
        RuntimeError
            If a child centroid cannot be located in any parent quad.
        """
        cell_data_old = np.asarray(cell_data_old)
        centroids = self.cell_barycenter()                    # (NC_new, 2)
        loc = self._locate_in_quads(centroids, node_old, cell_old)
        if (loc < 0).any():
            raise RuntimeError(
                f"inherit_cell_data: {(loc < 0).sum()} child centroid(s) outside "
                "every parent quad; check node_old/cell_old snapshot")
        return cell_data_old[loc]

    def refine_triangle_nvb(self, isMarkedCell=None, options={}):
        NC = self.number_of_cells()
        NHE = self.number_of_halfedges()

        color = self.halfedgedata.get('color')
        node = self._node_view()
        halfedge = self.halfedge

        if isMarkedCell is None:
            isMarkedCell = np.ones(NC, dtype=np.bool_)

        if color is None:
            color = np.zeros(NHE, dtype=np.int_)
            nex = halfedge[:, 2]
            pre = halfedge[:, 3]
            he = self._halfedge_view()
            l = node[he[:, 0]] - node[he[pre, 0]]
            l = np.linalg.norm(l, axis=1)
            color[(l > l[nex]) & (l > l[pre])] = 1
            self.halfedgedata['color'] = color

        if ('HB' in options) and (options['HB'] is not None):
            HB = np.zeros((NC, 2), dtype=np.int_)
            HB[:, 0] = np.arange(NC)
            HB[:, 1] = np.arange(NC)
            options['HB'] = HB

        isMarkedHEdge = self.mark_halfedge(isMarkedCell, method='nvb')
        flag = np.array(True)
        while flag.any():
            flag = isMarkedHEdge & (color == 1)
            flag = flag & flag[halfedge[:, 4]]

            NE1 = self.refine_halfedge(flag)

            NV = self.number_of_vertices_of_cells()
            isNewCell = NV == 4

            flag = np.r_[flag, np.zeros(NE1, dtype=np.bool_)]
            NE2 = int(flag.sum())
            isMarkedHEdge = np.r_[isMarkedHEdge, np.zeros(NE1 + NE2 * 2, dtype=np.bool_)]

            color = np.r_[color, np.zeros(NE1, dtype=np.int_)]
            color[flag] = 0
            color[halfedge[flag, 2]] = 1
            color[halfedge[halfedge[flag, 3], 3]] = 1
            color = np.r_[color, np.zeros(NE2 * 2, dtype=np.int_)]
            self.halfedgedata['color'] = color

            self._refine_tri_cell_(isNewCell, flag, options=options)

    def coarsen_triangle_nvb(self, isMarkedCell, options={}):
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']

        color = self.halfedgedata['color']
        halfedge = self.halfedge

        nex, opp = halfedge[:, 2], halfedge[:, 4]

        isMarkedHEdge = isMarkedCell[halfedge[:, 1]] & (hlevel[:] == 0) & (color[:] == 0)
        isMarkedHEdge = isMarkedHEdge & (clevel[halfedge[:, 1]] > 0)
        isMarkedHEdge = isMarkedHEdge & isMarkedHEdge[halfedge[:, 4]]

        isMarkedHEdge = isMarkedHEdge & (color[halfedge[:, 3]] == 1)

        isMarkedHEdge = isMarkedHEdge & (isMarkedHEdge[opp[nex[opp[nex]]]] | (opp[nex] == nex))
        isMarkedHEdge[opp[isMarkedHEdge]] = True

        isMarkedHEdge = self._coarsen_triangle_cell_(isMarkedCell, isMarkedHEdge, options=options)

        color = color[~isMarkedHEdge]
        flag = (color == 1) & (color == 1)[halfedge[:, 2]]
        color[flag] = 0
        color[halfedge[flag, 2]] = 0
        color[halfedge[flag, 3]] = 1

        NV = self.number_of_vertices_of_cells()
        nex = halfedge[:, 2]
        opp = halfedge[:, 4]
        isBDHEdge = self.boundary_halfedge_flag()
        flag = (opp[nex[opp[nex]]] == np.arange(len(halfedge))) | (isBDHEdge & isBDHEdge[nex])
        flag = flag & (color == 1)[halfedge[:, 2]] & (NV[halfedge[:, 1]] == 4)

        self.coarsen_halfedge(flag)

        color = color[~flag]
        self.halfedgedata['color'] = color

    def uniform_refine(self, n=1):
        if self.NV == 3:
            for _ in range(n):
                self.refine_triangle_rg()
        else:
            for _ in range(n):
                self.refine_triangle_rg()

    def tri_uniform_refine(self, n=1, method="rg"):
        if method == 'rg':
            for _ in range(n):
                self.refine_triangle_rg()
        elif method == 'nvb':
            for _ in range(n * 2):
                self.refine_triangle_nvb()
        else:
            raise ValueError('refine type error! "rg" or "nvb"')

    def halfedge_direction(self):
        node = self._node_view()
        halfedge = self._halfedge_view()
        return node[halfedge[:, 0]] - node[halfedge[halfedge[:, 3], 0]]

    def halfedge_length(self):
        return np.linalg.norm(self.halfedge_direction(), axis=1)

    # ------------------------------------------------------------------
    # topology helpers
    # ------------------------------------------------------------------

    def halfedge_to_edge(self, index=None):
        halfedge = self._halfedge_view()
        NE = self.number_of_edges()
        halfedge2edge = np.zeros(halfedge.shape[0], dtype=self.itype)
        halfedge2edge[self.hedge] = np.arange(NE, dtype=self.itype)
        halfedge2edge[halfedge[self.hedge, 4]] = np.arange(NE, dtype=self.itype)
        if index is None:
            return halfedge2edge
        return halfedge2edge[index]

    def edge_to_node(self):
        halfedge = self._halfedge_view()
        hedge = self.hedge
        edge = np.zeros((len(hedge), 2), dtype=self.itype)
        edge[:, 1] = halfedge[hedge, 0]
        edge[:, 0] = halfedge[halfedge[hedge, 4], 0]
        return edge

    def cell_to_node(self, return_sparse=False):
        halfedge = self._halfedge_view()
        cstart = self.cellstart
        NC = self.number_of_cells()

        NV = self.number_of_vertices_of_cells()
        if self.NV == 3:
            cn = np.zeros((NC, 3), dtype=self.itype)
            current = halfedge[self.hcell[cstart:], 2]
            cn[:, 0] = halfedge[current, 0]
            current = halfedge[current, 2]
            cn[:, 1] = halfedge[current, 0]
            current = halfedge[current, 2]
            cn[:, 2] = halfedge[current, 0]
            return cn
        elif self.NV == 4:
            cn = np.zeros((NC, 4), dtype=self.itype)
            current = halfedge[self.hcell[cstart:], 3]
            cn[:, 0] = halfedge[current, 0]
            current = halfedge[current, 2]
            cn[:, 1] = halfedge[current, 0]
            current = halfedge[current, 2]
            cn[:, 2] = halfedge[current, 0]
            current = halfedge[current, 2]
            cn[:, 3] = halfedge[current, 0]
            return cn
        else:
            # polygon: CSR-like (cell2node_flat, cellLocation)
            cellLocation = np.zeros(NC + 1, dtype=self.itype)
            cellLocation[1:] = np.cumsum(NV)
            cell2node = np.zeros(cellLocation[-1], dtype=self.itype)
            current = self.hcell[cstart:].copy()
            idx = cellLocation[:-1].copy()
            cell2node[idx] = halfedge[halfedge[current, 3], 0]
            NV0 = np.ones(NC, dtype=self.itype)
            isNotOK = NV0 < NV
            while np.any(isNotOK):
                idx[isNotOK] += 1
                NV0[isNotOK] += 1
                cell2node[idx[isNotOK]] = halfedge[current[isNotOK], 0]
                current[isNotOK] = halfedge[current[isNotOK], 2]
                isNotOK = NV0 < NV
            return cell2node, cellLocation

    # ------------------------------------------------------------------
    # Polygon refine / coarsen  (Stage 3)
    # ------------------------------------------------------------------

    def _refine_poly_cell_(self, isMarkedCell, isStartHEdge, options={},
                           center=None):
        """
        ``isMarkedCell`` has length NC_all (with outer/hole entries False).
        ``isStartHEdge`` has length NHE.
        New cell ids are appended at NC_all onward, keeping the outer/hole
        entries stable in position.

        Parameters
        ----------
        center : (NN1, 2) float ndarray, optional
            Explicit new-node coordinates, one per marked cell, in the same
            order as ``np.where(isMarkedCell)`` over the full NC_all range.
            When ``None`` (default) the area-weighted cell centroid is used
            (``cell_barycenter``). Pass this for cells whose centroid must not
            be the shoelace barycenter (e.g. NV==6 quad transition cells, whose
            split node should sit at the quad center, not the polygon centroid).
        """
        NC_all = self.number_of_all_cells()
        NN = self.number_of_nodes()
        NC1 = int(isStartHEdge.sum())
        NN1 = int(isMarkedCell.sum())
        NHE = self.number_of_halfedges()
        cstart = self.cellstart

        clevel = self.celldata['level']
        hlevel = self.halfedgedata['level']
        halfedge = self.halfedge

        if center is None:
            # cell_barycenter returns NC-length (interior only); map isMarkedCell
            # back into interior-cell indexing to pick the right centroids.
            interior_marked = isMarkedCell[cstart:]
            self.node.extend(self.cell_barycenter(index=interior_marked))
        else:
            center = np.asarray(center, dtype=self.ftype)
            if center.shape != (NN1, 2):
                raise ValueError(
                    f"center must have shape ({NN1}, 2) (one per marked cell), "
                    f"got {center.shape}")
            self.node.extend(center)

        cell2newNode = np.zeros(NC_all, dtype=np.int_)
        cell2newNode[isMarkedCell] = np.arange(NN, NN + NN1)

        current = np.where(isStartHEdge)[0]
        pre = halfedge[current, 3]
        nex = halfedge[current, 2]
        cidx = halfedge[current, 1]
        halfedge[current, 1] = np.arange(NC_all, NC_all + NC1)
        halfedge[nex, 1] = np.arange(NC_all, NC_all + NC1)
        isNotOK = np.ones_like(nex, dtype=np.bool_)
        while np.any(isNotOK):
            isNotOK = ~isStartHEdge[halfedge[nex, 2]]
            nex[isNotOK] = halfedge[nex[isNotOK], 2]
            halfedge[nex, 1] = np.arange(NC_all, NC_all + NC1)

        ppre = halfedge[halfedge[:, 3], 3]
        ppre[halfedge[nex, 2]] = current
        ppre = ppre[current]
        halfedge[current, 3] = np.arange(NHE + NC1, NHE + NC1 * 2)
        halfedge[pre, 2] = np.arange(NHE, NHE + NC1)

        halfedgeNew = halfedge.increase_size(NC1 * 2)
        halfedgeNew[:NC1, 0] = cell2newNode[cidx]
        halfedgeNew[:NC1, 1] = halfedge[pre, 1]
        halfedgeNew[:NC1, 2] = halfedge[ppre, 3]
        halfedgeNew[:NC1, 3] = pre
        halfedgeNew[:NC1, 4] = halfedge[current, 3]

        halfedgeNew[NC1:, 0] = halfedge[pre, 0]
        halfedgeNew[NC1:, 1] = halfedge[current, 1]
        halfedgeNew[NC1:, 2] = current
        halfedgeNew[NC1:, 3] = halfedge[nex, 2]
        halfedgeNew[NC1:, 4] = halfedge[pre, 2]

        # Reassign contiguous interior cell ids; outer/hole slots [0, cstart)
        # must stay in place. Only ids >= cstart get compacted.
        idx = np.unique(halfedge[:, 1])
        cidxmap = np.arange(NC_all + NC1)
        interior_idx = idx[idx >= cstart]
        cidxmap[interior_idx] = np.arange(cstart, cstart + interior_idx.size)
        halfedge[:, 1] = cidxmap[halfedge[:, 1]]

        clevel1 = clevel[cidx]
        clevelNew = clevel.adjust_size(isMarkedCell, int(NC1))
        clevelNew[:] = clevel1 + 1

        hlevel.extend(np.zeros(NC1 * 2, dtype=np.int_))

        if self.subdomain is not None:
            sdNew = self.subdomain.adjust_size(isMarkedCell, int(NC1))
            sdNew[:] = self.subdomain[cidx]

        self._reinit_after_edit()

        if 'numrefine' in options:
            num = options['numrefine']
            num = np.r_[num, np.zeros(NC1, dtype=np.int_)]
            num[-NC1:] = num[cidx] - 1
            flag = np.zeros(len(num), dtype=np.bool_)
            flag[-NC1:] = True
            flag = flag & (num < 0)
            num[flag] = 0
            options['numrefine'] = num[idx]

    def _coarsen_poly_cell_(self, isMarkedCell, isRNode, options={}):
        """
        ``isMarkedCell`` has length NC_all (with outer/hole entries False).
        Cells that are removed give up their ids; the remaining cells are
        remapped so that [0, cstart) outer/hole slots stay put and interior
        cells stay contiguous starting from ``cstart``.
        """
        NC_all = self.number_of_all_cells()
        NN = self.number_of_nodes()
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']
        cstart = self.cellstart

        node = self.node
        halfedge = self.halfedge

        flag = isMarkedCell[halfedge[:, 1]]
        np.logical_and.at(isRNode, halfedge[:, 0], flag)
        nn = int(isRNode.sum())
        newNode = node[isRNode]

        isMarkedHEdge = isRNode[halfedge[:, 0]]
        isMarkedHEdge[halfedge[isMarkedHEdge, 4]] = True
        isMarkedCell = np.zeros(NC_all + nn, dtype=np.bool_)
        isMarkedCell[halfedge[isMarkedHEdge, 1]] = True
        isMarkedCell[:cstart] = False
        # Number of interior cells that survive
        nc = int(np.sum(~isMarkedCell[cstart:NC_all]))

        ncl = np.zeros(NN, dtype=self.itype)
        ncl[halfedge[:, 0]] = clevel[halfedge[:, 1]] - 1
        # Only touch interior entries in clevel; outer/hole stay at level 0
        interior_del = isMarkedCell[cstart:NC_all]
        clevel_int = clevel[cstart:NC_all]
        clevel_new_int = clevel_int[~interior_del]
        clevel_new_int = np.concatenate([clevel_new_int, ncl[isRNode]])
        # Replace celldata['level'] wholesale
        clevel_new_full = np.concatenate(
            [clevel[:cstart], clevel_new_int]).astype(self.itype)
        self.celldata['level'] = DynamicArray(clevel_new_full, dtype=self.itype)
        clevel = self.celldata['level']

        # Build cidxmap of length NC_all: interior cells get contiguous ids
        # starting at cstart, "removed" cells map to their merged-into id.
        cidxmap = np.arange(NC_all)
        nidxmap = np.arange(NN)
        nidxmap[isRNode] = range(cstart + nc, cstart + nc + nn)
        isRHEdge = isRNode[halfedge[:, 0]]
        cidxmap[halfedge[isRHEdge, 1]] = nidxmap[halfedge[isRHEdge, 0]]
        cidxmap[cstart:cstart + (NC_all - cstart)][~interior_del] = \
            np.arange(cstart, cstart + nc)
        halfedge[:, 1] = cidxmap[halfedge[:, 1]]

        if 'numrefine' in options:
            num0 = np.zeros(nc + nn) - 10000
            num = options['numrefine']
            num[interior_del] += 1
            np.maximum.at(num0, cidxmap[cstart:NC_all], num)
            options['numrefine'] = num0

        nex = halfedge[:, 2]
        pre = halfedge[:, 3]
        opp = halfedge[:, 4]
        flag1 = isMarkedHEdge[nex]
        halfedge[flag1, 2] = nex[opp[nex[flag1]]]
        flag2 = isMarkedHEdge[pre]
        halfedge[flag2, 3] = pre[opp[pre[flag2]]]

        self.delete_entity(isMarkedHEdge, etype='halfedge')
        self.delete_entity(isRNode, etype='node')
        halfedge.adjust_size(isMarkedHEdge)
        hlevel.adjust_size(isMarkedHEdge)
        node.adjust_size(isRNode)

        if self.subdomain is not None:
            # Rebuild subdomain: keep [0, cstart) plus surviving interior
            # entries plus one per merged-in node (inherit any parent value).
            sd_arr = self.subdomain[:]
            sd_new_int = sd_arr[cstart:NC_all][~interior_del]
            # Inherit +1 default for the newly merged interior cells
            sd_new_add = np.ones(nn, dtype=self.itype)
            sd_new_full = np.concatenate(
                [sd_arr[:cstart], sd_new_int, sd_new_add]).astype(self.itype)
            self.subdomain = DynamicArray(sd_new_full, dtype=self.itype)

        self._reinit_after_edit()
        return isMarkedHEdge, isRNode, newNode

    def refine_poly(self, isMarkedCell=None, options=None):
        options = options or {'disp': True}
        clevel = self.celldata['level']
        hlevel = self.halfedgedata['level']
        halfedge = self.halfedge
        NC_all = self.number_of_all_cells()
        cstart = self.cellstart

        if isMarkedCell is None:
            isMarkedCell = np.ones(NC_all, dtype=np.bool_)
        else:
            isMarkedCell = np.asarray(isMarkedCell).astype(np.bool_)
            if isMarkedCell.size == self.number_of_cells():
                full = np.zeros(NC_all, dtype=np.bool_)
                full[cstart:] = isMarkedCell
                isMarkedCell = full
            else:
                isMarkedCell = isMarkedCell.copy()
        isMarkedCell[:cstart] = False

        isMarkedHEdge = self.mark_halfedge(isMarkedCell, method='poly')

        opp = halfedge[:, 4]
        cidx = halfedge[:, 1]
        # clevel is NC_all length so cidx directly indexes it
        flag = (clevel[cidx] + hlevel[:] + 1 == clevel[cidx[opp]]) & isMarkedCell[cidx]
        mark0 = np.where(flag)[0]
        mark1 = np.where(isMarkedHEdge)[0]

        isMarkedHEdge[halfedge[isMarkedHEdge, 4]] = True
        self.refine_halfedge(isMarkedHEdge)

        NHE = self.number_of_halfedges()
        isStartHEdge = np.zeros(NHE, dtype=np.bool_)
        isStartHEdge[mark1] = True
        isStartHEdge[halfedge[mark0, 2]] = True

        self._refine_poly_cell_(isMarkedCell, isStartHEdge, options=options)

    def coarsen_poly(self, isMarkedCell, i=0, options=None):
        options = options or {'disp': True}
        NN = self.number_of_nodes()
        NHE = self.number_of_halfedges()
        hlevel = self.halfedgedata['level']
        clevel = self.celldata['level']
        NC_all = self.number_of_all_cells()
        cstart = self.cellstart

        halfedge = self.halfedge

        isMarkedCell = np.asarray(isMarkedCell).astype(np.bool_)
        if isMarkedCell.size == self.number_of_cells():
            full = np.zeros(NC_all, dtype=np.bool_)
            full[cstart:] = isMarkedCell
            isMarkedCell = full
        else:
            isMarkedCell = isMarkedCell.copy()
        isMarkedCell[:cstart] = False

        # propagate marks to satisfy the "adjacent levels differ by <=1" rule
        while True:
            opp = halfedge[:, 4]
            isRNode = np.ones(NN, dtype=np.bool_)
            flag = (hlevel == hlevel[opp]) & (
                clevel[halfedge[:, 1]] > 0) & (opp != np.arange(NHE))
            flag = flag & isMarkedCell[halfedge[:, 1]]
            np.logical_and.at(isRNode, halfedge[:, 0], flag)
            flag = isRNode[halfedge[:, 0]]
            isMarkedCell[:] = False
            isMarkedCell[halfedge[flag, 1]] = True
            isMarkedCell[:cstart] = False

            isMarked = isMarkedCell[halfedge[:, 1]] & (~isMarkedCell[halfedge[opp, 1]])
            isMarked = isMarked & (clevel[halfedge[opp, 1]] > clevel[halfedge[:, 1]])
            isMarkedCell[halfedge[isMarked, 1]] = False
            if np.all(~isMarked):
                break

        isRNode = np.ones(NN, dtype=np.bool_)
        flag = (hlevel == hlevel[halfedge[:, 4]]) & (clevel[halfedge[:, 1]] > 0) & (
            halfedge[:, 4] != np.arange(NHE))
        np.logical_and.at(isRNode, halfedge[:, 0], flag)

        self._coarsen_poly_cell_(isMarkedCell, isRNode, options=options)

        NHE = self.number_of_halfedges()
        halfedge = self.halfedge
        nex = halfedge[:, 2]
        opp = halfedge[:, 4]
        flag = (opp[nex[opp[nex]]] == np.arange(NHE)) | (halfedge[:, 4] == np.arange(NHE))
        flag = flag & (hlevel[:] == 0) & (hlevel[halfedge[:, 2]] > 0)
        self.coarsen_halfedge(flag)

    # ------------------------------------------------------------------
    # Interface cutting  (Stage 4)
    # ------------------------------------------------------------------

    def cut_interface(self, interface, keep_feature=False):
        """Cut the mesh with a signed-distance function ``interface(x)``.

        Each cell must intersect the interface in exactly two points or not at
        all; every intersected cell is split into two cells. The child cell
        with negative ``interface`` value gets a new cell id, the other keeps
        the parent's id.

        Requires the mesh to have subdomain information so cell ownership can
        be tracked.
        """
        NN = self.number_of_nodes()
        NC = self.number_of_all_cells()

        node = self.node
        halfedge = self.halfedge
        isMainHEdge = self.main_halfedge_flag()

        phiValue = interface(node[:])
        edge = self.entity('edge')
        isCutHEdge = phiValue[halfedge[:, 0]] * phiValue[halfedge[halfedge[:, 3], 0]] < 0
        cutHEdge, = np.where(isCutHEdge & isMainHEdge)
        cutEdge = self.halfedge_to_edge(cutHEdge)

        e0 = node[edge[cutEdge, 0]]
        e1 = node[edge[cutEdge, 1]]
        cutNode = find_cut_point(interface, e0.copy(), e1.copy())

        self.refine_halfedge(isCutHEdge, newnode=cutNode)

        NHE = self.number_of_halfedges()
        newHE = np.where(halfedge[:, 0] >= NN)[0]

        cen = 0.5 * (
            self._node_view()[halfedge[newHE, 0]]
            + self._node_view()[halfedge[halfedge[newHE, 3], 0]]
        )
        isinnewHE = interface(cen) < 0
        newHEin = newHE[isinnewHE]
        newHEout = newHE[~isinnewHE]

        idx = np.argsort(halfedge[newHEout, 1])
        newHE[::2] = newHEout[idx]
        idx = np.argsort(halfedge[newHEin, 1])
        newHE[1::2] = newHEin[idx]
        newHE = newHE.reshape(-1, 2)

        ne = len(newHE)
        self.number_cut_cell = ne

        # Capture parent cell ids for later subdomain assignment
        parent_cell = halfedge[newHE[:, 1], 1].copy()

        halfedgeNew = halfedge.increase_size(ne * 2)
        halfedgeNew[:ne, 0] = halfedge[newHE[:, 1], 0]
        halfedgeNew[:ne, 1] = halfedge[newHE[:, 1], 1]
        halfedgeNew[:ne, 2] = halfedge[newHE[:, 1], 2]
        halfedgeNew[:ne, 3] = newHE[:, 0]
        halfedgeNew[:ne, 4] = np.arange(NHE + ne, NHE + ne * 2)

        halfedgeNew[ne:, 0] = halfedge[newHE[:, 0], 0]
        halfedgeNew[ne:, 1] = np.arange(NC, NC + ne)
        halfedgeNew[ne:, 2] = halfedge[newHE[:, 0], 2]
        halfedgeNew[ne:, 3] = newHE[:, 1]
        halfedgeNew[ne:, 4] = np.arange(NHE, NHE + ne)

        halfedge[halfedge[newHE[:, 0], 2], 3] = np.arange(NHE + ne, NHE + ne * 2)
        halfedge[halfedge[newHE[:, 1], 2], 3] = np.arange(NHE, NHE + ne)
        halfedge[newHE[:, 0], 2] = np.arange(NHE, NHE + ne)
        halfedge[newHE[:, 1], 2] = np.arange(NHE + ne, NHE + ne * 2)

        isNotOK = np.ones(ne, dtype=np.bool_)
        current = np.arange(NHE + ne, NHE + ne * 2)
        while np.any(isNotOK):
            halfedge[current[isNotOK], 1] = np.arange(NC, NC + ne)[isNotOK]
            current[isNotOK] = halfedge[current[isNotOK], 2]
            isNotOK = current != np.arange(NHE + ne, NHE + ne * 2)

        # subdomain: append entries for the newly created ``inside`` cells;
        # they inherit their parent's subdomain id, so callers can distinguish
        # inside/outside of the interface via the geometric centroid rather
        # than via subdomain marker.
        if self.subdomain is not None:
            new_sd = self.subdomain.increase_size(ne)
            new_sd[:] = self.subdomain[parent_cell]

        self._reinit_after_edit()
        self._init_level_info()

    @classmethod
    def from_interface_cut_box(cls, interface, box, nx=10, ny=10,
                               keep_feature=False):
        """Cartesian quad mesh -> cut by ``interface`` in one shot."""
        N = (nx + 1) * (ny + 1)
        NC = nx * ny
        node = np.zeros((N, 2))
        X, Y = np.mgrid[box[0]:box[1]:complex(0, nx + 1),
                        box[2]:box[3]:complex(0, ny + 1)]
        node[:, 0] = X.flatten()
        node[:, 1] = Y.flatten()

        idx = np.arange(N).reshape(nx + 1, ny + 1)
        cell = np.zeros((NC, 4), dtype=np.int_)
        cell[:, 0] = idx[0:-1, 0:-1].flat
        cell[:, 1] = idx[1:, 0:-1].flat
        cell[:, 2] = idx[1:, 1:].flat
        cell[:, 3] = idx[0:-1, 1:].flat

        mesh = cls._from_quad_cells(node, cell)
        mesh.cut_interface(interface, keep_feature=keep_feature)
        return mesh

    @classmethod
    def _from_quad_cells(cls, node, cell):
        """Build a subdomain-aware halfedge mesh from (node, quadcell)."""
        NC = cell.shape[0]
        NV = cell.shape[1]

        halfedge = np.zeros((NC * NV, 5), dtype=np.int_)
        for c in range(NC):
            for i in range(NV):
                he = c * NV + i
                halfedge[he, 0] = cell[c, (i + 1) % NV]
                halfedge[he, 1] = c + 1  # cellstart=1, cell 0 is unbounded outer
                halfedge[he, 2] = c * NV + (i + 1) % NV
                halfedge[he, 3] = c * NV + (i - 1) % NV
                halfedge[he, 4] = he

        edge_map = {}
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

        subdomain = np.ones(NC + 1, dtype=np.int_)
        subdomain[0] = 0
        return cls(node, halfedge, subdomain=subdomain, NV=NV)


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def find_cut_point(phi, p0, p1, max_iter=100):
    """Bisection root-finder for interface intersection on a segment (p0, p1).

    Signed-distance / level-set function ``phi`` is evaluated on both endpoints;
    the endpoint whose sign matches the midpoint is moved to the midpoint each
    iteration.  Vectorised over the leading axis of ``p0, p1``.
    """
    p0 = np.array(p0, dtype=float, copy=True)
    p1 = np.array(p1, dtype=float, copy=True)
    cutPoint = 0.5 * (p0 + p1)
    phi0 = phi(p0)
    phi1 = phi(p1)
    phic = phi(cutPoint)

    vec = p1 - p0
    h = np.sqrt(np.sum(vec ** 2, axis=1))
    eps = np.finfo(p0.dtype).eps
    tol = np.sqrt(eps) * h * h
    isNotOK = (h > tol) & (phic != 0)
    for _ in range(max_iter):
        if not np.any(isNotOK):
            break
        cutPoint[isNotOK] = 0.5 * (p0[isNotOK] + p1[isNotOK])
        phic[isNotOK] = phi(cutPoint[isNotOK])
        isLeft = np.zeros_like(isNotOK)
        isRight = np.zeros_like(isNotOK)
        isLeft[isNotOK] = phi0[isNotOK] * phic[isNotOK] > 0
        isRight[isNotOK] = phi1[isNotOK] * phic[isNotOK] > 0
        p0[isLeft] = cutPoint[isLeft]
        p1[isRight] = cutPoint[isRight]
        phi0[isLeft] = phic[isLeft]
        phi1[isRight] = phic[isRight]
        h[isNotOK] /= 2
        isNotOK[isNotOK] = (h[isNotOK] > tol[isNotOK]) & (phic[isNotOK] != 0)
    return cutPoint
