"""Hu-Zhang 元角点松弛 wrapper（fracturex 侧实现，不修改 fealpy）。

Background
==========
fealpy 自带的 ``HuZhangFESpace2d(use_relaxation=True)`` 只实现 [HM18] §4.2 的
"两单元 (m=2) + 两条入边端点 trace 接管"模式，并且实际上 ``cell_to_dof`` 的顶点段
没有暴露新增 DOF 的注入路径——结果是 ``_filter_active_corners_by_support`` 会把
**所有** NN 角点过滤掉（NCP=0），松弛实际上从未生效。

诊断脚本：``fracturex/tests/lshape_corner_diagnose.py``。

本模块绕开上面这个限制，按 [HM18] §4 的统一思路：

  在每个 NN 角点 x_c（被 m 个三角形 K_1,…,K_m 共享，m-1 条内部边
  e_1,…,e_{m-1} 经过 x_c），把"顶点处共享的 3 个对称张量分量"放松为
  "每个 cell 上独立的 3 个 cartesian 分量"，再用内部边的法向连续条件
  σ^{(j)} n_j = σ^{(j+1)} n_j（每条 2 个标量约束）压缩回 m+2 个真自由度。

公式 (m+2 = 3m - 2(m-1)) 在 m=2 时退化为 4，正好是 [HM18] §4.2；m=3 时是 5，
对应 §4.4。

Architecture
============
两套 DOF：

  - **unconstrained DOF**（``gdof_unc``）：
    在 fealpy 原 ``gdof_base`` 之外，每个 NN 角点追加 3(m-1) 个 cell-local 顶点
    DOF。``cell_to_dof_unc`` 在角点所在 cell 的"该顶点 3 列"上替换为这些新 DOF。

  - **relaxed DOF**（``gdof_rel``）：
    比 ``gdof_base`` 多 ``sum_c (m_c - 1)`` 个：每个角点多 m-1 个真自由度。

变换矩阵 ``TM ∈ R^{gdof_unc × gdof_rel}``：非角点 DOF 处恒等，角点块由内部边法
向连续条件 C z = 0（C ∈ R^{2(m-1) × 3m}）的零空间给出，scipy/numpy 的 SVD 算得。

Assembler pipeline
==================
  M, B = assemble(base_space, with cell_to_dof = cell_to_dof_unc)
  M2  = TM.T @ M @ TM
  B2  = TM.T @ B
  ...求解 X = [σ_rel; u]
  σ_unc = TM @ X[:gdof_rel]      # 需要回 cell-local 顶点 DOF 时
  σ_phys（按节点）= 在每个 cell 上 σ_unc[cell_to_dof_unc[K]] 直接读

Usage
=====
::

    from fracturex.discretization.huzhang_corner_relax import HuZhangCornerRelax
    relax = HuZhangCornerRelax(mesh, p=3, isNedge=isNedge)
    space = relax.base_space       # fealpy 原始空间（use_relaxation=False）
    c2d   = relax.cell_to_dof_unc  # 用这个替换 base.cell_to_dof()
    TM    = relax.TM               # scipy.sparse, (gdof_unc, gdof_rel)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from fealpy.backend import backend_manager as bm
from fealpy.functionspace.huzhang_fe_space_2d import HuZhangFESpace2d


@dataclass
class _CornerInfo:
    """单个 NN 角点的几何与 DOF 拓扑（unconstrained 视图）。"""
    nid: int                          # 角点节点 id
    cells: np.ndarray                 # incident cells, fan 序, shape (m,)
    local_vid: np.ndarray             # 每个 cell 上该角点的 local vertex idx, shape (m,)
    interior_edges: List[Tuple[int, int, np.ndarray]]
    # interior_edges[j] = (k_j, k_{j+1}, n_j)：K_cells[k_j] 与 K_cells[k_{j+1}]
    # 共享的内部边的单位法向 n_j（从 k_j 指向 k_{j+1}，方向规约任选）
    unc_dofs: np.ndarray              # 该角点的全部 3m 个 unconstrained 顶点 DOF, shape (m, 3)
    rel_dofs: np.ndarray              # 该角点的 m+2 个 relaxed DOF, shape (m+2,)


class HuZhangCornerRelax:
    """Build NN-corner relaxation overlay over an unrelaxed HuZhangFESpace2d."""

    def __init__(self, mesh, p: int, *, isNedge: np.ndarray,
                 base_space: Optional[HuZhangFESpace2d] = None,
                 angle_tol: float = 1e-10,
                 verbose: bool = False):
        """Build the relaxation overlay.

        Args:
            mesh: TriangleMesh (geometry/topology will be queried).
            p: HuZhang polynomial order.
            isNedge: (NE,) bool, marks Neumann edges (essential boundary on σ·n).
            base_space: optional pre-built HuZhangFESpace2d(use_relaxation=False);
                if None, this class builds one with the given mesh/p/isNedge.
            angle_tol: collinearity tolerance when detecting geometric corners.
            verbose: if True, print per-corner geometry/DOF diagnostics.
        """
        self.mesh = mesh
        self.p = int(p)
        self.isNedge = np.asarray(isNedge, dtype=bool)
        self.verbose = verbose
        self.angle_tol = float(angle_tol)

        if base_space is None:
            base_space = HuZhangFESpace2d(
                mesh, p=self.p, use_relaxation=False, bd_stress=self.isNedge,
            )
        self.base_space = base_space
        self.gdof_base = int(base_space.number_of_global_dofs())

        # 基础数据
        self._node = np.asarray(mesh.entity('node'))
        self._edge = np.asarray(mesh.entity('edge'))
        self._cell = np.asarray(mesh.entity('cell'))
        self._isBd = np.asarray(mesh.boundary_edge_flag())
        self._NN = mesh.number_of_nodes()
        self._NE = mesh.number_of_edges()
        self._NC = mesh.number_of_cells()

        # 1) 找出所有 NN 角点
        self.corners: List[_CornerInfo] = self._detect_nn_corners()

        # 2) 给每个角点分配 unconstrained + relaxed DOF id
        self._assign_dof_ids()

        # 3) 构造 cell_to_dof_unc
        self.cell_to_dof_unc = self._build_cell_to_dof_unc()

        # 4) 构造 TM
        self.TM = self._build_TM()

    # =====================================================================
    # 1) 角点识别
    # =====================================================================
    def _detect_nn_corners(self) -> List[_CornerInfo]:
        """Detect NN corners and gather per-corner topology.

        A node is an NN corner if it has ≥2 incident Neumann boundary edges
        whose tangent directions are not antiparallel (genuine corner).
        """
        # boundary node -> incident boundary edges (as global eid list)
        adj: List[List[int]] = [[] for _ in range(self._NN)]
        bd_eids = np.where(self._isBd)[0]
        for eid in bd_eids:
            a, b = int(self._edge[eid, 0]), int(self._edge[eid, 1])
            adj[a].append(int(eid))
            adj[b].append(int(eid))

        corners: List[_CornerInfo] = []
        for nid in range(self._NN):
            inc = adj[nid]
            if len(inc) < 2:
                continue

            # 至少 2 条 ΓN 边
            inc_N = [e for e in inc if bool(self.isNedge[e])]
            if len(inc_N) < 2:
                continue

            # 几何角点：找两条不反向共线的 ΓN 边即可
            if not self._has_noncollinear_pair(inc_N, nid):
                continue

            info = self._build_corner_topology(nid)
            if info is None:
                continue
            # m=1 的角点没有内部边 → 没有 traction inconsistency，无需松弛
            if len(info.cells) < 2:
                if self.verbose:
                    print(f"[corner-relax] skip nid={nid}: m=1 (no interior edge)")
                continue
            corners.append(info)

        if self.verbose:
            print(f"[corner-relax] detected NN corners = {len(corners)}; "
                  f"node ids = {[c.nid for c in corners]}")
        return corners

    def _has_noncollinear_pair(self, eids: Sequence[int], nid: int) -> bool:
        if len(eids) < 2:
            return False
        v0 = self._edge_dir_from_node(eids[0], nid)
        for k in range(1, len(eids)):
            v1 = self._edge_dir_from_node(eids[k], nid)
            if abs(float(np.dot(v0, v1)) + 1.0) > self.angle_tol:
                return True
        return False

    def _edge_dir_from_node(self, eid: int, nid: int) -> np.ndarray:
        a, b = int(self._edge[eid, 0]), int(self._edge[eid, 1])
        other = b if a == nid else a
        v = self._node[other] - self._node[nid]
        return v / (np.linalg.norm(v) + 1e-30)

    def _build_corner_topology(self, nid: int) -> Optional[_CornerInfo]:
        """Order incident cells in a fan around nid; collect interior-edge normals."""
        inc_cells = np.where(np.any(self._cell == nid, axis=1))[0]
        m = len(inc_cells)
        if m < 1:
            return None

        # 每个 cell 上 nid 的 local vertex idx
        local_vid = np.array([int(np.where(self._cell[c] == nid)[0][0]) for c in inc_cells], dtype=np.int32)

        # 通过"共享内部边"邻接关系把 fan 链接起来
        cell_edges = np.asarray(self.mesh.cell_to_edge())  # (NC, 3)
        # 对该 nid 周围的 cells 构造邻接：cell_a -- internal_edge --> cell_b
        # 同时记录 cell -> 哪些 ΓN 边在 nid 上（开口入口/出口）
        cell_to_neighbors = {int(c): [] for c in inc_cells}  # cell -> [(neighbor_cell, eid, n)]
        cell_to_bdedges = {int(c): [] for c in inc_cells}    # cell -> [eid] (boundary edges through nid)
        for c in inc_cells:
            for eid in cell_edges[c]:
                if nid not in self._edge[eid]:
                    continue
                if self._isBd[eid]:
                    cell_to_bdedges[int(c)].append(int(eid))
                    continue
                # 内部边：找另一侧 cell
                for c2 in inc_cells:
                    if int(c2) == int(c):
                        continue
                    if eid in cell_edges[c2]:
                        tau = self._edge_dir_from_node(int(eid), nid)
                        n_ij = np.array([-tau[1], tau[0]])
                        c2_centroid = self._node[self._cell[int(c2)]].mean(axis=0)
                        if np.dot(n_ij, c2_centroid - self._node[nid]) < 0:
                            n_ij = -n_ij
                        cell_to_neighbors[int(c)].append((int(c2), int(eid), n_ij))
                        break

        # 找开口 cell：含 ≥1 条 ΓN 边经过 nid 的 cell
        # 真正的 NN 角点：恰有 2 个开口 cell，连同 m-1 条内部边构成一条链
        opening_cells = [c for c, bds in cell_to_bdedges.items() if len(bds) >= 1]
        if len(opening_cells) < 2:
            if self.verbose:
                print(f"[corner-relax] nid={nid}: only {len(opening_cells)} opening cells; skip")
            return None

        # 从一个 opening cell 走链，按"最多一次访问"找到另一个 opening cell
        start = opening_cells[0]
        chain = [start]
        visited = {start}
        interior_edges: List[Tuple[int, int, np.ndarray]] = []
        cur = start
        while True:
            nbrs = [(c2, eid, n) for (c2, eid, n) in cell_to_neighbors[cur] if c2 not in visited]
            if not nbrs:
                break
            # 链应当唯一；若有分叉说明非流形
            if len(nbrs) > 1:
                if self.verbose:
                    print(f"[corner-relax] nid={nid}: branching fan at cell {cur}; skip")
                return None
            c2, eid, n = nbrs[0]
            interior_edges.append((len(chain) - 1, len(chain), n))
            chain.append(c2)
            visited.add(c2)
            cur = c2

        # 链终点必须也是 opening cell
        if cur not in opening_cells or len(chain) != m:
            if self.verbose:
                print(f"[corner-relax] nid={nid}: fan not a simple chain "
                      f"(len(chain)={len(chain)} vs m={m}, end={'opening' if cur in opening_cells else 'interior'}); skip")
            return None

        # 用 chain 序覆盖原 inc_cells / local_vid
        chain_arr = np.array(chain, dtype=np.int64)
        local_vid = np.array([int(np.where(self._cell[c] == nid)[0][0]) for c in chain_arr], dtype=np.int32)
        inc_cells = chain_arr

        return _CornerInfo(
            nid=int(nid),
            cells=inc_cells.astype(np.int64),
            local_vid=local_vid,
            interior_edges=interior_edges,
            unc_dofs=np.empty((m, 3), dtype=np.int64),  # 稍后填
            rel_dofs=np.empty((m + 2,), dtype=np.int64),
        )

    def _collect_node_edges(self, nid: int):
        rows = np.where(np.any(self._edge == nid, axis=1))[0]
        return rows.tolist()

    # =====================================================================
    # 2) DOF id 分配
    # =====================================================================
    def _assign_dof_ids(self):
        """Assign unconstrained + relaxed DOF ids.

        Layouts (both 1D index ranges, disjoint between unc and rel):

        unc space (size = gdof_unc):
          - [0, gdof_base): base ids unchanged. Corner cell-0 reuses its 3 base
            node DOFs here (so cell-to-dof for cell 0 needs no change).
          - [gdof_base, gdof_unc): 3*(m-1) fresh ids per corner for cells
            1..m-1's cell-local node DOFs.

        rel space (size = gdof_rel):
          - same-id mapping for *non-corner* base DOFs, but corner node DOFs
            are REMOVED from the rel space (they become cell-local) and replaced
            by m+2 fresh corner-relaxed DOFs.
          - To keep ids contiguous and disjoint, we lay out:
              [0, gdof_base): same id as base IF the base id is non-corner;
                              corner-node-3 base ids are LEFT EMPTY (Q.T then M2
                              will have zero rows/cols there, handled by row/col
                              elimination at solve time).
              [gdof_base, gdof_rel): fresh corner-relaxed ids (m+2 per corner).

        Leaving "empty slots" inside [0, gdof_base) keeps non-corner DOFs
        addressed identically and avoids any global re-permutation. The empty
        slots are eliminated as Dirichlet-fixed-to-zero before solving.
        """
        node2dof = np.asarray(self.base_space.dof.node_to_internal_dof())  # (NN, 3)

        next_unc = self.gdof_base
        next_rel = self.gdof_base

        empty_rel_slots: List[int] = []  # corner-node base ids that became "empty" in rel
        for c in self.corners:
            m = len(c.cells)
            c.unc_dofs = np.empty((m, 3), dtype=np.int64)
            c.unc_dofs[0] = node2dof[c.nid]
            for k in range(1, m):
                c.unc_dofs[k] = np.arange(next_unc, next_unc + 3, dtype=np.int64)
                next_unc += 3

            c.rel_dofs = np.arange(next_rel, next_rel + (m + 2), dtype=np.int64)
            next_rel += (m + 2)

            # corner-node base ids are now "empty" slots in rel space
            empty_rel_slots.extend(node2dof[c.nid].tolist())

        self.gdof_unc = next_unc
        self.gdof_rel = next_rel
        self.empty_rel_slots = np.array(sorted(set(empty_rel_slots)), dtype=np.int64)

        if self.verbose:
            print(f"[corner-relax] gdof_base={self.gdof_base}, "
                  f"gdof_unc={self.gdof_unc}, gdof_rel={self.gdof_rel}, "
                  f"empty_rel_slots={len(self.empty_rel_slots)}")

    # =====================================================================
    # 3) 扩展 cell_to_dof
    # =====================================================================
    def _build_cell_to_dof_unc(self) -> np.ndarray:
        """Replace per-vertex 3 DOFs at NN-corner cells with cell-local copies."""
        c2d = np.asarray(self.base_space.cell_to_dof()).copy()
        for c in self.corners:
            for k, cell_id in enumerate(c.cells):
                v = int(c.local_vid[k])
                c2d[cell_id, 3 * v: 3 * v + 3] = c.unc_dofs[k]
        return c2d

    # =====================================================================
    # 4) TM 构造
    # =====================================================================
    def _build_TM(self) -> sp.csr_matrix:
        """Build TM ∈ R^{gdof_unc × gdof_rel}.

        Layout (after disjoint id allocation):
        - non-corner DOFs with id d ∈ [0, gdof_base):
          identity in BOTH spaces, so TM[d, d] = 1.
        - corner block: all 3m unc DOFs are mapped from m+2 rel DOFs via the
          interior-edge normal-continuity null space.
        """
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []

        # 角点 unc id 集合（既含 cell-0 沿用的 base 3 DOF，也含 cells 1..m-1 新分配的）
        corner_unc_set = set()
        for c in self.corners:
            corner_unc_set.update(c.unc_dofs.reshape(-1).tolist())

        # 非角点 unc DOF：仅 [0, gdof_base) 段，逐个恒等映射
        for d in range(self.gdof_base):
            if d in corner_unc_set:
                continue
            rows.append(d); cols.append(d); vals.append(1.0)

        # 角点局部块
        for c in self.corners:
            m = len(c.cells)
            C = self._build_normal_continuity_matrix(c)
            N = self._nullspace(C)
            assert N.shape == (3 * m, m + 2), (
                f"nullspace shape mismatch: got {N.shape}, expected (3m, m+2) = ({3*m},{m+2})"
            )
            unc_ids = c.unc_dofs.reshape(-1)
            rel_ids = c.rel_dofs
            for i in range(3 * m):
                for j in range(m + 2):
                    if abs(N[i, j]) > 1e-14:
                        rows.append(int(unc_ids[i]))
                        cols.append(int(rel_ids[j]))
                        vals.append(float(N[i, j]))

        TM = sp.csr_matrix((vals, (rows, cols)),
                           shape=(self.gdof_unc, self.gdof_rel))
        return TM

    @staticmethod
    def _build_normal_continuity_matrix(c: _CornerInfo) -> np.ndarray:
        """Per-corner constraint matrix.

        Vector z ∈ R^{3m} is ordered as (S^{(1)}_xx, S^{(1)}_xy, S^{(1)}_yy,
        S^{(2)}_xx, ..., S^{(m)}_yy).

        For interior edge between cells k_j and k_{j+1} with normal n_j:
            S^{(k_j)} n_j = S^{(k_{j+1})} n_j
        gives 2 scalar rows:
            row_x: S_xx^{(k_j)} n_x + S_xy^{(k_j)} n_y
                 - S_xx^{(k_{j+1})} n_x - S_xy^{(k_{j+1})} n_y = 0
            row_y: S_xy^{(k_j)} n_x + S_yy^{(k_j)} n_y
                 - S_xy^{(k_{j+1})} n_x - S_yy^{(k_{j+1})} n_y = 0
        """
        m = len(c.cells)
        rows = []
        for (k_j, k_jp1, n) in c.interior_edges:
            nx, ny = float(n[0]), float(n[1])
            r = np.zeros(3 * m)
            r[3 * k_j + 0] += nx
            r[3 * k_j + 1] += ny
            r[3 * k_jp1 + 0] -= nx
            r[3 * k_jp1 + 1] -= ny
            rows.append(r)
            r = np.zeros(3 * m)
            r[3 * k_j + 1] += nx
            r[3 * k_j + 2] += ny
            r[3 * k_jp1 + 1] -= nx
            r[3 * k_jp1 + 2] -= ny
            rows.append(r)
        return np.array(rows) if rows else np.zeros((0, 3 * m))

    @staticmethod
    def _nullspace(C: np.ndarray) -> np.ndarray:
        """Compute an orthonormal basis of null(C) via SVD."""
        n = C.shape[1]
        if C.shape[0] == 0:
            return np.eye(n)
        U, s, Vh = np.linalg.svd(C, full_matrices=True)
        tol = max(C.shape) * (s.max() if s.size > 0 else 0.0) * np.finfo(float).eps
        rank = int(np.sum(s > tol))
        N = Vh[rank:].T
        return N

    # =====================================================================
    # 投影矩阵 Q：直接把 gdof_base × gdof_base 的装配矩阵压到 gdof_rel 维
    # =====================================================================
    @property
    def Q(self) -> sp.csr_matrix:
        """Return Q = P^T @ TM ∈ R^{gdof_base × gdof_rel}.

        ``P ∈ R^{gdof_unc × gdof_base}`` is the prolongation that duplicates each
        corner's 3 base DOFs into each surrounding cell's 3 cell-local DOFs.
        Concretely, ``P[i, j] = 1`` iff unc DOF i corresponds (by component) to
        base DOF j.

        Use ``M2 = Q.T @ M_base @ Q`` and ``B2 = Q.T @ B_base`` directly.
        """
        if hasattr(self, "_Q_cache"):
            return self._Q_cache
        node2dof = np.asarray(self.base_space.dof.node_to_internal_dof())  # (NN, 3)

        rows: List[int] = []  # base indices
        cols: List[int] = []  # unc indices
        vals: List[float] = []

        # 角点的 base node DOF id 集合（即 cell 0 沿用的 3 个 id）
        corner_base = set()
        for c in self.corners:
            corner_base.update(node2dof[c.nid].tolist())

        # 非角点 base DOF：P[d, d] = 1（unc id == base id）
        for d in range(self.gdof_base):
            if d in corner_base:
                continue
            rows.append(d); cols.append(d); vals.append(1.0)

        # 角点：每个 cell 上 3 个 cell-local unc DOF -> 该节点的 3 个 base DOF
        for c in self.corners:
            base3 = node2dof[c.nid]
            for k in range(len(c.cells)):
                for comp in range(3):
                    unc_id = int(c.unc_dofs[k, comp])
                    base_id = int(base3[comp])
                    rows.append(base_id); cols.append(unc_id); vals.append(1.0)

        P = sp.csr_matrix((vals, (rows, cols)),
                          shape=(self.gdof_base, self.gdof_unc))
        self._Q_cache = P @ self.TM   # (gdof_base, gdof_rel)
        return self._Q_cache

    # =====================================================================
    # 角点本质边界条件（fealpy 不能直接处理 NN 角点两边 traction 不相容）
    # =====================================================================
    def apply_corner_essential_bc_unc(self, gd_callable) -> Tuple[np.ndarray, np.ndarray]:
        """Build corner-localized essential BC values in the unconstrained space.

        For each NN corner c and each ΓN-end cell (fan-end k=0 and k=m-1) at
        that corner:

          - identify the ΓN edge that end cell touches through x_c;
          - evaluate σ_g at x_c via ``gd_callable`` (with the point on THAT edge
            so if gd depends on which edge, it picks the right side);
          - lock the two node-trace DOFs on that end cell (σ_xx, σ_xy) —
            corresponding to fealpy's cartesian nsframe.

        This is the "each cell gets its own edge traction" essential lock that
        [HM18] §4.2 (4.1) prescribes at NN corners.

        Why only 2 of 3:
          For an interior boundary edge with arbitrary normal n=(n_x, n_y), the
          essential trace contains n^T σ n and n^T σ t, which involve all 3
          cartesian components in linear combinations. The HuZhang vertex DOFs
          are however cartesian (σ_xx, σ_xy, σ_yy). At an NN corner with two
          ΓN edges, locking σ_xx and σ_xy is equivalent (in the cartesian
          basis) to fixing two scalar components; the third (σ_yy) follows from
          the global mixed system. This matches what fealpy's set_essential_bc_v2
          does in the base space, lifted into per-cell copies here.
        """
        uh_unc = np.zeros(self.gdof_unc)
        isbd_unc = np.zeros(self.gdof_unc, dtype=bool)
        cell_edges = np.asarray(self.mesh.cell_to_edge())
        for c in self.corners:
            for end_k in (0, len(c.cells) - 1):
                end_cell = int(c.cells[end_k])
                for eid in cell_edges[end_cell]:
                    if c.nid not in self._edge[int(eid)]:
                        continue
                    if not self.isNedge[int(eid)]:
                        continue
                    # 用**边上的近端点**（不是角点本身）评估 gd_callable，避免"角点是两边
                    # 交点时 gd 优先某一边"的歧义。取沿边从角点向内 1e-6 的点。
                    a, b = int(self._edge[int(eid), 0]), int(self._edge[int(eid), 1])
                    other = b if a == c.nid else a
                    x_c = self._node[c.nid]; x_o = self._node[other]
                    t = 1e-6
                    pt = ((1 - t) * x_c + t * x_o).reshape(1, 2)
                    sig_voigt = np.asarray(gd_callable(pt)).reshape(-1)
                    if sig_voigt.size != 3:
                        raise ValueError(f"gd must return Voigt size 3, got {sig_voigt.size}")
                    for comp in (0, 1):
                        unc_id = int(c.unc_dofs[end_k, comp])
                        uh_unc[unc_id] = float(sig_voigt[comp])
                        isbd_unc[unc_id] = True
                    break
        return uh_unc, isbd_unc

    def lift_base_bc_to_rel(self, uh_base: np.ndarray, isbd_base: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """Lift fealpy's base-space stress essential BC to the rel space.

        Non-corner base DOFs (id ∉ corner_node_3_ids) are copied identically.
        Corner-node base DOFs (id ∈ corner_node_3_ids) are interpreted as the
        target σ value at that vertex (in cartesian) — we propagate that value
        to ALL cell-local copies (cells 0..m-1) on that corner, then project
        through TM to rel via least-squares.

        Returns:
            (uh_rel, isbd_rel) of length gdof_rel.
        """
        uh_rel = np.zeros(self.gdof_rel)
        isbd_rel = np.zeros(self.gdof_rel, dtype=bool)

        node2dof = np.asarray(self.base_space.dof.node_to_internal_dof())
        corner_base = set()
        for c in self.corners:
            corner_base.update(node2dof[c.nid].tolist())

        # 1) 非角点 DOFs：恒等复制
        for d in range(self.gdof_base):
            if d in corner_base:
                continue
            if isbd_base[d]:
                uh_rel[d] = float(uh_base[d])
                isbd_rel[d] = True

        # 2) 角点：用 base 上 σ_xx, σ_xy 的真值（isbds[base3]=[T,T,F]）复制到所有 cell
        #    上的对应 unc DOF，然后投到 rel
        uh_unc = np.zeros(self.gdof_unc)
        isbd_unc = np.zeros(self.gdof_unc, dtype=bool)
        for c in self.corners:
            base3 = node2dof[c.nid]
            for comp in range(3):
                if not isbd_base[int(base3[comp])]:
                    continue
                val = float(uh_base[int(base3[comp])])
                # 复制到 m 个 cell 上的 cell-local DOF 的 comp 位置
                for k in range(len(c.cells)):
                    uid = int(c.unc_dofs[k, comp])
                    uh_unc[uid] = val
                    isbd_unc[uid] = True
        # 投到 rel
        TM_dense = self.TM
        for c in self.corners:
            unc_ids = c.unc_dofs.reshape(-1)
            rel_ids = c.rel_dofs
            active = isbd_unc[unc_ids]
            if not np.any(active):
                continue
            T_full = np.asarray(TM_dense[unc_ids][:, rel_ids].todense())
            T_act = T_full[active]
            y_act = uh_unc[unc_ids[active]]
            x_rel, *_ = np.linalg.lstsq(T_act, y_act, rcond=None)
            uh_rel[rel_ids] = x_rel
            isbd_rel[rel_ids] = True
        return uh_rel, isbd_rel

    def project_unc_bc_to_rel(self, uh_unc: np.ndarray, isbd_unc: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """Project unc-space Dirichlet data onto rel-space DOFs.

        Strategy: for each corner, collect the active unc DOFs (those with
        ``isbd_unc=True``) and solve a small least-squares problem TM[corner] x
        = uh_corner. The resulting x gives the rel-DOF values; any rel DOF not
        determined by the LSQ is left free (isbd=False).
        """
        uh_rel = np.zeros(self.gdof_rel)
        isbd_rel = np.zeros(self.gdof_rel, dtype=bool)
        TM_dense = self.TM
        for c in self.corners:
            unc_ids = c.unc_dofs.reshape(-1)
            rel_ids = c.rel_dofs
            active = isbd_unc[unc_ids]
            if not np.any(active):
                continue
            # 取角点子块
            T_full = np.asarray(TM_dense[unc_ids][:, rel_ids].todense())
            T_act = T_full[active]
            y_act = uh_unc[unc_ids[active]]
            # 最小二乘解：x_rel = pinv(T_act) y_act
            x_rel, *_ = np.linalg.lstsq(T_act, y_act, rcond=None)
            uh_rel[rel_ids] = x_rel
            # 在 rel 空间中，这 m+2 个 DOF 都被本质边界条件锁住
            isbd_rel[rel_ids] = True
        return uh_rel, isbd_rel

    @property
    def C_constraint(self) -> sp.csr_matrix:
        """Stack all per-corner normal-continuity constraint matrices.

        Returns:
            C ∈ R^{n_constraints × gdof_unc}, where each row is one scalar
            constraint ``C_i · σ_unc = 0`` from the (m-1) interior-edge
            normal-continuity equations of a corner.
        """
        if hasattr(self, "_C_cache"):
            return self._C_cache
        rows = []
        cols = []
        vals = []
        next_row = 0
        for c in self.corners:
            C_local = self._build_normal_continuity_matrix(c)  # (2(m-1), 3m)
            unc_ids = c.unc_dofs.reshape(-1)                     # (3m,)
            nrows, ncols = C_local.shape
            for r in range(nrows):
                for j in range(ncols):
                    if abs(C_local[r, j]) > 1e-14:
                        rows.append(next_row + r)
                        cols.append(int(unc_ids[j]))
                        vals.append(float(C_local[r, j]))
            next_row += nrows
        C = sp.csr_matrix((vals, (rows, cols)), shape=(next_row, self.gdof_unc))
        self._C_cache = C
        return C

    @property
    def n_constraints(self) -> int:
        return int(self.C_constraint.shape[0])

    def apply_corner_essential_bc_unc_all(self, gd_callable) -> Tuple[np.ndarray, np.ndarray]:
        """Lock 2 cartesian components (σ_xx, σ_xy) at both ΓN-end cells per corner.

        Returns:
            (uh_unc, isbd_unc): vectors of length gdof_unc. Only the corner-end
            cells (fan ends) get isbd=True on their 2 cell-local DOFs.
        """
        return self.apply_corner_essential_bc_unc(gd_callable)

    def lift_base_bc_to_unc(self, uh_base: np.ndarray, isbd_base: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """Lift fealpy's base-space stress essential BC to the unc space.

        - Non-corner base DOFs: mapped identically (unc id == base id).
        - Corner-node base DOFs: only the two **fan-end cells** (which actually
          carry ΓN edges) get the cell-local DOF locked. Middle cells (cells 1
          ..m-2) are kept FREE — their cell-local DOFs are determined by the
          normal-continuity Lagrange constraints C σ = 0 and the variational
          equation.

        This is the correct minimal essential locking: 2 ends × (σ_xx, σ_xy) =
        4 scalar constraints per corner — exactly matching the 4 scalar σ·n
        values supplied by the two ΓN edges.
        """
        uh_unc = np.zeros(self.gdof_unc)
        isbd_unc = np.zeros(self.gdof_unc, dtype=bool)
        node2dof = np.asarray(self.base_space.dof.node_to_internal_dof())
        corner_base_ids = set()
        for c in self.corners:
            corner_base_ids.update(node2dof[c.nid].tolist())
        # 非角点
        for d in range(self.gdof_base):
            if d in corner_base_ids:
                continue
            if isbd_base[d]:
                uh_unc[d] = float(uh_base[d])
                isbd_unc[d] = True
        # 角点：只在 fan-end cells（k=0 和 k=m-1）上 lock
        for c in self.corners:
            base3 = node2dof[c.nid]
            m = len(c.cells)
            for comp in range(3):
                bd = int(base3[comp])
                if not isbd_base[bd]:
                    continue
                v = float(uh_base[bd])
                for k in (0, m - 1):
                    uid = int(c.unc_dofs[k, comp])
                    uh_unc[uid] = v
                    isbd_unc[uid] = True
        return uh_unc, isbd_unc

    def l2_error_unc(self, sigma, sigma_exact, q: int = None,
                     is_unc: bool = False) -> float:
        """Compute ``|σ - σ_h|_{L^2}`` using cell-local DOFs at corners.

        Args:
            sigma: either rel-space coefficient vector (``is_unc=False``, default)
                or unc-space coefficient vector (``is_unc=True``).
        """
        mesh = self.mesh
        space = self.base_space
        if q is None:
            q = 2 * space.p + 6
        sigma_unc = np.asarray(sigma) if is_unc else self.lift_stress_unc(sigma)

        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NC = mesh.number_of_cells()
        phi = np.asarray(space.basis(bcs))               # (NC, NQ, ldof, 3) Voigt
        c2d = self.cell_to_dof_unc                       # (NC, ldof)
        coeffs = sigma_unc[c2d]                          # (NC, ldof)
        # σ_h(x) Voigt = sum_l coeffs[c, l] * phi[c, q, l, :]
        sh = np.einsum('cl, cqld -> cqd', coeffs, phi)   # (NC, NQ, 3)

        pts = np.asarray(mesh.bc_to_point(bcs))
        # bc_to_point 形状有时是 (NQ, NC, 2) — 处理两种
        if pts.shape[0] == NC:
            pass
        else:
            pts = pts.transpose(1, 0, 2)
        se = sigma_exact(pts)                            # (NC, NQ, 3)
        diff = sh - se
        w = np.array([1.0, 2.0, 1.0])                    # Voigt Frobenius
        d2 = (diff * diff * w).sum(axis=-1)              # (NC, NQ)
        cm = np.asarray(mesh.entity_measure('cell'))     # (NC,)
        val = (np.asarray(ws)[None, :] * d2).sum(axis=1) * cm
        return float(np.sqrt(val.sum()))

    def lift_stress_unc(self, sigma_rel: np.ndarray) -> np.ndarray:
        """Return σ in the unconstrained (gdof_unc) DOF space.

        Each cell reads its own cell-local copy at the angular vertex by using
        ``cell_to_dof_unc`` — this is the **physically correct** way for the
        relaxed solution since each side of a corner can carry different
        tangential traction.
        """
        return np.asarray(self.TM @ sigma_rel)

    def lift_stress_base_averaged(self, sigma_rel: np.ndarray) -> np.ndarray:
        """Project σ back to the base (fealpy) DOF by averaging cell copies.

        Useful for visualization / scalar errors against a globally continuous
        Hu-Zhang reconstruction. Loses the per-cell tangential information at
        corners — for accuracy assessment near corners, use ``cell_to_dof_unc``
        directly through a custom cell-quadrature error routine.
        """
        sigma_unc = self.lift_stress_unc(sigma_rel)
        sigma_base = np.zeros(self.gdof_base)
        node2dof = np.asarray(self.base_space.dof.node_to_internal_dof())
        corner_base = set()
        for c in self.corners:
            corner_base.update(node2dof[c.nid].tolist())
        for d in range(self.gdof_base):
            if d in corner_base:
                continue
            sigma_base[d] = sigma_unc[d]
        for c in self.corners:
            base3 = node2dof[c.nid]
            m = len(c.cells)
            for comp in range(3):
                vals = [float(sigma_unc[int(c.unc_dofs[k, comp])]) for k in range(m)]
                sigma_base[int(base3[comp])] = float(np.mean(vals))
        return sigma_base

    # =====================================================================
    # 工具
    # =====================================================================
    def diag(self) -> dict:
        """Return basic diagnostics for tests."""
        return dict(
            n_corners=len(self.corners),
            corner_node_ids=[c.nid for c in self.corners],
            corner_m=[int(len(c.cells)) for c in self.corners],
            gdof_base=self.gdof_base,
            gdof_unc=self.gdof_unc,
            gdof_rel=self.gdof_rel,
        )
