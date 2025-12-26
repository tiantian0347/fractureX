from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.typing import TensorLike

from .base import CaseBase, DirichletPiece


@dataclass
class SquareTensionCase(CaseBase):
    """
    Square tension:
      - y=0: u = 0
      - y=1: u_y = load
      - other boundary (including crack faces if with_fracture=True): Neumann (traction/stress)
    """
    name: str = "square_tension"
    box = (0.0, 1.0, 0.0, 1.0)
    tol: float = 1e-9

    # mesh options
    nx: int = 16
    ny: int = 16
    with_fracture: bool = False   # use TriangleMesh.from_square_domain_with_fracture
    refine: int = 0               # uniform refine times for fracture mesh

    # debug
    debug_mesh: bool = False

    _model: object = None

    def model(self):
        if self._model is None:
            raise RuntimeError("SquareTensionCase requires _model (material model instance).")
        return self._model

    # -----------------------
    # mesh
    # -----------------------
    def make_mesh(self, nx: Optional[int] = None, ny: Optional[int] = None):
        x0, x1, y0, y1 = self.box

        if self.with_fracture:
            mesh = TriangleMesh.from_square_domain_with_fracture(device=None)
            for _ in range(int(self.refine)):
                if hasattr(mesh, "uniform_refine"):
                    mesh.uniform_refine()
            self._mark_boundary_sets(mesh)
            return mesh

        if nx is None: nx = self.nx
        if ny is None: ny = self.ny
        mesh = TriangleMesh.from_box([x0, x1, y0, y1], nx=nx, ny=ny)
        self._mark_boundary_sets(mesh)  # 统一：box mesh 也标记一下（方便通用）
        return mesh

    def _mark_boundary_sets(self, mesh: TriangleMesh):
        """
        Cache boundary sets into mesh.edgedata:
          - is_bd_edge: geometry boundary edges
          - is_box_edge: edges on outer box boundary
          - is_crack_edge: boundary edges not on outer box boundary (fracture faces)
          - crack_edge_index: global eids for crack edges
        """
        NE = mesh.number_of_edges()
        isBd = mesh.boundary_edge_flag()
        bdedge = bm.where(isBd)[0]

        # no boundary edges: do nothing
        if int(bdedge.shape[0]) == 0:
            mesh.edgedata['is_bd_edge'] = isBd
            mesh.edgedata['is_box_edge'] = bm.zeros(NE, dtype=bm.bool)
            mesh.edgedata['is_crack_edge'] = bm.zeros(NE, dtype=bm.bool)
            mesh.edgedata['crack_edge_index'] = bm.zeros((0,), dtype=bm.int32)
            return

        bc = mesh.entity_barycenter('edge', index=bdedge)

        x0, x1, y0, y1 = self.box
        tol = self.tol
        on_box = ((bm.abs(bc[:, 0] - x0) < tol) | (bm.abs(bc[:, 0] - x1) < tol) |
                  (bm.abs(bc[:, 1] - y0) < tol) | (bm.abs(bc[:, 1] - y1) < tol))
        crack = ~on_box
        crack_eids = bdedge[crack]

        is_box_edge = bm.zeros(NE, dtype=bm.bool)
        is_crack_edge = bm.zeros(NE, dtype=bm.bool)
        is_box_edge = bm.set_at(is_box_edge, bdedge[on_box], True)
        is_crack_edge = bm.set_at(is_crack_edge, crack_eids, True)

        mesh.edgedata['is_bd_edge'] = isBd
        mesh.edgedata['is_box_edge'] = is_box_edge
        mesh.edgedata['is_crack_edge'] = is_crack_edge
        mesh.edgedata['crack_edge_index'] = crack_eids

        if self.debug_mesh:
            bc_all = bc
            print(f"[mesh debug] NEb={int(bdedge.shape[0])}, "
                  f"bc xmin/xmax=({float(bm.min(bc_all[:,0]))},{float(bm.max(bc_all[:,0]))}), "
                  f"ymin/ymax=({float(bm.min(bc_all[:,1]))},{float(bm.max(bc_all[:,1]))})")
            print(f"[mesh debug] crack edges = {int(crack_eids.shape[0])}")
            if int(crack_eids.shape[0]) > 0:
                bc_cr = mesh.entity_barycenter('edge', index=crack_eids)
                print(f"[mesh debug] crack bc xmin/xmax=({float(bm.min(bc_cr[:,0]))},{float(bm.max(bc_cr[:,0]))}), "
                      f"ymin/ymax=({float(bm.min(bc_cr[:,1]))},{float(bm.max(bc_cr[:,1]))})")

    # -----------------------
    # boundary: Dirichlet pieces
    # -----------------------
    def _on_y0(self, points: TensorLike) -> TensorLike:
        y0 = self.box[2]
        return bm.abs(points[:, 1] - y0) < self.tol

    def _on_y1(self, points: TensorLike) -> TensorLike:
        y1 = self.box[3]
        return bm.abs(points[:, 1] - y1) < self.tol

    def isD_bd(self, points: TensorLike) -> TensorLike:
        # points are edge barycenters (or face barycenters)
        return self._on_y0(points) | self._on_y1(points)
    
    def load_boundary_threshold(self):
        # y=1 作为加载边界（边重心输入）
        return self._on_y1


    def dirichlet_pieces(self, load: float) -> List[DirichletPiece]:
        def u_fix(points: TensorLike):
            GD = points.shape[-1]
            return bm.zeros(points.shape[:-1] + (GD,), dtype=bm.float64)

        def u_load(points: TensorLike):
            GD = points.shape[-1]
            out = bm.zeros(points.shape[:-1] + (GD,), dtype=bm.float64)
            out[..., 1] = load  # u_y = load
            return out

        return [
            DirichletPiece(threshold=self._on_y0, value=u_fix, direction=None),
            DirichletPiece(threshold=self._on_y1, value=u_load, direction=None),
        ]

    # -----------------------
    # boundary: Neumann selector (唯一接口)
    # -----------------------
    def build_isNedge(self, mesh: TriangleMesh):
        """
        Return (NE,) bool for ΓN edges.
        Rule: all boundary edges except Dirichlet edges are Neumann.
        This includes crack faces when with_fracture=True.
        """
        isBd = mesh.boundary_edge_flag()
        bdedge = bm.where(isBd)[0]
        bc = mesh.entity_barycenter('edge', index=bdedge)

        isD_on_bd = bm.asarray(self.isD_bd(bc)).astype(bm.bool)
        isN_on_bd = ~isD_on_bd

        isN = bm.zeros(mesh.number_of_edges(), dtype=bm.bool)
        isN = bm.set_at(isN, bdedge[isN_on_bd], True)
        isN = isN & isBd  # safety

        if self.debug_mesh and ('crack_edge_index' in mesh.edgedata):
            crack_eids = mesh.edgedata['crack_edge_index']
            if int(crack_eids.shape[0]) > 0:
                print("[isNedge debug] isNedge on crack:",
                      [bool(isN[int(e)]) for e in crack_eids])

        return isN
