from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.typing import TensorLike

from .base import CaseBase, DirichletPiece
from fracturex.boundarycondition.huzhang_boundary_condition import build_isNedge_from_isD


@dataclass
class Model0CircularNotchCase(CaseBase):
    """
    HuZhang + phase-field case adapted from `cases/phase_field/model0_example.py`.

    Geometry:
      - unit square with an internal circular hole boundary
      - hole center=(0.5, 0.5), radius=0.2 by default

    Boundary conditions:
      - top boundary y=1: imposed displacement u_y = load (u_x free)
      - inner circle boundary: fixed displacement u=0
      - phase-field on inner circle: d=0
      - all other boundaries: natural/traction-free
    """

    name: str = "model0_circular_notch"
    box = (0.0, 1.0, 0.0, 1.0)

    # distmesh params (same spirit as model0_example)
    hmin: float = 0.01
    distmesh_maxit: int = 100

    # circle notch params
    circle_cx: float = 0.5
    circle_cy: float = 0.5
    circle_r: float = 0.2
    circle_tol: float = 1e-3

    # top boundary tolerance
    top_tol: float = 1e-12

    debug_mesh: bool = False
    _model: object = None

    def model(self):
        """Return material model bound to this case.

        Input:
            None.
        Output:
            Material model instance set in `_model`.
        """
        if self._model is None:
            raise RuntimeError("Model0CircularNotchCase requires _model (material model instance).")
        return self._model

    def make_mesh(self, nx: Optional[int] = None, ny: Optional[int] = None):
        """Build distmesh-based square-with-hole mesh.

        Inputs:
            nx, ny: Unused compatibility arguments kept for `CaseBase` signature.
        Output:
            Generated `TriangleMesh` assigned to `self.mesh`.
        """
        # keep signature compatible with CaseBase, but this case uses distmesh.
        from fealpy.old.geometry.domain_2d import SquareWithCircleHoleDomain

        domain = SquareWithCircleHoleDomain(hmin=self.hmin)
        mesh = TriangleMesh.from_domain_distmesh(
            domain,
            maxit=int(self.distmesh_maxit),
            ftype=bm.float64,
        )
        self.mesh = mesh
        return mesh

    # -----------------------
    # boundary selectors
    # -----------------------
    def _on_top(self, points: TensorLike) -> TensorLike:
        """Top boundary selector `y=box[3]`.

        Input:
            points: Point array `(N, GD)`.
        Output:
            Boolean mask `(N,)` indicating top-boundary points.
        """
        y1 = self.box[3]
        return bm.abs(points[:, 1] - y1) < self.top_tol

    def _on_inner_circle(self, points: TensorLike) -> TensorLike:
        """Inner-circle boundary selector.

        Input:
            points: Point array `(N, GD)`.
        Output:
            Boolean mask `(N,)` indicating points on circular notch boundary.
        """
        x = points[:, 0]
        y = points[:, 1]
        rr = (x - self.circle_cx) ** 2 + (y - self.circle_cy) ** 2
        return bm.abs(rr - self.circle_r ** 2) < self.circle_tol

    def isD_bd(self, points: TensorLike) -> TensorLike:
        """HuZhang Dirichlet boundary selector.

        Input:
            points: Edge/point barycenters `(N, GD)`.
        Output:
            Boolean mask of Dirichlet boundary locations.
        """
        # HuZhang boundary classification: top load + inner hole fixed
        return self._on_top(points) | self._on_inner_circle(points)

    # -----------------------
    # displacement BC pieces
    # -----------------------
    def dirichlet_pieces(self, load: float) -> List[DirichletPiece]:
        """Build piecewise displacement Dirichlet descriptors.

        Input:
            load: Prescribed vertical displacement on top boundary.
        Output:
            List of `DirichletPiece` entries for fixed hole and loaded top edge.
        """
        def u_zero(points: TensorLike):
            GD = points.shape[-1]
            return bm.zeros(points.shape[:-1] + (GD,), dtype=bm.float64)

        def u_top(points: TensorLike):
            GD = points.shape[-1]
            out = bm.zeros(points.shape[:-1] + (GD,), dtype=bm.float64)
            out[..., 1] = load
            return out

        return [
            DirichletPiece(threshold=self._on_inner_circle, value=u_zero, direction=None, tag="fix"),
            DirichletPiece(threshold=self._on_top, value=u_top, direction="y", tag="load"),
        ]

    def neumann_data(self, load: float = 0.0):
        """Build Neumann/essential-stress descriptors for HuZhang BC handler.

        Input:
            load: Current load value (not explicitly used in this case).
        Output:
            Piecewise descriptor list consumed by stress boundary condition utility.
        """
        isNedge_free = build_isNedge_from_isD(self.mesh, self.isD_bd)
        gd0 = bm.array([0.0, 0.0], dtype=bm.float64)
        return [
            (isNedge_free, gd0, "nt", None),
            (self._on_top, gd0, "nt", "t"),
        ]

    def phasefield_dirichlet_data(self, load: float) -> Optional[Any]:
        """Provide phase-field Dirichlet BC descriptor.

        Input:
            load: Current load value.
        Output:
            List of dict descriptors fixing `d=0` on inner circle.
        """
        # model0_example: phase on inner circle fixed to 0
        return [{"bcdof": self._on_inner_circle, "value": 0.0}]

    def default_loads(self):
        """Return default displacement load schedule.

        Input:
            None.
        Output:
            Backend array of monotonic displacement values.
        """
        # same schedule as model0_example.is_force()
        return bm.concatenate(
            (
                bm.linspace(0, 70e-3, 6, dtype=bm.float64),
                bm.linspace(70e-3, 125e-3, 26, dtype=bm.float64)[1:],
            )
        )
