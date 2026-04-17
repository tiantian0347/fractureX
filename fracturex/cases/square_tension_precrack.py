from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.typing import TensorLike

from .square_tension import SquareTensionCase
from fracturex.boundarycondition.huzhang_boundary_condition import build_isNedge_from_isD


@dataclass
class SquareTensionPreCrackCase(SquareTensionCase):
    """
    Square tension with an initial pre-crack represented by d=1.

    Geometry:
      - intact square domain, no geometric notch/cut in the mesh

    Pre-crack:
      - a horizontal segment at y=0.5, x in [0.0, crack_length]
      - enforced by phase-field Dirichlet condition d = 1

    Boundary conditions:
      - y=0: fixed displacement
      - y=1: vertical displacement load
      - other outer boundaries: traction-free / natural BC
    """

    name: str = "square_tension_precrack"
    with_fracture: bool = False

    # initial crack segment
    crack_y: float = 0.5
    crack_length: float = 0.5
    crack_tol: float = 1e-9

    def make_mesh(self, nx: Optional[int] = None, ny: Optional[int] = None):
        x0, x1, y0, y1 = self.box
        if nx is None:
            nx = self.nx
        if ny is None:
            ny = self.ny

        mesh = TriangleMesh.from_box([x0, x1, y0, y1], nx=nx, ny=ny)
        self.mesh = mesh
        self._mark_boundary_sets(mesh)
        return mesh

    def crack_edge_mask(self, mesh: TriangleMesh) -> TensorLike:
        """No geometric crack edges are built into the mesh."""
        return bm.zeros(mesh.number_of_edges(), dtype=bm.bool)

    def _on_precrack(self, points: TensorLike) -> TensorLike:
        """
        Mark the initial crack segment in the intact square.

        The segment is y = crack_y, x in [0, crack_length].
        """
        x = points[:, 0]
        y = points[:, 1]
        return (
            bm.abs(y - self.crack_y) < self.crack_tol
        ) & (
            x >= -self.crack_tol
        ) & (
            x <= self.crack_length + self.crack_tol
        )

    def phasefield_initial_damage_data(self, load: float):
        """
        One-time initialization: set d = 1 on the pre-crack segment.
        """
        return [
            {
                "bcdof": self._on_precrack,
                "value": 1.0,
            }
        ]

    def phasefield_dirichlet_data(self, load: float):
        """
        Increment BC for each iteration: keep pre-crack as dd = 0.

        Since phase equation is assembled in increment form (A dd = ...),
        dd=0 preserves initialized d=1 on the pre-crack line.
        """
        return [
            {
                "bcdof": self._on_precrack,
                "value": 0.0,
            }
        ]

    def neumann_data(self, load: float = 0.0):
        # Same outer-boundary classification as SquareTensionCase.
        isNedge_free = build_isNedge_from_isD(self.mesh, self.isD_bd)
        gd0 = bm.array([0.0, 0.0], dtype=bm.float64)

        return [
            (isNedge_free, gd0, "nt", None),
            (self._on_y1, gd0, "nt", "t"),
        ]
