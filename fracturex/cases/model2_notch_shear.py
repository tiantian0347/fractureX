from __future__ import annotations

from dataclasses import dataclass
from typing import List

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .base import DirichletPiece
from .square_tension_precrack import SquareTensionPreCrackCase
from fracturex.boundarycondition.huzhang_boundary_condition import build_isNedge_from_isD


@dataclass
class Model2NotchXStretchCase(SquareTensionPreCrackCase):
    """
    Paper model-2: unit square, top x-stretch, pre-crack via phase-field (no geometric notch).

    Same mesh/phase-field treatment as ``SquareTensionPreCrackCase`` / ``phasefield_square_tension``
    style intact ``from_box`` mesh: geometric cut meshes are avoided (Hu-Zhang corner issues).

    Geometry:
      - [0,1]×[0,1] uniform triangle mesh
      - initial crack: d = 1 on y = crack_y, x in [0, crack_length] (default: mid-line left half)

    Boundary conditions (adaptive_paper.tex, lines 979–982):
      - y = 0: u_x = u_y = 0
      - y = 1: u_x = load, u_y = 0
      - other outer boundaries: traction-free

    Loading:
      - Δu_x = 1e-5 mm/step, 2400 steps, u_x,tot = 2.4e-2 mm（对齐 FEALPy 参照程序）
    """

    name: str = "model2_notch_x_stretch"

    # ~2048 triangles when nx=ny=32 (2*nx*ny)
    nx: int = 32
    ny: int = 32

    crack_y: float = 0.5
    crack_length: float = 0.5
    crack_tol: float = 1e-9

    du_x: float = 1.0e-5
    n_load_steps: int = 2000
    u_x_total: float = 2.0e-2

    def reaction_direction(self):
        return "x"

    def default_loads(self):
        n = int(self.n_load_steps)
        return bm.linspace(0.0, float(self.u_x_total), n + 1, dtype=bm.float64)

    def dirichlet_pieces(self, load: float) -> List[DirichletPiece]:
        def u_zero(points: TensorLike):
            gd = points.shape[-1]
            return bm.zeros(points.shape[:-1] + (gd,), dtype=bm.float64)

        def u_top_x(points: TensorLike):
            gd = points.shape[-1]
            out = bm.zeros(points.shape[:-1] + (gd,), dtype=bm.float64)
            out[..., 0] = load
            return out

        return [
            DirichletPiece(threshold=self._on_y0, value=u_zero, direction="x", tag="fix_bottom_ux"),
            DirichletPiece(threshold=self._on_y0, value=u_zero, direction="y", tag="fix_bottom_uy"),
            DirichletPiece(threshold=self._on_y1, value=u_top_x, direction="x", tag="load"),
            DirichletPiece(threshold=self._on_y1, value=u_zero, direction="y", tag="fix_top_uy"),
        ]

    def neumann_data(self, load: float = 0.0):
        is_nedge_free = build_isNedge_from_isD(self.mesh, self.isD_bd)
        gd0 = bm.array([0.0, 0.0], dtype=bm.float64)
        # NOTE: the top edge y=1 prescribes BOTH displacement components here
        # (u_x = load AND u_y = 0, see dirichlet_pieces), so it is a *full*
        # displacement boundary -- in the Hu-Zhang mixed form the whole traction
        # sigma.n on y=1 is an unknown reaction and must NOT be fixed as an
        # essential stress BC. (Contrast SquareTensionCase, whose top slides in x
        # with u_x free, where fixing the tangential traction gt=0 is correct.)
        # Keeping the old `(_on_y1, gd0, "nt", "t")` entry zeroed the tangential
        # reaction that carries the x-stretch load -> trivial solution u==0.
        return [
            (is_nedge_free, gd0, "nt", None),
        ]


Model2NotchShearCase = Model2NotchXStretchCase
