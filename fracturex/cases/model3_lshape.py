"""L-shaped panel (Winkler / Ambati 2015 §4.3).

Geometry (mm)
-------------
* Outer box ``[0, 500] × [0, 500]`` with the lower-right quadrant
  ``(x > 250) ∩ (y < 250)`` removed → L domain.
* Re-entrant corner at ``(250, 250)``; crack nucleates there (no pre-crack).

Boundary conditions
-------------------
* Bottom ``y = 0``: ``u = 0``.
* Load point ``(470, 250)`` on the inner horizontal arm: ``u_y = load``
  (window of half-width ``load_half_width`` so Hu–Zhang edge midpoints hit).
* Remaining faces: traction-free.
* Phase field: homogeneous Neumann (no essential ``d`` BC).

Material (Ambati §4.3, kN–mm)
-----------------------------
``lam = 6.16``, ``mu = 10.95``, ``Gc = 8.9e-5``, ``l0 = 1.1875``
(IP-FEM default in this repo uses ``l0 = 1.18``).

Loading (Ambati cyclic closure test)
------------------------------------
Default schedule: ``0 → 0.3 → −0.2 → 1.0`` mm.

Hu–Zhang note
--------------
The re-entrant corner is a known Hu–Zhang dangling-corner stress singularity;
prefer ``use_relaxation=True`` (corner relaxation) on the Hu–Zhang runner.
Standard Lagrange / IP-FEM use this case as-is (see
``interior_penalty/cases/model3_lshape.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.typing import TensorLike

from .base import CaseBase, DirichletPiece
from fracturex.boundarycondition.huzhang_boundary_condition import build_isNedge_from_isD


@dataclass
class Model3LShapeCase(CaseBase):
    """Ambati / Winkler L-shaped panel under cyclic point displacement."""

    name: str = "model3_lshape"

    box: Tuple[float, float, float, float] = (0.0, 500.0, 0.0, 500.0)
    cut_x: float = 250.0
    cut_y: float = 250.0
    load_point: Tuple[float, float] = (470.0, 250.0)
    load_half_width: float = 2.0  # mm; mark a short segment around the load point
    bottom_tol: float = 1e-9

    nx: int = 50
    ny: int = 50

    # Ambati material (exposed for runners)
    lam: float = 6.16
    mu: float = 10.95
    Gc: float = 8.9e-5
    l0: float = 1.1875

    debug_mesh: bool = False
    _model: object = None
    mesh: Any = field(default=None, repr=False)

    def model(self):
        if self._model is None:
            raise RuntimeError(
                "Model3LShapeCase requires _model (material model instance)."
            )
        return self._model

    def reaction_direction(self):
        return "y"

    def make_mesh(self, nx: Optional[int] = None, ny: Optional[int] = None):
        nx = int(self.nx if nx is None else nx)
        ny = int(self.ny if ny is None else ny)

        def _thr(p):
            return (p[..., 0] > self.cut_x) & (p[..., 1] < self.cut_y)

        mesh = TriangleMesh.from_box(
            box=list(self.box), nx=nx, ny=ny, threshold=_thr
        )
        self.mesh = mesh
        if self.debug_mesh:
            print(
                f"[model3 mesh] NN={mesh.number_of_nodes()}, "
                f"NC={mesh.number_of_cells()}, nx={nx}, ny={ny}"
            )
        return mesh

    def _load_hw(self) -> float:
        hx = (self.box[1] - self.box[0]) / max(int(self.nx), 1)
        return max(float(self.load_half_width), 0.6 * hx)

    def _on_bottom(self, points: TensorLike) -> TensorLike:
        y0 = self.box[2]
        return bm.abs(points[:, 1] - y0) < self.bottom_tol

    def _on_load(self, points: TensorLike) -> TensorLike:
        px, py = self.load_point
        # Inner horizontal face of the L is y = cut_y, x ∈ [cut_x, box_x1].
        return (
            (bm.abs(points[:, 1] - py) < self.bottom_tol)
            & (bm.abs(points[:, 0] - px) <= self._load_hw())
        )

    def isD_bd(self, points: TensorLike) -> TensorLike:
        return self._on_bottom(points) | self._on_load(points)

    def dirichlet_pieces(self, load: float) -> List[DirichletPiece]:
        def u_zero(points: TensorLike):
            GD = points.shape[-1]
            return bm.zeros(points.shape[:-1] + (GD,), dtype=bm.float64)

        def u_load(points: TensorLike):
            GD = points.shape[-1]
            out = bm.zeros(points.shape[:-1] + (GD,), dtype=bm.float64)
            out[..., 1] = load
            return out

        return [
            DirichletPiece(
                threshold=self._on_bottom,
                value=u_zero,
                direction=None,
                tag="fix",
            ),
            DirichletPiece(
                threshold=self._on_load,
                value=u_load,
                direction="y",
                tag="load",
            ),
        ]

    def neumann_data(self, load: float = 0.0):
        isNedge_free = build_isNedge_from_isD(self.mesh, self.isD_bd)
        gd0 = bm.array([0.0, 0.0], dtype=bm.float64)
        # Load point: uy prescribed, ux free → fix only tangential traction.
        return [
            (isNedge_free, gd0, "nt", None),
            (self._on_load, gd0, "nt", "t"),
        ]

    def phasefield_dirichlet_data(self, load: float) -> Optional[Any]:
        return None

    def default_loads(self):
        # Ambati cyclic: 0 → 0.3 → −0.2 → 1.0
        return bm.concatenate(
            (
                bm.linspace(0.0, 0.3, 301, dtype=bm.float64),
                bm.linspace(0.3, -0.2, 501, dtype=bm.float64)[1:],
                bm.linspace(-0.2, 1.0, 1201, dtype=bm.float64)[1:],
            )
        )

    def material_dict(self) -> dict:
        E = self.mu * (3.0 * self.lam + 2.0 * self.mu) / (self.lam + self.mu)
        nu = self.lam / (2.0 * (self.lam + self.mu))
        return dict(
            lam=self.lam,
            mu=self.mu,
            E=E,
            nu=nu,
            Gc=self.Gc,
            l0=self.l0,
        )


if __name__ == "__main__":
    case = Model3LShapeCase(nx=20, ny=20, debug_mesh=True)
    mesh = case.make_mesh()
    print("material:", case.material_dict())
    isBd = mesh.boundary_edge_flag()
    bdedge = bm.where(isBd)[0]
    bc = mesh.entity_barycenter("edge", index=bdedge)
    n_bot = int(bm.sum(case._on_bottom(bc)))
    n_load = int(bm.sum(case._on_load(bc)))
    print(f"boundary edges={int(bdedge.shape[0])}, bottom={n_bot}, load={n_load}")
    assert n_bot > 0 and n_load > 0, "bottom / load edges not captured"
