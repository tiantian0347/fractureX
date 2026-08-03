"""Asymmetrically notched beam with three holes (Ambati 2015 §4.5).

Origin
------
* Experiment: Bittencourt et al., Eng. Fract. Mech. 55:321–334 (1996)
  (Ambati ref. [40]).
* Phase-field setup: Ambati, Gerasimov, De Lorenzis, Comput. Mech.
  55:383–405 (2015) §4.5 / Fig. 23.

Geometry (mm) — Ambati Fig. 23
------------------------------
* Beam ``[0, 20] × [0, 8]``.
* Supports at ``(1, 0)`` and ``(19, 0)``; load at top centre ``(10, 8)``.
* Bottom notch at ``x = 4``, depth ``1`` (Hu–Zhang: ``d = 1`` pre-crack;
  FEM/IPFEM: geometric slit / small mouth).
* Three holes of diameter ``0.5`` (radius ``0.25``) on a diagonal in the
  left half (Ambati Fig. 23 labels: top clearance ``1.25``, vertical
  pitch ``2``, centres at ``x = 5, 7, 9``):

  ``(5.0, 6.5)``, ``(7.0, 4.5)``, ``(9.0, 2.5)``.

Material (Ambati §4.5, kN–mm)
-----------------------------
``lam = 12.0``, ``mu = 8.0``, ``Gc = 1e-3``, ``l0 = 0.01``.

Boundary conditions
-------------------
* Left support: ``u = 0``; right support: ``u_y = 0``.
* Top centre: ``u_y = −load`` (``load ≥ 0`` downward).
* Hole boundaries and free faces: traction-free.

Expected path
-------------
Crack from bottom notch curves toward the second hole (Ambati Fig. 23b /
Fig. 24); peak force ~0.6–0.7 kN (Fig. 25).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.typing import TensorLike

from .base import CaseBase, DirichletPiece
from fracturex.boundarycondition.huzhang_boundary_condition import build_isNedge_from_isD


def _build_gmsh_asymmetric_beam(
    *,
    mesh_size: float,
    length: float,
    height: float,
    notch_x: float,
    notch_depth: float,
    notch_mouth: float,
    with_geometric_notch: bool,
    left_support_x: float,
    right_support_x: float,
    holes: Sequence[Tuple[float, float]],
    hole_radius: float,
) -> TriangleMesh:
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("model6_asymmetric_notched_beam")

    mid = 0.5 * length
    half_m = 0.5 * max(notch_mouth, 1e-6)
    crack_line = None

    p = {}
    p["BL"] = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
    p["BR"] = gmsh.model.geo.addPoint(length, 0.0, 0.0)
    p["TR"] = gmsh.model.geo.addPoint(length, height, 0.0)
    p["TL"] = gmsh.model.geo.addPoint(0.0, height, 0.0)
    p["LS"] = gmsh.model.geo.addPoint(left_support_x, 0.0, 0.0)
    p["RS"] = gmsh.model.geo.addPoint(right_support_x, 0.0, 0.0)
    p["LOAD"] = gmsh.model.geo.addPoint(mid, height, 0.0)

    if with_geometric_notch:
        p["NL"] = gmsh.model.geo.addPoint(notch_x - half_m, 0.0, 0.0)
        p["NR"] = gmsh.model.geo.addPoint(notch_x + half_m, 0.0, 0.0)
        p["TIP"] = gmsh.model.geo.addPoint(notch_x, notch_depth, 0.0)
        # Assume 0 < LS < NL < NR < RS < L
        lines = [
            gmsh.model.geo.addLine(p["BL"], p["LS"]),
            gmsh.model.geo.addLine(p["LS"], p["NL"]),
            gmsh.model.geo.addLine(p["NL"], p["TIP"]),
            gmsh.model.geo.addLine(p["TIP"], p["NR"]),
            gmsh.model.geo.addLine(p["NR"], p["RS"]),
            gmsh.model.geo.addLine(p["RS"], p["BR"]),
            gmsh.model.geo.addLine(p["BR"], p["TR"]),
            gmsh.model.geo.addLine(p["TR"], p["LOAD"]),
            gmsh.model.geo.addLine(p["LOAD"], p["TL"]),
            gmsh.model.geo.addLine(p["TL"], p["BL"]),
        ]
    else:
        p["NX"] = gmsh.model.geo.addPoint(notch_x, 0.0, 0.0)
        p["TIP"] = gmsh.model.geo.addPoint(notch_x, notch_depth, 0.0)
        lines = [
            gmsh.model.geo.addLine(p["BL"], p["LS"]),
            gmsh.model.geo.addLine(p["LS"], p["NX"]),
            gmsh.model.geo.addLine(p["NX"], p["RS"]),
            gmsh.model.geo.addLine(p["RS"], p["BR"]),
            gmsh.model.geo.addLine(p["BR"], p["TR"]),
            gmsh.model.geo.addLine(p["TR"], p["LOAD"]),
            gmsh.model.geo.addLine(p["LOAD"], p["TL"]),
            gmsh.model.geo.addLine(p["TL"], p["BL"]),
        ]
        crack_line = gmsh.model.geo.addLine(p["NX"], p["TIP"])

    outer = gmsh.model.geo.addCurveLoop(lines)

    def _circle_loop(cx: float, cy: float, r: float):
        c = gmsh.model.geo.addPoint(cx, cy, 0.0)
        a = gmsh.model.geo.addPoint(cx - r, cy, 0.0)
        b = gmsh.model.geo.addPoint(cx + r, cy, 0.0)
        arc0 = gmsh.model.geo.addCircleArc(a, c, b)
        arc1 = gmsh.model.geo.addCircleArc(b, c, a)
        return gmsh.model.geo.addCurveLoop([arc0, arc1])

    hole_loops = [_circle_loop(cx, cy, hole_radius) for cx, cy in holes]
    gmsh.model.geo.addPlaneSurface([outer, *hole_loops], 1)
    gmsh.model.geo.synchronize()
    if crack_line is not None:
        gmsh.model.mesh.embed(1, [crack_line], 2, 1)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), float(mesh_size))
    gmsh.model.mesh.generate(2)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node = node_coords.reshape((-1, 3))[:, :2]
    nodetags_map = {int(j): i for i, j in enumerate(node_tags)}
    cell_tags, cell_connectivity = gmsh.model.mesh.getElementsByType(2)
    evid = bm.array(
        [nodetags_map[int(j)] for j in cell_connectivity], dtype=bm.int32
    )
    cell = evid.reshape((cell_tags.shape[-1], -1))
    gmsh.finalize()
    return TriangleMesh(bm.asarray(node, dtype=bm.float64), cell)


@dataclass
class Model6AsymmetricNotchedBeamCase(CaseBase):
    """Ambati §4.5 / Bittencourt asymmetrically notched beam with holes."""

    name: str = "model6_asymmetric_notched_beam"

    length: float = 20.0
    height: float = 8.0
    notch_x: float = 4.0
    notch_depth: float = 1.0
    notch_mouth: float = 0.05
    with_geometric_notch: bool = False
    crack_tol: float = 1e-9

    left_support_x: float = 1.0
    right_support_x: float = 19.0
    support_half_width: float = 0.05
    load_half_width: float = 0.05
    bd_tol: float = 1e-9

    hole_radius: float = 0.25
    holes: Tuple[Tuple[float, float], ...] = (
        (5.0, 6.5),
        (7.0, 4.5),
        (9.0, 2.5),
    )

    mesh_size: float = 0.2

    lam: float = 12.0
    mu: float = 8.0
    Gc: float = 1.0e-3
    l0: float = 0.01

    debug_mesh: bool = False
    _model: object = None
    mesh: Any = field(default=None, repr=False)

    def model(self):
        if self._model is None:
            raise RuntimeError(
                "Model6AsymmetricNotchedBeamCase requires _model."
            )
        return self._model

    def reaction_direction(self):
        return "y"

    def reaction_boundary(self, load: float = 0.0):
        def on_supports(points):
            return self._on_left_support(points) | self._on_right_support(points)

        return on_supports, "y", 1.0

    def make_mesh(self, nx: Optional[int] = None, ny: Optional[int] = None):
        mesh = _build_gmsh_asymmetric_beam(
            mesh_size=self.mesh_size,
            length=self.length,
            height=self.height,
            notch_x=self.notch_x,
            notch_depth=self.notch_depth,
            notch_mouth=self.notch_mouth,
            with_geometric_notch=bool(self.with_geometric_notch),
            left_support_x=self.left_support_x,
            right_support_x=self.right_support_x,
            holes=self.holes,
            hole_radius=self.hole_radius,
        )
        self.mesh = mesh
        if self.debug_mesh:
            mode = (
                "geometric_notch"
                if self.with_geometric_notch
                else "precrack_d=1"
            )
            print(
                f"[model6 mesh] NN={mesh.number_of_nodes()}, "
                f"NC={mesh.number_of_cells()}, mesh_size={self.mesh_size}, "
                f"mode={mode}"
            )
        return mesh

    def _support_hw(self) -> float:
        return max(float(self.support_half_width), 0.6 * float(self.mesh_size))

    def _load_hw(self) -> float:
        return max(float(self.load_half_width), 0.6 * float(self.mesh_size))

    def _on_left_support(self, points: TensorLike) -> TensorLike:
        return (
            (bm.abs(points[:, 1] - 0.0) < self.bd_tol)
            & (bm.abs(points[:, 0] - self.left_support_x) <= self._support_hw())
        )

    def _on_right_support(self, points: TensorLike) -> TensorLike:
        return (
            (bm.abs(points[:, 1] - 0.0) < self.bd_tol)
            & (bm.abs(points[:, 0] - self.right_support_x) <= self._support_hw())
        )

    def _on_load(self, points: TensorLike) -> TensorLike:
        mid = 0.5 * self.length
        return (
            (bm.abs(points[:, 1] - self.height) < self.bd_tol)
            & (bm.abs(points[:, 0] - mid) <= self._load_hw())
        )

    def _on_precrack(self, points: TensorLike) -> TensorLike:
        tol = float(self.crack_tol)
        x = points[:, 0]
        y = points[:, 1]
        return (
            (bm.abs(x - self.notch_x) < tol)
            & (y >= -tol)
            & (y <= self.notch_depth + tol)
        )

    def isD_bd(self, points: TensorLike) -> TensorLike:
        return (
            self._on_left_support(points)
            | self._on_right_support(points)
            | self._on_load(points)
        )

    def dirichlet_pieces(self, load: float) -> List[DirichletPiece]:
        def u_zero(points: TensorLike):
            GD = points.shape[-1]
            return bm.zeros(points.shape[:-1] + (GD,), dtype=bm.float64)

        def u_load(points: TensorLike):
            GD = points.shape[-1]
            out = bm.zeros(points.shape[:-1] + (GD,), dtype=bm.float64)
            out[..., 1] = -float(load)
            return out

        return [
            DirichletPiece(
                threshold=self._on_left_support,
                value=u_zero,
                direction=None,
                tag="fix_left",
            ),
            DirichletPiece(
                threshold=self._on_right_support,
                value=u_zero,
                direction="y",
                tag="fix_right",
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
        return [
            (isNedge_free, gd0, "nt", None),
            (self._on_right_support, gd0, "nt", "t"),
            (self._on_load, gd0, "nt", "t"),
        ]

    def phasefield_initial_damage_data(self, load: float) -> Optional[Any]:
        if self.with_geometric_notch:
            return None
        return [{"bcdof": self._on_precrack, "value": 1.0}]

    def phasefield_dirichlet_data(self, load: float) -> Optional[Any]:
        if self.with_geometric_notch:
            return None
        return [{"bcdof": self._on_precrack, "value": 0.0}]

    def default_loads(self):
        # Ambati: Δu=1e-3 for 200 steps, then 1e-4
        return bm.concatenate(
            (
                bm.linspace(0.0, 0.2, 201, dtype=bm.float64),
                bm.linspace(0.2, 0.3, 1001, dtype=bm.float64)[1:],
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
    case = Model6AsymmetricNotchedBeamCase(mesh_size=0.4, debug_mesh=True)
    mesh = case.make_mesh()
    print("material:", case.material_dict())
    isBd = mesh.boundary_edge_flag()
    bdedge = bm.where(isBd)[0]
    bc = mesh.entity_barycenter("edge", index=bdedge)
    print(
        "L/R/load",
        int(bm.sum(case._on_left_support(bc))),
        int(bm.sum(case._on_right_support(bc))),
        int(bm.sum(case._on_load(bc))),
    )
