"""Three-point bending of a notched beam (Ambati 2015 §4.4 / Miehe).

Geometry (mm)
-------------
* Beam: ``[0, length] × [0, height]`` with defaults ``8 × 2``.
* Bottom-centre notch depth ``notch_depth = 0.4``.
* Supports at beam ends ``(0,0)`` and ``(length,0)`` (Ambati Fig. 20);
  top-centre prescribed downward displacement.

Notch representation (discretization-dependent)
-----------------------------------------------
* **Hu–Zhang** (default ``with_geometric_notch=False``): intact rectangle;
  pre-crack ``d = 1`` on the vertical segment ``x = length/2``,
  ``y ∈ [0, notch_depth]`` (embedded mesh line).
* **Standard FEM / IP-FEM** (``with_geometric_notch=True``): V-notch cut
  matching the old ``fracture_pu`` gmsh script (mouth width
  ``notch_mouth``, tip at mid-span).

Boundary conditions
-------------------
* Left support: ``u_y = 0`` (and ``u_x = 0`` to kill rigid motion).
* Right support: ``u_y = 0``.
* Top midspan: ``u_y = −load`` (``load ≥ 0`` means downward magnitude).
* Remaining faces: traction-free.

Material (Ambati §4.4, kN–mm)
-----------------------------
``lam = 12.0``, ``mu = 8.0``, ``Gc = 5.4e-4``, ``l0 = 0.03``.

Loading (Ambati sketch)
-----------------------
Default: coarse then fine downward schedule up to ``0.1`` mm; runners may
override. Sign convention: case stores **positive** downward magnitude;
Dirichlet applies ``u_y = −load``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.typing import TensorLike

from .base import CaseBase, DirichletPiece
from fracturex.boundarycondition.huzhang_boundary_condition import build_isNedge_from_isD


def _build_gmsh_beam(
    *,
    mesh_size: float,
    local_mesh_size: Optional[float] = None,
    local_refine_half_width: float = 0.5,
    local_refine_ymax: float = 1.6,
    local_refine_transition: float = 0.15,
    length: float,
    height: float,
    notch_depth: float,
    notch_mouth: float,
    with_geometric_notch: bool,
    left_support_x: float,
    right_support_x: float,
) -> TriangleMesh:
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("model5_three_point_bending")

    mid = 0.5 * length
    half_m = 0.5 * notch_mouth
    crack_line = None

    p = {}
    p["BL"] = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
    p["BR"] = gmsh.model.geo.addPoint(length, 0.0, 0.0)
    p["TR"] = gmsh.model.geo.addPoint(length, height, 0.0)
    p["TL"] = gmsh.model.geo.addPoint(0.0, height, 0.0)
    p["LOAD"] = gmsh.model.geo.addPoint(mid, height, 0.0)
    # Reuse corners when Ambati end-supports coincide with BL / BR.
    eps = 1e-12
    p["LS"] = (
        p["BL"]
        if abs(left_support_x) < eps
        else gmsh.model.geo.addPoint(left_support_x, 0.0, 0.0)
    )
    p["RS"] = (
        p["BR"]
        if abs(right_support_x - length) < eps
        else gmsh.model.geo.addPoint(right_support_x, 0.0, 0.0)
    )

    def _seg(a, b):
        return None if a == b else gmsh.model.geo.addLine(a, b)

    if with_geometric_notch:
        p["ML"] = gmsh.model.geo.addPoint(mid - half_m, 0.0, 0.0)
        p["MR"] = gmsh.model.geo.addPoint(mid + half_m, 0.0, 0.0)
        p["TIP"] = gmsh.model.geo.addPoint(mid, notch_depth, 0.0)
        bot = [
            _seg(p["BL"], p["LS"]),
            _seg(p["LS"], p["ML"]),
            gmsh.model.geo.addLine(p["ML"], p["TIP"]),
            gmsh.model.geo.addLine(p["TIP"], p["MR"]),
            _seg(p["MR"], p["RS"]),
            _seg(p["RS"], p["BR"]),
        ]
    else:
        p["BOT_MID"] = gmsh.model.geo.addPoint(mid, 0.0, 0.0)
        p["CRACK"] = gmsh.model.geo.addPoint(mid, notch_depth, 0.0)
        bot = [
            _seg(p["BL"], p["LS"]),
            _seg(p["LS"], p["BOT_MID"]),
            _seg(p["BOT_MID"], p["RS"]),
            _seg(p["RS"], p["BR"]),
        ]
        crack_line = gmsh.model.geo.addLine(p["BOT_MID"], p["CRACK"])

    lines = [ln for ln in bot if ln is not None] + [
        gmsh.model.geo.addLine(p["BR"], p["TR"]),
        gmsh.model.geo.addLine(p["TR"], p["LOAD"]),
        gmsh.model.geo.addLine(p["LOAD"], p["TL"]),
        gmsh.model.geo.addLine(p["TL"], p["BL"]),
    ]
    loop = gmsh.model.geo.addCurveLoop(lines)
    gmsh.model.geo.addPlaneSurface([loop], 1)
    gmsh.model.geo.synchronize()
    if crack_line is not None:
        gmsh.model.mesh.embed(1, [crack_line], 2, 1)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), float(mesh_size))
    if local_mesh_size is not None:
        if local_mesh_size <= 0.0 or local_mesh_size >= mesh_size:
            raise ValueError("local_mesh_size must satisfy 0 < local_mesh_size < mesh_size")
        if local_refine_half_width <= 0.0 or local_refine_ymax <= 0.0:
            raise ValueError("local refinement box dimensions must be positive")
        # A symmetric box field resolves the process zone near the notch while
        # retaining a coarser far field.  This mirrors the a-priori refinement
        # used in the reference three-point-bending calculation.
        field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(field, "VIn", float(local_mesh_size))
        gmsh.model.mesh.field.setNumber(field, "VOut", float(mesh_size))
        gmsh.model.mesh.field.setNumber(field, "XMin", mid - float(local_refine_half_width))
        gmsh.model.mesh.field.setNumber(field, "XMax", mid + float(local_refine_half_width))
        gmsh.model.mesh.field.setNumber(field, "YMin", 0.0)
        gmsh.model.mesh.field.setNumber(field, "YMax", float(local_refine_ymax))
        gmsh.model.mesh.field.setNumber(field, "ZMin", -1.0)
        gmsh.model.mesh.field.setNumber(field, "ZMax", 1.0)
        gmsh.model.mesh.field.setNumber(field, "Thickness", float(local_refine_transition))
        gmsh.model.mesh.field.setAsBackgroundMesh(field)
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
class Model5ThreePointBendingCase(CaseBase):
    """Ambati / Miehe three-point bending notched beam."""

    name: str = "model5_three_point_bending"

    length: float = 8.0
    height: float = 2.0
    notch_depth: float = 0.4
    notch_mouth: float = 0.2  # geometric V only
    with_geometric_notch: bool = False
    crack_tol: float = 1e-9

    left_support_x: float = 0.0
    right_support_x: float = 8.0  # Ambati Fig.20: supports at beam ends
    support_half_width: float = 0.05
    load_half_width: float = 0.05
    bd_tol: float = 1e-9

    mesh_size: float = 0.1

    # Ambati §4.4 / Miehe
    lam: float = 12.0
    mu: float = 8.0
    Gc: float = 5.4e-4
    l0: float = 0.03

    debug_mesh: bool = False
    _model: object = None
    mesh: Any = field(default=None, repr=False)

    def model(self):
        if self._model is None:
            raise RuntimeError(
                "Model5ThreePointBendingCase requires _model "
                "(material model instance)."
            )
        return self._model

    def reaction_direction(self):
        return "y"

    def reaction_boundary(self, load: float = 0.0):
        """Report force via support reactions (not the load-patch edges)."""

        def on_supports(points: TensorLike) -> TensorLike:
            return self._on_left_support(points) | self._on_right_support(points)

        # Bottom outward normal is -e_y; sign=+1 yields upward support force.
        return on_supports, "y", 1.0

    def make_mesh(self, nx: Optional[int] = None, ny: Optional[int] = None):
        mesh = _build_gmsh_beam(
            mesh_size=self.mesh_size,
            length=self.length,
            height=self.height,
            notch_depth=self.notch_depth,
            notch_mouth=self.notch_mouth,
            with_geometric_notch=bool(self.with_geometric_notch),
            left_support_x=self.left_support_x,
            right_support_x=self.right_support_x,
        )
        self.mesh = mesh
        if self.debug_mesh:
            mode = (
                "geometric_notch"
                if self.with_geometric_notch
                else "precrack_d=1"
            )
            print(
                f"[model5 mesh] NN={mesh.number_of_nodes()}, "
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
        mid = 0.5 * self.length
        tol = float(self.crack_tol)
        x = points[:, 0]
        y = points[:, 1]
        return (
            (bm.abs(x - mid) < tol)
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
        # ``load`` is the downward magnitude (≥ 0); apply u_y = −load.
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
        # Right support / load: uy fixed, ux free → tangential traction fixed.
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
        # Compact Ambati-like schedule (downward magnitude).
        return bm.concatenate(
            (
                bm.linspace(0.0, 0.04, 41, dtype=bm.float64),
                bm.linspace(0.04, 0.1, 601, dtype=bm.float64)[1:],
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
    case = Model5ThreePointBendingCase(mesh_size=0.2, debug_mesh=True)
    mesh = case.make_mesh()
    print("material:", case.material_dict())
    isBd = mesh.boundary_edge_flag()
    bdedge = bm.where(isBd)[0]
    bc = mesh.entity_barycenter("edge", index=bdedge)
    n_L = int(bm.sum(case._on_left_support(bc)))
    n_R = int(bm.sum(case._on_right_support(bc)))
    n_load = int(bm.sum(case._on_load(bc)))
    print(f"supports L/R={n_L}/{n_R}, load={n_load}")
    assert n_L > 0 and n_R > 0 and n_load > 0
