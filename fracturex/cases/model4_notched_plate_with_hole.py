"""Ambati / COMSOL notched plate with an offset hole (2D).

Classic mixed-mode phase-field benchmark (Ambati et al., Comput. Mech. 2015
§4.6; COMSOL *Brittle Fracture of a Holed Plate*). Geometry in millimetres.

Geometry (mm)
-------------
* Outer plate: width 65, height 120 (origin at bottom-left).
* Left pre-crack / notch: length 10, centre height 65.
* Large hole: radius 10, centre (36.5, 51) — offset to induce mixed mode.
* Lower pin hole: radius 5, centre (20, 20).
* Upper pin hole: radius 5, centre (20, 100).

Notch representation (discretization-dependent)
-----------------------------------------------
* **Hu–Zhang** (default ``with_geometric_notch=False``): intact left edge;
  pre-crack is enforced by phase-field ``d = 1`` on the segment
  ``y = notch_y``, ``x ∈ [0, notch_width]`` (same recipe as
  ``SquareTensionPreCrackCase`` / model-2). Geometric cuts create dangling
  corners that break the Hu–Zhang stress space.
* **Standard FEM / IP-FEM** (``with_geometric_notch=True``): finite-height
  geometric notch (COMSOL ``notchHeight``, default 0.5 mm) cut into the mesh.

Boundary conditions
-------------------
* Lower pin circle: ``u = 0`` (fixed; Ambati "fixed lower pin").
* Upper pin circle: ``u_y = load``, ``u_x`` free (displacement-controlled
  pin; a full rigid-connector free-to-rotate model is not used here).
* Outer boundary, (optional) notch faces, and large hole: traction-free.
* Phase field: homogeneous Neumann, except Hu–Zhang pre-crack ``d`` BC above.

Material (Ambati §4.6, kN–mm units)
-----------------------------------
``lam = 1.94``, ``mu = 2.45``, ``Gc = 2.28e-3``, ``l0 = 0.1``
(equivalent to ``E ≈ 6 GPa``, ``ν ≈ 0.22``).

Loading
-------
Ambati: fixed increment ``Δu = 1e-3`` mm. Default schedule
``np.linspace(0, 2.0, 2001)`` (same span as the old ``fracture_pu`` script).

Bugs fixed relative to ``ttthesis/.../model4_fracture_...py``
------------------------------------------------------------
1. Geometric-notch path uses finite height 0.5 mm (COMSOL ``notchHeight``),
   not a zero-thickness out-and-back slit.
2. Pin selectors use OR of the two pin circles only where intended;
   the old ``is_inter_boundary`` used AND and was empty.
3. No interactive ``gmsh.fltk.run()``.
4. Mesh size is an explicit parameter (default 2 mm; refine for production).
5. Hu–Zhang path avoids the geometric notch (pre-crack via ``d=1``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.typing import TensorLike

from .base import CaseBase, DirichletPiece
from fracturex.boundarycondition.huzhang_boundary_condition import build_isNedge_from_isD


def _build_gmsh_mesh(
    *,
    mesh_size: float,
    width: float,
    height: float,
    notch_width: float,
    notch_height: float,
    notch_y: float,
    hole_center: Tuple[float, float],
    hole_radius: float,
    lower_pin_center: Tuple[float, float],
    upper_pin_center: Tuple[float, float],
    pin_radius: float,
    with_geometric_notch: bool,
) -> TriangleMesh:
    """Build the Ambati/COMSOL plate with gmsh; return a TriangleMesh."""
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("model4_notched_plate_with_hole")

    p = {}
    p[1] = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)      # BL
    p[2] = gmsh.model.geo.addPoint(width, 0.0, 0.0)     # BR
    p[3] = gmsh.model.geo.addPoint(width, height, 0.0)  # TR
    p[4] = gmsh.model.geo.addPoint(0.0, height, 0.0)    # TL

    crack_line = None  # embedded only in Hu–Zhang (pre-crack) mode
    if with_geometric_notch:
        # Counter-clockwise outer contour with left notch cut.
        # Left above notch → notch top → tip → notch bottom → left below.
        y_n0 = notch_y - 0.5 * notch_height
        y_n1 = notch_y + 0.5 * notch_height
        p[5] = gmsh.model.geo.addPoint(0.0, y_n1, 0.0)
        p[6] = gmsh.model.geo.addPoint(notch_width, y_n1, 0.0)
        p[7] = gmsh.model.geo.addPoint(notch_width, y_n0, 0.0)
        p[8] = gmsh.model.geo.addPoint(0.0, y_n0, 0.0)
        lines = [
            gmsh.model.geo.addLine(p[1], p[2]),  # bottom
            gmsh.model.geo.addLine(p[2], p[3]),  # right
            gmsh.model.geo.addLine(p[3], p[4]),  # top
            gmsh.model.geo.addLine(p[4], p[5]),  # left above notch
            gmsh.model.geo.addLine(p[5], p[6]),  # notch top face
            gmsh.model.geo.addLine(p[6], p[7]),  # notch tip
            gmsh.model.geo.addLine(p[7], p[8]),  # notch bottom face
            gmsh.model.geo.addLine(p[8], p[1]),  # left below notch
        ]
    else:
        # Intact left edge, but split at notch_y so the pre-crack line
        # can start from a boundary vertex and be embedded in the surface
        # (Lagrange DOFs then land on y = notch_y for d = 1 seeding).
        p[5] = gmsh.model.geo.addPoint(0.0, notch_y, 0.0)           # left mouth
        p[6] = gmsh.model.geo.addPoint(notch_width, notch_y, 0.0)    # tip
        lines = [
            gmsh.model.geo.addLine(p[1], p[2]),  # bottom
            gmsh.model.geo.addLine(p[2], p[3]),  # right
            gmsh.model.geo.addLine(p[3], p[4]),  # top
            gmsh.model.geo.addLine(p[4], p[5]),  # left above crack
            gmsh.model.geo.addLine(p[5], p[1]),  # left below crack
        ]
        crack_line = gmsh.model.geo.addLine(p[5], p[6])

    outer = gmsh.model.geo.addCurveLoop(lines)

    def _circle_loop(cx: float, cy: float, r: float):
        c = gmsh.model.geo.addPoint(cx, cy, 0.0)
        a = gmsh.model.geo.addPoint(cx - r, cy, 0.0)
        b = gmsh.model.geo.addPoint(cx + r, cy, 0.0)
        arc0 = gmsh.model.geo.addCircleArc(a, c, b)
        arc1 = gmsh.model.geo.addCircleArc(b, c, a)
        return gmsh.model.geo.addCurveLoop([arc0, arc1])

    hole_loop = _circle_loop(*hole_center, hole_radius)
    lower_pin_loop = _circle_loop(*lower_pin_center, pin_radius)
    upper_pin_loop = _circle_loop(*upper_pin_center, pin_radius)

    # Plane surface = outer minus three holes.
    gmsh.model.geo.addPlaneSurface(
        [outer, hole_loop, lower_pin_loop, upper_pin_loop], 1
    )
    gmsh.model.geo.synchronize()
    if crack_line is not None:
        # Force mesh edges along the pre-crack segment (no geometric cut).
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
class Model4NotchedPlateWithHoleCase(CaseBase):
    """
    Ambati notched plate with offset hole under pin loading (2D plane stress
    in the literature; the Hu–Zhang driver may still assemble plane-strain
    Lamé parameters — pass the Ambati ``lam``/``mu`` through the material
    model and keep geometry in mm).

    Set ``with_geometric_notch=False`` (default) for Hu–Zhang; set ``True``
    for standard Lagrange FEM / IP-FEM geometric-notch meshes.
    """

    name: str = "model4_notched_plate_with_hole"

    # plate
    width: float = 65.0
    height: float = 120.0

    # left notch / pre-crack length and height
    notch_width: float = 10.0
    notch_height: float = 0.5  # used only when with_geometric_notch=True
    notch_y: float = 65.0
    # Hu–Zhang: no geometric cut; crack via d=1 on this segment.
    # FEM / IP-FEM: set True to cut a finite-height notch into the mesh.
    with_geometric_notch: bool = False
    crack_tol: float = 1e-9

    # large hole
    hole_center: Tuple[float, float] = (36.5, 51.0)
    hole_radius: float = 10.0

    # pins
    lower_pin_center: Tuple[float, float] = (20.0, 20.0)
    upper_pin_center: Tuple[float, float] = (20.0, 100.0)
    pin_radius: float = 5.0

    # mesh
    mesh_size: float = 2.0
    pin_tol: float = 0.05  # absolute radial tol; also scaled with mesh_size

    # Ambati material (exposed for runners; case itself does not own the model)
    lam: float = 1.94
    mu: float = 2.45
    Gc: float = 2.28e-3
    l0: float = 0.1

    debug_mesh: bool = False
    _model: object = None
    mesh: Any = field(default=None, repr=False)

    def model(self):
        if self._model is None:
            raise RuntimeError(
                "Model4NotchedPlateWithHoleCase requires _model "
                "(material model instance)."
            )
        return self._model

    def reaction_direction(self):
        return "y"

    # -----------------------
    # mesh
    # -----------------------
    def make_mesh(self, nx: Optional[int] = None, ny: Optional[int] = None):
        # nx/ny ignored; kept for CaseBase signature compatibility.
        mesh = _build_gmsh_mesh(
            mesh_size=self.mesh_size,
            width=self.width,
            height=self.height,
            notch_width=self.notch_width,
            notch_height=self.notch_height,
            notch_y=self.notch_y,
            hole_center=self.hole_center,
            hole_radius=self.hole_radius,
            lower_pin_center=self.lower_pin_center,
            upper_pin_center=self.upper_pin_center,
            pin_radius=self.pin_radius,
            with_geometric_notch=bool(self.with_geometric_notch),
        )
        self.mesh = mesh
        if self.debug_mesh:
            NN = mesh.number_of_nodes()
            NC = mesh.number_of_cells()
            mode = (
                "geometric_notch"
                if self.with_geometric_notch
                else "precrack_d=1"
            )
            print(
                f"[model4 mesh] NN={NN}, NC={NC}, "
                f"mesh_size={self.mesh_size}, mode={mode}"
            )
        return mesh

    # -----------------------
    # boundary selectors
    # -----------------------
    def _pin_tol(self) -> float:
        return max(float(self.pin_tol), 0.35 * float(self.mesh_size))

    def _on_circle(
        self,
        points: TensorLike,
        center: Tuple[float, float],
        radius: float,
    ) -> TensorLike:
        x = points[:, 0]
        y = points[:, 1]
        r = bm.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        return bm.abs(r - radius) < self._pin_tol()

    def _on_lower_pin(self, points: TensorLike) -> TensorLike:
        return self._on_circle(points, self.lower_pin_center, self.pin_radius)

    def _on_upper_pin(self, points: TensorLike) -> TensorLike:
        return self._on_circle(points, self.upper_pin_center, self.pin_radius)

    def _on_precrack(self, points: TensorLike) -> TensorLike:
        """Initial crack segment: y = notch_y, x in [0, notch_width]."""
        x = points[:, 0]
        y = points[:, 1]
        tol = float(self.crack_tol)
        return (
            (bm.abs(y - self.notch_y) < tol)
            & (x >= -tol)
            & (x <= self.notch_width + tol)
        )

    def isD_bd(self, points: TensorLike) -> TensorLike:
        # Hu–Zhang D/N split: both pins are essential displacement boundaries.
        return self._on_lower_pin(points) | self._on_upper_pin(points)

    def dirichlet_pieces(self, load: float) -> List[DirichletPiece]:
        def u_zero(points: TensorLike):
            GD = points.shape[-1]
            return bm.zeros(points.shape[:-1] + (GD,), dtype=bm.float64)

        def u_top_pin(points: TensorLike):
            GD = points.shape[-1]
            out = bm.zeros(points.shape[:-1] + (GD,), dtype=bm.float64)
            out[..., 1] = load
            return out

        return [
            DirichletPiece(
                threshold=self._on_lower_pin,
                value=u_zero,
                direction=None,
                tag="fix",
            ),
            DirichletPiece(
                threshold=self._on_upper_pin,
                value=u_top_pin,
                direction="y",
                tag="load",
            ),
        ]

    def neumann_data(self, load: float = 0.0):
        isNedge_free = build_isNedge_from_isD(self.mesh, self.isD_bd)
        gd0 = bm.array([0.0, 0.0], dtype=bm.float64)
        # Free faces (outer / optional notch / large hole): full traction-free.
        # Upper pin: uy prescribed, ux free → fix only tangential traction.
        return [
            (isNedge_free, gd0, "nt", None),
            (self._on_upper_pin, gd0, "nt", "t"),
        ]

    def phasefield_initial_damage_data(self, load: float) -> Optional[Any]:
        # Hu–Zhang: seed pre-crack with d = 1 (no geometric notch).
        if self.with_geometric_notch:
            return None
        return [
            {
                "bcdof": self._on_precrack,
                "value": 1.0,
            }
        ]

    def phasefield_dirichlet_data(self, load: float) -> Optional[Any]:
        # Geometric-notch (FEM/IPFEM): no essential d-BC.
        # Pre-crack (Hu–Zhang): keep dd = 0 so initialized d = 1 is preserved.
        if self.with_geometric_notch:
            return None
        return [
            {
                "bcdof": self._on_precrack,
                "value": 0.0,
            }
        ]

    def default_loads(self):
        # Ambati: Δu = 1e-3 mm up to 2 mm (2000 steps).
        return bm.linspace(0.0, 2.0, 2001, dtype=bm.float64)

    def material_dict(self) -> dict:
        """Convenience for runners that build PhaseFieldDamageModel."""
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


# Backwards-friendly alias
Model4HoledPlateCase = Model4NotchedPlateWithHoleCase


if __name__ == "__main__":
    # Mesh / BC smoke check (no solve). Prefer:
    #   PYTHONPATH=../fealpy:. python -m fracturex.cases.model4_notched_plate_with_hole
    case = Model4NotchedPlateWithHoleCase(mesh_size=2.0, debug_mesh=True)
    mesh = case.make_mesh()
    print("material:", case.material_dict())
    print("with_geometric_notch:", case.with_geometric_notch)

    isBd = mesh.boundary_edge_flag()
    bdedge = bm.where(isBd)[0]
    bc = mesh.entity_barycenter("edge", index=bdedge)
    n_lower = int(bm.sum(case._on_lower_pin(bc)))
    n_upper = int(bm.sum(case._on_upper_pin(bc)))
    n_D = int(bm.sum(case.isD_bd(bc)))
    print(
        f"boundary edges={int(bdedge.shape[0])}, "
        f"lower_pin={n_lower}, upper_pin={n_upper}, isD={n_D}"
    )
    assert n_lower > 0 and n_upper > 0, "pin circles not captured on mesh"
    assert n_D == n_lower + n_upper, "pin masks should be disjoint"
