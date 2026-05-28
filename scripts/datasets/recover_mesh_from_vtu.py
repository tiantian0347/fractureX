"""Recover a recorder-compatible mesh.npz from a vtu frame for legacy runs.

Use case: a recorder dir produced before 2026-05-28 (no save_mesh) has its
FE mesh embedded in vtk/step_XXXX_load_*.vtu (corner-only triangles). This
script reads frame 0, rebuilds a TriangleMesh, recomputes the boundary
edge masks via the original case class, and writes
``<recorder_dir>/mesh.npz`` so that
:func:`fracturex.postprocess.dataset_export.load_discr_from_dir` can rebuild
the discretization.

Run:
    PYTHONPATH=$PWD $FEALPY_PYTHON \\
      scripts/datasets/recover_mesh_from_vtu.py \\
      --recorder-dir results/phasefield/model0_circular_notch/paper_aux_h1/epsg_1e-06 \\
      --case model0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def _read_vtu_p1_triangle(path: Path) -> tuple[np.ndarray, np.ndarray]:
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    ug = reader.GetOutput()
    node = vtk_to_numpy(ug.GetPoints().GetData())[:, :2].astype(np.float64).copy()
    n_cells = ug.GetNumberOfCells()
    cells: list[list[int]] = []
    for i in range(n_cells):
        c = ug.GetCell(i)
        if c.GetCellType() != vtk.VTK_TRIANGLE:
            raise ValueError(
                f"non-triangle cell at index {i} (type={c.GetCellType()}); "
                "only P1 triangle vtu is supported."
            )
        cells.append([c.GetPointId(0), c.GetPointId(1), c.GetPointId(2)])
    cell = np.asarray(cells, dtype=np.int64)
    return node, cell


def _build_case(name: str):
    if name == "model0":
        from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
        return Model0CircularNotchCase()
    raise ValueError(f"unknown case {name!r}; supported: model0")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recorder-dir", required=True, type=Path)
    ap.add_argument("--case", required=True, choices=["model0"])
    ap.add_argument("--p-sigma", type=int, default=3)
    ap.add_argument("--damage-p", type=int, default=1)
    ap.add_argument("--u-order", type=int, default=2)
    ap.add_argument("--use-relaxation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--vtk-frame", default="step_0000_load_0.000000e+00.vtu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rec_dir: Path = args.recorder_dir
    out_path = rec_dir / "mesh.npz"
    if out_path.exists() and not args.force:
        print(f"{out_path} exists; pass --force to overwrite")
        return 1

    vtu_path = rec_dir / "vtk" / args.vtk_frame
    if not vtu_path.exists():
        # fall back to lexicographic first
        candidates = sorted((rec_dir / "vtk").glob("step_*.vtu"))
        if not candidates:
            print(f"no vtu under {rec_dir/'vtk'}")
            return 1
        vtu_path = candidates[0]
    print(f"reading {vtu_path}")
    node, cell = _read_vtu_p1_triangle(vtu_path)
    print(f"  NN={node.shape[0]}  NC={cell.shape[0]}")

    # Build a real TriangleMesh, augment boundary, then run the same isD_bd
    # logic as HuZhangDiscretization.build to derive the boundary masks.
    from fealpy.backend import backend_manager as bm
    from fealpy.mesh import TriangleMesh
    from fealpy.functionspace.huzhang_fe_space_2d import HuZhangFESpace2d

    from fracturex.boundarycondition.huzhang_boundary_condition import (
        build_isNedge_from_isD,
    )
    from fracturex.utilfuc.mesh_patch import augment_boundary_edges_inplace

    case = _build_case(args.case)
    mesh = TriangleMesh(bm.asarray(node), bm.asarray(cell))

    extra = None
    if hasattr(case, "crack_edge_mask") and getattr(case, "crack_edge_mask", None) is not None:
        extra = case.crack_edge_mask(mesh)
    augment_boundary_edges_inplace(mesh, extra)

    isNedge_bd = build_isNedge_from_isD(mesh, case.isD_bd)
    space_sigma = HuZhangFESpace2d(
        mesh, p=args.p_sigma, use_relaxation=args.use_relaxation, bd_stress=isNedge_bd
    )
    isNedge_full = np.asarray(bm.asarray(space_sigma.isNedge)).reshape(-1).astype(bool)
    be_aug = np.asarray(bm.asarray(mesh.boundary_edge_flag())).reshape(-1).astype(bool)

    np.savez_compressed(
        out_path,
        node=node,
        cell=cell,
        p_sigma=int(args.p_sigma),
        damage_p=int(args.damage_p),
        u_space_order=int(args.u_order),
        use_relaxation=bool(args.use_relaxation),
        boundary_edge_flag_aug=be_aug,
        is_neumann_edge=isNedge_full,
    )
    print(f"wrote {out_path}  (gdof_sigma={space_sigma.number_of_global_dofs()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
