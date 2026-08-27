"""Unit tests for fracturex.postprocess.vtu_plot.

Covers fan triangulation (no vtk) and a synthetic mesh plot. VTU round-trip
runs only when vtk is importable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fracturex.postprocess.vtu_plot import (
    VtuMesh,
    plot_mesh_scalar,
    triangulate_polygon_cells,
)


def test_triangulate_triangle_passthrough() -> None:
    tris = triangulate_polygon_cells([[0, 1, 2], [2, 3, 0]])
    assert tris.shape == (2, 3)
    np.testing.assert_array_equal(tris, [[0, 1, 2], [2, 3, 0]])


def test_triangulate_quad_splits_into_two() -> None:
    tris = triangulate_polygon_cells([[0, 1, 2, 3]])
    np.testing.assert_array_equal(tris, [[0, 1, 2], [0, 2, 3]])


def test_triangulate_rejects_short_cell() -> None:
    with pytest.raises(ValueError, match="need >= 3"):
        triangulate_polygon_cells([[0, 1]])


def test_scalar_missing_field_lists_available() -> None:
    mesh = VtuMesh(
        xy=np.zeros((3, 2)),
        triangles=np.array([[0, 1, 2]], dtype=np.int64),
        n_cells=1,
        point_data={"damage": np.zeros(3)},
    )
    with pytest.raises(KeyError, match="uh"):
        mesh.scalar("uh")


def test_plot_mesh_scalar_writes_png(tmp_path: Path) -> None:
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    d = np.array([0.0, 0.5, 1.0])
    mesh = VtuMesh(
        xy=xy,
        triangles=np.array([[0, 1, 2]], dtype=np.int64),
        n_cells=1,
        point_data={"damage": d},
    )
    out = tmp_path / "one_tri.png"
    path = plot_mesh_scalar(mesh, "damage", out, dpi=80)
    assert path.is_file()
    assert path.stat().st_size > 0


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("vtk") is None,
    reason="vtk not installed",
)
def test_read_vtu_mesh_roundtrip(tmp_path: Path) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    from fracturex.postprocess.vtu_plot import read_vtu_mesh

    points = vtk.vtkPoints()
    for x, y in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
        points.InsertNextPoint(x, y, 0.0)
    ug = vtk.vtkUnstructuredGrid()
    ug.SetPoints(points)
    quad = vtk.vtkQuad()
    for i in range(4):
        quad.GetPointIds().SetId(i, i)
    ug.InsertNextCell(quad.GetCellType(), quad.GetPointIds())
    d = numpy_to_vtk(np.array([0.0, 0.2, 0.8, 1.0]), deep=True)
    d.SetName("damage")
    ug.GetPointData().AddArray(d)

    vtu = tmp_path / "quad.vtu"
    w = vtk.vtkXMLUnstructuredGridWriter()
    w.SetFileName(str(vtu))
    w.SetInputData(ug)
    w.Write()

    mesh = read_vtu_mesh(vtu)
    assert mesh.n_cells == 1
    assert mesh.triangles.shape == (2, 3)
    np.testing.assert_allclose(mesh.scalar("damage"), [0.0, 0.2, 0.8, 1.0])
