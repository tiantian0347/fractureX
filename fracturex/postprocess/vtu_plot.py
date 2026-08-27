"""Read a 2D VTU mesh and plot a nodal scalar with optional mesh overlay.

Boundary: 2D unstructured grids (triangles, quads, convex polygons). Does not
render 3D cells, warped displacement, or time series (see ``vtu_animation``).

Library:
    from fracturex.postprocess.vtu_plot import read_vtu_mesh, plot_mesh_scalar

CLI:
    python -m fracturex.postprocess.vtu_plot --vtu path/to/step.vtu --out out.png
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

# VTK cell type ids used by UnstructuredGrid (legacy linear 2D).
_VTK_TRIANGLE = 5
_VTK_QUAD = 9
_VTK_POLYGON = 7


@dataclass
class VtuMesh:
    """2D mesh plus nodal fields extracted from one VTU file.

    Attributes:
        xy: node coordinates, shape ``(n_nodes, 2)``, float64.
        triangles: fan-triangulated connectivity for plotting, ``(n_tri, 3)``.
        n_cells: original cell count (before triangulation).
        point_data: nodal arrays, each shape ``(n_nodes,)`` or ``(n_nodes, ncomp)``.
        path: source file if read from disk.
    """

    xy: np.ndarray
    triangles: np.ndarray
    n_cells: int
    point_data: dict[str, np.ndarray] = field(default_factory=dict)
    path: Path | None = None

    def scalar(self, name: str) -> np.ndarray:
        """Return a 1-D nodal field of length ``n_nodes``.

        Raises:
            KeyError: field missing.
            ValueError: field is not a scalar nodal array.
        """
        if name not in self.point_data:
            available = ", ".join(sorted(self.point_data)) or "(none)"
            raise KeyError(f"point field {name!r} not in VTU; available: {available}")
        arr = np.asarray(self.point_data[name])
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.shape != (self.xy.shape[0],):
            raise ValueError(
                f"field {name!r} has shape {arr.shape}, expected ({self.xy.shape[0]},)"
            )
        return np.asarray(arr, dtype=np.float64)


_SCALAR_ALIASES = ("damage", "d", "phase", "phasefield", "pf", "c")


def guess_scalar_field(mesh: VtuMesh, preferred: str | None = None) -> str:
    """Pick a nodal scalar name for phase-field plots.

    Tries ``preferred``, then common damage/phase aliases, then the first
    1-D point array.

    Args:
        mesh: one VTU snapshot.
        preferred: explicit point-data name, or ``None`` to guess.

    Returns:
        A name accepted by :meth:`VtuMesh.scalar`.

    Raises:
        KeyError: no usable scalar point field.
    """
    if preferred:
        mesh.scalar(preferred)
        return preferred
    for name in _SCALAR_ALIASES:
        if name in mesh.point_data:
            try:
                mesh.scalar(name)
                return name
            except ValueError:
                continue
    for name in mesh.point_data:
        try:
            mesh.scalar(name)
            return name
        except ValueError:
            continue
    available = ", ".join(sorted(mesh.point_data)) or "(none)"
    raise KeyError(f"no scalar point field in VTU; available: {available}")


def triangulate_polygon_cells(cells: Sequence[Sequence[int]]) -> np.ndarray:
    """Fan-triangulate 2D cells for matplotlib ``tripcolor`` / ``triplot``.

    A triangle is kept as one cell. An n-gon ``(v0,...,v_{n-1})`` becomes
    ``(v0, v_i, v_{i+1})`` for ``i = 1..n-2``. Convex cells only.

    Args:
        cells: each entry is the vertex ids of one cell, length >= 3.

    Returns:
        ``(n_tri, 3)`` int64 connectivity.

    Raises:
        ValueError: empty input or a cell with fewer than 3 vertices.
    """
    if not cells:
        raise ValueError("cells is empty")
    tris: list[tuple[int, int, int]] = []
    for cell in cells:
        ids = [int(v) for v in cell]
        if len(ids) < 3:
            raise ValueError(f"cell has {len(ids)} vertices, need >= 3")
        v0 = ids[0]
        for i in range(1, len(ids) - 1):
            tris.append((v0, ids[i], ids[i + 1]))
    return np.asarray(tris, dtype=np.int64)


def read_vtu_mesh(path: str | Path) -> VtuMesh:
    """Load a 2D VTU unstructured grid into :class:`VtuMesh`.

    Args:
        path: ``.vtu`` file.

    Returns:
        Mesh with all point-data arrays copied to numpy.

    Raises:
        FileNotFoundError: path missing.
        ImportError: ``vtk`` is not installed.
        ValueError: grid has no cells, or a cell is not a 2D polygon.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as exc:
        raise ImportError("read_vtu_mesh requires the vtk package") from exc

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    n_cells = int(grid.GetNumberOfCells())
    if n_cells == 0:
        raise ValueError(f"{path} has 0 cells")

    xy = np.asarray(vtk_to_numpy(grid.GetPoints().GetData())[:, :2], dtype=np.float64)
    cells: list[list[int]] = []
    for i in range(n_cells):
        cell = grid.GetCell(i)
        ctype = int(cell.GetCellType())
        npts = int(cell.GetNumberOfPoints())
        if ctype not in (_VTK_TRIANGLE, _VTK_QUAD, _VTK_POLYGON) and npts < 3:
            raise ValueError(f"{path} cell {i} type={ctype} npts={npts} is not a 2D polygon")
        if npts < 3:
            raise ValueError(f"{path} cell {i} has npts={npts}")
        cells.append([int(cell.GetPointId(j)) for j in range(npts)])

    point_data: dict[str, np.ndarray] = {}
    pd = grid.GetPointData()
    for k in range(pd.GetNumberOfArrays()):
        name = pd.GetArrayName(k)
        if name is None:
            continue
        point_data[str(name)] = np.asarray(vtk_to_numpy(pd.GetArray(k)))

    return VtuMesh(
        xy=xy,
        triangles=triangulate_polygon_cells(cells),
        n_cells=n_cells,
        point_data=point_data,
        path=path,
    )


def draw_mesh_scalar(
    ax,
    mesh: VtuMesh,
    field: str | np.ndarray,
    *,
    show_mesh: bool = True,
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    cmap: str = "YlOrRd",
    mesh_color: str = "0.25",
    mesh_lw: float = 0.18,
    mesh_alpha: float = 0.55,
):
    """Draw a nodal scalar on an existing matplotlib axes.

    Args:
        ax: target axes.
        mesh: 2D mesh.
        field: point-data name, or length-``n_nodes`` array.
        show_mesh: overlay element edges.
        vmin, vmax: color limits; ``None`` uses data min/max.
        xlim, ylim: axis limits; default is the mesh bounding box.
        cmap, mesh_color, mesh_lw, mesh_alpha: style.

    Returns:
        The ``tripcolor`` mappable (for a colorbar).

    Raises:
        ValueError: field length does not match node count.
    """
    import matplotlib.tri as mtri

    scalar = mesh.scalar(field) if isinstance(field, str) else np.asarray(field, dtype=np.float64)
    if scalar.shape != (mesh.xy.shape[0],):
        raise ValueError(f"scalar shape {scalar.shape} != ({mesh.xy.shape[0]},)")
    triang = mtri.Triangulation(mesh.xy[:, 0], mesh.xy[:, 1], mesh.triangles)
    tpc = ax.tripcolor(
        triang,
        scalar,
        cmap=cmap,
        shading="gouraud",
        vmin=scalar.min() if vmin is None else vmin,
        vmax=scalar.max() if vmax is None else vmax,
    )
    if show_mesh:
        ax.triplot(triang, color=mesh_color, lw=mesh_lw, alpha=mesh_alpha)
    ax.set_aspect("equal")
    if xlim is None:
        ax.set_xlim(float(mesh.xy[:, 0].min()), float(mesh.xy[:, 0].max()))
    else:
        ax.set_xlim(*xlim)
    if ylim is None:
        ax.set_ylim(float(mesh.xy[:, 1].min()), float(mesh.xy[:, 1].max()))
    else:
        ax.set_ylim(*ylim)
    return tpc


def plot_mesh_scalar(
    mesh: VtuMesh,
    field: str | np.ndarray,
    out: str | Path,
    *,
    show_mesh: bool = True,
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    cmap: str = "YlOrRd",
    mesh_color: str = "0.25",
    mesh_lw: float = 0.18,
    dpi: int = 220,
    colorbar_label: str | None = None,
    figsize: tuple[float, float] = (5.2, 5.0),
) -> Path:
    """Save a 2D mesh plot of a nodal scalar.

    Args:
        mesh: data from :func:`read_vtu_mesh` or a constructed :class:`VtuMesh`.
        field: point-data name, or a length-``n_nodes`` array.
        out: output ``.png`` / ``.pdf`` path.
        show_mesh: overlay ``triplot`` edges.
        vmin, vmax: color limits; ``None`` uses data min/max.
        xlim, ylim: axis limits; default is the mesh bounding box.
        title: figure title. ``None`` uses the source filename.
        cmap: matplotlib colormap name.
        mesh_color, mesh_lw: edge style.
        dpi: raster resolution.
        colorbar_label: colorbar text; default is the field name.
        figsize: matplotlib figure size in inches ``(width, height)``.

    Returns:
        Resolved output path.

    Raises:
        ValueError: field length does not match node count.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scalar = mesh.scalar(field) if isinstance(field, str) else np.asarray(field, dtype=np.float64)
    if scalar.shape != (mesh.xy.shape[0],):
        raise ValueError(f"scalar shape {scalar.shape} != ({mesh.xy.shape[0]},)")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    tpc = draw_mesh_scalar(
        ax,
        mesh,
        scalar,
        show_mesh=show_mesh,
        vmin=vmin,
        vmax=vmax,
        xlim=xlim,
        ylim=ylim,
        cmap=cmap,
        mesh_color=mesh_color,
        mesh_lw=mesh_lw,
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    field_name = field if isinstance(field, str) else "scalar"
    if title is None:
        src = mesh.path.name if mesh.path is not None else "mesh"
        title = f"{src}  NC={mesh.n_cells}"
    ax.set_title(title)
    cbar = fig.colorbar(tpc, ax=ax, shrink=0.82, pad=0.03)
    cbar.set_label(colorbar_label if colorbar_label is not None else field_name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path.resolve()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot a VTU nodal field with mesh overlay.")
    p.add_argument("--vtu", required=True, type=Path, help="input .vtu")
    p.add_argument("--out", required=True, type=Path, help="output image (.png/.pdf)")
    p.add_argument("--field", default="damage", help="point-data array name")
    p.add_argument("--title", default=None)
    p.add_argument("--no-mesh", action="store_true", help="hide element edges")
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=1.0)
    p.add_argument("--dpi", type=int, default=220)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry. Returns process exit code."""
    args = _build_parser().parse_args(argv)
    mesh = read_vtu_mesh(args.vtu)
    plot_mesh_scalar(
        mesh,
        args.field,
        args.out,
        show_mesh=not args.no_mesh,
        vmin=args.vmin,
        vmax=args.vmax,
        title=args.title,
        dpi=args.dpi,
    )
    print(f"wrote {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
