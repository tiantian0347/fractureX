from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# VTK enum for high-order triangle cell.
VTK_LAGRANGE_TRIANGLE = 69
VTK_QUAD = 9
VTK_HEXAHEDRON = 12


def _triangle_lagrange_barycentric_order(order: int) -> np.ndarray:
    """
    Build barycentric coordinates in VTK Lagrange-triangle node ordering.

    Ordering convention:
    1) 3 vertices
    2) edge nodes on edges (0-1), (1-2), (2-0)
    3) face interior nodes in recursive shell ordering
    """
    p = int(order)
    if p < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    # Corner nodes: [(1,0,0), (0,1,0), (0,0,1)]
    out: List[Tuple[float, float, float]] = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    if p == 1:
        return np.asarray(out, dtype=float)

    # Edge (0,1): (a,b,0), a+b=1
    for i in range(1, p):
        out.append(((p - i) / p, i / p, 0.0))
    # Edge (1,2): (0,a,b), a+b=1
    for i in range(1, p):
        out.append((0.0, (p - i) / p, i / p))
    # Edge (2,0): (b,0,a), a+b=1
    for i in range(1, p):
        out.append((i / p, 0.0, (p - i) / p))

    # Interior nodes: recursive shrink
    if p >= 3:
        inner = _triangle_lagrange_barycentric_order(p - 3)
        # Map inner triangle corners from (1,0,0)/(0,1,0)/(0,0,1)
        # to ( (p-2)/p,1/p,1/p ), (1/p,(p-2)/p,1/p), (1/p,1/p,(p-2)/p ).
        mapped = (inner * (p - 3) + 1.0) / p
        out.extend([tuple(v.tolist()) for v in mapped])

    return np.asarray(out, dtype=float)


def _format_data_array(name: str, arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        ncomp = 1
        flat = arr
    elif arr.ndim == 2:
        ncomp = arr.shape[1]
        flat = arr.reshape(-1)
    else:
        raise ValueError(f"Data array {name} must be 1D or 2D, got shape {arr.shape}")

    txt = " ".join(f"{float(v):.16e}" for v in flat)
    if ncomp == 1:
        return (
            f'        <DataArray type="Float64" Name="{name}" format="ascii">\n'
            f"          {txt}\n"
            f"        </DataArray>\n"
        )
    return (
        f'        <DataArray type="Float64" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">\n'
        f"          {txt}\n"
        f"        </DataArray>\n"
    )


def write_lagrange_triangle_vtu(
    *,
    fname: str,
    mesh,
    order: int,
    point_data: Dict[str, np.ndarray],
) -> None:
    """
    Write a VTU with VTK_LAGRANGE_TRIANGLE cells from sampled point data.

    Notes
    -----
    - This function duplicates high-order interpolation points per parent cell.
      It avoids global point dedup/reindex complexity and is robust for output.
    - `point_data` arrays must have shape (NC * Np,) or (NC * Np, ncomp), where
      Np=(order+1)*(order+2)//2.
    """
    d = os.path.dirname(fname)
    if d:
        os.makedirs(d, exist_ok=True)

    order = int(order)
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    cell = np.asarray(mesh.entity("cell"), dtype=np.int64)
    node = np.asarray(mesh.entity("node"), dtype=float)
    if cell.ndim != 2 or cell.shape[1] != 3:
        raise NotImplementedError("write_lagrange_triangle_vtu only supports 2D triangle mesh.")

    NC = int(mesh.number_of_cells())
    GD = int(mesh.geo_dimension())
    Np = (order + 1) * (order + 2) // 2

    bcs = _triangle_lagrange_barycentric_order(order)  # (Np, 3)
    if bcs.shape[0] != Np:
        raise RuntimeError("Unexpected barycentric node count for Lagrange triangle.")

    v0 = node[cell[:, 0], :]
    v1 = node[cell[:, 1], :]
    v2 = node[cell[:, 2], :]

    p0 = bcs[:, 0][None, :, None]
    p1 = bcs[:, 1][None, :, None]
    p2 = bcs[:, 2][None, :, None]
    xyz = p0 * v0[:, None, :] + p1 * v1[:, None, :] + p2 * v2[:, None, :]
    if GD == 2:
        xyz = np.concatenate([xyz, np.zeros((NC, Np, 1), dtype=xyz.dtype)], axis=-1)
    points = xyz.reshape(NC * Np, 3)

    connectivity = np.arange(NC * Np, dtype=np.int64).reshape(NC, Np)
    offsets = np.arange(1, NC + 1, dtype=np.int64) * Np
    types = np.full((NC,), VTK_LAGRANGE_TRIANGLE, dtype=np.uint8)
    higher_order_degrees = np.full((NC,), order, dtype=np.int64)

    # Validate point-data shape.
    for key, val in point_data.items():
        arr = np.asarray(val)
        if arr.shape[0] != NC * Np:
            raise ValueError(
                f"Point data `{key}` first dimension mismatch: expected {NC * Np}, got {arr.shape[0]}"
            )

    points_txt = " ".join(f"{float(v):.16e}" for v in points.reshape(-1))
    conn_txt = " ".join(str(int(v)) for v in connectivity.reshape(-1))
    off_txt = " ".join(str(int(v)) for v in offsets)
    type_txt = " ".join(str(int(v)) for v in types)
    deg_txt = " ".join(str(int(v)) for v in higher_order_degrees)

    pieces: List[str] = []
    pieces.append('<?xml version="1.0"?>\n')
    pieces.append(
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n'
    )
    pieces.append("  <UnstructuredGrid>\n")
    pieces.append(f'    <Piece NumberOfPoints="{NC * Np}" NumberOfCells="{NC}">\n')
    pieces.append("      <Points>\n")
    pieces.append(
        '        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n'
        f"          {points_txt}\n"
        "        </DataArray>\n"
    )
    pieces.append("      </Points>\n")

    pieces.append("      <Cells>\n")
    pieces.append(
        '        <DataArray type="Int64" Name="connectivity" format="ascii">\n'
        f"          {conn_txt}\n"
        "        </DataArray>\n"
    )
    pieces.append(
        '        <DataArray type="Int64" Name="offsets" format="ascii">\n'
        f"          {off_txt}\n"
        "        </DataArray>\n"
    )
    pieces.append(
        '        <DataArray type="UInt8" Name="types" format="ascii">\n'
        f"          {type_txt}\n"
        "        </DataArray>\n"
    )
    pieces.append(
        '        <DataArray type="Int64" Name="HigherOrderDegrees" format="ascii">\n'
        f"          {deg_txt}\n"
        "        </DataArray>\n"
    )
    pieces.append("      </Cells>\n")

    pieces.append("      <PointData>\n")
    for k, v in point_data.items():
        pieces.append(_format_data_array(k, np.asarray(v)))
    pieces.append("      </PointData>\n")
    pieces.append("    </Piece>\n")
    pieces.append("  </UnstructuredGrid>\n")
    pieces.append("</VTKFile>\n")

    with open(fname, "w", encoding="utf-8") as f:
        f.write("".join(pieces))


def sample_fields_for_lagrange_triangle(
    *,
    mesh,
    order: int,
    field_specs: Iterable[Tuple[str, object]],
) -> Dict[str, np.ndarray]:
    """
    Sample FE functions on VTK Lagrange-triangle nodes (duplicated per cell).

    Parameters
    ----------
    mesh : Triangle mesh
    order : int
        Lagrange order.
    field_specs : iterable of (name, callable_fe_function)
        Each function should support `fun(bcs, index=...)`.
    """
    order = int(order)
    bcs = _triangle_lagrange_barycentric_order(order)
    npts = (order + 1) * (order + 2) // 2
    NC = int(mesh.number_of_cells())
    out: Dict[str, np.ndarray] = {}
    for name, fun in field_specs:
        raw = fun(bcs, index=np.arange(NC, dtype=np.int64))
        arr = np.asarray(raw)
        if arr.ndim == 3:
            if arr.shape[0] == NC and arr.shape[1] == npts:
                arr = arr.reshape(NC * npts, arr.shape[2])
            elif arr.shape[0] == npts and arr.shape[1] == NC:
                arr = np.transpose(arr, (1, 0, 2)).reshape(NC * npts, arr.shape[2])
            else:
                arr = arr.reshape(NC * arr.shape[1], arr.shape[2])
        elif arr.ndim == 2:
            if arr.shape[0] == NC and arr.shape[1] == npts:
                arr = arr.reshape(NC * npts)
            elif arr.shape[0] == npts and arr.shape[1] == NC:
                arr = arr.T.reshape(NC * npts)
            elif arr.shape[0] == NC * npts:
                pass
            else:
                raise ValueError(
                    f"Unsupported 2D sampled shape for field {name}: {arr.shape}, "
                    f"expected ({NC}, {npts}), ({npts}, {NC}), or ({NC * npts}, *)"
                )
        elif arr.ndim == 1:
            if arr.size != NC * npts:
                raise ValueError(
                    f"Unsupported 1D sampled shape for field {name}: {arr.shape}, "
                    f"expected total size {NC * npts}"
                )
        else:
            raise ValueError(f"Unsupported sampled shape for field {name}: {arr.shape}")
        if arr.shape[0] != NC * npts:
            raise ValueError(
                f"Field `{name}` first dimension mismatch: expected {NC * npts}, got {arr.shape[0]}"
            )
        out[name] = arr
    return out


def _interval_lagrange_barycentric(order: int) -> np.ndarray:
    """
    1D reference Lagrange nodes as fealpy-style barycentric pairs (lambda0, lambda1),
    degree ``order`` (same convention as ``Mesh.multi_index_matrix(p, 1) / p``).
    """
    p = int(order)
    if p < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    ldof = p + 1
    multi_index = np.zeros((ldof, 2), dtype=np.float64)
    multi_index[:, 0] = np.arange(p, -1, -1, dtype=np.float64)
    multi_index[:, 1] = p - multi_index[:, 0]
    return multi_index / float(p)


def sample_fields_tensor_product_fe(
    *,
    mesh,
    order: int,
    cell_dim: int,
    field_specs: Iterable[Tuple[str, object]],
) -> Dict[str, np.ndarray]:
    """
    Sample scalar / vector FE functions on a tensor product of 1D Lagrange nodes
    (per cell), matching fealpy ``QuadrangleMesh`` / ``HexahedronMesh`` shape
    evaluation with ``bc = (line, line[, line])``.

    Parameters
    ----------
    mesh
        Quadrilateral (``cell_dim == 2``) or hexahedral (``cell_dim == 3``) mesh.
    order : int
        Polynomial degree ``p`` (number of 1D nodes = ``p + 1``).
    cell_dim : int
        2 for quads, 3 for hexes.
    field_specs : iterable of (name, fe_function)
        Each ``fe_function`` must support ``fun(bcs, index=...)`` with tuple ``bcs``.
    """
    order = int(order)
    line = _interval_lagrange_barycentric(order)
    if cell_dim == 2:
        bcs = (line, line)
        npts = (order + 1) ** 2
    elif cell_dim == 3:
        bcs = (line, line, line)
        npts = (order + 1) ** 3
    else:
        raise ValueError("cell_dim must be 2 or 3")

    nc = int(mesh.number_of_cells())
    idx = np.arange(nc, dtype=np.int64)
    out: Dict[str, np.ndarray] = {}
    for name, fun in field_specs:
        raw = fun(bcs, index=idx)
        arr = np.asarray(raw)
        if arr.ndim == 3:
            if arr.shape[0] == nc and arr.shape[1] == npts:
                arr = arr.reshape(nc * npts, arr.shape[2])
            elif arr.shape[0] == npts and arr.shape[1] == nc:
                arr = np.transpose(arr, (1, 0, 2)).reshape(nc * npts, arr.shape[2])
            else:
                arr = arr.reshape(nc * arr.shape[1], arr.shape[2])
        elif arr.ndim == 2:
            if arr.shape[0] == nc and arr.shape[1] == npts:
                arr = arr.reshape(nc * npts)
            elif arr.shape[0] == npts and arr.shape[1] == nc:
                arr = arr.T.reshape(nc * npts)
            elif arr.shape[0] == nc * npts:
                pass
            else:
                raise ValueError(
                    f"Unsupported 2D sampled shape for field {name}: {arr.shape}, "
                    f"expected ({nc}, {npts}), ({npts}, {nc}), or ({nc * npts}, *)"
                )
        elif arr.ndim == 1:
            if arr.size != nc * npts:
                raise ValueError(
                    f"Unsupported 1D sampled shape for field {name}: {arr.shape}, "
                    f"expected total size {nc * npts}"
                )
        else:
            raise ValueError(f"Unsupported sampled shape for field {name}: {arr.shape}")
        if arr.shape[0] != nc * npts:
            raise ValueError(
                f"Field `{name}` first dimension mismatch: expected {nc * npts}, got {arr.shape[0]}"
            )
        out[name] = arr
    return out


def _parent_dir_makedirs(fname: str) -> None:
    d = os.path.dirname(fname)
    if d:
        os.makedirs(d, exist_ok=True)


def _write_unstructured_vtu_ascii(
    fname: str,
    points_xyz: np.ndarray,
    connectivity: np.ndarray,
    offsets: np.ndarray,
    types: np.ndarray,
    point_data: Dict[str, np.ndarray],
    *,
    higher_order_degrees: Optional[np.ndarray] = None,
) -> None:
    """Minimal ASCII VTU writer (single Piece, linear or high-order cells)."""
    _parent_dir_makedirs(fname)
    points_xyz = np.asarray(points_xyz, dtype=float)
    connectivity = np.asarray(connectivity, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    types = np.asarray(types, dtype=np.uint8)
    np_pts = int(points_xyz.shape[0])
    np_cells = int(connectivity.shape[0])

    pts_flat = points_xyz.reshape(-1)
    points_txt = " ".join(f"{float(v):.16e}" for v in pts_flat)
    conn_txt = " ".join(str(int(v)) for v in connectivity.reshape(-1))
    off_txt = " ".join(str(int(v)) for v in offsets)
    type_txt = " ".join(str(int(v)) for v in types)

    pieces: List[str] = []
    pieces.append('<?xml version="1.0"?>\n')
    pieces.append(
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n'
    )
    pieces.append("  <UnstructuredGrid>\n")
    pieces.append(f'    <Piece NumberOfPoints="{np_pts}" NumberOfCells="{np_cells}">\n')
    pieces.append("      <Points>\n")
    pieces.append(
        '        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n'
        f"          {points_txt}\n"
        "        </DataArray>\n"
    )
    pieces.append("      </Points>\n")
    pieces.append("      <Cells>\n")
    pieces.append(
        '        <DataArray type="Int64" Name="connectivity" format="ascii">\n'
        f"          {conn_txt}\n"
        "        </DataArray>\n"
    )
    pieces.append(
        '        <DataArray type="Int64" Name="offsets" format="ascii">\n'
        f"          {off_txt}\n"
        "        </DataArray>\n"
    )
    pieces.append(
        '        <DataArray type="UInt8" Name="types" format="ascii">\n'
        f"          {type_txt}\n"
        "        </DataArray>\n"
    )
    if higher_order_degrees is not None:
        hod = np.asarray(higher_order_degrees, dtype=np.int64)
        deg_txt = " ".join(str(int(v)) for v in hod)
        pieces.append(
            '        <DataArray type="Int64" Name="HigherOrderDegrees" format="ascii">\n'
            f"          {deg_txt}\n"
            "        </DataArray>\n"
        )
    pieces.append("      </Cells>\n")
    pieces.append("      <PointData>\n")
    for k, v in point_data.items():
        pieces.append(_format_data_array(k, np.asarray(v)))
    pieces.append("      </PointData>\n")
    pieces.append("    </Piece>\n")
    pieces.append("  </UnstructuredGrid>\n")
    pieces.append("</VTKFile>\n")
    with open(fname, "w", encoding="utf-8") as f:
        f.write("".join(pieces))


def write_subdivided_quadrilateral_mesh_vtu(
    *,
    fname: str,
    mesh,
    order: int,
    point_data: Dict[str, np.ndarray],
) -> None:
    """
    Write a **linear** quad VTU that refines each Q_p macro-cell into ``p*p``
    bilinear quads on the tensor grid of interpolation points (duplicated per cell).

    ``point_data`` values must have length ``NC * (p+1)^2``, in the same order as
    :func:`sample_fields_tensor_product_fe` with ``cell_dim=2``.
    """
    order = int(order)
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    cell = np.asarray(mesh.entity("cell"), dtype=np.int64)
    if cell.ndim != 2 or cell.shape[1] != 4:
        raise NotImplementedError("write_subdivided_quadrilateral_mesh_vtu expects 2D quad cells (4 vertices).")

    p = order
    nc = int(mesh.number_of_cells())
    npp = (p + 1) * (p + 1)
    gd = int(mesh.geo_dimension())

    ip = np.asarray(mesh.interpolation_points(p), dtype=float)
    c2d = np.asarray(mesh.cell_to_ipoint(p), dtype=np.int64)
    pts = ip[c2d].reshape(nc * npp, gd)
    pts3 = np.zeros((pts.shape[0], 3), dtype=float)
    pts3[:, : min(gd, 3)] = pts[:, : min(gd, 3)]

    for key, val in point_data.items():
        arr = np.asarray(val)
        if arr.shape[0] != nc * npp:
            raise ValueError(
                f"Point data `{key}` first dimension mismatch: expected {nc * npp}, got {arr.shape[0]}"
            )

    n_sub = nc * p * p
    conn = np.empty((n_sub, 4), dtype=np.int64)
    row = 0
    base = 0
    for _ic in range(nc):
        for i in range(p):
            for j in range(p):
                conn[row, 0] = base + i * (p + 1) + j
                conn[row, 1] = base + i * (p + 1) + (j + 1)
                conn[row, 2] = base + (i + 1) * (p + 1) + (j + 1)
                conn[row, 3] = base + (i + 1) * (p + 1) + j
                row += 1
        base += npp

    types = np.full((n_sub,), VTK_QUAD, dtype=np.uint8)
    offsets = np.arange(1, n_sub + 1, dtype=np.int64) * 4
    _write_unstructured_vtu_ascii(fname, pts3, conn, offsets, types, point_data)


def write_subdivided_hexahedron_mesh_vtu(
    *,
    fname: str,
    mesh,
    order: int,
    point_data: Dict[str, np.ndarray],
) -> None:
    """
    Write a **linear** hex VTU that refines each Q_p macro-cell into ``p^3``
    trilinear hexes on the tensor grid of interpolation points (duplicated per cell).

    ``point_data`` values must have length ``NC * (p+1)^3``, same order as
    :func:`sample_fields_tensor_product_fe` with ``cell_dim=3``.
    """
    order = int(order)
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    cell = np.asarray(mesh.entity("cell"), dtype=np.int64)
    if cell.ndim != 2 or cell.shape[1] != 8:
        raise NotImplementedError("write_subdivided_hexahedron_mesh_vtu expects 3D hex cells (8 vertices).")

    p = order
    nc = int(mesh.number_of_cells())
    npp = (p + 1) ** 3
    gd = int(mesh.geo_dimension())
    if gd != 3:
        raise NotImplementedError("write_subdivided_hexahedron_mesh_vtu currently supports GD==3 only.")

    ip = np.asarray(mesh.interpolation_points(p), dtype=float)
    c2d = np.asarray(mesh.cell_to_ipoint(p), dtype=np.int64)
    pts3 = ip[c2d].reshape(nc * npp, 3)

    for key, val in point_data.items():
        arr = np.asarray(val)
        if arr.shape[0] != nc * npp:
            raise ValueError(
                f"Point data `{key}` first dimension mismatch: expected {nc * npp}, got {arr.shape[0]}"
            )

    n_sub = nc * p * p * p
    conn = np.empty((n_sub, 8), dtype=np.int64)
    row = 0
    base = 0

    def _loc(ii: int, jj: int, kk: int) -> int:
        return ii * (p + 1) * (p + 1) + jj * (p + 1) + kk

    for ic in range(nc):
        for i in range(p):
            for j in range(p):
                for k in range(p):
                    conn[row, 0] = base + _loc(i, j, k)
                    conn[row, 1] = base + _loc(i + 1, j, k)
                    conn[row, 2] = base + _loc(i + 1, j + 1, k)
                    conn[row, 3] = base + _loc(i, j + 1, k)
                    conn[row, 4] = base + _loc(i, j, k + 1)
                    conn[row, 5] = base + _loc(i + 1, j, k + 1)
                    conn[row, 6] = base + _loc(i + 1, j + 1, k + 1)
                    conn[row, 7] = base + _loc(i, j + 1, k + 1)
                    row += 1
        base += npp

    types = np.full((n_sub,), VTK_HEXAHEDRON, dtype=np.uint8)
    offsets = np.arange(1, n_sub + 1, dtype=np.int64) * 8
    _write_unstructured_vtu_ascii(fname, pts3, conn, offsets, types, point_data)
