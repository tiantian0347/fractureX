from __future__ import annotations

import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


# VTK enum for high-order triangle cell.
VTK_LAGRANGE_TRIANGLE = 69


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
    os.makedirs(os.path.dirname(fname), exist_ok=True)

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
    NC = int(mesh.number_of_cells())
    out: Dict[str, np.ndarray] = {}
    for name, fun in field_specs:
        raw = fun(bcs, index=np.arange(NC, dtype=np.int64))
        arr = np.asarray(raw)
        if arr.ndim == 3:
            arr = arr.reshape(NC * arr.shape[1], arr.shape[2])
        elif arr.ndim == 2:
            arr = arr.reshape(NC * arr.shape[1])
        else:
            raise ValueError(f"Unsupported sampled shape for field {name}: {arr.shape}")
        out[name] = arr
    return out
