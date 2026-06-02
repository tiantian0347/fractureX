"""Phase-field mesh sizing: enforce h_max < safety * l0/2."""
from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import numpy as np
from fealpy.mesh import TriangleMesh


def mesh_h_stats(mesh) -> dict:
    """统计网格边长。输入 mesh，返回含 ``h_min``/``h_max``/``h_mean`` 的 dict。"""
    h = np.asarray(mesh.edge_length(), dtype=float)
    return {
        "h_min": float(h.min()),
        "h_max": float(h.max()),
        "h_mean": float(h.mean()),
    }


def phasefield_h_target(l0: float, *, safety: Optional[float] = None) -> float:
    """相场网格目标最大边长 ``safety · l0/2``。

    Args:
        l0: 相场长度尺度。
        safety: 安全系数；缺省读环境变量 ``FRACTUREX_H_SAFETY``（默认 0.92）。
    Returns:
        目标 ``h_max`` 上界。
    """
    if safety is None:
        safety = float(os.environ.get("FRACTUREX_H_SAFETY", "0.92"))
    return float(safety) * (float(l0) / 2.0)


def resolve_model0_distmesh_hmin(
    l0: float,
    make_mesh,
    *,
    hmin0: Optional[float] = None,
    hmin_min: float = 1e-4,
) -> Tuple[float, dict]:
    """
    Refine distmesh ``hmin`` until ``h_max < phasefield_h_target(l0)``.

    Parameters
    ----------
    make_mesh
        Callable ``(hmin: float) -> mesh``.
    """
    target = phasefield_h_target(l0)
    hmin = float(hmin0) if hmin0 is not None else target * 0.45
    last: dict = {}
    for _ in range(24):
        mesh = make_mesh(hmin)
        stats = mesh_h_stats(mesh)
        last = {
            "hmin": hmin,
            **stats,
            "h_target": target,
            "l0": float(l0),
            "h_ok": stats["h_max"] < target,
            "NC": int(mesh.number_of_cells()),
            "NN": int(mesh.number_of_nodes()),
        }
        if stats["h_max"] < target:
            return hmin, last
        hmin *= 0.72
        if hmin < hmin_min:
            break
    raise RuntimeError(
        f"model0 distmesh: could not satisfy h_max < {target:.6e}; last={last}"
    )


def resolve_box_nx(
    l0: float,
    *,
    nx0: int = 32,
    nx_max: int = 1024,
) -> Tuple[int, dict]:
    """Refine uniform box ``nx`` until ``h_max < phasefield_h_target(l0)``."""
    target = phasefield_h_target(l0)
    nx = max(4, int(nx0))
    last: dict = {}
    while nx <= nx_max:
        mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=nx, ny=nx)
        stats = mesh_h_stats(mesh)
        last = {"nx": nx, "ny": nx, **stats, "h_target": target, "h_ok": stats["h_max"] < target}
        if stats["h_max"] < target:
            return nx, last
        scale = target / max(stats["h_max"], 1e-30)
        nx = max(int(np.ceil(nx / scale * 1.05)), nx + 4)
    raise RuntimeError(
        f"box mesh: could not satisfy h_max < {target:.6e} with nx <= {nx_max}; last={last}"
    )
