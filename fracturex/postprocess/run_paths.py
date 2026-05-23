"""Unified output directories for Hu-Zhang phase-field runs and tests."""
from __future__ import annotations

import os
from typing import Optional


def results_root() -> str:
    """Top-level results directory (override with ``FRACTUREX_RESULTS_ROOT``)."""
    return os.environ.get("FRACTUREX_RESULTS_ROOT", "results")


def epsg_tag(eps_g: float) -> str:
    return f"epsg_{float(eps_g):.0e}"


def phasefield_tag_dir(
    case_name: str,
    *run_path: str,
    eps_g: float = 1e-6,
    root: Optional[str] = None,
    mkdir: bool = False,
) -> str:
    """
    Standard run directory for CSV/NPZ/VTK:

        ``<root>/phasefield/<case_name>/<run_path...>/<epsg_tag>/``
    """
    path = os.path.join(
        root or results_root(),
        "phasefield",
        case_name,
        *run_path,
        epsg_tag(eps_g),
    )
    if mkdir:
        os.makedirs(path, exist_ok=True)
        os.makedirs(vtk_dir(path), exist_ok=True)
    return path


def vtk_dir(tag_dir: str) -> str:
    return os.path.join(tag_dir, "vtk")


def vtk_step_path(tag_dir: str, step: int) -> str:
    return os.path.join(vtk_dir(tag_dir), f"step_{int(step):03d}.vtu")


def resolve_run_output_dir(
    *,
    output_dir: Optional[str] = None,
    recorder: Optional[object] = None,
) -> Optional[str]:
    """Prefer explicit ``output_dir``, else ``recorder.outdir``."""
    if output_dir:
        return str(output_dir)
    if recorder is not None:
        out = getattr(recorder, "outdir", None)
        if out:
            return str(out)
    return None


def resolve_vtk_step_path(
    *,
    step: int,
    output_dir: Optional[str] = None,
    recorder: Optional[object] = None,
) -> Optional[str]:
    base = resolve_run_output_dir(output_dir=output_dir, recorder=recorder)
    if not base:
        return None
    return vtk_step_path(base, step)
