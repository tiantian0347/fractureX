# fracturex/postprocess/dataset_export/__init__.py
"""Encode a RunRecorder output dir into training-ready npz samples.

External protocol: docs/operator_learning/SURROGATE_DATA_SCHEMA.md.
Math: docs/operator_learning/plan_operator_learning.md §3.3 (E_h^in, E_h^out, masks).

Architecture (the portability seam):
  - grid / geometry / sampling / meta : **model-agnostic** building blocks;
  - adapter.SolverAdapter             : the interface a model must implement;
  - core                              : the generic export engine;
  - adapters/huzhang_phasefield       : the reference (Hu-Zhang) adapter.

Porting to another physical model means writing one adapter under
``adapters/`` — the core and the ``fracturex/learn/`` training side stay put.

This package keeps the historic flat import surface
(``from fracturex.postprocess.dataset_export import GridSpec, ...``) intact;
``export_recorder_to_sample`` here preserves the legacy positional signature.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .adapter import FieldSpec, SolverAdapter
from .adapters.huzhang_phasefield import (
    HuZhangPhaseFieldAdapter,
    _evaluate_huzhang_on_grid,
    evaluate_huzhang_on_grid,
    load_discr_from_dir,
)
from .core import (
    ExportConfig,
    encode_inputs,
    encode_outputs,
)
from .core import export_recorder_to_sample as _core_export_recorder_to_sample
from .geometry import (
    CircularNotchDomain,
    Geometry,
    GeometryLike,
    signed_distance,
    _signed_distance,
)
from .grid import (
    GridSpec,
    compute_coords,
    compute_sdf,
    compute_valid_mask,
    grid_points,
    _grid_points,
)
from .meta import SCHEMA_VERSION, build_sample_meta, git_commit_short
from .sampling import (
    PixelLocator,
    _PixelLocator,
    _build_pixel_locator,
    _evaluate_lagrange_on_grid,
    _group_pixels_by_cell,
    build_pixel_locator,
    evaluate_lagrange_on_grid,
    group_pixels_by_cell,
    sample_field_l2_projection,
    sample_field_nearest_quad,
)


def export_recorder_to_sample(
    recorder_dir: Path,
    out_npz: Path,
    out_meta: Path,
    cfg: ExportConfig,
    discr,
    geometry: GeometryLike,
    *,
    adapter: Optional[SolverAdapter] = None,
    sample_id: Optional[str] = None,
    git_commit: Optional[str] = None,
    config_hash: Optional[str] = None,
    load_history: Optional[np.ndarray] = None,
    material_overrides: Optional[dict] = None,
    extra_meta: Optional[dict] = None,
) -> dict:
    """Legacy-signature wrapper around :func:`core.export_recorder_to_sample`.

    Keeps the historic positional order ``(..., cfg, discr, geometry)`` and a
    pre-built ``discr``. New code may call the core directly with
    ``(..., cfg, geometry, adapter=..., discr=...)``.
    """
    return _core_export_recorder_to_sample(
        recorder_dir, out_npz, out_meta, cfg, geometry,
        adapter=adapter, discr=discr,
        sample_id=sample_id, git_commit=git_commit, config_hash=config_hash,
        load_history=load_history, material_overrides=material_overrides,
        extra_meta=extra_meta,
    )


__all__ = [
    "SCHEMA_VERSION",
    # grid
    "GridSpec", "grid_points", "compute_sdf", "compute_valid_mask", "compute_coords",
    # geometry
    "Geometry", "GeometryLike", "CircularNotchDomain", "signed_distance",
    # sampling
    "PixelLocator", "build_pixel_locator", "group_pixels_by_cell",
    "evaluate_lagrange_on_grid", "sample_field_nearest_quad", "sample_field_l2_projection",
    # adapter interface + reference adapter
    "FieldSpec", "SolverAdapter", "HuZhangPhaseFieldAdapter",
    "load_discr_from_dir", "evaluate_huzhang_on_grid",
    # core
    "ExportConfig", "encode_inputs", "encode_outputs", "export_recorder_to_sample",
    "build_sample_meta", "git_commit_short",
]
