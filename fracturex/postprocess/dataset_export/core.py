# fracturex/postprocess/dataset_export/core.py
"""Model-agnostic export engine.

Orchestrates: geometric inputs (grid.py) + material/outputs via a
:class:`SolverAdapter` → masked, normalized, schema-packed npz + meta.json.
No model-specific code lives here; swap the adapter to retarget the pipeline.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .adapter import SolverAdapter
from .adapters.huzhang_phasefield import HuZhangPhaseFieldAdapter
from .geometry import GeometryLike
from .grid import GridSpec, compute_coords, compute_sdf, compute_valid_mask
from .meta import (
    SCHEMA_VERSION,
    build_sample_meta,
    git_commit_short,
    read_recorder_meta,
)
from .sampling import PixelLocator, build_pixel_locator


@dataclass(frozen=True)
class ExportConfig:
    """Knobs that change schema content; written to metadata.json."""

    grid: GridSpec
    formulation: str = "standard"               # 'standard' | 'effective_stress'
    interpolation: str = "I2_L2_projection"     # 'I1_nearest_quad' | 'I2_L2_projection'
    include_history: bool = False               # plan §3.5 (b) variant
    include_material_field: bool = False        # heterogeneous material
    stress_scale: Optional[float] = None        # None → dataset-level autoscale later
    u_scale: Optional[float] = None
    boundary_codes: Optional[dict[int, str]] = None


def _default_adapter() -> SolverAdapter:
    return HuZhangPhaseFieldAdapter()


# ---------------------------------------------------------------------------
# recorder readers
# ---------------------------------------------------------------------------

def _read_history_csv_loads(recorder_dir: Path) -> np.ndarray:
    """Pull the per-step prescribed load from history.csv (the ``load`` column)."""
    csv_path = Path(recorder_dir) / "history.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing history.csv under {recorder_dir}")
    loads: list[float] = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        if "load" not in reader.fieldnames:
            raise ValueError(
                f"history.csv at {csv_path} has no 'load' column; "
                f"available columns: {reader.fieldnames}"
            )
        for row in reader:
            loads.append(float(row["load"]))
    return np.asarray(loads, dtype=np.float32)


def _read_history_csv_iter_status(
    recorder_dir: Path, T: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read ``iters`` and ``converged`` columns from history.csv (if present)."""
    csv_path = Path(recorder_dir) / "history.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    iters: list[int] = []
    converged: list[int] = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        if "iters" not in reader.fieldnames or "converged" not in reader.fieldnames:
            raise KeyError("history.csv missing iters / converged column")
        for row in reader:
            iters.append(int(row["iters"]))
            cv = str(row["converged"]).strip().lower()
            converged.append(1 if cv in ("1", "true", "yes") else 0)
    if len(iters) != T:
        # Recorder may have written more rows than checkpoints when
        # save_every>1; truncate / pad.
        iters = iters[:T] + [0] * max(0, T - len(iters))
        converged = converged[:T] + [1] * max(0, T - len(converged))
    return (
        np.asarray(iters, dtype=np.int32),
        np.asarray(converged, dtype=np.uint8),
    )


def _normalized_time(T: int) -> np.ndarray:
    if T <= 0:
        return np.zeros((0,), dtype=np.float32)
    if T == 1:
        return np.zeros((1,), dtype=np.float32)
    return np.linspace(0.0, 1.0, T, dtype=np.float32)


# ---------------------------------------------------------------------------
# Inputs encoding  (E_h^in)
# ---------------------------------------------------------------------------

def encode_inputs(
    recorder_dir: Path,
    cfg: ExportConfig,
    geometry: GeometryLike,
    adapter: Optional[SolverAdapter] = None,
    *,
    load_history: Optional[np.ndarray] = None,
    material_overrides: Optional[dict] = None,
) -> dict[str, np.ndarray]:
    """Build the input portion of one sample's npz payload (schema §3.1).

    Reads ``recorder_dir/meta.json`` for material parameters (decoded by the
    adapter) and the load history from ``history.csv`` (when not supplied).
    Computes ``sdf``/``mask``/``coords`` from ``geometry`` on ``cfg.grid``.
    """
    recorder_dir = Path(recorder_dir)
    adapter = adapter or _default_adapter()
    meta = read_recorder_meta(recorder_dir)
    material_vec = adapter.material_vector(meta, material_overrides)

    if load_history is None:
        load_history = _read_history_csv_loads(recorder_dir)
    load_history = np.asarray(load_history, dtype=np.float32)
    if load_history.ndim == 1:
        load_history = load_history[:, None]
    elif load_history.ndim != 2:
        raise ValueError(
            f"load_history must be 1D or 2D, got shape {load_history.shape}"
        )

    return {
        "sdf": compute_sdf(cfg.grid, geometry),
        "mask": compute_valid_mask(cfg.grid, geometry),
        "coords": compute_coords(cfg.grid),
        "material": material_vec.astype(np.float32),
        "load_history": load_history.astype(np.float32),
        "time": _normalized_time(load_history.shape[0]),
    }


# ---------------------------------------------------------------------------
# Outputs encoding  (E_h^out)
# ---------------------------------------------------------------------------

def encode_outputs(
    recorder_dir: Path,
    cfg: ExportConfig,
    discr,
    adapter: Optional[SolverAdapter] = None,
    *,
    geometry: Optional[GeometryLike] = None,
    locator: Optional[PixelLocator] = None,
    stress_scale: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """Build the output portion of one sample's npz payload (schema §3.2).

    Iterates the adapter's checkpoints, asks the adapter to evaluate each
    declared output field on the grid, masks pixels outside Ω to zero, packs
    ``(T, C, H, W)`` tensors, and applies the per-field scaling policy.

    Fields whose :class:`FieldSpec` declares ``scaling="stress_scale"`` are
    divided by ``stress_scale`` (cfg.stress_scale / arg, else auto from the
    last frame's 95th percentile of the first such field over Ω).
    """
    recorder_dir = Path(recorder_dir)
    adapter = adapter or _default_adapter()
    specs = adapter.output_field_specs
    ckpts = adapter.list_checkpoints(recorder_dir)

    if locator is None:
        locator = build_pixel_locator(adapter.mesh(discr), cfg.grid)

    # In-Ω mask: prefer geometry-based valid_mask; fall back to locator.cell_id.
    if geometry is not None:
        mask_hw = compute_valid_mask(cfg.grid, geometry)[0].astype(bool)  # (H, W)
    else:
        mask_hw = (locator.cell_id.reshape(cfg.grid.H, cfg.grid.W) >= 0)

    T = len(ckpts)
    H, W = cfg.grid.H, cfg.grid.W
    fields: dict[str, np.ndarray] = {
        spec.name: np.zeros((T, spec.channels, H, W), dtype=np.float32) for spec in specs
    }

    step_iters = np.zeros(T, dtype=np.int32)
    step_converged = np.ones(T, dtype=np.uint8)
    try:
        step_iters, step_converged = _read_history_csv_iter_status(recorder_dir, T)
    except (FileNotFoundError, KeyError, ValueError):
        pass

    has_reaction = hasattr(adapter, "reaction")
    reaction_rows: list[np.ndarray] = []
    for t, ckpt in enumerate(ckpts):
        z = np.load(ckpt, allow_pickle=False)
        frame = adapter.evaluate_outputs(discr, z, locator, cfg.grid)
        for spec in specs:
            arr = np.asarray(frame[spec.name], dtype=np.float32)
            arr[:, ~mask_hw] = 0.0  # outside-Ω zero (schema §3.4)
            fields[spec.name][t] = arr
        if has_reaction:
            reaction_rows.append(np.asarray(adapter.reaction(discr, z), dtype=np.float32))

    # Resolve stress scale from the first stress-scaled field (if any).
    scaled_specs = [s for s in specs if s.scaling == "stress_scale"]
    if stress_scale is None:
        stress_scale = cfg.stress_scale
    if stress_scale is None and scaled_specs:
        ref = fields[scaled_specs[0].name][-1]  # last frame, (C, H, W)
        magnitude = np.percentile(np.abs(ref[:, mask_hw]), 95.0) if mask_hw.any() else 0.0
        stress_scale = float(magnitude) if magnitude > 0 else 1.0
    stress_scale = float(stress_scale) if stress_scale is not None else 1.0

    for spec in specs:
        if spec.scaling == "stress_scale":
            fields[spec.name] = (fields[spec.name] / stress_scale).astype(np.float32)

    out: dict[str, np.ndarray] = dict(fields)
    out["step_iters"] = step_iters
    out["step_converged"] = step_converged
    out["_stress_scale"] = np.float32(stress_scale)  # consumed by export_*; popped.
    if reaction_rows:
        out["reaction"] = np.stack(reaction_rows, axis=0).astype(np.float32)  # (T, r)
    return out


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def export_recorder_to_sample(
    recorder_dir: Path,
    out_npz: Path,
    out_meta: Path,
    cfg: ExportConfig,
    geometry: GeometryLike,
    *,
    adapter: Optional[SolverAdapter] = None,
    discr=None,
    sample_id: Optional[str] = None,
    git_commit: Optional[str] = None,
    config_hash: Optional[str] = None,
    load_history: Optional[np.ndarray] = None,
    material_overrides: Optional[dict] = None,
    extra_meta: Optional[dict] = None,
) -> dict:
    """Encode one recorder run into <sample>.npz + <sample>.meta.json.

    Writes per SURROGATE_DATA_SCHEMA.md §2-§4. Side effect: out_npz and
    out_meta are written atomically (write to .tmp then rename) so partial
    failures don't leave half-written samples.

    ``adapter`` defaults to :class:`HuZhangPhaseFieldAdapter`. ``discr`` may be
    passed directly (legacy call sites); otherwise it is rebuilt via
    ``adapter.load_discretization(recorder_dir)``.

    Returns the metadata dict that was written (for caller-side bookkeeping).
    """
    recorder_dir = Path(recorder_dir)
    out_npz = Path(out_npz)
    out_meta = Path(out_meta)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    adapter = adapter or _default_adapter()
    if discr is None:
        discr = adapter.load_discretization(recorder_dir)

    locator = build_pixel_locator(adapter.mesh(discr), cfg.grid)

    inputs = encode_inputs(
        recorder_dir, cfg, geometry, adapter,
        load_history=load_history, material_overrides=material_overrides,
    )
    outputs = encode_outputs(
        recorder_dir, cfg, discr, adapter,
        geometry=geometry, locator=locator, stress_scale=cfg.stress_scale,
    )

    stress_scale = float(outputs.pop("_stress_scale"))
    valid_mask = inputs["mask"].copy()

    payload = {
        "sdf": inputs["sdf"],
        "mask": inputs["mask"],
        "coords": inputs["coords"],
        "material": inputs["material"],
        "load_history": inputs["load_history"],
        "time": inputs["time"],
        "valid_mask": valid_mask,
        "step_iters": outputs["step_iters"],
        "step_converged": outputs["step_converged"],
    }
    for spec in adapter.output_field_specs:
        payload[spec.name] = outputs[spec.name]
    if "reaction" in outputs:
        payload["reaction"] = outputs["reaction"]  # (T, r) physical force, schema §3.2

    npz_tmp = out_npz.with_suffix(out_npz.suffix + ".tmp")
    # np.savez_compressed appends '.npz' to a path that doesn't end in '.npz';
    # pass an open handle to bypass that and keep the atomic-rename contract.
    with npz_tmp.open("wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(npz_tmp, out_npz)

    # `damage` is the conventional first output; fall back to first spec.
    damage_key = "damage" if "damage" in payload else adapter.output_field_specs[0].name
    meta = build_sample_meta(
        recorder_dir=recorder_dir,
        cfg=cfg,
        discr=discr,
        adapter=adapter,
        sample_id=sample_id or out_npz.stem,
        git_commit=git_commit or git_commit_short(),
        config_hash=config_hash,
        stress_scale=stress_scale,
        damage=outputs[damage_key],
        load_history=inputs["load_history"],
        valid_mask=valid_mask,
        material_overrides=material_overrides,
        extra_meta=extra_meta,
    )
    meta_tmp = out_meta.with_suffix(out_meta.suffix + ".tmp")
    with meta_tmp.open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    os.replace(meta_tmp, out_meta)

    return meta
