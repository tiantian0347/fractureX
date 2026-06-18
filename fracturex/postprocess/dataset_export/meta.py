# fracturex/postprocess/dataset_export/meta.py
"""Build the ``<sample>.meta.json`` payload (schema §4).

Model-agnostic: material ordering and the geometry descriptor are pulled from
the active :class:`SolverAdapter`, so this writer never hard-codes a model's
parameters.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

SCHEMA_VERSION = "0.1"


def git_commit_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# Backwards-compatible private alias.
_git_commit_short = git_commit_short


def read_recorder_meta(recorder_dir: Path) -> dict:
    meta_path = Path(recorder_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing meta.json under {recorder_dir}")
    with meta_path.open("r") as f:
        return json.load(f)


_read_recorder_meta = read_recorder_meta


def build_sample_meta(
    *,
    recorder_dir: Path,
    cfg,
    discr,
    adapter,
    sample_id: str,
    git_commit: str,
    config_hash: Optional[str],
    stress_scale: float,
    damage: np.ndarray,
    load_history: np.ndarray,
    valid_mask: np.ndarray,
    material_overrides: Optional[dict],
    extra_meta: Optional[dict],
) -> dict:
    rec_meta = read_recorder_meta(recorder_dir)
    material_meta = dict(rec_meta.get("material") or {})
    if material_overrides:
        material_meta.update(material_overrides)
    mesh = adapter.mesh(discr)

    # Mesh stats; h_min/h_max from edge lengths if available.
    try:
        node = np.asarray(mesh.entity("node"))
        cell = np.asarray(mesh.entity("cell"))
        edge_vec = node[cell[:, [1, 2, 0]]] - node[cell[:, [0, 1, 2]]]
        edge_len = np.linalg.norm(edge_vec, axis=-1)
        h_min = float(edge_len.min())
        h_max = float(edge_len.max())
    except Exception:
        h_min = float("nan")
        h_max = float("nan")

    n_steps = int(damage.shape[0])
    n_converged = int(load_history.shape[0])  # checkpoint count as proxy

    meta = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "geometry_params": adapter.geometry_meta(recorder_dir, rec_meta, cfg),
        "material_params": {
            k: float(material_meta.get(k))
            for k in material_meta
            if isinstance(material_meta.get(k), (int, float))
        },
        "material_order": list(adapter.material_order),
        "formulation": cfg.formulation,
        "interpolation": cfg.interpolation,
        "mesh_info": {
            "NC": int(mesh.number_of_cells()),
            "NN": int(mesh.number_of_nodes()),
            "h_min": h_min,
            "h_max": h_max,
            "p_sigma": int(getattr(discr, "p", -1)),
            "p_d": int(getattr(discr, "damage_p", -1)),
            "p_u": int(getattr(discr, "u_space_order", -1)),
        },
        "grid": {
            "H": int(cfg.grid.H),
            "W": int(cfg.grid.W),
            "domain_bbox": [list(cfg.grid.bbox[0]), list(cfg.grid.bbox[1])],
        },
        "load": {
            "kind": rec_meta.get("load_kind", "monotone"),
            "u_max": float(np.max(np.abs(load_history))) if load_history.size else 0.0,
            "N_steps": n_steps,
            "load_surfaces": rec_meta.get("load_surfaces", []),
        },
        "scaling": {
            "stress_scale": stress_scale,
            "u_scale": float(cfg.u_scale) if cfg.u_scale is not None else 1.0,
            "length_scale": 1.0,
            "time_scale": 1.0,
        },
        "boundary_codes": cfg.boundary_codes or {},
        "solver_config": rec_meta.get("solver", {}),
        "git_commit": git_commit,
        "config_hash": config_hash or "unknown",
        "run_paths": {"recorder_dir": str(recorder_dir)},
        "stats": {
            "max_damage": float(damage.max()) if damage.size else 0.0,
            "n_inside_pixels": int(valid_mask.sum()),
            "n_valid_steps": int(n_converged),
            "converged_step_ratio": 1.0 if n_steps == 0 else float(n_converged) / n_steps,
        },
    }
    if extra_meta:
        meta.update(extra_meta)
    return meta


_build_sample_meta = build_sample_meta
