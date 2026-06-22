"""Generate a Model-0 phase-field dataset from a parameter-grid config.

Reads a JSON (or YAML if PyYAML is available) describing a parameter grid,
runs ``run_model0_one`` for each combination, then calls
``export_recorder_to_sample`` to write a schema v0.1 sample npz + meta.json
under ``<dataset_dir>/samples/``. A ``dataset_manifest.json`` indexes the
result.

Config format:
    {
      "dataset_name": "m0_smoke",
      "n_steps_override": 2,           # optional cap on load count for speed
      "fixed": {
        "hmin": 0.05,
        "p_sigma": 3,
        "damage_p": 2,
        "circle_cx": 0.5,
        "circle_cy": 0.5,
        "elastic_mode": "direct"
      },
      "grid": {                        # cartesian product
        "circle_r": [0.15, 0.20, 0.25],
        "Gc": [0.5, 1.0, 2.0],
        "l0": [0.015, 0.02, 0.025]
      },
      "export": {
        "H": 64, "W": 64, "bbox": [[0,1],[0,1]]
      }
    }

Run:
    PYTHONPATH=$PWD $FEALPY_PYTHON \\
      scripts/datasets/generate_phasefield_dataset.py \\
      --config scripts/datasets/configs/m0_smoke.json \\
      --dataset-dir results/datasets/m0_smoke
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fracturex.postprocess.dataset_export import (
    CircularNotchDomain,
    ExportConfig,
    GridSpec,
    export_recorder_to_sample,
    load_discr_from_dir,
)
from fracturex.tests.case_runners.model0_runner import (
    Model0RunArgs,
    run_model0_one,
)


def _load_config(path: Path) -> dict:
    text = path.read_text()
    if path.suffix in (".yml", ".yaml"):
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as e:
            raise SystemExit(
                "PyYAML not installed; convert to JSON or `pip install pyyaml`"
            ) from e
        return yaml.safe_load(text)
    return json.loads(text)


def _enumerate_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    combos: list[dict] = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        combos.append(dict(zip(keys, vals)))
    return combos


def _build_run_args(
    fixed: dict, combo: dict, n_steps_override: int | None, outdir: Path
) -> Model0RunArgs:
    args = Model0RunArgs(outdir=outdir)
    for k, v in fixed.items():
        if not hasattr(args, k):
            raise KeyError(f"unknown Model0RunArgs field {k!r} in fixed:")
        setattr(args, k, v)
    for k, v in combo.items():
        if not hasattr(args, k):
            raise KeyError(f"unknown Model0RunArgs field {k!r} in grid:")
        setattr(args, k, v)
    if n_steps_override is not None:
        loads = args.loads or list(np.linspace(0.0, 0.05, n_steps_override))
        args.loads = list(loads[: int(n_steps_override)])
    return args


def _build_export_cfg(export_cfg: dict) -> tuple[ExportConfig, CircularNotchDomain]:
    bbox = export_cfg.get("bbox", [[0.0, 1.0], [0.0, 1.0]])
    grid = GridSpec(
        H=int(export_cfg["H"]),
        W=int(export_cfg["W"]),
        bbox=((bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1])),
    )
    return ExportConfig(grid=grid), bbox


def _make_geometry(combo: dict, fixed: dict, bbox) -> CircularNotchDomain:
    cx = combo.get("circle_cx", fixed.get("circle_cx", 0.5))
    cy = combo.get("circle_cy", fixed.get("circle_cy", 0.5))
    r = combo.get("circle_r", fixed.get("circle_r", 0.2))
    return CircularNotchDomain(
        box=(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]),
        cx=float(cx), cy=float(cy), r=float(r),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap how many grid combos to run (debug)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip samples whose npz already exists")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the grid across N parallel processes")
    ap.add_argument("--shard", type=int, default=0,
                    help="this process's shard index in [0, num-shards)")
    ap.add_argument("--cleanup-runs", action="store_true",
                    help="delete each run dir after its sample is exported "
                         "(bounds disk to samples/; the npz is self-contained)")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    dataset_name = cfg.get("dataset_name", args.dataset_dir.name)
    fixed = cfg.get("fixed", {})
    grid = cfg.get("grid", {})
    n_steps_override = cfg.get("n_steps_override")
    export_dict = cfg["export"]
    cfg_export, bbox = _build_export_cfg(export_dict)

    combos = _enumerate_grid(grid)
    if args.max_samples is not None:
        combos = combos[: args.max_samples]
    # Shard by GLOBAL index so sample ids never collide across parallel shards
    # writing into the same dataset dir.
    indexed = list(enumerate(combos))
    if args.num_shards > 1:
        indexed = [(i, c) for (i, c) in indexed if i % args.num_shards == args.shard]
    print(f"dataset={dataset_name}  combos={len(combos)}  "
          f"shard={args.shard}/{args.num_shards}  this_shard={len(indexed)}")

    samples_dir = args.dataset_dir / "samples"
    runs_dir = args.dataset_dir / "runs"
    samples_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    manifest_samples: list[dict] = []
    failures: list[dict] = []
    t_start = time.perf_counter()

    for i, combo in indexed:
        sample_id = f"sample_{i:06d}"
        npz_path = samples_dir / f"{sample_id}.npz"
        meta_path = samples_dir / f"{sample_id}.meta.json"
        run_dir = runs_dir / sample_id

        if args.skip_existing and npz_path.exists() and meta_path.exists():
            print(f"[{i+1}/{len(combos)}] {sample_id}  SKIP (exists)")
            with meta_path.open() as f:
                manifest_samples.append({
                    "id": sample_id,
                    "npz": f"samples/{sample_id}.npz",
                    "meta": f"samples/{sample_id}.meta.json",
                    "ok": True,
                    "params": combo,
                })
            continue

        t0 = time.perf_counter()
        try:
            run_args = _build_run_args(fixed, combo, n_steps_override, run_dir)
            run_model0_one(run_args)
            discr = load_discr_from_dir(run_dir)
            geom = _make_geometry(combo, fixed, bbox)
            sample_meta = export_recorder_to_sample(
                run_dir, npz_path, meta_path, cfg_export, discr, geom,
                sample_id=sample_id,
                extra_meta={"grid_params": combo},
            )
            wall = time.perf_counter() - t0
            manifest_samples.append({
                "id": sample_id,
                "npz": f"samples/{sample_id}.npz",
                "meta": f"samples/{sample_id}.meta.json",
                "ok": True,
                "params": combo,
                "wall_s": round(wall, 2),
                "max_damage": sample_meta["stats"]["max_damage"],
            })
            print(f"[{i+1}/{len(combos)}] {sample_id}  OK  ({wall:.1f}s)  "
                  f"max_d={sample_meta['stats']['max_damage']:.3f}")
            if args.cleanup_runs:
                shutil.rmtree(run_dir, ignore_errors=True)
        except Exception as e:
            tb = traceback.format_exc(limit=3)
            print(f"[{i+1}/{len(combos)}] {sample_id}  FAIL: {e}")
            failures.append({"id": sample_id, "params": combo, "error": str(e), "trace": tb})
            manifest_samples.append({
                "id": sample_id, "params": combo, "ok": False, "error": str(e),
            })

    manifest = {
        "schema_version": "0.1",
        "dataset_name": dataset_name,
        "config_path": str(args.config),
        "fixed": fixed,
        "grid": grid,
        "n_samples": len(indexed),
        "n_ok": sum(1 for s in manifest_samples if s.get("ok")),
        "n_fail": len(failures),
        "samples": manifest_samples,
        "wall_s_total": round(time.perf_counter() - t_start, 2),
    }
    if args.num_shards > 1:
        manifest["shard"] = args.shard
        manifest["num_shards"] = args.num_shards
    manifest_name = ("dataset_manifest.json" if args.num_shards == 1
                     else f"dataset_manifest.shard{args.shard}.json")
    (args.dataset_dir / manifest_name).write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {args.dataset_dir/manifest_name}")
    print(f"ok={manifest['n_ok']}  fail={manifest['n_fail']}  "
          f"total wall = {manifest['wall_s_total']}s")
    return 0 if manifest["n_fail"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
