"""Merge per-shard manifests into one ``dataset_manifest.json``.

Parallel ``generate_phasefield_dataset.py --num-shards N --shard k`` runs all
write into the same dataset dir (sample ids are global, so no collision) but
each emits its own ``dataset_manifest.shard{k}.json``. This concatenates them
into the unified manifest the training side expects.

Run:
    python scripts/datasets/merge_shard_manifests.py --dataset-dir <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-dir", required=True, type=Path)
    args = ap.parse_args()

    shard_files = sorted(args.dataset_dir.glob("dataset_manifest.shard*.json"))
    if not shard_files:
        raise SystemExit(f"no dataset_manifest.shard*.json under {args.dataset_dir}")

    samples: list[dict] = []
    base: dict | None = None
    for sf in shard_files:
        m = json.loads(sf.read_text())
        base = base or m
        samples.extend(m.get("samples", []))

    # Deduplicate by id (last wins), then sort by id for stable ordering.
    by_id = {s["id"]: s for s in samples}
    merged_samples = [by_id[k] for k in sorted(by_id)]
    n_ok = sum(1 for s in merged_samples if s.get("ok"))

    manifest = {
        "schema_version": base.get("schema_version", "0.1"),
        "dataset_name": base.get("dataset_name", args.dataset_dir.name),
        "config_path": base.get("config_path"),
        "fixed": base.get("fixed", {}),
        "grid": base.get("grid", {}),
        "n_samples": len(merged_samples),
        "n_ok": n_ok,
        "n_fail": len(merged_samples) - n_ok,
        "n_shards": len(shard_files),
        "samples": merged_samples,
    }
    out = args.dataset_dir / "dataset_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"merged {len(shard_files)} shards -> {out}  "
          f"(n={len(merged_samples)}, ok={n_ok}, fail={manifest['n_fail']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
