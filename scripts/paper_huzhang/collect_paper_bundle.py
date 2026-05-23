#!/usr/bin/env python3
"""Aggregate paper_results_huzhang into a single index for writing / plotting."""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _read_json(path: Path):
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tail_history(path: Path, n: int = 5) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-n:]


def collect(root: Path) -> dict:
    index: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "cases": {},
    }

    phasefield_root = root / "phasefield"
    scan_root = phasefield_root if phasefield_root.is_dir() else root

    for case_dir in sorted(p for p in scan_root.iterdir() if p.is_dir()):
        case_id = case_dir.name
        entry: dict = {"runs": {}}
        for run_dir in sorted(p for p in case_dir.iterdir() if p.is_dir()):
            run_label = run_dir.name
            manifest = _read_json(run_dir / "run_manifest.json") or {}
            eps_dirs = sorted(d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("epsg_"))
            run_data = None
            for eps_d in eps_dirs:
                summary = _read_json(eps_d / "summary.json")
                meta = _read_json(eps_d / "meta.json")
                run_data = {
                    "tag_dir": str(eps_d.relative_to(root)),
                    "summary": summary,
                    "meta": meta,
                    "history_tail": _tail_history(eps_d / "history.csv"),
                    "vtk_glob": str((eps_d / "vtk").relative_to(root)) + "/*.vtu",
                    "checkpoints": str((eps_d / "checkpoints").relative_to(root)),
                }
            run_entry = {
                "manifest": manifest,
                "run": run_data,
            }
            if run_label.startswith("paper_baseline") or run_label == "reference_direct_serial":
                ref = _read_json(run_dir / "baseline_reference.json")
                if ref:
                    run_entry["baseline_reference"] = ref
            entry["runs"][run_label] = run_entry
        index["cases"][case_id] = entry

    out = root / "PAPER_INDEX.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# Hu-Zhang phase-field paper results",
        "",
        f"- Root: `{root}`",
        f"- Generated: {index['generated_at']}",
        "",
        "| Case | Run | DOF (σ/u/d) | h_max | h<l₀/2 | Steps | Wall [s] |",
        "|------|-----|-------------|-------|--------|-------|----------|",
    ]
    for case_id, data in index["cases"].items():
        for run_label, mdata in data.get("runs", {}).items():
            man = mdata.get("manifest") or {}
            gd = (man.get("gdof") or {})
            mesh = man.get("mesh") or {}
            run = mdata.get("run") or {}
            summ = (run.get("summary") or {}) if run else {}
            md_lines.append(
                "| {case} | {mode} | {sg}/{gu}/{gd} | {hmax:.4e} | {hok} | {nstep} | {wall:.2f} |".format(
                    case=case_id,
                    mode=run_label,
                    sg=gd.get("sigma", "-"),
                    gu=gd.get("u", "-"),
                    gd=gd.get("d", "-"),
                    hmax=float(mesh.get("h_max", float("nan"))),
                    hok="yes" if mesh.get("h_ok") else "no",
                    nstep=summ.get("n_load_steps", man.get("n_load_steps", "-")),
                    wall=float(summ.get("total_wall_s", man.get("wall_s", float("nan")))),
                )
            )
    md_path = root / "PAPER_INDEX.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    index["index_md"] = str(md_path)
    print(f"Wrote {out}")
    print(f"Wrote {md_path}")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=os.environ.get("FRACTUREX_RESULTS_ROOT", "results"),
    )
    args = parser.parse_args()
    collect(Path(args.root).resolve())


if __name__ == "__main__":
    main()
