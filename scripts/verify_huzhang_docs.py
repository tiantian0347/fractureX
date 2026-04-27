#!/usr/bin/env python3
"""
Verify that Hu–Zhang + phase-field architecture docs point to existing files.

Run from the repository root:
  python scripts/verify_huzhang_docs.py

When you add, rename, or move modules referenced in the docs, update:
  - docs/HUZHANG_PHASEFIELD_ARCHITECTURE.md
  - docs/HUZHANG_PHASEFIELD_ARCHITECTURE.en.md
  - the REQUIRED list below
Exit code 0 if all paths exist, 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Critical modules and docs for the architecture narrative (keep in sync with both .md files).
REQUIRED: list[str] = [
    "docs/HUZHANG_PHASEFIELD_ARCHITECTURE.md",
    "docs/HUZHANG_PHASEFIELD_ARCHITECTURE.en.md",
    "docs/README.md",
    "fracturex/cases/base.py",
    "fracturex/discretization/huzhang_discretization.py",
    "fracturex/damage/phasefield_damage.py",
    "fracturex/assemblers/huzhang_elastic_assembler.py",
    "fracturex/assemblers/phasefield_assembler.py",
    "fracturex/boundarycondition/huzhang_boundary_condition.py",
    "fracturex/drivers/huzhang_phasefield_staggered.py",
    "fracturex/utilfuc/linear_solvers.py",
    "fracturex/postprocess/recorder.py",
    "fracturex/postprocess/reaction.py",
    "fracturex/postprocess/run_report.py",
    "fracturex/tests/phasefield_model0_huzhang.py",
    "fracturex/phasefield/main_solve.py",
    "scripts/verify_huzhang_docs.py",
]


def main() -> int:
    missing: list[str] = []
    for rel in REQUIRED:
        path = ROOT / rel
        if not path.is_file():
            missing.append(rel)
    if missing:
        print("verify_huzhang_docs: missing files (update docs and REQUIRED list):", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 1
    print(f"verify_huzhang_docs: OK ({len(REQUIRED)} paths).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
