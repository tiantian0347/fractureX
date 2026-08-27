#!/usr/bin/env python3
"""CLI wrapper for fracturex.postprocess.vtu_plot.

Same flags as ``python -m fracturex.postprocess.vtu_plot``.

Example:
  python scripts/paper_huzhang/plot_vtu_mesh_damage.py \\
    --vtu path/to/step_032.vtu --out docs/adaptive/figures/final.png
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fracturex.postprocess.vtu_plot import main

if __name__ == "__main__":
    raise SystemExit(main())
