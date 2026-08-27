#!/usr/bin/env python3
"""CLI wrapper for fracturex.postprocess.vtu_animation.

Same flags as ``python -m fracturex.postprocess.vtu_animation``.

Examples:
  python scripts/paper_huzhang/plot_vtu_phasefield_gif.py \\
      --vtu-dir results/.../vtu --out docs/benchmarks/figures/phasefield/run.gif

  python scripts/paper_huzhang/plot_vtu_phasefield_gif.py \\
      --vtu-dir results/... --glob 'model5_std*.vtu' --out tpb.gif --stride 2

  python scripts/paper_huzhang/plot_vtu_phasefield_gif.py \\
      --vtu-dir results/.../vtu --out run.gif --mesh-every 10
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fracturex.postprocess.vtu_animation import main

if __name__ == "__main__":
    raise SystemExit(main())
