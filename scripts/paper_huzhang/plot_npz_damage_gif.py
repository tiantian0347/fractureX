#!/usr/bin/env python3
"""CLI wrapper for fracturex.postprocess.npz_animation.

Example (model5 final restart, α-ramp of d):
  python scripts/paper_huzhang/plot_npz_damage_gif.py \\
      --npz results/.../model5_std_state.npz \\
      --case model5 --mesh-size 0.015 \\
      --ramp-frames 32 --out docs/benchmarks/figures/phasefield/model5_d.gif
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fracturex.postprocess.npz_animation import main

if __name__ == "__main__":
    raise SystemExit(main())
