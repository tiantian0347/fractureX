"""Render an 8-panel grid of last-frame damage from the m0_smoke_2x2x2 dataset.

Output: docs/figures/m0/fig_dataset_smoke_grid.png

Run:
    PYTHONPATH=$PWD $FEALPY_PYTHON \\
      scripts/datasets/render_m0_dataset_smoke_grid.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASET = Path("results/datasets/m0_smoke_2x2x2")
OUT = Path("docs/figures/m0/fig_dataset_smoke_grid.png")


def main() -> None:
    manifest = json.loads((DATASET / "dataset_manifest.json").read_text())
    samples = manifest["samples"]
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for ax, s in zip(axes.flatten(), samples):
        z = np.load(DATASET / s["npz"])
        d = z["damage"][-1, 0]
        mask = z["mask"][0].astype(bool)
        d_masked = np.where(mask, d, np.nan)
        im = ax.imshow(d_masked, origin="lower", extent=[0, 1, 0, 1],
                       cmap="hot_r", vmin=0, vmax=1)
        # Draw the notch.
        p = s["params"]
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(0.5 + p["circle_r"] * np.cos(theta),
                0.5 + p["circle_r"] * np.sin(theta),
                color="black", linewidth=0.8)
        ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(
            f"{s['id']}\n"
            f"r={p['circle_r']}  Gc={p['Gc']}  l0={p['l0']}\n"
            f"max d = {s.get('max_damage', float('nan')):.3f}",
            fontsize=9,
        )
    fig.suptitle(
        "m0_smoke_2x2x2  (2 load steps, HW=64²) — last-frame damage on the 2×2×2 grid",
        y=1.0,
    )
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02,
                 shrink=0.85, label="damage  d")
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
