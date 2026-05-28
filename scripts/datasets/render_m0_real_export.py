"""Render real (damage, stress) on the schema grid from the paper_aux_h1 npz.

Loads ``results/operator_learning_smoke/sample_paper_aux_h1.npz`` (produced
by render_m0_real_export_npz.py / load_discr_from_dir + export_recorder_to_sample)
and plots last-frame damage + 3 stress channels masked by `valid_mask`.

Run:
    PYTHONPATH=$PWD $FEALPY_PYTHON \\
      scripts/datasets/render_m0_real_export.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NPZ = Path("results/operator_learning_smoke/sample_paper_aux_h1.npz")
OUT = Path("docs/figures/m0/fig_real_export_last_frame.png")


def main() -> None:
    if not NPZ.exists():
        raise FileNotFoundError(
            f"missing {NPZ}; run render_m0_real_export_npz.py first."
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    z = np.load(NPZ)

    damage = z["damage"]      # (T, 1, H, W)
    stress = z["stress"]      # (T, 3, H, W) -- (xx, yy, xy)
    mask = z["mask"][0].astype(bool)
    T = damage.shape[0]

    # Pick last-frame view; mask out-of-Ω with NaN so the colormap shows white.
    d_last = damage[-1, 0].astype(np.float32)
    s_last = stress[-1].astype(np.float32)
    d_last_masked = np.where(mask, d_last, np.nan)
    s_last_masked = np.where(mask[None], s_last, np.nan)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))

    im = axes[0].imshow(
        d_last_masked, origin="lower", extent=[0, 1, 0, 1],
        cmap="hot_r", vmin=0, vmax=1,
    )
    axes[0].contour(
        np.linspace(0, 1, mask.shape[1]), np.linspace(0, 1, mask.shape[0]),
        d_last, levels=[0.5], colors="cyan", linewidths=1.5,
    )
    theta = np.linspace(0, 2 * np.pi, 200)
    for ax in axes:
        ax.plot(0.5 + 0.2 * np.cos(theta), 0.5 + 0.2 * np.sin(theta),
                color="black", linewidth=0.8)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_title(f"damage  (T={T-1})  max={float(d_last[mask].max()):.3f}")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    titles = ["σ_xx (norm.)", "σ_yy (norm.)", "σ_xy (norm.)"]
    for k, ax in enumerate(axes[1:]):
        ch = s_last_masked[k]
        v = float(np.nanpercentile(np.abs(ch), 99))
        v = max(v, 1e-6)
        im = ax.imshow(
            ch, origin="lower", extent=[0, 1, 0, 1],
            cmap="RdBu_r", vmin=-v, vmax=v,
        )
        ax.set_title(titles[k])
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        "Fig 4. Real model0 export on H=W=128: last-frame damage + stress "
        "channels (norm.). Source: paper_aux_h1, checkpoint step 30.",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
