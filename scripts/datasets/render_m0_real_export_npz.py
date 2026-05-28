"""Run dataset_export.export_recorder_to_sample on paper_aux_h1 → npz.

Prereq: ``mesh.npz`` exists in the recorder dir. For legacy runs run
``recover_mesh_from_vtu.py`` first.

Run:
    PYTHONPATH=$PWD $FEALPY_PYTHON \\
      scripts/datasets/render_m0_real_export_npz.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from fracturex.postprocess.dataset_export import (
    CircularNotchDomain,
    ExportConfig,
    GridSpec,
    export_recorder_to_sample,
    load_discr_from_dir,
)


REC = Path(
    "results/phasefield/model0_circular_notch/paper_aux_h1/epsg_1e-06"
)
OUT = Path("results/operator_learning_smoke")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    discr = load_discr_from_dir(REC)
    print(
        f"rebuilt discr  gdof_sigma={discr.gdof_sigma}  "
        f"NN={discr.mesh.number_of_nodes()}  NC={discr.mesh.number_of_cells()}"
    )
    cfg = ExportConfig(grid=GridSpec(H=128, W=128, bbox=((0.0, 1.0), (0.0, 1.0))))
    geom = CircularNotchDomain(box=(0, 1, 0, 1), cx=0.5, cy=0.5, r=0.2)

    t0 = time.perf_counter()
    meta = export_recorder_to_sample(
        REC,
        OUT / "sample_paper_aux_h1.npz",
        OUT / "sample_paper_aux_h1.meta.json",
        cfg,
        discr,
        geom,
    )
    print(f"export wall = {time.perf_counter() - t0:.2f}s")

    z = np.load(OUT / "sample_paper_aux_h1.npz")
    print(f"  damage {z['damage'].shape}  max d = {z['damage'].max():.4f}")
    print(
        f"  stress {z['stress'].shape}  range = "
        f"[{z['stress'].min():.3e}, {z['stress'].max():.3e}]"
    )
    print(f"  mask = {int(z['mask'].sum())}/{z['mask'].size}")
    print(f"  meta.scaling.stress_scale = {meta['scaling']['stress_scale']:.3e}")


if __name__ == "__main__":
    main()
