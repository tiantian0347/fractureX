"""Generate the three figures embedded in
docs/m0_kickoff_report_2026-05-28.md §3.3.

Run from repo root with the project python:

    FEALPY_PYTHON=/path/to/python PYTHONPATH=$PWD \
        $FEALPY_PYTHON scripts/datasets/render_m0_kickoff_figures.py

Output: docs/figures/m0/{fig_geometry,fig_evaluator_d_error,fig_evaluator_sigma_zero}.png

The script is self-contained — it builds a tiny HuZhang discretization and
exercises the dataset_export evaluators on synthetic DOFs. No real recorder
data is read.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh

from fracturex.cases.base import CaseBase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.postprocess.dataset_export import (
    CircularNotchDomain,
    GridSpec,
    _build_pixel_locator,
    _evaluate_huzhang_on_grid,
    _evaluate_lagrange_on_grid,
    compute_coords,
    compute_sdf,
    compute_valid_mask,
)


OUT = "docs/figures/m0"


def _imshow(ax, data, title, cmap="viridis", vmin=None, vmax=None):
    im = ax.imshow(
        data, origin="lower", extent=[0, 1, 0, 1], cmap=cmap, vmin=vmin, vmax=vmax
    )
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)


def fig_geometry(out_path: str) -> None:
    grid = GridSpec(H=64, W=64, bbox=((0.0, 1.0), (0.0, 1.0)))
    geom = CircularNotchDomain(box=(0, 1, 0, 1), cx=0.5, cy=0.5, r=0.2)
    sdf = compute_sdf(grid, geom)[0]
    mask = compute_valid_mask(grid, geom)[0]
    coords = compute_coords(grid)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    _imshow(axes[0], sdf, "sdf  (positive inside Ω)", cmap="RdBu_r")
    _imshow(axes[1], mask, "mask  (1 = inside Ω)", cmap="gray", vmin=0, vmax=1)
    _imshow(axes[2], coords[0], "coords[0]  (norm. x)")
    _imshow(axes[3], coords[1], "coords[1]  (norm. y)")
    fig.suptitle(
        "Fig 1. Input encoding on H=W=64 grid; CircularNotchDomain(r=0.2)", y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


class _DummyBoxCase(CaseBase):
    name = "dummy_box"

    def make_mesh(self, **kw):
        return TriangleMesh.from_box([0, 1, 0, 1], nx=8, ny=8)

    def isD_bd(self, points):
        return np.zeros(points.shape[0], dtype=bool)

    def model(self):
        raise NotImplementedError


def _build_box_discr():
    return HuZhangDiscretization(_DummyBoxCase(), p=3, damage_p=1).build()


def fig_evaluator_d(out_path: str) -> None:
    discr = _build_box_discr()
    W = discr.space_d
    node = np.asarray(discr.mesh.entity("node"))
    d = np.asarray(discr.state.d)
    d[:] = node[:, 0]

    grid = GridSpec(H=64, W=64, bbox=((0.0, 1.0), (0.0, 1.0)))
    loc = _build_pixel_locator(discr.mesh, grid)
    d_grid = _evaluate_lagrange_on_grid(W, d, loc)[0]

    xs = np.linspace(0, 1, 64)
    ys = np.linspace(0, 1, 64)
    X, _ = np.meshgrid(xs, ys, indexing="xy")
    err = d_grid - X

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    _imshow(axes[0], d_grid, "evaluator output  d_grid")
    _imshow(axes[1], X, "analytic  d(x,y) = x")
    _imshow(
        axes[2],
        err,
        f"error  max|.| = {np.max(np.abs(err)):.2e}",
        cmap="RdBu_r",
    )
    fig.suptitle(
        "Fig 2. Lagrange P1 evaluator: d nodal=x, analytic d=x", y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_evaluator_sigma_zero(out_path: str) -> None:
    discr = _build_box_discr()
    sigma = np.asarray(discr.state.sigma)
    sigma[:] = 0.0

    grid = GridSpec(H=64, W=64, bbox=((0.0, 1.0), (0.0, 1.0)))
    loc = _build_pixel_locator(discr.mesh, grid)
    field_hz = _evaluate_huzhang_on_grid(discr.space_sigma, sigma, loc)

    titles = ["σ_xx (HuZhang)", "σ_xy (HuZhang)", "σ_yy (HuZhang)"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for k, ax in enumerate(axes):
        _imshow(
            ax,
            field_hz[k],
            f"{titles[k]}  max|.|={np.max(np.abs(field_hz[k])):.1e}",
            cmap="RdBu_r",
            vmin=-1e-12,
            vmax=1e-12,
        )
    fig.suptitle("Fig 3. HuZhang evaluator on σ DOFs ≡ 0", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    fig_geometry(os.path.join(OUT, "fig_geometry.png"))
    fig_evaluator_d(os.path.join(OUT, "fig_evaluator_d_error.png"))
    fig_evaluator_sigma_zero(os.path.join(OUT, "fig_evaluator_sigma_zero.png"))
    print(f"saved 3 figures into {OUT}/")


if __name__ == "__main__":
    main()
