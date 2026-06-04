"""§4.2 Measure 𝓘₁ vs 𝓘₂ interpolation error on the H quadrature-history field.

Background (m0_interpolation_error.md §4.2): unlike σ, the history field 𝓗
lives only on the quadrature points — there is no FE basis representation of
𝓗 itself. So the σ measurement template ("truth on grid = basis × dofs")
does not apply directly. Approach used here:

    truth (on qp)        := H_qp (the run's own value)
    𝓘₁(H) on grid        := nearest-quadrature-point scatter (existing helper)
    𝓘₂(H) on grid        := L²-project H_qp onto space_d, evaluate on grid

  Roundtrip-error metric:

      H̃_qp_*  := bilinearly resample 𝓘_*(H) at xq                  (qp side)
      e_L²    := ‖m_qp ⊙ (H̃_qp_* − H_qp)‖₂ / ‖m_qp ⊙ H_qp‖₂
      e_L∞    := max_qp |H̃_qp_* − H_qp| / max_qp |H_qp|

  m_qp marks qp points that fell inside Ω on the structured grid (a qp very
  close to the notch hole may map to a masked-out pixel). Reported as
  ``frac_qp_in_grid``.

  Auxiliary diagnostics:
      max_grid    : max of 𝓘_* (H) over Ω-pixels;
      max_qp_truth: max of H_qp;
      max_ratio   : max_grid / max_qp_truth (𝓘₁ ≈ 1 by construction, 𝓘₂ ≤ 1
                    when the L²-projection smooths out spikes).

Outputs:
    <out>/h_interp_error.csv          long form
    <out>/h_interp_summary.json       per-step / per-scheme aggregate
    <out>/h_interp_compare.png        side-by-side H field on grid for both
                                      schemes at the chosen step
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fracturex.postprocess.dataset_export import (
    CircularNotchDomain,
    GridSpec,
    compute_valid_mask,
    load_discr_from_dir,
    sample_field_l2_projection,
    sample_field_nearest_quad,
)
from fracturex.learn.eval.metrics import relative_l2, relative_linf

SCHEMES = ("I1", "I2", "const")


@dataclass
class HQpStep:
    label: str
    step: int
    qp_path: Path


def _to_hw(field: np.ndarray) -> np.ndarray:
    """Coerce sampler output to (H, W).

    ``sample_field_nearest_quad`` returns (H, W) for scalar trailing dim,
    while ``sample_field_l2_projection`` returns (H, W) for scalar inputs;
    both helpers also accept multi-channel input and add a leading channel
    axis. Defensive squeeze keeps the call site shape-agnostic.
    """
    f = np.asarray(field)
    if f.ndim == 3 and f.shape[0] == 1:
        f = f[0]
    return f


def _bilinear_resample(grid_field: np.ndarray, xq: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Resample a structured-grid field (H, W) at qp coords ``xq``.

    Uses bilinear interpolation in pixel space; out-of-bbox qp gets 0.
    Matches the "qp → grid → qp" roundtrip definition in §4.2.

    Args:
        grid_field: (H, W) float (after :func:`_to_hw`).
        xq:         (NC, NQ, 2) physical coords.
        grid:       GridSpec (defines bbox + grid pixel centers).

    Returns:
        (NC, NQ) float64.
    """
    f = _to_hw(grid_field)
    H, W = f.shape
    (x0, x1), (y0, y1) = grid.bbox
    # Same convention as _grid_points: pixel-center sampling.
    xs = np.linspace(x0, x1, W)
    ys = np.linspace(y0, y1, H)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    px = (xq[..., 0] - x0) / dx
    py = (xq[..., 1] - y0) / dy
    NC, NQ = xq.shape[:2]

    out = np.zeros((NC, NQ), dtype=np.float64)
    in_box = (
        (px >= 0.0) & (px <= W - 1) & (py >= 0.0) & (py <= H - 1)
    )
    if not in_box.any():
        return out
    px_c = np.clip(px, 0.0, W - 1 - 1e-9)
    py_c = np.clip(py, 0.0, H - 1 - 1e-9)
    i0 = np.floor(py_c).astype(np.int64)
    j0 = np.floor(px_c).astype(np.int64)
    fy = py_c - i0
    fx = px_c - j0
    i1 = np.minimum(i0 + 1, H - 1)
    j1 = np.minimum(j0 + 1, W - 1)
    v = (
        f[i0, j0] * (1 - fy) * (1 - fx)
        + f[i0, j1] * (1 - fy) * fx
        + f[i1, j0] * fy * (1 - fx)
        + f[i1, j1] * fy * fx
    )
    out[in_box] = v[in_box]
    return out


def _qp_inside_mask(xq: np.ndarray, mask_hw: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Mark qp points whose nearest pixel-center falls inside the Ω-mask.

    A qp very close to the notch hole may map to a masked-out pixel. Such qp
    cannot be reconstructed via either 𝓘 scheme; we exclude them from the
    error norms but report the count as ``frac_qp_in_grid`` for sanity.
    """
    (x0, x1), (y0, y1) = grid.bbox
    H, W = grid.H, grid.W
    xs = np.linspace(x0, x1, W)
    ys = np.linspace(y0, y1, H)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    j = np.clip(np.round((xq[..., 0] - x0) / dx).astype(np.int64), 0, W - 1)
    i = np.clip(np.round((xq[..., 1] - y0) / dy).astype(np.int64), 0, H - 1)
    return mask_hw[i, j].astype(bool)


def _load_qp_steps(recorder_dir: Path) -> list[HQpStep]:
    files = sorted((recorder_dir / "checkpoints").glob("step_*_qp.npz"))
    steps: list[HQpStep] = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        H_qp = np.asarray(z["H_qp"])
        if not np.any(H_qp > 0.0):
            continue  # skip step_000 / pre-driving frames
        steps.append(HQpStep(label=f.stem, step=int(z["step"]), qp_path=f))
    return steps


def _pick_three(steps: list[HQpStep]) -> dict[str, HQpStep]:
    if not steps:
        return {}
    if len(steps) == 1:
        return {"t_a": steps[0]}
    if len(steps) == 2:
        return {"t_a": steps[0], "t_c": steps[-1]}
    return {"t_a": steps[0], "t_b": steps[len(steps) // 2], "t_c": steps[-1]}


def _measure_one(
    discr,
    H_qp: np.ndarray,
    xq: np.ndarray,
    grid: GridSpec,
    mask_hw: np.ndarray,
) -> dict:
    """Returns {scheme: {rel_L2, rel_Linf, max_grid, ...}}.

    Three rows are produced per call:

    * ``I1``  — nearest-quadrature scatter then bilinear roundtrip;
    * ``I2``  — L²-project to space_d then bilinear roundtrip;
    * ``const`` — predict every qp with ``mean(H_qp)`` over Ω.
                  Acts as a "zero-information" baseline. If ``I1.rel_L2``
                  is close to ``const.rel_L2``, the residual is bounded
                  by grid representational capacity (the cusp does not
                  fit on the grid) rather than a poor 𝓘 choice.
    """
    qp_in = _qp_inside_mask(xq, mask_hw.astype(bool), grid)
    out: dict = {"frac_qp_in_grid": float(qp_in.mean())}

    truth_qp = H_qp.copy()
    truth_qp_masked = np.where(qp_in, truth_qp, 0.0)

    # 𝓘₁
    H_grid_i1 = sample_field_nearest_quad(H_qp, xq, grid, mask=mask_hw)  # (1, H, W)
    H_qp_back_i1 = _bilinear_resample(H_grid_i1, xq, grid)
    H_qp_back_i1_masked = np.where(qp_in, H_qp_back_i1, 0.0)

    out["I1"] = {
        "rel_L2": float(relative_l2(H_qp_back_i1_masked, truth_qp_masked, qp_in)),
        "rel_Linf": float(relative_linf(H_qp_back_i1_masked, truth_qp_masked, qp_in)),
        "max_grid": float(np.max(H_grid_i1)),
        "min_grid": float(np.min(H_grid_i1)),
    }

    # 𝓘₂
    H_grid_i2 = sample_field_l2_projection(
        H_qp, discr, grid, mask=mask_hw, quadrature_order=5
    )  # (1, H, W)
    H_qp_back_i2 = _bilinear_resample(H_grid_i2, xq, grid)
    H_qp_back_i2_masked = np.where(qp_in, H_qp_back_i2, 0.0)
    out["I2"] = {
        "rel_L2": float(relative_l2(H_qp_back_i2_masked, truth_qp_masked, qp_in)),
        "rel_Linf": float(relative_linf(H_qp_back_i2_masked, truth_qp_masked, qp_in)),
        "max_grid": float(np.max(H_grid_i2)),
        "min_grid": float(np.min(H_grid_i2)),
    }

    out["max_qp_truth"] = float(np.max(truth_qp))
    out["min_qp_truth"] = float(np.min(truth_qp))
    out["I1"]["max_ratio"] = out["I1"]["max_grid"] / max(out["max_qp_truth"], 1e-30)
    out["I2"]["max_ratio"] = out["I2"]["max_grid"] / max(out["max_qp_truth"], 1e-30)

    # const baseline: replace every qp value with the inside-Ω mean.
    if qp_in.any():
        mean_in = float(truth_qp[qp_in].mean())
    else:
        mean_in = 0.0
    pred_const = np.where(qp_in, mean_in, 0.0)
    out["const"] = {
        "rel_L2": float(relative_l2(pred_const, truth_qp_masked, qp_in)),
        "rel_Linf": float(relative_linf(pred_const, truth_qp_masked, qp_in)),
        "max_grid": mean_in,
        "min_grid": mean_in,
        "max_ratio": mean_in / max(out["max_qp_truth"], 1e-30),
    }

    out["_grids"] = {"I1": _to_hw(H_grid_i1), "I2": _to_hw(H_grid_i2)}
    return out


PLOT_SCHEMES = ("I1", "I2")  # const is a scalar baseline; not plotted.


def _plot_compare(
    out_path: Path,
    grid_pairs: dict[str, dict[str, np.ndarray]],
    geometry: CircularNotchDomain,
) -> None:
    """grid_pairs: {step_label: {'I1': (H,W), 'I2': (H,W)}} (post-_to_hw)."""
    steps = list(grid_pairs.keys())
    n = len(steps)
    fig, axes = plt.subplots(n, 2, figsize=(7.5, 3.6 * n), squeeze=False)
    for r, t in enumerate(steps):
        for c, scheme in enumerate(PLOT_SCHEMES):
            ax = axes[r, c]
            field = grid_pairs[t][scheme]
            im = ax.imshow(
                field,
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap="magma",
            )
            theta = np.linspace(0, 2 * np.pi, 200)
            ax.plot(
                geometry.cx + geometry.r * np.cos(theta),
                geometry.cy + geometry.r * np.sin(theta),
                "w-",
                lw=0.8,
            )
            ax.set_title(f"{t}, {scheme}: max={field.max():.2e}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("𝓗 field on grid — 𝓘₁ vs 𝓘₂ (h_qp_patch_h1)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def measure(recorder_dir: Path, grid: GridSpec, geometry, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    discr = load_discr_from_dir(recorder_dir)
    print(f"[h-interp] NC={discr.mesh.number_of_cells()} damage_p={discr.damage_p}")
    mask_hw = compute_valid_mask(grid, geometry)[0]

    steps_all = _load_qp_steps(recorder_dir)
    chosen = _pick_three(steps_all)
    if not chosen:
        raise SystemExit("no usable H_qp steps in {recorder_dir}")
    print(f"[h-interp] chosen: {sorted(chosen)} from {[s.label for s in steps_all]}")

    csv_path = out_dir / "h_interp_error.csv"
    summary: dict = {"by_step": {}}
    grid_for_plot: dict[str, dict[str, np.ndarray]] = {}

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step_label",
                "step_file",
                "scheme",
                "rel_L2",
                "rel_Linf",
                "max_grid",
                "max_ratio",
                "max_qp_truth",
                "frac_qp_in_grid",
            ],
        )
        writer.writeheader()

        for tlabel, hstep in chosen.items():
            z = np.load(hstep.qp_path, allow_pickle=False)
            H_qp = np.asarray(z["H_qp"])
            xq = np.asarray(z["xq"])
            res = _measure_one(discr, H_qp, xq, grid, mask_hw)
            grid_for_plot[tlabel] = res.pop("_grids")
            block: dict = {"frac_qp_in_grid": res["frac_qp_in_grid"]}
            for s in SCHEMES:
                rec = res[s]
                writer.writerow({
                    "step_label": tlabel,
                    "step_file": hstep.qp_path.name,
                    "scheme": s,
                    "rel_L2": rec["rel_L2"],
                    "rel_Linf": rec["rel_Linf"],
                    "max_grid": rec["max_grid"],
                    "max_ratio": rec["max_ratio"],
                    "max_qp_truth": res["max_qp_truth"],
                    "frac_qp_in_grid": res["frac_qp_in_grid"],
                })
                block[s] = {
                    "rel_L2": rec["rel_L2"],
                    "rel_Linf": rec["rel_Linf"],
                    "max_ratio": rec["max_ratio"],
                }
                print(
                    f"   {tlabel} {s}: rel_L2={rec['rel_L2']:.3e} "
                    f"rel_Linf={rec['rel_Linf']:.3e} "
                    f"max_ratio={rec['max_ratio']:.3f}"
                )
            block["max_qp_truth"] = res["max_qp_truth"]
            summary["by_step"][tlabel] = block

    (out_dir / "h_interp_summary.json").write_text(json.dumps(summary, indent=2))
    _plot_compare(out_dir / "h_interp_compare.png", grid_for_plot, geometry)
    print(f"== wrote {csv_path}")
    print(f"== wrote {out_dir / 'h_interp_summary.json'}")
    print(f"== wrote {out_dir / 'h_interp_compare.png'}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--recorder-dir",
        type=Path,
        default=Path(
            "/home/gongshihua/tian/fracturex/results/operator_learning_runs/h_qp_patch_h1"
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/gongshihua/tian/fracturex/docs/figures/m0/interp_error"),
    )
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    args = ap.parse_args()

    grid = GridSpec(H=args.H, W=args.W, bbox=((0.0, 1.0), (0.0, 1.0)))
    geometry = CircularNotchDomain(box=(0.0, 1.0, 0.0, 1.0), cx=0.5, cy=0.5, r=0.2)
    measure(args.recorder_dir, grid, geometry, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
