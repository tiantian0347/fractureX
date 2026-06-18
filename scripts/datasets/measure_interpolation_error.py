"""Measure 𝓘₁ vs 𝓘₂ interpolation error on real paper_aux runs.

Scans `paper_aux_h{1,2,3}` (gdof_sigma 10924 / 48092 / 183524). For each h:
  1) load_discr_from_dir → HuZhang discretization
  2) For each chosen time step t ∈ {t_a, t_b, t_c}:
       σ_truth(grid)  := HuZhang basis · σ_dofs evaluated directly on grid
                         (this is the FE solution itself; no reconstruction)
       σ_qp           := σ_dofs evaluated at this run's quadrature points
       𝓘₁(σ)(grid)    := nearest-quadrature-point sampling
       𝓘₂(σ)(grid)    := L²-project σ_qp to space_d, then sample on grid
  3) Report rel_L²(𝓘_*(σ), σ_truth) and rel_L^∞ per channel + per t,
     plus 𝓘₂/𝓘₁ ratio. Writes:
       - <out>/sigma_interp_error.csv        (long form)
       - <out>/sigma_interp_convergence.png  (rel L² vs h, log-log)
       - <out>/sigma_interp_summary.json     (geometric mean across channels/t)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fracturex.postprocess.dataset_export import (
    CircularNotchDomain,
    GridSpec,
    _build_pixel_locator,
    _evaluate_huzhang_on_grid,
    compute_valid_mask,
    load_discr_from_dir,
    sample_field_l2_projection,
    sample_field_nearest_quad,
)
from fracturex.learn.eval.metrics import relative_l2, relative_linf

CHANNEL_NAMES_HZ = ("sigma_xx", "sigma_xy", "sigma_yy")  # HuZhang Voigt order
SCHEMES = ("I1", "I2")
DEFAULT_HS = ("h1", "h2", "h3")
QUAD_ORDER = 5


@dataclass
class CaseSpec:
    name: str
    recorder_dir: Path
    h_label: str   # "h1" / "h2" / "h3"


def _checkpoint_files(recorder_dir: Path) -> list[Path]:
    return sorted((recorder_dir / "checkpoints").glob("step_*.npz"))


def _pick_time_steps(ckpts: Sequence[Path]) -> dict[str, Path]:
    """Choose t_a (early elastic), t_b (mid), t_c (late) checkpoints.

    step_000 is always the load=0 frame (σ_h ≡ 0); skipping it avoids a
    degenerate denominator in relative L². With ≥3 non-zero checkpoints we
    use first/middle/last of the non-zero subset; with 2 we pick t_a and t_c.
    """
    n = len(ckpts)
    if n == 0:
        return {}
    nonzero = list(ckpts[1:]) if n > 1 else list(ckpts)
    k = len(nonzero)
    if k == 0:
        return {"t_a": ckpts[0]}
    if k == 1:
        return {"t_a": nonzero[0]}
    if k == 2:
        return {"t_a": nonzero[0], "t_c": nonzero[1]}
    return {"t_a": nonzero[0], "t_b": nonzero[k // 2], "t_c": nonzero[-1]}


def _evaluate_huzhang_on_qp(space, sigma_dofs: np.ndarray) -> np.ndarray:
    """σ at every quadrature point: shape (NC, NQ, 3) in HuZhang order [xx, xy, yy]."""
    mesh = space.mesh
    qf = mesh.quadrature_formula(QUAD_ORDER)
    bcs, _ = qf.get_quadrature_points_and_weights()
    bcs = np.asarray(bcs)
    phi = np.asarray(space.basis(bcs))                       # (NC, NQ, ldof, 3)
    c2d = np.asarray(space.dof.cell_to_dof())                # (NC, ldof)
    dofs = np.asarray(sigma_dofs)
    cell_dofs = dofs[c2d]                                    # (NC, ldof)
    return np.einsum("cqld,cl->cqd", phi, cell_dofs)         # (NC, NQ, 3)


def _ground_truth_on_grid(
    space, sigma_dofs: np.ndarray, locator
) -> np.ndarray:
    """σ_h evaluated on grid via HuZhang basis directly (no reconstruction).

    This is what 𝓘_*(σ) is being compared against — the σ_h field that the
    surrogate would reconstruct ideally if interpolation were lossless.
    Returns (3, H, W) HuZhang-ordered.
    """
    return _evaluate_huzhang_on_grid(space, sigma_dofs, locator)


def _measure_one_step(
    discr,
    sigma_dofs: np.ndarray,
    grid: GridSpec,
    locator,
    mask_hw: np.ndarray,
) -> dict:
    """Returns rows of (channel, scheme, rel_L2, rel_Linf)."""
    qp_phys = np.asarray(discr.mesh.bc_to_point(
        discr.mesh.quadrature_formula(QUAD_ORDER).get_quadrature_points_and_weights()[0]
    ))                                                          # (NC, NQ, 2)
    sigma_qp = _evaluate_huzhang_on_qp(discr.space_sigma, sigma_dofs)   # (NC, NQ, 3)
    sigma_truth = _ground_truth_on_grid(discr.space_sigma, sigma_dofs, locator)  # (3, H, W)

    rows = []

    # 𝓘₁: nearest-quadrature scatter.  sample_field_nearest_quad returns
    # (3, H, W) for trailing dim 3.
    t0 = time.time()
    sigma_i1 = sample_field_nearest_quad(sigma_qp, qp_phys, grid, mask=mask_hw)
    t_i1 = time.time() - t0

    # 𝓘₂: L² projection per channel, reusing one mass solve internally.
    t0 = time.time()
    sigma_i2 = sample_field_l2_projection(
        sigma_qp, discr, grid, mask=mask_hw, quadrature_order=QUAD_ORDER
    )                                                                              # (3, H, W)
    t_i2 = time.time() - t0

    mask_bool = mask_hw.astype(bool)
    sigma_truth_masked = sigma_truth.copy()
    sigma_truth_masked[:, ~mask_bool] = 0.0

    for ci, ch in enumerate(CHANNEL_NAMES_HZ):
        rows.append({
            "scheme": "I1",
            "channel": ch,
            "rel_L2": relative_l2(sigma_i1[ci], sigma_truth_masked[ci], mask_bool),
            "rel_Linf": relative_linf(sigma_i1[ci], sigma_truth_masked[ci], mask_bool),
            "wall_s": t_i1,
        })
        rows.append({
            "scheme": "I2",
            "channel": ch,
            "rel_L2": relative_l2(sigma_i2[ci], sigma_truth_masked[ci], mask_bool),
            "rel_Linf": relative_linf(sigma_i2[ci], sigma_truth_masked[ci], mask_bool),
            "wall_s": t_i2,
        })
    return rows


def measure(cases: Sequence[CaseSpec], grid: GridSpec, geometry, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sigma_interp_error.csv"
    summary: dict = {}

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case", "h_label", "step_label", "step_file",
                "scheme", "channel", "rel_L2", "rel_Linf",
                "wall_s", "h_proxy",
            ],
        )
        writer.writeheader()

        for case in cases:
            print(f"== {case.name} {case.h_label} ==")
            discr = load_discr_from_dir(case.recorder_dir)
            NC = int(discr.mesh.number_of_cells())
            h_proxy = float(np.sqrt(2.0 / NC))   # rough average h on unit square
            print(f"   NC={NC}, h_proxy={h_proxy:.4e}, gdof_d={discr.space_d.number_of_global_dofs()}")
            locator = _build_pixel_locator(discr.mesh, grid)
            mask_hw = compute_valid_mask(grid, geometry)[0]    # (H, W) uint8

            ckpts = _checkpoint_files(case.recorder_dir)
            chosen = _pick_time_steps(ckpts)
            for tlabel, ckpt in chosen.items():
                z = np.load(ckpt, allow_pickle=False)
                sigma_dofs = np.asarray(z["sigma"])
                rows = _measure_one_step(discr, sigma_dofs, grid, locator, mask_hw)
                for r in rows:
                    writer.writerow({
                        "case": case.name,
                        "h_label": case.h_label,
                        "step_label": tlabel,
                        "step_file": ckpt.name,
                        "h_proxy": h_proxy,
                        **r,
                    })
                # tiny console dump for the t_c step (most informative)
                if tlabel == "t_c" or (tlabel == "t_a" and "t_c" not in chosen):
                    by_scheme = {s: [r for r in rows if r["scheme"] == s] for s in SCHEMES}
                    for s in SCHEMES:
                        gm = math.exp(np.mean(np.log([r["rel_L2"] for r in by_scheme[s]])))
                        print(f"   {tlabel} {s} geom-mean rel L² (3 ch) = {gm:.3e}")

    rows = list(csv.DictReader(csv_path.open()))
    for r in rows:
        for k in ("rel_L2", "rel_Linf", "wall_s", "h_proxy"):
            r[k] = float(r[k])
    summary = _build_summary(rows)
    (out_dir / "sigma_interp_summary.json").write_text(json.dumps(summary, indent=2))
    _plot_convergence(rows, out_dir / "sigma_interp_convergence.png")
    print(f"== wrote {csv_path}")
    print(f"== wrote {out_dir / 'sigma_interp_summary.json'}")
    print(f"== wrote {out_dir / 'sigma_interp_convergence.png'}")
    return summary


def _build_summary(rows: list[dict]) -> dict:
    out: dict = {"by_h": {}, "by_h_t_a_only": {}, "rates_t_a": {}, "rates_all": {}}
    eps = 1e-30

    def collect(filter_fn, dest_key: str) -> None:
        for h in DEFAULT_HS:
            block = {}
            for s in SCHEMES:
                sub = [r for r in rows if r["h_label"] == h and r["scheme"] == s and filter_fn(r)]
                if not sub:
                    continue
                l2_vals = [max(r["rel_L2"], eps) for r in sub]
                block[s] = {
                    "rel_L2_geom_mean": math.exp(np.mean(np.log(l2_vals))),
                    "rel_Linf_max": max(r["rel_Linf"] for r in sub),
                    "h_proxy": sub[0]["h_proxy"],
                    "n_rows": len(sub),
                }
            if block:
                out[dest_key][h] = block

    collect(lambda r: True, "by_h")
    collect(lambda r: r["step_label"] == "t_a", "by_h_t_a_only")

    for src_key, dst_key in (("by_h", "rates_all"), ("by_h_t_a_only", "rates_t_a")):
        for s in SCHEMES:
            rates = []
            prev = None
            for h in DEFAULT_HS:
                blk = out[src_key].get(h, {}).get(s)
                if blk is None:
                    continue
                if prev is not None:
                    ratio_h = prev["h_proxy"] / blk["h_proxy"]
                    e_prev = max(prev["rel_L2_geom_mean"], eps)
                    e_curr = max(blk["rel_L2_geom_mean"], eps)
                    ratio_e = e_prev / e_curr
                    rate = (math.log(ratio_e) / math.log(ratio_h)
                            if ratio_h > 1 and ratio_e > 0 else float("nan"))
                    rates.append({"from": prev["h_proxy"], "to": blk["h_proxy"], "rate": rate})
                prev = blk
            out[dst_key][s] = rates
    return out


def _plot_convergence(rows: list[dict], path: Path) -> None:
    """Convergence plot restricted to t_a (elastic) — only step shared across h's.

    h2/h3 paper_aux runs are short (only 2 checkpoints, both elastic-stage), so
    a fair h-rate can only be measured at the early step; t_b/t_c data on h2/h3
    is missing and would mix physics regimes if averaged in.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    h_vals_ref = None
    for s, color in zip(SCHEMES, ("tab:orange", "tab:blue")):
        h_vals, e_vals = [], []
        for h in DEFAULT_HS:
            sub = [r for r in rows if r["h_label"] == h and r["scheme"] == s
                   and r["step_label"] == "t_a"]
            if not sub:
                continue
            h_vals.append(sub[0]["h_proxy"])
            e_vals.append(math.exp(np.mean(np.log([max(r["rel_L2"], 1e-30) for r in sub]))))
        if h_vals:
            ax.loglog(h_vals, e_vals, "o-", color=color, label=f"{s} (rel L², t_a)")
            h_vals_ref = h_vals
            e0 = e_vals[0]
    if h_vals_ref:
        h_arr = np.array(h_vals_ref)
        ax.loglog(h_arr, e0 * (h_arr / h_arr[0]), "k--", lw=0.8, label="O(h) ref")
        ax.loglog(h_arr, e0 * (h_arr / h_arr[0]) ** 2, "k:", lw=0.8, label="O(h²) ref")
    ax.set_xlabel("h_proxy = √(2/NC)")
    ax.set_ylabel("rel L² (geom-mean over 3 σ channels at t_a)")
    ax.set_title("σ interpolation error: 𝓘₁ vs 𝓘₂ on paper_aux runs (t_a only)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

    # Companion plot: h1 only, σ channel-wise across t_a/t_b/t_c (shows the
    # "𝓘₂ penalised by sharp gradients" effect that is the headline finding).
    h1_path = path.with_name(path.stem + "_h1_time.png")
    _plot_h1_time_breakdown(rows, h1_path)


def _plot_h1_time_breakdown(rows: list[dict], path: Path) -> None:
    h1_rows = [r for r in rows if r["h_label"] == "h1"]
    if not h1_rows:
        return
    t_order = ["t_a", "t_b", "t_c"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, ch in zip(axes, CHANNEL_NAMES_HZ):
        for s, color in zip(SCHEMES, ("tab:orange", "tab:blue")):
            xs, ys = [], []
            for t in t_order:
                sub = [r for r in h1_rows if r["channel"] == ch
                       and r["scheme"] == s and r["step_label"] == t]
                if sub:
                    xs.append(t)
                    ys.append(sub[0]["rel_L2"])
            if xs:
                ax.plot(xs, ys, "o-", color=color, label=s)
        ax.set_title(ch)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("rel L² (h1)")
    axes[-1].legend()
    fig.suptitle("h1 only — interpolation error by σ channel × time step")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _default_cases(root: Path) -> list[CaseSpec]:
    """Resolve one CaseSpec per h-label, preferring the canonical paper_aux runs.

    For each `h` in DEFAULT_HS, look first at `paper_aux_<h>/epsg_1e-06`
    (the canonical paper experiment dirs being written by the long-running
    PID 2475* `run_case.py --case model0 --mode aux` jobs with
    FRACTUREX_RUN_LABEL_SUFFIX=h2/h3 as of 2026-05-29). If that is missing
    or has no mesh.npz yet, fall back to the `_dB` variant produced by
    out-of-band reruns (e.g. nice'd background patches). The non-`_dB`
    canonical path is preferred so this script auto-uses the user's main
    paper data without further intervention once those jobs land their
    full t_a/t_b/t_c checkpoints.
    """
    base = root / "results" / "phasefield" / "model0_circular_notch"
    out: list[CaseSpec] = []
    for h in DEFAULT_HS:
        candidates = [
            (f"paper_aux_{h}",    base / f"paper_aux_{h}"    / "epsg_1e-06"),
            (f"paper_aux_{h}_dB", base / f"paper_aux_{h}_dB" / "epsg_1e-06"),
        ]
        picked = None
        for name, rec_dir in candidates:
            if (rec_dir / "mesh.npz").exists() and (rec_dir / "checkpoints").exists():
                picked = (name, rec_dir)
                break
        if picked is None:
            print(f"[skip] {h}: no mesh.npz/checkpoints in any of {[c[0] for c in candidates]}")
            continue
        name, rec_dir = picked
        print(f"[case] {h} -> {name}")
        out.append(CaseSpec(name=name, recorder_dir=rec_dir, h_label=h))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--out-dir", type=Path,
                    default=Path("docs/figures/m0/interp_error"))
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    args = ap.parse_args()

    cases = _default_cases(args.root)
    if not cases:
        print("no usable paper_aux runs found")
        return 1
    grid = GridSpec(H=args.H, W=args.W, bbox=((0.0, 1.0), (0.0, 1.0)))
    geometry = CircularNotchDomain(box=(0.0, 1.0, 0.0, 1.0), cx=0.5, cy=0.5, r=0.2)
    summary = measure(cases, grid, geometry, args.out_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
