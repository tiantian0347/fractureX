#!/usr/bin/env python3
"""NEPIN spike: quantify hat_omega contraction at a paper_aux localization state.

Loads a phase-field checkpoint at ``max d ~ d_c + delta`` (just above the
localization hard-wall) and runs *one* NEPIN local elimination on
``Omega_s = {d > d_c}``. Reports:

  - subset size |S|
  - local Newton iters + residual reduction
  - the outer-Newton damage step ||dd|| before vs. after NEPIN
  - hat_omega surrogate = 2 * ||dd_after|| / ||dd_before||^2 (Deuflhard's
    a-posteriori estimator applied at a single point; a proper
    hat_omega_k would need two consecutive outer iters, but the
    single-point ratio is the cheapest witness of contraction).

Usage
-----
    PYTHONPATH=/Users/tian00/repository/fealpy:. python scripts/paper_precond/nepin_spike.py \\
        --checkpoint-dir results/phasefield/model0_circular_notch/paper_aux_h2/epsg_1e-06 \\
        [--step 014] [--d-c 0.82] [--load 0.092]

The default checkpoint dir mirrors the tag ``d12_recheck.py`` uses. If
``--step`` is omitted, the script picks the earliest step whose
``max_d`` exceeds ``d_c + 0.02`` (i.e. the localization onset).

Reads only; never mutates the checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse.linalg as spla

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from fealpy.backend import backend_manager as bm  # noqa: E402

from fracturex.analysis import (  # noqa: E402
    NEPINConfig,
    NEPINEliminator,
    build_nepin_callbacks,
)
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler  # noqa: E402
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel  # noqa: E402
from fracturex.postprocess.dataset_export.adapters.huzhang_phasefield import (  # noqa: E402
    load_discr_from_dir,
)
from fracturex.utilfuc.linear_solvers import as_scipy_csr  # noqa: E402


def _pick_step(ckdir: Path, target_min_maxd: float) -> Optional[str]:
    """Pick the earliest checkpoint step whose max_d exceeds target."""
    for p in sorted(ckdir.glob("step_*.npz")):
        try:
            z = np.load(p)
            if "d" in z.files:
                if float(np.asarray(z["d"], dtype=float).max()) > target_min_maxd:
                    return p.stem.split("_")[1]
        except Exception:
            continue
    return None


def _load_state_into(discr, ck_path: Path) -> float:
    z = np.load(ck_path, allow_pickle=False)
    discr.state.d[:] = bm.asarray(np.asarray(z["d"], dtype=np.float64))
    discr.state.u[:] = bm.asarray(np.asarray(z["u"], dtype=np.float64))
    discr.state.sigma[:] = bm.asarray(np.asarray(z["sigma"], dtype=np.float64))
    if "r_hist" in z.files:
        discr.state.r_hist[:] = bm.asarray(np.asarray(z["r_hist"], dtype=np.float64))
    return float(np.asarray(z["d"], dtype=float).max())


def _solve_dd(A_csr, F_np) -> tuple[np.ndarray, int, bool]:
    """One outer Newton step: solve A dd = F via LGMRES."""
    n = int(F_np.shape[0])
    x, info = spla.lgmres(
        A_csr, F_np, atol=1e-10, rtol=1e-8, maxiter=400, inner_m=200,
        x0=np.zeros(n),
    )
    return np.asarray(x, dtype=np.float64).reshape(-1), int(info), bool(info == 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--checkpoint-dir",
        default=str(_REPO / "results/phasefield/model0_circular_notch/paper_aux_h2/epsg_1e-06"),
        help="recorder root containing mesh.npz + checkpoints/",
    )
    ap.add_argument("--step", default=None, help="e.g. 014; auto-selected if omitted")
    ap.add_argument("--d-c", type=float, default=0.82, dest="d_c")
    ap.add_argument("--load", type=float, default=0.092)
    ap.add_argument("--max-local-iter", type=int, default=5)
    ap.add_argument("--local-tol", type=float, default=1e-2)
    ap.add_argument("--out-csv", default=str(_REPO / "spike_nepin.csv"))
    args = ap.parse_args()

    ckdir_root = Path(args.checkpoint_dir)
    ckdir = ckdir_root / "checkpoints"
    if not ckdir.exists():
        print(f"[spike] ERROR: {ckdir} not found", file=sys.stderr)
        return 2

    step = args.step or _pick_step(ckdir, target_min_maxd=args.d_c + 0.02)
    if step is None:
        print(
            f"[spike] no checkpoint has max_d > {args.d_c + 0.02}; nothing to do.",
            file=sys.stderr,
        )
        return 3
    ck_path = ckdir / f"step_{step}.npz"
    print(f"[spike] loading {ck_path}")

    discr = load_discr_from_dir(ckdir_root)
    max_d_loaded = _load_state_into(discr, ck_path)
    print(f"[spike] state loaded: max d = {max_d_loaded:.4f}")

    # Bind a damage model consistent with the d12_recheck / paper_aux run.
    damage = PhaseFieldDamageModel(
        density_type="AT2", degradation_type="quadratic",
        split="hybrid", eps_g=1e-6,
    )
    damage.on_build(discr, discr.state, discr.case if hasattr(discr, "case") else None)

    phase_asm = PhaseFieldAssembler(discr, discr.case, damage, assembly_parallel=False)

    # ---- BEFORE ---------------------------------------------------------
    d_before = np.asarray(bm.to_numpy(discr.state.d[:]), dtype=np.float64).copy()
    t0 = time.perf_counter()
    sys_d = phase_asm.assemble(args.load)
    t_asm_before = time.perf_counter() - t0
    A_before = as_scipy_csr(sys_d.A)
    F_before = np.asarray(bm.to_numpy(sys_d.F), dtype=np.float64).reshape(-1)

    t0 = time.perf_counter()
    dd_before, info_b, conv_b = _solve_dd(A_before, F_before)
    t_solve_before = time.perf_counter() - t0
    norm_dd_before = float(np.linalg.norm(dd_before))
    print(
        f"[spike] before NEPIN: assemble={t_asm_before:.2f}s, "
        f"solve={t_solve_before:.2f}s, info={info_b}, "
        f"||dd||={norm_dd_before:.3e}"
    )

    # ---- NEPIN ----------------------------------------------------------
    residual, jacobian = build_nepin_callbacks(sys_d, discr.state.d[:])
    cfg = NEPINConfig(
        d_c=args.d_c, max_local_iter=args.max_local_iter, local_tol=args.local_tol,
    )
    elim = NEPINEliminator(residual, jacobian, cfg)
    res = elim.eliminate(discr.state.d[:], None)
    print(
        f"[spike] NEPIN: |S|={res.subset_size}, "
        f"local_iters={res.local_iters}, "
        f"reduction={res.local_res_reduction:.3e}, "
        f"wall={res.wall_time:.2f}s, converged={res.converged}"
    )
    if res.subset_size == 0:
        print("[spike] empty subset; nothing eliminated.")
        return 1

    # Write the eliminated damage back and re-assemble the outer system.
    discr.state.d[:] = bm.clip(bm.asarray(res.d_corrected), 0.0, 1.0)

    # ---- AFTER ----------------------------------------------------------
    t0 = time.perf_counter()
    sys_d2 = phase_asm.assemble(args.load)
    t_asm_after = time.perf_counter() - t0
    A_after = as_scipy_csr(sys_d2.A)
    F_after = np.asarray(bm.to_numpy(sys_d2.F), dtype=np.float64).reshape(-1)
    t0 = time.perf_counter()
    dd_after, info_a, conv_a = _solve_dd(A_after, F_after)
    t_solve_after = time.perf_counter() - t0
    norm_dd_after = float(np.linalg.norm(dd_after))
    print(
        f"[spike] after NEPIN:  assemble={t_asm_after:.2f}s, "
        f"solve={t_solve_after:.2f}s, info={info_a}, "
        f"||dd||={norm_dd_after:.3e}"
    )

    # Contraction: how much smaller is ||dd_after|| relative to ||dd_before||?
    # For Deuflhard's estimator we would need TWO outer iters; a single-step
    # witness is the ratio (>1 = NEPIN reduced the outer increment).
    contraction = (
        norm_dd_before / max(norm_dd_after, 1e-30)
    )
    hat_omega_surrogate = 2.0 * norm_dd_after / max(norm_dd_before * norm_dd_before, 1e-30)
    print(
        f"[spike] contraction = ||dd_before|| / ||dd_after|| = {contraction:.3f}x"
    )
    print(f"[spike] hat_omega surrogate            = {hat_omega_surrogate:.3e}")

    # Restore d_before for reproducibility (this is a spike, not a fix).
    discr.state.d[:] = bm.asarray(d_before)

    # ---- CSV ------------------------------------------------------------
    row = dict(
        step=step,
        max_d=max_d_loaded,
        d_c=args.d_c,
        subset_size=res.subset_size,
        local_iters=res.local_iters,
        local_res_reduction=res.local_res_reduction,
        norm_dd_before=norm_dd_before,
        norm_dd_after=norm_dd_after,
        contraction=contraction,
        hat_omega_surrogate=hat_omega_surrogate,
        outer_info_before=info_b,
        outer_info_after=info_a,
        t_nepin_s=res.wall_time,
        t_asm_before_s=t_asm_before,
        t_asm_after_s=t_asm_after,
        t_solve_before_s=t_solve_before,
        t_solve_after_s=t_solve_after,
    )
    out_path = Path(args.out_csv)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"[spike] wrote {out_path}")

    # Decision aid: does the paper paragraph get written?
    contraction_pct = 1.0 - (norm_dd_after / max(norm_dd_before, 1e-30))
    print(
        f"[spike] outer-increment relative reduction = {100.0 * contraction_pct:.2f}%  "
        f"({'PAPER PARAGRAPH' if contraction_pct >= 0.05 else 'DESIGN DOC ONLY'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
