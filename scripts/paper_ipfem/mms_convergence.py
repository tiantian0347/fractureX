"""MMS convergence driver for the ip_fracture paper.

Two studies:

(A) Scalar C0-IP biharmonic: verifies the Theorem stated for the frozen-history
    phase-field subproblem (broken H^2 rate O(h^{p-1})).

(B) Length-scale-augmented displacement (SG elasticity, d=0): verifies
    Theorem 4.3 for the length-scale-augmented displacement subproblem.

Both studies write CSVs and log-log PNGs under
``$FRACTUREX_RESULTS_ROOT/paper_ipfem/`` (default ``results/paper_ipfem/``).

Usage
-----
    python scripts/paper_ipfem/mms_convergence.py
    python scripts/paper_ipfem/mms_convergence.py --p 2 3 4 --maxit 5
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure repo root on path when invoked as a plain script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fealpy.backend import backend_manager as bm  # noqa: E402

from fracturex.interior_penalty import (  # noqa: E402
    convergence_study,
    sin_sq_pde,
)
from fracturex.interior_penalty.tests.test_sg_elastic_mms import (  # noqa: E402
    h1_error as sg_h1_error,
    l2_error as sg_l2_error,
    solve_sg_elastic,
)


# ---------------------------------------------------------------------------

def _order(errs: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Observed rate between successive mesh refinements."""
    out = np.full_like(errs, np.nan, dtype=np.float64)
    for i in range(1, len(errs)):
        if errs[i - 1] > 0 and errs[i] > 0 and h[i - 1] != h[i]:
            out[i] = np.log2(errs[i - 1] / errs[i]) / np.log2(h[i - 1] / h[i])
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _loglog_plot(png_path: Path, h_by_p: dict, err_by_p: dict,
                 ylabel: str, ref_slopes: list[int]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[warn] matplotlib not available, skipping {png_path.name}")
        return

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for p in sorted(h_by_p):
        h = np.asarray(h_by_p[p])
        e = np.asarray(err_by_p[p])
        m = e > 0
        ax.loglog(h[m], e[m], marker="o", label=f"p={p}")
    # Reference slopes anchored at the finest point of the smallest p curve.
    p_ref = min(h_by_p)
    h_ref = np.asarray(h_by_p[p_ref])
    e_ref = np.asarray(err_by_p[p_ref])
    if len(h_ref) > 0 and e_ref[-1] > 0:
        x0, y0 = h_ref[-1], e_ref[-1]
        for s in ref_slopes:
            xs = np.array([x0 * 0.5, x0 * 2.0])
            ys = y0 * (xs / x0) ** s
            ax.loglog(xs, ys, "k--", lw=0.8, alpha=0.5)
            ax.annotate(f"O(h^{s})", (xs[1], ys[1]), fontsize=8, alpha=0.6)
    ax.set_xlabel("h")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------

def study_biharmonic(out_dir: Path, ps: list[int], maxit: int,
                     nx0: int, gamma_by_p: dict[int, float]) -> None:
    print("== (A) Scalar C0-IP biharmonic MMS ==", flush=True)
    print(f"[info] biharmonic γ by p: {gamma_by_p}", flush=True)
    pde = sin_sq_pde()
    rows: list[dict] = []
    h_by_p, l2_by_p, h1_by_p, h2_by_p = {}, {}, {}, {}
    for p in ps:
        gamma_p = gamma_by_p[p]
        t0 = time.time()
        h, eL2, eH1, eH2 = convergence_study(
            pde, p=p, maxit=maxit, nx0=nx0, gamma=gamma_p
        )
        dt = time.time() - t0
        h_by_p[p] = h
        l2_by_p[p] = eL2
        h1_by_p[p] = eH1
        h2_by_p[p] = eH2
        ord_l2 = _order(eL2, h)
        ord_h1 = _order(eH1, h)
        ord_h2 = _order(eH2, h)
        for i in range(maxit):
            rows.append({
                "p": p, "level": i, "h": float(h[i]),
                "errL2": float(eL2[i]), "orderL2": float(ord_l2[i]),
                "errH1": float(eH1[i]), "orderH1": float(ord_h1[i]),
                "errH2_broken": float(eH2[i]), "orderH2": float(ord_h2[i]),
            })
        print(f"  p={p} (γ={gamma_p}) finished in {dt:.1f}s: "
              f"final H2 err={eH2[-1]:.3e}, rate~{ord_h2[-1]:.2f}",
              flush=True)

    _write_csv(out_dir / "biharmonic_mms.csv", rows)
    _loglog_plot(
        out_dir / "biharmonic_mms_H2.png",
        h_by_p, h2_by_p,
        ylabel=r"broken $H^2$ semi-norm error",
        ref_slopes=[p - 1 for p in ps],
    )
    _loglog_plot(
        out_dir / "biharmonic_mms_L2.png",
        h_by_p, l2_by_p,
        ylabel=r"$L^2$ error",
        ref_slopes=[p + 1 for p in ps],
    )


# ---------------------------------------------------------------------------

def study_sg_elasticity(out_dir: Path, ps: list[int], nx_list: list[int],
                        ell_s_list: list[float],
                        gamma_by_p: dict[int, float]) -> None:
    print("== (B) Length-scale-augmented elasticity MMS (d=0) ==", flush=True)
    rows: list[dict] = []
    for ell_s in ell_s_list:
        h_by_p, l2_by_p, h1_by_p = {}, {}, {}
        for p in ps:
            gamma_p = gamma_by_p[p]
            hs, l2s, h1s = [], [], []
            for nx in nx_list:
                t0 = time.time()
                uh, tspace, space, mesh = solve_sg_elastic(
                    nx=nx, p_u=p, ell_s=ell_s, gamma=gamma_p
                )
                dt = time.time() - t0
                cm = np.asarray(mesh.entity_measure("cell"))
                h = float(np.sqrt(np.max(cm)))
                l2e = sg_l2_error(uh, space, mesh, p)
                h1e = sg_h1_error(uh, space, mesh, p)
                hs.append(h)
                l2s.append(l2e)
                h1s.append(h1e)
                print(f"  ell_s={ell_s:.3f} p={p} (γ={gamma_p}) nx={nx}: "
                      f"h={h:.4f} L2={l2e:.3e} H1={h1e:.3e} ({dt:.1f}s)",
                      flush=True)
            hs = np.array(hs); l2s = np.array(l2s); h1s = np.array(h1s)
            h_by_p[p] = hs
            l2_by_p[p] = l2s
            h1_by_p[p] = h1s
            ord_l2 = _order(l2s, hs)
            ord_h1 = _order(h1s, hs)
            for i, nx in enumerate(nx_list):
                rows.append({
                    "ell_s": ell_s, "p": p, "nx": nx, "h": float(hs[i]),
                    "errL2": float(l2s[i]), "orderL2": float(ord_l2[i]),
                    "errH1": float(h1s[i]), "orderH1": float(ord_h1[i]),
                })
        tag = f"ell{ell_s:.2f}".replace(".", "p")
        _loglog_plot(
            out_dir / f"sg_mms_H1_{tag}.png",
            h_by_p, h1_by_p,
            ylabel=r"$H^1$ semi-norm error",
            ref_slopes=[p for p in ps],
        )
        _loglog_plot(
            out_dir / f"sg_mms_L2_{tag}.png",
            h_by_p, l2_by_p,
            ylabel=r"$L^2$ error",
            ref_slopes=[p + 1 for p in ps],
        )

    _write_csv(out_dir / "sg_elasticity_mms.csv", rows)


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--p", type=int, nargs="+", default=[2, 3, 4],
                        help="polynomial degrees to sweep")
    parser.add_argument("--maxit", type=int, default=5,
                        help="biharmonic study: number of h-refinement levels")
    parser.add_argument("--nx0", type=int, default=4,
                        help="biharmonic study: coarsest nx")
    parser.add_argument("--sg-nx", type=int, nargs="+",
                        default=[4, 8, 16, 32, 64],
                        help="SG study: sequence of nx values")
    parser.add_argument("--ell-s", type=float, nargs="+",
                        default=[0.0, 0.05, 0.1],
                        help="SG study: length-scale sweep")
    parser.add_argument("--bih-gamma-p2", type=float, default=5.0,
                        help="biharmonic study γ for p=2 (fracture experiments)")
    parser.add_argument("--bih-gamma-p3", type=float, default=10.0,
                        help="biharmonic study γ for p=3 (fracture experiments)")
    parser.add_argument("--bih-gamma-p4", type=float, default=20.0,
                        help="biharmonic study γ for p=4 (fracture experiments)")
    parser.add_argument("--sg-gamma-p2", type=float, default=5.0,
                        help="SG study γ for p_u=2 (matches fracture experiments)")
    parser.add_argument("--sg-gamma-p3", type=float, default=10.0,
                        help="SG study γ for p_u=3 (matches degree-adaptive "
                             "fracture experiments)")
    parser.add_argument("--out", type=str, default=None,
                        help="output directory (default $FRACTUREX_RESULTS_ROOT/paper_ipfem)")
    parser.add_argument("--skip-biharmonic", action="store_true")
    parser.add_argument("--skip-sg", action="store_true")
    args = parser.parse_args()

    bm.set_backend("numpy")

    if args.out is not None:
        out_dir = Path(args.out)
    else:
        results_root = os.environ.get("FRACTUREX_RESULTS_ROOT",
                                      str(_REPO_ROOT / "results"))
        out_dir = Path(results_root) / "paper_ipfem"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] writing outputs to {out_dir}", flush=True)

    t_all = time.time()
    if not args.skip_biharmonic:
        bih_gamma_by_p = {2: args.bih_gamma_p2, 3: args.bih_gamma_p3,
                          4: args.bih_gamma_p4}
        # for any p outside {2,3,4}, fall back to the highest known γ
        max_bih_gamma = max(bih_gamma_by_p.values())
        for p in args.p:
            bih_gamma_by_p.setdefault(p, max_bih_gamma)
        study_biharmonic(out_dir, args.p, args.maxit, args.nx0,
                         bih_gamma_by_p)
    if not args.skip_sg:
        # SG solve_sg_elastic currently supports p=2,3 in existing tests; keep
        # separate to allow finer control from CLI.
        sg_ps = [p for p in args.p if p in (2, 3)]
        if not sg_ps:
            print("[warn] SG study needs p in {2,3}; skipping.", flush=True)
        else:
            gamma_by_p = {2: args.sg_gamma_p2, 3: args.sg_gamma_p3}
            print(f"[info] SG study γ by p: {gamma_by_p}", flush=True)
            study_sg_elasticity(out_dir, sg_ps, args.sg_nx, args.ell_s,
                                gamma_by_p)

    print(f"[done] total {time.time()-t_all:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
