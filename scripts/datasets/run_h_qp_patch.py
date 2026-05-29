"""Short patch run that produces ``H_qp.npz`` quadrature snapshots at t_a / t_b / t_c.

Mirrors the ``paper_aux_h1`` configuration (Model0CircularNotchCase,
hmin≈0.05, p_sigma=3, AT2/quadratic/hybrid phase field) but with
``save_quadrature_fields=True`` and ``save_every=10`` so only
``step_000 / 010 / 020 / 030`` checkpoints are kept. Used to feed
``measure_interpolation_error.py``'s §4.2 H-channel comparison.

Wall-time on a single CPU core is roughly the same per-step as
paper_aux_h1; 31 steps total. Output:

    results/operator_learning_runs/h_qp_patch_h1/
        meta.json
        history.csv
        iterations.csv
        mesh.npz
        checkpoints/
            step_000.npz
            step_000_qp.npz   <- H_qp + xq, q_order
            step_010.npz
            step_010_qp.npz
            ...
"""

from __future__ import annotations

from pathlib import Path

from fracturex.tests.case_runners.model0_runner import Model0RunArgs, run_model0_one


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    outdir = repo_root / "results" / "operator_learning_runs" / "h_qp_patch_h1"

    args = Model0RunArgs(
        circle_r=0.2,
        circle_cx=0.5,
        circle_cy=0.5,
        hmin=0.05,
        E=200.0,
        nu=0.2,
        Gc=1.0,
        l0=0.02,
        p_sigma=3,
        damage_p=2,
        use_relaxation=True,
        elastic_mode="direct",
        eps_g=1e-6,
        save_every=10,
        save_quadrature_fields=True,
        outdir=outdir,
    )
    out = run_model0_one(args)
    print(f"[h_qp_patch] recorder dir: {out}")


if __name__ == "__main__":
    main()
