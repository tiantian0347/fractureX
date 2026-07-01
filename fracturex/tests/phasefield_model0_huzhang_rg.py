"""Model-0 Hu-Zhang phase-field with red-green mesh adaptivity.

Compared to the paper driver ``phasefield_model0_huzhang.py`` (uses fealpy's
bisect) or ``adaptive_staggered.py`` (bisect + equilibrated estimator), this
driver:

  - marks cells with the M-DF driving-force indicator (Θ) already available
    in ``fracturex.adaptivity.adaptive_staggered``;
  - refines the triangle mesh using the newly ported red-green (rg)
    refinement from ``fracturex/mesh/halfedge_mesh.py`` via
    ``rg_refine_bridge.refine_rg_p1``;
  - transfers the P1 damage field d and history r_hist exactly on the new
    midpoint nodes (linear interpolation on midpoints of old edges);
  - rebuilds the Hu-Zhang discretization and resumes the staggered solve.

Usage
-----

    python -m fracturex.tests.phasefield_model0_huzhang_rg [--hmin 0.06] \
        [--max-cells 8000] [--adapt-every 1] [--adapt-until 30] \
        [--n-loads 15] [--outdir results/model0_rg]

The default schedule is a truncated version of the paper's:
    linspace(0, 0.07, 6) ∪ linspace(0.07, 0.125, 26)[1:]

so use ``--n-loads`` to limit the number of steps executed.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from fealpy.backend import backend_manager as bm

from fracturex.adaptivity.adaptive_staggered import (
    driving_force_per_cell,
    mark_driving_force,
)
from fracturex.adaptivity.rg_refine_bridge import refine_rg_p1
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.drivers.huzhang_phasefield_staggered import (
    HuZhangPhaseFieldStaggeredDriver,
)


def default_loads():
    return bm.concatenate(
        (
            bm.linspace(0.0, 70e-3, 6, dtype=bm.float64),
            bm.linspace(70e-3, 125e-3, 26, dtype=bm.float64)[1:],
        )
    )


def build_pipeline(hmin: float, *, l0: float = 0.02):
    """Build case/discr/damage/assemblers for Model-0."""

    class _Mat:
        E = 200.0
        nu = 0.2
        Gc = 1.0
        l0 = 0.02

        @property
        def mu(self):
            return self.E / (2.0 * (1.0 + self.nu))

        @property
        def lam(self):
            return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    mat = _Mat()
    mat.l0 = float(l0)
    case = Model0CircularNotchCase(
        _model=mat,
        circle_cx=0.5,
        circle_cy=0.5,
        circle_r=0.20,
        hmin=hmin,
        distmesh_maxit=100,
        debug_mesh=False,
    )
    discr = HuZhangDiscretization(case=case, p=3, damage_p=1).build()
    damage = PhaseFieldDamageModel(
        density_type="AT2",
        degradation_type="quadratic",
        split="hybrid",
        eps_g=1e-6,
        debug=False,
    )
    elastic_asm = HuZhangElasticAssembler(
        discr, case, damage, formulation="standard", assembly_parallel=False
    )
    phase_asm = PhaseFieldAssembler(
        discr, case, damage, debug=False, assembly_parallel=False
    )
    return case, discr, damage, elastic_asm, phase_asm


def make_driver(case, discr, damage, elastic_asm, phase_asm, *, tol=1e-4, maxit=30):
    driver = HuZhangPhaseFieldStaggeredDriver(
        case=case,
        discr=discr,
        damage=damage,
        elastic_assembler=elastic_asm,
        phase_assembler=phase_asm,
        tol=tol,
        maxit=maxit,
    )
    driver.initialize()
    return driver


def do_rg_adapt(
    case,
    discr,
    damage,
    elastic_asm,
    phase_asm,
    *,
    beta: float,
    c_h: float,
    max_cells: int,
) -> dict:
    """One M-DF mark + red-green refine + rebuild cycle.

    Returns a diagnostics dict.
    """
    NC_before = discr.mesh.number_of_cells()
    if NC_before >= max_cells:
        return {"refined": False, "reason": "max_cells", "nc_before": NC_before,
                "nc_after": NC_before, "n_marked": 0}

    Dcell = driving_force_per_cell(
        discr, damage, Gc=case._model.Gc, l0=case._model.l0
    )
    marked = mark_driving_force(
        discr, Dcell, l0=case._model.l0, beta=beta, c_h=c_h
    )
    n_marked = int(bm.sum(marked))
    if n_marked == 0:
        return {"refined": False, "reason": "no_cells", "nc_before": NC_before,
                "nc_after": NC_before, "n_marked": 0}

    d_old = np.asarray(discr.state.d[:]).astype(np.float64)
    r_old = np.asarray(discr.state.r_hist[:]).astype(np.float64)

    new_mesh, new_fields, _ = refine_rg_p1(
        discr.mesh,
        np.asarray(marked),
        fields={"d": d_old, "r_hist": r_old},
    )

    d_new = bm.asarray(new_fields["d"])
    r_new = bm.asarray(new_fields["r_hist"])

    def _transfer(old_discr, new_discr, old_st, new_st):
        new_st.d[:] = d_new
        new_st.r_hist[:] = r_new

    case.mesh = new_mesh
    discr.rebuild_on_new_mesh(new_mesh, transfer=_transfer)
    damage.on_mesh_changed(None, discr, None, discr.state, case)
    # Assemblers hold references to discr fields; refresh their view.
    elastic_asm.discr = discr
    phase_asm.discr = discr

    return {
        "refined": True,
        "reason": "marked",
        "nc_before": NC_before,
        "nc_after": discr.mesh.number_of_cells(),
        "n_marked": n_marked,
    }


def run(args) -> Path:
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[model0-rg] hmin={args.hmin}  max_cells={args.max_cells}  "
          f"adapt_every={args.adapt_every}  adapt_until={args.adapt_until}")

    t_total = time.time()
    case, discr, damage, elastic_asm, phase_asm = build_pipeline(
        hmin=args.hmin, l0=args.l0
    )
    print(f"[model0-rg] initial mesh: NC={discr.mesh.number_of_cells()} "
          f"NN={discr.mesh.number_of_nodes()}")

    loads = np.asarray(default_loads(), dtype=np.float64)
    if args.n_loads is not None:
        loads = loads[: int(args.n_loads)]

    log = []
    driver = make_driver(
        case, discr, damage, elastic_asm, phase_asm,
        tol=args.tol, maxit=args.maxit,
    )

    for k, load in enumerate(loads):
        t0 = time.time()
        info = driver.solve_one_step(step=k, load=float(load))
        t_solve = time.time() - t0
        max_d = float(bm.max(bm.abs(discr.state.d[:])))
        record = {
            "step": k,
            "load": float(load),
            "solve_s": t_solve,
            "nc_solve": discr.mesh.number_of_cells(),
            "nn_solve": discr.mesh.number_of_nodes(),
            "max_d": max_d,
        }

        if (
            k >= args.adapt_until
            or k % args.adapt_every != 0
            or k == len(loads) - 1
        ):
            record["refined"] = False
            log.append(record)
            print(f"[step {k}] load={load:.4e}  NC={record['nc_solve']}  "
                  f"solve={t_solve:.2f}s  max_d={max_d:.3e}")
            continue

        t0 = time.time()
        adapt_info = do_rg_adapt(
            case, discr, damage, elastic_asm, phase_asm,
            beta=args.beta, c_h=args.c_h, max_cells=args.max_cells,
        )
        t_adapt = time.time() - t0
        record.update({
            "refined": adapt_info["refined"],
            "refine_reason": adapt_info["reason"],
            "n_marked": adapt_info["n_marked"],
            "nc_after_adapt": adapt_info["nc_after"],
            "adapt_s": t_adapt,
        })
        log.append(record)
        # Recreate driver so it sees the (possibly rebuilt) discr / assemblers.
        driver = make_driver(
            case, discr, damage, elastic_asm, phase_asm,
            tol=args.tol, maxit=args.maxit,
        )
        print(f"[step {k}] load={load:.4e}  NC={record['nc_solve']}  "
              f"solve={t_solve:.2f}s  max_d={max_d:.3e}  "
              f"{'refined' if adapt_info['refined'] else adapt_info['reason']} "
              f"({adapt_info['n_marked']} marked, "
              f"NC->{adapt_info['nc_after']}, {t_adapt:.2f}s)")

    total_s = time.time() - t_total
    print(f"\n[model0-rg] finished {len(loads)} steps in {total_s:.1f}s "
          f"(final NC={discr.mesh.number_of_cells()})")

    # Dump log
    import json
    (outdir / "run_log.json").write_text(json.dumps(log, indent=2))
    print(f"[model0-rg] log saved to {outdir / 'run_log.json'}")

    return outdir


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hmin", type=float, default=0.06,
                    help="initial distmesh target edge length")
    ap.add_argument("--l0", type=float, default=0.02,
                    help="phase-field length scale (Gc, AT2 constants use this)")
    ap.add_argument("--max-cells", type=int, default=6000,
                    help="stop refining once NC exceeds this")
    ap.add_argument("--adapt-every", type=int, default=1,
                    help="refine every N load steps")
    ap.add_argument("--adapt-until", type=int, default=1_000_000,
                    help="disable adaptivity after this step index")
    ap.add_argument("--n-loads", type=int, default=None,
                    help="only run the first N loads (default: full schedule)")
    ap.add_argument("--beta", type=float, default=0.6,
                    help="M-DF driving-force threshold parameter (see mark_driving_force)")
    ap.add_argument("--c-h", type=float, default=2.0,
                    help="M-DF cell-size coupling parameter")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="staggered convergence tolerance")
    ap.add_argument("--maxit", type=int, default=30,
                    help="staggered iterations per load step")
    ap.add_argument("--outdir", type=str,
                    default="results/model0_rg")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
