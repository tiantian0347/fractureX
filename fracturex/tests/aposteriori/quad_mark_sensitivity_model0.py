"""Quadrature sensitivity of η_{T,q} and marked sets on Model-0 snaps.

For each selected snap (frozen mesh + d), reassemble and re-solve the mixed
Hu–Zhang system at several assembly degrees q, compute η_{T,q} via a conforming
re-solve, apply the paper Dörfler marker (L2, θ=0.5 on eligible cells), and
report successive relative changes and Jaccard overlap of marked sets.

Usage on lab (py312 + fracturex on PYTHONPATH)::

    python fracturex/tests/aposteriori/quad_mark_sensitivity_model0.py \\
        --snap-dir ~/tian/Frac_huzhang/figures/model0_rg_rerun \\
        --targets 0.077,0.088,0.10 --qs 14,22,30
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

from fracturex.adaptivity.adaptive_staggered import (
    eta_from_state,
    mark_eta_T_indicator,
)
from fracturex.adaptivity.primal_resolve_real import solve_primal_real
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.drivers.huzhang_phasefield_staggered import (
    HuZhangPhaseFieldStaggeredDriver,
)


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


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return inter / union


def _load_history(snap_dir: Path):
    hist = snap_dir / "history.csv"
    if not hist.exists():
        return []
    return list(csv.DictReader(open(hist)))


def _pick_snaps(snap_dir: Path, targets: list[float], tol: float):
    """Pick snap nearest each target load; return list of (label, path, load, row)."""
    rows = _load_history(snap_dir)
    if not rows:
        # fall back to snap metadata
        snaps = sorted(snap_dir.glob("snap_*.npz"))
        picked = []
        for t in targets:
            best = None
            best_err = 1e9
            for p in snaps:
                z = np.load(p)
                load = float(z["load"])
                err = abs(load - t)
                if err < best_err:
                    best_err = err
                    best = (p, load)
            if best is not None and best_err <= tol:
                picked.append((f"uy≈{targets[targets.index(t)]}", best[0], best[1], None))
        return picked

    by_step = {int(r["step"]): r for r in rows}
    picked = []
    used = set()
    for t in targets:
        best_step = None
        best_err = 1e9
        for step, r in by_step.items():
            load = float(r["load"])
            err = abs(load - t)
            if err < best_err:
                best_err = err
                best_step = step
        if best_step is None or best_err > tol:
            print(f"[pick] no snap within tol={tol} of target {t} "
                  f"(best_err={best_err})", flush=True)
            continue
        if best_step in used:
            continue
        used.add(best_step)
        path = snap_dir / f"snap_{best_step:03d}.npz"
        if not path.exists():
            print(f"[pick] missing {path}", flush=True)
            continue
        picked.append((f"uy≈{t}", path, float(by_step[best_step]["load"]),
                       by_step[best_step]))
    return picked


def _build_from_snap(path: Path):
    z = np.load(path)
    node = np.asarray(z["node"], dtype=np.float64)
    cell = np.asarray(z["cell"], dtype=np.int64)
    d = np.asarray(z["d"], dtype=np.float64)
    load = float(z["load"])
    mesh = TriangleMesh(bm.asarray(node), bm.asarray(cell))
    mat = _Mat()
    case = Model0CircularNotchCase(
        _model=mat,
        circle_cx=0.5,
        circle_cy=0.5,
        circle_r=0.20,
        hmin=0.05,
        distmesh_maxit=100,
        debug_mesh=False,
    )
    discr = HuZhangDiscretization(case=case, p=3, damage_p=1).build(mesh=mesh)
    case.mesh = discr.mesh
    damage = PhaseFieldDamageModel(
        density_type="AT2",
        degradation_type="quadratic",
        split="hybrid",
        eps_g=1e-6,
        debug=False,
    )
    discr.state.d[:] = bm.asarray(d)
    # Bind degradation/crack-density functions (same as driver.initialize).
    from fracturex.damage.base import DamageStateView
    view = DamageStateView(
        d=discr.state.d,
        sigma=discr.state.sigma,
        u=discr.state.u,
        r_hist=discr.state.r_hist,
        H=discr.state.H,
    )
    damage.on_build(discr, view, case)
    # Restore snap damage after on_build clip of the initial zero field.
    discr.state.d[:] = bm.clip(bm.asarray(d), 0.0, damage.clamp_max)
    return case, discr, damage, mat, load


def _solve_mixed(discr, case, damage, load: float, *, q: int, elastic_mode: str):
    # FEALPy Gauss–Legendre face rules only go up to index 20; the production
    # Hu–Zhang assembler uses the same q for volume and face BC terms.
    q_use = min(int(q), 20)
    if q_use != int(q):
        print(f"[warn] requested assembly q={q} capped to {q_use} "
              f"(face quadrature limit)", flush=True)
    elastic_asm = HuZhangElasticAssembler(
        discr, case, damage, formulation="standard",
        assembly_parallel=False, q=q_use,
    )
    phase_asm = PhaseFieldAssembler(
        discr, case, damage, debug=False, assembly_parallel=False,
    )
    if elastic_mode == "direct":
        solver = HuZhangPhaseFieldStaggeredDriver._default_spsolve
    else:
        from fracturex.utilfuc.linear_solvers import (
            solve_huzhang_block_gmres_auxspace,
        )

        def solver(A, F):
            x, _ = solve_huzhang_block_gmres_auxspace(
                A, F,
                gdof_sigma=discr.gdof_sigma,
                vspace=discr.space_u,
                atol=1e-12,
                rtol=1e-8,
            )
            return x

    if hasattr(elastic_asm, "begin_load_step"):
        elastic_asm.begin_load_step(load)
    sys_e = elastic_asm.assemble(load)
    Xe = solver(sys_e.A, sys_e.F)
    sigma, u, _ = sys_e.decode(Xe)
    discr.state.sigma[:] = sigma[:]
    discr.state.u[:] = u[:]
    return elastic_asm, phase_asm


def _analyze_snap(label: str, path: Path, qs: list[int], *,
                  theta: float, c_h: float, d_hi: float,
                  strategy: str, elastic_mode: str, p_u: int):
    print(f"\n=== {label}  {path.name} ===", flush=True)
    case, discr, damage, mat, load = _build_from_snap(path)
    nc = int(discr.mesh.number_of_cells())
    max_d = float(bm.max(bm.abs(discr.state.d[:])))
    print(f"[state] load={load:.6g} NC={nc} max_d={max_d:.6g} "
          f"mode={elastic_mode} strategy={strategy}", flush=True)

    rows = []
    marked_by_q = {}
    for q in qs:
        t0 = time.time()
        _solve_mixed(discr, case, damage, load, q=q, elastic_mode=elastic_mode)
        prim = solve_primal_real(
            discr, case, lam=mat.lam, mu=mat.mu, load=load,
            k_res=1e-6, p=p_u, q=min(int(q), 20),
        )
        ind = eta_from_state(
            discr, lam=mat.lam, mu=mat.mu, k_res=1e-6, q=min(int(q), 20),
            u_override=prim["uh"],
        )
        eta = float(ind["eta"])
        eta_T = np.asarray(bm.to_numpy(ind["eta_T"]), dtype=np.float64)
        eta_T2 = eta_T ** 2
        marked = mark_eta_T_indicator(
            discr, bm.asarray(eta_T2),
            l0=mat.l0, c_h=c_h, theta=theta, d_hi=d_hi, strategy=strategy,
        )
        marked_np = np.asarray(bm.to_numpy(marked), dtype=bool)
        marked_by_q[q] = marked_np
        n_mark = int(marked_np.sum())
        dt = time.time() - t0
        row = dict(q=q, eta=eta, n_marked=n_mark, t=dt, NC=nc, load=load)
        rows.append(row)
        print(f"[q={q:2d}] eta={eta:.10e}  n_marked={n_mark}  ({dt:.1f}s)",
              flush=True)

    for i in range(len(rows) - 1):
        q0, q1 = rows[i]["q"], rows[i + 1]["q"]
        e0, e1 = rows[i]["eta"], rows[i + 1]["eta"]
        rel = abs(e1 - e0) / max(abs(e1), 1e-30)
        jac = _jaccard(marked_by_q[q0], marked_by_q[q1])
        print(f"[rel] q={q0}->{q1}:  rel|Δη|/η={rel:.3e}  "
              f"Jaccard={jac:.6f}  n_mark={rows[i]['n_marked']}->"
              f"{rows[i+1]['n_marked']}", flush=True)

    return {
        "label": label,
        "snap": path.name,
        "load": load,
        "NC": nc,
        "max_d": max_d,
        "rows": rows,
        "jaccard": {
            f"{rows[i]['q']}->{rows[i+1]['q']}": _jaccard(
                marked_by_q[rows[i]["q"]], marked_by_q[rows[i + 1]["q"]]
            )
            for i in range(len(rows) - 1)
        },
        "rel_eta": {
            f"{rows[i]['q']}->{rows[i+1]['q']}":
                abs(rows[i + 1]["eta"] - rows[i]["eta"])
                / max(abs(rows[i + 1]["eta"]), 1e-30)
            for i in range(len(rows) - 1)
        },
    }


def main():
    bm.set_backend("numpy")
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-dir", type=str,
                    default=os.path.expanduser(
                        "~/tian/Frac_huzhang/figures/model0_rg_rerun"))
    ap.add_argument("--targets", type=str, default="0.077,0.088,0.10")
    ap.add_argument("--tol", type=float, default=0.004,
                    help="max |load-target| to accept a snap")
    ap.add_argument("--qs", type=str, default="10,14,18",
                    help="assembly/estimator quadrature degrees "
                         "(face rules cap at 20 in FEALPy)")
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--c-h", type=float, default=3.0)
    ap.add_argument("--d-hi", type=float, default=0.995)
    ap.add_argument("--strategy", type=str, default="L2",
                    choices=("L2", "max"))
    ap.add_argument("--elastic-mode", type=str, default="direct",
                    choices=("direct", "aux"))
    ap.add_argument("--p-u", type=int, default=2)
    ap.add_argument("--snap", type=str, default="",
                    help="optional explicit snap_XXX.npz (skip target pick)")
    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()

    snap_dir = Path(args.snap_dir).expanduser()
    qs = [int(x) for x in args.qs.split(",") if x.strip()]
    outdir = Path(args.outdir).expanduser() if args.outdir else snap_dir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.snap:
        path = Path(args.snap).expanduser()
        if not path.is_absolute():
            path = snap_dir / path
        picks = [("explicit", path, float(np.load(path)["load"]), None)]
    else:
        targets = [float(x) for x in args.targets.split(",") if x.strip()]
        picks = _pick_snaps(snap_dir, targets, args.tol)

    if not picks:
        print("[quad] no snaps selected; abort", flush=True)
        # list available loads for the monitor
        rows = _load_history(snap_dir)
        if rows:
            loads = [float(r["load"]) for r in rows]
            print(f"[quad] available loads: {loads}", flush=True)
        raise SystemExit(2)

    reports = []
    for label, path, _load, _row in picks:
        rep = _analyze_snap(
            label, path, qs,
            theta=args.theta, c_h=args.c_h, d_hi=args.d_hi,
            strategy=args.strategy, elastic_mode=args.elastic_mode,
            p_u=args.p_u,
        )
        reports.append(rep)

    out = outdir / "quad_mark_sensitivity.json"
    out.write_text(json.dumps(reports, indent=2))
    print(f"\n[quad] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
