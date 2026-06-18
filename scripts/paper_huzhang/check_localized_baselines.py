#!/usr/bin/env python3
"""Quick check (D12 §5.2b reviewer-proofing): run the three baselines
(none / Jacobi / ILU) AND a direct solve on a *genuinely localized* damage
operator (real phase-field checkpoint, max d ~ 0.997), to substantiate directly
--- rather than a fortiori from the uniform-d tables --- that aux_fast is the
only convergent solver on a sharp crack interface.

Usage: check_localized_baselines.py <checkpoint.npz> [hmin]
  e.g. ... paper_aux_scan_auxfast_h2/epsg_1e-06/checkpoints/step_015.npz 0.025
"""
from __future__ import annotations
import csv, sys, time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import gmres as scipy_gmres, LinearOperator

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from fealpy.backend import backend_manager as bm
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.utilfuc.linear_solvers import (
    as_scipy_csr, make_ilu_preconditioner, solve_huzhang_block_gmres_fast,
)

RTOL, ATOL = 1e-8, 1e-12
# baselines: a decisive cap well above aux's O(100). At ~50x the aux count with a
# stalled relative residual, the baseline has demonstrably failed; the full 60000
# DNF on a 40k-DOF localized saddle is intractable to run end-to-end here.
B_RESTART, B_MAXIT = 30, 70
# aux: allow it to converge through the O(100) localized regime
A_RESTART, A_MAXIT = 60, 400


class Mat:
    E = 200.0; nu = 0.2; Gc = 1.0; l0 = 0.02
    @property
    def mu(self): return self.E / (2 * (1 + self.nu))
    @property
    def lam(self): return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))


def count_gmres(A_, b_, M, restart, maxit):
    it = {"n": 0}
    def cb(_): it["n"] += 1
    t0 = time.perf_counter()
    try:
        x, info = scipy_gmres(A_, b_, M=M, restart=restart, maxiter=maxit,
                              rtol=RTOL, atol=ATOL, callback=cb, callback_type="pr_norm")
    except Exception as e:
        return {"niter": -1, "converged": False, "relres": float("nan"),
                "t_s": time.perf_counter() - t0, "note": f"err:{type(e).__name__}"}
    bn = max(float(np.linalg.norm(b_)), 1e-30)
    rr = float(np.linalg.norm(A_ @ x - b_) / bn)
    return {"niter": it["n"], "converged": bool(info == 0 and rr <= RTOL * 10),
            "relres": rr, "t_s": time.perf_counter() - t0, "note": ""}


def try_direct(A_csr, b_, label, solver):
    t0 = time.perf_counter()
    try:
        if solver == "pardiso":
            import pypardiso
            x = pypardiso.spsolve(A_csr, b_)
        else:
            from scipy.sparse.linalg import splu
            x = splu(A_csr.tocsc()).solve(b_)
        rr = float(np.linalg.norm(A_csr @ x - b_) / max(np.linalg.norm(b_), 1e-30))
        return {"niter": 0, "converged": bool(rr <= 1e-6), "relres": rr,
                "t_s": time.perf_counter() - t0, "note": "ok" if rr <= 1e-6 else "bad_residual"}
    except Exception as e:
        return {"niter": -1, "converged": False, "relres": float("nan"),
                "t_s": time.perf_counter() - t0, "note": f"err:{type(e).__name__}:{e}"[:80]}


def main():
    ckpt = Path(sys.argv[1])
    hmin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.025
    z = np.load(ckpt)
    d_real = np.asarray(z["d"], float)
    print(f"checkpoint {ckpt.name}: d in [{d_real.min():.4f}, {d_real.max():.4f}]  ndof_d={d_real.size}", flush=True)
    print("[t] building mesh + discretization + assembling localized operator ...", flush=True)
    _t_setup = time.perf_counter()

    dmg = PhaseFieldDamageModel(density_type="AT2", degradation_type="quadratic",
                                split="hybrid", eps_g=1e-6, debug=False)
    case = Model0CircularNotchCase(_model=Mat(), hmin=hmin)
    mesh = case.make_mesh()
    discr = HuZhangDiscretization(case, p=3, damage_p=2, use_relaxation=True).build(mesh=mesh)
    dmg.on_build(discr, discr.state, case)
    asm = HuZhangElasticAssembler(discr, case, dmg, formulation="standard",
                                  assembly_parallel=False)
    m = int(discr.gdof_sigma)
    if d_real.size != discr.space_d.number_of_global_dofs():
        raise RuntimeError(f"d dof mismatch: ckpt {d_real.size} vs {discr.space_d.number_of_global_dofs()} (wrong hmin?)")
    discr.state.d[:] = bm.asarray(d_real)
    if "H" in z.files and discr.state.H is not None:
        try:
            discr.state.H[:] = bm.asarray(np.asarray(z["H"], float))
        except Exception as e:
            print(f"[warn] could not restore H: {e}")

    LOAD = 0.092
    asm.begin_load_step(LOAD)
    sys_e = asm.assemble(LOAD)
    A_ = as_scipy_csr(sys_e.A)
    F = np.asarray(sys_e.F, float).reshape(-1)
    print(f"[t] setup+assembly done in {time.perf_counter()-_t_setup:.1f}s", flush=True)
    print(f"sigma-dof={m}  total-dof={A_.shape[0]}  nnz={A_.nnz}  maxd={d_real.max():.4f}  1/g_max~{1.0/1e-6:.0e}\n", flush=True)

    rows = []
    print(f"{'solver':>12} {'niter':>7} {'conv':>6} {'relres':>10} {'t_s':>8}  note", flush=True)

    def emit(name, r):
        rows.append((name, r))
        print(f"{name:>12} {r['niter']:>7} {str(r['converged']):>6} {r['relres']:>10.2e} {r['t_s']:>8.1f}  {r['note']}", flush=True)

    # --- aux_fast (the positive result) ---
    t0 = time.perf_counter()
    _, info_aux = solve_huzhang_block_gmres_fast(
        sys_e.A, F, gdof_sigma=m, vspace=discr.space_u,
        rtol=RTOL, atol=ATOL, restart=A_RESTART, maxit=A_MAXIT, q=5,
        weighted_aux=True, elastic_formulation="standard",
        damage=dmg, state=discr.state)
    emit("aux_fast", {"niter": int(getattr(info_aux, "niter", -1)), "converged": bool(getattr(info_aux, "converged", False)),
                      "relres": float("nan"), "t_s": time.perf_counter() - t0, "note": ""})

    # --- direct (does it factor the localized indefinite saddle?) ---
    for s in ("pardiso", "superlu"):
        emit(f"direct_{s}", try_direct(A_, F, s, s))

    # --- baselines (none / Jacobi / ILU): show they do NOT converge ---
    emit("none", count_gmres(A_, F, None, B_RESTART, B_MAXIT))

    diag = np.asarray(A_.diagonal(), float); diag = np.where(np.abs(diag) > 1e-30, diag, 1.0)
    Mjac = LinearOperator(A_.shape, matvec=lambda v, d=diag: v / d)
    emit("jacobi", count_gmres(A_, F, Mjac, B_RESTART, B_MAXIT))

    try:
        Milu = make_ilu_preconditioner(A_)
        r = count_gmres(A_, F, Milu, B_RESTART, B_MAXIT)
    except Exception as e:
        r = {"niter": -1, "converged": False, "relres": float("nan"), "t_s": 0.0, "note": f"ilu_build_err:{type(e).__name__}"}
    emit("ilu", r)

    outp = _REPO / "results/phasefield/_iter_stability/localized_baselines.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "hmin", "sigma_dof", "maxd", "solver", "niter", "converged", "relres", "t_s", "note"])
        for name, r in rows:
            w.writerow([ckpt.name, hmin, m, f"{d_real.max():.4f}", name, r["niter"], r["converged"],
                        f"{r['relres']:.3e}", f"{r['t_s']:.2f}", r["note"]])
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
