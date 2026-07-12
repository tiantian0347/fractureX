"""Three-discretization p-version assembly benchmark (T9 / framework paper).

Companion to ``hz_assembly_scaling_benchmark.py`` (pillar 1, Hu-Zhang). Where the
HZ script sweeps threads/DOF for the mixed assembler, this one sweeps the
**polynomial order p** for the other two discretization paradigms that share the
FEALPy backend:

  * standard Lagrange displacement PFM   (``phasefield.main_solve.MainSolve``)
  * C0 interior-penalty 4th-order PFM     (``interior_penalty.IPFEMPhaseFieldSolver``)

Assembly is timed in isolation (no linear solve), on a shared structured unit
square so DOF scaling is clean (no distmesh variance). Kernels timed:

  standard-elastic : LinearElasticIntegrator displacement stiffness vs p_disp
  ip-elastic       : same elastic kernel on the IP solver's tspace vs p_disp
                     (comparison point: identical kernel, different discretization)
  ip-phase4th      : IPFEMPhaseFieldSolver._assemble_phase_lhs() vs p_phase
                     (biharmonic + interior penalty = the 4th-order distinctive cost)

The HZ pillar is p-swept by its own script (HZ element needs p>=3 and a different
assembler object); it is intentionally NOT re-run here.

Run (repo root, venv_fealpy3 active):
    python -m fracturex.tests.discretization_pversion_benchmark
    FRACTUREX_PVER_NX=32 FRACTUREX_PVER_PDISP=1,2,3 python -m fracturex.tests.discretization_pversion_benchmark
Results -> results/benchmarks/pversion/discretization_pversion.json + stdout table.
"""

from __future__ import annotations

import os
import json
import time

import numpy as np

from fealpy.mesh import TriangleMesh
from fealpy.fem import BilinearForm, LinearElasticIntegrator

from fracturex.phasefield.main_solve import MainSolve
from fracturex.interior_penalty.ipfem_phasefield_solver import IPFEMPhaseFieldSolver


# ---- material (matches the HZ benchmark values) -------------------------------

E, NU, GC, L0 = 200.0, 0.2, 1.0, 0.02
MU = E / (2.0 * (1.0 + NU))
LAM = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MATERIAL = {"E": E, "nu": NU, "Gc": GC, "l0": L0, "lam": LAM, "mu": MU}

WARMUP_REPEAT = 3


def _shared_mesh(nx: int):
    return TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=nx, ny=nx)


def _time_call(fn, *, warmup_repeat: int) -> dict:
    """Time one cold call (builds caches) + `warmup_repeat` warm calls."""
    t0 = time.perf_counter()
    fn()
    cold_s = float(time.perf_counter() - t0)
    warm: list[float] = []
    for _ in range(max(warmup_repeat, 1)):
        t1 = time.perf_counter()
        fn()
        warm.append(float(time.perf_counter() - t1))
    warm_s = float(np.median(warm))
    return {
        "cold_s": cold_s,
        "warm_s": warm_s,
        "cache_speedup": (cold_s / warm_s) if warm_s > 0 else float("nan"),
    }


# ---- standard Lagrange --------------------------------------------------------

def _standard_elastic_timer(mesh, p_disp: int):
    """Build MainSolve at p_disp; return (assemble_fn, tensor_gdof)."""
    ms = MainSolve(mesh=mesh, material_params=MATERIAL, model_type="HybridModel")
    ms._method = "lfem"                # normally set inside MainSolve.solve()
    ms.initialize_settings(p=p_disp)   # builds pfcm, spaces, uh=0, d=0

    def _assemble():
        # identical to MainSolve.solve_displacement()'s assembly (main_solve.py L330-332)
        ubform = BilinearForm(ms.tspace)
        ubform.add_integrator(LinearElasticIntegrator(ms.pfcm, q=ms.q, method="voigt"))
        return ubform.assembly()

    return _assemble, int(ms.tspace.number_of_global_dofs())


# ---- C0 interior-penalty ------------------------------------------------------

def _ip_solver(mesh, p_disp: int, p_phase: int) -> IPFEMPhaseFieldSolver:
    return IPFEMPhaseFieldSolver(
        mesh, MATERIAL, p_disp=p_disp, p_phase=p_phase,
        gamma=5.0, model_type="HybridModel",
    )


def _ip_elastic_timer(mesh, p_disp: int):
    solver = _ip_solver(mesh, p_disp, p_phase=max(p_disp, 2))

    def _assemble():
        # identical to IPFEMPhaseFieldSolver.newton_raphson displacement block (L335-339)
        ubform = BilinearForm(solver.tspace)
        ubform.add_integrator(LinearElasticIntegrator(solver.pfcm, q=solver.q, method="voigt"))
        return ubform.assembly()

    return _assemble, int(solver.tspace.number_of_global_dofs())


def _ip_phase4th_timer(mesh, p_phase: int):
    solver = _ip_solver(mesh, p_disp=1, p_phase=p_phase)
    # History field is 0 at u=0 (newton_raphson sets it from pfcm before assembling);
    # the biharmonic+IP 4th-order cost we time here is H-independent anyway.
    solver.H = 0.0
    # biharmonic + interior-penalty phase LHS = the 4th-order distinctive kernel
    return solver._assemble_phase_lhs, int(solver.dspace.number_of_global_dofs())


# ---- axes ---------------------------------------------------------------------

def axis_standard_elastic(mesh, p_list: list[int]) -> list[dict]:
    rows = []
    for p in p_list:
        fn, gdof = _standard_elastic_timer(mesh, p)
        t = _time_call(fn, warmup_repeat=WARMUP_REPEAT)
        rows.append({"kernel": "standard-elastic", "p": p, "gdof": gdof, **t})
    return rows


def axis_ip_elastic(mesh, p_list: list[int]) -> list[dict]:
    rows = []
    for p in p_list:
        fn, gdof = _ip_elastic_timer(mesh, p)
        t = _time_call(fn, warmup_repeat=WARMUP_REPEAT)
        rows.append({"kernel": "ip-elastic", "p": p, "gdof": gdof, **t})
    return rows


def axis_ip_phase4th(mesh, p_list: list[int]) -> list[dict]:
    rows = []
    for p in p_list:
        fn, gdof = _ip_phase4th_timer(mesh, p)
        t = _time_call(fn, warmup_repeat=WARMUP_REPEAT)
        rows.append({"kernel": "ip-phase4th", "p": p, "gdof": gdof, **t})
    return rows


def _env_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return [int(x) for x in raw.split(",") if x.strip()]


def main() -> None:
    nx = int(os.environ.get("FRACTUREX_PVER_NX", "24"))
    # p_phase >= 2 for the biharmonic/IP space; p_disp can start at 1.
    pdisp_list = _env_int_list("FRACTUREX_PVER_PDISP", [1, 2, 3])
    pphase_list = _env_int_list("FRACTUREX_PVER_PPHASE", [2, 3, 4])
    kernels = os.environ.get("FRACTUREX_PVER_KERNELS", "standard-elastic,ip-elastic,ip-phase4th")
    kernels = {k.strip() for k in kernels.split(",") if k.strip()}

    mesh = _shared_mesh(nx)
    ncells = int(mesh.number_of_cells())
    print("===== three-discretization p-version assembly benchmark =====")
    print(f"shared mesh: unit square nx=ny={nx}  ncells={ncells}")
    print(f"kernels={sorted(kernels)}  p_disp={pdisp_list}  p_phase={pphase_list}")
    print(f"warmup_repeat={WARMUP_REPEAT}")

    rows: list[dict] = []
    if "standard-elastic" in kernels:
        print("\n-- standard Lagrange elastic stiffness (vs p_disp) --")
        rows += axis_standard_elastic(mesh, pdisp_list)
    if "ip-elastic" in kernels:
        print("\n-- C0-IP elastic stiffness (vs p_disp) --")
        rows += axis_ip_elastic(mesh, pdisp_list)
    if "ip-phase4th" in kernels:
        print("\n-- C0-IP 4th-order phase (biharmonic + interior penalty, vs p_phase) --")
        rows += axis_ip_phase4th(mesh, pphase_list)

    print("\n===== results =====")
    hdr = f"{'kernel':<20}{'p':>4}{'gdof':>10}{'cold_s':>10}{'warm_s':>10}{'cache_x':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['kernel']:<20}{r['p']:>4}{r['gdof']:>10}"
              f"{r['cold_s']:>10.3f}{r['warm_s']:>10.3f}{r['cache_speedup']:>9.2f}")

    outdir = os.path.join("results", "benchmarks", "pversion")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "discretization_pversion.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump({
            "description": "p-version assembly cost, standard Lagrange + C0-IP, shared unit-square mesh",
            "mesh": {"unit_square_nx": nx, "ncells": ncells},
            "material": MATERIAL,
            "warmup_repeat": WARMUP_REPEAT,
            "rows": rows,
        }, f, indent=2)
    print(f"\nWrote {outpath}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
