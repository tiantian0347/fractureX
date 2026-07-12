"""Hu-Zhang assembly scaling benchmark (T9 / framework paper, pillar 1).

Isolated **assembly** timing driver — does NOT run the staggered solve. It builds
the real case/discretization/damage/assembler stack exactly like
``tests/phasefield_model0_huzhang.py::_build_driver`` (up to the assembler) and
times ``HuZhangElasticAssembler.assemble(load)`` directly, so the wall clock is
pure assembly, not linear-solver time.

Axes (see T9_FRAMEWORK_PAPER_PLAN.md §3):
  1. strong scaling   — fixed mesh, wall vs assembly_nproc (1 -> N threads)
  2. size / DOF        — wall vs mesh DOF (sweep distmesh hmin)
  3. p-version         — wall vs polynomial order p (HZ stress space)
  4. cache cold/warm   — first assemble (cold, builds d-independent kernel cache)
                          vs subsequent assembles (warm, reuse cache)
  5. formulation       — standard vs effective_stress branch cost

Scope: this covers the HZ pillar only (the mature core + paper headline figure).
The three-discretization p-version sweep (standard Lagrange / C0-IP) needs each
solver's own API and lives in a separate follow-up script.

Run (from repo root, venv_fealpy3 active):
    python -m fracturex.tests.hz_assembly_scaling_benchmark
    FRACTUREX_BENCH_AXES=strong,pver python -m fracturex.tests.hz_assembly_scaling_benchmark
Results -> results/benchmarks/hz_assembly/<timestamp-free>.json + stdout table.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import json
import time

import numpy as np

from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.damage.base import DamageStateView
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler


@dataclass
class Model0Material:
    """Model-0 phase-field material (mirrors the harness values)."""

    E: float = 200.0
    nu: float = 0.2
    Gc: float = 1.0
    l0: float = 0.02

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self) -> float:
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


# ---- benchmark tunables (env-overridable) -------------------------------------

BENCH_LOAD = 1.0e-3          # representative scalar load fed to assemble()
WARMUP_REPEAT = 3            # warm assembles timed after the cold one
DEFAULT_HMIN = 0.02          # matches harness production mesh


def _build_hz_assembler(
    *,
    hmin: float,
    p: int,
    damage_p: int,
    formulation: str,
    assembly_parallel: bool,
    assembly_nproc: int | None,
):
    """Build the HZ elastic assembler + its mesh DOF count, no driver/solver.

    Returns (assembler, gdof_total, ncells).
    """
    mat = Model0Material()
    case = Model0CircularNotchCase(
        _model=mat,
        hmin=hmin,
        distmesh_maxit=100,
        debug_mesh=False,
    )
    mesh = case.make_mesh()

    discr = HuZhangDiscretization(
        case=case,
        p=p,
        damage_p=damage_p,
        use_relaxation=True,
    ).build(mesh=mesh)

    damage = PhaseFieldDamageModel(
        density_type="AT2",
        degradation_type="quadratic",
        split="hybrid",
        eps_g=1e-6,
        debug=False,
    )

    assembler = HuZhangElasticAssembler(
        discr,
        case,
        damage,
        formulation=formulation,
        assembly_parallel=assembly_parallel,
        assembly_nproc=assembly_nproc,
    )
    # Driver normally does this once (huzhang_phasefield_staggered.initialize):
    # binds Gc/l0/Lame + degradation function and clips the initial damage field.
    state = discr.state
    view = DamageStateView(
        d=state.d, sigma=state.sigma, u=state.u, r_hist=state.r_hist, H=state.H,
    )
    damage.on_build(discr, view, case)

    gdof_sigma = int(discr.space_sigma.number_of_global_dofs())
    gdof_u = int(discr.space_u.number_of_global_dofs())
    ncells = int(mesh.number_of_cells())
    return assembler, gdof_sigma + gdof_u, ncells


def _time_assemble(assembler, *, load: float, warmup_repeat: int) -> dict:
    """Time one cold assemble + `warmup_repeat` warm assembles.

    Cold call populates the d-independent geometric kernel cache; warm calls
    reuse it (that speedup is axis 4).
    """
    t0 = time.perf_counter()
    assembler.assemble(load)
    cold_s = float(time.perf_counter() - t0)

    warm_times: list[float] = []
    for _ in range(max(warmup_repeat, 1)):
        t1 = time.perf_counter()
        assembler.assemble(load)
        warm_times.append(float(time.perf_counter() - t1))
    warm_s = float(np.median(warm_times))
    return {
        "cold_s": cold_s,
        "warm_s": warm_s,
        "warm_times_s": warm_times,
        "cache_speedup": (cold_s / warm_s) if warm_s > 0 else float("nan"),
    }


# ---- axes ---------------------------------------------------------------------

def axis_strong_scaling(hmin: float, p: int, nproc_list: list[int]) -> list[dict]:
    """Fixed mesh, sweep assembly_nproc. Serial baseline first (parallel=False)."""
    rows: list[dict] = []
    # serial reference
    asm, gdof, nc = _build_hz_assembler(
        hmin=hmin, p=p, damage_p=2, formulation="standard",
        assembly_parallel=False, assembly_nproc=1,
    )
    t = _time_assemble(asm, load=BENCH_LOAD, warmup_repeat=WARMUP_REPEAT)
    serial_warm = t["warm_s"]
    rows.append({"axis": "strong", "nproc": 1, "parallel": False,
                 "gdof": gdof, "ncells": nc, **t, "speedup_vs_serial": 1.0})
    for nproc in nproc_list:
        if nproc <= 1:
            continue
        asm, gdof, nc = _build_hz_assembler(
            hmin=hmin, p=p, damage_p=2, formulation="standard",
            assembly_parallel=True, assembly_nproc=nproc,
        )
        t = _time_assemble(asm, load=BENCH_LOAD, warmup_repeat=WARMUP_REPEAT)
        rows.append({"axis": "strong", "nproc": nproc, "parallel": True,
                     "gdof": gdof, "ncells": nc, **t,
                     "speedup_vs_serial": (serial_warm / t["warm_s"]) if t["warm_s"] > 0 else float("nan")})
    return rows


def axis_size(hmin_list: list[float], p: int, nproc: int) -> list[dict]:
    """Sweep mesh resolution (hmin). Wall vs DOF."""
    rows: list[dict] = []
    for hmin in hmin_list:
        asm, gdof, nc = _build_hz_assembler(
            hmin=hmin, p=p, damage_p=2, formulation="standard",
            assembly_parallel=(nproc > 1), assembly_nproc=nproc,
        )
        t = _time_assemble(asm, load=BENCH_LOAD, warmup_repeat=WARMUP_REPEAT)
        rows.append({"axis": "size", "hmin": hmin, "nproc": nproc,
                     "gdof": gdof, "ncells": nc, **t})
    return rows


def axis_pversion(hmin: float, p_list: list[int], nproc: int) -> list[dict]:
    """Sweep HZ stress-space polynomial order p. Wall vs p."""
    rows: list[dict] = []
    for p in p_list:
        asm, gdof, nc = _build_hz_assembler(
            hmin=hmin, p=p, damage_p=2, formulation="standard",
            assembly_parallel=(nproc > 1), assembly_nproc=nproc,
        )
        t = _time_assemble(asm, load=BENCH_LOAD, warmup_repeat=WARMUP_REPEAT)
        rows.append({"axis": "pver", "p": p, "nproc": nproc,
                     "gdof": gdof, "ncells": nc, **t})
    return rows


def axis_formulation(hmin: float, p: int, nproc: int) -> list[dict]:
    """standard vs effective_stress assembly cost, same mesh/p."""
    rows: list[dict] = []
    for formulation in ("standard", "effective_stress"):
        asm, gdof, nc = _build_hz_assembler(
            hmin=hmin, p=p, damage_p=2, formulation=formulation,
            assembly_parallel=(nproc > 1), assembly_nproc=nproc,
        )
        t = _time_assemble(asm, load=BENCH_LOAD, warmup_repeat=WARMUP_REPEAT)
        rows.append({"axis": "formulation", "formulation": formulation,
                     "p": p, "nproc": nproc, "gdof": gdof, "ncells": nc, **t})
    return rows


def _env_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return [int(x) for x in raw.split(",") if x.strip()]


def _env_float_list(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return [float(x) for x in raw.split(",") if x.strip()]


def main() -> None:
    axes = os.environ.get("FRACTUREX_BENCH_AXES", "strong,size,pver,formulation")
    axes = {a.strip() for a in axes.split(",") if a.strip()}

    base_hmin = float(os.environ.get("FRACTUREX_BENCH_HMIN", str(DEFAULT_HMIN)))
    base_p = int(os.environ.get("FRACTUREX_BENCH_P", "3"))
    base_nproc = int(os.environ.get("FRACTUREX_BENCH_NPROC", str(os.cpu_count() or 4)))

    nproc_list = _env_int_list("FRACTUREX_BENCH_NPROC_LIST", [1, 2, 4, base_nproc])
    # coarse -> fine; effective_stress at p=3 is heavy, keep size sweep modest.
    hmin_list = _env_float_list("FRACTUREX_BENCH_HMIN_LIST", [0.05, 0.035, 0.02])
    # HZ symmetric element in this fealpy build is stable at p>=3 (u-space order
    # = p-1; p=2 hits a div_basis corner-dof bug). Sweep p in {3,4}.
    p_list = _env_int_list("FRACTUREX_BENCH_P_LIST", [3, 4])

    print("===== HZ assembly scaling benchmark (T9 pillar 1) =====")
    print(f"axes={sorted(axes)}  base_hmin={base_hmin}  base_p={base_p}  base_nproc={base_nproc}")
    print(f"cpu_count={os.cpu_count()}  load={BENCH_LOAD}  warmup_repeat={WARMUP_REPEAT}")

    all_rows: list[dict] = []
    if "strong" in axes:
        print("\n-- strong scaling (fixed mesh, vs nproc) --")
        all_rows += axis_strong_scaling(base_hmin, base_p, sorted(set(nproc_list)))
    if "size" in axes:
        print("\n-- size scaling (vs DOF) --")
        all_rows += axis_size(hmin_list, base_p, base_nproc)
    if "pver" in axes:
        print("\n-- p-version (vs polynomial order) --")
        all_rows += axis_pversion(base_hmin, p_list, base_nproc)
    if "formulation" in axes:
        print("\n-- formulation (standard vs effective_stress) --")
        all_rows += axis_formulation(base_hmin, base_p, base_nproc)

    # stdout table
    print("\n===== results =====")
    hdr = f"{'axis':<12}{'key':<20}{'gdof':>9}{'ncells':>9}{'cold_s':>10}{'warm_s':>10}{'cache_x':>9}{'extra':>16}"
    print(hdr)
    print("-" * len(hdr))
    for r in all_rows:
        if r["axis"] == "strong":
            key = f"nproc={r['nproc']}"
            extra = f"sp={r.get('speedup_vs_serial', float('nan')):.2f}x"
        elif r["axis"] == "size":
            key = f"hmin={r['hmin']}"
            extra = ""
        elif r["axis"] == "pver":
            key = f"p={r['p']}"
            extra = ""
        else:
            key = r.get("formulation", "")
            extra = f"p={r.get('p')}"
        print(f"{r['axis']:<12}{key:<20}{r['gdof']:>9}{r['ncells']:>9}"
              f"{r['cold_s']:>10.3f}{r['warm_s']:>10.3f}{r['cache_speedup']:>9.2f}{extra:>16}")

    outdir = os.path.join("results", "benchmarks", "hz_assembly")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "hz_assembly_scaling.json")
    payload = {
        "description": "HZ elastic assembly scaling; wall = pure assemble() time (no solve)",
        "case": "model0_circular_notch",
        "load": BENCH_LOAD,
        "warmup_repeat": WARMUP_REPEAT,
        "base": {"hmin": base_hmin, "p": base_p, "nproc": base_nproc, "cpu_count": os.cpu_count()},
        "rows": all_rows,
    }
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {outpath}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
