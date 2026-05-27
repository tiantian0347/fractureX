#!/usr/bin/env python3
"""
Hu-Zhang staggered phase-field paper runs (server batch).

Production (``--mode direct``, default for ``run_main.sh``):
  - parallel assembly
  - elastic: sparse direct (``spsolve`` / ``pardiso`` / ``mumps`` via env)
  - phase: unpreconditioned GMRES

Aux-space test (``--mode aux``, model0 recommended):
  - parallel assembly
  - elastic: aux-space preconditioned GMRES
  - phase: unpreconditioned GMRES

Baseline (``--mode baseline``):
  - serial assembly
  - elastic: direct (one load step only)
  - phase: unpreconditioned GMRES

Outputs under ``<root>/phasefield/<case>/<run_label>/epsg_<tag>/`` (same layout as tests):
  history.csv, iterations.csv, meta.json, summary.*, checkpoints/, vtk/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scipy.sparse.linalg import gmres as scipy_gmres
from scipy.sparse.linalg import spsolve as scipy_spsolve

from fealpy.mesh import TriangleMesh

from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
from fracturex.cases.model2_notch_shear import Model2NotchXStretchCase
from fracturex.cases.square_tension_precrack import SquareTensionPreCrackCase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.drivers.huzhang_phasefield_staggered import HuZhangPhaseFieldStaggeredDriver
from fracturex.postprocess.recorder import RunRecorder
from fracturex.postprocess.run_paths import epsg_tag, phasefield_tag_dir, vtk_dir
from fracturex.postprocess.run_report import (
    export_paper_summary,
    export_residual_force_displacement_curve,
)
from fracturex.utilfuc.linear_solvers import (
    KrylovInfo,
    _extract_converged_from_info,
    solve_huzhang_block_gmres_auxspace,
    solve_huzhang_block_gmres_fast,
)
from fracturex.utilfuc.phasefield_mesh import (
    mesh_h_stats as _mesh_h_stats,
    phasefield_h_target as _h_target,
    resolve_box_nx as _resolve_box_nx,
    resolve_model0_distmesh_hmin as _resolve_model0_hmin,
)


EPS_G = 1e-6
ELASTIC_FORMULATION = "standard"


def _run_short_smoke() -> bool:
    return os.environ.get("FRACTUREX_RUN_SHORT", "0") == "1"


def _normalize_mode(mode: str) -> str:
    m = mode.lower()
    if m == "main":
        # Legacy alias: production path is now direct elastic.
        return "direct"
    return m


def _parallel_assembly(mode: str) -> bool:
    return _normalize_mode(mode) in ("direct", "aux")


def _use_elastic_fast(mode: str) -> bool:
    if _normalize_mode(mode) != "aux":
        return False
    raw = os.environ.get("FRACTUREX_ELASTIC_FAST", "").strip()
    if raw == "1":
        return True
    if raw == "0":
        return False
    return _run_short_smoke()


def _elastic_direct_backend() -> str:
    raw = os.environ.get("FRACTUREX_ELASTIC_DIRECT_BACKEND", "spsolve").strip().lower()
    if raw in ("spsolve", "direct", "superlu", "pardiso", "mumps"):
        return raw
    return "spsolve"


def _run_label_for_mode(mode: str) -> str:
    mode = _normalize_mode(mode)
    base = {
        "baseline": "paper_baseline",
        "direct": "paper_direct",
        "aux": "paper_aux",
    }[mode]
    suffix = os.environ.get("FRACTUREX_RUN_LABEL_SUFFIX", "").strip()
    if suffix:
        return f"{base}_{suffix}"
    return base


def _assembly_nproc() -> int:
    raw = os.environ.get("FRACTUREX_ASSEMBLY_NPROC", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, int(os.cpu_count() or 1))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _aux_gmres_settings() -> dict:
    """GMRES parameters for the aux-space / fast elastic solver, overridable
    via FRACTUREX_GMRES_{RTOL,ATOL,RESTART,MAXIT}. Used for rtol/maxit tuning
    experiments without touching the rest of the pipeline.
    """
    return {
        "rtol": _env_float("FRACTUREX_GMRES_RTOL", 1e-8),
        "atol": _env_float("FRACTUREX_GMRES_ATOL", 1e-12),
        "restart": _env_int("FRACTUREX_GMRES_RESTART", 60),
        "maxit": _env_int("FRACTUREX_GMRES_MAXIT", 200),
    }


def _phase_gmres(
    A,
    F,
    *,
    restart: int = 80,
    maxiter: int = 3000,
    atol: float = 1e-20,
    rtol: float = 1e-10,
    check_rtol: float = 1e-8,
    fallback_to_spsolve: bool = True,
):
    if hasattr(A, "to_scipy"):
        a_ = A.to_scipy().tocsr()
    else:
        a_ = A.tocsr() if hasattr(A, "tocsr") else A
    f_ = np.asarray(F, dtype=float).reshape(-1)
    residuals: list[float] = []

    def _cb(rk):
        residuals.append(float(rk))

    x, info = scipy_gmres(
        a_,
        f_,
        restart=int(restart),
        maxiter=int(maxiter),
        atol=atol,
        rtol=rtol,
        callback=_cb,
        callback_type="pr_norm",
    )
    x = np.asarray(x, dtype=float).reshape(-1)
    bnorm = max(float(np.linalg.norm(f_)), 1e-30)
    rrel = float(np.linalg.norm(a_ @ x - f_) / bnorm)
    bad = (info != 0) or (not np.isfinite(x).all()) or (not np.isfinite(rrel)) or (rrel > check_rtol)
    niter = int(len(residuals)) if residuals else 0
    if bad and fallback_to_spsolve:
        x = scipy_spsolve(a_, f_)
        return x, KrylovInfo(
            solver="spsolve-phase-fallback",
            niter=1,
            converged=True,
            residual_norm=0.0,
            atol=atol,
            rtol=rtol,
        )
    return x, KrylovInfo(
        solver="gmres-phase",
        niter=max(niter, 1),
        converged=bool(_extract_converged_from_info(info)),
        residual_norm=float(residuals[-1]) if residuals else float("nan"),
        atol=atol,
        rtol=rtol,
    )


@dataclass
class Model0Material:
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


@dataclass
class SquareMaterial:
    E: float = 210.0
    nu: float = 0.3
    Gc: float = 2.7e-3
    l0: float = 0.015
    ft: float = 3.0

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self) -> float:
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


@dataclass
class Model2Material:
    E: float = 210.0
    nu: float = 0.3
    Gc: float = 2.7e-3
    l0: float = 1.33e-2

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self) -> float:
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


def _model0_loads() -> List[float]:
    return np.concatenate(
        [
            np.linspace(0.0, 70e-3, 6, dtype=float),
            np.linspace(70e-3, 125e-3, 26, dtype=float)[1:],
        ]
    ).tolist()


def _square_loads() -> List[float]:
    return np.concatenate(
        [
            np.linspace(0.0, 5e-3, 51, dtype=float),
            np.linspace(5e-3, 6.1e-3, 111, dtype=float)[1:],
        ]
    ).tolist()


def _baseline_one_load(loads: List[float]) -> List[float]:
    if not loads:
        return [0.0]
    raw = os.environ.get("FRACTUREX_BASELINE_LOAD_INDEX", "").strip()
    if raw:
        k = max(0, min(int(raw), len(loads) - 1))
    else:
        k = 1 if len(loads) > 1 and float(loads[0]) == 0.0 else 0
    return [float(loads[k])]


def _truncate_loads(loads: List[float]) -> List[float]:
    if os.environ.get("FRACTUREX_RUN_SHORT", "0") == "1":
        return loads[:3]
    raw = os.environ.get("FRACTUREX_RUN_NSTEPS", "").strip()
    if raw:
        n = max(1, int(raw))
        if loads and float(loads[0]) == 0.0:
            return loads[: n + 1]
        return loads[:n]
    return loads


def _vtu_step_set(n_steps: int) -> set[int]:
    raw = os.environ.get("FRACTUREX_VTU_STEPS", "").strip()
    if raw:
        return {int(x) for x in raw.split(",") if x.strip()}
    every = int(os.environ.get("FRACTUREX_VTU_EVERY", "0"))
    if every <= 0:
        every = max(1, n_steps // 20) if n_steps > 40 else 1
    steps = set(range(0, n_steps, every))
    steps.add(n_steps - 1)
    return steps


def _write_run_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _build_case(case_id: str):
    case_id = case_id.lower()
    if case_id == "model0":
        mat = Model0Material()
        hmin_env = os.environ.get("FRACTUREX_HMIN", "").strip()
        if hmin_env:
            hmin = float(hmin_env)
            mesh_info = {"hmin": hmin, "mesh_note": "FRACTUREX_HMIN override"}
        elif _run_short_smoke() and os.environ.get("FRACTUREX_FAST_COARSE_MESH", "1") == "1":
            hmin = 0.05
            mesh_info = {
                "hmin": hmin,
                "mesh_note": "RUN_SHORT smoke: coarse hmin (set FRACTUREX_FAST_COARSE_MESH=0 for paper mesh)",
            }
        else:
            hmin, mesh_info = _resolve_model0_hmin(
                mat.l0,
                lambda h: Model0CircularNotchCase(
                    _model=mat, hmin=h, debug_mesh=False
                ).make_mesh(),
                hmin0=None,
            )
        case = Model0CircularNotchCase(_model=mat, hmin=hmin, debug_mesh=False)
        case.output_enabled = False
        mesh = case.make_mesh()
        loads = _truncate_loads(_model0_loads())
        mesh_param = {"type": "distmesh", "hmin": hmin, **mesh_info}
    elif case_id in ("square", "square_tension", "model1"):
        mat = SquareMaterial()
        nx_env = os.environ.get("FRACTUREX_NX", "").strip()
        if nx_env:
            nx = int(nx_env)
            mesh_info = {"nx": nx, "ny": nx}
        else:
            nx, mesh_info = _resolve_box_nx(mat.l0)
        case = SquareTensionPreCrackCase(_model=mat, nx=nx, ny=nx, debug_mesh=False)
        case.output_enabled = False
        mesh = case.make_mesh()
        loads = _truncate_loads(_square_loads())
        mesh_param = {"type": "box", **mesh_info}
    elif case_id == "model2":
        mat = Model2Material()
        nx_env = os.environ.get("FRACTUREX_NX", "").strip()
        if nx_env:
            nx = ny = int(nx_env)
            mesh_info = {"nx": nx, "ny": ny}
        else:
            nx, mesh_info = _resolve_box_nx(mat.l0)
            ny = nx
        case = Model2NotchXStretchCase(_model=mat, nx=nx, ny=ny, debug_mesh=False)
        case.output_enabled = False
        mesh = case.make_mesh()
        loads = _truncate_loads(np.asarray(case.default_loads(), dtype=float).tolist())
        mesh_param = {"type": "box", **mesh_info}
    else:
        raise ValueError(f"Unknown case {case_id!r}; use model0 | square | model2")

    h_stats = _mesh_h_stats(mesh)
    mesh_param.update(h_stats)
    mesh_param["h_target"] = _h_target(mat.l0)
    mesh_param["l0"] = float(mat.l0)
    mesh_param["h_ok"] = bool(h_stats["h_max"] < mesh_param["h_target"])

    return case, mesh, mat, loads, mesh_param


def _build_driver(
    *,
    case,
    mesh,
    mode: str,
    run_dir: Path,
    save_npz: bool,
    save_every: int,
) -> Tuple[HuZhangPhaseFieldStaggeredDriver, HuZhangDiscretization, PhaseFieldDamageModel]:
    mode = _normalize_mode(mode)
    assembly_parallel = _parallel_assembly(mode)
    d_relaxation = HuZhangPhaseFieldStaggeredDriver._resolve_d_relaxation(None)

    discr = HuZhangDiscretization(
        case=case,
        p=3,
        damage_p=2,
        use_relaxation=True,
    ).build(mesh=mesh)

    damage = PhaseFieldDamageModel(
        density_type="AT2",
        degradation_type="quadratic",
        split="hybrid",
        eps_g=float(EPS_G),
        debug=False,
    )

    elastic_assembler = HuZhangElasticAssembler(
        discr,
        case,
        damage,
        formulation=ELASTIC_FORMULATION,
        assembly_parallel=assembly_parallel,
    )
    phase_assembler = PhaseFieldAssembler(
        discr,
        case,
        damage,
        debug=False,
        assembly_parallel=assembly_parallel,
    )

    recorder = RunRecorder(
        str(run_dir),
        save_npz=save_npz,
        save_every=max(1, int(save_every)),
    )

    if mode == "baseline":
        backend = _elastic_direct_backend()
        elastic_solver = HuZhangPhaseFieldStaggeredDriver.linear_solver(backend)
        solver_mode_tag = (
            f"{ELASTIC_FORMULATION}:baseline_elastic_{backend}_serial_asm_1step/phase_gmres"
        )
    elif mode == "direct":
        backend = _elastic_direct_backend()
        elastic_solver = HuZhangPhaseFieldStaggeredDriver.linear_solver(backend)
        solver_mode_tag = (
            f"{ELASTIC_FORMULATION}:elastic_{backend}_parallel/phase_gmres_no_precond"
        )
    elif _use_elastic_fast(mode):

        _gmres_kw = _aux_gmres_settings()

        def elastic_solver(A, F):
            return solve_huzhang_block_gmres_fast(
                A,
                F,
                gdof_sigma=discr.gdof_sigma,
                vspace=discr.space_u,
                atol=_gmres_kw["atol"],
                rtol=_gmres_kw["rtol"],
                restart=_gmres_kw["restart"],
                maxit=_gmres_kw["maxit"],
                q=3,
                weighted_aux=True,
                elastic_formulation=ELASTIC_FORMULATION,
                damage=damage,
                state=discr.state,
            )

        solver_mode_tag = (
            f"{ELASTIC_FORMULATION}:elastic_fast_schur_gmres_parallel/phase_gmres_no_precond"
        )
    else:

        _gmres_kw = _aux_gmres_settings()

        def elastic_solver(A, F):
            return solve_huzhang_block_gmres_auxspace(
                A,
                F,
                gdof_sigma=discr.gdof_sigma,
                vspace=discr.space_u,
                atol=_gmres_kw["atol"],
                rtol=_gmres_kw["rtol"],
                restart=_gmres_kw["restart"],
                maxit=_gmres_kw["maxit"],
                sstep=3,
                theta=0.25,
                q=3,
                schur_rebuild_interval=5,
                coarse_rebuild_interval=5,
                weighted_aux=True,
                elastic_formulation=ELASTIC_FORMULATION,
                damage=damage,
                state=discr.state,
                schur_ilu_in_precond=False,
            )

        solver_mode_tag = f"{ELASTIC_FORMULATION}:elastic_auxspace_gmres_parallel/phase_gmres_no_precond"

    driver = HuZhangPhaseFieldStaggeredDriver(
        case=case,
        discr=discr,
        damage=damage,
        elastic_assembler=elastic_assembler,
        phase_assembler=phase_assembler,
        tol=1e-5,
        maxit=500,
        d_relaxation=d_relaxation,
        elastic_solver=elastic_solver,
        phase_solver=_phase_gmres,
        compute_linear_residual=True,
        debug=False,
        timing=True,
        recorder=recorder,
        output_dir=str(run_dir),
    )
    driver._solver_mode_tag = solver_mode_tag  # type: ignore[attr-defined]
    return driver, discr, damage


def _run_with_vtu(
    driver: HuZhangPhaseFieldStaggeredDriver,
    case,
    discr: HuZhangDiscretization,
    loads: List[float],
    vtu_dir: Path,
    vtu_steps: set[int],
) -> Tuple[list, List[str]]:
    """Run load history once; write VTU only on selected steps."""
    from fracturex.drivers.huzhang_phasefield_staggered import StepInfo

    vtu_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    q = discr.damage_p + 3
    driver.initialize()
    if driver.recorder is not None:
        m = case.model()
        driver.recorder.write_meta(
            dict(
                case=case.name,
                p=int(discr.p),
                use_relaxation=bool(discr.use_relaxation),
                elastic_formulation=ELASTIC_FORMULATION,
                history_source=str(getattr(driver.damage, "history_source", "from_u")),
                mesh=dict(
                    NN=int(discr.mesh.number_of_nodes()),
                    NE=int(discr.mesh.number_of_edges()),
                    NC=int(discr.mesh.number_of_cells()),
                ),
                material={
                    k: float(getattr(m, k))
                    for k in ["lam", "mu", "E", "nu", "Gc", "l0", "ft"]
                    if hasattr(m, k)
                },
                gdof={
                    "sigma": int(discr.gdof_sigma),
                    "u": int(discr.gdof_u),
                    "d": int(discr.space_d.number_of_global_dofs()),
                },
                timing_enabled=bool(driver.timing),
                compute_linear_residual=bool(driver.compute_linear_residual),
            )
        )
    infos: List[StepInfo] = []
    for s, load in enumerate(loads):
        info = driver.solve_one_step(step=int(s), load=float(load))
        infos.append(info)
        if int(s) in vtu_steps:
            fname = vtu_dir / f"step_{s:04d}_load_{float(load):.6e}.vtu"
            driver._save_vtkfile(str(fname), cell_mode="mean", q=q)
            written.append(str(fname))
    return infos, written


def run(case_id: str, mode: str, out_root: Path) -> Path:
    mode = _normalize_mode(mode.lower())
    if mode not in ("direct", "aux", "baseline"):
        raise ValueError(f"mode must be direct, aux, or baseline, got {mode!r}")

    case, mesh, mat, loads_full, mesh_param = _build_case(case_id)
    loads = _baseline_one_load(loads_full) if mode == "baseline" else loads_full

    run_label = _run_label_for_mode(mode)
    tag_dir = phasefield_tag_dir(
        case.name,
        run_label,
        eps_g=EPS_G,
        root=str(out_root),
        mkdir=True,
    )
    run_parent = Path(tag_dir).parent
    vtu_path = Path(vtk_dir(tag_dir))

    save_npz = os.environ.get("FRACTUREX_SAVE_NPZ", "1" if mode != "baseline" else "0") == "1"
    save_every = int(os.environ.get("FRACTUREX_SAVE_EVERY", "10" if mode != "baseline" else "1"))

    driver, discr, damage = _build_driver(
        case=case,
        mesh=mesh,
        mode=mode,
        run_dir=Path(tag_dir),
        save_npz=save_npz,
        save_every=save_every,
    )

    gdof_block = int(discr.gdof_sigma + discr.gdof_u)
    gdof_total = int(gdof_block + discr.space_d.number_of_global_dofs())
    manifest = {
        "case": case_id,
        "mode": mode,
        "solver_mode": getattr(driver, "_solver_mode_tag", mode),
        "assembly_parallel": _parallel_assembly(mode),
        "elastic_direct_backend": _elastic_direct_backend() if mode in ("direct", "baseline") else None,
        "elastic_formulation": ELASTIC_FORMULATION,
        "eps_g": EPS_G,
        "mesh": mesh_param,
        "gdof": {
            "sigma": int(discr.gdof_sigma),
            "u": int(discr.gdof_u),
            "d": int(discr.space_d.number_of_global_dofs()),
            "elastic_block": gdof_block,
            "total": gdof_total,
        },
        "n_load_steps": len(loads),
        "loads_preview": loads[:5] + (["..."] if len(loads) > 5 else []),
    }
    _write_run_manifest(run_parent / "run_manifest.json", manifest)

    print(f"\n===== paper_huzhang {case_id} / {mode} =====")
    if mode in ("direct", "aux"):
        print(
            f"parallel: assembly_parallel=True, FRACTUREX_ASSEMBLY_NPROC={_assembly_nproc()}, "
            f"BLAS/OpenMP threads=1 (see env.sh); mode={mode}"
        )
        if mode == "direct":
            print(f"elastic: direct ({_elastic_direct_backend()})")
        else:
            print(f"elastic: aux-space GMRES (fast={_use_elastic_fast(mode)})")
    print(f"mesh: {mesh_param}")
    print(f"gdof: sigma={discr.gdof_sigma}, u={discr.gdof_u}, d={discr.space_d.number_of_global_dofs()}, total={gdof_total}")
    print(f"h_ok (h_max < {mesh_param['h_target']:.6e}): {mesh_param['h_ok']}")
    print(f"load steps: {len(loads)}")
    print(f"output: {tag_dir}")

    t0 = time.perf_counter()
    infos, vtu_paths = _run_with_vtu(
        driver,
        case,
        discr,
        loads,
        vtu_path,
        _vtu_step_set(len(loads)),
    )
    wall_s = float(time.perf_counter() - t0)

    solver_mode = getattr(driver, "_solver_mode_tag", mode)
    export_paper_summary(
        infos,
        outdir=tag_dir,
        total_wall_s=wall_s,
        solver_mode=solver_mode,
        history_source=str(getattr(damage, "history_source", "from_u")),
    )
    export_residual_force_displacement_curve(infos, outdir=tag_dir)

    if mode == "baseline":
        snap = {
            "description": "One load step: serial assembly + elastic spsolve vs main iterative path",
            "baseline_load_value": float(loads[0]),
            "full_run_n_load_steps": len(loads_full),
            "wall_s": wall_s,
            "reaction": [float(i.meta.get("residual_force", 0.0)) for i in infos],
            "final_max_d": float(np.max(np.asarray(discr.state.d[:]))),
            "gdof": manifest["gdof"],
            "mesh": mesh_param,
        }
        with open(run_parent / "baseline_reference.json", "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2, ensure_ascii=False)

    manifest["wall_s"] = wall_s
    manifest["vtu_files"] = vtu_paths
    manifest["final_max_d"] = float(np.max(np.asarray(discr.state.d[:])))
    _write_run_manifest(run_parent / "run_manifest.json", manifest)

    print(f"done in {wall_s:.2f}s -> {tag_dir}")
    return Path(tag_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hu-Zhang phase-field paper batch runner")
    parser.add_argument(
        "--case",
        required=True,
        choices=["model0", "square", "model2"],
        help="Benchmark case id",
    )
    parser.add_argument(
        "--mode",
        default="direct",
        choices=["direct", "aux", "main", "baseline"],
        help="direct=elastic sparse direct+parallel; aux=aux-space GMRES; baseline=1-step serial",
    )
    parser.add_argument(
        "--out-root",
        default=os.environ.get(
            "FRACTUREX_PAPER_ROOT",
            os.environ.get("FRACTUREX_RESULTS_ROOT", "results"),
        ),
        help="Results root (same as tests: <root>/phasefield/<case>/...)",
    )
    args = parser.parse_args()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    run(args.case, args.mode, out_root)


if __name__ == "__main__":
    main()
