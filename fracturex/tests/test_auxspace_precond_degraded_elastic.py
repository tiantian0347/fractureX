"""
Elastic-only verification: auxiliary-space preconditioned GMRES vs direct solve.

Focus: standard Hu-Zhang formulation with stress block M(d) scaled by 1/g(d).
When d=1 on part of the domain, g(d) is floored to eps_g (e.g. 1e-6 or 1e-8),
so local rows of A_sigma are amplified by ~1/eps_g — the ill-conditioned regime
that appears in phase-field fracture after crack formation.

Run (repo root):
  bash scripts/run_python.sh fracturex/tests/test_auxspace_precond_degraded_elastic.py

Environment:
  FRACTUREX_HMIN=0.02          coarser mesh for faster CI (default 0.02)
  FRACTUREX_RUN_SHORT=1        fewer loads / skip 1e-8 sweep
  FRACTUREX_EPS_G_LIST=1e-6,1e-8   comma-separated eps_g values
  FRACTUREX_OUTDIR=results/tests/auxspace_degraded_elastic
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse.linalg import spsolve as scipy_spsolve

from fealpy.backend import backend_manager as bm

from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
from fracturex.damage.base import DamageStateView
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.postprocess.run_paths import epsg_tag
from fracturex.utilfuc.linear_solvers import (
    _AUXSPACE_COARSE_CACHE,
    _AUXSPACE_SCHUR_CACHE,
    solve_huzhang_block_gmres_auxspace,
)
from fracturex.utilfuc.phasefield_mesh import mesh_h_stats


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


DAMAGE_PATTERNS: Dict[str, str] = {
    "intact": "d=0 everywhere (g≈1, baseline)",
    "half_cracked": "d=1 on x>=0.5, d=0 elsewhere (mixed 1/eps_g scaling)",
    "band_cracked": "d=1 on |x-0.5|<0.05 band, d=0 elsewhere",
    "patch_cracked": "d=1 on disk center (0.75,0.5) r<0.12, d=0 elsewhere",
}


def _clear_auxspace_solve_caches() -> None:
    _AUXSPACE_COARSE_CACHE.clear()
    _AUXSPACE_SCHUR_CACHE.clear()


def _apply_damage_pattern(discr: HuZhangDiscretization, pattern: str) -> Dict[str, float]:
    mesh = discr.mesh
    state = discr.state
    assert mesh is not None and state is not None

    node = bm.asarray(mesh.node)
    x = bm.to_numpy(node[:, 0])
    y = bm.to_numpy(node[:, 1])
    darr = np.zeros_like(x, dtype=float)

    if pattern == "intact":
        pass
    elif pattern == "half_cracked":
        darr[x >= 0.5] = 1.0
    elif pattern == "band_cracked":
        darr[np.abs(x - 0.5) < 0.05] = 1.0
    elif pattern == "patch_cracked":
        darr[(x - 0.75) ** 2 + (y - 0.5) ** 2 < 0.12**2] = 1.0
    else:
        raise ValueError(f"unknown damage pattern: {pattern}")

    state.d[:] = bm.asarray(darr)
    frac_d1 = float(np.mean(darr >= 0.999))
    return {"frac_nodes_d_eq_1": frac_d1, "d_min": float(darr.min()), "d_max": float(darr.max())}


def _gd_stats(damage: PhaseFieldDamageModel, discr: HuZhangDiscretization) -> Dict[str, float]:
    mesh = discr.mesh
    state = discr.state
    assert mesh is not None and state is not None
    qf = mesh.quadrature_formula(discr.p + 3, "cell")
    bcs, _ = qf.get_quadrature_points_and_weights()
    view = DamageStateView(
        d=state.d,
        sigma=state.sigma,
        u=state.u,
        r_hist=state.r_hist,
        H=state.H,
    )
    gd = damage.coef_bary(view, bcs, index=bm.arange(mesh.number_of_cells()))
    gd_np = np.asarray(bm.to_numpy(gd), dtype=float).ravel()
    inv_gd = 1.0 / np.maximum(gd_np, 1e-30)
    return {
        "g_min": float(gd_np.min()),
        "g_max": float(gd_np.max()),
        "inv_g_max": float(inv_gd.max()),
        "inv_g_mean": float(inv_gd.mean()),
    }


def _relative_residual(A, x: np.ndarray, b: np.ndarray) -> float:
    if hasattr(A, "to_scipy"):
        A_ = A.to_scipy().tocsr()
    else:
        A_ = A.tocsr() if hasattr(A, "tocsr") else A
    b_ = np.asarray(b, dtype=float).reshape(-1)
    x_ = np.asarray(x, dtype=float).reshape(-1)
    return float(np.linalg.norm(A_ @ x_ - b_) / max(np.linalg.norm(b_), 1e-30))


def _solve_direct(A, b: np.ndarray) -> np.ndarray:
    if hasattr(A, "to_scipy"):
        A_ = A.to_scipy().tocsr()
    else:
        A_ = A.tocsr() if hasattr(A, "tocsr") else A
    return np.asarray(scipy_spsolve(A_, np.asarray(b, dtype=float).reshape(-1)), dtype=float)


def _solve_aux(
    A,
    b: np.ndarray,
    *,
    discr: HuZhangDiscretization,
    damage: PhaseFieldDamageModel,
    elastic_formulation: str = "standard",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x, stats = solve_huzhang_block_gmres_auxspace(
        A,
        b,
        gdof_sigma=discr.gdof_sigma,
        vspace=discr.space_u,
        atol=1e-12,
        rtol=1e-8,
        restart=60,
        maxit=400,
        sstep=3,
        theta=0.25,
        q=3,
        schur_rebuild_interval=1,
        coarse_rebuild_interval=1,
        weighted_aux=True,
        elastic_formulation=elastic_formulation,
        damage=damage,
        state=discr.state,
        schur_ilu_in_precond=False,
    )
    meta = {
        "solver": getattr(stats, "solver", "gmres-auxspace"),
        "niter": int(getattr(stats, "niter", -1)),
        "converged": bool(getattr(stats, "converged", False)),
        "residual_norm": float(getattr(stats, "residual_norm", float("nan"))),
    }
    return np.asarray(x, dtype=float).reshape(-1), meta


def _compare_solutions(x_ref: np.ndarray, x_test: np.ndarray) -> Dict[str, float]:
    nref = float(np.linalg.norm(x_ref))
    diff = x_test - x_ref
    nd = float(np.linalg.norm(diff))
    return {
        "norm_diff": nd,
        "rel_diff": nd / max(nref, 1e-30),
        "max_abs_diff": float(np.max(np.abs(diff))),
    }


def _build_session(
    *,
    mat: Model0Material,
    eps_g: float,
    hmin: float,
) -> Tuple[Model0CircularNotchCase, HuZhangDiscretization, PhaseFieldDamageModel, HuZhangElasticAssembler]:
    damage = PhaseFieldDamageModel(
        density_type="AT2",
        degradation_type="quadratic",
        split="hybrid",
        eps_g=float(eps_g),
        debug=False,
    )
    case = Model0CircularNotchCase(_model=mat, hmin=float(hmin))
    mesh = case.make_mesh()
    discr = HuZhangDiscretization(case, p=3, use_relaxation=True).build(mesh=mesh)
    damage.on_build(discr, discr.state, case)
    assembler = HuZhangElasticAssembler(
        discr,
        case,
        damage,
        formulation="standard",
        assembly_parallel=False,
    )
    return case, discr, damage, assembler


def _run_one_case(
    *,
    pattern: str,
    eps_g: float,
    loads: List[float],
    hmin: float,
    outdir: str,
) -> List[Dict[str, Any]]:
    mat = Model0Material()
    case, discr, damage, assembler = _build_session(mat=mat, eps_g=eps_g, hmin=hmin)
    mesh = discr.mesh
    assert mesh is not None

    _clear_auxspace_solve_caches()
    pat_info = _apply_damage_pattern(discr, pattern)
    gd_info = _gd_stats(damage, discr)

    rows: List[Dict[str, Any]] = []
    for load in loads:
        assembler.begin_load_step(float(load))
        sys = assembler.assemble(float(load))
        A, F = sys.A, np.asarray(bm.to_numpy(sys.F), dtype=float).reshape(-1)

        t0 = time.perf_counter()
        x_ref = _solve_direct(A, F)
        t_direct = time.perf_counter() - t0

        t0 = time.perf_counter()
        x_aux, aux_meta = _solve_aux(A, F, discr=discr, damage=damage)
        t_aux = time.perf_counter() - t0

        cmp = _compare_solutions(x_ref, x_aux)
        row = {
            "pattern": pattern,
            "eps_g": float(eps_g),
            "load": float(load),
            "epsg_tag": epsg_tag(eps_g),
            **pat_info,
            **gd_info,
            **cmp,
            "rel_res_direct": _relative_residual(A, x_ref, F),
            "rel_res_aux": _relative_residual(A, x_aux, F),
            "wall_s_direct": float(t_direct),
            "wall_s_aux": float(t_aux),
            **aux_meta,
            "passed": bool(
                cmp["rel_diff"] < 1e-5
                and _relative_residual(A, x_aux, F) < 1e-7
                and aux_meta.get("converged", False)
            ),
        }
        rows.append(row)
        print(
            f"[{pattern} eps_g={eps_g:.0e} load={load:.4f}] "
            f"rel_diff={row['rel_diff']:.3e} rel_res_aux={row['rel_res_aux']:.3e} "
            f"gmres_iters={row['niter']} pass={row['passed']}"
        )
    return rows


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _write_test_report(
    *,
    outdir: str,
    rows: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> str:
    os.makedirs(outdir, exist_ok=True)
    report_path = os.path.join(outdir, "TEST_REPORT.md")

    n_pass = sum(1 for r in rows if r.get("passed"))
    n_total = len(rows)
    all_pass = n_pass == n_total and n_total > 0

    lines = [
        "# Auxiliary-Space Preconditioner — Degraded Elastic Block Test",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Output directory: `{outdir}`",
        f"- Overall: **{'PASS' if all_pass else 'FAIL'}** ({n_pass}/{n_total} cases)",
        "",
        "## Purpose",
        "",
        "Verify `solve_huzhang_block_gmres_auxspace` on the Hu-Zhang mixed elastic system",
        "with **standard** formulation: stress block `M(d)` uses integrator coefficient `1/g(d)`.",
        "Nodes/cells with `d=1` yield `g(d)=eps_g`, hence local `1/g(d) ~ 1/eps_g` (e.g. `1e6` or `1e8`).",
        "Reference solution: sparse direct `spsolve`. Phase-field and damage evolution are **frozen**.",
        "",
        "## Configuration",
        "",
        f"- elastic_formulation: `{meta.get('elastic_formulation', 'standard')}`",
        f"- weighted_aux: `{meta.get('weighted_aux', True)}`",
        f"- GMRES rtol: `{meta.get('gmres_rtol', 1e-8)}`",
        f"- hmin: `{meta.get('hmin')}`",
        f"- mesh: NN={meta.get('NN')}, NC={meta.get('NC')}",
        f"- eps_g values: `{meta.get('eps_g_list')}`",
        f"- damage patterns: {', '.join(meta.get('patterns', []))}",
        "",
        "## Damage patterns",
        "",
        "| Pattern | Description |",
        "|---|---|",
    ]
    for name, desc in DAMAGE_PATTERNS.items():
        lines.append(f"| `{name}` | {desc} |")

    lines.extend(
        [
            "",
            "## Results (per load / pattern / eps_g)",
            "",
            "| pattern | eps_g | load | frac d=1 | g_min | inv_g_max | rel_diff | rel_res_aux | gmres_iters | PASS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r['pattern']} | {r['eps_g']:.0e} | {r['load']:.4f} | "
            f"{r.get('frac_nodes_d_eq_1', 0):.3f} | {r.get('g_min', 0):.2e} | "
            f"{r.get('inv_g_max', 0):.2e} | {r['rel_diff']:.2e} | {r['rel_res_aux']:.2e} | "
            f"{r.get('niter', -1)} | {'yes' if r.get('passed') else '**no**'} |"
        )

    lines.extend(
        [
            "",
            "## Acceptance criteria",
            "",
            "- `rel_diff = ||x_aux - x_direct|| / ||x_direct|| < 1e-5`",
            "- `rel_res_aux = ||A x_aux - b|| / ||b|| < 1e-7`",
            "- GMRES reports converged (`converged=True`)",
            "",
            "## Artifacts",
            "",
            "- `results_detail.csv`: all numeric rows",
            "- `summary.json`: aggregate pass/fail and metadata",
            "- `comparison_by_epsg.json`: grouped metrics",
            "",
            "> Auto-generated; re-run the test script to refresh.",
            "",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def main() -> int:
    hmin = float(os.environ.get("FRACTUREX_HMIN", "0.02"))

    run_short = os.environ.get("FRACTUREX_RUN_SHORT", "0") == "1"
    eps_g_raw = os.environ.get("FRACTUREX_EPS_G_LIST", "1e-6,1e-8")
    eps_g_list = [float(s.strip()) for s in eps_g_raw.split(",") if s.strip()]
    if run_short:
        eps_g_list = eps_g_list[:1]

    patterns = list(DAMAGE_PATTERNS.keys())
    if run_short:
        patterns = ["intact", "half_cracked"]

    loads = [0.001, 0.005, 0.01] if not run_short else [0.005]
    out_root = os.environ.get(
        "FRACTUREX_OUTDIR",
        os.path.join("results", "tests", "auxspace_degraded_elastic"),
    )

    # probe mesh once for report header
    mat = Model0Material()
    probe_case = Model0CircularNotchCase(_model=mat, hmin=hmin)
    probe_mesh = probe_case.make_mesh()
    hstats = mesh_h_stats(probe_mesh)

    all_rows: List[Dict[str, Any]] = []
    for eps_g in eps_g_list:
        tag = epsg_tag(eps_g)
        tag_dir = os.path.join(out_root, tag)
        rows = []
        for pattern in patterns:
            rows.extend(
                _run_one_case(
                    pattern=pattern,
                    eps_g=eps_g,
                    loads=loads,
                    hmin=hmin,
                    outdir=tag_dir,
                )
            )
        _write_csv(os.path.join(tag_dir, "results_detail.csv"), rows)
        by_pat: Dict[str, Any] = {}
        for r in rows:
            by_pat.setdefault(r["pattern"], []).append(
                {k: r[k] for k in ("load", "rel_diff", "rel_res_aux", "niter", "passed", "inv_g_max")}
            )
        with open(os.path.join(tag_dir, "comparison_by_epsg.json"), "w", encoding="utf-8") as f:
            json.dump({"eps_g": eps_g, "patterns": by_pat}, f, indent=2)
        all_rows.extend(rows)

    n_pass = sum(1 for r in all_rows if r.get("passed"))
    meta = {
        "elastic_formulation": "standard",
        "weighted_aux": True,
        "gmres_rtol": 1e-8,
        "hmin": float(hmin),
        "NN": int(probe_mesh.number_of_nodes()),
        "NC": int(probe_mesh.number_of_cells()),
        "h_stats": hstats,
        "eps_g_list": eps_g_list,
        "patterns": patterns,
        "loads": loads,
        "n_cases": len(all_rows),
        "n_pass": n_pass,
        "all_pass": n_pass == len(all_rows) and len(all_rows) > 0,
    }
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "rows": all_rows}, f, indent=2)
    _write_csv(os.path.join(out_root, "results_all.csv"), all_rows)

    report_path = _write_test_report(outdir=out_root, rows=all_rows, meta=meta)
    print(f"\nWrote report: {report_path}")
    print(f"Summary: {n_pass}/{len(all_rows)} passed, all_pass={meta['all_pass']}")

    return 0 if meta["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
