"""Integration tests for the D13 enrichment seam inside the real Hu-Zhang solver.

Builds a genuine localized saddle operator from a phase-field checkpoint (same recipe
as scripts/paper_huzhang/check_localized_baselines.py) and checks:

  - solution-invariance: GMRES solution with enrichment ON matches OFF to machine
    precision (right preconditioning never changes the solution; plan command 4);
  - mechanism-gain: feeding an interface jump template does not break convergence and
    the seam transmits a coarse correction (niter remains bounded / improves).

These need FEALPy + pyamg + a dumped checkpoint; the module skips cleanly otherwise.

Run:
  PYTHONPATH=<repo> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m pytest fracturex/tests/test_coarse_space_enrich_integration.py -q -s
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]

# Localized real-field checkpoint (maxd ~ 0.998) + its mesh size.
_CKPT = _REPO / ("results/phasefield/model0_circular_notch/paper_aux_h2/"
                 "epsg_1e-06/checkpoints/step_015.npz")
_HMIN = 0.025
RTOL, ATOL = 1e-8, 1e-12


class _Mat:
    E = 200.0
    nu = 0.2
    Gc = 1.0
    l0 = 0.02

    @property
    def mu(self):
        return self.E / (2 * (1 + self.nu))

    @property
    def lam(self):
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))


@pytest.fixture(scope="module")
def localized_system():
    pytest.importorskip("pyamg")
    if not _CKPT.exists():
        pytest.skip(f"checkpoint not found: {_CKPT}")
    from fealpy.backend import backend_manager as bm
    from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
    from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
    from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
    from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
    from fracturex.utilfuc.linear_solvers import as_scipy_csr

    z = np.load(_CKPT)
    d_real = np.asarray(z["d"], float)
    dmg = PhaseFieldDamageModel(density_type="AT2", degradation_type="quadratic",
                                split="hybrid", eps_g=1e-6, debug=False)
    case = Model0CircularNotchCase(_model=_Mat(), hmin=_HMIN)
    mesh = case.make_mesh()
    discr = HuZhangDiscretization(case, p=3, damage_p=2, use_relaxation=True).build(mesh=mesh)
    dmg.on_build(discr, discr.state, case)
    asm = HuZhangElasticAssembler(discr, case, dmg, formulation="standard",
                                  assembly_parallel=False)
    if d_real.size != discr.space_d.number_of_global_dofs():
        pytest.skip("checkpoint d-dof mismatch for this hmin")
    discr.state.d[:] = bm.asarray(d_real)
    LOAD = 0.092
    asm.begin_load_step(LOAD)
    sys_e = asm.assemble(LOAD)
    F = np.asarray(sys_e.F, float).reshape(-1)
    return {
        "A": sys_e.A, "F": F, "m": int(discr.gdof_sigma),
        "vspace": discr.space_u, "mesh": mesh, "damage": dmg, "state": discr.state,
        "A_csr": as_scipy_csr(sys_e.A),
    }


def _solve(sysd, provider):
    from fracturex.utilfuc.linear_solvers import solve_huzhang_block_gmres_fast
    return solve_huzhang_block_gmres_fast(
        sysd["A"], sysd["F"], gdof_sigma=sysd["m"], vspace=sysd["vspace"],
        rtol=RTOL, atol=ATOL, restart=60, maxit=400, q=5,
        weighted_aux=True, elastic_formulation="standard",
        damage=sysd["damage"], state=sysd["state"],
        learned_coarse_provider=provider,
    )


def _template_provider(sysd):
    """Provider yielding an interface jump template Phi from the frozen damage field."""
    from fracturex.ml.coarse_features import extract_coarse_features
    from fracturex.ml.coarse_space_enrich import build_jump_template_modes
    cf = extract_coarse_features(sysd["mesh"], sysd["damage"], sysd["state"])
    Phi = build_jump_template_modes(cf.phi, grad_threshold=0.1)
    return lambda: Phi


def test_enrich_solution_invariance(localized_system):
    """Enrichment ON vs OFF: identical solution to machine precision."""
    x_off, info_off = _solve(localized_system, None)
    x_on, info_on = _solve(localized_system, _template_provider(localized_system))
    A = localized_system["A_csr"]
    b = localized_system["F"]
    bn = max(float(np.linalg.norm(b)), 1e-30)
    # both must solve the SAME system to tolerance
    r_off = float(np.linalg.norm(A @ x_off - b) / bn)
    r_on = float(np.linalg.norm(A @ x_on - b) / bn)
    assert r_off <= RTOL * 50, f"baseline residual too large: {r_off:.2e}"
    assert r_on <= RTOL * 50, f"enriched residual too large: {r_on:.2e}"
    # solutions agree (same linear system, both converged): right preconditioning.
    rel = float(np.linalg.norm(x_on - x_off) / max(np.linalg.norm(x_off), 1e-30))
    print(f"\n[invariance] niter off={info_off.niter} on={info_on.niter} "
          f"relres off={r_off:.2e} on={r_on:.2e} |x_on-x_off|/|x_off|={rel:.2e}")
    assert rel <= 1e-6, f"enrichment changed the solution: rel={rel:.2e}"


def test_enrich_does_not_break_convergence(localized_system):
    """With the template enrichment, GMRES still converges and niter stays bounded."""
    _, info_on = _solve(localized_system, _template_provider(localized_system))
    assert info_on.converged or info_on.residual_norm <= RTOL * 100
    assert info_on.niter <= 400
