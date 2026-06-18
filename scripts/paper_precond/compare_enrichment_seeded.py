#!/usr/bin/env python3
"""Controlled (deterministic) enrichment niter comparison.

pyamg 5.3's smoothed_aggregation_solver consumes the GLOBAL numpy RNG, so consecutive
solves build different AMG hierarchies -> ~19% niter noise that swamps any enrichment
signal (D13_IMPL §6.4). Fix: reseed numpy's global RNG to the SAME value before every
solve, so all strategies share one hierarchy realization and differ ONLY by enrichment.
We first verify determinism (baseline twice must match), then compare.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
_REPO=Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path: sys.path.insert(0,str(_REPO))
from fealpy.backend import backend_manager as bm
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.utilfuc.linear_solvers import (as_scipy_csr,_extract_mechanical_blocks,
  _diag_inv_stress_block,_approximate_schur_spd,_get_or_build_auxspace_pi_operators,
  solve_huzhang_block_gmres_fast, _FAST_PRECOND_CACHE, _AUXSPACE_COARSE_CACHE,
  _FAST_PYAMG_P1_ISO_CACHE)
from fracturex.ml.coarse_features import extract_coarse_features
from fracturex.ml.spectral_labels import worst_mode_amplitude
from fracturex.ml.coarse_space_enrich import build_jump_template_modes
from fracturex.ml.spectral_labels import ideal_interface_amplitude

class Mat:
    E=200.0;nu=0.2;Gc=1.0;l0=0.02
    @property
    def mu(self): return self.E/(2*(1+self.nu))
    @property
    def lam(self): return self.E*self.nu/((1+self.nu)*(1-2*self.nu))

SEED=12345
def _clear_caches():
    # force fresh pyamg builds so the reseed actually controls the hierarchy
    _FAST_PRECOND_CACHE.clear(); _AUXSPACE_COARSE_CACHE.clear(); _FAST_PYAMG_P1_ISO_CACHE.clear()

def main():
    ckpt=_REPO/"results/phasefield/model0_circular_notch/paper_aux_h2/epsg_1e-06/checkpoints/step_015.npz"
    hmin=0.025
    z=np.load(ckpt);d=np.asarray(z["d"],float)
    dmg=PhaseFieldDamageModel(density_type="AT2",degradation_type="quadratic",split="hybrid",eps_g=1e-6)
    case=Model0CircularNotchCase(_model=Mat(),hmin=hmin);mesh=case.make_mesh()
    discr=HuZhangDiscretization(case,p=3,damage_p=2,use_relaxation=True).build(mesh=mesh)
    dmg.on_build(discr,discr.state,case)
    asm=HuZhangElasticAssembler(discr,case,dmg,formulation="standard",assembly_parallel=False)
    discr.state.d[:]=bm.asarray(d);asm.begin_load_step(0.092);sys_e=asm.assemble(0.092)
    A_=as_scipy_csr(sys_e.A);F=np.asarray(sys_e.F,float).reshape(-1);m=int(discr.gdof_sigma)
    _,M,B=_extract_mechanical_blocks(A_,m);Dinv=_diag_inv_stress_block(M)
    S=_approximate_schur_spd(M,B,Dinv)
    cached=_get_or_build_auxspace_pi_operators(mesh,discr.space_u,5);sgdof=cached["sgdof"];PI_s=cached["PI_s"]
    Sb=S[:sgdof,:sgdof].tocsr()
    cf=extract_coarse_features(mesh,dmg,discr.state)
    tmpl=build_jump_template_modes(cf.phi).reshape(-1)
    amp_worst=worst_mode_amplitude(Sb,PI_s,iters=60)
    amp_heur=ideal_interface_amplitude(cf.phi)
    print(f"maxd={d.max():.4f} sigma-dof={m} SEED={SEED}",flush=True)
    def run(prov,tag):
        _clear_caches(); np.random.seed(SEED)   # deterministic hierarchy per solve
        t0=time.perf_counter()
        _,info=solve_huzhang_block_gmres_fast(sys_e.A,F,gdof_sigma=m,vspace=discr.space_u,
          rtol=1e-8,atol=1e-12,restart=60,maxit=400,q=5,weighted_aux=True,
          elastic_formulation="standard",damage=dmg,state=discr.state,learned_coarse_provider=prov)
        print(f"{tag:>16} niter={info.niter} conv={info.converged} t={time.perf_counter()-t0:.0f}s",flush=True)
        return info.niter
    b1=run(None,"baseline_1"); b2=run(None,"baseline_2")
    print(f"  determinism: baseline_1={b1} baseline_2={b2} -> {'DETERMINISTIC' if b1==b2 else 'STILL NOISY'}",flush=True)
    run(lambda:(tmpl*amp_heur).reshape(-1,1),"heuristic")
    run(lambda:amp_worst.reshape(-1,1),"worst_mode")

if __name__=="__main__": main()
