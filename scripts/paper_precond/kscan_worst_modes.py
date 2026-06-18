#!/usr/bin/env python3
"""k-scan: niter vs number of GenEO worst modes on the real localized operator.

Deterministic (seed before each solve; pyamg SA consumes global RNG -- D13_IMPL §6.4).
Builds top-k worst modes (top_k_worst_modes) as the deflation subspace and measures real
GMRES niter for k = 0(baseline),1,2,4,8,16,32. Answers: how many modes compress the
localized O(100) niter, and whether multi-mode breaks the rank-1 ceiling (§6.5).
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
  solve_huzhang_block_gmres_fast,_FAST_PRECOND_CACHE,_AUXSPACE_COARSE_CACHE,_FAST_PYAMG_P1_ISO_CACHE)
from fracturex.ml.spectral_labels import top_k_worst_modes
class Mat:
    E=200.0;nu=0.2;Gc=1.0;l0=0.02
    @property
    def mu(self): return self.E/(2*(1+self.nu))
    @property
    def lam(self): return self.E*self.nu/((1+self.nu)*(1-2*self.nu))
SEED=12345
def _clear(): _FAST_PRECOND_CACHE.clear();_AUXSPACE_COARSE_CACHE.clear();_FAST_PYAMG_P1_ISO_CACHE.clear()
def main():
    ks=[int(x) for x in (sys.argv[1].split(",") if len(sys.argv)>1 else ["0","1","2","4","8","16","32"])]
    ckpt=_REPO/"results/phasefield/model0_circular_notch/paper_aux_h2/epsg_1e-06/checkpoints/step_015.npz"
    z=np.load(ckpt);d=np.asarray(z["d"],float)
    dmg=PhaseFieldDamageModel(density_type="AT2",degradation_type="quadratic",split="hybrid",eps_g=1e-6)
    case=Model0CircularNotchCase(_model=Mat(),hmin=0.025);mesh=case.make_mesh()
    discr=HuZhangDiscretization(case,p=3,damage_p=2,use_relaxation=True).build(mesh=mesh)
    dmg.on_build(discr,discr.state,case)
    asm=HuZhangElasticAssembler(discr,case,dmg,formulation="standard",assembly_parallel=False)
    discr.state.d[:]=bm.asarray(d);asm.begin_load_step(0.092);sys_e=asm.assemble(0.092)
    A_=as_scipy_csr(sys_e.A);F=np.asarray(sys_e.F,float).reshape(-1);m=int(discr.gdof_sigma)
    _,M,B=_extract_mechanical_blocks(A_,m);Dinv=_diag_inv_stress_block(M)
    S=_approximate_schur_spd(M,B,Dinv)
    cached=_get_or_build_auxspace_pi_operators(mesh,discr.space_u,5);sgdof=cached["sgdof"];PI_s=cached["PI_s"]
    Sb=S[:sgdof,:sgdof].tocsr()
    print(f"maxd={d.max():.4f} sigma-dof={m} sgdof={sgdof} SEED={SEED}",flush=True)
    kmax=max(ks)
    Vall=top_k_worst_modes(Sb,PI_s,kmax,iters=40) if kmax>0 else None
    print(f"built {kmax} worst modes: {None if Vall is None else Vall.shape}",flush=True)
    print(f"{'k':>4} {'niter':>7} {'conv':>6} {'t_s':>8}",flush=True)
    for k in ks:
        prov=None if k==0 else (lambda kk=k: Vall[:,:kk])
        _clear(); np.random.seed(SEED)
        t0=time.perf_counter()
        _,info=solve_huzhang_block_gmres_fast(sys_e.A,F,gdof_sigma=m,vspace=discr.space_u,
          rtol=1e-8,atol=1e-12,restart=60,maxit=400,q=5,weighted_aux=True,
          elastic_formulation="standard",damage=dmg,state=discr.state,learned_coarse_provider=prov)
        print(f"{k:>4} {info.niter:>7} {str(info.converged):>6} {time.perf_counter()-t0:>8.0f}",flush=True)
if __name__=="__main__": main()
