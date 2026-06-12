#!/usr/bin/env python3
"""Worker: load the pre-assembled localized operator and solve with one knob config.

Config via CLI flags. Deterministic (seed + clear pyamg caches). Prints one result line.
Usage: _solve_knob.py <tag> [--gs N] [--schur MODE] [--restart R] [--maxit M] [--deflate K]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
_REPO=Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path: sys.path.insert(0,str(_REPO))
from fealpy.backend import backend_manager as bm
from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.utilfuc.linear_solvers import (solve_huzhang_block_gmres_fast,
  _extract_mechanical_blocks,_diag_inv_stress_block,_approximate_schur_spd,
  _get_or_build_auxspace_pi_operators,_FAST_PRECOND_CACHE,_AUXSPACE_COARSE_CACHE,_FAST_PYAMG_P1_ISO_CACHE)
class Mat:
    E=200.0;nu=0.2;Gc=1.0;l0=0.02
    @property
    def mu(self): return self.E/(2*(1+self.nu))
    @property
    def lam(self): return self.E*self.nu/((1+self.nu)*(1-2*self.nu))
SEED=12345
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--gs",type=int,default=2)
    ap.add_argument("--schur",default="auto")
    ap.add_argument("--restart",type=int,default=60)
    ap.add_argument("--maxit",type=int,default=400)
    ap.add_argument("--deflate",type=int,default=0)
    ap.add_argument("--cheb-degree",type=int,default=2)
    args=ap.parse_args()
    d=_REPO/"results/phasefield/_precond_knob"
    A_=sp.load_npz(d/"A.npz"); meta=np.load(d/"meta.npz",allow_pickle=True)
    F=np.asarray(meta["F"],float); dfield=np.asarray(meta["d"],float)
    m=int(meta["gdof_sigma"]); hmin=float(meta["hmin"])
    case=Model0CircularNotchCase(_model=Mat(),hmin=hmin); mesh=case.make_mesh()
    discr=HuZhangDiscretization(case,p=3,damage_p=2,use_relaxation=True).build(mesh=mesh)
    dmg=PhaseFieldDamageModel(density_type="AT2",degradation_type="quadratic",split="hybrid",eps_g=1e-6)
    dmg.on_build(discr,discr.state,case); discr.state.d[:]=bm.asarray(dfield)
    prov=None
    if args.deflate>0:
        from fracturex.ml.spectral_labels import top_k_worst_modes
        _,M,B=_extract_mechanical_blocks(A_,m);Dinv=_diag_inv_stress_block(M)
        S=_approximate_schur_spd(M,B,Dinv)
        cached=_get_or_build_auxspace_pi_operators(mesh,discr.space_u,5);sgdof=cached["sgdof"];PI_s=cached["PI_s"]
        V=top_k_worst_modes(S[:sgdof,:sgdof].tocsr(),PI_s,args.deflate,iters=40)
        prov=lambda:V
    _FAST_PRECOND_CACHE.clear();_AUXSPACE_COARSE_CACHE.clear();_FAST_PYAMG_P1_ISO_CACHE.clear()
    np.random.seed(SEED)
    t0=time.perf_counter()
    _,info=solve_huzhang_block_gmres_fast(A_,F,gdof_sigma=m,vspace=discr.space_u,
      rtol=1e-8,atol=1e-12,restart=args.restart,maxit=args.maxit,q=5,weighted_aux=True,
      elastic_formulation="standard",damage=dmg,state=discr.state,
      gs_iterations=args.gs,schur_precond=args.schur,cheb_degree=args.cheb_degree,
      learned_coarse_provider=prov)
    print(f"RESULT {args.tag:>22} niter={info.niter:>4} conv={info.converged} t={time.perf_counter()-t0:.0f}s "
          f"[gs={args.gs} schur={args.schur} restart={args.restart} deflate={args.deflate}]",flush=True)
if __name__=="__main__": main()
