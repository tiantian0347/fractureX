#!/usr/bin/env python3
"""D12 §5.2b re-check: is the localized 'O(100) niter' a GMRES-restart=60 artifact?

For several real h2 checkpoints spanning pre-localization -> fully localized, assemble
the operator and solve the fast aux preconditioner at restart in {60(D12 default),200,300}.
Deterministic (seed + clear pyamg caches). One operator assembly per checkpoint (~2s),
all restart solves serial within a checkpoint (cheap). Prints niter table.
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
from fracturex.utilfuc.linear_solvers import (as_scipy_csr, solve_huzhang_block_gmres_fast,
  _FAST_PRECOND_CACHE,_AUXSPACE_COARSE_CACHE,_FAST_PYAMG_P1_ISO_CACHE)
class Mat:
    E=200.0;nu=0.2;Gc=1.0;l0=0.02
    @property
    def mu(self): return self.E/(2*(1+self.nu))
    @property
    def lam(self): return self.E*self.nu/((1+self.nu)*(1-2*self.nu))
SEED=12345
import os
_TAG=os.environ.get("D12_TAG","paper_aux_h2")
CKDIR=_REPO/f"results/phasefield/model0_circular_notch/{_TAG}/epsg_1e-06/checkpoints"
def solve_one(A,F,m,vspace,dmg,state,restart,maxit):
    _FAST_PRECOND_CACHE.clear();_AUXSPACE_COARSE_CACHE.clear();_FAST_PYAMG_P1_ISO_CACHE.clear()
    np.random.seed(SEED)
    t0=time.perf_counter()
    _,info=solve_huzhang_block_gmres_fast(A,F,gdof_sigma=m,vspace=vspace,rtol=1e-8,atol=1e-12,
      restart=restart,maxit=maxit,q=5,weighted_aux=True,elastic_formulation="standard",
      damage=dmg,state=state)
    return info.niter,info.converged,time.perf_counter()-t0
def _discover_steps(localized_only: bool):
    """List checkpoint step ids under CKDIR. localized_only -> only maxd>0.9 (sorted)."""
    out=[]
    for p in sorted(CKDIR.glob("step_*.npz")):
        st=p.stem.split("_")[1]
        if localized_only:
            try:
                if float(np.asarray(np.load(p)["d"],float).max())<=0.9: continue
            except Exception: continue
        out.append(st)
    return out

def main():
    # arg1: step list, "auto"(all), or "localized"(only maxd>0.9). default h2 known steps.
    a1=sys.argv[1] if len(sys.argv)>1 else "013,014,015,017,020"
    if a1=="auto": steps=_discover_steps(localized_only=False)
    elif a1=="localized": steps=_discover_steps(localized_only=True)
    else: steps=a1.split(",")
    restarts=[int(x) for x in (sys.argv[2].split(",") if len(sys.argv)>2 else ["60","200","300"])]
    hmin=float(os.environ.get("D12_HMIN","0.025"))
    print(f"# TAG={_TAG} HMIN={hmin} steps={steps}",flush=True)
    print(f"{'step':>6} {'maxd':>7} " + " ".join(f"r{r:>4}".rjust(14) for r in restarts),flush=True)
    for st in steps:
        ck=CKDIR/f"step_{st}.npz"
        if not ck.exists(): print(f"step_{st}: MISSING",flush=True); continue
        z=np.load(ck); d=np.asarray(z["d"],float)
        dmg=PhaseFieldDamageModel(density_type="AT2",degradation_type="quadratic",split="hybrid",eps_g=1e-6)
        case=Model0CircularNotchCase(_model=Mat(),hmin=hmin); mesh=case.make_mesh()
        discr=HuZhangDiscretization(case,p=3,damage_p=2,use_relaxation=True).build(mesh=mesh)
        dmg.on_build(discr,discr.state,case)
        asm=HuZhangElasticAssembler(discr,case,dmg,formulation="standard",assembly_parallel=False)
        discr.state.d[:]=bm.asarray(d); asm.begin_load_step(0.092); sys_e=asm.assemble(0.092)
        A_=as_scipy_csr(sys_e.A); F=np.asarray(sys_e.F,float).reshape(-1); m=int(discr.gdof_sigma)
        cells=[]
        for r in restarts:
            maxit=max(400, 3*r)
            niter,conv,t=solve_one(A_,F,m,discr.space_u,dmg,discr.state,r,maxit)
            cells.append(f"{niter}/{('Y' if conv else 'N')}/{t:.0f}s")
        print(f"{st:>6} {d.max():>7.4f} " + " ".join(c.rjust(14) for c in cells),flush=True)
if __name__=="__main__": main()
