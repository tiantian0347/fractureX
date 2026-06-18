#!/usr/bin/env python3
"""Cross-case restart re-check: is 'localized O(100) niter = restart artifact' universal?

Loads ONE localized checkpoint for square / model2, assembles the operator, and solves
the fast aux preconditioner at restart in {60, 200, 300}. Deterministic (seed + clear
pyamg caches). Confirms the model0 finding (D13_IMPL §9.4) generalizes across crack
modes (I-type square, II-type model2). Usage: restart_recheck_xcase.py <case> <step>
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
_REPO=Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path: sys.path.insert(0,str(_REPO))
from fealpy.backend import backend_manager as bm
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.cases.square_tension_precrack import SquareTensionPreCrackCase
from fracturex.cases.model2_notch_shear import Model2NotchXStretchCase
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.utilfuc.linear_solvers import (as_scipy_csr, solve_huzhang_block_gmres_fast,
  _FAST_PRECOND_CACHE,_AUXSPACE_COARSE_CACHE,_FAST_PYAMG_P1_ISO_CACHE)
class SquareMat:
    E=210.0;nu=0.3;Gc=2.7e-3;l0=0.015;ft=3.0
    @property
    def mu(self): return self.E/(2*(1+self.nu))
    @property
    def lam(self): return self.E*self.nu/((1+self.nu)*(1-2*self.nu))
class Model2Mat:
    E=210.0;nu=0.3;Gc=2.7e-3;l0=1.5e-2
    @property
    def mu(self): return self.E/(2*(1+self.nu))
    @property
    def lam(self): return self.E*self.nu/((1+self.nu)*(1-2*self.nu))
SEED=12345
def build(case_id):
    if case_id=="square":
        return SquareTensionPreCrackCase(_model=SquareMat(),nx=216,ny=216,debug_mesh=False), 5e-3
    if case_id=="model2":
        return Model2NotchXStretchCase(_model=Model2Mat(),nx=160,ny=160,debug_mesh=False), 0.092
    raise ValueError(case_id)
def main():
    case_id=sys.argv[1]; step=sys.argv[2]
    restarts=[int(x) for x in (sys.argv[3].split(",") if len(sys.argv)>3 else ["60","200","300"])]
    ck=_REPO/f"results/phasefield/{'square_tension_precrack' if case_id=='square' else 'model2_notch_x_stretch'}/paper_aux/epsg_1e-06/checkpoints/step_{step}.npz"
    z=np.load(ck); d=np.asarray(z["d"],float)
    case,load=build(case_id); mesh=case.make_mesh()
    discr=HuZhangDiscretization(case,p=3,damage_p=2,use_relaxation=True).build(mesh=mesh)
    dmg=PhaseFieldDamageModel(density_type="AT2",degradation_type="quadratic",split="hybrid",eps_g=1e-6)
    dmg.on_build(discr,discr.state,case)
    if d.size!=discr.space_d.number_of_global_dofs():
        print(f"DOF MISMATCH ckpt {d.size} vs discr {discr.space_d.number_of_global_dofs()}"); return
    asm=HuZhangElasticAssembler(discr,case,dmg,formulation="standard",assembly_parallel=False)
    discr.state.d[:]=bm.asarray(d); asm.begin_load_step(load)
    t0=time.perf_counter(); sys_e=asm.assemble(load)
    A_=as_scipy_csr(sys_e.A); F=np.asarray(sys_e.F,float).reshape(-1); m=int(discr.gdof_sigma)
    print(f"{case_id} step_{step}: maxd={d.max():.4f} sigma-dof={m} total-dof={A_.shape[0]} assembled {time.perf_counter()-t0:.0f}s",flush=True)
    for r in restarts:
        _FAST_PRECOND_CACHE.clear();_AUXSPACE_COARSE_CACHE.clear();_FAST_PYAMG_P1_ISO_CACHE.clear()
        np.random.seed(SEED)
        t0=time.perf_counter()
        _,info=solve_huzhang_block_gmres_fast(A_,F,gdof_sigma=m,vspace=discr.space_u,rtol=1e-8,atol=1e-12,
          restart=r,maxit=max(400,3*r),q=5,weighted_aux=True,elastic_formulation="standard",
          damage=dmg,state=discr.state)
        print(f"  restart={r:>4} niter={info.niter:>4} conv={info.converged} t={time.perf_counter()-t0:.0f}s",flush=True)
if __name__=="__main__": main()
