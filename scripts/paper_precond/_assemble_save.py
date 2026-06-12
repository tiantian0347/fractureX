#!/usr/bin/env python3
"""Assemble the localized Hu-Zhang operator ONCE and save (A, F) for parallel knob sweeps.

The per-element assembly is the slow setup (~minutes); discr rebuild is ~2s. So we
assemble once here, save the scipy CSR A + rhs F + meta, and let _solve_knob.py workers
load them and solve with different solver knobs in parallel (the machine has free cores).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
_REPO=Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path: sys.path.insert(0,str(_REPO))
from fealpy.backend import backend_manager as bm
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.utilfuc.linear_solvers import as_scipy_csr
class Mat:
    E=200.0;nu=0.2;Gc=1.0;l0=0.02
    @property
    def mu(self): return self.E/(2*(1+self.nu))
    @property
    def lam(self): return self.E*self.nu/((1+self.nu)*(1-2*self.nu))
def main():
    ckpt=Path(sys.argv[1]) if len(sys.argv)>1 else _REPO/"results/phasefield/model0_circular_notch/paper_aux_h2/epsg_1e-06/checkpoints/step_015.npz"
    hmin=float(sys.argv[2]) if len(sys.argv)>2 else 0.025
    outdir=_REPO/"results/phasefield/_precond_knob"; outdir.mkdir(parents=True,exist_ok=True)
    z=np.load(ckpt); d=np.asarray(z["d"],float)
    dmg=PhaseFieldDamageModel(density_type="AT2",degradation_type="quadratic",split="hybrid",eps_g=1e-6)
    case=Model0CircularNotchCase(_model=Mat(),hmin=hmin); mesh=case.make_mesh()
    discr=HuZhangDiscretization(case,p=3,damage_p=2,use_relaxation=True).build(mesh=mesh)
    dmg.on_build(discr,discr.state,case)
    asm=HuZhangElasticAssembler(discr,case,dmg,formulation="standard",assembly_parallel=False)
    discr.state.d[:]=bm.asarray(d)
    t0=time.perf_counter(); asm.begin_load_step(0.092); sys_e=asm.assemble(0.092)
    A_=as_scipy_csr(sys_e.A); F=np.asarray(sys_e.F,float).reshape(-1)
    print(f"assembled in {time.perf_counter()-t0:.0f}s: A {A_.shape} nnz={A_.nnz} gdof_sigma={discr.gdof_sigma}",flush=True)
    sp.save_npz(outdir/"A.npz", A_)
    np.savez(outdir/"meta.npz", F=F, d=d, gdof_sigma=int(discr.gdof_sigma), hmin=hmin,
             ckpt=str(ckpt))
    print(f"saved to {outdir}/A.npz + meta.npz",flush=True)
if __name__=="__main__": main()
