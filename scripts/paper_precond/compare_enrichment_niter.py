#!/usr/bin/env python3
"""Compare D13 enrichment label strategies by REAL GMRES niter on a localized operator.

The two-level kappa proxy is unreliable here (non-symmetric M^{-1}S; ARPACK SM does not
converge -- see D12_RESULTS §5.6), so we use the honest metric: actual GMRES iteration
count on the assembled localized Hu-Zhang saddle system. Compares:
  baseline (no enrichment) / heuristic template / worst-mode spectral label.

Usage: compare_enrichment_niter.py [ckpt.npz] [hmin]
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path: sys.path.insert(0, str(_REPO))
from fealpy.backend import backend_manager as bm
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.utilfuc.linear_solvers import (
    as_scipy_csr, _extract_mechanical_blocks, _diag_inv_stress_block,
    _approximate_schur_spd, _get_or_build_auxspace_pi_operators,
    solve_huzhang_block_gmres_fast)
from fracturex.ml.coarse_features import extract_coarse_features
from fracturex.ml.spectral_labels import worst_mode_amplitude, ideal_interface_amplitude
from fracturex.ml.coarse_space_enrich import build_jump_template_modes

class Mat:
    E=200.0;nu=0.2;Gc=1.0;l0=0.02
    @property
    def mu(self): return self.E/(2*(1+self.nu))
    @property
    def lam(self): return self.E*self.nu/((1+self.nu)*(1-2*self.nu))

def main():
    ckpt=Path(sys.argv[1]) if len(sys.argv)>1 else _REPO/"results/phasefield/model0_circular_notch/paper_aux_h2/epsg_1e-06/checkpoints/step_015.npz"
    hmin=float(sys.argv[2]) if len(sys.argv)>2 else 0.025
    z=np.load(ckpt); d_real=np.asarray(z["d"],float)
    dmg=PhaseFieldDamageModel(density_type="AT2",degradation_type="quadratic",split="hybrid",eps_g=1e-6,debug=False)
    case=Model0CircularNotchCase(_model=Mat(),hmin=hmin); mesh=case.make_mesh()
    discr=HuZhangDiscretization(case,p=3,damage_p=2,use_relaxation=True).build(mesh=mesh)
    dmg.on_build(discr,discr.state,case)
    asm=HuZhangElasticAssembler(discr,case,dmg,formulation="standard",assembly_parallel=False)
    discr.state.d[:]=bm.asarray(d_real)
    asm.begin_load_step(0.092); sys_e=asm.assemble(0.092)
    A_=as_scipy_csr(sys_e.A); F=np.asarray(sys_e.F,float).reshape(-1); m=int(discr.gdof_sigma)
    mm,M,B=_extract_mechanical_blocks(A_,m); Dinv=_diag_inv_stress_block(M)
    S=_approximate_schur_spd(M,B,Dinv)
    cached=_get_or_build_auxspace_pi_operators(mesh,discr.space_u,5); sgdof=cached["sgdof"]; PI_s=cached["PI_s"]
    Sb=S[:sgdof,:sgdof].tocsr()
    cf=extract_coarse_features(mesh,dmg,discr.state)
    print(f"maxd={d_real.max():.4f} sigma-dof={m} total-dof={A_.shape[0]}",flush=True)

    # labels -> coarse-space column Phi (NN,1)
    tmpl=build_jump_template_modes(cf.phi).reshape(-1)
    labels={
        "baseline": None,
        "heuristic": (tmpl*ideal_interface_amplitude(cf.phi)).reshape(-1,1),
        "worst_mode": worst_mode_amplitude(Sb,PI_s,iters=60).reshape(-1,1),
    }
    print(f"{'strategy':>12} {'niter':>7} {'conv':>6} {'t_s':>8}",flush=True)
    for name,Phi in labels.items():
        prov=None if Phi is None else (lambda P=Phi: P)
        t0=time.perf_counter()
        _,info=solve_huzhang_block_gmres_fast(
            sys_e.A,F,gdof_sigma=m,vspace=discr.space_u,rtol=1e-8,atol=1e-12,
            restart=60,maxit=400,q=5,weighted_aux=True,elastic_formulation="standard",
            damage=dmg,state=discr.state,learned_coarse_provider=prov)
        print(f"{name:>12} {info.niter:>7} {str(info.converged):>6} {time.perf_counter()-t0:>8.1f}",flush=True)

if __name__=="__main__": main()
