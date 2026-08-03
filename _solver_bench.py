"""Elastic-solve backend benchmark (model2 nx=24, precrack, load>0).
Assemble once -> dump A,F npz -> time pardiso + mumps(orderings) + reuse_analysis.
All results accumulated and printed as a clean SUMMARY at the very end
(so MUMPS native stdout cannot clobber the numbers).
"""
import os, time
os.environ.setdefault("FRACTUREX_ASSEMBLY_PARALLEL", "0")
import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite  # noqa
from fealpy.backend import backend_manager as bm
bm.set_backend("numpy")
from fracturex.cases.model2_notch_shear import Model2NotchXStretchCase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.drivers.huzhang_phasefield_staggered import HuZhangPhaseFieldStaggeredDriver

class _Mat:
    E, nu, Gc, l0, ft = 210.0, 0.3, 2.7e-3, 0.015, 3.0
    @property
    def mu(self): return self.E/(2*(1+self.nu))
    @property
    def lam(self): return self.E*self.nu/((1+self.nu)*(1-2*self.nu))

RESULTS=[]
def rec(name, dt, x, Acsr, b):
    if dt is None: RESULTS.append((name, None, None)); return
    r=float(np.linalg.norm(Acsr@np.asarray(x).reshape(-1)-b)/max(np.linalg.norm(b),1e-30))
    RESULTS.append((name, dt, r))
def run(name, fn, Acsr, b):
    try:
        t=time.perf_counter(); x=fn(); dt=time.perf_counter()-t; rec(name,dt,x,Acsr,b)
    except Exception as e:
        RESULTS.append((name, None, f"{type(e).__name__}: {e}"))

NPZ="results/_solver_bench_A.npz"
if os.path.exists(NPZ):
    z=np.load(NPZ)
    Acsr=sp.csr_matrix((z["data"],z["indices"],z["indptr"]),shape=tuple(z["shape"])); b=z["b"]
    print(f"[load] reused {NPZ}", flush=True)
else:
    LOAD=1.0e-2
    mat=_Mat()
    case=Model2NotchXStretchCase(_model=mat,nx=24,ny=24,crack_y=0.5,crack_length=0.5)
    mesh=case.make_mesh()
    discr=HuZhangDiscretization(case=case,p=3,damage_p=1,use_relaxation=True).build(mesh=mesh)
    damage=PhaseFieldDamageModel(density_type="AT2",degradation_type="quadratic",split="hybrid",eps_g=1e-6)
    el=HuZhangElasticAssembler(discr,case,damage,formulation="standard")
    ph=PhaseFieldAssembler(discr,case,damage)
    drv=HuZhangPhaseFieldStaggeredDriver(case=case,discr=discr,damage=damage,
        elastic_assembler=el,phase_assembler=ph,tol=1e-4,maxit=200,
        elastic_solver=HuZhangPhaseFieldStaggeredDriver._default_spsolve,
        compute_linear_residual=False,debug=False,timing=False,save_vtu_per_step=False)
    drv.initialize()
    t=time.perf_counter(); sys_e=el.assemble(LOAD); print(f"[assemble] {time.perf_counter()-t:.1f}s", flush=True)
    A=sys_e.A; F=sys_e.F
    Acsr=(A.to_scipy().tocsr() if hasattr(A,"to_scipy") else A.tocsr()).astype(np.float64)
    b=np.asarray(F,dtype=np.float64).reshape(-1)
    np.savez(NPZ,data=Acsr.data,indices=Acsr.indices,indptr=Acsr.indptr,shape=Acsr.shape,b=b)
    print(f"[dump] {NPZ}", flush=True)
n=Acsr.shape[0]; se=abs(Acsr-Acsr.T); se=se.max() if se.nnz else 0.0
print(f"[matrix] n={n} nnz={Acsr.nnz} sym_err={se:.2e} |b|={np.linalg.norm(b):.3e}", flush=True)

import pypardiso
from pypardiso import PyPardisoSolver
run("pardiso_mtype11", lambda: pypardiso.spsolve(Acsr,b), Acsr, b)

import mumps
def mumps_ord(order):
    ctx=mumps.Context(verbose=False)
    Acsc=Acsr.tocsc()
    ctx.analyze(Acsc,ordering=order); ctx.factor(Acsc)
    return np.asarray(ctx.solve(b.copy())).reshape(-1)
for od in ("auto","metis","scotch","pord","amd"):
    run(f"mumps_{od}", (lambda o=od: mumps_ord(o)), Acsr, b)

# symbolic-reuse: analyze once (metis), factor two DIFFERENT value-sets
reuse_line=None
try:
    ctx=mumps.Context(verbose=False); Acsc=Acsr.tocsc()
    t=time.perf_counter(); ctx.analyze(Acsc,ordering="metis"); ta=time.perf_counter()-t
    t=time.perf_counter(); ctx.factor(Acsc); tf1=time.perf_counter()-t
    Acsc2=(Acsr*0.97).tocsc()
    t=time.perf_counter(); ctx.factor(Acsc2,reuse_analysis=True); tf2=time.perf_counter()-t
    reuse_line=f"analyze(metis)={ta:.2f}s  factor#1={tf1:.2f}s  factor#2(reuse)={tf2:.2f}s"
except Exception as e:
    reuse_line=f"FAILED {type(e).__name__}: {e}"

print("\n===SUMMARY_BEGIN===", flush=True)
for name,dt,r in RESULTS:
    if dt is None: print(f"{name:20s}  ERROR: {r}", flush=True)
    else: print(f"{name:20s}  t={dt:8.2f}s  rel_resid={r:.2e}", flush=True)
print(f"reuse_analysis:  {reuse_line}", flush=True)
print("===SUMMARY_END===", flush=True)
