#!/usr/bin/env python3
"""Dump the REAL phase-field linear system A dd = F from a LOCAL post-crack checkpoint.
Usage: dump_phase_local.py <case> <run_label> <out.npz>
"""
import sys, os, numpy as np
from pathlib import Path
REPO='/Users/tian00/repository/fractureX'
sys.path.insert(0, REPO)
os.environ.setdefault('FRACTUREX_ASSEMBLY_NPROC','1')
import scripts.paper_huzhang.run_case as rc
from scipy.sparse import save_npz, csr_matrix

case_id, run_label, out = sys.argv[1], sys.argv[2], sys.argv[3]
case, mesh, mat, loads, mesh_param = rc._build_case(case_id)
tag = rc.phasefield_tag_dir(case.name, run_label, eps_g=rc.EPS_G,
                            root=REPO+'/results', mkdir=False)
# fall back to the results dir under repository/results (mirror) if not under fractureX/results
if not Path(tag).exists():
    tag = rc.phasefield_tag_dir(case.name, run_label, eps_g=rc.EPS_G,
                                root='/Users/tian00/repository/results', mkdir=False)
print('[dump] tag=', tag)
driver, discr, damage = rc._build_driver(case=case, mesh=mesh, mode='direct',
                                         run_dir=Path(tag), save_npz=False, save_every=1)
driver.initialize()  # triggers damage.on_build -> sets Gc/l0 (H reset to None here)
ck = rc._latest_checkpoint(Path(tag))
assert ck is not None, f'no checkpoint under {tag}'
ck_path, ck_step = ck
print(f'[dump] restore {ck_path.name} step={ck_step}')
rc._restore_state_from_checkpoint(discr, ck_path)
load = float(loads[min(ck_step, len(loads)-1)])
st = discr.state
dmax=float(np.max(np.asarray(st.d[:]))); Hmax=float(np.max(np.asarray(st.H))) if st.H is not None else float('nan')
print(f'[dump] load={load:.5e} max_d={dmax:.4f} max_H={Hmax:.3e}')
sys_d = driver.phase_assembler.assemble(load)
A = sys_d.A.to_scipy().tocsr() if hasattr(sys_d.A,'to_scipy') else csr_matrix(sys_d.A)
F = np.asarray(sys_d.F,dtype=float).reshape(-1)
d_old = np.asarray(st.d[:],dtype=float).reshape(-1)
asym = abs((A-A.T)).max() if A.nnz else 0.0
print(f'[dump] A shape={A.shape} nnz={A.nnz} |A-A^T|_max={asym:.2e} ||F||={np.linalg.norm(F):.4e}')
save_npz(out.replace('.npz','_A.npz'), A)
np.savez(out, F=F, d_old=d_old, load=load, step=ck_step, max_d=dmax, max_H=Hmax,
         gdof_d=A.shape[0], A_file=os.path.basename(out.replace('.npz','_A.npz')))
print(f'[dump] wrote {out} + _A.npz')
