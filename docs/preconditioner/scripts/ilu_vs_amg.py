#!/usr/bin/env python3
"""ILU fill-in / setup vs AMG operator-complexity across meshes -> why not ILU."""
import time, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, pyamg
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="/Users/tian00/repository/fractureX/docs/preconditioner"
tags=["m0h1","m0h2","m0h3"]; ns=[]; ilu_fill={dt:[] for dt in (1e-2,1e-4)}
ilu_it={dt:[] for dt in (1e-2,1e-4)}; amg_op=[]; amg_it=[]
for t in tags:
    A=sp.load_npz(f"{OUT}/data/phase_{t}_A.npz").tocsc(); F=np.load(f"{OUT}/data/phase_{t}.npz")["F"].astype(float)
    n=A.shape[0]; ns.append(n); nnzA=A.nnz
    def it_of(M):
        c=[0]; spla.cg(A,F,rtol=1e-8,atol=0,maxiter=5000,M=M,callback=lambda x:c.__setitem__(0,c[0]+1)); return c[0]
    for dt_ in (1e-2,1e-4):
        ilu=spla.spilu(A,drop_tol=dt_,fill_factor=20)
        ilu_fill[dt_].append((ilu.L.nnz+ilu.U.nnz)/nnzA); ilu_it[dt_].append(it_of(spla.LinearOperator((n,n),ilu.solve)))
    ml=pyamg.smoothed_aggregation_solver(A.tocsr(),max_coarse=10)
    amg_op.append(ml.operator_complexity()); amg_it.append(it_of(ml.aspreconditioner()))
fig,axs=plt.subplots(1,2,figsize=(12,4.4))
ax=axs[0]
ax.plot(ns,ilu_fill[1e-4],"-o",label="ILU drop=1e-4  (nnz(LU)/nnz(A))")
ax.plot(ns,ilu_fill[1e-2],"-o",label="ILU drop=1e-2  (nnz(LU)/nnz(A))")
ax.plot(ns,amg_op,"-s",label="AMG-SA operator complexity")
ax.set_xscale("log"); ax.set_xlabel("n"); ax.set_ylabel("storage blow-up (x nnz(A))")
ax.set_title("Memory / operator complexity vs refinement"); ax.grid(True,which="both",alpha=.3); ax.legend(fontsize=8)
ax=axs[1]
ax.plot(ns,ilu_it[1e-4],"-o",label="CG+ILU drop=1e-4")
ax.plot(ns,ilu_it[1e-2],"-o",label="CG+ILU drop=1e-2 (cheap)")
ax.plot(ns,amg_it,"-s",label="CG+AMG-SA")
ax.set_xscale("log"); ax.set_xlabel("n"); ax.set_ylabel("CG iters (rtol=1e-8)")
ax.set_title("Iterations vs refinement: cheap-ILU diverges, AMG stable"); ax.grid(True,which="both",alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/ilu_vs_amg.png",dpi=140); print("wrote ilu_vs_amg.png")
print("ns=",ns); print("ilu_fill_1e-4=",[f"{x:.1f}" for x in ilu_fill[1e-4]])
print("ilu_it_1e-2=",ilu_it[1e-2]," amg_it=",amg_it," amg_op=",[f"{x:.2f}" for x in amg_op])
