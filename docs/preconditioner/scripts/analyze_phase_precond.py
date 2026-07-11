#!/usr/bin/env python3
"""Local analysis + preconditioner benchmark for the REAL post-crack phase system.

Loads A (SPD), F from the dumped npz. Produces:
  1) matrix-form diagnostics (diag contrast = reaction-term contrast, sparsity)
  2) full eigenvalue spectrum (n small) of A, D^{-1/2} A D^{-1/2} (Jacobi),
     and M_amg^{-1} A (AMG-preconditioned) -> condition numbers
  3) preconditioner benchmark: reference spsolve, then CG/GMRES with
     none / Jacobi / ICC(ILU) / AMG-SA / AMG-RS -> iters, time, relerr (correctness)
  4) plots: spectrum comparison + CG convergence histories
Outputs JSON + PNGs into docs/preconditioner/.
"""
import os, sys, json, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/Users/tian00/repository/fractureX/docs/preconditioner/data"
OUT  = "/Users/tian00/repository/fractureX/docs/preconditioner"
tag  = sys.argv[1] if len(sys.argv) > 1 else "m0h1"

meta = np.load(f"{DATA}/phase_{tag}.npz")
A = sp.load_npz(f"{DATA}/phase_{tag}_A.npz").tocsr()
F = meta["F"].astype(float)
n = A.shape[0]
res = {"tag": tag, "n": int(n), "nnz": int(A.nnz),
       "max_d": float(meta["max_d"]), "load": float(meta["load"]),
       "max_H": float(meta["max_H"]), "step": int(meta["step"])}

# ---------- 1) matrix form ----------
d = A.diagonal()
sym = abs((A - A.T)).max()
res["symmetry_maxabs"] = float(sym)
res["diag_min"] = float(d.min()); res["diag_max"] = float(d.max())
res["diag_contrast"] = float(d.max() / d.min())
res["row_nnz_mean"] = float(A.nnz / n)
print(f"[form] n={n} nnz={A.nnz} sym={sym:.1e} "
      f"diag[min,max]=[{d.min():.3e},{d.max():.3e}] contrast={d.max()/d.min():.2e}")

# ---------- 2) spectrum (dense; n is small) ----------
DENSE_CAP = 8000
dsafe = np.where(d > 0, d, 1.0)
Dm12 = 1.0 / np.sqrt(dsafe)
if n <= DENSE_CAP:
    Ad = A.toarray(); Ad = 0.5 * (Ad + Ad.T)  # full dense spectrum
    ev = np.clip(eigh(Ad, eigvals_only=True), 1e-300, None)
    Aj = (Dm12[:, None] * Ad) * Dm12[None, :]; Aj = 0.5 * (Aj + Aj.T)
    evj = np.clip(eigh(Aj, eigvals_only=True), 1e-300, None)
    res["spectrum_method"] = "dense"
else:
    Ad = None
    # sparse extremal eigenvalues. Build explicit sparse operators so shift-invert
    # (sigma=0) can factorize -> fast & reliable, unlike a matrix-free LinearOperator.
    lm = float(spla.eigsh(A, k=1, which="LM", return_eigenvectors=False)[0])
    ls = float(spla.eigsh(A, k=1, sigma=0.0, which="LM", return_eigenvectors=False)[0])
    ev = np.array([ls, lm])
    Dm12_mat = sp.diags(Dm12)
    Aj_sp = (Dm12_mat @ A @ Dm12_mat).tocsr()   # explicit sparse D^{-1/2} A D^{-1/2}
    ljm = float(spla.eigsh(Aj_sp, k=1, which="LM", return_eigenvectors=False)[0])
    ljs = float(spla.eigsh(Aj_sp, k=1, sigma=0.0, which="LM", return_eigenvectors=False)[0])
    evj = np.array([ljs, ljm])
    res["spectrum_method"] = "sparse_extremal"
res["lam_min"] = float(ev[0]); res["lam_max"] = float(ev[-1])
res["kappa_A"] = float(ev[-1] / ev[0])
res["kappa_jacobi"] = float(evj[-1] / evj[0])
res["lam_min_jac"] = float(evj[0]); res["lam_max_jac"] = float(evj[-1])

# AMG-preconditioned spectrum proxy: eigs of M^{-1}A
evamg = None
try:
    import pyamg
    ml = pyamg.smoothed_aggregation_solver(A.tocsr(), max_coarse=10)
    M = ml.aspreconditioner()  # LinearOperator
    if n <= DENSE_CAP:
        MA = np.zeros((n, n))
        for j in range(n):
            MA[:, j] = M.matvec(Ad[:, j])
        evamg = np.clip(np.sort(np.real(np.linalg.eigvals(MA))), 1e-300, None)
        res["kappa_amg"] = float(evamg[-1] / evamg[0])
    else:
        # For large n, an exact λmin of the nonsymmetric M^{-1}A via shift-invert
        # needs a factorization we don't have. Estimate the effective condition
        # number from preconditioned-CG's Lanczos tridiagonal (Ritz values) instead
        # -- this is exactly the spectrum CG "sees".
        from scipy.sparse.linalg import cg as _cg
        # Lanczos/Ritz via the preconditioned CG three-term recurrence:
        # run PCG and collect Ritz values of the projected operator.
        # Simpler robust proxy: power iteration for λmax(M^{-1}A), and
        # inverse-free λmin proxy = 1/λmax(A^{-1} M) is unavailable; so we
        # report κ_eff measured from PCG iteration count vs theory later, and
        # give λmax(M^{-1}A) exactly (power iteration).
        hi = float(np.real(spla.eigs(spla.LinearOperator((n, n),
                    matvec=lambda x: M.matvec(A @ x)),
                    k=1, which="LM", return_eigenvectors=False, maxiter=2000)[0]))
        res["lam_max_amg"] = hi
        res["kappa_amg"] = None  # exact κ not computed for large n; see PCG iters
    res["amg_levels"] = int(len(ml.levels))
    res["amg_grid_complexity"] = float(ml.grid_complexity())
    res["amg_op_complexity"] = float(ml.operator_complexity())
except Exception as e:
    res["amg_error"] = repr(e)
    print("[amg] spectrum proxy failed:", e)

print(f"[spec] kappa_A={res['kappa_A']:.3e}  kappa_jacobi={res['kappa_jacobi']:.3e}"
      + (f"  kappa_amg={res.get('kappa_amg',float('nan')):.3e}" if evamg is not None else ""))

# ---------- 3) benchmark ----------
b = F.copy()
x_ref = spla.spsolve(A.tocsc(), b)
nref = np.linalg.norm(x_ref)

def relerr(x):
    return float(np.linalg.norm(x - x_ref) / (nref + 1e-300))

def run_cg(M=None, label=""):
    hist = []
    def cb(xk):
        hist.append(np.linalg.norm(b - A @ xk) / np.linalg.norm(b))
    t = time.perf_counter()
    x, info = spla.cg(A, b, rtol=1e-10, atol=0.0, maxiter=5000, M=M, callback=cb)
    dt = time.perf_counter() - t
    return {"label": label, "solver": "CG", "iters": len(hist), "info": int(info),
            "time_s": dt, "relerr": relerr(x)}, hist

def run_gmres(M=None, label=""):
    hist = []
    def cb(pr):  # scipy gmres callback gives residual norm (rel) by default
        hist.append(float(pr))
    t = time.perf_counter()
    x, info = spla.gmres(A, b, rtol=1e-10, atol=0.0, restart=80, maxiter=5000,
                         M=M, callback=cb, callback_type="pr_norm")
    dt = time.perf_counter() - t
    return {"label": label, "solver": "GMRES", "iters": len(hist), "info": int(info),
            "time_s": dt, "relerr": relerr(x)}, hist

bench = []
histories = {}

# no precond
r, h = run_cg(None, "CG none"); bench.append(r); histories["CG none"] = h
r, h = run_gmres(None, "GMRES none"); bench.append(r); histories["GMRES none"] = h

# Jacobi
Minv = sp.diags(1.0 / dsafe)
r, h = run_cg(spla.aslinearoperator(Minv), "CG Jacobi"); bench.append(r); histories["CG Jacobi"] = h

# ILU (as approx IC for SPD)
try:
    ilu = spla.spilu(A.tocsc(), drop_tol=1e-4, fill_factor=10)
    Milu = spla.LinearOperator((n, n), ilu.solve)
    r, h = run_cg(Milu, "CG ILU"); bench.append(r); histories["CG ILU"] = h
except Exception as e:
    bench.append({"label": "CG ILU", "error": repr(e)})

# AMG-SA
try:
    import pyamg
    ml_sa = pyamg.smoothed_aggregation_solver(A.tocsr(), max_coarse=10)
    r, h = run_cg(ml_sa.aspreconditioner(), "CG AMG-SA"); bench.append(r); histories["CG AMG-SA"] = h
except Exception as e:
    bench.append({"label": "CG AMG-SA", "error": repr(e)})

# AMG-RS
try:
    import pyamg
    ml_rs = pyamg.ruge_stuben_solver(A.tocsr(), max_coarse=10)
    r, h = run_cg(ml_rs.aspreconditioner(), "CG AMG-RS"); bench.append(r); histories["CG AMG-RS"] = h
except Exception as e:
    bench.append({"label": "CG AMG-RS", "error": repr(e)})

# AMG standalone as direct-ish solver (V-cycles to tol)
try:
    import pyamg
    ml_sa2 = pyamg.smoothed_aggregation_solver(A.tocsr(), max_coarse=10)
    resid = []
    t = time.perf_counter()
    x = ml_sa2.solve(b, tol=1e-10, residuals=resid, accel=None, maxiter=200)
    dt = time.perf_counter() - t
    bench.append({"label": "AMG-SA standalone", "solver": "Vcycle", "iters": len(resid),
                  "info": 0, "time_s": dt, "relerr": relerr(x)})
    histories["AMG-SA standalone"] = [r/ resid[0] for r in resid] if resid else []
except Exception as e:
    bench.append({"label": "AMG-SA standalone", "error": repr(e)})

res["bench"] = bench
for x in bench:
    if "iters" in x:
        print(f"  {x['label']:20s} iters={x['iters']:4d} time={x['time_s']*1e3:7.1f}ms "
              f"relerr={x['relerr']:.2e} info={x.get('info')}")
    else:
        print(f"  {x['label']:20s} ERROR {x.get('error')}")

# ---------- 4) plots ----------
# spectrum plot
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
ax = axs[0]
if res.get("spectrum_method") == "dense":
    idx = np.arange(1, n + 1)
    ax.semilogy(idx, ev, ".", ms=3, label=f"A  (κ={res['kappa_A']:.1e})")
    ax.semilogy(idx, evj, ".", ms=3, label=f"Jacobi  (κ={res['kappa_jacobi']:.1e})")
    if evamg is not None:
        ax.semilogy(idx, evamg, ".", ms=3, label=f"AMG-SA  (κ={res['kappa_amg']:.1e})")
    ax.set_xlabel("index (sorted)"); ax.set_ylabel("eigenvalue")
else:
    # only extremal eigenvalues available: show [lam_min, lam_max] bands + kappa
    labels = ["A", "Jacobi"] + (["AMG-SA"] if evamg is not None else [])
    los = [res["lam_min"], res["lam_min_jac"]] + ([res["lam_min_amg"]] if evamg is not None else [])
    his = [res["lam_max"], res["lam_max_jac"]] + ([res["lam_max_amg"]] if evamg is not None else [])
    kaps = [res["kappa_A"], res["kappa_jacobi"]] + ([res.get("kappa_amg")] if evamg is not None else [])
    xs = np.arange(len(labels))
    for xi, lo, hi, lab in zip(xs, los, his, labels):
        ax.plot([xi, xi], [lo, hi], "-o", lw=3, ms=6, label=lab)
    ax.set_yscale("log"); ax.set_xticks(xs)
    ax.set_xticklabels([f"{l}\nκ={k:.1e}" for l, k in zip(labels, kaps)], fontsize=8)
    ax.set_ylabel("eigenvalue [λmin, λmax]")
ax.set_title(f"Phase operator spectrum  {tag}  n={n}  max_d={res['max_d']:.3f}")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

ax = axs[1]
for lab, h in histories.items():
    if h and len(h) > 0:
        ax.semilogy(np.arange(1, len(h) + 1), h, label=f"{lab} ({len(h)})", lw=1.3)
ax.set_xlabel("iteration"); ax.set_ylabel("rel. residual")
ax.set_title("Krylov convergence (rtol=1e-10)")
ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/spectrum_and_convergence_{tag}.png", dpi=130)
print(f"[plot] wrote {OUT}/spectrum_and_convergence_{tag}.png")

# eigenvalue histogram (density) for A vs Jacobi -- only when full spectrum available
if res.get("spectrum_method") == "dense":
    fig2, ax2 = plt.subplots(figsize=(7, 4.2))
    ax2.hist(np.log10(ev), bins=60, alpha=0.5, label="A")
    ax2.hist(np.log10(evj), bins=60, alpha=0.5, label="Jacobi-scaled")
    if evamg is not None:
        ax2.hist(np.log10(evamg), bins=60, alpha=0.5, label="AMG-preconditioned")
    ax2.set_xlabel("log10(eigenvalue)"); ax2.set_ylabel("count")
    ax2.set_title(f"Eigenvalue distribution  {tag}")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f"{OUT}/eig_hist_{tag}.png", dpi=130)
    print(f"[plot] wrote {OUT}/eig_hist_{tag}.png")

with open(f"{OUT}/analysis_{tag}.json", "w") as f:
    json.dump(res, f, indent=2)
print(f"[json] wrote {OUT}/analysis_{tag}.json")
