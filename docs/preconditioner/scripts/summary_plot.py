#!/usr/bin/env python3
"""Cross-mesh summary: iteration count & condition number vs problem size.
Reads analysis_{tag}.json for the three mesh levels -> the mesh-independence figure.
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/Users/tian00/repository/fractureX/docs/preconditioner"
tags = ["m0h1", "m0h2", "m0h3"]
data = {t: json.load(open(f"{OUT}/analysis_{t}.json")) for t in tags}
ns = [data[t]["n"] for t in tags]

def iters_of(t, label):
    for b in data[t]["bench"]:
        if b.get("label") == label and "iters" in b:
            return b["iters"]
    return None

methods = ["CG none", "CG Jacobi", "CG AMG-SA", "CG AMG-RS", "CG ILU"]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))

ax = axs[0]
for m in methods:
    ys = [iters_of(t, m) for t in tags]
    ax.plot(ns, ys, "-o", label=m)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("problem size  n (phase dof)")
ax.set_ylabel("Krylov iterations to rtol=1e-10")
ax.set_title("Mesh-(in)dependence of iteration count\n(real post-crack systems, model0 h1/h2/h3)")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

ax = axs[1]
kA = [data[t]["kappa_A"] for t in tags]
kJ = [data[t]["kappa_jacobi"] for t in tags]
ax.plot(ns, kA, "-o", label="κ(A)  (unpreconditioned)")
ax.plot(ns, kJ, "-o", label="κ(D^{-1}A)  (Jacobi)")
# O(h^-2) ~ O(n) reference through first point
ref = kA[0] * np.array(ns) / ns[0]
ax.plot(ns, ref, "k--", lw=1, label="O(n) ~ O(h^{-2}) ref")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("problem size  n"); ax.set_ylabel("condition number κ")
ax.set_title("Condition number growth under refinement")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(f"{OUT}/mesh_independence_summary.png", dpi=140)
print(f"[plot] wrote {OUT}/mesh_independence_summary.png")

# emit a compact markdown table to stdout for the doc
print("\n| n | κ(A) | κ(Jacobi) | " + " | ".join(methods) + " |")
print("|" + "---|" * (2 + 1 + len(methods)))
for t in tags:
    row = [data[t]["n"], f"{data[t]['kappa_A']:.0f}", f"{data[t]['kappa_jacobi']:.1f}"]
    row += [iters_of(t, m) for m in methods]
    print("| " + " | ".join(str(x) for x in row) + " |")
