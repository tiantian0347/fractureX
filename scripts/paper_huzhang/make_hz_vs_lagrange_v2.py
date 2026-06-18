#!/usr/bin/env python3
# make_hz_vs_lagrange_v2.py
#
# Paper section 7.6 (V2): h-convergence of the stress in L2 for the Hu-Zhang
# mixed element (p_sigma=3) vs a standard displacement-based Lagrange element
# (p_u=2, recovered stress sigma = C:eps(u_h)), on a MANUFACTURED linear-elastic
# solution on the unit square. This gives true convergence RATES (cleaner than
# "finest Hu-Zhang as reference"): Hu-Zhang stress should attain order
# p_sigma+1 = 4, while the Lagrange recovered stress lags at order p_u = 2.
#
# Both errors use the SAME Voigt-Frobenius L2 norm (weights [1,2,1] on
# [xx,xy,yy]) so the two curves are directly comparable.
#
# MUST be run under the py312 env that has fealpy:
#   /home/gongshihua/miniconda3/envs/py312/bin/python \
#       scripts/paper_huzhang/make_hz_vs_lagrange_v2.py [--ns 4,8,16,32]
# or:  bash scripts/run_python.sh scripts/paper_huzhang/make_hz_vs_lagrange_v2.py
#
# Outputs under Frac_huzhang/figures/:
#   hz_vs_lagrange_v2.{png,pdf}     loglog stress error vs h, both elements
#   hz_vs_lagrange_v2.csv           N, h, err_hz, err_lag, rate_hz, rate_lag

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- repo + fealpy paths -----------------------------------------------------
REPO = Path("/home/gongshihua/tian/fracturex")
TESTS = REPO / "fracturex" / "tests"
for p in (str(REPO), str(TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearForm, LinearElasticIntegrator, VectorSourceIntegrator
from fealpy.fem import DirichletBC
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.decorator import cartesian
from scipy.sparse.linalg import spsolve as scipy_spsolve

# Hu-Zhang manufactured-solution machinery (proven test code)
import linear_elastic_with_huzhang as HZ
from linear_elastic_pde import LinearElasticPDE

OUT = Path("/home/gongshihua/tian/Frac_huzhang/figures")
OUT.mkdir(parents=True, exist_ok=True)

# manufactured material: the Hu-Zhang test parametrizes the *compliance* by
# (lambda0, lambda1) via sigma = (1/l0) eps + (1/l0)(l1/(l0-2 l1)) tr(eps) I.
# The matching Lame parameters for the displacement (stiffness) form
# sigma = 2 mu eps + lam tr(eps) I are:
LAMBDA0 = 4.0
LAMBDA1 = 1.0
MU = 1.0 / (2.0 * LAMBDA0)
LAM = (1.0 / LAMBDA0) * (LAMBDA1 / (LAMBDA0 - 2.0 * LAMBDA1))


def lagrange_stress_error(pde, N, p_u=2, q=12):
    """Solve displacement-based Lagrange elasticity for the manufactured pde and
    return the Voigt-Frobenius L2 error of the recovered stress."""
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=N, ny=N)
    scalar = LagrangeFESpace(mesh, p=p_u)
    space = TensorFunctionSpace(scalar, shape=(2, -1))  # (u_x, u_y)
    uh = space.function()

    material = LinearElasticMaterial(
        name="manufactured", lame_lambda=LAM, shear_modulus=MU, hypo="plane_strain"
    )

    bform = BilinearForm(space)
    bform.add_integrator(LinearElasticIntegrator(material, q=q))
    K = bform.assembly()

    @cartesian
    def source(x, index=None):
        return pde.source(x)

    lform = LinearForm(space)
    lform.add_integrator(VectorSourceIntegrator(source=source, q=q))
    F = lform.assembly()

    @cartesian
    def gd(x, index=None):
        return pde.displacement(x)

    bc = DirichletBC(space, gd=gd)
    K, F = bc.apply(K, F, uh)

    Ks = K.to_scipy().tocsr() if hasattr(K, "to_scipy") else K
    uh[:] = scipy_spsolve(Ks, bm.to_numpy(F) if hasattr(bm, "to_numpy") else np.asarray(F))

    def sigmah(bcs, index=None):
        g = uh.grad_value(bcs)              # (NC, NQ, 2, 2) = d u_i / d x_j
        exx = g[..., 0, 0]
        eyy = g[..., 1, 1]
        exy = 0.5 * (g[..., 0, 1] + g[..., 1, 0])
        tr = exx + eyy
        sxx = LAM * tr + 2.0 * MU * exx
        syy = LAM * tr + 2.0 * MU * eyy
        sxy = 2.0 * MU * exy
        return bm.stack([sxx, sxy, syy], axis=-1)  # [xx, xy, yy]

    return HZ.l2_error_sigma(mesh, sigmah, pde.stress, q=max(q, 2 * p_u + 6))


def rates(h, err):
    h = np.asarray(h, float); err = np.asarray(err, float)
    r = np.full_like(err, np.nan)
    r[1:] = np.log(err[1:] / err[:-1]) / np.log(h[1:] / h[:-1])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="4,8,16,32", help="comma-separated mesh sizes N (nx=ny=N)")
    ap.add_argument("--p-hz", type=int, default=3, help="Hu-Zhang stress degree")
    ap.add_argument("--p-u", type=int, default=2, help="Lagrange displacement degree")
    args = ap.parse_args()
    Ns = [int(x) for x in args.ns.split(",") if x.strip()]

    x, y = HZ.symbols("x y")
    pi = float(np.pi)
    u = [(HZ.sin(pi * x) * HZ.sin(pi * y)) ** 2, (HZ.sin(pi * x) * HZ.sin(pi * y)) ** 2]
    pde = LinearElasticPDE(u, LAMBDA0, LAMBDA1)

    hs, e_hz, e_lag = [], [], []
    for N in Ns:
        # Hu-Zhang mixed stress (proven solve from the test module)
        sigmah, _ = HZ.solve(pde, N, args.p_hz)
        mesh = sigmah.space.mesh
        ehz = HZ.l2_error_sigma(mesh, sigmah, pde.stress, q=2 * args.p_hz + 6)
        # Lagrange recovered stress
        elag = lagrange_stress_error(pde, N, p_u=args.p_u)
        hs.append(1.0 / N); e_hz.append(ehz); e_lag.append(elag)
        print(f"N={N:4d}  h={1.0/N:.4f}  err_hz={ehz:.4e}  err_lag={elag:.4e}")

    r_hz, r_lag = rates(hs, e_hz), rates(hs, e_lag)

    with open(OUT / "hz_vs_lagrange_v2.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "h", "err_hz_sigma", "err_lag_sigma", "rate_hz", "rate_lag"])
        for i, N in enumerate(Ns):
            w.writerow([N, f"{hs[i]:.6e}", f"{e_hz[i]:.6e}", f"{e_lag[i]:.6e}",
                        f"{r_hz[i]:.3f}", f"{r_lag[i]:.3f}"])

    hs = np.array(hs)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.loglog(hs, e_hz, "o-", color="#1f3b73", lw=1.8, label=f"Hu--Zhang $\\sigma_h$ ($p={args.p_hz}$)")
    ax.loglog(hs, e_lag, "s-", color="#c0392b", lw=1.8, label=f"Lagrange recovered $\\sigma$ ($p={args.p_u}$)")
    # reference slopes
    c4 = e_hz[0] / hs[0] ** (args.p_hz + 1)
    c2 = e_lag[0] / hs[0] ** args.p_u
    ax.loglog(hs, c4 * hs ** (args.p_hz + 1), "k--", lw=0.9, alpha=0.7,
              label=f"$O(h^{args.p_hz + 1})$")
    ax.loglog(hs, c2 * hs ** args.p_u, "k:", lw=0.9, alpha=0.7, label=f"$O(h^{args.p_u})$")
    ax.set_xlabel(r"mesh size $h$")
    ax.set_ylabel(r"$\|\sigma-\sigma_h\|_{L^2}$")
    ax.set_title("V2: stress $L^2$ convergence (manufactured)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"hz_vs_lagrange_v2.{ext}", dpi=200)
    plt.close(fig)
    print("asymptotic rate  Hu-Zhang:", f"{r_hz[-1]:.2f}", " Lagrange:", f"{r_lag[-1]:.2f}")
    print("wrote", OUT / "hz_vs_lagrange_v2.pdf")


if __name__ == "__main__":
    main()
