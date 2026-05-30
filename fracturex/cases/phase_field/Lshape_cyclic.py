"""L-shaped panel under cyclic loading — standard-FEM phase-field case.

Role
----
This is the **operator-learning correctness test case** (see
``docs/plan_operator_learning.md`` §"L 型标准 FEM 校验算例"). It deliberately
lives outside the Hu-Zhang data pipeline:

* The L-shape has a **re-entrant corner** at ``(250, 250)`` — a stress
  singularity. The Hu-Zhang mixed element would need AFEM refinement there,
  producing a per-step DOF layout that changes across the load history. That
  breaks the static field-layout assumption the Hu-Zhang dataset path relies
  on, so we cannot generate this geometry through
  ``fracturex/tests/case_runners/model0_runner.py``.
* It can, however, be solved with the **standard displacement-based FEM**
  (``fracturex.phasefield.main_solve.MainSolve`` with ``HybridModel``), exactly
  like ``square_domian_with_fracture.py``.

Why it tests *correctness* of an operator-learning surrogate:

* The load schedule is **cyclic** (0 -> 0.3 -> -0.2 -> 1.0 mm). At a given
  displacement the loading branch differs from the unloading branch — the only
  thing distinguishing them is the irreversible history field ``H``. A surrogate
  that ignores history is visibly wrong on the reversal.
* The singular corner gives a **non-smooth field**, stressing a held-out
  geometry the (square, smooth-notch) training distribution never contains.

This file produces the classic L-shape force–displacement curve plus per-step
VTU; it does **not** (yet) write schema-v0.1 samples. The data-export bridge
(standard-FEM RunRecorder + an L-shape SDF domain) is sketched in the plan doc
as follow-up work.

Geometry / loading (Winkler L-shape benchmark)
----------------------------------------------
* domain: ``[0,500]^2`` mm with the lower-right quadrant removed;
* clamp: bottom edge ``y = 0`` (left foot), all components fixed;
* load:  prescribed vertical displacement at the single point ``(470, 250)``.

Material (Lamé, GPa) — passed straight through as ``lam``/``mu`` so no E/nu
round-trip is needed:
    lam = 6.16, mu = 10.95, Gc = 8.9e-5, l0 = 1.18

Run
---
    conda activate py312                 # this server's fealpy env
    PYTHONPATH=$PWD python \\
        fracturex/cases/phase_field/Lshape_cyclic.py --n 50 --max-steps 40

``n`` must keep the load point (470, 250) on a mesh node: ``500/n`` has to
divide gcd(470, 250) = 10, i.e. ``n in {50, 100, 250, 500}`` (the case asserts
this). Default ``--n 50`` is verified to converge.
"""
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

from fracturex.phasefield.main_solve import MainSolve


class LShapeCyclicModel:
    """L-shaped panel, cyclic vertical point load (standard FEM)."""

    # Lamé material (GPa) + fracture params; MainSolve accepts lam/mu directly.
    params = {'lam': 6.16, 'mu': 10.95, 'Gc': 8.9e-5, 'l0': 1.18}

    # geometry: square [0,500]^2 with the lower-right quadrant removed
    box = [0, 500, 0, 500]

    def is_force(self):
        """Cyclic prescribed-displacement schedule at the loading point (mm).

        0 -> 0.3 (301 steps), 0.3 -> -0.2 (500 steps), -0.2 -> 1.0 (1200 steps).
        The sign reversal is what makes the history field observable; a monotone
        ramp would not exercise irreversibility.
        """
        return bm.concatenate((
            bm.linspace(0, 0.3, 301, dtype=bm.float64),
            bm.linspace(0.3, -0.2, 501, dtype=bm.float64)[1:],
            bm.linspace(-0.2, 1.0, 1201, dtype=bm.float64)[1:],
        ))

    def is_force_boundary(self, p):
        """Single loading point on the right arm: (470, 250)."""
        return (bm.abs(p[..., 1] - 250) < 1e-12) & (bm.abs(p[..., 0] - 470) < 1e-5)

    def is_dirchlet_boundary(self, p):
        """Clamped bottom edge (y = 0) — the left foot of the L."""
        return bm.abs(p[..., 1]) < 1e-12

    @staticmethod
    def _removed_quadrant(p):
        # lower-right quadrant is cut out to form the L-shape
        return (p[..., 0] > 250) & (p[..., 1] < 250)

    def build_mesh(self, n=50):
        return TriangleMesh.from_box(box=self.box, nx=n, ny=n,
                                     threshold=self._removed_quadrant)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="L-shape cyclic phase-field fracture (standard FEM).")
    parser.add_argument('--degree', default=1, type=int,
                        help='Lagrange degree, default 1.')
    parser.add_argument('--maxit', default=100, type=int,
                        help='staggered max iterations per load step, default 100.')
    parser.add_argument('--backend', default='numpy', type=str)
    parser.add_argument('--model_type', default='HybridModel', type=str)
    parser.add_argument('--n', default=50, type=int,
                        help='mesh subdivisions per axis, default 50.')
    parser.add_argument('--max-steps', default=None, type=int,
                        help='truncate the cyclic schedule to this many steps '
                             '(for smoke runs); default uses the full ~2000.')
    parser.add_argument('--vtkname', default='Lshape', type=str)
    parser.add_argument('--save_vtkfile', default=True, type=bool)
    args = parser.parse_args(argv)

    bm.set_backend(args.backend)
    model = LShapeCyclicModel()
    mesh = model.build_mesh(n=args.n)
    mesh.to_vtk(fname=f'{args.vtkname}_init.vtu')

    # Guard: the load is a *single point* (470, 250). If the mesh has no node
    # there, MainSolve's force BC matches nothing and silently returns u == 0
    # (a bogus trivial solution). Fail loudly instead. The point is a node iff
    # the spacing 500/n divides gcd(470, 250) = 10, i.e. n in {50, 100, 250, 500}.
    _node = np.asarray(mesh.entity('node'))
    _n_load = int(np.asarray(model.is_force_boundary(_node)).sum())
    if _n_load < 1:
        raise SystemExit(
            f"loading point (470, 250) is not a mesh node for n={args.n} "
            f"(matched {_n_load}); the single-point load would be ignored (u=0). "
            f"Choose n so that 500/n divides 470 and 250 — e.g. n in {{50,100,250,500}}."
        )
    print(f"[Lshape] loading-point nodes matched: {_n_load}")

    disp = model.is_force()
    if args.max_steps is not None:
        disp = disp[: args.max_steps + 1]

    ms = MainSolve(mesh=mesh, material_params=model.params,
                   model_type=args.model_type)

    # prescribed vertical displacement at the loading point
    ms.add_boundary_condition('force', 'Dirichlet',
                              model.is_force_boundary, disp, 'y')
    # clamp the bottom foot
    ms.add_boundary_condition('displacement', 'Dirichlet',
                              model.is_dirchlet_boundary, 0)

    if args.save_vtkfile:
        ms.save_vtkfile(fname=args.vtkname)

    start = time.time()
    ms.solve(p=args.degree, maxit=args.maxit)
    print(f"Time: {time.time() - start:.1f}s")

    force = bm.to_numpy(ms.get_residual_force())
    disp_np = bm.to_numpy(disp)

    # Sanity: a real run must move load and accumulate history. A flat-zero
    # reaction or max_H == 0 means the trivial solution slipped through.
    max_H = float(np.nanmax(np.abs(bm.to_numpy(ms.H))))
    print(f"[Lshape] |reaction|_max={np.nanmax(np.abs(force)):.3e}  max_H={max_H:.3e}")
    if np.nanmax(np.abs(force)) == 0.0:
        print("[Lshape] WARNING: reaction is identically zero — check the load BC.")

    np.savetxt(f'{args.vtkname}_force_disp.txt',
               np.c_[disp_np, force],
               header='disp(mm)  reaction_force', comments='')

    fig, axs = plt.subplots()
    plt.plot(disp_np, force, label='Force')
    plt.xlabel('Displacement (mm)')
    plt.ylabel('Reaction force')
    plt.title('L-shape cyclic — force vs. displacement')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{args.vtkname}_force.png', dpi=300)
    print(f"wrote {args.vtkname}_force.png / {args.vtkname}_force_disp.txt")


if __name__ == '__main__':
    main()
