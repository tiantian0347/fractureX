#!/usr/bin/env python3
"""D13 feature-dump side-channel: extract per-coarse-dof dimensionless features phi
from frozen damage fields, reusing D12 phase-field checkpoints with ZERO extra
simulation. One sample = one (checkpoint) -> (NN, N_FEATURES) feature block.

The damage field is loaded into a freshly built discretization (same recipe as
check_localized_baselines.py); we only rebuild geometry + restore the frozen ``d``,
never advance the solver. Output is an .npz per checkpoint with the feature matrix,
node coords and provenance, ready for the learned coarse-space dataset (datasets.py).

With ``--with-worst-mode-label`` the script ALSO assembles the localized operator and
computes the spectral worst-mode amplitude target (spectral_labels.worst_mode_amplitude)
per coarse node, stored as ``target`` in the npz. This is the supervised label the
amplitude model regresses to (so online inference reproduces a worst-mode-like Phi with
NO power iteration). It costs one assembly (~minutes), hence it is opt-in.

Usage:
  dump_features.py --case model0 --hmin 0.025 --eps-g 1e-6 \
      [--with-worst-mode-label] --out results/phasefield/_precond_features \
      <ckpt1.npz> [<ckpt2.npz> ...]

Env (see scripts/paper_huzhang/env.sh):
  PYTHONPATH=<repo>  OMP_NUM_THREADS=1  OPENBLAS_NUM_THREADS=1
  python = conda env py312
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from fealpy.backend import backend_manager as bm
from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.ml.coarse_features import extract_coarse_features, FEATURE_NAMES

# Load level for the worst-mode label assembly (matches the localized-baseline recipe).
_LABEL_LOAD = 0.092


class _Model0Mat:
    E = 200.0
    nu = 0.2
    Gc = 1.0
    l0 = 0.02

    @property
    def mu(self):
        return self.E / (2 * (1 + self.nu))

    @property
    def lam(self):
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))


def _build_case(case_name: str, hmin: float):
    """Return (case, mesh) for a supported example. Mirrors the D12 baseline recipe."""
    name = case_name.strip().lower()
    if name in ("model0", "model0_circular_notch"):
        case = Model0CircularNotchCase(_model=_Model0Mat(), hmin=hmin)
        return case, case.make_mesh()
    raise ValueError(
        f"Unsupported --case {case_name!r}; only 'model0' wired so far "
        "(add square/model2 cases here as needed)."
    )


def _worst_mode_label(discr, mesh, dmg):
    """Assemble the localized operator and compute the spectral worst-mode amplitude.

    Returns ``(target_nn,)`` per coarse-node label in [0,1], or raises on failure. This
    is the only expensive path (one assembly + power iteration on the Schur block).
    """
    from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
    from fracturex.utilfuc.linear_solvers import (
        as_scipy_csr, _extract_mechanical_blocks, _diag_inv_stress_block,
        _approximate_schur_spd, _get_or_build_auxspace_pi_operators,
    )
    from fracturex.ml.spectral_labels import worst_mode_amplitude

    asm = HuZhangElasticAssembler(discr, discr._case, dmg, formulation="standard",
                                  assembly_parallel=False)
    asm.begin_load_step(_LABEL_LOAD)
    sys_e = asm.assemble(_LABEL_LOAD)
    A_ = as_scipy_csr(sys_e.A)
    m = int(discr.gdof_sigma)
    _, M, B = _extract_mechanical_blocks(A_, m)
    D_inv = _diag_inv_stress_block(M)
    S = _approximate_schur_spd(M, B, D_inv)
    cached = _get_or_build_auxspace_pi_operators(mesh, discr.space_u, 5)
    sgdof = int(cached["sgdof"])
    PI_s = cached["PI_s"]
    Sb = S[:sgdof, :sgdof].tocsr()  # component-0 Schur block (geometry shared per comp)
    return worst_mode_amplitude(Sb, PI_s, iters=60)


def process_one(ckpt: Path, case_name: str, hmin: float, eps_g: float, out_dir: Path,
                with_worst_mode: bool = False) -> dict:
    """Load a checkpoint's frozen d, extract features, write an .npz.

    No solve unless ``with_worst_mode`` (then one assembly to build the worst-mode label).
    """
    z = np.load(ckpt)
    d_real = np.asarray(z["d"], dtype=float)
    maxd = float(d_real.max())

    dmg = PhaseFieldDamageModel(
        density_type="AT2", degradation_type="quadratic",
        split="hybrid", eps_g=eps_g, debug=False,
    )
    case, mesh = _build_case(case_name, hmin)
    discr = HuZhangDiscretization(case, p=3, damage_p=2, use_relaxation=True).build(mesh=mesh)
    dmg.on_build(discr, discr.state, case)
    discr._case = case  # stash so _worst_mode_label can recover it

    ndof_d = int(discr.space_d.number_of_global_dofs())
    if d_real.size != ndof_d:
        raise RuntimeError(
            f"{ckpt.name}: d dof mismatch ckpt {d_real.size} vs discr {ndof_d} "
            f"(wrong --hmin {hmin}?)"
        )
    discr.state.d[:] = bm.asarray(d_real)

    cf = extract_coarse_features(mesh, dmg, discr.state, l0=float(dmg.l0), eps_g=1e-10)

    extra = {}
    if with_worst_mode:
        target = _worst_mode_label(discr, mesh, dmg)
        if target.shape[0] != cf.phi.shape[0]:
            raise RuntimeError(
                f"worst-mode target NN {target.shape[0]} != features NN {cf.phi.shape[0]}"
            )
        extra["target"] = np.asarray(target, dtype=np.float32)
        extra["target_kind"] = np.asarray("worst_mode_amplitude")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"feat_{case_name}_h{hmin:g}_{ckpt.stem}.npz"
    # Convert at the numpy/npz I/O boundary so this works on any FEALPy backend
    # (torch/jax tensors lack numpy's .astype).
    np.savez_compressed(
        out,
        phi=bm.to_numpy(cf.phi).astype(np.float32),
        feature_names=np.asarray(FEATURE_NAMES),
        node=bm.to_numpy(cf.node).astype(np.float32),
        d_node=bm.to_numpy(cf.d).astype(np.float32),
        g_node=bm.to_numpy(cf.g).astype(np.float32),
        l0=np.float64(cf.l0),
        hmin=np.float64(hmin),
        eps_g=np.float64(eps_g),
        maxd=np.float64(maxd),
        case=np.asarray(case_name),
        source_ckpt=np.asarray(str(ckpt)),
        nn=np.int64(cf.phi.shape[0]),
        **extra,
    )
    return {"out": out, "nn": cf.phi.shape[0], "maxd": maxd,
            "labelled": bool(with_worst_mode)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--case", default="model0")
    ap.add_argument("--hmin", type=float, default=0.025)
    ap.add_argument("--eps-g", type=float, default=1e-6)
    ap.add_argument("--out", type=Path,
                    default=_REPO / "results/phasefield/_precond_features")
    ap.add_argument("--with-worst-mode-label", action="store_true",
                    help="also assemble the operator and store the spectral worst-mode "
                         "amplitude target (one assembly per ckpt, ~minutes).")
    args = ap.parse_args()

    print(f"{'checkpoint':>28} {'NN':>8} {'maxd':>8} {'lbl':>4}   out", flush=True)
    for ckpt in args.checkpoints:
        t0 = time.perf_counter()
        r = process_one(ckpt, args.case, args.hmin, args.eps_g, args.out,
                        with_worst_mode=args.with_worst_mode_label)
        dt = time.perf_counter() - t0
        print(f"{ckpt.name:>28} {r['nn']:>8d} {r['maxd']:>8.4f} "
              f"{'Y' if r['labelled'] else '-':>4}   {r['out'].name}  ({dt:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
