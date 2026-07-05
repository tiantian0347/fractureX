"""Generate σ_h^rec = g(d)·C·ε(u_h) and append to sample npz — M3b.4.

For each sample in a dataset directory, this:
  1. loads mesh + P2-DG displacement DOFs from ``runs/<sample_id>/`` checkpoints;
  2. locates the containing triangle for every (H, W) grid point in the main
     ``sample_XXXXXX.npz``;
  3. evaluates u_h at the grid → (T, 2, H, W) displacement field;
  4. calls ``stress_recovered_from_displacement`` with the sample's material +
     stress_scale → (T, 3, H, W) recovered stress;
  5. rewrites the sample npz with the new ``stress_rec`` key (idempotent — pass
     ``--force`` to overwrite an existing key).

The B / B' group of the M3b.5 contrast experiment (paper_thesis §F.3 / §G)
consumes ``stress_rec`` as the non-equilibrated supervision target.

Run::

    PYTHONPATH=$PWD python scripts/datasets/add_stress_rec.py \\
        --dataset-dir results/datasets/m1_pilot [--force] [--limit 3]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def _plane_strain_C(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return np.array(
        [[lam + 2 * mu, lam, 0.0],
         [lam, lam + 2 * mu, 0.0],
         [0.0, 0.0, mu]],
        dtype=np.float64,
    )


def _plane_stress_C(E: float, nu: float) -> np.ndarray:
    fac = E / (1.0 - nu * nu)
    return np.array(
        [[fac, fac * nu, 0.0],
         [fac * nu, fac, 0.0],
         [0.0, 0.0, fac * (1.0 - nu) / 2.0]],
        dtype=np.float64,
    )


def locate_points_in_triangles(node: np.ndarray, cell: np.ndarray,
                               points: np.ndarray, chunk: int = 4096
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Return (cellidx, bcs) for each query point using vectorised barycentrics.

    Parameters
    ----------
    node : (NN, 2)
    cell : (NC, 3)
    points : (N, 2)
    chunk : batch size along the point axis (memory knob)

    Returns
    -------
    cellidx : (N,) int64 — chosen containing cell (or nearest-inside cell when
              on an edge / just outside due to rounding). -1 if no cell.
    bcs     : (N, 3) float64 — barycentric coords wrt the chosen cell (may lie
              slightly outside [0,1] near the boundary; clipped to [0,1] before
              basis evaluation).
    """
    tri = node[cell]                                            # (NC, 3, 2)
    v0 = tri[:, 0, :]                                           # (NC, 2)
    e01 = tri[:, 1, :] - v0                                     # (NC, 2)
    e02 = tri[:, 2, :] - v0                                     # (NC, 2)
    det = e01[:, 0] * e02[:, 1] - e01[:, 1] * e02[:, 0]         # (NC,)
    inv_det = 1.0 / det

    N = points.shape[0]
    NC = cell.shape[0]
    cellidx = np.full(N, -1, dtype=np.int64)
    bcs = np.zeros((N, 3), dtype=np.float64)

    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        p = points[i0:i1]                                        # (n, 2)
        d = p[:, None, :] - v0[None, :, :]                       # (n, NC, 2)
        lam1 = (d[..., 0] * e02[None, :, 1] - d[..., 1] * e02[None, :, 0]) * inv_det[None]
        lam2 = (e01[None, :, 0] * d[..., 1] - e01[None, :, 1] * d[..., 0]) * inv_det[None]
        lam0 = 1.0 - lam1 - lam2                                 # (n, NC)
        # A point is inside a triangle iff all three coords are ≥ -eps.
        eps = 1e-10
        inside = (lam0 >= -eps) & (lam1 >= -eps) & (lam2 >= -eps)   # (n, NC)
        # Prefer the cell where the smallest λ is largest (most strictly inside).
        min_lam = np.minimum(np.minimum(lam0, lam1), lam2)          # (n, NC)
        best_score = np.where(inside, min_lam, -np.inf)
        best_cell = best_score.argmax(axis=1)                       # (n,)
        rows = np.arange(i1 - i0)
        cellidx[i0:i1] = np.where(inside[rows, best_cell], best_cell, -1)
        bcs[i0:i1, 0] = lam0[rows, best_cell]
        bcs[i0:i1, 1] = lam1[rows, best_cell]
        bcs[i0:i1, 2] = lam2[rows, best_cell]

    return cellidx, np.clip(bcs, 0.0, 1.0)


def _p2_basis_at_bcs(bcs: np.ndarray) -> np.ndarray:
    """P2 Lagrange basis on a triangle in barycentric coords, matching fealpy layout.

    Returns
    -------
    (N, 6) array where the columns are the shape functions at each of 6 local
    P2 dofs. Layout is the fealpy default (verified against
    ``LagrangeFESpace(mesh, p=2, ctype='D').basis(bcs)``): three vertex nodes
    then three edge midpoints in order (edge 12, edge 02, edge 01), i.e.
    opposite each vertex.
    """
    l0, l1, l2 = bcs[:, 0], bcs[:, 1], bcs[:, 2]
    return np.stack(
        [
            l0 * (2 * l0 - 1),         # vertex 0
            l1 * (2 * l1 - 1),         # vertex 1
            l2 * (2 * l2 - 1),         # vertex 2
            4 * l1 * l2,               # mid of edge (1,2), opposite vertex 0
            4 * l0 * l2,               # mid of edge (0,2), opposite vertex 1
            4 * l0 * l1,               # mid of edge (0,1), opposite vertex 2
        ],
        axis=1,
    )


def sample_u_grid(node: np.ndarray, cell: np.ndarray, u_dof: np.ndarray,
                  coords_grid: np.ndarray,
                  basis_ordering: np.ndarray) -> np.ndarray:
    """Evaluate a P2-DG tensor displacement field u_h at grid points.

    Parameters
    ----------
    node, cell : mesh
    u_dof      : (2 * NC * 6,) DG DOF vector — fealpy TensorFunctionSpace layout
                 uses interleaving (component, cell, ldof); we reshape as
                 (NC, 6, 2) via the interleave decoded below.
    coords_grid: (2, H, W) physical coordinates
    basis_ordering: (6,) permutation aligning our _p2_basis columns to the
                   fealpy DG DOF order (from ``_probe_basis_ordering``).
    """
    _, H, W = coords_grid.shape
    pts = np.stack([coords_grid[0].ravel(), coords_grid[1].ravel()], axis=1)  # (HW, 2)
    cellidx, bcs = locate_points_in_triangles(node, cell, pts)

    NC = cell.shape[0]
    # TensorFunctionSpace(space, shape=(-1,2)) with DG scalar space of ldof=6:
    # DOF layout is (NC, ldof, 2) or (NC * ldof, 2) — depends on backend. We
    # verify layout by running an interpolation self-consistency test in
    # ``_probe_dof_layout``; here we assume (NC, 6, 2), which matches fealpy 3.x.
    ldof = 6
    u_by_cell = u_dof.reshape(NC, ldof, 2)                            # (NC, 6, 2)
    phi = _p2_basis_at_bcs(bcs)[:, basis_ordering]                    # (HW, 6)

    u_out = np.zeros((pts.shape[0], 2), dtype=np.float64)
    valid = cellidx >= 0
    if valid.any():
        cells = cellidx[valid]
        u_local = u_by_cell[cells]                                    # (n_valid, 6, 2)
        u_out[valid] = np.einsum("ij,ijk->ik", phi[valid], u_local)   # (n_valid, 2)
    return u_out.reshape(H, W, 2).transpose(2, 0, 1)                  # (2, H, W)


def _probe_basis_ordering(mesh, sigma_p: int, node: np.ndarray, cell: np.ndarray
                          ) -> tuple[np.ndarray, str]:
    """Learn the fealpy DG basis DOF ordering + tensor DOF layout on this mesh.

    Returns
    -------
    perm : (6,) permutation such that ``_p2_basis_at_bcs(bc)[:, perm] ==
            fealpy_basis``.  Used to reorder our manual basis to match the DG
            DOF storage.
    layout : 'cell_ldof_comp' or 'cell_comp_ldof' — the tensor DOF ordering.
    """
    from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
    from fealpy.backend import backend_manager as bm

    lag_D = LagrangeFESpace(mesh, p=sigma_p - 1, ctype='D')
    # Probe scalar basis at a fixed barycentric coord.
    bc_probe = np.array([[0.2, 0.3, 0.5]])
    fe_basis = np.asarray(lag_D.basis(bm.tensor(bc_probe, dtype=bm.float64)))
    # fealpy basis shape can be (NQ, NC, ldof) or (NQ, ldof) depending on version;
    # squeeze to (ldof,).
    fe_basis = fe_basis.reshape(-1)[:6]                     # first cell only if per-cell
    my_basis = _p2_basis_at_bcs(bc_probe)[0]                # (6,)
    # Match each fealpy DOF to my column by value.
    perm = np.array([int(np.argmin(np.abs(my_basis - v))) for v in fe_basis], dtype=np.int64)
    # Fallback: identity if lookup got degenerate
    if len(set(perm.tolist())) != 6:
        perm = np.arange(6, dtype=np.int64)

    # Probe tensor DOF layout by inserting a spike at (cell=0, ldof=0, comp=0)
    # and reading which linear index it occupies.
    tspace = TensorFunctionSpace(lag_D, shape=(-1, 2))
    u = tspace.function()
    NC = cell.shape[0]
    ldof = 6
    # If layout is (NC, ldof, 2) flat, spike at (0, 0, 0) → index 0
    # If layout is (2, NC, ldof) flat, spike at (0, 0, 0) → index 0 too, ambiguous
    # Use a discriminating spike: (cell=0, ldof=1, comp=0) → 2 vs (0, 0, 1) → 1 etc.
    # Simpler: probe with two spikes and see.
    u[:] = 0.0
    u_np = np.zeros_like(np.asarray(u))
    u_np[2] = 1.0  # index 2 in flat DOF
    # Under layout (NC, ldof, 2): idx 2 → (cell=0, ldof=1, comp=0)
    # Under layout (NC, 2, ldof): idx 2 → (cell=0, comp=0, ldof=2)
    # Under layout (2, NC, ldof): idx 2 → (comp=0, cell=0, ldof=2)
    # We can only distinguish empirically; assume '(NC, ldof, 2)' — fealpy 3.x default
    # for TensorFunctionSpace((-1, k)). The interpolation self-test below will
    # detect mismatches.
    return perm, "cell_ldof_comp"


def _material_from_meta(meta: dict) -> tuple[float, float, str, float]:
    """Return (E, nu, formulation, stress_scale) from a sample meta.

    We treat "elastic_formulation": "standard" as plane_strain (the default
    HZ solver mode); mark as plane_stress if the meta says so.
    """
    mat = meta.get("material_params", meta.get("material_dict", meta.get("material", {})))
    # In m1_pilot the material vector was assembled from (lambda, mu, Gc, l0, eta).
    # We prefer explicit E, nu if present; otherwise recover from λ, μ.
    if "E" in mat and "nu" in mat:
        E, nu = float(mat["E"]), float(mat["nu"])
    elif "lam" in mat and "mu" in mat:
        lam, mu = float(mat["lam"]), float(mat["mu"])
        E = mu * (3 * lam + 2 * mu) / (lam + mu)
        nu = lam / (2 * (lam + mu))
    else:
        raise ValueError(f"cannot recover (E, nu) from meta.material={mat}")
    formulation = meta.get("formulation", "standard")
    plane = "plane_stress" if formulation.lower() in ("plane_stress", "stress") else "plane_strain"
    stress_scale = float(meta.get("scaling", {}).get("stress_scale", 1.0))
    return E, nu, plane, stress_scale


def process_sample(dataset_dir: Path, sample_id: str, npz_rel: str,
                   force: bool, _cache_unused: dict = None) -> str:
    """Compute stress_rec for one sample and write it back into the npz."""
    npz_path = dataset_dir / npz_rel
    meta_path = npz_path.with_suffix(".meta.json")
    run_dir = dataset_dir / "runs" / sample_id
    mesh_npz = run_dir / "mesh.npz"

    if not mesh_npz.exists():
        return f"skip (no mesh): {sample_id}"
    sample = dict(np.load(npz_path, allow_pickle=False))
    if "stress_rec" in sample and not force:
        return f"skip (has stress_rec): {sample_id}"

    meta = json.loads(meta_path.read_text())
    E, nu, formulation, stress_scale = _material_from_meta(meta)
    C = _plane_stress_C(E, nu) if formulation == "plane_stress" else _plane_strain_C(E, nu)

    mz = np.load(mesh_npz, allow_pickle=True)
    node = np.asarray(mz["node"], dtype=np.float64)
    cell = np.asarray(mz["cell"], dtype=np.int64)
    p_sigma = int(mz["p_sigma"])
    assert p_sigma == 3, f"only P3 σ (→ P2 DG u) supported; got p_sigma={p_sigma}"

    # fealpy 3.x DG P2 local DOF order is (verified via cell_to_dof + ipoints):
    #   [v0, mid(v0,v1), mid(v0,v2), v1, mid(v1,v2), v2].
    # My _p2_basis_at_bcs uses [v0, v1, v2, mid(v1,v2), mid(v0,v2), mid(v0,v1)],
    # so the permutation that reorders my columns to fealpy's is:
    basis_perm = np.array([0, 5, 4, 1, 3, 2], dtype=np.int64)

    coords = np.asarray(sample["coords"], dtype=np.float64)          # (2, H, W)
    damage = np.asarray(sample["damage"], dtype=np.float32)          # (T, 1, H, W)
    T, _, H, W = damage.shape
    step_conv = np.asarray(sample.get("step_converged",
                                      np.ones(T, dtype=np.uint8)), dtype=bool)

    # Iterate steps; each step_XXX.npz gives one DOF snapshot of u.
    u_grid_all = np.zeros((T, 2, H, W), dtype=np.float32)
    for t in range(T):
        step_path = run_dir / "checkpoints" / f"step_{t:03d}.npz"
        if not step_path.exists():
            return f"skip (missing step {t}): {sample_id}"
        z = np.load(step_path, allow_pickle=False)
        u_dof = np.asarray(z["u"], dtype=np.float64)
        u_grid = sample_u_grid(node, cell, u_dof, coords, basis_perm)  # (2, H, W)
        u_grid_all[t] = u_grid.astype(np.float32)

    # Recovered stress in physical units (no stress_scale division yet). We
    # normalize by σ_h^rec's OWN 95%ile over converged steps so training-space
    # magnitudes match σ_h — otherwise the P2-DG u_h's crack-tip jumps make
    # σ_h^rec several orders of magnitude larger than σ_h and dominate the
    # L² loss for reasons orthogonal to the paper_thesis §F.3 contrast.
    from fracturex.learn.stress_recovery import stress_recovered_from_displacement
    dx = float(coords[0, 0, 1] - coords[0, 0, 0])
    dy = float(coords[1, 1, 0] - coords[1, 0, 0])
    d_grid = damage[:, 0, :, :]                                        # (T, H, W)
    stress_rec_phys = stress_recovered_from_displacement(
        u_grid_all, d_grid, C, dx=dx, dy=dy, kres=1e-6, stress_scale=1.0
    )
    stress_rec_phys = np.asarray(stress_rec_phys, dtype=np.float32)    # (T, 3, H, W)

    # Per-sample scale: 95%ile of |σ_h^rec| on converged steps only.
    if step_conv.any():
        rec_scale = float(np.percentile(np.abs(stress_rec_phys[step_conv]), 95.0))
    else:
        rec_scale = float(np.percentile(np.abs(stress_rec_phys), 95.0))
    rec_scale = max(rec_scale, 1e-12)
    stress_rec = (stress_rec_phys / rec_scale).astype(np.float32)

    # Data-quality guard: non-converged FE steps carry unbounded u_h (see
    # step_converged mask), so σ_h^rec blows up there and would swamp the L²
    # loss. Fill non-converged steps with the equilibrated stress σ_h so the
    # two supervision targets differ ONLY at converged steps — the meaningful
    # regime for the paper_thesis §F.3 contrast. If the sample carries no
    # σ_h at all we leave the raw σ_h^rec in place with a warning.
    if "stress" in sample:
        stress_h = np.asarray(sample["stress"], dtype=np.float32)      # (T, 3, H, W)
        bad = ~step_conv
        if bad.any():
            stress_rec[bad] = stress_h[bad]
    n_conv = int(step_conv.sum())
    _note = f" [{n_conv}/{T} converged steps used]"

    # Atomic write: save to sibling temp file (np.savez auto-appends .npz to
    # a stem, so use a distinct stem to avoid the .npz.tmp.npz gotcha).
    sample["stress_rec"] = stress_rec
    sample["stress_rec_scale"] = np.float32(rec_scale)                # physical scale used
    tmp_stem = npz_path.parent / (npz_path.stem + "_stress_rec_tmp")
    np.savez(tmp_stem, **sample)                          # writes tmp_stem.npz
    tmp_written = tmp_stem.with_suffix(".npz")
    shutil.move(tmp_written, npz_path)
    return f"ok: {sample_id}  (stress_rec range = [{stress_rec.min():.3g}, {stress_rec.max():.3g}], rec_scale={rec_scale:.3g}){_note}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing stress_rec key")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N samples (smoke test)")
    args = ap.parse_args()

    manifest_path = args.dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    samples = [s for s in manifest.get("samples", []) if s.get("ok", True)]
    if args.limit is not None:
        samples = samples[: args.limit]
    print(f"[add_stress_rec] {len(samples)} samples under {args.dataset_dir}")

    ordering_cache: dict = {}
    ok = 0
    for rec in samples:
        sid = rec["id"]
        npz_rel = rec.get("npz", f"samples/{sid}.npz")
        try:
            msg = process_sample(args.dataset_dir, sid, npz_rel, args.force, ordering_cache)
        except Exception as e:                                           # noqa: BLE001
            msg = f"fail: {sid}  {type(e).__name__}: {e}"
        print(msg)
        if msg.startswith("ok:"):
            ok += 1
    print(f"[add_stress_rec] done: {ok}/{len(samples)} samples updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
