# fracturex/postprocess/dataset_export/adapters/huzhang_phasefield.py
"""SolverAdapter for the Hu-Zhang mixed-element phase-field fracture model.

This is the reference adapter and the **only** place the export pipeline
touches FEALPy / Hu-Zhang specifics: rebuilding the discretization from a
recorder dir, evaluating σ_h on the Hu-Zhang stress space (with its Voigt
ordering), and the phase-field material vector. Everything else in the
``dataset_export`` package is model-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..adapter import FieldSpec, SolverAdapter
from ..grid import GridSpec
from ..sampling import (
    PixelLocator,
    build_pixel_locator,
    evaluate_lagrange_on_grid,
    group_pixels_by_cell,
)

# Voigt convention used by HuZhangFESpace2d: stress channels [σxx, σxy, σyy].
# Schema §3.2 stores stress as (σxx, σyy, σxy); reorder when packing.
_HZ_TO_SCHEMA_STRESS = (0, 2, 1)  # [xx, xy, yy] -> [xx, yy, xy]

# Material-name aliases (recorder meta may use any of these keys).
_MATERIAL_ALIASES = {
    "lambda": ("lambda", "lam", "lambda0"),
    "mu": ("mu", "lambda1"),
    "Gc": ("Gc", "G_c"),
    "l0": ("l0", "ell0", "ell_0"),
    "eta": ("eta",),
}


def evaluate_huzhang_on_grid(space, dofs: np.ndarray, locator: PixelLocator) -> np.ndarray:
    """Evaluate a Hu-Zhang stress function at every inside pixel.

    Args:
        space:   ``HuZhangFESpace2d`` instance.
        dofs:    (gdof,) DOF vector (one snapshot, single field).
        locator: pixel→cell mapping.

    Returns:
        (3, H, W) float32 in HuZhang Voigt order [σxx, σxy, σyy]. Outside-Ω
        pixels are zero.
    """
    HW = locator.H * locator.W
    out = np.zeros((HW, 3), dtype=np.float64)
    cell_to_dof = np.asarray(space.dof.cell_to_dof())
    groups = group_pixels_by_cell(locator)
    for cid, pix in groups.items():
        bc = locator.bary[pix].astype(np.float64)  # (n, 3)
        idx = np.array([cid], dtype=np.int64)
        phi = np.asarray(space.basis(bc, index=idx))  # (1, n, ldof, 3)
        local_dofs = np.asarray(dofs)[cell_to_dof[cid]]  # (ldof,)
        out[pix] = np.einsum("qld,l->qd", phi[0], local_dofs)
    return out.reshape(locator.H, locator.W, 3).transpose(2, 0, 1).astype(np.float32)


# Backwards-compatible private alias.
_evaluate_huzhang_on_grid = evaluate_huzhang_on_grid


def load_discr_from_dir(recorder_dir: Path):
    """Rebuild a HuZhangDiscretization from ``<recorder_dir>/mesh.npz``.

    Mirrors :meth:`fracturex.postprocess.recorder.RunRecorder.save_mesh`.
    Avoids ``case.make_mesh`` (distmesh isn't reproducible) and avoids
    ``case.isD_bd`` by reusing the persisted Neumann/Dirichlet edge mask.

    Returns:
        A built ``HuZhangDiscretization`` whose ``mesh`` matches the one used
        to produce the run's checkpoints. ``state`` fields are zero-initialized.
    """
    from fealpy.backend import backend_manager as bm
    from fealpy.functionspace.huzhang_fe_space_2d import HuZhangFESpace2d
    from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
    from fealpy.mesh import TriangleMesh

    from fracturex.cases.base import CaseBase
    from fracturex.discretization.huzhang_discretization import (
        HuZhangDiscretization,
        HuZhangState,
    )

    recorder_dir = Path(recorder_dir)
    mesh_path = recorder_dir / "mesh.npz"
    if not mesh_path.exists():
        raise FileNotFoundError(
            f"missing {mesh_path}; run with a recorder that calls "
            "RunRecorder.save_mesh (driver auto-emits since 2026-05-28). "
            "Older runs need to be rerun or reconstructed from `case` "
            "directly."
        )

    z = np.load(mesh_path, allow_pickle=False)
    node = np.asarray(z["node"], dtype=np.float64)
    cell = np.asarray(z["cell"], dtype=np.int64)
    p_sigma = int(z["p_sigma"])
    damage_p = int(z["damage_p"])
    u_space_order = int(z["u_space_order"])
    use_relaxation = bool(z["use_relaxation"])
    be_aug = np.asarray(z["boundary_edge_flag_aug"], dtype=bool)
    is_n_bd = np.asarray(z["is_neumann_edge"], dtype=bool)

    mesh = TriangleMesh(bm.asarray(node), bm.asarray(cell))

    # Patch boundary_edge_flag so HuZhangFESpace2d sees the augmented set.
    def _be_aug():
        return bm.asarray(be_aug)

    mesh.boundary_edge_flag = _be_aug
    if hasattr(mesh, "boundary_face_index"):
        def _bfi_aug():
            return bm.where(_be_aug())[0]
        mesh.boundary_face_index = _bfi_aug

    space_sigma = HuZhangFESpace2d(
        mesh, p=p_sigma, use_relaxation=use_relaxation, bd_stress=is_n_bd
    )
    u_scalar = LagrangeFESpace(mesh, p=u_space_order, ctype="D")
    space_u = TensorFunctionSpace(u_scalar, shape=(2, -1))
    space_d = LagrangeFESpace(mesh, p=damage_p, ctype="C")

    sigma = space_sigma.function()
    u = space_u.function()
    d = space_d.function()
    r_hist = space_d.function()

    # Use a dummy case purely to satisfy CaseBase typing; it is never invoked
    # because we hand-build mesh + spaces below.
    class _RebuiltCase(CaseBase):
        name = "rebuilt_from_recorder"
        def make_mesh(self, **kw):
            raise RuntimeError("rebuilt discr already has a mesh")
        def isD_bd(self, points):
            raise RuntimeError("rebuilt discr already has Neumann edges")
        def model(self):
            raise RuntimeError("rebuilt discr has no material model attached")

    discr = HuZhangDiscretization(
        _RebuiltCase(),
        p=p_sigma,
        use_relaxation=use_relaxation,
        damage_p=damage_p,
        u_space_order=u_space_order,
    )
    discr.mesh = mesh
    discr.space_sigma = space_sigma
    discr.space_u = space_u
    discr.space_d = space_d
    discr.state = HuZhangState(sigma=sigma, u=u, d=d, r_hist=r_hist, H=None)
    return discr


def verify_discr_matches_checkpoint(discr, checkpoint_path) -> None:
    """Raise ValueError if ``discr`` cannot host ``checkpoint_path``'s arrays.

    Guard against stale ``mesh.npz`` fields (in particular the
    ``damage_p=1`` written by pre-2026-05-31 runs while the state was
    actually P2). Compares the checkpoint's ``d``/``u``/``sigma``
    shapes against the rebuilt ``discr``'s space gdofs.

    Args:
        discr: HuZhangDiscretization from load_discr_from_dir.
        checkpoint_path: Path to a ``step_XXX.npz`` under ``checkpoints/``.

    Raises:
        ValueError: with an actionable message pointing at the offending
            field and its recorded vs. expected size.
    """
    ck = np.load(str(checkpoint_path), allow_pickle=False)
    mismatches = []
    checks = [
        ("d", discr.space_d.number_of_global_dofs(),
         "damage_p (rebuild) vs state.d (checkpoint)"),
        ("u", discr.space_u.number_of_global_dofs(),
         "u_space_order (rebuild) vs state.u (checkpoint)"),
        ("sigma", discr.space_sigma.number_of_global_dofs(),
         "p_sigma (rebuild) vs state.sigma (checkpoint)"),
    ]
    for field, expected, hint in checks:
        if field not in ck.files:
            continue
        got = int(np.asarray(ck[field]).shape[0])
        if got != int(expected):
            mismatches.append(
                f"  {field}: ck.shape={got}  expected={int(expected)}  ({hint})"
            )
    if mismatches:
        raise ValueError(
            "load_discr_from_dir: rebuilt spaces do not match checkpoint "
            f"{checkpoint_path}. This usually means mesh.npz has a stale "
            "damage_p / p_sigma / u_space_order field (pre-2026-05-31 runs "
            "recorded damage_p=1 while state.d was P2). Run "
            "`scripts/fix_stale_mesh_damage_p.py --apply <recorder_dir>` "
            "to rewrite mesh.npz consistent with the checkpoint sizes.\n"
            + "\n".join(mismatches)
        )


@dataclass(frozen=True)
class HuZhangPhaseFieldAdapter:
    """:class:`SolverAdapter` for Hu-Zhang mixed-element phase-field fracture.

    Stateless; per-run state is the ``discr`` handle returned by
    :meth:`load_discretization`.
    """

    schema_version: str = "0.1"
    material_order: tuple[str, ...] = ("lambda", "mu", "Gc", "l0", "eta")
    output_field_specs: tuple[FieldSpec, ...] = (
        FieldSpec("damage", 1, "none"),
        FieldSpec("stress", 3, "stress_scale"),
    )
    eta_default: float = 1e-9

    # --- discretization handling --------------------------------------------
    def load_discretization(self, recorder_dir: Path):
        return load_discr_from_dir(recorder_dir)

    def mesh(self, discr):
        return discr.mesh

    def list_checkpoints(self, recorder_dir: Path) -> list[Path]:
        ckpt_dir = Path(recorder_dir) / "checkpoints"
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"missing {ckpt_dir}")
        ckpts = sorted(ckpt_dir.glob("step_*.npz"))
        if not ckpts:
            raise FileNotFoundError(f"no step_*.npz under {ckpt_dir}")
        return ckpts

    # --- material ------------------------------------------------------------
    def material_vector(
        self, recorder_meta: dict, overrides: Optional[dict] = None
    ) -> np.ndarray:
        material = dict(recorder_meta.get("material") or {})
        if overrides:
            material.update(overrides)
        vec = np.zeros(len(self.material_order), dtype=np.float64)
        for i, key in enumerate(self.material_order):
            candidates = _MATERIAL_ALIASES.get(key, (key,))
            for c in candidates:
                if c in material:
                    vec[i] = float(material[c])
                    break
            else:
                if key == "eta":
                    vec[i] = float(self.eta_default)
                else:
                    raise KeyError(
                        f"material[{key!r}] missing in recorder meta (looked for "
                        f"{candidates}); cannot build material vector for schema."
                    )
        return vec

    # --- output fields -------------------------------------------------------
    def evaluate_outputs(
        self, discr, checkpoint: dict, locator: PixelLocator, grid: GridSpec
    ) -> dict[str, np.ndarray]:
        space_d = discr.space_d
        if space_d is None:
            raise RuntimeError("discr.space_d is None; build() the discretization first.")
        d_dofs = np.asarray(checkpoint["d"])
        sigma_dofs = np.asarray(checkpoint["sigma"])

        damage = evaluate_lagrange_on_grid(space_d, d_dofs, locator)  # (1, H, W)
        stress_hz = evaluate_huzhang_on_grid(
            discr.space_sigma, sigma_dofs, locator
        )  # (3, H, W) in [xx, xy, yy]
        stress = stress_hz[_HZ_TO_SCHEMA_STRESS, :, :]  # -> schema [xx, yy, xy]
        return {"damage": damage, "stress": stress}

    # --- reaction (boundary force on the loaded surface) ---------------------
    def reaction(self, discr, checkpoint: dict) -> np.ndarray:
        """Reaction force on the loaded boundary for one frame (schema §3.2).

        Model-0 loads ``u_y`` on the top edge (y=ymax); the reaction is
        ``R_y = ∫_top (σ·n)_y ds`` integrated from the Hu-Zhang σ_h via
        :func:`fracturex.postprocess.reaction.reaction_from_sigma`. Physical
        units (not divided by stress_scale). Returns shape ``(r,)`` with r=1.
        """
        from fealpy.backend import backend_manager as bm
        from fracturex.postprocess.reaction import reaction_from_sigma

        space = discr.space_sigma
        sf = space.function()
        sf[:] = bm.asarray(np.asarray(checkpoint["sigma"]))
        mesh = discr.mesh
        y = np.asarray(mesh.entity("node"))[:, 1]
        ymax = float(y.max())
        tol = 1e-6 * (float(y.max()) - float(y.min()) + 1.0)

        def _on_top(bc):
            return bm.abs(bc[:, 1] - ymax) < tol

        ry = reaction_from_sigma(mesh, sf, _on_top, direction="y")
        return np.asarray([ry], dtype=np.float32)

    # --- metadata ------------------------------------------------------------
    def geometry_meta(self, recorder_dir: Path, recorder_meta: dict, cfg) -> dict:
        return {
            "case": recorder_meta.get("case", "unknown"),
            "domain_bbox": [list(cfg.grid.bbox[0]), list(cfg.grid.bbox[1])],
        }
