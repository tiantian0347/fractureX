# fracturex/postprocess/recorder.py
from __future__ import annotations
import os, json, csv
import numpy as np


def _read_self_rss_mb() -> float:
    """Read current process RSS (MB) from /proc/self/status; NaN if unavailable."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return float("nan")


def _read_self_peak_rss_mb() -> float:
    """Read peak process RSS (MB) from /proc/self/status (VmHWM); NaN if unavailable.

    VmHWM is the resident-set high-water mark since process start, so it captures
    the transient solve/factorization peak even when sampled at the end of a load
    step (after that memory was freed). This is the figure C4 (memory vs N) needs;
    the instantaneous ``rss_mb`` from VmRSS systematically undercounts it.
    """
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return float("nan")


class RunRecorder:
    """
    Minimal persistent recorder:
      - meta.json (once)
      - history.csv (append each step)
      - checkpoints/step_XXX.npz (optional)
    """

    def __init__(
        self,
        outdir: str,
        *,
        save_npz: bool = True,
        save_every: int = 1,
        save_quadrature_fields: bool = False,
        save_recovered_strain: bool = False,
    ):
        """Create persistent run recorder.

        Inputs:
            outdir: Output directory path.
            save_npz: Whether to save checkpoint `.npz` snapshots.
            save_every: Save checkpoint every `save_every` load steps.
            save_quadrature_fields: Reserved switch for quadrature-level dumps
                (σ_qp, d_qp, H_qp) used by the operator-learning dataset
                pipeline. Default False; consumers (driver / dataset_export)
                read this flag and decide whether to populate the extra
                fields. The checkpoint format itself stays unchanged.
            save_recovered_strain: Reserved switch for ε^h = A(d) σ
                quadrature-level dumps (plan §3.3'). Default False; same
                consumer-driven semantics as ``save_quadrature_fields``.
        Output:
            None. Creates output directories and initializes CSV headers.
        """
        self.outdir = outdir
        self.save_npz = bool(save_npz)
        self.save_every = int(save_every)
        self.save_quadrature_fields = bool(save_quadrature_fields)
        self.save_recovered_strain = bool(save_recovered_strain)
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(os.path.join(outdir, "checkpoints"), exist_ok=True)
        self.csv_path = os.path.join(outdir, "history.csv")
        self.iter_csv_path = os.path.join(outdir, "iterations.csv")
        self._csv_header: list[str] | None = None
        self._iter_csv_header: list[str] | None = None

    def write_meta(self, meta: dict):
        """Write run-level metadata to `meta.json`.

        Input:
            meta: Metadata dictionary.
        Output:
            None. Overwrites `meta.json`.
        """
        path = os.path.join(self.outdir, "meta.json")
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)

    def append_history(self, row: dict):
        """Append one load-step record to `history.csv`.

        Input:
            row: Per-step metrics row.
        Output:
            None. Creates file/header on first call, then appends rows.
        """
        row.setdefault("rss_mb", _read_self_rss_mb())
        row.setdefault("peak_rss_mb", _read_self_peak_rss_mb())
        # Keep a stable header from the first row
        if self._csv_header is None:
            self._csv_header = list(row.keys())
            with open(self.csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._csv_header)
                w.writeheader()
                w.writerow(row)
        else:
            # if new keys appear, you can either ignore or extend header; here we extend safely
            for k in row.keys():
                if k not in self._csv_header:
                    self._csv_header.append(k)
            # rewrite file with new header is heavy; simplest: keep header fixed
            # so here we only write existing header fields
            out = {k: row.get(k, "") for k in self._csv_header}
            with open(self.csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._csv_header)
                w.writerow(out)

    def write_step(self, row: dict):
        """Backward-compatible alias used by phase-field driver."""
        self.append_history(row)

    def append_iteration(self, row: dict):
        """Append one nonlinear-iteration diagnostics row.

        Input:
            row: Per-iteration metrics row.
        Output:
            None. Writes to `iterations.csv`.
        """
        if self._iter_csv_header is None:
            self._iter_csv_header = list(row.keys())
            with open(self.iter_csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._iter_csv_header)
                w.writeheader()
                w.writerow(row)
        else:
            for k in row.keys():
                if k not in self._iter_csv_header:
                    self._iter_csv_header.append(k)
            out = {k: row.get(k, "") for k in self._iter_csv_header}
            with open(self.iter_csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._iter_csv_header)
                w.writerow(out)

    def save_mesh(self, discr) -> None:
        """Persist enough mesh + discretization info to rebuild ``discr`` later.

        Writes ``<outdir>/mesh.npz`` with ``node`` / ``cell`` (triangle), the
        polynomial orders (`p_sigma`, `damage_p`, `u_space_order`), the
        ``use_relaxation`` flag, plus ``is_neumann_edge`` (per
        boundary-edge bool) and the augmented ``boundary_edge_flag``. With
        these fields :func:`fracturex.postprocess.dataset_export.load_discr_from_dir`
        can rebuild a :class:`HuZhangDiscretization` byte-equivalent to the
        one that produced the checkpoints — no ``case`` instance required.

        No-op when ``save_npz`` is False or ``discr.mesh`` is None.
        """
        if not self.save_npz:
            return
        mesh = getattr(discr, "mesh", None)
        if mesh is None:
            return
        from fealpy.backend import backend_manager as bm

        node = np.asarray(mesh.entity("node"), dtype=np.float64)
        cell = np.asarray(mesh.entity("cell"), dtype=np.int64)

        # Augmented boundary-edge mask captures both the natural mesh boundary
        # and any crack edges injected via mesh_patch.augment_boundary_edges_inplace.
        be_aug = np.asarray(bm.asarray(mesh.boundary_edge_flag())).reshape(-1).astype(bool)

        # HuZhang's `bd_stress` argument is consumed at __init__ time and
        # processed into `space_sigma.isNedge` (NE-long bool mask). We dump
        # that processed mask — _build_isNedge handles NE-long input fine,
        # so the rebuild path passes `bd_stress=isNedge` directly.
        is_n_bd: np.ndarray
        space_sigma = getattr(discr, "space_sigma", None)
        if space_sigma is not None and hasattr(space_sigma, "isNedge"):
            try:
                is_n_bd = np.asarray(bm.asarray(space_sigma.isNedge)).reshape(-1).astype(bool)
            except Exception:
                is_n_bd = np.zeros(0, dtype=bool)
        else:
            is_n_bd = np.zeros(0, dtype=bool)

        path = os.path.join(self.outdir, "mesh.npz")
        np.savez_compressed(
            path,
            node=node,
            cell=cell,
            p_sigma=int(discr.p),
            damage_p=int(getattr(discr, "damage_p", 1)),
            u_space_order=int(getattr(discr, "u_space_order", max(int(discr.p) - 1, 1))),
            use_relaxation=bool(getattr(discr, "use_relaxation", True)),
            boundary_edge_flag_aug=be_aug,
            is_neumann_edge=is_n_bd,
        )

    def dump_quadrature_fields(self, step: int, discr, state) -> None:
        """Persist quadrature-level history field ``H_qp`` and physical points ``xq``.

        Writes ``<outdir>/checkpoints/step_XXX_qp.npz`` when both
        ``self.save_npz`` and ``self.save_quadrature_fields`` are True and
        ``state.H`` is populated. The quadrature rule mirrors
        :class:`PhaseFieldAssembler` — order ``damage_p + 3`` on the cell
        formula, so ``H_qp`` and ``xq`` index identical points.

        Note: the *method* is intentionally named ``dump_quadrature_fields``
        because ``save_quadrature_fields`` is already used as the bool
        switch on ``__init__`` (see §1.6 of the M0 kickoff report); a method
        of the same name would be shadowed by the attribute.

        Fields:
            ``H_qp`` (NC, NQ) float64,
            ``xq`` (NC, NQ, 2) float64,
            ``q_order`` int (the quadrature order actually used),
            ``step`` int.

        No-op when any precondition fails, so this is safe to call
        unconditionally from the driver.
        """
        if not self.save_npz:
            return
        if not self.save_quadrature_fields:
            return
        if step % self.save_every != 0:
            return
        H = getattr(state, "H", None)
        if H is None:
            return
        mesh = getattr(discr, "mesh", None)
        if mesh is None:
            return

        from fealpy.backend import backend_manager as bm

        H_np = np.asarray(bm.asarray(H), dtype=np.float64)
        if H_np.ndim != 2:
            return

        q_order = int(getattr(discr, "damage_p", 1)) + 3
        try:
            qf = mesh.quadrature_formula(q_order, "cell")
            bcs, _ = qf.get_quadrature_points_and_weights()
            xq = np.asarray(bm.asarray(mesh.bc_to_point(bcs)), dtype=np.float64)
        except Exception:
            return

        if xq.shape[:2] != H_np.shape:
            return

        path = os.path.join(self.outdir, "checkpoints", f"step_{step:03d}_qp.npz")
        np.savez_compressed(
            path,
            H_qp=H_np,
            xq=xq,
            q_order=q_order,
            step=int(step),
        )

    def save_checkpoint(self, step: int, discr, state):
        """Save checkpoint snapshot (`npz`) for current step.

        Inputs:
            step: Load step index.
            discr: Discretization object (for mesh metadata).
            state: Current FE state (`sigma`, `u`, `d`, `r_hist`, `H`).
        Output:
            None. Writes compressed checkpoint file when enabled.
        """
        if (not self.save_npz) or (step % self.save_every != 0):
            return
        path = os.path.join(self.outdir, "checkpoints", f"step_{step:03d}.npz")
        np.savez_compressed(
            path,
            sigma=np.asarray(state.sigma[:]),
            u=np.asarray(state.u[:]),
            d=np.asarray(state.d[:]),
            r_hist=np.asarray(state.r_hist[:]),
            H=np.asarray(state.H[:]) if getattr(state, "H", None) is not None else None,
            # light mesh info
            NN=int(discr.mesh.number_of_nodes()),
            NE=int(discr.mesh.number_of_edges()),
            NC=int(discr.mesh.number_of_cells()),
            p=int(discr.p),
        )
