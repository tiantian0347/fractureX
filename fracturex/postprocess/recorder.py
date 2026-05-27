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


class RunRecorder:
    """
    Minimal persistent recorder:
      - meta.json (once)
      - history.csv (append each step)
      - checkpoints/step_XXX.npz (optional)
    """

    def __init__(self, outdir: str, *, save_npz: bool = True, save_every: int = 1):
        """Create persistent run recorder.

        Inputs:
            outdir: Output directory path.
            save_npz: Whether to save checkpoint `.npz` snapshots.
            save_every: Save checkpoint every `save_every` load steps.
        Output:
            None. Creates output directories and initializes CSV headers.
        """
        self.outdir = outdir
        self.save_npz = bool(save_npz)
        self.save_every = int(save_every)
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
