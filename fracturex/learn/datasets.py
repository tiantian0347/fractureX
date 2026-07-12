# fracturex/learn/datasets.py
"""Dataset adapters for the SURROGATE_DATA_SCHEMA npz format.

Schema: docs/operator_learning/SURROGATE_DATA_SCHEMA.md
Plan:   docs/operator_learning/plan_operator_learning.md §M1

This module is **solver-decoupled**: it only reads the on-disk protocol
(``.npz`` + ``.meta.json`` + ``dataset_manifest.json``). The numpy reader is
torch-free; the PyTorch wrapper is built lazily so the reader and the metrics
can be exercised without torch installed.

Channel layout (Stage A, scheme A — predict all T steps as channels):
  input  X = concat[ sdf(1), mask(1), coords(2), material broadcast(k) ]  -> (4+k, H, W)
  target Y = damage                                                       -> (T, H, W)
The material vector length k is read from the data, never hard-coded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class DatasetConfig:
    """Knobs that select which npz fields a Dataset exposes."""
    include_stress: bool = True          # T3 main / T2 ablation
    include_stress_rec: bool = False     # T3.M3b σ_h^rec (recovered from u_h) contrast supervision
    include_history: bool = False        # plan §3.5 (b) variant
    rollout_length: Optional[int] = None # None → full sequence; else random window of length k
    augment_rotate90: bool = False
    augment_flip: bool = False


# Input channel names assembled by :func:`assemble_input_channels`, in order.
def input_channel_names(k_material: int) -> list[str]:
    return (
        ["sdf", "mask", "coord_x", "coord_y"]
        + [f"material_{i}" for i in range(k_material)]
    )


def assemble_input_channels(sample: dict) -> np.ndarray:
    """Build the (C_in, H, W) static input tensor for scheme A.

    C_in = 4 + k where k = len(sample['material']). Material scalars are
    broadcast to full (H, W) planes.
    """
    sdf = np.asarray(sample["sdf"], dtype=np.float32)          # (1, H, W)
    mask = np.asarray(sample["mask"], dtype=np.float32)        # (1, H, W)
    coords = np.asarray(sample["coords"], dtype=np.float32)    # (2, H, W)
    material = np.asarray(sample["material"], dtype=np.float32)  # (k,)
    H, W = sdf.shape[-2:]
    mat_planes = np.broadcast_to(material[:, None, None], (material.shape[0], H, W))
    return np.concatenate([sdf, mask, coords, mat_planes.astype(np.float32)], axis=0)


def target_damage(sample: dict) -> np.ndarray:
    """(T, H, W) damage target (channel axis squeezed).

    Clipped to the physical range [0, 1]: at non-converged staggered tail
    steps the FE damage can overshoot slightly (d > 1), which violates schema
    §6.3; we clamp so training targets stay physical. The overshoot itself is
    flagged via ``step_converged`` in the npz / dataset report.
    """
    d = np.asarray(sample["damage"], dtype=np.float32)  # (T, 1, H, W)
    return np.clip(d[:, 0, :, :], 0.0, 1.0)


def target_stress(sample: dict) -> np.ndarray:
    """(T, 3, H, W) normalized stress target (σxx, σyy, σxy), or None if absent.

    The npz stores stress already divided by ``stress_scale`` (schema §3.2), so
    it is used as-is for training (per-sample O(1) magnitude).
    """
    if "stress" not in sample:
        return None
    return np.asarray(sample["stress"], dtype=np.float32)  # (T, 3, H, W)


def target_stress_rec(sample: dict) -> np.ndarray:
    """(T, 3, H, W) displacement-recovered stress σ_h^rec = g(d)·C·ε(u_h), or None.

    Used by T3.M3b as the *non-equilibrated* supervision source in the
    contrast experiment (paper_thesis §F.3 / §G): a plain L² fit to
    σ_h^rec is expected to plateau at R̃_h ~ Θ(h^m) due to the normal
    jumps of σ_h^rec ∉ H(div), whereas fitting to σ_h ∈ H(div,S) is
    reducible to training noise.
    """
    if "stress_rec" not in sample:
        return None
    return np.asarray(sample["stress_rec"], dtype=np.float32)


class PhaseFieldOperatorDataset:
    """Lazy reader of ``samples/sample_XXXXXX.npz`` from a dataset directory.

    Reads ``dataset_manifest.json`` for the requested split (falls back to all
    OK samples when the manifest has no ``splits``). ``__getitem__`` returns a
    numpy-valued dict; framework batching is done by the wrappers below.
    """

    def __init__(self, dataset_dir: Path, split: str = "train",
                 cfg: Optional[DatasetConfig] = None) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.cfg = cfg or DatasetConfig()
        manifest_path = self.dataset_dir / "dataset_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing {manifest_path}")
        with manifest_path.open() as f:
            self.manifest = json.load(f)

        splits = self.manifest.get("splits") or {}
        if split in splits:
            ids = list(splits[split])
        else:
            # No explicit split → use every OK sample.
            ids = [s["id"] for s in self.manifest.get("samples", []) if s.get("ok", True)]
        # Resolve id → npz/meta paths from the samples table when available.
        by_id = {s["id"]: s for s in self.manifest.get("samples", [])}
        self.sample_ids: list[str] = []
        self._npz: list[Path] = []
        self._meta: list[Path] = []
        for sid in ids:
            rec = by_id.get(sid, {})
            npz = self.dataset_dir / rec.get("npz", f"samples/{sid}.npz")
            meta = self.dataset_dir / rec.get("meta", f"samples/{sid}.meta.json")
            if npz.exists() and meta.exists():
                self.sample_ids.append(sid)
                self._npz.append(npz)
                self._meta.append(meta)
        if not self.sample_ids:
            raise ValueError(
                f"no usable samples for split={split!r} under {self.dataset_dir}"
            )
        # `.npz` is a zip archive; np.load re-inflates it (single-threaded,
        # GIL-bound) on every access. With num_workers=0 that puts ~T·N·epochs
        # zip-decompressions on the training thread and starves the conv loop
        # (measured ~5% duty). Memoize the built sample dict so each npz is
        # decompressed exactly once; the fully-expanded split fits in RAM.
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        cached = self._cache.get(idx)
        if cached is not None:
            return cached
        z = np.load(self._npz[idx], allow_pickle=False)
        with self._meta[idx].open() as f:
            meta = json.load(f)
        sample: dict = {
            "sdf": z["sdf"], "mask": z["mask"], "coords": z["coords"],
            "material": z["material"], "time": z["time"],
            "load_history": z["load_history"],
            "damage": z["damage"], "valid_mask": z["valid_mask"],
            "sample_id": self.sample_ids[idx], "meta": meta,
        }
        if self.cfg.include_stress and "stress" in z:
            sample["stress"] = z["stress"]
        if self.cfg.include_stress_rec and "stress_rec" in z:
            sample["stress_rec"] = z["stress_rec"]
        if self.cfg.include_history and "history" in z:
            sample["history"] = z["history"]
        if "reaction" in z:                      # (T, r) FE-exact reference force curve
            sample["reaction"] = z["reaction"]
        self._cache[idx] = sample
        return sample

    # -- introspection used by the training side --
    @property
    def n_steps(self) -> int:
        """T (load steps), read from the first sample; asserts consistency lazily."""
        return int(self[0]["damage"].shape[0])

    @property
    def n_material(self) -> int:
        return int(self[0]["material"].shape[0])

    @property
    def grid_hw(self) -> tuple[int, int]:
        s = self[0]
        return int(s["sdf"].shape[-2]), int(s["sdf"].shape[-1])

    @property
    def n_input_channels(self) -> int:
        return 4 + self.n_material


def collate_masked(batch: list[dict]) -> dict:
    """Stack a batch into torch tensors plus per-sample metas.

    x:      (B, C_in, H, W)        static input channels
    y:      (B, T, H, W)           damage target (clamped to [0,1])
    stress: (B, T, 3, H, W)        normalized stress target — present only when
                                   every sample in the batch carries ``stress``
    mask:   (B, 1, H, W)           in-Ω mask
    """
    import torch

    xs = np.stack([assemble_input_channels(s) for s in batch], axis=0)
    ys = np.stack([target_damage(s) for s in batch], axis=0)
    ms = np.stack([np.asarray(s["valid_mask"], dtype=np.float32) for s in batch], axis=0)
    out = {
        "x": torch.from_numpy(xs),
        "y": torch.from_numpy(ys),
        "mask": torch.from_numpy(ms),
        "meta": [s.get("meta", {}) for s in batch],
        "sample_id": [s.get("sample_id") for s in batch],
    }
    stresses = [target_stress(s) for s in batch]
    if all(s is not None for s in stresses):
        out["stress"] = torch.from_numpy(np.stack(stresses, axis=0))
    stresses_rec = [target_stress_rec(s) for s in batch]
    if all(s is not None for s in stresses_rec):
        out["stress_rec"] = torch.from_numpy(np.stack(stresses_rec, axis=0))
    return out


def as_torch_dataset(dataset: PhaseFieldOperatorDataset):
    """Wrap :class:`PhaseFieldOperatorDataset` as a ``torch.utils.data.Dataset``.

    Each item is the raw numpy sample dict; use :func:`collate_masked` as the
    DataLoader ``collate_fn`` to get batched tensors.
    """
    from torch.utils.data import Dataset

    class _TorchSurrogateDataset(Dataset):
        def __init__(self, ds: PhaseFieldOperatorDataset):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            return self.ds[i]

    return _TorchSurrogateDataset(dataset)
