"""Dataset assembly + held-out splitting for the learned coarse-space (D13 §5.3).

A *sample* is one frozen-d snapshot: the per-node feature block ``phi`` produced by
``coarse_features`` / ``scripts/paper_precond/dump_features.py``, plus provenance
metadata (case, hmin, l0, maxd, eps_g) and an OPTIONAL learning target. Targets are
attached later by ``spectral_labels`` (κ of the two-level operator / enrichment
labels); this module never computes them and never imports a solver or torch.

Splitting follows the four generalization protocols in the plan (§5.3):

    protocol         train axis                       test axis
    cross_damage     maxd small (pre-localization)     maxd -> 1 (localized)
    cross_mesh*      coarse meshes (large hmin)        fine meshes (small hmin)
    cross_case       some cases                        a held-out case
    cross_l0         some l0                           a held-out l0

(*cross_mesh is the headline: learned mesh-independence.)

Standardization statistics are computed on the TRAIN split only and applied to test,
so no information leaks from the held-out set.

Backend policy (docs/architecture/multibackend_convention.md): npz read is the numpy
I/O boundary; feature tensors are converted to ``bm`` for any downstream compute.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np  # I/O boundary only (npz load); compute uses bm.

from fealpy.backend import backend_manager as bm

from fracturex.ml.coarse_features import FEATURE_NAMES, N_FEATURES


@dataclass
class FeatureSample:
    """One frozen-d snapshot: features + provenance + optional target.

    Attributes:
        phi: ``(NN, N_FEATURES)`` per-node feature block (bm tensor).
        node: ``(NN, gdim)`` node coordinates (bm tensor; debug/plot only).
        case: example name (e.g. ``"model0"``).
        hmin: mesh size knob (cross-mesh axis).
        l0: phase-field length scale (cross-l0 axis).
        maxd: max damage in the snapshot (cross-damage axis).
        eps_g: degradation floor used at dump time.
        source: source checkpoint path (provenance).
        target: optional per-node or scalar learning target (attached by
            spectral_labels). ``None`` until labelled.
    """

    phi: Any
    node: Any
    case: str
    hmin: float
    l0: float
    maxd: float
    eps_g: float
    source: str
    target: Optional[Any] = None

    @property
    def n_nodes(self) -> int:
        return int(self.phi.shape[0])


def load_sample(path: str | Path) -> FeatureSample:
    """Load one feature .npz (written by dump_features.py) into a FeatureSample.

    Args:
        path: path to a ``feat_*.npz`` file.
    Returns:
        a :class:`FeatureSample` with ``target=None``.
    Raises:
        ValueError: if the feature width / names disagree with the current contract.
    """
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    names = [str(x) for x in z["feature_names"]]
    if names != list(FEATURE_NAMES):
        raise ValueError(
            f"{path.name}: feature_names {names} != contract {list(FEATURE_NAMES)} "
            "(re-dump features after a schema change)."
        )
    phi = bm.asarray(z["phi"])
    if phi.shape[1] != N_FEATURES:
        raise ValueError(f"{path.name}: phi width {phi.shape[1]} != {N_FEATURES}")
    target = bm.asarray(z["target"]) if "target" in z.files else None
    return FeatureSample(
        phi=phi,
        node=bm.asarray(z["node"]),
        case=str(z["case"]),
        hmin=float(z["hmin"]),
        l0=float(z["l0"]),
        maxd=float(z["maxd"]),
        eps_g=float(z["eps_g"]),
        source=str(z["source_ckpt"]),
        target=target,
    )


def load_samples(paths: Sequence[str | Path] | str | Path) -> list[FeatureSample]:
    """Load many feature .npz files (a dir is globbed for ``feat_*.npz``).

    Args:
        paths: a directory, a single file, or a sequence of files.
    Returns:
        list of :class:`FeatureSample`, sorted by source for determinism.
    """
    if isinstance(paths, (str, Path)):
        p = Path(paths)
        files = sorted(p.glob("feat_*.npz")) if p.is_dir() else [p]
    else:
        files = [Path(x) for x in paths]
    return sorted((load_sample(f) for f in files), key=lambda s: s.source)


# ---------------------------------------------------------------------------
# Held-out splitting (§5.3)
# ---------------------------------------------------------------------------

@dataclass
class DataSplit:
    """A train/test partition plus the standardization fitted on train.

    Attributes:
        train, test: sample lists.
        mean, std: ``(N_FEATURES,)`` bm tensors fitted on TRAIN nodes only.
        protocol: name of the protocol used.
    """

    train: list[FeatureSample]
    test: list[FeatureSample]
    mean: Any
    std: Any
    protocol: str

    def standardize(self, phi: Any) -> Any:
        """Apply the train-fitted z-score to a feature block."""
        return (phi - self.mean) / self.std


def _fit_standardizer(train: Sequence[FeatureSample]) -> tuple[Any, Any]:
    """Per-feature mean/std over all TRAIN nodes (test excluded to avoid leakage)."""
    if not train:
        raise ValueError("cannot fit standardizer on an empty train split")
    stacked = bm.concatenate([s.phi for s in train], axis=0)  # (sum_NN, N_FEATURES)
    mean = bm.mean(stacked, axis=0)
    std = bm.std(stacked, axis=0)
    std = bm.where(std > 1e-12, std, bm.ones_like(std))  # guard constant columns
    return mean, std


def split_by_predicate(
    samples: Sequence[FeatureSample],
    is_test: Callable[[FeatureSample], bool],
    *,
    protocol: str = "custom",
) -> DataSplit:
    """Generic split: a sample goes to test iff ``is_test(sample)`` is true.

    Args:
        samples: all loaded samples.
        is_test: predicate selecting the held-out set.
        protocol: label stored on the result.
    Returns:
        a :class:`DataSplit` with standardizer fitted on train.
    """
    train = [s for s in samples if not is_test(s)]
    test = [s for s in samples if is_test(s)]
    mean, std = _fit_standardizer(train)
    return DataSplit(train=train, test=test, mean=mean, std=std, protocol=protocol)


def split_cross_damage(
    samples: Sequence[FeatureSample], *, test_maxd_min: float = 0.95
) -> DataSplit:
    """Train on pre-localization snapshots, test on localized ones (maxd -> 1)."""
    return split_by_predicate(
        samples, lambda s: s.maxd >= test_maxd_min, protocol="cross_damage"
    )


def split_cross_mesh(
    samples: Sequence[FeatureSample], *, test_hmin_max: float
) -> DataSplit:
    """Headline split: train on coarse meshes, test on fine ones.

    Args:
        test_hmin_max: a sample is held out for test iff ``hmin <= test_hmin_max``
            (finer meshes have smaller hmin).
    """
    return split_by_predicate(
        samples, lambda s: s.hmin <= test_hmin_max, protocol="cross_mesh"
    )


def split_cross_case(
    samples: Sequence[FeatureSample], *, test_cases: Sequence[str]
) -> DataSplit:
    """Train on some cases, test on a held-out case (geometry generalization)."""
    held = {c.lower() for c in test_cases}
    return split_by_predicate(
        samples, lambda s: s.case.lower() in held, protocol="cross_case"
    )


def split_cross_l0(
    samples: Sequence[FeatureSample], *, test_l0: Sequence[float], rtol: float = 1e-3
) -> DataSplit:
    """Train on some l0, test on a held-out l0 (length-scale generalization)."""
    targets = [float(x) for x in test_l0]

    def is_test(s: FeatureSample) -> bool:
        return any(abs(s.l0 - t) <= rtol * max(abs(t), 1e-30) for t in targets)

    return split_by_predicate(samples, is_test, protocol="cross_l0")
