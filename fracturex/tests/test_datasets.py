"""Unit tests for fracturex.ml.datasets (D13 dataset assembly + held-out splits).

Covers:
  - FeatureSample construction / n_nodes;
  - the four §5.3 hold-out protocols partition correctly by their axis;
  - standardizer is fitted on TRAIN only (no test leakage) and z-scores train data;
  - constant feature columns do not produce div-by-zero;
  - load_sample round-trips a real dump_features .npz when present.

Run:
  PYTHONPATH=<repo> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m pytest fracturex/tests/test_datasets.py -q
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fealpy.backend import backend_manager as bm

from fracturex.ml.coarse_features import FEATURE_NAMES, N_FEATURES
from fracturex.ml.datasets import (
    FeatureSample,
    DataSplit,
    load_sample,
    split_cross_damage,
    split_cross_mesh,
    split_cross_case,
    split_cross_l0,
    split_by_predicate,
)

_REPO = Path(__file__).resolve().parents[2]


def _sample(*, case="model0", hmin=0.025, l0=0.02, maxd=0.5, nn=10, fill=None):
    rng = np.random.default_rng(int(maxd * 1000) + int(hmin * 1000) + len(case))
    phi = rng.standard_normal((nn, N_FEATURES)) if fill is None else np.full((nn, N_FEATURES), fill)
    return FeatureSample(
        phi=bm.asarray(phi.astype(np.float64)),
        node=bm.asarray(rng.standard_normal((nn, 2))),
        case=case, hmin=hmin, l0=l0, maxd=maxd, eps_g=1e-6,
        source=f"{case}_h{hmin}_l{l0}_d{maxd}",
    )


def test_feature_sample_basic():
    s = _sample(nn=7)
    assert s.n_nodes == 7
    assert s.phi.shape == (7, N_FEATURES)
    assert s.target is None


def test_cross_damage_split():
    samples = [_sample(maxd=m) for m in (0.1, 0.5, 0.9, 0.97, 0.999)]
    sp = split_cross_damage(samples, test_maxd_min=0.95)
    assert sp.protocol == "cross_damage"
    assert sorted(s.maxd for s in sp.test) == [0.97, 0.999]
    assert sorted(s.maxd for s in sp.train) == [0.1, 0.5, 0.9]


def test_cross_mesh_split_headline():
    # finer mesh => smaller hmin; held out for test.
    samples = [_sample(hmin=h) for h in (0.05, 0.025, 0.013, 0.008)]
    sp = split_cross_mesh(samples, test_hmin_max=0.013)
    assert sp.protocol == "cross_mesh"
    assert sorted(s.hmin for s in sp.test) == [0.008, 0.013]
    assert sorted(s.hmin for s in sp.train) == [0.025, 0.05]


def test_cross_case_split():
    samples = [_sample(case=c) for c in ("model0", "square", "model2")]
    sp = split_cross_case(samples, test_cases=["model2"])
    assert sp.protocol == "cross_case"
    assert [s.case for s in sp.test] == ["model2"]
    assert {s.case for s in sp.train} == {"model0", "square"}


def test_cross_l0_split():
    samples = [_sample(l0=v) for v in (2e-3, 1e-3, 5e-4)]
    sp = split_cross_l0(samples, test_l0=[5e-4])
    assert sp.protocol == "cross_l0"
    assert [pytest.approx(s.l0) for s in sp.test] == [pytest.approx(5e-4)]
    assert len(sp.train) == 2


def test_standardizer_fitted_on_train_only():
    # Train nodes all ~ N(0,1); inject an extreme test sample. The standardizer
    # must reflect ONLY train stats (mean ~0, std ~1), unaffected by the test outlier.
    train = [_sample(maxd=m, nn=500) for m in (0.1, 0.5, 0.9)]
    test = [_sample(maxd=0.999, nn=500, fill=1000.0)]
    sp = split_by_predicate(train + test, lambda s: s.maxd >= 0.95)
    mean = np.asarray(bm.to_numpy(sp.mean))
    std = np.asarray(bm.to_numpy(sp.std))
    assert np.all(np.abs(mean) < 0.5), f"train mean leaked: {mean}"
    assert np.all(np.abs(std - 1.0) < 0.5), f"train std leaked: {std}"
    # standardized train data is ~ zero-mean unit-std.
    z = np.asarray(bm.to_numpy(sp.standardize(sp.train[0].phi)))
    assert abs(z.mean()) < 0.3


def test_standardizer_handles_constant_column():
    # all-constant features => std guarded to 1, no NaN/inf.
    samples = [_sample(nn=50, fill=3.0) for _ in range(3)]
    sp = split_by_predicate(samples, lambda s: False)
    z = bm.to_numpy(sp.standardize(samples[0].phi))
    assert np.isfinite(np.asarray(z)).all()


def test_empty_train_raises():
    samples = [_sample(maxd=0.99)]
    with pytest.raises(ValueError):
        split_cross_damage(samples, test_maxd_min=0.95)  # everything is test


def test_load_real_npz_roundtrip():
    feat_dir = _REPO / "results/phasefield/_precond_features"
    files = sorted(feat_dir.glob("feat_*.npz")) if feat_dir.is_dir() else []
    if not files:
        pytest.skip("no dumped feature npz available")
    s = load_sample(files[0])
    assert s.phi.shape[1] == N_FEATURES
    assert s.n_nodes > 0
    assert s.case and s.l0 > 0 and 0.0 <= s.maxd <= 1.0
    assert np.isfinite(np.asarray(bm.to_numpy(s.phi))).all()
