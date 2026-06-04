"""End-to-end smoke test for fracturex.postprocess.dataset_export.

Verifies a sampling roundtrip: build a tiny HuZhang discretization, fabricate
a few `step_*.npz` checkpoints, run :func:`export_recorder_to_sample`, then
assert the schema invariants from `docs/operator_learning/SURROGATE_DATA_SCHEMA.md` §6.

The simulator itself isn't exercised here — that's the job of the larger
integration test that is out of M0's scope. The point of this file is to
catch shape / dtype / mask drift inside dataset_export.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _build_dummy_recorder(tmpdir: Path, discr) -> tuple[Path, np.ndarray]:
    """Drop a 2-step recorder dir under tmpdir; return path + load array."""
    sigma = np.zeros_like(np.asarray(discr.state.sigma))
    u = np.zeros_like(np.asarray(discr.state.u))
    d = np.zeros_like(np.asarray(discr.state.d))
    r_hist = np.zeros_like(np.asarray(discr.state.r_hist))
    ck_dir = tmpdir / "checkpoints"
    ck_dir.mkdir()
    for s in (0, 1):
        np.savez(
            ck_dir / f"step_{s:03d}.npz",
            sigma=sigma, u=u, d=d, r_hist=r_hist, H=None,
            NN=int(discr.mesh.number_of_nodes()),
            NE=int(discr.mesh.number_of_edges()),
            NC=int(discr.mesh.number_of_cells()),
            p=int(discr.p),
        )
    (tmpdir / "meta.json").write_text(json.dumps({
        "case": "dummy_box",
        "p": int(discr.p),
        "mesh": {
            "NN": int(discr.mesh.number_of_nodes()),
            "NE": int(discr.mesh.number_of_edges()),
            "NC": int(discr.mesh.number_of_cells()),
        },
        "material": {"lambda": 55.5, "mu": 83.3, "Gc": 1.0, "l0": 0.02, "eta": 1e-9},
    }))
    (tmpdir / "history.csv").write_text(
        "load,iters,converged\n0.0,1,True\n0.01,4,True\n"
    )
    return tmpdir, np.array([0.0, 0.01], dtype=np.float32)


@pytest.fixture
def tmp_dataset(tmp_path):
    pytest.importorskip("fealpy")
    from fealpy.mesh import TriangleMesh
    from fracturex.cases.base import CaseBase
    from fracturex.discretization.huzhang_discretization import HuZhangDiscretization

    class _DummyCase(CaseBase):
        name = "dummy_box"

        def make_mesh(self, **kw):
            return TriangleMesh.from_box([0, 1, 0, 1], nx=8, ny=8)

        def isD_bd(self, points):
            return np.zeros(points.shape[0], dtype=bool)

        def model(self):
            raise NotImplementedError

    case = _DummyCase()
    discr = HuZhangDiscretization(case, p=3, damage_p=1).build()
    rec_dir = tmp_path / "rec"
    rec_dir.mkdir()
    _build_dummy_recorder(rec_dir, discr)
    return discr, rec_dir, tmp_path


def test_export_roundtrip_invariants(tmp_dataset):
    pytest.importorskip("matplotlib")
    from fracturex.postprocess.dataset_export import (
        SCHEMA_VERSION, GridSpec, ExportConfig, CircularNotchDomain,
        export_recorder_to_sample,
    )

    discr, rec_dir, tmp_path = tmp_dataset
    cfg = ExportConfig(grid=GridSpec(H=16, W=16, bbox=((0.0, 1.0), (0.0, 1.0))))
    geom = CircularNotchDomain(box=(0, 1, 0, 1), cx=0.5, cy=0.5, r=0.0)

    out_npz = tmp_path / "sample_000000.npz"
    out_meta = tmp_path / "sample_000000.meta.json"
    meta = export_recorder_to_sample(rec_dir, out_npz, out_meta, cfg, discr, geom)

    z = np.load(out_npz)
    H, W, T = 16, 16, 2

    # Shapes (schema §3.1, §3.2, §6.1)
    assert z["damage"].shape == (T, 1, H, W)
    assert z["stress"].shape == (T, 3, H, W)
    assert z["sdf"].shape == (1, H, W)
    assert z["mask"].shape == (1, H, W)
    assert z["valid_mask"].shape == (1, H, W)
    assert z["coords"].shape == (2, H, W)
    assert z["load_history"].shape == (T, 1)
    assert z["time"].shape == (T,)
    assert z["material"].shape == (5,)
    assert z["step_iters"].shape == (T,)
    assert z["step_converged"].shape == (T,)

    # Dtypes
    assert z["damage"].dtype == np.float32
    assert z["stress"].dtype == np.float32
    assert z["sdf"].dtype == np.float32
    assert z["mask"].dtype == np.uint8
    assert z["valid_mask"].dtype == np.uint8

    # Mask consistency (§6.2)
    assert np.array_equal(z["mask"], z["valid_mask"])
    # No notch ⇒ all pixels inside box
    assert int(z["mask"].sum()) == H * W

    # Damage invariants (§6.3): zeroed DOFs ⇒ damage ≡ 0, monotone trivially
    assert z["damage"].min() == 0.0
    assert z["damage"].max() <= 1.0 + 1e-6
    assert (z["damage"][1] >= z["damage"][0] - 1e-6).all()

    # Outside-Ω zero (§6.4): with no notch trivial; check the rule still fires
    # by verifying domain interior values aren't unexpectedly nonzero.
    assert np.abs(z["damage"] * (1 - z["mask"][None])).max() < 1e-6
    assert np.abs(z["stress"] * (1 - z["mask"][None])).max() < 1e-6

    # Schema version (§6.6)
    assert meta["schema_version"] == SCHEMA_VERSION

    # Meta required fields (§4)
    for key in (
        "schema_version", "sample_id", "grid", "material_params",
        "material_order", "scaling", "git_commit", "config_hash",
        "formulation", "interpolation", "solver_config", "stats",
    ):
        assert key in meta, f"missing required meta key {key!r}"

    assert meta["formulation"] in {"standard", "effective_stress"}
    assert meta["interpolation"] in {"I1_nearest_quad", "I2_L2_projection"}


def test_meta_json_written(tmp_dataset):
    pytest.importorskip("matplotlib")
    from fracturex.postprocess.dataset_export import (
        GridSpec, ExportConfig, CircularNotchDomain, export_recorder_to_sample,
    )

    discr, rec_dir, tmp_path = tmp_dataset
    cfg = ExportConfig(grid=GridSpec(H=8, W=8, bbox=((0.0, 1.0), (0.0, 1.0))))
    geom = CircularNotchDomain(box=(0, 1, 0, 1), cx=0.5, cy=0.5, r=0.0)
    out_npz = tmp_path / "s.npz"
    out_meta = tmp_path / "s.meta.json"
    export_recorder_to_sample(rec_dir, out_npz, out_meta, cfg, discr, geom)
    assert out_npz.exists()
    assert out_meta.exists()
    with out_meta.open() as f:
        meta = json.load(f)
    assert meta["grid"] == {"H": 8, "W": 8, "domain_bbox": [[0.0, 1.0], [0.0, 1.0]]}
