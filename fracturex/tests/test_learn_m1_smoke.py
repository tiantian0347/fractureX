"""M1 smoke test for fracturex.learn (operator-learning surrogate).

Fabricates a tiny SURROGATE_DATA_SCHEMA dataset directly (no solver run), then
trains each M1 baseline for one epoch through the shared train loop and asserts
the deliverables exist (config.json / metrics.csv / checkpoints / eval_report.md)
and the metric table is computable. This exercises the schema-decoupled training
side in isolation; the export side is covered by test_dataset_roundtrip.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _make_mini_dataset(root: Path, n: int = 4, H: int = 16, W: int = 16,
                       T: int = 3, k: int = 5) -> Path:
    """Write n schema-v0.1 samples + manifest with train/val splits."""
    samples_dir = root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    sample_recs = []
    for i in range(n):
        sid = f"sample_{i:06d}"
        mask = np.ones((1, H, W), dtype=np.uint8)
        # Monotone-in-time damage blob (schema §6.3).
        base = rng.random((H, W)).astype(np.float32) * 0.3
        damage = np.stack([np.clip(base * (s + 1) / T, 0, 1) for s in range(T)], axis=0)
        damage = damage[:, None, :, :]  # (T,1,H,W)
        stress = rng.standard_normal((T, 3, H, W)).astype(np.float32) * 0.1
        coords = np.stack(np.meshgrid(
            np.linspace(0, 1, W, dtype=np.float32),
            np.linspace(0, 1, H, dtype=np.float32), indexing="xy"), axis=0)
        np.savez_compressed(
            samples_dir / f"{sid}.npz",
            sdf=np.ones((1, H, W), np.float32),
            mask=mask,
            coords=coords.astype(np.float32),
            material=np.array([1.0, 0.8, 1.0, 0.02, 1e-9][:k], dtype=np.float32),
            load_history=np.linspace(0, 0.01, T, dtype=np.float32)[:, None],
            time=np.linspace(0, 1, T, dtype=np.float32),
            damage=damage.astype(np.float32),
            stress=stress,
            valid_mask=mask,
        )
        (samples_dir / f"{sid}.meta.json").write_text(json.dumps({
            "schema_version": "0.1", "sample_id": sid,
            "grid": {"H": H, "W": W, "domain_bbox": [[0, 1], [0, 1]]},
            "material_params": {"lambda": 1.0}, "material_order": ["lambda"],
            "scaling": {"stress_scale": 1.0}, "git_commit": "test",
            "config_hash": "test", "formulation": "standard",
            "interpolation": "I1_nearest_quad", "solver_config": {},
            "stats": {"max_damage": float(damage.max())},
        }))
        sample_recs.append({"id": sid, "npz": f"samples/{sid}.npz",
                            "meta": f"samples/{sid}.meta.json", "ok": True})
    ids = [r["id"] for r in sample_recs]
    (root / "dataset_manifest.json").write_text(json.dumps({
        "schema_version": "0.1", "dataset_name": "mini",
        "samples": sample_recs,
        "splits": {"train": ids[:-1], "val": ids[-1:]},
    }))
    return root


@pytest.mark.parametrize("model_name", ["unet", "fno_2d", "deeponet"])
def test_train_one_epoch(tmp_path, model_name):
    pytest.importorskip("torch")
    from fracturex.learn.train import TrainConfig, train

    ds_dir = _make_mini_dataset(tmp_path / "ds")
    out_dir = tmp_path / "runs" / model_name
    cfg = TrainConfig(
        dataset_dir=ds_dir, out_dir=out_dir, model=model_name,
        stage="A", epochs=1, batch_size=2, lr=1e-3, device="cpu",
    )
    final = train(cfg)

    assert (out_dir / "config.json").exists()
    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "eval_report.md").exists()
    assert (out_dir / "checkpoints" / "model_final.pt").exists()
    # metric table computed and finite where expected
    assert np.isfinite(final["relative_l2"])
    assert 0.0 <= final["crack_set_iou"] <= 1.0
    assert np.isfinite(final["ssim"])


@pytest.mark.parametrize("model_name", ["multioutput_fno", "fno_2d", "unet"])
def test_train_stage_b_one_epoch(tmp_path, model_name):
    """Stage B (d + σ): multi-output models train and report σ metrics."""
    pytest.importorskip("torch")
    from fracturex.learn.train import TrainConfig, train

    ds_dir = _make_mini_dataset(tmp_path / "ds")
    out_dir = tmp_path / "runs" / f"B_{model_name}"
    cfg = TrainConfig(
        dataset_dir=ds_dir, out_dir=out_dir, model=model_name,
        stage="B", epochs=1, batch_size=2, lr=1e-3, device="cpu",
        lambda_sigma=1.0,
    )
    final = train(cfg)
    assert (out_dir / "eval_report.md").exists()
    # σ metrics present and finite in Stage B
    assert np.isfinite(final["sigma_relative_l2"])
    assert np.isfinite(final["principal_stress_l2"])
    assert np.isfinite(final["relative_l2"])  # damage still scored


def test_sigma_arcsinh_transform_roundtrip():
    """arcsinh forward∘inverse is identity; compresses the heavy tail."""
    from fracturex.learn.transforms import sigma_forward, sigma_inverse

    x = np.array([-344.0, -14.0, -0.11, 0.0, 0.11, 14.0, 344.0], dtype=np.float64)
    y = sigma_forward(x, "arcsinh", 1.0)
    assert np.allclose(sigma_inverse(y, "arcsinh", 1.0), x, atol=1e-6)
    assert abs(y[-1]) < 7.0                      # asinh(344) ≈ 6.5 (tail compressed)
    assert np.allclose(y[4], 0.11, atol=2e-3)    # asinh(0.11) ≈ 0.11, ~linear near 0


def test_train_stage_b_arcsinh(tmp_path):
    """Stage B with arcsinh σ transform runs and reports physical + train σ error."""
    pytest.importorskip("torch")
    from fracturex.learn.train import TrainConfig, train

    ds_dir = _make_mini_dataset(tmp_path / "ds")
    out_dir = tmp_path / "runs" / "B_arcsinh"
    cfg = TrainConfig(
        dataset_dir=ds_dir, out_dir=out_dir, model="multioutput_fno",
        stage="B", epochs=1, batch_size=2, device="cpu",
        lambda_sigma=1.0, sigma_transform="arcsinh",
    )
    final = train(cfg)
    assert np.isfinite(final["sigma_relative_l2"])         # physical
    assert np.isfinite(final["sigma_relative_l2_train"])   # arcsinh space


def test_train_stage_b_peak_weighted(tmp_path):
    """Stage B with peak-weighted σ loss runs and reports the peak-region metric."""
    pytest.importorskip("torch")
    from fracturex.learn.train import TrainConfig, train

    ds_dir = _make_mini_dataset(tmp_path / "ds")
    out_dir = tmp_path / "runs" / "B_peak"
    cfg = TrainConfig(
        dataset_dir=ds_dir, out_dir=out_dir, model="multioutput_unet",
        stage="B", epochs=1, batch_size=2, device="cpu",
        lambda_sigma=1.0, sigma_loss="peak_weighted", sigma_peak_alpha=4.0,
    )
    final = train(cfg)
    assert np.isfinite(final["sigma_relative_l2"])
    assert np.isfinite(final["sigma_peak_relative_l2"])


def test_peak_load_error_grid():
    """Grid peak-load error: 0 for identical σ, scale-free, finite for scaled."""
    import numpy as np
    from fracturex.learn.eval.metrics import peak_load_error_grid
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((2, 5, 3, 8, 8)).astype(np.float32)
    mask = np.ones((2, 1, 8, 8), np.uint8)
    assert peak_load_error_grid(sig, sig, mask) == 0.0          # identical → 0
    assert peak_load_error_grid(2 * sig, sig, mask) > 0.0       # under/over → nonzero
    assert np.isfinite(peak_load_error_grid(sig * 0, sig, mask))


def test_dataset_reader_and_metrics_are_torch_free(tmp_path):
    """The numpy reader + metrics work without importing torch."""
    from fracturex.learn.datasets import (
        PhaseFieldOperatorDataset, assemble_input_channels, target_damage,
    )
    from fracturex.learn.eval import metrics as M

    ds_dir = _make_mini_dataset(tmp_path / "ds")
    ds = PhaseFieldOperatorDataset(ds_dir, split="train")
    assert len(ds) == 3
    assert ds.n_material == 5
    assert ds.n_input_channels == 9  # 4 + k
    s = ds[0]
    x = assemble_input_channels(s)
    y = target_damage(s)
    assert x.shape == (9, 16, 16)
    assert y.shape == (ds.n_steps, 16, 16)
    # metric sanity: identical arrays ⇒ zero error, perfect IoU
    assert M.relative_l2(y, y, s["valid_mask"]) == 0.0
    assert M.crack_set_iou(y, y, s["valid_mask"]) == 1.0
