# fracturex/learn/train.py
"""Training loop entry point — model-agnostic, schema-driven.

PyTorch is the default backend. Each stage from plan §3.5 (A → E) is a
separate ``train(...)`` invocation, selected by ``TrainConfig.stage``:

  - Stage A (M1): damage-only, loss ``L_d``. Output (B, T, H, W).
  - Stage B (M2): damage + Hu-Zhang stress, loss ``L_d + λ_σ L_σ``.
    Output (B, T, 4, H, W) with fields (d, σxx, σyy, σxy); needs a
    multi-output model (``multioutput_fno`` / ``multioutput_unet``).

Models are built through :func:`fracturex.learn.models.build_model`.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .datasets import (
    DatasetConfig,
    PhaseFieldOperatorDataset,
    as_torch_dataset,
    collate_masked,
)
from .eval import metrics as M
from .losses import masked_relative_l2, peak_weighted_relative_l2, stage_loss
from .models import build_model
from .transforms import sigma_forward, sigma_inverse

# Stage A predicts damage only; Stage B predicts (d, σxx, σyy, σxy).
_STAGE_N_FIELDS = {"A": 1, "B": 4, "C": 4, "D": 4, "E": 4}
# Auto-map a plain backbone name to its multi-output variant for n_fields>1.
_MULTI_MAP = {"fno_2d": "multioutput_fno", "fno": "multioutput_fno",
              "fno2d": "multioutput_fno", "unet": "multioutput_unet"}


@dataclass
class TrainConfig:
    dataset_dir: Path
    out_dir: Path
    model: str                  # 'unet' | 'fno_2d' | 'deeponet' | 'multioutput_fno' | 'multioutput_unet'
    stage: str = "A"            # A | B | C | D | E   (plan §3.5)
    epochs: int = 100
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 0.0
    rollout_length: Optional[int] = None
    seed: int = 0
    device: str = "cpu"
    train_split: str = "train"
    val_split: str = "val"
    model_kwargs: Optional[dict] = None
    # σ target transform (Stage B): 'none' | 'arcsinh' (heavy-tail compression)
    sigma_transform: str = "none"
    sigma_transform_scale: float = 1.0
    # σ loss (Stage B): 'rel_l2' | 'peak_weighted' (up-weight crack-tip peak)
    sigma_loss: str = "rel_l2"
    sigma_peak_alpha: float = 4.0
    # Loss weights (overrides plan §3.5 defaults if non-None)
    lambda_sigma: Optional[float] = None
    lambda_h1: Optional[float] = None
    lambda_front: Optional[float] = None
    lambda_eq: Optional[float] = None
    lambda_irr: Optional[float] = None


def _make_loader(dataset_dir, split, batch_size, shuffle):
    from torch.utils.data import DataLoader

    ds = PhaseFieldOperatorDataset(dataset_dir, split=split, cfg=DatasetConfig())
    loader = DataLoader(
        as_torch_dataset(ds), batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_masked, num_workers=0,
    )
    return ds, loader


def _split_pred(pred, stage: str):
    """Split a model output into (damage, stress-or-None) per stage.

    Stage A: pred is (B, T, H, W) → (damage, None).
    Stage B: pred is (B, T, 4, H, W) → (d=pred[:,:,0], σ=pred[:,:,1:4]).
    """
    if _STAGE_N_FIELDS.get(stage.upper(), 1) == 1:
        return pred, None
    return pred[:, :, 0], pred[:, :, 1:4]


def _compute_loss(pred, batch, cfg, device):
    """Stage-aware loss; returns (total_loss, components_dict of floats)."""
    y = batch["y"].to(device)
    mask = batch["mask"].to(device)
    d_pred, sigma_pred = _split_pred(pred, cfg.stage)
    l_d = masked_relative_l2(d_pred, y, mask)
    if sigma_pred is None:
        return stage_loss(cfg.stage, l_d=l_d), {"l_d": float(l_d.detach())}
    # Model predicts σ in the (possibly transformed) training space; supervise
    # against the transformed target so a heavy tail doesn't dominate L².
    sigma_t = sigma_forward(batch["stress"].to(device),
                            cfg.sigma_transform, cfg.sigma_transform_scale)
    if cfg.sigma_loss == "peak_weighted":
        l_sigma = peak_weighted_relative_l2(sigma_pred, sigma_t, mask, alpha=cfg.sigma_peak_alpha)
    else:
        l_sigma = masked_relative_l2(sigma_pred, sigma_t, mask)
    total = stage_loss(cfg.stage, l_d=l_d, l_sigma=l_sigma, lambda_sigma=cfg.lambda_sigma)
    return total, {"l_d": float(l_d.detach()), "l_sigma": float(l_sigma.detach())}


def _evaluate(model, loader, stage, device,
              sigma_transform="none", sigma_scale=1.0) -> dict:
    """Average the metric table over a loader (adds σ metrics for Stage B).

    σ predictions are inverted from the training space (e.g. arcsinh) back to
    physical/normalized units before scoring, so ``sigma_relative_l2`` /
    ``principal_stress_l2`` are honest physical-unit errors. The training-space
    error is also reported as ``sigma_relative_l2_train`` (what the loss saw).
    """
    import torch

    model.eval()
    agg: dict[str, list[float]] = {
        "relative_l2": [], "relative_h1": [], "crack_set_iou": [],
        "crack_front_hausdorff": [], "ssim": [],
    }
    stage_b = _STAGE_N_FIELDS.get(stage.upper(), 1) > 1
    if stage_b:
        agg.update({"sigma_relative_l2": [], "principal_stress_l2": [],
                    "sigma_peak_relative_l2": [], "peak_load_error": [],
                    "sigma_relative_l2_train": []})
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["mask"].cpu().numpy()
            d_pred, sigma_pred = _split_pred(model(x), stage)
            dp = d_pred.cpu().numpy()
            dt = batch["y"].cpu().numpy()
            agg["relative_l2"].append(M.relative_l2(dp, dt, mask))
            agg["relative_h1"].append(M.relative_h1(dp, dt, mask))
            agg["crack_set_iou"].append(M.crack_set_iou(dp, dt, mask))
            hd = M.crack_front_hausdorff(dp, dt, mask)
            if np.isfinite(hd):
                agg["crack_front_hausdorff"].append(hd)
            agg["ssim"].append(M.ssim_masked(dp, dt, mask))
            if stage_b and sigma_pred is not None:
                sp_train = sigma_pred.cpu().numpy()                 # training space
                st_phys = batch["stress"].cpu().numpy()            # physical(normalized)
                sp_phys = sigma_inverse(sp_train, sigma_transform, sigma_scale)
                st_train = sigma_forward(st_phys, sigma_transform, sigma_scale)
                agg["sigma_relative_l2"].append(M.stress_relative_l2(sp_phys, st_phys, mask))
                agg["principal_stress_l2"].append(
                    M.principal_stress_relative_l2(sp_phys, st_phys, mask))
                agg["sigma_peak_relative_l2"].append(
                    M.sigma_peak_relative_l2(sp_phys, st_phys, mask, q=95.0))
                agg["peak_load_error"].append(
                    M.peak_load_error_grid(sp_phys, st_phys, mask))
                agg["sigma_relative_l2_train"].append(
                    M.stress_relative_l2(sp_train, st_train, mask))
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in agg.items()}


def _build_for_stage(cfg, in_ch, T, hw, device):
    """Build the model, auto-selecting a multi-output variant for Stage B."""
    n_fields = _STAGE_N_FIELDS.get(cfg.stage.upper(), 1)
    name = cfg.model
    if n_fields > 1 and not name.lower().startswith(("multioutput", "mo_")):
        if name.lower() not in _MULTI_MAP:
            raise ValueError(
                f"stage {cfg.stage} needs a multi-output model; "
                f"{name!r} has no multi-output variant (use fno_2d/unet or multioutput_*)"
            )
        name = _MULTI_MAP[name.lower()]
    model = build_model(name, in_ch, T, grid_hw=hw, n_fields=n_fields,
                        **(cfg.model_kwargs or {}))
    return model.to(device), name, n_fields


def train(cfg: TrainConfig) -> dict:
    """Single-stage training run.

    Writes:
      out_dir/{config.json, metrics.csv, checkpoints/model_epoch_XXX.pt,
               eval_report.md}

    Returns the final evaluation metric dict (on the val/test split).
    """
    import torch

    out_dir = Path(cfg.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    train_ds, train_loader = _make_loader(
        cfg.dataset_dir, cfg.train_split, cfg.batch_size, shuffle=True
    )
    try:
        val_ds, val_loader = _make_loader(
            cfg.dataset_dir, cfg.val_split, cfg.batch_size, shuffle=False
        )
    except (ValueError, FileNotFoundError):
        val_ds, val_loader = train_ds, train_loader

    in_ch = train_ds.n_input_channels
    T = train_ds.n_steps
    H, W = train_ds.grid_hw
    model, resolved_name, n_fields = _build_for_stage(cfg, in_ch, T, (H, W), device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    cfg_dump = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    cfg_dump.update({"in_channels": in_ch, "T": T, "grid": [H, W],
                     "n_train": len(train_ds), "resolved_model": resolved_name,
                     "n_fields": n_fields})
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2))

    metrics_rows = []
    for epoch in range(cfg.epochs):
        model.train()
        ep_loss, n = 0.0, 0
        for batch in train_loader:
            pred = model(batch["x"].to(device))
            loss, _ = _compute_loss(pred, batch, cfg, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = batch["x"].shape[0]
            ep_loss += float(loss.detach()) * bs
            n += bs
        train_loss = ep_loss / max(n, 1)

        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for batch in val_loader:
                loss, _ = _compute_loss(model(batch["x"].to(device)), batch, cfg, device)
                bs = batch["x"].shape[0]
                vl += float(loss) * bs
                vn += bs
            val_loss = vl / max(vn, 1)

        metrics_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    # Save only the final model — per-epoch checkpoints are ~64 MB each and
    # bloat the disk (300 epochs × N models). The metric curves live in
    # metrics.csv; reload model_final.pt for diagnostics / inference.
    torch.save(model.state_dict(), out_dir / "checkpoints" / "model_final.pt")

    with (out_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        w.writeheader()
        w.writerows(metrics_rows)

    final = _evaluate(model, val_loader, cfg.stage, device,
                      sigma_transform=cfg.sigma_transform,
                      sigma_scale=cfg.sigma_transform_scale)
    lines = [
        f"# Eval report — model={resolved_name}, stage={cfg.stage}",
        "",
        f"- dataset: `{cfg.dataset_dir}`",
        f"- epochs: {cfg.epochs}, batch_size: {cfg.batch_size}, lr: {cfg.lr}"
        + (f", λ_σ={cfg.lambda_sigma}" if n_fields > 1 else ""),
        f"- val split: `{cfg.val_split}` (n={len(val_ds)})",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for k, v in final.items():
        lines.append(f"| {k} | {v:.6g} |")
    lines.append("")
    if n_fields > 1:
        lines.append("> σ metrics on normalized stress (per-sample stress_scale). "
                     "reaction / equilibrium residual land in Stage D.")
    else:
        lines.append("> peak_load / equilibrium need reaction & σ (Stage B/D).")
    (out_dir / "eval_report.md").write_text("\n".join(lines))

    return final
