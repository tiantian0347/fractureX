"""Run the M1 damage-only baseline comparison (plan §M1).

Given a generated dataset directory, this:
  1. assigns deterministic train/test splits and records them in the manifest
     (schema §5 requires splits to be persisted for reproducibility);
  2. trains each baseline (U-Net / FNO2d / DeepONet, Stage A, damage-only)
     through the shared `fracturex.learn.train.train`;
  3. writes a combined comparison table + a training-curve plot.

Deliverable (plan §M1): three-baseline metric table + training curves +
per-model eval_report.md.

Run:
    PYTHONPATH=$PWD $FEALPY_PYTHON scripts/datasets/run_m1_experiment.py \\
      --dataset-dir results/datasets/m1_pilot \\
      --out-dir results/learn/m1_pilot --epochs 300
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def assign_splits(dataset_dir: Path, test_frac: float = 0.3, seed: int = 0) -> dict:
    """Add deterministic {train,test} splits to the manifest (idempotent-ish).

    Overwrites any existing splits so re-runs are reproducible for a given seed.
    """
    man_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(man_path.read_text())
    ids = [s["id"] for s in manifest.get("samples", []) if s.get("ok", True)]
    ids = sorted(ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_test = max(1, int(round(len(ids) * test_frac)))
    test_idx = set(perm[:n_test].tolist())
    train = [ids[i] for i in range(len(ids)) if i not in test_idx]
    test = [ids[i] for i in range(len(ids)) if i in test_idx]
    manifest["splits"] = {"train": train, "test": test}
    man_path.write_text(json.dumps(manifest, indent=2))
    return manifest["splits"]


def read_curve(model_out: Path) -> list[dict]:
    rows = []
    with (model_out / "metrics.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--models", nargs="+", default=None,
                    help="default: stage A → unet/fno_2d/deeponet; "
                         "stage B → multioutput_fno/multioutput_unet")
    ap.add_argument("--stage", default="A", help="A (damage-only) | B (d+σ)")
    ap.add_argument("--lambda-sigma", type=float, default=1.0,
                    help="σ loss weight (Stage B)")
    ap.add_argument("--sigma-transform", default="none",
                    help="σ target transform (Stage B): none | arcsinh")
    ap.add_argument("--sigma-loss", default="rel_l2",
                    help="σ loss (Stage B): rel_l2 | peak_weighted")
    ap.add_argument("--sigma-peak-alpha", type=float, default=4.0,
                    help="peak up-weight α for --sigma-loss peak_weighted")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from fracturex.learn.train import TrainConfig, train

    stage = args.stage.upper()
    if args.models is None:
        args.models = (["multioutput_fno", "multioutput_unet"] if stage != "A"
                       else ["unet", "fno_2d", "deeponet"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = assign_splits(args.dataset_dir, args.test_frac, args.seed)
    print(f"stage={stage}  splits: train={len(splits['train'])}  test={len(splits['test'])}")

    results: dict[str, dict] = {}
    curves: dict[str, list[dict]] = {}
    for model in args.models:
        print(f"\n=== training {model} (stage {stage}) ===")
        out = args.out_dir / model
        cfg = TrainConfig(
            dataset_dir=args.dataset_dir, out_dir=out, model=model,
            stage=stage, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, device="cpu", lambda_sigma=args.lambda_sigma,
            sigma_transform=args.sigma_transform,
            sigma_loss=args.sigma_loss, sigma_peak_alpha=args.sigma_peak_alpha,
            train_split="train", val_split="test", seed=args.seed,
        )
        results[model] = train(cfg)
        curves[model] = read_curve(out)
        print(f"{model} final: " + ", ".join(f"{k}={v:.4g}" for k, v in results[model].items()))

    # comparison table — metric columns taken from whatever the stage reported
    base_keys = ["relative_l2", "relative_h1", "crack_set_iou",
                 "crack_front_hausdorff", "ssim"]
    extra = [k for k in ("sigma_relative_l2", "sigma_peak_relative_l2",
                          "peak_load_error", "principal_stress_l2",
                          "sigma_relative_l2_train")
             if any(k in results[m] for m in args.models)]
    metric_keys = base_keys + extra
    title = ("# M1 baseline comparison (Stage A, damage-only)" if stage == "A"
             else f"# M2 Stage {stage} comparison (damage + Hu-Zhang σ supervision)")
    lines = [
        title, "",
        f"- dataset: `{args.dataset_dir}` (train={len(splits['train'])}, test={len(splits['test'])})",
        f"- epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.lr}"
        + (f", λ_σ={args.lambda_sigma}" if stage != "A" else ""),
        f"- metrics evaluated on held-out **test** split",
        "",
        "| model | " + " | ".join(metric_keys) + " |",
        "| --- |" + " --- |" * len(metric_keys),
    ]
    for model in args.models:
        row = [model] + [f"{results[model].get(k, float('nan')):.4g}" for k in metric_keys]
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "> lower is better for relative_l2/_h1/hausdorff/σ; "
              "higher for crack_set_iou/ssim.",
              "> σ metrics on normalized stress; reaction/equilibrium → Stage D."]
    (args.out_dir / "comparison_table.md").write_text("\n".join(lines))
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # training curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for model in args.models:
            ep = [r["epoch"] for r in curves[model]]
            axes[0].plot(ep, [r["train_loss"] for r in curves[model]], label=model)
            axes[1].plot(ep, [r["val_loss"] for r in curves[model]], label=model)
        for ax, title in zip(axes, ["train loss (masked rel. L2)", "test loss (masked rel. L2)"]):
            ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_yscale("log")
            ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "training_curves.png", dpi=120)
        print(f"\ncurves -> {args.out_dir/'training_curves.png'}")
    except Exception as e:  # plotting is best-effort
        print(f"[warn] could not plot curves: {e}")

    print(f"table   -> {args.out_dir/'comparison_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
