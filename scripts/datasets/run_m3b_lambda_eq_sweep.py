"""M3b Stage D sweep — HZ supervision × λ_eq ∈ {0, 0.01, 0.1, 1.0}.

Runs the A / A' groups of the M3b.5 experiment (paper_thesis §F.3 / §G):

  A  : σ_h supervision, λ_eq = 0            (baseline, plain Stage B loss)
  A' : σ_h supervision, λ_eq ∈ {0.01, 0.1, 1.0}  (Stage D balance regularizer)

Group B / B' (σ_h^rec supervision) needs stress_rec in the dataset — that
lands after M3b.4 is done. Run:

    PYTHONPATH=$PWD python scripts/datasets/run_m3b_lambda_eq_sweep.py \\
        --dataset-dir results/datasets/m1_pilot \\
        --out-dir     results/learn/m3b_sweep \\
        --epochs 100 --model multioutput_fno
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def assign_splits(dataset_dir: Path, test_frac: float = 0.3, seed: int = 0) -> dict:
    man_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(man_path.read_text())
    if manifest.get("splits") and "train" in manifest["splits"] and "test" in manifest["splits"]:
        return manifest["splits"]
    ids = sorted(s["id"] for s in manifest.get("samples", []) if s.get("ok", True))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_test = max(1, int(round(len(ids) * test_frac)))
    test_idx = set(perm[:n_test].tolist())
    train = [ids[i] for i in range(len(ids)) if i not in test_idx]
    test = [ids[i] for i in range(len(ids)) if i in test_idx]
    manifest["splits"] = {"train": train, "test": test}
    man_path.write_text(json.dumps(manifest, indent=2))
    return manifest["splits"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model", default="multioutput_fno")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-sigma", type=float, default=1.0)
    ap.add_argument("--lambda-eq", nargs="+", type=float,
                    default=[0.0, 0.01, 0.1, 1.0])
    ap.add_argument("--supervision", default="sigma_h",
                    choices=["sigma_h", "sigma_h_rec"])
    ap.add_argument("--sigma-transform", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from fracturex.learn.train import TrainConfig, train

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = assign_splits(args.dataset_dir, seed=args.seed)
    print(f"[m3b] dataset={args.dataset_dir} splits: train={len(splits['train'])} test={len(splits['test'])}")

    summary = []
    for lam in args.lambda_eq:
        tag = f"leq{lam:g}".replace(".", "p")
        out = args.out_dir / f"{args.supervision}_{tag}"
        print(f"\n=== M3b: {args.supervision}  λ_eq={lam}  → {out}")
        cfg = TrainConfig(
            dataset_dir=args.dataset_dir, out_dir=out,
            model=args.model, stage="B",
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            device=args.device, seed=args.seed,
            train_split="train", val_split="test",
            sigma_transform=args.sigma_transform,
            lambda_sigma=args.lambda_sigma,
            lambda_eq=lam if lam > 0.0 else None,
            supervision_source=args.supervision,
        )
        final = train(cfg)
        summary.append({"lambda_eq": lam, "out": str(out), **final})
        print(f"→ {final}")

    (args.out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[m3b] wrote {args.out_dir / 'sweep_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
