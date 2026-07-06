"""通用 ℓ_s sweep 脚本，可传入任何 SG 版 case 类。

内部使用与 `sweep_ells_model1.py` 相同的 plot 逻辑，用于 model0 (circular
hole) 与 model2 (shear) 上重复的应变梯度尺寸效应实验。

用法::

    python fracturex/interior_penalty/scripts/sweep_ells_generic.py model0
    python fracturex/interior_penalty/scripts/sweep_ells_generic.py model2
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm

bm.set_backend("numpy")


def _get_case_config(name: str):
    """按 case name 返回 (case_class, base_kwargs, l0)。"""
    if name == "model0":
        from fracturex.interior_penalty.cases import Model0SGCase
        return Model0SGCase, dict(hmin=0.1, distmesh_maxit=50), 0.02
    if name == "model1":
        from fracturex.interior_penalty.cases import Model1SGCase
        return Model1SGCase, dict(refine=3), 0.0133
    if name == "model2":
        from fracturex.interior_penalty.cases import Model2SGCase
        return Model2SGCase, dict(refine=3), 0.004
    raise ValueError(f"unknown case name '{name}'")


def sweep(name: str, ell_s_list, max_steps: int, maxit_per_step: int, rtol: float,
          disp_range) -> dict:
    case_cls, base_kwargs, l0 = _get_case_config(name)
    disp_seq = np.linspace(disp_range[0], disp_range[1], max_steps + 1)

    all_force = np.zeros((len(ell_s_list), len(disp_seq)))
    peak_force = []
    peak_disp = []
    all_time = []
    for i, ell_s in enumerate(ell_s_list):
        t0 = time.time()
        case = case_cls(load_sequence=disp_seq, ell_s=ell_s, **base_kwargs)
        out = case.run(maxit_per_step=maxit_per_step, rtol=rtol, verbose=False)
        elapsed = time.time() - t0
        all_time.append(elapsed)
        all_force[i, :] = out["force"]
        pk = int(np.argmax(np.abs(out["force"])))
        peak_force.append(float(out["force"][pk]))
        peak_disp.append(float(out["disp"][pk]))
        print(
            f"[{name}] ell_s={ell_s:.4f}  elapsed={elapsed:.1f}s  "
            f"peak_force={peak_force[-1]:.4f} @ disp={peak_disp[-1]:.4e}"
        )

    return dict(
        name=name,
        disp=disp_seq,
        force=all_force,
        ell_s_list=np.array(ell_s_list),
        peak_force=np.array(peak_force),
        peak_disp=np.array(peak_disp),
        time_s=np.array(all_time),
        l0=l0,
    )


def plot(result: dict, prefix: str):
    disp = result["disp"]
    force = result["force"]
    ell_s_list = result["ell_s_list"]
    l0 = result["l0"]
    name = result["name"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, ell_s in enumerate(ell_s_list):
        ax.plot(disp, force[i], "-o", ms=3, label=f"ℓ_s={ell_s:.3f}")
    ax.set_xlabel("displacement")
    ax.set_ylabel("force")
    ax.set_title(f"{name}: strain-gradient size effect (ℓ_0={l0})")
    ax.grid(True)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fname = f"{prefix}_sweep.png"
    fig.savefig(fname, dpi=150)
    print(f"[plot] saved {fname}")
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = ell_s_list / l0
    ax1.plot(x, result["peak_force"], "o-", color="C0")
    ax1.set_xlabel(r"$\ell_s / \ell_0$")
    ax1.set_ylabel("peak force")
    ax1.set_title(rf"{name}: Peak force vs $\ell_s/\ell_0$ ($\ell_0={l0}$)")
    ax1.grid(True)
    baseline = result["peak_force"][0]
    for xi, fi in zip(x, result["peak_force"]):
        pct = 100 * (fi - baseline) / max(abs(baseline), 1e-16)
        ax1.annotate(
            f"+{pct:.1f}%" if pct > 0.1 else ("0%" if abs(pct) <= 0.1 else f"{pct:.1f}%"),
            (xi, fi), textcoords="offset points", xytext=(6, 0), fontsize=8,
        )
    ax2.plot(x, result["peak_disp"], "s-", color="C1")
    ax2.set_xlabel(r"$\ell_s / \ell_0$")
    ax2.set_ylabel("peak displacement")
    ax2.set_title(rf"{name}: Peak-force disp vs $\ell_s/\ell_0$")
    ax2.grid(True)
    fig.tight_layout()
    fname = f"{prefix}_peak_normalized.png"
    fig.savefig(fname, dpi=150)
    print(f"[plot] saved {fname}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=["model0", "model1", "model2"])
    ap.add_argument("--ell-s", nargs="+", type=float,
                    default=[0.0, 0.02, 0.05, 0.1, 0.15, 0.2])
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--maxit", type=int, default=15)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--disp-max", type=float, default=5e-3)
    ap.add_argument("--outdir", type=str, default="/tmp")
    args = ap.parse_args()

    result = sweep(
        name=args.case,
        ell_s_list=args.ell_s,
        max_steps=args.max_steps,
        maxit_per_step=args.maxit,
        rtol=args.rtol,
        disp_range=(0.0, args.disp_max),
    )
    prefix = f"{args.outdir}/{args.case}_ells"
    np.savez(f"{prefix}_sweep.npz", **result)
    print(f"[sweep] saved {prefix}_sweep.npz")
    plot(result, prefix=prefix)


if __name__ == "__main__":
    main()
