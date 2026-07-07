"""扫描 strain-gradient 长度尺度 ℓ_s，产 model1 拉伸的 force-disp 曲线族。

对齐 `ipfem_paper.tex` §Model / §Discretization 加的应变梯度小节，
提供 ℓ_s → 力-位移响应的量化结果，展示尺寸效应：
- 较大 ℓ_s → 峰值力更高，起裂延后
- ℓ_s = 0 恢复经典 Aifantis-off (原 IPFEM 相场)

用法::

    source ~/venv_fealpy3/bin/activate
    python fracturex/interior_penalty/scripts/sweep_ells_model1.py

产出:
    /tmp/model1_ells_sweep.npz        (disp, force[per ell_s])
    /tmp/model1_ells_sweep.png        force-disp 族曲线
    /tmp/model1_ells_peak.png         peak force / peak disp vs ell_s
"""
from __future__ import annotations

import time
import numpy as np
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm

bm.set_backend("numpy")

from fracturex.interior_penalty.cases.model1_sg import Model1SGCase


def sweep(
    ell_s_list=(0.0, 0.005, 0.01, 0.02, 0.05),
    refine: int = 3,
    max_steps: int = 50,
    maxit_per_step: int = 20,
    rtol: float = 1e-4,
    sg_split: bool = False,
    disp_range=(0.0, 5e-3),
) -> dict:
    disp_seq = np.linspace(disp_range[0], disp_range[1], max_steps + 1)
    all_force = np.zeros((len(ell_s_list), len(disp_seq)))
    all_time = []
    peak_force = []
    peak_disp = []

    for i, ell_s in enumerate(ell_s_list):
        t0 = time.time()
        case = Model1SGCase(
            refine=refine,
            ell_s=ell_s,
            sg_split=sg_split,
            load_sequence=disp_seq,
        )
        out = case.run(maxit_per_step=maxit_per_step, rtol=rtol, verbose=False)
        elapsed = time.time() - t0
        all_time.append(elapsed)
        all_force[i, :] = out["force"]
        pk = int(np.argmax(np.abs(out["force"])))
        peak_force.append(float(out["force"][pk]))
        peak_disp.append(float(out["disp"][pk]))
        print(
            f"[sweep] ell_s={ell_s:.4f}  elapsed={elapsed:.1f}s  "
            f"peak_force={peak_force[-1]:.4f} @ disp={peak_disp[-1]:.4e}"
        )
    return dict(
        disp=disp_seq,
        force=all_force,
        ell_s_list=np.array(ell_s_list),
        peak_force=np.array(peak_force),
        peak_disp=np.array(peak_disp),
        time_s=np.array(all_time),
    )


def plot(result: dict, prefix: str = "/tmp/model1_ells"):
    disp = result["disp"]
    force = result["force"]
    ell_s_list = result["ell_s_list"]

    # Force-disp curves
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, ell_s in enumerate(ell_s_list):
        ax.plot(disp, force[i], "-o", ms=3,
                label=f"ℓ_s={ell_s:.3f} (ℓ_0=0.0133)")
    ax.set_xlabel("displacement")
    ax.set_ylabel("force")
    ax.set_title("Notched-square SENT tension: elastic length-scale size effect (ℓ_s sweep)")
    ax.grid(True)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fname = f"{prefix}_sweep.png"
    fig.savefig(fname, dpi=150)
    print(f"[plot] saved {fname}")
    plt.close(fig)

    # Peak force + peak disp vs ell_s
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ell_s_list, result["peak_force"], "o-", color="C0")
    ax1.set_xlabel("ℓ_s")
    ax1.set_ylabel("peak force")
    ax1.set_title("Peak force vs ℓ_s")
    ax1.grid(True)
    ax2.plot(ell_s_list, result["peak_disp"], "s-", color="C1")
    ax2.set_xlabel("ℓ_s")
    ax2.set_ylabel("peak displacement")
    ax2.set_title("Peak-force displacement vs ℓ_s")
    ax2.grid(True)
    fig.tight_layout()
    fname = f"{prefix}_peak.png"
    fig.savefig(fname, dpi=150)
    print(f"[plot] saved {fname}")
    plt.close(fig)


def plot_normalized(result: dict, l0: float, prefix: str = "/tmp/model1_ells"):
    """把 ell_s 归一化到 ℓ_0，展示应变梯度/裂纹尺度比对峰值力的调制。"""
    ell_s = result["ell_s_list"]
    peak_f = result["peak_force"]
    peak_d = result["peak_disp"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = ell_s / l0
    ax1.plot(x, peak_f, "o-", color="C0")
    ax1.set_xlabel(r"$\ell_s / \ell_0$")
    ax1.set_ylabel("peak force")
    ax1.set_title(r"Peak force vs $\ell_s / \ell_0$ (SENT tension, $\ell_0=0.0133$)")
    ax1.grid(True)
    # 相对 ell_s=0 的 stiffening 百分比
    baseline = peak_f[0]
    for i, (xi, fi) in enumerate(zip(x, peak_f)):
        pct = 100 * (fi - baseline) / baseline
        ax1.annotate(f"+{pct:.1f}%" if pct > 0 else "0%",
                     (xi, fi), textcoords="offset points",
                     xytext=(6, 0), fontsize=8)

    ax2.plot(x, peak_d, "s-", color="C1")
    ax2.set_xlabel(r"$\ell_s / \ell_0$")
    ax2.set_ylabel("peak displacement")
    ax2.set_title(r"Peak-force displacement vs $\ell_s / \ell_0$")
    ax2.grid(True)
    fig.tight_layout()
    fname = f"{prefix}_peak_normalized.png"
    fig.savefig(fname, dpi=150)
    print(f"[plot] saved {fname}")
    plt.close(fig)


def main():
    result = sweep(
        ell_s_list=(0.0, 0.02, 0.05, 0.1, 0.15, 0.2),
        refine=3,
        max_steps=40,
        maxit_per_step=15,
        rtol=1e-4,
        disp_range=(0.0, 5e-3),
    )
    np.savez("/tmp/model1_ells_sweep.npz", **result)
    print("[sweep] saved /tmp/model1_ells_sweep.npz")
    plot(result, prefix="/tmp/model1_ells")
    plot_normalized(result, l0=0.0133, prefix="/tmp/model1_ells")


if __name__ == "__main__":
    main()
