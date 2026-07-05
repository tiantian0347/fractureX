"""扫描内罚参数 γ ∈ {1, 3, 5, 10, 20, 50} 对 model1 SENT 的鲁棒性。

对齐 `ipfem_paper.tex` §Conclusion.② 里点名的 open extension 之一：
"A systematic sensitivity study for the penalty parameter would also clarify
the robustness of the chosen values."

产出:
    /tmp/model1_gamma_sweep.npz     每步 disp / force[per γ]
    /tmp/model1_gamma_sweep.png     force-disp 曲线族
    /tmp/model1_gamma_peak.png      peak force / peak disp vs γ
    /tmp/model1_gamma_cond.png      A_biharm+IP 矩阵条件数估计 vs γ (可选)

用法::

    source ~/venv_fealpy3/bin/activate
    python fracturex/interior_penalty/scripts/sweep_gamma_model1.py
"""
from __future__ import annotations

import time
import numpy as np
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm

bm.set_backend("numpy")

from fracturex.interior_penalty.cases.model1_square_tension import Model1SquareTensionCase


def sweep_gamma(
    gamma_list=(1.0, 3.0, 5.0, 10.0, 20.0, 50.0),
    refine: int = 3,
    max_steps: int = 30,
    maxit_per_step: int = 15,
    rtol: float = 1e-4,
    disp_range=(0.0, 5e-3),
) -> dict:
    disp_seq = np.linspace(disp_range[0], disp_range[1], max_steps + 1)
    all_force = np.zeros((len(gamma_list), len(disp_seq)))
    all_time = []
    peak_force = []
    peak_disp = []
    converged = []
    for i, gamma in enumerate(gamma_list):
        t0 = time.time()
        case = Model1SquareTensionCase(
            refine=refine, gamma=gamma, load_sequence=disp_seq
        )
        try:
            out = case.run(maxit_per_step=maxit_per_step, rtol=rtol, verbose=False)
            elapsed = time.time() - t0
            all_time.append(elapsed)
            all_force[i, :] = out["force"]
            # 简单收敛判定：force 不 NaN 且末段力有意义
            ok = np.all(np.isfinite(out["force"]))
            converged.append(ok)
            pk = int(np.argmax(np.abs(out["force"])))
            peak_force.append(float(out["force"][pk]))
            peak_disp.append(float(out["disp"][pk]))
            print(
                f"[γ-sweep] γ={gamma:6.1f}  elapsed={elapsed:.1f}s  "
                f"peak={peak_force[-1]:.4f} @ disp={peak_disp[-1]:.4e}"
                f"  converged={ok}"
            )
        except Exception as exc:
            elapsed = time.time() - t0
            all_time.append(elapsed)
            all_force[i, :] = np.nan
            converged.append(False)
            peak_force.append(np.nan)
            peak_disp.append(np.nan)
            print(f"[γ-sweep] γ={gamma:6.1f}  FAILED: {exc}")

    return dict(
        disp=disp_seq,
        force=all_force,
        gamma_list=np.array(gamma_list),
        peak_force=np.array(peak_force),
        peak_disp=np.array(peak_disp),
        time_s=np.array(all_time),
        converged=np.array(converged),
    )


def plot(result: dict, prefix: str = "/tmp/model1_gamma"):
    disp = result["disp"]
    force = result["force"]
    gamma_list = result["gamma_list"]

    # Force-disp curves
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, gamma in enumerate(gamma_list):
        if not result["converged"][i]:
            continue
        ax.plot(disp, force[i], "-o", ms=3, label=f"γ={gamma:.0f}")
    ax.set_xlabel("displacement")
    ax.set_ylabel("force")
    ax.set_title("model1: penalty γ sensitivity")
    ax.grid(True)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fname = f"{prefix}_sweep.png"
    fig.savefig(fname, dpi=150)
    print(f"[plot] saved {fname}")
    plt.close(fig)

    # Peak force / disp vs γ (semilogx)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ok = result["converged"]
    gvalid = gamma_list[ok]
    pf = np.array(result["peak_force"])[ok]
    pd = np.array(result["peak_disp"])[ok]
    ax1.semilogx(gvalid, pf, "o-", color="C0")
    ax1.set_xlabel(r"$\gamma$ (log scale)")
    ax1.set_ylabel("peak force")
    ax1.set_title(r"Peak force vs $\gamma$")
    ax1.grid(True, which="both")
    ax2.semilogx(gvalid, pd, "s-", color="C1")
    ax2.set_xlabel(r"$\gamma$ (log scale)")
    ax2.set_ylabel("peak displacement")
    ax2.set_title(r"Peak-force displacement vs $\gamma$")
    ax2.grid(True, which="both")
    fig.tight_layout()
    fname = f"{prefix}_peak.png"
    fig.savefig(fname, dpi=150)
    print(f"[plot] saved {fname}")
    plt.close(fig)


def compute_condition_numbers(gamma_list, refine: int = 2):
    """粗网格上估计 A_bh + A_IP(γ) 的条件数（谱范数）。

    只是初步定性——用小 dense eigvalue 计算，仅适用于小矩阵。
    """
    from scipy.sparse.linalg import eigsh
    from fracturex.interior_penalty.solver import _to_scipy_csr, _assemble_interior_penalty
    from fealpy.fem import BilinearForm, ScalarBiharmonicIntegrator
    from fealpy.functionspace import InteriorPenaltyFESpace2d
    from fracturex.interior_penalty.cases.model1_square_tension import _default_init_mesh

    mesh = _default_init_mesh(refine)
    ipspace = InteriorPenaltyFESpace2d(mesh, p=2)
    q = 7

    bform = BilinearForm(ipspace)
    bform.add_integrator(ScalarBiharmonicIntegrator(q=q))
    A_bh = _to_scipy_csr(bform.assembly())

    conds = []
    for gamma in gamma_list:
        A_ip = _assemble_interior_penalty(ipspace, gamma=gamma, q=q)
        A = A_bh + A_ip
        # 用 shift 让 A 保持正定（消除 kernel）
        A_shifted = A + 1e-8 * np.eye(A.shape[0])
        try:
            # 谱两端
            lam_max, _ = eigsh(A_shifted, k=1, which="LM", return_eigenvectors=True, tol=1e-4)
            lam_min, _ = eigsh(A_shifted, k=1, which="SM", return_eigenvectors=True, tol=1e-4, sigma=0.0)
            cond = float(np.abs(lam_max) / max(np.abs(lam_min), 1e-16))
        except Exception:
            # SM 有时不稳，退化到 dense
            arr = A_shifted.toarray() if hasattr(A_shifted, "toarray") else A_shifted
            eigs = np.linalg.eigvalsh(arr)
            cond = float(eigs.max() / max(eigs.min(), 1e-16))
        conds.append(cond)
        print(f"[cond] γ={gamma:6.1f}  cond≈{cond:.3e}")

    return np.array(conds)


def plot_condition(gamma_list, conds, prefix: str = "/tmp/model1_gamma"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(gamma_list, conds, "o-", color="C2")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\kappa(A_{\rm bh} + A_{\rm IP})$ estimate")
    ax.set_title(r"IP matrix conditioning vs $\gamma$")
    ax.grid(True, which="both")
    fig.tight_layout()
    fname = f"{prefix}_cond.png"
    fig.savefig(fname, dpi=150)
    print(f"[plot] saved {fname}")
    plt.close(fig)


def main():
    gamma_list = (1.0, 3.0, 5.0, 10.0, 20.0, 50.0)

    # 1) Physical sweep on model1
    result = sweep_gamma(
        gamma_list=gamma_list,
        refine=3,
        max_steps=30,
        maxit_per_step=15,
        rtol=1e-4,
        disp_range=(0.0, 5e-3),
    )
    np.savez("/tmp/model1_gamma_sweep.npz", **result)
    print("[sweep] saved /tmp/model1_gamma_sweep.npz")
    plot(result, prefix="/tmp/model1_gamma")

    # 2) Matrix conditioning (small mesh)
    conds = compute_condition_numbers(list(gamma_list), refine=2)
    plot_condition(np.array(gamma_list), conds, prefix="/tmp/model1_gamma")
    np.savez("/tmp/model1_gamma_cond.npz", gamma=np.array(gamma_list), cond=conds)


if __name__ == "__main__":
    main()
