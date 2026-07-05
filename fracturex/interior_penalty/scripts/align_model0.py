"""对齐脚本：model0 完整 30 步跑 + 与老代码结果 (p2_h1) 对比。

用法::

    source ~/venv_fealpy3/bin/activate
    python fracturex/interior_penalty/scripts/align_model0.py

产出:
    /tmp/model0_align.png                 force-disp 对比图
    /tmp/model0_mine_30steps.npz          我们跑出的 (disp, force, stored, dissipated)
"""
from __future__ import annotations

import time
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm

bm.set_backend("numpy")

from fracturex.interior_penalty.cases.model0_circular_hole import (
    Model0CircularHoleCase,
)


REF_TXT = Path(
    "/Users/tian00/repository/Tian/thesis/ip_fracture/ipfem_fp_results/"
    "ipfem_fp_model0/p2_h1_results_model0_ada.txt"
)


def _parse_reference(path: Path) -> np.ndarray:
    text = path.read_text()
    # Extract the [ ... ] block after "force:"
    m = re.search(r"force:\s*\[([^\]]*)\]", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"cannot parse {path}")
    tokens = re.findall(r"[-+]?\d*\.\d+(?:e[-+]?\d+)?|[-+]?\d+", m.group(1))
    return np.array([float(t) for t in tokens], dtype=np.float64)


def main(hmin: float = 0.05, distmesh_maxit: int = 100, maxit_per_step: int = 30,
         rtol: float = 1e-5):
    print(f"[align_model0] hmin={hmin} distmesh_maxit={distmesh_maxit}")
    case = Model0CircularHoleCase(hmin=hmin, distmesh_maxit=distmesh_maxit)
    solver = case.build_solver()
    NN = solver.mesh.number_of_nodes()
    NC = solver.mesh.number_of_cells()
    print(f"[align_model0] mine mesh NN={NN} NC={NC}")

    t0 = time.time()
    out = case.run(maxit_per_step=maxit_per_step, rtol=rtol, verbose=False)
    elapsed = time.time() - t0

    disp = out["disp"]
    force = out["force"]
    stored = out["stored_energy"]
    dissipated = out["dissipated_energy"]

    print(f"[align_model0] elapsed {elapsed:.1f}s over {len(disp)} steps")
    print(f"[align_model0] peak force = {force.max():.4f} at step "
          f"{int(np.argmax(force))} disp={disp[int(np.argmax(force))]:.4e}")

    ref_force = _parse_reference(REF_TXT)
    print(f"[align_model0] ref peak force = {ref_force.max():.4f} at step "
          f"{int(np.argmax(ref_force))}")

    n = min(len(disp), len(ref_force))
    disp_s = disp[:n]

    # ---- plot ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(disp_s, ref_force[:n], "o-", color="k", label=f"ref (p2_h1)")
    ax1.plot(disp_s, force[:n], "s--", color="C1",
             label=f"IPFEMPhaseFieldSolver (NN={NN}, NC={NC})")
    ax1.set_xlabel("disp")
    ax1.set_ylabel("force")
    ax1.set_title("model0: force-disp")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(disp_s, stored[:n], "-", label="stored (ours)")
    ax2.plot(disp_s, dissipated[:n], "-", label="dissipated (ours)")
    ax2.plot(disp_s, stored[:n] + dissipated[:n], "-", label="total (ours)")
    ax2.set_xlabel("disp")
    ax2.set_ylabel("energy")
    ax2.set_title("model0: energy")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()
    out_png = "/tmp/model0_align.png"
    fig.savefig(out_png, dpi=150)
    print(f"[align_model0] saved plot to {out_png}")

    np.savez(
        "/tmp/model0_mine_30steps.npz",
        disp=disp, force=force,
        stored_energy=stored, dissipated_energy=dissipated,
        ref_force=ref_force,
    )
    print("[align_model0] saved data to /tmp/model0_mine_30steps.npz")


if __name__ == "__main__":
    main()
