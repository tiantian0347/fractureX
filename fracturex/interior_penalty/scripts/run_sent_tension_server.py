"""SENT tension benchmark server entry (Miehe 2010).

服务器版：完整 30-step 载荷序列 + vtu 每 20 步一次 + 全过程数据保存。
本地一般不跑（时间较长），本地冒烟见
`fracturex/interior_penalty/tests/test_ipfem_phasefield_smoke.py`。

用法::

    source ~/venv_fealpy3/bin/activate
    nohup python fracturex/interior_penalty/scripts/run_sent_tension_server.py \
        --refine 6 --outdir /path/to/output \
        > sent_tension.log 2>&1 &

    # 或用 tmux
    tmux new -s sent
    source ~/venv_fealpy3/bin/activate
    python fracturex/interior_penalty/scripts/run_sent_tension_server.py --refine 6

参数（默认对齐 Miehe 2010 §5.1）：
    E=210, ν=0.3, Gc=2.7e-3, ℓ_0=0.015
    load: 0..5e-3 (500 步) ∪ 5e-3..6e-3 (1000 步) = 1500 步
    p_disp=1, p_phase=2, gamma=5

产出:
    <outdir>/sent_tension_history.npz       每步 disp / force / stored / diss
    <outdir>/sent_tension_stepXXXXXX.vtu    vtu 快照
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from fealpy.backend import backend_manager as bm
bm.set_backend("numpy")

from fracturex.interior_penalty.cases import SentTensionMieheCase


def main():
    ap = argparse.ArgumentParser(description="SENT tension server run")
    ap.add_argument("--refine", type=int, default=6,
                    help="uniform_refine 层数; 6 → NC≈32k, 7 → NC≈130k")
    ap.add_argument("--outdir", type=str, default="/tmp/sent_tension_server")
    ap.add_argument("--maxit", type=int, default=100)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--vtk-every", type=int, default=20,
                    help="每多少载荷步存一次 vtu (0=不存)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="调试用: 只跑前 N 步 (默认跑全 ~1500 步)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[SENT] outdir={outdir}")
    print(f"[SENT] refine={args.refine}, maxit={args.maxit}, rtol={args.rtol}")

    case = SentTensionMieheCase(refine=args.refine)
    print(f"[SENT] material: E={case.E}, nu={case.nu}, "
          f"Gc={case.Gc}, l0={case.l0}")

    mesh = case.build_mesh()
    print(f"[SENT] mesh: NN={mesh.number_of_nodes()}, "
          f"NC={mesh.number_of_cells()}")

    t0 = time.time()
    out = case.run(
        max_steps=args.max_steps,
        maxit_per_step=args.maxit,
        rtol=args.rtol,
        verbose=True,
        vtk_every=args.vtk_every if args.vtk_every > 0 else None,
        vtk_prefix=str(outdir / "sent_tension") if args.vtk_every > 0 else None,
    )
    elapsed = time.time() - t0

    print(f"[SENT] elapsed {elapsed:.1f}s")
    peak_i = int(np.argmax(np.abs(out["force"])))
    print(f"[SENT] peak force = {out['force'][peak_i]:.4f} "
          f"@ disp={out['disp'][peak_i]:.4e} (step {peak_i})")

    np.savez(
        outdir / "sent_tension_history.npz",
        disp=out["disp"],
        force=out["force"],
        stored_energy=out["stored_energy"],
        dissipated_energy=out["dissipated_energy"],
        refine=args.refine,
        maxit=args.maxit,
        rtol=args.rtol,
        elapsed_s=elapsed,
    )
    print(f"[SENT] saved history {outdir/'sent_tension_history.npz'}")


if __name__ == "__main__":
    main()
