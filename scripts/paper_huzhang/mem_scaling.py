#!/usr/bin/env python3
"""B2 内存扩展：peak RSS vs σ-DOF（matrix-free vs direct 分解）。

每档网格独立进程跑一次（ru_maxrss 取干净峰值）：
  mode=mf      : matrix-free(recompute) 装配 + 几次 matvec —— 不存 M2/因子，近线性内存；
  mode=direct  : 精简装配(关 Phi 缓存) + pardiso 分解求解 —— fill-in 超线性，大 N OOM。
论点：mf 近线性、direct 超线性且在 h4/h5 撑爆 → "迭代法在大规模/受限内存下唯一可行"。

用法: mem_scaling.py <mf|direct> <hmin>
输出: 一行 `mode,hmin,sigma,peak_rss_mb,t_s,note` 到 stdout。
"""
from __future__ import annotations
import os, sys, time, resource
import numpy as np
from fealpy.backend import backend_manager as bm
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.cases.model0_circular_notch import Model0CircularNotchCase
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization


class Mat:
    E = 200.0; nu = 0.2; Gc = 1.0; l0 = 0.02
    @property
    def mu(self): return self.E / (2 * (1 + self.nu))
    @property
    def lam(self): return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    mode, hmin = sys.argv[1], float(sys.argv[2])
    dmg = PhaseFieldDamageModel(density_type="AT2", degradation_type="quadratic",
                                split="hybrid", eps_g=1e-6, debug=False)
    case = Model0CircularNotchCase(_model=Mat(), hmin=hmin)
    mesh = case.make_mesh()
    discr = HuZhangDiscretization(case, p=3, use_relaxation=True).build(mesh=mesh)
    dmg.on_build(discr, discr.state, case)
    # representative damaged state (d=0.9 half) — conditioning ~ production
    node = bm.asarray(mesh.node); x = bm.to_numpy(node[:, 0])
    d = np.zeros_like(x); d[x >= 0.5] = 0.9
    discr.state.d[:] = bm.asarray(d)
    m = int(discr.gdof_sigma)
    note = ""
    t0 = time.perf_counter()
    try:
        if mode == "mf":
            asm = HuZhangElasticAssembler(discr, case, dmg, formulation="standard",
                                          assembly_parallel=False)
            asm._matfree = True; asm._matfree_recompute = True; asm._matfree_chunk = 2048
            asm.begin_load_step(0.09); s = asm.assemble(0.09)
            n = s.A.shape[0]; rng = np.random.default_rng(0)
            for _ in range(3):
                s.A.matvec(rng.standard_normal(n))
        elif mode == "direct":
            os.environ["FRACTUREX_M_KERNEL_CACHE"] = "0"  # lean assembly (no Phi cache)
            asm = HuZhangElasticAssembler(discr, case, dmg, formulation="standard",
                                          assembly_parallel=False)
            asm.begin_load_step(0.09); s = asm.assemble(0.09)
            A_ = s.A.tocsr() if hasattr(s.A, "tocsr") else s.A
            import pypardiso
            x_sol = pypardiso.spsolve(A_, np.asarray(s.F, float).reshape(-1))
            note = f"solvenorm={float(np.linalg.norm(x_sol)):.3e}"
        else:
            raise SystemExit(f"unknown mode {mode}")
    except MemoryError:
        note = "OOM"
    except Exception as e:
        note = f"err:{type(e).__name__}:{str(e)[:40]}"
    t = time.perf_counter() - t0
    print(f"{mode},{hmin},{m},{peak_mb():.0f},{t:.1f},{note}")


if __name__ == "__main__":
    main()
