"""Phase 3 最后一块：surrogate-truth 效率 Θ=η_τ/err（真实算例无解析解）。

真实裂纹无解析真解 ⇒ 用**嵌套 uniform-refine 细网格全分辨解**当参照真解 u_ref，量
Θ = η_τ / ‖ε(u_coarse)-ε(u_ref)‖_{C_d}。THEORY §5：可靠性要求 Θ≥1，Θ→1 表示估计子尖锐。

干净点在**嵌套精确延拓**（避免跨网格点定位）：
  - uniform_refine 是嵌套的 ⇒ 粗 P1 空间 ⊂ 细 P1 空间。粗解 u_coarse 的节点值经
    prolongation 矩阵 IM 延拓到细网格 ⇒ **同一函数**（P1 嵌套精确），与 u_ref 同在细网格 P1。
  - d 场（P1 节点）同样经 IM 延拓到细网格（精确）⇒ 细网格逐元 g(d)。
  - 真误差在细网格积分点上算（u_coarse_on_fine.grad vs u_ref.grad），无需把粗解定位回粗单元。

取舍（论文须标注）：本 Θ 研究用 **P1 位移**（非生产 P2），因嵌套延拓对 P1 精确、对 P2 需点
定位易错；这与 T6 用一致空间做 effectivity 同类。η_τ 仍是生产 Hu–Zhang σ_h（p=3）驱动。

约定：计算走 bm；线性解 scipy；np 仅 I/O。环境 py312 + PYTHONPATH。
运行: PYTHONPATH=$PWD python fracturex/tests/aposteriori/run_theta_surrogate_model1.py
  SMOKE=1 快速冒烟（nx=8, 2 步, refine 1）。配置 env: NX DU NSTEP NREF KRES。
"""
from __future__ import annotations

import os

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearElasticIntegrator

from fracturex.cases.square_tension_precrack import SquareTensionPreCrackCase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.drivers.huzhang_phasefield_staggered import HuZhangPhaseFieldStaggeredDriver
from fracturex.adaptivity.adaptive_staggered import make_assemblers, eta_from_state
from fracturex.adaptivity.primal_elastic_solve import DegradedElasticMaterial
from fracturex.phasefield.vector_Dirichlet_bc import VectorDirichletBC
from fracturex.adaptivity.equilibrated_estimator import (
    strain_to_voigt, stress_voigt_from_strain, voigt_inner, degradation,
)


class _Mat:
    E, nu, Gc, l0, ft = 210.0, 0.3, 2.7e-3, 0.015, 3.0
    @property
    def mu(self): return self.E / (2.0 * (1.0 + self.nu))
    @property
    def lam(self): return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


def _i(n, d):
    try: return int(os.environ.get(n, d))
    except (TypeError, ValueError): return int(d)
def _f(n, d):
    try: return float(os.environ.get(n, d))
    except (TypeError, ValueError): return float(d)


def _g_cell_from_dnode(mesh, d_node, *, k_res):
    """逐元退化 g(d_T)=(1-d_T)²+k_res，d_T 取单元重心（P1 节点场重心插值=三顶点均值）。"""
    cell = mesh.entity("cell")
    d_cell = bm.mean(d_node[cell], axis=1)               # (NC,) 重心 d
    return (1.0 - d_cell) ** 2 + k_res


def _solve_primal_on_mesh(mesh, g_cell, case, *, lam, mu, load, p, q=None):
    """退化弹性连续 primal（真实分量式 BC，体力 0）——solve_primal_real 的 mesh-level 内核。

    与 primal_resolve_real.solve_primal_real 同逻辑，但参数化 mesh+g_cell+p（供细网格 truth）。
    """
    q = (p + 3) if q is None else q
    scalar = LagrangeFESpace(mesh, p=p)
    space = TensorFunctionSpace(scalar, shape=(2, -1))   # 分量优先 (GD,-1)
    material = DegradedElasticMaterial(lam, mu, g_cell)
    bform = BilinearForm(space)
    bform.add_integrator(LinearElasticIntegrator(material, q=q))
    A = bform.assembly()
    ndof = space.number_of_global_dofs()
    f = bm.zeros(ndof, dtype=bm.float64)
    uh = space.function()
    _axis = {"x": 0, "y": 1, "z": 2}
    probe = bm.array([[0.5, 0.5]], dtype=bm.float64)
    for piece in case.dirichlet_pieces(load):
        vec = bm.asarray(piece.value(probe)).reshape(-1)
        comp = _axis[piece.direction] if piece.direction is not None else None
        gd = float(vec[comp]) if comp is not None else float(vec[0])
        bc = VectorDirichletBC(space, gd, piece.threshold, direction=piece.direction)
        A, f = bc.apply(A, f)
    from fealpy.solver import spsolve
    uh[:] = spsolve(A, f, solver="scipy")
    return uh, space


def _prolong_u(IM, u_coarse_flat, nn_c, nn_f):
    """粗 P1 向量位移 (2*nn_c,) 分量优先 → 细网格 (2*nn_f,)，逐分量 IM 延拓（嵌套精确）。"""
    uc = bm.to_numpy(u_coarse_flat).reshape(2, nn_c)     # [x行, y行]
    P = IM                                                # (nn_f, nn_c)
    uf = np.stack([P @ uc[0], P @ uc[1]], axis=0)        # (2, nn_f)
    return bm.asarray(uf.reshape(-1))


def main():
    bm.set_backend("numpy")
    smoke = os.environ.get("SMOKE", "0") == "1"
    mat = _Mat()
    k_res = _f("KRES", 1e-6)
    nx = _i("NX", 8 if smoke else 24)
    du = _f("DU", 1e-3 if smoke else 2.5e-4)
    nstep = _i("NSTEP", 2 if smoke else 20)              # 跑到接近峰值的固定接受态
    nref = _i("NREF", 1 if smoke else 2)                 # uniform-refine 层数（truth 分辨）
    p_u = _i("PU", 1)                                    # P1 位移（嵌套延拓精确，见 docstring）

    # ---------- 1) 粗网格接受态：staggered 到固定 load，发育 d 场 + σ_h ----------
    case = SquareTensionPreCrackCase(_model=mat, nx=nx, ny=nx,
                                     crack_y=0.5, crack_length=0.5)
    mesh = case.make_mesh()
    discr = HuZhangDiscretization(case=case, p=3, damage_p=1,
                                  use_relaxation=True).build(mesh=mesh)
    damage = PhaseFieldDamageModel(density_type="AT2", degradation_type="quadratic",
                                   split="hybrid", eps_g=k_res)
    el_asm, ph_asm = make_assemblers(discr, case, damage)
    driver = HuZhangPhaseFieldStaggeredDriver(
        case=case, discr=discr, damage=damage,
        elastic_assembler=el_asm, phase_assembler=ph_asm,
        tol=1e-4, maxit=200, d_relaxation=1.0,
        compute_linear_residual=False, debug=False, timing=False,
        save_vtu_per_step=False, stagger_print_interval=0,
    )
    driver.initialize()
    load = 0.0
    for s in range(nstep):
        load = float((s + 1) * du)
        info = driver.solve_one_step(step=s, load=load)
    nc_c = int(discr.mesh.number_of_cells())
    print(f"[theta] coarse accept state: nx={nx} load={load:.3e} "
          f"max_d={float(info.max_d):.4f} R={abs(float(info.meta.get('R',0))):.4e} nc={nc_c}",
          flush=True)

    # ---------- 2) 粗网格 η_τ：连续 primal（生产真实 BC）喂估计子（严格，常数=1）----------
    g_cell_c = _g_cell_from_dnode(discr.mesh, discr.state.d[:], k_res=k_res)
    u_coarse, space_c = _solve_primal_on_mesh(
        discr.mesh, g_cell_c, case, lam=mat.lam, mu=mat.mu, load=load, p=p_u)
    eta = float(eta_from_state(discr, lam=mat.lam, mu=mat.mu, k_res=k_res,
                               u_override=u_coarse)["eta"])
    print(f"[theta] eta_tau (coarse, strict) = {eta:.6e}", flush=True)

    # ---------- 3) 细网格 truth：uniform-refine + 嵌套精确延拓 d/u ----------
    mesh_c = discr.mesh
    nn_c = int(mesh_c.number_of_nodes())
    d_node_c = discr.state.d[:]
    mesh_f = type(mesh_c)(bm.copy(mesh_c.entity("node")), bm.copy(mesh_c.entity("cell")))
    IMs = mesh_f.uniform_refine(n=nref, returnim=True)   # finest→coarsest 顺序
    # 组合多层 prolongation：P_total = P_1 @ P_2 @ ... （node 值 finest = P_total @ coarse）
    P_total = IMs[0]
    for k in range(1, len(IMs)):
        P_total = P_total @ IMs[k]
    P_total = P_total.toarray() if hasattr(P_total, "toarray") else np.asarray(bm.to_numpy(P_total))
    nn_f = int(mesh_f.number_of_nodes())
    d_node_f = bm.asarray(P_total @ bm.to_numpy(d_node_c))     # d 延拓（P1 嵌套精确）
    g_cell_f = _g_cell_from_dnode(mesh_f, d_node_f, k_res=k_res)
    u_ref, space_f = _solve_primal_on_mesh(
        mesh_f, g_cell_f, case, lam=mat.lam, mu=mat.mu, load=load, p=p_u)
    print(f"[theta] fine truth: nref={nref} nc_fine={int(mesh_f.number_of_cells())} "
          f"nn_fine={nn_f}", flush=True)

    # ---------- 4) 真能量误差（细网格积分点）：u_coarse 延拓 vs u_ref ----------
    u_coarse_on_fine_flat = _prolong_u(P_total, u_coarse[:], nn_c, nn_f)
    u_cf = space_f.function(); u_cf[:] = u_coarse_on_fine_flat
    q = p_u + 3
    qf = mesh_f.quadrature_formula(q)
    bcs, ws = qf.get_quadrature_points_and_weights()
    cm = mesh_f.entity_measure("cell")
    grad_cf = u_cf.grad_value(bcs)        # (NC,NQ,2,2)
    grad_rf = u_ref.grad_value(bcs)
    # d 在细网格积分点（P1 标量插值）
    dscalar_f = LagrangeFESpace(mesh_f, p=1).function(); dscalar_f[:] = d_node_f
    d_qp = dscalar_f(bcs)                  # (NC,NQ)
    g = degradation(d_qp, k_res)
    de = strain_to_voigt(grad_cf) - strain_to_voigt(grad_rf)
    Cde = stress_voigt_from_strain(de, mat.lam, mat.mu)
    integrand = g * voigt_inner(de, Cde)
    val = bm.sum(cm * bm.einsum('q,cq->c', ws, integrand))
    err = float(bm.sqrt(bm.where(val < 0, bm.zeros_like(val), val)))

    theta = eta / err if err > 1e-300 else float("inf")
    print(f"[theta] energy err ‖ε(u_c)-ε(u_ref)‖_Cd = {err:.6e}", flush=True)
    print(f"[theta] ===> Theta = eta_tau / err = {theta:.4f}  (reliability needs >=1)",
          flush=True)
    print(f"[theta] DONE nx={nx} load={load:.3e} nref={nref} p_u={p_u} "
          f"eta={eta:.4e} err={err:.4e} Theta={theta:.4f}", flush=True)


if __name__ == "__main__":
    main()
