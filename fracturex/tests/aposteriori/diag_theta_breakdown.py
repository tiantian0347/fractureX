"""诊断 Θ=η_τ/err<1（2026-06-18 surrogate run，Θ=0.57）的根因。

粗态只算一次，一并报三项嫌疑（见 RESULTS §Phase3 待查）：
  (A) 真解收敛性  : err 随 nref∈{1,2,3} 是否稳（不稳 ⇒ truth 欠分辨，参照不可信）。
  (B) 牵引泄漏    : b_j=∫σ_h:ε(φ_j)（连续 P1 测试）。f=0 时 Hu–Zhang div σ_h=0 精确 ⇒
                    b_j=∮_∂(σ_h·n)φ_j。Γ_N（左/右边 + 顶边 x 分量，t_N=0）上的 b 应≈0；
                    若 ‖b|Γ_N‖ 与 η 同量级 ⇒ σ_h∉Σ_f，破坏可靠性常数=1（T4 留的 osc(t_N) 坑）。
  (C) g 口径      : η_τ 现用逐元常数 g(重心)；σ_h 用 p=3 逐积分点 g。改逐积分点 g 重算 η 看是否抬。

约定：计算走 bm，scipy 解线性，np 仅 I/O。运行:
  PYTHONPATH=$PWD python fracturex/tests/aposteriori/diag_theta_breakdown.py
env: NX(24) DU(2.5e-4) NSTEP(20) KRES(1e-6) NREFS("1,2,3")
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
from fracturex.adaptivity.primal_resolve_real import apply_dirichlet_pieces_lifted
from fracturex.adaptivity.equilibrated_estimator import (
    strain_to_voigt, stress_voigt_from_strain, voigt_inner, degradation,
    compliance_apply_voigt,
)


class _Mat:
    E, nu, Gc, l0, ft = 210.0, 0.3, 2.7e-3, 0.015, 3.0
    @property
    def mu(self): return self.E / (2.0 * (1.0 + self.nu))
    @property
    def lam(self): return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


def _f(n, d):
    try: return float(os.environ.get(n, d))
    except (TypeError, ValueError): return float(d)
def _i(n, d):
    try: return int(os.environ.get(n, d))
    except (TypeError, ValueError): return int(d)


def _g_cell_from_dnode(mesh, d_node, *, k_res):
    cell = mesh.entity("cell")
    d_cell = bm.mean(d_node[cell], axis=1)
    return (1.0 - d_cell) ** 2 + k_res


def _solve_primal_on_mesh(mesh, g_cell, case, *, lam, mu, load, p, q=None):
    q = (p + 3) if q is None else q
    scalar = LagrangeFESpace(mesh, p=p)
    space = TensorFunctionSpace(scalar, shape=(2, -1))
    material = DegradedElasticMaterial(lam, mu, g_cell)
    bform = BilinearForm(space)
    bform.add_integrator(LinearElasticIntegrator(material, q=q))
    A = bform.assembly()
    ndof = space.number_of_global_dofs()
    f = bm.zeros(ndof, dtype=bm.float64)
    uh, _, _ = apply_dirichlet_pieces_lifted(A, f, space, case, load)  # 带非齐次提升（2026-06-21 修复）
    return uh, space


def _err_vs_truth(discr, case, mat, u_coarse, *, k_res, nref, p_u, coeff="fine"):
    """nref 层 uniform-refine 细 primal 当 truth；嵌套精确延拓 d/u；细积分点算能量误差。

    coeff: 'fine'    = 逐细元重心 g（旧口径，系数随网格再分辨 ⇒ 非嵌套 ⇒ err 发散）。
           'inherit' = 细元继承父粗元的分片常数 g（parent(k)=k%NC0，uniform_refine 交错序）。
                       粗/细同一固定系数问题、位移空间嵌套 ⇒ err 应单调收敛。
    """
    mesh_c = discr.mesh
    NC0 = int(mesh_c.number_of_cells())
    nn_c = int(mesh_c.number_of_nodes())
    d_node_c = discr.state.d[:]
    g_cell_c = _g_cell_from_dnode(mesh_c, d_node_c, k_res=k_res)
    mesh_f = type(mesh_c)(bm.copy(mesh_c.entity("node")), bm.copy(mesh_c.entity("cell")))
    IMs = mesh_f.uniform_refine(n=nref, returnim=True)
    P = IMs[0]
    for k in range(1, len(IMs)):
        P = P @ IMs[k]
    P = P.toarray() if hasattr(P, "toarray") else np.asarray(bm.to_numpy(P))
    nn_f = int(mesh_f.number_of_nodes())
    d_node_f = bm.asarray(P @ bm.to_numpy(d_node_c))
    NC_f = int(mesh_f.number_of_cells())
    if coeff == "inherit":
        parent = bm.arange(NC_f) % NC0                    # uniform_refine 交错序父映射
        g_cell_f = g_cell_c[parent]                       # 固定分片常数系数（嵌套）
    else:
        g_cell_f = _g_cell_from_dnode(mesh_f, d_node_f, k_res=k_res)
    u_ref, space_f = _solve_primal_on_mesh(mesh_f, g_cell_f, case,
                                           lam=mat.lam, mu=mat.mu, load=case._load, p=p_u)
    uc = bm.to_numpy(u_coarse[:]).reshape(2, nn_c)
    u_cf = space_f.function()
    u_cf[:] = bm.asarray(np.stack([P @ uc[0], P @ uc[1]], axis=0).reshape(-1))
    q = p_u + 3
    bcs, ws = mesh_f.quadrature_formula(q).get_quadrature_points_and_weights()
    cm = mesh_f.entity_measure("cell")
    de = strain_to_voigt(u_cf.grad_value(bcs)) - strain_to_voigt(u_ref.grad_value(bcs))
    if coeff == "inherit":
        g = g_cell_f[:, None] * bm.ones((1, bcs.shape[0]), dtype=bm.float64)  # 分片常数能量范数
    else:
        dsc = LagrangeFESpace(mesh_f, p=1).function(); dsc[:] = d_node_f
        g = degradation(dsc(bcs), k_res)
    val = bm.sum(cm * bm.einsum('q,cq->c', ws, g * voigt_inner(de, stress_voigt_from_strain(de, mat.lam, mat.mu))))
    return float(bm.sqrt(bm.where(val < 0, bm.zeros_like(val), val))), int(mesh_f.number_of_cells())


def _traction_leak(discr, case, *, q=8):
    """b_j=∫σ_h:ε(φ_j)（连续 P1），按 Γ_D/Γ_N/interior 分组报范数。"""
    mesh = discr.mesh
    scalar = LagrangeFESpace(mesh, p=1)
    bcs, ws = mesh.quadrature_formula(q).get_quadrature_points_and_weights()
    cm = mesh.entity_measure("cell")
    gphi = scalar.grad_basis(bcs)                    # (NC,NQ,ldof,2)
    c2d = scalar.cell_to_dof()                       # (NC,ldof)
    sig = discr.state.sigma(bcs)                     # (NC,NQ,3) Voigt(xx,xy,yy)
    sxx, sxy, syy = sig[..., 0], sig[..., 1], sig[..., 2]
    # σ:ε(φ) for x-comp basis = sxx ∂xφ + sxy ∂yφ ; y-comp = syy ∂yφ + sxy ∂xφ
    gx, gy = gphi[..., 0], gphi[..., 1]              # (NC,NQ,ldof)
    wq = bm.einsum('q,c->cq', ws, cm)               # (NC,NQ)
    bx_loc = bm.einsum('cq,cql->cl', wq, sxx[..., None] * gx + sxy[..., None] * gy)
    by_loc = bm.einsum('cq,cql->cl', wq, syy[..., None] * gy + sxy[..., None] * gx)
    nn = int(mesh.number_of_nodes())
    bx = bm.zeros(nn, dtype=bm.float64); by = bm.zeros(nn, dtype=bm.float64)
    bx = bm.index_add(bx, c2d.reshape(-1), bx_loc.reshape(-1))
    by = bm.index_add(by, c2d.reshape(-1), by_loc.reshape(-1))
    node = mesh.entity("node")
    x, y = node[:, 0], node[:, 1]
    x0, x1, y0, y1 = case.box; tol = 1e-9
    on_bot = bm.abs(y - y0) < tol; on_top = bm.abs(y - y1) < tol
    on_lr = (bm.abs(x - x0) < tol) | (bm.abs(x - x1) < tol)
    # 约束: y on bot|top ; x on bot only. Γ_N = 自由方向上的边界 dof。
    xfree_bd = (on_lr | on_top) & (~on_bot)          # 左右(x) + 顶(x); 排底
    yfree_bd = on_lr & (~on_bot) & (~on_top)         # 左右(y); 排底顶
    is_bd = on_bot | on_top | on_lr
    interior = ~is_bd
    xD = on_bot; yD = on_bot | on_top                # Dirichlet 约束 dof（反力所在）
    def nrm(mask_x, mask_y):
        v = bm.concatenate([bx[mask_x], by[mask_y]]) if (int(bm.sum(mask_x)) + int(bm.sum(mask_y))) else bm.zeros(1)
        return float(bm.sqrt(bm.sum(v * v)))
    return {
        "leak_GammaN": nrm(xfree_bd, yfree_bd),
        "reaction_GammaD": nrm(xD, yD),
        "interior": nrm(interior, interior),
        "total": float(bm.sqrt(bm.sum(bx * bx) + bm.sum(by * by))),
        "n_xfree": int(bm.sum(xfree_bd)), "n_yfree": int(bm.sum(yfree_bd)),
    }


def _eta_quadrature_g(discr, mat, u_coarse, *, k_res, q=8, percell=False):
    """用逐积分点 g(d_qp)（与 σ_h 一致）重算 η，对比 const-g。
    percell=True 时用逐元重心 g（与 inherit 分片常数系数口径匹配）。"""
    mesh = discr.mesh
    bcs, ws = mesh.quadrature_formula(q).get_quadrature_points_and_weights()
    cm = mesh.entity_measure("cell")
    grad = u_coarse.grad_value(bcs)
    sig = discr.state.sigma(bcs)
    dqp = discr.state.d(bcs)
    if dqp.ndim == 3: dqp = dqp[..., 0]
    if percell:
        d_node = discr.state.d[:]; cell = mesh.entity("cell")
        d_cell = bm.mean(d_node[cell], axis=1)
        g = ((1.0 - d_cell) ** 2 + k_res)[:, None] * bm.ones((1, bcs.shape[0]), dtype=bm.float64)
    else:
        g = degradation(dqp, k_res)
    eps = strain_to_voigt(grad)
    r = g[..., None] * stress_voigt_from_strain(eps, mat.lam, mat.mu) - sig
    Cinv_r = compliance_apply_voigt(r, mat.lam, mat.mu)
    integ = (1.0 / g) * voigt_inner(r, Cinv_r)
    val = bm.sum(cm * bm.einsum('q,cq->c', ws, integ))
    return float(bm.sqrt(bm.where(val < 0, bm.zeros_like(val), val)))


def main():
    bm.set_backend("numpy")
    mat = _Mat()
    k_res = _f("KRES", 1e-6)
    nx = _i("NX", 24); du = _f("DU", 2.5e-4); nstep = _i("NSTEP", 20); p_u = 1
    nrefs = [int(s) for s in os.environ.get("NREFS", "1,2,3").split(",")]

    case = SquareTensionPreCrackCase(_model=mat, nx=nx, ny=nx, crack_y=0.5, crack_length=0.5)
    mesh = case.make_mesh()
    discr = HuZhangDiscretization(case=case, p=3, damage_p=1, use_relaxation=True).build(mesh=mesh)
    damage = PhaseFieldDamageModel(density_type="AT2", degradation_type="quadratic", split="hybrid", eps_g=k_res)
    el_asm, ph_asm = make_assemblers(discr, case, damage)
    driver = HuZhangPhaseFieldStaggeredDriver(
        case=case, discr=discr, damage=damage, elastic_assembler=el_asm, phase_assembler=ph_asm,
        tol=1e-4, maxit=200, d_relaxation=1.0, compute_linear_residual=False,
        debug=False, timing=False, save_vtu_per_step=False, stagger_print_interval=0)
    driver.initialize()
    load = 0.0
    for s in range(nstep):
        load = float((s + 1) * du)
        info = driver.solve_one_step(step=s, load=load)
    case._load = load
    print(f"[diag] coarse accept: nx={nx} load={load:.3e} max_d={float(info.max_d):.4f} "
          f"nc={int(discr.mesh.number_of_cells())}", flush=True)

    g_cell_c = _g_cell_from_dnode(discr.mesh, discr.state.d[:], k_res=k_res)
    u_coarse, _ = _solve_primal_on_mesh(discr.mesh, g_cell_c, case, lam=mat.lam, mu=mat.mu, load=load, p=p_u)
    eta_const = float(eta_from_state(discr, lam=mat.lam, mu=mat.mu, k_res=k_res, u_override=u_coarse)["eta"])
    eta_quad = _eta_quadrature_g(discr, mat, u_coarse, k_res=k_res)
    print(f"[diag] eta_tau: const-g={eta_const:.6e}  quad-g={eta_quad:.6e}  "
          f"ratio={eta_quad/eta_const:.3f}", flush=True)

    print("[diag] --- (B) traction leak: b_j = int sigma_h:eps(phi_j) ---", flush=True)
    lk = _traction_leak(discr, case)
    print(f"[diag]   ||b|Gamma_N(free)||  = {lk['leak_GammaN']:.6e}   "
          f"(n_xfree={lk['n_xfree']} n_yfree={lk['n_yfree']})", flush=True)
    print(f"[diag]   ||b|Gamma_D(react)|| = {lk['reaction_GammaD']:.6e}", flush=True)
    print(f"[diag]   ||b|interior||       = {lk['interior']:.6e}", flush=True)
    print(f"[diag]   leak/eta_const = {lk['leak_GammaN']/eta_const:.4f}   "
          f"interior/eta = {lk['interior']/eta_const:.4f}", flush=True)

    print("[diag] --- (A) truth convergence: err vs nref (fine-coeff vs inherit-coeff) ---", flush=True)
    # per-cell（重心 g）η，与 inherit 分片常数系数口径匹配
    eta_pc = _eta_quadrature_g(discr, mat, u_coarse, k_res=k_res, percell=True)
    print(f"[diag]   eta_percell(barycentric g) = {eta_pc:.6e}", flush=True)
    for mode, eref in (("fine", eta_quad), ("inherit", eta_pc)):
        for nr in nrefs:
            err, ncf = _err_vs_truth(discr, case, mat, u_coarse, k_res=k_res,
                                     nref=nr, p_u=p_u, coeff=mode)
            print(f"[diag]   [{mode:7s}] nref={nr} nc_fine={ncf:>7d}  err={err:.6e}  "
                  f"Theta={eref/err:.4f}", flush=True)
    print("[diag] DONE", flush=True)


if __name__ == "__main__":
    main()
