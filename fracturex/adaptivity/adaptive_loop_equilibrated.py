"""M2: 平衡型估计子驱动的自适应加密循环。

理论见 docs/adaptive/THEORY_equilibrated_aposteriori.md，设计见 docs/adaptive/DESIGN_program_and_tests.md。
循环（单载荷步，冻结 d 场；多载荷步演化是 M3）：
  while 未达停机:
    u_h ← 标准 FEM 退化解（solve_primal_degraded）
    σ_h ← Hu–Zhang 混合退化解（solve_degraded_huzhang，平衡）
    η_T ← equilibrated_indicator（逐元指示子）
    marked ← Dörfler bulk marking（Mesh.mark, method='L2', θ）
    bisect 加密；d/H 转移（u/σ 下轮重解，见 test_refine_interp 结论）

输出每轮 (DOF, η, 真误差, Θ)，供 T8 对比自适应 vs 均匀。
注：σ/u 是瞬时量每步重解；本循环冻结解析 d(x)，故 d 用解析重求值（无需转移）。
运行 p=3（Hu–Zhang 要求）。计算走 bm；线性求解经各 solver 内部 scipy。
"""
from __future__ import annotations

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.mesh.mesh_base import Mesh

from fracturex.adaptivity.degraded_mms import DegradedElasticMMS
from fracturex.adaptivity.degraded_huzhang_solve import solve_degraded_huzhang
from fracturex.adaptivity.primal_elastic_solve import solve_primal_degraded
from fracturex.adaptivity.equilibrated_estimator import (
    equilibrated_indicator, energy_error_Cd, effectivity_index,
)


def _all_dirichlet(bc):
    tol = 1e-12
    return ((bm.abs(bc[:, 0] - 0) < tol) | (bm.abs(bc[:, 0] - 1) < tol)
            | (bm.abs(bc[:, 1] - 0) < tol) | (bm.abs(bc[:, 1] - 1) < tol))


def solve_and_estimate(mesh, p, pde, *, lam, mu, k_res, q=12):
    """单网格：解 u_h+σ_h，算 η_T / η / 真误差 / Θ。

    输入:
      mesh,p,pde,lam,mu,k_res,q
    输出 dict:
      eta_T:(NC,) 逐元指示子, eta:标量, err:真能量误差, theta:Θ, dof:总自由度
    """
    prim = solve_primal_degraded(mesh, p, pde, lam=lam, mu=mu, k_res=k_res,
                                 dirichlet=pde.displacement)
    hz = solve_degraded_huzhang(mesh, p, pde, lam=lam, mu=mu, k_res=k_res,
                                isD_bd=_all_dirichlet)
    qf = mesh.quadrature_formula(q, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    cm = mesh.entity_measure('cell')
    pts = mesh.bc_to_point(bcs)

    grad_uh = prim["uh"].grad_value(bcs)
    sigmah_qp = hz["sigmah"](bcs)
    d_qp = pde.damage(pts)
    grad_u_exact = pde.grad_u(pts)

    ind = equilibrated_indicator(mesh, grad_uh, sigmah_qp, d_qp,
                                 lam=lam, mu=mu, k_res=k_res,
                                 weights=ws, cellmeasure=cm)
    err = energy_error_Cd(grad_uh, grad_u_exact, d_qp,
                          lam=lam, mu=mu, k_res=k_res,
                          weights=ws, cellmeasure=cm)
    dof = (prim["space"].number_of_global_dofs()
           + hz["space_sigma"].number_of_global_dofs())
    return {"eta_T": ind["eta_T"], "eta": ind["eta"], "err": err,
            "theta": effectivity_index(ind["eta"], err), "dof": dof}


def adaptive_loop(pde, *, lam, mu, k_res, p=3, N0=8, theta=0.4,
                  max_iter=6, verbose=True):
    """平衡型估计子驱动的自适应加密循环。

    输入:
      pde     : DegradedElasticMMS（解析 d(x) 冻结）
      lam,mu,k_res : 材料/退化
      p       : 次数（Hu–Zhang 要求 p≥3）
      N0      : 初始 N×N 网格
      theta   : Dörfler bulk 比例
      max_iter: 最大加密轮数
    输出:
      history: list of dict（每轮 dof/eta/err/theta），含初始网格
    """
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=N0, ny=N0)
    history = []
    for it in range(max_iter):
        res = solve_and_estimate(mesh, p, pde, lam=lam, mu=mu, k_res=k_res)
        history.append({"iter": it, "dof": res["dof"], "eta": res["eta"],
                        "err": res["err"], "theta": res["theta"],
                        "NC": mesh.number_of_cells()})
        if verbose:
            print(f"[M2] it={it} NC={mesh.number_of_cells()} dof={res['dof']} "
                  f"η={res['eta']:.4e} err={res['err']:.4e} Θ={res['theta']:.4f}")
        if it == max_iter - 1:
            break
        # Dörfler 标记 + bisect 加密
        marked = Mesh.mark(eta=res["eta_T"], theta=theta, method='L2')
        opt = mesh.bisect_options(disp=False)
        mesh.bisect(marked, options=opt)
    return history


def uniform_refine_loop(pde, *, lam, mu, k_res, p=3, N_list=(8, 12, 16, 24, 32),
                        verbose=True):
    """均匀网格对照：各 N 独立解，量 dof/η/err。"""
    history = []
    for N in N_list:
        mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=N, ny=N)
        res = solve_and_estimate(mesh, p, pde, lam=lam, mu=mu, k_res=k_res)
        history.append({"N": N, "dof": res["dof"], "eta": res["eta"],
                        "err": res["err"], "theta": res["theta"],
                        "NC": mesh.number_of_cells()})
        if verbose:
            print(f"[uniform] N={N} NC={mesh.number_of_cells()} dof={res['dof']} "
                  f"η={res['eta']:.4e} err={res['err']:.4e} Θ={res['theta']:.4f}")
    return history
