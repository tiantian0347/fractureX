"""Phase 3 最后一块：surrogate-truth 效率 Θ=η_τ/err（真实算例无解析解）。

真实裂纹无解析真解 ⇒ 用**嵌套 uniform-refine 细网格全分辨解**当参照真解 u_ref，量
Θ = η_τ / ‖ε(u_coarse)-ε(u_ref)‖_{C_d}。THEORY §5：可靠性要求 Θ≥1，Θ→1 表示估计子尖锐。

干净点在**嵌套精确延拓**（避免跨网格点定位）：
  - uniform_refine 是嵌套的 ⇒ 粗 P1 空间 ⊂ 细 P1 空间。粗解 u_coarse 的节点值经
    prolongation 矩阵 IM 延拓到细网格 ⇒ **同一函数**（P1 嵌套精确），与 u_ref 同在细网格 P1。
  - d 场（P1 节点）同样经 IM 延拓到细网格（精确）；所有层均在积分点直接
    评价同一个连续函数 g_h(x)=(1-d_h(x))²+k_res，不做逐元平均或重新投影。
  - 真误差在细网格积分点上算（u_coarse_on_fine.grad vs u_ref.grad），无需把粗解定位回粗单元。

取舍（论文须标注）：本 Θ 研究用 **P1 位移**（非论文默认 P3），因当前嵌套延拓实现对 P1
精确；这与 T6 用一致空间做 effectivity 同类。η_τ 仍使用 Hu–Zhang σ_h（p=3）。

约定：计算走 bm；线性解 scipy；np 仅 I/O。环境 py312 + PYTHONPATH。
运行: PYTHONPATH=$PWD python fracturex/tests/aposteriori/run_theta_surrogate_model1.py
  SMOKE=1 快速冒烟（nx=8, 2 步, refine 1）。
  配置 env: NX DU NSTEP NREFS("1,2,3,4") KRES PU Q ETA_QS("10,14,18,22")。
"""
from __future__ import annotations

import os

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearElasticIntegrator
from fealpy.material.elastic_material import LinearElasticMaterial

from fracturex.cases.square_tension_precrack import SquareTensionPreCrackCase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.drivers.huzhang_phasefield_staggered import HuZhangPhaseFieldStaggeredDriver
from fracturex.adaptivity.adaptive_staggered import make_assemblers, eta_from_state
from fracturex.adaptivity.primal_resolve_real import apply_dirichlet_pieces_lifted
from fracturex.adaptivity.equilibrated_estimator import (
    strain_to_voigt, stress_voigt_from_strain, voigt_inner, degradation,
    equilibrated_indicator,
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


class _NodalDegradedElasticMaterial(LinearElasticMaterial):
    """Variable material g(d_h) C evaluated directly at quadrature points."""

    def __init__(self, lam, mu, d_field, *, k_res):
        super().__init__(name="degraded_nodal", lame_lambda=lam,
                         shear_modulus=mu, hypo="plane_strain")
        self._d_field = d_field
        self._k_res = k_res

    def elastic_matrix(self, bcs=None):
        d_qp = self._d_field(bcs)
        if d_qp.ndim == 3:
            d_qp = d_qp[..., 0]
        g_qp = degradation(d_qp, self._k_res)
        return g_qp[..., None, None] * self.D[None, None, ...]


def _solve_primal_on_mesh(mesh, d_node, case, *, lam, mu, k_res, load, p, q=10):
    """退化弹性连续 primal（真实分量式 BC，体力 0）——solve_primal_real 的 mesh-level 内核。

    d_node is the exact nested P1 representation of the fixed parent-mesh
    damage field.  The coefficient is evaluated at quadrature points.
    BC 经 apply_dirichlet_pieces_lifted **带非齐次提升**消元（2026-06-21 修复：直接 apply
    漏 −A·u_D 致载荷边能量假发散 ⇒ Θ<1，见 RESULTS §Θ<1 根因诊断）。
    """
    scalar = LagrangeFESpace(mesh, p=p)
    space = TensorFunctionSpace(scalar, shape=(2, -1))   # 分量优先 (GD,-1)
    d_field = LagrangeFESpace(mesh, p=1).function()
    d_field[:] = d_node
    material = _NodalDegradedElasticMaterial(
        lam, mu, d_field, k_res=k_res)
    bform = BilinearForm(space)
    bform.add_integrator(
        LinearElasticIntegrator(material, q=q, method="voigt"))
    A = bform.assembly()
    ndof = space.number_of_global_dofs()
    f = bm.zeros(ndof, dtype=bm.float64)
    uh, _, _ = apply_dirichlet_pieces_lifted(A, f, space, case, load)
    return uh, space


def _prolong_u(IM, u_coarse_flat, nn_c, nn_f):
    """粗 P1 向量位移 (2*nn_c,) 分量优先 → 细网格 (2*nn_f,)，逐分量 IM 延拓（嵌套精确）。"""
    uc = bm.to_numpy(u_coarse_flat).reshape(2, nn_c)     # [x行, y行]
    P = IM                                                # (nn_f, nn_c)
    uf = np.stack([P @ uc[0], P @ uc[1]], axis=0)        # (2, nn_f)
    return bm.asarray(uf.reshape(-1))


def _nested_damage_check(mesh_c, d_node_c, mesh_f, d_node_f, bcs):
    """Compare fine P1 values with direct evaluation of the parent P1 field."""
    node_c = np.asarray(bm.to_numpy(mesh_c.entity("node")))
    cell_c = np.asarray(bm.to_numpy(mesh_c.entity("cell")), dtype=np.int64)
    node_f = np.asarray(bm.to_numpy(mesh_f.entity("node")))
    cell_f = np.asarray(bm.to_numpy(mesh_f.entity("cell")), dtype=np.int64)
    dc = np.asarray(bm.to_numpy(d_node_c))
    df = np.asarray(bm.to_numpy(d_node_f))
    nc0 = cell_c.shape[0]
    ncf = cell_f.shape[0]

    # FEALPy uniform_refine interleaves children so child k has parent k % NC0.
    sample = np.linspace(0, ncf - 1, min(256, ncf), dtype=np.int64)
    parent = sample % nc0
    pts = np.einsum("qj,sjd->sqd", bcs, node_f[cell_f[sample]])
    tri = node_c[cell_c[parent]]
    mat = np.stack((tri[:, 0] - tri[:, 2],
                    tri[:, 1] - tri[:, 2]), axis=-1)
    rhs = pts - tri[:, None, 2]
    lam01 = np.einsum("sij,sqj->sqi", np.linalg.inv(mat), rhs)
    lam = np.concatenate(
        (lam01, (1.0 - lam01.sum(axis=-1))[..., None]), axis=-1)
    d_parent = np.einsum("sqj,sj->sq", lam, dc[cell_c[parent]])
    d_fine = np.einsum("qj,sj->sq", bcs, df[cell_f[sample]])
    return float(np.max(np.abs(d_fine - d_parent)))


def _reference_subtriangles(nsub):
    """Uniform nsub^2 partition of the barycentric reference triangle."""
    triangles = []
    for i in range(nsub):
        for j in range(nsub - i):
            xy = np.array(((i, j), (i + 1, j), (i, j + 1))) / nsub
            triangles.append(np.column_stack(
                (xy, 1.0 - xy.sum(axis=1))))
    for i in range(nsub - 1):
        for j in range(nsub - 1 - i):
            xy = np.array(
                ((i + 1, j), (i + 1, j + 1), (i, j + 1))) / nsub
            triangles.append(np.column_stack(
                (xy, 1.0 - xy.sum(axis=1))))
    return triangles


def _eta_composite(discr, u_field, *, lam, mu, k_res, q, nsub):
    """Evaluate eta with degree-q quadrature on nsub^2 subtriangles/cell."""
    mesh = discr.mesh
    qf = mesh.quadrature_formula(q, "cell")
    bcs0, ws = qf.get_quadrature_points_and_weights()
    bcs0 = np.asarray(bm.to_numpy(bcs0))
    cm = mesh.entity_measure("cell") / (nsub * nsub)
    eta_t2 = np.zeros(int(mesh.number_of_cells()))
    for vertices in _reference_subtriangles(nsub):
        bcs = bm.asarray(bcs0 @ vertices)
        grad_uh = u_field.grad_value(bcs)
        sigmah_qp = discr.state.sigma(bcs)
        d_qp = discr.state.d(bcs)
        if d_qp.ndim == 3:
            d_qp = d_qp[..., 0]
        ind = equilibrated_indicator(
            mesh, grad_uh, sigmah_qp, d_qp,
            lam=lam, mu=mu, k_res=k_res, weights=ws, cellmeasure=cm)
        eta_t2 += np.asarray(bm.to_numpy(ind["eta_T"])) ** 2
    return float(np.sqrt(eta_t2.sum()))


def main():
    bm.set_backend("numpy")
    smoke = os.environ.get("SMOKE", "0") == "1"
    mat = _Mat()
    k_res = _f("KRES", 1e-6)
    nx = _i("NX", 8 if smoke else 24)
    du = _f("DU", 1e-3 if smoke else 2.5e-4)
    nstep = _i("NSTEP", 2 if smoke else 20)              # 跑到接近峰值的固定接受态
    default_nrefs = "1" if smoke else "1,2,3,4"
    nrefs = [int(v) for v in os.environ.get(
        "NREFS", os.environ.get("NREF", default_nrefs)).split(",")]
    p_u = _i("PU", 1)                                    # P1 位移（嵌套延拓精确，见 docstring）
    q = _i("Q", 10)
    eta_qs = [int(v) for v in os.environ.get(
        "ETA_QS", str(q)).split(",")]
    eta_subdivs = [int(v) for v in os.environ.get(
        "ETA_SUBDIVS", "").split(",") if v]

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
    d_node_c = discr.state.d[:]
    u_coarse, space_c = _solve_primal_on_mesh(
        discr.mesh, d_node_c, case, lam=mat.lam, mu=mat.mu,
        k_res=k_res, load=load, p=p_u, q=q)
    eta_by_q = {}
    for eta_q in eta_qs:
        eta_by_q[eta_q] = float(eta_from_state(
            discr, lam=mat.lam, mu=mat.mu, k_res=k_res,
            q=eta_q, u_override=u_coarse)["eta"])
        print(f"[theta] eta_tau q={eta_q} = {eta_by_q[eta_q]:.10e}",
              flush=True)
    for q0, q1 in zip(eta_qs[:-1], eta_qs[1:]):
        rel = abs(eta_by_q[q1] - eta_by_q[q0]) / eta_by_q[q1]
        print(f"[theta] eta quadrature change q={q0}->{q1}: "
              f"rel={rel:.3e}", flush=True)
    eta_composite = {}
    for nsub in eta_subdivs:
        eta_composite[nsub] = _eta_composite(
            discr, u_coarse, lam=mat.lam, mu=mat.mu, k_res=k_res,
            q=q, nsub=nsub)
        print(f"[theta] eta composite q={q} nsub={nsub}: "
              f"{eta_composite[nsub]:.10e}", flush=True)
    for s0, s1 in zip(eta_subdivs[:-1], eta_subdivs[1:]):
        rel = abs(eta_composite[s1] - eta_composite[s0]) / eta_composite[s1]
        print(f"[theta] eta composite change nsub={s0}->{s1}: "
              f"rel={rel:.3e}", flush=True)
    eta = (eta_composite[eta_subdivs[-1]] if eta_subdivs
           else eta_by_q.get(q, eta_by_q[eta_qs[-1]]))

    # ---------- 3) 细网格 truth：uniform-refine + 嵌套精确延拓 d/u ----------
    mesh_c = discr.mesh
    nn_c = int(mesh_c.number_of_nodes())
    d_node_c_np = np.asarray(bm.to_numpy(d_node_c))
    results = []
    for nref in nrefs:
        mesh_f = type(mesh_c)(
            bm.copy(mesh_c.entity("node")), bm.copy(mesh_c.entity("cell")))
        IMs = mesh_f.uniform_refine(n=nref, returnim=True)
        P_total = IMs[0]
        for k in range(1, len(IMs)):
            P_total = P_total @ IMs[k]
        nn_f = int(mesh_f.number_of_nodes())
        d_node_f = bm.asarray(P_total @ d_node_c_np)

        qf = mesh_f.quadrature_formula(q)
        bcs, ws = qf.get_quadrature_points_and_weights()
        consistency = _nested_damage_check(
            mesh_c, d_node_c, mesh_f, d_node_f,
            np.asarray(bm.to_numpy(bcs)))
        print(f"[theta] nested-d consistency: nref={nref} "
              f"max_abs={consistency:.3e}", flush=True)

        u_ref, space_f = _solve_primal_on_mesh(
            mesh_f, d_node_f, case, lam=mat.lam, mu=mat.mu,
            k_res=k_res, load=load, p=p_u, q=q)
        nc_f = int(mesh_f.number_of_cells())
        print(f"[theta] fine truth: nref={nref} nc_fine={nc_f} "
              f"nn_fine={nn_f}", flush=True)

        # ---------- 4) 能量误差：所有量在同一细网格积分点评价 ----------
        u_coarse_on_fine_flat = _prolong_u(
            P_total, u_coarse[:], nn_c, nn_f)
        u_cf = space_f.function()
        u_cf[:] = u_coarse_on_fine_flat
        cm = mesh_f.entity_measure("cell")
        grad_cf = u_cf.grad_value(bcs)
        grad_rf = u_ref.grad_value(bcs)
        dscalar_f = LagrangeFESpace(mesh_f, p=1).function()
        dscalar_f[:] = d_node_f
        d_qp = dscalar_f(bcs)
        g = degradation(d_qp, k_res)
        de = strain_to_voigt(grad_cf) - strain_to_voigt(grad_rf)
        Cde = stress_voigt_from_strain(de, mat.lam, mat.mu)
        integrand = g * voigt_inner(de, Cde)
        val = bm.sum(cm * bm.einsum("q,cq->c", ws, integrand))
        err = float(bm.sqrt(bm.where(
            val < 0, bm.zeros_like(val), val)))
        theta = eta / err if err > 1e-300 else float("inf")
        results.append((nref, nc_f, err, theta, consistency))
        print(f"[theta] RESULT nref={nref} nc={nc_f} eta={eta:.8e} "
              f"err={err:.8e} theta={theta:.8f} "
              f"dcheck={consistency:.3e}", flush=True)

    err_values = np.array([r[2] for r in results])
    theta_values = np.array([r[3] for r in results])
    err_monotone = bool(np.all(np.diff(err_values) >= -1e-12))
    theta_monotone = bool(np.all(np.diff(theta_values) <= 1e-12))
    print(f"[theta] monotonicity: err_up={err_monotone} "
          f"theta_down={theta_monotone}", flush=True)
    if not err_monotone or not theta_monotone:
        raise RuntimeError("nested-reference monotonicity check failed")
    print(f"[theta] DONE nx={nx} load={load:.3e} p_u={p_u} q={q}",
          flush=True)


if __name__ == "__main__":
    main()
