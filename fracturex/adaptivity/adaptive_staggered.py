"""M3 full（完整版）：平衡型 η_T 估计子驱动的自适应加密，耦合进真实 staggered 求解。

与 `tests/aposteriori/test_m3_real_staggered.py`（M3 full 最小集成，只验 η_T 定位、不加密）
的区别：本模块把「估计 → Dörfler 标记 → bisect 加密 → d/r_hist 转移 → 重建离散」整条链
接进 `HuZhangPhaseFieldStaggeredDriver` 的载荷推进，做到**网格跟随裂尖**。

设计依据（仓库已为自适应预留的接缝，本模块只做装配）：
  - driver 每个载荷步后调 `adapt_hook`，并在 hook 后重连 assembler.discr + 复位预裂纹
    （见 huzhang_phasefield_staggered.py run()）。本模块不走 adapt_hook，而是由 runner
    显式调用 `solve_one_step` + `refine_and_rebuild`，以便控制自适应停机与逐步输出。
  - `HuZhangDiscretization.rebuild_on_new_mesh(new_mesh, transfer=...)` 重建空间/state。
  - `PhaseFieldDamageModel.on_mesh_changed(...)` 在换网格后重置 quadrature 历史场 H=None
    （单调准静态加载下安全，见该方法注释）。

数据转移策略（见 RESULTS_aposteriori.md §interp 与 test_refine_interp）：
  - d, r_hist : 连续 Lagrange 节点场 → bisect `options['data']` 边中点平均转移（P1 线性精确）。
  - H         : quadrature 历史 (NC,NQ) → **不转移**，重建后置 None，下一步弹性解重算。
  - σ, u      : H(div)/DG，gdof 变、无节点对应 → **不转移**，下一载荷步重解（瞬时量）。

约定：计算走 `bm`，numpy 仅限文件 I/O（fracturex_multibackend_convention）。p=3（Hu–Zhang 要求）。
本模块用 damage_p=1 的 d 场（P1 转移已验证；见 runner 默认）。
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from fealpy.backend import backend_manager as bm
from fealpy.mesh.mesh_base import Mesh

from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.adaptivity.equilibrated_estimator import equilibrated_indicator


def eta_from_state(discr, *, lam: float, mu: float, k_res: float = 1e-6,
                   q: int = 8, u_override=None) -> Dict[str, object]:
    """从当前离散 state 的 (σ_h, d, u) 算逐元平衡型指示子 η_T。

    输入:
      discr : 已 build 的 HuZhangDiscretization（state.sigma/d/u 为生产退化混合解）
      lam,mu: Lamé 参数
      k_res : 残余刚度 g=(1-d)^2+k_res（>0）
      q     : 求积阶（默认 8，高于 σ_h(p=3)/u(p=2) 以免欠积分污染 η）
      u_override : 可选连续位移 FE function（如 solve_primal_real 的 u_h^cont）。
                   给定时用其 grad_value 算残差 r=ℂ_d ε(u_override)−σ_h ⇒ **严格 Θ**
                   （Prager–Synge 需 H¹-协调 primal；默认 None 用 DG-u = 非严格，仅趋势）。
    输出 dict:
      eta   : 标量全局 η
      eta_T : (NC,) 逐元指示子
      cen   : (NC,2) 单元重心坐标（标记定位诊断用）
    """
    mesh = discr.mesh
    qf = mesh.quadrature_formula(q, "cell")
    bcs, ws = qf.get_quadrature_points_and_weights()
    cm = mesh.entity_measure("cell")

    u_field = discr.state.u if u_override is None else u_override
    grad_uh = u_field.grad_value(bcs)                  # (NC,NQ,2,2) 位移梯度
    sigmah_qp = discr.state.sigma(bcs)                 # (NC,NQ,3) σ_h（Voigt）
    d_qp = discr.state.d(bcs)                          # (NC,NQ[,1]) 相场
    if d_qp.ndim == 3:
        d_qp = d_qp[..., 0]

    ind = equilibrated_indicator(mesh, grad_uh, sigmah_qp, d_qp,
                                 lam=lam, mu=mu, k_res=k_res,
                                 weights=ws, cellmeasure=cm)
    cen = mesh.bc_to_point(bm.array([[1 / 3, 1 / 3, 1 / 3]]))[:, 0, :]
    return {"eta": ind["eta"], "eta_T": ind["eta_T"], "cen": cen}


def make_assemblers(discr, case, damage, *, parallel: Optional[bool] = None
                    ) -> Tuple[HuZhangElasticAssembler, PhaseFieldAssembler]:
    """在（重建后的）discr 上新建弹性/相场装配器，丢弃旧的空间相关缓存。

    输入: discr/case/damage；parallel 是否并行装配（None 时用装配器默认，即并行，
          可用环境变量 FRACTUREX_ASSEMBLY_PARALLEL=0 强制串行）。
    输出: (elastic_assembler, phase_assembler)，均绑定到传入的 discr。
    """
    el = HuZhangElasticAssembler(discr, case, damage,
                                 formulation="standard", assembly_parallel=parallel)
    ph = PhaseFieldAssembler(discr, case, damage, assembly_parallel=parallel)
    return el, ph


def initial_hmin_floor(mesh, *, max_levels: int = 4) -> float:
    """由初始网格最小单元面积定 hmin 下限：base_area / 2^max_levels。

    bisect（最长边二分）每级约把单元面积减半；`cm > floor` 的过滤使任一单元最多被加密
    max_levels 次。返回标量面积阈值。
    """
    cm = mesh.entity_measure("cell")
    base = float(bm.min(cm))
    return base / float(2 ** int(max_levels))


def refine_masked(discr, damage, case, marked) -> Dict[str, object]:
    """给定 (NC,) bool 掩码：bisect 加密 + 转移 d/r_hist + 重建离散 + 重置历史场 H。

    所有标记策略（Dörfler-η_T / 驱动力 M-DF / 应力重构）共用的加密内核。
    输入:
      discr  : 当前 HuZhangDiscretization（其 mesh 就地 bisect、state 被重建）
      damage : 相场损伤模型（提供 on_mesh_changed 重置历史场）
      case   : 算例（重建时重取边界/预裂纹掩码）
      marked : (NC,) bool 掩码（已含尺寸下限过滤）
    输出 dict:
      refined : bool；nc_before/nc_after：加密前后单元数；n_marked：标记单元数
    数据转移（见 THEORY_marking_strategy §7、RESULTS §interp）：
      d, r_hist（连续 P1 节点场）经 bisect 边中点平均转移；σ/u/H 不转移（σ/u 下轮重解，H 重置）。
    """
    mesh = discr.mesh
    nc_before = int(mesh.number_of_cells())
    n_marked = int(bm.sum(marked))
    if n_marked == 0:
        return {"refined": False, "nc_before": nc_before,
                "nc_after": nc_before, "n_marked": 0}

    old_state = discr.state
    d_arr = bm.copy(old_state.d[:])
    r_arr = bm.copy(old_state.r_hist[:])
    opt = mesh.bisect_options(data={"d": d_arr, "r_hist": r_arr}, disp=False)
    mesh.bisect(marked, options=opt)
    new_d = bm.asarray(opt["data"]["d"]).reshape(-1)
    new_r = bm.asarray(opt["data"]["r_hist"]).reshape(-1)

    def _transfer(old_discr, new_discr, old_st, new_st):
        # 不可逆裁剪：d∈[0,1]；r_hist 直接继承（其不可逆由损伤模型/驱动保证）
        new_st.d[:] = bm.clip(new_d, 0.0, 1.0)
        new_st.r_hist[:] = new_r

    discr.rebuild_on_new_mesh(mesh, transfer=_transfer)
    # 历史场 H 重置（quadrature 布局换网格后无对应；单调加载下安全）
    damage.on_mesh_changed(discr, discr, old_state, discr.state, case)

    return {"refined": True, "nc_before": nc_before,
            "nc_after": int(mesh.number_of_cells()), "n_marked": n_marked}


def refine_and_rebuild(discr, damage, case, eta_T, *, theta: float,
                       hmin_floor: float) -> Dict[str, object]:
    """Dörfler(η_T) 标记 + 加密（旧 per-step 路径；M2/旧 M3-full 用）。

    输入: eta_T (NC,) 指示子；theta Dörfler bulk 比例；hmin_floor 单元面积下限。
    输出: 同 refine_masked。
    """
    mesh = discr.mesh
    cm = mesh.entity_measure("cell")
    marked = Mesh.mark(eta=eta_T, theta=theta, method="L2")
    marked = bm.logical_and(marked, cm > hmin_floor)
    return refine_masked(discr, damage, case, marked)


# ---------------------------------------------------------------------------
# M-DF：Hu–Zhang 应力驱动的预测型标记（THEORY_marking_strategy §2/§6）
# ---------------------------------------------------------------------------
def driving_force_per_cell(discr, damage, *, Gc=None, l0=None):
    """无量纲裂纹驱动力 𝒟_τ = (2 l0/G_c)·max_q H_{τ,q}（THEORY §1 (2)、§2 (3)）。

    H 是相场历史场（quadrature 布局 (NC,NQ)，= max 历史 ψ⁺），由 Hu–Zhang 精确应力算出
    （命题 2：用 σ_h 比位移应力误标带窄）。𝒟_c=1/3 是软化触发临界值 (§2 (4))。
    输入:
      discr : 已解的 HuZhangDiscretization（state.H 为 (NC,NQ)；None 则未解）
      damage: 提供 Gc,l0（on_build 后为 float）；可显式覆盖
    输出: (NC,) 𝒟_τ；state.H 为 None 时返回全 0。
    """
    Gc = float(Gc if Gc is not None else damage.Gc)
    l0 = float(l0 if l0 is not None else damage.l0)
    H = discr.state.H
    if H is None:
        return bm.zeros(discr.mesh.number_of_cells(), dtype=bm.float64)
    Hcell = bm.max(bm.asarray(H), axis=1)            # (NC,) 逐元历史峰
    return (2.0 * l0 / Gc) * Hcell


def mark_driving_force(discr, Dcell, *, l0, beta: float = 0.6, c_h: float = 2.0):
    """M-DF 标记掩码：𝒟_τ ≥ θ_D=β·𝒟_c=β/3 且 h_τ > l0/c_h（THEORY §6、§3 (6)）。

    h_τ=√(2·area_τ)（结构右三角网格的边长尺度）⇒ 下限 h_τ>l0/c_h ⟺ area_τ>(l0/c_h)²/2。
    输入:
      discr : 当前离散（取单元面积）
      Dcell : (NC,) 无量纲驱动力 𝒟_τ
      l0    : 相场长度尺度
      beta  : β∈(0,1)，θ_D=β/3；越小越提前加密（更预测，更多 DOF）
      c_h   : h_τ≤l0/c_h 即停（c_h=2 ⇒ h≤l0/2；更准用 4）
    输出: (NC,) bool 掩码。
    """
    theta_D = float(beta) / 3.0                       # 𝒟_c = 1/3 (THEORY §2 (4))
    cm = discr.mesh.entity_measure("cell")
    area_floor = (float(l0) / float(c_h)) ** 2 / 2.0  # h=√(2 area) ≤ l0/c_h ⟺ area ≤ floor
    return bm.logical_and(Dcell >= theta_D, cm > area_floor)


# ---------------------------------------------------------------------------
# Recovery marker on d（tian2024 重构型指示子；抗 g^{-2} 噪声）
# ---------------------------------------------------------------------------
def recovery_indicator_d(discr, *, method: str = "area"):
    """相场重构型指示子 η_τ^d = ‖R_h d_h − ∇d_h‖_{0,τ}（见 tian2024 §recovery）。

    直接用 FEALPy `RecoveryAlg().recovery_estimate(state.d, method)`；返回 (NC,) 逐元误差²。
    与 σ 驱动指示子相比：整个量只涉及 ∇d，无 g(d)^{-2} 权重；d≡1 的 seed 预裂缝上 ∇d_h=0、
    R_h d_h=0 ⇒ η_τ^d=0，天然屏蔽了 Section 4 里 g^{-2} 数值噪声放大源。
    输入:
      discr : 已 build 的离散（state.d 是 damage_p 阶 P1/P2 连续 Lagrange FE 函数）
      method: 权重方案 simple/area/area_harmonic/distance/distance_harmonic（tian2024 §Recovery
              各方案；area 默认，稳且便宜）
    输出: (NC,) η_τ^d²（fealpy 的 celltype=True error 返回逐元 L² 平方，直接送 Dörfler 即可）。
    """
    from fealpy.fem.recovery_alg import RecoveryAlg
    return RecoveryAlg().recovery_estimate(discr.state.d, method=method)


def _cell_max_d(discr):
    """逐元最大节点 d：用于 σ-mask 的 d_cut 过滤（"任一 q 点 d>d_cut 即视为已饱和"）。返回 (NC,)。"""
    d = discr.state.d[:]
    c2d = discr.space_d.cell_to_dof()
    return bm.max(d[c2d], axis=-1)


def _cell_min_d(discr):
    """逐元最小节点 d：用于 recovery 的 d_hi 过滤（"至少一个顶点未坏 ⇒ 可细分"）。返回 (NC,)。
    区别 _cell_max_d：seed 边界胞（一顶点 d=1，其余 d=0）min_d=0，会被保留——正是 ∇d 跳变、
    η_τ^d 最大处；若用 max_d 过滤会连过渡带一起误杀，导致 Dörfler 空标（见 mark_recovery 备注）。
    """
    d = discr.state.d[:]
    c2d = discr.space_d.cell_to_dof()
    return bm.min(d[c2d], axis=-1)


def mark_recovery(discr, eta_d, *, l0, c_h: float = 2.0, theta: float = 0.5,
                  d_hi: float = 0.995):
    """基于 η_τ^d 的 Dörfler L² 标记，尺寸下限 h_τ>l0/c_h，且跳过完全断裂胞（min_d>d_hi）。

    过滤放在 Dörfler **之前**：先把已完全断裂胞（min d>d_hi，所有顶点都已坏）的 η 清零，再算
    L² 累积；否则 seed 内部（若 eta 非零）会挤占 bulk 配额，把过渡带真正想细分的胞漏掉。
    输入:
      discr : 当前离散
      eta_d : (NC,) 来自 recovery_indicator_d 的逐元误差²
      l0    : 相场长度尺度
      c_h   : h_τ≤l0/c_h 停加密
      theta : Dörfler bulk 比例（0.5 与 model0 rg 一致）
      d_hi  : 逐元 min d 上限；>d_hi 视为完全断裂，Dörfler 前清零
    输出: (NC,) bool 掩码。
    """
    cm = discr.mesh.entity_measure("cell")
    area_floor = (float(l0) / float(c_h)) ** 2 / 2.0
    keep = bm.logical_and(cm > area_floor, _cell_min_d(discr) <= d_hi)
    eta_masked = bm.where(keep, eta_d, bm.zeros_like(eta_d))
    marked = Mesh.mark(eta=eta_masked, theta=theta, method="L2")
    return bm.logical_and(marked, keep)


def mark_hybrid(discr, damage, Dcell, eta_d, *, l0,
                beta: float = 0.6, c_h: float = 2.0, d_cut: float = 0.9,
                theta_rec: float = 0.5, d_hi_rec: float = 0.995):
    """混合标记：σ-mask（d_cut on cell max d，屏蔽 g^{-2} 噪声）∪ recovery Dörfler（d_hi on cell min d）。

    两个子掩码各自的域：
      - σ-mask   : {τ : 𝒟_τ≥β/3, cell max d ≤ d_cut, h_τ>l0/c_h}
                   d_cut<1 排除任一 q 点 d>d_cut 的胞（Section 4 Htau 处方；屏蔽 g^{-2} 噪声）。
      - recovery : mark_recovery 的输出（d_hi 用 cell min d，保留过渡带；预清零后 Dörfler）。
    合并: σ-mask ∨ recovery-mask。

    输入:
      discr, damage : 当前离散/损伤
      Dcell         : (NC,) driving_force_per_cell 的返回
      eta_d         : (NC,) recovery_indicator_d 的返回
      l0, beta, c_h : 同 mark_driving_force
      d_cut         : σ-mask 上限（默认 0.9；作用于 cell max d）
      theta_rec     : recovery Dörfler bulk
      d_hi_rec      : recovery 上限（默认 0.995；作用于 cell min d）
    输出: (NC,) bool 掩码。
    """
    cm = discr.mesh.entity_measure("cell")
    area_floor = (float(l0) / float(c_h)) ** 2 / 2.0
    theta_D = float(beta) / 3.0
    mask_sigma = bm.logical_and.reduce([Dcell >= theta_D,
                                        _cell_max_d(discr) <= d_cut,
                                        cm > area_floor])
    mask_rec = mark_recovery(discr, eta_d, l0=l0, c_h=c_h,
                             theta=theta_rec, d_hi=d_hi_rec)
    return bm.logical_or(mask_sigma, mask_rec)


# ---------------------------------------------------------------------------
# η_T 直接作 marker（Prager–Synge，§3 Cor.5.3 reliability=1 + Remark 5.6 局部下界）
# ---------------------------------------------------------------------------
def mark_eta_T_indicator(discr, eta_T2, *, l0, c_h: float = 2.0,
                         theta: float = 0.5, d_hi: float = 0.995,
                         strategy: str = "max"):
    """基于 η_T² = ‖ℂ_d ε(u_h^c) − σ_h‖_{𝔸_d,T}² 的元素标记。

    理论支撑（论文 §3 Cor.5.3 + Remark 5.6）：
      - 全局: ‖ε(u_h^c − u)‖_{ℂ_d} ≤ η = (Σ η_T²)^{1/2}，reliability constant = 1；
      - 局部: η_{ω_z}² ≤ C_{κ_z} (‖ε(u_h^c − u)‖²_{ℂ_d,ω_z} + ‖σ_h − σ‖²_{𝔸_d,ω_z})，
              patch contrast κ_{ω_z} 有界 ⇒ η_T 作 marker 满足 efficiency-up-to-osc。
    因此 marker on η_T² 是理论上 justified 的（区别于 §4 heuristic 𝒟_τ）。

    标记策略（strategy 参数）：
      - "max"（默认, tian2024 adaptive_paper §Recovery L669 同款）: {τ: η_T²(τ) ≥ θ·max_τ η_T²(τ)}
        在弹性阶段 η_T 空间平缓时天然收敛：max 是被少数裂尖胞主导时才会有大量胞入选。
      - "L2" (Dörfler bulk): 累积 η² ≥ θ·total。在裂纹带高度局部化后收敛快，但在弹性阶段
        bulk-η 均匀时会一次性挑近半数胞（SENT 4 步实测 76%）。
    完全断裂胞（min d>d_hi）预清零。尺寸下限 h_τ>l0/c_h。

    输入:
      discr : 当前离散
      eta_T2: (NC,) 逐元 η_T²（由 eta_from_state(u_override=u_h^c) 返回）
      l0    : 相场长度尺度
      c_h   : 尺寸下限系数
      theta : 阈值参数（max: 0.3–0.5 常用；L2: 0.1–0.2 保守）
      d_hi  : cell min d 上限
      strategy: "max" | "L2"
    输出: (NC,) bool 掩码。
    """
    cm = discr.mesh.entity_measure("cell")
    area_floor = (float(l0) / float(c_h)) ** 2 / 2.0
    keep = bm.logical_and(cm > area_floor, _cell_min_d(discr) <= d_hi)
    eta_masked = bm.where(keep, eta_T2, bm.zeros_like(eta_T2))
    if strategy == "max":
        eta_max = float(bm.max(eta_masked))
        if eta_max <= 0.0:
            return bm.zeros_like(eta_masked, dtype=bool)
        marked = eta_masked >= float(theta) * eta_max
    elif strategy == "L2":
        marked = Mesh.mark(eta=eta_masked, theta=theta, method="L2")
    else:
        raise ValueError(f"unknown strategy={strategy!r}; expected max/L2")
    return bm.logical_and(marked, keep)
