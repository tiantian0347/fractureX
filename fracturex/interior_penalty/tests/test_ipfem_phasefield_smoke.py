"""IP-FEM 相场断裂 solver 烟雾测试（4 个 2D 算例）。

对每个算例跑 3-4 个粗网格载荷步，检查：
- solver 能构造起来、边界能挂上；
- newton_raphson 不抛异常；
- force / stored_energy / dissipated_energy 是有限值；
- 能量非负、单调递增；
- 末段力量级合理（非零）。

不检查收敛精度和物理曲线，那是大网格长时间跑的事。
"""
import numpy as np
import pytest

from fealpy.backend import backend_manager as bm

from fracturex.interior_penalty.cases import (
    Model0CircularHoleCase,
    Model0SGCase,
    Model1SquareTensionCase,
    Model2ShearCase,
    Model2SGCase,
    Model3LShapeCase,
    SentTensionMieheCase,
)


def _assert_outputs_sane(out, min_last_force: float = 1e-3):
    force = out["force"]
    stored = out["stored_energy"]
    dissipated = out["dissipated_energy"]

    assert np.all(np.isfinite(force)), f"force has NaN/Inf: {force}"
    assert np.all(np.isfinite(stored)), f"stored has NaN/Inf: {stored}"
    assert np.all(np.isfinite(dissipated)), f"dissipated has NaN/Inf: {dissipated}"
    assert np.all(stored >= -1e-9), f"stored_energy 出现负值: {stored}"
    assert np.all(dissipated >= -1e-9), f"dissipated_energy 出现负值: {dissipated}"
    assert abs(force[-1]) > min_last_force, f"末段力太小: {force[-1]}"
    for i in range(1, len(stored)):
        assert stored[i] >= stored[i - 1] - 1e-9, f"stored_energy 非单调: {stored}"
    for i in range(1, len(dissipated)):
        assert dissipated[i] >= dissipated[i - 1] - 1e-9, f"dissipated 非单调: {dissipated}"


@pytest.fixture(autouse=True)
def _numpy_backend():
    bm.set_backend("numpy")
    yield


# --------------------------------------------------------------------- model1

def test_model1_solver_construction():
    case = Model1SquareTensionCase(refine=1)
    mesh = case.build_mesh()
    solver = case.build_solver(mesh)
    assert solver.uh is not None
    assert solver.d is not None
    assert solver.mesh.number_of_cells() > 0
    assert solver.force == 0.0
    assert solver.stored_energy == 0.0
    assert solver.dissipated_energy == 0.0


def test_model1_smoke():
    case = Model1SquareTensionCase(refine=2, load_sequence=np.linspace(0, 3e-3, 4))
    out = case.run(maxit_per_step=4, rtol=1e-3, verbose=False)
    _assert_outputs_sane(out, min_last_force=1e-3)


# --------------------------------------------------------------------- model0

def test_model0_smoke():
    case = Model0CircularHoleCase(
        hmin=0.15, distmesh_maxit=20,
        load_sequence=np.linspace(0, 40e-3, 4),
    )
    out = case.run(maxit_per_step=4, rtol=1e-3, verbose=False)
    _assert_outputs_sane(out, min_last_force=1.0)


# --------------------------------------------------------------------- model2

def test_model2_smoke():
    case = Model2ShearCase(refine=2, load_sequence=np.linspace(0, 3e-3, 4))
    out = case.run(maxit_per_step=4, rtol=1e-3, verbose=False)
    _assert_outputs_sane(out, min_last_force=1e-3)


# --------------------------------------------------------------------- model3

def test_model3_smoke():
    # 用较粗网格 (nx=20) + 相应对齐的 load_point 让节点落在其上
    case = Model3LShapeCase(
        nx=20, ny=20,
        load_point=(200.0, 100.0),  # 20 分格 → dx=25 → 落在节点上
        load_sequence=np.linspace(0, 0.05, 4),
    )
    out = case.run(maxit_per_step=4, rtol=1e-3, verbose=False)
    _assert_outputs_sane(out, min_last_force=1e-3)


# --------------------------------------------------------------------- SENT

def test_sent_tension_smoke():
    """SENT tension (Miehe 2010) 冒烟：3 步小 disp，粗网格。
    服务器版跑全 1500 步的入口在
    `fracturex/interior_penalty/scripts/run_sent_tension_server.py`。
    """
    case = SentTensionMieheCase(
        refine=2, load_sequence=np.linspace(0, 3e-3, 4)
    )
    out = case.run(maxit_per_step=4, rtol=1e-3, verbose=False)
    _assert_outputs_sane(out, min_last_force=1e-3)


# --------------------------------------------------------------------- SG variants

def test_model0_sg_smoke():
    """model0 SG (内圆孔 + 应变梯度) 冒烟。"""
    case = Model0SGCase(
        hmin=0.15, distmesh_maxit=20, ell_s=0.05,
        load_sequence=np.linspace(0, 40e-3, 4),
    )
    out = case.run(maxit_per_step=4, rtol=1e-3, verbose=False)
    _assert_outputs_sane(out, min_last_force=1.0)


def test_model2_sg_smoke():
    """model2 SG (剪切 + 应变梯度) 冒烟。"""
    case = Model2SGCase(
        refine=2, ell_s=0.05,
        load_sequence=np.linspace(0, 3e-3, 4),
    )
    out = case.run(maxit_per_step=4, rtol=1e-3, verbose=False)
    _assert_outputs_sane(out, min_last_force=1e-3)


def test_sg_split_reduces_to_base_under_pure_tension():
    """Ali 2024 谱分解版 sanity check:
    - 均匀拉伸下 (无预裂纹), sg_split=True 与 sg_split=False 应给出
      同一数量级的 force，且 |f_split| ≤ |f_nosplit|（因 SG 项在 χ_+<1 处被削）。
    """
    from fealpy.mesh import TriangleMesh
    from fracturex.interior_penalty import IPFEMPhaseFieldSGSolver

    mat = dict(E=210, nu=0.3, Gc=2.7e-3, l0=0.0133)
    mat["mu"] = 210 / (2 * 1.3)
    mat["lam"] = 210 * 0.3 / (1.3 * 0.4)

    def is_disp_bd(p):
        p = np.asarray(p)
        return np.c_[np.zeros(p.shape[0], dtype=bool), np.abs(p[..., 1] - 1) < 1e-12]

    def is_fix(p):
        p = np.asarray(p)
        return np.abs(p[..., 1]) < 1e-12

    mesh1 = TriangleMesh.from_box([0, 1, 0, 1], nx=4, ny=4)
    sg1 = IPFEMPhaseFieldSGSolver(mesh1, mat, ell_s=0.2, p_disp=2, sg_split=False)
    sg1.attach_boundary(is_disp_boundary=is_disp_bd, is_dirchlet_boundary=is_fix)
    sg1.newton_raphson(1e-3, maxit=5, rtol=1e-4)
    f1 = sg1.force

    mesh2 = TriangleMesh.from_box([0, 1, 0, 1], nx=4, ny=4)
    sg2 = IPFEMPhaseFieldSGSolver(mesh2, mat, ell_s=0.2, p_disp=2, sg_split=True)
    sg2.attach_boundary(is_disp_boundary=is_disp_bd, is_dirchlet_boundary=is_fix)
    sg2.newton_raphson(1e-3, maxit=5, rtol=1e-4)
    f2 = sg2.force

    # 拉伸主导下 sg_split 前后 force 差异不应太大 (< 15%)：
    # 底部固定层 + Poisson 效应会让一些 cell tr(ε) 变小，
    # χ_+ < 1 时 SG 贡献被 attenuate。这里检查方向一致性
    # (split → 更小 force, 因为 SG 阻力减弱)。
    rel_diff = abs(f1 - f2) / max(abs(f1), 1e-10)
    assert 0 <= rel_diff < 0.15, (
        f"pure tension 下 sg_split 前后 force 差异异常: "
        f"f_nosplit={f1:.4f}, f_split={f2:.4f}, rel={rel_diff:.3f}"
    )
    # 拉伸下 split=True 使 SG 贡献变小 → force 应 ≤ split=False
    assert abs(f2) <= abs(f1) + 1e-6, (
        f"split=True 在拉伸下应给出 |force| ≤ split=False, "
        f"f_nosplit={f1:.4f}, f_split={f2:.4f}"
    )


def test_sg_split_disp_softens_under_compression():
    """在有预裂纹的算例下，``sg_split=True`` 应使 |force| 小于 ``sg_split=False``
    (因为 χ_+ 在压缩区变小，SG 贡献被 attenuated)。"""
    from fracturex.interior_penalty import IPFEMPhaseFieldSGSolver
    from fracturex.interior_penalty.cases.model1_square_tension import _default_init_mesh

    mat = dict(E=210, nu=0.3, Gc=2.7e-3, l0=0.0133)
    mat["mu"] = 210 / (2 * 1.3)
    mat["lam"] = 210 * 0.3 / (1.3 * 0.4)

    def is_disp_bd(p):
        p = np.asarray(p)
        return np.c_[np.zeros(p.shape[0], dtype=bool), np.abs(p[..., 1] - 1) < 1e-12]

    def is_fix(p):
        p = np.asarray(p)
        return np.abs(p[..., 1]) < 1e-12

    ell_s = 0.5  # 显著 SG 影响
    f_by_split = {}
    for split in [False, True]:
        mesh = _default_init_mesh(refine=1)
        sg = IPFEMPhaseFieldSGSolver(mesh, mat, ell_s=ell_s, p_disp=2, sg_split=split)
        sg.attach_boundary(is_disp_boundary=is_disp_bd, is_dirchlet_boundary=is_fix)
        sg.newton_raphson(-1e-3, maxit=5, rtol=1e-4)
        f_by_split[split] = abs(sg.force)

    assert f_by_split[True] < f_by_split[False], (
        f"压缩情形下 sg_split=True 应给出更小 |force|, 得到 "
        f"split=False → {f_by_split[False]:.4f}, split=True → {f_by_split[True]:.4f}"
    )


def test_model1_smoke_with_adaptive():
    """model1 + 自适应加密烟雾测试。"""
    case = Model1SquareTensionCase(refine=1, load_sequence=np.linspace(0, 3e-3, 4))
    solver = case.build_solver()
    solver.enable_adaptive_refinement(theta=0.2)
    NC_before = solver.mesh.number_of_cells()
    disp = case.loads()
    force = np.zeros_like(disp)
    for i in range(1, len(disp)):
        solver.newton_raphson(disp[i], maxit=4, rtol=1e-3, verbose=False)
        force[i] = solver.force
    NC_after = solver.mesh.number_of_cells()
    assert np.all(np.isfinite(force))
    # 自适应加密要么保持要么变多，不允许减少
    assert NC_after >= NC_before, f"NC 减少: {NC_before} -> {NC_after}"
    assert abs(force[-1]) > 1e-3


if __name__ == "__main__":
    bm.set_backend("numpy")
    test_model1_solver_construction()
    test_model1_smoke()
    test_model0_smoke()
    test_model2_smoke()
    test_model3_smoke()
    print("OK")
