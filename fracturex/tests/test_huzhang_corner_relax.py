"""Structural sanity tests for HuZhangCornerRelax (no PDE solve yet)."""
from __future__ import annotations
import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

from fracturex.discretization.huzhang_corner_relax import HuZhangCornerRelax
from fracturex.boundarycondition.huzhang_boundary_condition import build_isNedge_from_isD


def make_lshape_mesh(N):
    if N % 2 != 0:
        N += 1
    mesh = TriangleMesh.from_box([-1.0, 1.0, -1.0, 1.0], nx=N, ny=N)
    cell = mesh.entity('cell')
    node = mesh.entity('node')
    bary = node[cell].mean(axis=1)
    keep = ~((bary[:, 0] > 0) & (bary[:, 1] > 0))
    new_cell = cell[keep]
    used = bm.unique(new_cell.reshape(-1))
    remap = -bm.ones(node.shape[0], dtype=new_cell.dtype)
    remap[used] = bm.arange(used.shape[0], dtype=new_cell.dtype)
    new_node = node[used]
    new_cell = remap[new_cell]
    return TriangleMesh(new_node, new_cell)


def isD_lshape_reentrant(bc):
    tol = 1e-9
    x = bc[:, 0]; y = bc[:, 1]
    on_xeq0 = (bm.abs(x) < tol) & (y > -tol) & (y < 1 + tol)
    on_yeq0 = (bm.abs(y) < tol) & (x > -tol) & (x < 1 + tol)
    return ~(on_xeq0 | on_yeq0)


def isD_square_top_only_dirichlet(bc):
    return bm.abs(bc[:, 1] - 0.0) < 1e-12


def test_square_NN_corners():
    """Square [0,1]^2 with bottom = ΓD.

    from_box 的对角剖分使两个对角顶点 m=1（单 cell，无内部边，跳过），
    另两个 m=2。
    """
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=4, ny=4)
    isNedge = build_isNedge_from_isD(mesh, isD_square_top_only_dirichlet)
    relax = HuZhangCornerRelax(mesh, p=3, isNedge=isNedge, verbose=True)
    d = relax.diag()
    print("[square NN]", d)
    # 期望 1 个 m=2 真正的 NN 角点（另一个 m=1 被跳过）
    n_m_ge_2 = sum(1 for m in d['corner_m'] if m >= 2)
    assert n_m_ge_2 == d['n_corners'], f"all retained corners must have m>=2; corner_m={d['corner_m']}"
    for m in d['corner_m']:
        assert m >= 2, f"got m={m} < 2"
    # rel 全部新分配 m+2 个；unc 新分配 3*(m-1) 个
    new_rel = sum(m + 2 for m in d['corner_m'])
    new_unc = sum(3 * (m - 1) for m in d['corner_m'])
    assert d['gdof_rel'] - d['gdof_base'] == new_rel, (d['gdof_rel'], d['gdof_base'], new_rel)
    assert d['gdof_unc'] - d['gdof_base'] == new_unc, (d['gdof_unc'], d['gdof_base'], new_unc)


def test_lshape_reentrant_corner():
    """L-shape: re-entrant corner at (0,0) shared by m=4 cells when N is even-and-doubled mesh."""
    for N in [2, 4, 8]:
        mesh = make_lshape_mesh(N)
        isNedge = build_isNedge_from_isD(mesh, isD_lshape_reentrant)
        relax = HuZhangCornerRelax(mesh, p=3, isNedge=isNedge, verbose=False)
        d = relax.diag()
        print(f"[L N={N}]", d)
        assert d['n_corners'] == 1, f"N={N}: expected 1 NN corner at (0,0), got {d['n_corners']}"
        m = d['corner_m'][0]
        assert d['gdof_rel'] - d['gdof_base'] == m + 2
        assert d['gdof_unc'] - d['gdof_base'] == 3 * (m - 1)


def test_nullspace_dimensions():
    """For each corner, TM column block must span null(C) of dim m+2 exactly."""
    mesh = make_lshape_mesh(2)
    isNedge = build_isNedge_from_isD(mesh, isD_lshape_reentrant)
    relax = HuZhangCornerRelax(mesh, p=3, isNedge=isNedge, verbose=False)
    for c in relax.corners:
        C = relax._build_normal_continuity_matrix(c)
        N = relax._nullspace(C)
        m = len(c.cells)
        assert C.shape == (2 * (m - 1), 3 * m), \
            f"C shape: got {C.shape}, expected (2(m-1), 3m)=({2*(m-1)},{3*m})"
        assert N.shape == (3 * m, m + 2), \
            f"N shape: got {N.shape}, expected (3m, m+2)=({3*m},{m+2})"
        # 验证 C N ≈ 0
        residual = np.linalg.norm(C @ N)
        assert residual < 1e-10, f"C@N residual {residual:.2e}, expected ~0"
        print(f"  corner nid={c.nid}, m={m}: C{C.shape} N{N.shape}, |CN|={residual:.2e}")


def test_cell_to_dof_unc_uniqueness():
    """Every DOF id must appear in cell_to_dof_unc (no dangling new DOFs)."""
    mesh = make_lshape_mesh(4)
    isNedge = build_isNedge_from_isD(mesh, isD_lshape_reentrant)
    relax = HuZhangCornerRelax(mesh, p=3, isNedge=isNedge, verbose=False)
    c2d = relax.cell_to_dof_unc
    seen = np.unique(c2d.reshape(-1))
    expected = np.arange(relax.gdof_unc)
    missing = np.setdiff1d(expected, seen)
    print(f"[c2d_unc] gdof_unc={relax.gdof_unc}, seen={seen.size}, missing={missing.size}")
    assert missing.size == 0, f"missing DOF ids in cell_to_dof_unc: {missing[:20]}"


def test_TM_shape_and_identity_block():
    """TM shape correct, non-corner DOFs map identically."""
    mesh = make_lshape_mesh(4)
    isNedge = build_isNedge_from_isD(mesh, isD_lshape_reentrant)
    relax = HuZhangCornerRelax(mesh, p=3, isNedge=isNedge, verbose=False)
    TM = relax.TM
    assert TM.shape == (relax.gdof_unc, relax.gdof_rel), TM.shape

    # 角点 unc DOFs
    corner_unc = set()
    for c in relax.corners:
        corner_unc.update(c.unc_dofs.reshape(-1).tolist())

    TM_dense = TM.toarray() if relax.gdof_unc < 1500 else None
    if TM_dense is not None:
        for d in range(relax.gdof_unc):
            if d in corner_unc:
                continue
            if d < relax.gdof_base:
                # 应当映射到自己（rel DOF id 与 unc DOF id 相同）
                col = TM_dense[d]
                expect = np.zeros(relax.gdof_rel); expect[d] = 1.0
                assert np.allclose(col, expect), \
                    f"non-corner DOF {d} not identity: TM row sum {col.sum()}"
        print("[TM] non-corner identity block OK")


if __name__ == "__main__":
    test_square_NN_corners()
    test_lshape_reentrant_corner()
    test_nullspace_dimensions()
    test_cell_to_dof_unc_uniqueness()
    test_TM_shape_and_identity_block()
    print("\nAll structural sanity tests PASSED")
