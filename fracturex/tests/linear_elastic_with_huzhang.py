
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace.huzhang_fe_space_2d import HuZhangFESpace2d
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.fem.huzhang_stress_integrator import HuZhangStressIntegrator
#from fealpy.fem.huzhang_displacement_integrator import HuZhangDisplacementIntegrator
from fealpy.fem.huzhang_mix_integrator import HuZhangMixIntegrator
from fealpy.fem import VectorSourceIntegrator

from fealpy.decorator import cartesian, barycentric

from fealpy.fem import BilinearForm,ScalarMassIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator,BoundaryFaceSourceIntegrator
from fealpy.fem import DivIntegrator
from fealpy.fem import BlockForm,LinearBlockForm

from fracturex.boundarycondition.huzhang_boundary_condition import HuzhangBoundaryCondition, HuzhangStressBoundaryCondition, build_isNedge_from_isD


from linear_elastic_pde import LinearElasticPDE

from sympy import symbols, sin, cos, Matrix, lambdify

from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table

from fealpy.solver import spsolve
from scipy.sparse.linalg import spsolve as scipy_spsolve
from scipy.sparse import csr_matrix, coo_matrix, bmat, spdiags



import sys
import time

import numpy as np

def l2_error_sigma(mesh, sigmah, sigma_exact, q=30):
    """
    sigmah: FE function, 返回 (NC,NQ,3) 或 (NQ,NC,3) 视你的接口
    sigma_exact(points): 返回 (...,3) 对应 [xx,xy,yy]
    """
    index = None
    qf = mesh.quadrature_formula(q, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()      # (NQ,3), (NQ,)
    cm = mesh.entity_measure('cell', index=index)         # (NC,)

    # 物理点 (NC,NQ,2)
    pts = mesh.bc_to_point(bcs, index=index)

    # exact: (NC,NQ,3)
    se = sigma_exact(pts)

    # approx: (NC,NQ,3) ——这里按你的 FEALPy 接口调整
    sh = sigmah(bcs)  # 或 sigmah.value(bcs) 等

    diff = se - sh

    # Voigt Frobenius 权重
    w = np.array([1.0, 2.0, 1.0])

    # (NC,NQ)
    d2 = diff[...,0]**2*w[0] + diff[...,1]**2*w[1] + diff[...,2]**2*w[2]

    # ∑_cell ∑_q ws * |K| * d2
    val = np.sum(ws[None,:] * d2, axis=1) * cm
    return float(np.sqrt(np.sum(val)))




def source_vector(space : LagrangeFESpace, f : callable):
    p = space.p
    mesh = space.mesh
    TD = mesh.top_dimension()
    gdof = space.number_of_global_dofs()

    cellmeasure = mesh.entity_measure('cell')
    qf = mesh.quadrature_formula(p+3, 'cell')

    bcs, ws = qf.get_quadrature_points_and_weights()
    points = mesh.bc_to_point(bcs)

    phi  = space.basis(bcs)
    fval = f(points)
    b = bm.einsum('q, c, cql, cqd->cld', ws, cellmeasure, phi, fval)

    cell2dof = space.cell_to_dof()
    r = bm.zeros(gdof*TD, dtype=phi.dtype)
    for i in range(TD):
        bm.add.at(r, gdof*i + cell2dof, b[..., i]) 
    return r



def solve(pde, N, p):
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=N, ny=N)
    node = mesh.entity('node')
    q = 2*p + 6
    # Find corner points
    is_x_bd = (bm.abs(node[:, 0] - 0) < 1e-9) | (bm.abs(node[:, 0] - 1) < 1e-9)
    is_y_bd = (bm.abs(node[:, 1] - 0) < 1e-9) | (bm.abs(node[:, 1] - 1) < 1e-9)
    is_corner = is_x_bd & is_y_bd
    corner_coords = node[is_corner]
    mesh.meshdata['corner'] = corner_coords
    def isD_bd(bc):
        tol = 1e-12
        return bm.abs(bc[:, 1] - 0.0) < tol  # bc: (NEb,2) 边重心
    
    isNedge = build_isNedge_from_isD(mesh, isD_bd)

    space0 = HuZhangFESpace2d(mesh, p=p, use_relaxation=True, 
                              bd_stress=isNedge)

    #space0 = HuZhangFESpace2d(mesh, p=p, use_relaxation=True)

    space = LagrangeFESpace(mesh, p=p-1, ctype='D')
    space1 = TensorFunctionSpace(space, shape=(-1, 2))

    lambda0 = pde.lambda0
    lambda1 = pde.lambda1

    gdof0 = space0.number_of_global_dofs()
    gdof1 = space1.number_of_global_dofs()
    uh = space1.function()

    @cartesian
    def coef_func(p, index=None):
        return bm.ones(p.shape[:-1], dtype=bm.float64)
    
    @barycentric
    def coef_func_bary(bcs, index=None):
        return bm.ones((1, )+bcs.shape[:-1], dtype=bm.float64)

    bform1 = BilinearForm(space0)
    bform1.add_integrator(HuZhangStressIntegrator(coef=coef_func_bary, lambda0=lambda0, lambda1=lambda1))

    bform2 = BilinearForm((space1,space0))
    bform2.add_integrator(HuZhangMixIntegrator())
    
    M = bform1.assembly()
    B = bform2.assembly()

    M = bform1.assembly().to_scipy().tocsr()
    B = bform2.assembly().to_scipy().tocsr()
    TM = space0.TM.to_scipy().tocsr()

    M2 = TM.T @ M @ TM
    B2 = TM.T @ B
    A = bmat([[M2,  B2],
          [B2.T, None]], format="csr")


    # A = BlockForm([[bform1,bform2],
    #                 [bform2.T,None]])
    # A = A.assembly()

    lform1 = LinearForm(space1)

    @cartesian
    def source(x, index=None):
        return pde.source(x)
    lform1.add_integrator(VectorSourceIntegrator(source=source))

    b = lform1.assembly()


    HBC = HuzhangBoundaryCondition(space=space0)
    a = HBC.displacement_boundary_condition(value=pde.displacement, threshold=isD_bd)
    #a = displacement_boundary_condition(space0, pde.displacement)

    F = bm.zeros(A.shape[0], dtype=A.dtype)
    F[:gdof0] = TM.T @ a
    #F[:gdof0] = a
    F[gdof0:] = -b


    HSBC = HuzhangStressBoundaryCondition(space=space0)

    def is_dir_dof(point):
        x = point[:, 0]
        y = point[:, 1]
        tol = 1e-9
        flag = (bm.abs(x - 0) < tol) | (bm.abs(x - 1) < tol) | (bm.abs(y - 1) < tol) 
        return flag
    
    #mesh.edgedata['dirichlet'] = is_dir_dof
    #uh_stress, isbddof_stress = space0.set_dirichlet_bc(pde.stress, threshold=isNedge)
    uh_stress, isbddof_stress = HSBC.set_essential_bc(pde.stress, threshold=isNedge, coord="auto")

    # 扩展全系统向量
    uh_global = bm.zeros(A.shape[0], dtype=A.dtype)
    uh_global[:gdof0] = uh_stress
    
    isbddof_global = bm.zeros(A.shape[0], dtype=bool)
    isbddof_global[:gdof0] = isbddof_stress
    
    # 修改右端项: F = F - A * u_known
    F = F - A @ uh_global
    
    # 强加边界值
    F[isbddof_global] = uh_global[isbddof_global]
    
    # 修改矩阵 A (置 1 置 0)
    # 构造对角掩码矩阵
    total_dof = A.shape[0]
    bdIdx = bm.zeros(total_dof, dtype=bm.int32)
    bdIdx[isbddof_global] = 1
    
    Tbd = spdiags(bdIdx, 0, total_dof, total_dof)
    T = spdiags(1 - bdIdx, 0, total_dof, total_dof)
    
    A = T @ A @ T + Tbd

    #isBdDof = HSBC.set_essential_bc(uh, 0, A, F)
    

    X = scipy_spsolve(A, F)

    sigmaval = TM @ X[:gdof0]
    #sigmaval = X[:gdof0]
    uval = X[gdof0:]

    sigmah = space0.function()
    sigmah[:] = sigmaval

    
    uh[:] = uval
    return sigmah, uh


if __name__ == "__main__":
    lambda0 = 4
    lambda1 = 1
    maxit = 5
    p = int(sys.argv[1])

    errorType = [
                 '$|| \\boldsymbol{\\sigma} - \\boldsymbol{\\sigma}_h||_{\\Omega,0}$',
                 '$|| \\boldsymbol{u} - \\boldsymbol{u}_h||_{\\Omega,0}$',
                 ]
    errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
    h = bm.zeros(maxit, dtype=bm.float64)

    x, y = symbols('x y')

    pi = bm.pi 
    u0 = (sin(pi*x)*sin(pi*y))**2
    u1 = (sin(pi*x)*sin(pi*y))**2
    #u0 = sin(5*x)*sin(7*y)
    #u1 = cos(5*x)*cos(4*y)

    u = [u0, u1]
    pde = LinearElasticPDE(u, lambda0, lambda1)

    for i in range(maxit):
        N = 2**(i+1) 
        #N = 2**i
        print("Solving with N =", N)
        sigmah, uh = solve(pde, N, p)
        mesh = sigmah.space.mesh

        e0 = mesh.error(uh, pde.displacement) 
        e1 = mesh.error(sigmah, pde.stress)
        #e1 = l2_error_sigma(mesh, sigmah, pde.stress, q=30)

        h[i] = 1/N
        errorMatrix[0, i] = e1
        errorMatrix[1, i] = e0 
        print(N, e0, e1)

    show_error_table(h, errorType, errorMatrix)
    showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
    plt.show()























