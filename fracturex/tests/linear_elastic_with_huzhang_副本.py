
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

from fracturex.damagemodel.huzhang_boundary_condition import HuzhangBoundaryCondition, HuzhangStressBoundaryCondition


from linear_elastic_pde import LinearElasticPDE

from sympy import symbols, sin, cos, Matrix, lambdify

from fealpy.tools.show import showmultirate
from fealpy.tools.show import show_error_table

#from fealpy.solver import spsolve
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, coo_matrix, bmat, spdiags

import sys
import time

def set_essential_stress_bc(A, F, space0, pde, threshold=None):
    """
    Apply the essential stress boundary conditions in the context of the assembled system matrix `A`
    and the right-hand side vector `F`.

    Parameters:
    - A: Assembled system matrix (combined stiffness and mixed integrators).
    - F: Right-hand side vector (combined forces).
    - space0: Function space object, used to get the necessary dof information.
    - pde: The problem definition object, used to get boundary conditions.
    - threshold: Function to determine which boundary faces to apply the conditions to.
    
    Returns:
    - isBdDof: A boolean array indicating which dofs are on the boundary and have been fixed.
    """
    mesh = space0.mesh
    gdim = mesh.geo_dimension()
    gdof = space0.number_of_global_dofs()
    
    # Determine boundary face indices based on threshold
    if threshold is not None:
        index = mesh.boundary_face_index()
        bc = mesh.entity_barycenter('face', index=index)
        flag = threshold(bc)
        index = index[flag]

    # Determine boundary degrees of freedom (dof)
    face2dof = space0.face_to_dof()[index]
    isBdDof = bm.zeros(gdof, dtype=bm.bool_)

    # Initialize the frame for stress projection (normal and tangential)
    normal = mesh.face_unit_normal(index)
    tangent = mesh.edge_unit_tangent(index)
    
    # Placeholder for stress values (sigma) and corresponding right-hand side terms
    val = pde.stress_function(index, normal, tangent)
    
    # Apply boundary conditions at interior boundary points (internal faces)
    for i, bd_idx in enumerate(index):
        # Calculate stress projections
        stress_proj = val[i]
        
        # Apply stress boundary condition to the dof corresponding to this boundary face
        dof = face2dof[i]
        F[dof] -= stress_proj
        isBdDof[dof] = True  # Mark these dofs as boundary dofs

    # Apply boundary conditions to corner points if necessary
    corner_indices = space0.boundary_corner_index(index)  # Call the new function
    for corner_idx in corner_indices:
        corner_stress = pde.corner_stress(corner_idx)
        corner_dof = space0.corner_dof(corner_idx)
        
        # Apply corner stress condition
        F[corner_dof] -= corner_stress
        isBdDof[corner_dof] = True
    
    return isBdDof



def solve(pde, N, p):
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=N, ny=N)
    node = mesh.entity('node')
    edge = mesh.entity('edge')
    bd_e = mesh.boundary_face_index()
    bc_e = mesh.entity_barycenter('edge', index=bd_e)
    # ΓN: y = 1
    flagN = bm.abs(bc_e[:, 1] - 1.0) < 1e-12

    isNedge = bm.zeros(mesh.number_of_edges(), dtype=bm.bool)
    isNedge = bm.set_at(isNedge, bd_e[flagN], True)

    space0 = HuZhangFESpace2d(mesh, p=p, use_relaxation=True, isNedge=isNedge, corner_mode="auto")
    print("NCP =", space0.NCP)
    print("corner types count:",
        (space0.corner['type'] == 2).sum(),
        (space0.corner['type'] == 1).sum(),
        (space0.corner['type'] == 0).sum())


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
    #                [bform2.T,None]])
    # A = A.assembly()

    lform1 = LinearForm(space1)

    @cartesian
    def source(x, index=None):
        return pde.source(x)
    lform1.add_integrator(VectorSourceIntegrator(source=source))

    b = lform1.assembly()

    def is_displacement_bc(point):
        x = point[:, 0]
        y = point[:, 1]
        tol = 1e-9
        flag = (bm.abs(x - 0) < tol) | (bm.abs(x - 1) < tol) 
        return flag

    HBC = HuzhangBoundaryCondition(space=space0)
    a = HBC.displacement_boundary_condition(value=pde.displacement, threshold=is_displacement_bc)
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
        flag = (bm.abs(y - 0) < tol) | (bm.abs(y - 1) < tol) 
        return flag
    uh_stress, isbddof_stress = space0.set_dirichlet_bc(pde.stress, threshold=is_dir_dof)
    
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
    

    X = spsolve(A, F)

    sigmaval = TM @ X[:gdof0]
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
        N = 2**(i+2) 
        #N = 2**i
        print("Solving with N =", N)
        sigmah, uh = solve(pde, N, p)
        mesh = sigmah.space.mesh

        e0 = mesh.error(uh, pde.displacement) 
        e1 = mesh.error(sigmah, pde.stress)

        h[i] = 1/N
        errorMatrix[0, i] = e1
        errorMatrix[1, i] = e0 
        print(N, e0, e1)

    show_error_table(h, errorType, errorMatrix)
    showmultirate(plt, 2, h, errorMatrix,  errorType, propsize=20)
    plt.show()























