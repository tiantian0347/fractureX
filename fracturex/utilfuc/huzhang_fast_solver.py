from typing import Optional, TypeVar, Union, Generic, Callable
from fealpy.typing import TensorLike, Index, _S, Threshold

import pyamg
import time

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, HuZhangFESpace2d
from fealpy.mesh import TriangleMesh, TetrahedronMesh
from fealpy.decorator import barycentric, cartesian
from fealpy.fem import BilinearForm, ScalarDiffusionIntegrator, DirichletBC

from scipy.sparse import csr_matrix, coo_matrix, spdiags, bmat
from scipy.sparse.linalg import LinearOperator, gmres, minres, lgmres, spsolve

from orthogonal_hzfe_space import HuZhangOrthogonalFEspace2d
from orthogonal_hzfe_space import HuZhangOrthogonalFEspace3d
from orthogonal_L2fe_space import L2OrthogonalFEspace 
from orthogonal_fe_space import H1OrthogonalFEspace2d, H1OrthogonalFEspace3d

from pyamg.relaxation.relaxation import gauss_seidel
from scipy.linalg import sqrtm, eigh 
from scipy.sparse import eye
from scipy.linalg import cho_factor, cho_solve

from smoother import JacobiPreconditioner, GaussSeidelPreconditioner

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

iter_count = 0

def mumps_solve(A, b):
    from mumps import DMumpsContext
    from scipy.sparse import coo_matrix
    x = b.copy()

    ctx = DMumpsContext()
    ctx.set_silent()
    ctx.set_centralized_sparse(A)

    ctx.set_rhs(x)
    ctx.run(job=6)
    ctx.destroy()
    return x


import numpy as np

def _get_diag(A, bm):
    """尽量兼容 scipy sparse / numpy / backend matrix."""
    if hasattr(A, "diagonal"):
        d = A.diagonal()
    else:
        d = bm.diag(A)
    return bm.asarray(d, dtype=bm.float64)


def estimate_lambda_max_DinvA(A, dinv, bm, iters=10):
    """
    幂迭代估计 lambda_max(D^{-1} A)
    这里只用于 Chebyshev 磨光，不要求特别精确。
    """
    n = A.shape[0]
    x = bm.ones(n, dtype=bm.float64)
    x = x / bm.linalg.norm(x)

    lam = 1.0
    print("Starting lambda_max estimate...")
    for _ in range(iters):
        y = dinv * (A @ x)
        ny = bm.linalg.norm(y)
        ny_val = float(ny)
        if ny_val < 1e-30:
            return 1.0
        x = y / ny
        Ax = dinv * (A @ x)
        lam = float(bm.sum(x * Ax) / bm.sum(x * x))
    print("Ending lambda_max estimate:", lam)
    return max(lam, 1e-12)


def chebyshev_smooth(A, x, b, dinv, lmax, bm, degree=2, lower_ratio=0.1):
    """
    Chebyshev smoother:
        x <- x + p_m(D^{-1}A)(b - A x)

    参数:
        A          : 线性算子 / 矩阵
        x          : 当前近似（会返回新值）
        b          : 右端
        dinv       : 对角预条件 D^{-1}
        lmax       : lambda_max(D^{-1}A) 估计
        degree     : Chebyshev 次数，通常 2~4
        lower_ratio: 取 lambda_min = lower_ratio * lambda_max
                     做磨光时常用 0.1 ~ 0.3
    """
    if degree <= 0:
        return x

    lmin = lower_ratio * lmax
    lmin = max(lmin, 1e-12 * lmax)

    theta = 0.5 * (lmax + lmin)
    delta = 0.5 * (lmax - lmin)

    # 极端情况下退化成 Richardson/Jacobi
    if abs(delta) < 1e-30:
        r = b - A @ x
        return x + (dinv / theta) * r

    sigma = theta / delta
    rho = 1.0 / sigma

    # 第一步
    r = b - A @ x
    z = dinv * r
    p = z / theta
    x = x + p

    # 后续递推
    for _ in range(1, degree):
        r = b - A @ x
        z = dinv * r
        rho_new = 1.0 / (2.0 * sigma - rho)
        p = rho_new * rho * p + (2.0 * rho_new / delta) * z
        x = x + p
        rho = rho_new

    return x

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

def sparse_matrix_rank(matrix, tol=1e-10):
    """
    计算稀疏矩阵的秩

    参数:
    matrix: scipy稀疏矩阵或numpy数组
    tol: 判断奇异值为零的容差

    返回:
    矩阵的秩
    """
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)

    m, n = matrix.shape
    k = min(m, n) - 1  # 最大可能的奇异值数量

    if k <= 0:
        return 0 if m * n == 0 else min(m, n)

    try:
        # 计算奇异值
        singular_values = svds(matrix.astype(float), k=k,
                               return_singular_vectors=False)
        # 统计非零奇异值数量
        rank = np.sum(singular_values > tol)
        # 确保至少返回1（如果矩阵有非零元素）
        if rank == 0 and matrix.nnz > 0:
            rank = 1
        return rank
    except:
        # 如果SVD失败，使用QR分解（对稀疏矩阵也有效）
        from scipy.sparse.linalg import qr
        Q, R, P = qr(matrix, pivoting=True)
        # 统计对角线非零元素数量
        diag_R = R.diagonal()
        rank = np.sum(np.abs(diag_R) > tol)
        return rank

def optimal_decomposition(S_diag, v):
    """
    Solve:
        min sum_i v_i^T S_i v_i
        s.t. sum_i R_i^T v_i = v
    """
    IS = []
    JS = []
    DS = []
    IR = []
    JR = []
    DR = []

    L = 0
    for S_i, R_i in S_diag:
        l = len(R_i)

        I = bm.arange(l, dtype=bm.int32)+L
        IS.append(bm.broadcast_to(I[:, None], (l, l)).reshape(-1))
        JS.append(bm.broadcast_to(I[None, :], (l, l)).reshape(-1))
        DS.append(S_i.reshape(-1))

        IR.append(R_i)
        JR.append(I)
        DR.append(bm.ones(l, dtype=bm.float64))
        L += l

    I = bm.concatenate(IS)
    J = bm.concatenate(JS)
    D = bm.concatenate(DS)
    S = coo_matrix((D, (I, J)), shape=(L, L)).tocsc()

    I = bm.concatenate(IR)
    J = bm.concatenate(JR)
    D = bm.concatenate(DR)
    R = coo_matrix((D, (I, J)), shape=(v.shape[0], L)).tocsc()

    print(R.shape)

    A = bmat([[S, R.T],
               [R, None]], format='csc')

    rank_S = sparse_matrix_rank(S)
    rank_R = sparse_matrix_rank(R)
    print("rank of S:", rank_S, "S's shape :", S.shape)
    print("rank of R:", rank_R, "R's shape :", R.shape)

    v = bm.concatenate([bm.zeros(L, dtype=bm.float64), v])

    lam = mumps_solve(A, v)
    nv = lam[:L]

    v_local = []
    v_all = v[L:]
    for _, R_i in S_diag:
        l = len(R_i)
        v_all[R_i] -= nv[:l]
        v_local.append(nv[:l])
        nv = nv[l:] 
    print("Right ?", bm.linalg.norm(v_all))
    return v_local


def plot_optimal_decomposition(space, S, v):
    import matplotlib.pyplot as plt

    S_diag = vertex_preconditioner(space, S)

    v_local = optimal_decomposition(S_diag, v)
    mesh = space.mesh
    NN = mesh.number_of_nodes()
    for i in range(NN):
        vi = v_local[i]
        _, dofs = S_diag[i]
        ui = space.function()
        ui[dofs] = vi
        plot_function(ui, com=0)
    plt.show()

def plot_on_tri_lattice(x, y, z, axes):
    """
    Plot values z given at points (x,y) that form a triangular lattice (or arbitrary scattered points).
    Produces two figures: a 3D surface and a 2D filled contour.
    Inputs:
      x, y, z : 1D arrays of the same length
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()
    assert x.shape == y.shape == z.shape, "x, y, z must be same length"
    
    # Build triangulation (matplotlib will triangulate the scattered points)
    tri = mtri.Triangulation(x, y)
    
    # --- 3D surface ---
    surf = axes.plot_trisurf(tri, z, linewidth=0.2, edgecolor='gray')  # no explicit color map
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')

def plot_function(uh, com):

    fig1 = plt.figure(figsize=(8,6))
    axes = fig1.add_subplot(111, projection='3d')

    space = uh.space
    mesh = space.mesh
    NC = mesh.number_of_cells()

    p = 5 
    bcs = bm.multi_index_matrix(p, 2)/p
    points = mesh.bc_to_point(bcs)
    vals = uh(bcs)
    for c in range(NC):
        ptsc = points[c]
        valsc = vals[c]
        plot_on_tri_lattice(ptsc[:,0], ptsc[:,1], valsc, axes)

def vertex_dof_collocation(space, alldof = True):
    p  = space.p
    NN = space.mesh.number_of_nodes()
    NC = space.mesh.number_of_cells()
    gdof = space.number_of_global_dofs()
    mesh = space.mesh
    cell = mesh.entity("cell")

    c2d = space.dof.cell_to_dof()
    c2n = space.mesh.entity("cell")
    ldof  = space.dof.number_of_local_dofs()


    flag0 = bm.ones(ldof, dtype=bm.bool)
    flag1 = bm.ones(ldof, dtype=bm.bool)
    flag2 = bm.ones(ldof, dtype=bm.bool)

    if not alldof:

        nldof = space.dof.number_of_internal_local_dofs("node")
        eldof = space.dof.number_of_internal_local_dofs("edge")
        cldof = space.dof.number_of_internal_local_dofs("cell")

        flag0[nldof*1:nldof*2] = 0
        flag0[nldof*2:nldof*3] = 0
        flag0[3*nldof:3*nldof+eldof] = 0

        flag1[:nldof*1] = 0
        flag1[nldof*2:nldof*3] = 0
        flag1[3*nldof+eldof:3*nldof+2*eldof] = 0

        flag2[:nldof*2] = 0
        flag2[3*nldof+2*eldof:3*nldof+3*eldof] = 0

    N = flag0.sum() 
    c2dforv = bm.zeros((NC, 3, N), dtype=bm.int32)
    c2dforv[:, 0] = c2d[:, flag0]
    c2dforv[:, 1] = c2d[:, flag1]
    c2dforv[:, 2] = c2d[:, flag2]

    k = 1
    ndof = bm.zeros((NN, k), dtype=bm.int32)
    ndof[cell] = c2d[:, None, :k]

    N = flag0.sum() 
    c2dforv = bm.zeros((NC, 3, N), dtype=bm.int32)
    c2dforv[:, 0] = c2d[:, flag0]
    c2dforv[:, 1] = c2d[:, flag1]
    c2dforv[:, 2] = c2d[:, flag2]

    I = bm.broadcast_to(c2n[:, :, None], (NC, 3, N)).reshape(-1)
    J = c2dforv.reshape(-1)
    V = bm.ones_like(I, dtype=bm.float64)
    vertex_dof = csr_matrix((V, (I, J)), shape=(NN, gdof))
    return vertex_dof

def vertex_preconditioner_all(huspace, uspace, B, Minv):
    """顶点预处理"""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import LinearOperator

    hzvertex_dof = vertex_dof_collocation(huspace, alldof=True)
    uvertex_dof  = vertex_dof_collocation(uspace,  alldof=True)

    mesh = huspace.mesh
    NN = mesh.number_of_nodes()
    ugdof = uspace.number_of_global_dofs()

    S_diag = []
    S = B.T @ Minv @ B
    for i in range(NN):
        # 获得与顶点i相关的所有自由度
        start = hzvertex_dof.indptr[i]
        end = hzvertex_dof.indptr[i+1]
        dofs_hz = hzvertex_dof.indices[start:end]

        start = uvertex_dof.indptr[i]
        end = uvertex_dof.indptr[i+1]
        dofs_u = uvertex_dof.indices[start:end]
        dofs_u = bm.concatenate([dofs_u, dofs_u + ugdof])

        #B_sub = B[dofs_hz, :][:, dofs_u]
        B_sub = B[:, dofs_u]
        #Minv_sub = spdiags(Minv.diagonal()[dofs_hz], 0, len(dofs_hz), len(dofs_hz)) 
        Minv_sub = Minv 

        S_i = B_sub.T @ Minv_sub @ B_sub

        S_diag.append([bm.linalg.inv(S_i.toarray()), dofs_u])

    def preconditioner(r):
        """
        @brief Jacobi 型的预处理
        """
        r = r.astype(bm.float64)
        e = bm.zeros_like(r)
        for i in range(NN):
            M_i, dofs = S_diag[i]
            e[dofs] += M_i @ r[dofs]
        return e
    shape = (ugdof*2, ugdof*2)
    pre = LinearOperator(shape=shape, matvec=preconditioner)
    return pre

def vertex_preconditioner(space, A):
    """顶点预处理"""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import LinearOperator

    A = A.tocsr()
    n = A.shape[0]
    A_diag = [] 

    p  = space.p
    NN = space.mesh.number_of_nodes()
    NC = space.mesh.number_of_cells()
    gdof = space.number_of_global_dofs()
    mesh = space.mesh
    cell = mesh.entity("cell")

    c2d = space.dof.cell_to_dof()
    c2n = space.mesh.entity("cell")
    ldof  = space.dof.number_of_local_dofs()

    vertex_dof = vertex_dof_collocation(space, alldof=True)

    isbdnode = space.mesh.boundary_node_flag()
    for i in range(NN):
        start = vertex_dof.indptr[i]
        end = vertex_dof.indptr[i+1]
        # 获得与顶点i相关的所有自由度
        dofs = vertex_dof.indices[start:end]

        A_sub = A[dofs, :][:, dofs]
        # Cholesky 分解
        L = cho_factor(A_sub.toarray())

        #flag = bm.abs(A_sub.toarray()) < 1e-12
        #A_diag.append([bm.linalg.inv(A_sub.toarray()), dofs])
        A_diag.append([L, dofs])





    def preconditioner(r):
        """
        @brief Jacobi 型的预处理
        """
        r = r.astype(bm.float64)
        e = bm.zeros_like(r)
        for i in range(NN):
            #M_i, dofs = A_diag[i]
            #e[dofs] += M_i @ r[dofs]
            L, dofs = A_diag[i]
            e[dofs] += cho_solve(L, r[dofs])


            #val = M_i @ r[dofs]
            #flag = dofs!=n2d[i]
            #iidd = dofs[flag]
            #e[iidd] += val[flag]

        return e
    pre = LinearOperator(shape=A.shape, matvec=preconditioner)
    return pre
    #return A_diag 

def edge_preconditioner(space, A):
    """顶点预处理"""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import LinearOperator

    A = A.tocsr()
    n = A.shape[0]
    A_diag = [] 

    p  = space.p
    NE = space.mesh.number_of_edges()
    NC = space.mesh.number_of_cells()
    gdof = space.number_of_global_dofs()

    c2d = space.dof.cell_to_dof()
    c2e = space.mesh.cell_to_edge()
    ldof  = space.dof.number_of_local_dofs()
    nldof = space.dof.number_of_internal_local_dofs("node")
    eldof = space.dof.number_of_internal_local_dofs("edge")
    cldof = space.dof.number_of_internal_local_dofs("cell")

    flag0 = bm.ones(ldof, dtype=bm.bool)
    flag1 = bm.ones(ldof, dtype=bm.bool)
    flag2 = bm.ones(ldof, dtype=bm.bool)
    flag0[:3] = 0
    flag0[3+eldof:3+2*eldof] = 0
    flag0[3+2*eldof:3+3*eldof] = 0
    flag1[:3] = 0
    flag1[3:3+eldof] = 0
    flag1[3+2*eldof:3+3*eldof] = 0
    flag2[:3] = 0
    flag2[3:3+eldof] = 0
    flag2[3+eldof:3+2*eldof] = 0

    #mindex = bm.multi_index_matrix(p, 2)
    #flag0 = mindex[:, 0] != 0
    #flag1 = mindex[:, 1] != 0
    #flag2 = mindex[:, 2] != 0

    N = flag0.sum() 
    c2dfore = bm.zeros((NC, 3, N), dtype=bm.int32)
    c2dfore[:, 0] = c2d[:, flag0]
    c2dfore[:, 1] = c2d[:, flag1]
    c2dfore[:, 2] = c2d[:, flag2]

    I = bm.broadcast_to(c2e[:, :, None], (NC, 3, N)).reshape(-1)
    J = c2dfore.reshape(-1)
    V = bm.ones_like(I, dtype=bm.float64)
    edge_dof = csr_matrix((V, (I, J)), shape=(NE, gdof))
    for i in range(NE):
        start = edge_dof.indptr[i]
        end = edge_dof.indptr[i+1]
        # 获得与顶点i相关的所有自由度
        dofs = edge_dof.indices[start:end]

        I = bm.arange(len(dofs), dtype=bm.int32)
        d = bm.ones(len(dofs), dtype=bm.float64)
        L = csr_matrix((d, (I, dofs)), shape=(len(dofs), gdof))
        A_sub = L @ A @ L.T

        A_diag.append([bm.linalg.inv(A_sub.toarray()), dofs])

    def preconditioner(r):
        """
        @brief Jacobi 型的预处理
        """
        r = r.astype(bm.float64)
        e = bm.zeros_like(r)
        for i in range(NE):
            M_i, dofs = A_diag[i]
            e[dofs] += M_i @ r[dofs]
        return e
    pre = LinearOperator(shape=A.shape, matvec=preconditioner)
    return pre

def project_space_to_huzhang(uspace0, uspace):
    """
    Projects a given function uspace0 to the Hu-Zhang orthogonal finite element
    uspace0.
    """

    p = uspace.p
    q = p + 3

    mesh = uspace0.mesh

    qf = mesh.quadrature_formula(q, "cell") 
    bcs, ws = qf.get_quadrature_points_and_weights()

    phi  = uspace.basis(bcs) # (NC, NQ, ldof0)
    phi0 =  uspace0.basis(bcs)

    cm = mesh.entity_measure("cell")  # (NC, )
    F  = bm.einsum('cql, cqm, q, c-> clm', phi0, phi, ws, cm)
    M  = bm.einsum('cql, cqm, q, c-> clm', phi, phi, ws, cm)
    Minv = bm.linalg.inv(M)
    F = bm.einsum('clm, cmd->cld', F, Minv)

    c2d0   = uspace0.cell_to_dof()
    gdof0  = uspace0.number_of_global_dofs()
    c2d  = uspace.cell_to_dof()
    gdof = uspace.number_of_global_dofs()

    I = bm.broadcast_to(c2d[:, None], F.shape).reshape(-1)
    J = bm.broadcast_to(c2d0[..., None], F.shape).reshape(-1)
    M = coo_matrix((F.reshape(-1), (I, J)), shape=(gdof, gdof0),
                   dtype=phi.dtype)
    return M.tocsr()

class HZFEMFastSolve():

    def __init__(self, pde, mesh, p, smooth_num=20):
        self.p   = p
        self.pde = pde
        self.mesh = mesh
        self.smooth_num = smooth_num
        TD = mesh.top_dimension()

        self.sspace = HuZhangOrthogonalFEspace2d(mesh, p)
        #self.sspace = HuZhangFEspace2d(mesh, p)
        self.uspace = H1OrthogonalFEspace2d(mesh, p-1, ctype="D") 
        #self.uspace = L2OrthogonalFEspace(mesh, p-1) 
        self.uspace0 = LagrangeFESpace(mesh, 1)

        self.ugdof = self.uspace.number_of_global_dofs()
        self.sgdof = self.sspace.number_of_global_dofs()
        self.ugdof0 = self.uspace0.number_of_global_dofs()
        self.TD    = TD
        self.gdof  = self.sgdof + self.ugdof*TD

        self.get_fast_solver_of_coarse_schur()
        
        ###################### 构造计算所需矩阵 ######################
        self.PI = project_space_to_huzhang(self.uspace0, self.uspace) 
        self.M, self.D = self.mass_matrix()
        self.B = self.mix_matrix()

        self.BT = self.B.T.tocsr()
        self.PIT = self.PI.T.tocsr()

        self.OP = bmat([[self.M, self.B], [self.BT, None]], format='csr')

        b = self.source_vector(self.uspace, pde.source)
        a = self.displacement_boundary_condition(self.sspace, pde.displacement)

        F = bm.zeros(self.gdof, dtype=bm.float64)
        F[:self.sgdof] = a
        F[self.sgdof:] = -b
        self.F = F

        self.D_inv = 1/self.D

        # S 相当于间断元的刚度矩阵
        sgdof  = self.M.shape[0]
        M_inv = spdiags(self.D_inv, 0, sgdof, sgdof)
        S = self.BT@M_inv@self.B
        S = S.tocsr()
        self.S = S

        #Sden = S.toarray()
        #PW = M_inv@self.B@bm.linalg.inv(Sden)@self.BT
        #import numpy as np
        #np.savetxt("PW.csv", PW, delimiter=",")
        #for i in range(len(PW)):
        #    flag = bm.abs(PW[i])<1e-10
        #    print(i, ":", bm.sum(flag))
        #exit()

        ####################### 构造顶点预处理子 ##########################
        ugdof = self.ugdof
        I = bm.arange(ugdof)
        d = bm.ones(ugdof, dtype=bm.float64)

        L = csr_matrix((d, (I, I)), shape=(ugdof, ugdof*2))
        S00 = L@self.S@L.T
        S00.data[bm.abs(S00.data) < 1e-12] = 0.0
        S00.eliminate_zeros()

        L = csr_matrix((d, (I, I+ugdof)), shape=(ugdof, ugdof*2))
        S11 = L@self.S@L.T
        S11.data[bm.abs(S11.data) < 1e-12] = 0.0
        S11.eliminate_zeros()

        t0 = time.time()
        self.P0 = vertex_preconditioner(self.uspace, S00)
        t1 = time.time()
        self.P1 = vertex_preconditioner(self.uspace, S11)
        t2 = time.time()
        #self.PA = vertex_preconditioner_all(self.sspace, self.uspace, self.B,
        #                                    spdiags(self.D_inv, 0, sgdof, sgdof))

        #Ps = vertex_preconditioner1(self.uspace, S)
        #self.PI_vertex, _, _ = Ps
        t3 = time.time()

        print("构造 P0 顶点预处理器时间：", t1 - t0, "秒")
        print("构造 P1 顶点预处理器时间：", t2 - t1, "秒")
        print("构造 S 顶点预处理器时间：", t3 - t2, "秒")

        self.S00 = S00
        self.S11 = S11
        ####################### Jacobi 平滑子 ########################## 
        def vertex_pre(v):
            u = bm.zeros_like(v)
            u[:ugdof] = self.P0(v[:ugdof])
            u[ugdof:] = self.P1(v[ugdof:])
            #u = self.PA(v)
            return u

        #PIA = bmat([[self.PI, None],
        #            [None, self.PI]], format='csr')
        #SS = PIA.T@self.S@PIA

        #flag = bm.abs(SS.data) < 1e-12
        #SS.data[flag] = 0.0
        #SS.eliminate_zeros()

        #ml = pyamg.smoothed_aggregation_solver(SS)

        isbddof = ~mesh.boundary_node_flag()
        J = bm.where(isbddof)[0]
        V = bm.ones_like(J, dtype=bm.float64)
        bd_dof = csr_matrix((V, (J, J)), shape=(self.ugdof0, self.ugdof0))

        I = eye(self.ugdof0, format='csr', dtype=bm.float64)

        S_co = self.PIT@S00@self.PI

        S_ = bd_dof@self.S_@(bd_dof.T)
        np.savetxt("S.csv", S_.toarray(), delimiter=",")
        np.savetxt("S00.csv", S_co.toarray(), delimiter=",")


        #S_bd = (I-bd_dof)@S_co@(I-bd_dof.T)
        #S_ += 2*spdiags(S_bd.diagonal(), 0, self.ugdof0, self.ugdof0) 
        #S_ += S_bd 

        flag = bm.abs(S_.data) < 1e-12
        S_.data[flag] = 0.0
        S_.eliminate_zeros()

        print("S_ 的非零元个数：", S_.nnz)

        #self.S_ = S_
        #S_ = self.S_

        ### TEST ##
        #@cartesian
        #def ux(p):
        #    x = p[..., 0]
        #    y = p[..., 1]
        #    return x**2
        #uh = self.uspace.L2Projector(ux)
        #plot_optimal_decomposition(self.uspace, S00, uh)

        self.ml = pyamg.smoothed_aggregation_solver(S_)
        def pre_of_S(r1):
            num = 2
            e1 = bm.zeros_like(r1, dtype=bm.float64)
            gauss_seidel(self.S, e1, r1, iterations=num, sweep='forward')
            r2 = r1 - self.S@e1 # 更新残差
            #e1 += 0.5*vertex_pre(r2)  # 应用光滑子 
            #r2 = r1 - self.S@e1 # 更新残差

            X = self.ml.solve(self.PIT@r2[:ugdof], maxiter=1, cycle='V')
            Y = self.ml.solve(self.PIT@r2[ugdof:], maxiter=1, cycle='V') 
            e1[:ugdof] += self.PI@X
            e1[ugdof:] += self.PI@Y

            #r2 = r1 - self.S@e1 # 更新残差
            #e1 += 0.5*vertex_pre(r2)  # 应用光滑子 

            gauss_seidel(self.S, e1, r1, iterations=num, sweep='backward')
            #e1 = mumps_solve(self.S, r1) # 最后一个精确解算器
            return e1 

        def pre_of_S0(r1):
            #t0 = time.time() 
            e1 = bm.zeros_like(r1, dtype=bm.float64)

            X = self.ml.solve(self.PIT@r1[:ugdof], maxiter=1, cycle='V')
            Y = self.ml.solve(self.PIT@r1[ugdof:], maxiter=1, cycle='V') 
            e1[:ugdof] = self.PI@X
            e1[ugdof:] = self.PI@Y
            #X = ml.solve(PIA.T@r1, maxiter=10, cycle='V')
            #e1 = PIA@X

            #t1 = time.time()
            e1 += 0.5*vertex_pre(r1)  # 应用光滑子 
            #t2 = time.time()
            #print("Schur 预处理子时间：", t2 - t1, "秒") 
            #print("Schur 平滑子时间：", t1 - t0, "秒")
            #e1 = mumps_solve(self.S, r1)
            return e1 

        def pre_of_S(r1):
            num = 2
            e1 = bm.zeros_like(r1, dtype=bm.float64)

            if not hasattr(self, "_S_cheb_ready") or not self._S_cheb_ready:
                d = _get_diag(self.S, bm)
                dmax = float(bm.max(bm.abs(d)))
                eps = max(1e-14 * dmax, 1e-30)
                d = bm.where(bm.abs(d) > eps, d, 1.0)

                self.S_dinv = 1.0 / d
                self.S_lmax = estimate_lambda_max_DinvA(self.S, self.S_dinv, bm, iters=10)
                self.S_cheb_lower_ratio = 0.1
                self._S_cheb_ready = True

            # pre smooth
            e1 = chebyshev_smooth(
                self.S, e1, r1,
                self.S_dinv, self.S_lmax, bm,
                degree=num,
                lower_ratio=self.S_cheb_lower_ratio
            )

            r2 = r1 - self.S @ e1

            X = self.ml.solve(self.PIT @ r2[:ugdof], maxiter=1, cycle='V')
            Y = self.ml.solve(self.PIT @ r2[ugdof:], maxiter=1, cycle='V')

            e1[:ugdof] += self.PI @ X
            e1[ugdof:] += self.PI @ Y

            # post smooth
            e1 = chebyshev_smooth(
                self.S, e1, r1,
                self.S_dinv, self.S_lmax, bm,
                degree=num,
                lower_ratio=self.S_cheb_lower_ratio
            )
            return e1

        self.pre_of_S = pre_of_S

        def SS0_inv_SS(v):
            v0 = S@v
            e = self.pre_of_S(v0)
            return e

        #from scipy.sparse.linalg import LinearOperator, eigsh
        #MM = LinearOperator(S.shape, matvec=SS0_inv_SS)
        #eigmax = eigsh(MM, k=1, which='LM', return_eigenvectors=False)
        #eigmin = eigsh(MM, k=1, which='SM', return_eigenvectors=False)
        #print("max eigenvalue of SS0^{-1}SS:", eigmax)
        #print("min eigenvalue of SS0^{-1}SS:", eigmin)
        #print("condition number:", eigmax/eigmin)
        #exit()

    def get_fast_solver_of_coarse_schur(self):
        space = self.uspace0

        bform = BilinearForm(space)
        bform.add_integrator(ScalarDiffusionIntegrator(q=3))
        S_ = bform.assembly()

        bc = DirichletBC(space, bm.zeros(S_.shape[0], dtype = S_.dtype))
        S_, _ = bc.apply(S_,bm.zeros(S_.shape[0], dtype = S_.dtype))
        S_    = S_.to_scipy()
        self.S_ = S_

        self.ml = pyamg.smoothed_aggregation_solver(S_)
        #self.ml = pyamg.ruge_stuben_solver(S_) # 这里要求必须有网格内部节点 
        #self.ml = MumpsSolver(S_) 

    def linear_operator(self, b):
        t0 = time.time()
        m = self.sgdof
        r = bm.zeros_like(b)
        r[:m] = self.M@b[:m]+self.B@b[m:]
        r[m:] = self.BT@b[:m]
        t1 = time.time()
        print("线性算子时间：", t1 - t0, "秒")
        return r

    def preconditioner_of_S(self, r1):
        e2 = self.pre_of_S(r1)
        return e2

    def preconditioner_of_S0(self, r1):
        e1 = mumps_solve(self.S, r1)
        return e1

    def precondieitoner(self,r):
        t0 = time.time()
        r = r.astype(bm.float64)
        sgdof = self.sgdof
        ugdof = self.ugdof

        r0 = r[:sgdof]
        r1 = r[sgdof:]

        e0 = self.D_inv*r0
        r1 -= self.BT@e0 

        e1 = bm.zeros_like(r1, dtype=bm.float64)
        e1 = self.preconditioner_of_S(r1)
        e0 += (self.B@e1)*self.D_inv
        t1 = time.time()
        print("GMRES 预处理时间：", t1 - t0, "秒")
        return bm.concatenate([e0, -e1])

    def preconditioner_minres(self,r):
        t0 = time.time()
        r = r.astype(bm.float64)
        sgdof = self.sgdof
        ugdof = self.ugdof

        r0 = r[:sgdof]
        r1 = r[sgdof:]

        e0 = self.D_inv*r0
        e1 = self.preconditioner_of_S(r1)
        t1 = time.time()
        print("minres 预处理时间：", t1 - t0, "秒")
        return bm.concatenate([e0, e1])
        
    def solve(self):
        sgdof = self.sgdof
        ugdof = self.ugdof
        gdof  = self.gdof 
        print("总自由度：", gdof)
        print("非零元个数：", (self.M!=0).sum() + (self.B!=0).sum() + (self.BT!=0).sum())
        F = self.F
        A = LinearOperator((gdof, gdof), matvec=self.linear_operator)
        P = LinearOperator((gdof, gdof), matvec=self.precondieitoner)
        #P = LinearOperator((gdof, gdof), matvec=self.preconditioner_minres)
        t0 = time.time() 
        x = gmres(A, F, M = P, rtol=1e-8, maxiter=100,
                  restart=20,
                  callback_type='pr_norm',
                  callback=self.count_iter) 

        #x = minres(A, F, M = P, rtol=1e-8, maxiter=100,
        #           callback = self.count_iter)
        t1 = time.time()
        r = F - A@x[0]

        print(f'iter count = {iter_count}, relative residual = {bm.linalg.norm(r)/bm.linalg.norm(F)}')
        print("GMRES 迭代时间：", t1 - t0, "秒")

        sigmah = self.sspace.function()
        sigmah[:] = x[0][:self.sgdof]

        uhx = self.uspace.function()
        uhy = self.uspace.function()
        uhx[:] = x[0][sgdof:sgdof+ugdof]
        uhy[:] = x[0][sgdof+ugdof:]
        return sigmah, uhx, uhy

    def count_iter(self, x):
        global iter_count
        iter_count += 1
        print(f'iter count = {iter_count}, residual = {bm.max(bm.abs(x))}')
        print("---------------------------------------------------------------------------------------")

    def mass_matrix(self):
        lambda0 = self.pde.lambda0
        lambda1 = self.pde.lambda1
        space = self.sspace
        p = space.p
        mesh = space.mesh
        gdof = space.number_of_global_dofs()

        cellmeasure = mesh.entity_measure('cell')
        qf = mesh.quadrature_formula(p+2, 'cell')

        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs)
        trphi = phi[..., 0] + phi[..., -1]
        Aphi = lambda0*phi
        Aphi[..., 0] += -lambda1*trphi
        Aphi[..., -1] += -lambda1*trphi
        Aphi[..., 1] *= 2

        num = bm.array([1, 2, 1], dtype=space.ftype)
        A  = bm.einsum('q, c, cqld, cqmd->clm', ws, cellmeasure, Aphi, phi)
        #A -= lambda1*bm.einsum('q, c, cql, cqm->clm', ws, cellmeasure, trphi, trphi)

        Dc = bm.einsum('q, c, cqld, cqld, d->cl', ws, cellmeasure, phi, phi, num)

        cell2dof = space.cell_to_dof()
        I = bm.broadcast_to(cell2dof[:, None], A.shape)
        J = bm.broadcast_to(cell2dof[..., None], A.shape)
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof), dtype=phi.dtype)

        D = bm.zeros(gdof, dtype=phi.dtype)
        bm.add.at(D, cell2dof, Dc)
        return A, D

    def mix_matrix(self):
        space0 = self.sspace
        space1 = self.uspace

        p = space0.p
        mesh = space0.mesh
        gdof0 = space0.number_of_global_dofs()
        gdof1 = space1.number_of_global_dofs()

        cell2dof0 = space0.cell_to_dof()
        cell2dof1 = space1.cell_to_dof()

        cellmeasure = mesh.entity_measure('cell') 
        qf = mesh.quadrature_formula(p+2, 'cell')

        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space0.div_basis(bcs)
        psi = space1.basis(bcs)
        B_ = bm.einsum('q, c, cqld, cqm->clmd', ws, cellmeasure, phi, psi)

        shape = B_.shape[:-1]

        B = coo_matrix((gdof0, gdof1*2), dtype=phi.dtype)
        I = bm.broadcast_to(cell2dof0[..., None], shape)
        for i in range(2):
            J = bm.broadcast_to(gdof1*i + cell2dof1[:, None], shape)
            B += coo_matrix((B_[..., i].flat, (I.flat, J.flat)), 
                            shape=(gdof0, gdof1*2), dtype=phi.dtype)
        return B.tocsr()

    def source_vector(self, space : LagrangeFESpace, f : callable):
        p = space.p
        mesh = space.mesh
        gdof = space.number_of_global_dofs()

        cellmeasure = mesh.entity_measure('cell')
        qf = mesh.quadrature_formula(p+2, 'cell')

        bcs, ws = qf.get_quadrature_points_and_weights()
        points = mesh.bc_to_point(bcs)

        phi  = space.basis(bcs)
        fval = f(points)
        b = bm.einsum('q, c, cql, cqd->cld', ws, cellmeasure, phi, fval)

        cell2dof = space.cell_to_dof()
        r = bm.zeros(gdof*2, dtype=phi.dtype)
        for i in range(2):
            bm.add.at(r, gdof*i + cell2dof, b[..., i])
        return r

    def displacement_boundary_condition(self, space, g : callable):
        p = space.p
        mesh = space.mesh
        TD = mesh.top_dimension()
        ldof = space.dof.number_of_local_dofs()
        gdof = space.dof.number_of_global_dofs()

        bdedge = mesh.boundary_edge_flag()
        e2c = mesh.edge_to_cell()[bdedge]
        en  = mesh.edge_unit_normal()[bdedge]
        cell2dof = space.cell_to_dof()[e2c[:, 0]]
        NBF = bdedge.sum()

        cellmeasure = mesh.entity_measure('edge')[bdedge]
        qf = mesh.quadrature_formula(p+2, 'edge')

        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(bcs)

        bcsi = [bm.insert(bcs, i, 0, axis=-1) for i in range(3)]

        symidx = [[0, 1], [1, 2]]
        phin = bm.zeros((NBF, NQ, ldof, 2), dtype=space.ftype)
        gval = bm.zeros((NBF, NQ, 2), dtype=space.ftype)
        for i in range(3):
            flag = e2c[:, 2] == i
            phi = space.basis(bcsi[i])[e2c[flag, 0]]
            phin[flag, ..., 0] = bm.sum(phi[..., symidx[0]] * en[flag, None, None], axis=-1)
            phin[flag, ..., 1] = bm.sum(phi[..., symidx[1]] * en[flag, None, None], axis=-1)
            points = mesh.bc_to_point(bcsi[i])[e2c[flag, 0]]
            gval[flag] = g(points)

        b = bm.einsum('q, c, cqld, cqd->cl', ws, cellmeasure, phin, gval)
        cell2dof = space.cell_to_dof()[e2c[:, 0]]
        r = bm.zeros(gdof, dtype=phi.dtype)
        bm.add.at(r, cell2dof, b)
        return r


