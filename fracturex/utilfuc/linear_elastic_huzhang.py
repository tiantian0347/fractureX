"""Hu-Zhang 快速求解器的线弹性收敛性验证脚本（2D/3D 制造解，命令行运行）。

注：依赖 ``smoother``、``huzhang_fast_solver`` 等包外模块，作为手动验证脚本留存，
非 fracturex 包的可导入模块。
"""

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh, TetrahedronMesh

from smoother import JacobiPreconditioner, GaussSeidelPreconditioner
from huzhang_fast_solver import HZFEMFastSolve 
from huzhang_fast_solver3d import HZFEMFastSolve3d
from linear_elastic_pde import LinearElasticPDE, LinearElasticPDE3d, LinearElasticPDE0
from linear_elastic_pde import LinearElasticPDE3d0

from sympy import symbols, sin, cos, exp
import sys

def test_2d():
    """2D 制造解收敛测试：命令行参数 ``n p lam``，打印位移/应力的 L2 误差。"""
    n = int(sys.argv[1])
    p = int(sys.argv[2])
    nu = 0.48
    lam = 2.0*nu/(1-2.0*nu)
    lam = float(sys.argv[3])
    mu  = 0.5

    lambda0 = 1/(2*mu)
    lambda1 = lam/(2*mu*(2*lam + 2*mu))
    print('lambda0 = ', lambda0)
    print('lambda1 = ', lambda1)

    #lambda0, lambda1 = 1, 1.0/4.0
    k = bm.pi/20

    x, y = symbols('x y')
    pi = bm.pi
    u0 = (sin(pi*x)*sin(pi*y))
    u1 = (sin(pi*x)*sin(pi*y))
    u0 = sin(5*k*x)*sin(7*y)*exp(x-y)
    u1 = sin(5*k*x)*sin(4*y)*exp(x+y)

    u = [u0, u1]
    #pde = LinearElasticPDE(u, lambda0, lambda1)
    pde = LinearElasticPDE0(u, lambda0, lambda1)

    ux = lambda p : pde.displacement(p)[..., 0]
    uy = lambda p : pde.displacement(p)[..., 1]

    mesh = TriangleMesh.from_box([0, 1, 0, 1], n, n)
    #nodes = bm.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]],
    #                 dtype=bm.float64)
    #cells = bm.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=bm.int32)
    #mesh = TriangleMesh(nodes, cells)
    #mesh.uniform_bisect(n)
    #mesh = TriangleMesh.from_polygon_gmsh([[0, 0], [1, 0], [1, 1], [0, 1]], 1/n)

    node = bm.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=bm.float64)
    cell = bm.array([[0, 1, 2]], dtype=bm.int32)
    #mesh = TriangleMesh(node, cell)
    
    solver = HZFEMFastSolve(pde, mesh, p)

    sigmah, uhx, uhy = solver.solve()

    e0 = mesh.error(uhx, ux)
    e1 = mesh.error(uhy, uy)
    e2 = mesh.error(sigmah, pde.stress)

    print('L2 error of u0:', e0)
    print('L2 error of u1:', e1)
    print('L2 error of stress:', e2)



def test_3d():
    """3D 制造解收敛测试：命令行参数 ``n``，打印位移/应力的 L2 误差。"""
    n = int(sys.argv[1])

    lambda0, lambda1 = 1.0, 1.0/4.0
    lam = 100
    mu  = 0.5

    lambda0 = 1/(2*mu)
    lambda1 = lam/(2*mu*(3*lam + 2*mu))

    x, y, z = symbols('x y z')
    pi = bm.pi
    u0 = (sin(pi*x)*sin(pi*y)*sin(pi*z))
    u1 = (sin(pi*x)*sin(pi*y)*sin(pi*z))
    u2 = (sin(pi*x)*sin(pi*y)*sin(pi*z))
    #u0 = sin(5*x)*sin(7*y)*sin(6*z)
    #u1 = cos(5*x)*cos(4*y)*cos(3*z)
    #u2 = sin(3*x)*cos(4*y)*sin(5*z)

    u = [u0, u1, u2]
    pde = LinearElasticPDE3d(u, lambda0, lambda1)
    pde = LinearElasticPDE3d0(u, lambda0, lambda1)

    ux = lambda p : pde.displacement(p)[..., 0]
    uy = lambda p : pde.displacement(p)[..., 1]
    uz = lambda p : pde.displacement(p)[..., 2]

    mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], n, n, n)
    solver = HZFEMFastSolve3d(pde, mesh, 4)
    print("asdasdasd")

    sigmah, uhx, uhy, uhz = solver.solve()

    e0 = mesh.error(uhx, ux)
    e1 = mesh.error(uhy, uy)
    e2 = mesh.error(uhz, uz)
    e3 = mesh.error(sigmah, pde.stress)

    print('L2 error of u0:', e0)
    print('L2 error of u1:', e1)
    print('L2 error of u2:', e2)
    print('L2 error of stress:', e3)

if __name__ == '__main__':
    #test_2d()
    test_3d()
















