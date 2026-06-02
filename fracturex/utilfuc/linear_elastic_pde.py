"""线弹性制造解（manufactured solution）PDE 数据。

给定一个 sympy 位移场 ``u``，本模块按柔度（compliance）型本构推出解析应力 ``σ``、
对应的体力 ``f = -div σ`` 以及边界位移/面力，供 Hu-Zhang 混合元的收敛率验证使用。

应力按对称 Voigt 顺序存储：
  - 2D：``(σxx, σxy, σyy)``，最后一维长度 3；
  - 3D：``(σxx, σxy, σxz, σyy, σyz, σzz)``，最后一维长度 6。

本构采用柔度形式 ``ε = C^{-1} σ`` 的逆写法，材料参数 ``lambda0``/``lambda1`` 见各类
``__init__``（注意这里 ``lambda0, lambda1`` 是该柔度参数化下的系数，不是标准 Lamé λ、μ）。

类一览：
  - ``LinearElasticPDE`` / ``LinearElasticPDE3d``：2D/3D 完整制造解（由 ``u`` 符号求导得到）。
  - ``LinearElasticPDE0`` / ``LinearElasticPDE3d0``：占位/平凡算例（常体力、零解析场），用于冒烟测试。
"""
import numpy as np
from sympy import symbols, sin, cos, Matrix, lambdify
from sympy import derive_by_array, eye, tensorcontraction

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

class LinearElasticPDE0():
    """2D 平凡算例：常体力 f=(1,0)，解析应力/位移取零（占位，用于冒烟测试）。"""

    def __init__(self, u, lambda0, lambda1):
        """Args:
            u: 位移符号表达式（此平凡类不实际使用，仅保存）。
            lambda0, lambda1: 柔度本构参数（保存备用）。
        """
        x, y = symbols('x y')
        self.u = u
        self.lambda0 = lambda0
        self.lambda1 = lambda1

    def source(self, p):
        """体力 f。

        Args:
            p: 坐标 ``(..., 2)``。
        Returns:
            ``(..., 2)`` 体力，恒为 ``(1, 0)``。
        """
        x = p[..., 0]
        y = p[..., 1]
        f = bm.zeros(p.shape[:-1] + (2, ), dtype=bm.float64)
        f[..., 0] = 1
        f[..., 1] = 0
        return f

    def boundart_displacement(self, p):
        """边界位移（零 Dirichlet）。

        Args:
            p: 坐标 ``(..., 2)``。
        Returns:
            与 ``p`` 同形状的零数组。
        """
        return bm.zeros(p.shape, dtype=bm.float64)

    def stress(self, p):
        """解析应力（恒零）。

        Args:
            p: 坐标 ``(..., 2)``。
        Returns:
            ``(..., 3)`` Voigt 应力 ``(σxx, σxy, σyy)``，恒零。
        """
        x = p[..., 0]
        y = p[..., 1]
        sigma = bm.zeros(p.shape[:-1] + (3, ), dtype=bm.float64)
        return sigma

    def displacement(self, p):
        """解析位移（恒零）。

        Args:
            p: 坐标 ``(..., 2)``。
        Returns:
            ``(..., 2)`` 位移，恒零。
        """
        x = p[..., 0]
        y = p[..., 1]
        u = bm.zeros(p.shape[:-1] + (2, ), dtype=bm.float64)
        return u

class LinearElasticPDE3d0():
    """3D 平凡算例：常体力 f=(1,0,0)，解析应力/位移取零（占位，用于冒烟测试）。"""

    def __init__(self, u, lambda0, lambda1):
        """Args:
            u: 位移符号表达式（此平凡类不实际使用，仅保存）。
            lambda0, lambda1: 柔度本构参数（保存备用）。
        """
        x, y, z = symbols('x y z')
        self.u = u
        self.lambda0 = lambda0
        self.lambda1 = lambda1

    def source(self, p):
        """体力 f。

        Args:
            p: 坐标 ``(..., 3)``。
        Returns:
            ``(..., 3)`` 体力，恒为 ``(1, 0, 0)``。
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        f = bm.zeros(p.shape[:-1] + (3, ), dtype=bm.float64)
        f[..., 0] = 1
        f[..., 1] = 0
        f[..., 2] = 0
        return f

    def boundart_displacement(self, p):
        """边界位移（零 Dirichlet），返回与 ``p`` 同形状的零数组。"""
        return bm.zeros(p.shape, dtype=bm.float64)

    def stress(self, p):
        """解析应力（恒零），返回 ``(..., 6)`` Voigt 应力。"""
        sigma = bm.zeros(p.shape[:-1] + (6, ), dtype=bm.float64)
        return sigma

    def displacement(self, p):
        """解析位移（恒零），返回 ``(..., 3)``。"""
        u = bm.zeros(p.shape[:-1] + (3, ), dtype=bm.float64)
        return u


class LinearElasticPDE():
    """2D 线弹性制造解：由符号位移 ``u`` 推出解析应力与体力 ``f=-div σ``。"""

    def __init__(self, u, lambda0, lambda1):
        """构造制造解，预编译各解析场为 numpy 可调用函数。

        计算流程：应变 ``ε=sym(∇u)`` → 柔度型本构
        ``σ = (1/λ0)(c1·tr(ε)·I + ε)``，``c1=λ1/(λ0-2λ1)`` → 体力 ``f=-div σ``。

        Args:
            u: 长度 2 的 sympy 表达式列表 ``[u_x(x,y), u_y(x,y)]``。
            lambda0, lambda1: 柔度本构参数（非标准 Lamé λ、μ）。
        """
        x, y = symbols('x y')
        self.u = u
        self.lambda0 = lambda0
        self.lambda1 = lambda1

        grad_u = Matrix([[0, 0], [0, 0]])
        grad_u[0, 0] = u[0].diff(x)
        grad_u[0, 1] = u[0].diff(y)
        grad_u[1, 0] = u[1].diff(x)
        grad_u[1, 1] = u[1].diff(y)

        epsilon = (grad_u + grad_u.T) / 2     # 对称化操作

        trepsilon = tensorcontraction(epsilon, (0, 1))

        c0 = 1/lambda0
        c1 = lambda1/(lambda0 - 2*lambda1)
        sigma = c0*(c1*trepsilon*eye(2) + epsilon) 

        f = [-sigma[0, 0].diff(x) - sigma[0, 1].diff(y), 
             -sigma[1, 0].diff(x) - sigma[1, 1].diff(y)]

        self.sigmaxx = lambdify((x, y), sigma[0, 0], 'numpy')
        self.sigmayy = lambdify((x, y), sigma[1, 1], 'numpy')
        self.sigmaxy = lambdify((x, y), sigma[0, 1], 'numpy')

        self.fx = lambdify((x, y), f[0], 'numpy')
        self.fy = lambdify((x, y), f[1], 'numpy')

        self.ux = lambdify((x, y), u[0], 'numpy')
        self.uy = lambdify((x, y), u[1], 'numpy')

    def stress(self, p):
        """解析应力。

        Args:
            p: 坐标 ``(..., 2)``。
        Returns:
            ``(..., 3)`` Voigt 应力 ``(σxx, σxy, σyy)``。
        """
        x = p[..., 0]
        y = p[..., 1]
        sigma = bm.zeros(p.shape[:-1] + (3, ), dtype=bm.float64)
        sigma[..., 0] = self.sigmaxx(x, y)
        sigma[..., 1] = self.sigmaxy(x, y)
        sigma[..., 2] = self.sigmayy(x, y)
        return sigma

    def source(self, p):
        """体力 ``f = -div σ``。

        Args:
            p: 坐标 ``(..., 2)``。
        Returns:
            ``(..., 2)`` 体力 ``(f_x, f_y)``。
        """
        x = p[..., 0]
        y = p[..., 1]
        f = bm.zeros(p.shape[:-1] + (2, ), dtype=bm.float64)
        f[..., 0] = self.fx(x, y)
        f[..., 1] = self.fy(x, y)
        return f

    def displacement(self, p):
        """解析位移。

        Args:
            p: 坐标 ``(..., 2)``。
        Returns:
            ``(..., 2)`` 位移 ``(u_x, u_y)``。
        """
        x = p[..., 0]
        y = p[..., 1]
        u = bm.zeros(p.shape[:-1] + (2, ), dtype=bm.float64)
        u[..., 0] = self.ux(x, y)
        u[..., 1] = self.uy(x, y)
        return u

    def boundart_displacement(self, p):
        """边界位移，等于解析位移 :meth:`displacement`。"""
        return self.displacement(p)

    def boundart_stress(self, p, n):
        """边界面力（traction）``t = σ·n``。

        Args:
            p: 边界坐标 ``(..., 2)``。
            n: 外法向 ``(..., 2)``。
        Returns:
            ``(..., 2)`` 面力 ``(t_x, t_y)``。

        说明：按 ``t = σ·n`` 展开，``t_x = σxx·n_x + σxy·n_y``、
        ``t_y = σxy·n_x + σyy·n_y``（σ Voigt 存储为 [σxx, σxy, σyy]）。
        """
        sigma = self.stress(p)
        bs = bm.zeros(p.shape, dtype=bm.float64)
        bs[..., 0] = sigma[..., 0]*n[..., 0] + sigma[..., 1]*n[..., 1]
        bs[..., 1] = sigma[..., 1]*n[..., 0] + sigma[..., 2]*n[..., 1]
        return bs

class LinearElasticPDE3d():
    """3D 线弹性制造解：由符号位移 ``u`` 推出解析应力与体力 ``f=-div σ``。"""

    def __init__(self, u, lambda0, lambda1):
        """构造制造解，预编译各解析场为 numpy 可调用函数。

        计算流程与 2D 版一致，本构 ``c1=λ1/(λ0-3λ1)``（3D 体积项系数）。

        Args:
            u: 长度 3 的 sympy 表达式列表 ``[u_x, u_y, u_z](x,y,z)``。
            lambda0, lambda1: 柔度本构参数（非标准 Lamé λ、μ）。
        """
        x, y, z = symbols('x y z')
        self.u = u
        self.lambda0 = lambda0
        self.lambda1 = lambda1

        variables = [x, y, z]

        grad_u = Matrix([[u[i].diff(variables[j]) for j in range(3)] for i in range(3)])
        epsilon = (grad_u + grad_u.T) / 2     # 对称化操作

        trepsilon = tensorcontraction(epsilon, (0, 1))

        c0 = 1/lambda0
        c1 = lambda1/(lambda0 - 3*lambda1)
        sigma = c0*(c1*trepsilon*eye(3) + epsilon) 

        f = [-sigma[0, 0].diff(x) - sigma[0, 1].diff(y) - sigma[0, 2].diff(z),
             -sigma[1, 0].diff(x) - sigma[1, 1].diff(y) - sigma[1, 2].diff(z),
             -sigma[2, 0].diff(x) - sigma[2, 1].diff(y) - sigma[2, 2].diff(z)]

        self.sigmaxx = lambdify((x, y, z), sigma[0, 0], 'numpy')
        self.sigmaxy = lambdify((x, y, z), sigma[0, 1], 'numpy')
        self.sigmaxz = lambdify((x, y, z), sigma[0, 2], 'numpy')
        self.sigmayy = lambdify((x, y, z), sigma[1, 1], 'numpy')
        self.sigmayz = lambdify((x, y, z), sigma[1, 2], 'numpy')
        self.sigmazz = lambdify((x, y, z), sigma[2, 2], 'numpy')

        self.fx = lambdify((x, y, z), f[0], 'numpy')
        self.fy = lambdify((x, y, z), f[1], 'numpy')
        self.fz = lambdify((x, y, z), f[2], 'numpy')

        self.ux = lambdify((x, y, z), u[0], 'numpy')
        self.uy = lambdify((x, y, z), u[1], 'numpy')
        self.uz = lambdify((x, y, z), u[2], 'numpy')

    def stress(self, p):
        """解析应力。

        Args:
            p: 坐标 ``(..., 3)``。
        Returns:
            ``(..., 6)`` Voigt 应力 ``(σxx, σxy, σxz, σyy, σyz, σzz)``。
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        sigma = bm.zeros(p.shape[:-1] + (6, ), dtype=bm.float64)
        sigma[..., 0] = self.sigmaxx(x, y, z)
        sigma[..., 1] = self.sigmaxy(x, y, z)
        sigma[..., 2] = self.sigmaxz(x, y, z)
        sigma[..., 3] = self.sigmayy(x, y, z)
        sigma[..., 4] = self.sigmayz(x, y, z)
        sigma[..., 5] = self.sigmazz(x, y, z)
        return sigma

    def source(self, p):
        """体力 ``f = -div σ``。

        Args:
            p: 坐标 ``(..., 3)``。
        Returns:
            ``(..., 3)`` 体力 ``(f_x, f_y, f_z)``。
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        f = bm.zeros(p.shape[:-1] + (3, ), dtype=bm.float64)
        f[..., 0] = self.fx(x, y, z)
        f[..., 1] = self.fy(x, y, z)
        f[..., 2] = self.fz(x, y, z)
        return f

    def displacement(self, p):
        """解析位移。

        Args:
            p: 坐标 ``(..., 3)``。
        Returns:
            ``(..., 3)`` 位移 ``(u_x, u_y, u_z)``。
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        u = bm.zeros(p.shape[:-1] + (3, ), dtype=bm.float64)
        u[..., 0] = self.ux(x, y, z)
        u[..., 1] = self.uy(x, y, z)
        u[..., 2] = self.uz(x, y, z)
        return u

    def boundart_displacement(self, p):
        """边界位移，等于解析位移 :meth:`displacement`。"""
        return self.displacement(p)

    def boundart_stress(self, p, n):
        """边界面力（traction）``t = σ·n``。

        借助对称索引 ``symidx`` 把 Voigt 应力还原为对称张量行再与法向点乘。

        Args:
            p: 边界坐标 ``(..., 3)``。
            n: 外法向 ``(..., 3)``。
        Returns:
            ``(..., 3)`` 面力 ``(t_x, t_y, t_z)``。
        """
        # symidx[i] 给出 σ 第 i 行 (σ_i0, σ_i1, σ_i2) 在 Voigt 存储中的下标
        symidx = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
        sigma = self.stress(p)
        bs = bm.zeros(p.shape, dtype=bm.float64)
        bs[..., 0] = bm.sum(sigma[..., symidx[0]]*n, axis=-1) 
        bs[..., 1] = bm.sum(sigma[..., symidx[1]]*n, axis=-1)
        bs[..., 2] = bm.sum(sigma[..., symidx[2]]*n, axis=-1)
        return bs


if __name__ == '__main__':
    x, y = symbols('x y')
    u0 = sin(x)
    u1 = cos(y)

    u = [u0, u1]
    pde = LinearElasticPDE(u, 1, 1)


    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=10, ny=10)
    points = mesh.entity('node')
    cell = mesh.entity('cell')

    a = pde.displacement(points)
    print(a)






















