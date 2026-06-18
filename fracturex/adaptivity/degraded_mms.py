"""退化 MMS（manufactured solution）：带空间变化裂纹带 d(x) 的退化弹性。

T6 门槛专用（THEORY §5 第 3 层）：需要**空间变化**的冻结 d(x)，
常数 d 下局部对比度 κ≡1、Θ→1 与 k_res 无关，测不到核心命题。

退化柔度/刚度（平面应变，Lamé λ,μ；退化 g(d)=(1-d)²+k_res）：
  σ = g(d) C ε(u),   C ε = λ tr(ε) I + 2μ ε
  f = -div σ = -div(g(d) C ε(u))   （含 ∇g 项，sympy 自动求导）

用 sympy 造解，lambdify 到 numpy。记号:
  分量序 Voigt (xx, xy, yy)，与 HuZhangFESpace2d / equilibrated_estimator 一致。
"""
from __future__ import annotations

from sympy import symbols, sin, cos, exp, Matrix, eye, lambdify, sqrt, tanh

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian


class DegradedElasticMMS:
    """带裂纹带 d(x) 的退化弹性制造解。

    输入:
      u_expr   : [u0(x,y), u1(x,y)] sympy 位移表达式
      d_expr   : d(x,y) sympy 相场表达式 ∈[0,1]（裂纹带）
      lam, mu  : Lamé 参数
      k_res    : 残余刚度 (>0)
    属性方法:
      displacement(p) / stress(p) / source(p) / damage(p) : (...,2)/(...,3)/(...,2)/(...,)
      grad_u(p) : (...,2,2) 位移梯度（estimator 算 ε(u) 用）
    """

    def __init__(self, u_expr, d_expr, *, lam: float, mu: float, k_res: float):
        x, y = symbols('x y')
        self.lam, self.mu, self.k_res = lam, mu, k_res

        g = (1 - d_expr) ** 2 + k_res                      # 退化函数 g(d)

        gu = Matrix([[u_expr[0].diff(x), u_expr[0].diff(y)],
                     [u_expr[1].diff(x), u_expr[1].diff(y)]])
        eps = (gu + gu.T) / 2
        tr = eps[0, 0] + eps[1, 1]
        Ceps = lam * tr * eye(2) + 2 * mu * eps            # 未退化 C ε
        sigma = g * Ceps                                   # 退化应力 σ = g C ε

        f = [-(sigma[0, 0].diff(x) + sigma[0, 1].diff(y)),
             -(sigma[1, 0].diff(x) + sigma[1, 1].diff(y))]

        self._ux = lambdify((x, y), u_expr[0], 'numpy')
        self._uy = lambdify((x, y), u_expr[1], 'numpy')
        self._gu = [[lambdify((x, y), gu[i, j], 'numpy') for j in range(2)]
                    for i in range(2)]
        self._sxx = lambdify((x, y), sigma[0, 0], 'numpy')
        self._sxy = lambdify((x, y), sigma[0, 1], 'numpy')
        self._syy = lambdify((x, y), sigma[1, 1], 'numpy')
        self._fx = lambdify((x, y), f[0], 'numpy')
        self._fy = lambdify((x, y), f[1], 'numpy')
        self._d = lambdify((x, y), d_expr, 'numpy')

    def displacement(self, p):
        x, y = p[..., 0], p[..., 1]
        return bm.stack([bm.array(self._ux(x, y) * bm.ones_like(x)),
                         bm.array(self._uy(x, y) * bm.ones_like(x))], axis=-1)

    @cartesian
    def source(self, p):
        x, y = p[..., 0], p[..., 1]
        return bm.stack([bm.array(self._fx(x, y) * bm.ones_like(x)),
                         bm.array(self._fy(x, y) * bm.ones_like(x))], axis=-1)

    def stress(self, p):
        x, y = p[..., 0], p[..., 1]
        return bm.stack([bm.array(self._sxx(x, y) * bm.ones_like(x)),
                         bm.array(self._sxy(x, y) * bm.ones_like(x)),
                         bm.array(self._syy(x, y) * bm.ones_like(x))], axis=-1)

    def damage(self, p):
        x, y = p[..., 0], p[..., 1]
        return bm.array(self._d(x, y) * bm.ones_like(x))

    def grad_u(self, p):
        x, y = p[..., 0], p[..., 1]
        g00 = bm.array(self._gu[0][0](x, y) * bm.ones_like(x))
        g01 = bm.array(self._gu[0][1](x, y) * bm.ones_like(x))
        g10 = bm.array(self._gu[1][0](x, y) * bm.ones_like(x))
        g11 = bm.array(self._gu[1][1](x, y) * bm.ones_like(x))
        row0 = bm.stack([g00, g01], axis=-1)
        row1 = bm.stack([g10, g11], axis=-1)
        return bm.stack([row0, row1], axis=-2)


def crack_band_d(x, y, *, x0=0.5, width=0.1, dmax=0.95):
    """sympy 裂纹带 d(x,y)：沿 x=x0 的竖直带，高斯廓形（用于造解）。

    d = dmax · exp(-((x-x0)/width)²)，∈[0,dmax]，光滑（κ patch O(1)）。
    返回 sympy 表达式。
    """
    return dmax * exp(-((x - x0) / width) ** 2)
