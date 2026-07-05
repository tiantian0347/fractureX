"""fealpy_old_2 里的 SquareWithCircleHoleDomain 在 fealpy v3 里没有，
这里按最小接口 (符号距离 + fh) 复刻一份，只服务于 IP-FEM 的 model0。"""
from __future__ import annotations

import numpy as np

from fealpy.geometry.domain import Domain
from fealpy.geometry.signed_distance_function import dcircle, drectangle, ddiff
from fealpy.geometry.sizing_function import huniform


class SquareWithCircleHoleDomain(Domain):
    """[0,1]² 减去 (0.5, 0.5, r=0.2) 圆的 2D 域，signed-distance 表达。"""

    def __init__(self, hmin: float = 0.1, hmax: float | None = None, fh=None):
        super().__init__(hmin=hmin, hmax=hmax if hmax is not None else hmin,
                         GD=2, fh=fh if fh is not None else huniform)
        self.box = [0.0, 1.0, 0.0, 1.0]
        self._vertices = np.array(
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            dtype=np.float64,
        )
        self._fd_circle = lambda p: dcircle(p, [0.5, 0.5], 0.2)
        self._fd_square = lambda p: drectangle(p, [0.0, 1.0, 0.0, 1.0])
        self._fd = lambda p: ddiff(self._fd_square(p), self._fd_circle(p))
        self.facets = {0: self._vertices, 1: self._fd}

    def __call__(self, p):
        return self._fd(p)

    def signed_dist_function(self, p):
        return self._fd(p)

    def sizing_function(self, p):
        return self.fh(p, self)

    def facet(self, dim):
        return self.facets[dim]

    def meshing_facet_0d(self):
        return self.facets[0]

    def meshing_facet_1d(self, hmin, fh=None):
        return None
