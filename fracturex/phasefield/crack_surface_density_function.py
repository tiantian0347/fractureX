"""裂纹面密度函数 ``h(d)`` 及其一/二阶导（AT1、AT2、用户自定义）。

约定：每个求值方法返回 ``(value, c_d)`` 元组，其中 ``c_d`` 是该模型的归一化常数
（AT2 为 2，AT1 为 8/3），供相场能量泛函的尺度归一化使用。
"""
from fealpy.backend import backend_manager as bm


class CrackSurfaceDensityFunction:
    """裂纹面密度函数族的统一接口，按 ``density_type`` 分派到 AT1/AT2/用户自定义实现。"""

    def __init__(self, density_type='AT2', **kwargs):
        """Args:
            density_type: ``'AT2'`` | ``'AT1'`` | ``'user_defined'``。
            **kwargs: 用户自定义模型的回调（``density_func`` 等），存入 ``self.params``。
        """
        self.density_type = density_type
        self.params = kwargs


    def density_function(self, d):
        """裂纹面密度 ``h(d)``。输入损伤 ``d``，返回 ``(h(d), c_d)``。"""
        if self.density_type == 'AT2':
            return self._AT2_density(d)
        elif self.density_type == 'AT1':
            return self._AT1_density(d)
        elif self.density_type == 'user_defined':
            return self._user_defined_density(d)
        else:
            raise ValueError(f"Unknown density type: {self.density_type}")

    def grad_density_function(self, d):
        """裂纹面密度一阶导 ``h'(d)``。输入 ``d``，返回 ``(h'(d), c_d)``。"""
        if self.density_type == 'AT2':
            return self._AT2_grad_density(d)
        elif self.density_type == 'AT1':
            return self._AT1_grad_density(d)
        elif self.density_type == 'user_defined':
            return self._user_defined_grad_density(d)
        else:
            raise ValueError(f"Unknown density type: {self.density_type}")

    def grad_grad_density_function(self, d):
        """裂纹面密度二阶导 ``h''(d)``。输入 ``d``，返回 ``(h''(d), c_d)``。"""
        if self.density_type == 'AT2':
            return self._AT2_grad_grad_density(d)
        elif self.density_type == 'AT1':
            return self._AT1_grad_grad_density(d)
        elif self.density_type == 'user_defined':
            return self._user_defined_grad_grad_density(d)
        else:
            raise ValueError(f"Unknown density type: {self.density_type}")
        
    def _AT2_density(self, d):
        """
        The AT2 crack surface density function h(d) = d^2, c_d=2.
        """
        return d**2, 2
    
    def _AT2_grad_density(self, d):
        """
        The derivative of the AT2 crack surface density function h'(d) = 2d, c_d=2.
        """
        return 2*d, 2
    
    def _AT2_grad_grad_density(self, d):
        """
        The second derivative of the AT2 crack surface density function h''(d) = 2.
        """
        return 2, 2
    
    def _AT1_density(self, d):
        """
        The AT1 crack surface density function g(d) = d.
        """
        return d, 8/3
    
    def _AT1_grad_density(self, d):
        """
        The derivative of the AT1 crack surface density function g'(d) = 1.
        """
        return 1, 8/3
    
    def _AT1_grad_grad_density(self, d):
        """
        The second derivative of the AT1 crack surface density function g''(d) = 0.
        """
        return 0, 8/3
    
    # User-defined model implementations
    def _user_defined_density(self, d):
        """
        The user-defined crack surface density function h(d).
        """
        if 'density_func' in self.params:
            return self.params['density_func'](d)
        raise NotImplementedError("User-defined density function is not provided.")

    def _user_defined_grad_density(self, d):
        """
        The first derivative of the user-defined crack surface density function h'(d).
        """
        if 'grad_density_func' in self.params:
            return self.params['grad_density_func'](d)
        raise NotImplementedError("User-defined first derivative function is not provided.")

    def _user_defined_grad_grad_density(self, d):
        """
        The second derivative of the user-defined crack surface density function h''(d).
        """
        if 'grad_grad_density_func' in self.params:
            return self.params['grad_grad_density_func'](d)
        raise NotImplementedError("User-defined second derivative function is not provided.")
    