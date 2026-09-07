"""能量退化函数 ``g(d)`` 及其一/二阶导（quadratic、thrice、用户自定义）。

退化函数把损伤 ``d∈[0,1]`` 映射到刚度折减系数 ``g(d)``（``d=0`` 完好 → ``g≈1``，
``d=1`` 完全破坏 → ``g≈0``）。quadratic 默认保留历史 additive floor；显式选择
``floor_mode='convex'`` 可与 Hu--Zhang 的残余刚度约定完全一致。
"""


class EnergyDegradationFunction:
    """能量退化函数族的统一接口，按 ``degradation_type`` 分派到对应实现。"""

    def __init__(self, degradation_type='quadratic', **kwargs):
        """Initialize an energy degradation law.

        Parameters
        ----------
        degradation_type : str
            ``quadratic``, ``thrice`` or ``user_defined``.
        residual_stiffness : float, optional
            Dimensionless positive floor ``k``. Default ``1e-10`` preserves
            the historical standard-FEM quadratic law.
        floor_mode : {``additive``, ``convex``}, optional
            ``additive`` gives ``(1-d)^2+k``; ``convex`` gives
            ``(1-k)(1-d)^2+k`` and therefore exactly ``g(0)=1``.
        **kwargs
            User-defined degradation callbacks and their parameters.

        Raises
        ------
        ValueError
            If the residual stiffness or floor mode is invalid.
        """
        self.degradation_type = degradation_type
        self.params = kwargs
        self.residual_stiffness = float(kwargs.get('residual_stiffness', 1.0e-10))
        self.floor_mode = str(kwargs.get('floor_mode', 'additive'))
        if not 0.0 < self.residual_stiffness <= 1.0:
            raise ValueError("residual_stiffness must lie in (0, 1]")
        if self.floor_mode not in {'additive', 'convex'}:
            raise ValueError("floor_mode must be 'additive' or 'convex'")

    def degradation_function(self, d):
        """
        Calculate the energy degradation factor g (d) based on the phase field value d.

        Parameters:
        d (float or numpy array): Phase field value, with a value range of [0,1].

        return:
        g (float or numpy array): Degradation factor g (d).
        """
        if self.degradation_type == 'quadratic':
            return self._quadratic_degradation(d)
        elif self.degradation_type == 'thrice':
            return self._thrice_degradation(d)
        elif self.degradation_type == 'user_defined':
            return self._user_defined_degradation(d)
        else:
            raise ValueError(f"Unknown degradation type: {self.degradation_type}")

    def grad_degradation_function(self, d):
        """退化函数一阶导 ``g'(d)``。输入相场值 ``d``，返回 ``g'(d)``。"""
        if self.degradation_type == 'quadratic':
            return self._quadratic_grad_degradation(d)
        elif self.degradation_type == 'user_defined':
            return self._user_defined_grad_degradation(d)
        else:
            raise ValueError(f"Unknown degradation type: {self.degradation_type}")

    def grad_grad_degradation_function(self, d):
        """退化函数二阶导 ``g''(d)``。输入相场值 ``d``，返回 ``g''(d)``。"""
        if self.degradation_type == 'quadratic':
            return self._quadratice_grad_grad_degradation(d)
        elif self.degradation_type == 'user_defined':
            return self._user_defined_grad_grad_degradation(d)
        else:
            raise ValueError(f"Unknown degradation type: {self.degradation_type}")
        
    def grad_degradation_function_constant_coef(self):
        """
        Get the constant coefficient in the gradient of the energy degradation function.

        Parameters:
        d (float or numpy array): phase field value.

        return:
        c (float or numpy array): The constant coefficient in the gradient of the energy degradation function.
        """
        if self.degradation_type == 'quadratic':
            factor = 1.0 - self.residual_stiffness if self.floor_mode == 'convex' else 1.0
            return -2 * factor
        elif self.degradation_type == 'user_defined':
            return self.params.get('constant_coef')
        else:
            raise ValueError(f"Unknown degradation type: {self.degradation_type}")


    def _quadratic_degradation(self, d):
        """
        The quadratic energy degradation function g (d)=(1-d) ^ 2.

        Parameters:
        d (float or numpy array): phase field value.

        return:
        g (float or numpy array): Degradation factor g (d).
        """
        k = self.residual_stiffness
        if self.floor_mode == 'convex':
            return (1.0 - k) * (1 - d)**2 + k
        return (1 - d)**2 + k
    
    def _quadratic_grad_degradation(self, d):
        """
        The derivative of the quadratic energy degradation function g'(d) = -2(1 - d)。
        """
        factor = 1.0 - self.residual_stiffness if self.floor_mode == 'convex' else 1.0
        return factor * (-2 + 2*d)
    
    def _quadratice_grad_grad_degradation(self, d):
        """
        The second derivative of the quadratic energy degradation function g''(d) = 2。
        """
        factor = 1.0 - self.residual_stiffness if self.floor_mode == 'convex' else 1.0
        return 2 * factor

    def _thrice_degradation(self, d):
        """
        The thrice degradation function g(d) = 3(1-d)^2-2(1-d)^3。

        Parameters:
        d (float or numpy array): phase field value.

        return:
        g (float or numpy array): Degradation factor g (d).
        """
        eps = 1e-10
        gd = 3*(1 - d)**2 - 2*(1-d)**3 + eps
        return gd

    def _user_defined_degradation(self, d):
        """
        User defined energy degradation function, which can be in any user-defined form.

        Parameters:
        d (float or numpy array): phase field value.

        return:
        g (float or numpy array): Degradation factor g (d).
        """
        custom_function = self.params.get('custom_function')
        if custom_function is None:
            raise ValueError("For user_defined degradation, 'custom_function' must be provided.")
        return custom_function(d)
    
    def _user_defined_grad_degradation(self, d):
        """
        The derivative of the user-defined energy degradation function.
        """
        custom_grad_function = self.params.get('custom_grad_function')
        if custom_grad_function is None:
            raise ValueError("For user_defined degradation, 'custom_grad_function' must be provided.")
        return custom_grad_function(d)
    
    def _user_defined_grad_grad_degradation(self, d):
        """
        The second derivative of the user-defined energy degradation function.
        """
        custom_grad_grad_function = self.params.get('custom_grad_grad_function')
        if custom_grad_grad_function is None:
            raise ValueError("For user_defined degradation, 'custom_grad_grad_function' must be provided.")
        return custom_grad_grad_function(d)

    def plot_degradation_function(self, d_values):
        """
        绘制能量退化函数的曲线。

        参数:
        d_values (numpy array): 一组相场值 d 用于绘制函数曲线。
        """
        import matplotlib.pyplot as plt

        g_values = self.degradation_function(d_values)
        plt.plot(d_values, g_values, label=f'{self.degradation_type} degradation')
        plt.xlabel('Phase field variable d')
        plt.ylabel('Degradation function g(d)')
        plt.title('Energy Degradation Function')
        plt.legend()
        plt.grid(True)
        plt.show()
