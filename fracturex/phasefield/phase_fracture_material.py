"""相场断裂本构模型（基于位移的相场框架）。

定义损伤耦合的弹性本构家族：有效应力、退化应力/切线刚度、正负能量分裂与不可逆历史场。
各向同性（Isotropic）、谱分解（Spectral）、混合（Hybrid）模型已实现；各向异性
（Anisotropic）、偏量（Deviatoric）为占位 stub。``PhaseFractureMaterialFactory`` 按名字
创建对应模型。

约定：``bc`` 为重心坐标；应变/应力张量形状一般为 ``(NC, NQ, GD, GD)``，能量密度为
``(NC, NQ)``。注意本模块属基于位移的相场实现，与 Hu-Zhang 混合元侧的
``fracturex.damage.phasefield_damage`` 各自独立。
"""
from typing import Optional

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.material.elastic_material import LinearElasticMaterial

from fealpy.decorator import barycentric

from fracturex.utilfuc.utils import flatten_symmetric_matrices

class BasedPhaseFractureMaterial(LinearElasticMaterial):
    """相场断裂本构基类：持有材料参数、退化函数及当前位移/损伤/历史场状态。

    子类需实现 ``stress_value``、``elastic_matrix`` 等给出具体的退化应力与切线刚度。
    """

    def __init__(self, material, energy_degradation_fun):
        """
        Parameters
        ----------
        material : dict
            材料参数，需含 ``{'lam','mu'}`` 或 ``{'E','nu'}``。
        energy_degradation_fun :
            能量退化函数对象 ``g(d)``（提供 ``degradation_function`` 等）。
        """
        self._gd = energy_degradation_fun # 能量退化函数
        if 'lam' in material and 'mu' in material:
            self.lam = material['lam']
            self.mu = material['mu']
        elif 'E' in material and 'nu' in material:
            self.E = material['E']
            self.nu = material['nu']

            self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
            self.mu = self.E / (2 * (1 + self.nu))
        else:
            raise ValueError("The material parameters are not correct.")


        self.uh = None
        self.d = None

        self.H = None # 谱分解模型下的最大历史场

    def update_disp(self, uh):
        """更新当前位移场 ``self.uh``（输入有限元位移函数 ``uh``）。"""
        self.uh = uh

    def update_phase(self, d):
        """更新当前损伤场 ``self.d``（输入损伤有限元函数 ``d``）。"""
        self.d = d

    def update_historical_field(self, H):
        """更新最大历史场 ``self.H``（输入历史场张量 ``H``）。"""
        self.H = H

    @ barycentric
    def effective_stress(self, bc) -> TensorLike:
        """
        Compute the effective stress tensor, which is the stress tensor without the damage effect.

        Parameters
        ----------
        u : TensorLike
            The displacement field.
        strain : TensorLike 
            The strain tensor.
        Returns
        -------
        TensorLike
            The effective stress tensor.
        """
        strain = self.strain_value(bc)

        lam = self.lam
        mu = self.mu
        trace_e = bm.einsum('...ii', strain)
        I = bm.eye(strain.shape[-1])
        stress = lam * trace_e[..., None, None] * I + 2 * mu * strain
        
        return stress

    @ barycentric
    def strain_value(self,bc=None) -> TensorLike:
        """
        Compute the strain tensor.
        """ 
    
        uh = self.uh
        guh = uh.grad_value(bc)
        
        strain = 0.5 * (guh + bm.swapaxes(guh, -2, -1))
        return strain
    
    @ barycentric
    def linear_elastic_matrix(self, bc=None) -> TensorLike:
        """
        Compute the linear elastic matrix.
        """
        strain = self.strain_value(bc)

        GD = strain.shape[-1]
       
        lam = self.lam
        mu = self.mu
        if GD == 2:
            D0 = bm.tensor([[lam + 2 * mu, lam, 0],
                          [lam, lam + 2 * mu, 0],
                          [0, 0, mu]], dtype=bm.float64)
        elif GD == 3:
            D0 = bm.tensor([[lam + 2 * mu, lam, lam, 0, 0, 0],
                            [lam, lam + 2 * mu, lam, 0, 0, 0],
                            [lam, lam, lam + 2 * mu, 0, 0, 0],
                            [0, 0, 0, mu, 0, 0],
                            [0, 0, 0, 0, mu, 0],
                            [0, 0, 0, 0, 0, mu]], dtype=bm.float64)
        else:
            raise NotImplementedError("This dim is not correct, we cannot give the linear elastic matrix.")
        return D0


class IsotropicModel(BasedPhaseFractureMaterial):
    """各向同性退化模型：应力与切线刚度整体乘以退化因子 ``g(d)``，不做拉压分裂。"""

    @ barycentric
    def stress_value(self, bc=None) -> TensorLike:
        """
        Compute the fracture stress tensor.
        """
        d = self.d

        gd = self._gd.degradation_function(d(bc)) # 能量退化函数 (NC, NQ)
        stress = self.effective_stress(bc=bc) * gd[..., None, None]
        return stress

    @ barycentric
    def elastic_matrix(self, bc) -> TensorLike: 
        """
        Compute the tangent matrix.
        """
        d = self.d
        gd = self._gd.degradation_function(d(bc)) # 能量退化函数 (NC, NQ)
        D0 = self.linear_elastic_matrix(bc=bc) # 线弹性矩阵 
        D = D0 * gd[..., None, None]
        return D
    
    def positive_stress_func(self, guh) -> TensorLike:
        """
        @brief Compute the stress tensor from the grad displacement tensor.
        ----------
        guh : TensorLike
            The grad displacement tensor.
        Returns
        -------
        TensorLike
            The flattened stress tensor.
        """
        GD = guh.shape[-1]
        lam = self.lam
        mu = self.mu
        strain = 0.5 * (guh + bm.swapaxes(guh, -2, -1))
        
        trace_e = bm.einsum('...ii', strain)
        
        I = bm.eye(GD)
        stress = lam * trace_e[..., None, None] * I + 2 * mu * strain
        flat_stress = flatten_symmetric_matrices(stress)
        return flat_stress
    
    def negative_stress_func(self, guh) -> TensorLike:
        """
        @brief Compute the stress tensor from the grad displacement tensor.
        ----------
        guh : TensorLike
            The grad displacement tensor.
        Returns
        -------
        TensorLike
            The flattened stress tensor.
        """
        GD = guh.shape[-1]
        stress = bm.zeros(guh.shape, dtype=bm.float64)
        flat_stress = flatten_symmetric_matrices(stress)
        return flat_stress
    
class AnisotropicModel(BasedPhaseFractureMaterial):
    """各向异性退化模型（占位，未实现）。"""

    def stress_value(self, bc) -> TensorLike:
        """各向异性模型应力（未实现）。输入重心坐标 ``bc``。"""
        # 计算各向异性模型下的应力
        pass

    def elastic_matrix(self, bc) -> TensorLike:
        """各向异性模型切线刚度（未实现）。输入重心坐标 ``bc``。"""
        # 计算各向异性模型下的切线刚度矩阵
        pass

    def positive_stress_func(self, guh) -> TensorLike:
        """
        @brief Compute the stress tensor from the grad displacement tensor.
        ----------
        guh : TensorLike
            The grad displacement tensor.
        Returns
        -------
        TensorLike
            The flattened stress tensor.
        """
        pass
    
    def negative_stress_func(self, guh) -> TensorLike:
        """
        @brief Compute the stress tensor from the grad displacement tensor.
        ----------
        guh : TensorLike
            The grad displacement tensor.
        Returns
        -------
        TensorLike
            The flattened stress tensor.
        """
        pass

class DeviatoricModel(BasedPhaseFractureMaterial):
    """偏量/体积分裂退化模型（占位，未实现）。"""

    def stress_value(self, bc) -> TensorLike:
        """偏量模型应力（未实现）。输入重心坐标 ``bc``。"""
        # 计算偏应力模型下的应力
        pass

    def elastic_matrix(self, bc) -> TensorLike:
        """偏量模型切线刚度（未实现）。输入重心坐标 ``bc``。"""
        # 计算偏应力模型下的切线刚度矩阵
        pass

    def positive_stress_func(self, guh) -> TensorLike:
        """
        @brief Compute the stress tensor from the grad displacement tensor.
        ----------
        guh : TensorLike
            The grad displacement tensor.
        Returns
        -------
        TensorLike
            The flattened stress tensor.
        """
        pass

    def negative_stress_func(self, guh) -> TensorLike:
        """
        @brief Compute the stress tensor from the grad displacement tensor.
        ----------
        guh : TensorLike
            The grad displacement tensor.
        Returns
        -------
        TensorLike
            The flattened stress tensor.
        """
        pass
        

class SpectralModel(BasedPhaseFractureMaterial):
    """谱分解（Miehe）模型：基于应变正负部分裂计算 ``ψ⁺/ψ⁻`` 与不可逆历史场。

    ``stress_value`` / ``elastic_matrix`` 暂未实现；本模型主要供 Hybrid 提供历史场驱动力
    （:meth:`maximum_historical_field`）。
    """

    def stress_value(self, bc) -> TensorLike:
        """谱分解模型应力（未实现）。输入重心坐标 ``bc``。"""
        # 计算谱分解模型下的应力
        pass

    def elastic_matrix(self, bc) -> TensorLike:
        """谱分解模型切线刚度（未实现）。输入重心坐标 ``bc``。"""
        # 计算谱分解模型下的切线刚度矩阵
        pass

    def strain_energy_density_decomposition(self, s: TensorLike):
        """
        @brief Strain energy density decomposition from Miehe Spectral
        decomposition method.
        @param[in] s strain，（NC, NQ, GD, GD）
        """

        lam = self.lam
        mu = self.mu

        # 应变正负分解
        sp, sm = self.strain_pm_eig_decomposition(s)
        
        #ts = bm.trace(s, axis1=-2, axis2=-1)
        ts = bm.einsum('...ii', s)

        tp, tm = self.macaulay_operation(ts)
        #tsp = bm.trace(sp**2, axis1=-2, axis2=-1)
        #tsm = bm.trace(sm**2, axis1=-2, axis2=-1)
        tsp = bm.einsum('...ii', sp**2)
        tsm = bm.einsum('...ii', sm**2)

        phi_p = lam * tp ** 2 / 2.0 + mu * tsp
        phi_m = lam * tm ** 2 / 2.0 + mu * tsm
        return phi_p, phi_m

    def strain_pm_eig_decomposition(self, s: TensorLike):
        """
        @brief Decomposition of Positive and Negative Characteristics of Strain.
        varespilon_{\\pm} = \\sum_{a=0}^{GD-1} <varespilon_a>_{\\pm} n_a \\otimes n_a
        varespilon_a is the a-th eigenvalue of strain tensor.
        n_a is the a-th eigenvector of strain tensor.
        
        @param[in] s strain，（NC, NQ, GD, GD）
        """
        '''
        if bm.device_type(s) == 'cuda':
            torch.cuda.empty_cache()
            try:
                w, v = bm.linalg.eigh(s)  # w 特征值, v 特征向量
            except torch.cuda.OutOfMemoryError as e:
                print("CUDA out of memory. Attempting to free cache.")
                torch.cuda.empty_cache()
        else:
            w, v = bm.linalg.eigh(s) # w 特征值, v 特征向量
        '''
        w, v = bm.linalg.eigh(s)
        p, m = self.macaulay_operation(w)

        sp = bm.zeros_like(s)
        sm = bm.zeros_like(s)
        
        GD = s.shape[-1]
        for i in range(GD):
            n0 = v[..., i]  # (NC, NQ, GD)
            n1 = p[..., i, None] * n0  # (NC, NQ, GD)
            sp += n1[..., None] * n0[..., None, :]

            n1 = m[..., i, None] * n0
            sm += n1[..., None] * n0[..., None, :]
        return sp, sm

    
    def macaulay_operation(self, alpha):
        """
        @brief Macaulay operation
        """
        val = bm.abs(alpha)
        p = (alpha + val) / 2.0
        m = (alpha - val) / 2.0
        return p, m

    def heaviside(self, x):
        """
        @brief
        """
        val = bm.zeros_like(x)
        val[x > 1e-13] = 1
        val[bm.abs(x) < 1e-13] = 0.5
        val[x < -1e-13] = 0
        return val
    
    def linear_strain_value(self, bc):
        """
        Compute the linear strain tensor.
        """
        bc = bm.array([[1/3, 1/3, 1/3]], dtype=bm.float64)
        uh = self.uh
        guh = uh.grad_value(bc)
        strain = 0.5 * (guh + bm.swapaxes(guh, -2, -1))
        return strain
    
    @ barycentric
    def maximum_historical_field(self, bc):
        """
        Irreversible driver for the tensile strain-energy density.

        Updates ``self.H`` by a pointwise maximum of the positive-mode elastic
        energy density :math:`\\psi^+` (from spectral strain decomposition) against
        the previous field. This is evaluated during phase-field matrix assembly
        (quadrature sweeps); quasi-static "time" is advanced by the outer load loop
        in :class:`fracturex.phasefield.main_solve.MainSolve`, not by a separate ODE
        for ``H``. AT1/AT2 only affect the crack surface density through
        ``CrackSurfaceDensityFunction``, not :math:`\\psi^+` here.
        """
        strain = self.strain_value(bc)
       
        phip, _ = self.strain_energy_density_decomposition(strain)
        
        if self.H is None:
            self.H = phip[:]
        else:
            self.H = bm.maximum(self.H, phip)
        return self.H
    
    def positive_stress_func(self, guh) -> TensorLike:
        """
        @brief Compute the stress tensor from the grad displacement tensor.
        ----------
        guh : TensorLike
            The grad displacement tensor.
        Returns
        -------
        TensorLike
            The flattened stress tensor.
        """
        pass
    
    def negative_stress_func(self, guh) -> TensorLike:
        """
        @brief Compute the stress tensor from the grad displacement tensor.
        ----------
        guh : TensorLike
            The grad displacement tensor.
        Returns
        -------
        TensorLike
            The flattened stress tensor.
        """
        pass     

class HybridModel(BasedPhaseFractureMaterial):
    """混合模型：应力/刚度用各向同性退化，历史场驱动力用谱分解 ``ψ⁺``。

    内部组合 ``IsotropicModel``（应力/刚度）与 ``SpectralModel``（历史场），兼顾计算效率与
    拉压不对称的不可逆性。
    """

    def __init__(self, material, energy_degradation_fun):
        """
        Parameters
        ----------
        material : dict
            材料参数（``{'lam','mu'}`` 或 ``{'E','nu'}``）。
        energy_degradation_fun :
            能量退化函数对象。
        """

        self._isotropic_model = IsotropicModel(material, energy_degradation_fun)
        self._spectral_model = SpectralModel(material, energy_degradation_fun)
        super().__init__(material, energy_degradation_fun)

    @ barycentric
    def stress_value(self, bc) -> TensorLike:
        """
        Compute the fracture stress tensor.
        """
        self._isotropic_model.uh = self.uh
        self._isotropic_model.d = self.d
        return self._isotropic_model.stress_value(bc=bc)

    @ barycentric
    def elastic_matrix(self, bc) -> TensorLike:
        """退化切线刚度矩阵，委托各向同性模型计算。输入重心坐标 ``bc``，返回 ``D``。"""
        self._isotropic_model.uh = self.uh
        self._isotropic_model.d = self.d
        return self._isotropic_model.elastic_matrix(bc=bc)
    
    def positive_stress_func(self, guh) -> TensorLike:
        """
        @brief Compute the stress tensor from the grad displacement tensor.
        ----------
        guh : TensorLike
            The grad displacement tensor.
        Returns
        -------
        TensorLike
            The flattened stress tensor.
        """
        return self._isotropic_model.positive_stress_func(guh)
    
    def negative_stress_func(self, guh) -> TensorLike:
        """
        @brief Compute the stress tensor from the grad displacement tensor.
        ----------
        guh : TensorLike
            The grad displacement tensor.
        Returns
        -------
        TensorLike
            The flattened stress tensor.
        """
        return self._isotropic_model.negative_stress_func(guh)

    @ barycentric
    def maximum_historical_field(self, bc):
        """
        Delegate irreversible :math:`\\psi^+` history to the spectral model; see
        :meth:`SpectralModel.maximum_historical_field` for the update rule.
        """
        self._spectral_model.uh = self.uh
        self._spectral_model.d = self.d
        self._spectral_model.H = self.H
        
        self.H = self._spectral_model.maximum_historical_field(bc)
        return self.H
        

class PhaseFractureMaterialFactory:
    """
    工厂类，用于创建不同的本构模型
    """
    @staticmethod
    def create(model_type, material, energy_degradation_fun):
        """
        Parameters
        ----------
        model_type : str
            本构模型类型
        material : dict
        """
        if model_type == 'IsotropicModel':
            return IsotropicModel(material, energy_degradation_fun)
        elif model_type == 'AnisotropicModel':
            return AnisotropicModel(material, energy_degradation_fun)
        elif model_type == 'SpectralModel':
            return SpectralModel(material, energy_degradation_fun) 
        elif model_type == 'DeviatoricModel':
            return DeviatoricModel(material, energy_degradation_fun)
        elif model_type == 'HybridModel':
            return HybridModel(material, energy_degradation_fun)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
