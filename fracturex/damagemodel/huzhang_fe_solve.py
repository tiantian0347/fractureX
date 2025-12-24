from fealpy.backend import backend_manager as bm
from fealpy.utils import timer

from fealpy.functionspace import HuZhangFESpace2d, HuZhangFESpace3d
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.fem.huzhang_stress_integrator import HuZhangStressIntegrator
#from fealpy.fem.huzhang_displacement_integrator import HuZhangDisplacementIntegrator
from fealpy.fem.huzhang_mix_integrator import HuZhangMixIntegrator
from fealpy.fem import BilinearForm, LinearForm
from fealpy.fem import VectorSourceIntegrator
from fracturex.damagemodel.huzhang_boundary_condition import HuzhangBoundaryCondition, HuzhangBoundaryCondition, HuzhangStressBoundaryCondition, build_isNedge_from_isD

from fealpy.typing import TensorLike, CoefLike, Threshold

from fealpy.decorator import cartesian, barycentric

from fealpy.fem import BlockForm
from fealpy.solver import cg, spsolve, gmres
from scipy.sparse.linalg import lgmres
from scipy.sparse import bmat

class HuZhangFESolve():
    def __init__(self, model, mesh, p:int, q:int=None, use_relaxation:bool=True, isNedge:TensorLike=None):
        """
        @brief 

        @param[in] model 算例模型
        @param[in] mesh 连续体离散网格
        @param[in] p 有限元空间次数
        @param[in] q 积分次数
        @param[in] use_relaxation 是否使用松弛
        @param[in] isNedge 自由边界标记
        """
        self. model = model
        self.mesh = mesh
        self.p = p
        self.q = q if q else p + 3
        self.use_relaxation = use_relaxation
        self.isNedge = isNedge

        GD = mesh.geo_dimension()
        if GD == 2:
            self.hspace = HuZhangFESpace2d(mesh, p=p, use_relaxation=use_relaxation, bd_stress=isNedge)
        elif GD == 3:
            raise NotImplementedError("3D Hu-Zhang FE space is not implemented yet.")
            #self.hspace = HuZhangFESpace3d(mesh, p=p, use_relaxation=True, bd_stress=isNedge)
        else:
            raise ValueError("Unsupported mesh dimension: {}".format(mesh.dim))
        self.lspace = LagrangeFESpace(mesh, p=p)
        self.tspace = TensorFunctionSpace(self.lspace, shape=(-1, GD))

        self.uh = self.tspace.function()
        self.sigma = self.hspace.function()
        
        self.d = self.lspace.function()
        self.get_model_parameters()
        self.damage_function()


        self.tmr = timer()
        next(self.tmr)

    def get_model_parameters(self):
        """
        @brief 获取模型参数
        @details 该函数用于设置模型的参数，例如材料属性等。
        """
        model = self.model
        self.Gc = model.Gc
        self.l0 = model.l0
        self.E = model.E
        self.nv = model.nv
        self.ft = model.ft
        self.lam = model.lam
        self.mu = model.mu

    def iteration_solve(self, disp):
        """
        @brief 迭代求解
        """
        max_iter = 50
        tol = 1e-6

        tmr = self.tmr
        

        for iter in range(max_iter):
            sigma_old = self.sigma.copy()
            uh_old = self.uh.copy()
            d_old = self.d.copy()

            tmr.send('start')

            self.solve(disp)

            tmr.send('one_iteration_end')

            norm_sigma = bm.linalg.norm(self.sigma - sigma_old)
            norm_uh = bm.linalg.norm(self.uh - uh_old)
            norm_d = bm.linalg.norm(self.d - d_old)

            tmr.send(None)

            print(f"Iteration {iter+1}: ||sigma - sigma_old|| = {norm_sigma:.6e}, "
                  f"||uh - uh_old|| = {norm_uh:.6e}, ||d - d_old|| = {norm_d:.6e}")

            #if norm_sigma < tol and norm_uh < tol and norm_d < tol:
            if norm_d < tol:
                print("Convergence achieved.")
                break
        else:
            print("Maximum iterations reached without convergence.")

    def solve(self, disp):
        """
        @brief 求解
        """
        tmr = self.tmr
        mesh = self.mesh
        hspace = self.hspace
        lspace = self.lspace
        tspace = self.tspace
        print('disp', disp)
        
        uh = self.uh
        sigma = self.sigma
        d = self.d
        
        GD = mesh.geo_dimension()
        gdof0 = hspace.number_of_global_dofs()

        tmr.send('one_solve_start')

        @barycentric
        def coef_func(bcs, index=None):
            d_coef = (1-d(bcs)+1e-15)
            return 1/d_coef

        c0 = 1/(2 * self.mu)
        c1 = self.lam / (2 * self.mu)*(GD * self.lam+ 2*self.mu)

        bform1 = BilinearForm(hspace)
        bform1.add_integrator(HuZhangStressIntegrator(coef=coef_func,
                                                      lambda0=c0, lambda1=c1))

        bform2 = BilinearForm((tspace, hspace))
        bform2.add_integrator(HuZhangMixIntegrator())

        if self.use_relaxation:
            M = bform1.assembly().to_scipy().tocsr()
            B = bform2.assembly().to_scipy().tocsr()
            TM = hspace.TM.to_scipy().tocsr()

            M2 = TM.T @ M @ TM
            B2 = TM.T @ B
            A = bmat([[M2,  B2],
                  [B2.T, None]], format="csr")
        else:
            A = BlockForm([[bform1,bform2],
                            [bform2.T,None]])
            A = A.assembly()

        tmr.send('matrix_assembly')

        # lform1 = LinearForm(space1)
        # @cartesian
        # def source(x, index=None):
        #     return self.model.source(x)
        # lform1.add_integrator(VectorSourceIntegrator(source=source))

        HBC = HuzhangBoundaryCondition(space=hspace, q=self.q)
        b = HBC.displacement_boundary_condition(value=disp, threshold=self.model.is_disp_boundary, direction='y')
        
        
        F = bm.zeros(A.shape[0], dtype=A.dtype)

        F[:gdof0] = TM.T @ b
        #F[gdof0:] = -a

        tmr.send('disp_boundary_assembly')
        HSBC = HuzhangStressBoundaryCondition(space=hspace)
        A, F = HSBC.apply_essential_bc_to_system(
            A, F,
            gd=0.0,
            threshold=self.isNedge,    # 直接用 mask 最稳
            coord="auto",
            sigma_offset=0,
            sigma_gdof=gdof0
        )


        tmr.send('stress_boundary_assembly')
        
        _A = A.to_scipy()
        _R = bm.to_numpy(F)

        x, info = lgmres(_A, _R, atol=1e-12)
        _x = bm.tensor(x)
        #_x,  = lgmres(A, F)
        sigma = _x[:gdof0]
        uh[:] = _x[gdof0:]
        tmr.send('linear_solve')

        self.sigma[:] = sigma
        self.uh[:] = uh

        self.damage_function()
        print('uh',bm.sum(uh[:]))
        print('d', bm.sum(self.d[:]))

    def damage_function(self):
        """
        @brief 计算损伤函数
        @details 该函数计算材料的损伤函数，基于应力和材料参数。
        @note 损伤函数的值在 [0, 1] 范围内，表示材料的损伤程度。
        """
        stress = self.sigma
        Hd = self.moderation_parameter()
        r = self.damage_threshold(stress)
        d = 1 - bm.exp(-2*Hd*(r-self.ft)/self.ft)
        self.d[:] = d

    def effective_stress(self, stress):
        """
        @brief 计算有效应力
        """
        d = self.d
        stress_eff = (1-d)*stress
        return stress_eff

    def max_diagonal_element(self, stress):
        """
        计算展开后的对称矩阵的对角线元素的最大值。

        参数:
            flattened_matrix: 展开后的对称矩阵数组，形状为 (3,) 或 (6,)。

        返回:
            对角线元素的最大值。
        """
        # 根据数组长度确定对角线元素的索引
        GD = self.mesh.geo_dimension()
        if GD == 2:
            diag_indices = [0, -1]  # 2x2 矩阵的对角线索引
        elif GD == 3:
            diag_indices = [0, -3, -1]  # 3x3 矩阵的对角线索引
        else:
            raise ValueError("Invalid stress tensor dimension.")

        # 提取对角线元素并计算最大值
        return bm.max(stress[..., diag_indices])

    def equivalent_stress(self, stress):
        """
        @brief 计算等效应力
        """
#        stress_eff = self.effective_stress(stress)
        d = self.d
        stress_eq = (1-d)*self.max_diagonal_element(stress)
        return stress_eq

    def damage_threshold(self, stress):
        """
        @brief 计算损伤阈值
        """
        stress_eq = self.equivalent_stress(stress)
        try:
            r = bm.maximum.reduce(self.ft, stress_eq, r)
        except NameError:
            r = bm.maximum(self.ft, stress_eq[:])
        return r

    def moderation_parameter(self):
        """
        @brief 调和参数
        """
        lch = self.characteristic_length()
        Hd = self.l0 / (2*lch + self.l0)
        return Hd
        

    def characteristic_length(self):
        """
        @brief 特征长度
        """
        lch = self.Gc*self.E / (self.ft**2)
        return lch