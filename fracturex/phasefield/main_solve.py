"""相场断裂准静态交错（staggered）主求解驱动。

``MainSolve`` 按载荷步推进规定位移边界，每步用 Newton-Raphson 交错求解位移块与相场块，
直至残差指标收敛；支持自适应加密、自动微分装配、多种稀疏线性求解后端与 VTK 输出。
类级 docstring 详述历史场不可逆性、载荷步进与线性代数选项。
"""
from typing import Callable, Optional, Dict
import os
import numpy as np
from fealpy.utils import timer

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.decorator import barycentric
from fealpy.fem import BilinearForm, LinearForm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import LinearElasticIntegrator, ScalarDiffusionIntegrator, ScalarMassIntegrator, ScalarSourceIntegrator
from fealpy.fem import ScalarNeumannBCIntegrator, VectorNeumannBCIntegrator

# 自动微分模块
from fealpy.fem import NonlinearForm
from fealpy.fem import ScalarNonlinearMassIntegrator, ScalarNonlinearDiffusionIntegrator
from fealpy.fem import NonlinearElasticIntegrator

# 边界处理模块
from fealpy.fem import DirichletBC

from fealpy.solver import gmres as fealpy_gmres
from scipy.sparse.linalg import cg as scipy_sparse_cg
from scipy.sparse.linalg import gmres as scipy_sparse_gmres
from scipy.sparse.linalg import lgmres, spsolve as scipy_sparse_spsolve

from fracturex.phasefield.energy_degradation_function import EnergyDegradationFunction as EDFunc
from fracturex.phasefield.crack_surface_density_function import CrackSurfaceDensityFunction as CSDFunc
from fracturex.phasefield.phase_fracture_material import PhaseFractureMaterialFactory
from fracturex.adaptivity.adaptive_refinement import AdaptiveRefinement
from fracturex.phasefield.vector_Dirichlet_bc import VectorDirichletBC
from fracturex.utilfuc.vtk_lagrange_writer import (
    sample_fields_for_lagrange_triangle,
    sample_fields_tensor_product_fe,
    write_lagrange_triangle_vtu,
    write_subdivided_hexahedron_mesh_vtu,
    write_subdivided_quadrilateral_mesh_vtu,
)
from fracturex.utilfuc.sparse_direct_backends import solve_direct_mumps, solve_direct_pardiso


class MainSolve:
    """
    Staggered quasi-static phase-field driver (displacement block, then phase block).

    **Historical strain energy density** ``H`` (irreversibility)
        Within each Newton step the phase equation assembles coefficients that call
        ``PhaseFractureMaterial.maximum_historical_field(bc)``. That routine updates
        the stored field with a pointwise maximum of the tensile strain-energy density
        driver :math:`\\psi^+` (``SpectralModel`` / ``HybridModel``) against the previous
        iterate, which is the usual Miehe-type irreversibility for spectral models.
        There is **no separate explicit time integrator** for ``H``: quasi-static
        "time" is the outer load sequence in :meth:`solve`, and ``H`` advances whenever
        the phase subsystem is assembled. AT1/AT2 enter only through
        ``CrackSurfaceDensityFunction`` in the phase diffusion/mass coefficients, not
        in the definition of :math:`\\psi^+`.

    **Load stepping**
        :meth:`solve` walks the entries of the prescribed ``'force'`` boundary values
        (see :meth:`add_boundary_condition`). Those values are applied as **Dirichlet
        displacements** on the driving boundary (reaction is reported, not dead-load
        control). There is **no built-in arc-length or force-controlled continuation**;
        extend :meth:`solve` or wrap this class if you need snap-back tracking.

    **Neumann / natural boundary data**
        Optional ``'Neumann'`` entries add contributions to the residual before Dirichlet
        rows are eliminated; see :meth:`add_boundary_condition`.

        **Linear algebra**
        Sparse solves use :meth:`set_scipy_solver` / :meth:`set_cupy_solver` for the
        backend and :meth:`set_linear_solver_options` for method (``auto`` chooses
        SciPy ``spsolve`` below ``direct_max_dof``, otherwise GMRES), tolerances, and
        iteration caps. Optional **direct** packages: ``method='pardiso'`` (``pypardiso``)
        and ``method='mumps'`` (``python-mumps`` / ``import mumps``). Pass
        ``linear_solver_options`` into :meth:`solve` for one-shot overrides.
    """

    def __init__(self, mesh, material_params: Dict,
                 model_type: str = 'HybridModel'):
        """
        Initialize the MainSolver class with more customization options.

        Parameters
        ----------
        mesh : object
            The mesh for the problem.
        material_params : dict
            Dictionary containing material properties: 'lam', 'mu', 'E', 'nu', 'Gc', 'l0'.
        p : int, optional
            Polynomial order for the function space, by default 1.
        q : int, optional
            Quadrature degree, by default p + 3.
        model_type : str, optional
            Stress decomposition model, by default 'HybridModel'.
        method : str, optional
            The method for solving the problem, by default 'lfem'.
        """
        self.mesh = mesh

        self.model_type = model_type

        self.material_params = material_params

        # Material parameters
        self.Gc = material_params['Gc']
        self.l0 = material_params['l0']
        
        self.bc_dict = {}

        self.CSDFunc = None
        self.EDFunc = None

        self.enable_refinement = False

        self._save_vtk = False
        self._atype = None
        self._timer = False

        self._solver = None

        # Sparse linear solve (inside each Newton linearization)
        self._lin_method = 'auto'  # 'auto' | 'direct' | 'gmres' | 'lgmres' | 'cg'
        self._lin_rtol = 1e-8
        self._lin_atol = 1e-12
        self._lin_maxiter = None  # default: min(2000, max(100, 5*n))
        self._lin_restart = 30
        self._direct_max_dof = 8000
        self._lin_use_ilu = False

        # Initialize the timer
        self.tmr = timer()
        next(self.tmr)
    
    def initialize_settings(self, p: int = 1, q: int = None, ):
        """
        Initialize the settings for the problem.
        """
        # Material and energy degradation function
        if self.EDFunc is None:
            self.set_energy_degradation(degradation_type='quadratic')
        if self.CSDFunc is None:
            self.set_crack_surface_density(density_type='AT2')

        self.pfcm = PhaseFractureMaterialFactory.create(self.model_type, self.material_params, self.EDFunc)

        # Initialize spaces
        if self._method == 'lfem':
            self.set_lfe_space(p=p, q=q)
        else:
            raise ValueError(f"Unknown method: {self._method}")
        
        self._scalar_gdof = int(self.space.number_of_global_dofs())
        self._tensor_gdof = int(self.tspace.number_of_global_dofs())
        if self._scalar_gdof <= 0 or self._tensor_gdof <= 0:
            raise RuntimeError("Invalid global DOF count after building FE spaces.")

        self.pfcm.update_disp(self.uh)
        self.pfcm.update_phase(self.d)

        # solver backend (scipy sparse vs fealpy/cupy gmres)
        if self._solver is None:
            if bm.device_type(self.uh) == 'cpu':
                print('Using scipy solver.')
                self._solver = 'scipy'
            elif bm.device_type(self.uh) == 'cuda':
                print('Using cupy solver.')
                self._solver = 'cupy'
            else:
                print('Using scipy solver.')
                self._solver = 'scipy'
        elif self._solver == 'cupy' and bm.device_type(self.uh) == 'cpu':
            print(
                "[MainSolve] cupy linear solver was selected but FE tensors are on CPU; "
                "falling back to scipy. Use CUDA tensors for cupy assembly/solve."
            )
            self._solver = 'scipy'

    def solve(
        self,
        method: str = 'lfem',
        p: int = 1,
        q: int = None,
        maxit: int = 50,
        *,
        linear_solver_options: Optional[Dict] = None,
    ):
        """
        Solve the phase-field fracture problem.

        Outer ``for`` loop advances the **prescribed displacement** samples stored on
        the ``'force'`` boundary (see :meth:`add_boundary_condition`): each entry is a
        new target displacement magnitude for that boundary. Within each sample,
        :meth:`newton_raphson` performs a displacement-then-phase split until the
        scaled residual indicators drop below ``1e-5``. This is **not** an arc-length
        or force-controlled Riks implementation; softening branches that need those
        schemes must be added externally.

        Parameters
        ----------
        maxit : int, optional
            Maximum number of iterations, by default 30.
        atype : str, optional
            Type of the solver, by default None. if 'auto', using automatic differentiation to assemble matrix. 
        vtkname : str, optional
            VTK output file name, by default None.
        linear_solver_options : dict, optional
            If given, forwarded once to :meth:`set_linear_solver_options` before the
            load-step loop (e.g. ``{'method': 'gmres', 'rtol': 1e-6}``).
        """
        if linear_solver_options:
            self.set_linear_solver_options(**linear_solver_options)
        self._method = method
        self.initialize_settings(p=p, q=q)
        self._initialize_force_boundary()
        self._Rforce = bm.zeros_like(self._force_value)
        
        #for i in range(2):
        for i in range(len(self._force_value)-1):
            print('i', i)
            self._currt_force_value = self._force_value[i+1]

            # Run Newton-Raphson iteration
            self.newton_raphson(maxit)
            
            if self._save_vtk:
                if self._vtkfname is None:
                    fname = f'test{i:010d}.vtu'
                else:
                    fname = f'{self._vtkfname}{i:010d}.vtu'
                self._save_vtkfile(fname=fname)
            
            bm.set_at(self._Rforce, i+1, self._Rfu)
            

    def newton_raphson(self, maxit: int = 50):
        """
        Perform the Newton-Raphson iteration for solving the problem.

        Parameters
        ----------
        maxit : int, optional
            Maximum number of iterations, by default 30.
        force_value : TensorLike
            Value of the force boundary condition.
        """
        tmr = self.tmr
        for k in range(maxit):
            print(f"Newton-Raphson Iteration {k + 1}/{maxit}:")
            
            tmr.send('start')

            # Solve the displacement field
            if self._method == 'lfem':
                if self._atype == 'auto':
                    er0 = self.solve_displacement_auto()
                else:
                    er0 = self.solve_displacement()
            else:
                raise ValueError(f"Unknown method: {self._method}")

            # Solve the phase field
            if self._method == 'lfem':
                if self._atype == 'auto':
                    print(f"Using automatic differentiation to assemble phase field matrix.")
                    er1 = self.solve_phase_field_auto()
                else:
                    er1 = self.solve_phase_field()
            else:
                raise ValueError(f"Unknown method: {self._method}")

            # Adaptive refinement
            if self.enable_refinement:
                data = self.set_interpolation_data()
                self.mesh, new_data = self.adaptive.perform_refinement(self.mesh, self.d, data, self.l0)
                if new_data:
                    if self._method == 'lfem':
                        self.set_lfe_space()
                    else:
                        raise ValueError(f"Unknown method: {self._method}")
                    self.update_interpolation_data(new_data)
                    print(f"Refinement after iteration {k + 1}")

                tmr.send('refine')

            # Check for convergence
            if k == 0:
                e0, e1 = er0, er1
            
            error = max(er0/e0, er1/e1)
            if self._timer:
                tmr.send(None)

            print(f"Displacement error after iteration {k + 1}: {er0/e0}")
            print(f"Phase field error after iteration {k + 1}: {er1/e1}")
            print(f'Iteration {k+1}, Error: {error}')
            if error < 1e-5:
                print(f"Convergence achieved after {k + 1} iterations.")
                break

    def solve_displacement(self) -> float:
        """
        Solve the displacement field and return the residual norm.

        Returns
        -------
        float
            The norm of the residual.
        """
        uh = self.uh
        tmr = self.tmr
        tmr.send('disp_start')

        fbc = VectorDirichletBC(self.tspace, self._currt_force_value, self._force_dof, direction=self._force_direction)
        uh, force_index = fbc.apply_value(uh)
        self.pfcm.update_disp(uh)

        ubform = BilinearForm(self.tspace)
        ubform.add_integrator(LinearElasticIntegrator(self.pfcm, q=self.q, method='voigt'))
        A = ubform.assembly()
        
        R = -A @ uh[:]
        self._Rfu = bm.sum(-R[force_index])
        tmr.send('disp_assemble')

        # Apply force boundary conditions
        ubc = VectorDirichletBC(self.tspace, 0, threshold=self._force_dof, direction=self._force_direction)
        A, R = ubc.apply(A, R)
        R = self._append_neumann_to_residual(R, 'displacement')

        # Apply displacement boundary conditions
        A, R = self._apply_boundary_conditions(A, R, field='displacement')
        tmr.send('apply_bc')
        
        du = self.solver(A, R)
        uh += du[:]
        self.uh = uh
        
        self.pfcm.update_disp(uh)
        tmr.send('disp_solver')
        return bm.linalg.norm(R)

    def solve_phase_field(self) -> float:
        """
        Solve the phase field and return the residual norm.

        Returns
        -------
        float
            The norm of the residual.
        """
        Gc, l0, d = self.Gc, self.l0, self.d

        @barycentric
        def diff_coef(bc, index):
            gg_hd, c_d = self.CSDFunc.grad_grad_density_function(d(bc))
            return Gc * l0 * 2 / c_d

        @barycentric
        def mass_coef1(bc, index):
            gg_hd, c_d = self.CSDFunc.grad_grad_density_function(d(bc))
            return gg_hd * Gc / (l0 * c_d)
        
        
        @barycentric
        def mass_coef2(bc, index):
            gg_gd = self.EDFunc.grad_grad_degradation_function(d(bc))
            self.H = self.pfcm.maximum_historical_field(bc)
            return gg_gd * self.H
        
        @barycentric
        def source_coef(bc, index):
            gc_gd = self.EDFunc.grad_degradation_function_constant_coef()
            return -1 * gc_gd * self.H

        tmr = self.tmr
        tmr.send('phase_start')

        dbform = BilinearForm(self.space)
        dbform.add_integrator(ScalarDiffusionIntegrator(coef=diff_coef, q=self.q), ScalarMassIntegrator(coef=mass_coef1, q=self.q), ScalarMassIntegrator(coef=mass_coef2, q=self.q))
        A = dbform.assembly()
        tmr.send('phase_matrix_assemble')

        dlform = LinearForm(self.space)
        dlform.add_integrator(ScalarSourceIntegrator(source=source_coef, q=self.q))
        R = dlform.assembly()
        R -= A @ d[:]
        R = self._append_neumann_to_residual(R, 'phase')

        tmr.send('phase_R_assemble')

        A, R = self._apply_boundary_conditions(A, R, field='phase')
        tmr.send('phase_apply_bc')
        
        dd = self.solver(A, R)
        d += dd[:]
  

        self.d = d
        self.pfcm.update_phase(d)
        #self.H = self.pfcm.H

        tmr.send('phase_solver')
        return bm.linalg.norm(R)
    
    def solve_displacement_auto(self) -> float:
        """
        Using automatic differentiation to assemble matrix and solve the displacement field and return the residual norm.

        Returns
        -------
        float
            The norm of the residual.
        """
        uh = self.uh
        tmr = self.tmr
        tmr.send('diap_start')

        fbc = VectorDirichletBC(self.tspace, self._currt_force_value, self._force_dof, direction=self._force_direction)
        uh, force_index = fbc.apply_value(uh)
        self.pfcm.update_disp(uh)

        @barycentric
        def postive_coef(bc, **kwargs):
            return self.EDFunc.degradation_function(self.d(bc))
        
        postive_coef.uh = uh
        postive_coef.kernel_func = self.pfcm.positive_stress_func

        ubform = NonlinearForm(self.tspace)

        if self.model_type == 'HybridModel' or self.model_type == 'IsotropicModel':
            ubform.add_integrator(NonlinearElasticIntegrator(coef=postive_coef, material=self.pfcm, q=self.q))
        else:
            @barycentric
            def negative_coef(bc, **kwargs):
                return 1
            
            negative_coef.uh = uh
            negative_coef.kernel_func = self.pfcm.negative_stress_func

            ubform.add_integrator(NonlinearElasticIntegrator(coef=postive_coef, material=self.pfcm, q=self.q), NonlinearElasticIntegrator(coef=negative_coef, material=self.pfcm, q=self.q))
            
        A, R = ubform.assembly()

        self._Rfu = bm.sum(-R[force_index])
        tmr.send('disp_assemble')

        # Apply force boundary conditions
        ubc = VectorDirichletBC(self.tspace, 0, threshold=self._force_dof, direction=self._force_direction)
        A, R = ubc.apply(A, R)
        R = self._append_neumann_to_residual(R, 'displacement')

        # Apply displacement boundary conditions
        A, R = self._apply_boundary_conditions(A, R, field='displacement')
        tmr.send('apply_bc')

        du = self.solver(A, R)

        uh += du[:]
        self.uh = uh
        
        self.pfcm.update_disp(uh)
        tmr.send('disp_solver')
        return bm.linalg.norm(R)
    
    def solve_phase_field_auto(self) -> float:
        """
        Solve the phase field and return the residual norm.

        Returns
        -------
        float
            The norm of the residual.
        """
        Gc, l0, d = self.Gc, self.l0, self.d
        
        c_d = self.CSDFunc.grad_density_function(d)[1]
        @barycentric
        def diffusion_coef(bc, **kwargs):
            return Gc * l0 * 2 / c_d
        
        def diffusion_kernel_func(u):
            return u
        
        def diffusion_grad_kernel_func(u):
            return 1

        @barycentric
        def mass_coef1(bc, **kwargs):
            return Gc / (l0 * c_d)
        
        @barycentric
        def mass_kernel_func1(u):
            return self.CSDFunc.grad_density_function(u)[0]

        def mass_grad_kernel_func1(u):
            return self.CSDFunc.grad_grad_density_function(u)[0]
                
        @barycentric
        def mass_coef2(bc, **kwargs):
            self.H = self.pfcm.maximum_historical_field(bc)
            return self.H
        
        @barycentric
        def mass_kernel_func2(u):
            return self.EDFunc.grad_degradation_function(u)
        
        @barycentric
        def mass_grad_kernel_func2(u):
            return self.EDFunc.grad_grad_degradation_function(u)
        
        diffusion_coef.kernel_func = diffusion_kernel_func
        mass_coef1.kernel_func = mass_kernel_func1
        mass_coef2.kernel_func = mass_kernel_func2

        if bm.backend_name == 'numpy':
            diffusion_coef.grad_kernel_func = diffusion_grad_kernel_func
            mass_coef1.grad_kernel_func = mass_grad_kernel_func1
            mass_coef2.grad_kernel_func = mass_grad_kernel_func2

        mass_coef1.uh = d
        mass_coef2.uh = d
        diffusion_coef.uh = d

        tmr = self.tmr
        tmr.send('phase_start')

        # using automatic differentiation to assemble the phase field system        
        dform = NonlinearForm(self.space)
        dform.add_integrator(ScalarNonlinearDiffusionIntegrator(diffusion_coef, q=self.q), ScalarNonlinearMassIntegrator(mass_coef1, q=self.q), ScalarNonlinearMassIntegrator(mass_coef2, q=self.q)) 
        #dform.add_integrator(ScalarNonlinearMassIntegrator(mass_coef1, q=self.q))
        #dform.add_integrator(ScalarNonlinearMassIntegrator(mass_coef2, q=self.q))
        #dform.add_integrator(ScalarSourceIntegrator(source_coef, q=self.q))

        A, R = dform.assembly()
        tmr.send('phase_matrix_assemble')

        R = self._append_neumann_to_residual(R, 'phase')

        A, R = self._apply_boundary_conditions(A, R, field='phase')
        tmr.send('phase_apply_bc')

        dd = self.solver(A, R)
        d += dd[:]

        self.d = d
        self.pfcm.update_phase(d)
        #self.H = self.pfcm.H

        tmr.send('phase_solver')
        return bm.linalg.norm(R)


    def set_lfe_space(self, p: int = 1, q: int = None):
        """
        Set the finite element spaces for displacement and phase fields.
        """
        self.p = p
        self.q = self.p + 3 if q is None else q
        self.space = LagrangeFESpace(self.mesh, self.p)
        self.tspace = TensorFunctionSpace(self.space, (self.mesh.geo_dimension(), -1))
        self.d = self.space.function()
        self.uh = self.tspace.function()

    def set_adaptive_refinement(self, marking_strategy: str = 'recovery', refine_method: str = 'bisect', theta: float = 0.2):
        """
        Set the adaptive refinement parameters.
        ----------
        marking_strategy : str, optional
            The marking strategy for refinement, by default 'recovery'.
        refine_method : str, optional
            The refinement method, by default 'bisect'.
        theta : float, optional
            Mark threshold parameter, by default 0.2.        
        """
        # Adaptive refinement settings
        self.enable_refinement = True
        self.adaptive = AdaptiveRefinement(marking_strategy=marking_strategy, refine_method=refine_method, theta=theta)

    def set_interpolation_data(self):
        """
        Set the interpolation data to refine.
        """
        GD = self.mesh.geo_dimension()
        #NQ = self.H.shape[-1] if len(self.H) > 1 else 1
        if GD == 2:
            dcell2dof = self.space.cell_to_dof()
            ucell2dof = self.tspace.cell_to_dof()
            data = {'uh': self.uh[ucell2dof], 'd': self.d[dcell2dof], 'H': self.H}
        elif GD == 3:
            data = {'nodedata':[self.uh, self.d], 'celldata':self.H}
        return data

    def update_interpolation_data(self, data):
        """
        Update the data after refinement
        """
        GD = self.mesh.geo_dimension()
        if GD == 2:
            dcell2dof = self.space.cell_to_dof()
            ucell2dof = self.tspace.cell_to_dof()
            self.uh[ucell2dof.reshape(-1)] = data['uh']
            self.d[dcell2dof.reshape(-1)] = data['d']
            H = data['H']
        elif GD == 3:
            assert self.p == 1
            self.uh = data['uh']
            self.d = data['d']

            H = data['H']
        self.H = H
        self.pfcm.update_historical_field(self.H)
        self.pfcm.update_disp(self.uh)
        self.pfcm.update_phase(self.d)

    def _save_vtkfile(self, fname: str):
        """
        Save the solution to a VTK file.

        Exports **full Lagrange dof data** (per macro-cell, points duplicated):

        - 2D triangles: ``VTK_LAGRANGE_TRIANGLE`` via
          :func:`~fracturex.utilfuc.vtk_lagrange_writer.write_lagrange_triangle_vtu`.
        - 2D quadrilaterals (4 corner vertices): tensor-product sampling and a
          refined **linear quad** grid via
          :func:`~fracturex.utilfuc.vtk_lagrange_writer.write_subdivided_quadrilateral_mesh_vtu`.
        - 3D hexahedra (8 corner vertices): tensor-product sampling and refined
          **linear hex** sub-cells via
          :func:`~fracturex.utilfuc.vtk_lagrange_writer.write_subdivided_hexahedron_mesh_vtu`.

        If a specialized path fails, falls back to ``mesh.nodedata`` +
        ``mesh.to_vtk`` (vertex-only geometry nodes).

        Parameters
        ----------
        fname : str
            File name for saving the VTK output.
        """
        mesh = self.mesh
        GD = int(mesh.geo_dimension())
        cell = np.asarray(bm.to_numpy(mesh.entity("cell")))
        ncv = int(cell.shape[1]) if cell.ndim == 2 else -1
        p_order = int(getattr(self, "p", 1))

        if GD == 2 and ncv == 3:
            try:
                sampled = sample_fields_for_lagrange_triangle(
                    mesh=mesh,
                    order=p_order,
                    field_specs=(
                        ("damage", self.d),
                        ("uh", self.uh),
                    ),
                )
                uh_s = sampled["uh"]
                if uh_s.ndim == 2 and uh_s.shape[1] >= 2:
                    sampled["ux"] = uh_s[:, 0]
                    sampled["uy"] = uh_s[:, 1]
                    if uh_s.shape[1] > 2:
                        sampled["uz"] = uh_s[:, 2]
                dname = os.path.dirname(fname)
                if dname:
                    os.makedirs(dname, exist_ok=True)
                write_lagrange_triangle_vtu(
                    fname=fname,
                    mesh=mesh,
                    order=p_order,
                    point_data=sampled,
                )
                return
            except Exception as e:
                print(
                    f"[MainSolve._save_vtkfile] Lagrange triangle VTU failed ({e}); "
                    "trying other exporters or vertex-based to_vtk."
                )

        if GD == 2 and ncv == 4:
            try:
                sampled = sample_fields_tensor_product_fe(
                    mesh=mesh,
                    order=p_order,
                    cell_dim=2,
                    field_specs=(
                        ("damage", self.d),
                        ("uh", self.uh),
                    ),
                )
                uh_s = sampled["uh"]
                if uh_s.ndim == 2 and uh_s.shape[1] >= 2:
                    sampled["ux"] = uh_s[:, 0]
                    sampled["uy"] = uh_s[:, 1]
                    if uh_s.shape[1] > 2:
                        sampled["uz"] = uh_s[:, 2]
                dname = os.path.dirname(fname)
                if dname:
                    os.makedirs(dname, exist_ok=True)
                write_subdivided_quadrilateral_mesh_vtu(
                    fname=fname,
                    mesh=mesh,
                    order=p_order,
                    point_data=sampled,
                )
                return
            except Exception as e:
                print(
                    f"[MainSolve._save_vtkfile] Quad high-order VTU failed ({e}); "
                    "falling back to vertex-based to_vtk."
                )

        if GD == 3 and ncv == 8:
            try:
                sampled = sample_fields_tensor_product_fe(
                    mesh=mesh,
                    order=p_order,
                    cell_dim=3,
                    field_specs=(
                        ("damage", self.d),
                        ("uh", self.uh),
                    ),
                )
                uh_s = sampled["uh"]
                if uh_s.ndim == 2 and uh_s.shape[1] >= 2:
                    sampled["ux"] = uh_s[:, 0]
                    sampled["uy"] = uh_s[:, 1]
                    if uh_s.shape[1] > 2:
                        sampled["uz"] = uh_s[:, 2]
                dname = os.path.dirname(fname)
                if dname:
                    os.makedirs(dname, exist_ok=True)
                write_subdivided_hexahedron_mesh_vtu(
                    fname=fname,
                    mesh=mesh,
                    order=p_order,
                    point_data=sampled,
                )
                return
            except Exception as e:
                print(
                    f"[MainSolve._save_vtkfile] Hex high-order VTU failed ({e}); "
                    "falling back to vertex-based to_vtk."
                )

        mesh.nodedata['damage'] = self.d
        mesh.nodedata['uh'] = self.uh.reshape(GD, -1).T
        dname = os.path.dirname(fname)
        if dname:
            os.makedirs(dname, exist_ok=True)
        mesh.to_vtk(fname=fname)

    def save_vtkfile(self, fname: str):
        """
        Save the solution to a VTK file.

        Parameters
        ----------
        fname : str
            File name for saving the VTK output.
        """
        self._vtkfname = fname
        self._save_vtk = True

    def auto_assembly_matrix(self):
        """
        Assemble the system matrix using automatic differentiation.
        """
        assert bm.backend_name != "numpy", "In the numpy backend, you cannot use automatic differentiation method to assembly matrix."
        self._atype = 'auto'

    def fast_assembly_matrix(self):
        """
        Assemble the system matrix using fast assembly.
        """
        self._atype = 'fast'

    def add_boundary_condition(
        self,
        field_type: str,
        bc_type: str,
        boundary_dof: Optional[TensorLike] = None,
        value: Optional[TensorLike] = None,
        direction: Optional[str] = None,
        *,
        gN: Optional[Callable] = None,
        threshold=None,
        neumann_q: Optional[int] = None,
    ):
        """
        Add boundary condition for a specific field and boundary.

        Parameters
        ----------
        field_type : str
            ``'force'``, ``'displacement'``, or ``'phase'``.
        bc_type : str
            ``'Dirichlet'`` or ``'Neumann'``.
        boundary_dof : TensorLike, optional
            For Dirichlet: DOF mask / indices (same as before). For Neumann **without**
            ``gN``: global DOF indices receiving a **lumped** flux / load vector ``value``.
        value : TensorLike, optional
            Prescribed value (Dirichlet) or lumped Neumann data matching ``boundary_dof``.
        direction : str, optional
            For vector Dirichlet: ``'x'``, ``'y'``, or ``'z'``.
        gN : callable, optional
            Natural boundary data in **fealpy** convention: for phase, signature like
            ``gN(ps, n)`` (Cartesian) or ``gN(bcs, index=...)`` (barycentric) as accepted by
            ``ScalarNeumannBCIntegrator``; for displacement, traction data for
            ``VectorNeumannBCIntegrator`` (see fealpy docs). When ``gN`` is given,
            ``boundary_dof`` / ``value`` are ignored for face assembly.
        threshold : optional
            Optional boundary-face selector passed to fealpy Neumann integrators
            (callable on face barycenters, index array, or ``None`` for all boundary faces).
        neumann_q : int, optional
            Face quadrature order for Neumann assembly; defaults to ``self.q`` once spaces exist.
            Ignored for lumped nodal Neumann.

        Notes
        -----
        Neumann contributions are assembled into the residual **after** the elastic /
        phase volume terms and **after** the special ``'force'`` displacement row
        replacement in elasticity, but **before** additional Dirichlet elimination.
        """
        if field_type not in self.bc_dict:
            self.bc_dict[field_type] = []

        if bc_type == 'Neumann':
            if field_type not in ('phase', 'displacement'):
                raise ValueError("Neumann BC is only supported for 'phase' or 'displacement'.")
            if gN is None and (boundary_dof is None or value is None):
                raise ValueError(
                    "Neumann BC requires either keyword `gN` (face natural data) "
                    "or both `boundary_dof` and `value` (lumped nodal loads)."
                )

        self.bc_dict[field_type].append({
            'type': bc_type,
            'bcdof': boundary_dof,
            'value': value,
            'direction': direction,
            'gN': gN,
            'threshold': threshold,
            'neumann_q': neumann_q,
        })

    def _append_neumann_to_residual(self, R: TensorLike, field: str) -> TensorLike:
        """Add all Neumann (natural) contributions for ``field`` to residual ``R``."""
        bc_list = self._get_boundary_conditions(field)
        if not bc_list:
            return R
        R = bm.asarray(R)
        load = bm.zeros_like(R)
        for bcdata in bc_list:
            if bcdata.get('type') != 'Neumann':
                continue
            if field == 'phase':
                load = load + self._assemble_scalar_neumann_load(bcdata)
            elif field == 'displacement':
                load = load + self._assemble_vector_neumann_load(bcdata)
            else:
                raise ValueError(f"Unknown field '{field}' for Neumann BC.")
        return R + load

    def _assemble_scalar_neumann_load(self, bcdata: Dict) -> TensorLike:
        """装配相场标量 Neumann（自然边界）载荷向量。

        Args:
            bcdata: 边界条件 dict；若含 ``gN`` 则按面积分装配，否则按 ``bcdof``/``value``
                做集中节点载荷。
        Returns:
            ``(scalar_gdof,)`` 的载荷向量。
        """
        if bcdata.get('gN') is not None:
            qn = bcdata.get('neumann_q')
            if qn is None:
                qn = self.q
            nbc = ScalarNeumannBCIntegrator(bcdata['gN'], threshold=bcdata.get('threshold'), q=int(qn))
            nbc.ftype = self.space.ftype
            F = nbc.assembly_face_vector(self.space)
            return bm.asarray(F, dtype=self.d.dtype).reshape(-1)
        dof = bm.asarray(bcdata['bcdof'], dtype=bm.int64).reshape(-1)
        val = bm.asarray(bcdata['value']).reshape(-1)
        if dof.size != val.size:
            raise ValueError(
                f"Phase Neumann: len(boundary_dof)={dof.size} != len(value)={val.size}."
            )
        F = bm.zeros((self.space.number_of_global_dofs(),), dtype=val.dtype)
        bm.add.at(F, dof, val)
        return F

    def _assemble_vector_neumann_load(self, bcdata: Dict) -> TensorLike:
        """装配位移向量 Neumann（面力/自然边界）载荷向量。

        Args:
            bcdata: 边界条件 dict；若含 ``gN`` 则按面积分装配面力，否则按 ``bcdof``/``value``
                做集中节点载荷。
        Returns:
            ``(tensor_gdof,)`` 的载荷向量。
        """
        GD = int(self.mesh.geo_dimension())
        gdof = int(self.tspace.number_of_global_dofs())
        if bcdata.get('gN') is not None:
            qn = bcdata.get('neumann_q')
            if qn is None:
                qn = self.q
            vn = VectorNeumannBCIntegrator(bcdata['gN'], threshold=bcdata.get('threshold'), q=int(qn))
            vn.ftype = self.space.ftype
            spaces = (self.space,) * GD
            F = vn.assembly_face_vector(spaces)
            F = bm.asarray(F).reshape(-1)
            if F.shape[0] != gdof:
                raise ValueError(
                    f"Displacement Neumann face load length {F.shape[0]} != tspace gdof {gdof}."
                )
            return F
        dof = bm.asarray(bcdata['bcdof'], dtype=bm.int64).reshape(-1)
        val = bm.asarray(bcdata['value']).reshape(-1)
        if dof.size != val.size:
            raise ValueError(
                f"Displacement Neumann: len(boundary_dof)={dof.size} != len(value)={val.size}."
            )
        F = bm.zeros((gdof,), dtype=val.dtype)
        bm.add.at(F, dof, val)
        return F

    def _get_boundary_conditions(self, field_type: str):
        """
        Get the boundary conditions for a specific field.

        Parameters
        ----------
        field_type : str
            'force', 'displacement', or 'phase'.

        Returns
        -------
        list
            A list of boundary condition data for the specified field.
        """
        return self.bc_dict.get(field_type, [])


    def _initialize_force_boundary(self):
        """
        Initialize the force boundary conditions.
        """
        force_data = self._get_boundary_conditions('force')
        force_data = force_data[0]
        self._force_type = force_data.get('type')
        self._force_dof = force_data.get('bcdof')
        self._force_value = force_data.get('value')
        self._force_direction = force_data.get('direction')

    def _apply_boundary_conditions(self, A, R, field: str):
        """
        Apply Dirichlet boundary conditions to ``A`` and ``R``.

        Neumann data are **not** handled here; they are assembled in
        :meth:`_append_neumann_to_residual` before this method runs.

        Parameters
        ----------
        A : sparse matrix
            System matrix.
        R : ndarray
            Residual vector.
        field : str
            Field type: ``'displacement'`` or ``'phase'``.
        """
        bc_list = self._get_boundary_conditions(field)

        if bc_list:
            for bcdata in bc_list:
                if bcdata.get('type') != 'Dirichlet':
                    continue
                if field == 'displacement':
                    bc = VectorDirichletBC(
                        self.tspace,
                        bcdata['value'],
                        bcdata['bcdof'],
                        direction=bcdata['direction'],
                    )
                    A, R = bc.apply(A, R)
                elif field == 'phase':
                    bc = DirichletBC(self.space, gd=bcdata['value'], threshold=bcdata['bcdof'])
                    A, R = bc.apply(A, R)
                else:
                    raise ValueError(f"Unknown field '{field}' for Dirichlet BC.")
        return A, R
    
    def get_residual_force(self):
        """
        Get the residual vector.
        """
        return self._Rforce
    
    def output_timer(self):
        """开启计时输出（置 ``self._timer=True``），后续迭代打印各阶段耗时。"""
        self._timer = True


    def set_energy_degradation(self, degradation_type='quadratic', EDfunc=None, **kwargs):
        """
        Set the energy degradation function.

        Parameters
        ----------
        EDfunc : callable, optional
            Energy degradation function class or factory. If None, a default energy degradation function is used.
        degradation_type : str, optional
            Type of energy degradation function. Default is 'quadratic'.
        **kwargs : dict
            Additional parameters passed to the energy degradation function.
        """
        if EDfunc is not None:
            self.EDFunc = EDfunc(degradation_type='user_defined', **kwargs)
        else:
            self.EDFunc = EDFunc(degradation_type=degradation_type)

    def set_crack_surface_density(self, density_type='AT2', CSDfunc=None, **kwargs):
        """
        Set the crack surface density function.

        Parameters
        ----------
        CSDFunc : callable, optional
            Crack surface density function class or factory. If None, a default crack surface density function is used.
        density_type : str, optional
            Type of crack surface density function. Default is 'AT2'.
        **kwargs : dict
            Additional parameters passed to the crack surface density function.
        """
        if CSDfunc is not None:
            self.CSDFunc = CSDfunc(density_type='user_defined', **kwargs)
        else:
            self.CSDFunc = CSDFunc(density_type=density_type)

    def set_cupy_solver(self):
        """
        Use the fealpy / CuPy-backed Krylov path for linear solves (requires CUDA tensors).
        """
        print('Using cupy solver.')
        self._solver = 'cupy'

    def set_scipy_solver(self):
        """
        Use SciPy sparse linear algebra on the CPU (default when tensors are on CPU).
        """
        print('Using scipy solver.')
        self._solver = 'scipy'

    def set_linear_solver_options(
        self,
        *,
        method: Optional[str] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        maxiter: Optional[int] = None,
        restart: Optional[int] = None,
        direct_max_dof: Optional[int] = None,
        use_ilu_preconditioner: Optional[bool] = None,
    ) -> None:
        """
        Configure the sparse linear solver used by :meth:`solver`.

        Parameters
        ----------
        method : {'auto', 'direct', 'gmres', 'lgmres', 'cg', 'pardiso', 'mumps'}, optional
            ``auto`` uses SciPy ``spsolve`` (SuperLU) when the matrix dimension is at
            most ``direct_max_dof``, otherwise ``gmres``. ``pardiso`` calls
            ``pypardiso.spsolve`` (Intel MKL PARDISO; install ``pypardiso``). ``mumps``
            uses the ``python-mumps`` bindings (``import mumps``) with a sequential
            ``mumps.Context`` factorization. MUMPS/PARDISO are **not** selected by
            ``auto``; set ``method`` explicitly when the optional packages are available.
        rtol, atol : float, optional
            Relative and absolute tolerances for Krylov methods (SciPy semantics).
        maxiter : int, optional
            Maximum Krylov iterations; ``None`` selects ``min(2000, max(100, 5*n))``.
        restart : int, optional
            GMRES restart length.
        direct_max_dof : int, optional
            Matrix order threshold for the ``auto`` branch to pick a direct solve.
        use_ilu_preconditioner : bool, optional
            If True, attempt an incomplete LU preconditioner with GMRES/LGMRES (SciPy
            path only); failures fall back to no preconditioning.
        """
        if method is not None:
            m = str(method).lower()
            allowed = ('auto', 'direct', 'gmres', 'lgmres', 'cg', 'pardiso', 'mumps')
            if m not in allowed:
                raise ValueError(f"method must be one of {allowed}, got {method!r}")
            self._lin_method = m
        if rtol is not None:
            self._lin_rtol = float(rtol)
        if atol is not None:
            self._lin_atol = float(atol)
        if maxiter is not None:
            self._lin_maxiter = int(maxiter)
        if restart is not None:
            self._lin_restart = int(restart)
        if direct_max_dof is not None:
            self._direct_max_dof = int(direct_max_dof)
        if use_ilu_preconditioner is not None:
            self._lin_use_ilu = bool(use_ilu_preconditioner)

    def _scipy_try_ilu(self, A_sp):
        """尝试为 SciPy 稀疏矩阵构造 ILU 预条件子。

        Args:
            A_sp: SciPy 稀疏系统矩阵。
        Returns:
            ``LinearOperator`` 形式的 ILU 预条件子；构造失败则返回 ``None``。
        """
        try:
            from scipy.sparse.linalg import LinearOperator, spilu

            ilu = spilu(A_sp.tocsc(), drop_tol=1e-4, fill_factor=12)

            def _mv(v):
                return ilu.solve(np.asarray(v, dtype=np.float64))

            return LinearOperator(A_sp.shape, matvec=_mv, dtype=np.float64)
        except Exception as exc:
            print(f"[MainSolve] ILU preconditioner disabled ({exc}).")
            return None

    def solver(self, A, R, atol=None, rtol=None, maxiter=None):
        """
        Solve ``A x = R`` with backend chosen by :meth:`set_scipy_solver` /
        :meth:`set_cupy_solver` and options from :meth:`set_linear_solver_options`.

        Optional ``atol`` / ``rtol`` / ``maxiter`` override instance defaults for this
        call only.
        """
        atol = self._lin_atol if atol is None else float(atol)
        rtol = self._lin_rtol if rtol is None else float(rtol)
        eff_maxiter = self._lin_maxiter if maxiter is None else int(maxiter)
        if eff_maxiter is not None and eff_maxiter <= 0:
            eff_maxiter = None

        if self._solver == 'scipy':
            A_sp = A.to_scipy()
            n = int(A_sp.shape[0])
            R_np = np.asarray(bm.to_numpy(R), dtype=np.float64).reshape(-1)

            method = self._lin_method
            if method == 'auto':
                method = 'direct' if n <= self._direct_max_dof else 'gmres'

            if method == 'direct':
                x = scipy_sparse_spsolve(A_sp.tocsc(), R_np)
                info = 0
            elif method == 'pardiso':
                x = solve_direct_pardiso(A_sp, R_np)
                info = 0
            elif method == 'mumps':
                x = solve_direct_mumps(A_sp, R_np)
                info = 0
            elif method in ('lgmres', 'gmres', 'cg'):
                m_it = eff_maxiter if eff_maxiter is not None else min(2000, max(100, 5 * n))
                M = None
                if self._lin_use_ilu and method in ('gmres', 'lgmres'):
                    M = self._scipy_try_ilu(A_sp)

                if method == 'lgmres':
                    x, info = lgmres(A_sp, R_np, rtol=rtol, atol=atol, maxiter=m_it, M=M)
                elif method == 'gmres':
                    x, info = scipy_sparse_gmres(
                        A_sp,
                        R_np,
                        rtol=rtol,
                        atol=atol,
                        maxiter=m_it,
                        restart=self._lin_restart,
                        M=M,
                    )
                elif method == 'cg':
                    x, info = scipy_sparse_cg(A_sp, R_np, rtol=rtol, atol=atol, maxiter=m_it, M=M)
                if info != 0:
                    print(f"[MainSolve.solver] scipy {method} finished with info={info}")
            else:
                raise ValueError(
                    f"Unknown scipy linear method {method!r}. "
                    f"Expected one of: direct, pardiso, mumps, lgmres, gmres, cg (or auto)."
                )

            x = bm.tensor(x)

        elif self._solver == 'cupy':
            R_b = bm.asarray(R)
            m_it = eff_maxiter if eff_maxiter is not None else min(2000, max(200, 10 * int(A.shape[0])))
            try:
                x = fealpy_gmres(
                    A,
                    R_b,
                    rtol=rtol,
                    atol=atol,
                    maxiter=m_it,
                    solver=self._solver,
                )
            except TypeError:
                x = fealpy_gmres(A, R_b, atol=atol, maxiter=m_it, solver=self._solver)
        else:
            raise ValueError(f"Unknown solver backend: {self._solver}")
        return x
