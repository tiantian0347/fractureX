from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from fracturex.cases.square_tension_precrack import SquareTensionPreCrackCase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.drivers.huzhang_phasefield_staggered import HuZhangPhaseFieldStaggeredDriver
from fracturex.postprocess.recorder import RunRecorder
from fealpy.backend import backend_manager as bm

@dataclass
class PhaseFieldMaterial:
	E: float = 210e3
	nu: float = 0.3
	Gc: float = 2.7
	l0: float = 0.1
	ft: float = 3.0

	@property
	def mu(self):
		return self.E / (2.0 * (1.0 + self.nu))

	@property
	def lam(self):
		return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


def main():
	material = PhaseFieldMaterial()

	case = SquareTensionPreCrackCase(
		_model=material,
		with_fracture=False,
		nx=32,
		ny=32,
		crack_y=0.5,
		crack_length=0.5,
		debug_mesh=True,
	)

	discr = HuZhangDiscretization(
		case=case,
		p=3,
		damage_p=1,
		use_relaxation=True,
	).build()

	damage = PhaseFieldDamageModel(
		density_type="AT2",
		degradation_type="quadratic",
		split="hybrid",
		debug=False,
	)

	elastic_assembler = HuZhangElasticAssembler(discr, case, damage)
	phase_assembler = PhaseFieldAssembler(discr, case, damage, debug=False)
	recorder = RunRecorder("results/square_precrack_stats", save_npz=True, save_every=1)

	driver = HuZhangPhaseFieldStaggeredDriver(
		case=case,
		discr=discr,
		damage=damage,
		elastic_assembler=elastic_assembler,
		phase_assembler=phase_assembler,
		tol=1e-5,
		maxit=1000,
		elastic_krylov="minres",
		phase_krylov="gmres",
		phase_precond="ilu",
		debug=True,
		timing=True,
		recorder=recorder,
	)

	# 在求解前，检查预裂纹初始化
	print("\n===== initial state before solve =====")
	initial_state = discr.state
	if initial_state is not None:
		print("initial max(d) =", float(np.max(cast(Any, initial_state.d)[:])))
		print("initial min(d) =", float(np.min(cast(Any, initial_state.d)[:])))
	
	loads = np.linspace(0.0, 1.0e-3, 11).tolist()
	#loads = bm.concatenate((bm.linspace(0, 5e-3, 501, dtype=bm.float64), bm.linspace(5e-3, 6.1e-3, 1101, dtype=bm.float64)[1:]))

	infos = driver.run(loads)

	print("\n===== solve summary =====")
	for info in infos:
		print(
			f"step={info.step:02d}, "
			f"load={info.load:.4e}, "
			f"iters={info.iters:02d}, "
			f"conv={info.converged}, "
			f"err_u={info.err_u:.3e}, "
			f"err_d={info.err_d:.3e}, "
			f"max_d={info.max_d:.3e}"
		)

	state = discr.state
	print("\n===== final state =====")
	assert state is not None
	print("max(d) =", float(np.max(cast(Any, state.d)[:])))
	print("min(d) =", float(np.min(cast(Any, state.d)[:])))
	print("max(H) =", float(np.max(cast(Any, state.H)[:])) if state.H is not None else 0.0)
	
	# 保存VTK来可视化
	driver._save_vtkfile("results/test_domain_precrack_final.vtu", cell_mode="mean")
	print("\nVTK file saved to: results/test_domain_precrack_final.vtu")
	
	# 额外调试：检查预裂纹线上的d值
	print("\n===== pre-crack line d values (debugging) =====")
	mesh = discr.mesh
	space_d = discr.space_d
	if space_d is not None:
		ip = space_d.interpolation_points()  # (NDOF, 2)
		crack_mask = case._on_precrack(ip)
		crack_indices = np.where(crack_mask)[0]
		if len(crack_indices) > 0:
			d_on_precrack = np.asarray(state.d[:])[crack_indices]
			print(f"Number of DOFs on pre-crack: {len(crack_indices)}")
			print(f"d values on pre-crack - min: {np.min(d_on_precrack):.6f}, max: {np.max(d_on_precrack):.6f}, mean: {np.mean(d_on_precrack):.6f}")
			print(f"First 10 d values on pre-crack: {d_on_precrack[:10]}")
		else:
			print("WARNING: No DOFs found on pre-crack line!")
	else:
		print("WARNING: space_d is None!")


if __name__ == "__main__":
	main()

