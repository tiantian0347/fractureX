# test_huzhang_phasefield_square_tension.py

from dataclasses import dataclass
import numpy as np

from fracturex.cases.square_tension import SquareTensionCase
from fracturex.discretization.huzhang_discretization import HuZhangDiscretization
from fracturex.damage.phasefield_damage import PhaseFieldDamageModel
from fracturex.assemblers.huzhang_elastic_assembler import HuZhangElasticAssembler
from fracturex.assemblers.phasefield_assembler import PhaseFieldAssembler
from fracturex.drivers.huzhang_phasefield_staggered import HuZhangPhaseFieldStaggeredDriver


# ------------------------------------------------------------
# 1. 材料参数对象
#    只要能满足 case.model() / damage.on_build() / assembler 的读取即可
# ------------------------------------------------------------
@dataclass
class PhaseFieldMaterial:
    E: float
    nu: float
    Gc: float
    l0: float
    ft: float = 1.0   # 先占位；当前 phase-field 版本不一定用到


# ------------------------------------------------------------
# 2. 基于现有 SquareTensionCase 做一个 phase-field 版本
#    注意：这里不是预制裂纹，切口已经在网格里，所以 phasefield_dirichlet_data 返回 None
# ------------------------------------------------------------
class SquareTensionPhaseFieldCase(SquareTensionCase):
    def phasefield_dirichlet_data(self, load: float):
        # 几何切口已经存在，不再额外强制 d=1
        return None


def main():
    # --------------------------------------------------------
    # 3. 材料
    # --------------------------------------------------------
    material = PhaseFieldMaterial(
        E=210e3,     # 你可以改成自己的量纲/参数
        nu=0.3,
        Gc=2.7,
        l0=0.03,
        ft=3.0,
    )

    # --------------------------------------------------------
    # 4. case：注意 with_fracture=True
    #    表示网格里直接有切口
    # --------------------------------------------------------
    case = SquareTensionPhaseFieldCase(
        _model=material,
        with_fracture=True,
        refine=4,          # 先适中；太粗不容易看到切口尖端效应，太细先别上
        debug_mesh=True,
    )

    # --------------------------------------------------------
    # 5. 离散：Hu-Zhang + 独立 phase-field 空间
    # --------------------------------------------------------
    discr = HuZhangDiscretization(
        case=case,
        p=3,                 # 先从低阶开始，先跑通
        damage_p=1,          # phase-field space order
        use_relaxation=True,
    )

    # build mesh/state
    discr.build()

    # --------------------------------------------------------
    # 6. damage model：hybrid
    #    - elasticity degradation: isotropic
    #    - history field: spectral
    # --------------------------------------------------------
    damage = PhaseFieldDamageModel(
        density_type="AT2",
        degradation_type="quadratic",
        split="hybrid",
        debug=True,
    )

    # --------------------------------------------------------
    # 7. assemblers
    # --------------------------------------------------------
    elastic_assembler = HuZhangElasticAssembler(discr, case, damage)
    phase_assembler = PhaseFieldAssembler(discr, case, damage, debug=True)

    # --------------------------------------------------------
    # 8. driver
    # --------------------------------------------------------
    driver = HuZhangPhaseFieldStaggeredDriver(
        case=case,
        discr=discr,
        damage=damage,
        elastic_assembler=elastic_assembler,
        phase_assembler=phase_assembler,
        tol=1e-5,
        maxit=30,
        debug=True,
    )

    # --------------------------------------------------------
    # 9. 载荷步
    #    先用很小的位移加载，验证程序稳定性
    # --------------------------------------------------------
    loads = np.linspace(0.0, 1.0e-3, 11).tolist()

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

    # 最后一步结果简单检查
    state = discr.state
    print("\n===== final state =====")
    print("max(d) =", float(np.max(state.d[:])))
    print("max(H) =", float(np.max(state.H[:])))


if __name__ == "__main__":
    main()