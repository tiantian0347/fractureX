"""IP-FEM 相场断裂算例集合。

移植自 ttthesis/code/ip_hybrid_mix/test_model{0,1,2,3}.py，只保留 2D 算例
(内罚 FEM 只在 InteriorPenaltyFESpace2d 上有实现)。

用法示例::

    from fracturex.interior_penalty.cases import Model1SquareTensionCase
    case = Model1SquareTensionCase()
    result = case.run(max_steps=5, maxit_per_step=30)
"""
from .model0_circular_hole import Model0CircularHoleCase
from .model1_square_tension import Model1SquareTensionCase
from .model1_sg import Model1SGCase
from .model2_notch_shear import Model2ShearCase
from .model3_lshape import Model3LShapeCase
from .sent_tension import SentTensionMieheCase

__all__ = [
    "Model0CircularHoleCase",
    "Model1SquareTensionCase",
    "Model1SGCase",
    "Model2ShearCase",
    "Model3LShapeCase",
    "SentTensionMieheCase",
]
