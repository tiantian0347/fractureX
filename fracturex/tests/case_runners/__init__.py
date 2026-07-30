"""Parametrized case runners for operator-learning / smoke datasets.

Paper experiment scripts live beside this package as
``phasefield_model*_huzhang.py`` and keep their own env-var surface.
"""
from fracturex.tests.case_runners.model0_runner import (
    Model0Material,
    Model0RunArgs,
    run_model0_one,
)
from fracturex.tests.case_runners.model2_runner import (
    Model2Material,
    Model2RunArgs,
    run_model2_one,
)
from fracturex.tests.case_runners.model3_runner import (
    Model3Material,
    Model3RunArgs,
    run_model3_one,
)
from fracturex.tests.case_runners.model4_runner import (
    Model4Material,
    Model4RunArgs,
    run_model4_one,
)
from fracturex.tests.case_runners.model5_runner import (
    Model5Material,
    Model5RunArgs,
    run_model5_one,
)
from fracturex.tests.case_runners.model6_runner import (
    Model6Material,
    Model6RunArgs,
    run_model6_one,
)

__all__ = [
    "Model0Material",
    "Model0RunArgs",
    "run_model0_one",
    "Model2Material",
    "Model2RunArgs",
    "run_model2_one",
    "Model3Material",
    "Model3RunArgs",
    "run_model3_one",
    "Model4Material",
    "Model4RunArgs",
    "run_model4_one",
    "Model5Material",
    "Model5RunArgs",
    "run_model5_one",
    "Model6Material",
    "Model6RunArgs",
    "run_model6_one",
]
