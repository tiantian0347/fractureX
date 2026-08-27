#!/usr/bin/env python3
"""Verify the coupled slow-subspace and local-elimination identities.

Purpose
-------
Check the algebraic chain used by the coupled slow-subspace paper:
T -> G -> lifted mode -> Q_omega G -> captured-mode decay.

Inputs
------
None. The script uses one deterministic symmetric positive-definite block
Jacobian. All matrices are dimensionless float64 fixtures.

Outputs
-------
One JSON object on stdout. A nonzero exit status indicates a failed numerical
contract.
"""
from __future__ import annotations

import json
import platform
from typing import Any

import numpy as np

from fracturex.analysis.staggered_slow_mode import (
    coupled_mode_lift,
    dominant_mode,
    local_elimination_projection,
    weighted_survival_factor,
)


SCRIPT_VERSION = "1.0"


def verify_coupled_decay() -> dict[str, Any]:
    """Verify the coupled eigenmode and SPD local-elimination decay identity."""
    matrix_a = np.array([[2.0, 0.1], [0.1, 1.5]], dtype=np.float64)
    matrix_d = np.diag([1.2, 1.0, 1.4])
    matrix_b = np.array(
        [[0.9, 0.25, 0.0], [0.0, 0.75, 0.15]], dtype=np.float64
    )
    matrix_c = matrix_b.T
    propagation = np.linalg.solve(
        matrix_d, matrix_c @ np.linalg.solve(matrix_a, matrix_b)
    )
    full_sweep = np.block(
        [
            [np.zeros((2, 2)), -np.linalg.solve(matrix_a, matrix_b)],
            [np.zeros((3, 2)), propagation],
        ]
    )
    jacobian = np.block([[matrix_a, matrix_b], [matrix_c, matrix_d]])
    if np.min(np.linalg.eigvalsh(jacobian)) <= 0.0:
        raise AssertionError("fixture Jacobian is not symmetric positive definite")

    mode = dominant_mode(propagation)
    eigenvalue = mode.eigenvalue
    lifted = coupled_mode_lift(matrix_a, matrix_b, mode.mode, eigenvalue)
    lifting_residual = float(
        np.linalg.norm(full_sweep @ lifted - eigenvalue * lifted)
    )

    patch_dofs = np.array([0, 1, 2, 3], dtype=np.int64)
    projection_complement = local_elimination_projection(jacobian, patch_dofs)
    identity = np.eye(jacobian.shape[0])
    local_projection = identity - projection_complement

    def j_norm(vector: np.ndarray) -> float:
        """Return the norm induced by the monolithic SPD Jacobian."""
        value = np.real_if_close(vector).astype(np.float64)
        return float(np.sqrt(value @ jacobian @ value))

    survival = weighted_survival_factor(
        projection_complement, np.real_if_close(lifted), jacobian
    )
    captured = 1.0 - survival**2
    composite = projection_complement @ full_sweep
    measured = j_norm(composite @ lifted) / j_norm(lifted)
    predicted = abs(eigenvalue) * np.sqrt(max(0.0, 1.0 - captured))
    decay_error = abs(measured - predicted)

    if lifting_residual > 1.0e-12:
        raise AssertionError("coupled eigenmode lifting identity failed")
    if decay_error > 1.0e-12:
        raise AssertionError("local-elimination decay identity failed")
    if np.linalg.norm(
        projection_complement @ projection_complement - projection_complement
    ) > 1.0e-12:
        raise AssertionError("local-elimination derivative is not a projection")

    return {
        "spectral_radius": mode.spectral_radius,
        "eigenvalue_real": float(np.real(eigenvalue)),
        "eigenvalue_imag": float(np.imag(eigenvalue)),
        "lifting_residual_l2": lifting_residual,
        "patch_dofs": patch_dofs.tolist(),
        "capture_rate": float(captured),
        "survival_factor": float(survival),
        "measured_decay": float(measured),
        "predicted_decay": float(predicted),
        "decay_identity_error": float(decay_error),
        "status": "passed",
    }


def main() -> None:
    """Run the deterministic identity checks and print machine-readable JSON."""
    result = {
        "script_version": SCRIPT_VERSION,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "float_dtype": "float64",
        "coupled_slow_subspace": verify_coupled_decay(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
