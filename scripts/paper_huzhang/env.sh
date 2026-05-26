#!/usr/bin/env bash
# fractureX paper / batch runs — no venv activate needed.
# Usage from repo root:
#   bash scripts/paper_huzhang/run_all.sh model0
#
# Optional override:
#   export FEALPY_PYTHON=/path/to/python

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${_REPO_ROOT}:${PYTHONPATH:-}"
export FRACTUREX_RESULTS_ROOT="${FRACTUREX_RESULTS_ROOT:-${_REPO_ROOT}/results}"
export FRACTUREX_PAPER_ROOT="${FRACTUREX_PAPER_ROOT:-${FRACTUREX_RESULTS_ROOT}}"
export FRACTUREX_PAPER_LOG_DIR="${FRACTUREX_PAPER_LOG_DIR:-${FRACTUREX_RESULTS_ROOT}/logs}"

# Avoid OpenBLAS "too many threads" segfault on high-core hosts during parallel assembly.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export FRACTUREX_ASSEMBLY_NPROC="${FRACTUREX_ASSEMBLY_NPROC:-64}"

# shellcheck source=/dev/null
source "${_REPO_ROOT}/scripts/resolve_fealpy_python.sh"

if [[ -n "${FRACTUREX_PYTHON:-}" ]] && _fracturex_runtime_ok "${FRACTUREX_PYTHON}"; then
  export FEALPY_PYTHON="${FRACTUREX_PYTHON}"
elif ! ensure_fealpy_python "${_REPO_ROOT}"; then
  return 1 2>/dev/null || exit 1
fi

export FRACTUREX_PYTHON="${FEALPY_PYTHON}"

if [[ "${FRACTUREX_ENV_QUIET:-0}" != "1" ]]; then
  echo "FRACTUREX_PYTHON=${FRACTUREX_PYTHON}"
  echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} FRACTUREX_ASSEMBLY_NPROC=${FRACTUREX_ASSEMBLY_NPROC}"
  "${FRACTUREX_PYTHON}" -c "
import numpy, scipy
from fealpy.backend import backend_manager
parts = ['numpy', numpy.__version__, 'scipy', scipy.__version__]
try:
    import pyamg
    parts += ['pyamg', pyamg.__version__]
except ImportError:
    parts.append('pyamg MISSING (pip install pyamg)')
print('runtime OK', ' '.join(parts))
"
fi
