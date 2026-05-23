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
  "${FRACTUREX_PYTHON}" -c "import numpy, scipy; from fealpy.backend import backend_manager; print('runtime OK (numpy', numpy.__version__, ')')"
fi
