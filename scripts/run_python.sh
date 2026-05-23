#!/usr/bin/env bash
# Shared entry: resolve Python + repo paths (no venv activate).
set -euo pipefail

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${_REPO_ROOT}:${PYTHONPATH:-}"

# shellcheck source=/dev/null
source "${_REPO_ROOT}/scripts/resolve_fealpy_python.sh"
if ! ensure_fealpy_python "${_REPO_ROOT}"; then
  exit 127
fi
export FRACTUREX_PYTHON="${FEALPY_PYTHON}"

exec "${FRACTUREX_PYTHON}" "$@"
