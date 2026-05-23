#!/usr/bin/env bash
# Main paper path: parallel assembly + elastic aux-space GMRES + phase GMRES (no precond).
set -euo pipefail

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${_REPO_ROOT}/scripts/paper_huzhang/env.sh"

CASE="${1:-all}"
OUT="${FRACTUREX_PAPER_ROOT}"

_run_one() {
  echo ">>> main / ${1}"
  "${FRACTUREX_PYTHON}" "${_REPO_ROOT}/scripts/paper_huzhang/run_case.py" \
    --case "${1}" --mode main --out-root "${OUT}"
}

case "${CASE}" in
  model0|square|model2) _run_one "${CASE}" ;;
  all)
    _run_one model0
    _run_one square
    _run_one model2
    ;;
  *)
    echo "Usage: $0 [model0|square|model2|all]" >&2
    exit 2
    ;;
esac
