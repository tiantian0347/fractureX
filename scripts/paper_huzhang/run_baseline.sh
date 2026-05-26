#!/usr/bin/env bash
# Baseline: ONE load step, serial assembly, elastic scipy spsolve (for paper comparison).
set -euo pipefail

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${_REPO_ROOT}/scripts/paper_huzhang/_case_id.sh"
# shellcheck source=/dev/null
source "${_REPO_ROOT}/scripts/paper_huzhang/env.sh"

_raw="${1:-all}"
CASE="$(_normalize_paper_case "${_raw}")" || {
  echo "Usage: $0 [model0|model1|square|model2|all]" >&2
  exit 2
}
OUT="${FRACTUREX_PAPER_ROOT}"

_run_one() {
  echo ">>> baseline (1 step, direct elastic) / ${1}"
  "${FRACTUREX_PYTHON}" "${_REPO_ROOT}/scripts/paper_huzhang/run_case.py" \
    --case "${1}" --mode baseline --out-root "${OUT}"
}

case "${CASE}" in
  model0|square|model2) _run_one "${CASE}" ;;
  all)
    _run_one model0
    _run_one square
    _run_one model2
    ;;
  *)
    echo "Usage: $0 [model0|model1|square|model2|all]" >&2
    exit 2
    ;;
esac
