#!/usr/bin/env bash
# Wait for background paper jobs, aggregate results, sync summary into docs/.
#
# Usage:
#   bash scripts/paper_huzhang/wait_and_collect.sh
#   bash scripts/paper_huzhang/wait_and_collect.sh --no-wait   # collect only
#   bash scripts/paper_huzhang/wait_and_collect.sh model0 model1 model2
set -euo pipefail

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${_REPO_ROOT}/scripts/paper_huzhang/_case_id.sh"
# shellcheck source=/dev/null
source "${_REPO_ROOT}/scripts/paper_huzhang/env.sh"

WAIT=1
CASES=()
for arg in "$@"; do
  case "${arg}" in
    --no-wait) WAIT=0 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      CASES+=("$(_paper_log_slug "${arg}")")
      ;;
  esac
done

if [[ ${#CASES[@]} -eq 0 ]]; then
  CASES=(model0 model1 model2 model0_aux model1_aux)
fi

LOG_DIR="${FRACTUREX_PAPER_LOG_DIR:-${FRACTUREX_RESULTS_ROOT}/logs}"
DOCS_OUT="${FRACTUREX_PAPER_DOCS:-${_REPO_ROOT}/docs/HUZHANG_PAPER_RESULTS.md}"

_wait_one() {
  local case_id="$1"
  local pid_file="${LOG_DIR}/${case_id}.pid"
  local status_file="${LOG_DIR}/${case_id}.status"

  if [[ ! -f "${pid_file}" ]]; then
    echo "[wait] ${case_id}: no pid file (${pid_file}); skip wait" >&2
    return 0
  fi

  local pid
  pid="$(<"${pid_file}")"
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "[wait] ${case_id}: pid ${pid} not running"
    return 0
  fi

  echo "[wait] ${case_id}: waiting for pid ${pid} ..."
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 30
  done

  if [[ -f "${status_file}" ]]; then
    echo "[wait] ${case_id}: finished with status $(<"${status_file}")"
  fi
}

if [[ "${WAIT}" -eq 1 ]]; then
  for c in "${CASES[@]}"; do
    _wait_one "${c}"
  done
fi

_fail=0
_optional_slugs=(model0_aux model1_aux)
for c in "${CASES[@]}"; do
  st="${LOG_DIR}/${c}.status"
  if [[ ! -f "${st}" ]]; then
    if [[ " ${_optional_slugs[*]} " == *" ${c} "* ]]; then
      echo "[collect] ${c}: optional job not submitted; skip" >&2
      continue
    fi
    echo "[collect] ${c}: missing ${st}" >&2
    _fail=1
    continue
  fi
  if [[ "$(<"${st}")" != ok ]]; then
    if [[ " ${_optional_slugs[*]} " == *" ${c} "* ]]; then
      echo "[collect] ${c}: optional job status '$(<"${st}")'; skip" >&2
      continue
    fi
    echo "[collect] ${c}: status is '$(<"${st}")' (expected ok)" >&2
    _fail=1
  fi
done

"${FRACTUREX_PYTHON}" "${_REPO_ROOT}/scripts/paper_huzhang/collect_paper_bundle.py" \
  --root "${FRACTUREX_PAPER_ROOT}" \
  --docs-out "${DOCS_OUT}"

if [[ "${_fail}" -ne 0 ]]; then
  echo "Some cases did not finish successfully; see ${LOG_DIR}/*.status" >&2
  exit 1
fi

echo "Done. Index: ${FRACTUREX_PAPER_ROOT}/PAPER_INDEX.md"
echo "Docs:     ${DOCS_OUT}"
