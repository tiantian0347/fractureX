#!/usr/bin/env bash
# Foreground runner for the model0 2nd-vs-4th-order phase-field comparison.
# Wrap with nohup/screen/sbatch when launching in the background.
#
# Usage:
#   bash scripts/paper_ipfem/run_model0_order_compare.sh
#   nohup bash scripts/paper_ipfem/run_model0_order_compare.sh > /tmp/m0cmp.log 2>&1 &
set -euo pipefail

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${_REPO_ROOT}/scripts/paper_huzhang/env.sh"

SLUG="paper_ipfem_model0_order"
LOG_DIR="${FRACTUREX_PAPER_LOG_DIR:-${FRACTUREX_RESULTS_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
STATUS="${LOG_DIR}/${SLUG}.status"
EXIT_FILE="${LOG_DIR}/${SLUG}.exit"
PID_FILE="${LOG_DIR}/${SLUG}.pid"
STARTED="${LOG_DIR}/${SLUG}.started_at"
JOB_LOG="${LOG_DIR}/${SLUG}.log"

if [[ -f "${PID_FILE}" ]]; then
  _old_pid="$(<"${PID_FILE}")"
  if [[ -n "${_old_pid}" ]] && kill -0 "${_old_pid}" 2>/dev/null; then
    _old_status=""
    [[ -f "${STATUS}" ]] && _old_status="$(<"${STATUS}")"
    if [[ "${_old_status}" == "running" ]]; then
      echo "Refuse to start: '${SLUG}' still running (pid ${_old_pid})." >&2
      exit 3
    fi
  fi
fi

echo $$ > "${PID_FILE}"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "${STARTED}"
echo running > "${STATUS}"

_rc=0
"${FRACTUREX_PYTHON}" \
  "${_REPO_ROOT}/scripts/paper_ipfem/model0_order_compare.py" \
  "$@" >> "${JOB_LOG}" 2>&1 || _rc=$?

echo "${_rc}" > "${EXIT_FILE}"
if [[ "${_rc}" -eq 0 ]]; then
  echo ok > "${STATUS}"
else
  echo fail > "${STATUS}"
fi
exit "${_rc}"
