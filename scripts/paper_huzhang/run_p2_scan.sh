#!/usr/bin/env bash
# Launch a single P2 mesh-scan run for model0 in the background.
#
# Usage:
#   bash scripts/paper_huzhang/run_p2_scan.sh <h_tag> <hmin> <mode>
#
# Example (h2 aux):
#   bash scripts/paper_huzhang/run_p2_scan.sh h2 0.025 aux
#
# Outputs go to:
#   results/phasefield/model0_circular_notch/paper_<mode>_<h_tag>/epsg_1e-06/
#   results/logs/p2_model0_<mode>_<h_tag>.log
#   results/logs/p2_model0_<mode>_<h_tag>.pid

set -euo pipefail

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Auto-detect a working Python before env.sh runs its self-check.
if [[ -z "${FRACTUREX_PYTHON:-}" ]]; then
  for _candidate in \
    /home/gongshihua/miniconda3/envs/py312/bin/python \
    "$(command -v python 2>/dev/null || true)" \
    "$(command -v python3 2>/dev/null || true)"; do
    [[ -z "${_candidate}" || ! -x "${_candidate}" ]] && continue
    if "${_candidate}" -c "import numpy, scipy; from fealpy.backend import backend_manager" 2>/dev/null; then
      export FRACTUREX_PYTHON="${_candidate}"
      break
    fi
  done
fi

# shellcheck source=/dev/null
source "${_REPO_ROOT}/scripts/paper_huzhang/env.sh"

H_TAG="${1:-}"
HMIN="${2:-}"
MODE="${3:-}"

if [[ -z "${H_TAG}" || -z "${HMIN}" || -z "${MODE}" ]]; then
  echo "Usage: $0 <h_tag> <hmin> <mode>" >&2
  echo "  Example: $0 h2 0.025 aux" >&2
  exit 2
fi

case "${MODE}" in
  direct|aux) ;;
  *) echo "MODE must be 'direct' or 'aux'." >&2; exit 2 ;;
esac

LOG_DIR="${FRACTUREX_PAPER_LOG_DIR:-${FRACTUREX_RESULTS_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

SLUG="p2_model0_${MODE}_${H_TAG}"
LOG="${LOG_DIR}/${SLUG}.log"
PID_FILE="${LOG_DIR}/${SLUG}.pid"
STARTED="${LOG_DIR}/${SLUG}.started_at"
STATUS="${LOG_DIR}/${SLUG}.status"
EXIT_FILE="${LOG_DIR}/${SLUG}.exit"

if [[ -f "${PID_FILE}" ]]; then
  _old_pid="$(<"${PID_FILE}")"
  if [[ -n "${_old_pid}" ]] && kill -0 "${_old_pid}" 2>/dev/null; then
    echo "Refuse to start: ${SLUG} is still running (pid ${_old_pid})." >&2
    exit 3
  fi
fi

export FRACTUREX_HMIN="${HMIN}"
export FRACTUREX_RUN_LABEL_SUFFIX="${H_TAG}"
export FRACTUREX_ELASTIC_FAST=0   # use the auxiliary-space preconditioner (not the fast variant)

OUT_ROOT="${FRACTUREX_PAPER_ROOT:-${FRACTUREX_RESULTS_ROOT}}"

date -u +"%Y-%m-%dT%H:%M:%SZ" > "${STARTED}"
echo "running" > "${STATUS}"
rm -f "${EXIT_FILE}"

# Launch a detached supervisor subshell. It starts the Python worker, records
# its pid, waits for it to finish, and writes the final status / exit code.
# The outer script returns immediately so the caller is not blocked.
(
  "${FRACTUREX_PYTHON}" \
    "${_REPO_ROOT}/scripts/paper_huzhang/run_case.py" \
    --case model0 --mode "${MODE}" --out-root "${OUT_ROOT}" \
    >> "${LOG}" 2>&1 &
  _child=$!
  echo "${_child}" > "${PID_FILE}"
  wait "${_child}"
  _rc=$?
  echo "${_rc}" > "${EXIT_FILE}"
  if [[ "${_rc}" -eq 0 ]]; then
    echo ok > "${STATUS}"
  else
    echo fail > "${STATUS}"
  fi
) </dev/null >/dev/null 2>&1 &
disown

# Wait briefly for the supervisor to write the PID_FILE, so the caller can
# report a meaningful pid.
NEW_PID=""
for _try in 1 2 3 4 5 6 7 8 9 10; do
  if [[ -s "${PID_FILE}" ]]; then
    NEW_PID="$(<"${PID_FILE}")"
    break
  fi
  sleep 0.2
done

echo "Launched ${SLUG} pid=${NEW_PID:-?}"
echo "  hmin=${HMIN}  mode=${MODE}  label_suffix=${H_TAG}"
echo "  log:    ${LOG}"
echo "  status: ${STATUS}"
echo "  exit:   ${EXIT_FILE} (written on completion)"
