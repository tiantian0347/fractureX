#!/usr/bin/env bash
# Run one paper case in the foreground (wrap with nohup/sbatch yourself).
# Writes status under results/logs/ for wait_and_collect.sh.
#
# Usage:
#   run_background_job.sh <case> [direct|aux|all]
#     case : model0 | model1 | square | model2
#            model0_aux | model1_aux         (alias: pass `<case> aux`)
#
# Examples:
#   nohup bash scripts/paper_huzhang/run_background_job.sh model0 direct ...
#   nohup bash scripts/paper_huzhang/run_background_job.sh model1 aux ...
set -euo pipefail

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${_REPO_ROOT}/scripts/paper_huzhang/_case_id.sh"
# shellcheck source=/dev/null
source "${_REPO_ROOT}/scripts/paper_huzhang/env.sh"

_raw="${1:-}"
_kind="${2:-direct}"
if [[ -z "${_raw}" ]]; then
  echo "Usage: $0 <model0|model1|square|model2|model0_aux|model1_aux> [direct|aux|all]" >&2
  exit 2
fi

case "${_kind}" in
  direct|aux|all) ;;
  *)
    echo "Unknown job kind: ${_kind} (use direct, aux, or all)" >&2
    exit 2
    ;;
esac

if [[ "${_kind}" == "aux" ]]; then
  CASE_NORM="$(_normalize_paper_case "${_raw}")" || {
    echo "Unknown case for aux job: ${_raw}" >&2
    exit 2
  }
  case "${CASE_NORM}" in
    model0) SLUG="model0_aux" ;;
    square) SLUG="model1_aux" ;;
    model2)
      echo "aux-space job is currently scripted only for model0 and model1 (got model2)." >&2
      echo "If you really need model2 aux, run scripts/paper_huzhang/run_case.py --case model2 --mode aux directly." >&2
      exit 2
      ;;
    *)
      echo "aux-space job is only defined for model0 and model1 (got: ${_raw})" >&2
      exit 2
      ;;
  esac
else
  SLUG="$(_paper_log_slug "${_raw}")" || {
    echo "Unknown case: ${_raw}" >&2
    exit 2
  }
fi

LOG_DIR="${FRACTUREX_PAPER_LOG_DIR:-${FRACTUREX_RESULTS_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

STATUS="${LOG_DIR}/${SLUG}.status"
EXIT_FILE="${LOG_DIR}/${SLUG}.exit"
PID_FILE="${LOG_DIR}/${SLUG}.pid"
STARTED="${LOG_DIR}/${SLUG}.started_at"

if [[ -f "${PID_FILE}" ]]; then
  _old_pid="$(<"${PID_FILE}")"
  if [[ -n "${_old_pid}" ]] && kill -0 "${_old_pid}" 2>/dev/null; then
    _old_status=""
    [[ -f "${STATUS}" ]] && _old_status="$(<"${STATUS}")"
    if [[ "${_old_status}" == "running" ]]; then
      echo "Refuse to start: another job for slug '${SLUG}' is still running (pid ${_old_pid})." >&2
      echo "If it is stale, kill it or remove ${PID_FILE} before retrying." >&2
      exit 3
    fi
  fi
fi

echo $$ > "${PID_FILE}"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "${STARTED}"
echo running > "${STATUS}"

_run() {
  case "${_kind}" in
    direct)
      CASE="$(_normalize_paper_case "${_raw}")" || exit 2
      export FRACTUREX_RUN_MODEL0_AUX=0
      bash "${_REPO_ROOT}/scripts/paper_huzhang/run_direct.sh" "${CASE}"
      ;;
    aux)
      export FRACTUREX_ELASTIC_FAST=0
      case "${SLUG}" in
        model0_aux)
          bash "${_REPO_ROOT}/scripts/paper_huzhang/run_aux_model0.sh"
          ;;
        model1_aux)
          bash "${_REPO_ROOT}/scripts/paper_huzhang/run_aux_model1.sh"
          ;;
      esac
      ;;
    all)
      CASE="$(_normalize_paper_case "${_raw}")" || exit 2
      bash "${_REPO_ROOT}/scripts/paper_huzhang/run_all.sh" "${CASE}"
      ;;
  esac
}

_rc=0
if [[ -n "${FRACTUREX_BG_JOB_LOG:-}" ]]; then
  mkdir -p "$(dirname "${FRACTUREX_BG_JOB_LOG}")"
  _run >> "${FRACTUREX_BG_JOB_LOG}" 2>&1 || _rc=$?
else
  _run || _rc=$?
fi

echo "${_rc}" > "${EXIT_FILE}"
if [[ "${_rc}" -eq 0 ]]; then
  echo ok > "${STATUS}"
else
  echo fail > "${STATUS}"
fi
exit "${_rc}"
