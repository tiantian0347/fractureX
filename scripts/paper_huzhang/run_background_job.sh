#!/usr/bin/env bash
# Run one paper case in the foreground (wrap with nohup/sbatch yourself).
# Writes status under results/logs/ for wait_and_collect.sh.
#
# Usage:
#   run_background_job.sh model0 [direct|aux|all]
#   run_background_job.sh model0_aux [aux]
#
# Examples:
#   nohup bash scripts/paper_huzhang/run_background_job.sh model0 direct ...
#   nohup bash scripts/paper_huzhang/run_background_job.sh model0 aux ...
set -euo pipefail

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${_REPO_ROOT}/scripts/paper_huzhang/_case_id.sh"
# shellcheck source=/dev/null
source "${_REPO_ROOT}/scripts/paper_huzhang/env.sh"

_raw="${1:-}"
_kind="${2:-direct}"
if [[ -z "${_raw}" ]]; then
  echo "Usage: $0 <model0|model1|model2|model0_aux> [direct|aux|all]" >&2
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
  SLUG="model0_aux"
  if [[ "${_raw}" != "model0" && "${_raw}" != "model0_aux" && "${_raw}" != "model0-aux" && "${_raw}" != "model0aux" ]]; then
    echo "aux-space job is only defined for model0 (use: run_background_job.sh model0 aux)" >&2
    exit 2
  fi
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
      bash "${_REPO_ROOT}/scripts/paper_huzhang/run_aux_model0.sh"
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
