#!/usr/bin/env bash
# Helpers for parallel background paper runs.
# Default layout: model0/1/2 elastic direct + model0 aux-space (4 jobs).
#
#   bash scripts/paper_huzhang/background_batch.sh print-cmds
#   bash scripts/paper_huzhang/background_batch.sh status
#   bash scripts/paper_huzhang/background_batch.sh watch-and-collect
set -euo pipefail

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_ROOT="${FRACTUREX_RESULTS_ROOT:-${_REPO_ROOT}/results}"
LOG_DIR="${FRACTUREX_PAPER_LOG_DIR:-${RESULTS_ROOT}/logs}"
# Slugs for logs/status files
CASES=(model0 model1 model2 model0_aux)

_source_env() {
  # shellcheck source=/dev/null
  source "${_REPO_ROOT}/scripts/paper_huzhang/env.sh"
}

_print_cmds() {
  cat <<'EOF'
# From repo root — activate env once, then start four independent jobs:
EOF
  echo "cd ${_REPO_ROOT}"
  echo "source scripts/paper_huzhang/env.sh"
  echo "mkdir -p ${LOG_DIR}"
  echo ""
  echo "# Full-resolution direct elastic (recommend pardiso if scipy SuperLU OOMs):"
  echo "# export FRACTUREX_ELASTIC_DIRECT_BACKEND=pardiso"
  echo ""
  for c in model0 model1 model2; do
    cat <<EOF
# ${c} — elastic direct
nohup env FRACTUREX_BG_JOB_LOG=${LOG_DIR}/${c}.log \\
  bash scripts/paper_huzhang/run_background_job.sh ${c} direct \\
  >> ${LOG_DIR}/${c}.nohup 2>&1 &
echo \$! > ${LOG_DIR}/${c}.nohup_pid

EOF
  done
  cat <<EOF
# model0 — elastic aux-space precondition (validation run)
nohup env FRACTUREX_BG_JOB_LOG=${LOG_DIR}/model0_aux.log FRACTUREX_ELASTIC_FAST=0 \\
  bash scripts/paper_huzhang/run_background_job.sh model0 aux \\
  >> ${LOG_DIR}/model0_aux.nohup 2>&1 &
echo \$! > ${LOG_DIR}/model0_aux.nohup_pid

EOF
  cat <<EOF
# After all jobs finish:
bash scripts/paper_huzhang/wait_and_collect.sh model0 model1 model2 model0_aux
EOF
}

_status() {
  mkdir -p "${LOG_DIR}"
  printf "%-12s %-10s %-8s %s\n" CASE STATUS PID LOG
  for c in "${CASES[@]}"; do
    local st pid log_line
    st="-"
    [[ -f "${LOG_DIR}/${c}.status" ]] && st="$(<"${LOG_DIR}/${c}.status")"
    pid="-"
    [[ -f "${LOG_DIR}/${c}.pid" ]] && pid="$(<"${LOG_DIR}/${c}.pid")"
    log_line="${LOG_DIR}/${c}.log"
    if [[ -f "${LOG_DIR}/${c}.nohup" ]]; then
      log_line="${LOG_DIR}/${c}.nohup"
    fi
    printf "%-12s %-10s %-8s %s\n" "${c}" "${st}" "${pid}" "${log_line}"
  done
}

_subcmd="${1:-print-cmds}"
shift || true

case "${_subcmd}" in
  print-cmds|print)
    _print_cmds
    ;;
  status)
    _status
    ;;
  watch|watch-and-collect|collect)
    _source_env
    exec bash "${_REPO_ROOT}/scripts/paper_huzhang/wait_and_collect.sh" "$@"
    ;;
  *)
    echo "Usage: $0 {print-cmds|status|watch-and-collect}" >&2
    exit 2
    ;;
esac
