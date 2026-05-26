#!/usr/bin/env bash
# Full paper batch: optional baseline, direct elastic (3 cases), optional model0 aux.
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

# Baseline uses scipy SuperLU direct factorization on the full paper mesh (~3M+ DOFs).
# That exceeds SuperLU's practical limits; skip for smoke runs unless explicitly forced.
_skip_baseline="${FRACTUREX_SKIP_BASELINE:-0}"
if [[ "${FRACTUREX_RUN_SHORT:-0}" == "1" ]]; then
  _skip_baseline=1
fi
if [[ "${_skip_baseline}" == "1" ]]; then
  echo ">>> skipping baseline (FRACTUREX_SKIP_BASELINE=1 or FRACTUREX_RUN_SHORT=1)"
else
  "${_REPO_ROOT}/scripts/paper_huzhang/run_baseline.sh" "${CASE}"
fi

"${_REPO_ROOT}/scripts/paper_huzhang/run_direct.sh" "${CASE}"

_run_aux="${FRACTUREX_RUN_MODEL0_AUX:-1}"
if [[ "${FRACTUREX_RUN_SHORT:-0}" == "1" ]]; then
  _run_aux="${FRACTUREX_RUN_MODEL0_AUX:-0}"
fi
if [[ "${_run_aux}" == "1" ]]; then
  if [[ "${CASE}" == "model0" || "${CASE}" == "all" ]]; then
    "${_REPO_ROOT}/scripts/paper_huzhang/run_aux_model0.sh"
  fi
fi

"${FRACTUREX_PYTHON}" "${_REPO_ROOT}/scripts/paper_huzhang/collect_paper_bundle.py" \
  --root "${FRACTUREX_PAPER_ROOT}"
