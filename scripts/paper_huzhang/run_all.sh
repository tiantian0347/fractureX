#!/usr/bin/env bash
# Full paper batch: baselines (1 load step each) then main production runs.
set -euo pipefail

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${_REPO_ROOT}/scripts/paper_huzhang/env.sh"
CASE="${1:-all}"

"${_REPO_ROOT}/scripts/paper_huzhang/run_baseline.sh" "${CASE}"
"${_REPO_ROOT}/scripts/paper_huzhang/run_main.sh" "${CASE}"
"${FRACTUREX_PYTHON}" "${_REPO_ROOT}/scripts/paper_huzhang/collect_paper_bundle.py" \
  --root "${FRACTUREX_PAPER_ROOT}"
