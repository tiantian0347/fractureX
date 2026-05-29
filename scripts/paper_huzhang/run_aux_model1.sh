#!/usr/bin/env bash
# model1 (square_tension_precrack) only:
# parallel assembly + elastic aux-space preconditioned GMRES.
set -euo pipefail

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${_REPO_ROOT}/scripts/paper_huzhang/env.sh"

OUT="${FRACTUREX_PAPER_ROOT}"

echo ">>> aux (aux-space elastic) / model1 (square_tension_precrack)"
exec "${FRACTUREX_PYTHON}" "${_REPO_ROOT}/scripts/paper_huzhang/run_case.py" \
  --case square --mode aux --out-root "${OUT}"
