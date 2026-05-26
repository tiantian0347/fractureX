#!/usr/bin/env bash
# Legacy entry: same as run_direct.sh (elastic sparse direct).
set -euo pipefail
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${_REPO_ROOT}/scripts/paper_huzhang/run_direct.sh" "$@"
