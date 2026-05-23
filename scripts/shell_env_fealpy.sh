# Optional local dev helper — NOT required on servers.
# Paper / test scripts use scripts/resolve_fealpy_python.sh instead.
#
# Usage: source scripts/shell_env_fealpy.sh

if [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -x "${VIRTUAL_ENV}/bin/python" ]]; then
  export PATH="${VIRTUAL_ENV}/bin:${PATH}"
elif [[ -d "${HOME}/venv_fealpy3/bin" ]]; then
  export PATH="${HOME}/venv_fealpy3/bin:${PATH}"
fi

# shellcheck source=scripts/resolve_fealpy_python.sh
if [[ -f "$(dirname "${BASH_SOURCE[0]}")/resolve_fealpy_python.sh" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/resolve_fealpy_python.sh"
  ensure_fealpy_python "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" 2>/dev/null || true
fi
