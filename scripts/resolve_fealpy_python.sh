#!/usr/bin/env bash
# Pick a Python with fractureX runtime deps — no manual venv activate required.
#
# Prefers (in order): working FEALPY_PYTHON, conda base/env, PATH python3/python,
# then optional local venvs (.venv, ~/venv_fealpy3).
#
# Override: export FEALPY_PYTHON=/path/to/python
# One-time install in that python:
#   python3 -m pip install fealpy && python3 -m pip install -e /path/to/fractureX

_init_conda_if_needed() {
  [[ -n "${CONDA_PREFIX:-}" ]] && return 0
  local conda_sh=""
  if [[ -n "${CONDA_EXE:-}" ]]; then
    conda_sh="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    conda_sh="${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    conda_sh="${HOME}/anaconda3/etc/profile.d/conda.sh"
  fi
  if [[ -n "${conda_sh}" && -f "${conda_sh}" ]]; then
    # shellcheck source=/dev/null
    source "${conda_sh}"
    # Use base when no env active; harmless if already in (base).
    conda activate base 2>/dev/null || true
  fi
}

_fracturex_runtime_ok() {
  local py="$1"
  [[ -x "$py" ]] || return 1
  if [[ "${FEALPY_SKIP_IMPORT_CHECK:-0}" == "1" ]]; then
    return 0
  fi
  "$py" -c "
import numpy
import scipy
from fealpy.backend import backend_manager
" >/dev/null 2>&1
}

_fracturex_runtime_diag() {
  local py="$1"
  [[ -x "$py" ]] || { echo "not executable: $py"; return; }
  "$py" -c "
import sys
print('executable:', sys.executable)
for mod in ('numpy', 'scipy', 'fealpy'):
    try:
        __import__(mod)
        print(mod + ': OK')
    except Exception as e:
        print(mod + ': FAIL', e)
" 2>&1 || true
}

print_fracturex_install_hint() {
  local repo_root="${1:-.}"
  cat >&2 <<EOF
No Python with numpy + scipy + fealpy found.

Install once into the interpreter you want to use (conda base is fine; no venv required):

  cd ${repo_root}
  python3 -m pip install -U pip
  python3 -m pip install fealpy
  python3 -m pip install -e .

Verify:

  python3 -c "import numpy, scipy; from fealpy.backend import backend_manager; print('ok')"

Then rerun (scripts auto-detect python3 / conda):

  bash scripts/paper_huzhang/run_all.sh model0

Optional: export FEALPY_PYTHON=/absolute/path/to/python
EOF
}

resolve_fealpy_python() {
  local repo_root="${1:-}"
  _init_conda_if_needed

  local -a candidates=()

  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    candidates+=("${CONDA_PREFIX}/bin/python" "${CONDA_PREFIX}/bin/python3")
  fi
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi
  if command -v python >/dev/null 2>&1; then
    candidates+=("$(command -v python)")
  fi
  if [[ -n "${FEALPY_PYTHON:-}" ]]; then
    candidates+=("${FEALPY_PYTHON}")
  fi
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    candidates+=("${VIRTUAL_ENV}/bin/python" "${VIRTUAL_ENV}/bin/python3")
  fi
  if [[ -n "${repo_root}" ]]; then
    candidates+=(
      "${repo_root}/.venv/bin/python"
      "${repo_root}/venv/bin/python"
    )
  fi
  candidates+=(
    "${HOME}/venv_fealpy3/bin/python"
    "${HOME}/.venv/bin/python"
  )

  local seen="" cand
  for cand in "${candidates[@]}"; do
    [[ -n "$cand" ]] || continue
    if [[ ":${seen}:" == *":${cand}:"* ]]; then
      continue
    fi
    seen="${seen}:${cand}"
    if _fracturex_runtime_ok "$cand"; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}

# Backward-compatible alias used by older snippets
_fealpy_import_ok() {
  _fracturex_runtime_ok "$1"
}

ensure_fealpy_python() {
  local repo_root="${1:-}"
  _init_conda_if_needed

  if [[ -n "${FEALPY_PYTHON:-}" ]] && ! _fracturex_runtime_ok "${FEALPY_PYTHON}"; then
    echo "resolve_fealpy_python: FEALPY_PYTHON=${FEALPY_PYTHON} missing deps; auto-detecting..." >&2
    _fracturex_runtime_diag "${FEALPY_PYTHON}" >&2
    unset FEALPY_PYTHON
  fi

  if [[ -z "${FEALPY_PYTHON:-}" ]] || ! _fracturex_runtime_ok "${FEALPY_PYTHON}"; then
    if ! FEALPY_PYTHON="$(resolve_fealpy_python "${repo_root}")"; then
      print_fracturex_install_hint "${repo_root}"
      return 1
    fi
    export FEALPY_PYTHON
  fi
  return 0
}

# When sourced: export FEALPY_PYTHON if possible.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  _repo_for_resolve="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  ensure_fealpy_python "${_repo_for_resolve}" || true
fi
