#!/usr/bin/env bash
# Launch a debug aux-space model0 run with custom inner GMRES tolerances,
# writing to its own results directory so it does not clobber an in-flight
# production run.
#
# Motivated by the aux_h1 staggered-divergence incident on 2026-05-27 (see
# /home/gongshihua/tian/Frac_huzhang/PAPER_REVISION_PLAN.md §7.7): error_d at
# step 26 grew geometrically while the inner GMRES kept hitting maxit=200 with
# true_relres ~ 1.9e-8. Hypothesis: residual noise at rtol=1e-8 is amplified by
# the outer staggered fixed point. This script lets us sweep rtol/maxit cheaply.
#
# Usage:
#   debug_aux_rtol.sh <h_tag> <hmin> <rtol> [maxit] [restart]
#
# Example (re-run h1 with rtol=1e-10):
#   bash scripts/paper_huzhang/debug_aux_rtol.sh h1 0.05 1e-10
#
# The output goes to:
#   results/phasefield/model0_circular_notch/paper_aux_<h_tag>_dbg<rtol_tag>/
#   results/logs/dbg_model0_aux_<h_tag>_<rtol_tag>.log
#
# (The companion p2 production run is untouched.)

set -euo pipefail

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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
RTOL="${3:-}"
MAXIT="${4:-200}"
RESTART="${5:-60}"

if [[ -z "${H_TAG}" || -z "${HMIN}" || -z "${RTOL}" ]]; then
  echo "Usage: $0 <h_tag> <hmin> <rtol> [maxit=200] [restart=60]" >&2
  echo "  Example: $0 h1 0.05 1e-10" >&2
  exit 2
fi

# Build an rtol tag like "1em10" for filesystem-friendly naming.
# e.g. 1e-10 -> 1em10, 1.5e-9 -> 1d5em9
RTOL_TAG="$(echo "${RTOL}" | sed -e 's/-/m/g' -e 's/+/p/g' -e 's/\./d/g')"
LABEL_SUFFIX="${H_TAG}_dbg${RTOL_TAG}"
SLUG="dbg_model0_aux_${H_TAG}_${RTOL_TAG}"

LOG_DIR="${FRACTUREX_PAPER_LOG_DIR:-${FRACTUREX_RESULTS_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

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
export FRACTUREX_RUN_LABEL_SUFFIX="${LABEL_SUFFIX}"
export FRACTUREX_ELASTIC_FAST=0
export FRACTUREX_GMRES_RTOL="${RTOL}"
export FRACTUREX_GMRES_MAXIT="${MAXIT}"
export FRACTUREX_GMRES_RESTART="${RESTART}"

OUT_ROOT="${FRACTUREX_PAPER_ROOT:-${FRACTUREX_RESULTS_ROOT}}"

date -u +"%Y-%m-%dT%H:%M:%SZ" > "${STARTED}"
echo running > "${STATUS}"
rm -f "${EXIT_FILE}"

(
  "${FRACTUREX_PYTHON}" \
    "${_REPO_ROOT}/scripts/paper_huzhang/run_case.py" \
    --case model0 --mode aux --out-root "${OUT_ROOT}" \
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

NEW_PID=""
for _try in 1 2 3 4 5 6 7 8 9 10; do
  if [[ -s "${PID_FILE}" ]]; then
    NEW_PID="$(<"${PID_FILE}")"
    break
  fi
  sleep 0.2
done

echo "Launched ${SLUG} pid=${NEW_PID:-?}"
echo "  hmin=${HMIN}  rtol=${RTOL}  maxit=${MAXIT}  restart=${RESTART}"
echo "  label_suffix=${LABEL_SUFFIX}"
echo "  log:    ${LOG}"
echo "  status: ${STATUS}"
echo "  exit:   ${EXIT_FILE} (written on completion)"
echo "  out:    ${OUT_ROOT}/phasefield/model0_circular_notch/paper_aux_${LABEL_SUFFIX}/"
