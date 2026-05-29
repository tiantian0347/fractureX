#!/usr/bin/env bash
# Parameter sweep for the D12 preconditioner paper.
#
# Drives ``python -m fracturex.tests.precond_sweep`` over the matrix described
# in docs/D12_PRECONDITIONER_PAPER_PLAN.md §4. Reuses the env-bootstrap from
# scripts/paper_huzhang/env.sh so no virtualenv activation is needed.
#
# Usage from repo root:
#   bash scripts/paper_precond/run_sweep.sh                    # full matrix
#   bash scripts/paper_precond/run_sweep.sh --quick            # tiny smoke
#   bash scripts/paper_precond/run_sweep.sh --case model0 \    # restrict
#       --algorithms aux_weighted,direct \
#       --hmins 0.04,0.02
#
# Output: $FRACTUREX_RESULTS_ROOT/paper_precond/sweep.csv (header + rows).

set -euo pipefail

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${_REPO_ROOT}/scripts/paper_huzhang/env.sh"

OUT_DIR="${FRACTUREX_RESULTS_ROOT}/paper_precond"
mkdir -p "${OUT_DIR}"
CSV_OUT="${OUT_DIR}/sweep.csv"
LOG_DIR="${FRACTUREX_PAPER_LOG_DIR}/paper_precond"
mkdir -p "${LOG_DIR}"

# -------- defaults (paper §4 full matrix) --------
CASES_DEFAULT="model0"                          # extend after adding builders
ALGOS_DEFAULT="direct,ilu_gmres,aux_unweighted,aux_weighted"
FORMS_DEFAULT="standard,effective_stress"
HMINS_DEFAULT="0.04,0.02,0.01,0.005"
L0S_DEFAULT="2e-3,1e-3,5e-4"
EPSGS_DEFAULT="1e-3,1e-6,1e-9"
MAXDS_DEFAULT="0.1,0.5,0.9,0.99,0.999"

# -------- quick smoke profile (one cell per axis) --------
QUICK=0
CASES="${CASES_DEFAULT}"
ALGOS="${ALGOS_DEFAULT}"
FORMS="${FORMS_DEFAULT}"
HMINS="${HMINS_DEFAULT}"
L0S="${L0S_DEFAULT}"
EPSGS="${EPSGS_DEFAULT}"
MAXDS="${MAXDS_DEFAULT}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)        QUICK=1; shift ;;
    --case)         CASES="$2"; shift 2 ;;
    --algorithms)   ALGOS="$2"; shift 2 ;;
    --formulations) FORMS="$2"; shift 2 ;;
    --hmins)        HMINS="$2"; shift 2 ;;
    --l0s)          L0S="$2"; shift 2 ;;
    --eps-gs)       EPSGS="$2"; shift 2 ;;
    --max-ds)       MAXDS="$2"; shift 2 ;;
    --csv)          CSV_OUT="$2"; shift 2 ;;
    -h|--help)      grep "^# " "$0" | sed 's/^# //'; exit 0 ;;
    *)              echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ "${QUICK}" == "1" ]]; then
  HMINS="0.04,0.02"
  L0S="1e-3"
  EPSGS="1e-6"
  MAXDS="0.1,0.9"
  ALGOS="direct,aux_weighted"
  FORMS="standard"
fi

# Fresh CSV (keep header consistent across rerun)
: > "${CSV_OUT}"

IFS=',' read -r -a CASE_ARR  <<< "${CASES}"
IFS=',' read -r -a ALGO_ARR  <<< "${ALGOS}"
IFS=',' read -r -a FORM_ARR  <<< "${FORMS}"
IFS=',' read -r -a HMIN_ARR  <<< "${HMINS}"
IFS=',' read -r -a L0_ARR    <<< "${L0S}"
IFS=',' read -r -a EPSG_ARR  <<< "${EPSGS}"
IFS=',' read -r -a MAXD_ARR  <<< "${MAXDS}"

TOTAL=$(( ${#CASE_ARR[@]} * ${#ALGO_ARR[@]} * ${#FORM_ARR[@]} * \
          ${#HMIN_ARR[@]} * ${#L0_ARR[@]} * ${#EPSG_ARR[@]} * ${#MAXD_ARR[@]} ))
echo "[paper_precond] total combinations: ${TOTAL}"
echo "[paper_precond] csv:                ${CSV_OUT}"
echo "[paper_precond] log dir:            ${LOG_DIR}"

COUNTER=0
for CASE in "${CASE_ARR[@]}"; do
for FORM in "${FORM_ARR[@]}"; do
for HMIN in "${HMIN_ARR[@]}"; do
for L0   in "${L0_ARR[@]}"; do
for EPSG in "${EPSG_ARR[@]}"; do
for MAXD in "${MAXD_ARR[@]}"; do
for ALGO in "${ALGO_ARR[@]}"; do
  COUNTER=$(( COUNTER + 1 ))
  TAG="${CASE}_${ALGO}_${FORM}_h${HMIN}_l${L0}_eg${EPSG}_d${MAXD}"
  LOG="${LOG_DIR}/${TAG}.log"
  printf "[%5d/%5d] %s\n" "${COUNTER}" "${TOTAL}" "${TAG}"
  if ! "${FRACTUREX_PYTHON}" -m fracturex.tests.precond_sweep \
        --case "${CASE}" \
        --algorithm "${ALGO}" \
        --formulation "${FORM}" \
        --hmin "${HMIN}" \
        --l0 "${L0}" \
        --eps-g "${EPSG}" \
        --max-d "${MAXD}" \
        --csv-out "${CSV_OUT}" \
        >"${LOG}" 2>&1; then
    echo "  -> failed; see ${LOG}"
  fi
done
done
done
done
done
done
done

echo "[paper_precond] done. CSV: ${CSV_OUT}"
