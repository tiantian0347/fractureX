#!/usr/bin/env bash
# Model5 standard FEM (MainSolve) — lab background launcher.
#
# Mesh guide (ℓ₀=0.03 mm):
#   h=0.10  NC≈4k     (done: std_bg_h010_full, ~22 h full schedule)
#   h=0.015 NC≈166k   (~44× cost vs h=0.1)
#   h=0.010 NC≈372k   (~100× cost; recommended for peak/u alignment)
#   h=0.008 NC≈579k   (~156× cost; diminishing returns)
#
# Usage (on lab, from repo root):
#   setsid bash scripts/paper_huzhang/run_model5_std_fem_lab.sh smoke_a \
#     > results/logs/model5_std_fem_h015_smoke_a.nohup 2>&1 < /dev/null &
#   setsid bash scripts/paper_huzhang/run_model5_std_fem_lab.sh smoke \
#     > results/logs/model5_std_fem_h001_smoke.nohup 2>&1 < /dev/null &
#   setsid bash scripts/paper_huzhang/run_model5_std_fem_lab.sh full \
#     > results/logs/model5_std_fem_h001_full.nohup 2>&1 < /dev/null &

set -euo pipefail

MODE="${1:-smoke_a}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-/home/gongshihua/miniconda3/envs/py312/bin/python}"
export PYTHONPATH="${ROOT}:${ROOT}/../fealpy:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Serial only (no OpenMP / BLAS threading).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

MESH=0.015
MAXIT=80
DEG=1

case "$MODE" in
  smoke_a_cont)
    MESH=0.015
    OUT="results/phasefield/model5_standard_fem/std_bg_h015_smoke_a_cont"
    PREV="results/phasefield/model5_standard_fem/std_bg_h015_smoke_a/model5_std_force_disp.txt"
    EXTRA=(
      --u-start 0.03
      --u-end 0.06
      --n-steps 30
      --merge-with "$PREV"
      --save-state-npz "${OUT}/model5_std_state.npz"
    )
    ;;
  smoke_a)
    MESH=0.015
    OUT="results/phasefield/model5_standard_fem/std_bg_h015_smoke_a"
    EXTRA=(--max-steps 30)   # u ≈ 0 → 0.03 mm; bracket Ambati peak region
    ;;
  smoke)
    MESH=0.01
    OUT="results/phasefield/model5_standard_fem/std_bg_h001_smoke"
    EXTRA=(--max-steps 80)   # u ≈ 0 → 0.05 mm; enough to bracket Ambati peak
    ;;
  full)
    OUT="results/phasefield/model5_standard_fem/std_bg_h001_full"
    EXTRA=()
    ;;
  *)
    echo "usage: $0 {smoke_a|smoke_a_cont|smoke|full}" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT" results/logs

echo "[$(date -Iseconds)] model5 std FEM h=${MESH} mode=${MODE} out=${OUT}"

exec "$PY" fracturex/cases/phase_field/model5_three_point_bending.py \
  --mesh-size "$MESH" \
  --degree "$DEG" \
  --maxit "$MAXIT" \
  --outdir "$OUT" \
  --vtkname model5_std \
  "${EXTRA[@]}"
