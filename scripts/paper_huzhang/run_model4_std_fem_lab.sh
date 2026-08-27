#!/usr/bin/env bash
# Model4 standard FEM (MainSolve) — lab launcher.
#
# Ambati §4.6 plate 65×120, ℓ₀=0.1 mm. Full h≲ℓ₀/2 is millions of cells;
# first production uses h=1.0 (coarse path) with Δu=0.01 mm to 2 mm.
#
# Usage (on lab, from repo root):
#   setsid bash scripts/paper_huzhang/run_model4_std_fem_lab.sh smoke \
#     > results/logs/model4_std_fem_smoke.nohup 2>&1 < /dev/null &
#   setsid bash scripts/paper_huzhang/run_model4_std_fem_lab.sh path \
#     > results/logs/model4_std_fem_path.nohup 2>&1 < /dev/null &

set -euo pipefail

MODE="${1:-path}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-/home/gongshihua/miniconda3/envs/py312/bin/python}"
export PYTHONPATH="${ROOT}:${ROOT}/../fealpy:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

MAXIT=80
DEG=1

case "$MODE" in
  smoke)
    MESH=2.0
    OUT="results/phasefield/model4_standard_fem/std_h2_smoke"
    EXTRA=(--max-steps 8)
    ;;
  path)
    MESH=1.0
    OUT="results/phasefield/model4_standard_fem/std_h1_path"
    EXTRA=(--u-start 0 --u-end 2.0 --n-steps 200 --save-state-npz "${OUT}/model4_std_state.npz")
    ;;
  *)
    echo "usage: $0 {smoke|path}" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT" results/logs
echo "[$(date -Iseconds)] model4 std FEM h=${MESH} mode=${MODE} out=${OUT}"

exec "$PY" fracturex/cases/phase_field/model4_notched_plate.py \
  --mesh-size "$MESH" \
  --degree "$DEG" \
  --maxit "$MAXIT" \
  --lin-method gmres \
  --outdir "$OUT" \
  --vtkname model4_std \
  --save-vtk \
  "${EXTRA[@]}"
