#!/usr/bin/env bash
# Model6 standard FEM (MainSolve) — lab launcher.
#
# Ambati §4.5 beam 20×8, ℓ₀=0.01 mm. Full h≲ℓ₀/2 is millions of cells;
# first production uses h=0.15 (resolves r=0.25 holes) through the ~0.22 mm peak.
#
# Usage (on lab, from repo root):
#   setsid bash scripts/paper_huzhang/run_model6_std_fem_lab.sh smoke \
#     > results/logs/model6_std_fem_smoke.nohup 2>&1 < /dev/null &
#   setsid bash scripts/paper_huzhang/run_model6_std_fem_lab.sh path \
#     > results/logs/model6_std_fem_path.nohup 2>&1 < /dev/null &

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
    MESH=0.4
    OUT="results/phasefield/model6_standard_fem/std_h04_smoke"
    EXTRA=(--max-steps 8)
    ;;
  path)
    MESH=0.15
    OUT="results/phasefield/model6_standard_fem/std_h015_path"
    EXTRA=(--u-start 0 --u-end 0.25 --n-steps 250 --save-state-npz "${OUT}/model6_std_state.npz")
    ;;
  *)
    echo "usage: $0 {smoke|path}" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT" results/logs
echo "[$(date -Iseconds)] model6 std FEM h=${MESH} mode=${MODE} out=${OUT}"

exec "$PY" fracturex/cases/phase_field/model6_asymmetric_beam.py \
  --mesh-size "$MESH" \
  --degree "$DEG" \
  --maxit "$MAXIT" \
  --lin-method gmres \
  --outdir "$OUT" \
  --vtkname model6_std \
  --save-vtk \
  "${EXTRA[@]}"
