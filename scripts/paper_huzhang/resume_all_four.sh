#!/usr/bin/env bash
# resume_all_four.sh — 并行续跑 aux_h2 / aux_h3 / model1 / model2(→step200)
set -u

REPO=/home/gongshihua/tian/fracturex
PY=/home/gongshihua/miniconda3/envs/py312/bin/python
export PYTHONPATH=$REPO
cd "$REPO" || exit 1

STATUS=$REPO/docs/preconditioner/PIPELINE_STATUS.md
M2_STEPS="${FRACTUREX_RUN_NSTEPS:-200}"

ts() { date '+%F %T'; }
note() { printf -- '- [%s] %s\n' "$(ts)" "$1" >> "$STATUS"; }

launch_aux() {
  local tier="$1" hmin="$2"
  local log="$REPO/results/phasefield/model0_circular_notch/paper_aux_${tier}/epsg_1e-06/resume.log"
  FRACTUREX_RESUME=1 FRACTUREX_SAVE_EVERY=1 \
  FRACTUREX_RUN_LABEL_SUFFIX="$tier" FRACTUREX_HMIN="$hmin" \
  MKL_NUM_THREADS=32 OMP_NUM_THREADS=32 \
    nohup "$PY" "$REPO/scripts/paper_huzhang/run_case.py" --case model0 --mode aux --out-root results \
      >> "$log" 2>&1 &
  echo $!
}

launch_model1() {
  local log="$REPO/results/phasefield/square_tension_precrack/paper_direct_full_nx120/epsg_1e-06/resume.log"
  FRACTUREX_RESUME=1 FRACTUREX_SAVE_EVERY=1 \
  FRACTUREX_RUN_LABEL_SUFFIX=full_nx120 FRACTUREX_NX=120 \
  FRACTUREX_ELASTIC_DIRECT_BACKEND=pardiso \
  MKL_NUM_THREADS=48 OMP_NUM_THREADS=48 \
    nohup "$PY" "$REPO/scripts/paper_huzhang/run_case.py" --case square --mode direct --out-root results \
      >> "$log" 2>&1 &
  echo $!
}

launch_model2() {
  local log="$REPO/results/phasefield/model2_notch_x_stretch/paper_direct_full/epsg_1e-06/resume.log"
  FRACTUREX_RESUME=1 FRACTUREX_SAVE_EVERY=1 \
  FRACTUREX_RUN_LABEL_SUFFIX=full \
  FRACTUREX_NX=160 FRACTUREX_N_LOAD_STEPS=240 FRACTUREX_RUN_NSTEPS="$M2_STEPS" \
  FRACTUREX_ELASTIC_DIRECT_BACKEND=pardiso \
  FRACTUREX_ASSEMBLY_NPROC=24 MKL_NUM_THREADS=24 OMP_NUM_THREADS=24 \
    nohup "$PY" "$REPO/scripts/paper_huzhang/run_case.py" --case model2 --mode direct --out-root results \
      >> "$log" 2>&1 &
  echo $!
}

note "=== 四路续跑 (aux_h2 ‖ aux_h3 ‖ model1 ‖ model2→step${M2_STEPS}, SAVE_EVERY=1) ==="
PID_H2=$(launch_aux h2 0.025)
PID_H3=$(launch_aux h3 0.013)
PID_M1=$(launch_model1)
PID_M2=$(launch_model2)
note "PIDs: h2=$PID_H2 h3=$PID_H3 model1=$PID_M1 model2=$PID_M2"
echo "PIDs: h2=$PID_H2 h3=$PID_H3 m1=$PID_M1 m2=$PID_M2"
