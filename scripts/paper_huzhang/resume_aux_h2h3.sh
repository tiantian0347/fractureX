#!/usr/bin/env bash
# resume_aux_h2h3.sh — 续跑 aux_h2 ‖ aux_h3 至 step30 (model1 已完成,不在此列)
set -u
REPO=/home/gongshihua/tian/fracturex
PY=/home/gongshihua/miniconda3/envs/py312/bin/python
export PYTHONPATH=$REPO
cd "$REPO" || exit 1
STATUS=$REPO/docs/preconditioner/PIPELINE_STATUS.md
ts() { date '+%F %T'; }
note() { printf -- '- [%s] %s\n' "$(ts)" "$1" >> "$STATUS"; }

launch_aux() {  # $1=tier $2=hmin
  local tier="$1" hmin="$2"
  local log="$REPO/results/phasefield/model0_circular_notch/paper_aux_${tier}/epsg_1e-06/resume.log"
  FRACTUREX_RESUME=1 FRACTUREX_SAVE_EVERY=1 \
  FRACTUREX_RUN_LABEL_SUFFIX="$tier" FRACTUREX_HMIN="$hmin" \
  MKL_NUM_THREADS=32 OMP_NUM_THREADS=32 \
    nohup "$PY" "$REPO/scripts/paper_huzhang/run_case.py" --case model0 --mode aux --out-root results \
      >> "$log" 2>&1 &
  echo $!
}

note "=== 续跑 aux_h2 ‖ aux_h3 → step30 (model1 已完成排除, SAVE_EVERY=1) ==="
PID_H2=$(launch_aux h2 0.025)
PID_H3=$(launch_aux h3 0.013)
note "PIDs: h2=$PID_H2 h3=$PID_H3"
echo "PIDs: h2=$PID_H2 h3=$PID_H3"
