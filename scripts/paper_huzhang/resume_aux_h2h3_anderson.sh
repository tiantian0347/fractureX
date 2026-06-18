#!/usr/bin/env bash
# resume_aux_h2h3_anderson.sh — 续跑 aux_h2 ‖ aux_h3 至 step30，开 Anderson 加速。
#
# 背景：2026-06-09 无加速续算在裂纹起裂处 staggered 外层发散（h3 step14 振荡 262 次
# error 爬到 1.68 后极限环、h2 step17 卡死）。按 D12 §14 / 路线 B 开 Anderson 破起裂
# 极限环。参数取 run_aux_model0.sh 已验证口径（稳态反力 vs plain AM 差 ≤0.007%）。
# 各自从最新 checkpoint 续（h2@step16, h3@step13；发散的 step14/17 未存盘）。
set -u
REPO=/home/gongshihua/tian/fracturex
PY=/home/gongshihua/miniconda3/envs/py312/bin/python
export PYTHONPATH=$REPO
cd "$REPO" || exit 1
STATUS=$REPO/docs/preconditioner/PIPELINE_STATUS.md
ts() { date '+%F %T'; }
note() { printf -- '- [%s] %s\n' "$(ts)" "$1" >> "$STATUS"; }

# ---- Anderson 加速（D12 §14 验证口径，破起裂极限环）----
export FRACTUREX_ANDERSON_DEPTH=5            # 窗口大小 m>0 开启
export FRACTUREX_ANDERSON_OMEGA=1.0          # 稳态播种 plain，不 over-relax（保反力）
export FRACTUREX_ANDERSON_TR_FACTOR=20       # 信赖域：加速步长 <= tr*||f||，挡过冲
export FRACTUREX_ANDERSON_RESTART_OMEGA=1.6  # 仅 restart 后第一步 over-relax kick

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

note "=== 续跑 aux_h2 ‖ aux_h3 → step30，开 Anderson(depth=5,omega=1,tr=20,restart_omega=1.6) 破起裂发散 ==="
PID_H2=$(launch_aux h2 0.025)
PID_H3=$(launch_aux h3 0.013)
note "PIDs(anderson): h2=$PID_H2 h3=$PID_H3 (resume h2@step16 h3@step13)"
echo "PIDs: h2=$PID_H2 h3=$PID_H3"
