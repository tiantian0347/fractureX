#!/usr/bin/env bash
# resume_aux_h2_noanderson.sh — 从 step15 续 aux_h2，restart=200，关 Anderson（诊断）。
#
# 背景：从 step15 续(restart=200+Anderson)跑 17h，step16(maxd=1.0 近奇异)仍 DNF
# (restart=200 打满 maxit=400 残差0.46、staggered 外层震荡 error 0.4-5.6)。
# 假设：Anderson 把 d 场外推到更病态状态，反而害了弹性鞍点。本脚本关 Anderson
# (DEPTH=0，纯朴素 staggered + 欠松弛)，看 step16 是否恢复收敛。
set -u
REPO=/home/gongshihua/tian/fracturex
PY=/home/gongshihua/miniconda3/envs/py312/bin/python
export PYTHONPATH=$REPO
cd "$REPO" || exit 1
STATUS=$REPO/docs/preconditioner/PIPELINE_STATUS.md
LOG=$REPO/results/phasefield/model0_circular_notch/paper_aux_h2/epsg_1e-06/resume.log
ts() { date '+%F %T'; }
note() { printf -- '- [%s] %s\n' "$(ts)" "$1" >> "$STATUS"; }

# Anderson OFF（DEPTH=0 = legacy 朴素 staggered 路径）
export FRACTUREX_ANDERSON_DEPTH=0
# restart=200 保留
export FRACTUREX_GMRES_RESTART=200
export FRACTUREX_GMRES_MAXIT=400

note "=== aux_h2 从 step15 续，restart=200，**关 Anderson**（诊断 step16 DNF 是否 Anderson 致病态）==="
FRACTUREX_RESUME=1 FRACTUREX_SAVE_EVERY=1 \
FRACTUREX_RUN_LABEL_SUFFIX=h2 FRACTUREX_HMIN=0.025 \
MKL_NUM_THREADS=32 OMP_NUM_THREADS=32 \
  nohup "$PY" "$REPO/scripts/paper_huzhang/run_case.py" --case model0 --mode aux --out-root results \
    >> "$LOG" 2>&1 &
PID=$!
note "aux_h2 noAnderson PID=$PID (resume@step15, restart=200, ANDERSON_DEPTH=0)"
echo "PID=$PID"
