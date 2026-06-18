#!/usr/bin/env bash
# resume_aux_h2_r200.sh — 从 step15 干净 checkpoint 续 aux_h2，restart=200 + Anderson。
#
# 背景：上次续算(PID 1065680，旧内存 restart=60 版)在完全分离区(step17-30,maxd=1.0)
# 产出非物理反力(翻正爬到+102)——restart=60 在分离后奇异非正规鞍点上重启停滞。
# d12_recheck 证实同算子 restart=200 收敛(step17: r60→400/DNF, r200→25/conv)。
# step15(maxd=0.998)是最后一个 aux≡direct 的干净 checkpoint(反力差4e-4)；step16起偏离。
# 从 step15 续、显式 restart=200，重算 step16→30 拿物理正确反力。
# step16-30 旧(r60垃圾)checkpoint 已移到 /tmp/aux_h2_r60_ckpt_bak/；resume 会自动
# 把 history/iterations 截断到 step15 再续写。
set -u
REPO=/home/gongshihua/tian/fracturex
PY=/home/gongshihua/miniconda3/envs/py312/bin/python
export PYTHONPATH=$REPO
cd "$REPO" || exit 1
STATUS=$REPO/docs/preconditioner/PIPELINE_STATUS.md
LOG=$REPO/results/phasefield/model0_circular_notch/paper_aux_h2/epsg_1e-06/resume.log
ts() { date '+%F %T'; }
note() { printf -- '- [%s] %s\n' "$(ts)" "$1" >> "$STATUS"; }

# Anderson 加速（破起裂极限环；D12 §14 验证口径）
export FRACTUREX_ANDERSON_DEPTH=5
export FRACTUREX_ANDERSON_OMEGA=1.0
export FRACTUREX_ANDERSON_TR_FACTOR=20
export FRACTUREX_ANDERSON_RESTART_OMEGA=1.6
# 显式 restart=200（默认已 200，显式以防环境覆盖）；maxit=400
export FRACTUREX_GMRES_RESTART=200
export FRACTUREX_GMRES_MAXIT=400

note "=== 续跑 aux_h2 从 step15 → step30，restart=200 + Anderson（修 restart=60 分离区垃圾反力）==="
FRACTUREX_RESUME=1 FRACTUREX_SAVE_EVERY=1 \
FRACTUREX_RUN_LABEL_SUFFIX=h2 FRACTUREX_HMIN=0.025 \
MKL_NUM_THREADS=32 OMP_NUM_THREADS=32 \
  nohup "$PY" "$REPO/scripts/paper_huzhang/run_case.py" --case model0 --mode aux --out-root results \
    >> "$LOG" 2>&1 &
PID=$!
note "aux_h2 r200 PID=$PID (resume@step15, restart=200, Anderson on)"
echo "PID=$PID"
