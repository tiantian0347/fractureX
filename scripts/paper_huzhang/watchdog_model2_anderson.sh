#!/usr/bin/env bash
# watchdog_model2_anderson.sh — model2 direct_full (Anderson) 看门狗。
#
# 与 watchdog_model2.sh 同逻辑,但重启走 resume_model2_direct_full_anderson.sh
# (FRACTUREX_ANDERSON_DEPTH=5),保证每次自动重启都重新开启 Anderson,续算行为一致。
# 每 CHECK_INTERVAL 秒检查:
#   - summary.json 存在 → run 已跑到 step200,看门狗退出。
#   - run_case --case model2 进程活着 → 不动。
#   - 进程死了 + 无 summary.json → 从最新 checkpoint 重启(SAVE_EVERY=1 不丢步)。
#
# 用法: nohup bash scripts/paper_huzhang/watchdog_model2_anderson.sh \
#         > /tmp/model2_watchdog_anderson.log 2>&1 &
set -u
REPO=/home/gongshihua/tian/fracturex
cd "$REPO" || exit 1
D=$REPO/results/phasefield/model2_notch_x_stretch/paper_direct_full/epsg_1e-06
RESUME=$REPO/scripts/paper_huzhang/resume_model2_direct_full_anderson.sh
STATUS=$REPO/docs/preconditioner/PIPELINE_STATUS.md
CHECK_INTERVAL=600
ts() { date '+%F %T'; }
note() { printf -- '- [%s] %s\n' "$(ts)" "$1" >> "$STATUS"; }

note "model2 看门狗启动 [ANDERSON] (每 ${CHECK_INTERVAL}s 检查;死则从最新 checkpoint 重启,summary.json 出现即退出)"
restarts=0
while :; do
  if [ -f "$D/summary.json" ]; then
    note "model2 看门狗 [ANDERSON]:检测到 summary.json (run 已完成 step200),退出。共自动重启 ${restarts} 次。"
    break
  fi
  if pgrep -f "run_case.py --case model2" >/dev/null 2>&1; then
    sleep "$CHECK_INTERVAL"; continue
  fi
  laststep=$(python3 -c "import csv;r=list(csv.DictReader(open('$D/history.csv')));print(r[-1]['step'])" 2>/dev/null || echo "?")
  restarts=$((restarts+1))
  note "model2 看门狗 [ANDERSON]:进程不在 + 无 summary.json (history@step${laststep}),第 ${restarts} 次自动从最新 checkpoint 重启。"
  nohup bash "$RESUME" >> /tmp/model2_resume_anderson.log 2>&1 &
  sleep 90
done
