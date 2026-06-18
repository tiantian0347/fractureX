#!/usr/bin/env bash
# watchdog_model2.sh — model2 direct_full 看门狗：进程死了且未完成则自动从最新 checkpoint 重启。
#
# 背景：model2 direct run(nx=160, σ-DOF 847k)在 max_d=1.0 完全损伤区 staggered 外层
# 数百次迭代/步，已多次被外部静默终止(无 Traceback/OOM 日志)。看门狗每 N 分钟检查一次：
#   - summary.json 存在 → run 已正常跑到 step200，看门狗退出(任务完成)。
#   - 进程活着 → 啥也不做。
#   - 进程死了 + 无 summary.json → 从最新 checkpoint 重启(SAVE_EVERY=1 保证不丢步)。
# 重启复用 resume_model2_direct_full.sh(已 pin nx=160 + N_LOAD_STEPS=240 + RUN_NSTEPS=200)。
#
# 用法: nohup bash scripts/paper_huzhang/watchdog_model2.sh > /tmp/model2_watchdog.log 2>&1 &
set -u
REPO=/home/gongshihua/tian/fracturex
cd "$REPO" || exit 1
D=$REPO/results/phasefield/model2_notch_x_stretch/paper_direct_full/epsg_1e-06
RESUME=$REPO/scripts/paper_huzhang/resume_model2_direct_full.sh
STATUS=$REPO/docs/preconditioner/PIPELINE_STATUS.md
CHECK_INTERVAL=600   # 10 min
ts() { date '+%F %T'; }
note() { printf -- '- [%s] %s\n' "$(ts)" "$1" >> "$STATUS"; }

note "model2 看门狗启动 (每 ${CHECK_INTERVAL}s 检查；死则从最新 checkpoint 重启，summary.json 出现即退出)"
restarts=0
while :; do
  if [ -f "$D/summary.json" ]; then
    note "model2 看门狗：检测到 summary.json (run 已完成 step200)，看门狗退出。共自动重启 ${restarts} 次。"
    break
  fi
  # 是否有 model2 run_case 进程在跑
  if pgrep -f "run_case.py --case model2" >/dev/null 2>&1; then
    sleep "$CHECK_INTERVAL"; continue
  fi
  # 进程死了且没完成 → 重启
  laststep=$(python3 -c "import csv;r=list(csv.DictReader(open('$D/history.csv')));print(r[-1]['step'])" 2>/dev/null || echo "?")
  restarts=$((restarts+1))
  note "model2 看门狗：进程不在 + 无 summary.json (history@step${laststep})，第 ${restarts} 次自动从最新 checkpoint 重启。"
  nohup bash "$RESUME" >> /tmp/model2_resume.log 2>&1 &
  sleep 90   # 给它时间起来(resume + 装配)，避免下一轮误判又重启
done
