#!/usr/bin/env bash
# resume_model2_direct_full_anderson.sh — Anderson-accelerated 续算 (2026-06-14)
#
# 与 resume_model2_direct_full.sh 完全一致(pardiso, nx=160, N_LOAD_STEPS=240,
# RUN_NSTEPS=200, SAVE_EVERY=1, RESUME=1, SUFFIX=full),唯一区别:开启
# Storvik 风格 Anderson 加速(FRACTUREX_ANDERSON_DEPTH=5)。
#
# 动机: step137 单步 449 次 staggered 迭代 / ~8.5h,step138 在完全损伤区
#   (max_d=1.0)外层迭代爬行。Anderson 加速 d 不动点迭代,tol 仍 1e-5,收敛到
#   同一不动点 → reaction 曲线不变,只是迭代数大幅下降(driver 默认 omega=1.0
#   plain reseed,不损反力精度;信赖域 tr_factor=20 + restart 兜底防过冲)。
#   参数取 AndersonAccelerator 自带默认(depth=5,beta=1,omega=1,patience=3)。
#
# 用法: nohup bash scripts/paper_huzhang/resume_model2_direct_full_anderson.sh \
#         > /tmp/model2_full_anderson.log 2>&1 &
set -u

REPO=/home/gongshihua/tian/fracturex
PY=/home/gongshihua/miniconda3/envs/py312/bin/python
export PYTHONPATH=$REPO
cd "$REPO" || exit 1

RUN_DIR=results/phasefield/model2_notch_x_stretch/paper_direct_full/epsg_1e-06
HIST=$RUN_DIR/history.csv
LOG=$RUN_DIR/resume.log
STATUS=$REPO/docs/preconditioner/PIPELINE_STATUS.md

ts()   { date '+%F %T'; }
note() { printf -- '- [%s] %s\n' "$(ts)" "$1" >> "$STATUS"; }

note "model2 direct_full 续算启动 [ANDERSON depth=5] (pardiso, nx=160, RUN_NSTEPS=${FRACTUREX_RUN_NSTEPS:-200}, SAVE_EVERY=1, resume@latest checkpoint)"
FRACTUREX_RESUME=1 FRACTUREX_SAVE_EVERY=1 \
FRACTUREX_RUN_LABEL_SUFFIX=full \
FRACTUREX_NX=160 FRACTUREX_N_LOAD_STEPS=240 FRACTUREX_RUN_NSTEPS="${FRACTUREX_RUN_NSTEPS:-200}" \
FRACTUREX_ELASTIC_DIRECT_BACKEND=pardiso \
FRACTUREX_ANDERSON_DEPTH="${FRACTUREX_ANDERSON_DEPTH:-5}" \
FRACTUREX_ASSEMBLY_NPROC=24 MKL_NUM_THREADS=24 OMP_NUM_THREADS=24 \
  "$PY" scripts/paper_huzhang/run_case.py --case model2 --mode direct --out-root results \
    >> "$LOG" 2>&1 &
JOB=$!
note "model2 direct_full [ANDERSON] PID=$JOB"
echo "model2 direct_full [ANDERSON] launched: PID=$JOB  log=$LOG"

probe() {
  "$PY" - "$HIST" <<'PY' 2>/dev/null
import csv, sys
p = sys.argv[1]
try:
    rows = list(csv.DictReader(open(p)))
except Exception:
    print("rows=0"); sys.exit()
if not rows:
    print("rows=0"); sys.exit()
r = rows[-1]
print("rows=%d step=%s maxd=%.4f Rx=%s dispx=%s" % (
    len(rows), r.get("step","?"), float(r.get("max_d","nan") or "nan"),
    r.get("reaction_x","?"), r.get("disp_x","?")))
PY
}

while kill -0 "$JOB" 2>/dev/null; do
  sleep 1800
  kill -0 "$JOB" 2>/dev/null || break
  note "model2 direct_full [ANDERSON] 进行中 [pid $JOB] $(probe)"
done

note "model2 direct_full [ANDERSON] 进程已退出 [pid $JOB],最终 $(probe)"
# 退出后刷新 reaction_curve.csv 摘要(供 make_model2_figures.py 重绘)。
"$PY" - "$HIST" "$RUN_DIR/reaction_curve.csv" <<'PY' >> /tmp/model2_full_anderson.log 2>&1 && \
  note "model2 reaction 曲线摘要已刷新 $RUN_DIR/reaction_curve.csv" || \
  note "model2 reaction 摘要失败,见 /tmp/model2_full_anderson.log"
import csv, sys
src, dst = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
seen = {}
for r in rows:
    seen[r.get("step","")] = r  # dedup, last writer wins
with open(dst, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "disp_x", "reaction_x", "max_d", "max_H"])
    for k in sorted(seen, key=lambda s: int(s) if s.isdigit() else 1<<30):
        r = seen[k]
        w.writerow([r.get("step",""), r.get("disp_x",""), r.get("reaction_x",""),
                    r.get("max_d",""), r.get("max_H","")])
PY
