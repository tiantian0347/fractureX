#!/usr/bin/env bash
# resume_model2_direct_full.sh  —  续算 model2 完整 reaction-force 曲线 (2026-06-06)
#
# model2 (notch x-stretch) direct/pardiso 的 paper_direct_full run 之前只跑到
# step59/240(disp_x=0.0059)就停了,reaction-force 曲线不完整。本脚本从最新
# checkpoint 续算到 200 步(默认; 可设 FRACTUREX_RUN_NSTEPS=240 跑满),
# 拿到 reaction_x(disp_x) 曲线。
#
# 求解器选型: 2D 算例 pardiso direct 最快(记忆 aux_loses_to_pardiso_2d:
#   aux fast 在 2D 上时间输给 pardiso 11-14x),原 run 也是 solve_direct_pardiso
#   (niter=1)。不改求解器。
#
# 关键: FRACTUREX_SAVE_EVERY=1 每步存盘(原 run 是 10,checkpoint 只到 step50,
#   故 step51-59 会重算;以后绝不再丢)。FRACTUREX_RESUME=1 从 step50 续。
#   *必须* pin FRACTUREX_NX=160 + FRACTUREX_N_LOAD_STEPS=240: 原 run 之后这两个
#   默认值被改过(nx 160→216 NC 51200→93312, n_load_steps 240→2400),不 pin 会
#   NC 不匹配而拒绝续算。tag = paper_direct + SUFFIX=full,pardiso 后端,
#   与原 run 完全一致(已离线验证 NC=51200 / du=1e-4 / loads[59]=0.0059)。
#
# 机器已满载(model0 aux x2 + square direct 在跑),故 assembly/MKL 线程压到 24,
#   避免抢占正在跑的论文 job(pardiso solve 才是瓶颈,assembly 只是间歇峰值)。
#
# 用法: nohup bash scripts/paper_huzhang/resume_model2_direct_full.sh \
#         > /tmp/model2_full.log 2>&1 &
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

# ---- launch (backgrounded) -------------------------------------------------
note "model2 direct_full 续算启动 (pardiso, nx=160, RUN_NSTEPS=${FRACTUREX_RUN_NSTEPS:-200}, SAVE_EVERY=1, resume@latest checkpoint)"
FRACTUREX_RESUME=1 FRACTUREX_SAVE_EVERY=1 \
FRACTUREX_RUN_LABEL_SUFFIX=full \
FRACTUREX_NX=160 FRACTUREX_N_LOAD_STEPS=240 FRACTUREX_RUN_NSTEPS="${FRACTUREX_RUN_NSTEPS:-200}" \
FRACTUREX_ELASTIC_DIRECT_BACKEND=pardiso \
FRACTUREX_ASSEMBLY_NPROC=24 MKL_NUM_THREADS=24 OMP_NUM_THREADS=24 \
  "$PY" scripts/paper_huzhang/run_case.py --case model2 --mode direct --out-root results \
    >> "$LOG" 2>&1 &
JOB=$!
note "model2 direct_full PID=$JOB"
echo "model2 direct_full launched: PID=$JOB  log=$LOG"

# ---- progress probe --------------------------------------------------------
probe() {  # -> "rows=N step=.. maxd=.. Rx=.. dispx=.. niter=.."
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
print("rows=%d step=%s maxd=%.4f Rx=%s dispx=%s niter_e=%s" % (
    len(rows), r.get("step","?"), float(r.get("max_d","nan") or "nan"),
    r.get("reaction_x","?"), r.get("disp_x","?"), r.get("linear_niter_elastic","?")))
PY
}

# ---- monitor loop: 每 30min 写进度,进程退出后重绘图并收尾 ------------------
while kill -0 "$JOB" 2>/dev/null; do
  sleep 1800
  kill -0 "$JOB" 2>/dev/null || break
  note "model2 direct_full 进行中 [pid $JOB] $(probe)"
done

note "model2 direct_full 进程已退出 [pid $JOB],最终 $(probe)"
# 尚无 make_model2_figures.py;退出后把完整 reaction_x(disp_x) 曲线落一份 CSV
# 摘要到 run 目录,便于直接核对峰值载荷与软化段(绘图脚本后续再补)。
"$PY" - "$HIST" "$RUN_DIR/reaction_curve.csv" <<'PY' >> /tmp/model2_full.log 2>&1 && \
  note "model2 reaction 曲线摘要已写 $RUN_DIR/reaction_curve.csv" || \
  note "model2 reaction 摘要失败,见 /tmp/model2_full.log"
import csv, sys
src, dst = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
with open(dst, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "disp_x", "reaction_x", "max_d", "max_H"])
    for r in rows:
        w.writerow([r.get("step",""), r.get("disp_x",""), r.get("reaction_x",""),
                    r.get("max_d",""), r.get("max_H","")])
Rx = [abs(float(r["reaction_x"])) for r in rows if r.get("reaction_x") not in (None, "", "nan")]
if Rx:
    import builtins
    pk = builtins.max(Rx)
    print("peak |reaction_x| = %.6g  (over %d steps)" % (pk, len(rows)))
PY
