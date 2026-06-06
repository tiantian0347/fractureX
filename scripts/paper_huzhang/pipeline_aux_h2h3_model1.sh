#!/usr/bin/env bash
# pipeline_aux_h2h3_model1.sh  —  PARALLEL edition (2026-06-04)
#
# 无人值守并行流水线。三段同时后台跑(机器 2TB RAM/176核,三者峰值 RSS 合计
# ~25GB,无 OOM 风险,串行只白白浪费墙钟):
#   run A  model0 aux_h2  (hmin=0.025, NC=2868,  σ-dof 48k,  resume@step10)
#   run B  model0 aux_h3  (hmin=0.013, NC=11034, σ-dof 184k, resume@step10)
#   run C  model1/square  (nx=120,     NC=28800, σ-dof 477k, pardiso, resume@step50)
#
# 关键修复 (vs. 旧串行版):
#   - FRACTUREX_SAVE_EVERY=1 : 每个 load step 都存 checkpoint。旧版默认 10,
#     导致 h2 跑到 step15(已穿局部化墙,step14/15 各烧 11h/32h)却只有 step10
#     的 checkpoint,重启重烧 ~43h。以后绝不再丢局部化步。
#   - 三段并行,各自独立日志,互不阻塞。
#   - 末尾 monitor 循环:每 30min 把三段进度(history 行数 / 最新 maxd / niter /
#     是否存活)写进 PIPELINE_STATUS.md,并在某段完成时重绘对应图。
#
# 用法: nohup bash scripts/paper_huzhang/pipeline_aux_h2h3_model1.sh \
#         > /tmp/pipeline.log 2>&1 &
set -u

REPO=/home/gongshihua/tian/fracturex
PY=/home/gongshihua/miniconda3/envs/py312/bin/python
export PYTHONPATH=$REPO
cd "$REPO" || exit 1

CASEDIR=results/phasefield/model0_circular_notch
SQDIR=results/phasefield/square_tension_precrack
STATUS=$REPO/docs/preconditioner/PIPELINE_STATUS.md

H2_DIR=$CASEDIR/paper_aux_h2/epsg_1e-06
H3_DIR=$CASEDIR/paper_aux_h3/epsg_1e-06
M1_DIR=$SQDIR/paper_direct_full_nx120/epsg_1e-06

ts() { date '+%F %T'; }
note() { printf -- '- [%s] %s\n' "$(ts)" "$1" >> "$STATUS"; }

# ---- launchers (each backgrounded) -----------------------------------------
launch_aux() {  # $1=tier $2=hmin
  local tier="$1" hmin="$2" log="$CASEDIR/paper_aux_${1}/epsg_1e-06/resume.log"
  note "aux_${tier} 启动 (hmin=$hmin, SAVE_EVERY=1, resume@checkpoint)"
  FRACTUREX_RESUME=1 FRACTUREX_SAVE_EVERY=1 \
  FRACTUREX_RUN_LABEL_SUFFIX="$tier" FRACTUREX_HMIN="$hmin" \
  MKL_NUM_THREADS=32 OMP_NUM_THREADS=32 \
    "$PY" scripts/paper_huzhang/run_case.py --case model0 --mode aux --out-root results \
      >> "$log" 2>&1 &
  echo $!
}

launch_model1() {
  local log="$M1_DIR/resume.log"
  note "model1 nx=120 启动 (pardiso, SAVE_EVERY=1, resume@step50)"
  FRACTUREX_RESUME=1 FRACTUREX_SAVE_EVERY=1 \
  FRACTUREX_RUN_LABEL_SUFFIX=full_nx120 FRACTUREX_NX=120 \
  FRACTUREX_ELASTIC_DIRECT_BACKEND=pardiso \
  MKL_NUM_THREADS=48 OMP_NUM_THREADS=48 \
    "$PY" scripts/paper_huzhang/run_case.py --case square --mode direct --out-root results \
      >> "$log" 2>&1 &
  echo $!
}

# ---- figure hooks ----------------------------------------------------------
fig_model0() {
  "$PY" scripts/paper_huzhang/make_model0_figures.py >> /tmp/pipeline.log 2>&1 \
    && note "model0 图已重绘 (Frac_huzhang/figures/model0_loaddisp.*)" \
    || note "model0 绘图失败,见 /tmp/pipeline.log"
}
fig_model1() {
  FRACTUREX_MODEL1_RUN="$M1_DIR" \
    "$PY" scripts/paper_huzhang/make_model1_figures.py >> /tmp/pipeline.log 2>&1 \
    && note "model1 图已重绘 (Frac_huzhang/figures/model1_loaddisp.*)" \
    || note "model1 绘图失败,见 /tmp/pipeline.log"
}

# ---- progress probe --------------------------------------------------------
probe() {  # $1=name $2=history.csv  -> "rows=N maxd=.. niter=.. load=.."
  local name="$1" csv="$2"
  "$PY" - "$csv" <<'PY' 2>/dev/null
import csv, sys
p = sys.argv[1]
try:
    rows = list(csv.reader(open(p)))
except FileNotFoundError:
    print("rows=0 (no history)"); sys.exit()
if len(rows) < 2:
    print("rows=0"); sys.exit()
h, last = rows[0], rows[-1]
def col(n):
    try: return last[h.index(n)]
    except Exception: return "?"
print("rows=%d step=%s maxd=%.4f niter_e=%s load=%s" % (
    len(rows)-1, col("step"), float(col("max_d")), col("linear_niter_elastic"), col("load")))
PY
}

# ============================================================================
note "=== 并行流水线启动 (aux_h2 ‖ aux_h3 ‖ model1, SAVE_EVERY=1) ==="
PID_H2=$(launch_aux h2 0.025)
PID_H3=$(launch_aux h3 0.013)
PID_M1=$(launch_model1)
note "PIDs: h2=$PID_H2 h3=$PID_H3 model1=$PID_M1"
echo "[$(ts)] launched h2=$PID_H2 h3=$PID_H3 m1=$PID_M1"

# ---- monitor loop ----------------------------------------------------------
declare -A DONE=()
while :; do
  sleep 1800   # 30 min
  alive_any=0
  for tag in h2 h3 m1; do
    case $tag in
      h2) pid=$PID_H2; csv=$H2_DIR/history.csv; name="aux_h2";;
      h3) pid=$PID_H3; csv=$H3_DIR/history.csv; name="aux_h3";;
      m1) pid=$PID_M1; csv=$M1_DIR/history.csv; name="model1";;
    esac
    if kill -0 "$pid" 2>/dev/null; then
      alive_any=1
      note "$name 进行中 [pid $pid] $(probe "$name" "$csv")"
    elif [ -z "${DONE[$tag]:-}" ]; then
      DONE[$tag]=1
      note "$name 已结束 [pid $pid] $(probe "$name" "$csv")"
      case $tag in
        h2|h3) fig_model0;;
        m1)    fig_model1;;
      esac
    fi
  done
  [ "$alive_any" -eq 0 ] && break
done

note "=== 并行流水线全部结束 ==="
echo "[$(ts)] pipeline done"
