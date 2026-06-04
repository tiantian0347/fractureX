#!/usr/bin/env bash
# resume_aux_h2h3_serial.sh
#
# 错峰串行续算 model0 aux h2 -> h3(绝不并行,防 OOM)。
# 整机 2026-06-04 重启后所有 run 被清,checkpoint 仍在(step_010.npz)。
#   h2: hmin=0.025 -> NC=2868  (验证匹配 checkpoint)
#   h3: hmin=0.013 -> NC=11034 (resume 自核 NC,mismatch 直接 raise)
# aux 默认走 fast 两层 V-cycle(FRACTUREX_ELASTIC_FAST 不设=默认 fast,niter≈6;
# 但裂纹完全局部化处 niter 会 7->~121,单步几十小时,这是算法硬墙)。
#
# 用法: bash resume_aux_h2h3_serial.sh   (建议 nohup ... & 后台)
# 日志: results/.../paper_aux_h{2,3}/epsg_1e-06/resume.log
set -u

REPO=/home/gongshihua/tian/fracturex
PY=/home/gongshihua/miniconda3/envs/py312/bin/python
export PYTHONPATH=$REPO
cd "$REPO" || exit 1

CASEDIR=results/phasefield/model0_circular_notch
# 内存哨兵阈值(KiB):可用内存低于此值则中止,防 OOM 卡死整机(2TB 机器留 100GB)
MIN_AVAIL_KB=$((100 * 1024 * 1024))

mem_guard() {
  local avail
  avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "$avail" -lt "$MIN_AVAIL_KB" ]; then
    echo "[mem-guard] MemAvailable=$((avail/1024/1024))GB < 100GB 阈值,中止以防 OOM" >&2
    return 1
  fi
  echo "[mem-guard] MemAvailable=$((avail/1024/1024))GB,OK"
  return 0
}

run_tier() {
  local tier="$1" hmin="$2"
  local logdir="$CASEDIR/paper_aux_${tier}/epsg_1e-06"
  local log="$logdir/resume.log"
  echo "================================================================"
  echo "[$(date '+%F %T')] 启动 aux_${tier} (hmin=$hmin) 续算"
  echo "  日志: $log"
  mem_guard || return 1
  # aux 续算:RESUME=1 从 step_010 续;fast 路径默认开;pardiso 不涉及(aux 走 GMRES)
  FRACTUREX_RESUME=1 \
  FRACTUREX_RUN_LABEL_SUFFIX="$tier" \
  FRACTUREX_HMIN="$hmin" \
  MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 \
    "$PY" scripts/paper_huzhang/run_case.py \
      --case model0 --mode aux --out-root results \
      >> "$log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] aux_${tier} 退出码=$rc"
  return $rc
}

echo "[$(date '+%F %T')] === 错峰串行续算开始 (h2 -> h3) ==="
run_tier h2 0.025
rc2=$?
if [ "$rc2" -ne 0 ]; then
  echo "[$(date '+%F %T')] aux_h2 失败(rc=$rc2),不启动 h3。检查 resume.log"
  exit "$rc2"
fi
echo "[$(date '+%F %T')] aux_h2 完成,排队启动 h3"
run_tier h3 0.013
rc3=$?
echo "[$(date '+%F %T')] === 全部结束 (h2 rc=$rc2, h3 rc=$rc3) ==="
