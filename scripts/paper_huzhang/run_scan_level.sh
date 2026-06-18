#!/usr/bin/env bash
# 受控扫描:同一 case 同一网格级,direct(pardiso) 与 aux(FAST=1 对称V-cycle) 在
# 一致配置下对比 per-step 墙钟 + peak_rss。起裂前+起裂区 (NSTEPS=16, model0 起裂~step14)。
# 要求:每步存 checkpoint (save_every=1) + 每步写 VTU (默认) -> 可断点 + 完整 VTU 序列。
#
# 用法: run_scan_level.sh <case> <level h1|h2|h3|h4> [nsteps]
set -euo pipefail
CASE="${1:?case required: model0|square|model2}"
LEVEL="${2:?level required: h1|h2|h3|h4}"
NSTEPS="${3:-16}"

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# 显式指定 py312 解释器(base env 无 fealpy,env.sh 自动探测会失败)
export FEALPY_PYTHON="${FEALPY_PYTHON:-/home/gongshihua/miniconda3/envs/py312/bin/python}"
export FRACTUREX_PYTHON="${FRACTUREX_PYTHON:-${FEALPY_PYTHON}}"
FRACTUREX_ENV_QUIET=1 source "${_REPO_ROOT}/scripts/paper_huzhang/env.sh"

# ---- 一致的线程/装配配置 (整个扫描所有 run/级别用同一套,才可比) ----
# 装配不 fork(避开 OpenBLAS 超额订阅段错),求解走 MKL 多线程(pardiso 直接受益)。
export FRACTUREX_ASSEMBLY_NPROC=1
export MKL_NUM_THREADS=16
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
# ---- checkpoint + VTU 要求 ----
export FRACTUREX_SAVE_NPZ=1
export FRACTUREX_SAVE_EVERY=1          # 每步存 checkpoint (单步数分钟,写盘可忽略)
export FRACTUREX_RUN_NSTEPS="${NSTEPS}" # 截到起裂区,做规模化对比(非整跑)
# VTU 每步: FRACTUREX_VTU_EVERY 默认 1

case "${LEVEL}" in
  h1) HMIN=0.05 ;;
  h2) HMIN=0.025 ;;
  h3) HMIN=0.013 ;;
  h4) HMIN=0.008 ;;
  *) echo "unknown level ${LEVEL}"; exit 2 ;;
esac
export FRACTUREX_HMIN="${HMIN}"

LOGDIR="${FRACTUREX_PAPER_ROOT}/logs"
mkdir -p "${LOGDIR}"
PY="${FRACTUREX_PYTHON}"
RC="${_REPO_ROOT}/scripts/paper_huzhang/run_case.py"

run_one () {
  local mode="$1"; shift
  local suffix="$1"; shift
  local tag="${CASE}_${suffix}_${LEVEL}"
  local log="${LOGDIR}/scan_${tag}.log"
  echo ">>> [${tag}] mode=${mode} hmin=${HMIN} nsteps=${NSTEPS} MKL=${MKL_NUM_THREADS}  -> ${log}"
  FRACTUREX_RUN_LABEL_SUFFIX="${suffix}_${LEVEL}" "$@" \
    "${PY}" "${RC}" --case "${CASE}" --mode "${mode}" --out-root "${FRACTUREX_PAPER_ROOT}" \
    >"${log}" 2>&1
  echo "<<< [${tag}] done rc=$?"
}

echo "===== scan ${CASE} ${LEVEL} (hmin=${HMIN}) ====="
# direct: pardiso 强 baseline
run_one direct scan_pardiso  env FRACTUREX_ELASTIC_DIRECT_BACKEND=pardiso
# aux: FAST=1 对称 V-cycle (论文方法最优形态)
run_one aux    scan_auxfast  env FRACTUREX_ELASTIC_FAST=1
echo "===== scan ${CASE} ${LEVEL} complete ====="
