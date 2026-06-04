#!/usr/bin/env bash
# 等受控扫描的 3 个待完成 run 都写出 wall_s(run() 末尾标记)后,自动出对比表。
# 完成即退出 -> 触发任务通知。最长等 TIMEOUT 秒(默认 8h)。
set -uo pipefail
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="${_REPO_ROOT}/results/phasefield/model0_circular_notch"
PY=/home/gongshihua/miniconda3/envs/py312/bin/python
OUT="${_REPO_ROOT}/results/logs/scan_compare.txt"
TIMEOUT="${1:-28800}"
POLL=120

# 待完成的 run(manifest 含 wall_s 即视为跑完);direct_h1 已完成不必等
PENDING=(
  "paper_aux_scan_auxfast_h1"
  "paper_direct_scan_pardiso_h2"
  "paper_aux_scan_auxfast_h2"
  "paper_direct_scan_pardiso_h3"
  "paper_aux_scan_auxfast_h3"
)

done_one () {  # $1=run_label -> 0 if finished
  local mf="${BASE}/$1/run_manifest.json"
  [[ -f "$mf" ]] && grep -q '"wall_s"' "$mf"
}

t0=$(date +%s)
while :; do
  all=1
  for r in "${PENDING[@]}"; do done_one "$r" || { all=0; break; }; done
  if [[ $all -eq 1 ]]; then
    echo "[watch] 全部完成,生成对比表 -> ${OUT}"
    "${PY}" "${_REPO_ROOT}/scripts/paper_huzhang/scan_compare.py" h1 h2 | tee "${OUT}"
    exit 0
  fi
  now=$(date +%s)
  if (( now - t0 > TIMEOUT )); then
    echo "[watch] 超时 ${TIMEOUT}s 仍未全完,输出当前可得对比 -> ${OUT}"
    { echo "(TIMEOUT — 部分 run 未完成)"; "${PY}" "${_REPO_ROOT}/scripts/paper_huzhang/scan_compare.py" h1 h2; } | tee "${OUT}"
    exit 2
  fi
  sleep "${POLL}"
done
