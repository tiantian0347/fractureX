#!/usr/bin/env bash
# model1 (square_tension_precrack) only:
# parallel assembly + elastic aux-space preconditioned GMRES.
set -euo pipefail

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${_REPO_ROOT}/scripts/paper_huzhang/env.sh"

# 可选:开启 staggered Anderson 加速(aux 路径已验证,见 commit 6dc62ee/a779abe)。
# 默认关(FRACTUREX_ANDERSON_DEPTH=0)。开启后稳定段反力 vs plain AM 差 ≤0.007%,
# 现实加速 ~1.5–2×(峰前 ~2× + 起裂步可靠收敛;非历史记录里已丢配置的 38×)。
#   export FRACTUREX_ANDERSON_DEPTH=5           # >0 开启;窗口大小 m
#   export FRACTUREX_ANDERSON_OMEGA=1.0         # 稳态播种 plain,不 over-relax(保反力)
#   export FRACTUREX_ANDERSON_TR_FACTOR=20      # 信赖域:加速步长 <= tr*||f||,挡过冲
#   export FRACTUREX_ANDERSON_RESTART_OMEGA=1.6 # 仅 restart 后第一步 over-relax kick,破起裂极限环
#   # 其余可调:FRACTUREX_ANDERSON_{BETA,BLOWUP,PATIENCE}
# 注:square(model1)direct 跑大网格另需 FRACTUREX_ELASTIC_DIRECT_BACKEND=pardiso(见 commit 历史);aux 路径不受此影响。

OUT="${FRACTUREX_PAPER_ROOT}"

echo ">>> aux (aux-space elastic) / model1 (square_tension_precrack)"
exec "${FRACTUREX_PYTHON}" "${_REPO_ROOT}/scripts/paper_huzhang/run_case.py" \
  --case square --mode aux --out-root "${OUT}"
