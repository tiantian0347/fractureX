#!/usr/bin/env bash
# resume_aux_h2h3_dnfbreak_0619.sh
# 目标:为 model0 拿到完整 load 曲线(aux h2 ‖ h3),并以更长 Krylov basis 复试 step16 DNF。
#
# 背景/依据:
#  - 2026-06-12 结案:aux_h2 step16(max_d->1.0 完全分离瞬间)在 restart=200/maxit=400
#    (开/关 Anderson 各一次)均打满 maxit 不收敛 = 迭代法真实边界。
#  - 但 d12/d13 已证 restart 长度是局部化 niter 的主因(step_014 93->9@r300)。06-12 只试到
#    restart=200。本次拉到 restart=400/maxit=800(更长 basis),直接判定 step16 DNF 是否
#    仍是 restart 截断假象;若收敛即可推翻 §5.2d,若仍 DNF 则坐实"restart=400 也救不了"。
#  - mesh 经验证确定性复现(make_mesh(0.013)=NN5726/NC11034 与 direct_h3 逐字节一致;
#    make_mesh(0.025)=NC2868 与 direct_h2 一致),resume 的 NC 校验会兜底。
#
# 数据完整性(Jun-13 history 回退留下的坑):
#  - h2 history 现到 step12,checkpoints 13/14/15 已暂存到 checkpoints/_stashed_above_step12/,
#    使 step12 成为最新 -> 从 step12 续,连续重写 13,14,15,16... 无缺口。
#  - h3 history 回退到 step4 且 step5-9 既无 history 也无 checkpoint(无法在原网格恢复),
#    checkpoints 10-14 已暂存到 checkpoints/_stashed_above_step000/,使 step000 成为最新
#    -> 从 step000 续,在同一(确定性复现的)网格上重跑 1..30 拿完整连续曲线。
#  - 修改前完整备份在 results/phasefield/model0_circular_notch/_resume_backup_20260619/。
set -u
REPO=/home/gongshihua/tian/fracturex
PY=/home/gongshihua/miniconda3/envs/py312/bin/python
export PYTHONPATH=$REPO
cd "$REPO" || exit 1
STATUS=$REPO/docs/preconditioner/PIPELINE_STATUS.md
ts() { date '+%F %T'; }
note() { printf -- '- [%s] %s\n' "$(ts)" "$1" >> "$STATUS"; }

# ---- 更长 Krylov basis:复试 step16 DNF(06-12 只到 restart=200)----
export FRACTUREX_GMRES_RESTART=400
export FRACTUREX_GMRES_MAXIT=800
# ---- Anderson 加速(破起裂极限环;D12 §14 / run_aux_model0.sh 验证口径)----
export FRACTUREX_ANDERSON_DEPTH=5
export FRACTUREX_ANDERSON_OMEGA=1.0
export FRACTUREX_ANDERSON_TR_FACTOR=20
export FRACTUREX_ANDERSON_RESTART_OMEGA=1.6
# ---- 高核主机 BLAS 线程安全(env.sh 同口径,防 OpenBLAS 段错)----
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export FRACTUREX_ASSEMBLY_NPROC=64

launch_aux() {  # $1=tier $2=hmin
  local tier="$1" hmin="$2"
  local log="$REPO/results/phasefield/model0_circular_notch/paper_aux_${tier}/epsg_1e-06/resume.log"
  FRACTUREX_RESUME=1 FRACTUREX_SAVE_EVERY=1 \
  FRACTUREX_RUN_LABEL_SUFFIX="$tier" FRACTUREX_HMIN="$hmin" \
    nohup "$PY" "$REPO/scripts/paper_huzhang/run_case.py" --case model0 --mode aux --out-root results \
      >> "$log" 2>&1 &
  echo $!
}

note "=== [0619] 续跑 aux_h2(@step12) ‖ aux_h3(@step000) -> step30;restart=400/maxit=800 + Anderson(d5);复试 step16 DNF + 取完整 load 曲线 ==="
PID_H2=$(launch_aux h2 0.025)
PID_H3=$(launch_aux h3 0.013)
note "PIDs(dnfbreak_0619): h2=$PID_H2 (resume@step12) h3=$PID_H3 (resume@step000)"
echo "PIDs: h2=$PID_H2 h3=$PID_H3"
