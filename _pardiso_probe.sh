#!/bin/bash
cd ~/tian/fracturex
source ~/miniconda3/etc/profile.d/conda.sh && conda activate py312
export PYTHONPATH=. FRACTUREX_CASE=model2 FRACTUREX_NX=24 FRACTUREX_SMOKE=1 \
       FRACTUREX_MARKER=eta_T FRACTUREX_ELASTIC_SOLVER=pardiso \
       FRACTUREX_MAX_STEPS=1 FRACTUREX_NO_VTU=1
run() {  # name  MKL_T  ASM_PARALLEL  ASM_NPROC
  local name=$1 T=$2 P=$3 NP=$4
  export MKL_NUM_THREADS=$T OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
  export FRACTUREX_ASSEMBLY_PARALLEL=$P FRACTUREX_ASSEMBLY_NPROC=$NP
  export FRACTUREX_OUTDIR=results/_probe_$name
  echo "=== PROBE $name : MKL_T=$T asm_par=$P nproc=$NP ==="
  python fracturex/tests/aposteriori/run_m3_pc_model1.py 2>&1 | grep -E "\[PC\] step"
}
run mkl16_par16   16 1 16
run mkl16_ser      16 0 1
run mkl1_ser        1 0 1
run mkl8_ser        8 0 1
echo "=== PROBES DONE ==="
