#!/bin/bash
cd ~/tian/fracturex
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py312
export PYTHONPATH=.
export FRACTUREX_CASE=model2
export FRACTUREX_NX=48
export FRACTUREX_DU=2.5e-4
export FRACTUREX_MAX_STEPS=60
export FRACTUREX_MARKER=eta_T
export FRACTUREX_ETA_T_STRATEGY=max
export FRACTUREX_THETA_REC=0.9
export FRACTUREX_ETA_DECREMENT=0.7
export FRACTUREX_D_HI=0.999
export FRACTUREX_CH=4.0
export FRACTUREX_ELASTIC_SOLVER=pardiso
export FRACTUREX_RESTART_NPZ=sens_nx48_step39.npz
export FRACTUREX_RESTART_STEP=39
export FRACTUREX_PEAK_R0=0.1957
export FRACTUREX_OUTDIR=results/adaptive_m3_pc_model2_eta_T_nx48_cont
python fracturex/tests/aposteriori/run_m3_pc_model1.py
