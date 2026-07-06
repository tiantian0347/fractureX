#!/bin/bash
cd ~/tian/fracturex
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py312
export PYTHONPATH=.
export FRACTUREX_CASE=model2
export FRACTUREX_NX=48
export FRACTUREX_DU=2.5e-4
export FRACTUREX_MAX_STEPS=40
export FRACTUREX_MARKER=eta_T
export FRACTUREX_ETA_T_STRATEGY=max
export FRACTUREX_THETA_REC=0.7
export FRACTUREX_ETA_DECREMENT=0.7
export FRACTUREX_D_HI=0.999
export FRACTUREX_CH=4.0
export FRACTUREX_LINSOLVE=pardiso
export FRACTUREX_OUT=results/adaptive_m3_pc_model2_eta_T_nx48/
python fracturex/tests/aposteriori/run_m3_pc_model1.py
