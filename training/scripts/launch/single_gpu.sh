#!/bin/bash
# (10-26) one run on one GPU of a single machine (no Slurm): torchrun standalone, world size 1.
#   usage: single_gpu.sh <run_name> <config relative to repo/> [gpu index, default 0]     e.g. single_gpu.sh nxs_b200 configs/single/nxs_b200.yaml
# Writes runs/<run_name>.out (stdout) under the data root; the run's own log / checkpoints go to run.out_dir of the config.
set -u
NAME=$1; CFG=$2; GPU=${3:-0}; R=${JEPA_ROOT:-/data/llama_vjepa_lm}; REPO=$R/training; PY=${JEPA_PY:-$(dirname "$(which python)")}
cd $REPO
export PYTHONPATH=$REPO HF_HOME=$R/hf_home OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ARROW_DEFAULT_MEMORY_POOL=system
echo -e "$(date -Iseconds)\trun:$NAME\tlaunch\tsingle gpu=$GPU cfg=$CFG" >> $R/logs/status.tsv
CUDA_VISIBLE_DEVICES=$GPU $PY/torchrun --standalone --nnodes=1 --nproc_per_node=1 -m jepa_llm.training.train_single $CFG >> $R/runs/$NAME.out 2>&1
rc=$?; echo -e "$(date -Iseconds)\trun:$NAME\t$([ $rc = 0 ] && echo complete || echo FAILED)\trc=$rc" >> $R/logs/status.tsv; exit $rc
