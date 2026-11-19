#!/bin/bash
# TemporalBench (HF gated; token read from HF_TOKEN_PATH, never stored in the repo)
source /data/llama_vjepa_lm/scripts/env.sh
$PY -I /data/llama_vjepa_lm/training/scripts/download/dl_hf.py temporalbench dataset microsoft/TemporalBench $R/bench/TemporalBench --workers 8
echo -e "$(date -Iseconds)\tLANE_TB\tdone\t" >> $R/logs/status.tsv
