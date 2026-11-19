#!/bin/bash
# Alternative LLM backbones matching the original papers: Llama 3.1 8B (V-JEPA 2 scaling / V-JEPA 2.1),
# Llama 3.2 1B/3B (same-family small ladder, as in PLM).
source /data/llama_vjepa_lm/scripts/env.sh
H="$PY -I /data/llama_vjepa_lm/training/scripts/download/dl_hf.py"
$H llama3_2_1b_instruct model meta-llama/Llama-3.2-1B-Instruct $R/models/Llama-3.2-1B-Instruct --ignore "original/*"
$H llama3_2_3b_instruct model meta-llama/Llama-3.2-3B-Instruct $R/models/Llama-3.2-3B-Instruct --ignore "original/*"
$H llama3_1_8b_instruct model meta-llama/Llama-3.1-8B-Instruct $R/models/Llama-3.1-8B-Instruct --ignore "original/*"
$H llama3_1_8b model meta-llama/Llama-3.1-8B $R/models/Llama-3.1-8B --ignore "original/*"
echo -e "$(date -Iseconds)\tLANE_LLM_ALT\tdone\t" >> $R/logs/status.tsv
