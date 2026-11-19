#!/bin/bash
# NVIDIA physical interaction: everything except depths (11.5TB); small modalities first
source /data/llama_vjepa_lm/scripts/env.sh
for m in captions cameras physics scene videos segmentation; do
  $PY -I $R/scripts/dl_hf.py nvidia_$m dataset nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes \
     $R/data/nvidia_physical_interaction --allow "$m/*" README.md "*.gif" --workers 8
done
echo -e "$(date -Iseconds)\tLANE_NVIDIA\tdone\t" >> $R/logs/status.tsv
