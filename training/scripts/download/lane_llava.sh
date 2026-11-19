#!/bin/bash
# LLaVA-Video-178K: all annotation json first, then video archives except benchmark-derived folders
source /data/llama_vjepa_lm/scripts/env.sh
D=$R/data/LLaVA-Video-178K
$PY -I $R/scripts/dl_hf.py llava_video_json dataset lmms-lab/LLaVA-Video-178K $D --allow "*.json" "*.md" "*.py"
$PY -I $R/scripts/dl_hf.py llava_video_videos dataset lmms-lab/LLaVA-Video-178K $D \
   --ignore "*perceptiontest*/*.tar.gz" "*nextqa*/*.tar.gz" "*activitynetqa*/*.tar.gz" --workers 8
echo -e "$(date -Iseconds)\tLANE_LLAVA\tdone\t" >> $R/logs/status.tsv
