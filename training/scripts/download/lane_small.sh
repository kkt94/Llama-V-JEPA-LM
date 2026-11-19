#!/bin/bash
# Smaller pilots: V-JEPA 2.1 distilled L/B + Qwen3 0.6B/1.7B/4B (same hybrid family as Qwen3-8B)
source /data/llama_vjepa_lm/scripts/env.sh
U=$R/scripts/dl_url.sh; H="$PY -I $R/scripts/dl_hf.py"
for f in vjepa2_1_vitl_dist_vitG_384 vjepa2_1_vitb_dist_vitG_384; do
  $U $f https://dl.fbaipublicfiles.com/vjepa2/$f.pt $R/models/vjepa2_1/$f.pt; done
for s in 0.6B 1.7B 4B; do $H qwen3_${s,,} model Qwen/Qwen3-$s $R/models/Qwen3-$s; done
echo -e "$(date -Iseconds)\tLANE_SMALL\tdone\t" >> $R/logs/status.tsv
