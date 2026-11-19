#!/bin/bash
# Unpack archives into usable trees. Archives are kept; each unit leaves a .done marker so reruns skip it.
source /data/llama_vjepa_lm/scripts/env.sh
ST=$R/logs/status.tsv; M=$R/logs/extract_done; mkdir -p $M
mark() { echo -e "$(date -Iseconds)\textract:$1\t$2\t$3" >> $ST; }
# zip archives are handled by lane_unzip.sh (no unzip binary on this host)
X=$R/bench/mvp_sources/extracted; W=$R/bench/mvp_sources/raw
for t in inflevel_lab intphys_dev; do
  [ -f $M/mvp_$t.done ] || { mkdir -p $X/$t && pigz -dc $W/$t.tar.gz | tar -x -C $X/$t && touch $M/mvp_$t.done && mark mvp_$t complete $X/$t || mark mvp_$t FAILED $t; }
done
# LLaVA-Video videos (1.2 TB of tar.gz), 8 in parallel
L=$R/data/LLaVA-Video-178K; mkdir -p $L/videos
ls $L/*/*.tar.gz | xargs -P 8 -I{} bash -c 'f={}; k=$(basename $f .tar.gz); [ -f '$M'/llava_$k.done ] && exit 0; \
  if pigz -dc "$f" | tar -x -C '$L'/videos; then touch '$M'/llava_$k.done; echo -e "$(date -Iseconds)\textract:llava_$k\tcomplete\t" >> '$ST'; \
  else echo -e "$(date -Iseconds)\textract:llava_$k\tFAILED\t$f" >> '$ST'; fi'
echo -e "$(date -Iseconds)\tLANE_EXTRACT\tdone\tllava=$(ls $M/llava_*.done | wc -l)/$(ls $L/*/*.tar.gz | wc -l)" >> $ST
