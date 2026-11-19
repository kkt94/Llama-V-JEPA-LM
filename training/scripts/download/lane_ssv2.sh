#!/bin/bash
# Something-Something v2 (MVP human-object subset). Public link works only after the Qualcomm license is accepted.
source /data/llama_vjepa_lm/scripts/env.sh
B=https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/data/something-something-v2
for p in 00 01; do
  bash /data/llama_vjepa_lm/training/scripts/download/dl_url.sh ssv2:$p $B/20bn-something-something-v2-$p $R/bench/mvp_sources/raw/20bn-something-something-v2-$p &
done; wait
echo -e "$(date -Iseconds)\tLANE_SSV2\tdone\t" >> $R/logs/status.tsv
