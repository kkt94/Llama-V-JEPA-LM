#!/bin/bash
# (10-19 13:30) COIN videos (VideoChat-Flash TimeIT-COIN, mp4) -> data/VideoChat-Flash/x/coin; FGQA coin segments cut from them; FGQA manifest rebuilt
R=/data/llama_vjepa_lm; X=$R/data/VideoChat-Flash/x/coin; PY=python; L=$R/logs/dl/local_data.log; mkdir -p $X
for t in $R/data/VideoChat-Flash/timeaware_subset/timeit_COIN-TimeIT-COIN_10k/*.tar.zst; do
  k=$(basename $t .tar.zst); [ -f $X/.$k.done ] && continue
  nice -n 10 zstd -dc $t | tar -x --no-same-owner -C $X && touch $X/.$k.done
  echo "$(date -Iseconds) coin extracted $k" >> $L
done
echo "$(date -Iseconds) coin videos: $(find $X -name '*.mp4' | wc -l)" >> $L
nice -n 10 $PY -I $R/logs/zz_fgqa_local_cut.py coin $X 16 >> $L 2>&1
$PY -I $R/logs/zz_fgqa_manifest.py >> $L 2>&1
echo "$(date -Iseconds) COIN LOCAL DONE" >> $L
