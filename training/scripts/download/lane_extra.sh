#!/bin/bash
# Extra sources: ShareGPTVideo frames (llava_hound 255k QA) + Language Table tfrecords (MVP robot subset).
source /data/llama_vjepa_lm/scripts/env.sh
S=/data/llama_vjepa_lm/training/scripts/download
$PY -I $S/dl_hf.py sharegptvideo dataset ShareGPTVideo/train_video_and_instruction $R/data/ShareGPTVideo --workers 8
LT=$R/bench/mvp_sources/raw/language_table
mkdir -p $LT
curl -s "https://storage.googleapis.com/storage/v1/b/gresearch/o?prefix=robotics/language_table/0.0.1/&fields=items(name,size)&maxResults=1000" \
  | $PY -I -c "import json,sys; [print(x['name']) for x in json.load(sys.stdin)['items']]" > $LT/objects.txt
echo -e "$(date -Iseconds)\tlanguage_table\tstart\tobjects=$(wc -l < $LT/objects.txt)" >> $R/logs/status.tsv
xargs -P 8 -I{} bash -c 'n="{}"; out="'$LT'/${n#robotics/language_table/}"; [ -f "$out.ok" ] && exit 0; \
  bash '$S'/dl_url.sh "lt:$(basename $n)" "https://storage.googleapis.com/gresearch/$n" "$out" > /dev/null && touch "$out.ok"' < $LT/objects.txt
ok=$(ls $LT/0.0.1/*.ok 2>/dev/null | wc -l)
echo -e "$(date -Iseconds)\tlanguage_table\t$([ $ok -eq $(wc -l < $LT/objects.txt) ] && echo complete || echo INCOMPLETE)\tok=$ok" >> $R/logs/status.tsv
echo -e "$(date -Iseconds)\tLANE_EXTRA\tdone\t" >> $R/logs/status.tsv
