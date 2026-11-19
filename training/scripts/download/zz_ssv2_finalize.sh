#!/bin/bash
# The SSv2 server omits Content-Length on HEAD, so dl_url.sh cannot confirm completion. Expected totals come from
# the Content-Range of a ranged GET. When both files reach them: stop the retry loop, check zip integrity, log.
R=/data/llama_vjepa_lm; D=$R/bench/mvp_sources/raw; ST=$R/logs/status.tsv
declare -A W=([00]=10000000000 [01]=9444975475)
until [ "$(stat -c %s $D/20bn-something-something-v2-00 2>/dev/null)" = "${W[00]}" ] && \
      [ "$(stat -c %s $D/20bn-something-something-v2-01 2>/dev/null)" = "${W[01]}" ]; do sleep 30; done
pkill -f '^bash /data/llama_vjepa_lm/training/scripts/download/lane_ssv2.sh' ; pkill -f 'dl_url.sh ssv2:'
for p in 00 01; do
  if python -I -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); b=z.testzip(); sys.exit(1 if b else 0)" $D/20bn-something-something-v2-$p; then
    echo -e "$(date -Iseconds)\tssv2:$p\tcomplete\tsize=${W[$p]} zip_ok" >> $ST
  else echo -e "$(date -Iseconds)\tssv2:$p\tFAILED\tzip integrity" >> $ST; fi
done
