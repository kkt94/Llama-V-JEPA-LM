#!/bin/bash
# SSv2 parts are a split tar.gz (not zip): concatenate, decompress (pigz verifies gzip CRC) and untar.
R=/data/llama_vjepa_lm; D=$R/bench/mvp_sources/raw; X=$R/bench/mvp_sources/extracted/ssv2; ST=$R/logs/status.tsv
mkdir -p $X
if cat $D/20bn-something-something-v2-00 $D/20bn-something-something-v2-01 | pigz -dc | tar -x -C $X; then
  echo -e "$(date -Iseconds)\tssv2\tcomplete\tsplit tar.gz ok, webm=$(ls $X/20bn-something-something-v2 | wc -l)" >> $ST
else echo -e "$(date -Iseconds)\tssv2\tFAILED\textract" >> $ST; fi
