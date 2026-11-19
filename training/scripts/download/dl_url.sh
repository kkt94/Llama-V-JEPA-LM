#!/bin/bash
# usage: dl_url.sh <name> <url> <out_path>   resumable, verifies size against Content-Length
name=$1; url=$2; out=$3; ST=/data/llama_vjepa_lm/training/scripts/data_gen/status.tsv
mkdir -p "$(dirname "$out")"
want=$(curl -sIL -m 60 "$url" | grep -i '^content-length' | tail -1 | tr -dc 0-9)
echo -e "$(date -Iseconds)\t$name\tstart\t$url size=$want" >> $ST
for i in $(seq 1 30); do
  curl -sSL -C - --retry 5 --retry-delay 20 -o "$out" "$url"
  have=$(stat -c %s "$out" 2>/dev/null || echo 0)
  if [ -n "$want" ] && [ "$have" = "$want" ]; then
    echo -e "$(date -Iseconds)\t$name\tcomplete\tsize=$have" >> $ST; exit 0; fi
  echo -e "$(date -Iseconds)\t$name\tretry\tattempt=$i have=$have want=$want" >> $ST; sleep 30
done
echo -e "$(date -Iseconds)\t$name\tFAILED\t$url" >> $ST; exit 1
