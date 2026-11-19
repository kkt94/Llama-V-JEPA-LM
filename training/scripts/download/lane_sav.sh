#!/bin/bash
# SA-V from the signed CDN list (data/SA-V/links.tsv, kept out of git; links expire). 8 in parallel, then sha256 check
# against Meta's sav_sha256sum.chk.
source /data/llama_vjepa_lm/scripts/env.sh
S=/data/llama_vjepa_lm/training/scripts/download; D=$R/data/SA-V; ST=$R/logs/status.tsv
tail -n +2 $D/links.tsv | xargs -P 8 -L 1 bash -c 'n=$0; u=$1; [ -f "'$D'/$n.ok" ] && exit 0; \
  bash '$S'/dl_url.sh "sav:$n" "$u" "'$D'/raw/$n" > /dev/null && touch "'$D'/$n.ok"'
ok=$(ls $D/*.ok | wc -l); want=$(($(wc -l < $D/links.tsv) - 1))
echo -e "$(date -Iseconds)\tsav_download\t$([ $ok -eq $want ] && echo complete || echo INCOMPLETE)\tok=$ok/$want" >> $ST
cd $D/raw && if sha256sum -c --ignore-missing sav_sha256sum.chk > $D/sha256_check.txt 2>&1; then
  echo -e "$(date -Iseconds)\tsav_sha256\tcomplete\t$(grep -c ': OK' $D/sha256_check.txt) OK" >> $ST
else
  echo -e "$(date -Iseconds)\tsav_sha256\tFAILED\t$(grep -vc ': OK' $D/sha256_check.txt) not OK, see $D/sha256_check.txt" >> $ST
fi
echo -e "$(date -Iseconds)\tLANE_SAV\tdone\t" >> $ST
