#!/bin/bash
# SA-V: process each train tar as soon as its download is verified (.ok marker): masklet json -> data/SA-V/annotations,
# mp4 -> 384p at native 24 fps (masklets are annotated at 6 fps = every 4th frame), GOP 12. val/test tars are unpacked
# as-is. Multi-node safe via mkdir locks.
R=/data/llama_vjepa_lm; D=$R/data/SA-V; O=$R/prep/sav_384_24fps; ST=$R/logs/status.tsv; H=$(hostname -s)
mkdir -p $O/_tmp $O/_done $O/_lock $D/annotations $D/extracted
echo -e "$(date -Iseconds)\tprep_sav@$H\tstart\t" >> $ST
proc() {  # $1 = tar basename without .tar
  k=$1; t=$D/raw/$k.tar; [ -f $O/_done/$k ] && return 0; mkdir $O/_lock/$k 2>/dev/null || return 0
  if [[ $k == sav_[0-9]* ]]; then
    w=$O/_tmp/$k; rm -rf $w; mkdir -p $w
    tar -xf $t -C $D/annotations --wildcards '*.json'
    tar -xf $t -C $w --wildcards '*.mp4'
    (cd $w && find . -name '*.mp4' | sed 's#^\./##') > $w/_list.txt
    xargs -a $w/_list.txt -P ${JOBS:-6} -I@ bash $R/training/scripts/prep/transcode_one.sh "$w/@" "$O/@" native 12
    n=$(wc -l < $w/_list.txt); m=$(sed "s#^#$O/#" $w/_list.txt | xargs ls 2>/dev/null | wc -l)
    if [ "$n" = "$m" ]; then touch $O/_done/$k; echo -e "$(date -Iseconds)\tprep_sav:$k\tcomplete\tmp4=$n" >> $ST; else rmdir $O/_lock/$k; fi
    rm -rf $w
  else
    tar -xf $t -C $D/extracted && touch $O/_done/$k && echo -e "$(date -Iseconds)\tprep_sav:$k\tcomplete\tunpacked" >> $ST || rmdir $O/_lock/$k
  fi
}
want=$(tail -n +2 $D/links.tsv | cut -f1 | grep '\.tar$' | sed 's/\.tar$//')
while :; do
  left=0
  for k in $want; do
    [ -f $O/_done/$k ] && continue; left=$((left+1))
    [ -f $D/$k.tar.ok ] && proc $k
  done
  [ $left -eq 0 ] && break
  grep -qP '\tLANE_SAV\tdone' $ST && [ -z "$(for k in $want; do [ -f $D/$k.tar.ok ] || echo x; done)" ] || sleep 60
  # stop if the download lane finished but some tars never verified
  if grep -qP '\tLANE_SAV\tdone' $ST && [ -n "$(for k in $want; do [ -f $D/$k.tar.ok ] || echo x; done)" ]; then
    echo -e "$(date -Iseconds)\tprep_sav@$H\tINCOMPLETE\tsome tars missing .ok" >> $ST; break; fi
done
echo -e "$(date -Iseconds)\tprep_sav@$H\tdone\t$(ls $O/_done | wc -l) units" >> $ST
