#!/bin/bash
# NVIDIA physical interaction: RGB camera videos only (the *_{com,rot,spin,velocity}_per_clip.mp4 physics visualisations
# are excluded, spec 9.1). 1080p -> 384p, native 30 fps (physics timing), GOP 16.
# Safe to run on several nodes at once: each shard is claimed with an atomic mkdir lock; finished shards get a _done marker.
R=/data/llama_vjepa_lm; I=$R/data/nvidia_physical_interaction/videos; O=$R/prep/nvidia_rgb_384; ST=$R/logs/status.tsv
H=$(hostname -s); mkdir -p $O/_tmp $O/_done $O/_lock; echo -e "$(date -Iseconds)\tprep_nvidia@$H\tstart\t" >> $ST
ls $I/*/*.tar | xargs -P ${SHARDS:-4} -I{} bash -c '
  t={}; k=$(basename $t .tar); [ -f '$O'/_done/$k ] && exit 0
  mkdir '$O'/_lock/$k 2>/dev/null || exit 0
  w='$O'/_tmp/$k; rm -rf $w; mkdir -p $w
  tar -tf $t | grep -E "\.mp4$" | grep -vE "_(com|rot|spin|velocity)_per_clip\.mp4$" > $w/_rgb.txt
  tar -xf $t -C $w -T $w/_rgb.txt
  sc=$(basename $(dirname $t))
  xargs -a $w/_rgb.txt -P 4 -I@ bash '$R'/training/scripts/prep/transcode_one.sh "$w/@" "'$O'/$sc/@" native 16
  n=$(wc -l < $w/_rgb.txt); m=$(sed "s#^#'$O'/$sc/#" $w/_rgb.txt | xargs ls 2>/dev/null | wc -l)
  if [ "$n" = "$m" ]; then touch '$O'/_done/$k; else rmdir '$O'/_lock/$k; fi
  rm -rf $w'
n_sh=$(ls $I/*/*.tar | wc -l); n_done=$(ls $O/_done | wc -l)
echo -e "$(date -Iseconds)\tprep_nvidia@$H\t$([ $n_sh -eq $n_done ] && echo complete || echo partial)\tshards=$n_done/$n_sh" >> $ST
