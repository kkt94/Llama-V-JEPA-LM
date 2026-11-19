#!/bin/bash
# LLaVA-Video-178K -> 384p / 15 fps / GOP 15 (measured: 16-frame random access 1.0 s -> 0.24 s per sample, 36% smaller)
R=/data/llama_vjepa_lm; I=$R/data/LLaVA-Video-178K/videos; O=$R/prep/llava_384_15fps; ST=$R/logs/status.tsv
mkdir -p $O; echo -e "$(date -Iseconds)\tprep_llava\tstart\t" >> $ST
find $I -name '*.mp4' | sort > $O/_inputs.txt
sed "s#^$I/##" $O/_inputs.txt | xargs -P ${JOBS:-32} -I{} bash $R/training/scripts/prep/transcode_one.sh "$I/{}" "$O/{}" 15 15
n_in=$(wc -l < $O/_inputs.txt); n_out=$(find $O -name '*.mp4' | wc -l)
echo -e "$(date -Iseconds)\tprep_llava\t$([ $n_in -eq $n_out ] && echo complete || echo INCOMPLETE)\t$n_out/$n_in" >> $ST
