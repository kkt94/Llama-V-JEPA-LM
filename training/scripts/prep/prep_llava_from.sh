#!/bin/bash
# Extra LLaVA worker starting at a fraction of the input list (START_FRAC, default 0.5) and walking forward, then wrapping
# around. Finished outputs are skipped and temp names are per-process, so overlapping workers are harmless.
R=/data/llama_vjepa_lm; I=$R/data/LLaVA-Video-178K/videos; O=$R/prep/llava_384_15fps; ST=$R/logs/status.tsv; H=$(hostname -s)
n=$(wc -l < $O/_inputs.txt); s=$(awk -v n=$n -v f=${START_FRAC:-0.5} 'BEGIN{printf "%d", n*f}')
echo -e "$(date -Iseconds)\tprep_llava_from@$H\tstart\tfrac=${START_FRAC:-0.5}" >> $ST
{ tail -n +$((s+1)) $O/_inputs.txt; head -n $s $O/_inputs.txt; } | sed "s#^$I/##" | \
  xargs -P ${JOBS:-14} -I{} bash $R/training/scripts/prep/transcode_one.sh "$I/{}" "$O/{}" 15 15
echo -e "$(date -Iseconds)\tprep_llava_from@$H\tdone\t" >> $ST
