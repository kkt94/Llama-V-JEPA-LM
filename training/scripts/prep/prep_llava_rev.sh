#!/bin/bash
# Second LLaVA worker for another node: walks the same input list in REVERSE so it meets the forward worker in the
# middle; finished outputs are skipped and transcode_one.sh uses per-process temp names, so a late overlap is harmless.
R=/data/llama_vjepa_lm; I=$R/data/LLaVA-Video-178K/videos; O=$R/prep/llava_384_15fps; ST=$R/logs/status.tsv; H=$(hostname -s)
echo -e "$(date -Iseconds)\tprep_llava_rev@$H\tstart\t" >> $ST
tac $O/_inputs.txt | sed "s#^$I/##" | xargs -P ${JOBS:-14} -I{} bash $R/training/scripts/prep/transcode_one.sh "$I/{}" "$O/{}" 15 15
echo -e "$(date -Iseconds)\tprep_llava_rev@$H\tdone\t$(find $O -name '*.mp4' | wc -l) mp4" >> $ST
