#!/bin/bash
# LLaVA-Video folders restored to training on 2026-10-09 (nextqa, perceptiontest: ids proven train-only). Their video
# archives were excluded from the first download; fetch, extract, then transcode the new videos to 384p/15fps.
source /data/llama_vjepa_lm/scripts/env.sh
D=$R/data/LLaVA-Video-178K; ST=$R/logs/status.tsv; M=$R/logs/extract_done
$PY -I /data/llama_vjepa_lm/training/scripts/download/dl_hf.py llava_video_restored dataset lmms-lab/LLaVA-Video-178K $D \
   --allow "*nextqa*/*.tar.gz" "*perceptiontest*/*.tar.gz" --workers 8 || exit 1
ls $D/*nextqa*/*.tar.gz $D/*perceptiontest*/*.tar.gz | while read f; do k=$(basename $f .tar.gz)
  [ -f $M/llava_$k.done ] && continue
  pigz -dc "$f" | tar -x -C $D/videos && touch $M/llava_$k.done && echo -e "$(date -Iseconds)\textract:llava_$k\tcomplete\t" >> $ST \
    || echo -e "$(date -Iseconds)\textract:llava_$k\tFAILED\t$f" >> $ST
done
O=$R/prep/llava_384_15fps
ls $D/*nextqa*/*.tar.gz $D/*perceptiontest*/*.tar.gz | xargs -I{} tar -tzf {} | grep '\.mp4$' | sort -u > $O/_inputs_restored.txt
xargs -a $O/_inputs_restored.txt -P ${JOBS:-12} -I{} bash $R/training/scripts/prep/transcode_one.sh "$D/videos/{}" "$O/{}" 15 15
n=$(wc -l < $O/_inputs_restored.txt); m=$(sed "s#^#$O/#" $O/_inputs_restored.txt | xargs ls 2>/dev/null | wc -l)
echo -e "$(date -Iseconds)\tprep_llava_restored\t$([ $n -eq $m ] && echo complete || echo INCOMPLETE)\t$m/$n" >> $ST
