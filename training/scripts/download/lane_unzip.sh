#!/bin/bash
# Zip archives (unzip binary is absent here): extracted with python zipfile, 4 in parallel, .done markers.
source /data/llama_vjepa_lm/scripts/env.sh
ST=$R/logs/status.tsv; M=$R/logs/extract_done; mkdir -p $M
U="python -I /data/llama_vjepa_lm/training/scripts/download/unzip_py.py"
P=$R/bench/perception_test; X=$R/bench/mvp_sources/extracted; W=$R/bench/mvp_sources/raw
{
  for z in valid_videos valid_annotations mc_question_valid_annotations; do echo "pt_$z $P/$z.zip $P/extracted"; done
  for z in $R/bench/Video-MME-v2/videos/*.zip; do echo "vmme2_$(basename $z .zip) $z $R/bench/Video-MME-v2/extracted"; done
  echo "vmme_logical $R/bench/video-mme-logical/video_mme_logical.zip $R/bench/video-mme-logical/extracted"
  echo "mvp_clevrer $W/clevrer_video_validation.zip $X/clevrer"
  echo "mvp_grasp $W/grasp_videos.zip $X/grasp"
  echo "mvp_charades $W/Charades_v1_480.zip $X/charades"
  echo "mvp_vinoground $W/hf_vinoground/vinoground_videos.zip $X/vinoground"
} | xargs -P 4 -L 1 bash -c 'n=$0; z=$1; d=$2; [ -f '$M'/$n.done ] && exit 0; mkdir -p $d; \
  if '"$U"' $z $d; then touch '$M'/$n.done; echo -e "$(date -Iseconds)\textract:$n\tcomplete\t$d" >> '$ST'; \
  else echo -e "$(date -Iseconds)\textract:$n\tFAILED\t$z" >> '$ST'; fi'
echo -e "$(date -Iseconds)\tLANE_UNZIP\tdone\t$(ls $M | wc -l) markers" >> $ST
