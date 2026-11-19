#!/bin/bash
# models, code, benchmarks (small/medium)
source /data/llama_vjepa_lm/scripts/env.sh
U=$R/scripts/dl_url.sh; H="$PY -I $R/scripts/dl_hf.py"
$U vjepa2_1_vitg_384 https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt $R/models/vjepa2_1/vjepa2_1_vitg_384.pt
$H qwen3_8b model Qwen/Qwen3-8B $R/models/Qwen3-8B
mkdir -p $R/code; for g in facebookresearch/vjepa2 google-deepmind/perception_test facebookresearch/minimal_video_pairs EvolvingLMMs-Lab/lmms-eval; do
  d=$R/code/$(basename $g); [ -d $d/.git ] || git clone -q https://github.com/$g $d
  echo -e "$(date -Iseconds)\tgit:$g\tcomplete\t$(git -C $d rev-parse HEAD)" >> $R/logs/status.tsv
done
P=https://storage.googleapis.com/dm-perception-test
for f in zip_data/valid_videos.zip zip_data/valid_annotations.zip zip_data/mc_question_valid_annotations.zip \
         misc/cut_frame_mapping_valid.json misc/cut_frame_mapping_train.json misc/mc_question_valid_id_list.csv; do
  $U pt:$(basename $f) $P/$f $R/bench/perception_test/$(basename $f)
done
$H intphys2 dataset facebook/IntPhys2 $R/bench/IntPhys2
$H video_mme_logical dataset marcuskwan/video-mme-logical $R/bench/video-mme-logical
$H mvp_annotations dataset facebook/minimal_video_pairs $R/bench/minimal_video_pairs
$H plm_video_human dataset facebook/PLM-Video-Human $R/data/PLM-Video-Human
$H video_mme_v2 dataset MME-Benchmarks/Video-MME-v2 $R/bench/Video-MME-v2
$U vjepa2_1_vitG_384 https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt $R/models/vjepa2_1/vjepa2_1_vitG_384.pt
echo -e "$(date -Iseconds)\tLANE_CORE\tdone\t" >> $R/logs/status.tsv
