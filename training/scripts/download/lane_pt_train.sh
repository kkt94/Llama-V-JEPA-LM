#!/bin/bash
# PerceptionTest TRAIN split (allowed for training; validation/test stay evaluation-only). V-JEPA 2.1 post-trains on it.
source /data/llama_vjepa_lm/scripts/env.sh
U=$R/training/scripts/download/dl_url.sh; P=https://storage.googleapis.com/dm-perception-test/zip_data; D=$R/data/perception_test_train
for f in train_annotations.zip mc_question_train_annotations.zip train_videos.zip; do bash $U pt_train:$f $P/$f $D/$f || exit 1; done
for f in train_annotations mc_question_train_annotations train_videos; do
  [ -f $R/logs/extract_done/pt_train_$f.done ] && continue
  $PY -I $R/training/scripts/download/unzip_py.py $D/$f.zip $D/extracted && touch $R/logs/extract_done/pt_train_$f.done \
    && echo -e "$(date -Iseconds)\textract:pt_train_$f\tcomplete\t" >> $R/logs/status.tsv
done
echo -e "$(date -Iseconds)\tLANE_PT_TRAIN\tdone\t" >> $R/logs/status.tsv
