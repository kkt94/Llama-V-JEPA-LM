#!/bin/bash
# Data boost (2026-10-09): SSv2 labels (licence accepted by the user; public link works after login) and the CLEVRER TRAIN
# split (videos, QA incl. predictive/counterfactual, object trajectory + collision annotations). CLEVRER validation is
# NOT downloaded for training: MVP evaluates on CLEVRER validation videos.
source /data/llama_vjepa_lm/scripts/env.sh
U=$R/training/scripts/download/dl_url.sh
B=https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/data/something-something-v2
curl -sSL --retry 5 -o $R/data/ssv2_labels.zip "$B/20bn-something-something-download-package-labels.zip" && \
  $PY -I $R/training/scripts/download/unzip_py.py $R/data/ssv2_labels.zip $R/data/ssv2_labels && \
  echo -e "$(date -Iseconds)\tssv2_labels\tcomplete\t$(ls $R/data/ssv2_labels/* | tr '\n' ' ')" >> $R/logs/status.tsv
C=http://data.csail.mit.edu/clevrer; D=$R/data/clevrer
for f in questions/train.json annotations/train/annotation_train.zip videos/train/video_train.zip; do
  bash $U clevrer:$(basename $f) $C/$f $D/$(basename $f)
done
for z in annotation_train video_train; do
  $PY -I $R/training/scripts/download/unzip_py.py $D/$z.zip $D/$z && echo -e "$(date -Iseconds)\textract:clevrer_$z\tcomplete\t" >> $R/logs/status.tsv
done
echo -e "$(date -Iseconds)\tLANE_BOOST\tdone\t" >> $R/logs/status.tsv
