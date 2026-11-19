#!/bin/bash
# PerceptionTest TRAIN videos -> 384p, native 30 fps (object-track boxes are on 30-fps frame ids), GOP 15.
R=/data/llama_vjepa_lm; I=$R/data/perception_test_train/extracted/videos; O=$R/prep/pt_train_384; ST=$R/logs/status.tsv
mkdir -p $O; ls $I | grep '\.mp4$' | xargs -P ${JOBS:-16} -I{} bash $R/training/scripts/prep/transcode_one.sh "$I/{}" "$O/{}" native 15
n=$(ls $I | grep -c '\.mp4$'); m=$(ls $O | grep -c '\.mp4$')
echo -e "$(date -Iseconds)\tprep_pt_train\t$([ $n -eq $m ] && echo complete || echo INCOMPLETE)\t$m/$n" >> $ST
