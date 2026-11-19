#!/bin/bash
# MVP source videos: raw archives only (postprocessing per MVP Makefile is done later).
# Skipped tonight: SSv2 (Qualcomm license acceptance needed), Language Table (tensorflow + gs://).
source /data/llama_vjepa_lm/scripts/env.sh
U=$R/scripts/dl_url.sh; D=$R/bench/mvp_sources/raw
$U mvp:inflevel https://pub-7320908bcb5b4cdea63c22bc2a38600c.r2.dev/inflevel_lab.tar.gz $D/inflevel_lab.tar.gz
$U mvp:intphys_dev https://download-intphys.cognitive-ml.fr/dev.tar.gz $D/intphys_dev.tar.gz
$U mvp:clevrer_val http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip $D/clevrer_video_validation.zip
$U mvp:grasp 'https://drive.usercontent.google.com/download?id=1_xPzN0MS3vVlci4yHL5FBH8N_ZpL6mt8&export=download&confirm=t' $D/grasp_videos.zip
$U mvp:charades_480 https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip $D/Charades_v1_480.zip
$PY -I $R/scripts/dl_hf.py mvp_vinoground dataset HanSolo9682/Vinoground $D/hf_vinoground
echo -e "$(date -Iseconds)\tLANE_MVP\tdone\t" >> $R/logs/status.tsv
