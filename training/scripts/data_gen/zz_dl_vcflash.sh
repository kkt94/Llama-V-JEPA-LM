#!/bin/bash
# (10-19 11:20) VideoChat-Flash-Training-Data (OpenGVLab, Apache-2.0, videos in the repo), contamination-safe subsets only, priority first.
# Left out: STAR / NTU (MVBench, TVBench, MVP), TGIF (TOMATO), Kinetics (Vinoground via VATEX), SSv2 / CLEVRER / NExT-QA (held already).
source /data/llama_vjepa_lm/scripts/env.sh
D=$R/data/VideoChat-Flash; DL=$R/training/scripts/download/dl_hf.py
p1=("annotations/video/*" "timeaware_subset/timeit_COIN-TimeIT-COIN_10k/*" "shortvideo_subset/caption_youcook2-youcook2-train_debug_9k/*" \
    "longvid_subset/htstep_eventcount_2k.zip" "longvid_subset/htstep_eventrelationship_1k.zip" "longvid_subset/htstep_eventunderstanding_1k.zip" \
    "shortvideo_subset/vqa_ego_qa-ego_qa-train_8k/*" "shortvideo_subset/caption_vidln_oops-vidln-oops_train_11k/*" "shortvideo_subset/caption_vidln_uvo_sparse-vidln-uvo_sparse_train_6k/*" \
    "shortvideo_subset/caption_vidln_uvo_dense-vidln-uvo_dense_train_1k/*" "shortvideo_subset/caption_vidln_ovis-vidln-ovis_train_1k.zip" "shortvideo_subset/caption_sharegpt4o-sharegpt4o_3k/*")
p2=("longvid_subset/ego4dhcap_eventunderstanding_2k_zip/*" "shortvideo_subset/lsmdc-lsmdc_297k/*" "timeaware_subset/timeit_ANet-TimeIT-Activitynet_Captions_11k/*" \
    "timeaware_subset/grounding_ANetRTL-ActivityNet-RTL-ANet_RTL_34k/*" "timeaware_subset/timeit_DiDeMo-TimeIT-DiDeMo_33k/*" "timeaware_subset/timeit_QuerYD-TimeIT-QuerYD_15k/*" \
    "timeaware_subset/timeit_HiREST-TimeIT-HiREST_1k/*" "timeaware_subset/grounding_ANetHL-ANet-HL-ANet_HL2_11k/*")
nice -n 10 $PY -I $DL vcflash_p1 dataset OpenGVLab/VideoChat-Flash-Training-Data $D --allow "${p1[@]}" --workers 8
nice -n 10 $PY -I $DL vcflash_p2 dataset OpenGVLab/VideoChat-Flash-Training-Data $D --allow "${p2[@]}" --workers 8
echo "$(date -Iseconds) VCFLASH DONE $(du -sh $D | cut -f1)" >> $R/logs/dl/vcflash.log
