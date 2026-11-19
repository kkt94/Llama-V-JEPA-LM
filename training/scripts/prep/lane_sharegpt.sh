#!/bin/bash
# ShareGPTVideo frame chunks -> 2 fps mp4 per clip (prep_sharegpt_frames.py). Run the same lane on several hosts:
# chunks are claimed with mkdir locks, a finished chunk leaves a .done marker. train_300k first (QA + captions used by
# LLaVA-Video's llava_hound and ShareGPTVideo's 900k QA), then train_600k (caption-only videos).
#   usage: lane_sharegpt.sh <n_parallel_chunks>
R=/data/llama_vjepa_lm; D=$R/data/ShareGPTVideo; O=$R/prep/sharegpt_2fps; L=$O/_locks; ST=$R/logs/status.tsv
mkdir -p $L
run_chunk() {
  local f=$1 part=$(basename $(dirname $1)) k=$(basename $1 .tar.gz)
  mkdir $L/${part}_$k 2>/dev/null || return 0
  [ -f $L/${part}_$k.done ] && return 0
  echo -e "$(date -Iseconds)\tprep_sharegpt:${part}_$k@$(hostname)\tstart\t" >> $ST
  if ENC_THREADS=${ENC_THREADS:-6} python -I $R/training/scripts/prep/prep_sharegpt_frames.py $f $O/$part \
        > $O/_logs_${part}_$k.txt 2>&1; then
    touch $L/${part}_$k.done; echo -e "$(date -Iseconds)\tprep_sharegpt:${part}_$k\tcomplete\t$(tail -1 $O/_logs_${part}_$k.txt)" >> $ST
  else
    echo -e "$(date -Iseconds)\tprep_sharegpt:${part}_$k\tFAILED\t$(tail -1 $O/_logs_${part}_$k.txt)" >> $ST; rmdir $L/${part}_$k
  fi
}
export -f run_chunk; export R D O L ST
mkdir -p $O/_logs
ls $D/train_300k/chunk_*.tar.gz | sort -V > $O/_chunks_$(hostname).txt; ls $D/train_600k/chunk_*.tar.gz | sort -V >> $O/_chunks_$(hostname).txt
xargs -a $O/_chunks_$(hostname).txt -P ${1:-4} -I{} bash -c 'run_chunk {}'
echo -e "$(date -Iseconds)\tLANE_SHAREGPT@$(hostname)\tdone\t" >> $ST
