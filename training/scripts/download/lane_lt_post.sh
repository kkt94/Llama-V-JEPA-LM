#!/bin/bash
# MVP robot subset: wait for local Language Table tfrecords, then run MVP's download_lt.py / merge_lt.py against the
# local copy (gs:// path rewritten) in a separate TF env. Output: bench/mvp_sources/extracted/language_table_slow/*.mp4
source /data/llama_vjepa_lm/scripts/env.sh
ST=$R/logs/status.tsv; E=/path/to/venv
if [ ! -x $E/bin/python ]; then
  ~/.local/bin/uv venv --python /usr/bin/python3.12 $E && \
  ~/.local/bin/uv pip install --python $E/bin/python tensorflow-cpu tensorflow-datasets datasets pillow tqdm numpy opencv-python-headless \
    || { echo -e "$(date -Iseconds)\tlt_post\tFAILED\tenv" >> $ST; exit 1; }
fi
until grep -qP '\tlanguage_table\t(complete|INCOMPLETE)' $ST; do sleep 120; done
grep -qP '\tlanguage_table\tcomplete' $ST || { echo -e "$(date -Iseconds)\tlt_post\tFAILED\ttfrecords incomplete" >> $ST; exit 1; }
W=$R/bench/mvp_sources/extracted/lt_work; mkdir -p $W/language_table
C=/data/llama_vjepa_lm/code/minimal_video_pairs/setup
sed "s#gs://gresearch/robotics/language_table'#$R/bench/mvp_sources/raw/language_table'#; s#^import os#import os\nimport numpy as np\nfrom datasets import load_dataset#" $C/download_lt.py > $W/download_lt_local.py
cp $C/merge_lt.py $W/
echo -e "$(date -Iseconds)\tlt_post\tstart\t" >> $ST
cd $W && $E/bin/python download_lt_local.py > lt_download.log 2>&1 && $E/bin/python merge_lt.py > lt_merge.log 2>&1 \
  && mv $W/language_table_slow $R/bench/mvp_sources/extracted/language_table_slow \
  && echo -e "$(date -Iseconds)\tlt_post\tcomplete\tmp4=$(ls $R/bench/mvp_sources/extracted/language_table_slow | wc -l)" >> $ST \
  || echo -e "$(date -Iseconds)\tlt_post\tFAILED\tsee $W/*.log" >> $ST
