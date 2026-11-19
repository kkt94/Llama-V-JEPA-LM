#!/bin/bash
# (10-24 evening) right BEFORE launching nx5_Q: rebuild the open-data manifests from what is on disk (EPIC grew), keep rows with video, freeze both
# opendl and opendl2, build the nx4-style base (lr x1.5) then nx5_Q (zz_make_nx5_cfg.py), CPU-check every new / changed source.
cd /data/llama_vjepa_lm; R=/data/llama_vjepa_lm; PY=python; export ARROW_DEFAULT_MEMORY_POOL=system CUDA_VISIBLE_DEVICES=""
keep() { $PY -I -c "
import pandas as pd, sys
for f in sys.argv[1:]:
    d = pd.read_parquet(f); n = len(d); d = d[d.prep_exists.astype(str).isin(['True','true','1'])].reset_index(drop=True); d.to_parquet(f); print(f.split('/')[-1], n, '->', len(d), flush=True)" "$@"; }
for g in zz_qgen_charades_ego zz_qgen_epic zz_qgen_kinetics_countix zz_qgen_recog; do $PY -I logs/$g.py 2>&1 | grep -E "rows|missing|videos"; done
keep manifests/stage1/opendl/*.parquet manifests/stage1/opendl2/*.parquet
TS=$(date +%m%d_%H%M); F=manifests/stage1/opendl_frozen_$TS; F2=manifests/stage1/opendl2_frozen_$TS
mkdir -p $F $F2 && cp manifests/stage1/opendl/*.parquet $F/ && cp manifests/stage1/opendl2/*.parquet $F2/ && echo "frozen: $F $F2"
$PY -I logs/zz_make_nx4_cfg.py zz_nx5base --lr_mult 1.5 --mdir $R/$F | tail -1
$PY -I logs/zz_make_nx5_cfg.py nx5_Q --base repo/configs/single/zz_nx5base.yaml --mdir2 $R/$F2 && rm repo/configs/single/zz_nx5base.yaml
$PY -I logs/zz_nx4_datacheck.py configs/single/nx5_Q.yaml llava,ssv2_real,ssv2_count,ssv2_color,chego_done,chego_state,epic_,ssv2_obj 2>&1 | grep -vE "Warn|warn"
$PY -I logs/zz_lenbias_all.py repo/configs/single/nx5_Q.yaml 2>&1 | grep -E "^llava|^ssv2_|^chego_(done|state)"
