#!/bin/bash
# (10-27) Oops: wait for extraction -> transcode -> question generator (w3, CPU only)
R=/data/llama_vjepa_lm; cd $R; PY=python
until grep -q "OOPS EXTRACT DONE\|Traceback" logs/zz_oops_extract.out; do sleep 20; done
$PY -I logs/zz_oops_prep.py > logs/zz_oops_prep.out 2>&1
$PY -I logs/zz_qgen_oops.py > logs/zz_qgen_oops.out 2>&1
