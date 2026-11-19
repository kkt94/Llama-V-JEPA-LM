#!/bin/bash
# (10-26) after every EgoDex split is prepared (the worker that finishes last regenerates the EgoDex questions): Robo-format EgoDex questions over
# all splits, then the nx13 configs. Status line "nx13 data refreshed" when done.
R=/data/llama_vjepa_lm; ST=$R/data/opendl/_meta/status.tsv; PY=python; cd $R
until grep -q "egodex questions regenerated (all splits)" $ST; do sleep 300; done
(cd logs && nice -n 10 $PY -I zz_qgen_robo.py) > logs/zz_after_egodex.out 2>&1
for f in $R/manifests/stage1/opendl3/egodex_*.parquet; do echo "$f"; done >> logs/zz_after_egodex.out
$PY -I logs/zz_make_nx13_cfg.py nx13_Q >> logs/zz_after_egodex.out 2>&1; $PY -I logs/zz_make_nx13_cfg.py nx13w_Q --sqrt >> logs/zz_after_egodex.out 2>&1
echo -e "$(date -Iseconds)\tnx13 data refreshed (EgoDex all splits)" >> $ST
