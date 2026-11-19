#!/bin/bash
# (10-26) after the RoboVQA clips are extracted: questions (zz_qgen_robovqa.py), text-only audit of the yes / no sets, nx13 configs refreshed
R=/data/llama_vjepa_lm; ST=$R/data/opendl/_meta/status.tsv; PY=python; cd $R
until grep -q "robovqa clips extracted" $ST; do sleep 120; done
(cd logs && nice -n 10 $PY -I zz_qgen_robovqa.py) > logs/zz_after_robovqa.out 2>&1
sed 's/if os.path.basename(p).split("_")\[0\] in ("fmt", "egocam", "next", "phys")/if os.path.basename(p).startswith("rvqa_")/' logs/zz_balance_audit_v13.py > logs/zz_audit_rvqa.py
sed -i 's/    d = pd.read_parquet(f"{O}\/{k}.parquet"); P = \[parse(c) for c in d.conversations\]/    d = pd.read_parquet(f"{O}\/{k}.parquet"); P = [parse(c) for c in d.conversations]\n    if any(o is None and g not in ("yes", "no") for _, o, g in P): continue/' logs/zz_audit_rvqa.py
nice -n 10 $PY -I logs/zz_audit_rvqa.py >> logs/zz_after_robovqa.out 2>&1
$PY -I logs/zz_make_nx13_cfg.py nx13_Q >> logs/zz_after_robovqa.out 2>&1; $PY -I logs/zz_make_nx13_cfg.py nx13w_Q --sqrt >> logs/zz_after_robovqa.out 2>&1
echo -e "$(date -Iseconds)\tnx13 data refreshed (RoboVQA)" >> $ST
