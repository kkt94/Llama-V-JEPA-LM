#!/bin/bash
# (10-26, main run soon) every EPIC-derived question set in parallel lanes on w3 (same generators / audits as zz_w3_prep.sh, independent
# chains side by side): epic | seq | egoturn | v13 -> balance audit | v14 -> v16 -> v16 audit; then the nx13 config
R=/data/llama_vjepa_lm; PY=python; cd $R/logs
run() { for g in "$@"; do echo "== $(date -Iseconds) $g"; nice -n 5 $PY -I $g 2>&1 | grep -v -i warn | grep -v "example:" | tail -25; done; }
run zz_qgen_epic.py > zz_par_epic.out 2>&1 &
run zz_qgen_seq.py > zz_par_seq.out 2>&1 &
run zz_qgen_egoturn.py > zz_par_egoturn.out 2>&1 &
run zz_qgen_v13.py zz_balance_audit_v13.py > zz_par_v13.out 2>&1 &
run zz_qgen_v14.py zz_qgen_v16.py zz_audit_v16.py > zz_par_v14.out 2>&1 &
wait; $PY -I zz_make_nx13_cfg.py nx13_Q 2>&1 | grep -v "   new" | tail -3
echo -e "$(date -Iseconds)\tnx13 data ready (EPIC questions, parallel)" >> $R/data/opendl/_meta/status.tsv; echo "QGEN PAR DONE"
