#!/bin/bash
# allowlist (now incl. TempCompass / TOMATO / PT-test ids) -> llava + sharegpt manifests -> capcontrast rows kept only
# if their video is still in the rebuilt source manifest, prep_exists refreshed from disk
set -e
source /data/llama_vjepa_lm/scripts/env.sh
R=/data/llama_vjepa_lm; cd $R
$PY -I repo/scripts/data/build_allowlist.py > logs/rebuild_allowlist.log 2>&1
$PY -I repo/scripts/data/build_stage1_llava.py > logs/rebuild_llava.log 2>&1
$PY repo/scripts/data/build_stage1_sharegpt.py > logs/rebuild_sharegpt.log 2>&1
$PY - <<'PY'
import pandas as pd, os
R = "/data/llama_vjepa_lm/manifests/stage1"
for src, root in (("sharegpt", "/data/llama_vjepa_lm/prep/sharegpt_2fps"), ("llava", "/data/llama_vjepa_lm/prep/llava_384_15fps")):
    keep = set(pd.read_parquet(f"{R}/{src}.parquet", columns=["video"]).video)
    for suf in ("", "_heldout"):
        p = f"{R}/capcontrast_{src}{suf}.parquet"
        d = pd.read_parquet(p); n0 = len(d)
        d = d[d.video.isin(keep)].copy()
        d["prep_exists"] = d.video.map(lambda v: os.path.exists(f"{root}/{v}"))
        d.to_parquet(p, index=False)
        print(p.split("/")[-1], n0, "->", len(d), "ready", int(d.prep_exists.sum()))
PY
echo REBUILD_DONE
