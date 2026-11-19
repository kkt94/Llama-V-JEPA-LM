"""Tiny stage-1 manifest (first N allowed LLaVA records whose 384p clip already exists) for smoke / overfit runs."""
import json, os, sys
import pandas as pd
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data")
from build_stage1_llava import ALLOW, PREP, SRC, convert  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
OUT = f"/data/llama_vjepa_lm/manifests/stage1/llava_smoke{N}.parquet"
rows, cache = [], {}
with open(ALLOW) as f:
    for line in f:
        r = json.loads(line)
        if r["file"].startswith("llava_hound") or not os.path.exists(f"{PREP}/{r['video']}"):
            continue
        if r["file"] not in cache:
            cache = {r["file"]: json.load(open(f"{SRC}/{r['file']}"))}
        d = cache[r["file"]][r["idx"]]
        msgs = convert(d["conversations"])
        rows.append({"video": r["video"], "prep_exists": True, "conversations": json.dumps(msgs, ensure_ascii=False),
                     "file": r["file"], "idx": r["idx"]})
        if len(rows) >= N:
            break
os.makedirs(os.path.dirname(OUT), exist_ok=True)
pd.DataFrame(rows).to_parquet(OUT, index=False)
print(OUT, len(rows), rows[0]["video"], json.loads(rows[0]["conversations"])[0]["content"][:80])
