"""Stage-1 training manifest from LLaVA-Video-178K, restricted to the contamination allowlist.

Output: /data/llama_vjepa_lm/manifests/stage1/llava.parquet, one row per allowed record:
  source, kind (cap|mc|oe), file, idx, video (path relative to the 384p prep root), prep_exists, n_turns, conversations
  (JSON list of {"role", "content"}, first user turn carries the evidence placeholder).
llava_hound records point at ShareGPTVideo frame folders, not mp4 files; they are written with kind="hound" and
prep_exists=False until a frame loader exists. Re-run after preprocessing finishes to refresh prep_exists.
"""
import collections
import json
import os
import sys

import pandas as pd

sys.path.insert(0, "/data/llama_vjepa_lm/training")
from jepa_llm.data.chat_format import VIS  # noqa: E402

R = "/data/llama_vjepa_lm"
SRC = f"{R}/data/LLaVA-Video-178K"
PREP = f"{R}/prep/llava_384_15fps"
ALLOW = f"{R}/manifests/allowlist/llava_video_allowlist.jsonl"
OUT = f"{R}/manifests/stage1/llava.parquet"
ROLE = {"human": "user", "gpt": "assistant"}


def convert(convs):
    msgs = [{"role": ROLE[c["from"]], "content": c["value"]} for c in convs]
    first = msgs[0]["content"]
    first = first.replace("<image>\n", VIS + "\n", 1) if "<image>\n" in first else (
        first.replace("<image>", VIS, 1) if "<image>" in first else VIS + "\n" + first)
    msgs[0] = {"role": "user", "content": first.replace("<image>", "")}
    return msgs


def main():
    by_file = collections.defaultdict(list)
    with open(ALLOW) as f:
        for line in f:
            r = json.loads(line)
            by_file[r["file"]].append(r)
    rows = []
    for fname, recs in sorted(by_file.items()):
        data = json.load(open(f"{SRC}/{fname}"))
        kind = "hound" if fname.startswith("llava_hound") else ("cap" if "_cap_" in fname else ("mc" if "_mc_" in fname else "oe"))
        for r in recs:
            d = data[r["idx"]]
            assert d.get("video") == r["video"], (fname, r["idx"])
            msgs = convert(d["conversations"])
            exists = kind != "hound" and os.path.exists(f"{PREP}/{r['video']}")
            rows.append({"source": fname.split("/")[0], "kind": kind, "file": fname, "idx": r["idx"], "video": r["video"],
                         "prep_exists": exists, "n_turns": sum(m["role"] == "assistant" for m in msgs),
                         "conversations": json.dumps(msgs, ensure_ascii=False)})
        print(f"{fname}: {len(recs)}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(df.groupby("kind").agg(records=("idx", "size"), videos=("video", "nunique"), prep=("prep_exists", "sum"),
                                 turns=("n_turns", "sum")))


if __name__ == "__main__":
    main()
