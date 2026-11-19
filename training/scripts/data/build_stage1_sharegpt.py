"""Stage-1 / general data from ShareGPTVideo (GPT-4V annotated WebVid / VIDAL / ActivityNet clips), 2 fps mp4s made by
scripts/prep/prep_sharegpt_frames.py.

  qa     qa/chatgpt_qa_900k.jsonl         3 QA per train_300k video (LLaVA-Video's llava_hound uses a 240k subset;
                                          those llava.parquet rows stay off so nothing is counted twice)
  cap    sft/video_caption_300k.jsonl     detailed caption per train_300k video
  cap6   pretrain/video_caption_pretrain  detailed caption for the train_600k videos only (its other 300k overlap `cap`)
Only records in manifests/allowlist/sharegptvideo_allowlist.jsonl (benchmark video ids removed) are used.
prep_exists is read from disk: re-run after the frame conversion finishes.
Output: manifests/stage1/sharegpt.parquet (video = train_300k/<id>.mp4 or train_600k/<id>.mp4 under prep/sharegpt_2fps)
"""
import collections
import json
import os

import pandas as pd

R = "/data/llama_vjepa_lm"
SRC = f"{R}/data/ShareGPTVideo/video_instruction/train"
PREP = f"{R}/prep/sharegpt_2fps"
VIS = "<|jepa_evidence|>"
FILES = {"qa": "qa/chatgpt_qa_900k.jsonl", "cap": "sft/video_caption_300k.jsonl", "cap6": "pretrain/video_caption_pretrain.jsonl"}


def convert(convs):
    msgs = [{"role": {"human": "user", "gpt": "assistant"}[c["from"]], "content": c["value"]} for c in convs]
    q = msgs[0]["content"].replace("<video>\n", "").replace("\n<video>", "").replace("<video>", "").strip()
    msgs[0] = {"role": "user", "content": f"{VIS}\n{q}"}
    return msgs


def main():
    allow = collections.defaultdict(set)
    for line in open(f"{R}/manifests/allowlist/sharegptvideo_allowlist.jsonl"):
        r = json.loads(line)
        allow[r["file"]].add(r["idx"])
    assert all(allow[f] for f in FILES.values()), "allowlist has no records for one of the files: the filter would be vacuous"
    on_disk = {p: {f[:-4] for f in os.listdir(f"{PREP}/{p}") if f.endswith(".mp4") and ".part" not in f}
               if os.path.isdir(f"{PREP}/{p}") else set() for p in ("train_300k", "train_600k")}
    part300 = set()
    rows, stats = [], collections.Counter()
    for kind in ("qa", "cap", "cap6"):
        fname = FILES[kind]
        for idx, line in enumerate(open(f"{SRC}/{fname}")):
            if idx not in allow[fname]:
                stats[f"{kind}:not_allowed"] += 1
                continue
            d = json.loads(line)
            vid = d["video"]
            if kind == "cap6":
                if vid in part300:
                    stats["cap6:in_300k"] += 1        # caption already taken from video_caption_300k
                    continue
                part = "train_600k"
            else:
                part = "train_300k"; part300.add(vid)
            rows.append({"video": f"{part}/{vid}.mp4", "prep_exists": vid in on_disk[part], "kind": kind,
                         "conversations": json.dumps(convert(d["conversations"]), ensure_ascii=False)})
    df = pd.DataFrame(rows)
    df.to_parquet(f"{R}/manifests/stage1/sharegpt.parquet", index=False)
    print(dict(stats))
    print(df.groupby("kind").agg(rows=("video", "size"), videos=("video", "nunique"), ready=("prep_exists", "sum")).to_string())
    print(json.loads(df.conversations.iloc[0]))


if __name__ == "__main__":
    main()
