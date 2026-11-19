"""Training-data inventory per recipe stage, from the manifests actually consumed by the trainers.

For each manifest: rows, unique videos, rows whose 384p clip exists now (prep_exists re-checked on disk, not the value
stored at build time), answer turns, and the stage that consumes it. Writes manifests/inventory.json and prints a table.
"""
import json
import os
import sys

import pandas as pd

R = "/data/llama_vjepa_lm"
SOURCES = [
    # name, manifest, prep root, stage(s), kind
    ("llava_video", f"{R}/manifests/stage1/llava.parquet", f"{R}/prep/llava_384_15fps", "1 (+3 native)", "general video QA / captions"),
    ("plm_rtloc", f"{R}/manifests/stage1/plm_rtloc.parquet", f"{R}/prep/sav_384_24fps", "1 (+3 native)", "real-video temporal grounding"),
    ("nvidia_state_qa", f"{R}/manifests/stage1/nvidia_state.parquet", f"{R}/prep/nvidia_rgb_384", "1", "physics state QA (gold time/region)"),
    ("ssv2", f"{R}/manifests/stage1/ssv2.parquet", f"{R}/bench/mvp_sources/extracted/ssv2/20bn-something-something-v2", "1 (+2/3 general)", "minimal-contrast action MC + descriptions"),
    ("clevrer", f"{R}/manifests/stage1/clevrer.parquet", f"{R}/data/clevrer/video_train", "1 (+2/3 general)", "collision physics: descriptive / explanatory / predictive / counterfactual"),
    ("pt_train_mc", f"{R}/manifests/stage1/pt_train_mc.parquet", f"{R}/prep/pt_train_384", "1 (+2/3 general)", "PerceptionTest train MC"),
    ("nvidia_groups", f"{R}/manifests/stage2/nvidia_groups.parquet", f"{R}/prep/nvidia_rgb_384", "2, 3", "world groups / pairs / gold specs"),
    ("pt_tracks", f"{R}/manifests/stage2/pt_tracks.parquet", f"{R}/prep/pt_train_384", "2, 3", "real-video object tracks, gold specs"),
    ("sav_tracks", f"{R}/manifests/stage2/sav_tracks.parquet", f"{R}/prep/sav_384_24fps", "2, 3", "real-video masklets: direction / permanence, gold specs, 50% occluded"),
]


def main():
    out = []
    for name, path, root, stage, kind in SOURCES:
        if not os.path.exists(path):
            out.append({"name": name, "stage": stage, "kind": kind, "status": "not built"})
            continue
        df = pd.read_parquet(path)
        vids = df.video.drop_duplicates()
        exists = vids.map(lambda v: os.path.exists(f"{root}/{v}"))
        ok = set(vids[exists])
        rec = {"name": name, "stage": stage, "kind": kind, "rows": len(df), "videos": int(len(vids)),
               "videos_ready": int(exists.sum()), "rows_ready": int(df.video.isin(ok).sum())}
        if "conversations" in df:
            rec["answer_turns"] = int(df.conversations.map(lambda c: sum(m["role"] == "assistant" for m in json.loads(c))).sum())
        if "pair_id" in df:
            rec["pairs"] = int((df.pair_id >= 0).sum() // 2)
            rec["groups"] = int(df.group_id.nunique())
        out.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(out, open(f"{R}/manifests/inventory.json", "w"), indent=1)
    for r in out:
        if r.get("status"):
            print(f"{r['name']:16s} {r['stage']:14s} {r['status']}")


if __name__ == "__main__":
    sys.exit(main())
