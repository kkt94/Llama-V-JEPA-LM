"""Stage-1 manifest from PLM-Video-Human RTLoc on SA-V train videos (real video, human-written, time-grounded).

Only RTLoc is used. RCap/RDCap need the referred masklet's region, but PLM's `masklet_id` could not be mapped to the
public SA-V masklets: on 1,794 RDCap rows with "Out of frame." segments, visibility agreement was 0.72-0.76 for the
manual / auto / concatenated id hypotheses vs 0.62 for "always visible", and even the best-matching masklet of the video
beats the runner-up by a median of only 0.027 (scripts/checks/check_plm_masklet_namespace.py). A wrong region would
teach wrong evidence, so those tasks are left out. RTLoc only needs the clip: caption -> time span.

PLM frame indices are SA-V annotation frames at 6 fps (total_frames == number of annotated frames), seconds = frame / 6.
Evidence = the whole clip (t0 = 0, t1 = duration), sampled with the usual paired-uniform sampler.
Usage under the Llama backbone: PLM's "Llama" model-naming clause coincides with the Llama licence requirement.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, "/data/llama_vjepa_lm/training")
from jepa_llm.data.chat_format import VIS  # noqa: E402

R = "/data/llama_vjepa_lm"
PREP = f"{R}/prep/sav_384_24fps"
OUT = f"{R}/manifests/stage1/plm_rtloc.parquet"
FPS_ANN = 6.0


def main():
    df = pd.read_parquet(f"{R}/data/PLM-Video-Human/rtloc/plm_rtloc_train.parquet")
    allowed = {}  # member mp4 -> tar folder, SA-V train only (contamination allowlist)
    for l in open(f"{R}/manifests/allowlist/sav_allowlist.jsonl"):
        a = json.loads(l)
        allowed[a["member"]] = a["tar"][:-4]
    assert allowed, "empty SA-V allowlist"
    rows, skipped = [], 0
    for r in df.itertuples(index=False):
        vid = r.video[:-4]
        if r.video not in allowed:
            skipped += 1
            continue
        rel = f"sav_train/{allowed[r.video]}/{vid}.mp4"
        f0, f1, n = int(r.start_frame), int(r.end_frame), int(r.total_frames)
        q = f'When does this happen: "{r.caption.strip()}"? Answer with the start and end time in seconds.'
        a = f"From {f0 / FPS_ANN:.1f}s to {(f1 + 1) / FPS_ANN:.1f}s."
        rows.append({"task": "rtloc", "video": rel, "prep_exists": os.path.exists(f"{PREP}/{rel}"),
                     "t0": 0.0, "t1": n / FPS_ANN, "region": [],
                     "conversations": json.dumps([{"role": "user", "content": f"{VIS}\n{q}"},
                                                  {"role": "assistant", "content": a}], ensure_ascii=False)})
    assert rows, "no RTLoc row matched the SA-V allowlist"
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"rtloc rows={len(out)} videos={out.video.nunique()} prep_exists={int(out.prep_exists.sum())} "
          f"skipped_not_in_sav_train_allowlist={skipped}")


if __name__ == "__main__":
    main()
