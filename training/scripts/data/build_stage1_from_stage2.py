"""Single-stage recipe: the stage-2 grounded questions (NVIDIA worlds, SA-V masklets, PerceptionTest tracks) as ordinary
stage-1 QA rows. No composer / gold spec: the model answers from the whole clip.

- every question is kept, shown unoccluded (the blanked bin x cell only existed to exercise masked-prediction evidence;
  the occluded rows are independent questions, not twins); duplicates of (video, window, crop, question) are removed
- half of the rows become multiple choice in the evaluator's format (yes/no -> 2 options, left/right/up/down -> 4
  options, shuffled), the other half keep the short open answer
- window t0/t1 and the crop centre the question generator used are kept (positions in the question refer to that crop)
- only the train splits (the *_heldout manifests stay out)
Output: manifests/stage1/{nvidia_groups,sav_tracks,pt_tracks}_qa.parquet
"""
import json

import numpy as np
import pandas as pd

R = "/data/llama_vjepa_lm"
VIS = "<|jepa_evidence|>"
SUFFIX = "Answer with the option's letter from the given choices directly."
LETTERS = "ABCD"
SPACES = [["Yes.", "No."], ["Left.", "Right.", "Up.", "Down."]]


def main():
    rng = np.random.default_rng(0)
    for name in ("nvidia_groups", "sav_tracks", "pt_tracks"):
        d = pd.read_parquet(f"{R}/manifests/stage2/{name}.parquet")
        for c in ("crop_cx", "crop_cy"):
            if c not in d:
                d[c] = np.nan
        d = d.drop_duplicates(["video", "t0", "t1", "crop_cx", "crop_cy", "question"])
        rows = []
        for r in d.itertuples():
            space = next(s for s in SPACES if r.answer in s)
            if rng.random() < 0.5:
                opts = [o.rstrip(".") for o in rng.permutation(space)]
                q = r.question.split(" Answer left, right, up or down.")[0]
                q = q + "\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)) + f"\n{SUFFIX}"
                a, kind = LETTERS[opts.index(r.answer.rstrip("."))] + ".", "mc"
            else:
                q, a, kind = r.question, r.answer, "open"
            rows.append({"video": r.video, "prep_exists": bool(r.prep_exists), "t0": float(r.t0), "t1": float(r.t1),
                         "crop_cx": r.crop_cx, "crop_cy": r.crop_cy, "kind": kind, "op": r.op,
                         "conversations": json.dumps([{"role": "user", "content": f"{VIS}\n{q}"},
                                                      {"role": "assistant", "content": a}])})
        out = pd.DataFrame(rows)
        out.to_parquet(f"{R}/manifests/stage1/{name}_qa.parquet", index=False)
        print(name, "rows", len(out), "videos", out.video.nunique(), out.groupby(["op", "kind"]).size().to_dict())
        print("  ", json.loads(out[out.kind == "mc"].conversations.iloc[0])[0]["content"].replace("\n", " | "))


if __name__ == "__main__":
    main()
