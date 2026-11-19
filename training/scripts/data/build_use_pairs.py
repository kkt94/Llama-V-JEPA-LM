"""Paired videos for L_use (spec 6.5, 'use failure'): the same question on two videos whose correct answers differ.
Swapping the two videos' evidence must make the original answer less likely.
  use_pairs_plausibility.parquet  physics plausibility pairs (same source clip, possible vs impossible; same prompt)
  use_pairs_nvidia.parquet        NVIDIA world pairs of pair_type 1 (same question, another camera of the same world,
                                  the answer changes); one prompt per pair (MC with the SAME option order, or open)
Every row carries pair_id (two rows per id). Train splits only.
"""
import json

import numpy as np
import pandas as pd

R = "/data/llama_vjepa_lm/manifests"
VIS = "<|jepa_evidence|>"
SUFFIX = "Answer with the option's letter from the given choices directly."
LETTERS = "ABCD"
SPACES = [["Yes.", "No."], ["Left.", "Right.", "Up.", "Down."]]


def main():
    p = pd.read_parquet(f"{R}/stage1/plausibility.parquet")
    p = p[p.groupby("pair_id").pair_id.transform("size") == 2]
    p[["video", "prep_exists", "pair_id", "conversations"]].to_parquet(f"{R}/stage1/use_pairs_plausibility.parquet", index=False)
    print("plausibility pairs", p.pair_id.nunique())

    d = pd.read_parquet(f"{R}/stage2/nvidia_groups.parquet")
    d = d[(d.pair_id >= 0) & (d.pair_type == 1)]
    d = d[d.groupby("pair_id").pair_id.transform("size") == 2]
    rng = np.random.default_rng(0)
    rows = []
    for pid, g in d.groupby("pair_id"):
        space = next(s for s in SPACES if g.answer.iloc[0] in s)
        mc = rng.random() < 0.5
        opts = [o.rstrip(".") for o in rng.permutation(space)]
        for r in g.itertuples():
            if mc:
                q = r.question.split(" Answer left, right, up or down.")[0]
                q += "\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)) + f"\n{SUFFIX}"
                a = LETTERS[opts.index(r.answer.rstrip("."))] + "."
            else:
                q, a = r.question, r.answer
            rows.append({"video": r.video, "prep_exists": bool(r.prep_exists), "t0": float(r.t0), "t1": float(r.t1),
                         "crop_cx": r.crop_cx, "pair_id": int(pid),
                         "conversations": json.dumps([{"role": "user", "content": f"{VIS}\n{q}"}, {"role": "assistant", "content": a}])})
    out = pd.DataFrame(rows)
    out.to_parquet(f"{R}/stage1/use_pairs_nvidia.parquet", index=False)
    g = out.groupby("pair_id")
    same_q = (g.conversations.apply(lambda s: s.map(lambda c: json.loads(c)[0]["content"]).nunique()) == 1).mean()
    diff_a = (g.conversations.apply(lambda s: s.map(lambda c: json.loads(c)[1]["content"]).nunique()) == 2).mean()
    print("nvidia pairs", out.pair_id.nunique(), "same prompt", same_q, "answers differ", diff_a)


if __name__ == "__main__":
    main()
