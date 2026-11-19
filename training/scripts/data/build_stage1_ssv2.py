"""Stage-1 / general video-language data from Something-Something v2 TRAIN (human action templates with object names).

Two rows per video:
  mc    "Which action is shown?" with 4 options: the true label and 3 hard negatives = the most similar OTHER templates
        (word overlap, e.g. 'from left to right' vs 'from right to left', 'into' vs 'out of'), filled with the SAME
        objects, so the options differ only in the action/direction - a minimal-contrast question like MVP's.
        Same prompt format as the evaluator; answer = letter.
  open  "Describe the action in the video." -> the label sentence.
Videos: original 240p webm (decoded and upscaled on the fly, no transcode). Excluded: every SSv2 id used by MVP
(manifests/allowlist/benchmark_ids_mvp.txt). Validation/test splits are not used.
Output: manifests/stage1/ssv2.parquet
"""
import json
import os
import re

import numpy as np
import pandas as pd

R = "/data/llama_vjepa_lm"
LAB = f"{R}/data/ssv2_labels/labels"
VIDEOS = f"{R}/bench/mvp_sources/extracted/ssv2/20bn-something-something-v2"
LETTERS = "ABCD"
SUFFIX = "Answer with the option's letter from the given choices directly."
VIS = "<|jepa_evidence|>"


def fill(template, objs):
    out, k = template, 0
    while "[something]" in out:
        out = out.replace("[something]", objs[k] if k < len(objs) else "something", 1)
        k += 1
    return out[0].upper() + out[1:]


def words(t):
    return set(re.findall(r"[a-z]+", t.lower())) - {"something"}


def main():
    train = json.load(open(f"{LAB}/train.json"))
    templates = sorted({x["template"].replace("[something]", "[something]") for x in train})
    mvp = set()
    for line in open(f"{R}/manifests/allowlist/benchmark_ids_mvp.txt"):
        p = line.rstrip("\n").split("\t")
        if p and p[0] == "ssv2":
            mvp.add(p[1])
    assert mvp, "no MVP ssv2 ids read: the exclusion would be vacuous"
    # hard negatives: the 6 most similar other templates by Jaccard word overlap
    W = {t: words(t) for t in templates}
    near = {t: sorted((u for u in templates if u != t), key=lambda u: -len(W[t] & W[u]) / max(1, len(W[t] | W[u])))[:6]
            for t in templates}
    rng = np.random.default_rng(0)
    rows, excluded, missing = [], 0, 0
    for x in train:
        if x["id"] in mvp:
            excluded += 1
            continue
        path = f"{x['id']}.webm"
        if not os.path.exists(f"{VIDEOS}/{path}"):
            missing += 1
            continue
        t = x["template"]
        objs = x.get("placeholders", [])
        true = fill(t, objs)
        negs = [fill(u, objs) for u in rng.choice(near[t], 3, replace=False)]
        opts = [true] + [n for n in negs if n != true][:3]
        if len(opts) < 4:
            continue
        perm = rng.permutation(4)
        opts = [opts[i] for i in perm]
        ans = LETTERS[int(np.where(perm == 0)[0][0])]
        q = "Which action is shown in the video?\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)) + f"\n{SUFFIX}"
        for kind, qq, aa in (("mc", q, ans + "."), ("open", "Describe the action in the video.", true + ".")):
            rows.append({"video": path, "prep_exists": True, "kind": kind, "template": t,
                         "conversations": json.dumps([{"role": "user", "content": f"{VIS}\n{qq}"},
                                                      {"role": "assistant", "content": aa}])})
    df = pd.DataFrame(rows)
    df.to_parquet(f"{R}/manifests/stage1/ssv2.parquet", index=False)
    print(f"train {len(train)} | excluded (MVP ids) {excluded} | missing video {missing} | rows {len(df)} "
          f"videos {df.video.nunique()} templates {df.template.nunique()}")
    print(df[df.kind == "mc"].conversations.iloc[0])


if __name__ == "__main__":
    main()
