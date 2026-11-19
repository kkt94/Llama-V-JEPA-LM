"""Stage-1 / general data from CLEVRER TRAIN (video_00000 - video_09999; MVP evaluates on CLEVRER validation, 10000+).

Rows, all in the evaluator's prompt format (letters + SUFFIX) except `open`:
  desc_mc     descriptive question, 2 options = the answer + one distractor of the same type (count +-1, another
              colour / shape / material, yes vs no) - the same 2-option form MVP uses for CLEVRER
  desc_open   30% of descriptive questions also as a short open answer
  choice_mc   explanatory / predictive / counterfactual: "Which ..." with one correct and one wrong choice (up to 2 per
              question, CLEVRER choices are multi-label so 4-way single-answer MC is not well defined)
  choice_yn   the same choices one by one: "<question> Is this one of them: <choice>?" -> yes / no (2 options)
Video: the original 480x320 25 fps mp4 (128 frames), centre crop at load time like the evaluator.
Output: manifests/stage1/clevrer.parquet
"""
import glob
import json
import os

import numpy as np
import pandas as pd

R = "/data/llama_vjepa_lm"
D = f"{R}/data/clevrer"
VIDEOS = f"{D}/video_train"
LETTERS = "AB"
SUFFIX = "Answer with the option's letter from the given choices directly."
VIS = "<|jepa_evidence|>"
VOCAB = {"query_color": ["gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow"],
         "query_shape": ["sphere", "cylinder", "cube"], "query_material": ["rubber", "metal"], "exist": ["yes", "no"]}


def mc(question, right, wrong, rng):
    opts = [right, wrong]
    if rng.random() < 0.5:
        opts = opts[::-1]
    q = f"{question}\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)) + f"\n{SUFFIX}"
    return q, LETTERS[opts.index(right)] + "."


def distractor(sub, ans, rng):
    if sub == "count":
        n = int(ans)
        return str(n + 1 if n == 0 or rng.random() < 0.5 else n - 1)
    return str(rng.choice([v for v in VOCAB[sub] if v != ans]))


def main():
    paths = {os.path.basename(p): os.path.relpath(p, VIDEOS) for p in glob.glob(f"{VIDEOS}/**/*.mp4", recursive=True)}
    assert len(paths) > 9000, f"only {len(paths)} CLEVRER train videos extracted"
    mvp = {line.split("\t")[1] for line in open(f"{R}/manifests/allowlist/benchmark_ids_mvp.txt") if line.startswith("clevrer\t")}
    assert mvp, "no MVP clevrer ids read: the exclusion check would be vacuous"
    rng = np.random.default_rng(0)
    rows, excluded, missing = [], 0, 0

    def add(video, kind, sub, q, a):
        rows.append({"video": video, "prep_exists": True, "kind": kind, "subtype": sub,
                     "conversations": json.dumps([{"role": "user", "content": f"{VIS}\n{q}"},
                                                  {"role": "assistant", "content": a}])})

    for v in json.load(open(f"{D}/train.json")):
        fn = v["video_filename"]
        if fn[:-4] in mvp:
            excluded += 1
            continue
        if fn not in paths:
            missing += 1
            continue
        video = paths[fn]
        for x in v["questions"]:
            t = x["question_type"]
            if t == "descriptive":
                sub, ans = x["question_subtype"], x["answer"]
                add(video, "desc_mc", sub, *mc(x["question"], ans, distractor(sub, ans, rng), rng))
                if rng.random() < 0.3:
                    add(video, "desc_open", sub, x["question"], ans[0].upper() + ans[1:] + ".")
                continue
            right = [c["choice"] for c in x["choices"] if c["answer"] == "correct"]
            wrong = [c["choice"] for c in x["choices"] if c["answer"] == "wrong"]
            if right and wrong:
                for k in range(min(2, len(right) * len(wrong))):
                    add(video, "choice_mc", t, *mc(x["question"], right[k % len(right)], wrong[(k // len(right)) % len(wrong)], rng))
            for c in x["choices"]:
                q = f"{x['question']} Is this one of them: {c['choice'][0].lower() + c['choice'][1:]}?"
                add(video, "choice_yn", t, *mc(q, "yes" if c["answer"] == "correct" else "no",
                                               "no" if c["answer"] == "correct" else "yes", rng))
    df = pd.DataFrame(rows)
    assert not df.video.map(lambda p: os.path.basename(p)[:-4] in mvp).any()
    df.to_parquet(f"{R}/manifests/stage1/clevrer.parquet", index=False)
    print(f"videos {df.video.nunique()} | excluded (MVP ids) {excluded} | missing video {missing} | rows {len(df)}")
    print(df.groupby(["kind", "subtype"]).size().to_string())
    for k in ("desc_mc", "choice_mc", "choice_yn"):
        print(k, "| answer letters", df[df.kind == k].conversations.str.extract(r'"content": "([AB])\.')[0].value_counts().to_dict())
        print(json.loads(df[df.kind == k].conversations.iloc[0])[0]["content"])


if __name__ == "__main__":
    main()
