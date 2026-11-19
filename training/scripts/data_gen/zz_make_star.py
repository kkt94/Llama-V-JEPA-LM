"""(10-23, user approved: STAR train split, but never the evaluation videos) STAR train (45.7k situated-reasoning questions on Charades
clips: interaction, sequence, prediction, feasibility) -> multiple-choice training manifest. Every Charades video that appears in ANY
evaluation set - by name or by frame content (logs/zz_charades_dedupe.py -> logs/zz_charades_eval_ids_content.json) - is dropped with all
its questions. Options are shuffled per question (seeded) so the answer letter is balanced. Clip = the question's [start, end] window.
-> manifests/stage1/star_train.parquet"""
import ast, hashlib, json, os, random, collections
import pandas as pd
R = "/data/llama_vjepa_lm"; V = f"{R}/bench/CharadesSTA/videos"
ex = set(json.load(open(f"{R}/logs/zz_charades_eval_ids_content.json"))["ids"])
paths = {os.path.basename(p)[:-4]: p for p in __import__("glob").glob(f"{V}/**/*.mp4", recursive=True)}
d = json.load(open(f"{R}/data/STAR/STAR_train.json"))
rows, drop = [], collections.Counter()
for x in d:
    v = x["video_id"]
    if v in ex:
        drop["evaluation video"] += 1; continue
    if v not in paths:
        drop["no video"] += 1; continue
    ch = x["choices"]; ch = ast.literal_eval(ch) if isinstance(ch, str) else ch
    opts = [str(c["choice"]).strip() for c in ch]; ans = str(x["answer"]).strip()
    if ans not in opts or len(opts) < 2:
        drop["answer not in choices"] += 1; continue
    rng = random.Random(int(hashlib.md5(x["question_id"].encode()).hexdigest()[:8], 16)); rng.shuffle(opts)
    a = opts.index(ans); L = "ABCDEFG"
    q = "<|jepa_evidence|>\n" + str(x["question"]).strip() + "\n" + "\n".join(f"{L[i]}. {o}" for i, o in enumerate(opts)) + \
        "\nAnswer with the option's letter from the given choices directly."
    rows.append({"video": paths[v], "prep_exists": True, "t0": float(x["start"]), "t1": float(x["end"]), "kind": "mc",
                 "op": "star_" + x["question_id"].split("_")[0].lower(), "conversations": json.dumps([{"role": "user", "content": q}, {"role": "assistant", "content": f"{L[a]}."}])})
m = pd.DataFrame(rows); m.to_parquet(f"{R}/manifests/stage1/star_train.parquet")
print("STAR train", len(d), "-> kept", len(m), "videos", m.video.nunique(), "| dropped", dict(drop), "| excluded ids", len(ex))
print(m.op.value_counts().to_dict()); print(collections.Counter(json.loads(c)[1]["content"] for c in m.conversations))
print(json.loads(m.iloc[0].conversations))
