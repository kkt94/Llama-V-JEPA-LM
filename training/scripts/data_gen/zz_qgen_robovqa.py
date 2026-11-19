"""(10-26) RoboVQA (Google, Apache-2.0; clips + labels as redistributed in nvidia/Cosmos-Reason1-SFT-Dataset, CC-BY-4.0). ONLY the original
RoboVQA labels in metadata.task_metadata are used (question_processed / answer_processed, all split "train"); the model-written captions
("understanding") and <think> traces are dropped. The 101 benchmark clips (Cosmos-Reason1-Benchmark) do not occur in this train set.
 rvqa_success  task:success:discrete:{True,False}               yes / no 50/50, up to 30k per answer
 rvqa_afford   task:affordance:discriminative:discrete:{..}      yes / no 50/50, up to 30k per answer
 rvqa_plan     task:planning:freeform (+ immediate_planning_with_context20)   open answer, up to 30k
 rvqa_future   task:future_prediction:freeform                   open answer, up to 20k
 rvqa_past     task:past_description:freeform                    open answer, up to 20k
yes / no: half lettered (A/B), half open "Yes." / "No.". -> manifests/stage1/opendl3/rvqa_*.parquet (prep root data/opendl/robovqa/clips)
usage: zz_qgen_robovqa.py"""
import sys, os, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
R = "/data/llama_vjepa_lm"; D = f"{R}/data/opendl/robovqa"; O3 = f"{R}/manifests/stage1/opendl3"; g = Gen(113)
X = json.load(open(f"{D}/meta/robovqa_understanding.json")); have = set(os.listdir(f"{D}/clips"))
bench = {str(b.get("video", "")).split("/")[-1] for b in json.load(open(f"{D}/meta/benchmark_qa_pairs.json"))}
E = collections.defaultdict(list)
for x in X:
    for t in (x.get("metadata", {}).get("task_metadata") or []):
        v = str(t["video"]).split("/")[-1]
        if v in have and v not in bench and t.get("split") == "train" and t.get("question_processed") and t.get("answer_processed") is not None: E[t["task"]].append((v, t))
print({k: len(v) for k, v in E.items()}, flush=True)
out = collections.defaultdict(list)
def add(kind, v, conv, key): out[kind].append(dict(video=v, prep_exists=True, root2=None, tq=None, aug="", t0=0.0, t1=0.0, conversations=conv, _key=key))
def yn(kind, v, q, yes):
    if g.rng.random() < 0.5: add(kind, v, g.row(v, 0, 0, q, ["yes", "no"] if yes else ["no", "yes"], 0)["conversations"], "yes" if yes else "no")
    else: add(kind, v, json.dumps([{"role": "user", "content": f"<|jepa_evidence|>\n{q}\nAnswer yes or no."}, {"role": "assistant", "content": "Yes." if yes else "No."}]), "yes" if yes else "no")
def op(kind, v, q, a): a = str(a).strip(); add(kind, v, json.dumps([{"role": "user", "content": f"<|jepa_evidence|>\n{q}\nAnswer in a few words."}, {"role": "assistant", "content": a[:1].upper() + a[1:] + ("" if a.endswith(".") else ".")}]), "open")
def take(task, n): L = E.get(task, []); return [L[int(i)] for i in g.rng.permutation(len(L))[:n]]
for kind, pos, neg in (("rvqa_success", "task:success:discrete:True", "task:success:discrete:False"), ("rvqa_afford", "task:affordance:discriminative:discrete:True", "task:affordance:discriminative:discrete:False")):
    n = min(60000, len(E.get(pos, [])), len(E.get(neg, [])))   # (v2) 60k cap: the per-question balance below drops about a third
    for v, t in take(pos, n): yn(kind, v, t["question_processed"], str(t["answer_processed"]).strip().lower().startswith("y"))
    for v, t in take(neg, n): yn(kind, v, t["question_processed"], str(t["answer_processed"]).strip().lower().startswith("y"))
for kind, tasks, n in (("rvqa_plan", ["task:planning:freeform", "task:immediate_planning_with_context20:freeform"], 15000), ("rvqa_future", ["task:future_prediction:freeform"], 20000), ("rvqa_past", ["task:past_description:freeform"], 20000)):
    for task in tasks:
        for v, t in take(task, n): op(kind, v, t["question_processed"], t["answer_processed"])
for k, rows in sorted(out.items()):
    d = pd.DataFrame(rows)
    if k in ("rvqa_success", "rvqa_afford"):
        print(k, "answers before", d._key.value_counts().to_dict())
        # (v2) yes / no balanced PER QUESTION TEXT (the instruction): with a global balance only, the instruction alone answered rvqa_afford
        # 59.9 % of held-out rows (chance 50)
        d["_q"] = [json.loads(c)[0]["content"].split("\n")[1].strip().lower() for c in d.conversations]; keep = []
        for _, gq in d.groupby("_q"):
            if gq._key.nunique() == 2: m = gq._key.value_counts().min(); keep += [gg.sample(m, random_state=0) for _, gg in gq.groupby("_key")]
        d = pd.concat(keep).sort_index().drop(columns="_q"); print(k, "after per-question balance", d._key.value_counts().to_dict())
    d = d.drop(columns="_key").reset_index(drop=True); report(k, d.to_dict("records")); d.to_parquet(f"{O3}/{k}.parquet")
