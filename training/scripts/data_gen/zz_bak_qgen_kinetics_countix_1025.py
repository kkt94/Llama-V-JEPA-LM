"""(10-24, user: build questions from all the open data) Kinetics-700-2020 (human action classes, 10 s clips, CVDF mirror) and Countix
(human repetition counts on Kinetics clips; Molmo2-Countix csv, Ai2's mapping to Kinetics-700 files). Answers from the human labels.
  k700_act   : which action; distractors share a content word with the true class when possible (e.g. "playing guitar" / "playing ukulele"),
               else random; one question per downloaded train clip
  countix    : how many repetitions inside the counted span; counts 2..10 only (32 frames over the span must show every cycle: >= 3 frames
               per cycle); options the count and three nearby wrong counts; aug "" / rev / flip (a count does not change)
-> manifests/stage1/opendl/{k700_act,countix}.parquet (prep root data/opendl/kinetics700; clips used as downloaded)"""
import sys, os, re, json, glob, collections
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
import pandas as pd
R = "/data/llama_vjepa_lm"; K = f"{R}/data/opendl/kinetics700"; C = f"{R}/data/opendl/countix"; OUT = f"{R}/manifests/stage1/opendl"; os.makedirs(OUT, exist_ok=True)
g = Gen(47)
ann = pd.read_csv(f"{K}/annotations/train.csv")
classes = sorted(ann.label.unique()); STOP = {"a", "an", "the", "of", "on", "in", "with", "to", "and", "or", "at", "for", "up", "out", "off", "something", "someone"}
words = {c: {w for w in re.findall(r"[a-z]+", c.lower()) if w not in STOP and len(w) > 2} for c in classes}
by_word = collections.defaultdict(set)
for c, ws in words.items():
    for w in ws: by_word[w].add(c)
nice = lambda c: c[0].upper() + c[1:]
Q = ["What is the person doing?", "Which action is shown in the video?", "What activity is performed in the video?", "What are the people in the video doing?"]
files = {}
for p in glob.glob(f"{K}/train/*/*.mp4"):
    files[os.path.basename(p)[:-4]] = os.path.relpath(p, K)
rows = []
for r in ann.itertuples():
    key = f"{r.youtube_id}_{int(r.time_start):06d}_{int(r.time_end):06d}"
    if key not in files: continue
    hard = sorted({c for w in words[r.label] for c in by_word[w]} - {r.label})
    dis = list(g.rng.permutation(hard))[:3] if hard else []
    dis += [c for c in g.rng.permutation(classes) if c != r.label and c not in dis][:3 - len(dis)]
    rows.append(g.row(files[key], 0.0, 10.0, g.pick(Q), [nice(r.label)] + [nice(c) for c in dis], 0, "", True))
print("Kinetics train annotations", len(ann), "| clips on disk", len(files), flush=True)
report("k700_act", rows); pd.DataFrame(rows).to_parquet(f"{OUT}/k700_act.parquet")

cq = json.load(open(f"{C}/class_to_question.json")); cx = pd.read_csv(f"{C}/countix_train_mapped.csv")
rows = []; miss = 0
for i, r in enumerate(cx.itertuples()):
    c = int(r.count)
    if not 2 <= c <= 10 or r.repetition_end - r.repetition_start < 1.0: continue
    ex = os.path.exists(f"{K}/{r.video_path}"); miss += not ex
    wrong = [x for x in (c - 2, c - 1, c + 1, c + 2, c + 3, 2 * c) if x >= 1 and x != c]
    opts = [str(c)] + [str(x) for x in list(g.rng.permutation(wrong))[:3]]
    q = g.pick(cq.get(r._2, [f"How many times does the person do {r._2}?"]))
    rows.append(g.row(r.video_path, float(r.repetition_start) - float(r.kinetics_start), float(r.repetition_end) - float(r.kinetics_start), q, opts, 0,
                      ["", "rev", "flip"][i % 3], ex))
print("Countix rows", len(cx), "| counts 2..10 kept", len(rows), "| clip missing (Kinetics not downloaded yet?)", miss, flush=True)
report("countix", rows); pd.DataFrame(rows).to_parquet(f"{OUT}/countix.parquet")
