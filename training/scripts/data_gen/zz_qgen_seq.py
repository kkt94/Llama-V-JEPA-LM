"""(10-25, first-person data without Ego4D) multi-step ORDER questions in the format of TVBench egocentric_sequence ("What is the sequence of
actions shown in the video?"), answers from human-labelled action spans: 3-4 consecutive labelled actions that do not overlap (each starts
>= 0.2 s after the previous ends, all different), the clip cut from the first start to the last end (<= 30 s). Options = the true order + 3
other permutations of the SAME actions (equal length: no length cue). (v2) no 'rev' augmentation: played backwards, open looks like close -
the action names would no longer match the screen.
Sources: EPIC-KITCHENS-100 (first person), Charades-Ego (first- and third-person pairs; label = the class names).
-> manifests/stage1/opendl3/{epic_seq,chego_seq}.parquet   usage: zz_qgen_seq.py"""
import sys, csv, io, os, zipfile, itertools, collections
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
import pandas as pd
R = "/data/llama_vjepa_lm"; OUT = f"{R}/manifests/stage1/opendl3"; os.makedirs(OUT, exist_ok=True); g = Gen(71)
Q = ["What is the sequence of actions shown in the video?", "In what order are these actions performed?", "Which sequence of actions matches the video?",
     "What do I do in this video, in order?"]
def build(spans, path, ex, k_list=(3, 4), max_len=30.0, stride=2):
    """spans: list of (start, end, phrase) sorted by start -> rows"""
    rows = []; i = 0
    while i < len(spans):
        k = int(g.rng.choice(k_list)); seq = [spans[i]]
        for sp in spans[i + 1:]:
            if len(seq) == k: break
            if sp[0] >= seq[-1][1] + 0.2 and sp[2] not in {x[2] for x in seq}: seq.append(sp)
        if len(seq) == k and seq[-1][1] - seq[0][0] <= max_len:
            aug = ""; true = [x[2] for x in seq]
            perms = [list(p) for p in itertools.permutations(true) if list(p) != true]; others = [perms[j] for j in g.rng.permutation(len(perms))[:3]]
            opts = [", then ".join(true)] + [", then ".join(p) for p in others]
            rows.append(g.row(path, seq[0][0], seq[-1][1], g.pick(Q), opts, 0, aug, ex)); i += stride
        else:
            i += 1
    return rows
# ---- EPIC-KITCHENS-100
A = f"{R}/data/opendl/epic_kitchens_100/annotations"; PREP = f"{R}/prep/epic_384_15fps"
ts = lambda s: (lambda h, m, x: 3600 * int(h) + 60 * int(m) + float(x))(*s.split(":"))
VK = {r["id"]: r["key"] for r in csv.DictReader(open(f"{A}/EPIC_100_verb_classes.csv"))}; NK = {r["id"]: r["key"] for r in csv.DictReader(open(f"{A}/EPIC_100_noun_classes.csv"))}
nice_n = lambda k: " ".join(reversed(k.split(":"))) if ":" in k else k
phrase = lambda v, n: f"{VK[v].replace('-', ' ')} {nice_n(NK[n])}"
byvid = collections.defaultdict(list)
for f in ("EPIC_100_train.csv", "EPIC_100_validation.csv"):
    for r in csv.DictReader(open(f"{A}/{f}")):
        byvid[(r["participant_id"], r["video_id"])].append((ts(r["start_timestamp"]), ts(r["stop_timestamp"]), phrase(r["verb_class"], r["noun_class"])))
ep = []
for (p, vid), X in byvid.items():
    X.sort(); path = f"{p}/{vid}.mp4"; ep += build(X, path, os.path.exists(f"{PREP}/{path}"))
report("epic_seq", ep); pd.DataFrame(ep).to_parquet(f"{OUT}/epic_seq.parquet")
# ---- Charades-Ego
D = f"{R}/data/opendl/charades_ego"; VID = f"{D}/CharadesEgo_v1_480"
z = zipfile.ZipFile(f"{D}/CharadesEgo.zip"); rd = lambda n: io.TextIOWrapper(z.open("CharadesEgo/" + n), encoding="utf-8")
NAME = {l.split()[0]: l.strip()[5:] for l in rd("Charades_v1_classes.txt") if l.strip()}
FACE = {"c029", "c072", "c078", "c085", "c094", "c131", "c133", "c146", "c149", "c152", "c153"}   # face / bodily-state classes, as zz_qgen_charades_ego.py
lo = lambda c: NAME[c][0].lower() + NAME[c][1:]
ch = []
for r in csv.DictReader(rd("CharadesEgo_v1_train.csv")):
    X = []
    for a in (r["actions"] or "").split(";"):
        q_ = a.split()
        if len(q_) == 3 and q_[0] in NAME and q_[0] not in FACE and float(q_[2]) - float(q_[1]) >= 1.0: X.append((float(q_[1]), float(q_[2]), lo(q_[0])))
    X.sort(); v = f"{r['id']}.mp4"; ch += build(X, v, os.path.exists(f"{VID}/{v}"), stride=1)
report("chego_seq", ch); pd.DataFrame(ch).to_parquet(f"{OUT}/chego_seq.parquet")
