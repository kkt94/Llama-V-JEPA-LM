"""(10-24, user: build questions from all the open data) Charades-Ego (AI2; paired first-person / third-person recordings of the same scripts,
157 human-labelled action classes with time spans; train split only). Three kinds, answers from the human labels:
  act   : what the person (first-person: "the camera wearer") does in one labelled span; distractors share the verb or the object of the true
          class and are NOT labelled anywhere in that span (multi-label videos)
  order : two labelled actions that do not overlap, which came first; aug "" / "rev" (played backwards the order flips)
  rev   : a span of an action with a time-reversal counterpart (open <-> close, put <-> take, turn on <-> off, sit down <-> stand up, shoes on <->
          off); aug "" / "rev": played backwards, the answer is the counterpart
-> manifests/stage1/opendl/chego_{act,order,rev}.parquet (prep root data/opendl/charades_ego/CharadesEgo_v1_480)"""
import sys, io, csv, zipfile, os, collections, re
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
import pandas as pd
R = "/data/llama_vjepa_lm"; D = f"{R}/data/opendl/charades_ego"; VID = f"{D}/CharadesEgo_v1_480"; OUT = f"{R}/manifests/stage1/opendl"; os.makedirs(OUT, exist_ok=True)
z = zipfile.ZipFile(f"{D}/CharadesEgo.zip"); rd = lambda n: io.TextIOWrapper(z.open("CharadesEgo/" + n), encoding="utf-8")
NAME = {l.split()[0]: l.strip()[5:] for l in rd("Charades_v1_classes.txt") if l.strip()}
# (v2, after looking at items) facial / bodily-state classes cannot be seen from the camera wearer and barely at 480p third-person: never an
# answer, never a distractor (smiling, laughing, sneezing, awakening, snuggling)
FACE = {"c029", "c072", "c078", "c085", "c094", "c131", "c133", "c146", "c149", "c152", "c153"}
NAME = {c: n for c, n in NAME.items() if c not in FACE}
MAP = {l.split()[0]: (l.split()[1], l.split()[2]) for l in rd("Charades_v1_mapping.txt") if l.strip()}       # class -> (object, verb)
REV = {}
for a, b in [("c006", "c008"), ("c025", "c027"), ("c039", "c041"), ("c046", "c048"), ("c089", "c090"), ("c104", "c105"), ("c112", "c113"), ("c142", "c143"),
             ("c055", "c057"), ("c151", "c154"), ("c001", "c002"), ("c017", "c018"), ("c022", "c023"), ("c028", "c030"), ("c034", "c035"), ("c042", "c043"),
             ("c049", "c050"), ("c054", "c056"), ("c062", "c063"), ("c068", "c069"), ("c071", "c073"), ("c077", "c079"), ("c099", "c100"), ("c109", "c110"),
             ("c116", "c117"), ("c119", "c120")]:
    REV[a], REV[b] = b, a
rows = list(csv.DictReader(rd("CharadesEgo_v1_train.csv")))
g = Gen(41); act, order, rev = [], [], []
nm = lambda c: NAME[c][0].upper() + NAME[c][1:]
Q_ACT = {"ego": ["What does the camera wearer do in this first-person video?", "What am I doing in this video?", "Which action is performed in this first-person video?",
                 "What is the person wearing the camera doing?"],
         "exo": ["What is the person doing?", "Which action does the person perform?", "What happens in the video?", "What is the person in the video doing?"]}
Q_ORD = ["In what order do these actions happen?", "Which sequence matches the video?", "What happens first?", "Which order of events is shown?"]
missing = 0
for r in rows:
    ego = "ego" if r["egocentric"] == "Yes" else "exo"; v = f"{r['id']}.mp4"; ex = os.path.exists(f"{VID}/{v}"); missing += not ex
    A = []; F = []
    for a in (r["actions"] or "").split(";"):
        p = a.split()
        if len(p) == 3 and p[0] in FACE: F.append((float(p[1]), float(p[2])))
        if len(p) == 3 and p[0] in NAME: A.append((p[0], float(p[1]), float(p[2])))
    A = [a for a in A if a[2] - a[1] >= 1.0]
    present = lambda t0, t1: {c for c, s, e in A if s < t1 and e > t0}
    for c, s, e in A:                                                                   # act
        t0, t1 = s, min(e, s + 12.0); pres = present(t0, t1)
        o, vb = MAP[c]; hard = [x for x in NAME if x not in pres and x != c and (MAP[x][0] == o or MAP[x][1] == vb)]
        rest = [x for x in NAME if x not in pres and x != c and x not in hard]
        dis = list(g.rng.permutation(hard))[:3]; dis += list(g.rng.permutation(rest))[:3 - len(dis)]
        act.append(g.row(v, t0, t1, g.pick(Q_ACT[ego]), [nm(c)] + [nm(x) for x in dis], 0, "", ex))
        if c in REV and e - s <= 10.0:                                                  # rev
            aug = g.pick(["", "rev"]); ans = c if aug == "" else REV[c]
            others = [x for x in NAME if x not in pres and x not in (c, REV[c]) and (MAP[x][0] == o)]
            others = (list(g.rng.permutation(others)) + list(g.rng.permutation([x for x in NAME if x not in pres and x not in (c, REV[c])])))[:2]
            rev.append(g.row(v, s, e, g.pick(Q_ACT[ego]), [nm(ans), nm(REV[ans])] + [nm(x) for x in others], 0, aug, ex))
    pairs = [(a, b) for a in A for b in A if a[0] != b[0] and a[2] + 0.3 <= b[1] and b[2] - a[1] <= 20.0 and b[0] not in present(a[1], a[2]) and a[0] not in present(b[1], b[2])]
    for i in list(g.rng.permutation(len(pairs)))[:3]:
        a, b = pairs[i]; aug = g.pick(["", "rev"]); f, s_ = (a, b) if aug == "" else (b, a)
        lo = lambda c: NAME[c][0].lower() + NAME[c][1:]
        opts = [f"First {lo(f[0])}, then {lo(s_[0])}", f"First {lo(s_[0])}, then {lo(f[0])}"]
        order.append(g.row(v, a[1], b[2], g.pick(Q_ORD), opts, 0, aug, ex))
print("videos", len(rows), "| video file missing (not extracted yet?)", missing, flush=True)
for name, rr in (("chego_act", act), ("chego_order", order), ("chego_rev", rev)):
    report(name, rr); pd.DataFrame(rr).to_parquet(f"{OUT}/{name}.parquet")
