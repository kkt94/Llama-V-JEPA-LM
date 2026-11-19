"""(10-26, weak benchmarks whose formats nothing in our mix matches; answers from human-labelled spans or from the composition itself)
 vino_seg   Vinoground-video format: "The video has two segments separated by black frames. Which segment matches this caption: "first X, then
            Y"?" - one segment = the two labelled spans in their true order, the other = the same spans swapped (composed with a 2 s black clip
            between); which segment is first is random. Human labels: EPIC-KITCHENS-100, HoloAssist, Charades-Ego.
 loc_nat    MVBench / TVBench action-localization format: "During which part of the video does the action 'X' occur?" from Charades-Ego
            labelled spans in the whole video: end <= 40 % of the length -> beginning, start >= 60 % -> end, inside 25-75 % -> middle, span
            >= 85 % of the length -> throughout (others not asked); classes balanced
 loc_comp   same question on a COMPOSED video: the X span placed first / second / third among two spans of other actions of other videos
            (no "throughout": a lone span would be shorter - the clip length would answer); position known by construction
 rep_loop   repetition counting: one labelled span repeated N = 2..7 times back to back, total length drawn first (8-14 s) and the span
            chosen with length ~ total / N (so the clip length does not give N) ("Please count the number of repeated actions in
            the video." / "How many times does the person X?"); options count_options (gold rank uniform)
Composed rows use the loader's "seq" kind (tq = {"kind": "seq", "segs": [[abs path, t0, t1], ...]}, prep root "/").
-> manifests/stage1/opendl3/{vino_seg,loc_nat,loc_comp,rep_loop}.parquet   usage: zz_qgen_v14.py"""
import sys, os, csv, io, json, zipfile, collections
import numpy as np, pandas as pd
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report, count_options
R = "/data/llama_vjepa_lm"; O3 = f"{R}/manifests/stage1/opendl3"; g = Gen(107); BLACK = [f"{R}/prep/synthetic/black_2s.mp4", 0.0, 2.0]
ROOT = {"epic": f"{R}/prep/epic_384_15fps", "holo": f"{R}/prep/holoassist_384_15fps", "chego": f"{R}/data/opendl/charades_ego/CharadesEgo_v1_480"}
V = []; LEN = {}
A = f"{R}/data/opendl/epic_kitchens_100/annotations"; ts = lambda s: (lambda h, m, x: 3600 * int(h) + 60 * int(m) + float(x))(*s.split(":"))
VK = {r["id"]: r["key"] for r in csv.DictReader(open(f"{A}/EPIC_100_verb_classes.csv"))}; NK = {r["id"]: r["key"] for r in csv.DictReader(open(f"{A}/EPIC_100_noun_classes.csv"))}
nice_n = lambda k: " ".join(reversed(k.split(":"))) if ":" in k else k
by = collections.defaultdict(list)
for f in ("EPIC_100_train.csv", "EPIC_100_validation.csv"):
    for r in csv.DictReader(open(f"{A}/{f}")):
        by[(r["participant_id"], r["video_id"])].append((ts(r["start_timestamp"]), ts(r["stop_timestamp"]), f"{VK[r['verb_class']].replace('-', ' ')} the {nice_n(NK[r['noun_class']])}"))
for (p, vid), X in by.items():
    path = f"{ROOT['epic']}/{p}/{vid}.mp4"
    if os.path.exists(path): V.append(("epic", path, sorted(X)))
cl = lambda s: str(s).replace("_", " ").replace("-", " ").strip().lower()
for v in json.load(open(f"{R}/data/opendl/holoassist/data-annotation-trainval-v1_1.json")):
    path = f"{ROOT['holo']}/{v['video_name']}.mp4"
    if not os.path.exists(path): continue
    X = []
    for e in v["events"]:
        a = e.get("attributes") or {}
        if e["label"] == "Fine grained action" and "end" in e and cl(a.get("Verb", "none")) != "none" and cl(a.get("Noun", "none")) != "none":
            adj = cl(a.get("Adjective", "none")); X.append((float(e["start"]), float(e["end"]), f"{cl(a['Verb'])} the {'' if adj == 'none' else adj + ' '}{cl(a['Noun'])}"))
    V.append(("holo", path, sorted(X)))
z = zipfile.ZipFile(f"{R}/data/opendl/charades_ego/CharadesEgo.zip"); rd = lambda n: io.TextIOWrapper(z.open("CharadesEgo/" + n), encoding="utf-8")
NAME = {l.split()[0]: l.strip()[5:] for l in rd("Charades_v1_classes.txt") if l.strip()}
FACE = {"c029", "c072", "c078", "c085", "c094", "c131", "c133", "c146", "c149", "c152", "c153"}
for r in csv.DictReader(rd("CharadesEgo_v1_train.csv")):
    X = []
    for a in (r["actions"] or "").split(";"):
        q = a.split()
        if len(q) == 3 and q[0] in NAME and q[0] not in FACE and float(q[2]) - float(q[1]) >= 1.0: X.append((float(q[1]), float(q[2]), NAME[q[0]][0].lower() + NAME[q[0]][1:]))
    path = f"{ROOT['chego']}/{r['id']}.mp4"
    if X and os.path.exists(path): V.append(("chego", path, sorted(X))); LEN[path] = float(r["length"] or 0)
print("videos:", collections.Counter(s for s, _, _ in V), flush=True)
out = collections.defaultdict(list)
def row(kind, src, segs, q, opts, k, key):
    tq = json.dumps({"kind": "seq", "segs": [[str(p), float(a), float(b)] for p, a, b in segs]}) if segs else None
    rr = g.row(segs[0][0] if segs else src, 0.0, sum(b - a for _, a, b in segs) if segs else 0.0, q, opts, k)
    out[kind].append(dict(rr, tq=tq, _key=key))
# ---- vino_seg
VQ = 'The video has two segments separated by black frames. Which segment matches this caption: "{}"'
VO = ["The first segment (before the black frames)", "The second segment (after the black frames)"]
pool = []                                                  # all usable spans for fillers / loops: (src, path, s, e, phrase)
for src, path, X in V:
    X = [x for x in X if 1.0 <= x[1] - x[0] <= 8.0]; pool += [(src, path) + x for x in X]
    pairs = [(X[i], X[j]) for i in range(len(X)) for j in range(i + 1, min(i + 4, len(X))) if X[j][0] >= X[i][1] + 0.2 and X[i][2] != X[j][2]]
    for a, b in [pairs[int(k)] for k in g.rng.permutation(len(pairs))[:{"epic": 6, "holo": 6, "chego": 2}[src]]]:
        true_seg, swap_seg = [(path, a[0], a[1]), (path, b[0], b[1])], [(path, b[0], b[1]), (path, a[0], a[1])]
        first_true = g.rng.random() < 0.5
        segs = (true_seg + [tuple(BLACK)] + swap_seg) if first_true else (swap_seg + [tuple(BLACK)] + true_seg)
        cap = f"first {a[2]}, then {b[2]}"
        opts = VO if first_true else VO[::-1]               # gold always at index 0 here; g.row shuffles the options
        row("vino_seg", src, segs, VQ.format(cap), opts, 0, "first" if first_true else "second")
# ---- loc_nat (Charades-Ego, whole video)
LQ = "During which part of the video does the action '{}' occur?"
LO = {"beginning": "At the beginning of the video.", "middle": "In the middle of the video.", "end": "At the end of the video.", "throughout": "Throughout the entire video."}
for src, path, X in V:
    if src != "chego" or LEN.get(path, 0) < 8: continue
    D_ = LEN[path]
    for s, e, ph in X:
        if sum(x[2] == ph for x in X) > 1: continue
        pos = "throughout" if e - s >= 0.85 * D_ else "beginning" if e <= 0.4 * D_ else "end" if s >= 0.6 * D_ else "middle" if s >= 0.25 * D_ and e <= 0.75 * D_ else None
        if pos: row("loc_nat", src, [(path, 0.0, D_)], LQ.format(ph), [LO[pos]] + [LO[o] for o in LO if o != pos], 0, pos)
# ---- loc_comp (composed)
for _ in range(24000):
    t = pool[int(g.rng.integers(len(pool)))]; pos = g.pick(["beginning", "middle", "end"])   # (v2) no "throughout" here: one span alone is shorter than three - duration would answer
    tgt = (t[1], t[2], t[3])
    if pos == "throughout": segs = [tgt]
    else:
        fill = [pool[int(k)] for k in g.rng.choice(len(pool), 6, replace=False)]
        fill = [f for f in fill if f[1] != t[1] and f[4] != t[4]][:2]
        if len(fill) < 2: continue
        fl = [(f[1], f[2], f[3]) for f in fill]
        segs = [tgt] + fl if pos == "beginning" else [fl[0], tgt, fl[1]] if pos == "middle" else fl + [tgt]
    row("loc_comp", t[0], segs, LQ.format(t[4]), [LO[pos]] + [LO[o] for o in LO if o != pos], 0, pos)
# ---- rep_loop
L_ = np.array([x[3] - x[2] for x in pool])
for _ in range(12000):                                     # (v2) total length drawn first (8-14 s), unit length ~ total / n: the clip length
    n = int(g.rng.integers(2, 8)); T_ = g.rng.uniform(8.0, 14.0); ok = np.nonzero(np.abs(L_ - T_ / n) <= 0.15 * T_ / n)[0]   # says nothing about n
    if not len(ok): continue
    t = pool[int(ok[int(g.rng.integers(len(ok)))])]; segs = [(t[1], t[2], t[3])] * n
    q = g.pick(["Please count the number of repeated actions in the video.", f"How many times does the person {t[4]}?" if t[0] != "chego" else f"How many times is '{t[4]}' repeated?"])
    opts = [str(n)] + [str(int(x)) for x in g.rng.choice([m for m in range(2, 8) if m != n], 3, replace=False)]   # (v3) only values that occur as answers
    row("rep_loop", t[0], segs, q, opts, 0, str(n))
for k, rows in sorted(out.items()):
    d = pd.DataFrame(rows); cnt = d._key.value_counts().to_dict()
    if k in ("vino_seg", "loc_nat", "loc_comp"):
        m = d._key.value_counts().min(); d = pd.concat([gg.sample(m, random_state=0) for _, gg in d.groupby("_key")]).sort_index()
    print(k, "keys before", cnt, "after", d._key.value_counts().to_dict(), flush=True)
    d = d.drop(columns="_key").reset_index(drop=True); report(k, d.to_dict("records")); d.to_parquet(f"{O3}/{k}.parquet")
