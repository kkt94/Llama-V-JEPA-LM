"""(10-24, user: build questions from all the open data) EPIC-KITCHENS-100 (first-person kitchen video, 76,885 human-labelled action spans
with a verb and a noun class; train + validation annotations). Videos are used from prep/epic_384_15fps/Pxx/Pxx_yy.mp4 (transcoded by
logs/zz_prep_epic.sh as the download lands). Spans shorter than 2 s are widened symmetrically to 2 s. Four kinds, answers from the labels:
  act   : "verb noun" of the span; distractors: two with the same noun and another verb, one with the same verb and another noun, all seen
          elsewhere in the annotations and none labelled in an overlapping span
  noun  : which object is handled; distractors are nouns of the same video not labelled in an overlapping span (same kitchen)
  order : two consecutive non-overlapping spans (<= 15 s together), which came first; aug "" / "rev" (backwards the order flips)
  rev   : spans whose verb has a time-reversal counterpart (open/close, take/put, turn-on/turn-off, insert/remove, roll/unroll); aug "" / "rev"
-> manifests/stage1/opendl/epic_{act,noun,order,rev}.parquet (prep root prep/epic_384_15fps)"""
import sys, csv, os, collections
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
import pandas as pd
R = "/data/llama_vjepa_lm"; A = f"{R}/data/opendl/epic_kitchens_100/annotations"; PREP = f"{R}/prep/epic_384_15fps"; OUT = f"{R}/manifests/stage1/opendl"
os.makedirs(OUT, exist_ok=True)
ts = lambda s: (lambda h, m, x: 3600 * int(h) + 60 * int(m) + float(x))(*s.split(":"))
VK = {r["id"]: r["key"] for r in csv.DictReader(open(f"{A}/EPIC_100_verb_classes.csv"))}
NK = {r["id"]: r["key"] for r in csv.DictReader(open(f"{A}/EPIC_100_noun_classes.csv"))}
nice_n = lambda k: " ".join(reversed(k.split(":"))) if ":" in k else k
nice_v = lambda k: k.replace("-", " ")
phrase = lambda v, n: f"{nice_v(VK[v])} {nice_n(NK[n])}"
REVV = {"open": "close", "close": "open", "take": "put", "put": "take", "turn-on": "turn-off", "turn-off": "turn-on", "insert": "remove", "remove": "insert",
        "roll": "unroll", "unroll": "roll"}
KV = {k: i for i, k in VK.items()}
S = []
for f in ("EPIC_100_train.csv", "EPIC_100_validation.csv"):
    for r in csv.DictReader(open(f"{A}/{f}")):
        S.append(dict(vid=r["video_id"], p=r["participant_id"], s=ts(r["start_timestamp"]), e=ts(r["stop_timestamp"]), v=r["verb_class"], n=r["noun_class"]))
pairs_seen = collections.Counter((x["v"], x["n"]) for x in S)
by_noun = collections.defaultdict(set); by_verb = collections.defaultdict(set)
for (v, n), c in pairs_seen.items():
    if c >= 3: by_noun[n].add(v); by_verb[v].add(n)
byvid = collections.defaultdict(list)
for x in S: byvid[x["vid"]].append(x)
g = Gen(43); out = {k: [] for k in ("act", "noun", "order", "rev")}
Q_ACT = ["What does the camera wearer do?", "What am I doing in this video?", "Which action is performed in this first-person video?", "What is the person doing with their hands?"]
Q_NOUN = ["Which object does the camera wearer handle?", "What object am I interacting with?", "Which object is used in this clip?"]
Q_ORD = ["In what order do these actions happen?", "Which sequence matches the video?", "What happens first?"]
miss = 0
for vid, X in byvid.items():
    X.sort(key=lambda x: x["s"]); path = f"{X[0]['p']}/{vid}.mp4"; ex = os.path.exists(f"{PREP}/{path}"); miss += not ex
    nouns_here = sorted({x["n"] for x in X})
    def win(x):
        c, d = (x["s"] + x["e"]) / 2, max(x["e"] - x["s"], 2.0); return max(0.0, c - d / 2), c + d / 2
    def present(t0, t1): return {(y["v"], y["n"]) for y in X if y["s"] < t1 and y["e"] > t0}
    for x in X:
        t0, t1 = win(x); pr = present(t0, t1); prn = {n for _, n in pr}
        if x["e"] - x["s"] > 15: continue
        dv = [v for v in by_noun[x["n"]] if v != x["v"] and (v, x["n"]) not in pr]; dn = [n for n in by_verb[x["v"]] if n != x["n"] and (x["v"], n) not in pr]
        if len(dv) >= 2 and dn:
            dis = [(v, x["n"]) for v in list(g.rng.permutation(dv))[:2]] + [(x["v"], list(g.rng.permutation(dn))[0])]
            out["act"].append(g.row(path, t0, t1, g.pick(Q_ACT), [phrase(x["v"], x["n"])] + [phrase(*d) for d in dis], 0, "", ex))
        cand = [n for n in nouns_here if n not in prn]
        if len(cand) >= 3:
            out["noun"].append(g.row(path, t0, t1, g.pick(Q_NOUN), [nice_n(NK[x["n"]])] + [nice_n(NK[n]) for n in list(g.rng.permutation(cand))[:3]], 0, "", ex))
        vk = VK[x["v"]]
        if vk in REVV and REVV[vk] in KV:
            aug = g.pick(["", "rev"]); a = vk if aug == "" else REVV[vk]; b = REVV[a]
            others = [v for v in by_noun[x["n"]] if VK[v] not in (a, b)]
            others = list(g.rng.permutation(others))[:2]
            if len(others) == 2:
                out["rev"].append(g.row(path, t0, t1, g.pick(Q_ACT), [phrase(KV[a], x["n"]), phrase(KV[b], x["n"])] + [phrase(v, x["n"]) for v in others], 0, aug, ex))
    for a, b in zip(X[:-1], X[1:]):
        if b["s"] >= a["e"] + 0.2 and b["e"] - a["s"] <= 15 and (a["v"], a["n"]) != (b["v"], b["n"]) and g.rng.random() < 0.5:
            aug = g.pick(["", "rev"]); f, s_ = (a, b) if aug == "" else (b, a)
            opts = [f"First {phrase(f['v'], f['n'])}, then {phrase(s_['v'], s_['n'])}", f"First {phrase(s_['v'], s_['n'])}, then {phrase(f['v'], f['n'])}"]
            out["order"].append(g.row(path, a["s"], b["e"], g.pick(Q_ORD), opts, 0, aug, ex))
print("videos", len(byvid), "| transcoded file missing", miss, flush=True)
for k, rr in out.items():
    report("epic_" + k, rr); pd.DataFrame(rr).to_parquet(f"{OUT}/epic_{k}.parquet")
