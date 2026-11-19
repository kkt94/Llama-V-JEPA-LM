"""(10-27) capbin - TemporalBench binary caption choice ("Which caption best describes this video?" A / B) built from HUMAN action
sequences. Fixes a measured bias: on TemporalBench binary our model picks the LONGER caption 46 % (gold 33 %) and option B 64 % (gold 51 %).
Sources (human labels only, no model text):
  EPIC-KITCHENS-100  annotations/EPIC_100_{train,validation}.csv  verb_class / noun_class KEYS ("take the knife")   prep/epic_384_15fps
  HoloAssist         data-annotation-trainval-v1_1.json  "Fine grained action" Verb / Adjective / Noun (ordinal adjectives such as
                     "first" / "second" dropped: not visible)                                                       prep/holoassist_384_15fps
  Charades-Ego: no usable run (its labelled actions overlap - there is never a 3-step sequence with a defined order), so none.
Clip = a plain [t0, t1] window = the first step's start .. the last step's end of a run of N = 3-5 CONSECUTIVE labelled steps: each step
starts after the previous one (overlap <= 0.3 s, ends ordered too), gaps <= 4 s, all (verb, noun) different, and no other labelled action
of the video overlaps the window by more than 0.3 s (the caption lists everything the annotators marked in the clip); each labelled step
is used by at most one row.  Gold = the true steps in order: "The person first X, then Y, and then Z." (3 connective templates; the verb
put in the 3rd person - a fixed morphological transformation). The distractor differs minimally (same template):
  swap     two steps exchanged (adjacent 70 %)
  replace  one step replaced by another labelled action of the SAME video that does not occur in the window (same verb or same noun
           as the replaced step when the video has one)
  drop     one step left out (gold N = 4-5)
  insert   an action of the same video that does not occur in the window added at a random place (gold N = 3-4)
Shortcut control by construction: (i) the clip length is drawn FIRST (uniform 6-18 s) and a run with span within +-10 % is searched, so the
duration says nothing about N or the distractor type; (ii) drop uses N = 4, 5 and insert N = 3, 4 in equal numbers per step-count pair
({3,4}, {4,5}): the caption step counts never reveal which one is true; (iii) gold is the SHORTER caption in exactly 50 % of the rows whose
two captions differ in length (drop = insert counts, replace balanced shorter / longer); (iv) gold is option A in exactly 50 % of rows.
Then a held-out-by-video audit (option letter / option text prior, caption-text logistic regression, clip-length bins, structural
features, verb / phrase Markov plausibility); > chance + 5 must be fixed.
-> manifests/stage1/opendl3/v17/v17_capbin__{epic,holo}.parquet + roots.json   usage: zz_qgen_capbin.py > zz_qgen_capbin.out"""
import sys, os, csv, json, hashlib, collections, re
import numpy as np, pandas as pd
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
R = "/data/llama_vjepa_lm"; O = f"{R}/manifests/stage1/opendl3/v17"; os.makedirs(O, exist_ok=True); g = Gen(171); rng = g.rng
ROOT = {"epic": f"{R}/prep/epic_384_15fps", "holo": f"{R}/prep/holoassist_384_15fps"}
Q = "Which caption best describes this video?"


def third(v):
    """base verb phrase -> 3rd person singular ('turn on' -> 'turns on', 'dry' -> 'dries', 'wash' -> 'washes')"""
    w, rest = (v.split(" ", 1) + [""])[:2]
    w = {"do": "does", "go": "goes", "have": "has"}.get(w) or (w + "es" if re.search(r"(s|sh|ch|x|z)$", w) else w[:-1] + "ies" if re.search(r"[^aeiou]y$", w) else w + "s")
    return (w + " " + rest).strip()


# ---------------------------------------------------------------- labelled steps: (start, end, (verb, noun), phrase)
V = []
A = f"{R}/data/opendl/epic_kitchens_100/annotations"; ts = lambda s: (lambda h, m, x: 3600 * int(h) + 60 * int(m) + float(x))(*s.split(":"))
VK = {r["id"]: r["key"] for r in csv.DictReader(open(f"{A}/EPIC_100_verb_classes.csv"))}; NK = {r["id"]: r["key"] for r in csv.DictReader(open(f"{A}/EPIC_100_noun_classes.csv"))}
nice_n = lambda k: " ".join(reversed(k.split(":"))) if ":" in k else k
EV_SKIP = {"transition", "wait"}                           # not actions one can see as a step
by = collections.defaultdict(list)
for f in ("EPIC_100_train.csv", "EPIC_100_validation.csv"):
    for r in csv.DictReader(open(f"{A}/{f}")):
        v, n = VK[r["verb_class"]], nice_n(NK[r["noun_class"]]).replace("-", " ")
        if v in EV_SKIP: continue
        vb = "let go of" if v == "let-go" else v.replace("-", " ")
        by[(r["participant_id"], r["video_id"])].append((ts(r["start_timestamp"]), ts(r["stop_timestamp"]), (v, n), f"{third(vb)} the {n}"))
for (p, vid), X in sorted(by.items()):
    if os.path.exists(f"{ROOT['epic']}/{p}/{vid}.mp4"): V.append(("epic", f"{p}/{vid}.mp4", sorted(X)))
cl = lambda s: str(s).replace("_", " ").replace("-", " ").strip().lower()
ORD = {"first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "last", "next", "other", "another", "remaining", "none"}
HV = {"mix/stir": "stir", "stack/pile": "stack"}
for v in json.load(open(f"{R}/data/opendl/holoassist/data-annotation-trainval-v1_1.json")):
    if not os.path.exists(f"{ROOT['holo']}/{v['video_name']}.mp4"): continue
    X = []
    for e in v["events"]:
        a = e.get("attributes") or {}
        vb, n, adj = cl(a.get("Verb", "none")), cl(a.get("Noun", "none")), cl(a.get("Adjective", "none"))
        if e["label"] != "Fine grained action" or "end" not in e or vb == "none" or n == "none": continue
        vb = HV.get(vb, vb); adj = "" if adj in ORD or "/" in adj else adj + " "
        X.append((float(e["start"]), float(e["end"]), (vb, n), f"{third(vb)} the {adj}{n}"))
    if X: V.append(("holo", f"{v['video_name']}.mp4", sorted(X)))
print("videos:", collections.Counter(s for s, _, _ in V), flush=True)

# ---------------------------------------------------------------- runs of N consecutive steps (index per source / N, by span)
OV, GAP = 0.3, 4.0
SE = [(np.array([x[0] for x in X]), np.array([x[1] for x in X])) for _, _, X in V]


def qual(vi, i, N):
    """X[i:i+N] is a run: consecutive, ordered, gaps <= GAP, overlaps <= OV, distinct (verb, noun), nothing else labelled in the window"""
    X = V[vi][2]
    if i < 0 or i + N > len(X): return False
    run = X[i:i + N]
    if len({x[2] for x in run}) < N: return False
    if any(run[j + 1][0] < run[j][1] - OV or run[j + 1][0] - run[j][1] > GAP or run[j + 1][1] <= run[j][1] or run[j + 1][0] <= run[j][0] for j in range(N - 1)): return False
    S_, E_ = SE[vi]; s, e = run[0][0], run[-1][1]; ov = np.minimum(E_, e) - np.maximum(S_, s); ov[i:i + N] = 0
    return not (ov > OV).any()


RUNS = collections.defaultdict(list)                       # (src, N) -> [(span, video index, i)]
for vi, (src, path, X) in enumerate(V):
    for N in (3, 4, 5):
        for i in range(len(X) - N + 1):
            if qual(vi, i, N): RUNS[(src, N)].append((X[i + N - 1][1] - X[i][0], vi, i))
for k in RUNS: RUNS[k].sort()
print("runs:", {f"{k[0]}{k[1]}": len(v) for k, v in sorted(RUNS.items())}, flush=True)
SP = {k: np.array([r[0] for r in v]) for k, v in RUNS.items()}
used = collections.Counter()                               # (video index, step index): each labelled step in at most 2 rows
fold = lambda v: int(hashlib.md5(v.encode()).hexdigest(), 16) % 5   # = the audit's split (fold 0 held out)

# stream plausibility for CHOOSING distractors: phrase bigrams over every video's labelled action stream, cross-fitted (a row of fold f
# is scored by the streams of the other folds), so the distractor picked is the most plausible one a text-only model could know
BI = [collections.Counter() for _ in range(5)]; UNI = [collections.Counter() for _ in range(5)]
for _, path, X in V:
    f = fold(path); ph = [x[3] for x in X]; BI[f].update(zip(ph[:-1], ph[1:])); UNI[f].update(ph[:-1])
BIa, UNIa = sum(BI, collections.Counter()), sum(UNI, collections.Counter()); NV = len(UNIa) + 1
def splaus(seq, f):
    return float(np.mean([np.log((BIa[(a, b)] - BI[f][(a, b)] + 0.5) / (UNIa[a] - UNI[f][a] + 0.5 * NV)) for a, b in zip(seq[:-1], seq[1:])]))


def find_run(src, N, T):
    sp = SP[(src, N)]; lo, hi = np.searchsorted(sp, 0.9 * T), np.searchsorted(sp, 1.1 * T)
    if hi <= lo: return None
    for j in rng.integers(lo, hi, 40):
        _, vi, i = RUNS[(src, N)][int(j)]
        if all(used[(vi, i + k)] < 2 for k in range(N)): return vi, i
    return None


TPL = [lambda s: f"The person first {s[0]}, " + "".join(f"then {x}, " for x in s[1:-1]) + f"and then {s[-1]}.",
       lambda s: f"First the person {s[0]}, " + "".join(f"then {x}, " for x in s[1:-1]) + f"and finally {s[-1]}.",
       lambda s: f"The person {s[0]}, " + "".join(f"then {x}, " for x in s[1:-1]) + f"and then {s[-1]}."]
# drop / insert at the run's BOUNDARY only: the distractor of an insert is the true neighbouring labelled action added (the longer run
# must itself be a run, the neighbour outside the clip), that of a drop the run without its first / last step - both captions are then
# real label sequences of the video, so a text-only model cannot tell the true one (a middle drop / random insert makes the wrong
# caption implausible: text-only 78 % in the first audit)
NFOR = {"swap": (3, 4, 5), "replace": (3, 4, 5), "drop": (4, 5), "insert": (3, 4)}
cand = []
for it in range(600000):
    src = "epic" if rng.random() < 0.4 else "holo"; typ = g.pick(list(NFOR)); N = int(g.pick(NFOR[typ])); T = float(rng.uniform(6.0, 18.0))
    got = find_run(src, N, T)
    if got is None: continue
    vi, i = got; _, path, X = V[vi]; run = X[i:i + N]; s, e = run[0][0], run[-1][1]; f = fold(path)
    inwin = {x[2] for x in X if min(x[1], e) - max(x[0], s) > 0}   # every (verb, noun) labelled anywhere in the window
    steps = [x[3] for x in run]; tp = int(rng.integers(len(TPL))); pos = ""
    if typ == "swap":                                      # the swap whose result is most plausible (80 %) or a random one
        P = [(a, b) for a in range(N) for b in range(a + 1, N)]
        sw = lambda ab: [steps[ab[1]] if j == ab[0] else steps[ab[0]] if j == ab[1] else x for j, x in enumerate(steps)]
        ab = max(P, key=lambda ab: splaus(sw(ab), f) + 1e-6 * rng.random()) if rng.random() < 0.8 else P[int(rng.integers(len(P)))]
        d = sw(ab); info = f"{ab[0]}-{ab[1]}"
    elif typ == "replace":                                 # same verb or noun as the replaced step when possible, the most plausible result
        others = list({x[2]: x for x in X if x[2] not in inwin}.values())
        if not others: continue
        best = None
        for a in range(N):
            near = [x for x in others if x[2][0] == run[a][2][0] or x[2][1] == run[a][2][1]]
            for x in (near or others)[:60]:
                dd = steps[:a] + [x[3]] + steps[a + 1:]; sc = splaus(dd, f) + (1.0 if near else 0.0) + 1e-6 * rng.random()
                if best is None or sc > best[0]: best = (sc, dd, a)
        d, info = best[1], str(best[2])
    elif typ == "drop":
        pos = g.pick(["front", "back"]); d = steps[1:] if pos == "front" else steps[:-1]; info = pos
    else:
        pos = g.pick(["front", "back"]); j = i - 1 if pos == "front" else i + N
        if not qual(vi, min(i, j), N + 1) or X[j][2] in inwin or (pos == "front" and X[j][1] > s) or (pos == "back" and X[j][0] < e): continue
        d = [X[j][3]] + steps if pos == "front" else steps + [X[j][3]]; info = pos
    gold, dis = TPL[tp](steps), TPL[tp](d)
    if gold == dis: continue
    for k in range(N): used[(vi, i + k)] += 1
    cand.append(dict(src=src, vid=path, fold=f, t0=float(s), t1=float(e), typ=typ, pos=pos, N=N, Nd=len(d), gold=gold, dis=dis, gsteps=steps, dsteps=d,
                     info=info, cmp=int(np.sign(len(gold) - len(dis)))))
C = pd.DataFrame(cand)
C["pair"] = [f"{min(n, m)}{max(n, m)}" for n, m in zip(C.N, C.Nd)]
print("candidates:", len(C), C.groupby(["src", "typ"]).size().to_dict(), flush=True)

# ---------------------------------------------------------------- cross-fitted text-only predictors on the candidates
from sklearn.feature_extraction.text import TfidfVectorizer; from sklearn.linear_model import LogisticRegression
def markov(seqs):
    uni, bi = collections.Counter(), collections.Counter()
    for s in seqs:
        s = ["<s>"] + list(s) + ["</s>"]; uni.update(s[:-1]); bi.update(zip(s[:-1], s[1:]))
    nv = len(uni) + 1
    return lambda s: [np.log((bi[(a, b)] + 0.5) / (uni[a] + 0.5 * nv)) for a, b in zip(["<s>"] + list(s), list(s) + ["</s>"])]
vb_of = lambda ph: ph.split(" ")[0]
o = rng.random(len(C)) < 0.5                                # a random orientation for the LR (A = gold where True)
oA = np.where(o, C.gold, C.dis); oB = np.where(o, C.dis, C.gold)
SG = {k: np.zeros(len(C)) for k in ("pm", "ps", "vm", "lr")}
for f in range(5):
    tr_, te_ = (C.fold != f).to_numpy(), (C.fold == f).to_numpy()
    for nm, fx in (("p", lambda s: s), ("v", lambda s: [vb_of(x) for x in s])):
        M = markov([fx(s) for s in C.gsteps[tr_]])
        for agg in ("mean", "sum"):
            if nm == "v" and agg == "sum": continue
            sg = np.array([getattr(np, agg)(M(fx(a))) - getattr(np, agg)(M(fx(b))) for a, b in zip(C.gsteps[te_], C.dsteps[te_])])
            SG[nm + agg[0]][te_] = np.sign(np.round(sg, 9))
    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b").fit(list(oA[tr_]) + list(oB[tr_]))
    Xd = vec.transform(oA) - vec.transform(oB); pA = LogisticRegression(max_iter=1000).fit(Xd[tr_], o[tr_]).predict_proba(Xd[te_])[:, 1]
    SG["lr"][te_] = np.sign(np.where(o[te_], pA - 0.5, 0.5 - pA))
for k, v in SG.items(): C["s_" + k] = v.astype(int)
print("cross-fitted text predictors favour gold (before balancing):", {k: round(float((v > 0).mean() / max((v != 0).mean(), 1e-9)), 3) for k, v in SG.items()}, flush=True)

# ---------------------------------------------------------------- mirror balance
# within each (source, type group, step-count pair, boundary side): a row whose text-only signs (Markov phrase mean / sum, verb mean,
# tf-idf LR, gold longer / shorter) are k is kept only with as many rows whose signs are -k: every one of these predictors - and any
# odd combination of them - is then right exactly half the time, and gold is the shorter caption in exactly half the differing rows
DIMS = ["s_pm", "s_ps", "s_vm", "s_lr", "cmp"]
C["grp"] = [f"{s_}|{'di' if t in ('drop', 'insert') else t}|{p}|{q}" for s_, t, p, q in zip(C.src, C.typ, C.pair, C.pos.replace({"": "-"}))]
keep = []
for _, G in C.groupby("grp"):
    cell = {k: x for k, x in G.groupby(DIMS)}
    for k, x in cell.items():
        nk = tuple(-v for v in k)
        if k == nk: keep.append(x); continue
        if k > nk or nk not in cell: continue
        m = min(len(x), len(cell[nk])); keep += [x.sample(m, random_state=1), cell[nk].sample(m, random_state=2)]
D = pd.concat(keep).sample(frac=1.0, random_state=7).reset_index(drop=True)
D["goldA"] = False; D.loc[rng.permutation(len(D))[: len(D) // 2], "goldA"] = True
print("kept:", len(D), D.groupby(["src", "typ"]).size().to_dict(), flush=True)
print("pairs:", D.groupby(["typ", "pair"]).size().to_dict(), "| side:", D.groupby(["typ", "pos"]).size().to_dict(), flush=True)
dif = D[D.cmp != 0]
print(f"gold shorter among differing-length rows: {(dif.cmp < 0).mean():.4f} ({(dif.cmp < 0).sum()}/{len(dif)});  gold = A: {D.goldA.mean():.4f};  "
      f"clip length median {np.median(D.t1 - D.t0):.1f} s by N {D.groupby('N').apply(lambda x: round(float(np.median(x.t1 - x.t0)), 2)).to_dict()}", flush=True)

# ---------------------------------------------------------------- rows
rows = []
for r in D.itertuples():
    opts, k = ([r.gold, r.dis], 0) if r.goldA else ([r.dis, r.gold], 1)
    rows.append(dict(video=r.vid, prep_exists=True, root2=None, tq=None, aug="", t0=r.t0, t1=r.t1, conversations=g.render(Q, opts, k)))
D["conversations"] = [x["conversations"] for x in rows]

# ---------------------------------------------------------------- audit (held out by video)
te = np.array([int(hashlib.md5(v.encode()).hexdigest(), 16) % 5 == 0 for v in D.vid]); tr = ~te
yA = D.goldA.to_numpy()
optA = np.where(yA, D.gold, D.dis); optB = np.where(yA, D.dis, D.gold)
res = {}
res["option letter prior"] = np.mean(yA[te] == (yA[tr].mean() >= 0.5))
cnt_g = collections.Counter(np.where(yA[tr], optA[tr], optB[tr])); cnt_o = collections.Counter(list(optA[tr]) + list(optB[tr]))
sc = lambda o: cnt_g[o] / max(1, cnt_o[o]) + 1e-6 * (int(hashlib.md5(o.encode()).hexdigest(), 16) % 997)
res["option text prior"] = np.mean([(sc(a) > sc(b)) == y for a, b, y in zip(optA[te], optB[te], yA[te])])
vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b").fit(list(optA[tr]) + list(optB[tr]))
Xd = vec.transform(optA) - vec.transform(optB)
lr_pred = LogisticRegression(max_iter=1000, C=1.0).fit(Xd[tr], yA[tr]).predict(Xd); res["caption text LR (A-B tf-idf 1-2gram)"] = np.mean(lr_pred[te] == yA[te])
Lc = (D.t1 - D.t0).to_numpy(); nA = np.array([len(x) for x in np.where(yA, D.gsteps, D.dsteps)]); nB = np.array([len(x) for x in np.where(yA, D.dsteps, D.gsteps)])
cA, cB = np.array([len(x) for x in optA]), np.array([len(x) for x in optB])
qb = np.quantile(Lc[tr], np.linspace(0, 1, 21)[1:-1]); b_ = np.searchsorted(qb, Lc)
rule = collections.defaultdict(collections.Counter)                    # per clip-length bin x (A longer?) x (A more steps?): majority gold letter
key = list(zip(b_, np.sign(cA - cB), np.sign(nA - nB)))
for i in np.nonzero(tr)[0]: rule[key[i]][bool(yA[i])] += 1
res["clip-length bins x longer x more steps"] = np.mean([(rule[key[i]][True] >= rule[key[i]][False]) == yA[i] for i in np.nonzero(te)[0]])
F = np.stack([cA - cB, nA - nB, Lc, Lc * (nA - nB), Lc * (cA - cB), nA, nB, np.log(Lc)], 1); F = (F - F[tr].mean(0)) / (F[tr].std(0) + 1e-9)
res["structural LR (lengths, steps, clip length)"] = np.mean(LogisticRegression(max_iter=1000).fit(F[tr], yA[tr]).predict(F[te]) == yA[te])
for name, f_ in (("phrase", lambda s: s), ("verb", lambda s: [vb_of(x) for x in s])):
    M = markov([f_(s) for s in D.gsteps[tr]])
    for agg in ("mean", "sum"):
        sa = np.array([getattr(np, agg)(M(f_(s))) for s in np.where(yA, D.gsteps, D.dsteps)]); sb = np.array([getattr(np, agg)(M(f_(s))) for s in np.where(yA, D.dsteps, D.gsteps)])
        res[f"Markov {name} bigram ({agg} log p)"] = np.mean((sa[te] > sb[te]) == yA[te])
    Fm = np.stack([sa - sb, nA - nB, cA - cB], 1)
    res[f"Markov {name} + lengths LR"] = np.mean(LogisticRegression(max_iter=1000).fit(Fm[tr], yA[tr]).predict(Fm[te]) == yA[te])
print(f"\nAUDIT capbin (held out by video: test {te.sum()} / {len(D)}; chance 50.0; flag > 55.0)", flush=True)
for k, v in res.items(): print(f"  {k:46s} {100 * v:5.1f}{'   <-- FLAG' if v > 0.55 else ''}", flush=True)
for typ in ("drop+insert", "replace", "swap"):                         # (drop / insert apart would condition on the answer: drop = gold longer)
    m = D.typ.isin(typ.split("+")).to_numpy() & te
    print(f"  per type {typ:8s} test {m.sum():5d}: caption-text LR {100 * np.mean(lr_pred[m] == yA[m]):5.1f}   longer caption is gold {100 * np.mean((cA[m] > cB[m]) == yA[m]):5.1f}", flush=True)

# ---------------------------------------------------------------- write
ROOTS = {}
for src, G in D.groupby("src"):
    out = pd.DataFrame([rows[i] for i in G.index]).reset_index(drop=True); name = f"v17_capbin__{src}"
    report(name, out.to_dict("records")); out.to_parquet(f"{O}/{name}.parquet"); ROOTS[name] = [ROOT[src], 1.0]
for r in D.sample(6, random_state=0).itertuples(): print(f"  [{r.typ}/{r.info}] {r.vid} {r.t0:.1f}-{r.t1:.1f}\n     GOLD {r.gold}\n     DIS  {r.dis}", flush=True)
import fcntl
with open(f"{O}/.roots.lock", "w") as lk:                 # roots.json is shared with another builder: merge under a lock
    fcntl.flock(lk, fcntl.LOCK_EX); p = f"{O}/roots.json"; cur = json.load(open(p)) if os.path.exists(p) else {}
    cur.update(ROOTS); json.dump(cur, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
print("roots.json keys:", sorted(cur), flush=True)
