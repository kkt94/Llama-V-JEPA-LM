"""(10-27) finemotion - minimal pairs on subtle dynamics, answers by construction (known transformations of human-labelled clips):
 speed  TempCompass speed form on ONE Something-Something v2 clip shown twice, above and below (tq kind "stack": each panel 32 frames
        evenly over its own playback span, given as fractions of the clip; frame times nominal, jitter 0). The two playback rates differ
        only slightly - longer span / shorter span = r ~ U(1.15, 1.40) - or are equal (both spans the same length, different start
        points). Every clip gives a MINIMAL PAIR of two rows: an unequal clip as it is and with the two panels exchanged (faster <-> slower,
        same question text), or an equal-rate clip at two different offsets (same / same). faster : slower : same = 1 : 1 : 1; the two
        panels' start points always differ by >= 0.05 of the clip (identical panels would give "same" away).
        "Is the X above moving faster, slower, or at the same speed as the X below?" (50 %) and TempCompass's other speed wordings;
        X = the SSv2 object placeholder (human label). Clips: SSv2 TRAIN, moving templates, >= 3.5 s.
 dir    direction minimal pairs: SSv2 TRAIN clips whose human template states a horizontal direction - "Pushing / Pulling [something]
        from left to right / right to left", "Turning the camera left / right while filming [something]" - each clip twice: as it is and
        mirrored (aug "flip": video.flip(-1) in the loader; a mirror swaps left and right), SAME question text and options, opposite
        answers. TempCompass direction wordings (multi-choice / yes-no / caption matching), only the two horizontal answers offered
        (an option that is never the answer would be an option-prior shortcut).
SSv2 VALIDATION clips are not used (held-out pilot set). Held-out-by-video audit: option prior, question-text logistic regression, clip
duration bins (decord duration of the source clip; the frames' times are nominal for "stack" rows anyway).
-> manifests/stage1/opendl3/v17/v17_finemotion_{speed,dir}__ssv2.parquet + roots.json   usage: zz_qgen_finemotion.py > zz_qgen_finemotion.out"""
import sys, os, re, json, hashlib, collections
import numpy as np, pandas as pd
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
R = "/data/llama_vjepa_lm"; O = f"{R}/manifests/stage1/opendl3/v17"; os.makedirs(O, exist_ok=True); g = Gen(172); rng = g.rng
SS = f"{R}/bench/mvp_sources/extracted/ssv2/20bn-something-something-v2"
N_SPEED_CLIPS = 8700                                       # x 2 rows


def fill(t, ph):
    i = iter(ph); return re.sub(r"\[[^\]]*\]", lambda m: next(i, "something"), t)


def obj_name(o):
    o = re.sub(r"^(a|an|the|one of the|one|some|two|three|a pair of|a piece of)\s+", "", o.strip().lower())
    return o if 1 <= len(o.split()) <= 4 else None


D = pd.DataFrame(json.load(open(f"{R}/data/ssv2_labels/labels/train.json")))
D = D[[len(p) > 0 and not any(x.split()[:1] and x.split()[0].lower().endswith("ing") for x in p) for p in D.placeholders]]
D["obj"] = [obj_name(p[0]) for p in D.placeholders]; D = D[D.obj.notna()]
D["video"] = [f"{i}.webm" for i in D.id]; D = D[[os.path.exists(f"{SS}/{v}") for v in D.video]]
print("SSv2 train clips with an object name:", len(D), flush=True)

# ---------------------------------------------------------------- dir: mirrored minimal pairs
DIRT = {"Pushing [something] from left to right": ("obj", "right", "pushes"), "Pushing [something] from right to left": ("obj", "left", "pushes"),
        "Pulling [something] from left to right": ("obj", "right", "pulls"), "Pulling [something] from right to left": ("obj", "left", "pulls"),
        "Turning the camera left while filming [something]": ("cam", "left", ""), "Turning the camera right while filming [something]": ("cam", "right", "")}
OBJF = [("What is the direction of the {x}?", {"right": "moving from left to right", "left": "moving from right to left"}),
        ("From the camera's point of view, in which direction is the {x} moving?", {"right": "from left to right", "left": "from right to left"}),
        ("How does the {x} move in the horizontal direction?", {"right": "left to right", "left": "right to left"}),
        ("In relation to the camera's position, which way is the {x} headed?", {"right": "from left to right", "left": "from right to left"})]
CAMF = [("In which direction is the camera turning?", {"right": "to the right", "left": "to the left"}),
        ("What is the camera movement in the video?", {"right": "panning right", "left": "panning left"}),
        ("How does the camera move while filming the {x}?", {"right": "it turns to the right", "left": "it turns to the left"})]
out = collections.defaultdict(list); dirvid = set()
for r in D[D.template.isin(list(DIRT))].itertuples():
    kind, d0, verb = DIRT[r.template]; x = r.obj; f = rng.random()
    if f < 0.6:                                            # multi-choice
        q, W = (OBJF if kind == "obj" else CAMF)[int(rng.integers(4 if kind == "obj" else 3))]; q = q.format(x=x); opts = lambda d: ([W[d], W["left" if d == "right" else "right"]], 0)
    elif f < 0.8:                                          # yes / no: the asked direction drawn once per pair
        ask = g.pick(["left", "right"])
        q = (f"Is the {x} moving from {'left to right' if ask == 'right' else 'right to left'}?" if kind == "obj" else f"Is the camera panning to the {ask}?")
        opts = lambda d: (["Yes", "No"], 0 if d == ask else 1)
    else:                                                  # caption matching
        q = "Which caption matches the video better?"
        cap = (lambda d: f"Someone {verb} the {x} from {'left to right' if d == 'right' else 'right to left'}.") if kind == "obj" else (lambda d: f"The camera turns {d} while filming the {x}.")
        opts = lambda d: ([cap(d), cap("left" if d == "right" else "right")], 0)
    for aug in ("", "flip"):                               # the same clip as it is and mirrored: same text, opposite answer
        d = d0 if aug == "" else ("left" if d0 == "right" else "right"); o_, k = opts(d)
        out["v17_finemotion_dir__ssv2"].append(dict(g.row(r.video, 0, 0, q, o_, k, aug=aug), t0=None, t1=None, _key=d, _kind=kind))
    dirvid.add(r.video)
print("dir pairs:", len(dirvid), flush=True)

# ---------------------------------------------------------------- speed: stacked same-clip minimal pairs
STILL = ("Holding", "Showing", "Pretending", "Something", "Trying", "Poking", "Touching", "Pulling two ends of [something] but nothing",
         "Spinning [something] that quickly stops", "Turning the camera", "Approaching", "Moving away from")
M = D[~D.template.str.startswith(STILL) & ~D.video.isin(dirvid)]
pick = M.sample(min(len(M), 2 * N_SPEED_CLIPS), random_state=0)
import decord
from multiprocessing import Pool
def dur(v):
    try:
        vr = decord.VideoReader(f"{SS}/{v}", num_threads=1); return len(vr) / float(vr.get_avg_fps())
    except Exception:
        return 0.0
with Pool(16) as P: DUR = dict(zip(list(pick.video) + sorted(dirvid), P.map(dur, list(pick.video) + sorted(dirvid), chunksize=64)))
pick = pick[[DUR[v] >= 3.5 for v in pick.video]].head(N_SPEED_CLIPS)
print("speed clips:", len(pick), "duration median", round(float(np.median([DUR[v] for v in pick.video])), 2), flush=True)
SPQ = [("Is the {x} above moving faster, slower, or at the same speed as the {x} below?", {"faster": "faster", "slower": "slower", "same": "at the same speed"}),
       ("What is the speed of the {x} above?", {"faster": "moving faster than the {x} below", "slower": "moving slower than the {x} below", "same": "moving at the same speed as the {x} below"}),
       ("Which {x} is moving faster?", {"faster": "the {x} above", "slower": "the {x} below", "same": "both are moving at the same speed"}),
       ("How does the speed of the {x} above compare to the speed of the {x} below?", {"faster": "The {x} above is moving faster than the {x} below",
        "slower": "The {x} above is moving slower than the {x} below", "same": "The {x} above and the {x} below have the same speed"})]


def spans(L1, L2):
    while True:
        a, b = rng.uniform(0, 1 - L1), rng.uniform(0, 1 - L2)
        if abs(a - b) >= 0.05: return [round(a, 4), round(a + L1, 4)], [round(b, 4), round(b + L2, 4)]


cls = np.array((["unequal"] * 2 + ["same"]) * (len(pick) // 3 + 1))[: len(pick)]; rng.shuffle(cls)
for r, c in zip(pick.itertuples(), cls):
    j = 0 if rng.random() < 0.5 else int(rng.integers(1, 4)); q, W = SPQ[j]; x = r.obj; q = q.format(x=x); W = {k: v.format(x=x) for k, v in W.items()}
    if c == "unequal":
        ratio = float(rng.uniform(1.15, 1.40)); Lf = float(rng.uniform(0.55, 0.95)); fa, sl = spans(Lf, Lf / ratio)
        var = [("faster", fa, sl), ("slower", sl, fa)]     # the pair: panels exchanged
    else:
        L = float(rng.uniform(0.45, 0.95)); var = [("same",) + spans(L, L), ("same",) + spans(L, L)]
    for lab, top, bot in var:
        tq = json.dumps({"kind": "stack", "top": top, "bottom": bot})
        out["v17_finemotion_speed__ssv2"].append(dict(g.row(r.video, 0, 0, q, [W[lab]] + [W[k] for k in W if k != lab], 0), tq=tq, t0=None, t1=None, _key=lab, _kind="speed"))

# ---------------------------------------------------------------- audit (held out by video) + write
from sklearn.feature_extraction.text import TfidfVectorizer; from sklearn.linear_model import LogisticRegression
def parse(c):
    c = json.loads(c); u = c[0]["content"].replace("<|jepa_evidence|>\n", ""); a = c[1]["content"].strip(); opts = re.findall(r"^([A-H])\. (.+)$", u, flags=re.M)
    return u.split("\n")[0], [o for _, o in opts], dict(opts)[a[0]]
ROOTS = {}
for name, rows in sorted(out.items()):
    d = pd.DataFrame(rows); P = [parse(c) for c in d.conversations]
    te = np.array([int(hashlib.md5(v.encode()).hexdigest(), 16) % 5 == 0 for v in d.video]); tr = ~te
    cnt_g = collections.Counter(P[i][2] for i in np.nonzero(tr)[0]); cnt_o = collections.Counter(o for i in np.nonzero(tr)[0] for o in P[i][1])
    pri = np.mean([max(P[i][1], key=lambda o: cnt_g[o] / max(1, cnt_o[o]) + 1e-6 * (hash(o) % 997)) == P[i][2] for i in np.nonzero(te)[0]])
    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2)); X = vec.fit_transform([p[0] for p in P]); y = np.array([p[2] for p in P])
    clf = LogisticRegression(max_iter=500).fit(X[tr], y[tr]); pr = clf.predict_proba(X[te]); cl = {c: j for j, c in enumerate(clf.classes_)}
    txt = np.mean([max(P[i][1], key=lambda o: pr[n, cl[o]] if o in cl else -1) == P[i][2] for n, i in enumerate(np.nonzero(te)[0])])
    Lc = np.array([DUR[v] for v in d.video]); qb = np.quantile(Lc[tr], np.linspace(0, 1, 21)[1:-1]); b_ = np.searchsorted(qb, Lc)
    bc = collections.defaultdict(collections.Counter)
    for i in np.nonzero(tr)[0]: bc[b_[i]][P[i][2]] += 1
    du = np.mean([max(P[i][1], key=lambda o: bc[b_[i]][o] + 1e-6 * (hash(o) % 997)) == P[i][2] for i in np.nonzero(te)[0]])
    # answer CLASS (faster / slower / same, left / right) from the clip duration bin + template kind: a class-level shortcut check
    kk = list(zip(b_, d._kind)); bk = collections.defaultdict(collections.Counter)
    for i in np.nonzero(tr)[0]: bk[kk[i]][d._key.iloc[i]] += 1
    duk = np.mean([max(bk[kk[i]], key=lambda c: bk[kk[i]][c] + 1e-6 * (hash(c) % 997)) == d._key.iloc[i] for i in np.nonzero(te)[0]])
    ch = np.mean([1 / len(P[i][1]) for i in np.nonzero(te)[0]])
    fl = lambda v: f"{100 * v:5.1f}{'*' if v > ch + 0.05 else ' '}"
    print(f"AUDIT {name:28s} n {len(d):6d} test {te.sum():5d} chance {100 * ch:5.1f}  option-prior {fl(pri)} question-text {fl(txt)} clip-length {fl(du)} "
          f"| class from duration bin x kind {fl(duk)} (class chance {100 / d._key.nunique():.1f})  (* = > chance + 5)", flush=True)
    print("   answers:", d._key.value_counts().to_dict(), "| aug:", d.aug.value_counts().to_dict(), flush=True)
    d = d.drop(columns=["_key", "_kind"]).reset_index(drop=True); report(name, d.to_dict("records")); d.to_parquet(f"{O}/{name}.parquet"); ROOTS[name] = [SS, 0.0]
import fcntl
with open(f"{O}/.roots.lock", "w") as lk:                 # roots.json is shared with another builder: merge under a lock
    fcntl.flock(lk, fcntl.LOCK_EX); p = f"{O}/roots.json"; cur = json.load(open(p)) if os.path.exists(p) else {}
    cur.update(ROOTS); json.dump(cur, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)
print("roots.json keys:", sorted(cur), flush=True)
