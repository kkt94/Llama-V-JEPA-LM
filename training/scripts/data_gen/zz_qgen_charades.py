"""(10-27) Charades (third-person, human-labelled action spans) question sets aimed at the weak MVBench fine_grained_pose form ("What is the
action performed by the person in the video?" with NTU-style whole-body options). Every answer comes from the Charades human labels
(Charades_v1_train.csv / Charades_v1_test.csv `actions` = "cXXX start end;...", class names Charades_v1_classes.txt) or from the timestamps;
no model text anywhere. Videos: Charades_v1_480 transcoded to prep/charades_384_15fps/<id>.mp4 (384 short side, 15 fps).
EXCLUDED: every Charades id that appears in any eval manifest (manifests/eval/pilot/*.jsonl, `video` basename or any 5-char token that is a
Charades id - MVBench/STAR, TVBench, Charades-STA, FAVOR, heldout sets): 2,627 of 9,848 videos.

Pose concepts (Charades classes merged into one short NTU/MVBench-style verb phrase; mode = how a clip is cut from a labelled span):
  sit_down   "sit down"            c151                                                  exact  (clip = the labelled span, 1-12 s)
  stand_up   "stand up"            c154                                                  exact
  wake_up    "wake up"             c133 c146                                             exact
  sitting    "sit"                 c010 c011 c059 c123 c125 c135                         crop   (any sub-window of the span)
  lying      "lie down"            c122 c124 c134                                        crop
  walk       "walk"                c097                                                  crop
  run        "run"                 c150                                                  crop
  pick_up    "pick up"             c002 c018 c023 c030 c035 c043 c050 c056 c063 c069 c073 c079 c083 c100 c110 c117 c120 c138  exact
  put_down   "put down"            c001 c009 c017 c022 c028 c034 c042 c049 c054 c062 c068 c071 c077 c081 c086 c099 c109 c116 c119 c130  exact
  throw      "throw"               c003 c024 c031 c036 c045 c058 c064 c074 c080 c101 c126   exact
  dress      "put on clothes"      c148      undress "take off clothes"  c155             crop
  shoe_on    "put on shoes"        c055      shoe_off "take off shoes"   c057             crop
  drink      "drink"               c106                                                  semi   (sub-window covering >= 50 % of the span)
  eat        "eat"                 c156 c065                                             crop
  sneeze     "sneeze"              c153                                                  exact
  phone_call "talk on the phone"   c019      phone_play "play with a phone"  c016          crop
  read       "read a book"         c032      write "write on paper"  c145                 crop
  take_photo "take a picture"      c087                                                  semi
  fix_hair   "fix their hair"      c144                                                  semi
  sweep      "sweep the floor"     c102                                                  crop
(not used: object-specific holding / opening / closing / tidying / washing / watching / smiling / laughing / cooking / light / door /
 vacuum classes - not whole-body, and too rare: standing on a chair c060, washing hands c139)

Clip length is decoupled from the answer: one shared length distribution T (0.5 s bins over 1-12 s, T ~ the minimum over the exact-mode
concepts of their span-length densities); every concept gets the SAME number of clips per bin (exact: spans whose length is in the bin;
crop/semi: a window whose length is a pooled exact-span length from that bin). A span is reused at most R times per set (pose_mc /
pose_yn-yes R=4, pose_open R=3). pose_open uses its own T_open (same rule over the spans that pass its label filter).

Sets (-> manifests/stage1/opendl3/v17/v17_<set>__charades.parquet, prep root prep/charades_384_15fps, jitter 1.0):
 pose_mc    MVBench wordings ("What is the action performed by the person in the video?" / "Which one of these descriptions correctly
            matches the actions in the video?" + 2 more); 4 options = gold + 3 distractors from the concept set, chosen least-offered
            first (every option string gold ~1/4 of the times it is offered); a distractor is never labelled in a span overlapping the
            clip window and never "compatible" with the gold (CONFLICT: sit_down~sitting, walk~run, put_down~throw, dress~shoe_on, ...).
            Gold concept counts exactly equal (24 concepts x 368).
 pose_yn    "Is the person <ing> in the video?" etc.; yes / no exactly 50 / 50 PER ASKED CONCEPT; a 'no' clip is a clip of another concept
            in the SAME 0.5 s length bin as its paired 'yes' clip, on which the asked concept is not labelled (overlapping) and no labelled
            concept is compatible with it; half lettered (yes / no options), half "Answer yes or no."  (24 concepts x 186 yes + 186 no)
 pose_open  "What is the action performed by the person in the video?\nAnswer in a few words." -> the phrase; clip must have no other pose
            concept labelled in an overlapping span unless it is compatible (sit_down + sitting). Concepts exactly balanced; concepts
            with < 150 valid clips are left out of this set only (wake_up, shoe_off, fix_hair) -> 21 x 169.
 order2     2-4 labelled actions (any of the 157 classes) of the same video, pairwise non-overlapping with >= 1 s gaps, each class labelled
            exactly once in the video (all valid chains enumerated; up to 8 per video, option count drawn uniformly among the counts the
            video allows); "What does the person do first / last?" (base / past / gerund forms); clip = from the first start
            to the last end (+0-1.5 s). first / last exactly balanced; per class, times gold as first ~= times gold as last.
Gold LETTERS are balanced exactly per answer key (gold position assigned round-robin within each key). Second-turn reorders as zz_qgen_common.
Audit (text-only shortcuts, held out by md5(video id) % 5 == 0): option prior, question text LR, clip length (20 bins) -> appended.
usage: zz_qgen_charades.py  (run on w3 via srun; output log logs/zz_qgen_charades.out)"""
import sys, os, re, csv, json, glob, hashlib, collections, itertools
import numpy as np, pandas as pd
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
R = "/data/llama_vjepa_lm"; D = f"{R}/data/opendl/charades/Charades"; PREP = f"{R}/prep/charades_384_15fps"; O = f"{R}/manifests/stage1/opendl3/v17"
os.makedirs(O, exist_ok=True); g = Gen(117); rng = g.rng
CN = {l[:4]: l[5:].strip() for l in open(f"{D}/Charades_v1_classes.txt") if l.strip()}
# ---------------- videos, labels, eval exclusion
V = {}
for f in ("Charades_v1_train.csv", "Charades_v1_test.csv"):
    for r in csv.DictReader(open(f"{D}/{f}")):
        ln = float(r["length"]); sp = []
        for a in r["actions"].split(";"):
            if not a.strip(): continue
            c, s, e = a.split(); s, e = max(0.0, float(s)), min(ln, float(e))
            if e > s and c in CN: sp.append((c, s, e))
        V[r["id"]] = (ln, sorted(sp, key=lambda x: x[1]))
ids = set(V); EX = set()
for p in glob.glob(f"{R}/manifests/eval/pilot/*.jsonl"):
    for line in open(p):
        try: v = str(json.loads(line).get("video") or "")
        except Exception: v = ""
        s = os.path.splitext(os.path.basename(v))[0]
        if s in ids: EX.add(s)
        EX |= {t for t in re.findall(r"(?<![A-Za-z0-9])[A-Z0-9]{5}(?![A-Za-z0-9])", line) if t in ids}
noprep = {v for v in ids - EX if not os.path.exists(f"{PREP}/{v}.mp4")}
USE = sorted(ids - EX - noprep)
print(f"videos {len(ids)} | excluded (in eval manifests) {len(EX)} | missing prep {len(noprep)} | used {len(USE)}", flush=True)
# ---------------- pose concepts
C = {  # key: (option phrase, 'Does the person ...' base, gerund, 'The person ...' 3rd person, classes, mode)
    "sit_down": ("sit down", "sit down", "sitting down", "sits down", "c151", "exact"),
    "stand_up": ("stand up", "stand up", "standing up", "stands up", "c154", "exact"),
    "wake_up": ("wake up", "wake up", "waking up", "wakes up", "c133 c146", "exact"),
    "sitting": ("sit", "sit", "sitting", "is sitting", "c010 c011 c059 c123 c125 c135", "crop"),
    "lying": ("lie down", "lie down", "lying down", "is lying down", "c122 c124 c134", "crop"),
    "walk": ("walk", "walk", "walking", "walks", "c097", "crop"),
    "run": ("run", "run", "running", "runs", "c150", "crop"),
    "pick_up": ("pick up", "pick something up", "picking something up", "picks something up",
                "c002 c018 c023 c030 c035 c043 c050 c056 c063 c069 c073 c079 c083 c100 c110 c117 c120 c138", "exact"),
    "put_down": ("put down", "put something down", "putting something down", "puts something down",
                 "c001 c009 c017 c022 c028 c034 c042 c049 c054 c062 c068 c071 c077 c081 c086 c099 c109 c116 c119 c130", "exact"),
    "throw": ("throw", "throw something", "throwing something", "throws something", "c003 c024 c031 c036 c045 c058 c064 c074 c080 c101 c126", "exact"),
    "dress": ("put on clothes", "put on clothes", "putting on clothes", "puts on clothes", "c148", "crop"),
    "undress": ("take off clothes", "take off clothes", "taking off clothes", "takes off clothes", "c155", "crop"),
    "shoe_on": ("put on shoes", "put on shoes", "putting on shoes", "puts on shoes", "c055", "crop"),
    "shoe_off": ("take off shoes", "take off shoes", "taking off shoes", "takes off shoes", "c057", "crop"),
    "drink": ("drink", "drink something", "drinking", "drinks", "c106", "semi"),
    "eat": ("eat", "eat something", "eating", "eats", "c156 c065", "crop"),
    "sneeze": ("sneeze", "sneeze", "sneezing", "sneezes", "c153", "exact"),
    "phone_call": ("talk on the phone", "talk on the phone", "talking on the phone", "talks on the phone", "c019", "crop"),
    "phone_play": ("play with a phone", "play with a phone", "playing with a phone", "plays with a phone", "c016", "crop"),
    "read": ("read a book", "read a book", "reading a book", "reads a book", "c032", "crop"),
    "write": ("write on paper", "write on paper", "writing on paper", "writes on paper", "c145", "crop"),
    "take_photo": ("take a picture", "take a picture", "taking a picture", "takes a picture", "c087", "semi"),
    "fix_hair": ("fix their hair", "fix their hair", "fixing their hair", "fixes their hair", "c144", "semi"),
    "sweep": ("sweep the floor", "sweep the floor", "sweeping the floor", "sweeps the floor", "c102", "crop"),
}
K = list(C); CL2K = {c: k for k, v in C.items() for c in v[4].split()}
CONFLICT = {frozenset(p) for p in [("sit_down", "sitting"), ("stand_up", "sitting"), ("stand_up", "lying"), ("wake_up", "lying"), ("wake_up", "sitting"),
            ("wake_up", "stand_up"), ("walk", "run"), ("put_down", "throw"), ("phone_call", "phone_play"), ("take_photo", "phone_play"), ("read", "write"),
            ("dress", "shoe_on"), ("undress", "shoe_off"), ("eat", "drink"), ("sitting", "read"), ("sitting", "write"), ("sitting", "phone_play"),
            ("sitting", "phone_call"), ("sitting", "eat"), ("sitting", "drink"), ("sitting", "shoe_on"), ("sitting", "shoe_off"), ("lying", "read"),
            ("lying", "phone_play"), ("lying", "phone_call"), ("sit_down", "shoe_on"), ("sit_down", "shoe_off")]}
ok_pair = lambda a, b: a != b and frozenset((a, b)) not in CONFLICT
PL = {v: [(CL2K[c], s, e) for c, s, e in V[v][1] if c in CL2K] for v in USE}       # pose-concept labels per video
labelled = lambda v, t0, t1: {k for k, s, e in PL[v] if s < t1 and e > t0}
SP = collections.defaultdict(list)
for v in USE:
    for k, s, e in PL[v]: SP[k].append((v, s, e))
# ---------------- shared clip-length distribution
B = np.arange(1.0, 12.01, 0.5); nb = len(B) - 1; binof = lambda x: int(np.searchsorted(B, x, side="right") - 1)
EXL = collections.defaultdict(list)                                                 # pooled exact-span lengths per bin
dens = []
for k in K:
    if C[k][5] != "exact": continue
    L_ = np.array([e - s for _, s, e in SP[k]]); L_ = L_[(L_ >= 1) & (L_ < 12)]; h = np.histogram(L_, B)[0]; dens.append(h / h.sum())
    for x in L_: EXL[binof(x)].append(float(x))
T = np.min(dens, axis=0); T = T / T.sum()
print("T (0.5 s bins from 1 s):", " ".join(f"{x:.3f}" for x in T), flush=True)
def alloc(N, T=T):
    f = N * T; a = np.floor(f).astype(int); r = N - a.sum(); a[np.argsort(-(f - a), kind="stable")[:r]] += 1; return a
def sampler(k, N, Rmax, ok=None, seed=0, T=T):
    """N clips (video, t0, t1) of concept k, alloc(N) per length bin; each span used <= Rmax times; ok(v, t0, t1) filter. None if short."""
    r_ = np.random.default_rng(seed); mode = C[k][5]; out = []; use = collections.Counter()   # use: per span, across bins
    for b, n in enumerate(alloc(N, T)):
        if n == 0: continue
        lo, hi = B[b], B[b + 1]
        if mode == "exact": cand = [x for x in SP[k] if lo <= x[2] - x[1] < hi]
        elif mode == "crop": cand = [x for x in SP[k] if x[2] - x[1] >= lo]
        else: cand = [x for x in SP[k] if lo <= x[2] - x[1] < 2 * hi]
        got = 0
        for rep in range(Rmax):
            for i in r_.permutation(len(cand)):
                if got == n: break
                v, s, e = cand[i]
                if use[cand[i]] >= Rmax: continue
                if mode == "exact": t0, t1 = s, e
                else:
                    ln = float(r_.choice(EXL[b]))
                    if e - s < ln or (mode == "semi" and ln < 0.5 * (e - s)): continue
                    t0 = float(r_.uniform(s, max(s, e - ln))); t1 = t0 + ln
                if ok is not None and not ok(v, t0, t1): continue
                out.append((v, round(t0, 2), round(t1, 2))); got += 1; use[cand[i]] += 1
        if got < n: return None
    return out
def capacity(k, Rmax, ok=None, T=T):
    lo, hi = 50, 4000
    while lo < hi:
        m = (lo + hi + 1) // 2
        if sampler(k, m, Rmax, ok, 0, T) is not None: lo = m
        else: hi = m - 1
    return lo
def draw(k, N, Rmax, ok=None, seed=0, T=T):
    for t in range(30):
        x = sampler(k, N, Rmax, ok, seed + 7919 * t, T)
        if x is not None: return x
    raise RuntimeError((k, N))
# open answers: every other pose concept labelled in the clip must be compatible with the gold (e.g. sit_down + sitting)
uniq = lambda k: (lambda v, t0, t1: all(z == k or not ok_pair(k, z) for z in labelled(v, t0, t1)))
# pose_open has its own length distribution T_open: the same minimum-density rule over the exact-mode spans that pass uniq (with
# the shared T, rare short bins had no valid sit_down / stand_up clip)
dens_o = []
for k in K:
    if C[k][5] != "exact": continue
    L_ = np.array([e - s for v, s, e in SP[k] if 1 <= e - s < 12 and uniq(k)(v, s, e)])
    if len(L_) >= 100: h = np.histogram(L_, B)[0]; dens_o.append(h / h.sum())
T_open = np.min(dens_o, axis=0); T_open = T_open / T_open.sum()
print("T_open:", " ".join(f"{x:.3f}" for x in T_open), flush=True)
CAP = {k: (capacity(k, 4), capacity(k, 3, uniq(k), T_open)) for k in K}
print("capacity per concept (any, R=4 / only compatible pose labels in the clip, R=3):", CAP, flush=True)
# ---------------- rows: store (q, opts, gold index, key, video id, t0, t1, form) then balance + letter assignment
out = collections.defaultdict(list)
def put(name, v, t0, t1, q, opts, k, key, form="mc"):
    out[name].append(dict(q=q, opts=opts, k=k, _key=key, vid=v, t0=t0, t1=t1, form=form))
MCQ = ["What is the action performed by the person in the video?", "Which one of these descriptions correctly matches the actions in the video?",
       "What is the person doing in the video?", "Which action does the person perform in the video?"]
def opt_text(k, qi): return f"The person {C[k][3]}." if qi == 1 else C[k][0]
# pose_mc
N_MC = min(c[0] for c in CAP.values()) - 5; N_MC = min(N_MC, 40000 // len(K)); print("pose_mc per concept", N_MC, flush=True)
offered = collections.Counter(); rows_mc = []
for j, k in enumerate(K):
    for v, t0, t1 in draw(k, N_MC, 4, seed=1000 + j): rows_mc.append((k, v, t0, t1))
for i in rng.permutation(len(rows_mc)):
    k, v, t0, t1 = rows_mc[i]; lab = labelled(v, t0, t1)
    el = [x for x in K if x not in lab and ok_pair(k, x)]
    if len(el) < 3: continue
    ds = []
    for _ in range(3):                                                              # least offered first (ratio gold / offered flat)
        m = min(offered[x] for x in el if x not in ds); pool = [x for x in el if x not in ds and offered[x] <= m + 1]
        ds.append(pool[int(rng.integers(len(pool)))])
    for x in ds: offered[x] += 1
    offered[k] += 1; qi = int(rng.integers(len(MCQ)))
    put("v17_pose_mc__charades", v, t0, t1, MCQ[qi], [opt_text(x, qi) for x in [k] + ds], 0, k)
# pose_yn
YNQ = ["Is the person {ing} in the video?", "Does the person {base} in the video?", "Does the video show the person {ing}?", "Is the person in the video {ing}?"]
N_YN = min(min(c[0] for c in CAP.values()) // 2, 20000 // len(K)); print("pose_yn per concept per answer", N_YN, flush=True)
pool = {k: sampler(k, 3 * N_YN, 8, seed=2000 + j) or draw(k, N_YN, 8, seed=2000 + j) for j, k in enumerate(K)}
PB = collections.defaultdict(list)                                                  # (concept, length bin) -> shuffled clips
for y, X in pool.items():
    for x in X: PB[(y, binof(x[2] - x[1]))].append(x)
for key_ in list(PB): PB[key_] = [PB[key_][i] for i in rng.permutation(len(PB[key_]))]
ptr = collections.Counter()
for j, k in enumerate(K):
    yes = draw(k, N_YN, 4, seed=3000 + j); nos = []
    others = [y for y in K if ok_pair(k, y)]
    for x in yes:                                                                   # one 'no' clip per 'yes' clip, same length bin
        b = binof(x[2] - x[1]); ys = [y for y in others if PB.get((y, b))]
        for tries in range(400):
            y = ys[int(rng.integers(len(ys)))]; L_ = PB[(y, b)]; v, t0, t1 = L_[ptr[(y, b)] % len(L_)]; ptr[(y, b)] += 1
            if all(z != k and ok_pair(k, z) for z in labelled(v, t0, t1)): nos.append((v, t0, t1)); break
    assert len(nos) == N_YN, (k, len(nos))
    for ans, clips in (("yes", yes), ("no", nos)):
        for v, t0, t1 in clips:
            q = YNQ[int(rng.integers(len(YNQ)))].format(ing=C[k][2], base=C[k][1])
            if rng.random() < 0.5: put("v17_pose_yn__charades", v, t0, t1, q, ["yes", "no"], 0 if ans == "yes" else 1, ans)
            else: put("v17_pose_yn__charades", v, t0, t1, q, None, ans.capitalize() + ".", ans, form="yn_open")
# pose_open
OQ = [("What is the action performed by the person in the video?", 0), ("What is the person doing in the video?", 2), ("What action does the person perform?", 0)]
KO = [k for k in K if CAP[k][1] >= 150]; N_OP = min(min(CAP[k][1] for k in KO) - 5, 20000 // len(KO))
print("pose_open concepts", len(KO), "dropped (open capacity < 150):", [k for k in K if k not in KO], "per concept", N_OP, flush=True)
for j, k in enumerate(KO):
    for v, t0, t1 in draw(k, N_OP, 3, uniq(k), seed=4000 + j, T=T_open):
        q, f = OQ[int(rng.integers(len(OQ)))]; a = C[k][f]
        put("v17_pose_open__charades", v, t0, t1, q, None, a[0].upper() + a[1:] + ".", k, form="open")
# ---------------- order2 (any class, timestamps)
IRR = {"Holding": ("hold", "held"), "Putting": ("put", "put"), "Taking": ("take", "took"), "Throwing": ("throw", "threw"), "Tidying": ("tidy", "tidied"),
       "Washing": ("wash", "washed"), "Closing": ("close", "closed"), "Fixing": ("fix", "fixed"), "Opening": ("open", "opened"), "Sitting": ("sit", "sat"),
       "Working": ("work", "worked"), "Playing": ("play", "played"), "Talking": ("talk", "talked"), "Smiling": ("smile", "smiled"), "Watching": ("watch", "watched"),
       "Reading": ("read", "read"), "Looking": ("look", "looked"), "Eating": ("eat", "ate"), "Making": ("make", "made"), "Snuggling": ("snuggle", "snuggled"),
       "Reaching": ("reach", "reached"), "Laughing": ("laugh", "laughed"), "Walking": ("walk", "walked"), "Turning": ("turn", "turned"), "Drinking": ("drink", "drank"),
       "Pouring": ("pour", "poured"), "Lying": ("lie", "lay"), "Grasping": ("grasp", "grasped"), "Standing": ("stand", "stood"), "awakening": ("awaken", "awakened"),
       "cooking": ("cook", "cooked"), "dressing": ("dress", "dressed"), "running": ("run", "ran"), "going": ("go", "went"), "sneezing": ("sneeze", "sneezed"),
       "undressing": ("undress", "undressed"), "smiling": ("smile", "smiled"), "laughing": ("laugh", "laughed"), "eating": ("eat", "ate"),
       "standing": ("stand", "stood"), "holding": ("hold", "held"), "consuming": ("consume", "consumed"), "Wash": ("wash", "washed")}
def forms(c):
    n = re.sub(r"^Someone is ", "", CN[c]); w, rest = (n.split(" ", 1) + [""])[:2]; parts = w.split("/")
    base = "/".join(IRR[p][0] if p in IRR else IRR[p.capitalize()][0] for p in parts); past = "/".join(IRR[p][1] if p in IRR else IRR[p.capitalize()][1] for p in parts)
    ing = "/".join((p if p.endswith("ing") else "washing") for p in parts).lower()
    cap = lambda s: s[0].upper() + s[1:]
    return {"base": cap(f"{base} {rest}".strip()), "past": cap(f"{past} {rest}".strip()) + ".", "ing": cap(f"{ing} {rest}".strip())}
FORMS = {c: forms(c) for c in CN}
FORMS["c083"] = {"base": "Reach for and grab a picture", "past": "Reached for and grabbed a picture.", "ing": "Reaching for and grabbing a picture"}
OQS = {"first": [("What does the person do first?", "base"), ("Which action does the person perform first in the video?", "base"),
                 ("What did the person do first?", "past"), ("Which of these happens first in the video?", "ing")],
       "last": [("What does the person do last?", "base"), ("Which action does the person perform last in the video?", "base"),
                ("What did the person do last?", "past"), ("Which of these happens last in the video?", "ing")]}
cand_o = []
for v in USE:
    ln, sp = V[v]; cnt = collections.Counter(c for c, _, _ in sp); one = [x for x in sp if cnt[x[0]] == 1 and x[2] - x[1] >= 1.0]
    byn = {n: [c_ for c_ in itertools.combinations(one, n) if all(c_[i + 1][1] - c_[i][2] >= 1.0 for i in range(n - 1))] for n in (2, 3, 4)}
    for _ in range(8):                                                              # up to 8 sets per video, option count drawn
        ns = [n for n in (2, 3, 4) if byn[n]]                                       # uniformly among those the video allows
        if not ns: break
        n = ns[int(rng.integers(len(ns)))]; cand_o.append((v, byn[n].pop(int(rng.integers(len(byn[n]))))))
print("order candidates", len(cand_o), flush=True)
pos = collections.defaultdict(lambda: [0, 0])                                      # per class: gold-as-first, gold-as-last
for i in rng.permutation(len(cand_o)):
    v, s = cand_o[i]; first, last = s[0][0], s[-1][0]
    # ask the type that keeps each class's (gold as first) == (gold as last)
    df = pos[first][0] - pos[first][1]; dl = pos[last][1] - pos[last][0]
    typ = "first" if df < dl else "last" if dl < df else ("first" if rng.random() < 0.5 else "last")
    gc = first if typ == "first" else last; pos[gc][0 if typ == "first" else 1] += 1
    q, f = OQS[typ][int(rng.integers(4))]; opts = [FORMS[c][f] for c, _, _ in s]; k = [c for c, _, _ in s].index(gc)
    ln = V[v][0]; t0 = max(0.0, s[0][1] - float(rng.uniform(0, 1.5))); t1 = min(ln, max(e for _, _, e in s) + float(rng.uniform(0, 1.5)))
    put("v17_order2__charades", v, round(t0, 2), round(t1, 2), q, opts, k, typ)
# ---------------- balance keys exactly, assign gold letters round-robin per key, write
CAP_ROWS = 40000; ROOTS = {}
for name, rows in sorted(out.items()):
    d = pd.DataFrame(rows); before = d._key.value_counts().to_dict(); m = int(d._key.value_counts().min()); m = min(m, CAP_ROWS // d._key.nunique())
    d = pd.concat([gg.sample(m, random_state=0) for _, gg in d.groupby("_key")]).sample(frac=1, random_state=1).reset_index(drop=True)
    print(name, "keys before", dict(sorted(before.items())), "-> each", m, flush=True)
    recs = []; slot = collections.Counter()
    for r in d.to_dict("records"):
        if r["form"] == "mc":
            n = len(r["opts"]); gpos = slot[(r["_key"], n)] % n; slot[(r["_key"], n)] += 1
            rest = [o for i, o in enumerate(r["opts"]) if i != r["k"]]; rest = [rest[i] for i in rng.permutation(len(rest))]
            opts = rest[:gpos] + [r["opts"][r["k"]]] + rest[gpos:]; conv = g.render(r["q"], opts, gpos)
        else:
            tail = "\nAnswer yes or no." if r["form"] == "yn_open" else "\nAnswer in a few words."
            conv = json.dumps([{"role": "user", "content": f"<|jepa_evidence|>\n{r['q']}{tail}"}, {"role": "assistant", "content": r["k"]}])
        recs.append(dict(video=f"{r['vid']}.mp4", prep_exists=os.path.exists(f"{PREP}/{r['vid']}.mp4"), root2=None, tq=None, aug="",
                         t0=float(r["t0"]), t1=float(r["t1"]), conversations=conv))
    report(name, recs); pd.DataFrame(recs).to_parquet(f"{O}/{name}.parquet"); ROOTS[name] = [PREP, 1.0]
rp = f"{O}/roots.json"; old = json.load(open(rp)) if os.path.exists(rp) else {}
old.update(ROOTS); tmp = rp + ".tmp.charades"; json.dump(old, open(tmp, "w"), indent=1); os.replace(tmp, rp)
print("roots.json keys", sorted(old), flush=True)
# ---------------- audit (zz_audit_v16 style; non-lettered rows: offered = answers seen with the same question text)
from sklearn.feature_extraction.text import TfidfVectorizer; from sklearn.linear_model import LogisticRegression
def parse(c):
    c = json.loads(c); u = c[0]["content"].replace("<|jepa_evidence|>\n", ""); a = c[1]["content"].strip(); opts = re.findall(r"^([A-H])\. (.+)$", u, flags=re.M)
    return u.split("\n")[0], ([o for _, o in opts] if opts else None), (dict(opts)[a[0]] if opts else a)
print("\nAUDIT (held out by md5(video id) % 5 == 0; chance = mean 1/#offered)", flush=True)
for name in sorted(ROOTS):
    d = pd.read_parquet(f"{O}/{name}.parquet"); P = [list(parse(c)) for c in d.conversations]
    byq = collections.defaultdict(set)
    for q, o, a in P:
        if o is None: byq[q].add(a)
    for p in P:
        if p[1] is None: p[1] = ["No.", "Yes."] if p[2] in ("Yes.", "No.") else sorted(byq[p[0]])
    vid = [x[:-4] for x in d.video]; te = np.array([int(hashlib.md5(v.encode()).hexdigest(), 16) % 5 == 0 for v in vid]); tr = ~te
    cnt_g = collections.Counter(P[i][2] for i in np.nonzero(tr)[0]); cnt_o = collections.Counter(o for i in np.nonzero(tr)[0] for o in P[i][1])
    pri = np.mean([max(P[i][1], key=lambda o: cnt_g[o] / max(1, cnt_o[o]) + 1e-6 * (hash(o) % 997)) == P[i][2] for i in np.nonzero(te)[0]])
    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2)); X = vec.fit_transform([p[0] for p in P]); y = np.array([p[2] for p in P])
    clf = LogisticRegression(max_iter=300, C=1.0).fit(X[tr], y[tr]); pr = clf.predict_proba(X[te]); cl = {c: j for j, c in enumerate(clf.classes_)}
    txt = np.mean([max(P[i][1], key=lambda o: pr[n, cl[o]] if o in cl else -1) == P[i][2] for n, i in enumerate(np.nonzero(te)[0])])
    Lc = (d.t1 - d.t0).to_numpy(); qb = np.quantile(Lc[tr], np.linspace(0, 1, 21)[1:-1]); b_ = np.searchsorted(qb, Lc); bc = collections.defaultdict(collections.Counter)
    for i in np.nonzero(tr)[0]: bc[b_[i]][P[i][2]] += 1
    dur = np.mean([max(P[i][1], key=lambda o: bc[b_[i]][o] + 1e-6 * (hash(o) % 997)) == P[i][2] for i in np.nonzero(te)[0]])
    ch = np.mean([1 / len(P[i][1]) for i in np.nonzero(te)[0]]); flag = "  <-- FIX" if max(pri, txt, dur) > ch + 0.05 else ""
    print(f"{name:28s} n {len(d):6d} test {te.sum():5d} videos {len(set(vid)):5d} chance {100*ch:5.1f}  option-prior {100*pri:5.1f}  question-text {100*txt:5.1f}"
          f"  clip-length {100*dur:5.1f}  len mean {Lc.mean():.1f}s{flag}", flush=True)
