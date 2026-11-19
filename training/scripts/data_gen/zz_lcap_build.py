"""(10-28) lcap step 4: choose ONE negative per gold so that no text-only cue is informative, then write the v20 lcap manifests.
Inputs: zz_lcap_golds.parquet (human golds), zz_lcap_cands.parquet (rule-based single edits), zz_lcap_lp.parquet (base Qwen3-1.7B
log p(caption | 'Video caption: '), bf16 GPU).  Per candidate: d = lp(gold) - lp(neg) (gold more probable when > 0), the token / char length
signs of gold - neg, and the edit direction (which word of the exact word pair the gold holds; 0 for a sentence swap).
MIRROR BALANCE (as zz_qgen_capbin): inside each cell (source style group x edit sub-type x exact word pair x |d| bin) a row with signature k = (direction, sign d, token-length sign,
char-length sign, sign of a cross-fitted tf-idf text classifier (zz_lcap_textlr.py)) is kept only together with as many rows of signature -k. Every function of these features - LM prior, its size, length,
which word of the pair the gold holds, and any odd combination - is then right exactly half of the time, within every edit type.
Selection: golds in random order, each takes the candidate whose mirror signature is in deficit (preferring under-filled edit types,
TemporalBench-like mix), a few refill passes, then an exact trim.  Gold option A in exactly half the rows of every edit type; 30 % of rows
get the second turn with the options reordered (zz_qgen_common.Gen.render, as v17_capbin).  Audit held out by video.
-> manifests/stage1/v20/v20_lcap__<src>.parquet, roots.json, logs/zz_lcap_pairs.parquet   usage: python -I zz_lcap_build.py"""
import sys, os, json, hashlib, collections
import numpy as np, pandas as pd
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
R = "/data/llama_vjepa_lm"; O = os.environ.get("LCAP_OUT", f"{R}/manifests/stage1/v20"); os.makedirs(O, exist_ok=True); g = Gen(2020); rng = g.rng
Q = "Which caption best describes this video?"
G = pd.read_parquet(f"{R}/logs/zz_lcap_golds.parquet").set_index("gid", drop=False)
C = pd.read_parquet(f"{R}/logs/zz_lcap_cands.parquet")
LP = pd.read_parquet(os.environ.get("LCAP_LP", f"{R}/logs/zz_lcap_lp.parquet")); lp = dict(zip(LP.text, LP.lp)); nt = dict(zip(LP.text, LP.ntok))
C = C[C.neg.isin(lp.keys()) & C.gid.map(lambda i: G.cap[i] in lp)].reset_index(drop=True)
C = C[~((C["sub"] == "before_after") & (C.span == 0))].reset_index(drop=True)   # 'After that, ...' opening a caption refers to events outside the clip
C["src"] = G.src.reindex(C.gid).values; C["gold"] = G.cap.reindex(C.gid).values
C["d"] = [lp[a] - lp[b] for a, b in zip(C.gold, C.neg)]
C["tl"] = [int(np.sign(nt[a] - nt[b])) for a, b in zip(C.gold, C.neg)]
C["cl"] = [int(np.sign(len(a) - len(b))) for a, b in zip(C.gold, C.neg)]
C["ds"] = np.sign(C.d).astype(int)
# direction: +1 / -1 by the alphabetical order of the two words of the exact pair (gold word -> negative word); a sentence swap is 0.
# The mirror then pairs 'left>right' rows with 'right>left' rows (same exact word pair, the cell holds the pair)
C["pair"] = ["swap" if x == "swap" else "~".join(sorted(x.lower().split(">"))) for x in C.dirn]
C["dir"] = [0 if x == "swap" else (1 if x.lower().split(">")[0] < x.lower().split(">")[1] else -1) for x in C.dirn]

BINS = [0, 0.5, 1.0, 2.0, 4.0, np.inf]
C["bin"] = np.digitize(C.d.abs(), BINS[1:-1])
SG = {"plm": "plm", "plmd": "plm", "kin": "vidln", "oops": "vidln", "uvo": "vidln", "ovis": "vidln", "anet": "anet", "charades": "charades"}
C["cell"] = C.src.map(SG) + "|" + C["sub"] + "|" + C["pair"] + "|" + C.bin.astype(str)       # source style group in the cell too
SL = pd.read_parquet(f"{R}/logs/zz_lcap_slr.parquet")                   # cross-fitted tf-idf LR sign (zz_lcap_textlr.py)
C = C.merge(SL[["gid", "neg", "s_lr"]], on=["gid", "neg"], how="inner")
C["sig"] = list(zip(C.dir, C.ds, C.tl, C.cl, C.s_lr))
C = C[C.ds != 0].reset_index(drop=True)
neg_sig = lambda k: tuple(-x for x in k)
print("candidates scored", len(C), "golds", C.gid.nunique(), flush=True)
print("gold more probable among ALL candidates, by type:", C.groupby("typ").ds.apply(lambda x: round(float((x > 0).mean()), 3)).to_dict(), flush=True)

# TemporalBench short-QA category mix -> target shares of our types
CAP = 0.15
TARGET = {"action": 0.32, "order": 0.22, "dir": 0.22, "effector": 0.12, "count": 0.07, "speed": 0.05}
SUBN = C.groupby("typ")["sub"].nunique().to_dict()
def select(passes=6):
    """pass 0: every gold takes its best candidate (mirror deficit first, then the most under-filled edit type / sub-type); passes 1..: a gold
    whose current pick is in surplus in its cell switches to a candidate in deficit; finally an exact trim to the mirror"""
    cnt = collections.defaultdict(collections.Counter); chosen = {}; ntyp = collections.Counter(); nsub = collections.Counter()
    by_g = {k: list(v.itertuples()) for k, v in C.groupby("gid")}
    golds = list(by_g); rng.shuffle(golds)
    def pri(r, deficit):
        tot = max(1, sum(ntyp.values()))
        return (deficit > 0, -ntyp[r.typ] / tot / TARGET[r.typ], -nsub[r.sub] / max(1, ntyp[r.typ]) * SUBN[r.typ], rng.random())
    def put(r, k):
        cnt[r.cell][r.sig] += k; ntyp[r.typ] += k; nsub[r.sub] += k
    for gi in golds:
        best = max(((pri(r, cnt[r.cell][neg_sig(r.sig)] - cnt[r.cell][r.sig]), r) for r in by_g[gi]), key=lambda x: x[0])[1]
        chosen[gi] = best; put(best, 1)
    for p in range(1, passes):
        sw = 0; rng.shuffle(golds)
        for gi in golds:
            c = chosen[gi]
            if cnt[c.cell][c.sig] - cnt[c.cell][neg_sig(c.sig)] <= 0: continue          # current pick is matched: keep
            put(c, -1)
            opts = [(pri(r, cnt[r.cell][neg_sig(r.sig)] - cnt[r.cell][r.sig]), r) for r in by_g[gi] if r.Index != c.Index]
            opts = [x for x in opts if x[0][0]]
            if opts: c = max(opts, key=lambda x: x[0])[1]; sw += 1
            chosen[gi] = c; put(c, 1)
        bal = sum(min(v[k], v[neg_sig(k)]) for v in cnt.values() for k in v if k != neg_sig(k))
        print(f"pass {p}: switched {sw}, matched rows {bal}", flush=True)
    S = C.loc[[r.Index for r in chosen.values()]]; keep = []
    for cell, X in S.groupby("cell"):
        grp = {k: v for k, v in X.groupby("sig")}
        for k, v in grp.items():
            nk = neg_sig(k)
            if k == nk: continue
            m = min(len(v), len(grp.get(nk, [])))
            keep += list(v.sample(m, random_state=1).index)
    D = C.loc[sorted(keep)].copy()
    print(f"kept {len(D)} by type {D.typ.value_counts().to_dict()}", flush=True)
    # sub-type cap: no single edit sub-type above CAP of the rows (sit/stand and sentence swaps dominate the pool); removed as mirrored pairs
    cap = int(CAP * len(D))
    for sub, X in D.groupby("sub"):
        if len(X) <= cap: continue
        drop = []
        f = 1 - cap / len(X)
        for cell, Y in X.groupby("cell"):
            for sg, Z in Y.groupby("sig"):
                if sg > neg_sig(sg): continue
                M = Y[Y.sig == neg_sig(sg)]; m = int(round(f * min(len(Z), len(M))))
                drop += list(Z.sample(m, random_state=2).index) + list(M.sample(m, random_state=3).index)
        D = D.drop(index=drop)
    print(f"after sub-type cap {CAP}: {len(D)} by type {D.typ.value_counts().to_dict()}", flush=True)
    return D
D = select()
D = D.sample(frac=1.0, random_state=3).reset_index(drop=True)

# ---------------------------------------------------------------- A / B and rows
D["goldA"] = False
for t, X in D.groupby("typ"):
    ix = X.index.to_numpy().copy(); rng.shuffle(ix); D.loc[ix[: len(ix) // 2], "goldA"] = True
G2 = G.set_index("gid")
D["video"] = G2.video.reindex(D.gid).values; D["t0"] = G2.t0.reindex(D.gid).values; D["t1"] = G2.t1.reindex(D.gid).values
D["extra"] = G2.extra.reindex(D.gid).values
conv = []
for r in D.itertuples():
    opts, k = ([r.gold, r.neg], 0) if r.goldA else ([r.neg, r.gold], 1)
    conv.append(g.render(Q, opts, k))
D["conversations"] = conv

# ---------------------------------------------------------------- stats
print(f"\nPAIRS {len(D)}  golds unique {D.gid.nunique()}  videos {D.video.nunique()}", flush=True)
print("by source x type:\n" + pd.crosstab(D.src, D.typ, margins=True).to_string(), flush=True)
print("by sub-type:", D.groupby(["typ", "sub"]).size().to_dict(), flush=True)
print("gold more probable (LM prior):", round(float((D.d > 0).mean()), 4), "| by type", D.groupby("typ").d.apply(lambda x: round(float((x > 0).mean()), 3)).to_dict(),
      "| by source", D.groupby("src").d.apply(lambda x: round(float((x > 0).mean()), 3)).to_dict(), flush=True)
print("gold = A:", round(float(D.goldA.mean()), 4), D.groupby("typ").goldA.mean().round(3).to_dict(), flush=True)
nz = D[D.tl != 0]; cz = D[D.cl != 0]
print(f"gold longer (tokens) among differing rows: {(nz.tl > 0).mean():.4f} ({len(nz)} rows differ, {len(D) - len(nz)} equal) | "
      f"gold longer (chars): {(cz.cl > 0).mean():.4f} ({len(cz)} differ)", flush=True)
dtok = np.array([nt[a] - nt[b] for a, b in zip(D.gold, D.neg)])
print("|token diff| distribution:", collections.Counter(np.abs(dtok).tolist()).most_common(8), flush=True)
gl = np.array([nt[a] for a in D.gold])
print("gold tokens p0/5/25/50/75/95/100:", np.percentile(gl, [0, 5, 25, 50, 75, 95, 100]).round(0).tolist(), "| hist (10-token bins):",
      dict(sorted(collections.Counter((gl // 10 * 10).tolist()).items())), flush=True)
print("|d| bins (nats) by sign:", pd.crosstab(D.bin, D.ds).to_dict(), flush=True)

# ---------------------------------------------------------------- audit, held out by video
from sklearn.feature_extraction.text import TfidfVectorizer; from sklearn.linear_model import LogisticRegression
te = np.array([int(hashlib.md5(str(v).encode()).hexdigest(), 16) % 5 == 0 for v in D.video]); tr = ~te
y = D.goldA.to_numpy(); oA = np.where(y, D.gold, D.neg); oB = np.where(y, D.neg, D.gold)
res = {}
res["option letter prior"] = np.mean(y[te] == (y[tr].mean() >= 0.5))
lA = np.array([lp[a] for a in oA]); lB = np.array([lp[b] for b in oB])
res["LM prior: more probable option"] = np.mean((lA > lB)[te] == y[te])
res["LM prior: less probable option"] = np.mean((lA < lB)[te] == y[te])
res["longer option (tokens, ties->A)"] = np.mean((np.array([nt[a] for a in oA]) >= np.array([nt[b] for b in oB]))[te] == y[te])
vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b").fit(list(oA[tr]) + list(oB[tr]))
Xd = vec.transform(oA) - vec.transform(oB)
pred = LogisticRegression(max_iter=2000).fit(Xd[tr], y[tr]).predict(Xd); res["caption text LR (A-B tf-idf 1-2gram)"] = np.mean(pred[te] == y[te])
F = np.stack([lA - lB, np.abs(lA - lB), (lA - lB) * np.abs(lA - lB), np.array([nt[a] - nt[b] for a, b in zip(oA, oB)]),
              np.array([len(a) - len(b) for a, b in zip(oA, oB)])], 1)
F = (F - F[tr].mean(0)) / (F[tr].std(0) + 1e-9)
res["LR on LM-prior diff, |diff|, lengths"] = np.mean(LogisticRegression(max_iter=1000).fit(F[tr], y[tr]).predict(F[te]) == y[te])
print(f"\nAUDIT (held out by video: test {te.sum()} / {len(D)}; chance 50.0; flag > 55.0)", flush=True)
for k, v in res.items(): print(f"  {k:44s} {100 * v:5.1f}{'   <-- FLAG' if v > 0.55 else ''}", flush=True)
for t in sorted(D.typ.unique()):
    m = (D.typ == t).to_numpy() & te
    print(f"  per type {t:9s} test {m.sum():5d}: text LR {100 * np.mean(pred[m] == y[m]):5.1f}  LM-more-probable {100 * np.mean((lA > lB)[m] == y[m]):5.1f}", flush=True)

# ---------------------------------------------------------------- write
ROOT = {"plm": f"{R}/prep/sav_384_24fps", "oops": f"{R}/data/VideoChat-Flash/x/vidln_oops", "vidln": "/", "kin": f"{R}/data/opendl/kinetics700/train",
        "anet": f"{R}/data/VideoChat-Flash/x/timeit_ANet", "charades": f"{R}/prep/charades_384_15fps"}
GRP = {"plm": "plm", "plmd": "plm", "oops": "oops", "uvo": "vidln", "ovis": "vidln", "kin": "kin", "anet": "anet", "charades": "charades"}
D["grp"] = D.src.map(GRP)
D.drop(columns=["sig"]).to_parquet(f"{R}/logs/zz_lcap_pairs.parquet" if O.endswith("/v20") else f"{O}/pairs.parquet")
roots = {}
for grp, X in D.groupby("grp"):
    vid = [("/" + v) if grp == "vidln" and not v.startswith("/") else v for v in X.video]
    out = pd.DataFrame(dict(video=vid, prep_exists=True, root2=None, tq=None, aug="", t0=X.t0.astype(float).values, t1=X.t1.astype(float).values,
                            conversations=X.conversations.values))
    if grp == "plm":
        e = [json.loads(x) for x in X.extra]; out["crop_cx"] = [x["cx"] for x in e]; out["crop_cy"] = [x["cy"] for x in e]
    name = f"v20_lcap__{grp}"; report(name, out.to_dict("records"))
    out.to_parquet(f"{O}/{name}.parquet"); roots[name] = [ROOT[grp], len(out)]
json.dump(roots, open(f"{O}/roots.json", "w"), indent=1)
print("written:", roots, flush=True)
for t in sorted(D.typ.unique()):
    r = D[D.typ == t].sample(1, random_state=5).iloc[0]
    print(f"\n[{t}/{r['sub']} {r.dirn}] {r.src} {r.video} d={r.d:+.2f}\n  GOLD {r.gold}\n  NEG  {r.neg}", flush=True)
