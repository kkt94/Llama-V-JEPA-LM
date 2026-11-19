"""(10-26) text-only shortcut audit of the v16 format sets (logs/zz_qgen_v16.py), held out by VIDEO (80 / 20): (a) option prior - pick the
offered option most often gold in train; (b) question text -> gold option: logistic regression on the question words (instruction /
object / action names included), restricted to the offered options; (c) clip length (20 quantile bins) -> majority gold. Chance = mean 1 / #options.  usage: zz_audit_v16.py"""
import json, glob, os, re, hashlib, collections, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer; from sklearn.linear_model import LogisticRegression
O = "/data/llama_vjepa_lm/manifests/stage1/opendl3/v16"
def parse(c):
    c = json.loads(c); u = c[0]["content"].replace("<|jepa_evidence|>\n", ""); a = c[1]["content"].strip(); opts = re.findall(r"^([A-H])\. (.+)$", u, flags=re.M)
    return u.split("\n")[0], [o for _, o in opts], dict(opts)[a[0]]
for p in sorted(glob.glob(f"{O}/*.parquet")):
    d = pd.read_parquet(p); P = [parse(c) for c in d.conversations]
    vid = [json.loads(t)["segs"][0][0] if isinstance(t, str) and '"seq"' in t else v for v, t in zip(d.video, d.tq)]
    te = np.array([int(hashlib.md5(v.encode()).hexdigest(), 16) % 5 == 0 for v in vid]); tr = ~te
    cnt_g = collections.Counter(P[i][2] for i in np.nonzero(tr)[0]); cnt_o = collections.Counter(o for i in np.nonzero(tr)[0] for o in P[i][1])
    pri = np.mean([max(P[i][1], key=lambda o: cnt_g[o] / max(1, cnt_o[o]) + 1e-6 * (hash(o) % 997)) == P[i][2] for i in np.nonzero(te)[0]])
    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2)); X = vec.fit_transform([P[i][0] for i in range(len(P))]); y = np.array([P[i][2] for i in range(len(P))])
    clf = LogisticRegression(max_iter=300, C=1.0).fit(X[tr], y[tr]); pr = clf.predict_proba(X[te]); cl = {c: j for j, c in enumerate(clf.classes_)}
    txt = np.mean([max(P[i][1], key=lambda o: pr[n, cl[o]] if o in cl else -1) == P[i][2] for n, i in enumerate(np.nonzero(te)[0])])
    Lc = (d.t1 - d.t0).to_numpy(); qb = np.quantile(Lc[tr], np.linspace(0, 1, 21)[1:-1]); b_ = np.searchsorted(qb, Lc)   # (c) clip length -> answer
    bc = collections.defaultdict(collections.Counter)
    for i in np.nonzero(tr)[0]: bc[b_[i]][P[i][2]] += 1
    dur = np.mean([max(P[i][1], key=lambda o: bc[b_[i]][o] + 1e-6 * (hash(o) % 997)) == P[i][2] for i in np.nonzero(te)[0]])
    ch = np.mean([1 / len(P[i][1]) for i in np.nonzero(te)[0]])
    print(f"{os.path.basename(p)[:-8]:34s} n {len(d):6d} test {te.sum():5d} chance {100*ch:5.1f}  option-prior {100*pri:5.1f}  question-text {100*txt:5.1f}  clip-length {100*dur:5.1f}", flush=True)
