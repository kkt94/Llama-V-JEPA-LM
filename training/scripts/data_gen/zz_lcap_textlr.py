"""(10-28) lcap step 3b (CPU): cross-fitted text-only predictor on the candidate pairs, used as one more mirror-balance dimension in
zz_lcap_build.py (as s_lr in zz_qgen_capbin). Each candidate gets a random orientation (A = gold or A = negative); a logistic regression on
tf-idf(A) - tf-idf(B) (word 1-2-grams) is fit on 4 video folds and predicts the 5th. s_lr = +1 when it favours the gold, -1 otherwise.
-> logs/zz_lcap_slr.parquet (gid, neg, s_lr, p_gold)   usage: python -I zz_lcap_textlr.py"""
import hashlib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer; from sklearn.linear_model import LogisticRegression
R = "/data/llama_vjepa_lm"; rng = np.random.default_rng(7)
G = pd.read_parquet(f"{R}/logs/zz_lcap_golds.parquet").set_index("gid", drop=False); C = pd.read_parquet(f"{R}/logs/zz_lcap_cands.parquet")
C["gold"] = G.cap.reindex(C.gid).values; C["video"] = G.video.reindex(C.gid).values
fold = np.array([int(hashlib.md5(str(v).encode()).hexdigest(), 16) % 5 for v in C.video])
o = rng.random(len(C)) < 0.5; A = np.where(o, C.gold, C.neg); B = np.where(o, C.neg, C.gold)
p_gold = np.zeros(len(C))
for f in range(5):
    tr, te = fold != f, fold == f
    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b").fit(list(A[tr][::2]) + list(B[tr][1::2]))
    Xd = vec.transform(A) - vec.transform(B)
    pA = LogisticRegression(max_iter=2000).fit(Xd[tr], o[tr]).predict_proba(Xd[te])[:, 1]
    p_gold[te] = np.where(o[te], pA, 1 - pA)
    print(f"fold {f}: held-out acc {np.mean((pA > 0.5) == o[te]):.4f}", flush=True)
C["p_gold"] = p_gold; C["s_lr"] = np.where(p_gold > 0.5, 1, -1)
C[["gid", "neg", "s_lr", "p_gold"]].to_parquet(f"{R}/logs/zz_lcap_slr.parquet")
print("text LR favours gold (all candidates):", round(float((C.s_lr > 0).mean()), 4), flush=True)
