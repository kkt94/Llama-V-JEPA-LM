"""(10-26) new opendl3 sets (zz_qgen_v13.py, zz_qgen_phys.py): (1) yes / no kinds balanced to 50 / 50 by dropping the majority (seeded),
(2) text-only shortcut audit on held-out VIDEOS: yes/no - majority answer of the same question text in the train split; multiple choice -
pick the option most often gold (vs distractor) in the train split. Chance = 50 % / 1 / #options.  usage: zz_balance_audit_v13.py"""
import json, glob, os, re, collections, hashlib, numpy as np, pandas as pd
O = "/data/llama_vjepa_lm/manifests/stage1/opendl3"; rng = np.random.default_rng(7)
def parse(c):
    c = json.loads(c); u = c[0]["content"].replace("<|jepa_evidence|>\n", ""); a = c[1]["content"].strip()
    opts = re.findall(r"^([A-H])\. (.+)$", u, flags=re.M); q = u.split("\n")[0]
    if not opts: return q, None, a.rstrip(".").lower()                     # open yes / no
    gold = dict(opts)[a[0]]; return q, [o for _, o in opts], gold
kinds = sorted(os.path.basename(p)[:-8] for p in glob.glob(f"{O}/*.parquet") if os.path.basename(p).startswith("rvqa_"))
for k in kinds:
    d = pd.read_parquet(f"{O}/{k}.parquet"); P = [parse(c) for c in d.conversations]
    if any(o is None and g not in ("yes", "no") for _, o, g in P): continue
    yn = all((o is None and g in ("yes", "no")) or (o is not None and sorted(o) == ["no", "yes"]) for _, o, g in P)
    if yn:
        lab = np.array([g for _, _, g in P]); iy, ino = np.nonzero(lab == "yes")[0], np.nonzero(lab == "no")[0]; m = min(len(iy), len(ino))
        keep = np.sort(np.concatenate([rng.choice(iy, m, replace=False), rng.choice(ino, m, replace=False)]))
        if len(keep) < len(d): d = d.iloc[keep].reset_index(drop=True); d.to_parquet(f"{O}/{k}.parquet"); P = [P[i] for i in keep]
    te = np.array([int(hashlib.md5(v.encode()).hexdigest(), 16) % 4 == 0 for v in d.video])
    if yn:
        tr = collections.defaultdict(collections.Counter)
        for (q, _, g), t in zip(P, te):
            if not t: tr[q][g] += 1
        hit = [ (tr[q].most_common(1)[0][0] if tr[q] else "yes") == g for (q, _, g), t in zip(P, te) if t]
        print(f"{k:15s} rows {len(d):6d} (yes/no balanced)  text-only held-out {100*np.mean(hit):5.1f}%  (chance 50.0)")
    else:
        gc, dc = collections.Counter(), collections.Counter()
        for (q, o, g), t in zip(P, te):
            if not t: gc[g] += 1; dc.update(x for x in o if x != g)
        sc = lambda x: np.log((gc[x] + 1) / (dc[x] + 1))
        hit = [max(o, key=sc) == g for (q, o, g), t in zip(P, te) if t]; ch = np.mean([1 / len(o) for (q, o, g), t in zip(P, te) if t])
        print(f"{k:15s} rows {len(d):6d}                    option-prior held-out {100*np.mean(hit):5.1f}%  (chance {100*ch:.1f})")
