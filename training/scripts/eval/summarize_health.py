"""Summarise jepa_health.py outputs: pairwise surprise accuracy per source (impossible > possible), or native L1.
usage: summarize_health.py "<glob of jsonl>" [more globs ...]   (one line per glob)"""
import collections
import glob
import json
import sys
from math import comb

import numpy as np


def binom_p(w, n):
    return sum(comb(n, i) for i in range(w, n + 1)) / 2 ** n


for pat in sys.argv[1:]:
    rows = [json.loads(l) for f in sorted(glob.glob(pat)) for l in open(f)]
    if not rows:
        print(pat, "no rows"); continue
    if "native_l1" in rows[0]:
        x = np.array([r["native_l1"] for r in rows])
        print(f"{pat}: native self-consistency L1 {x.mean():.4f} +- {x.std() / np.sqrt(len(x)):.4f} (n={len(x)})"); continue
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["source"], r["group"])][r["possible"]] = r
    out = []
    for src in ("intphys", "grasp", "inflevel", "all"):
        pairs = [v for (s, g), v in by.items() if len(v) == 2 and (src == "all" or s == src)]
        if not pairs:
            continue
        res = []
        for key in ("pred", "pred_top10", "copy"):
            w = sum(max(v[False][key]) > max(v[True][key]) for v in pairs)
            res.append(f"{key} {w}/{len(pairs)}={w / len(pairs):.3f} (p {binom_p(w, len(pairs)):.3g})")
        out.append(f"  {src:8s} " + " | ".join(res))
    print(pat); print("\n".join(out))
