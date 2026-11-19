"""(10-27, user: "filter repetitive or unfinished captions") nx13 generation probe: video descriptions loop in 46 % of answers at 8k
(nx12 25 %). Scan every assistant turn of every nx13 source; a turn of >= 25 words is flagged when
 repeat   share of repeated word 4-grams > 0.3 (the same measure as logs/zz_gen_probe.py)
 cut      it stops mid-sentence: ends with , ; : - or its last word is a function word (and, the, of, with, ...). A complete sentence that only
          lacks the final period (common in human labels: "put strawberries into the juicer") is NOT flagged (first scan flagged those)
A row with any flagged turn is dropped. Sources with flagged rows get a filtered copy in manifests/stage1/clean1027/<name>.parquet
(same columns); the others are untouched.  usage: zz_capfilter.py <config name>   -> logs/zz_capfilter.out, clean1027/flags.json"""
import sys, os, json, yaml, numpy as np, pandas as pd
from multiprocessing import Pool
R = "/data/llama_vjepa_lm"; OUT = f"{R}/manifests/stage1/clean1027"; os.makedirs(OUT, exist_ok=True)
FUNC = set("a an the and or but of to in on at with from for by as into onto than then while is are was were be his her their its this that which who".split())
def flag(conv):
    rep = cut = False
    for m in json.loads(conv):
        if m.get("role") != "assistant": continue
        w = m["content"].split()
        if len(w) < 25: continue
        g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
        if 1 - len(set(g)) / len(g) > 0.3: rep = True
        if m["content"].rstrip()[-1] in ",;:-" or w[-1].lower().strip("\"'") in FUNC: cut = True
    return rep, cut
def flagged_turns(conv, kind):
    out = []
    for m in json.loads(conv):
        if m.get("role") == "assistant" and flag(json.dumps([m]))[kind]: out.append(m["content"])
    return out
def scan(s):
    d = pd.read_parquet(s["manifest"]); F = np.array([flag(c) for c in d.conversations], dtype=bool).reshape(-1, 2)
    bad = F.any(1); res = {"name": s["name"], "manifest": s["manifest"], "n": len(d), "repeat": int(F[:, 0].sum()), "cut": int(F[:, 1].sum()), "drop": int(bad.sum())}
    if bad.any():
        p = f"{OUT}/{s['name']}.parquet"; d[~bad].reset_index(drop=True).to_parquet(p + ".part"); os.replace(p + ".part", p); res["clean"] = p
        res["ex_repeat"] = [flagged_turns(c, 0)[0][:300] for c in d.conversations[F[:, 0]].sample(min(3, int(F[:, 0].sum())), random_state=0)]
        res["ex_cut"] = [flagged_turns(c, 1)[0][-200:] for c in d.conversations[F[:, 1] & ~F[:, 0]].sample(min(3, int((F[:, 1] & ~F[:, 0]).sum())), random_state=0)]
    return res
if __name__ == "__main__":
    c = yaml.safe_load(open(f"{R}/training/configs/single/{sys.argv[1]}.yaml")); S = c["data"]["sources"]
    with Pool(40) as P: out = P.map(scan, sorted(S, key=lambda s: -os.path.getsize(s["manifest"])), chunksize=1)
    json.dump(out, open(f"{OUT}/flags.json", "w"), indent=1)
    for r in sorted(out, key=lambda r: -r["drop"]):
        if r["drop"]: print(f"{r['name']:28s} rows {r['n']:8d} drop {r['drop']:7d} ({100 * r['drop'] / r['n']:5.2f} %)  repeat {r['repeat']:6d} cut {r['cut']:6d}")
    print("clean sources", sum(r["drop"] == 0 for r in out), "of", len(out)); print("CAPFILTER DONE")
