"""one EgoDex set from all splits (test, part1-5, extra)
for nx13: the MEASURED-dynamics questions first, task naming (appearance-led, 101 classes) only as a small share:
 hdir (hand direction on screen, measured wrist)  180k, answers balanced    hmore (which hand moves more, 3D wrist path) 100k, balanced
 turn (camera wearer's yaw)                        all                       task + "what am I doing" (task naming)          50k, balanced by task
Within a kind rows are taken across as many different videos as possible (each video's 1st row first, then 2nd, ...).
-> manifests/stage1/opendl3/egodex_sel.parquet (prep root prep/egodex_384_15fps)   usage: zz_egodex_select.py"""
import glob, json, re, collections, numpy as np, pandas as pd
O = "/data/llama_vjepa_lm/manifests/stage1/opendl3"; rng = np.random.default_rng(1026)
def kind(q):
    q = q.lower()
    if "which task" in q or "what task" in q or "what am i doing" in q or "which activity" in q: return "task"
    if "move more" in q or "moves more" in q: return "hmore"
    if "turn" in q or "facing" in q or "viewing direction" in q: return "turn"
    if "hand" in q: return "hdir"
    return "other"
def gold(c):
    c = json.loads(c); u = c[0]["content"]; a = c[1]["content"].strip(); o = dict(re.findall(r"^([A-H])\. (.+)$", u, flags=re.M))
    return (o.get(a[:1]) if a[:1] in o and len(a) <= 2 else a.rstrip(".")).lower(), u.replace("<|jepa_evidence|>\n", "").split("\n")[0]
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{O}/egodex_*.parquet")) if not f.endswith("_sel.parquet")], ignore_index=True)
G = [gold(c) for c in d.conversations]; d["_ans"] = [g[0] for g in G]; d["_k"] = [kind(g[1]) for g in G]
print(len(d), "rows", d._k.value_counts().to_dict())
# task naming: the answer is the task name -> balance by task; the other kinds: by answer
d["_key"] = d._ans
d = d.iloc[rng.permutation(len(d))].reset_index(drop=True)
d["_vrank"] = d.groupby(["_k", "video"]).cumcount()                   # 0 = the video's first row of this kind
CAP = {"hdir": 180000, "hmore": 100000, "turn": 10 ** 9, "task": 50000}
out = []
for k, cap in CAP.items():
    x = d[d._k == k].sort_values("_vrank", kind="stable"); keys = x._key.value_counts()
    per = cap // max(len(keys), 1) if cap < 10 ** 9 else 10 ** 9
    if k != "turn":     # balanced: the same number per answer (all of the rarer answers when below the share), spread over videos
        s = x.groupby("_key", group_keys=False).head(per)
        if len(s) < cap: s = pd.concat([s, x.drop(s.index).head(cap - len(s))])        # fill up, still most-distinct-videos first
        x = s.head(cap)
    out.append(x); print(f"{k:6s} kept {len(x):6d} of {int((d._k == k).sum()):6d}  videos {x.video.nunique():6d}  answers {x._key.value_counts().head(6).to_dict()}")
s = pd.concat(out).drop(columns=["_ans", "_k", "_key", "_vrank"]).sample(frac=1.0, random_state=0).reset_index(drop=True)
s.to_parquet(f"{O}/egodex_sel.parquet"); print("egodex_sel", len(s), "rows, videos", s.video.nunique())
