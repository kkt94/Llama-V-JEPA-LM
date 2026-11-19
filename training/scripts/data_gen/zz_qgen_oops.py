"""Oops "unexpected part" localisation (weak: TVBench unexpected_action 27). Human labels only: Oops (Epstein et al. 2020)
annotations/transition_times.json = the moment the action turns unintentional, marked by 3 annotators (-1 = not found).
 onset  tau = median of the found times; kept when >= 2 annotators found it and they agree within 1.0 s
 rows   a window of W seconds (W ~ U[3, 8], the SAME distribution for every answer) placed so that tau falls at a chosen relative position
        beginning u in [0.10, 0.25], middle [0.42, 0.58], end [0.75, 0.90]; every feasible position of every video (up to 3 rows / video)
        composed "seq" rows with one segment: frame times run from 0 over the window (a plain window passes the video's absolute seconds,
        and the window start then tells the answer: 'beginning' windows start later for the same tau)
 forms  loc3  TVBench wordings ("Locate the unexpected part of the video.") + beginning / middle / end ("throughout" cannot be defined by one
              onset; a never-correct option would itself be a prior shortcut)
        half  "In which half of the video does the action go wrong?" (u in [0.10, 0.40] or [0.60, 0.90]; separate draw)
 balance per form: answers equal within each window-length bin (10 bins) -> the clip length cannot tell the answer.
Videos used by any evaluation manifest (TemporalBench Oops clips) were never extracted (logs/zz_oops_extract.py).
-> manifests/stage1/opendl3/v17/v17_oops_{loc3,half}.parquet + roots.json entries (prep root "/", jitter 0)"""
import json, os, glob, collections, numpy as np, pandas as pd, decord
import sys; sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
R = "/data/llama_vjepa_lm"; O = f"{R}/manifests/stage1/opendl3/v17"; os.makedirs(O, exist_ok=True); g = Gen(1027)
P = f"{R}/prep/oops_384_15fps"; tt = json.load(open(f"{R}/data/opendl/oops_hf_x/ann/annotations/transition_times.json"))
LO = {"beginning": "At the beginning of the video.", "middle": "In the middle of the video.", "end": "At the end of the video."}
U3 = {"beginning": (0.10, 0.25), "middle": (0.42, 0.58), "end": (0.75, 0.90)}
U2 = {"first": (0.10, 0.40), "second": (0.60, 0.90)}; HO = {"first": "In the first half.", "second": "In the second half."}
Q3 = ["Locate the unexpected part of the video.", "Locate the accidental part of the video.", "Locate the part of the video where something goes wrong.",
      "When does the mishap happen in the video?", "At what point in the video does the action fail?", "Where in the video does the unintended event occur?"]
Q2 = ["In which half of the video does the action go wrong?", "Does the failure happen in the first or the second half of the video?",
      "When does the unexpected event occur?"]
def dur(p):
    try: vr = decord.VideoReader(p, num_threads=1); return len(vr) / float(vr.get_avg_fps())
    except Exception: return None
V = []
for k, a in tt.items():
    p = f"{P}/{k}.mp4"
    if not os.path.exists(p): continue
    f = [t for t in a["t"] if t >= 0]
    if len(f) < 2 or max(f) - min(f) > 1.0: continue
    V.append((p, float(np.median(f))))
print("videos with an agreed onset and a transcoded file", len(V), flush=True)
from multiprocessing import Pool
with Pool(32) as pool: D = pool.map(dur, [p for p, _ in V], chunksize=64)
cand = {"loc3": collections.defaultdict(list), "half": collections.defaultdict(list)}
for (p, tau), L in zip(V, D):
    if not L or tau >= L: continue
    for form, U in (("loc3", U3), ("half", U2)):
        for lab, (lo, hi) in U.items():
            for _ in range(10):
                W = float(g.rng.uniform(3.0, 8.0)); u = float(g.rng.uniform(lo, hi)); t0 = tau - u * W
                if t0 >= 0 and t0 + W <= L - 0.05: cand[form][lab].append((p, t0, W)); break
for form, C in cand.items():
    print(form, "feasible", {k: len(v) for k, v in C.items()}, flush=True)
    bins = np.linspace(3.0, 8.0, 11); by = collections.defaultdict(lambda: collections.defaultdict(list))
    for lab, X in C.items():
        for x in X: by[int(np.clip(np.digitize(x[2], bins) - 1, 0, 9))][lab].append(x)
    rows = []
    for b, G in by.items():
        if len(G) < len(C): continue
        m = min(len(v) for v in G.values())
        for lab, X in G.items():
            for j in g.rng.permutation(len(X))[:m]:
                p, t0, W = X[int(j)]
                base = dict(video=p, prep_exists=True, root2=None, tq=json.dumps({"kind": "seq", "segs": [[p, round(t0, 3), round(t0 + W, 3)]]}), aug="", t0=0.0, t1=W)
                if form == "loc3": opts = [LO[lab]] + [LO[o] for o in LO if o != lab]; q = g.pick(Q3)
                else: opts = [HO[lab]] + [HO[o] for o in HO if o != lab]; q = g.pick(Q2)
                r = dict(base); r["conversations"] = g.render(q, *g.shuffle(opts, 0)); r["_key"] = lab; rows.append(r)
    name = f"v17_oops_{form}"; d = pd.DataFrame(rows); report(name, rows)
    print(name, "answers", dict(collections.Counter(d._key)), "W mean by answer", d.groupby("_key").t1.mean().round(2).to_dict(), flush=True)
    d.to_parquet(f"{O}/{name}.parquet")
    rp = f"{O}/roots.json"; RT = json.load(open(rp)) if os.path.exists(rp) else {}; RT[name] = ["/", 0.0]; json.dump(RT, open(rp + ".tmp", "w"), indent=1); os.replace(rp + ".tmp", rp)
print("OOPS QGEN DONE")
