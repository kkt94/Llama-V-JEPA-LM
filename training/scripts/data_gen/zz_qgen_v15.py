"""one human label, many question / answer forms, over the
label sources that so far had ONE form (4-way multiple choice): Kinetics-700 classes (web human actions), Something-Something v2 filled
templates (hand-object), SSv2 objects, Language-Table instructions (real-robot tabletop; crowd-written, with a hard negative instruction).
 open      short free answer = the label ("Answer in a few words.")                                  k700, ssv2_obj, language_table (ssv2 has one)
 yn        "does the video show 'X'?" yes / no 50 / 50; the 'no' label drawn by label frequency (or the row's hard negative instruction)
 mcn       2 / 3 / 5 / 6 options: the row's own hard distractors first, then frequency-drawn labels
 bincap    "which caption describes the video better?" gold vs one of the row's hard distractors (TemporalBench-binary form)   ssv2, language_table
 whichseg  two 10 s Kinetics clips joined by 2 s of black: "which segment shows 'X'?"  (first / second, balanced)
 half      two 10 s clips: "what happens in the second part / first part?" - the other part's class is always among the options
 ndistinct four 2.5 s segments covering 1-4 distinct Kinetics classes (repeats = other clips of the same class): "how many different actions?"
           (total length always 10 s; answers 1-4 uniform)
-> manifests/stage1/opendl3/v15_<format>__<source>.parquet (prep root of the source; composed rows "seq" with absolute paths, root "/")
usage: zz_qgen_v15.py"""
import sys, os, json, re, collections, yaml
import numpy as np, pandas as pd
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
R = "/data/llama_vjepa_lm"; O3 = f"{R}/manifests/stage1/opendl3"; g = Gen(109); N = 12000; BLACK = [f"{R}/prep/synthetic/black_2s.mp4", 0.0, 2.0]
cfg = yaml.safe_load(open(f"{R}/training/configs/single/nx13_Q.yaml")); S = {s["name"]: s for s in cfg["data"]["sources"]}
def parse(c):
    c = json.loads(c); u = c[0]["content"]; opts = re.findall(r"^([A-H])\. (.+)$", u, flags=re.M); a = c[1]["content"].strip()[:1]
    d = dict(opts); return (d.get(a), [o for k, o in opts if k != a]) if a in d else (None, [])
SRC = {}
for name, kinds in (("k700_act", None), ("ssv2", "mc"), ("ssv2_obj", None), ("language_table", None)):
    d = pd.read_parquet(S[name]["manifest"]); d = d[d.prep_exists.astype(str).isin(["True", "true", "1"])]
    if kinds and "kind" in d: d = d[d.kind == kinds]
    d = d.drop_duplicates("video").sample(min(len(d), 80000), random_state=0)
    P = [parse(c) for c in d.conversations]; d = d.assign(label=[p[0] for p in P], hard=[p[1] for p in P]); d = d[d.label.notna()].reset_index(drop=True)
    SRC[name] = d; print(name, len(d), "labelled clips", flush=True)
out = collections.defaultdict(list)
def add(kind, src, video, t0, t1, conv, key, tq=None):
    out[f"v15_{kind}__{src}"].append(dict(video=video, prep_exists=True, root2=None, tq=tq, aug="", t0=float(t0), t1=float(t1), conversations=conv, _key=key))
def mc(kind, src, video, t0, t1, q, opts, key, tq=None): add(kind, src, video, t0, t1, g.row(video, t0, t1, q, opts, 0)["conversations"], key, tq)
def yn(kind, src, video, t0, t1, q, yes, tq=None):
    if g.rng.random() < 0.5: mc(kind, src, video, t0, t1, q, ["yes", "no"] if yes else ["no", "yes"], "yes" if yes else "no", tq)
    else: add(kind, src, video, t0, t1, json.dumps([{"role": "user", "content": f"<|jepa_evidence|>\n{q}\nAnswer yes or no."}, {"role": "assistant", "content": "Yes." if yes else "No."}]), "yes" if yes else "no", tq)
def openq(src, video, t0, t1, q, ans): add("open", src, video, t0, t1, json.dumps([{"role": "user", "content": f"<|jepa_evidence|>\n{q}\nAnswer in a few words."}, {"role": "assistant", "content": ans[0].upper() + ans[1:] + ("" if ans.endswith(".") else ".")}]), "open")
QY = {"k700_act": ["Is the person {}?", "Does the video show someone {}?", "Is this a video of '{}'?"], "ssv2": ["Does the video show '{}'?", "Is the action '{}' performed in this video?"],
      "ssv2_obj": ["Is the {} being manipulated in this video?", "Does the person handle a {}?"], "language_table": ["Is the robot carrying out the instruction '{}'?", "Does the robot {}?"]}
QM = {"k700_act": ["What is the person doing?", "Which activity is shown?", "What activity does the video show?"], "ssv2": ["Which action is shown in the video?", "What does the person do?"],
      "ssv2_obj": ["What object is being manipulated?", "Which object does the person handle?"], "language_table": ["Which robot instruction best describes the actions in the video?", "What is the robot instructed to do?"]}
QO = {"k700_act": "What is the person doing?", "ssv2_obj": "Which object is being manipulated?", "language_table": "What instruction is the robot carrying out?"}
for name, d in SRC.items():
    vid = lambda r: r.video; t0 = lambda r: float(getattr(r, "t0", 0.0) or 0.0) if "t0" in d else 0.0; t1 = lambda r: float(getattr(r, "t1", 0.0) or 0.0) if "t1" in d else 0.0
    freq = d.label.value_counts(); labs = freq.index.to_numpy(); p = (freq / freq.sum()).to_numpy()
    def draw(excl, k):
        outl = []
        while len(outl) < k:
            x = labs[int(g.rng.choice(len(labs), p=p))]
            if x not in excl and x not in outl: outl.append(x)
        return outl
    idx = g.rng.permutation(len(d))
    for j, i in enumerate(idx[: 4 * N]):
        r = d.iloc[int(i)]; lab = r.label; f = j % 4
        if f == 0 and name in QO: openq(name, vid(r), t0(r), t1(r), QO[name], lab.lower() if name != "language_table" else lab)
        elif f == 1:
            yes = g.rng.random() < 0.5
            neg = (str(r.neg_text) if name == "language_table" and isinstance(getattr(r, "neg_text", None), str) and g.rng.random() < 0.5 else draw({lab}, 1)[0])
            lab_ = lab.lower() if name == "k700_act" else lab
            yn("yn", name, vid(r), t0(r), t1(r), g.pick(QY[name]).format(lab_ if yes else (neg.lower() if name == "k700_act" else neg)), yes)
        elif f == 2:
            k = int(g.pick([2, 3, 5, 6])); hard = [] if name == "ssv2_obj" else [h for h in r.hard if h != lab][: (k - 1) // 2]   # (v2) at most half hard (their
            ds = hard + draw({lab, *hard}, k - 1 - len(hard))                                                                       # own frequencies are a prior)
            mc("mcn", name, vid(r), t0(r), t1(r), g.pick(QM[name]), [lab] + ds, str(k))
        elif f == 3 and name in ("ssv2", "language_table") and r.hard:
            alt = r.hard[int(g.rng.integers(len(r.hard)))] if g.rng.random() < 0.5 else draw({lab}, 1)[0]                # (v2) half hard, half frequency-drawn
            mc("bincap", name, vid(r), t0(r), t1(r), g.pick(["Which caption describes the video better?", "Which description matches the video?"]), [lab, alt], "mc")
# ---- composed Kinetics forms
K = SRC["k700_act"]; root = S["k700_act"]["prep_root"]; ab = lambda v: f"{root}/{v}"; bycls = collections.defaultdict(list)
for v, l in zip(K.video, K.label): bycls[l].append(v)
cls = [c for c, v in bycls.items() if len(v) >= 4]
for _ in range(N):
    a, b = g.rng.choice(len(K), 2, replace=False); ra, rb = K.iloc[int(a)], K.iloc[int(b)]
    if ra.label == rb.label: continue
    segs = [[ab(ra.video), 0.0, 10.0], BLACK, [ab(rb.video), 0.0, 10.0]]; tq = json.dumps({"kind": "seq", "segs": segs})
    if g.rng.random() < 0.5:
        ask_first = g.rng.random() < 0.5; lab = (ra if ask_first else rb).label.lower()
        o = ["The first segment (before the black frames)", "The second segment (after the black frames)"]
        mc("whichseg", "k700_act", ab(ra.video), 0, 22, f'The video has two segments separated by black frames. Which segment shows "{lab}"?', o if ask_first else o[::-1], "first" if ask_first else "second", tq)
    else:
        second = g.rng.random() < 0.5; tgt, oth = (rb, ra) if second else (ra, rb)
        ds = [oth.label] + [c for c in g.rng.choice(cls, 4, replace=False) if c not in (tgt.label, oth.label)][:2]
        mc("half", "k700_act", ab(ra.video), 0, 22, f"What happens in the {'second' if second else 'first'} part of the video (after / before the black frames)?", [tgt.label] + ds, "second" if second else "first", tq)
for _ in range(N):
    k = int(g.rng.integers(1, 5)); cs = list(g.rng.choice(cls, k, replace=False)); seq = cs + [cs[int(g.rng.integers(k))] for _ in range(4 - k)]; g.rng.shuffle(seq)
    used = collections.Counter(); segs = []
    for c in seq:
        v = bycls[c][used[c] % len(bycls[c])]; used[c] += 1; st = float(g.rng.uniform(0, 7.5)); segs.append([ab(v), st, st + 2.5])
    mc("ndistinct", "k700_act", segs[0][0], 0, 10, g.pick(["How many different actions are shown in the video?", "How many distinct activities appear in this video?"]), [str(k)] + [str(x) for x in range(1, 5) if x != k], str(k), json.dumps({"kind": "seq", "segs": segs}))
for kname, rows in sorted(out.items()):
    dd = pd.DataFrame(rows)
    if dd._key.isin(["yes", "no"]).all() or kname.startswith(("v15_whichseg", "v15_half", "v15_ndistinct")):
        m = dd._key.value_counts().min(); dd = pd.concat([gg.sample(m, random_state=0) for _, gg in dd.groupby("_key")]).sort_index()
    dd = dd.drop(columns="_key").reset_index(drop=True); report(kname, dd.to_dict("records")); dd.to_parquet(f"{O3}/{kname}.parquet")
