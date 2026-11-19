"""(10-28) lcap step 1: the pool of HUMAN-written long captions (gold candidates) for the long-caption single-edit pair set (v20 lcap).
Every caption is a human description of the clip (no model text):
  plm      PLM-Video-Human RCap (SA-V): human region captions of one subject over [t0, t1] (dataset card: "human-annotated");
           rows of manifests/stage1/clean1027/rcap.parquet (already filtered for repeated / cut text) -> video, t0, t1, crop centre
  oops     VidLN narratives (Oops clips)      manifests/stage1/clean1027/vcf_vidln_oops.parquet  (prep_root data/VideoChat-Flash/x/vidln_oops)
  uvo      VidLN narratives (UVO sparse+dense) manifests/stage1/clean1027/vidln_uvo.parquet       (absolute path, prep_root /)
  ovis     VidLN narratives (OVIS, frame dirs) manifests/stage1/vidln_ovis.parquet                (absolute path, prep_root /)
  kin      VidLN narratives (Kinetics)         VCF annotations caption_vidln_kinetics (videos data/opendl/kinetics700/train/<class>/)
  anet     ActivityNet Captions sentences with their human timestamps (VCF timeit_ANet annotation, videos data/VideoChat-Flash/x/timeit_ANet):
           a gold = a run of CONSECUTIVE sentences of one video (start-sorted), the shortest run from sentence i that reaches 50 tokens,
           clip = [first start, last end]; runs of a video do not share sentences
  charades Charades human descriptions (Charades_v1_{train,test}.csv `descriptions`, one gold per description), whole clip
Length rule: gold 50-150 Qwen3 tokens (no other cut). Excluded: any video whose id appears in an eval manifest
(manifests/eval/*.jsonl, eval/pilot/*.jsonl, eval/light/*.jsonl; basename stem, YouTube id, Charades id, SA-V id, Oops name).
-> logs/zz_lcap_golds.parquet   usage: python -I zz_lcap_golds.py > zz_lcap_golds.out"""
import os, re, csv, json, glob, collections
import numpy as np, pandas as pd
from transformers import AutoTokenizer
R = "/data/llama_vjepa_lm"; tok = AutoTokenizer.from_pretrained(f"{R}/models/Qwen3-1.7B")
ntok = lambda xs: [len(x) for x in tok(list(xs), add_special_tokens=False).input_ids]
clean = lambda s: re.sub(r"\s+", " ", str(s)).strip()

# ---------------------------------------------------------------- eval ids for exclusion
EV = []
for p in glob.glob(f"{R}/manifests/eval/*.jsonl") + glob.glob(f"{R}/manifests/eval/pilot/*.jsonl") + glob.glob(f"{R}/manifests/eval/light/*.jsonl"):
    EV.append(open(p, errors="ignore").read())
EV = "\n".join(EV)
TOK = set(re.findall(r"[A-Za-z0-9]+", EV)) | set(re.findall(r"[A-Za-z0-9_]+", EV))      # Charades / SA-V / OVIS ids anywhere in a line
PATHS = set(re.findall(r'"([^"]{1,400}?\.(?:mp4|webm|mkv|avi|mov|m4v|jpg|png))"', EV)) | set(re.findall(r'"video[a-z_]*": "([^"]{1,400})"', EV))
STEMS = {os.path.splitext(os.path.basename(x))[0] for x in PATHS} | {os.path.basename(x) for x in PATHS}
STEMS |= {re.sub(r"(_mute|_start_.*)$", "", x) for x in STEMS}           # TemporalBench clip names: <oops name>_mute, <id>_start_a_end_b
YT = set()                                                              # every 11-char window of every id-like run in a video path
for x in PATHS:
    for t in re.findall(r"[A-Za-z0-9_-]{11,}", x): YT |= {t[i:i + 11] for i in range(len(t) - 10)}
print("eval chars", len(EV), "paths", len(PATHS), "stems", len(STEMS), "yt windows", len(YT), flush=True)
del EV
def in_eval(key, short=False):
    if not key: return False
    if short: return key in TOK
    return key in STEMS or (len(key) == 11 and key in YT) or key in TOK

rows = []
def add(src, video, root, t0, t1, cap, key, extra=None, short=False):
    rows.append(dict(src=src, video=video, root=root, t0=t0, t1=t1, cap=clean(cap), key=key, short=short, extra=json.dumps(extra or {})))

# plm
d = pd.read_parquet(f"{R}/manifests/stage1/clean1027/rcap.parquet")
for v, t0, t1, cx, cy, c in zip(d.video, d.t0, d.t1, d.crop_cx, d.crop_cy, d.conversations):
    add("plm", v, f"{R}/prep/sav_384_24fps", float(t0), float(t1), json.loads(c)[1]["content"], os.path.basename(v)[:-4], dict(cx=float(cx), cy=float(cy)))
# plm dense (RDCap = the same human captions as RCap, every segment of one subject's whole video): a gold = a run of CONSECUTIVE segments
# of one subject (an 'Out of frame.' segment breaks a run), the shortest run from segment i that reaches 50 tokens; runs do not share segments
d = pd.read_parquet(f"{R}/manifests/stage1/clean1027/rdcap.parquet"); seen = set(); PD = 0
for v, cx, cy, c in zip(d.video, d.crop_cx, d.crop_cy, d.conversations):
    segs = [(float(a), float(b), clean(t)) for a, b, t in re.findall(r"From ([\d.]+)s to ([\d.]+)s: (.*)", json.loads(c)[1]["content"])]
    if (v, tuple(segs)) in seen or len(segs) < 2: continue
    seen.add((v, tuple(segs))); PD += 1
    runs, cur = [], []
    for sg in segs:
        if re.fullmatch(r"(?i)out of (the )?frame\.?", sg[2]) or not sg[2]: runs.append(cur); cur = []
        else: cur.append(sg)
    runs.append(cur)
    for run in runs:
        if len(run) < 2: continue
        sents = [x[2] if x[2][-1] in ".!?" else x[2] + "." for x in run]; nt = ntok(sents); i = 0
        while i < len(run):
            j, tot = i, 0
            while j < len(run) and tot < 50: tot += nt[j] + (1 if j > i else 0); j += 1
            if tot < 50: break
            if j - i >= 2: add("plmd", v, f"{R}/prep/sav_384_24fps", run[i][0], run[j - 1][1], " ".join(sents[i:j]), os.path.basename(v)[:-4],
                               dict(cx=float(cx), cy=float(cy), sents=sents[i:j], ts=[[x[0], x[1]] for x in run[i:j]]))
            i = j
print("plm dense subjects", PD, flush=True)
# vidln
for src, m, root in (("oops", "clean1027/vcf_vidln_oops", f"{R}/data/VideoChat-Flash/x/vidln_oops"), ("uvo", "clean1027/vidln_uvo", "/"), ("ovis", "vidln_ovis", "/")):
    d = pd.read_parquet(f"{R}/manifests/stage1/{m}.parquet")
    for v, c in zip(d.video, d.conversations):
        stem = os.path.splitext(os.path.basename(v))[0]
        if src == "uvo": v = v.lstrip("/")
        if src == "ovis": v = v.lstrip("/")
        add(src, v, root, None, None, json.loads(c)[1]["content"], stem, short=(src == "ovis"))
KR = f"{R}/data/opendl/kinetics700/train"
KI = {}                                                                  # YouTube id -> class/file of the local Kinetics-700 train clip
for cls in os.listdir(KR):
    for fn in os.listdir(f"{KR}/{cls}"): KI.setdefault(fn[:11], f"{cls}/{fn}")
for x in json.load(open(f"{R}/data/VideoChat-Flash/annotations/video/caption_vidln_kinetics-vidln-kinetics_train_28k.json")):
    yid = os.path.basename(x["video"])[:11]
    if yid not in KI: continue
    add("kin", KI[yid], KR, None, None, x["conversations"][1]["value"], yid)
# anet
AR = f"{R}/data/VideoChat-Flash/x/timeit_ANet"
AN = 0
for x in json.load(open(f"{R}/data/VideoChat-Flash/annotations/video/timeit_ANet-TimeIT-Activitynet_Captions_11k.json")):
    fn = os.path.basename(x["video"])
    if not os.path.exists(f"{AR}/{fn}"): continue
    ans = x["conversations"][1]["value"]
    segs = [(float(a), float(b), clean(s)) for a, b, s in re.findall(r"([\d.]+) - ([\d.]+) seconds, (.*?)(?=\s+[\d.]+ - [\d.]+ seconds,|$)", ans, flags=re.S)]
    segs = sorted([s for s in segs if s[1] > s[0] and s[2]], key=lambda s: (s[0], s[1]))
    if not segs: continue
    AN += 1
    sents = [s[2] if s[2][-1] in ".!?" else s[2] + "." for s in segs]; sents = [s[0].upper() + s[1:] for s in sents]
    nt = ntok(sents); i = 0
    while i < len(segs):
        j, tot = i, 0
        while j < len(segs) and tot < 50: tot += nt[j] + (1 if j > i else 0); j += 1
        if tot < 50: break
        add("anet", fn, AR, segs[i][0], max(s[1] for s in segs[i:j]), " ".join(sents[i:j]), fn[2:13],
            dict(sents=sents[i:j], ts=[[s[0], s[1]] for s in segs[i:j]]))
        i = j
print("anet videos with captions + video", AN, flush=True)
# charades
CD = f"{R}/data/opendl/charades/Charades"; CP = f"{R}/prep/charades_384_15fps"
for f in ("Charades_v1_train.csv", "Charades_v1_test.csv"):
    for r in csv.DictReader(open(f"{CD}/{f}")):
        if not os.path.exists(f"{CP}/{r['id']}.mp4"): continue
        for k, ds in enumerate(x for x in r["descriptions"].split(";") if x.strip()):
            add("charades", f"{r['id']}.mp4", CP, None, None, ds, r["id"], dict(k=k), short=True)

G = pd.DataFrame(rows)
print("raw pool", G.groupby("src").size().to_dict(), flush=True)
G = G.drop_duplicates(["src", "video", "cap"]).reset_index(drop=True)
G["ntok"] = ntok(G.cap)
keys = G[["key", "short"]].drop_duplicates()
ex = {k for k, s in zip(keys.key, keys.short) if in_eval(k, s)}
G["in_eval"] = G.key.isin(ex)
for s, x in G.groupby("src"):
    q = np.percentile(x.ntok, [5, 25, 50, 75, 95])
    print(f"{s:9s} caps {len(x):7d} videos {x.video.nunique():6d} | tokens p5/25/50/75/95 {q.round(0).tolist()} | 50-150: {int(x.ntok.between(50, 150).sum()):6d} "
          f"| in eval: {int(x.in_eval.sum()):5d} videos {x[x.in_eval].video.nunique()}", flush=True)
G = G[G.ntok.between(50, 150) & ~G.in_eval].reset_index(drop=True)
G["gid"] = np.arange(len(G))
G.to_parquet(f"{R}/logs/zz_lcap_golds.parquet")
print("kept golds", len(G), G.groupby("src").size().to_dict(), flush=True)
for s, x in G.groupby("src"): print(f"  [{s}] {x.sample(1, random_state=0).cap.iloc[0]}", flush=True)
