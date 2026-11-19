"""the benchmark question / option
forms that our mix lacked, built from labels / measurements / transformations we already have (no new answers are invented):
 nav_vln    MVBench egocentric_navigation form: 'This is a navigation video of an agent following instruction: "..." What is the next action
            it should take?'  Move forward / Turn left and move forward / Turn right and move forward / Stop.  EPIC Fields camera poses
            (measured): the clip = 4 s before t, the answer = the camera motion in [t, t + 2.5 s] (yaw > 20 deg right / < -20 left;
            |yaw| < 6 and forward travel > the video's median travel -> forward; |yaw| < 4 and travel < 0.3 x median -> stop); the
            "instruction" = the human EPIC narrations that start in [t, t + 12 s]. Classes balanced.
 loc_tv     TVBench action_localization wordings ("when does the action 'X' take place?") over the loc_nat / loc_comp rows (same answers)
 cam_mb     MotionBench camera forms ("Does the camera move / tilt / zoom at any point during the video?" -> "Yes, it tilts upward" /
            "No, it remains static"; "How does the camera move as the scene unfolds?") over camera_clevrer / camera_chexo (known
            synthetic moves) and egocam_mc_epic (measured EPIC yaw)
 tempo      TempCompass playback-speed form ("What is the tempo of the video playback?" slow / medium / fast; slow motion / standard speed /
            high speed; slower / equal / faster than real time): the same 32 frames over 2.5-3.5 s (slow), 8-12 s (standard), 32-48 s
            (fast) of continuous EPIC footage, as "rate" rows (same nominal frame times for all: the bridge sees no real seconds)
 rspeed_w   TempCompass compare wordings ("What is the speed of the X above?" -> "moving faster than the X below") over tq4_rspeed rows
            (answer from the two playback spans of the stacked video)
 rep6       TOMATO form: 5 / 6 numeric options over the rep_loop rows (counts 2-7, every value that occurs as an answer)
-> manifests/stage1/opendl3/v16/<name>.parquet + roots.json {name: [prep_root, jitter]}   usage: zz_qgen_v16.py"""
import sys, os, re, json, glob, csv, collections, yaml
import numpy as np, pandas as pd
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
R = "/data/llama_vjepa_lm"; O = f"{R}/manifests/stage1/opendl3/v16"; os.makedirs(O, exist_ok=True); g = Gen(116)
cfg = yaml.safe_load(open(f"{R}/training/configs/single/nx13_Q.yaml")); S = {s["name"]: s for s in cfg["data"]["sources"]}
BLACK = None; out = collections.defaultdict(list); ROOTS = {}
def gold_q(c):
    c = json.loads(c); u = c[0]["content"]; opts = dict(re.findall(r"^([A-H])\. (.+)$", u, flags=re.M)); q = u.replace("<|jepa_evidence|>", "").strip().split("\n")[0]
    return q, opts.get(c[1]["content"].strip()[:1])
def put(name, base, q, opts, k, key, root, jitter):
    r = dict(base); r["conversations"] = g.render(q, *g.shuffle(opts, k)); r["_key"] = key; out[name].append(r); ROOTS[name] = [root, jitter]
def take(name, n):
    d = pd.read_parquet(S[name]["manifest"]); d = d[d.prep_exists.astype(str).isin(["True", "true", "1"])]
    return d.sample(min(n, len(d)), random_state=0).to_dict("records")
# ---- nav_vln (EPIC Fields)
FPS = {r["video_id"]: (float(r["fps"]), float(r["duration"])) for r in csv.DictReader(open(f"{R}/data/opendl/epic_kitchens_100/annotations/EPIC_100_video_info.csv"))}
A = f"{R}/data/opendl/epic_kitchens_100/annotations"; ts = lambda s: (lambda h, m, x: 3600 * int(h) + 60 * int(m) + float(x))(*s.split(":"))
NARR = collections.defaultdict(list)
for f in ("EPIC_100_train.csv", "EPIC_100_validation.csv"):
    for r in csv.DictReader(open(f"{A}/{f}")): NARR[r["video_id"]].append((ts(r["start_timestamp"]), r["narration"].strip()))
def rot(q):
    w, x, y, z = q; return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
NAVO = {"forward": "Move forward", "left": "Turn left and move forward", "right": "Turn right and move forward", "stop": "Stop"}
NAVQ = ['This is a navigation video of an agent following instruction: "{}" What is the next action it should take?', 'This is a navigation video of an agent following instruction: "{}" What is the next action you should take?']
EP = f"{R}/prep/epic_384_15fps"; nav = collections.defaultdict(list); L = 2.5
for p in sorted(glob.glob(f"{R}/data/opendl/epic_fields/json-format/*.json")):
    vid = os.path.basename(p)[:-5]; path = f"{vid.split('_')[0]}/{vid}.mp4"
    if vid not in FPS or not os.path.exists(f"{EP}/{path}"): continue
    fps, dur = FPS[vid]; fr = {int(k[6:16]): np.array(v) for k, v in json.load(open(p))["images"].items()}; keys = np.array(sorted(fr))
    def pose(t):
        f = int(round(t * fps)) + 1; i = np.abs(keys - f).argmin()
        if abs(keys[i] - f) > fps * 0.1: return None
        Rm = rot(fr[keys[i]][:4]); return Rm, -Rm.T @ fr[keys[i]][4:7]
    W = []
    for t in np.arange(4.0, dur - L - 1.0, 1.0):
        a, b = pose(t), pose(t + L)
        if a is None or b is None: continue
        fwd = a[0] @ b[0].T @ np.array([0, 0, 1.0]); yaw = np.degrees(np.arctan2(fwd[0], fwd[2])); d = a[0] @ (b[1] - a[1]); W.append((t, yaw, d))
    if len(W) < 20: continue
    med = np.median([np.linalg.norm(d) for _, _, d in W]) + 1e-9; got = collections.defaultdict(list)
    for t, yaw, d in W:
        lab = ("right" if yaw > 20 else "left" if yaw < -20 else "forward" if abs(yaw) < 6 and d[2] > med and d[2] > 2 * abs(d[0]) else
               "stop" if abs(yaw) < 4 and np.linalg.norm(d) < 0.3 * med else None)
        ins = [n for s, n in NARR[vid] if t <= s <= t + 12.0][:3]
        if lab and ins: got[lab].append((t, ". ".join(x[:1].upper() + x[1:] for x in ins) + "."))
    for lab, X in got.items():
        for j in g.rng.permutation(len(X))[:15]: nav[lab].append((path, X[int(j)][0], X[int(j)][1]))
print("nav windows per class", {k: len(v) for k, v in nav.items()}, flush=True)
m = min(len(v) for v in nav.values()) if len(nav) == 4 else 0
INS = [x[2] for X in nav.values() for x in X]              # (v2) the instruction of a RANDOM window (any video, any class): the next upcoming
for lab, X in nav.items():                                 # narrations predicted the measured motion (text-only 38 % vs 25 %) - text must not answer
    for j in g.rng.permutation(len(X))[:m]:
        path, t, _ = X[int(j)]; ins = INS[int(g.rng.integers(len(INS)))]; base = dict(video=path, prep_exists=True, root2=None, tq=None, aug="", t0=float(t - 4.0), t1=float(t))
        put("v16_nav_vln__epic", base, g.pick(NAVQ).format(ins), [NAVO[lab]] + [NAVO[o] for o in NAVO if o != lab], 0, lab, EP, 1.0)
# ---- loc_tv
LO = {"beginning": "At the beginning of the video.", "middle": "In the middle of the video.", "end": "At the end of the video.", "throughout": "Throughout the entire video."}
TVQ = ["In the given video, when does the action '{}' take place?", "When in the video sequence do we observe the action '{}'?", "At what point in the video does the action '{}' happen?", "When does the action '{}' occur in the video?"]
for src in ("loc_nat", "loc_comp"):
    byp = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in take(src, 30000):
        q, gold = gold_q(r["conversations"]); ph = re.search(r"action '(.+)' occur", q)
        if ph and gold in LO.values(): byp[ph.group(1)][gold].append(r)
    if src == "loc_comp":                                  # (v3) "throughout" for composed rows too (it was never the answer: option prior 35 %):
        LOC = {"beginning": 0, "middle": 1, "end": 2}      # three spans of the SAME action phrase from different videos / times, same total form
        tgt = collections.defaultdict(list); inv = {v: k for k, v in LO.items()}
        for ph, G in byp.items():
            for gold, rs in G.items():
                for r in rs: tgt[ph].append((r, json.loads(r["tq"])["segs"][LOC[inv[gold]]]))
        Lref = np.array([float(r["t1"]) - float(r["t0"]) for G in byp.values() for rs in G.values() for r in rs])
        for ph, X in tgt.items():
            for _ in range(len(X) // 3):
                want = float(Lref[int(g.rng.integers(len(Lref)))])       # (v3) total length drawn from the other answers' lengths
                for _try in range(30):
                    pick = [X[int(i)] for i in g.rng.choice(len(X), 3, replace=False)]
                    if len({p[1][0] for p in pick}) == 3 and abs(sum(p[1][2] - p[1][1] for p in pick) - want) <= 0.1 * want: break
                else: continue
                r = dict(pick[0][0]); segs = [p[1] for p in pick]; r["tq"] = json.dumps({"kind": "seq", "segs": segs}); r["video"] = segs[0][0]; r["t1"] = float(sum(b - a for _, a, b in segs))
                byp[ph][LO["throughout"]].append(r)
    npos = 4                                               # (v2) balanced per action phrase: the phrase alone predicted the position
    for ph, G in byp.items():                              # (Charades-Ego loc_nat text-only 37 % vs 25 %)
        if len(G) < npos: continue
        m_ = min(len(v) for v in G.values())
        for gold, rs in G.items():
            for r in rs[:m_]: put("v16_loc_tv__" + src, r, g.pick(TVQ).format(ph), [gold] + [o for o in LO.values() if o != gold], 0, gold, "/", 1.0)
# (10-26) unexp (TVBench 'locate the ... part') dropped: a spliced-in clip is found by appearance, not by dynamics - against the data's purpose
# ---- cam_mb
D = {"pan_left": "pans to the left", "pan_right": "pans to the right", "tilt_up": "tilts upward", "tilt_down": "tilts downward", "zoom_in": "zooms in", "zoom_out": "zooms out"}
EGO = {"The camera pans to the left": "pan_left", "The camera pans to the right": "pan_right", "The camera stays still": "static"}
def cam_rows(name, n):
    for r in take(name, n):
        if r.get("tq"): mv = json.loads(r["tq"])["move"]
        else: mv = EGO.get(gold_q(r["conversations"])[1])
        if mv: yield r, mv, S[name]["prep_root"], S[name]["jitter"]
for name, n in (("camera_clevrer", 5000), ("camera_chexo", 3927), ("egocam_mc_epic", 6000)):
    for r, mv, root, jit in cam_rows(name, n):
        f = g.rng.random(); yes = lambda x: f"Yes, it {D[x]}"; NO = "No, it remains static"
        moves = [x for x in D if name != "egocam_mc_epic" or x.startswith("pan")]
        if f < 0.4:
            q = g.pick(["Does the camera move at any point during the video?", "Does the camera move during the video?", "Is there any camera movement in the video?"])
            gold = NO if mv == "static" else yes(mv); ds = [NO] if mv != "static" else []; ds += [yes(x) for x in g.rng.permutation(moves) if x != mv][: 3 - len(ds)]
        elif f < 0.7:                                      # (v2) the asked type is drawn independently of the true move (asking the true
            typ = g.pick(sorted({x.split("_")[0] for x in moves}))   # type made the question text answer: 31 % vs 25 %); a different move -> "No, it <move>"
            q = f"Does the camera {typ} at any point during the video?"; no_ = lambda x: NO if x == "static" else f"No, it {D[x]}"
            cand = [yes(x) if x.startswith(typ) else no_(x) for x in moves + ["static"]]
            gold = yes(mv) if mv.startswith(typ) else no_(mv); ds = [c for c in g.rng.permutation(cand) if c != gold][:3]
        else:
            q = g.pick(["How does the camera move as the scene unfolds?", "What camera movement occurs as the scene unfolds?", "What type of camera movement is used in the video?"])
            ph = lambda x: "The camera remains static" if x == "static" else f"The camera {D[x]}"
            gold = ph(mv); ds = [ph(x) for x in g.rng.permutation(moves + ["static"]) if x != mv][:3]
        put(f"v16_cam_mb__{name}", r, q, [gold] + ds, 0, mv, root, jit)
# ---- tempo (EPIC continuous footage)
TS = {"slow": (2.5, 3.5), "standard": (8.0, 12.0), "fast": (32.0, 48.0)}
TQ = [("What is the tempo of the video playback?", {"slow": "slow", "standard": "medium", "fast": "fast"}), ("What term describes the tempo of the video?", {"slow": "slow motion", "standard": "standard speed", "fast": "high speed"}),
      ("What is the relative speed of the video compared to real-time?", {"slow": "slower", "standard": "equal", "fast": "faster"}), ("At what speed is the video playing?", {"slow": "in slow motion", "standard": "at normal speed", "fast": "fast-forwarded"})]
ev = [(f"{v.split('_')[0]}/{v}.mp4", d) for v, (_, d) in FPS.items() if d > 120 and os.path.exists(f"{EP}/{v.split('_')[0]}/{v}.mp4")]
for _ in range(12000):
    path, dur = ev[int(g.rng.integers(len(ev)))]; lab = g.pick(list(TS)); span = float(g.rng.uniform(*TS[lab])); c = float(g.rng.uniform(30, dur - 30))
    # (v2) a "rate" transformation row, not a plain [t0, t1] clip: plain clips carry their real frame seconds into the bridge (the span
    # would answer by time, not by motion); transformation rows all get the same nominal times, so only the motion per frame tells the tempo
    q, M = TQ[int(g.rng.integers(len(TQ)))]; base = dict(video=path, prep_exists=True, root2=None, tq=json.dumps({"kind": "rate", "frac": span / 48.0}), aug="", t0=c - 24.0, t1=c + 24.0)
    put("v16_tempo__epic", base, q, [M[lab]] + [M[o] for o in M if o != lab], 0, lab, EP, 0.0)
# ---- rspeed_w
for name in ("tq4_rspeed_ssv2", "tq4_rspeed_sav"):
    for r in take(name, 8000):
        q, _ = gold_q(r["conversations"]); ob = re.search(r"speed of the (.+?) above", q) or re.search(r"the (.+?) above", q); tq = json.loads(r["tq"])
        if not ob: continue
        ob = ob.group(1); lt, lb = np.diff(tq["top"])[0], np.diff(tq["bottom"])[0]; lab = "same" if abs(lt - lb) < 1e-6 else "faster" if lt > lb else "slower"
        if g.rng.random() < 0.5:
            O_ = {"faster": f"moving faster than the {ob} below", "slower": f"moving slower than the {ob} below", "same": f"moving at the same speed as the {ob} below"}; q2 = f"What is the speed of the {ob} above?"
        else:
            O_ = {"faster": "It's faster", "slower": "It's slower", "same": "It moves at the same speed"}; q2 = f"Compared to the {ob} below, how fast is the {ob} above?"
        put(f"v16_rspeed_w__{name}", r, q2, [O_[lab]] + [O_[o] for o in O_ if o != lab], 0, lab, S[name]["prep_root"], 0.0)
# ---- rep6
for r in take("rep_loop", 12000):
    q, gold = gold_q(r["conversations"])
    if not gold or not gold.isdigit(): continue
    k = int(g.pick([5, 6])); ds = [str(x) for x in g.rng.permutation([m_ for m_ in range(2, 8) if str(m_) != gold])][: k - 1]
    put("v16_rep6__rep_loop", r, q, [gold] + ds, 0, gold, "/", 1.0)
# ---- balance per answer key, write
for k, rows in sorted(out.items()):
    d = pd.DataFrame(rows); before = d._key.value_counts().to_dict(); m_ = d._key.value_counts().min()
    d = pd.concat([gg.sample(m_, random_state=0) for _, gg in d.groupby("_key")]).sort_index()
    print(k, "keys before", before, "after", m_, "each", flush=True)
    d = d.drop(columns="_key").reset_index(drop=True); report(k, d.to_dict("records")); d.to_parquet(f"{O}/{k}.parquet")
json.dump(ROOTS, open(f"{O}/roots.json", "w"), indent=1)
