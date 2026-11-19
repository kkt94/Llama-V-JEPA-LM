"""(10-16) Transformation questions, round 3 (= round 2 with three fixes after tq2_A @2000, see the end of this note). Round 2: the benchmarks' own constructions, rebuilt on our training clips. m3b_A @6000 on TempCompass
by construction: event order on two joined clips 46%, relative speed on a clip stacked over itself at another speed 34% (3 options),
direction on clips played BACKWARDS 28-33% (below chance: it answers the usual direction, not the seen one); TOMATO asks speed change and
two-step directions with 4-6 options. Each kind here is built so the answer is exact (video.compose_clip), with the benchmarks' option sets
- the 'both / same / not moving' options are made TRUE by construction where they are offered (side-by-side clips, equal rates, a frozen
frame), never offered only as decoys:
 order    two clips joined (A then B / B then A) or side by side (together); 3 options                    SSv2 labels, LLaVA caption openings
 rspeed   one clip above itself: the panels over spans in ratio 3:1 (faster / slower) or equal spans (same speed)    SSv2, SA-V (6 s windows)
 dspeed   one clip, sampling rate 3x higher in its second half (accelerating) or first (decelerating), or a frozen frame (not moving)
 odir     SSv2 object movements (left / right / up / down / towards / away from the camera) as they are, backwards or mirrored; 4 options
 cdir     SSv2 camera movements (pan left / right, tilt up / down, forward / backward), as they are, backwards or mirrored; 4 options
 state    SSv2 actions with an inverse, as they are or backwards (= the inverse)
 dir2     two direction clips joined (each part as it is / mirrored / backwards): 'Leftwards then upwards.'; 5-6 options with single
          directions and 'Not moving at all' (true when both parts are a frozen frame)
Training: manifests/stage1/tq2/<kind>_<src>.parquet (one variant per clip). Held-out: manifests/eval/pilot/tq2_heldout.jsonl (SSv2
VALIDATION, LLaVA videos left out; each clip in all its variants = a group). Round 3 fixes: (1) 'together' as TempCompass shows it - two clips ABOVE / BELOW each other (tq2 had them side by side: TempCompass
'simultaneous' items stayed at 18%); side by side kept for a third. (2) relative speed: both panels CENTRED on the same moment (the clip's
own pace cancels), ratios 2:1 and 3:1, equal = same span shifted 5% (tq2_A leaned to 'same speed' on TempCompass: 6 -> 61 on the equal items,
37 -> 17 on the slower ones). (3) camera movement on NATURAL clips (SA-V 6 s windows) from optical flow (logs/zz_flow_label.py), as they are,
backwards or mirrored (TempCompass camera on clips played backwards: 28 -> 23).
usage: zz_tq3_build.py"""
import json, re, glob, os, collections, numpy as np, pandas as pd
R = "/data/llama_vjepa_lm"; OUT = f"{R}/manifests/stage1/tq3"; os.makedirs(OUT, exist_ok=True); rng = np.random.default_rng(0)
SS = f"{R}/bench/mvp_sources/extracted/ssv2/20bn-something-something-v2"; LV = f"{R}/prep/llava_384_15fps"
TAIL = "\nAnswer with the option's letter from the given choices directly."
L = lambda s: s[0].lower() + s[1:] if s[:2] != s[:2].upper() else s
U = lambda s: s[0].upper() + s[1:]
pick = lambda xs: xs[int(rng.integers(len(xs)))]


def conv(q, opts, a):
    o = list(dict.fromkeys(opts)); rng.shuffle(o); assert a in o
    u = f"<|jepa_evidence|>\n{q}\n" + "\n".join(f"{'ABCDEFG'[i]}. {x}" for i, x in enumerate(o)) + TAIL
    return json.dumps([{"role": "user", "content": u}, {"role": "assistant", "content": "ABCDEFG"[o.index(a)] + "."}]), o, o.index(a)


def fill(t, ph):
    i = iter(ph); return re.sub(r"\[[^\]]*\]", lambda m: next(i, "something"), t)


def obj_name(o):
    o = re.sub(r"^(a|an|the|one of the|one|some|two|three|a pair of|a piece of)\s+", "", o.strip().lower())
    return o if 1 <= len(o.split()) <= 4 else "object"


def ssv2(split):
    D = pd.DataFrame(json.load(open(f"{R}/data/ssv2_labels/labels/{split}.json")))
    D = D[[not any(x.split()[:1] and x.split()[0].lower().endswith("ing") for x in p) for p in D.placeholders]]   # 'opening a manicure set' as an object
    D["cap"] = [U(fill(t, p)) for t, p in zip(D.template, D.placeholders)]; D["obj"] = [obj_name(p[0]) if len(p) else "object" for p in D.placeholders]
    D["video"] = [f"{i}.webm" for i in D.id]
    return D[[os.path.exists(f"{SS}/{v}") for v in D.video]] if split == "validation" else D


STILL = ("Holding", "Showing", "Pretending", "Something", "Trying", "Poking", "Touching", "Pulling two ends of [something] but nothing", "Spinning [something] that quickly stops")
moving = lambda D: D[~D.template.str.startswith(STILL)]

# ---------- directions ----------
OPP = {"left": "right", "right": "left", "up": "down", "down": "up", "towards": "away", "away": "towards", "forward": "backward", "backward": "forward"}
MIR = {"left": "right", "right": "left"}                                     # a mirror swaps only left and right
ODIR = {"Pushing [something] from left to right": "right", "Pushing [something] from right to left": "left",
        "Pulling [something] from left to right": "right", "Pulling [something] from right to left": "left",
        "Moving [something] up": "up", "Moving [something] down": "down",
        "Moving [something] towards the camera": "towards", "Moving [something] away from the camera": "away"}
CDIR = {"Turning the camera left while filming [something]": "left", "Turning the camera right while filming [something]": "right",
        "Turning the camera upwards while filming [something]": "up", "Turning the camera downwards while filming [something]": "down",
        "Approaching [something] with your camera": "forward", "Moving away from [something] with your camera": "backward"}
OWORD = [{"left": "to the left", "right": "to the right", "up": "upwards", "down": "downwards", "towards": "towards the camera", "away": "away from the camera"},
         {"left": "going left", "right": "going right", "up": "going up", "down": "going down", "towards": "coming towards the screen", "away": "going away from the screen"}]
CWORD = [{"left": "panning left", "right": "panning right", "up": "tilting up", "down": "tilting down", "forward": "moving forward", "backward": "moving backward"},
         {"left": "turning leftward", "right": "turning rightward", "up": "turning upward", "down": "turning downward", "forward": "moving towards the scene", "backward": "moving away from the scene"}]
D2W = {"left": "Leftwards", "right": "Rightwards", "up": "Upwards", "down": "Downwards"}


def seen(d, aug):
    return OPP[d] if aug == "rev" else MIR.get(d, d) if aug == "flip" else d


def dir_opts(d, words, n=4):
    """the true direction, its opposite, and others"""
    rest = [x for x in words if x not in (d, OPP[d])]; rng.shuffle(rest)
    return [words[d], words[OPP[d]]] + [words[x] for x in rest[:n - 2]]


# ---------- question builders: each returns dict(video, tq / aug, qa=(q, opts, answer)) ----------
SPEC_ROOT = {}


def order_item(a, b, mode, root):
    """mode 'ab' (a then b), 'ba', 'side' (together)"""
    v = int(rng.integers(3))
    if mode == "side": tq = {"kind": "side", "v2": b.video} if rng.random() < 1 / 3 else {"kind": "stack", "v2": b.video, "top": [0, 1], "bottom": [0, 1]}; vid = a.video
    else:
        x, y = (a, b) if mode == "ab" else (b, a); tq = {"kind": "concat", "v2": y.video}; vid = x.video
    f, s = (a.cap, b.cap) if mode == "ab" else (b.cap, a.cap)
    if v == 0:
        both = pick(["Both events happen at the same time", "Both events happen simultaneously"])
        opts = [f"{a.cap} and then {L(b.cap)}", f"{b.cap} and then {L(a.cap)}", both]
        ans = both if mode == "side" else f"{f} and then {L(s)}"; q = pick(["What is the order of events in the video?", "What order do the events occur in?"])
    elif v == 1:
        both = "Both events happen together"; opts = [a.cap, b.cap, both]; ans = both if mode == "side" else f
        q = pick(["What is happening first in the video?", "What is happening in the first event of the video?", "Which event happens first?"])
    else:
        both = "Both events happen together"; opts = [a.cap, b.cap, both]; ans = both if mode == "side" else s
        q = pick(["What is happening at the end of the video?", "Which event happens last?"])
    return dict(video=vid, root2=root, tq=json.dumps(tq), qa=(q, opts, ans))


RS_Q = [("What is the speed of the {o} above compared to the {o} below?", {"f": "faster", "s": "slower", "e": "the same speed"}),
        ("How does the speed of the {o} above compare to the {o} below?", {"f": "faster", "s": "slower", "e": "same speed"}),
        ("Compared to the video below, how fast is the {o} above moving?", {"f": "faster than the one below", "s": "slower than the one below", "e": "at the same speed as the one below"})]


def rspeed_item(r, mode, obj):
    """mode f (above faster), s (slower), e (equal rates); 3:1 spans"""
    r_ = 2.0 if rng.random() < 0.5 else 3.0; Ls = 0.9 / r_; Lf = 0.9                  # both centred on the middle of the span
    sl, fa = [0.5 - Ls / 2, 0.5 + Ls / 2], [0.5 - Lf / 2, 0.5 + Lf / 2]
    if mode == "f": top, bot = fa, sl
    elif mode == "s": top, bot = sl, fa
    else: top, bot = [0.5 - 0.225, 0.5 + 0.225], [0.55 - 0.225, 0.55 + 0.225]
    q, w = pick(RS_Q)
    return dict(video=r.video, t0=getattr(r, "t0", None), t1=getattr(r, "t1", None), tq=json.dumps({"kind": "stack", "top": top, "bottom": bot}),
                qa=(q.format(o=obj), list(w.values()), w[mode]))


DS_Q = [("How does the speed of the {o} change?", {"up": "Accelerating.", "down": "Decelerating.", "none": "Not moving at all."}),
        ("What is the {o}'s speed pattern?", {"up": "Accelerating.", "down": "Decelerating.", "none": "Not moving at all."}),
        ("How does the motion of the {o} change over the video?", {"up": "It speeds up.", "down": "It slows down.", "none": "It does not move."})]


def dspeed_item(r, mode, obj):
    q, w = pick(DS_Q); tq = {"kind": "freeze"} if mode == "none" else {"kind": "speed", "change": mode}
    return dict(video=r.video, t0=getattr(r, "t0", None), t1=getattr(r, "t1", None), tq=json.dumps(tq), qa=(q.format(o=obj), list(w.values()), w[mode]))


def odir_item(r, aug):
    d = seen(ODIR[r.template], aug); w = pick(OWORD)
    q = pick([f"Which way is the {r.obj} moving?", f"In which direction does the {r.obj} move?", f"How is the {r.obj} moving?"])
    return dict(video=r.video, aug=aug, qa=(q, dir_opts(d, w), w[d]))


def cdir_item(r, aug):
    d = seen(CDIR[r.template], aug); w = pick(CWORD)
    q = pick(["What is the camera motion in the video?", "How is the camera moving?", "In what way is the primary motion of the camera depicted in the video?"])
    return dict(video=r.video, aug=aug, qa=(q, dir_opts(d, w), w[d]))


INV = {"Opening [something]": "Closing {0}", "Closing [something]": "Opening {0}", "Folding [something]": "Unfolding {0}", "Unfolding [something]": "Folding {0}",
       "Covering [something] with [something]": "Uncovering {0}", "Uncovering [something]": "Covering {0}",
       "Putting [something] into [something]": "Taking {0} out of {1}", "Taking [something] out of [something]": "Putting {0} into {1}",
       "Moving [something] and [something] closer to each other": "Moving {0} and {1} away from each other",
       "Moving [something] and [something] away from each other": "Moving {0} and {1} closer to each other",
       "Moving [something] closer to [something]": "Moving {0} away from {1}", "Moving [something] away from [something]": "Moving {0} closer to {1}",
       "Picking [something] up": "Putting {0} down", "Putting [something] on a surface": "Picking {0} up", "Putting [something] onto [something]": "Taking {0} off {1}",
       "Plugging [something] into [something]": "Unplugging {0} from {1}"}


def state_item(r, rev):
    a, b = r.cap, U(INV[r.template].format(*(list(r.placeholders) + ["something"] * 2)))
    q = pick(["What happens in the video?", "Which action is performed in the video?", "What is the person doing in the video?"])
    return dict(video=r.video, aug="rev" if rev else "", qa=(q, [a, b], b if rev else a))


LR4 = [d for d in ODIR.values() if d in D2W]


def dir2_item(a, b, a1, a2, freeze=False):
    """a, b: SSv2 rows with a left/right/up/down movement; a1 / a2: their transforms"""
    if freeze:
        tq = {"kind": "freeze"}; ans = pick(["Not moving at all.", "Not moving at all"]); d1 = d2 = None
        vid = a.video
    else:
        d1, d2 = seen(ODIR[a.template], a1), seen(ODIR[b.template], a2)
        tq = {"kind": "concat", "v2": b.video, "aug1": a1, "aug2": a2}; vid = a.video
        ans = f"{D2W[d1]} then {L(D2W[d2])}." if d1 != d2 else f"{D2W[d1]}."
    pool = [f"{D2W[x]} then {L(D2W[y])}." for x in D2W for y in D2W if x != y] + [f"{D2W[x]}." for x in D2W]
    # the answer and its mirror image in time (B then A; a single direction: its opposite) - a guess from the option list alone is 50%
    if d1 is None: core = [ans]
    elif d1 != d2: core = [ans, f"{D2W[d2]} then {L(D2W[d1])}."]
    else: core = [ans, f"{D2W[OPP[d1]]}."]
    n = int(rng.integers(5, 7)); opts = core + ([] if d1 is None else ["Not moving at all."])
    rest = [p for p in pool if p not in opts]; rng.shuffle(rest); opts += rest[: n - len(opts)]
    q = pick([f"How did the {a.obj} move in terms of direction?", f"What was the direction of the {a.obj}'s movement?", f"How would you describe the {a.obj}'s directional movement?"])
    return dict(video=vid, root2=SS, tq=json.dumps(tq), qa=(q, opts, ans))


# ---------- sources ----------
def llava_caps():
    M = pd.read_parquet(f"{R}/manifests/stage1/llava.parquet", columns=["source", "kind", "video", "conversations"])
    M = M[(M.kind == "cap") & M.source.str.startswith(("0_30_s", "30_60_s"))].drop_duplicates("video")
    pat = re.compile(r"^The video (?:begins with|opens with|starts with|showcases|shows|features|depicts|captures|presents)\s+(.*)$", re.I); out = []
    for v, c in zip(M.video, M.conversations):
        t = json.loads(c)[1]["content"].split(". ")[0].strip().rstrip("."); m = pat.match(t)
        if not m: continue
        s = m.group(1); w = s.split()
        if 5 <= len(w) <= 22 and "video" not in s.lower() and "frame" not in s.lower(): out.append((v, U(s)))
    return pd.DataFrame(out, columns=["video", "cap"])


def moving_llava():
    M = pd.read_parquet(f"{R}/feats/ic128/main/llava.parquet"); C = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{R}/runs/info/consist_*.parquet")])
    C = C[(C.source == "llava") & (C.arrow_org <= 0.1) & (C.arrow_rev >= 0.9)].drop_duplicates("video")
    return pd.DataFrame([{"video": M.video.iloc[r], "t0": M.t0.iloc[r], "t1": M.t1.iloc[r]} for r in C.row])


def save(name, rows):
    d = pd.DataFrame([{"video": r["video"], "prep_exists": True, "root2": r.get("root2"), "tq": r.get("tq"), "aug": r.get("aug", ""), "t0": r.get("t0"), "t1": r.get("t1"),
                       "conversations": conv(*r["qa"])[0]} for r in rows])
    d.to_parquet(f"{OUT}/{name}.parquet"); print(name, len(d), flush=True)


tr, va = ssv2("train"), ssv2("validation"); LC = llava_caps(); hold_v = set(LC.video.sample(600, random_state=1))
# LLaVA clips are not used for speed: the feature store keeps whole videos (no window), and over a whole video with cuts a 3x rate
# change separated only 85% / 35% (frame differences); SSv2 single actions 90% / 100%
cyc = lambda n, ks: [ks[i % len(ks)] for i in range(n)]


def order_rows(D, root, n):
    D = D.sample(frac=1, random_state=int(rng.integers(1 << 30))).reset_index(drop=True); out = []
    tcol = "template" if "template" in D else "cap"; modes = cyc(n, ["ab", "ba", "ab", "ba", "side"])
    for i in range(min(n, len(D) // 2)):
        a, b = D.iloc[2 * i], D.iloc[2 * i + 1]
        if a[tcol] != b[tcol] and a.cap != b.cap: out.append(order_item(a, b, modes[i], root))
    return out


save("order_ssv2", order_rows(moving(tr), SS, 30000))
save("order_llava", order_rows(LC[~LC.video.isin(hold_v)], LV, 20000))
mt = moving(tr).sample(frac=1, random_state=11).reset_index(drop=True)
save("rspeed_ssv2", [rspeed_item(r, m, r.obj) for r, m in zip(mt.iloc[:30000].itertuples(), cyc(30000, ["f", "s", "e"]))])
save("dspeed_ssv2", [dspeed_item(r, m, r.obj) for r, m in zip(mt.iloc[30000:60000].itertuples(), cyc(30000, ["up", "down", "up", "down", "none"]))])
# SA-V (everyday single-shot videos, 24 fps, 10-17 s): speed on a 6 s window; a 3x rate change separates 100% / 93% (frame differences)
SV = "/data/llama_vjepa_lm/prep/sav_384_24fps"
import decord
def sav_rows(n, seed):
    vids = pd.read_parquet(f"{R}/manifests/stage1/sav_tracks_qa.parquet", columns=["video"]).video.drop_duplicates().sample(n, random_state=seed); out = []
    for v in vids:
        try: vr = decord.VideoReader(f"{SV}/{v}"); D = len(vr) / vr.get_avg_fps(); del vr
        except Exception: continue
        t0 = float(rng.uniform(0, max(D - 6, 0))); out.append(dict(video=v, t0=t0, t1=min(D, t0 + 6)))
    return pd.DataFrame(out)
SVall = sav_rows(36600, 21); SVh = SVall.iloc[:600].reset_index(drop=True); SVt = SVall.iloc[600:].reset_index(drop=True)
save("rspeed_sav", [rspeed_item(r, m, "video") for r, m in zip(SVt.iloc[:18000].itertuples(), cyc(18000, ["f", "s", "e"]))])
save("dspeed_sav", [dspeed_item(r, m, "motion") for r, m in zip(SVt.iloc[18000:].itertuples(), cyc(len(SVt) - 18000, ["up", "down", "up", "down", "none"]))])
# camera movement on SA-V windows from optical flow (the measured movement of the content; zz_flow_label.py)
FL = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{R}/runs/info/flow_*_*.parquet")])
def flab(r, th=0.010, ts=0.008):
    ax, ay, az = abs(r.tx), abs(r.ty), abs(r.s) * 0.5                # the expansion's displacement at the edge, in widths
    if ax >= th and r.agree_x >= 0.875 and ax > 2 * ay and ax > 2 * az: return "left" if r.tx > 0 else "right"     # content right = pan left
    if ay >= th and r.agree_y >= 0.875 and ay > 2 * ax and ay > 2 * az: return "up" if r.ty > 0 else "down"
    if abs(r.s) >= ts and r.agree_s >= 0.875 and az > 1.5 * max(ax, ay): return "forward" if r.s > 0 else "backward"
    return None
FL["d"] = [flab(r) for r in FL.itertuples()]; FL = FL[FL.d.notna()].drop_duplicates("video").sample(frac=1, random_state=18).reset_index(drop=True)
print("flow-labelled SA-V windows", len(FL), FL.d.value_counts().to_dict(), flush=True)
def cnat_item(r, aug):
    d = seen(r.d, aug); w = pick(CWORD)
    q = pick(["What is the camera motion in the video?", "How is the camera moving?", "In what way is the primary motion of the camera depicted in the video?",
              "What direction is the camera moving in?"])
    return dict(video=r.video, t0=r.t0, t1=r.t1, aug=aug, qa=(q, dir_opts(d, w), w[d]))
FLh, FLt = FL.iloc[:300], FL.iloc[300:]
save("cnat_sav", [cnat_item(r, a) for r, a in zip(FLt.itertuples(), cyc(len(FLt), ["", "rev", "flip"]))])
O = tr[tr.template.isin(ODIR)].sample(frac=1, random_state=13)
save("odir_ssv2", [odir_item(r, a) for r, a in zip(O.itertuples(), cyc(len(O), ["", "rev", "flip"]))])
C_ = tr[tr.template.isin(CDIR)].sample(frac=1, random_state=14)
save("cdir_ssv2", [cdir_item(r, a) for r, a in zip(C_.itertuples(), cyc(len(C_), ["", "rev", "flip"]))])
S = tr[tr.template.isin(INV)].sample(frac=1, random_state=15)
save("state_ssv2", [state_item(r, k % 2 == 1) for k, r in enumerate(S.itertuples())])
LRt = tr[tr.template.isin([t for t, d in ODIR.items() if d in D2W])].sample(frac=1, random_state=16).reset_index(drop=True); rows = []
for i in range(min(20000, len(LRt) // 2)):
    a, b = LRt.iloc[2 * i], LRt.iloc[2 * i + 1]
    rows.append(dir2_item(a, b, pick(["", "rev", "flip"]), pick(["", "rev", "flip"]), freeze=i % 10 == 9))
save("dir2_ssv2", rows)

# ---------- held-out ----------
H = []; cnt = collections.Counter()
def item(kind, gid, r, root):
    _, o, k = conv(*r["qa"]); q = r["qa"][0]; cnt[gid] += 1
    d = {"bench": "tq3_heldout", "id": f"{gid}_{cnt[gid]}", "video": f"{root}/{r['video']}", "question": q, "options": o, "answer_index": k, "answer_text": o[k],
         "group_id": gid, "aug": r.get("aug", ""), "extra": {"kind": kind}}
    if r.get("tq"):
        t = json.loads(r["tq"])
        if "v2" in t: t["v2"] = f"{r.get('root2') or root}/{t['v2']}"
        d["tq"] = t
    if r.get("t0") is not None and r.get("t1") is not None and r["t1"] > r["t0"]: d["start_sec"], d["end_sec"] = float(r["t0"]), float(r["t1"])
    H.append(d)
vm = moving(va).sample(frac=1, random_state=7).reset_index(drop=True)
for i in range(200):
    a, b = vm.iloc[2 * i], vm.iloc[2 * i + 1]
    if a.template == b.template: continue
    st = rng.bit_generator.state
    for m in ("ab", "ba", "side"): rng.bit_generator.state = st; item("order_ssv2", f"order_ssv2_{i}", order_item(a, b, m, SS), SS)
LH = LC[LC.video.isin(hold_v)].reset_index(drop=True)
for i in range(200):
    a, b = LH.iloc[2 * i], LH.iloc[2 * i + 1]; st = rng.bit_generator.state
    for m in ("ab", "ba", "side"): rng.bit_generator.state = st; item("order_llava", f"order_llava_{i}", order_item(a, b, m, LV), LV)
for i, r in enumerate(vm.iloc[400:700].itertuples()):
    st = rng.bit_generator.state
    for m in ("f", "s", "e"): rng.bit_generator.state = st; item("rspeed_ssv2", f"rspeed_ssv2_{i}", rspeed_item(r, m, r.obj), SS)
for i, r in enumerate(SVh.iloc[:300].itertuples()):
    st = rng.bit_generator.state
    for m in ("f", "s", "e"): rng.bit_generator.state = st; item("rspeed_sav", f"rspeed_sav_{i}", rspeed_item(r, m, "video"), SV)
for i, r in enumerate(SVh.iloc[300:600].itertuples()):
    st = rng.bit_generator.state
    for m in ("up", "down", "none"): rng.bit_generator.state = st; item("dspeed_sav", f"dspeed_sav_{i}", dspeed_item(r, m, "motion"), SV)
for i, r in enumerate(vm.iloc[700:1000].itertuples()):
    st = rng.bit_generator.state
    for m in ("up", "down", "none"): rng.bit_generator.state = st; item("dspeed_ssv2", f"dspeed_ssv2_{i}", dspeed_item(r, m, r.obj), SS)
for i, r in enumerate(va[va.template.isin(ODIR)].sample(300, random_state=8).itertuples()):
    st = rng.bit_generator.state
    for a in ("", "rev", "flip"): rng.bit_generator.state = st; item("odir_ssv2", f"odir_ssv2_{i}", odir_item(r, a), SS)
for i, r in enumerate(FLh.itertuples()):
    st = rng.bit_generator.state
    for a in ("", "rev", "flip"): rng.bit_generator.state = st; item("cnat_sav", f"cnat_sav_{i}", cnat_item(r, a), SV)
CV = va[va.template.isin(CDIR)]
for i, r in enumerate(CV.sample(min(300, len(CV)), random_state=9).itertuples()):
    st = rng.bit_generator.state
    for a in ("", "rev", "flip"): rng.bit_generator.state = st; item("cdir_ssv2", f"cdir_ssv2_{i}", cdir_item(r, a), SS)
for i, r in enumerate(va[va.template.isin(INV)].sample(300, random_state=10).itertuples()):
    st = rng.bit_generator.state
    for rv in (False, True): rng.bit_generator.state = st; item("state_ssv2", f"state_ssv2_{i}", state_item(r, rv), SS)
LRv = va[va.template.isin([t for t, d in ODIR.items() if d in D2W])].sample(frac=1, random_state=17).reset_index(drop=True)
for i in range(200):
    a, b = LRv.iloc[2 * i], LRv.iloc[2 * i + 1]
    for a1, a2 in (("", ""), ("rev", "rev"), ("flip", "")): item("dir2_ssv2", f"dir2_ssv2_{i}", dir2_item(a, b, a1, a2), SS)
with open(f"{R}/manifests/eval/pilot/tq3_heldout.jsonl", "w") as f:
    for h in H: f.write(json.dumps(h) + "\n")
print("held-out", len(H), dict(collections.Counter(h["extra"]["kind"] for h in H)))
