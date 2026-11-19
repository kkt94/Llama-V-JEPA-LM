"""(10-26, after nx12@10k: new data moved only the benchmark category whose QUESTION FORMAT and domain matched it - TVBench egocentric
sequence +30 - and nothing named alike but asked differently) the same human labels / measurements asked in many formats, so the reading is
not tied to one template. Answers only from human-labelled action spans (EPIC-KITCHENS-100, HoloAssist, Charades-Ego) or measured camera
poses (EPIC Fields yaw: sign checked against optical flow, 91.8 %). All videos already transcoded (no new decoding work on training nodes).
 fmt_yn       "does the video show 'X'?"            yes: a labelled span; no: another action of the SAME video outside the window (half open
                                                    answer "Yes." / "No.", half lettered yes / no)
 fmt_before   "does 'A' happen before 'B'?"         two consecutive labelled actions in one window; asked both ways (yes / no 50 / 50)
 fmt_after    "what happens right after 'A'?"       (and "right before 'B'") 4 options, distractors from the same video
 fmt_caption  "which description matches?"         2 captions: the true order of 2-3 actions and the same actions reordered (Vinoground /
                                                    TemporalBench form)
 fmt_breakdown "describe the detailed breakdown"   3-4 options = orders of the same 3 actions, comma-joined (MotionBench form)
 egocam_mc / egocam_yn   real camera motion of EPIC first-person footage over 2.5 s: pans left / pans right / stays still (|yaw| > 20 deg
                         left/right, < 3 deg still), as options and as yes / no
 next_turn    EPIC: the clip shows 2.5 s with little turning (|yaw| < 10); "which way next?" = the measured yaw of the following 2.5 s
 next_action  HoloAssist: clip = up to 8 s ending with a labelled action; "what will the person do next?" = the next labelled action
-> manifests/stage1/opendl3/{fmt_*,egocam_*,next_*}.parquet  (prep roots: EPIC prep/epic_384_15fps, HoloAssist prep/holoassist_384_15fps,
   Charades-Ego data/opendl/charades_ego/CharadesEgo_v1_480; the `video` column is relative to it, column `src` names the source)
usage: zz_qgen_v13.py"""
import sys, os, csv, io, json, zipfile, glob, itertools, collections
import numpy as np, pandas as pd
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
R = "/data/llama_vjepa_lm"; OUT = f"{R}/manifests/stage1/opendl3"; g = Gen(97)
ROOT = {"epic": f"{R}/prep/epic_384_15fps", "holo": f"{R}/prep/holoassist_384_15fps", "chego": f"{R}/data/opendl/charades_ego/CharadesEgo_v1_480"}
# ---- labelled spans per video: (start, end, phrase)
V = []                                                     # (src, video rel path, [(s, e, phrase)])
A = f"{R}/data/opendl/epic_kitchens_100/annotations"; ts = lambda s: (lambda h, m, x: 3600 * int(h) + 60 * int(m) + float(x))(*s.split(":"))
VK = {r["id"]: r["key"] for r in csv.DictReader(open(f"{A}/EPIC_100_verb_classes.csv"))}; NK = {r["id"]: r["key"] for r in csv.DictReader(open(f"{A}/EPIC_100_noun_classes.csv"))}
nice_n = lambda k: " ".join(reversed(k.split(":"))) if ":" in k else k
by = collections.defaultdict(list)
for f in ("EPIC_100_train.csv", "EPIC_100_validation.csv"):
    for r in csv.DictReader(open(f"{A}/{f}")):
        by[(r["participant_id"], r["video_id"])].append((ts(r["start_timestamp"]), ts(r["stop_timestamp"]), f"{VK[r['verb_class']].replace('-', ' ')} the {nice_n(NK[r['noun_class']])}"))
for (p, vid), X in by.items():
    path = f"{p}/{vid}.mp4"
    if os.path.exists(f"{ROOT['epic']}/{path}"): V.append(("epic", path, sorted(X)))
cl = lambda s: str(s).replace("_", " ").replace("-", " ").strip().lower()
for v in json.load(open(f"{R}/data/opendl/holoassist/data-annotation-trainval-v1_1.json")):
    path = f"{v['video_name']}.mp4"
    if not os.path.exists(f"{ROOT['holo']}/{path}"): continue
    X = []
    for e in v["events"]:
        a = e.get("attributes") or {}
        if e["label"] == "Fine grained action" and "end" in e and cl(a.get("Verb", "none")) != "none" and cl(a.get("Noun", "none")) != "none":
            adj = cl(a.get("Adjective", "none")); X.append((float(e["start"]), float(e["end"]), f"{cl(a['Verb'])} the {'' if adj == 'none' else adj + ' '}{cl(a['Noun'])}"))
    V.append(("holo", path, sorted(X)))
D = f"{R}/data/opendl/charades_ego"; z = zipfile.ZipFile(f"{D}/CharadesEgo.zip"); rd = lambda n: io.TextIOWrapper(z.open("CharadesEgo/" + n), encoding="utf-8")
NAME = {l.split()[0]: l.strip()[5:] for l in rd("Charades_v1_classes.txt") if l.strip()}
FACE = {"c029", "c072", "c078", "c085", "c094", "c131", "c133", "c146", "c149", "c152", "c153"}
for r in csv.DictReader(rd("CharadesEgo_v1_train.csv")):
    X = []
    for a in (r["actions"] or "").split(";"):
        q = a.split()
        if len(q) == 3 and q[0] in NAME and q[0] not in FACE and float(q[2]) - float(q[1]) >= 1.0: X.append((float(q[1]), float(q[2]), NAME[q[0]][0].lower() + NAME[q[0]][1:]))
    path = f"{r['id']}.mp4"
    if X and os.path.exists(f"{ROOT['chego']}/{path}"): V.append(("chego", path, sorted(X)))
print("videos with labels:", collections.Counter(s for s, _, _ in V), flush=True)
out = collections.defaultdict(list)
def add(kind, src, path, t0, t1, conv):
    out[kind].append(dict(video=path, prep_exists=True, root2=None, tq=None, aug="", t0=float(t0), t1=float(t1), conversations=conv, src=src))
def mc(kind, src, path, t0, t1, q, opts, k): r = g.row(path, t0, t1, q, opts, k); add(kind, src, path, t0, t1, r["conversations"])
def yn(kind, src, path, t0, t1, q, yes):
    if g.rng.random() < 0.5: mc(kind, src, path, t0, t1, q, ["yes", "no"] if yes else ["no", "yes"], 0)
    else: add(kind, src, path, t0, t1, json.dumps([{"role": "user", "content": f"<|jepa_evidence|>\n{q}\nAnswer yes or no."}, {"role": "assistant", "content": "Yes." if yes else "No."}]))
Q_YN = ["Does the video show '{}'?", "Is '{}' shown in this video?", "Does the person {} in this video?"]
Q_BEF = ["Does '{}' happen before '{}'?", "Did the person {} before they {}?"]
Q_AFT = ["What happens right after '{}'?", "What does the person do right after they {}?"]
Q_BFR = ["What happens right before '{}'?", "What does the person do just before they {}?"]
Q_CAP = ["Which description matches the video?", "Which caption describes the video correctly?", "Which sentence matches what happens?"]
Q_BRK = ["Please describe the detailed breakdown of the action in the video.", "In what order are the steps performed?", "Which breakdown of the activity is correct?"]
def tpl(qs, src, *ph):                                     # natural phrasing only for base-form sources (EPIC / HoloAssist verb + noun)
    q = g.pick(qs if src != "chego" else [x for x in qs if "'" in x]); return q.format(*ph)
for src, path, X in V:
    X = [x for x in X if 0.5 <= x[1] - x[0] <= 12.0]; phr = sorted({x[2] for x in X})
    if len(phr) < 4: continue
    cap = {"epic": 8, "holo": 8, "chego": 3}[src]          # items per kind per video (EPIC / HoloAssist videos are long)
    idx = list(g.rng.permutation(len(X)))
    for i in idx[:cap]:                                    # fmt_yn
        s, e, p = X[i]; yes = g.rng.random() < 0.5
        if yes: yn("fmt_yn", src, path, s, e, tpl(Q_YN, src, p), True)
        else:
            neg = [x[2] for x in X if x[2] != p and (x[1] < s - 5 or x[0] > e + 5) and not any(y[2] == x[2] and y[0] < e and y[1] > s for y in X)]
            if neg: yn("fmt_yn", src, path, s, e, tpl(Q_YN, src, g.pick(neg)), False)
    seqs = []                                              # consecutive non-overlapping, different actions
    for i in range(len(X)):
        sq = [X[i]]
        for y in X[i + 1:]:
            if y[0] >= sq[-1][1] + 0.2 and y[2] not in {z[2] for z in sq}: sq.append(y)
            if len(sq) == 3: break
        if len(sq) >= 2 and sq[-1][1] - sq[0][0] <= 20: seqs.append(sq)
    for j in list(g.rng.permutation(len(seqs)))[:cap]:
        sq = seqs[j]; a, b = sq[0], sq[1]; t0, t1 = a[0], b[1]
        if g.rng.random() < 0.5: yn("fmt_before", src, path, t0, t1, tpl(Q_BEF, src, a[2], b[2]), True)
        else: yn("fmt_before", src, path, t0, t1, tpl(Q_BEF, src, b[2], a[2]), False)
        others = [p for p in phr if p not in (a[2], b[2])]
        if len(others) >= 3:
            ds = [others[int(k)] for k in g.rng.choice(len(others), 3, replace=False)]
            if g.rng.random() < 0.5: mc("fmt_after", src, path, t0, t1, tpl(Q_AFT, src, a[2]), [b[2]] + ds, 0)
            else: mc("fmt_after", src, path, t0, t1, tpl(Q_BFR, src, b[2]), [a[2]] + ds, 0)
        acts = [x[2] for x in sq]; tt1 = sq[-1][1]
        perm = [list(pp) for pp in itertools.permutations(acts) if list(pp) != acts]; alt = perm[int(g.rng.integers(len(perm)))]
        mc("fmt_caption", src, path, t0, tt1, g.pick(Q_CAP), ["First " + ", then ".join(acts) + ".", "First " + ", then ".join(alt) + "."], 0)
        if len(sq) == 3:
            pp = [perm[int(k)] for k in g.rng.choice(len(perm), min(3, len(perm)), replace=False)]
            mc("fmt_breakdown", src, path, t0, tt1, g.pick(Q_BRK), [", ".join(acts)] + [", ".join(x) for x in pp], 0)
    if src == "holo":                                      # next_action
        for i in list(g.rng.permutation(len(X) - 1))[:cap]:
            a = X[i]; nxt = [y for y in X[i + 1:] if y[0] >= a[1] - 0.1 and y[2] != a[2]]
            if not nxt: continue
            b = nxt[0]; others = [p for p in phr if p not in (a[2], b[2])]
            if len(others) < 3 or b[0] - a[1] > 3.0: continue
            ds = [others[int(k)] for k in g.rng.choice(len(others), 3, replace=False)]
            mc("next_action", src, path, max(0.0, a[1] - 8.0), a[1], g.pick(["What will the person do next?", "What is the next step?", "What does the person do after this?"]), [b[2]] + ds, 0)
# ---- EPIC Fields: real camera motion and the next turn
def rot(q):
    w, x, y, zz = q; return np.array([[1 - 2 * (y * y + zz * zz), 2 * (x * y - zz * w), 2 * (x * zz + y * w)], [2 * (x * y + zz * w), 1 - 2 * (x * x + zz * zz), 2 * (y * zz - x * w)], [2 * (x * zz - y * w), 2 * (y * zz + x * w), 1 - 2 * (x * x + y * y)]])
FPS = {r["video_id"]: (float(r["fps"]), float(r["duration"])) for r in csv.DictReader(open(f"{A}/EPIC_100_video_info.csv"))}
CAM = {"left": "The camera pans to the left", "right": "The camera pans to the right", "still": "The camera stays still"}
TURN = {"left": "Turn left", "right": "Turn right", "same": "Keep facing the same direction"}
for p in sorted(glob.glob(f"{R}/data/opendl/epic_fields/json-format/*.json")):
    vid = os.path.basename(p)[:-5]; path = f"{vid.split('_')[0]}/{vid}.mp4"
    if vid not in FPS or not os.path.exists(f"{ROOT['epic']}/{path}"): continue
    fps, dur = FPS[vid]; fr = {int(k[6:16]): np.array(v) for k, v in json.load(open(p))["images"].items()}; keys = np.array(sorted(fr))
    def yaw(t0, t1):
        f0, f1 = int(round(t0 * fps)) + 1, int(round(t1 * fps)) + 1; i0, i1 = np.abs(keys - f0).argmin(), np.abs(keys - f1).argmin()
        if abs(keys[i0] - f0) > fps * 0.1 or abs(keys[i1] - f1) > fps * 0.1: return None
        f = rot(fr[keys[i0]][:4]) @ rot(fr[keys[i1]][:4]).T @ np.array([0, 0, 1.0]); return float(np.degrees(np.arctan2(f[0], f[2])))
    got = collections.defaultdict(list)
    for t0 in np.arange(2.0, dur - 7.0, 1.25):
        y1 = yaw(t0, t0 + 2.5)
        if y1 is None: continue
        lab = "right" if y1 > 20 else "left" if y1 < -20 else "still" if abs(y1) < 3 else None
        if lab: got["cam_" + lab].append(t0)
        if abs(y1) < 10:
            y2 = yaw(t0 + 2.5, t0 + 5.0)
            if y2 is not None:
                l2 = "right" if y2 > 25 else "left" if y2 < -25 else "same" if abs(y2) < 3 else None
                if l2: got["next_" + l2].append(t0)
    k = min(len(got["cam_left"]), len(got["cam_right"]), len(got["cam_still"]), 12)
    for lab in ("left", "right", "still"):
        for t0 in [got["cam_" + lab][int(j)] for j in g.rng.permutation(len(got["cam_" + lab]))[:k]]:
            if g.rng.random() < 0.5: mc("egocam_mc", "epic", path, t0, t0 + 2.5, g.pick(["How does the camera move?", "What is the camera motion in this video?", "Which camera movement is shown?"]), [CAM[lab]] + [CAM[o] for o in CAM if o != lab], 0)
            else:
                ask = g.pick([o for o in CAM]); yn("egocam_yn", "epic", path, t0, t0 + 2.5, g.pick(["Does the camera pan to the left?", "Does the camera pan to the right?", "Does the camera stay still?"][["left", "right", "still"].index(ask):][:1]), ask == lab)
    k = min(len(got["next_left"]), len(got["next_right"]), len(got["next_same"]), 6)
    for lab in ("left", "right", "same"):
        for t0 in [got["next_" + lab][int(j)] for j in g.rng.permutation(len(got["next_" + lab]))[:k]]:
            mc("next_turn", "epic", path, t0, t0 + 2.5, g.pick(["Which way will the camera wearer turn next?", "What will I do next: turn left, turn right, or keep facing the same way?", "After this, which way does the person turn?"]), [TURN[lab]] + [TURN[o] for o in TURN if o != lab], 0)
for kind, rows in sorted(out.items()):
    report(kind, rows); d = pd.DataFrame(rows)
    print("   by source:", dict(collections.Counter(d.src)), flush=True)
    d.to_parquet(f"{OUT}/{kind}.parquet")
