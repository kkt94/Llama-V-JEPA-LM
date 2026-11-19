"""(10-26, user: target the physics benchmarks too - PhysBench dynamics / property, MMWorld, IntPhys-style prediction) physical-event
questions from the CLEVRER SIMULATION ground truth (annotation_train: per-frame location / velocity, collision frames, object properties) in
formats the existing `clevrer` source (CLEVRER's own question programs) does not use. 128 frames at 25 fps per video.
 phys_collide   "do the X and the Y collide?"            yes: a recorded collision pair; no: two objects in view that never collide (50/50)
 phys_count     "how many collisions happen?"            count_options (gold rank uniform)
 phys_first     "which object does the X hit first?"     4 options among the objects of the video
 phys_fastest   "which object moves fastest?"            mean speed over the clip, winner >= 1.5x the runner-up, all in view
 phys_moving    "is the X still moving at the end?"      speed over the last 0.4 s: > 0.3 moving / < 0.02 still (50/50)
 phys_predict   clip CUT at 2.0-3.4 s, "will the X and the Y collide later?"  yes: first collision of the pair >= 0.4 s after the cut, both in
                view before the cut; no: both in view, never collide (50/50)  - prediction from observed motion
Objects are named by colour + shape (+ material when needed to be unique). -> manifests/stage1/opendl3/phys_*.parquet (prep root
data/clevrer/video_train)   usage: zz_qgen_phys.py"""
import sys, os, json, glob, collections, numpy as np, pandas as pd
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report, count_options
R = "/data/llama_vjepa_lm"; C = f"{R}/data/clevrer"; OUT = f"{R}/manifests/stage1/opendl3"; g = Gen(101); FPS = 25.0
out = collections.defaultdict(list)
def add(kind, path, t0, t1, conv): out[kind].append(dict(video=path, prep_exists=True, root2=None, tq=None, aug="", t0=float(t0), t1=float(t1), conversations=conv))
def mc(kind, path, t0, t1, q, opts, k): add(kind, path, t0, t1, g.row(path, t0, t1, q, opts, k)["conversations"])
def yn(kind, path, t0, t1, q, yes):
    if g.rng.random() < 0.5: mc(kind, path, t0, t1, q, ["yes", "no"] if yes else ["no", "yes"], 0)
    else: add(kind, path, t0, t1, json.dumps([{"role": "user", "content": f"<|jepa_evidence|>\n{q}\nAnswer yes or no."}, {"role": "assistant", "content": "Yes." if yes else "No."}]))
for af in sorted(glob.glob(f"{C}/annotation_train/*/*.json")):
    a = json.load(open(af)); vid = a["video_filename"].replace(".mp4", ""); n_ = int(vid.split("_")[1]); grp = f"video_{n_ // 1000 * 1000:05d}-{n_ // 1000 * 1000 + 1000:05d}"
    path = f"{grp}/{vid}.mp4"
    if not os.path.exists(f"{C}/video_train/{path}"): continue
    P = {o["object_id"]: o for o in a["object_property"]}; T = a["motion_trajectory"]; nf = len(T); dur = nf / FPS
    key = collections.Counter((o["color"], o["shape"]) for o in P.values())
    name = {i: (f"the {o['color']} {o['shape']}" if key[(o["color"], o["shape"])] == 1 else f"the {o['color']} {o['material']} {o['shape']}") for i, o in P.items()}
    if len(set(name.values())) < len(name): continue
    vis = {i: np.array([any(ob["object_id"] == i and ob.get("inside_camera_view") for ob in fr["objects"]) for fr in T]) for i in P}
    spd = {i: np.array([next((np.hypot(*ob["velocity"][:2]) for ob in fr["objects"] if ob["object_id"] == i), np.nan) for fr in T]) for i in P}
    col = [(tuple(sorted(c["object_ids"])), c["frame_id"]) for c in a["collision"]]; pairs = {p for p, _ in col}
    seen = [i for i in P if vis[i].any()]
    # collide yes / no over the whole video
    if seen and len(seen) >= 2:
        allp = [(x, y) for i, x in enumerate(sorted(seen)) for y in sorted(seen)[i + 1:]]
        pos = [p for p in allp if p in pairs]; neg = [p for p in allp if p not in pairs]
        if pos and neg:
            p = pos[int(g.rng.integers(len(pos)))] if g.rng.random() < 0.5 else None
            if p: yn("phys_collide", path, 0, dur, g.pick(["Do {} and {} collide?", "Does {} hit {}?", "Is there a collision between {} and {}?"]).format(name[p[0]], name[p[1]]), True)
            else: p = neg[int(g.rng.integers(len(neg)))]; yn("phys_collide", path, 0, dur, g.pick(["Do {} and {} collide?", "Does {} hit {}?", "Is there a collision between {} and {}?"]).format(name[p[0]], name[p[1]]), False)
    # number of collisions
    if g.rng.random() < 0.5:
        mc("phys_count", path, 0, dur, g.pick(["How many collisions happen in the video?", "How many times do objects collide?"]), count_options(len(col), g.rng, lo=0, width=3), 0)
    # first object hit by X
    hit = collections.defaultdict(list)
    for (x, y), f in sorted(col, key=lambda c: c[1]): hit[x].append(y); hit[y].append(x)
    cand = [i for i in hit if len(seen) >= 4]
    if cand:
        i = cand[int(g.rng.integers(len(cand)))]; gold = hit[i][0]; others = [j for j in seen if j not in (i, gold)]
        if len(others) >= 3:
            ds = [others[int(k)] for k in g.rng.choice(len(others), 3, replace=False)]
            mc("phys_first", path, 0, dur, g.pick(["Which object does {} hit first?", "What is the first object that collides with {}?"]).format(name[i]), [name[gold].capitalize()] + [name[j].capitalize() for j in ds], 0)
    # fastest over a 2 s window where >= 4 objects are in view
    for _ in range(2):
        s = int(g.rng.integers(0, max(1, nf - 50))); e = s + 50; inview = [i for i in P if vis[i][s:e].all()]
        if len(inview) < 4: continue
        m = {i: np.nanmean(spd[i][s:e]) for i in inview}; order = sorted(inview, key=lambda i: -m[i])
        if m[order[0]] >= 1.5 * m[order[1]] and m[order[0]] > 0.3:
            ds = [order[int(k)] for k in 1 + g.rng.choice(len(order) - 1, 3, replace=False)]
            mc("phys_fastest", path, s / FPS, e / FPS, g.pick(["Which object moves fastest?", "Which object is the fastest in this clip?"]), [name[order[0]].capitalize()] + [name[j].capitalize() for j in ds], 0); break
    # moving at the end
    endv = {i: np.nanmean(spd[i][-10:]) for i in P if vis[i][-10:].all()}
    mov = [i for i, v in endv.items() if v > 0.3]; sti = [i for i, v in endv.items() if v < 0.02]
    if mov and sti:
        yes = g.rng.random() < 0.5; i = (mov if yes else sti)[int(g.rng.integers(len(mov if yes else sti)))]
        yn("phys_moving", path, 0, dur, g.pick(["Is {} still moving at the end of the video?", "At the end, is {} moving?"]).format(name[i]), yes)
    # prediction from a cut clip
    cut = int(g.rng.integers(50, 86)); before = [i for i in P if vis[i][:cut].sum() >= 10]
    firstf = {}
    for p, f in col: firstf[p] = min(f, firstf.get(p, 10 ** 6))
    allp = [(x, y) for i, x in enumerate(sorted(before)) for y in sorted(before)[i + 1:]]
    pos = [p for p in allp if p in firstf and firstf[p] >= cut + 10]; neg = [p for p in allp if p not in pairs]
    if pos and neg:
        yes = g.rng.random() < 0.5; p = (pos if yes else neg)[int(g.rng.integers(len(pos if yes else neg)))]
        yn("phys_predict", path, 0, cut / FPS, g.pick(["Will {} and {} collide later?", "Are {} and {} going to collide after this?", "Predict: will {} hit {}?"]).format(name[p[0]], name[p[1]]), yes)
for kind, rows in sorted(out.items()):
    report(kind, rows); pd.DataFrame(rows).to_parquet(f"{OUT}/{kind}.parquet")
