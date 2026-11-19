"""(10-25) EgoDex (Apple, CC BY-NC-ND; Vision Pro first-person, 30 fps, per-frame MEASURED camera + 25-joint hand poses). Answers only from the
collection-protocol task name and the measured poses; the description / llm_* attributes are NOT used. Convention checked by projecting the wrist
and fingertips onto the frames (camera looks along +z, y down; logs/zz_qgen_egodex.py header, scratchpad egodex_peek/proj.jpg).
 task   - which of the 101 recorded tasks (whole episode, <= 12 s window)
 hdir   - how one hand (left/right named) moves on screen over 2 s: left / right / up / down / stays still (image displacement of the wrist,
          moving: dominant axis > 15 % of image width and > 2x the other axis; still: < 3 %)
 hmore  - which hand moves more over 3 s (3D wrist path; ratio > 3, both > 15 cm with ratio < 1.25 = both about equally)
 turn   - camera wearer turns left / right / keeps facing (yaw over 2.5 s, same thresholds as epic_turn)
Wrist confidence >= 0.5 over the window.  -> manifests/stage1/opendl3/egodex_<split>.parquet (prep root prep/egodex_384_15fps)
usage: zz_qgen_egodex.py <split> [<split> ..]"""
import sys, glob, os, h5py, numpy as np, pandas as pd, collections, json
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
R = "/data/llama_vjepa_lm"; B = f"{R}/data/opendl/egodex"; PREP = f"{R}/prep/egodex_384_15fps"; OUT = f"{R}/manifests/stage1/opendl3"; g = Gen(83)
PAIR = {"add", "remove", "assemble", "disassemble", "open", "close", "insert", "stack", "unstack", "fold", "unfold", "charge", "uncharge", "clip",
        "unclip", "crumple", "flatten", "deal", "gather", "screw", "unscrew", "sleeve", "unsleeve", "thread", "unthread", "tie", "untie", "zip", "unzip",
        "wrap", "unwrap", "lock", "unlock", "throw", "catch", "collect", "braid", "unbraid", "build", "paint", "clean", "put", "load", "dispense",
        "scoop", "dump", "peel", "place", "pick", "play", "reset", "stock", "unstock", "wash", "rake", "smooth", "roll", "knead", "fry", "boil", "serve"}
def human(t):
    w = t.split("_"); s = w[0]
    for a, b in zip(w, w[1:]): s += ("/" if a in PAIR and b in PAIR else " ") + b
    return s.replace("basic ", "")
FPS = 30.0
Q = {"task": ["What task is the person doing?", "Which activity is shown in this first-person video?", "What am I doing in this video?"],
     "turn": ["Which way does the camera wearer turn?", "How does my viewing direction change in this video?"],
     "hmore": ["Which hand moves more in this video?", "Which of my hands is moving more?"]}
DIRS = {"left": "moves to the left", "right": "moves to the right", "up": "moves up", "down": "moves down", "still": "stays roughly still"}
TURN = {"left": "The camera wearer turns to the left", "right": "The camera wearer turns to the right", "same": "The camera wearer keeps facing the same direction"}
HM = {"left": "The left hand", "right": "The right hand", "both": "Both hands about equally"}
def yaw(Ca, Cb):
    f = Ca[:3, :3].T @ Cb[:3, :3] @ np.array([0, 0, 1.0]); return np.degrees(np.arctan2(f[0], f[2]))
for split in sys.argv[1:]:
    files = sorted(glob.glob(f"{B}/{split}/*/*.hdf5")); tasks = sorted({p.split("/")[-2] for p in files}); rows = []; kinds = collections.Counter()
    for p in files:
        task = p.split("/")[-2]; rel = f"{split}/{task}/{os.path.basename(p)[:-5]}.mp4"; ex = os.path.exists(f"{PREP}/{rel}")
        try:
            f = h5py.File(p, "r"); C = f["transforms/camera"][:]; K = f["camera/intrinsic"][:]; n = len(C)
            H = {h: f[f"transforms/{h}Hand"][:, :3, 3] for h in ("left", "right")}
            Cf = {h: f[f"confidences/{h}Hand"][:] for h in ("left", "right")} if "confidences" in f else None   # some tasks ship no confidences: no hand questions
        except Exception as e: kinds["read_err"] += 1; continue
        dur = n / FPS; W = 2 * K[0, 2]
        if dur < 2.0: kinds["short"] += 1; continue
        t0 = float(g.rng.uniform(0, max(0.0, dur - 12.0))); t1 = min(dur, t0 + 12.0)
        others = [human(t) for t in g.rng.permutation([t for t in tasks if t != task])[:3]]
        rows.append(g.row(rel, t0, t1, g.pick(Q["task"]), [human(task)] + others, 0, "", ex)); kinds["task"] += 1
        def img(h, i):
            x, y, z = (np.linalg.inv(C[i]) @ np.append(H[h][i], 1.0))[:3]; return (np.array([K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]]) if z > 0.05 else None)
        if Cf is None: kinds["no_conf"] += 1
        for a in (np.arange(0.0, dur - 2.0, 1.0) if Cf is not None else []):           # hand direction on screen, 2 s
            i0, i1 = int(a * FPS), min(n - 1, int((a + 2.0) * FPS))
            for h in ("left", "right"):
                if Cf[h][i0:i1 + 1].min() < 0.5: continue
                u0, u1 = img(h, i0), img(h, i1)
                if u0 is None or u1 is None or not (0 <= u0[0] < W and 0 <= u1[0] < W): continue
                dx, dy = (u1 - u0) / W
                lab = ("right" if dx > 0 else "left") if abs(dx) > 0.15 and abs(dx) > 2 * abs(dy) else ("down" if dy > 0 else "up") if abs(dy) > 0.15 and abs(dy) > 2 * abs(dx) else "still" if np.hypot(dx, dy) < 0.03 else None
                if lab is None or (lab == "still" and g.rng.random() > 0.25): continue
                opts = [DIRS[lab]] + [DIRS[o] for o in g.rng.permutation([o for o in DIRS if o != lab])[:3]]
                rows.append(dict(g.row(rel, a, a + 2.0, f"How does my {h} hand move in the view?", [f"The {h} hand {o}" for o in opts], 0, "", ex), _k="hdir_" + lab)); kinds["hdir_" + lab] += 1
        for a in (np.arange(0.0, dur - 3.0, 1.5) if Cf is not None else []):           # which hand moves more, 3 s
            i0, i1 = int(a * FPS), min(n - 1, int((a + 3.0) * FPS))
            if min(Cf["left"][i0:i1 + 1].min(), Cf["right"][i0:i1 + 1].min()) < 0.5: continue
            L = {h: np.linalg.norm(np.diff(H[h][i0:i1 + 1:3], axis=0), axis=1).sum() for h in ("left", "right")}
            lab = "left" if L["left"] > 3 * L["right"] and L["left"] > 0.15 else "right" if L["right"] > 3 * L["left"] and L["right"] > 0.15 else "both" if min(L.values()) > 0.15 and max(L.values()) < 1.25 * min(L.values()) else None
            if lab: rows.append(dict(g.row(rel, a, a + 3.0, g.pick(Q["hmore"]), [HM[lab]] + [HM[o] for o in HM if o != lab], 0, "", ex), _k="hmore_" + lab)); kinds["hmore_" + lab] += 1
        for a in np.arange(0.0, dur - 2.5, 1.25):                                    # head turn, 2.5 s
            y = yaw(C[int(a * FPS)], C[min(n - 1, int((a + 2.5) * FPS))])
            lab = "right" if y > 20 else "left" if y < -20 else "same" if abs(y) < 3 and g.rng.random() < 0.1 else None
            if lab: rows.append(dict(g.row(rel, a, a + 2.5, g.pick(Q["turn"]), [TURN[lab]] + [TURN[o] for o in TURN if o != lab], 0, "", ex), _k="turn_" + lab)); kinds["turn_" + lab] += 1
    # balance the answers within each kind: every class capped at 1.5x the rarest class of that kind (no class can be guessed from priors)
    by = collections.defaultdict(list)
    for r in rows: by[r.get("_k", "task").rsplit("_", 1)[0] if "_k" in r else "task"].append(r)
    keep = []
    for kd, rr in by.items():
        if kd == "task": keep += rr; continue
        cl = collections.defaultdict(list)
        for r in rr: cl[r["_k"]].append(r)
        cap = int(1.5 * min(len(v) for v in cl.values())) if len(cl) == len({"hdir": DIRS, "hmore": HM, "turn": TURN}[kd]) else 0
        for k_, v in cl.items(): keep += [v[j] for j in g.rng.permutation(len(v))[:cap]]; kinds["kept_" + k_] = min(cap, len(v))
    rows = [{k: v for k, v in r.items() if k != "_k"} for r in keep]
    report(f"egodex_{split}", rows); print(dict(sorted(kinds.items())), "| video exists", sum(r["prep_exists"] for r in rows), "tasks", len(tasks))
    pd.DataFrame(rows).to_parquet(f"{OUT}/egodex_{split}.parquet")
