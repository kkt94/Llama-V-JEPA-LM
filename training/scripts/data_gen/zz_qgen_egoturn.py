"""(10-25, first-person navigation without Ego4D) "which way does the camera wearer turn" from MEASURED camera poses: EPIC Fields (COLMAP
reconstructions of EPIC-KITCHENS-100, CC BY-NC 4.0). Yaw between the ends of a 2.5 s window (forward axis of the later camera in the earlier
camera frame); its sign was checked against the optical flow of the video itself (logs/zz_epicfields_check.py: 91.8 % agreement for |yaw| > 10
deg; pitch did NOT agree - hands dominate vertical flow - so no up/down questions). Labels: yaw > +20 deg right, < -20 deg left, |yaw| < 3 deg
same direction; classes balanced per video. -> manifests/stage1/opendl3/epic_turn.parquet (prep root prep/epic_384_15fps)  usage: zz_qgen_egoturn.py"""
import sys, json, glob, os, csv, numpy as np
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
import pandas as pd
R = "/data/llama_vjepa_lm"; F = f"{R}/data/opendl/epic_fields/json-format"; PREP = f"{R}/prep/epic_384_15fps"; OUT = f"{R}/manifests/stage1/opendl3"; g = Gen(79)
FPS = {r["video_id"]: (float(r["fps"]), float(r["duration"])) for r in csv.DictReader(open(f"{R}/data/opendl/epic_kitchens_100/annotations/EPIC_100_video_info.csv"))}
def rot(q):
    w, x, y, z = q; return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
PH = {"left": "The camera wearer turns to the left", "right": "The camera wearer turns to the right", "same": "The camera wearer keeps facing the same direction"}
Q = ["Which way does the camera wearer turn?", "How does my viewing direction change in this video?", "Which way do I turn in this first-person video?", "Does the camera wearer turn left, turn right, or keep facing the same way?"]
rows = []; L = 2.5
for p in sorted(glob.glob(f"{F}/*.json")):
    vid = os.path.basename(p)[:-5]
    if vid not in FPS: continue
    fps, dur = FPS[vid]; path = f"{vid.split('_')[0]}/{vid}.mp4"; ex = os.path.exists(f"{PREP}/{path}")
    fr = {int(k[6:16]): np.array(v) for k, v in json.load(open(p))["images"].items()}; keys = np.array(sorted(fr))
    got = {"left": [], "right": [], "same": []}
    for t0 in np.arange(2.0, dur - L - 2.0, 1.25):
        f0, f1 = int(round(t0 * fps)) + 1, int(round((t0 + L) * fps)) + 1
        i0, i1 = np.abs(keys - f0).argmin(), np.abs(keys - f1).argmin()
        if abs(keys[i0] - f0) > fps * 0.1 or abs(keys[i1] - f1) > fps * 0.1: continue
        fwd = rot(fr[keys[i0]][:4]) @ rot(fr[keys[i1]][:4]).T @ np.array([0, 0, 1.0]); yaw = np.degrees(np.arctan2(fwd[0], fwd[2]))
        lab = "right" if yaw > 20 else "left" if yaw < -20 else "same" if abs(yaw) < 3 else None
        if lab: got[lab].append(t0)
    k = min(len(got["left"]) + len(got["right"]), 2 * len(got["same"])) // 2
    for lab in ("left", "right", "same"):
        ts = got[lab]; take = [ts[j] for j in g.rng.permutation(len(ts))[: (k if lab != "same" else k)]]
        for t0 in take:
            opts = [PH[lab]] + [PH[o] for o in PH if o != lab]
            rows.append(g.row(path, float(t0), float(t0) + L, g.pick(Q), opts, 0, "", ex))
report("epic_turn", rows); pd.DataFrame(rows).to_parquet(f"{OUT}/epic_turn.parquet")
d = pd.DataFrame(rows); import collections, json as _j
print("answers:", collections.Counter(_j.loads(c)[1]["content"] for c in d.conversations).most_common(3), "| video exists", int(d.prep_exists.astype(bool).sum()))
