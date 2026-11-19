"""(10-25, first-person / camera motion weakness: MotionBench Camera Motion, MVBench egocentric navigation) camera-motion questions by a KNOWN
transformation (video.compose_clip kind "camera": a window slides / scales over a static-camera source; verified on a synthetic marker video:
pan_left -> content drifts right, tilt_up -> content drifts down, zoom_in -> content leaves the frame outward). Sources with a fixed camera:
CLEVRER (synthetic) and the third-person half of Charades-Ego. One 3-6 s window per source video, move drawn uniformly over 7, travel
"amt" in [0.4, 0.9]. Options: the move, its opposite (hard negative) and two others (static's opposite = a random move).
-> manifests/stage1/opendl3/{camera_clevrer,camera_chexo}.parquet   usage: zz_qgen_camera.py"""
import sys, os, glob, json
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
import pandas as pd
R = "/data/llama_vjepa_lm"; OUT = f"{R}/manifests/stage1/opendl3"; os.makedirs(OUT, exist_ok=True); g = Gen(73)
PH = {"pan_left": "The camera pans to the left", "pan_right": "The camera pans to the right", "tilt_up": "The camera tilts upward", "tilt_down": "The camera tilts downward",
      "zoom_in": "The camera zooms in", "zoom_out": "The camera zooms out", "static": "The camera stays still"}
OPP = {"pan_left": "pan_right", "pan_right": "pan_left", "tilt_up": "tilt_down", "tilt_down": "tilt_up", "zoom_in": "zoom_out", "zoom_out": "zoom_in"}
Q = ["How does the camera move in this video?", "What is the camera motion in this video?", "Which camera movement is shown?", "How does the viewpoint change over the video?"]
MOVES = list(PH)
def rows_for(paths, root, dur_of):
    out = []
    for p in paths:
        D = dur_of(p)
        if D is None or D < 3.5: continue
        L = float(g.rng.uniform(3.0, min(6.0, D))); t0 = float(g.rng.uniform(0, D - L)); mv = MOVES[int(g.rng.integers(len(MOVES)))]
        opp = OPP.get(mv) or MOVES[int(g.rng.integers(len(MOVES) - 1))]
        rest = [m for m in MOVES if m not in (mv, opp)]; others = [rest[j] for j in g.rng.permutation(len(rest))[:2]]
        r = g.row(os.path.relpath(p, root), t0, t0 + L, g.pick(Q), [PH[mv], PH[opp]] + [PH[o] for o in others], 0, "", True)
        r["tq"] = json.dumps({"kind": "camera", "move": mv, "amt": round(float(g.rng.uniform(0.4, 0.9)), 2)}); out.append(r)
    return out
import decord
def dur(p):
    try: vr = decord.VideoReader(p, num_threads=1); return len(vr) / float(vr.get_avg_fps())
    except Exception: return None
CL = f"{R}/data/clevrer/video_train"; cl = sorted(glob.glob(f"{CL}/*/*.mp4"))[::2]                       # every other CLEVRER video (~5k)
rc = rows_for(cl, CL, lambda p: 5.12)                                                                  # CLEVRER clips are 128 frames at 25 fps
report("camera_clevrer", rc); pd.DataFrame(rc).to_parquet(f"{OUT}/camera_clevrer.parquet")
CH = f"{R}/data/opendl/charades_ego/CharadesEgo_v1_480"; ch = sorted(p for p in glob.glob(f"{CH}/*.mp4") if not os.path.basename(p).split(".")[0].endswith("EGO"))
rh = rows_for(ch, CH, dur); report("camera_chexo", rh); pd.DataFrame(rh).to_parquet(f"{OUT}/camera_chexo.parquet")
print({m: sum(json.loads(r["tq"])["move"] == m for r in rc + rh) for m in MOVES})
