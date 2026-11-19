"""(10-23, nx3 prep) PLM-Video-Human RCap (179k region captions on SA-V train videos) -> training manifest.
Each row: the caption's time window [start_frame, end_frame] of the 6 fps annotation as the clip (t0, t1), the square crop centred on the
subject at the window's first visible frame (crop_cx, crop_cy), and the subject's box at that frame in percent of the CROP the model sees.
Segments under 1 s (6 annotation frames) are dropped.
Prompt: describe what the subject in that box does; answer = the human caption. 98% / 2% split by video hash (as build_stage2_sav.py).
-> manifests/stage1/rcap.parquet (+ rcap_heldout.parquet)
usage: zz_make_rcap.py [--limit N]"""
import hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import pycocotools.mask as mu

R = "/data/llama_vjepa_lm"
ANN = f"{R}/data/SA-V/annotations/sav_train"; PREP = f"{R}/prep/sav_384_24fps"; FPS = 6.0
TEMPL = ["Describe what the subject in the region {b} does in this video.",
         "What does the subject in the region {b} do over the course of the video?",
         "Describe the actions of the subject in the region {b}.",
         "How does the subject in the region {b} move and act in this video?"]
NOTE = " (The region is given as [left, top, right, bottom] in percent of the frame, at the start of the video.)"


def per_video(args):
    vid, rows = args
    folder = f"sav_{int(vid[4:]) // 1000:03d}"
    rel = f"sav_train/{folder}/{vid}.mp4"
    ap = f"{ANN}/{folder}/{vid}_manual.json"
    if not (os.path.exists(f"{PREP}/{rel}") and os.path.exists(ap)):
        return [], len(rows)
    d = json.load(open(ap)); W, H = float(d["video_width"]), float(d["video_height"]); F_ = len(d["masklet"])
    ids = list(d["masklet_id"]); out, bad = [], 0
    h = int(hashlib.md5(vid.encode()).hexdigest()[8:16], 16) % 100
    for r in rows:
        if r["masklet_id"] not in ids:
            bad += 1; continue
        i = ids.index(r["masklet_id"]); s, e = int(r["start_frame"]), min(int(r["end_frame"]), F_ - 1)
        box = None
        for f in range(s, e + 1):                       # first frame of the window where the subject is visible
            rle = d["masklet"][f][i]
            if rle is None:
                continue
            q = dict(rle); q["counts"] = q["counts"].encode() if isinstance(q["counts"], str) else q["counts"]
            x, y, w, hh = mu.toBbox(q)
            if w > 1 and hh > 1:
                box, f0 = (x, y, x + w, y + hh), f; break
        if box is None or e - s < 5:                     # (also) captions of segments under 1 s ("barely visible for a frame")
            bad += 1; continue
        cx, cy = (box[0] + box[2]) / 2 / W, (box[1] + box[3]) / 2 / H
        side = min(W, H)                                  # the square crop the model sees (video._finish with crop_center)
        x0 = min(max(cx * W - side / 2, 0), W - side); y0 = min(max(cy * H - side / 2, 0), H - side)
        bx = [np.clip((box[0] - x0) / side, 0, 1), np.clip((box[1] - y0) / side, 0, 1), np.clip((box[2] - x0) / side, 0, 1), np.clip((box[3] - y0) / side, 0, 1)]
        full = (box[2] - box[0]) * (box[3] - box[1]); vis = (bx[2] - bx[0]) * (bx[3] - bx[1]) * side * side
        if vis < 0.5 * full:                              # subject mostly outside the square: skip
            bad += 1; continue
        b = "[" + ", ".join(str(int(round(100 * v))) for v in bx) + "]"
        t = TEMPL[int(hashlib.md5(f"{vid}{r['uid']}".encode()).hexdigest()[:4], 16) % len(TEMPL)].format(b=b) + NOTE
        conv = [{"role": "user", "content": "<|jepa_evidence|>\n" + t}, {"role": "assistant", "content": str(r["caption"]).strip()}]
        out.append({"video": rel, "prep_exists": True, "t0": s / FPS, "t1": (e + 1) / FPS, "crop_cx": float(cx), "crop_cy": float(cy),
                    "kind": "rcap", "op": "rcap", "split": "train" if h < 98 else "heldout", "conversations": json.dumps(conv)})
    return out, bad


if __name__ == "__main__":
    df = pd.read_parquet(f"{R}/data/PLM-Video-Human/rcap/plm_rcap_train.parquet")
    if "--limit" in sys.argv:
        df = df.head(int(sys.argv[sys.argv.index("--limit") + 1]))
    groups = [(v.replace(".mp4", ""), g.to_dict("records")) for v, g in df.groupby("video")]
    rows, bad = [], 0
    with ProcessPoolExecutor(32) as ex:
        for o, b_ in ex.map(per_video, groups, chunksize=64):
            rows += o; bad += b_
    m = pd.DataFrame(rows)
    sfx = "_smoke" if "--limit" in sys.argv else ""
    for sp, name in (("train", "rcap"), ("heldout", "rcap_heldout")):
        part = m[m.split == sp].drop(columns="split").reset_index(drop=True)
        part.to_parquet(f"{R}/manifests/stage1/{name}{sfx}.parquet")
        print(name + sfx, len(part), "videos", part.video.nunique())
    print("input", len(df), "kept", len(m), "dropped", bad)
    print(m.iloc[0].to_dict())
