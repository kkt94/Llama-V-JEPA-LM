"""(10-23, nx3 prep) PLM-Video-Human RDCap (117k dense region captions on SA-V train videos) -> training manifest.
Each row: the whole video, the square crop centred on the subject at its first visible annotated frame, the subject's box at that frame
(percent of the crop the model sees) and its time; the answer = every (start, end, caption) of the subject over the whole video, in the
time format of plm_rtloc ("From 5.0s to 12.2s: ..."), including the periods it is out of view. Same 98 / 2 video split as RCap.
-> manifests/stage1/rdcap.parquet (+ rdcap_heldout.parquet)"""
import ast, hashlib, json, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
import pycocotools.mask as mu

R = "/data/llama_vjepa_lm"
ANN = f"{R}/data/SA-V/annotations/sav_train"; PREP = f"{R}/prep/sav_384_24fps"; FPS = 6.0
TEMPL = ["Describe, with start and end times in seconds, everything the subject in the region {b} at {t:.1f}s does over the whole video.",
         "Give a dense, time-stamped description of the subject in the region {b} at {t:.1f}s, covering the whole video.",
         "For the subject in the region {b} at {t:.1f}s, list the events it is involved in with their start and end times in seconds."]
NOTE = " (The region is given as [left, top, right, bottom] in percent of the frame.)"


def per_video(args):
    vid, rows = args
    folder = f"sav_{int(vid[4:]) // 1000:03d}"
    rel = f"sav_train/{folder}/{vid}.mp4"; ap = f"{ANN}/{folder}/{vid}_manual.json"
    if not (os.path.exists(f"{PREP}/{rel}") and os.path.exists(ap)):
        return [], len(rows)
    d = json.load(open(ap)); W, H = float(d["video_width"]), float(d["video_height"]); F_ = len(d["masklet"])
    ids = [int(x) for x in d["masklet_id"]]; out, bad = [], 0
    split = "train" if int(hashlib.md5(vid.encode()).hexdigest()[8:16], 16) % 100 < 98 else "heldout"
    for r in rows:
        m = int(r["masklet_id"])
        if m not in ids:
            bad += 1; continue
        i = ids.index(m); box = None
        for f in range(F_):
            rle = d["masklet"][f][i]
            if rle is None:
                continue
            q = dict(rle); q["counts"] = q["counts"].encode() if isinstance(q["counts"], str) else q["counts"]
            x, y, w, h = mu.toBbox(q)
            if w > 1 and h > 1:
                box, f0 = (x, y, x + w, y + h), f; break
        try:
            segs = r["dense_captions"]; segs = ast.literal_eval(segs.replace("}\n {", "}, {")) if isinstance(segs, str) else list(segs)
        except Exception:
            bad += 1; continue
        if box is None or not segs:
            bad += 1; continue
        cx, cy = (box[0] + box[2]) / 2 / W, (box[1] + box[3]) / 2 / H; side = min(W, H)
        x0 = min(max(cx * W - side / 2, 0), W - side); y0 = min(max(cy * H - side / 2, 0), H - side)
        bx = [np.clip((box[0] - x0) / side, 0, 1), np.clip((box[1] - y0) / side, 0, 1), np.clip((box[2] - x0) / side, 0, 1), np.clip((box[3] - y0) / side, 0, 1)]
        if (bx[2] - bx[0]) * (bx[3] - bx[1]) * side * side < 0.5 * (box[2] - box[0]) * (box[3] - box[1]):
            bad += 1; continue
        b = "[" + ", ".join(str(int(round(100 * v))) for v in bx) + "]"
        t = TEMPL[int(hashlib.md5(f"{vid}{r['uid']}".encode()).hexdigest()[:4], 16) % len(TEMPL)].format(b=b, t=f0 / FPS) + NOTE
        ans = "\n".join(f"From {int(s['start_frame']) / FPS:.1f}s to {(int(s['end_frame']) + 1) / FPS:.1f}s: {str(s['caption']).strip()}"
                        for s in sorted(segs, key=lambda s: int(s["start_frame"])))
        conv = [{"role": "user", "content": "<|jepa_evidence|>\n" + t}, {"role": "assistant", "content": ans}]
        out.append({"video": rel, "prep_exists": True, "t0": 0.0, "t1": F_ / FPS, "crop_cx": float(cx), "crop_cy": float(cy),
                    "kind": "rdcap", "op": "rdcap", "split": split, "conversations": json.dumps(conv)})
    return out, bad


if __name__ == "__main__":
    df = pd.read_parquet(f"{R}/data/PLM-Video-Human/rdcap/plm_rdcap_train.parquet")
    groups = [(v.replace(".mp4", ""), g.to_dict("records")) for v, g in df.groupby("video")]
    rows, bad = [], 0
    with ProcessPoolExecutor(32) as ex:
        for o, b_ in ex.map(per_video, groups, chunksize=64):
            rows += o; bad += b_
    m = pd.DataFrame(rows)
    for sp, name in (("train", "rdcap"), ("heldout", "rdcap_heldout")):
        part = m[m.split == sp].drop(columns="split").reset_index(drop=True)
        part.to_parquet(f"{R}/manifests/stage1/{name}.parquet"); print(name, len(part), "videos", part.video.nunique())
    print("input", len(df), "kept", len(m), "dropped", bad)
    print(json.loads(m.iloc[0].conversations))
