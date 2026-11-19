"""Stage-2 real-video evidence data from SA-V TRAIN manual masklets (human-verified masks, 6 fps annotation).

SA-V has no object names, so objects are referred to by WHERE they are at a given time ("the object in the top-left of
the frame at 2.0 s"), and only when no other masklet centre lies in that cell at that time. The cell is computed in the
square crop the model sees (random offset along the long side, keeping the object inside).
Ops (semantic ids continue the NVIDIA ones):
  4 sav_direction  "In which direction does the object in the <pos> at t0 s move between t0 and t1 s?"   bin of the span
  5 sav_visible    "Is the object that is in the <pos> at t s still visible at t2 s?"   bin containing t2 (t2 later)
A question that names the cell of BOTH objects ("is A left of B") would leak the answer, so it is not generated.
Gold spec = (kind, op, time bin, 3x3 cell); half the rows blank that bin x cell (masked prediction is the gold kind).
Answers balanced per op. Split: 98% train / 2% heldout by video hash (SA-V train videos only - the contamination
allowlist covers SA-V train tars). Output: manifests/stage2/sav_tracks.parquet (+ _heldout).
"""
import glob
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pycocotools.mask as mu

sys.path.insert(0, "/data/llama_vjepa_lm/training")
from jepa_llm.data.video import paired_uniform_times  # noqa: E402

R = "/data/llama_vjepa_lm"
ANN = f"{R}/data/SA-V/annotations/sav_train"
PREP = f"{R}/prep/sav_384_24fps"
DEST = f"{R}/manifests/stage2/sav_tracks.parquet"
FPS_ANN, NUM_FRAMES, N_BINS, RGRID = 6.0, 16, 4, 3
POS = ["top-left", "top", "top-right", "left", "centre", "right", "bottom-left", "bottom", "bottom-right"]
OPS = {"sav_direction": 4, "sav_visible": 5}


def boxes(d):
    """-> centres [n_obj, F, 2] (normalised x, y; nan when absent)."""
    n, F_ = int(d["masklet_num"]), len(d["masklet"])
    c = np.full((n, F_, 2), np.nan)
    W, H = d["video_width"], d["video_height"]
    for f, fr in enumerate(d["masklet"]):
        for i, rle in enumerate(fr):
            if rle is None:
                continue
            r = dict(rle); r["counts"] = r["counts"].encode() if isinstance(r["counts"], str) else r["counts"]
            x, y, w, h = mu.toBbox(r)
            if w > 0 and h > 0:
                c[i, f] = [(x + w / 2) / W, (y + h / 2) / H]
    return c


def video_rows(path):
    vid = os.path.basename(path)[:-12]
    folder = os.path.basename(os.path.dirname(path))
    rel = f"sav_train/{folder}/{vid}.mp4"
    if not os.path.exists(f"{PREP}/{rel}"):
        return []
    d = json.load(open(path))
    c = boxes(d)
    n, F_ = c.shape[:2]
    dur = F_ / FPS_ANN
    W, H = float(d["video_width"]), float(d["video_height"])
    portrait = H > W
    side = (W / H) if portrait else (H / W)              # square crop size as a fraction of the long side
    rng = np.random.default_rng(int(hashlib.md5(vid.encode()).hexdigest()[:8], 16))
    times = paired_uniform_times(dur, NUM_FRAMES, 0.25)
    bins = [times[b * 4:(b + 1) * 4] for b in range(N_BINS)]
    split = "train" if int(hashlib.md5(vid.encode()).hexdigest()[8:16], 16) % 100 < 98 else "heldout"
    out = []
    for _ in range(int(os.environ.get("TRIES", 15))):
        op = rng.choice(list(OPS))
        i = int(rng.integers(n))
        b = int(rng.integers(N_BINS - (1 if op == "sav_visible" else 0)))
        f0 = int(round(bins[b][0] * FPS_ANN)); f1 = int(round(bins[b][-1] * FPS_ANN))
        f0, f1 = min(f0, F_ - 1), min(f1, F_ - 1)
        p0 = c[i, f0]
        if np.isnan(p0).any():
            continue
        long0 = p0[1] if portrait else p0[0]
        lo, hi = max(0.0, long0 - 0.98 * side), min(1.0 - side, long0 - 0.02 * side)
        if lo > hi:
            continue
        off = float(rng.uniform(lo, hi))

        def cell(p):
            a = ((p[1] - off) / side) if portrait else ((p[0] - off) / side)   # along the long side, inside the crop
            s = p[0] if portrait else p[1]                                       # along the short side (full extent)
            if not (0 <= a < 1 and 0 <= s < 1):
                return None
            col, row = (int(s * RGRID), int(a * RGRID)) if portrait else (int(a * RGRID), int(s * RGRID))
            return row * RGRID + col + 1
        k = cell(p0)
        if k is None:
            continue
        others = [cell(c[j, f0]) for j in range(n) if j != i and not np.isnan(c[j, f0]).any()]
        if k in others:
            continue                                     # the location would not identify one object
        pos, t0 = POS[k - 1], f0 / FPS_ANN
        if op == "sav_direction":
            p1 = c[i, f1]
            if np.isnan(p1).any() or f1 - f0 < 3:
                continue
            du, dv = (p1[0] - p0[0]) * W, (p1[1] - p0[1]) * H
            scale = max(W, H) / 1920
            if max(abs(du), abs(dv)) < 60 * scale or not (abs(du) >= 1.5 * abs(dv) or abs(dv) >= 1.5 * abs(du)):
                continue
            ans = ("Right." if du > 0 else "Left.") if abs(du) > abs(dv) else ("Down." if dv > 0 else "Up.")
            q = (f"In which direction does the object in the {pos} of the frame at {t0:.1f} s move between {t0:.1f} s "
                 f"and {f1 / FPS_ANN:.1f} s? Answer left, right, up or down.")
            gb, gk = b, k
        else:
            b2 = int(rng.integers(b + 1, N_BINS))
            f2 = min(int(round(bins[b2][len(bins[b2]) // 2] * FPS_ANN)), F_ - 1)
            vis = not np.isnan(c[i, f2]).any()
            k2 = cell(c[i, f2]) if vis else None
            if vis and k2 is None:
                continue                                  # visible but outside the crop: not answerable from the input
            ans = "Yes." if vis else "No."
            q = f"Is the object that is in the {pos} of the frame at {t0:.1f} s still visible at {f2 / FPS_ANN:.1f} s?"
            gb, gk = b2, (k2 if vis else k)
        masked = bool(rng.integers(2))
        out.append({"video": rel, "prep_exists": True, "split": split, "t0": 0.0, "t1": dur,
                    "crop_cx": (off + side / 2) if not portrait else 0.5, "crop_cy": (off + side / 2) if portrait else 0.5,
                    "question": q, "answer": ans, "op": op, "gold_kind": int(masked), "gold_sem": OPS[op],
                    "gold_time_bin": gb, "gold_region": gk, "mask_bin": gb if masked else -1, "mask_cell": gk if masked else -1,
                    "group_id": f"sav|{op}|{pos}", "pair_id": -1, "pair_type": 0})
    return out


def main():
    fs = sorted(glob.glob(f"{ANN}/*/*_manual.json"))
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 8))) as ex:
        rows = [r for rs in ex.map(video_rows, fs, chunksize=64) for r in rs]
    df = pd.DataFrame(rows).drop_duplicates(["video", "question"])
    keep = []
    for op, g in df.groupby("op"):
        cnt = g.answer.value_counts(); cap = cnt.min() if op == "sav_visible" else int(1.5 * cnt.min())
        for _, gg in g.groupby("answer"):
            keep += gg.sample(min(cap, len(gg)), random_state=0).index.tolist()
    df = df.loc[sorted(keep)].reset_index(drop=True)
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    df[df.split == "train"].to_parquet(DEST, index=False)
    df[df.split != "train"].to_parquet(DEST.replace(".parquet", "_heldout.parquet"), index=False)
    tr = df[df.split == "train"]
    print(f"videos {len(fs)} rows {len(df)} train {len(tr)} masked {int(tr.gold_kind.sum())}")
    print(tr.groupby(["op", "answer"]).size().to_string()); print("cells", tr.gold_region.value_counts().sort_index().to_dict())
    print("bins", tr.gold_time_bin.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
