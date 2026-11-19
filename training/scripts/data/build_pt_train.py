"""PerceptionTest TRAIN split -> (a) stage-1 / general multiple-choice QA, (b) stage-2 real-video evidence data.

Only the official train split is used (validation is an evaluation benchmark; MVP uses validation videos only).
(a) manifests/stage1/pt_train_mc.parquet: mc_question_train with the same prompt the evaluator uses (options listed,
    "Answer with the option's letter...") and the letter as the answer; cut_frame_mapping_train applied as the visible
    window [0, cut). This is the data V-JEPA 2.1 post-trains on, so PerceptionTest-valid becomes in-distribution.
(b) manifests/stage2/pt_tracks.parquet: questions from the human object tracks (boxes every 30 frames, labels, is_masked):
      lr_at_t   "At t s, is the <label A> to the left of the <label B> on screen?"   (labels unique in the video)
      direction "In which direction does the <label> move on screen between t0 and t1?"  (static-camera videos only)
    Observed clip = an 8 s window (16 frames, 4 bins of 2 s), square crop at a random offset keeping the objects inside,
    gold spec = (kind, op, bin, 3x3 cell); half the rows blank that bin x cell in pixels so masked prediction is the gold
    kind. No pairs (single camera). Boxes flagged is_masked at the query frame are not used.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/llama_vjepa_lm/training")
from jepa_llm.data.chat_format import VIS  # noqa: E402
from jepa_llm.data.video import paired_uniform_times  # noqa: E402

R = "/data/llama_vjepa_lm"
SRC = f"{R}/data/perception_test_train/extracted"
PREP = f"{R}/prep/pt_train_384"
LETTERS = "ABCDEFGHIJ"
SUFFIX = "Answer with the option's letter from the given choices directly."
OPS = {"lr_at_t": 0, "direction": 1}
WIN, NUM_FRAMES, N_BINS, RGRID = 8.0, 16, 4, 3
CROP_W = 1080 / 1920


def mc_rows():
    d = json.load(open(f"{SRC}/mc_question_train.json"))
    cut = json.load(open(f"{R}/bench/perception_test/cut_frame_mapping_train.json"))
    rows = []
    for vid, v in d.items():
        fps = float(v["metadata"]["frame_rate"]); nf = int(v["metadata"]["num_frames"])
        cf = int(cut.get(vid, -1))
        t1 = (cf if cf > 0 else nf) / fps
        for q in v["mc_question"]:
            opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"]))
            prompt = f"{q['question']}\n{opts}\n{SUFFIX}"
            ans = f"{LETTERS[int(q['answer_id'])]}."
            rows.append({"video": f"{vid}.mp4", "prep_exists": os.path.exists(f"{PREP}/{vid}.mp4"), "t0": 0.0, "t1": t1,
                         "region": [], "area": q.get("area", ""),
                         "conversations": json.dumps([{"role": "user", "content": f"{VIS}\n{prompt}"},
                                                      {"role": "assistant", "content": ans}])})
    return pd.DataFrame(rows)


def track_rows():
    d = json.load(open(f"{SRC}/all_train.json"))
    rng = np.random.default_rng(0)
    rows = []
    for vid, v in d.items():
        md = v["metadata"]; fps = float(md["frame_rate"]); nf = int(md["num_frames"]); dur = nf / fps
        if dur < WIN + 1:
            continue
        tracks = v.get("object_tracking", [])
        labels = pd.Series([t["label"] for t in tracks]).value_counts()
        uniq = [t for t in tracks if labels[t["label"]] == 1 and t["label"] not in ("person", "table")]
        if not uniq:
            continue
        box = {}
        for t in uniq:
            for f, b, m in zip(t["frame_ids"], t["bounding_boxes"], t["is_masked"]):
                if not m:
                    box.setdefault(t["label"], {})[int(f)] = np.asarray(b, dtype=float)   # x0, y0, x1, y1 normalised
        for _ in range(int(os.environ.get("TRIES", 40))):
            w0 = float(rng.uniform(0, dur - WIN))
            times = w0 + paired_uniform_times(WIN, NUM_FRAMES, 0.25)
            b = int(rng.integers(N_BINS))
            bt = times[b * 4:(b + 1) * 4]
            op = "lr_at_t" if (md.get("is_camera_moving") or rng.random() < 0.5) else "direction"
            frames_in = lambda lab: sorted(f for f in box.get(lab, {}) if bt[0] - 0.15 <= f / fps <= bt[-1] + 0.15)
            if op == "lr_at_t":
                cand = [l for l in box if frames_in(l)]
                common = None
                if len(cand) < 2:
                    continue
                a, c = rng.choice(cand, 2, replace=False)
                common = sorted(set(frames_in(a)) & set(frames_in(c)))
                if not common:
                    continue
                f = common[len(common) // 2]
                ca, cc = box[a][f], box[c][f]
                ua, uc = (ca[0] + ca[2]) / 2, (cc[0] + cc[2]) / 2
                if abs(ua - uc) < 0.05:
                    continue
                q = f"At {f / fps - w0:.1f} s, is the {a} to the left of the {c} on screen?"
                ans = "Yes." if ua < uc else "No."
                pts = [((ca[0] + ca[2]) / 2, (ca[1] + ca[3]) / 2), ((cc[0] + cc[2]) / 2, (cc[1] + cc[3]) / 2)]
                group = f"pt|lr|{'-'.join(sorted([a, c]))}"
            else:
                cand = [l for l in box if len(frames_in(l)) >= 2]
                if not cand:
                    continue
                a = rng.choice(cand)
                fs = frames_in(a)
                c0, c1 = box[a][fs[0]], box[a][fs[-1]]
                du = (c1[0] + c1[2] - c0[0] - c0[2]) / 2 * 1920
                dv = (c1[1] + c1[3] - c0[1] - c0[3]) / 2 * 1080
                if max(abs(du), abs(dv)) < 60 or not (abs(du) >= 1.5 * abs(dv) or abs(dv) >= 1.5 * abs(du)):
                    continue
                ans = ("Right." if du > 0 else "Left.") if abs(du) > abs(dv) else ("Down." if dv > 0 else "Up.")
                q = (f"In which direction does the {a} move on screen between {fs[0] / fps - w0:.1f} s and "
                     f"{fs[-1] / fps - w0:.1f} s? Answer left, right, up or down.")
                pts = [((c0[0] + c0[2]) / 2, (c0[1] + c0[3]) / 2), ((c1[0] + c1[2]) / 2, (c1[1] + c1[3]) / 2)]
                group = f"pt|dir|{a}"
            us = [p[0] for p in pts]
            lo, hi = max(0.0, max(us) - 0.98 * CROP_W), min(1.0 - CROP_W, min(us) - 0.02 * CROP_W)
            if lo > hi:
                continue
            x0 = float(rng.uniform(lo, hi))
            cells = {int(p[1] * RGRID) * RGRID + int((p[0] - x0) / CROP_W * RGRID) + 1 for p in pts}
            if len(cells) != 1:
                continue
            cell = cells.pop()
            masked = bool(rng.integers(2))
            rows.append({"video": f"{vid}.mp4", "prep_exists": os.path.exists(f"{PREP}/{vid}.mp4"), "t0": w0, "t1": w0 + WIN,
                         "crop_cx": x0 + CROP_W / 2, "question": q, "answer": ans, "op": op, "gold_kind": int(masked),
                         "gold_sem": OPS[op], "gold_time_bin": b, "gold_region": cell, "mask_bin": b if masked else -1,
                         "mask_cell": cell if masked else -1, "group_id": group, "pair_id": -1, "pair_type": 0})
    return pd.DataFrame(rows).drop_duplicates(["video", "question"]).reset_index(drop=True)


def main():
    os.makedirs(f"{R}/manifests/stage2", exist_ok=True)
    mc = mc_rows()
    mc.to_parquet(f"{R}/manifests/stage1/pt_train_mc.parquet", index=False)
    print(f"pt_train_mc rows {len(mc)} videos {mc.video.nunique()} prep {int(mc.prep_exists.sum())}")
    tr = track_rows()
    # answer balance per op (yes/no equal; direction capped at 1.5x the rarest)
    keep = []
    for op, g in tr.groupby("op"):
        c = g.answer.value_counts(); cap = c.min() if op != "direction" else int(1.5 * c.min())
        for _, gg in g.groupby("answer"):
            keep += gg.sample(min(cap, len(gg)), random_state=0).index.tolist()
    tr = tr.loc[sorted(keep)].reset_index(drop=True)
    tr.to_parquet(f"{R}/manifests/stage2/pt_tracks.parquet", index=False)
    print(f"pt_tracks rows {len(tr)} videos {tr.video.nunique()} masked {int(tr.gold_kind.sum())}")
    print(tr.groupby(["op", "answer"]).size().to_string()); print("cells", tr.gold_region.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
