"""Stage-1 physics-state QA from the NVIDIA synthetic scenes (spec 8.3-8.4): answers computed from ground-truth
trajectories, evidence given as a time window + image region (gold evidence for the stage-1 readout).

Sources (built by nvidia_index.py, conventions in manifests/nvidia/report.md and nvidia_common.py):
  traj/<scenario>.npz   world-frame COM per USDA-resolved body (rows already matched to objects), dynamic flags
  world_index.parquet   per (world, camera): usd_transform (camera->world), fxn, ppx, ppy, frame_count, fps
Event definitions are nvidia_pairs.body_events (onset 1 cm, rest 0.05 m/s, stop = >= 10 rest frames before the end,
drop = COM 2 cm below start). Image-plane positions = pinhole projection of the COM (nvidia_common.project).
Objects with any per-frame COM jump > 0.5 m (15 m/s; the NVIDIA report finds such jumps in 100% of ball_mixer, 32% of
towers, 21% of bowling and 11% of billiards clips, e.g. balls falling off the table to z = -110 m) are never asked about.
No contact labels, no occlusion reasoning: an object counts as "in view" when its projected COM is inside the frame and in
front of the camera (it may still be occluded - a known limitation of v1).

Question families (one or two per clip, balanced answers):
  lr_at_t     "At 2.3 s, is the A to the left of the B on screen?"         window t+-0.5 s, region = both objects
  first_mover "Which starts moving first, the A or the B?"                  whole clip, region = both objects
  direction   "In which direction does the A move on screen between ..."     that span, region = A's path
  still_moving "Is the A still moving at the end of the video?"              last 1.5 s, region = A
  falls       "Does the A fall or topple during the video?"                 whole clip, region = A
Objects are named "<material> <noun>" from the prim name; objects whose name is not unique in the world are skipped.
Split: by world (all cameras of a world together), 90/5/5 by hash; only `train` rows feed stage 1 (others kept for
mechanism evaluation). Output: manifests/stage1/nvidia_state.parquet (train only) and nvidia_state_heldout.parquet (val/test).
"""
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/llama_vjepa_lm/training")
from nvidia_common import OUT, SCENARIOS, project, world_to_cam  # noqa: E402
from nvidia_pairs import body_events  # noqa: E402
from jepa_llm.data.chat_format import VIS  # noqa: E402

W_IMG, H_IMG = 1920, 1080
MAX_STEP_M = 0.5                     # per-frame COM jump above which a trajectory is treated as non-physical
PREP = "/data/llama_vjepa_lm/prep/nvidia_rgb_384"
DEST = "/data/llama_vjepa_lm/manifests/stage1/nvidia_state.parquet"
MATS = ["clear_plastic", "plastic", "glass", "metal", "wood", "rubber", "stone", "ceramic", "paper", "foam", "steel"]
COMPOUND = {"diningchair": "dining chair", "redonion": "red onion", "whitepackerbottle": "white bottle",
            "pumpkinlarge": "large pumpkin", "plywoodcrateassembly": "plywood crate", "stdwoodcrateasm": "wooden crate",
            "cubebox": "cube box"}
DROP_TOK = re.compile(r"^(S\d+|Prop|L\d+|A\d+|N\d+|\d+|[a-z]\d+|North|South|East|West|Left|Right|asm|assembly)$", re.I)


def object_name(prim):
    name = str(prim).split("/")[-1]
    mat = ""
    for m in MATS:
        if name.lower().endswith("_" + m):
            mat, name = m.replace("_", " "), name[: -len(m) - 1]
            break
    toks = [t for t in name.split("_") if t and not DROP_TOK.match(t)]
    words = []
    for t in toks:
        if t.lower() in MATS and not mat:
            mat = t.lower()
            continue
        t = re.sub(r"\d+$", "", t)
        words += [w.lower() for w in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", t)] if t else []
    words = " ".join(COMPOUND.get(w, w) for w in words).split()
    noun = " ".join(words[-2:]) if words else ""
    return (f"{mat} {noun}".strip() if noun else ""), noun


def split_of(world):
    h = int(hashlib.md5(world.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 90 else ("val" if h < 95 else "test")


def bbox(us, vs, margin=0.06):
    x0, x1 = float(np.min(us)) / W_IMG - margin, float(np.max(us)) / W_IMG + margin
    y0, y1 = float(np.min(vs)) / H_IMG - margin, float(np.max(vs)) / H_IMG + margin
    return [max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)]


def row(world, cam, sc, fam, q, a, t0, t1, region, extra):
    return {"scenario": sc, "world": world, "camera": cam, "video": f"{sc}/{world}/{cam}.mp4", "split": split_of(world),
            "family": fam, "question": q, "answer": a, "t0": float(t0), "t1": float(t1), "region": region,
            "conversations": json.dumps([{"role": "user", "content": f"{VIS}\n{q}"}, {"role": "assistant", "content": a}]),
            "extra": json.dumps(extra)}


def clip_questions(rng, world, cam, sc, names, tr, ev, M, fxn, ppx, ppy, T, fps):
    pc = world_to_cam(tr, M)                                   # [n, T, 3]
    u, v, front = project(pc, fxn, W_IMG, H_IMG, ppx, ppy)
    inside = front & (u >= 0) & (u < W_IMG) & (v >= 0) & (v < H_IMG)
    dur = T / fps
    named = [i for i, n in enumerate(names) if n]
    out = []
    fams = rng.permutation(["lr_at_t", "first_mover", "direction", "still_moving", "falls"])
    for fam in fams:
        if len(out) >= 2:
            break
        if fam == "lr_at_t" and len(named) >= 2:
            f = int(rng.integers(int(0.5 * fps), T - int(0.5 * fps)))
            cand = [i for i in named if inside[i, f]]
            if len(cand) < 2:
                continue
            a, b = rng.choice(cand, 2, replace=False)
            if abs(u[a, f] - u[b, f]) < 40:
                continue
            ans = "Yes." if u[a, f] < u[b, f] else "No."
            t = f / fps
            q = f"At {t:.1f} s, is the {names[a]} to the left of the {names[b]} on screen?"
            out.append(row(world, cam, sc, fam, q, ans, max(0, t - 0.5), min(dur, t + 0.5),
                           bbox([u[a, f], u[b, f]], [v[a, f], v[b, f]]), {"frame": f, "objects": [int(a), int(b)]}))
        elif fam == "first_mover":
            rest = [i for i in named if not ev["init_moving"][i] and ev["onset"][i] > 0 and inside[i, 0]]
            if len(rest) < 2:
                continue
            a, b = rng.choice(rest, 2, replace=False)
            if abs(int(ev["onset"][a]) - int(ev["onset"][b])) < 3:
                continue
            first = a if ev["onset"][a] < ev["onset"][b] else b
            q = f"Which starts moving first, the {names[a]} or the {names[b]}?"
            out.append(row(world, cam, sc, fam, q, f"The {names[first]}.", 0, dur,
                           bbox(np.r_[u[a, 0], u[b, 0]], np.r_[v[a, 0], v[b, 0]], 0.1),
                           {"onsets": [int(ev["onset"][a]), int(ev["onset"][b])], "objects": [int(a), int(b)]}))
        elif fam == "direction":
            movers = [i for i in named if ev["disp"][i].max() > 0.05]
            rng.shuffle(movers)
            for a in movers:
                f0 = max(int(ev["onset"][a]), 0)
                f1 = min(T - 1, f0 + int(1.5 * fps))
                if f1 - f0 < fps or not (inside[a, f0] and inside[a, f1]):
                    continue
                du, dv = u[a, f1] - u[a, f0], v[a, f1] - v[a, f0]
                if max(abs(du), abs(dv)) < 60 or not (abs(du) >= 1.5 * abs(dv) or abs(dv) >= 1.5 * abs(du)):
                    continue
                d = ("Right." if du > 0 else "Left.") if abs(du) > abs(dv) else ("Down." if dv > 0 else "Up.")
                t0, t1 = f0 / fps, f1 / fps
                q = f"In which direction does the {names[a]} move on screen between {t0:.1f} s and {t1:.1f} s? Answer left, right, up or down."
                seg = slice(f0, f1 + 1)
                out.append(row(world, cam, sc, fam, q, d, t0, t1, bbox(u[a, seg], v[a, seg]), {"objects": [int(a)]}))
                break
        elif fam == "still_moving":
            movers = [i for i in named if ev["onset"][i] >= 0 and inside[i, T - 1]]
            if not movers:
                continue
            a = int(rng.choice(movers))
            still = not bool(ev["stopped"][a])
            if still == bool(rng.integers(2)) and rng.random() < 0.5:   # soft balancing between yes and no
                continue
            q = f"Is the {names[a]} still moving at the end of the video?"
            out.append(row(world, cam, sc, fam, q, "Yes." if still else "No.", max(0, dur - 1.5), dur,
                           bbox(u[a, -int(1.5 * fps):], v[a, -int(1.5 * fps):]), {"objects": [a]}))
        elif fam == "falls":
            cand = [i for i in named if inside[i, 0]]
            if not cand:
                continue
            a = int(rng.choice(cand))
            q = f"Does the {names[a]} fall or topple during the video?"
            out.append(row(world, cam, sc, fam, q, "Yes." if ev["drop"][a] else "No.", 0, dur,
                           bbox(u[a], v[a]), {"objects": [a]}))
    return out


def build_scenario(sc):
    z = np.load(f"{OUT}/traj/{sc}.npz")
    worlds, nobj, prims, dyn, traj = z["worlds"], z["n_obj"], z["prims"], z["dynamic"], z["traj"]
    starts = np.r_[0, np.cumsum(nobj)[:-1]]
    idx = pq.read_table(f"{OUT}/world_index.parquet", columns=["scenario", "world", "camera", "usd_transform", "fxn", "ppx",
                                                                "ppy", "frame_count", "fps"],
                        filters=[("scenario", "=", sc)]).to_pandas()
    by_world = {w: g for w, g in idx.groupby("world")}
    rng = np.random.default_rng(int(hashlib.md5(sc.encode()).hexdigest()[:8], 16))
    rows, stats = [], {"worlds": 0, "clips": 0, "no_traj": 0, "named_objects": 0, "ambiguous_names": 0}
    limit = int(os.environ.get("LIMIT_WORLDS", 0)) or len(worlds)
    for wi, w in enumerate(worlds[:limit]):
        w = str(w)
        s, n = int(starts[wi]), int(nobj[wi])
        if n == 0 or w not in by_world:
            stats["no_traj"] += 1
            continue
        stats["worlds"] += 1
        P = [str(p) for p in prims[s:s + n]]
        tr = traj[s:s + n].astype(np.float64)
        nm = [object_name(p)[0] if dyn[s + k] else "" for k, p in enumerate(P)]
        cnt = pd.Series([object_name(p)[0] for p in P]).value_counts()
        amb = {k for k, c in cnt.items() if c > 1}
        jump = np.linalg.norm(np.diff(tr, axis=1), axis=-1).max(1) if tr.shape[1] > 1 else np.zeros(n)
        names = [x if x and x not in amb and jump[k] <= MAX_STEP_M else "" for k, x in enumerate(nm)]
        stats["implausible_objects"] = stats.get("implausible_objects", 0) + int((jump > MAX_STEP_M).sum())
        stats["named_objects"] += sum(bool(x) for x in names)
        stats["ambiguous_names"] += sum(bool(x) and x in amb for x in nm)
        ev = body_events(tr)
        for r in by_world[w].itertuples():
            if not os.path.exists(f"{PREP}/{sc}/{w}/{r.camera}.mp4"):
                continue
            stats["clips"] += 1
            T = min(int(r.frame_count), tr.shape[1])
            rows += clip_questions(rng, w, r.camera, sc, names, tr[:, :T], {k: (v[:, :T] if getattr(v, "ndim", 1) == 2 else v)
                                                                              for k, v in ev.items()},
                                   np.asarray(r.usd_transform, dtype=np.float64), float(r.fxn), float(r.ppx), float(r.ppy),
                                   T, float(r.fps))
    print(sc, stats, "rows", len(rows), flush=True)
    return rows, {sc: stats}


def write_splits(df):
    """train rows -> DEST (consumed by stage 1); val/test -> *_heldout.parquet (mechanism evaluation, never trained on)."""
    df[df.split == "train"].reset_index(drop=True).to_parquet(DEST, index=False)
    df[df.split != "train"].reset_index(drop=True).to_parquet(DEST.replace(".parquet", "_heldout.parquet"), index=False)


def balance(df, seed=0):
    """Remove answer-prior shortcuts per (scenario, family): yes/no exactly balanced, direction classes capped at 1.5x the
    rarest class, first_mover balanced between 'first-named object first' and 'second-named first'."""
    rng = np.random.default_rng(seed)
    keep = []
    for (sc, fam), g in df.groupby(["scenario", "family"]):
        if fam == "first_mover":
            key = g.apply(lambda r: r.answer[4:-1] == r.question.split("first, the ")[1].split(" or the ")[0], axis=1)
        else:
            key = g.answer
        counts = key.value_counts()
        if len(counts) < 2:
            continue                                   # a single-answer group would only teach the prior
        cap = counts.min() if fam != "direction" else int(1.5 * counts.min())
        for k, idx in g.groupby(key).groups.items():
            idx = np.asarray(list(idx))
            keep.extend(rng.choice(idx, min(cap, len(idx)), replace=False).tolist())
    return df.loc[sorted(keep)].reset_index(drop=True)


def main():
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 5))) as ex:
        res = list(ex.map(build_scenario, SCENARIOS))
    raw = pd.DataFrame([r for rows, _ in res for r in rows])
    df = balance(raw)
    print(f"rows before balancing {len(raw)}, after {len(df)}")
    df["prep_exists"] = True
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    write_splits(df)
    print(df.groupby(["split"]).size().to_dict())
    print(df.groupby(["family", "answer"]).size().unstack(fill_value=0).iloc[:, :8].to_string())
    print(df.groupby("scenario").size().to_dict())
    json.dump({k: v for _, s in res for k, v in s.items()}, open(DEST.replace(".parquet", "_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
