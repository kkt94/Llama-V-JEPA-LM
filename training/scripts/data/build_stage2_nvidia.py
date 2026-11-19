"""Stage-2 NVIDIA world-group data (spec 5.4, 6.4-6.5, 8): questions with a gold EvidenceSpec, occlusion rows that require
masked prediction, and camera pairs for L_use. Built only from definitions that are correct by construction.

Observed clip: the whole video, 16 frames paired-uniform (8 tubelets, 4 time bins of 2 tubelets), centre-cropped square
around the queried objects (crop_cx). Gold spec = (kind, semantic op, time bin, 3x3 region cell) in the crop the model sees.
  kind OBSERVED          the queried cell x bin is visible
  kind MASKED_PREDICTION the pixels of exactly that cell x bin are blanked in the input (mask_bin / mask_cell); reading the
                         observed tokens there gives nothing, the predictor must estimate them (spec 3.2, 7.2)
Semantic ops: 0 lr_at_t (image-plane left/right at time t), 1 direction (on-screen motion over a span), 2 still_moving (at
the end), 3 falls (3D drop). SHORT_FUTURE stays disabled (spec 5.2), so no question needs frames after the clip.
Pairs: two cameras of the same world asking the same question about the same objects. lr_at_t / direction are
view-dependent -> pair_type 1 (the answer must change) when the two cameras' answers differ; still_moving / falls are 3D
facts -> pair_type 0 (must stay). group_id = (scenario, op, object slots): one question meaning across worlds (spec 8.1).
Objects with a per-frame COM jump > 0.5 m are excluded (non-physical trajectories, see the NVIDIA report).
Split: same world-level hash split as stage-1 state QA; only train rows go to manifests/stage2/nvidia_groups.parquet.
"""
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/llama_vjepa_lm/training")
from build_stage1_nvidia_state import H_IMG, MAX_STEP_M, W_IMG, object_name, split_of  # noqa: E402
from nvidia_common import OUT, SCENARIOS, project, slot_key, world_to_cam  # noqa: E402
from nvidia_pairs import body_events  # noqa: E402
from jepa_llm.data.video import paired_uniform_times  # noqa: E402

NUM_FRAMES, N_BINS, RGRID = 16, 4, 3
OPS = {"lr_at_t": 0, "direction": 1, "still_moving": 2, "falls": 3}
DEST = "/data/llama_vjepa_lm/manifests/stage2/nvidia_groups.parquet"
PREP = "/data/llama_vjepa_lm/prep/nvidia_rgb_384"
CROP_W = H_IMG / W_IMG          # square crop width as a fraction of the 16:9 frame width (0.5625)


def crop_x0(cx):
    return float(np.clip(cx - CROP_W / 2, 0.0, 1.0 - CROP_W))


def cell_of(u, v, x0):
    """source pixel -> 3x3 cell index (1..9) inside the square crop starting at x0 (fraction of width); None if outside."""
    xs = (u / W_IMG - x0) / CROP_W
    ys = v / H_IMG
    if not (0 <= xs < 1 and 0 <= ys < 1):
        return None
    return int(ys * RGRID) * RGRID + int(xs * RGRID) + 1


def bin_frames(T, fps):
    """source frame indices of each time bin for the paired-uniform 16-frame sample of the whole clip."""
    t = paired_uniform_times(T / fps, NUM_FRAMES, 0.25)
    f = np.clip(np.round(t * fps).astype(int), 0, T - 1)
    per = NUM_FRAMES // N_BINS
    return [f[b * per:(b + 1) * per] for b in range(N_BINS)], t


def make_q(op, names, a, b=None, t=None, t0=None, t1=None):
    if op == "lr_at_t":
        return f"At {t:.1f} s, is the {names[a]} to the left of the {names[b]} on screen?"
    if op == "direction":
        return f"In which direction does the {names[a]} move on screen between {t0:.1f} s and {t1:.1f} s? Answer left, right, up or down."
    if op == "still_moving":
        return f"Is the {names[a]} still moving at the end of the video?"
    return f"Does the {names[a]} fall or topple during the video?"


def world_rows(rng, sc, world, cams, P, names, tr, ev):
    """cams: list of (camera, M, fxn, ppx, ppy, T, fps). Returns rows for this world (both cameras of a pair)."""
    out = []
    proj = {}
    for cam, M, fxn, ppx, ppy, T, fps in cams:
        u, v, front = project(world_to_cam(tr[:, :T], M), fxn, W_IMG, H_IMG, ppx, ppy)
        proj[cam] = (u, v, front & (u >= 0) & (u < W_IMG) & (v >= 0) & (v < H_IMG), T, fps)
    named = [i for i, n in enumerate(names) if n]
    if len(named) < 1 or len(cams) < 2:
        return out
    c1, c2 = rng.choice(len(cams), 2, replace=False)
    pair_cams = [cams[c1][0], cams[c2][0]]
    op = rng.choice(list(OPS))
    T, fps = proj[pair_cams[0]][3], proj[pair_cams[0]][4]
    bins, times = bin_frames(T, fps)
    b = int(rng.integers(N_BINS))
    fr = bins[b]
    a = bb = None
    if op == "lr_at_t":
        cand = [i for i in named if all(proj[c][2][i, fr].all() for c in pair_cams)]
        if len(cand) < 2:
            return out
        a, bb = (int(x) for x in rng.choice(cand, 2, replace=False))
        f = int(fr[len(fr) // 2]); t = f / fps
        answers = {c: ("Yes." if proj[c][0][a, f] < proj[c][0][bb, f] else "No.") for c in pair_cams}
        if any(abs(proj[c][0][a, f] - proj[c][0][bb, f]) < 40 for c in pair_cams):
            return out
        q = make_q(op, names, a, bb, t=t)
        objs = [a, bb]
    elif op == "direction":
        movers = [i for i in named if ev["disp"][i, fr[-1]] - ev["disp"][i, fr[0]] > 0.03]
        if not movers:
            return out
        a = int(rng.choice(movers)); objs = [a]
        answers = {}
        for c in pair_cams:
            u, v, ins = proj[c][:3]
            if not (ins[a, fr[0]] and ins[a, fr[-1]]):
                return out
            du, dv = u[a, fr[-1]] - u[a, fr[0]], v[a, fr[-1]] - v[a, fr[0]]
            if max(abs(du), abs(dv)) < 40 or not (abs(du) >= 1.5 * abs(dv) or abs(dv) >= 1.5 * abs(du)):
                return out
            answers[c] = ("Right." if du > 0 else "Left.") if abs(du) > abs(dv) else ("Down." if dv > 0 else "Up.")
        q = make_q(op, names, a, t0=fr[0] / fps, t1=fr[-1] / fps)
    elif op == "still_moving":
        b, fr = N_BINS - 1, bins[N_BINS - 1]
        movers = [i for i in named if ev["onset"][i] >= 0 and all(proj[c][2][i, fr].all() for c in pair_cams)]
        if not movers:
            return out
        a = int(rng.choice(movers)); objs = [a]
        ans = "No." if ev["stopped"][a] else "Yes."
        answers = {c: ans for c in pair_cams}
        q = make_q(op, names, a)
    else:  # falls: the bin in which the COM is lowest relative to its start
        cand = [i for i in named if all(proj[c][2][i].any() for c in pair_cams)]
        if not cand:
            return out
        a = int(rng.choice(cand)); objs = [a]
        drop_t = int(np.argmin(tr[a, :T, 2])) if ev["drop"][a] else int(rng.integers(T))
        b = int(np.argmin([abs(int(np.median(x)) - drop_t) for x in bins])); fr = bins[b]
        if not all(proj[c][2][a, fr].all() for c in pair_cams):
            return out
        ans = "Yes." if ev["drop"][a] else "No."
        answers = {c: ans for c in pair_cams}
        q = make_q(op, names, a)
    view_dep = op in ("lr_at_t", "direction")
    pair_type = int(view_dep and answers[pair_cams[0]] != answers[pair_cams[1]])
    if view_dep and not pair_type:
        pair_type = -1                                  # same answer on both views: keep rows, no pair constraint
    pair_id = f"{world}|{op}|{'-'.join(map(str, objs))}"
    slots = "-".join(slot_key(P[o]) for o in objs)
    masked = bool(rng.integers(2))
    for c in pair_cams:
        u, v, ins = proj[c][:3]
        fmid = int(fr[len(fr) // 2])
        us = [u[o, fmid] / W_IMG for o in objs]
        # random square-crop offset that keeps the objects inside (a centred crop would make the region cell a constant)
        lo, hi = max(0.0, max(us) - 0.98 * CROP_W), min(1.0 - CROP_W, min(us) - 0.02 * CROP_W)
        if lo > hi:
            return []
        x0 = float(rng.uniform(lo, hi))
        cells = {cell_of(u[o, fmid], v[o, fmid], x0) for o in objs}
        if None in cells or len(cells) != 1:            # objects must share one visible cell for a single-cell spec
            return []
        cell = cells.pop()
        out.append({"scenario": sc, "world": world, "camera": c, "video": f"{sc}/{world}/{c}.mp4", "split": split_of(world),
                    "prep_exists": True, "t0": 0.0, "t1": T / fps, "crop_cx": x0 + CROP_W / 2,
                    "question": q, "answer": answers[c], "op": op,
                    "gold_kind": 1 if masked else 0, "gold_sem": OPS[op], "gold_time_bin": b, "gold_region": cell,
                    "mask_bin": b if masked else -1, "mask_cell": cell if masked else -1,
                    "group_id": f"{sc}|{op}|{slots}", "pair_key": pair_id if pair_type >= 0 else "",
                    "pair_type": max(pair_type, 0)})
    return out


def build_scenario(sc):
    z = np.load(f"{OUT}/traj/{sc}.npz")
    worlds, nobj, prims, dyn, traj = z["worlds"], z["n_obj"], z["prims"], z["dynamic"], z["traj"]
    starts = np.r_[0, np.cumsum(nobj)[:-1]]
    idx = pq.read_table(f"{OUT}/world_index.parquet", columns=["world", "camera", "usd_transform", "fxn", "ppx", "ppy",
                                                                "frame_count", "fps"], filters=[("scenario", "=", sc)]).to_pandas()
    by_world = {w: g for w, g in idx.groupby("world")}
    rng = np.random.default_rng(int(hashlib.md5(("s2" + sc).encode()).hexdigest()[:8], 16))
    rows = []
    limit = int(os.environ.get("LIMIT_WORLDS", 0)) or len(worlds)
    for wi, w in enumerate(worlds[:limit]):
        w = str(w); s, n = int(starts[wi]), int(nobj[wi])
        if n == 0 or w not in by_world:
            continue
        P = [str(p) for p in prims[s:s + n]]
        tr = traj[s:s + n].astype(np.float64)
        full = [object_name(p)[0] for p in P]
        cnt = pd.Series(full).value_counts()
        jump = np.linalg.norm(np.diff(tr, axis=1), axis=-1).max(1) if tr.shape[1] > 1 else np.zeros(n)
        names = [x if dyn[s + k] and x and cnt[x] == 1 and jump[k] <= MAX_STEP_M else "" for k, x in enumerate(full)]
        cams = [(r.camera, np.asarray(r.usd_transform, dtype=np.float64), float(r.fxn), float(r.ppx), float(r.ppy),
                 min(int(r.frame_count), tr.shape[1]), float(r.fps)) for r in by_world[w].itertuples()
                if os.path.exists(f"{PREP}/{sc}/{w}/{r.camera}.mp4")]
        ev = body_events(tr)
        for _ in range(int(os.environ.get("TRIES", 8))):  # several question pairs per world
            rows += world_rows(rng, sc, w, cams, P, names, tr, ev)
    print(sc, "rows", len(rows), flush=True)
    return rows


def balance_units(df, seed=0):
    """Answer balance per (scenario, op) at the level of units (a camera pair = 2 rows kept or dropped together, the
    first row's answer is the key): yes/no equalised, direction classes capped at 1.5x the rarest; also caps region-cell
    classes at 2x the rarest non-empty cell per op so the composer cannot win by a location prior."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["unit"] = np.where(df.pair_key != "", df.pair_key, df.world + "|" + df.camera + "|" + df.question)
    first = df.groupby("unit").head(1)
    # note: per-scenario balance also removes families whose answer is constant inside a scenario (e.g. 'falls' is
    # always 'No.' in billiards and 'Yes.' in objects_falling) - those would only teach a scene-identity shortcut
    keep_units = []
    for (sc, op), g in first.groupby(["scenario", "op"]):
        counts = g.answer.value_counts()
        if len(counts) < 2:
            continue
        cap = counts.min() if op != "direction" else int(1.5 * counts.min())
        for _, gg in g.groupby("answer"):
            keep_units += rng.choice(gg.unit.values, min(cap, len(gg)), replace=False).tolist()
    out = df[df.unit.isin(set(keep_units))]
    keep = []
    for op, g in out.groupby("op"):
        c = g.gold_region.value_counts()
        c_ref = c[c >= 0.05 * c.sum()]                        # ignore near-empty cells when setting the cap
        cap = 3 * (c_ref.min() if len(c_ref) else c.min())
        for cell, gg in g.groupby("gold_region"):
            units = gg.unit.unique()
            keep += rng.choice(units, min(len(units), cap), replace=False).tolist()
    return out[out.unit.isin(set(keep))].drop(columns="unit").reset_index(drop=True)


def main():
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 5))) as ex:
        raw = pd.DataFrame([r for rows in ex.map(build_scenario, SCENARIOS) for r in rows])
    raw = raw.drop_duplicates(["world", "camera", "question"]).reset_index(drop=True)   # same question drawn twice
    df = balance_units(raw)
    print(f"rows before balancing {len(raw)}, after {len(df)}")
    keys = sorted(k for k in df.pair_key.unique() if k)
    pid = {k: i for i, k in enumerate(keys)}
    df["pair_id"] = df.pair_key.map(lambda k: pid.get(k, -1))
    sizes = df[df.pair_id >= 0].groupby("pair_id").size()
    df.loc[df.pair_id.isin(sizes[sizes != 2].index), "pair_id"] = -1   # pairs must have exactly two rows
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    df[df.split == "train"].reset_index(drop=True).to_parquet(DEST, index=False)
    df[df.split != "train"].reset_index(drop=True).to_parquet(DEST.replace(".parquet", "_heldout.parquet"), index=False)
    tr = df[df.split == "train"]
    print(f"rows {len(df)} train {len(tr)} | pairs {int((tr.pair_id >= 0).sum() // 2)} "
          f"(change {int(((tr.pair_id >= 0) & (tr.pair_type == 1)).sum() // 2)}, invariant {int(((tr.pair_id >= 0) & (tr.pair_type == 0)).sum() // 2)})"
          f" | masked {int((tr.gold_kind == 1).sum())} | groups {tr.group_id.nunique()}")
    print(tr.groupby(["op", "answer"]).size().to_string())


if __name__ == "__main__":
    main()
