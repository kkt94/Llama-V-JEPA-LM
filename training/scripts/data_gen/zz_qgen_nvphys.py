"""(10-26, PhysBench: 1350 of its 5305 video items are PREDICTION - "where will the ball drop", "will it collide" - and nothing in our mix
asks what happens after the visible part) prediction questions from the NVIDIA PhysX scenes' ground-truth trajectories (same sources,
naming, events, projection and world-hash split as repo/scripts/data/build_stage1_nvidia_state.py - train worlds only). The clip is CUT
before the event; the answer is what the trajectory does afterwards.
 nv_fall    "will the X fall or topple after this?"     yes: X drops >= 0.4 s after the cut (cut 0.4-1.5 s before the drop); no: X never drops
 nv_move    "will the X start moving after this?"       yes: X at rest until >= 0.4 s after the cut; no: X never moves
 nv_side    "at the end, where is the X relative to the Y on screen?"   cut before X moves; left / right of Y at the end, with the end side
            equal to the side at the cut in exactly half of the rows (the cut frame alone cannot answer)
 nv_far     "which object travels the farthest?"         whole clip, winner >= 1.5x the runner-up, 4 options
yes / no: half lettered, half open answer. -> manifests/stage1/opendl3/nv_*.parquet (prep root prep/nvidia_rgb_384)
usage: zz_qgen_nvphys.py"""
import sys, os, json, hashlib, collections
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data"); sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); sys.path.insert(0, "/data/llama_vjepa_lm/training")
from nvidia_common import OUT, SCENARIOS, project, world_to_cam
from nvidia_pairs import body_events, DROP_D
from build_stage1_nvidia_state import object_name, split_of, MAX_STEP_M, W_IMG, H_IMG, PREP
from zz_qgen_common import Gen, report
O3 = "/data/llama_vjepa_lm/manifests/stage1/opendl3"

def build(sc):
    g = Gen(int(hashlib.md5(sc.encode()).hexdigest()[:8], 16)); out = collections.defaultdict(list)
    def add(kind, path, t0, t1, conv, key): out[kind].append(dict(video=path, prep_exists=True, root2=None, tq=None, aug="", t0=float(t0), t1=float(t1), conversations=conv, _key=key))
    def mc(kind, path, t0, t1, q, opts, k, key): add(kind, path, t0, t1, g.row(path, t0, t1, q, opts, k)["conversations"], key)
    def yn(kind, path, t0, t1, q, yes):
        if g.rng.random() < 0.5: mc(kind, path, t0, t1, q, ["yes", "no"] if yes else ["no", "yes"], 0, "yes" if yes else "no")
        else: add(kind, path, t0, t1, json.dumps([{"role": "user", "content": f"<|jepa_evidence|>\n{q}\nAnswer yes or no."}, {"role": "assistant", "content": "Yes." if yes else "No."}]), "yes" if yes else "no")
    z = np.load(f"{OUT}/traj/{sc}.npz"); worlds, nobj, prims, dyn, traj = z["worlds"], z["n_obj"], z["prims"], z["dynamic"], z["traj"]
    starts = np.r_[0, np.cumsum(nobj)[:-1]]
    idx = pq.read_table(f"{OUT}/world_index.parquet", columns=["scenario", "world", "camera", "usd_transform", "fxn", "ppx", "ppy", "frame_count", "fps"], filters=[("scenario", "=", sc)]).to_pandas()
    by_world = {w: gg for w, gg in idx.groupby("world")}
    for wi, w in enumerate(worlds):
        w = str(w); s, n = int(starts[wi]), int(nobj[wi])
        if n == 0 or w not in by_world or split_of(w) != "train": continue
        P = [str(p) for p in prims[s:s + n]]; tr = traj[s:s + n].astype(np.float64)
        nm = [object_name(p)[0] if dyn[s + k] else "" for k, p in enumerate(P)]; cnt = pd.Series([object_name(p)[0] for p in P]).value_counts(); amb = {k for k, c in cnt.items() if c > 1}
        jump = np.linalg.norm(np.diff(tr, axis=1), axis=-1).max(1) if tr.shape[1] > 1 else np.zeros(n)
        names = [x if x and x not in amb and jump[k] <= MAX_STEP_M else "" for k, x in enumerate(nm)]; named = [i for i, x in enumerate(names) if x]
        if not named: continue
        for r in by_world[w].itertuples():
            path = f"{sc}/{w}/{r.camera}.mp4"
            if not os.path.exists(f"{PREP}/{path}"): continue
            T = min(int(r.frame_count), tr.shape[1]); fps = float(r.fps); trc = tr[:, :T]; ev = body_events(trc)
            u, v, front = project(world_to_cam(trc, np.asarray(r.usd_transform, dtype=np.float64)), float(r.fxn), W_IMG, H_IMG, float(r.ppx), float(r.ppy))
            inside = front & (u >= 0) & (u < W_IMG) & (v >= 0) & (v < H_IMG); dur = T / fps
            fall_f = {i: (int(np.argmax(trc[i, :, 2] < trc[i, 0, 2] - DROP_D)) if ev["drop"][i] else -1) for i in named}
            # nv_fall
            pos = [i for i in named if fall_f[i] > int(1.4 * fps) and inside[i, max(0, fall_f[i] - int(0.4 * fps))]]
            neg = [i for i in named if fall_f[i] < 0 and inside[i, 0]]
            if pos and neg and dur > 3:
                if g.rng.random() < 0.5:
                    i = pos[int(g.rng.integers(len(pos)))]; cut = (fall_f[i] - g.rng.uniform(0.4, 1.5) * fps) / fps
                    if cut >= 1.0: yn("nv_fall", path, 0, cut, g.pick(["Will the {} fall or topple after this?", "Is the {} going to fall over?", "Predict: does the {} fall later?"]).format(names[i]), True)
                else:
                    i = neg[int(g.rng.integers(len(neg)))]; cut = g.rng.uniform(1.0, max(1.1, dur - 1.0)); yn("nv_fall", path, 0, cut, g.pick(["Will the {} fall or topple after this?", "Is the {} going to fall over?", "Predict: does the {} fall later?"]).format(names[i]), False)
            # nv_move
            pos = [i for i in named if not ev["init_moving"][i] and ev["onset"][i] > int(1.2 * fps) and inside[i, 0]]
            neg = [i for i in named if ev["onset"][i] < 0 and inside[i, 0]]
            if pos and neg:
                if g.rng.random() < 0.5:
                    i = pos[int(g.rng.integers(len(pos)))]; cut = (ev["onset"][i] - g.rng.uniform(0.4, 1.2) * fps) / fps
                    if cut >= 0.8: yn("nv_move", path, 0, cut, g.pick(["Will the {} start moving after this?", "Is the {} going to be set in motion?"]).format(names[i]), True)
                else:
                    i = neg[int(g.rng.integers(len(neg)))]; cut = g.rng.uniform(0.8, max(0.9, dur - 1.0)); yn("nv_move", path, 0, cut, g.pick(["Will the {} start moving after this?", "Is the {} going to be set in motion?"]).format(names[i]), False)
            # nv_side
            movers = [i for i in named if ev["onset"][i] > int(1.0 * fps) and ev["disp"][i].max() > 0.1]
            for a in g.rng.permutation(movers)[:1] if movers else []:
                cf = int(ev["onset"][a]) - int(0.3 * fps); others = [b for b in named if b != a and inside[b, cf] and inside[b, T - 1]]
                if cf < int(0.8 * fps) or not inside[a, cf] or not inside[a, T - 1] or not others: continue
                b = others[int(g.rng.integers(len(others)))]
                if abs(u[a, T - 1] - u[b, T - 1]) < 60 or abs(u[a, cf] - u[b, cf]) < 30: continue
                end_left, cut_left = u[a, T - 1] < u[b, T - 1], u[a, cf] < u[b, cf]
                opts = [f"To the left of the {names[b]}", f"To the right of the {names[b]}"]
                mc("nv_side", path, 0, cf / fps, g.pick(["At the end of the video, where will the {} be relative to the {} on screen?", "Predict where the {} ends up relative to the {}."]).format(names[a], names[b]), opts if end_left else opts[::-1], 0, "same" if end_left == cut_left else "changed")
            # nv_far
            moved = [i for i in named if inside[i, 0] and inside[i, T - 1]]
            if len(moved) >= 4:
                d = {i: float(np.linalg.norm(trc[i, -1] - trc[i, 0])) for i in moved}; o = sorted(moved, key=lambda i: -d[i])
                if d[o[0]] > 0.2 and d[o[0]] >= 1.5 * d[o[1]]:
                    ds = [o[int(k)] for k in 1 + g.rng.choice(len(o) - 1, 3, replace=False)]
                    mc("nv_far", path, 0, dur, g.pick(["Which object travels the farthest?", "Which object moves the greatest distance?"]), [f"The {names[o[0]]}"] + [f"The {names[j]}" for j in ds], 0, "mc")
    return dict(out)

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    allr = collections.defaultdict(list)
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 4))) as ex:
        for res in ex.map(build, SCENARIOS):
            for k, v in res.items(): allr[k] += v
    rng = np.random.default_rng(3)
    for k, rows in sorted(allr.items()):
        d = pd.DataFrame(rows); cnt = d._key.value_counts().to_dict()
        if k != "nv_far":                                   # exact balance of the key (yes / no, or same / changed side)
            m = d._key.value_counts().min(); d = pd.concat([gg.sample(m, random_state=0) for _, gg in d.groupby("_key")]).sort_index()
        print(k, "before", cnt, "after", d._key.value_counts().to_dict())
        d = d.drop(columns="_key").reset_index(drop=True); report(k, d.to_dict("records")); d.to_parquet(f"{O3}/{k}.parquet")
