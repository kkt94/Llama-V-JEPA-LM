"""Stage-1 physical-plausibility pairs (plausible vs. implausible video from the SAME source clip; only the physics differs).

Target skills: MVP intuitive_physics (InfLevel, GRASP, IntPhys, CLEVRER), IntPhys2 (permanence, immutability, spatio-temporal
continuity, solidity), PerceptionTest physics. No benchmark video is read; only TRAIN-split sources are used.

Sources
  clevrer  CLEVRER TRAIN videos video_00000-09999 (480x320, 25 fps, 128 frames, static camera) + annotation_train
           motion_trajectory. MVP uses CLEVRER validation (10000+): ids >= 10000 and ids in the MVP allowlist are asserted
           absent. Image positions = the CLEVR/Blender render camera (loc (7.36,-6.93,4.96), euler (63.6,0.6,46.7) deg,
           35 mm lens / 32 mm sensor) applied to the annotated world location; checked against frame-difference blobs on
           a few train videos (median offset 10-25 px, the scale is right), so displacement thresholds carry a margin.
  nvidia   NVIDIA PhysicalAI simulations, 384p transcodes in prep/nvidia_rgb_384 (682x384, 30 fps, 150 frames). Only clips
           listed in manifests/stage1/nvidia_state.parquet (train worlds, md5(world) mod 100 < 90) are used; clips in
           nvidia_state_heldout.parquet are asserted absent. Trajectories: manifests/nvidia/traj/<scenario>.npz (world COM,
           USDA-resolved rows), cameras from world_index.parquet (all cameras static, cam_static == True everywhere).
           Projection (report.md): u = ppx*W + fxn*W*x/-z, v = ppy*H - fxn*W*y/-z at W,H = 1920,1080, scaled to 682 px.
           Exclusions: scenario ball_mixer (non-physical jumps in 100% of clips) and every WORLD in which any body moves
           > 0.5 m in one frame (MAX_STEP_M, e.g. billiards balls falling off the table to z = -110 m) - the plausible
           member of a pair must really be plausible.

Manipulations (both members are decoded from the same source and re-encoded with identical settings: libx264 veryfast
CRF 23 GOP 16 yuv420p, single thread, native fps; same frame count, so codec artefacts and duration are not cues)
  teleport  spatio-temporal continuity. A segment of L frames (0.4-1.0 s) starting at s is cut out; at least one dynamic
            object is inside the view at s-1 and s+L and its projected COM moves >= TP_MIN_FRAC of the image width across
            the cut (0.0625 = 30 px at 480 px; CLEVRER 30 px, NVIDIA 0.06*W). Implausible = frames [0,s) + [s+L,T);
            plausible partner = the same number of frames removed from the start or the end (random).
            Visibility gate (`measure` + `select`, before rendering): on the decoded source, the fraction of pixels whose
            green channel changes by > 25 between the two frames that become adjacent (splice) must be >= VIS_RATIO (2) x
            the natural one-frame change right before/after the cut AND exceed it by >= VIS_MIN_EXCESS (0.5% of the image,
            ~ a 32x32 px object at 682x384). This removes jumps of tiny / occluded objects (e.g. small balls behind posts).
            Up to MEASURE_PER_SCENARIO NVIDIA candidates per scenario are measured (all CLEVRER ones).
  reverse   gravity / time reversal, NVIDIA only (CLEVRER is excluded: reversed billiard-like collisions on a flat floor
            are not clearly against gravity). Free fall alone is time-symmetric, so a clip qualifies only when a dynamic
            object falls >= REV_MIN_DROP_M from its highest point and then is at REST for the last REV_REST_FRAMES frames,
            visible at both ends, with a downward on-screen displacement >= REV_MIN_FRAC x width. Reversed, the resting
            object spontaneously leaps upward. Implausible = clip played backwards, plausible = forwards.
  permanence (object vanishing while visible) is NOT built: CLEVRER has no segmentation and the NVIDIA mask colours cannot
            be matched to objects reliably (report.md); crude box blurs would add a non-physical cue.

Format: manifests/stage1/plausibility.parquet (+ _heldout.parquet: 2% by md5 of the source clip key = CLEVRER video id or
NVIDIA world, so all cameras/kinds of one world stay together). Columns: video (relative to PREP_ROOT), prep_exists, kind,
source, scenario, pair_id, possible, conversations (JSON [{"role":"user","content":"<|jepa_evidence|>\\n"+prompt},
{"role":"assistant","content":"A."|"B."}]) and extra (JSON with clip, frames, metrics). Both members of a pair get the
SAME question and the SAME option order (only the video differs), so Yes/No and A/B are exactly 50/50. Prompt pool:
MVP intuitive_physics wordings (generic, IntPhys, simulator/GRASP-InfLevel style) with MVP's long options, IntPhys2's wording
with Yes/No, and generic paraphrases; option order random per pair, evaluator suffix "Answer with the option's letter ...".

Selection: <= MAX_CAMS_PER_WORLD (2) cameras per world and kind, then TARGET per (source, kind) water-filled over scenarios.
Usage (heavy steps on CPU nodes 03/04 via srun, 32 workers): plan -> measure --shard k --nshard n -> select ->
render --shard k --nshard n -> manifest. Every stage is resumable (outputs written as *.tmp.* then renamed; per-shard jsonl).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvidia_common import OUT as NV_OUT, project, world_to_cam  # noqa: E402

R = "/data/llama_vjepa_lm"
PREP_ROOT = f"{R}/prep/plausibility"
CAND = f"{PREP_ROOT}/_candidates.parquet"
PLAN = f"{PREP_ROOT}/_plan.parquet"
MEAS_DIR = f"{PREP_ROOT}/_measure"
RES_DIR = f"{PREP_ROOT}/_results"
DEST = f"{R}/manifests/stage1/plausibility.parquet"
FF = f"{R}/tools/ffmpeg"
CLV = f"{R}/data/clevrer"
NV_PREP = f"{R}/prep/nvidia_rgb_384"
NV_SCEN = ["billiards", "bowling", "dominoes", "objects_falling", "obstruction", "rolling_ramp_objects",
           "rolling_ramp_obstruct", "towers", "wrecking_ball"]           # ball_mixer excluded (non-physical jumps)
MAX_STEP_M = 0.5
TP_MIN_FRAC_CLV = 30 / 480
TP_MIN_FRAC_NV = 0.06
REV_MIN_DROP_M, REV_REST_FRAMES, REV_REST_V, REV_MIN_FRAC = 0.20, 15, 0.05, 0.05
VIS_RATIO, VIS_MIN_EXCESS = 2.0, 0.005
MEASURE_PER_SCENARIO = 6000
MAX_CAMS_PER_WORLD = 2                      # per kind; the 4 cameras of a world show the same physics
TARGET = {"clevrer/teleport": 10000, "nvidia/teleport": 12000, "nvidia/reverse": 12000}
HELDOUT_PCT = 2
VIS = "<|jepa_evidence|>"
SUFFIX = "Answer with the option's letter from the given choices directly."
ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-g", "16", "-pix_fmt", "yuv420p", "-threads", "1",
       "-an", "-movflags", "+faststart"]

# ------------------------------------------------------------------ prompts
MVP_YES = "Yes, everything is behaving according to human intuitive physics understanding"
MVP_NO = "No, something in the video is off/strange or violates human intuitive physics understanding"
PROMPTS = [  # (question, yes option, no option)
    ("Is this video physically plausible/possible according to your understanding of e.g. object permanence, gravity, "
     "trajectories of objects? (Assume it is the normal laws of physics and there are no magic tricks. Sometimes the video "
     "might start with visually showing the general setup of objects before the important actions happen)", MVP_YES, MVP_NO),
    ("Is this video physically plausible/possible according to your understanding of e.g. object permanence, shape constancy "
     "(objects maintain shape over time), continuous trajectories of objects? Assume it is the normal laws of physics. Your "
     "answer should be based on the events in the video and ignore the quality of the simulation engine.", MVP_YES, MVP_NO),
    ("The video you're seeing was generated by a simulator. Given how objects behave on earth, is the {focus} plausible? "
     "Your answer should be based on the events in the video and ignore the quality of the simulation.", MVP_YES, MVP_NO),
    ("The video you’re seeing was generated by a simulator. Given how objects behave on earth, is the behaviour of the "
     "objects plausible? Your answer should be based on the events in the video and ignore the quality of the simulation. "
     "You should also ignore the framerate of the video, as it may be slowed down. Finish your answer by Yes or No", "Yes", "No"),
    ("Does everything in this video obey the laws of physics?", "Yes", "No"),
    ("Is the motion of the objects in this video physically possible?", "Yes, it is physically possible",
     "No, it is physically impossible"),
    ("Could the events in this video happen in the real world? Consider gravity, continuity of motion and object permanence.",
     "Yes, the events are physically plausible", "No, something physically impossible happens"),
    ("Watch how the objects move. Is their behaviour consistent with intuitive physics, or does something impossible happen?",
     "It is consistent with intuitive physics", "Something impossible happens"),
]
FOCUS = ["behaviour of the objects", "trajectory of the objects", "motion of the objects", "outcome of the experiment"]


def make_prompt(rng):
    q, yes, no = PROMPTS[int(rng.integers(len(PROMPTS)))]
    q = q.replace("{focus}", FOCUS[int(rng.integers(len(FOCUS)))])
    opts = [yes, no] if rng.random() < 0.5 else [no, yes]
    text = f"{q}\n" + "\n".join(f"{'AB'[i]}. {o}" for i, o in enumerate(opts)) + f"\n{SUFFIX}"
    return text, "AB"[opts.index(yes)] + ".", "AB"[opts.index(no)] + "."


def h100(key):
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 100


def seed_of(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ planning: teleport window search
def choose_teleport(rng, uv, vis, W, T, Lmin, Lmax, margin, min_frac):
    """uv [n,T,2] pixels, vis [n,T] bool. Returns (s, L, obj, frac) or None. Several random L tried."""
    for L in rng.permutation(np.arange(Lmin, Lmax + 1))[:4]:
        L = int(L)
        ss = np.arange(margin, T - L - margin + 1)
        if len(ss) == 0:
            continue
        a, b = ss - 1, ss + L
        d = np.linalg.norm(uv[:, b] - uv[:, a], axis=-1) / W           # [n, S]
        ok = vis[:, a] & vis[:, b] & np.isfinite(d)
        d = np.where(ok, d, 0.0)
        best = d.max(0)
        good = np.nonzero(best >= min_frac)[0]
        if len(good):
            k = int(rng.choice(good))
            return int(ss[k]), L, int(d[:, k].argmax()), float(best[k])
    return None


def tele_frames(T, s, L, drop_start):
    imp = list(range(0, s)) + list(range(s + L, T))
    pla = list(range(L, T)) if drop_start else list(range(0, T - L))
    return imp, pla


# CLEVRER camera
def clevrer_proj(p, W=480, H=320):
    rx, ry, rz = np.deg2rad([63.559, 0.62, 46.692])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    pc = (np.asarray(p) - np.array([7.358891487121582, -6.925790786743164, 4.958309173583984])) @ (Rz @ Ry @ Rx)
    f = 35.0 / 32.0 * W
    return np.stack([W / 2 + f * pc[..., 0] / -pc[..., 2], H / 2 - f * pc[..., 1] / -pc[..., 2]], -1)


def plan_clevrer_one(vid):
    sub = f"{vid // 1000 * 1000:05d}-{vid // 1000 * 1000 + 1000:05d}"
    src = f"{CLV}/video_train/video_{sub}/video_{vid:05d}.mp4"
    ann = f"{CLV}/annotation_train/annotation_{sub}/annotation_{vid:05d}.json"
    if not (os.path.exists(src) and os.path.exists(ann)):
        return None
    a = json.load(open(ann))
    mt = a["motion_trajectory"]
    T = len(mt)
    ids = sorted({o["object_id"] for f in mt for o in f["objects"]})
    loc = np.full((len(ids), T, 3), np.nan)
    vis = np.zeros((len(ids), T), bool)
    for t, f in enumerate(mt):
        for o in f["objects"]:
            k = ids.index(o["object_id"])
            loc[k, t] = o["location"]
            vis[k, t] = bool(o["inside_camera_view"])
    uv = clevrer_proj(loc)
    vis &= (uv[..., 0] > 10) & (uv[..., 0] < 470) & (uv[..., 1] > 10) & (uv[..., 1] < 310)
    rng = np.random.default_rng(seed_of("clevrer", vid))
    r = choose_teleport(rng, uv, vis, 480, T, 10, 25, 12, TP_MIN_FRAC_CLV)
    if r is None:
        return None
    s, L, obj, frac = r
    return {"source": "clevrer", "scenario": "clevrer", "clip": f"video_{vid:05d}", "group": f"video_{vid:05d}",
            "src": src, "kind": "teleport", "T": T, "fps": 25.0, "s": s, "L": L, "obj": obj, "frac": frac,
            "drop_start": bool(rng.random() < 0.5), "name": f"clevrer/video_{vid:05d}_teleport"}


def plan_nvidia_scenario(sc):
    train = set(pd.read_parquet(f"{R}/manifests/stage1/nvidia_state.parquet", columns=["video"]).video)
    held = set(pd.read_parquet(f"{R}/manifests/stage1/nvidia_state_heldout.parquet", columns=["video"]).video)
    z = np.load(f"{NV_OUT}/traj/{sc}.npz")
    worlds, nobj, dyn = z["worlds"], z["n_obj"], z["dynamic"]
    traj = z["traj"]
    starts = np.r_[0, np.cumsum(nobj)[:-1]]
    idx = pd.read_parquet(f"{NV_OUT}/world_index.parquet", columns=["scenario", "world", "camera", "usd_transform", "fxn",
                                                                  "ppx", "ppy", "frame_count", "fps", "cam_static"],
                          filters=[("scenario", "=", sc)])
    by_world = {w: g for w, g in idx.groupby("world")}
    out, st = [], {"worlds": 0, "jump_excluded": 0, "clips": 0, "tele": 0, "rev": 0}
    Wn, Hn = 1920, 1080
    for wi, w in enumerate(worlds):
        w = str(w)
        s0, n = int(starts[wi]), int(nobj[wi])
        if n == 0 or w not in by_world:
            continue
        cams = [r for r in by_world[w].itertuples() if f"{sc}/{w}/{r.camera}.mp4" in train]
        if not cams:
            continue
        assert not any(f"{sc}/{w}/{r.camera}.mp4" in held for r in cams)
        st["worlds"] += 1
        tr = traj[s0:s0 + n].astype(np.float64)
        if not np.isfinite(tr).all() or np.linalg.norm(np.diff(tr, axis=1), axis=-1).max() > MAX_STEP_M:
            st["jump_excluded"] += 1
            continue
        d = dyn[s0:s0 + n]
        if not d.any():
            continue
        trd = tr[d]
        T = trd.shape[1]
        # reverse candidates (camera independent part)
        sp = np.zeros(trd.shape[:2])
        sp[:, 1:] = np.linalg.norm(np.diff(trd, axis=1), axis=-1) * 30.0
        rest = (sp[:, -REV_REST_FRAMES:] < REV_REST_V).all(1)
        zz = trd[:, :, 2]
        thi = zz[:, :T - REV_REST_FRAMES].argmax(1)
        drop = zz[np.arange(len(zz)), thi] - zz[:, -1]
        rev_obj = np.nonzero(rest & (drop >= REV_MIN_DROP_M))[0]
        rng = np.random.default_rng(seed_of(sc, w))
        tele_rows, rev_rows = [], []
        for r in cams:
            T_c = min(int(r.frame_count), T)
            pc = world_to_cam(trd[:, :T_c], np.asarray(r.usd_transform, dtype=np.float64))
            u, v, front = project(pc, float(r.fxn), Wn, Hn, float(r.ppx), float(r.ppy))
            vis = front & (u > 0.03 * Wn) & (u < 0.97 * Wn) & (v > 0.03 * Hn) & (v < 0.97 * Hn)
            uv = np.stack([u, v], -1)
            base = {"source": "nvidia", "scenario": sc, "clip": f"{sc}/{w}/{r.camera}", "group": w,
                    "src": f"{NV_PREP}/{sc}/{w}/{r.camera}.mp4", "T": T_c, "fps": float(r.fps)}
            st["clips"] += 1
            tp = choose_teleport(rng, uv, vis, Wn, T_c, 12, 30, 15, TP_MIN_FRAC_NV)
            if tp is not None:
                s, L, obj, frac = tp
                tele_rows.append({**base, "kind": "teleport", "s": s, "L": L, "obj": obj, "frac": frac,
                                  "drop_start": bool(rng.random() < 0.5),
                                  "name": f"nvidia/{sc}/{w}__{r.camera}_teleport"})
            best = None
            for o in rev_obj:
                t0 = int(thi[o])
                if T_c != T or not (vis[o, t0] and vis[o, T_c - 1]):
                    continue
                dv = (v[o, T_c - 1] - v[o, t0]) / Wn
                if dv >= REV_MIN_FRAC and (best is None or dv > best[1]):
                    best = (int(o), float(dv), float(drop[o]))
            if best is not None:
                rev_rows.append({**base, "kind": "reverse", "s": -1, "L": 0, "obj": best[0], "frac": best[1],
                                 "drop_m": best[2], "drop_start": False, "name": f"nvidia/{sc}/{w}__{r.camera}_reverse"})
        out += tele_rows + rev_rows                 # all cameras; <= MAX_CAMS_PER_WORLD chosen in `select`
        st["tele"] += len(tele_rows)
        st["rev"] += len(rev_rows)
    print(sc, st, flush=True)
    return out


def cmd_plan(a):
    """All candidates (every train clip / camera that passes the trajectory criteria) -> CAND."""
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        nv = ex.map(plan_nvidia_scenario, NV_SCEN)
        cl = list(ex.map(plan_clevrer_one, range(10000), chunksize=200))
        nv = [r for rows in nv for r in rows]
    df = pd.DataFrame([r for r in cl if r] + nv).sort_values("name").reset_index(drop=True)
    df["pair_id"] = df.name.str.replace("/", "__")
    assert df.pair_id.is_unique
    os.makedirs(PREP_ROOT, exist_ok=True)
    df.to_parquet(CAND, index=False)
    print("candidates", df.groupby(["source", "kind", "scenario"]).size().to_string(), f"{time.time() - t0:.0f}s", flush=True)


def splice_visible(m):
    """Splice must change >= VIS_RATIO x the natural one-frame change right before / after it, and by >= VIS_MIN_EXCESS
    of the image (about a 32x32 px object at 682x384), i.e. a visible object really jumps."""
    return m["splice"] >= VIS_RATIO * max(m["nat_adj"], 1e-4) and m["splice"] - m["nat_adj"] >= VIS_MIN_EXCESS


def teleport_metrics(g, s, L, imp, pla):
    c = lambda i, j: change_frac(g[i], g[j])  # noqa: E731
    return {"splice": c(s - 1, s + L), "nat_adj": max(c(s - 2, s - 1), c(s + L, s + L + 1)),
            "nat_partner": max(c(i, i + 1) for i in pla[:-1]),
            "nat_imp": max(c(imp[i], imp[i + 1]) for i in range(len(imp) - 1) if imp[i + 1] == imp[i] + 1)}


def load_frames(row):
    import decord
    vr = decord.VideoReader(row["src"], num_threads=1)
    n = len(vr)
    T = min(n, int(row["T"]))
    if row["kind"] == "teleport" and int(row["s"]) + int(row["L"]) + 1 >= T:
        raise ValueError(f"video has {n} frames, too short for the planned cut")
    return vr.get_batch(list(range(T))).asnumpy(), n, T


def measure_one(row):
    try:
        fr, n, T = load_frames(row)
        g = grey(fr)
        imp, pla = tele_frames(T, int(row["s"]), int(row["L"]), bool(row["drop_start"]))
        return {"pair_id": row["pair_id"], "ok": True, "n_src": n, **teleport_metrics(g, int(row["s"]), int(row["L"]), imp, pla)}
    except Exception as e:  # noqa: BLE001
        return {"pair_id": row["pair_id"], "ok": False, "err": repr(e)[:300]}


def read_jsonl_dir(d):
    out = {}
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        for line in open(f"{d}/{f}"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok"):
                out[r["pair_id"]] = r
    return out


def cmd_measure(a):
    cand = pd.read_parquet(CAND)
    cand = cand[cand.kind == "teleport"]
    # measuring is decode-bound: a hash-ordered subset of MEASURE_PER_SCENARIO per NVIDIA scenario is plenty for the targets
    cand = cand.assign(_h=cand.pair_id.map(lambda p: seed_of("measure", p))).sort_values("_h")
    cand = cand[(cand.groupby("scenario").cumcount() < MEASURE_PER_SCENARIO) | (cand.source == "clevrer")].sort_values("name")
    cand = cand[cand.index % a.nshard == a.shard]
    os.makedirs(MEAS_DIR, exist_ok=True)
    done = read_jsonl_dir(MEAS_DIR)
    todo = [r for r in cand.to_dict("records") if r["pair_id"] not in done]
    print(f"measure shard {a.shard}/{a.nshard}: {len(cand)} teleport candidates, {len(todo)} to measure", flush=True)
    t0, k = time.time(), 0
    with ProcessPoolExecutor(a.workers) as ex, open(f"{MEAS_DIR}/shard{a.shard}of{a.nshard}.jsonl", "a") as fo:
        for r in ex.map(measure_one, todo, chunksize=8):
            fo.write(json.dumps(r) + "\n")
            k += 1
            if k % 2000 == 0:
                fo.flush()
                print(f"{k}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print(f"measured {k} {time.time() - t0:.0f}s", flush=True)


def cmd_select(a):
    """Visible teleports only; <= MAX_CAMS_PER_WORLD cameras per world and kind; per (source, kind) TARGET reached by
    water-filling over scenarios (small scenarios keep everything)."""
    cand = pd.read_parquet(CAND)
    meas = read_jsonl_dir(MEAS_DIR)
    tele = cand.kind == "teleport"
    vis = cand.pair_id.map(lambda p: p in meas and splice_visible(meas[p]))
    print("teleport measured", int(cand[tele].pair_id.isin(meas.keys()).sum()), "/", int(tele.sum()), "visible",
          cand[tele & vis].groupby("scenario").size().to_dict(), "of", cand[tele].groupby("scenario").size().to_dict())
    df = cand[~tele | vis]
    rng = np.random.default_rng(0)
    df = df.loc[rng.permutation(df.index)]
    df = df.groupby(["kind", "group"], sort=False).head(MAX_CAMS_PER_WORLD)
    keep = []
    for (src, kind), g in df.groupby(["source", "kind"]):
        tgt = TARGET.get(f"{src}/{kind}", 0)
        sizes = g.groupby("scenario").size().sort_values()
        cap, left, nleft = {}, tgt, len(sizes)
        for scn, nsz in sizes.items():
            cap[scn] = min(nsz, left // nleft)
            left -= cap[scn]
            nleft -= 1
        for scn, gg in g.groupby("scenario"):
            keep.append(gg.iloc[:cap[scn]])
    plan = pd.concat(keep).sort_values("name").reset_index(drop=True)
    plan.to_parquet(PLAN, index=False)
    print("plan", plan.groupby(["source", "kind", "scenario"]).size().to_string(), flush=True)


# ------------------------------------------------------------------ rendering
def grey(fr):
    """Green channel as the luma proxy for pixel-change metrics (a float channel mean was the measuring bottleneck)."""
    return fr[..., 1]


def change_frac(a, b):
    return float((np.abs(a.astype(np.int16) - b.astype(np.int16)) > 25).mean())


def encode(frames, path, fps):
    tmp = path[:-4] + f".tmp.{os.uname().nodename}.{os.getpid()}.mp4"
    T, H, W = frames.shape[:3]
    p = subprocess.Popen([FF, "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", f"{fps:g}", "-i", "-", *ENC, "-f", "mp4", tmp], stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(frames).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed {path}")
    os.replace(tmp, path)


def render_one(row):
    imp_p, pla_p = f"{PREP_ROOT}/{row['name']}_imp.mp4", f"{PREP_ROOT}/{row['name']}_pla.mp4"
    try:
        fr, n, T = load_frames(row)
        met = {"n_src": n}
        if row["kind"] == "teleport":
            imp, pla = tele_frames(T, int(row["s"]), int(row["L"]), bool(row["drop_start"]))
            met.update(teleport_metrics(grey(fr), int(row["s"]), int(row["L"]), imp, pla))
        else:
            pla = list(range(T))
            imp = pla[::-1]
        os.makedirs(os.path.dirname(imp_p), exist_ok=True)
        for path, idx in ((imp_p, imp), (pla_p, pla)):
            if not os.path.exists(path):
                encode(fr[idx], path, float(row["fps"]))
        met.update(n_imp=len(imp), n_pla=len(pla), H=int(fr.shape[1]), W=int(fr.shape[2]))
        return {"pair_id": row["pair_id"], "name": row["name"], "ok": True, **met}
    except Exception as e:  # noqa: BLE001
        return {"pair_id": row["pair_id"], "name": row["name"], "ok": False, "err": repr(e)[:300]}


def cmd_render(a):
    plan = pd.read_parquet(PLAN)
    plan = plan[plan.index % a.nshard == a.shard]
    os.makedirs(RES_DIR, exist_ok=True)
    res_path = f"{RES_DIR}/shard{a.shard}of{a.nshard}.jsonl"
    done = set()
    if os.path.exists(res_path):
        for line in open(res_path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok") and os.path.exists(f"{PREP_ROOT}/{r['name']}_imp.mp4") \
                    and os.path.exists(f"{PREP_ROOT}/{r['name']}_pla.mp4"):
                done.add(r["pair_id"])
    todo = [r for r in plan.to_dict("records") if r["pair_id"] not in done]
    print(f"shard {a.shard}/{a.nshard}: {len(plan)} planned, {len(done)} done, {len(todo)} to render", flush=True)
    t0, k, nerr = time.time(), 0, 0
    with ProcessPoolExecutor(a.workers) as ex, open(res_path, "a") as fo:
        for r in ex.map(render_one, todo, chunksize=4):
            fo.write(json.dumps(r) + "\n")
            k += 1
            nerr += not r["ok"]
            if k % 500 == 0:
                fo.flush()
                print(f"{k}/{len(todo)} err={nerr} {time.time() - t0:.0f}s", flush=True)
    print(f"finished {k} err={nerr} {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ manifest
def cmd_manifest(a):
    plan = pd.read_parquet(PLAN).set_index("pair_id")
    res = read_jsonl_dir(RES_DIR)
    rows, drop = [], {"not_rendered": 0, "splice_invisible": 0, "missing_file": 0, "frame_mismatch": 0}
    for pid, p in plan.iterrows():
        r = res.get(pid)
        if r is None:
            drop["not_rendered"] += 1
            continue
        if r["n_imp"] != r["n_pla"]:
            drop["frame_mismatch"] += 1
            continue
        if p.kind == "teleport" and not splice_visible(r):
            drop["splice_invisible"] += 1
            continue
        vids = {True: f"{p['name']}_pla.mp4", False: f"{p['name']}_imp.mp4"}
        if not all(os.path.exists(f"{PREP_ROOT}/{v}") for v in vids.values()):
            drop["missing_file"] += 1
            continue
        rng = np.random.default_rng(seed_of("prompt", pid))
        q, yes, no = make_prompt(rng)
        extra = {"clip": p["clip"], "src": p["src"], "T": int(p["T"]), "fps": float(p.fps), "n_frames": int(r["n_imp"]),
                 "obj": int(p.obj), "disp_frac_width": round(float(p.frac), 4)}
        if p.kind == "teleport":
            extra.update(s=int(p.s), L=int(p.L), drop_start=bool(p.drop_start), splice_change=round(r["splice"], 5), nat_change_adjacent=round(r["nat_adj"], 5),
                         nat_change_partner=round(r["nat_partner"], 5), nat_change_imp=round(r["nat_imp"], 5))
        else:
            extra.update(drop_m=round(float(p.drop_m), 3))
        for possible in (True, False):
            rows.append({"video": vids[possible], "prep_exists": True, "kind": p.kind, "source": p.source,
                         "scenario": p.scenario, "pair_id": pid, "possible": possible, "group": p["group"],
                         "conversations": json.dumps([{"role": "user", "content": f"{VIS}\n{q}"},
                                                      {"role": "assistant", "content": yes if possible else no}]),
                         "extra": json.dumps(extra)})
    df = pd.DataFrame(rows)
    df["prep_exists"] = [os.path.exists(f"{PREP_ROOT}/{v}") for v in df.video]
    held = df.group.map(lambda g: h100("plaus:" + g) < HELDOUT_PCT)
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    df[~held].drop(columns="group").reset_index(drop=True).to_parquet(DEST, index=False)
    df[held].drop(columns="group").reset_index(drop=True).to_parquet(DEST.replace(".parquet", "_heldout.parquet"), index=False)
    print("dropped", drop)
    print("rows", len(df), "train", int((~held).sum()), "heldout", int(held.sum()))
    print(df.groupby(["source", "kind", "scenario"]).pair_id.nunique().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "measure", "select", "render", "manifest"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    {"plan": cmd_plan, "measure": cmd_measure, "select": cmd_select, "render": cmd_render,
     "manifest": cmd_manifest}[a.cmd](a)


if __name__ == "__main__":
    main()
