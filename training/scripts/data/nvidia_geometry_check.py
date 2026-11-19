"""Geometry check: project physics COM into the image and compare with the segmentation masks.

usage: python -I nvidia_geometry_check.py [--per-scenario 8] [--frames 6] [--seed 0] [--workers 12]
Needs world_index.parquet and tar_listing/ (nvidia_index.py list/build). Read-only on the data, deterministic given --seed.

For every sampled clip (scenario, world, camera) and frames F = linspace(0, T-1, --frames):
  * every rigid body (group of npz rows with identical trajectory) is projected with the camera json (square-pixel model, and
    the json's (fx, fy) model for comparison);
  * the PNG colour under the projected COM pixel is collected; its per-body majority over frames ("pixel colour") is the
    label-free mask association;
  * each colour-label source (npz['segmentation_colors'][row], static.json colour, raw.json colour(s) of the USDA-resolved prim)
    is compared with the pixel colour;
  * pixel error = |mask centroid - projected COM| for the pixel-colour mask, stratified by mask area and by whether the COM pixel
    falls on the object's own mask (unoccluded at the COM) or not;
  * visibility.npz rows are compared with mask presence (area > 0) of the same colour.
  * RGB/mask temporal alignment (on clips that have a 384p transcode): the lag in {-3..3} maximising the correlation between
    |RGB_t - RGB_{t-1}| and the mask-change map.
Outputs: geometry_objects.parquet (one row per body x frame), geometry_clips.parquet.
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # shared host: one BLAS thread per process
    _os.environ.setdefault(_v, "1")

import argparse, collections, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvidia_common import OUT, read_member, load_npz_bytes, listing_path, world_to_cam, cam_to_world, project, prim_top

cv2.setNumThreads(1)


def ckey(c):
    c = [int(x) for x in c]
    return (c[0] << 24) | (c[1] << 16) | (c[2] << 8) | (c[3] if len(c) > 3 else 255)


def png_keys(b):
    im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    if im.ndim == 2 or im.shape[2] == 3:
        raise ValueError("expected RGBA segmentation png")
    im = im.astype(np.uint32)
    return (im[..., 2] << 24) | (im[..., 1] << 16) | (im[..., 0] << 8) | im[..., 3]


def member_ref(listing_cache, tar, member):
    if tar not in listing_cache:
        t = pq.read_table(listing_path(tar)).to_pandas().set_index("member")
        listing_cache[tar] = t
    r = listing_cache[tar].loc[member]
    return tar, int(r["offset"]), int(r["size"])


def rgb_alignment(prep_path, seg_tar, world, cam, lc, T, center):
    """Correlation of RGB frame differences with mask changes for lags -3..3 over a 24-frame window with motion."""
    import av
    frames = []
    with av.open(prep_path) as c:
        for f in c.decode(video=0):
            frames.append(f.to_ndarray(format="gray").astype(np.float32))
    if len(frames) < 10:
        return None
    H, W = frames[0].shape
    t0 = int(np.clip(center - 12, 4, max(4, min(len(frames), T) - 28)))
    ts = list(range(t0, min(t0 + 24, len(frames) - 4, T - 4)))
    need = sorted({t + d for t in ts for d in range(-4, 5)} & set(range(T)))
    masks = {}
    for t in need:
        k = png_keys(read_member(*member_ref(lc, seg_tar, f"{world}/{cam}_{t:04d}.png")))
        masks[t] = cv2.resize(k.astype(np.float64), (W, H), interpolation=cv2.INTER_NEAREST)
    out = {}
    for lag in range(-3, 4):
        num = den1 = den2 = 0.0
        for t in ts:
            if t + lag - 1 not in masks or t + lag not in masks:
                continue
            a = np.abs(frames[t] - frames[t - 1]).ravel()
            b = (masks[t + lag] != masks[t + lag - 1]).astype(np.float32).ravel()
            a = a - a.mean(); b = b - b.mean()
            num += (a * b).sum(); den1 += (a * a).sum(); den2 += (b * b).sum()
        out[lag] = float(num / np.sqrt(den1 * den2)) if den1 > 0 and den2 > 0 else np.nan
    return out


def check_clip(job):
    row, n_frames, do_rgb = job
    lc = {}
    world, cam = row["world"], row["camera"]
    phys = row["physics_tar"]
    com = load_npz_bytes(read_member(*member_ref(lc, phys, f"{world}/{cam}_com.npz")))
    d = com["data"].astype(np.float64)
    n, T, _ = d.shape
    npz_cols = [ckey(c) for c in com["segmentation_colors"]]
    seg_tar = row["segmentation_tar"]
    raw = json.loads(read_member(*member_ref(lc, seg_tar, row["seg_raw_member"])))["objects"]
    vis = load_npz_bytes(read_member(*member_ref(lc, seg_tar, row["seg_vis_member"])))
    raw_by_top = collections.defaultdict(set)
    for o in raw:
        raw_by_top[prim_top(o["prim_path"])].add(ckey(o["color"]))
    # rigid-body groups
    flat = d.reshape(n, -1)
    gid = -np.ones(n, int); groups = []
    for i in range(n):
        if gid[i] < 0:
            same = np.where((np.abs(flat - flat[i]).max(1) < 1e-6) & (gid < 0))[0]
            gid[same] = len(groups); groups.append(same)
    # identity of groups from the index (USDA-resolved static.json objects)
    ident = {}
    for o in row["objects"]:
        rows_ = list(o["npz_rows"])
        if rows_ and o["identity_resid_m"] is not None and o["identity_resid_m"] < 0.02 and o["identity_margin_m"] > 0.01:
            ident[int(gid[rows_[0]])] = o
    fxn, fyn = row["fxn"], row["fyn"]
    frames = np.unique(np.linspace(0, T - 1, n_frames).round().astype(int))
    recs = []
    pix_votes = collections.defaultdict(collections.Counter)
    keymaps, cstats, bg = {}, {}, {}
    for t in frames:
        K = png_keys(read_member(*member_ref(lc, seg_tar, f"{world}/{cam}_{t:04d}.png")))
        keymaps[t] = K
        H, W = K.shape
        keys, inv, cnt = np.unique(K.ravel(), return_inverse=True, return_counts=True)
        ys, xs = np.divmod(np.arange(K.size), W)
        sx = np.bincount(inv, weights=xs, minlength=len(keys)); sy = np.bincount(inv, weights=ys, minlength=len(keys))
        cstats[t] = {int(k): (int(c), sx[i] / c, sy[i] / c) for i, (k, c) in enumerate(zip(keys, cnt))}
        # background = any colour covering > 15% of the frame (floor, table, ramp, sky/empty): never the mask of a dynamic body
        bg[t] = {k for k, (c, _, _) in cstats[t].items() if c > 0.15 * H * W}
        u, v, front = project(d[:, t], fxn, W, H)
        for g, rows_ in enumerate(groups):
            r0 = rows_[0]
            if front[r0] and 0 <= u[r0] < W and 0 <= v[r0] < H:
                c_ = int(K[int(v[r0]), int(u[r0])])
                if c_ not in bg[t]:
                    pix_votes[g][c_] += 1
    # the background/ground colour: largest area at frame 0 is excluded from votes being "the object"
    for g, rows_ in enumerate(groups):
        r0 = rows_[0]
        votes = pix_votes.get(g, collections.Counter())
        pix_col, pix_n = (votes.most_common(1)[0] if votes else (None, 0))
        o = ident.get(g)
        st_col = ckey(o["segmentation_color"]) if o is not None else None
        raw_cols = raw_by_top.get(o["prim_path"], set()) if o is not None else set()
        npzc = {npz_cols[i] for i in rows_}
        for t in frames:
            K = keymaps[t]; H, W = K.shape
            u, v, front = project(d[r0:r0 + 1, t], fxn, W, H)
            uA = u[0]; vA = 0.5 * H - fyn * H * d[r0, t, 1] / (-d[r0, t, 2])  # json (fx, fy) model, for comparison
            inside = bool(front[0] and 0 <= u[0] < W and 0 <= v[0] < H)
            under = int(K[int(v[0]), int(u[0])]) if inside else None
            rec = dict(scenario=row["scenario"], world=world, camera=cam, frame=int(t), body=g, n_rows=len(rows_),
                       prim=(o["prim_path"] if o is not None else None), dynamic=o is not None,
                       u=float(u[0]), v=float(v[0]), depth=float(-d[r0, t, 2]), proj_inside=inside,
                       pix_col=pix_col, pix_col_votes=pix_n, n_votes=sum(votes.values()),
                       under_is_pixcol=(under == pix_col) if inside and pix_col is not None else None,
                       npz_label_eq_pix=(pix_col in npzc) if pix_col is not None else None,
                       static_eq_pix=(st_col == pix_col) if (pix_col is not None and st_col is not None) else None,
                       raw_eq_pix=(pix_col in raw_cols) if (pix_col is not None and o is not None) else None)
            # label-free nearest mask: the non-background colour whose centroid is closest to the projected COM
            cand = [(np.hypot(cx - u[0], cy - v[0]), k_) for k_, (a_, cx, cy) in cstats[t].items() if k_ not in bg[t] and a_ >= 30]
            rec["err_nearest"] = float(min(cand)[0]) if (cand and inside) else None
            rs = [cstats[t][c_] for c_ in raw_cols if c_ in cstats[t]]
            if rs:   # raw.json may list several part colours for one prim: area-weighted centroid of all of them
                A = sum(r_[0] for r_ in rs)
                rec["area_raw"] = A
                rec["err_raw"] = float(np.hypot(sum(r_[0] * r_[1] for r_ in rs) / A - u[0], sum(r_[0] * r_[2] for r_ in rs) / A - v[0]))
            else:
                rec["area_raw"], rec["err_raw"] = 0, None
            for name, col in (("pix", pix_col), ("static", st_col), ("npz", npz_cols[r0])):
                s = cstats[t].get(col) if col is not None else None
                rec[f"area_{name}"] = s[0] if s else 0
                rec[f"err_{name}"] = float(np.hypot(s[1] - u[0], s[2] - v[0])) if s else None
                if name == "pix":
                    rec["err_pix_jsonfy"] = float(np.hypot(s[1] - uA, s[2] - vA)) if s else None
            recs.append(rec)
    # visibility.npz vs mask presence
    vcols = [ckey(c) for c in vis["segmentation_colors"]]
    agree = tot = 0
    for t in frames:
        for i, c in enumerate(vcols):
            pres = cstats[t].get(c, (0,))[0] > 0
            agree += bool(vis["visibility"][i, t]) == pres; tot += 1
    clip = dict(scenario=row["scenario"], world=world, camera=cam, T=T, n_rows=n, n_bodies=len(groups), n_identified=len(ident),
                n_frames_checked=len(frames), vis_agree=agree, vis_total=tot,
                vis_order_eq_raw=[ckey(o["color"]) for o in raw] == vcols, vis_order_eq_npz=vcols == npz_cols)
    if do_rgb and row.get("prep_video"):
        try:
            # window centred on the frame with the largest summed COM image motion of all bodies (physics-based)
            u_, v_, f_ = project(d, fxn, 1920, 1080)
            mot = np.nansum(np.hypot(np.diff(u_, axis=1), np.diff(v_, axis=1)) * f_[:, 1:], axis=0)
            mot = np.where(np.isfinite(mot), mot, 0); mot[:3] = 0
            clip["rgb_window_center"] = int(mot.argmax()) + 1
            al = rgb_alignment(row["prep_video"], seg_tar, world, cam, lc, T, clip["rgb_window_center"])
            if al:
                clip.update({f"rgb_lag_corr_{k}": v for k, v in al.items()})
                clip["rgb_best_lag"] = int(max(al, key=lambda k: -np.inf if np.isnan(al[k]) else al[k]))
        except Exception as e:  # noqa: BLE001 - report, do not hide
            clip["rgb_error"] = repr(e)
    return recs, clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scenario", type=int, default=8)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--rgb-per-scenario", type=int, default=3)
    ap.add_argument("--index", default=f"{OUT}/world_index.parquet")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    idx = pd.read_parquet(a.index)
    have = idx[idx.segmentation_tar.notna() & idx.seg_raw_member.notna() & idx.seg_vis_member.notna()]
    have = have[have.segmentation_tar.map(lambda t: os.path.exists(listing_path(t)))]
    print(f"[geom] index rows {len(idx)}; rows with listed segmentation {len(have)}", flush=True)
    rng = np.random.default_rng(a.seed)
    jobs = []
    for sc, g in have.groupby("scenario", sort=True):
        worlds = np.array(sorted(g.world.unique()))
        pick = rng.choice(worlds, size=min(a.per_scenario, len(worlds)), replace=False)
        for k, w in enumerate(sorted(pick)):
            cams = sorted(g[g.world == w].camera)
            cam = cams[int(rng.integers(len(cams)))]
            r = g[(g.world == w) & (g.camera == cam)].iloc[0].to_dict()
            r["objects"] = json.loads(r["objects_json"])
            jobs.append((r, a.frames, k < a.rgb_per_scenario))
    print(f"[geom] sampled clips {len(jobs)} over {len({j[0]['scenario'] for j in jobs})} scenarios", flush=True)
    if not jobs:
        raise SystemExit("0 clips sampled: nothing to compare (bug, not a result)")
    recs, clips = [], []
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (r, c) in enumerate(ex.map(check_clip, jobs)):
            recs.extend(r); clips.append(c)
            print(f"[geom] {i+1}/{len(jobs)} {c['world']} {c['camera']} bodies={c['n_bodies']} t={time.time()-t0:.0f}s", flush=True)
    R = pd.DataFrame(recs); C = pd.DataFrame(clips)
    R.to_parquet(f"{OUT}/geometry_objects{a.tag}.parquet"); C.to_parquet(f"{OUT}/geometry_clips{a.tag}.parquet")
    print(f"[geom] wrote {len(R)} body-frame records, {len(C)} clips")


if __name__ == "__main__":
    main()
