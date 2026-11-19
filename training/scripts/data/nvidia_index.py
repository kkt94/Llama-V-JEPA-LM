"""World index for the NVIDIA PhysicalAI synthetic-physics data (one row per scenario x world x camera).

usage (CPU only, read-only on the downloaded tars; run under `nice -n 19`):
  python -I nvidia_index.py list  [--workers 16] [--modalities physics,cameras,...]   # tar member listings with byte offsets
  python -I nvidia_index.py build [--workers 16] [--scenarios billiards,...]           # per-scenario index + trajectory cache
  python -I nvidia_index.py merge [--workers 10]                                        # world_index.parquet / objects.parquet

Outputs (all under /data/llama_vjepa_lm/manifests/nvidia/):
  tar_listing/<modality>/<tar>.parquet   member, offset (data start), size        -> O(1) random access into any tar
  world_index.parquet                    one row per (scenario, world, camera); objects_json / colliders_json hold the per-object
                                         lists (static.json properties + USDA-resolved npz rows)
  objects.parquet                        the same objects flattened: one row per (world, camera, dynamic object)
  world_index_<sc>.parquet, objects_<sc>.parquet   per-scenario parts of the two tables above
  traj/<scenario>.npz                    world-frame COM trajectories (every frame, float32) of every USDA-resolved body;
                                         `dynamic` marks bodies listed in static.json (the rest are static colliders)
  build_orphans.json                     worlds present in a modality but missing physics (all zero in the 2026-10-09 build)
Deterministic: inputs are processed in sorted order, there is no randomness in this script.
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # shared host: one BLAS thread per process
    _os.environ.setdefault(_v, "1")

import argparse, bisect, collections, json, os, sys, tarfile, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvidia_common import (DATA, OUT, PREP, SCENARIOS, MODALITIES, DERIVED_VIDEO, split_world, tar_paths, listing_path,
                           read_member, load_npz_bytes, cam_to_world, usda_top_prims, prim_top, slot_key)


# ====================================================================== list
def list_one(tar):
    out = listing_path(tar)
    if os.path.exists(out):
        return tar, -1
    names, offs, sizes = [], [], []
    with tarfile.open(tar) as tf:
        for m in tf:
            if m.isfile():
                names.append(m.name); offs.append(m.offset_data); sizes.append(m.size)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + f".part{os.getpid()}"
    pq.write_table(pa.table({"member": names, "offset": pa.array(offs, pa.int64()), "size": pa.array(sizes, pa.int64())}), tmp)
    os.replace(tmp, out)
    return tar, len(names)


def cmd_list(a):
    mods = a.modalities.split(",")
    tars = [t for m in mods for s in SCENARIOS for t in tar_paths(m, s)]
    todo = [t for t in tars if not os.path.exists(listing_path(t))]
    print(f"[list] tars total {len(tars)}, to list {len(todo)}, workers {a.workers}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (t, n) in enumerate(ex.map(list_one, todo)):
            print(f"[list] {i+1}/{len(todo)} {os.path.basename(t)} members={n} t={time.time()-t0:.0f}s", flush=True)


def first_member(tar):
    with tarfile.open(tar) as tf:
        m = tf.next()
        return m.name


def check_seg_boundaries(scenario):
    """For listed segmentation tars: every world's members are contiguous, worlds are in lexicographic order inside the tar,
    and the first world of tar i+1 sorts after the last world of tar i. Returns a list of violations (empty = OK)."""
    bad, prev_last = [], None
    for t in tar_paths("segmentation", scenario):
        if not os.path.exists(listing_path(t)):
            prev_last = None
            continue
        ws = [m.split("/", 1)[0] for m in pq.read_table(listing_path(t), columns=["member"]).column(0).to_pylist()]
        runs = [w for i, w in enumerate(ws) if i == 0 or w != ws[i - 1]]
        if len(runs) != len(set(runs)):
            bad.append(f"{t}: non-contiguous world blocks")
        if runs != sorted(runs):
            bad.append(f"{t}: worlds not in lexicographic order")
        if prev_last is not None and runs and runs[0] <= prev_last:
            bad.append(f"{t}: first world {runs[0]} <= previous tar last world {prev_last}")
        prev_last = runs[-1] if runs else prev_last
    return bad


def load_listing(modality, scenario, allow_missing=False):
    """member -> (tar, offset, size) for all tars of one modality/scenario."""
    d = {}
    for t in tar_paths(modality, scenario):
        if not os.path.exists(listing_path(t)):
            if modality == "segmentation":
                continue          # handled by the boundary fallback in build_jobs
            if allow_missing:
                print(f"[warn] no listing for {t} (allowed for test runs only)", flush=True)
                continue
            raise SystemExit(f"missing listing for {t}: run `list` first")
        tb = pq.read_table(listing_path(t)).to_pydict()
        for m, o, s in zip(tb["member"], tb["offset"], tb["size"]):
            if m in d:
                raise SystemExit(f"duplicate member {m} in {t} and {d[m][0]}")
            d[m] = (t, o, s)
    return d


# ====================================================================== build
def resolve_rows(com0, vel0, prims, _unused=None):
    """Assign npz rows (distinct rigid-body trajectories) to USDA top-level prims by initial pose.

    com0/vel0: [n,3] world-frame position/velocity at frame 0 of one representative row per rigid body.
    Cost = min over dt in [0, 0.2] s of |p0 - v0*dt - translate|  (straight-line back-extrapolation: exact for constant velocity
    at dt = t, for free fall from rest at dt = t/2). Returns {prim: (row_index, residual_m, margin_m)} for assigned pairs."""
    names = sorted(prims)
    if not names or len(com0) == 0:
        return {}
    P = np.stack([prims[n]["translate"] for n in names])            # [m,3]
    a = com0[:, None, :] - P[None]                                    # [n,m,3]
    vv = (vel0 * vel0).sum(-1)[:, None]                               # [n,1]
    with np.errstate(divide="ignore", invalid="ignore"):
        dt = np.where(vv > 1e-12, (a * vel0[:, None, :]).sum(-1) / vv, 0.0)
    dt = np.clip(dt, 0.0, 0.2)
    C = np.linalg.norm(a - vel0[:, None, :] * dt[..., None], axis=-1)  # [n,m]
    r, c = linear_sum_assignment(C)
    out = {}
    for i, j in zip(r, c):
        others = np.delete(C[i], j)
        margin = float(others.min() - C[i, j]) if len(others) else np.inf
        out[names[j]] = (int(i), float(C[i, j]), margin)
    return out


def process_world(job):
    """job = (scenario, world, {cam: {kind: (tar, off, size)}}, scene (tar,off,size) or None, metadata (tar,off,size) or None,
    seg {cam: {'raw':..., 'vis':..., 'n_png': int, 'tar': str}}, video {cam: (tar, member)}, captions {cam: [members]})."""
    scenario, world, phys, scene_ref, meta_ref, seg, vid, cap = job
    _, thash, variant = split_world(world)
    prims = {}
    meta = {}
    if scene_ref:
        prims = usda_top_prims(read_member(*scene_ref).decode("utf-8", "replace"))
    if meta_ref:
        meta = json.loads(read_member(*meta_ref))
    rows, objs, traj = [], [], None
    world_tr = {}          # cam -> world-frame trajectories of all npz rows [n,T,3]
    cam_info = {}
    for cam in sorted(phys):
        p = phys[cam]
        com = load_npz_bytes(read_member(*p["com"]))
        vel = load_npz_bytes(read_member(*p["velocity"]))
        st = json.loads(read_member(*p["static"]))
        cj = json.loads(read_member(*p["camera"])) if "camera" in p else None
        d = com["data"].astype(np.float64)
        n, T, _ = d.shape
        info = dict(n_rows=n, T=T, frame_count=int(com["frame_count"]), static_frame_count=int(st.get("frame_count", -1)),
                    vel_same_colors=bool(np.array_equal(vel["segmentation_colors"], com["segmentation_colors"])),
                    vel_shape_ok=bool(vel["velocities"].shape == com["data"].shape))
        if cj is not None:
            U = np.asarray(cj["usd_transform"], dtype=np.float64)
            c = cj["camera"]
            fl = np.asarray(c["focal_length"], dtype=np.float64)
            pp = np.asarray(c["principal_point"], dtype=np.float64)
            info.update(cam_static=bool(np.abs(U - U[0]).max() < 1e-9 and np.abs(fl - fl[0]).max() < 1e-12),
                        fxn=float(fl[0, 0]), fyn=float(fl[0, 1]), ppx=float(pp[0, 0]), ppy=float(pp[0, 1]),
                        distortion_zero=bool(np.allclose(c.get("distortion", 0), 0)), usd_transform=U[0].tolist(),
                        pose_world2cam=list(map(float, c["pose_world2cam"][0])), cam_frames=len(U))
            wtr = cam_to_world(d, U[0])
            wvel = vel["velocities"].astype(np.float64) @ U[0].reshape(4, 4)[:3, :3]
            world_tr[cam] = (wtr, wvel)
            steps = np.linalg.norm(np.diff(d, axis=1), axis=-1)          # [n, T-1]
            info.update(last_step_m=float(steps[:, -1].max()) if steps.size else None,
                        max_step_before_last_m=float(steps[:, :-1].max()) if steps.shape[1] > 1 else None)
        info.update(gravity=st.get("gravity", {}).get("world_gravity"), npz_colors=com["segmentation_colors"].tolist(),
                    static_objects=st["objects"])
        cam_info[cam] = info
    cams = sorted(cam_info)
    # ---- per camera: rigid bodies = groups of rows with identical trajectories (multi-mesh props have one row per mesh),
    # identity of each group from the USDA initial poses. The row order can differ between cameras of the same world, so this
    # is done for every camera separately.
    resolved_by_cam, groups_by_cam = {}, {}
    for cam in cams:
        if cam not in world_tr:
            resolved_by_cam[cam], groups_by_cam[cam] = {}, []
            continue
        W0, V0 = world_tr[cam]
        n = len(W0)
        flat = W0.reshape(n, -1)
        assigned = -np.ones(n, int)
        groups = []
        for i in range(n):
            if assigned[i] >= 0:
                continue
            same = np.where((np.abs(flat - flat[i]).max(1) < 1e-6) & (assigned < 0))[0]
            assigned[same] = len(groups)
            groups.append(same.tolist())
        rep = np.array([g[0] for g in groups])
        res = resolve_rows(W0[rep, 0], V0[rep, 0], prims, None)
        resolved_by_cam[cam] = {prim: dict(rows=groups[gi], resid=resid, margin=margin) for prim, (gi, resid, margin) in res.items()}
        groups_by_cam[cam] = groups
    ref = cams[0]
    good = lambda r: r["resid"] < 0.02 and r["margin"] > 0.01  # noqa: E731
    # ---- cross-camera consistency: same row order as the reference camera? same world trajectory per resolved prim?
    xcam, same_order, first_div = {}, {}, {}
    for cam in cams:
        if cam not in world_tr or ref not in world_tr:
            xcam[cam] = same_order[cam] = None
            continue
        same_order[cam] = bool(world_tr[cam][0].shape == world_tr[ref][0].shape and
                               np.abs(world_tr[cam][0] - world_tr[ref][0]).max() < 1e-4)
        diffs = [np.abs(world_tr[cam][0][r["rows"][0]] - world_tr[ref][0][resolved_by_cam[ref][p]["rows"][0]]).max()
                 for p, r in resolved_by_cam[cam].items() if good(r) and p in resolved_by_cam[ref] and good(resolved_by_cam[ref][p])]
        xcam[cam] = float(max(diffs)) if diffs else None
        div = [np.linalg.norm(world_tr[cam][0][r["rows"][0]] - world_tr[ref][0][resolved_by_cam[ref][p]["rows"][0]], axis=-1)
               for p, r in resolved_by_cam[cam].items() if good(r) and p in resolved_by_cam[ref] and good(resolved_by_cam[ref][p])
               and world_tr[cam][0].shape[1] == world_tr[ref][0].shape[1]]
        if div:
            bad = (np.stack(div) > 1e-3).any(0)
            first_div[cam] = int(bad.argmax()) if bad.any() else -1
        else:
            first_div[cam] = None
    static_names = {o["prim_path"] for o in cam_info[ref]["static_objects"]}
    resolved = resolved_by_cam.get(ref, {})
    groups = groups_by_cam.get(ref, [])
    if ref in world_tr:
        W0 = world_tr[ref][0]
        # world-frame trajectories for the cache: every USDA-resolved, unambiguous prim (dynamic = listed in static.json;
        # static colliders such as bumpers / obstacles are kept because their layout is part of the observable scene)
        tr_objs, tr_dyn, tr_arr = [], [], []
        for prim in sorted(resolved):
            r = resolved[prim]
            if good(r):
                tr_objs.append(prim); tr_dyn.append(prim in static_names); tr_arr.append(W0[r["rows"][0]].astype(np.float32))
        traj = (world, tr_objs, tr_dyn, np.stack(tr_arr) if tr_arr else np.zeros((0, W0.shape[1], 3), np.float32))
    # ---- label-source agreement with the USDA identity (per camera)
    for cam in cams:
        info = cam_info[cam]
        npz_cols = [tuple(c) for c in info["npz_colors"]]
        st_col = {o["prim_path"]: tuple(o["segmentation_color"]) for o in info["static_objects"]}
        raw = seg.get(cam, {}).get("raw_objects")
        raw_col2top = {tuple(o["color"]): prim_top(o["prim_path"]) for o in raw} if raw else {}
        n_res = n_npz_static = n_npz_raw = n_st_raw = n_st = 0
        for prim, r in resolved_by_cam[cam].items():
            if not good(r):
                continue
            n_res += 1
            row_cols = {npz_cols[i] for i in r["rows"]}
            if raw_col2top:
                n_npz_raw += any(raw_col2top.get(c) == prim for c in row_cols)
            if prim in st_col:
                n_st += 1
                n_npz_static += st_col[prim] in row_cols
                if raw_col2top:
                    n_st_raw += raw_col2top.get(st_col[prim]) == prim
        info.update(n_resolved=n_res, n_npz_label_eq_raw=n_npz_raw if raw_col2top else None, n_resolved_dyn=n_st,
                    n_npz_label_eq_static=n_npz_static, n_static_eq_raw=n_st_raw if raw_col2top else None,
                    n_rigid_bodies=len(groups_by_cam[cam]))
    # ---- rows of the index
    for cam in cams:
        info = cam_info[cam]
        so = info["static_objects"]
        s = seg.get(cam, {})
        objs_cam = []
        for o in so:
            pr = o.get("properties", {})
            r = resolved_by_cam[cam].get(o["prim_path"])
            objs_cam.append(dict(
                prim_path=o["prim_path"], slot=slot_key(o["prim_path"]), segmentation_color=list(o["segmentation_color"]),
                mass=_num(pr.get("mass")), static_friction=_num(pr.get("static_friction")),
                dynamic_friction=_num(pr.get("dynamic_friction")), restitution=_num(pr.get("restitution")),
                density=_num(pr.get("density")), inertia=_vec(pr.get("inertia")), center_of_mass_local=_vec(pr.get("center_of_mass")),
                collision_enabled=bool(pr.get("collision_enabled", False)),
                npz_rows=(r["rows"] if r else []), identity_resid_m=(r["resid"] if r else None),
                identity_margin_m=(r["margin"] if r else None)))
        vt = vid.get(cam)
        prep = f"{PREP}/{scenario}/{world}/{cam}.mp4"
        rows.append(dict(
            scenario=scenario, world=world, template_hash=thash, variant=variant, camera=cam,
            frame_count=info["frame_count"], T_npz=info["T"], static_frame_count=info["static_frame_count"],
            fps=float(meta.get("config", {}).get("capture_fps", np.nan)) if meta else np.nan,
            physics_substeps=meta.get("config", {}).get("physics_substeps") if meta else None,
            scene_seed=str(meta.get("reproducibility", {}).get("scene_seed")) if meta else None,
            n_rows_npz=info["n_rows"], n_rigid_bodies=info["n_rigid_bodies"], n_objects=len(so),
            row_order_same_as_ref_cam=same_order.get(cam), xcam_first_diverge_frame=first_div.get(cam),
            last_step_m=info.get("last_step_m"), max_step_before_last_m=info.get("max_step_before_last_m"),
            n_raw_seg_objects=(len(s["raw_objects"]) if s.get("raw_objects") is not None else None),
            gravity=info["gravity"], cam_static=info.get("cam_static"), fxn=info.get("fxn"), fyn=info.get("fyn"),
            ppx=info.get("ppx"), ppy=info.get("ppy"), distortion_zero=info.get("distortion_zero"),
            usd_transform=info.get("usd_transform"), pose_world2cam=info.get("pose_world2cam"), cam_frames=info.get("cam_frames"),
            vel_same_colors=info["vel_same_colors"], vel_shape_ok=info["vel_shape_ok"],
            xcam_world_maxdiff_m=xcam.get(cam), n_usda_prims=len(prims), n_resolved=info["n_resolved"],
            n_resolved_dyn=info["n_resolved_dyn"], n_npz_label_eq_static=info["n_npz_label_eq_static"],
            n_npz_label_eq_raw=info["n_npz_label_eq_raw"], n_static_eq_raw=info["n_static_eq_raw"],
            objects=objs_cam,
            colliders=[dict(prim_path=pp, npz_rows=r["rows"], identity_resid_m=r["resid"], identity_margin_m=r["margin"])
                       for pp, r in sorted(resolved_by_cam[cam].items()) if pp not in {o["prim_path"] for o in so}],
            physics_tar=phys[cam]["com"][0], physics_members=[f"{world}/{cam}_{k}" for k in
                                                              ("com.npz", "velocity.npz", "rot.npz", "spin.npz", "static.json")],
            cameras_tar=phys[cam]["camera"][0] if "camera" in phys[cam] else None,
            cameras_member=f"{world}/{cam}.json" if "camera" in phys[cam] else None,
            segmentation_tar=s.get("tar"), seg_raw_member=s.get("raw_member"), seg_vis_member=s.get("vis_member"),
            seg_png_pattern=f"{world}/{cam}_{{frame:04d}}.png" if s.get("tar") else None, seg_n_png=s.get("n_png"),
            seg_source=s.get("seg_source"),
            video_tar=vt[0] if vt else None, video_member=vt[1] if vt else None,
            captions_tar=cap.get(cam, (None, []))[0], caption_members=cap.get(cam, (None, []))[1],
            scene_tar=scene_ref[0] if scene_ref else None, prep_video=prep if os.path.exists(prep) else None))
    return rows, traj


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _vec(x):
    try:
        return [float(v) for v in x]
    except (TypeError, ValueError):
        return None


def build_jobs(scenario, allow_missing=False):
    L = {m: load_listing(m, scenario, allow_missing) for m in MODALITIES}
    per = collections.defaultdict(lambda: dict(phys=collections.defaultdict(dict), scene=None, meta=None, seg={}, vid={},
                                               cap=collections.defaultdict(lambda: [None, []])))
    for mem, ref in L["physics"].items():
        world, fn = mem.split("/", 1)
        for kind in ("com", "velocity", "rot", "spin"):
            if fn.endswith(f"_{kind}.npz"):
                per[world]["phys"][fn[: -len(kind) - 5]][kind] = ref
        if fn.endswith("_static.json"):
            per[world]["phys"][fn[:-12]]["static"] = ref
    for mem, ref in L["cameras"].items():
        world, fn = mem.split("/", 1)
        if world in per and fn.endswith(".json") and fn[:-5] in per[world]["phys"]:
            per[world]["phys"][fn[:-5]]["camera"] = ref
    for mem, ref in L["scene"].items():
        world, fn = mem.split("/", 1)
        if world in per and fn.endswith(".usda"):
            per[world]["scene"] = ref
        if world in per and fn == "metadata.json":
            per[world]["meta"] = ref
    seg_png = collections.Counter()
    for mem, ref in L["segmentation"].items():
        world, fn = mem.split("/", 1)
        if world not in per:
            continue
        if fn.endswith("_raw.json"):
            per[world]["seg"].setdefault(fn[:-9], {}).update(raw_member=mem, raw_ref=ref, tar=ref[0])
        elif fn.endswith("_visibility.npz"):
            per[world]["seg"].setdefault(fn[:-15], {}).update(vis_member=mem, tar=ref[0])
        elif fn.endswith(".png"):
            seg_png[(world, fn.rsplit("_", 1)[0])] += 1
    for (world, cam), k in seg_png.items():
        per[world]["seg"].setdefault(cam, {})["n_png"] = k
    for w in per.values():
        for s in w["seg"].values():
            s["seg_source"] = "listing"
    # Segmentation tars not listed yet (the full listing of 770 tars / 47M members is slow on this NFS): the world -> tar
    # membership is inferred from tar boundaries (first member of every tar; tars are filled in lexicographic world order,
    # checked on every listed tar by check_seg_boundaries). Member names follow the fixed pattern seen in listed tars.
    unlisted = [t for t in tar_paths("segmentation", scenario) if not os.path.exists(listing_path(t))]
    if unlisted:
        bounds = sorted((first_member(t).split("/", 1)[0], t) for t in tar_paths("segmentation", scenario))
        firsts = [b[0] for b in bounds]
        for world, w in per.items():
            k = bisect.bisect_right(firsts, world) - 1
            if k < 0 or bounds[k][1] not in unlisted:
                continue
            for cam in w["phys"]:
                w["seg"][cam] = dict(tar=bounds[k][1], raw_member=f"{world}/{cam}_raw.json", vis_member=f"{world}/{cam}_visibility.npz",
                                     n_png=None, seg_source="boundary")
    for mem, ref in L["videos"].items():
        world, fn = mem.split("/", 1)
        if world in per and fn.endswith(".mp4") and not DERIVED_VIDEO.search(fn):
            per[world]["vid"][fn[:-4]] = (ref[0], mem)
    for mem, ref in L["captions"].items():
        world, fn = mem.split("/", 1)
        if world in per:
            cam = next((c for c in per[world]["phys"] if fn.startswith(c + ".") or fn.startswith(c + "_")), None)
            if cam:
                per[world]["cap"][cam][0] = ref[0]; per[world]["cap"][cam][1].append(mem)
    orphans = {m: sorted({k.split("/", 1)[0] for k in L[m]} - set(per)) for m in MODALITIES if m != "physics"}
    jobs = []
    for world in sorted(per, key=lambda w: split_world(w)[2]):
        w = per[world]
        phys = {c: v for c, v in w["phys"].items() if {"com", "velocity", "static"} <= set(v)}
        jobs.append((scenario, world, phys, w["scene"], w["meta"], w["seg"], w["vid"], {c: tuple(v) for c, v in w["cap"].items()}))
    return jobs, orphans


def seg_raw_loader(job):
    """Small segmentation json are read here (they live inside the 1.9 TB segmentation tars, random access by offset)."""
    scenario, world, phys, scene_ref, meta_ref, seg, vid, cap = job
    for cam, s in seg.items():
        if s.get("raw_ref"):
            s["raw_objects"] = json.loads(read_member(*s.pop("raw_ref")))["objects"]
    return process_world(job)


def cmd_build(a):
    os.makedirs(f"{OUT}/traj", exist_ok=True)
    summary_orphans = {}
    for scenario in a.scenarios.split(","):
        t0 = time.time()
        jobs, orphans = build_jobs(scenario, a.allow_missing_listing)
        summary_orphans[scenario] = {m: len(v) for m, v in orphans.items()}
        print(f"[build] {scenario}: worlds with physics {len(jobs)}; worlds present only in other modalities: "
              f"{ {m: len(v) for m, v in orphans.items()} }", flush=True)
        if a.limit:
            jobs = jobs[: a.limit]
        rows, trajs = [], []
        with ProcessPoolExecutor(a.workers) as ex:
            for i, (r, tr) in enumerate(ex.map(seg_raw_loader, jobs, chunksize=8)):
                rows.extend(r)
                if tr is not None:
                    trajs.append(tr)
                if (i + 1) % 1000 == 0:
                    print(f"[build] {scenario} {i+1}/{len(jobs)} t={time.time()-t0:.0f}s", flush=True)
        tag = f"_limit{a.limit}" if a.limit else ""
        write_scenario_tables(rows, scenario, tag)
        # trajectory cache: ragged objects -> flat arrays
        worlds = [t[0] for t in trajs]
        nobj = np.array([len(t[1]) for t in trajs], np.int32)
        T = max((t[3].shape[1] for t in trajs), default=0)
        arr = np.concatenate([t[3] for t in trajs]) if trajs else np.zeros((0, T, 3), np.float32)
        np.savez(f"{OUT}/traj/{scenario}{tag}.npz", worlds=np.array(worlds), n_obj=nobj,
                 prims=np.array([p for t in trajs for p in t[1]]), dynamic=np.array([x for t in trajs for x in t[2]], bool), traj=arr)
        print(f"[build] {scenario}: {len(rows)} camera rows, traj objects {arr.shape} in {time.time()-t0:.0f}s", flush=True)
    with open(f"{OUT}/build_orphans.json", "w") as f:
        json.dump(summary_orphans, f, indent=1)


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    raise TypeError(type(x))


OBJ_COLS = ["prim_path", "slot", "segmentation_color", "mass", "static_friction", "dynamic_friction", "restitution", "density",
            "inertia", "center_of_mass_local", "collision_enabled", "npz_rows", "identity_resid_m", "identity_margin_m"]


def write_scenario_tables(rows, scenario, tag=""):
    """world_index_<sc>.parquet: one row per clip, object lists stored as JSON strings (objects_json, colliders_json) because a
    deeply nested list<struct<list<...>>> column made pyarrow take minutes per column read; objects_<sc>.parquet: the same
    objects flattened, one row per (world, camera, dynamic object), plain columns."""
    flat = []
    for r in rows:
        objs = r.pop("objects"); cols = r.pop("colliders")
        r["objects_json"] = json.dumps(objs, default=_jsonable)
        r["colliders_json"] = json.dumps(cols, default=_jsonable)
        for k, o in enumerate(objs):
            f = {c: o.get(c) for c in OBJ_COLS}
            for c in ("segmentation_color", "inertia", "center_of_mass_local", "npz_rows"):
                v = f[c]
                f[c] = None if v is None else [float(x) if c in ("inertia", "center_of_mass_local") else int(x) for x in v]
            flat.append(dict(scenario=r["scenario"], world=r["world"], camera=r["camera"], obj_index=k, **f))
    pq.write_table(pa.Table.from_pylist(rows), f"{OUT}/world_index_{scenario}{tag}.parquet", row_group_size=8192)
    pq.write_table(pa.Table.from_pylist(flat), f"{OUT}/objects_{scenario}{tag}.parquet", row_group_size=65536)


def convert_legacy(scenario):
    """Per-scenario files written before write_scenario_tables existed (nested 'objects' column) -> new layout."""
    p = f"{OUT}/world_index_{scenario}.parquet"
    if "objects" not in pq.read_schema(p).names:
        return scenario, "already converted"
    rows = pq.read_table(p).to_pylist()
    write_scenario_tables(rows, scenario)
    return scenario, f"converted {len(rows)} rows"


def cmd_merge(a):
    todo = [s for s in SCENARIOS if os.path.exists(f"{OUT}/world_index_{s}.parquet")]
    with ProcessPoolExecutor(min(a.workers, len(todo))) as ex:
        for sc, msg in ex.map(convert_legacy, todo):
            print(f"[merge] {sc}: {msg}", flush=True)
    for name in ("world_index", "objects"):
        tabs = [pq.read_table(f"{OUT}/{name}_{s}.parquet") for s in todo]
        t = pa.concat_tables(tabs, promote_options="default")
        pq.write_table(t, f"{OUT}/{name}.parquet", row_group_size=8192 if name == "world_index" else 65536)
        print(f"[merge] {name}.parquet rows={t.num_rows} from {len(tabs)} scenarios", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("list"); p.add_argument("--workers", type=int, default=16); p.add_argument("--modalities", default=",".join(MODALITIES))
    p = sp.add_parser("build"); p.add_argument("--workers", type=int, default=16); p.add_argument("--scenarios", default=",".join(SCENARIOS))
    p.add_argument("--limit", type=int, default=0); p.add_argument("--allow-missing-listing", action="store_true")
    p = sp.add_parser("merge"); p.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    {"list": cmd_list, "build": cmd_build, "merge": cmd_merge}[a.cmd](a)
