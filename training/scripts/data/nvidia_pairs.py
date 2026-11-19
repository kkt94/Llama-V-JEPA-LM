"""Event observability, view (in)variance, "same past / different future" pairs and split proposal for the NVIDIA physics data.

usage: python -I nvidia_pairs.py [--workers 10] [--scenarios billiards,...] [--max-group 6000] [--seed 0]
Needs world_index.parquet + traj/<scenario>.npz (nvidia_index.py build). CPU only, deterministic given --seed.

Definitions (all from world-frame COM trajectories of USDA-resolved bodies; contact is never inferred):
  speed      |central finite difference of COM| * fps                      (m/s)
  onset      first frame with |p_t - p_0| > 1 cm, for bodies at rest at frame 0 (speed < 0.05 m/s over frames 0..2)
  initially moving   speed >= 0.05 m/s at frames 0..2 (launched before the clip starts: cue ball, projectiles, ...)
  stop       last frame with speed > 0.05 m/s, if followed by >= 10 frames at rest before the clip ends
  drop       min_t z_t < z_0 - 2 cm (COM lowered: fall / topple); world up = +Z (USDA upAxis, confirmed by free fall)
  leave frame   projected COM inside the image at frame 0 and outside at a later frame (per camera)
  first mover   for two bodies at rest at frame 0 that both start moving: decidable if onsets differ by >= 3 frames
Pairs (spec 6.5 L_change and 11.4B): inside one scenario, worlds are compared only when they have the same set of object slots
(prim name without the randomised material token), per camera name. Observable past = image-plane trajectories (u/W, v/W)
of all slots over frames [0, k) subsampled every 3 frames; distance = RMS over slots x frames. Future difference at frame k+h:
  lr_flip   some slot pair has opposite image-plane left/right order in the two worlds (|du| >= 20 px in both), with the same order
            at frame k-1 in both worlds
  z_flip    some slot pair has opposite 3D height order (|dz| >= 3 cm in both), same order at frame k-1
  move_flip some dynamic slot moves > 5 cm between k and k+h in one world and < 1 cm in the other
Outputs: events_summary.json, invariance_summary.json, pairs_counts.parquet, pairs_examples.json, splits.parquet, splits_summary.json
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # shared host: one BLAS thread per process
    _os.environ.setdefault(_v, "1")

import argparse, collections, hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvidia_common import OUT, SCENARIOS, world_to_cam, project, slot_key, MATERIALS

FPS = 30.0
W_IMG, H_IMG = 1920, 1080
REST_V, ONSET_D, DROP_D = 0.05, 0.01, 0.02
LR_PX, DZ_M, GAP_FR = 20.0, 0.03, 3
K_GRID, H_GRID, EPS_GRID = (15, 30, 45), (15, 30, 60), (0.01, 0.02, 0.04)   # image space: fraction of image width
EPS_M_GRID = (0.02, 0.05, 0.10)                                                  # world space: metres (RMS)
SUB = 3
MAXQ = 40
MAX_EVAL = 100_000
TAG = os.environ.get("NV_TAG", "")      # test runs only: suffix of traj/<scenario><TAG>.npz


def load_scenario(sc):
    z = np.load(f"{OUT}/traj/{sc}{TAG}.npz")
    worlds, nobj, prims, dyn, traj = z["worlds"], z["n_obj"], z["prims"], z["dynamic"], z["traj"]
    off = np.concatenate([[0], np.cumsum(nobj)])
    W = {}
    for i, w in enumerate(worlds):
        s, e = off[i], off[i + 1]
        W[str(w)] = dict(prims=[str(p) for p in prims[s:e]], dyn=dyn[s:e].copy(), traj=traj[s:e].astype(np.float64))
    return W


def body_events(tr):
    """tr [n,T,3] world frame -> dict of per-body arrays."""
    n, T, _ = tr.shape
    v = np.zeros((n, T))
    if T > 2:
        v[:, 1:-1] = np.linalg.norm(tr[:, 2:] - tr[:, :-2], axis=-1) * FPS / 2
        v[:, 0] = np.linalg.norm(tr[:, 1] - tr[:, 0], axis=-1) * FPS
        v[:, -1] = np.linalg.norm(tr[:, -1] - tr[:, -2], axis=-1) * FPS
    disp = np.linalg.norm(tr - tr[:, :1], axis=-1)
    init_moving = v[:, :3].max(1) >= REST_V
    moved = disp > ONSET_D
    onset = np.where(moved.any(1), moved.argmax(1), -1)
    onset[init_moving] = 0
    moving = v > REST_V
    last = np.where(moving.any(1), T - 1 - moving[:, ::-1].argmax(1), -1)
    stopped = (last >= 0) & (last <= T - 11)
    drop = tr[:, :, 2].min(1) < tr[:, 0, 2] - DROP_D
    return dict(v=v, disp=disp, init_moving=init_moving, onset=onset, last_move=last, stopped=stopped, drop=drop)


def close_pairs(X, ndim, grid, chunk=2048):
    """All pairs (i<j) with RMS distance sqrt(|xi-xj|^2/ndim) < eps for each eps in grid, plus every row's NN distance.
    Exact (chunked brute force). Returns ({eps: int64 [n,3] = (i, j, round(dist*1e6))}, nn)."""
    N = len(X)
    X = np.nan_to_num(X.astype(np.float64), nan=10.0, posinf=10.0, neginf=-10.0)
    sq = (X * X).sum(1)
    out = {e: [] for e in grid}
    nn = np.full(N, np.inf)
    emax = max(grid)
    for i0 in range(0, N, chunk):
        D = np.sqrt(np.maximum(sq[i0:i0 + chunk, None] + sq[None] - 2 * X[i0:i0 + chunk] @ X.T, 0) / ndim)
        r = np.arange(len(D))
        D[r, r + i0] = np.inf
        nn[i0:i0 + len(D)] = D.min(1)
        ii, jj = np.nonzero(D < emax)
        keep = jj > ii + i0
        ii, jj = ii[keep], jj[keep]
        dd = D[ii, jj]
        for e in grid:
            m = dd < e
            out[e].append(np.stack([ii[m] + i0, jj[m], np.round(dd[m] * 1e6).astype(np.int64)], 1))
    return {e: (np.concatenate(v) if v else np.zeros((0, 3), np.int64)) for e, v in out.items()}, nn


def descriptor(prim):
    """'(type, material)' visible description used to check whether an object can be referred to unambiguously."""
    name = prim.split("/")[-1]
    mat = next((m for m in MATERIALS if name.lower().endswith("_" + m)), "")
    head = name.split("_")[0]
    return head, mat


# ====================================================================== per-scenario analysis
def analyse_scenario(args):
    sc, max_group, seed, index_path = args
    t0 = time.time()
    idx = pd.read_parquet(index_path, filters=[("scenario", "==", sc)],
                          columns=["world", "camera", "usd_transform", "fxn", "T_npz"])
    Wd = load_scenario(sc)
    cams = sorted(idx.camera.unique())
    camtab = {(r.world, r.camera): (np.asarray(r.usd_transform), r.fxn) for r in idx.itertuples()}
    print(f"[{sc}] worlds with trajectories {len(Wd)}, index clips {len(idx)}, cameras {cams}", flush=True)
    ev = collections.Counter(); hist = collections.defaultdict(list)
    inv = collections.Counter()
    proj = {}   # (world, cam) -> u,v arrays [n,T]
    for w, d in Wd.items():
        tr = d["traj"]; dyn = d["dyn"]
        if tr.shape[0] == 0:
            ev["worlds_without_resolved_bodies"] += 1
            continue
        e = body_events(tr[dyn]) if dyn.any() else None
        ev["worlds"] += 1
        if e is None:
            continue
        nd = int(dyn.sum())
        ev["dyn_bodies"] += nd
        ev["init_moving"] += int(e["init_moving"].sum())
        rest_moved = (~e["init_moving"]) & (e["onset"] > 0)
        ev["rest_then_moved"] += int(rest_moved.sum())
        ev["never_moved"] += int((e["onset"] < 0).sum())
        ev["stopped_before_end"] += int(e["stopped"].sum())
        ev["moved_and_still_moving_at_end"] += int(((e["onset"] >= 0) & ~e["stopped"]).sum())
        ev["drop"] += int(e["drop"].sum())
        hist["onset_frames"].extend(e["onset"][rest_moved].tolist())
        on = e["onset"][rest_moved]
        if len(on) >= 2:
            dif = np.abs(on[:, None] - on[None])[np.triu_indices(len(on), 1)]
            ev["first_mover_pairs"] += len(dif); ev["first_mover_decidable"] += int((dif >= GAP_FR).sum())
            ev["first_mover_ties_0_1fr"] += int((dif <= 1).sum())
        # unique visible descriptors -> referable objects
        desc = [descriptor(p) for p, dd in zip(d["prims"], dyn) if dd]
        cnt = collections.Counter(desc)
        ev["dyn_bodies_unique_descriptor"] += sum(cnt[x] == 1 for x in desc)
        # ---------- invariance: 3D questions at mid / end frame (same answer in every camera)
        T = tr.shape[1]
        # pair-level counts are O(n^2 T); worlds with > MAXQ dynamic bodies use a fixed per-world subsample of MAXQ bodies
        sel = np.arange(nd)
        if nd > MAXQ:
            sel = np.sort(np.random.default_rng(int(hashlib.sha1(w.encode()).hexdigest()[:8], 16)).choice(nd, MAXQ, replace=False))
            ev["worlds_pair_counts_subsampled"] += 1
        trd = tr[dyn][sel]
        iu = np.triu_indices(len(sel), 1)
        for t in (T // 2, T - 1):
            sp = e["v"][sel, t]
            a, b = sp[iu[0]], sp[iu[1]]
            ok = (np.maximum(a, b) > 0.1) & (np.maximum(a, b) >= 1.2 * np.minimum(a, b))
            inv["speed_rank_3d"] += int(ok.sum())
            dz = np.abs(trd[iu[0], t, 2] - trd[iu[1], t, 2])
            inv["height_rank_3d"] += int((dz >= DZ_M).sum())
        if len(on) >= 2:
            inv["first_moved_3d"] += int((dif >= GAP_FR).sum())
        # ---------- view-dependent image-plane questions
        lr_sign = []
        for cam in cams:
            if (w, cam) not in camtab:
                continue
            M, fxn = camtab[(w, cam)]
            pc = world_to_cam(tr, M)
            u, v, front = project(pc, fxn, W_IMG, H_IMG)
            proj[(w, cam)] = (np.where(front, u, np.nan).astype(np.float32), np.where(front, v, np.nan).astype(np.float32))
            inside = front & (u >= 0) & (u < W_IMG) & (v >= 0) & (v < H_IMG)
            ud, ins_d = u[dyn], inside[dyn]
            ev["cam_dyn_bodies"] += nd
            ev["cam_dyn_inside_f0"] += int(ins_d[:, 0].sum())
            ev["cam_dyn_leave_frame"] += int((ins_d[:, 0] & ~ins_d.all(1)).sum())
            ev["cam_dyn_never_inside"] += int((~ins_d.any(1)).sum())
            ud, ins_d = ud[sel], ins_d[sel]
            t = T - 1
            du = ud[iu[0], t] - ud[iu[1], t]
            both = ins_d[iu[0], t] & ins_d[iu[1], t]
            okp = both & (np.abs(du) >= LR_PX)
            inv["lr_image_instances"] += int(okp.sum())
            lr_sign.append(np.where(okp, np.sign(du), 0))
            # crossings in the image plane (left/right order changes with >= LR_PX margin on both sides)
            duT = ud[iu[0]] - ud[iu[1]]
            bothT = ins_d[iu[0]] & ins_d[iu[1]]
            s = np.where(bothT & (np.abs(duT) >= LR_PX), np.sign(duT), 0)
            nz = s != 0
            rr = np.arange(len(s))
            first = s[rr, nz.argmax(1)] * nz.any(1) if len(s) else np.zeros(0)
            lastv = s[rr, s.shape[1] - 1 - nz[:, ::-1].argmax(1)] * nz.any(1) if len(s) else np.zeros(0)
            ev["cam_pair_lr_crossings"] += int(((first * lastv) < 0).sum())
            ev["cam_pairs"] += len(first)
        if len(lr_sign) >= 2:
            S = np.stack(lr_sign)                       # [cams, pairs]
            defined = (S != 0).sum(0)
            pos = (S > 0).sum(0); neg = (S < 0).sum(0)
            inv["lr_pairs_defined_in_>=2_cams"] += int((defined >= 2).sum())
            inv["lr_pairs_answer_differs_across_cams"] += int(((pos > 0) & (neg > 0)).sum())
            inv["lr_pairs_same_answer_all_defined_cams"] += int(((defined >= 2) & ~((pos > 0) & (neg > 0))).sum())
    hist_s = {k: np.percentile(v, [10, 50, 90]).tolist() if v else None for k, v in hist.items()}
    print(f"[{sc}] events/invariance done in {time.time()-t0:.0f}s", flush=True)

    # ====================================================================== pairs
    slot_sig = {}
    for w, d in Wd.items():
        if d["traj"].shape[0] == 0:
            continue
        slots = [slot_key(p) for p in d["prims"]]
        if len(set(slots)) != len(slots):
            continue
        slot_sig[w] = tuple(sorted(zip(slots, d["dyn"].tolist())))
    groups = collections.defaultdict(list)
    for w, s in slot_sig.items():
        groups[s].append(w)
    # a slot is "stable" if it appears in > 1 world; random-asset scenes have unique names per world
    slot_worlds = collections.Counter(s for sig in groups for s, _ in sig for _ in range(len(groups[sig])))
    stable_frac = np.mean([slot_worlds[s] > 1 for sig in groups for s, _ in sig]) if groups else 0.0
    gsz = sorted((len(v) for v in groups.values()), reverse=True)
    pair_rows, examples = [], []
    n_ex = collections.Counter()
    rng = np.random.default_rng(seed)
    n_pairable_worlds = sum(n for n in gsz if n >= 2)
    print(f"[{sc}] slot-signature groups {len(groups)}, largest {gsz[:5]}, worlds in groups>=2: {n_pairable_worlds}, "
          f"stable-slot fraction {stable_frac:.3f}", flush=True)
    T_all = min((d["traj"].shape[1] for d in Wd.values() if d["traj"].shape[0]), default=0)
    for sig, ws in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(ws) < 2:
            continue
        ws = sorted(ws)
        if len(ws) > max_group:
            ws = sorted(rng.choice(ws, max_group, replace=False).tolist())
        order = [s_ for s_, _ in sig]
        dynm = np.array([dd for _, dd in sig])
        perm = {w: [{slot_key(p): i for i, p in enumerate(Wd[w]["prims"])}[s_] for s_ in order] for w in ws}
        TR = np.stack([Wd[w]["traj"][perm[w]] for w in ws])          # [N, S, T, 3] world frame, slot order
        N, S = TR.shape[:2]
        iu = np.triu_indices(S, 1)
        ON = np.stack([body_events(TR[i][dynm])["onset"] for i in range(N)]) if dynm.any() else np.zeros((N, 0), int)
        cam_ok = [c for c in cams if all((w, c) in proj for w in ws)]
        UV = {c: (np.stack([proj[(w, c)][0][perm[w]] for w in ws]), np.stack([proj[(w, c)][1][perm[w]] for w in ws]))
              for c in cam_ok}
        campos = {c: np.stack([np.asarray(camtab[(w, c)][0]).reshape(4, 4)[3, :3] for w in ws]) for c in cam_ok}

        dpair = dynm[iu[0]] & dynm[iu[1]]          # questions are asked about dynamic objects only (not table / bumpers)

        def lr(c, t):
            """image-plane left/right sign of every slot pair; 0 if undecided (|du| < LR_PX, a COM outside the image, or not
            a dynamic pair)."""
            U, V = UV[c]
            ins = (U[:, :, t] >= 0) & (U[:, :, t] < W_IMG) & (V[:, :, t] >= 0) & (V[:, :, t] < H_IMG)
            du = U[:, iu[0], t] - U[:, iu[1], t]
            ok = ins[:, iu[0]] & ins[:, iu[1]] & (np.abs(du) >= LR_PX) & dpair[None]
            return np.where(ok, np.sign(du), 0)

        def zo(t):
            dz = TR[:, iu[0], t, 2] - TR[:, iu[1], t, 2]
            return np.where((np.abs(dz) >= DZ_M) & dpair[None], np.sign(dz), 0)

        def flips(P, k, tf, cam_list):
            """future-difference indicators for close pairs P (rows i, j)."""
            a, b = P[:, 0], P[:, 1]
            out = {}
            Zp, Zf = zo(k - 1), zo(tf)
            out["z"] = ((Zf[a] * Zf[b] < 0) & (Zp[a] == Zp[b]) & (Zp[a] != 0)).any(1)
            mv = np.linalg.norm(TR[:, :, tf] - TR[:, :, k - 1], axis=-1)
            out["move"] = ((((mv[a] > 0.05) & (mv[b] < 0.01)) | ((mv[b] > 0.05) & (mv[a] < 0.01))) & dynm[None]).any(1)
            lrf = np.zeros(len(P), bool); lr_detail = {}
            for c in cam_list:
                Lp, Lf = lr(c, k - 1), lr(c, tf)
                f = (Lf[a] * Lf[b] < 0) & (Lp[a] == Lp[b]) & (Lp[a] != 0)
                lr_detail[c] = f
                lrf |= f.any(1)
            out["lr"] = lrf
            ps = (np.linalg.norm(TR[:, :, k - 1] - TR[:, :, 0], axis=-1)[:, dynm] < ONSET_D).all(1) if dynm.any() else np.ones(N, bool)
            out["static_prefix"] = ps[a] & ps[b]
            return out, lr_detail, mv

        for space in ["image:" + c for c in cam_ok] + ["world"]:
            cam = space.split(":", 1)[1] if space.startswith("image:") else None
            for k in K_GRID:
                fr = np.arange(0, k, SUB)
                if cam:
                    U, V = UV[cam]
                    # normalised image coordinates (both divided by the width); points behind the camera or far outside the
                    # image are clipped to a band around the frame so that one off-screen collider cannot dominate
                    Un = np.clip(np.nan_to_num(U / W_IMG, nan=-0.25), -0.25, 1.25)
                    Vn = np.clip(np.nan_to_num(V / W_IMG, nan=-0.25), -0.25, H_IMG / W_IMG + 0.25)
                    X = np.concatenate([Un[:, :, fr].reshape(N, -1), Vn[:, :, fr].reshape(N, -1)], 1)
                    ndim, grid = S * len(fr), EPS_GRID
                else:
                    X = TR[:, :, fr].reshape(N, -1)
                    ndim, grid = S * len(fr), EPS_M_GRID
                P_by, nn = close_pairs(X, ndim, grid)
                pair_rows.append(dict(scenario=sc, space=space, camera=cam, k=k, h=0, eps=np.nan, group_size=N, n_slots=S,
                                      kind="nn_dist", nn_p10=float(np.percentile(nn, 10)), nn_p50=float(np.percentile(nn, 50)),
                                      nn_p90=float(np.percentile(nn, 90))))
                for e_ in grid:
                    P_all = P_by[e_]
                    for h in H_GRID:
                        P = P_all
                        tf = k + h
                        if tf >= T_all:
                            continue
                        rec = dict(scenario=sc, space=space, camera=cam, k=k, h=h, eps=e_, group_size=N, n_slots=S,
                                   kind="same_past_diff_future", n_close_pairs=len(P),
                                   n_worlds_in_close=len(np.unique(P[:, :2])) if len(P) else 0)
                        if len(P) > MAX_EVAL:
                            # flip statistics on a fixed random subset of the close pairs (counts NOT scaled; eval_n says how many)
                            P = P[np.sort(np.random.default_rng(seed).choice(len(P), MAX_EVAL, replace=False))]
                        rec["eval_n"] = len(P)
                        if len(P):
                            f, lr_detail, mv = flips(P, k, tf, [cam] if cam else cam_ok)
                            anyf = f["lr"] | f["z"] | f["move"]
                            rec.update(n_lr_flip=int(f["lr"].sum()), n_z_flip=int(f["z"].sum()), n_move_flip=int(f["move"].sum()),
                                       n_any_flip=int(anyf.sum()), n_close_static_prefix=int(f["static_prefix"].sum()),
                                       n_worlds_any_flip=len(np.unique(P[anyf][:, :2])) if anyf.any() else 0)
                            if cam is None and cam_ok:
                                cd = np.mean([np.linalg.norm(campos[c][P[:, 0]] - campos[c][P[:, 1]], axis=1) for c in cam_ok], 0)
                                rec["median_cam_pos_diff_m"] = float(np.median(cd))
                            if f["lr"].any() and n_ex[(space, k, h, e_)] < 2:
                                for q in np.nonzero(f["lr"])[0][: 2 - n_ex[(space, k, h, e_)]]:
                                    n_ex[(space, k, h, e_)] += 1
                                    c = cam if cam else next(c_ for c_ in cam_ok if lr_detail[c_][q].any())
                                    pi = int(np.nonzero(lr_detail[c][q])[0][0])
                                    ia, ib = int(P[q, 0]), int(P[q, 1]); Uc = UV[c][0]
                                    sa, sb = order[iu[0][pi]], order[iu[1][pi]]
                                    examples.append(dict(
                                        scenario=sc, space=space, camera=c, world_A=ws[ia], world_B=ws[ib], k=k, h=h, eps=e_,
                                        past_rms_dist=float(P[q, 2] * 1e-6), slot_pair=[sa, sb], frame_past_end=k - 1, frame_future=tf,
                                        du_px_A_past=float(Uc[ia, iu[0][pi], k - 1] - Uc[ia, iu[1][pi], k - 1]),
                                        du_px_B_past=float(Uc[ib, iu[0][pi], k - 1] - Uc[ib, iu[1][pi], k - 1]),
                                        du_px_A_future=float(Uc[ia, iu[0][pi], tf] - Uc[ia, iu[1][pi], tf]),
                                        du_px_B_future=float(Uc[ib, iu[0][pi], tf] - Uc[ib, iu[1][pi], tf]),
                                        cam_pos_diff_m=float(np.linalg.norm(campos[c][ia] - campos[c][ib])),
                                        max_move_after_k_A_m=float(mv[ia][dynm].max()), max_move_after_k_B_m=float(mv[ib][dynm].max())))
                        pair_rows.append(rec)
            # spec 11.4B mirror: similar final window (last 30 frames), different past (first-mover order flips)
            fr = np.arange(T_all - 30, T_all, SUB)
            if cam:
                U, V = UV[cam]
                Un = np.clip(np.nan_to_num(U / W_IMG, nan=-0.25), -0.25, 1.25)
                Vn = np.clip(np.nan_to_num(V / W_IMG, nan=-0.25), -0.25, H_IMG / W_IMG + 0.25)
                X = np.concatenate([Un[:, :, fr].reshape(N, -1), Vn[:, :, fr].reshape(N, -1)], 1); grid = EPS_GRID
            else:
                X = TR[:, :, fr].reshape(N, -1); grid = EPS_M_GRID
            P_by, _ = close_pairs(X, S * len(fr), grid)
            iu2 = np.triu_indices(ON.shape[1], 1)
            for e_ in grid:
                P = P_by[e_]; flips_ = 0
                if len(P) and ON.shape[1] >= 2:
                    oa, ob = ON[P[:, 0]], ON[P[:, 1]]
                    da = oa[:, iu2[0]] - oa[:, iu2[1]]; db = ob[:, iu2[0]] - ob[:, iu2[1]]
                    valid = (oa[:, iu2[0]] > 0) & (oa[:, iu2[1]] > 0) & (ob[:, iu2[0]] > 0) & (ob[:, iu2[1]] > 0)
                    fl = valid & (np.abs(da) >= GAP_FR) & (np.abs(db) >= GAP_FR) & (np.sign(da) != np.sign(db))
                    flips_ = int(fl.any(1).sum())
                pair_rows.append(dict(scenario=sc, space=space, camera=cam, k=-30, h=0, eps=e_, group_size=N, n_slots=S,
                                      kind="same_future_diff_past", n_close_pairs=len(P), n_onset_order_flip=flips_))
    print(f"[{sc}] pairs done in {time.time()-t0:.0f}s", flush=True)
    return sc, dict(events=dict(ev), hist=hist_s), dict(inv), pair_rows, examples, dict(
        n_groups=len(groups), group_sizes_top=gsz[:10], worlds_in_groups_ge2=n_pairable_worlds, stable_slot_fraction=float(stable_frac),
        n_worlds_with_slot_sig=len(slot_sig), n_worlds=len(Wd))


def make_splits():
    idx = pd.read_parquet(f"{OUT}/world_index.parquet", columns=["scenario", "world", "camera"])
    w = idx.groupby(["scenario", "world"]).camera.count().rename("n_cams").reset_index()
    h = w.world.map(lambda s: int(hashlib.sha1(s.encode()).hexdigest()[:8], 16) % 100)
    w["split"] = np.where(h < 90, "train", np.where(h < 95, "val", "test"))
    w["hash100"] = h
    w.to_parquet(f"{OUT}/splits.parquet")
    by = w.groupby(["scenario", "split"]).agg(worlds=("world", "count"), clips=("n_cams", "sum")).reset_index()
    loso = {sc: dict(test_worlds=int((w.scenario == sc).sum()), test_clips=int(w[w.scenario == sc].n_cams.sum()),
                     train_worlds=int((w.scenario != sc).sum()), train_clips=int(w[w.scenario != sc].n_cams.sum()))
            for sc in sorted(w.scenario.unique())}
    return dict(by_world=by.to_dict("records"), totals=w.groupby("split").agg(worlds=("world", "count"), clips=("n_cams", "sum")).reset_index().to_dict("records"),
                loso=loso)


def verify_examples(n_per=1, settings=((30, 30, 0.02), (30, 30, 0.04), (15, 30, 0.02), (45, 30, 0.04))):
    """Independent re-computation of example pairs from the raw physics npz + camera json in the tars (not the trajectory cache):
    for each picked example, the image-plane left/right offset du of the slot pair at frame_past_end and frame_future."""
    from nvidia_common import read_member, load_npz_bytes, listing_path, project as proj_
    import pyarrow.parquet as pq
    ex = json.load(open(f"{OUT}/pairs_examples.json"))
    print(f"[verify] examples available: {len(ex)}", flush=True)
    picked, seen = [], set()
    for st in settings:
        for e in ex:
            if (e["k"], e["h"], e["eps"]) == st and e["space"].startswith("image") and e["scenario"] not in seen:
                picked.append(e); seen.add(e["scenario"])
    idx = pd.read_parquet(f"{OUT}/world_index.parquet", columns=["world", "camera", "physics_tar", "cameras_tar", "objects_json",
                                                                  "colliders_json"],
                          filters=[("world", "in", sorted({e["world_A"] for e in picked} | {e["world_B"] for e in picked}))])
    lst = {}

    def ref(tar, member):
        if tar not in lst:
            lst[tar] = pq.read_table(listing_path(tar)).to_pandas().set_index("member")
        r = lst[tar].loc[member]
        return tar, int(r["offset"]), int(r["size"])

    out = []
    for e in picked:
        res = dict(scenario=e["scenario"], camera=e["camera"], world_A=e["world_A"], world_B=e["world_B"], slots=e["slot_pair"],
                   frames=[e["frame_past_end"], e["frame_future"]])
        for side in "AB":
            w = e[f"world_{side}"]
            r = idx[(idx.world == w) & (idx.camera == e["camera"])].iloc[0]
            com = load_npz_bytes(read_member(*ref(r.physics_tar, f"{w}/{e['camera']}_com.npz")))["data"].astype(float)
            cj = json.loads(read_member(*ref(r.cameras_tar, f"{w}/{e['camera']}.json")))
            fxn = cj["camera"]["focal_length"][0][0]
            objs = json.loads(r.objects_json) + json.loads(r.colliders_json)
            row = {slot_key(o["prim_path"]): o["npz_rows"][0] for o in objs if o["npz_rows"]}
            ia, ib = row[e["slot_pair"][0]], row[e["slot_pair"][1]]
            for lab, t in (("past", e["frame_past_end"]), ("future", e["frame_future"])):
                u, v, _ = proj_(com[[ia, ib], t], fxn)
                res[f"du_{side}_{lab}_raw"] = float(u[0] - u[1])
                res[f"du_{side}_{lab}_cache"] = e[f"du_px_{side}_{lab}"]
            res[f"prims_{side}"] = [o["prim_path"] for o in objs if slot_key(o["prim_path"]) in e["slot_pair"]]
        res["past_rms_dist_px"] = e["past_rms_dist"] * W_IMG
        out.append(res)
    json.dump(out, open(f"{OUT}/pairs_examples_verified.json", "w"), indent=1)
    for r in out:
        print(json.dumps(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--max-group", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index", default=f"{OUT}/world_index.parquet")
    ap.add_argument("--verify-examples", action="store_true", help="only re-check example pairs against the raw tars")
    a = ap.parse_args()
    if a.verify_examples:
        return verify_examples()
    scs = [s for s in a.scenarios.split(",") if os.path.exists(f"{OUT}/traj/{s}{TAG}.npz")]
    print(f"[pairs] scenarios with trajectory cache: {len(scs)} {scs}", flush=True)
    if not scs:
        raise SystemExit("0 scenarios to analyse (bug, not a result)")
    out_ev, out_inv, rows, ex_all, grp = {}, {}, [], [], {}
    with ProcessPoolExecutor(min(a.workers, len(scs))) as ex:
        for sc, ev, inv, pr, exs, g in ex.map(analyse_scenario, [(s, a.max_group, a.seed, a.index) for s in scs]):
            out_ev[sc], out_inv[sc], grp[sc] = ev, inv, g
            rows.extend(pr); ex_all.extend(exs)
    json.dump(dict(events=out_ev, groups=grp), open(f"{OUT}/events_summary{TAG}.json", "w"), indent=1)
    json.dump(out_inv, open(f"{OUT}/invariance_summary{TAG}.json", "w"), indent=1)
    pd.DataFrame(rows).to_parquet(f"{OUT}/pairs_counts{TAG}.parquet")
    json.dump(ex_all, open(f"{OUT}/pairs_examples{TAG}.json", "w"), indent=1)
    if not TAG:
        json.dump(make_splits(), open(f"{OUT}/splits_summary.json", "w"), indent=1)
    print(f"[pairs] wrote {len(rows)} count rows, {len(ex_all)} examples")


if __name__ == "__main__":
    main()
