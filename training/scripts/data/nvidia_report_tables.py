"""Markdown tables for manifests/nvidia/report.md from the outputs of nvidia_index / nvidia_geometry_check / nvidia_pairs.

usage: python -I nvidia_report_tables.py > /data/llama_vjepa_lm/manifests/nvidia/report_tables.md
Read-only; prints every table with the number of items it was computed from.
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvidia_common import OUT


def md(df, floatfmt="{:.3g}"):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda x: "" if pd.isna(x) else floatfmt.format(x))
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def ok_ident(o):
    return o["identity_resid_m"] is not None and o["identity_resid_m"] < 0.02 and o["identity_margin_m"] > 0.01


def index_tables():
    cols = ["scenario", "world", "camera", "frame_count", "T_npz", "fps", "n_rows_npz", "n_rigid_bodies", "n_objects", "n_usda_prims",
            "n_resolved", "n_resolved_dyn", "n_npz_label_eq_static", "n_npz_label_eq_raw", "n_static_eq_raw", "cam_static",
            "row_order_same_as_ref_cam", "xcam_first_diverge_frame", "last_step_m", "max_step_before_last_m", "prep_video",
            "video_member", "seg_n_png", "fxn", "fyn", "distortion_zero", "gravity"]
    d = pd.read_parquet(f"{OUT}/world_index.parquet", columns=cols)
    O = pd.read_parquet(f"{OUT}/objects.parquet", columns=["scenario", "identity_resid_m", "identity_margin_m"])
    O = pd.DataFrame(dict(scenario=O.scenario, resid=O.identity_resid_m.astype(float),
                          ok=(O.identity_resid_m < 0.02) & (O.identity_margin_m > 0.01)))
    print(f"## index: {len(d)} camera clips, {d.world.nunique()} worlds, {len(O)} (clip, dynamic object) entries\n")
    g = d.groupby("scenario")
    t = pd.DataFrame(dict(
        worlds=g.world.nunique(), clips=g.size(), cameras=g.camera.agg(lambda s: ",".join(sorted(s.unique()))),
        T=g.T_npz.agg(lambda s: "/".join(map(str, sorted(s.unique())))), fps=g.fps.agg(lambda s: "/".join(map(str, sorted(s.unique())))),
        dyn_objects_median=g.n_objects.median(), rigid_bodies_median=g.n_rigid_bodies.median(),
        npz_rows_median=g.n_rows_npz.median(), rgb_member=g.video_member.agg(lambda s: s.notna().mean()),
        seg_png_eq_T=g.apply(lambda x: (x.seg_n_png == x.T_npz).mean()), prep384=g.prep_video.agg(lambda s: s.notna().mean()),
    )).reset_index()
    print("### T1 per scenario\n"); print(md(t)); print()
    rows = []
    og = O.groupby("scenario")
    for sc, x in g:
        o = og.get_group(sc)
        rows.append(dict(
            scenario=sc, dyn_object_entries=len(o), resolved_frac=o.ok.mean(), resid_p50_m=np.nanmedian(o.resid),
            resid_p99_m=np.nanpercentile(o.resid, 99),
            clips_npz_eq_static_all=((x.n_npz_label_eq_static == x.n_resolved_dyn) & (x.n_resolved_dyn > 0)).mean(),
            clips_static_eq_raw_all=((x.n_static_eq_raw == x.n_resolved_dyn) & (x.n_resolved_dyn > 0)).mean(),
            clips_static_eq_raw_none=((x.n_static_eq_raw == 0) & (x.n_resolved_dyn > 0)).mean(),
            row_order_differs_from_cam0=(x.row_order_same_as_ref_cam == False).mean(),  # noqa: E712
            traj_diverges_from_cam0=(x.xcam_first_diverge_frame >= 0).mean(),
            diverge_only_last_frame=(x.xcam_first_diverge_frame == x.T_npz - 1).mean(),
            cam_static=x.cam_static.mean(), fyn_over_fxn=(x.fyn / x.fxn).median(),
            step_gt_0p5m=(x.max_step_before_last_m > 0.5).mean()))
    print("### T2 identity and consistency (fractions of clips unless noted)\n"); print(md(pd.DataFrame(rows))); print()
    grav = d.groupby("scenario").gravity.agg(lambda s: dict(sorted(pd.Series(
        [str([float(round(x, 2)) for x in v]) for v in s if v is not None]).value_counts().items())))
    print("### static.json gravity vectors per scenario\n"); print(md(grav.reset_index())); print()
    return d


def geometry_tables():
    p = f"{OUT}/geometry_objects.parquet"
    if not os.path.exists(p):
        print("## geometry: not run\n"); return
    R = pd.read_parquet(p); C = pd.read_parquet(f"{OUT}/geometry_clips.parquet")
    print(f"## geometry: {len(C)} clips, {len(R)} body-frame records, {R[R.dynamic].groupby(['world','body']).ngroups} dynamic bodies\n")
    D = R[R.dynamic].copy()
    conf = D.pix_col_votes >= 0.6 * D.n_votes.clip(lower=1)
    D["conf"] = conf & (D.n_votes >= 3)
    # a colour label is "consistent" at a body-frame if the centroid of the mask with that label lies within
    # max(25 px, sqrt(mask area)) of the projected COM (COM inside the image)
    Din = D[D.proj_inside].copy()
    for src in ["pix", "static", "npz", "raw"]:
        Din[f"ok_{src}"] = Din[f"err_{src}"] < np.maximum(25, np.sqrt(Din[f"area_{src}"].astype(float)))
    Din["ok_nearest"] = Din.err_nearest < 25
    t = Din.groupby("scenario").agg(body_frames=("body", "size"), static_json=("ok_static", "mean"), npz_colors=("ok_npz", "mean"),
                                    raw_json=("ok_raw", "mean"), label_free_pixel_vote=("ok_pix", "mean"),
                                    nearest_mask_lt25px=("ok_nearest", "mean")).reset_index()
    print("### G1 fraction of dynamic body-frames whose label mask centroid is consistent with the projected COM\n")
    print(md(t)); print()
    C2 = Din.groupby(["scenario", "world", "camera"])[["ok_static", "ok_npz", "ok_raw", "ok_pix"]].mean().reset_index()
    C2["best"] = np.where(C2.ok_static >= C2.ok_raw, "static/npz", "raw")
    C2["best_ok"] = C2[["ok_static", "ok_raw"]].max(axis=1)
    C2["clear"] = (C2.best_ok >= 0.75) & ((C2.ok_static - C2.ok_raw).abs() >= 0.3)
    t2 = C2.groupby("scenario").agg(clips=("world", "size"), static_ge_0p75=("ok_static", lambda s: (s >= .75).mean()),
                                    raw_ge_0p75=("ok_raw", lambda s: (s >= .75).mean()),
                                    both_ge_0p75=("best_ok", lambda s: 0), clear_winner=("clear", "mean"),
                                    winner_raw=("best", lambda s: (s == "raw").mean())).reset_index()
    t2["both_ge_0p75"] = C2.assign(b=(C2.ok_static >= .75) & (C2.ok_raw >= .75)).groupby("scenario").b.mean().values
    print("### G2 per clip: which label source is consistent (fraction of clips)\n"); print(md(t2)); print()
    V = D[D.conf & D.err_pix.notna() & (D.area_pix >= 30)]
    V = V.assign(own=V.under_is_pixcol.fillna(False))
    t3 = V.groupby(["scenario", "own"]).err_pix.describe(percentiles=[.5, .9])[["count", "50%", "90%"]].reset_index()
    print("### G3 |mask centroid - projected COM| (px, 1920x1080) by COM pixel on own mask (own=True) or occluded/off-mask\n")
    print(md(t3)); print()
    V2 = V[V.own]
    t4 = pd.DataFrame(dict(model=["square px (fx)", "json (fx, fy)"],
                           p50=[V2.err_pix.median(), V2.err_pix_jsonfy.median()], p90=[V2.err_pix.quantile(.9), V2.err_pix_jsonfy.quantile(.9)],
                           n=[len(V2), len(V2)]))
    print("### G4 intrinsics model comparison (unoccluded COM)\n"); print(md(t4)); print()
    V3 = V2.assign(area_bin=pd.cut(V2.area_pix, [0, 300, 3000, 30000, 1e9], labels=["<300", "300-3k", "3k-30k", ">30k"]))
    print("### G5 error vs mask area (unoccluded COM)\n")
    print(md(V3.groupby("area_bin", observed=True).err_pix.describe(percentiles=[.5, .9])[["count", "50%", "90%"]].reset_index())); print()
    vis = C.assign(vis_rate=C.vis_agree / C.vis_total).groupby("scenario").agg(
        clips=("world", "size"), vis_agree=("vis_rate", "mean"), vis_order_eq_raw=("vis_order_eq_raw", "mean"),
        vis_order_eq_npz=("vis_order_eq_npz", "mean")).reset_index()
    print("### G6 visibility.npz vs mask presence (same colour)\n"); print(md(vis)); print()
    if "rgb_best_lag" in C:
        r = C[C.rgb_best_lag.notna()]
        lagc = [c for c in C.columns if c.startswith("rgb_lag_corr_")]
        print(f"### G7 RGB vs mask temporal alignment: {len(r)} clips; best lag counts {r.rgb_best_lag.value_counts().to_dict()}; "
              f"mean corr per lag { {c[13:]: round(float(r[c].mean()), 3) for c in lagc} }\n")
    worst = V2.sort_values("err_pix", ascending=False).head(8)[["scenario", "world", "camera", "frame", "prim", "area_pix", "err_pix", "depth"]]
    print("### G8 largest errors with COM on own mask\n"); print(md(worst)); print()


def events_tables():
    p = f"{OUT}/events_summary.json"
    E = json.load(open(p)); I = json.load(open(f"{OUT}/invariance_summary.json"))
    rows = []
    for sc, v in sorted(E["events"].items()):
        e = v["events"]; n = max(e.get("dyn_bodies", 0), 1); nc = max(e.get("cam_dyn_bodies", 0), 1)
        rows.append(dict(scenario=sc, worlds=e.get("worlds", 0), dyn_bodies=e.get("dyn_bodies", 0),
                         init_moving=e.get("init_moving", 0) / n, rest_then_moved=e.get("rest_then_moved", 0) / n,
                         never_moved=e.get("never_moved", 0) / n, stopped_before_end=e.get("stopped_before_end", 0) / n,
                         drop=e.get("drop", 0) / n, onset_p10_p50_p90=v["hist"].get("onset_frames"),
                         first_mover_pairs=e.get("first_mover_pairs", 0),
                         first_mover_decidable=e.get("first_mover_decidable", 0) / max(e.get("first_mover_pairs", 0), 1),
                         ties_le1fr=e.get("first_mover_ties_0_1fr", 0) / max(e.get("first_mover_pairs", 0), 1),
                         leave_frame_per_cam=e.get("cam_dyn_leave_frame", 0) / nc, inside_f0=e.get("cam_dyn_inside_f0", 0) / nc,
                         lr_crossing_pairs=e.get("cam_pair_lr_crossings", 0) / max(e.get("cam_pairs", 0), 1),
                         unique_descriptor=e.get("dyn_bodies_unique_descriptor", 0) / n))
    print("## events (fractions of dynamic bodies unless noted)\n"); print(md(pd.DataFrame(rows))); print()
    print("### groups for pair search\n"); print(md(pd.DataFrame(E["groups"]).T.reset_index())); print()
    print("## invariance instance counts (mid + end frame, dynamic pairs)\n"); print(md(pd.DataFrame(I).T.reset_index())); print()


def pair_tables():
    P = pd.read_parquet(f"{OUT}/pairs_counts.parquet")
    nn = P[P.kind == "nn_dist"].groupby(["scenario", "space", "k"])[["group_size", "nn_p10", "nn_p50", "nn_p90"]].median().reset_index()
    print("## pair search: nearest-neighbour past distance (median over slot groups)\n"); print(md(nn)); print()
    S = P[P.kind == "same_past_diff_future"].copy()
    S["space_kind"] = np.where(S.space == "world", "world", "image")
    agg = S.groupby(["scenario", "space_kind", "k", "h", "eps"])[["n_close_pairs", "eval_n", "n_lr_flip", "n_z_flip", "n_move_flip",
                                                                   "n_any_flip", "n_close_static_prefix"]].sum().reset_index()
    for c in ["n_lr_flip", "n_z_flip", "n_move_flip", "n_any_flip"]:
        agg[c.replace("n_", "rate_")] = agg[c] / agg.eval_n.clip(lower=1)
    agg["est_any_flip_pairs"] = (agg.rate_any_flip * agg.n_close_pairs).round()
    print("## same past / different future: summed over slot groups (image rows summed over the 4 camera names); flip counts are "
          "over eval_n evaluated pairs (<= 100k per group/setting), est_any_flip_pairs = rate x n_close_pairs\n")
    print(md(agg[agg.n_close_pairs > 0].drop(columns=["n_lr_flip", "n_z_flip", "n_move_flip", "n_any_flip"]))); print()
    F = P[P.kind == "same_future_diff_past"].copy()
    F["space_kind"] = np.where(F.space == "world", "world", "image")
    print("## same future / different past (last 30 frames close; first-mover order flips)\n")
    print(md(F.groupby(["scenario", "space_kind", "eps"])[["n_close_pairs", "n_onset_order_flip"]].sum().reset_index())); print()


def split_tables():
    S = json.load(open(f"{OUT}/splits_summary.json"))
    print("## splits\n"); print(md(pd.DataFrame(S["totals"]))); print()
    print(md(pd.DataFrame(S["by_world"]).pivot(index="scenario", columns="split", values="worlds").reset_index())); print()
    print(md(pd.DataFrame(S["loso"]).T.reset_index())); print()


if __name__ == "__main__":
    index_tables(); geometry_tables(); events_tables(); pair_tables(); split_tables()
