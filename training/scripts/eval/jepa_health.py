"""Does V-JEPA still work as V-JEPA after language alignment? Zero-shot diagnostics of the encoder + predictor alone.

  surprise  intuitive-physics violation detection without any training (Garrido et al. 2025): MVP's IntPhys / GRASP /
            InfLevel minimal pairs (same scene, possible vs impossible). Contiguous 4 fps windows of 16 frames, stride
            2 s; the predictor sees the first 12 frames (tokens removed before the encoder) and predicts the last 2
            tubelets; surprise = L1 to the per-level LayerNorm of the same model's encoder on the full window. Video
            score = max over windows (also the mean of the top 10% tokens). Pairwise accuracy = impossible > possible.
  native    masked-prediction self-consistency on held-out clips with the official cooldown masks: L1 between the
            model's predictor output and its own full-clip encoder states at the masked tokens (no reference involved,
            so it does not favour the arm trained against the reference).
The encoder / predictor come from the official checkpoint, or from a training checkpoint (`vision.*` keys) when --ckpt
is given. Diagnostics use V-JEPA only - the LLM is never run - so they measure what language alignment did to V-JEPA.
usage: jepa_health.py --task surprise|native --out <jsonl> [--ckpt runs/<run>/ckpt/latest] [--pairs 200] [--shard i --nshard n]
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/llama_vjepa_lm/training")
from jepa_llm.data.video import load_clip  # noqa: E402
from jepa_llm.models.vision_backbone import VJEPABackbone  # noqa: E402

R = "/data/llama_vjepa_lm"
G, FPS, WIN, CTX, STRIDE = 576, 4.0, 16, 12, 2.0


def load_vision(variant, ckpt, device):
    v = VJEPABackbone(variant).eval()
    if ckpt:
        from jepa_llm.eval.runner import load_weights
        load_weights(torch.nn.ModuleDict({"vision": v}), ckpt)
    return v.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)


def ln4(x):
    return F.layer_norm(x.float().view(*x.shape[:-1], 4, 1408), (1408,))


def physics_pairs(n_per_source, seed=0):
    items = [json.loads(l) for l in open(f"{R}/manifests/eval/mvp_full.jsonl")]
    by = collections.defaultdict(list)
    for x in items:
        if x["extra"]["subset"] == "intuitive_physics" and x["extra"]["source"] in ("intphys", "grasp", "inflevel"):
            by[x["group_id"]].append(x)
    rng = np.random.default_rng(seed)
    out = []
    for src in ("intphys", "grasp", "inflevel"):
        groups = sorted(g for g, v in by.items() if v[0]["extra"]["source"] == src and len(v) == 2)
        for g in rng.permutation(groups)[:n_per_source]:
            for x in by[g]:
                possible = x["options"][x["answer_index"]].lower().startswith("yes")
                out.append({"group": g, "source": src, "video": x["video"], "possible": possible})
    return out


@torch.no_grad()
def surprise(m, path, device, dtype):
    import decord
    vr = decord.VideoReader(path, num_threads=2)
    dur = len(vr) / vr.get_avg_fps()
    vis = VJEPABackbone.tokens_before_frame(CTX, WIN).to(device)
    tgt = torch.arange(len(vis), WIN // 2 * G, device=device)
    pred_s, top_s, copy_s = [], [], []
    for t0 in np.arange(0.0, max(dur - WIN / FPS, 0.0) + 1e-6, STRIDE):
        v, _, _ = load_clip(path, WIN, times=t0 + np.arange(WIN) / FPS)
        v = v.unsqueeze(0).to(device, dtype)
        full = ln4(m.encoder(v, training=True))[0]
        p, _ = m.predict(m.encode_masked_context(v, vis.unsqueeze(0)), vis.unsqueeze(0), tgt.unsqueeze(0))
        err = (F.layer_norm(p[0].float(), (1408,)) - full[tgt]).abs().mean((1, 2))
        pred_s.append(float(err.mean())); top_s.append(float(err.topk(max(1, len(err) // 10)).values.mean()))
        last = full[(CTX // 2 - 1) * G:(CTX // 2) * G].repeat(2, 1, 1)
        copy_s.append(float((last - full[tgt]).abs().mean()))
    return {"pred": pred_s, "pred_top10": top_s, "copy": copy_s}


@torch.no_grad()
def native_consistency(m, paths, device, dtype, n_frames=16, seed=0):
    from jepa_llm.losses.native import make_mask_generators
    gens = make_mask_generators(n_frames)
    torch.manual_seed(seed)
    rows = []
    for path in paths:
        import decord
        vr = decord.VideoReader(path, num_threads=2)
        dur = len(vr) / vr.get_avg_fps()
        t0 = max(0.0, dur / 2 - n_frames / FPS / 2)
        v, _, _ = load_clip(path, n_frames, times=t0 + np.arange(n_frames) / FPS)
        v = v.unsqueeze(0).to(device, dtype)
        target = ln4(m.encoder(v, training=True))[0]
        errs = []
        for gen in gens:
            enc, pred = gen(1)
            enc = (torch.stack(enc) if isinstance(enc, list) else enc).to(device)
            pred = (torch.stack(pred) if isinstance(pred, list) else pred).to(device)
            zp, _ = m.predictor(m.encode_masked_context(v, enc), enc, pred, mod="video")
            zp = ln4(zp)[0]
            errs.append(float((zp - target[pred[0]]).abs().mean()))
        rows.append({"video": path, "native_l1": float(np.mean(errs))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["surprise", "native"])
    ap.add_argument("--out", required=True); ap.add_argument("--ckpt"); ap.add_argument("--variant", default="vitg_1b")
    ap.add_argument("--pairs", type=int, default=200); ap.add_argument("--clips", type=int, default=300)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("THREADS", 16)))
    m = load_vision(a.variant, a.ckpt, device)
    dtype = next(m.parameters()).dtype
    fo = open(a.out, "a")
    if a.task == "surprise":
        done = {json.loads(l)["video"] for l in open(a.out)} if os.path.getsize(a.out) else set()
        for r in physics_pairs(a.pairs)[a.shard::a.nshard]:
            if r["video"] in done:
                continue
            fo.write(json.dumps({**r, **surprise(m, r["video"], device, dtype), "ckpt": a.ckpt}) + "\n"); fo.flush()
    else:
        # held-out clips never used for training: NVIDIA heldout worlds + SA-V heldout videos
        import pandas as pd
        paths = []
        for mf, root in ((f"{R}/manifests/stage1/nvidia_state_heldout.parquet", f"{R}/prep/nvidia_rgb_384"),
                         (f"{R}/manifests/stage2/sav_tracks_heldout.parquet", f"{R}/prep/sav_384_24fps")):
            if os.path.exists(mf):
                vids = pd.read_parquet(mf, columns=["video"]).video.drop_duplicates().sample(frac=1, random_state=0)
                paths += [f"{root}/{v}" for v in vids[: a.clips // 2]]
        for r in native_consistency(m, paths[a.shard::a.nshard], device, dtype):
            fo.write(json.dumps({**r, "ckpt": a.ckpt}) + "\n"); fo.flush()


if __name__ == "__main__":
    main()
