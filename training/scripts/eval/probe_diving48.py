"""Diving48 frozen-feature attentive probe: does the V-JEPA 2.1 g1B encoder keep its motion understanding after it was
fine-tuned inside the video-LLM? Same tool for the official encoder and for training checkpoints (`vision.*` keys).

Subcommands
  unpack   HF mteb/diving48v2 parquet -> one mp4 per clip (bench/diving48/videos/<split>/<vid_name>.mp4, temp file +
           rename, resumable) and manifests bench/diving48/<split>.jsonl {path, label, vid_name, start_frame, end_frame,
           n_frames, fps, height, width}. The parquet bytes are untrusted: vid_name must match [A-Za-z0-9_-]+, the bytes
           are only written to disk and opened with decord (header check: frame count, fps, size).
           Facts (checked at unpack, 15,027 + 1,970 clips, 0 decode errors): start_frame is always 0 and end_frame ==
           n_frames - 1 (inclusive last frame), i.e. every file IS the clip; all 640x480; fps 25/30/60; duration median
           4.1 s (1.0-7.8 s); label 30 never occurs in either split (47 classes present; head stays 48-way). `extract`
           still samples inside [start_frame, end_frame] so a manifest that ever disagrees is cropped, not read whole.
  extract  encoder features. Per clip: 16 frames at evenly spaced frame indices over the clip
           (idx_i = start + (i + 0.5) * L / 16, L = end - start + 1, rounded; i.e. the centres of 16 equal bins). This is
           the fallback branch of jepa_llm.data.video.paired_uniform_times; the paired 0.25 s scheme is not used because
           Diving48 clips are short (median 4.1 s) and the dive needs uniform temporal coverage - here the two frames
           of a tubelet are duration/16 apart (median ~0.26 s), close to the 0.25 s pretraining step anyway. Sampling
           is by frame index, so it is independent of the source fps (25/30/60). Resize short side to 384
           (bicubic), centre crop 384x384, ImageNet normalisation (load_clip). Single view (no multi-clip / 3-crop TTA:
           upstream uses 4 segments x 3 views x 32 frames; this is a cheaper, fixed protocol for *relative* comparisons).
           Forward: encoder(video, training=True) under no_grad in fp32 (TF32 matmuls on GPU, never bf16: the same
           forward precision for every evaluated encoder) -> the 4 hierarchical levels (blocks 9/19/29/39, each through
           its own trained LayerNorm, i.e. exactly the upstream multilevel probe features) [8 tubelets, 24, 24, 4, 1408]
           -> 4x4 average pool in space -> [288 = 8 x 6 x 6, 4, 1408], stored float16 (3.24 MB / clip; 55 GB for all
           16,997 clips). Output: <out>/<split>/c<chunk>.pt, 32 clips per chunk; chunk c belongs to shard c % nshard, so
           the result does not depend on nshard and an interrupted run resumes by skipping finished chunks. Every chunk
           records the encoder fingerprint; `train` refuses a directory that mixes encoders.
  train    attentive probe on the stored features (V-JEPA AttentivePooler, depth 1): one learnable query,
           a CrossAttentionBlock (pre-LN cross-attention with 16 heads + residual + MLP ratio 4, as upstream
           complete_block=True) -> LayerNorm -> linear 48-way head. Token layout (--layout):
             tokens (default, upstream multilevel protocol): each level is its own token, 288 x 4 = 1152 tokens of 1408
             concat: 288 tokens of 5632 (4 levels concatenated); ~380M probe params, slower, and at lr 1e-3 its first
                     steps blow up (CPU test: epoch-1 loss 33 vs 4-8 for tokens) - kept only as an option
           AdamW (lr 1e-3, wd 0.01, betas 0.9/0.999), cosine to 0 per step, no warmup, 20 epochs, batch 256, bf16
           autocast on GPU (fp32 on CPU), 3 seeds (init + data order). Test top-1 is read once, after the last epoch
           (no checkpoint selection on test). Reports mean +- sd (sample sd, ddof=1) over seeds.
usage
  probe_diving48.py unpack [--split train test]
  probe_diving48.py extract --split train|test --out <featdir> [--ckpt runs/<run>/ckpt/latest] [--shard i --nshard n]
                            [--device cuda|cpu] [--manifest <jsonl>] [--threads 16]
  probe_diving48.py train --feat <featdir> --out <json> [--layout tokens|concat] [--seeds 0 1 2] [--device cuda|cpu]
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, "/data/llama_vjepa_lm/training")

BENCH = "/data/llama_vjepa_lm/bench/diving48"
NUM_FRAMES, POOL, CHUNK, NUM_CLASSES = 16, 4, 32, 48
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


# ---------------------------------------------------------------------------------------------------------------- unpack
def _probe_video(path):
    import decord
    try:
        vr = decord.VideoReader(path, num_threads=1)
        h, w = vr[0].shape[:2]
        return {"n_frames": len(vr), "fps": float(vr.get_avg_fps()), "height": int(h), "width": int(w)}
    except Exception as e:  # noqa: BLE001 - report, never crash on one bad file
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def cmd_unpack(a):
    import pyarrow.parquet as pq
    from multiprocessing import Pool
    for split in a.split:
        vdir = f"{BENCH}/videos/{split}"
        os.makedirs(vdir, exist_ok=True)
        rows, written = [], 0
        for f in sorted(glob.glob(f"{BENCH}/data/{split}-*.parquet")):
            pf = pq.ParquetFile(f)
            for g in range(pf.metadata.num_row_groups):
                t = pf.read_row_group(g, columns=["vid_name", "video", "start_frame", "end_frame", "label"]).to_pylist()
                for r in t:
                    name = r["vid_name"]
                    if not isinstance(name, str) or not NAME_RE.match(name):
                        raise ValueError(f"unsafe vid_name {name!r} in {f}")
                    lab = int(r["label"])
                    assert 0 <= lab < NUM_CLASSES, lab
                    dst = f"{vdir}/{name}.mp4"
                    data = r["video"]["bytes"]
                    if not (os.path.exists(dst) and os.path.getsize(dst) == len(data)):
                        tmp = f"{dst}.tmp{os.getpid()}"
                        with open(tmp, "wb") as fh:
                            fh.write(data)
                        os.replace(tmp, dst)
                        written += 1
                    rows.append({"path": dst, "label": lab, "vid_name": name,
                                 "start_frame": int(r["start_frame"]), "end_frame": int(r["end_frame"])})
        names = [r["vid_name"] for r in rows]
        assert len(set(names)) == len(names), "duplicate vid_name"
        with Pool(a.workers) as p:
            info = p.map(_probe_video, [r["path"] for r in rows], chunksize=16)
        bad = 0
        for r, i in zip(rows, info):
            r.update(i)
            bad += "error" in i
        tmp = f"{BENCH}/{split}.jsonl.tmp"
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        os.replace(tmp, f"{BENCH}/{split}.jsonl")
        ok = [r for r in rows if "error" not in r]
        d = np.array([r["n_frames"] - 1 - r["end_frame"] for r in ok])
        print(json.dumps({"split": split, "rows": len(rows), "written": written, "decode_errors": bad,
                          "start_frame_nonzero": sum(r["start_frame"] != 0 for r in rows),
                          "n_frames-1-end_frame": {str(k): int(v) for k, v in zip(*np.unique(d, return_counts=True))},
                          "labels": len(set(r["label"] for r in rows)),
                          "fps": {str(k): int(v) for k, v in zip(*np.unique([round(r["fps"], 2) for r in ok],
                                                                          return_counts=True))},
                          "size": {str(k): int(v) for k, v in zip(*np.unique([f"{r['width']}x{r['height']}" for r in ok],
                                                                           return_counts=True))},
                          "n_frames_median": float(np.median([r["n_frames"] for r in ok]))}))


# --------------------------------------------------------------------------------------------------------------- extract
def frame_indices(start, end, n_frames, num=NUM_FRAMES):
    """Centres of `num` equal bins over the inclusive frame range [start, end] (clamped to the file)."""
    s, e = max(0, int(start)), min(int(end), n_frames - 1)
    if e < s:
        s, e = 0, n_frames - 1
    L = e - s + 1
    return np.clip(np.floor(s + (np.arange(num) + 0.5) * L / num).astype(int), s, e)


def load_encoder(ckpt, device):
    from jepa_llm.models.vision_backbone import VJEPABackbone
    v = VJEPABackbone("vitg_1b").eval()
    if ckpt:
        # same loader as scripts/eval/jepa_health.py: `vision.*` keys of a training checkpoint (DCP dir or pointer file)
        from jepa_llm.eval.runner import load_weights
        load_weights(nn.ModuleDict({"vision": v}), ckpt)
    enc = v.encoder
    del v.predictor
    # md5 of the raw fp32 weight bytes: exact and independent of thread count (a float sum is not)
    h = hashlib.md5()
    with torch.no_grad():
        for k, p in sorted(enc.state_dict().items()):
            h.update(k.encode())
            h.update(p.detach().float().contiguous().view(torch.uint8).numpy().tobytes())
    enc = enc.to(device=device, dtype=torch.float32)   # fp32 forward on every device (see module docstring)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, h.hexdigest()


@torch.no_grad()
def encode(enc, video):
    """video [B,3,16,384,384] -> [B, 288, 4, 1408] (4x4 spatial average pool of the 4 normed hierarchical levels)."""
    x = enc(video, training=True)                                   # [B, 8*24*24, 5632]
    b, n, d = x.shape
    t = video.shape[2] // 2
    x = x.view(b, t, 24, 24, d).permute(0, 1, 4, 2, 3).reshape(b * t, d, 24, 24)
    x = F.avg_pool2d(x, POOL)                                       # [b*t, d, 6, 6]
    g = 24 // POOL
    x = x.view(b, t, d, g * g).permute(0, 1, 3, 2).reshape(b, t * g * g, 4, d // 4)
    return x


def cmd_extract(a):
    from jepa_llm.data.video import load_clip
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    man = a.manifest or f"{BENCH}/{a.split}.jsonl"
    items = [json.loads(l) for l in open(man)]
    odir = f"{a.out}/{a.split}"
    os.makedirs(odir, exist_ok=True)
    chunks = [c for c in range(math.ceil(len(items) / CHUNK)) if c % a.nshard == a.shard]
    todo = [c for c in chunks if not os.path.exists(f"{odir}/c{c:05d}.pt")]
    print(f"[extract] {a.split}: {len(items)} clips, shard {a.shard}/{a.nshard}: {len(chunks)} chunks, "
          f"{len(todo)} to do", flush=True)
    if not todo:
        return
    enc, fp = load_encoder(a.ckpt, device)
    meta = {"ckpt": a.ckpt or "official:vjepa2_1_vitg_384.pt/target_encoder", "fingerprint": fp,
            "num_frames": NUM_FRAMES, "pool": POOL, "precision": "fp32" + ("+tf32" if
            device.type == "cuda" else ""), "sampling": "evenly spaced bin centres over [start_frame, end_frame]"}
    mpath = f"{a.out}/meta.json"
    if os.path.exists(mpath):
        old = json.load(open(mpath))
        assert old["fingerprint"] == fp, f"{a.out} holds features of another encoder ({old['ckpt']})"
    elif a.shard == 0:
        with open(f"{mpath}.tmp", "w") as fh:
            json.dump(meta, fh, indent=1)
        os.replace(f"{mpath}.tmp", mpath)
    print(f"[extract] encoder fingerprint {fp} ({meta['ckpt']})", flush=True)
    from concurrent.futures import ThreadPoolExecutor

    def load(it):
        idx = frame_indices(it["start_frame"], it["end_frame"], it["n_frames"])
        return load_clip(it["path"], NUM_FRAMES, times=idx / it["fps"])[0]

    # decode runs in a thread pool one chunk ahead of the encoder (decord / torch resize release the GIL)
    pool = ThreadPoolExecutor(a.workers)
    submit = lambda c: [pool.submit(load, it) for it in items[c * CHUNK:(c + 1) * CHUNK]]  # noqa: E731
    pending = submit(todo[0])
    t_start, n_done = time.time(), 0
    for k, c in enumerate(todo):
        sel = items[c * CHUNK:(c + 1) * CHUNK]
        futs, pending = pending, (submit(todo[k + 1]) if k + 1 < len(todo) else None)
        feats, t_dec, t_enc = [], 0.0, 0.0
        for i in range(0, len(sel), a.batch):
            t0 = time.time()
            vids = [f.result() for f in futs[i:i + a.batch]]
            t1 = time.time()
            x = torch.stack(vids).to(device)
            feats.append(encode(enc, x).half().cpu())
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_dec, t_enc = t_dec + t1 - t0, t_enc + time.time() - t1
        out = {"feats": torch.cat(feats), "labels": torch.tensor([it["label"] for it in sel]),
               "vid_names": [it["vid_name"] for it in sel], "fingerprint": fp, "ckpt": meta["ckpt"],
               "manifest": man}
        tmp = f"{odir}/c{c:05d}.pt.tmp{os.getpid()}"
        torch.save(out, tmp)
        os.replace(tmp, f"{odir}/c{c:05d}.pt")
        n_done += len(sel)
        print(f"[extract] chunk {c}: {len(sel)} clips, decode wait {t_dec / len(sel):.2f} s/clip, "
              f"encode {t_enc / len(sel):.2f} s/clip, total {(time.time() - t_start) / n_done:.2f} s/clip", flush=True)


# ----------------------------------------------------------------------------------------------------------------- train
class CrossAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.q, self.kv, self.proj = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)

    def forward(self, q, x):
        b, n, d = x.shape
        q = self.q(q).view(b, -1, self.h, d // self.h).transpose(1, 2)
        k, v = self.kv(x).view(b, n, 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(b, -1, d))


class AttentiveProbe(nn.Module):
    """V-JEPA AttentiveClassifier with depth 1 (query -> CrossAttentionBlock) + final LayerNorm + linear head."""

    def __init__(self, dim, heads=16, mlp_ratio=4.0, num_classes=NUM_CLASSES):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm1, self.attn = nn.LayerNorm(dim), CrossAttention(dim, heads)
        hid = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))
        self.norm_out, self.head = nn.LayerNorm(dim), nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.query, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        self.mlp[2].weight.data.div_(math.sqrt(2.0))   # upstream _rescale_blocks for the single block

    def forward(self, x):
        q = self.query.expand(x.shape[0], -1, -1)
        q = q + self.attn(q, self.norm1(x))
        q = q + self.mlp(self.norm2(q))
        return self.head(self.norm_out(q[:, 0]))


def load_split(fdir, split):
    files = sorted(glob.glob(f"{fdir}/{split}/c*.pt"))
    assert files, f"no features in {fdir}/{split}"
    x, labels, names, fps = None, [], [], set()
    for j, f in enumerate(files):   # fill a preallocated array: peak RAM = one split, not twice it
        p = torch.load(f, map_location="cpu", weights_only=False)
        if x is None:
            x = torch.empty((len(files) * CHUNK, *p["feats"].shape[1:]), dtype=torch.float16)
        x[len(labels):len(labels) + len(p["labels"])] = p["feats"]
        labels += p["labels"].tolist()
        names += p["vid_names"]
        fps.add(p["fingerprint"])
    assert len(fps) == 1, f"{fdir}/{split} mixes encoders: {fps}"
    assert len(set(names)) == len(names), "duplicate clips"
    return x[:len(labels)], torch.tensor(labels), fps.pop(), names


def to_tokens(x, layout):
    """[B, 288, 4, 1408] -> tokens."""
    if layout == "concat":
        return x.flatten(2)                      # [B, 288, 5632]
    return x.flatten(1, 2)                       # [B, 1152, 1408]


def cmd_train(a):
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    xtr, ytr, fp_tr, n_tr = load_split(a.feat, "train")
    xte, yte, fp_te, n_te = load_split(a.feat, "test")
    assert fp_tr == fp_te, "train/test features come from different encoders"
    assert not set(n_tr) & set(n_te), "train/test overlap"
    man = {s: f"{BENCH}/{s}.jsonl" for s in ("train", "test")}
    full = {s: sum(1 for _ in open(p)) for s, p in man.items() if os.path.exists(p)}
    print(f"[train] train {len(ytr)} / test {len(yte)} clips (full splits {full}), fingerprint {fp_tr}", flush=True)
    if device.type == "cuda":
        free = torch.cuda.mem_get_info()[0]
        need = (xtr.numel() + xte.numel()) * 2
        if need + (16 << 30) < free:          # keep the features GPU-resident when they fit
            xtr, xte = xtr.to(device), xte.to(device)
    ytr, yte = ytr.to(device), yte.to(device)
    dim = 5632 if a.layout == "concat" else 1408
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu",
                                                                                                    enabled=False)
    results = []
    for seed in a.seeds:
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        model = AttentiveProbe(dim, a.heads, a.mlp_ratio).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.999))
        steps_per_epoch = math.ceil(len(ytr) / a.batch)
        total = steps_per_epoch * a.epochs
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
        hist, t0 = [], time.time()
        for ep in range(a.epochs):
            model.train()
            perm = torch.randperm(len(ytr), generator=g)
            tl, tc, tn = 0.0, 0, 0
            for i in range(0, len(perm), a.batch):
                idx = perm[i:i + a.batch]
                x = to_tokens(xtr[idx.to(xtr.device)].to(device, non_blocking=True).float(), a.layout)
                y = ytr[idx.to(ytr.device)]
                with amp:
                    logits = model(x)
                loss = F.cross_entropy(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tl += loss.item() * len(y)
                tc += (logits.argmax(1) == y).sum().item()
                tn += len(y)
            hist.append({"epoch": ep + 1, "train_loss": tl / tn, "train_acc": tc / tn})
            print(f"[train] seed {seed} ep {ep + 1}: loss {tl / tn:.4f} acc {tc / tn:.3f} ({time.time() - t0:.0f}s)",
                  flush=True)
        model.eval()
        correct = 0
        with torch.no_grad(), amp:
            for i in range(0, len(yte), a.batch):
                x = to_tokens(xte[i:i + a.batch].to(device).float(), a.layout)
                correct += (model(x).argmax(1) == yte[i:i + a.batch]).sum().item()
        acc = correct / len(yte)
        print(f"[train] seed {seed}: test top-1 {acc:.4f}", flush=True)
        results.append({"seed": seed, "test_top1": acc, "history": hist, "seconds": time.time() - t0})
    accs = np.array([r["test_top1"] for r in results])
    out = {"feat": os.path.abspath(a.feat), "fingerprint": fp_tr,
           "meta": json.load(open(f"{a.feat}/meta.json")) if os.path.exists(f"{a.feat}/meta.json") else None,
           "n_train": len(ytr), "n_test": len(yte), "n_classes_train": len(set(ytr.tolist())),
           "complete": full == {"train": len(ytr), "test": len(yte)},   # False = subset / unfinished extraction
           "chance_majority": float(torch.bincount(yte.cpu()).max() / len(yte)),
           "config": {k: v for k, v in vars(a).items() if k not in ("func",)},
           "test_top1_mean": float(accs.mean()), "test_top1_sd": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
           "per_seed": results}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[train] test top-1 {out['test_top1_mean']:.4f} +- {out['test_top1_sd']:.4f} over seeds {a.seeds} "
          f"(n_test {len(yte)}, majority-class {out['chance_majority']:.3f}) -> {a.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("unpack")
    u.add_argument("--split", nargs="+", default=["test", "train"])
    u.add_argument("--workers", type=int, default=16)
    u.set_defaults(func=cmd_unpack)
    e = sub.add_parser("extract")
    e.add_argument("--split", required=True, choices=["train", "test"])
    e.add_argument("--out", required=True)
    e.add_argument("--ckpt", default=None, help="training checkpoint (DCP dir or ckpt/latest pointer); default official")
    e.add_argument("--manifest", default=None, help="override bench/diving48/<split>.jsonl (e.g. a test subset)")
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--nshard", type=int, default=1)
    e.add_argument("--device", default="cuda")
    e.add_argument("--batch", type=int, default=4)
    e.add_argument("--threads", type=int, default=16)
    e.add_argument("--workers", type=int, default=6, help="decode threads")
    e.set_defaults(func=cmd_extract)
    t = sub.add_parser("train")
    t.add_argument("--feat", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--layout", default="tokens", choices=["tokens", "concat"])
    t.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--wd", type=float, default=0.01)
    t.add_argument("--heads", type=int, default=16)
    t.add_argument("--mlp_ratio", type=float, default=4.0)
    t.add_argument("--device", default="cuda")
    t.add_argument("--threads", type=int, default=16)
    t.set_defaults(func=cmd_train)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
