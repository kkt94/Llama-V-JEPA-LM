"""Model-side evaluation: multiple-choice by option-letter likelihood (spec 10.3: one scoring rule for every model).

    python -m jepa_llm.eval.runner --config <training cfg> --ckpt <ckpt dir or ckpt/latest> --items <eval.jsonl> \
        --out <preds.jsonl> [--stage 1|2] [--limit N]
    (torchrun --nproc_per_node K ... splits the items across GPUs; each rank holds a full bf16 replica)

Item format (from jepa_llm.eval.adapters): {"id", "video", "question", "options": [...], "answer_index",
"start_sec"/"end_sec" or "cut_sec" (optional visible span), "group_id", ...}.
Prompt: "<evidence>\\n{question}\\nA. ...\\nB. ...\\nAnswer with the option's letter from the given choices directly."
The prediction is argmax over the letter tokens of the first answer position.
Stage 1 model: whole visible clip as evidence. Stage 2/3 model: pass 1 + composer choose the EvidenceSpec (no gold),
the spec is executed and pass 2 answers - the real inference path (spec 4.3); the chosen spec is logged per item.
"""
import argparse
import json
from jepa_llm.models.evidence_bridge import normalize_levels, pool_regions
import numpy as np
import math
import os

import torch
import torch.distributed as dist
import yaml
from transformers import AutoTokenizer

from jepa_llm.data.chat_format import IGNORE, VIS, build_example, build_generation_prompt, build_pass1
from jepa_llm.data.video import load_clip, compose_clip
from jepa_llm.models.vision_backbone import VJEPABackbone, build_vision
from jepa_llm.training.train_stage1 import build_llm, compute_dtype

LETTERS = "ABCDEFGHIJ"
SUFFIX = "Answer with the option's letter from the given choices directly."


def prompt(item):
    if not item.get("options"):        # open-ended (Video-MME-Logical): the official prompt is used verbatim
        return item["question"]
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(item["options"]))
    return f"{item['question']}\n{opts}\n{SUFFIX}"


@torch.no_grad()
def parse_letter(resp, n):
    """(11-12, review Q3) generation scoring: the option letter a free-form answer names (first standalone A..; else a unique option
    text it contains); None if neither (counted wrong)."""
    import re
    L = "ABCDEFGHIJ"[:n]; r = (resp or "").strip()
    m = re.match(r"^[\(\[]?([A-J])(?:[\)\].:,\s]|$)", r)
    if m and m.group(1) in L: return L.index(m.group(1))
    m = re.search(r"(?:answer|option)\s*(?:is|:)?\s*[\(\[]?([A-J])\b", r, re.I)
    if m and m.group(1).upper() in L: return L.index(m.group(1).upper())
    m = re.search(r"\b([A-J])\b", r)
    if m and m.group(1) in L: return L.index(m.group(1))
    return None


def parse_option(resp, options):
    i = parse_letter(resp, len(options))
    if i is not None: return i
    low = (resp or "").lower(); hits = [k for k, o in enumerate(options) if str(o).lower().strip() and str(o).lower().strip() in low]
    return hits[0] if len(hits) == 1 else None


def generate(model, tok, ev, q, device, max_new_tokens=64):
    """Greedy decoding with the evidence spliced into the prompt (eval only; the model is a plain replica)."""
    p = build_generation_prompt(tok, f"{VIS}\n{q}", ev.shape[1])
    ids = torch.tensor([p["input_ids"]], device=device)
    vis = torch.tensor([p["vis_mask"]], device=device).bool()
    llm = model.core.llm
    emb = llm.get_input_embeddings()(ids)
    if model.core.hybrid:                                        # pooled tokens in the prompt + the full grid as memory
        emb = emb.masked_scatter(vis.unsqueeze(-1), ev.to(emb.dtype)); model.core._mem = (model.mem_tokens(), None)
    elif model.core.memory:                                      # the video as memory read by every layer (no video slots)
        model.core._mem = (ev, None)
    elif model.core.entry_layer:                                 # V-JEPA tokens enter at a middle layer (LanguageCore.enable_entry)
        emb = emb.masked_fill(vis.unsqueeze(-1), 0.0); model.core._entry = (vis, ev.view(-1, ev.shape[-1]))
    else:
        emb = emb.masked_scatter(vis.unsqueeze(-1), ev.to(emb.dtype))
    model.core.gate.set(vis)                                     # video positions (GatedLinear); the new tokens below are text
    try:
        out = llm(inputs_embeds=emb, use_cache=True)
    finally:
        model.core.gate.set(None); model.core._entry = None
        if not model.core.memory:
            model.core._mem = None
    stop = {tok.eos_token_id, *(tok.convert_tokens_to_ids(t) for t in ("<|eot_id|>", "<|im_end|>") if t in tok.get_vocab())}
    gen = []
    for _ in range(max_new_tokens):
        nxt = int(out.logits[0, -1].argmax())
        if nxt in stop:
            break
        gen.append(nxt)
        out = llm(input_ids=torch.tensor([[nxt]], device=device), past_key_values=out.past_key_values, use_cache=True)
    model.core._mem = None
    return tok.decode(gen, skip_special_tokens=True)


def letter_ids(tok, n):
    ids = [tok(LETTERS[i], add_special_tokens=False)["input_ids"] for i in range(n)]
    assert all(len(x) == 1 for x in ids), "option letters must be single tokens"
    return [x[0] for x in ids]


def load_weights(mods, ckpt):
    """mods: ModuleDict of trainable modules (same names as in training). Accepts single-process state.pt checkpoints
    or DCP directories (converted once to a torch file next to them)."""
    if os.path.isfile(ckpt):
        ckpt = open(ckpt).read().strip()
    if os.path.exists(f"{ckpt}/state.pt"):
        sd = torch.load(f"{ckpt}/state.pt", map_location="cpu", weights_only=False)["model"]
    else:
        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
        tmp = f"{ckpt}.full.pt"
        # (10-28) one converter per checkpoint: several jobs starting together each converted and os.replace'd the file while the others were
        # reading it (NFS "Stale file handle"). The first to create <tmp>.lock converts; the rest wait for <tmp> (a lock older than 40 min is
        # taken as a dead converter's and removed)
        import time as _time
        own = False
        while not os.path.exists(tmp) and (not dist.is_initialized() or dist.get_rank() == 0):
            try:
                os.close(os.open(f"{tmp}.lock", os.O_CREAT | os.O_EXCL)); own = True; break
            except FileExistsError:
                try:
                    if _time.time() - os.path.getmtime(f"{tmp}.lock") > 2400:
                        os.remove(f"{tmp}.lock"); continue
                except FileNotFoundError:
                    continue
                _time.sleep(15)
        if own and not os.path.exists(tmp):
            # (10-22) written under a private name and renamed: two evaluations of the same checkpoint (main + controls on the other node)
            # must never read a half-written file or truncate each other's
            import socket
            part = f"{tmp}.part.{socket.gethostname()}.{os.getpid()}"
            # (10-26) read the model entries only (the DCP directory also holds the optimizer: 50 GB vs ~16 GB for the 4B run; converting
            # all of it and dropping the optimizer wrote ~70 GB per checkpoint to NFS and took ~15 min when several ran at once)
            from torch.distributed.checkpoint import FileSystemReader
            from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
            from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
            full = {}; _load_state_dict(full, storage_reader=FileSystemReader(ckpt), planner=_EmptyStateDictLoadPlanner(keys={"model"}), no_dist=True)
            torch.save({"model": full["model"]}, part); del full
            os.replace(part, tmp)
        if own:
            os.remove(f"{tmp}.lock")
        if dist.is_initialized():
            # (11-01) explicit device: with uneven GPUs per node (8 / 7 / 8 / 7) the guessed device id hung the 30-GPU init_from start
            dist.barrier(device_ids=[torch.cuda.current_device()]) if torch.cuda.is_available() else dist.barrier()
        sd = torch.load(tmp, map_location="cpu", weights_only=False)["model"]
    want = mods.state_dict()
    from jepa_llm.training.train_stage2 import adapt_legacy, stage1_to_stage2_keys
    sd = adapt_legacy(stage1_to_stage2_keys(sd) if "composer" in mods else sd, want)
    missing = [k for k in want if k not in sd and not k.startswith("retention.")]
    assert not missing, f"checkpoint lacks {missing[:5]}"
    mods.load_state_dict({k: v for k, v in sd.items() if k in want}, strict=False)


def build_model(cfg, stage, device):
    m = cfg["model"]
    os.environ["JEPA_VIS_LAST"] = "1" if cfg.get("data", {}).get("vis_last") else "0"      # prompt layout of this run (chat_format.place_vis)
    os.environ["JEPA_SAMPLING"] = cfg.get("data", {}).get("sampling", "paired")      # (10-21) frame times of this run (video.paired_uniform_times)
    if os.environ.get("JEPA_EVAL_SAMPLING"):              # (10-22) measurement only: other frame times than the run's (e.g. paired 0.25 s tubelets)
        os.environ["JEPA_SAMPLING"] = os.environ["JEPA_EVAL_SAMPLING"]
    vision = build_vision(m["vision"]).eval()
    llm = build_llm(cfg, device)
    if stage == 1 and m.get("graft"):
        from jepa_llm.models.graft import GraftModel
        model = GraftModel(vision, llm, evidence_cells=cfg["data"].get("cells", 7), bridge_hidden=m.get("bridge_hidden", 2048), **m["graft"])
        return model, torch.nn.ModuleDict({"core": model.core, "bridge": model.bridge})
    if stage == 1:
        from jepa_llm.models.stage1 import Stage1Model
        # single-stage runs may fine-tune V-JEPA: its weights come from the checkpoint, while the reference copy
        # (built here from the official weights) keeps supplying the surprise expectations, as in training
        tv = m.get("train_vision", "none") if isinstance(m.get("train_vision", "none"), str) else "none"
        model = Stage1Model(vision, llm, None, evidence_cells=cfg["data"].get("cells", 4),
                            surprise_segments=cfg["model"].get("surprise_segments", 0),
                            bridge_hidden=m.get("bridge_hidden", 2048), train_vision=tv, spectral=m.get("spectral"),
                            token_order=m.get("token_order", "tyx"), channel=m.get("channel"), expert=m.get("expert"),
                            vcode=m.get("vcode"), entry_layer=m.get("entry_layer"), memory=m.get("memory", False))      # resizes the vocabulary as in training, so the weights load
        os.environ["JEPA_MEMORY"] = "1" if (m.get("memory") is True or (isinstance(m.get("memory"), dict) and "cells" not in m["memory"])) else "0"     # no video slots in the prompts (chat_format)
        model.causal_win = int(m.get("causal_window", 0))
        model.time_groups = int(m.get("time_groups", 0))
        model.chunk_frames = int(m.get("chunk_frames", 0))   # long clips encoded in chunks of this many frames
        model.vjepa_last = m.get("vjepa_levels") == "last"   # V-JEPA final output only
        model.flow_clean = bool(m.get("flow_clean", False))
        model.flow_global = bool(m.get("flow_global", False))   # (11-03) frame-level flow only   # (11-03) confidence-weighted, camera-relative flow
        if m.get("flow_lags"): model.flow_lags = tuple(int(x) for x in m["flow_lags"])
        if m.get("flow_feats"): model.enable_flow(9 * len(getattr(model, "flow_lags", (1,))))   # (11-02) the latent-flow embedding is part of the bridge's weights
        mods = torch.nn.ModuleDict({"core": model.core, "bridge": model.bridge})
        if getattr(model, "mem_bridge", None) is not None:
            mods["mem_bridge"] = model.mem_bridge
        if m.get("temporal_memory"):                      # (10-17) Gated DeltaNet memory; evaluation clips per item: up to JEPA_TMEM_KMAX
            model.enable_tmem(m["temporal_memory"]); mods["tmem"] = model.tmem
            os.environ.setdefault("JEPA_TMEM_KMAX", str(int(m["temporal_memory"].get("eval_kmax", 32))))
            os.environ.setdefault("JEPA_TMEM_CLIPS", str(float(m["temporal_memory"].get("clip_s", 8.0))))
        if tv != "none":
            mods["vision"] = model.vision
    else:
        from jepa_llm.models.stage2 import Stage2Model
        model = Stage2Model(vision, llm, n_sem=m["n_sem"], retention_layers=None, n_time_bins=m.get("n_time_bins", 4),
                            region_grid=m.get("region_grid", 3), bridge_hidden=m.get("bridge_hidden", 2048))
        mods = torch.nn.ModuleDict({"core": model.core, "bridge": model.bridge, "composer": model.composer})
        if m.get("train_vision"):
            mods["vision"] = model.vision
    return model, mods


@torch.no_grad()
def score_item(model, stage, tok, item, num_frames, device, dtype, lids):
    if (item.get("extra") or {}).get("image"):           # (10-21) still image (OCR): high-resolution image path, open answer
        img = load_clip(item["video"], 2, size=int(os.environ.get("JEPA_IMAGE_SIZE", 1152)), frame="pad")[0].unsqueeze(0).to(device, dtype)
        ev = model.image_tokens(img, int(os.environ.get("JEPA_IMAGE_POOL", 3)))
        return None, None, {"response": generate(model, tok, ev, item["question"], device, max_new_tokens=32)}
    window = None
    if item.get("cut_sec") is not None:
        window = (0.0, float(item["cut_sec"]))
    elif item.get("start_sec") is not None and item.get("end_sec") is not None:
        window = (float(item["start_sec"]), float(item["end_sec"]))
    video, times = _load(item, num_frames)
    video = _aug(video, item)
    video, times = video.unsqueeze(0).to(device, dtype), times.unsqueeze(0).to(device)
    q = prompt(item)
    info = {}
    if stage == 1 and getattr(model, "graft", False):
        ev, _ = model.embed(video, times)                # whole clip seen + V-JEPA's prediction of what follows
    elif stage == 1:
        pooled, coords, sec, geom, *res = model.evidence(video, times)
        ev = model.bridge(pooled, coords, sec, 0, geom, residual=res[0] if res else None)
    else:
        z = model.observed(video)
        ov_p, ov_c, ov_s, ov_g = model.overview(z, times)
        ov = model.bridge(ov_p, ov_c, ov_s, 0, ov_g)
        p1 = build_pass1(tok, q, ov.shape[1])
        t = lambda k: torch.tensor([p1[k]], device=device)
        read = (t("sem_mask") + t("loc_mask")).bool()
        h = model.core(t("input_ids"), torch.ones_like(t("input_ids")), ov, t("vis_mask"), labels=None, read_mask=read).view(1, 2, -1)
        logits, _ = model.composer(h[:, 0], h[:, 1])
        spec = model.composer.decode(logits)[0]
        info["spec"] = [spec.kind, spec.semantic_id, spec.time_bin, spec.region]
        ex = model.execute(video, z, times, [spec])
        ev = model.embed_evidence(ex)
    if not item.get("options"):
        return None, None, {**info, "response": generate(model, tok, ev, q, device)}
    p2 = build_example(tok, [{"role": "user", "content": f"{VIS}\n{q}"}, {"role": "assistant", "content": "A"}],
                       ev.shape[1], 100000)
    t2 = lambda k: torch.tensor([p2[k]], device=device)
    if getattr(model, "mem_cells", 0):
        model.core.mem_input = model.mem_tokens()
    lg, _, _, _ = model.core(t2("input_ids"), torch.ones_like(t2("input_ids")), ev, t2("vis_mask"), labels=t2("labels"))
    scores = lg[0, lids[: len(item["options"])]]
    return int(scores.argmax()), scores.tolist(), info


def _aug(v, item):
    """(10-15) counterfactual held-out items: the clip mirrored ("flip") or played backwards ("rev"); v [3, T, H, W]"""
    a = item.get("aug") or ""
    return v.flip(-1) if a == "flip" else v.flip(1) if a == "rev" else v


def _load(item, nf):
    """the clip, or (10-16, item["tq"]: transformation questions) the clip composed by video.compose_clip"""
    if item.get("image"):                                    # (10-28) image QA items: a still image as a static clip (_load_image)
        return _load_image(item, nf)
    sp = item.get("tq")
    if sp and sp.get("kind") == "seq":                       # (10-18) long video built from segments (absolute paths)
        from jepa_llm.data.video import seq_clip
        return seq_clip(sp["segs"], "/", nf)
    if sp:
        return compose_clip(sp, item["video"], nf, window=_window(item), path2=sp.get("v2"), window2=tuple(sp["w2"]) if sp.get("w2") else None)
    if os.environ.get("JEPA_EVAL_SPEED"):                    # (10-28) measurement only: playback speed (as baseline_eval BASE_EVAL_SPEED)
        return _load_speed(item, nf, float(os.environ["JEPA_EVAL_SPEED"]))
    return load_clip(item["video"], nf, window=_window(item))[:2]


def _img_frame():
    """(10-28) image items: how the image becomes the 384 square - JEPA_IMG_FRAME crop (default: short side to 384, centre square, as
    every video) | pad (letterbox: long side to 384, the bars filled with the ImageNet mean colour = 0 after normalising)"""
    m = os.environ.get("JEPA_IMG_FRAME", "crop")
    assert m in ("crop", "pad"), m
    return m


def _img_span():
    """(10-28) image items: the static clip's frames are evenly spaced over this many seconds (JEPA_IMG_SPAN, default 8: 0.25 s apart,
    the frame times a 8 s video gets under uniform 32-frame sampling and the V-JEPA tubelet step; load_clip's still-image convention)"""
    return float(os.environ.get("JEPA_IMG_SPAN", 8.0))


def _load_image(item, nf):
    """(10-28) item["image"] (jpg / png / webp file) -> (video [3, nf, 384, 384] normalised, times [nf]): the image repeated over the nf
    frames, resized / squared / ImageNet-normalised exactly as video frames (video.load_clip -> _finish), times from _img_span"""
    mode = os.environ.get("JEPA_IMG_MODE", "repeat")         # (10-28) repeat (default) | single | tiles
    if mode == "single":                                     # one V-JEPA tubelet (2 identical frames): 1 moment x 8 x 8 = 64 tokens
        v = load_clip(item["image"], 1, frame=_img_frame())[0].expand(-1, 2, -1, -1).contiguous()
        return v, torch.tensor([0.125, 0.375], dtype=torch.float32)
    if mode == "native":                                     # V-JEPA 2.1's own image path: ONE frame [3, 1, 384, 384] -> the encoder's 2D image
        v = load_clip(item["image"], 1, frame=_img_frame())[0]   # tokenizer (patch_embed_img + img_mod_embed, img_temporal_dim_size=1): 24 x 24
        return v, torch.tensor([0.125, 0.375], dtype=torch.float32)   # patches = one moment's grid -> 64 tokens; times as one tubelet
    if mode == "tiles":                                      # the image at 4x the side (1536), cut into 4 x 4 tiles of 384 visited in a snake
        g = 4                                                # order (a camera panning over it); each tile = one moment (2 identical frames)
        big = load_clip(item["image"], 1, size=384 * g, frame=_img_frame())[0][:, 0]                 # [3, 1536, 1536]
        tiles = [big[:, r * 384:(r + 1) * 384, c * 384:(c + 1) * 384] for r in range(g) for c in (range(g) if r % 2 == 0 else range(g - 1, -1, -1))]
        v = torch.stack([t for t in tiles for _ in range(2)], 1).contiguous()                       # [3, 32, 384, 384]
        return v, (torch.arange(v.shape[1], dtype=torch.float32) + 0.5) * _img_span() / v.shape[1]
    v = load_clip(item["image"], 1, frame=_img_frame())[0].expand(-1, nf, -1, -1).contiguous()   # resized once, then repeated
    return v, (torch.arange(nf, dtype=torch.float32) + 0.5) * _img_span() / nf


def _load_speed(item, nf, s):
    """(10-28) the clip played at speed s: frames over a centred sub-window of min(1, 0.5 s) of the item's span (s = 1: the middle half),
    the tubelet pair gap 0.25 s x s of real time, and frame times divided by s around the sub-window start, so every condition shows the
    same apparent duration (half the span) and only the real motion per step changes"""
    import decord
    vr = decord.VideoReader(item["video"], num_threads=1); dur = len(vr) / float(vr.get_avg_fps()); del vr
    w = _window(item); t0, t1 = (0.0, dur) if w is None else (max(0.0, w[0]), min(dur, w[1]))
    if t1 <= t0:
        t0, t1 = 0.0, dur
    c, L = (t0 + t1) / 2, (t1 - t0) * min(1.0, 0.5 * s)
    v, t = load_clip(item["video"], nf, window=(c - L / 2, c + L / 2), pair_gap_sec=0.25 * s)[:2]
    return v, (c - L / 2) + (t - (c - L / 2)) / s


def _shuffle_index(item_id, nf, unit=2):
    """(10-28) fixed per-item permutation of the frame order in 2-frame units (tubelets); identical to baseline_eval.shuffle_index"""
    import hashlib
    perm = np.random.RandomState(int(hashlib.md5(str(item_id).encode()).hexdigest()[:8], 16)).permutation(nf // unit)
    return torch.from_numpy((perm[:, None] * unit + np.arange(unit)[None]).reshape(-1))


def _window(item):
    if item.get("cut_sec") is not None:
        return (0.0, float(item["cut_sec"]))
    if item.get("start_sec") is not None and item.get("end_sec") is not None:
        return (float(item["start_sec"]), float(item["end_sec"]))
    return None


def _mem_frames(item, kmax, nf=16, clip_s=8.0):
    """(10-17) temporal-memory clips of an evaluation item, as in the training store (logs/zz_mfeat_build.py): the item's window (whole
    video by default) cut into K = clamp(ceil(span / clip_s), 1, kmax) equal sub-windows, 16 paired-uniform frames each, the video opened
    once -> (uint8 frames [K * 16, H, W, 3], clip centres [K] seconds). Composed clips (item['tq']) are not split: K = 0 (the model uses
    the overview clip itself)."""
    import decord
    from jepa_llm.data.video import paired_uniform_times
    if item.get("tq"):
        return None, None
    vr = decord.VideoReader(item["video"], num_threads=1); fps = float(vr.get_avg_fps()); n = len(vr)
    w = _window(item); t0, t1 = (0.0, n / fps) if w is None else (max(0.0, w[0]), min(n / fps, w[1]))
    K = int(min(max(math.ceil((t1 - t0) / clip_s), 1), kmax)); gap = max(0.25, 1.0 / fps); edges = np.linspace(t0, t1, K + 1); T = []
    for k in range(K):
        ww = max(edges[k + 1] - edges[k], 1e-3); T.append(edges[k] + paired_uniform_times(ww, nf, min(gap, ww / 2)))
    idx = np.clip(np.round(np.concatenate(T) * fps).astype(int), 0, n - 1); u, inv = np.unique(idx, return_inverse=True)
    fr = torch.from_numpy(vr.get_batch(u.tolist()).asnumpy())[torch.from_numpy(inv)]
    return fr, torch.tensor((edges[:-1] + edges[1:]) / 2, dtype=torch.float32)


@torch.no_grad()
def _mem_encode(model, fr, device, alt="", nf=16, size=384):
    """(memory resolution: model.tmem_res; frames cropped to the overview's 384 field of view first, as the training store)"""
    return model.live_memory(_mem_prep(fr, device, alt, nf, size))


def _mem_prep(fr, device, alt="", nf=16, size=384):
    """uint8 frames [K*16, H, W, 3] -> [K, 128, 1408] (frozen V-JEPA, final level, 4 x 4 cells), preprocessing as load_clip, on the GPU"""
    import torch.nn.functional as F
    from jepa_llm.data.video import MEAN, STD
    from jepa_llm.models.evidence_bridge import normalize_levels, pool_regions
    x = fr.to(device).permute(0, 3, 1, 2).float() / 255; h, w = x.shape[-2:]
    if min(h, w) != size:
        sc = size / min(h, w); x = F.interpolate(x, size=(round(h * sc), round(w * sc)), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1); h, w = x.shape[-2:]
    y0, x0 = (h - size) // 2, (w - size) // 2; x = x[:, :, y0:y0 + size, x0:x0 + size]
    x = ((x - MEAN.view(1, 3, 1, 1).to(device)) / STD.view(1, 3, 1, 1).to(device)).view(-1, nf, 3, size, size).transpose(1, 2)   # [K, 3, 16, H, W]
    if alt == "held":
        x = x[:, :, nf // 2: nf // 2 + 1].expand(-1, -1, nf, -1, -1).contiguous()
    elif alt == "backwards":
        x = x.flip(0).flip(2)                               # clips in reverse order, each played backwards
    return x.to(torch.bfloat16)


class _ClipSet(torch.utils.data.Dataset):
    """Decodes clips in DataLoader workers; returns the cached V-JEPA evidence instead when it exists."""

    def __init__(self, items, nf, cache_dir, tag=""):
        self.items, self.nf, self.cache_dir, self.tag = items, nf, cache_dir, tag

    def key(self, item):
        import hashlib
        # (10-28) image items: keyed by the image file, its framing and frame times (video items: the key is unchanged)
        src = [item["video"], _window(item), self.nf, self.tag] if not item.get("image") else \
            [item["image"], "image", _img_frame(), _img_span(), self.nf, self.tag] + \
            ([os.environ["JEPA_IMG_MODE"]] if os.environ.get("JEPA_IMG_MODE", "repeat") != "repeat" else [])   # (10-28) single / tiles
        return hashlib.md5(json.dumps(src + (["uniform"] if os.environ.get("JEPA_SAMPLING") == "uniform" else []) + ([item["aug"]] if item.get("aug") else [])
                                      + ([item["tq"]] if item.get("tq") else []), sort_keys=True).encode()).hexdigest()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        if os.environ.get("JEPA_EVAL_VIDEO", "") == "blank":   # (10-18) no visual content: nothing to decode (run_batched zeroes the video anyway)
            return {"i": i, "video": torch.zeros(3, self.nf, 384, 384), "times": torch.arange(self.nf).float() * 0.5, "cache_path": None}
        cp = f"{self.cache_dir}/{self.key(it)}.pt" if self.cache_dir else None
        try:
            kmax = int(os.environ.get("JEPA_TMEM_KMAX", 0))
            if kmax:                                          # (10-17) temporal memory: the item's memory clips (cached separately)
                cs = float(os.environ.get("JEPA_TMEM_CLIPS", 8.0)); res_ = int(os.environ.get("JEPA_TMEM_RES", 256))
                mp = cp[:-3] + f".mem{kmax}_{cs:g}s_{res_}.pt" if cp else None   # memory features depend on clip length and resolution
                if mp and os.path.exists(mp):
                    mem = {"memfeat": torch.load(mp, map_location="cpu", weights_only=False)}
                else:
                    fr, msec = _mem_frames(it, kmax, self.nf, cs)
                    mem = {"memfr": fr, "memsec": msec, "mem_path": mp} if fr is not None else {}
            else:
                mem = {}
            if cp and os.path.exists(cp):
                return {"i": i, "cached": torch.load(cp, map_location="cpu", weights_only=False), "cache_path": cp, **mem}
            v, t = _load(it, self.nf)
            v = _aug(v, it)
            return {"i": i, "video": v, "times": t, "cache_path": cp, **mem}
        except Exception as e:
            return {"i": i, "error": f"{type(e).__name__}: {str(e)[:200]}"}


@torch.no_grad()
def run_batched(model, tok, items, nf, device, dtype, lids, out, bs=8, workers=6, cache_dir=None):
    """Option-letter scoring in batches with prefetched decoding and (frozen V-JEPA only) a per-clip evidence cache:
    the pooled V-JEPA states of a clip do not depend on the checkpoint when V-JEPA is frozen, so they are computed once.
    Same prompt, same letter scoring as score_item (stage 1)."""
    graft = getattr(model, "graft", False)
    alt = os.environ.get("JEPA_EVAL_VIDEO", "")     # measurement only: "held" (the middle frame repeated: no motion) | "backwards" | "blank"
    use_cache = cache_dir is not None and getattr(model, "train_vision", "none") == "none" and not graft and not alt and not getattr(model, "mem_cells", 0) \
        and not os.environ.get("JEPA_EVAL_SPEED")         # (10-28) speed measurement: other frames than the cached ones
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
    tag = f"{model.vision.variant}|cells{model.cells}|surprise{model.surprise_segments}"   # what the pooled states depend on
    if getattr(model, "causal_win", 0) or getattr(model, "time_groups", 0) or getattr(model, "chunk_frames", 0):
        tag += f"|causal{model.causal_win}|tgroups{model.time_groups}|chunk{getattr(model, 'chunk_frames', 0)}"
    if getattr(model.bridge, "split", 0):                     # (10-17) band-split tokens: the cache holds equalised tokens (and their gain), not pooled states
        tag += f"|split{model.bridge.split}|pg{model.bridge.patch_gain:.4f}"
    if getattr(model, "flow_feats", False):
        tag += "|flow" + "".join(str(x) for x in getattr(model, "flow_lags", (1,))) + ("c" if getattr(model, "flow_clean", False) else "") + ("g" if getattr(model, "flow_global", False) else "")   # (11-02) the cache also holds the latent-flow features
    if getattr(model, "vjepa_last", False):
        tag += "|last"                                    # (10-16) final V-JEPA level only: other pooled states than the 4-level ones
    ds = _ClipSet(items, nf, cache_dir if use_cache else None, tag)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=workers, collate_fn=lambda b: b, prefetch_factor=4 if workers else None)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for batch in dl:
        for r in [r for r in batch if "error" in r]:
            out.write(json.dumps({"id": items[r["i"]]["id"], "pred_index": None, "error": r["error"]}) + "\n")
        batch = [r for r in batch if "error" not in r]
        if not batch:
            continue
        fresh = [r for r in batch if "video" in r]
        if graft:
            v = torch.stack([r["video"] for r in batch]).to(device, dtype); t = torch.stack([r["times"] for r in batch]).to(device)
            ev, _ = model.embed(v, t)
        elif fresh:
            v = torch.stack([r["video"] for r in fresh]).to(device, dtype); t = torch.stack([r["times"] for r in fresh]).to(device)
            if alt == "held":
                v = v[:, :, v.shape[2] // 2: v.shape[2] // 2 + 1].expand(-1, -1, v.shape[2], -1, -1).contiguous()
            elif alt == "backwards":
                v = v.flip(2)
            elif alt == "blank":                                 # (10-17) measurement only: no visual content (the language prior)
                v = torch.zeros_like(v)
            elif alt == "shuffle":                               # (10-28) measurement only: frame order shuffled in 2-frame units (tubelets),
                v = torch.stack([v[j][:, _shuffle_index(items[r["i"]]["id"], v.shape[2]).to(v.device)] for j, r in enumerate(fresh)])   # fixed per item
            elif alt == "hflip":                                 # (10-28) measurement only: the clip mirrored left <-> right
                v = v.flip(-1)
            pooled, coords, sec, geom, *res = model.evidence(v, t)
            for j, r in enumerate(fresh):
                r["cached"] = {"pooled": pooled[j].to(torch.bfloat16).cpu(), "coords": coords.cpu(), "sec": sec[j].cpu(),
                               "geom": geom[j].cpu(), "res": res[0][j].to(torch.bfloat16).cpu() if res else None,
                               "flow": model.bridge._flow[j].float().cpu() if getattr(model.bridge, "_flow", None) is not None else None}
                if use_cache and r["cache_path"] and not os.path.exists(r["cache_path"]):
                    tmp = r["cache_path"] + f".tmp{os.getpid()}"; torch.save(r["cached"], tmp); os.replace(tmp, r["cache_path"])
        if not graft:
            c = [r["cached"] for r in batch]
            pooled = torch.stack([x["pooled"] for x in c]).to(device, dtype)
            sec = torch.stack([x["sec"] for x in c]).to(device); geom = torch.stack([x["geom"] for x in c]).to(device)
            res = torch.stack([x["res"] for x in c]).to(device, dtype) if c[0]["res"] is not None else None
            model.bridge._flow = torch.stack([x["flow"] for x in c]).to(device) if c[0].get("flow") is not None else None
            ev = model.bridge(pooled, c[0]["coords"].to(device), sec, 0, geom, residual=res)
        if getattr(model, "tmem", None) is not None:          # (10-17) temporal memory tokens after the overview tokens
            feats, secs = [], []
            for r in batch:
                if "memfeat" in r:
                    f_ = r["memfeat"]["feat"].to(device).float(); s_ = r["memfeat"]["sec"]
                elif "memfr" in r:
                    f_ = _mem_encode(model, r["memfr"], device, alt); s_ = r["memsec"]
                    if r.get("mem_path") and not alt and not os.path.exists(r["mem_path"]):
                        tmp = r["mem_path"] + f".tmp{os.getpid()}"; torch.save({"feat": f_.half().cpu(), "sec": s_}, tmp); os.replace(tmp, r["mem_path"])
                else:
                    f_, s_ = None, None                       # composed clip: the overview clip itself (pooled live; needs the overview pass)
                feats.append(f_); secs.append(s_)
            if any(f_ is None for f_ in feats):              # rows without memory clips: encode their overview clip once more as one memory clip
                for j, r in enumerate(batch):
                    if feats[j] is None:
                        it = items[r["i"]]; vv, tt = _load(it, nf); vv = _aug(vv, it)
                        vv = vv[:, vv.shape[1] // 2: vv.shape[1] // 2 + 1].expand(-1, vv.shape[1], -1, -1) if alt == "held" else vv.flip(1) if alt == "backwards" else vv
                        feats[j] = model.live_memory(vv.unsqueeze(0).to(device, dtype)).float()
                        secs[j] = tt.float().mean().view(1)
            km = max(f_.shape[0] for f_ in feats); B = len(batch)
            mf = torch.zeros(B, km, 128, 1408, device=device); ms = torch.zeros(B, km, device=device); mk = torch.zeros(B, dtype=torch.long, device=device)
            for j in range(B):
                n_ = feats[j].shape[0]; mf[j, :n_] = feats[j]; ms[j, :n_] = secs[j].to(device); mk[j] = n_
            ev = model.append_memory(ev, None, {"feat": mf, "k": mk, "sec": ms})
        # (10-22) open-ended video items (no options: temporal grounding, numeric answers) ride the same prefetch / evidence cache and
        # are answered by greedy decoding, one row at a time; the option rows of the batch are scored below as before
        GEN = os.environ.get("JEPA_EVAL_GEN") == "1"               # (11-12, review Q3) measurement only: option items answered by generation + parsing
        openj = [j for j, r in enumerate(batch) if GEN or not items[r["i"]].get("options")]
        for j in openj:
            it = items[batch[j]["i"]]
            resp = generate(model, tok, ev[j:j + 1], prompt(it), device, max_new_tokens=16 if it.get("options") else 64)
            pi = parse_option(resp, it["options"]) if it.get("options") else None
            out.write(json.dumps({"id": it["id"], "pred_index": pi, "response": resp}) + "\n")
        if openj:
            keep = [j for j in range(len(batch)) if j not in openj]
            if not keep:
                out.flush(); continue
            ev = ev[keep]; batch = [batch[j] for j in keep]
        exs = [build_example(tok, [{"role": "user", "content": f"{VIS}\n{prompt(items[r['i']])}"}, {"role": "assistant", "content": "A"}],
                             ev.shape[1], 100000) for r in batch]
        L = max(len(e["input_ids"]) for e in exs)
        ids = torch.tensor([e["input_ids"] + [pad] * (L - len(e["input_ids"])) for e in exs], device=device)
        lab = torch.tensor([e["labels"] + [IGNORE] * (L - len(e["labels"])) for e in exs], device=device)
        vm = torch.tensor([e["vis_mask"] + [0] * (L - len(e["vis_mask"])) for e in exs], device=device)
        am = torch.tensor([[1] * len(e["input_ids"]) + [0] * (L - len(e["input_ids"])) for e in exs], device=device)
        if getattr(model, "mem_cells", 0):                     # hybrid: the full grid of this batch (no cache: every clip encoded above)
            model.core.mem_input = model.mem_tokens()
        lg, _, ex_id, _ = model.core(ids, am, ev, vm, lab)
        first = {}
        for row, e in enumerate(ex_id.tolist()):
            first.setdefault(e, row)
        for j, r in enumerate(batch):
            it = items[r["i"]]
            sc = lg[first[j], lids[: len(it["options"])]]
            out.write(json.dumps({"id": it["id"], "pred_index": int(sc.argmax()), "scores": sc.float().tolist()}) + "\n")
        out.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--stage", type=int, default=1); ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int); ap.add_argument("--num-frames", type=int)
    ap.add_argument("--batch", type=int, default=1, help=">1: batched scoring with prefetched decoding (stage 1, options)")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--cache_dir", help="frozen V-JEPA evidence cache")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    distributed = "WORLD_SIZE" in os.environ and a.device == "cuda"
    if distributed:
        dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"])); torch.cuda.set_device(device)
    else:
        rank, world, device = 0, 1, torch.device(a.device)
    dtype = compute_dtype(device)
    from jepa_llm.data.chat_format import setup_tokenizer
    tok = setup_tokenizer(AutoTokenizer.from_pretrained(cfg["model"]["llm"]), cfg["model"])
    model, mods = build_model(cfg, a.stage, device)
    load_weights(mods, a.ckpt)
    model.to(device=device, dtype=dtype).eval()
    nf = a.num_frames or int(os.environ.get("JEPA_EVAL_NF", 0)) or cfg["data"]["num_frames"]   # (10-17) JEPA_EVAL_NF: frame-count measurement
    # (10-19) --items / --out may be comma-separated lists: several benchmarks on one loaded model (loading took ~4 min per benchmark)
    for ipath, opath in zip(a.items.split(","), a.out.split(",")):
        run_items(model, tok, a, ipath, opath, nf, device, dtype, distributed)
    if distributed:
        dist.barrier(); dist.destroy_process_group()


def run_items(model, tok, a, ipath, opath, nf, device, dtype, distributed):
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    items = [json.loads(l) for l in open(ipath)][: a.limit]
    lids = letter_ids(tok, max([len(i["options"]) for i in items if i.get("options")] or [2]))
    out = open(f"{opath}.rank{rank}" if distributed else opath, "w")
    if a.batch > 1 and a.stage == 1:
        mine = [it for k, it in enumerate(items) if k % world == rank]
        isimg = lambda it: bool((it.get("extra") or {}).get("image")) if isinstance(it.get("extra"), dict) else False
        mc = [it for it in mine if not isimg(it)]                 # (10-22) open-ended video items too (generated inside run_batched)
        run_batched(model, tok, mc, nf, device, dtype, lids, out, a.batch, a.workers, a.cache_dir)
        items = [it for it in mine if isimg(it)]; world, rank = 1, 0       # still images (OCR): per item below
    for k, item in enumerate(items):
        if k % world != rank:
            continue
        try:
            pred, scores, info = score_item(model, a.stage, tok, item, nf, device, dtype, lids)
            rec = {"id": item["id"], "pred_index": pred, "scores": scores, **info}
        except Exception as e:  # unreadable video etc.: counted as missing, never silently dropped
            rec = {"id": item["id"], "pred_index": None, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        out.write(json.dumps(rec) + "\n"); out.flush()
    out.close()
    if distributed:
        dist.barrier()


if __name__ == "__main__":
    main()
