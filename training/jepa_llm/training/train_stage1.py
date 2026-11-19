"""Stage-1 trainer (spec 5.3, 13.5, 14): full fine-tuning of the LLM + bridge (+ retention), frozen V-JEPA.

    torchrun --nproc_per_node 8 -m jepa_llm.training.train_stage1 configs/stage1/<run>.yaml
    python -m jepa_llm.training.train_stage1 <cfg> --device cpu      # single-process smoke run, no FSDP

FSDP2 shards every decoder layer, the language core root (embeddings / final norm / LM head), the bridge and R.
Parameters are fp32 masters with bf16 compute; AdamW state is fp32. The frozen V-JEPA stays outside FSDP in bf16.
Checkpoints (torch.distributed.checkpoint) hold only trainable modules + optimizer + a train_state.json; resuming
continues the same sample order without decoding skipped batches.
"""
import argparse
import functools
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from jepa_llm.data.stage1_dataset import ResumableShardSampler, Stage1Dataset, collate
from jepa_llm.models.stage1 import Stage1Model
from jepa_llm.models.vision_backbone import VJEPABackbone


def log0(*a):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*a, flush=True)


def setup_dist(device_arg):
    """-> (distributed, rank, world, device, mesh). torchrun + cuda -> NCCL; torchrun + cpu -> gloo with an explicit CPU
    mesh (used to test the FSDP / DCP code paths without GPUs)."""
    if "WORLD_SIZE" not in os.environ:
        return False, 0, 1, torch.device(device_arg), None
    from torch.distributed.device_mesh import init_device_mesh
    if device_arg == "cuda":
        import datetime   # (10-29) JEPA_DIST_TIMEOUT_MIN: collective timeout (default 10 min as before); compile warm-up can exceed it
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=int(os.environ.get("JEPA_DIST_TIMEOUT_MIN", "10"))))
        local = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        dist.init_process_group("gloo")
        device = torch.device("cpu")
    mesh = init_device_mesh(device.type, (dist.get_world_size(),))
    return True, dist.get_rank(), dist.get_world_size(), device, mesh


def low_precision(device):
    """bf16 compute on GPU. JEPA_EMULATE_BF16=1 gives the CPU smoke runs the same dtypes as the GPU path (the first GPU run
    failed on a dtype mismatch that the fp32 CPU smoke runs could not see)."""
    return device.type == "cuda" or os.environ.get("JEPA_EMULATE_BF16") == "1"


def compute_dtype(device):
    return torch.bfloat16 if low_precision(device) else torch.float32


def shard(module, mesh, device, reshard: bool = True):
    """FSDP2 fully_shard; bf16 compute with fp32 reduce on GPU (or under JEPA_EMULATE_BF16), plain fp32 on CPU."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    # (10-29) JEPA_REDUCE_DTYPE=bf16: gradients reduce-scattered in bf16 (half the traffic; the profile of nx16_Q8 showed the fp32
    # reduce-scatter as 38 % of GPU time). Default fp32 as before
    rd = torch.bfloat16 if os.environ.get("JEPA_REDUCE_DTYPE") == "bf16" else torch.float32
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=rd) if low_precision(device) else MixedPrecisionPolicy()
    # reshard=False keeps the gathered parameters after forward (no re-gather in backward): faster for small models
    fully_shard(module, mesh=mesh, mp_policy=mp, reshard_after_forward=reshard)


def build_llm(cfg, device):
    path = cfg["model"]["llm"]
    if cfg["model"].get("llm_random_tiny"):  # tests / smoke only: same tokenizer & architecture family, random weights
        c = AutoConfig.from_pretrained(path)
        for k, v in cfg["model"]["llm_random_tiny"].items():
            setattr(c, k, v)
        return AutoModelForCausalLM.from_config(c, dtype=torch.float32)
    # (11-16) MoE backbones (Qwen3-30B-A3B): grouped_mm experts kernel (the eager per-expert loop was 15x slower, runs/analysis/bench_30b.jsonl)
    moe = {"experts_implementation": "grouped_mm"} if getattr(AutoConfig.from_pretrained(path), "num_experts", None) else {}
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, **moe)


def param_groups(model, cfg):
    """llm / new (bridge, R, C) / enc / pred groups, each split into decay and no-decay; `base_lr` drives the schedule.
    The frozen reference copy never enters (requires_grad False)."""
    o = cfg["optim"]
    spec = {"llm": (o["lr_llm"], o.get("wd_llm", 0.01)), "new": (o["lr_new"], o.get("wd_new", 0.05)),
            "enc": (o.get("lr_enc", 0.0), o.get("wd_llm", 0.01)), "pred": (o.get("lr_pred", 0.0), o.get("wd_llm", 0.01))}
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad or n.startswith("reference."):
            continue
        if n.startswith(("bridge.", "retention.", "composer.", "predictor_heads.", "geo.", "critic.", "eq_heads.", "corr_head.", "next_head.", "tr_head.", "mem_bridge.", "tmem.", "dyn_head.")):   # (10-21) dyn_head was missing: it trained at the LLM rate
            g = "new"
        elif n.startswith("vision.encoder."):
            g = "enc"
        elif n.startswith("vision.predictor."):
            g = "pred"
        else:
            g = "llm"
        nd = p.ndim < 2 or "norm" in n
        groups.setdefault((g, nd), []).append(p)
    return [{"params": ps, "lr": spec[g][0], "base_lr": spec[g][0], "weight_decay": 0.0 if nd else spec[g][1],
             "name": g + ("_nd" if nd else "")} for (g, nd), ps in sorted(groups.items())]


def check_full_finetune(model, optim):
    """spec 13.5: every LLM parameter trainable and in the optimizer exactly once; V-JEPA frozen."""
    llm_params = list(model.core.llm.parameters())
    assert all(p.requires_grad for p in llm_params), "LLM must be fully trainable (no partial / LoRA training)"
    assert not any(p.requires_grad for p in model.vision.parameters()), "V-JEPA must be frozen in stage 1"
    ids = [id(p) for g in optim.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "a parameter appears twice in the optimizer"
    assert {id(p) for p in llm_params} <= set(ids), "some LLM parameters are missing from the optimizer"
    n_llm = sum(p.numel() for p in llm_params)
    n_train = sum(p.numel() for g in optim.param_groups for p in g["params"])
    return n_llm, n_train


def lr_factor(step, total, warmup_frac=0.03, floor=0.1):
    w = max(1, int(total * warmup_frac))
    if step < w:
        return (step + 1) / w
    t = (step - w) / max(1, total - w)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def trainable(model):
    mods = {"core": model.core, "bridge": model.bridge}
    if model.retention is not None:
        mods["retention"] = model.retention
    return torch.nn.ModuleDict(mods)


def save_ckpt(out, step, mods, optim, state, distributed):
    """mods: ModuleDict of the trainable modules (checkpoint never stores the frozen V-JEPA)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict
    path = f"{out}/ckpt/step_{step:07d}"
    msd, osd = get_state_dict(mods, optim)
    if distributed:
        dcp.save({"model": msd, "optim": osd}, checkpoint_id=path)
    else:
        os.makedirs(path, exist_ok=True)
        torch.save({"model": msd, "optim": osd}, f"{path}/state.pt")
    if not distributed or dist.get_rank() == 0:
        json.dump(state, open(f"{path}/train_state.json", "w"), indent=1)
        with open(f"{out}/ckpt/latest", "w") as f:
            f.write(path)
    return path


def load_ckpt(path, mods, optim, distributed):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    msd, osd = get_state_dict(mods, optim)
    if distributed:
        sd = {"model": msd, "optim": osd}
        dcp.load(sd, checkpoint_id=path)
    else:
        sd = torch.load(f"{path}/state.pt", map_location="cpu", weights_only=False)
    set_state_dict(mods, optim, model_state_dict=sd["model"], optim_state_dict=sd["optim"])
    return json.load(open(f"{path}/train_state.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run, o = cfg["run"], cfg["optim"]
    distributed, rank, world, device, mesh = setup_dist(args.device)
    seed = run.get("seed", 0)
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed)
    out = run["out_dir"]
    os.makedirs(out, exist_ok=True)

    # ---- model ----------------------------------------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(cfg["model"]["llm"])
    vision = VJEPABackbone(cfg["model"]["vision"]).eval()
    llm = build_llm(cfg, device)
    lam = float(cfg["loss"].get("lambda_state", 0.0))
    ret_layers = cfg["model"].get("retention_layers") if lam > 0 else None
    model = Stage1Model(vision, llm, ret_layers, evidence_cells=cfg["data"].get("cells", 4),
                        surprise_segments=cfg["model"].get("surprise_segments", 0),
                        bridge_hidden=cfg["model"].get("bridge_hidden", 2048))
    if run.get("activation_checkpointing", device.type == "cuda"):   # default on for GPU; CPU smoke can force it
        model.core.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vis_dtype = compute_dtype(device)
    model.vision.to(device=device, dtype=vis_dtype)
    if distributed:
        model.core.to(device); model.bridge.to(device)
        if model.retention is not None:
            model.retention.to(device)
        for layer in model.core.llm.model.layers:
            shard(layer, mesh, device)
        shard(model.core, mesh, device)
        shard(model.bridge, mesh, device)
        if model.retention is not None:
            shard(model.retention, mesh, device)
    else:
        model.core.to(device); model.bridge.to(device)
        if model.retention is not None:
            model.retention.to(device)
    optim = torch.optim.AdamW(param_groups(model, cfg), betas=(0.9, 0.95), eps=1e-8,
                              fused=device.type == "cuda")
    n_llm, n_train = check_full_finetune(model, optim)
    log0(f"[init] world={world} llm_params={n_llm/1e6:.1f}M trainable={n_train/1e6:.1f}M "
         f"vision={cfg['model']['vision']} lambda_state={lam} retention_layers={ret_layers}")

    # ---- data -----------------------------------------------------------------------------------------------------
    d = cfg["data"]
    ds = Stage1Dataset(d["sources"], tok, d["num_frames"], d.get("cells", 4), d["max_len"], d.get("chat_kwargs"),
                       d.get("pair_gap_sec", 0.25))
    bs, accum = o["batch_per_gpu"], o.get("grad_accum", 1)
    steps_per_epoch = len(ds) // (bs * accum * world)
    total = args.max_steps or o.get("max_steps") or steps_per_epoch * o.get("epochs", 1)
    log0(f"[data] samples={len(ds)} evidence_tokens={ds.n_vis} steps/epoch={steps_per_epoch} total_steps={total} "
         f"per_source={ds.df.source.value_counts().to_dict()} weights={ {s.get('name'): s.get('weight') for s in d['sources']} }")

    state = {"step": 0, "epoch": 0, "sample_offset": 0, "skipped": {}}
    latest = f"{out}/ckpt/latest"
    if os.path.exists(latest):
        state = load_ckpt(open(latest).read().strip(), trainable(model), optim, distributed)
        log0(f"[resume] {state}")

    def loader(epoch, start):
        w = {src.get("name", src["manifest"]): src["weight"] for src in d["sources"] if "weight" in src}
        smp = ResumableShardSampler(len(ds), rank, world, seed, epoch, start,
                                    source_ids=ds.df.source.tolist() if w else None, weights=w or None)
        return iter(DataLoader(ds, batch_size=bs, sampler=smp, num_workers=d.get("workers", 8), pin_memory=True,
                               collate_fn=functools.partial(collate, pad_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id),
                               persistent_workers=False, prefetch_factor=4 if d.get("workers", 8) else None))

    it = loader(state["epoch"], state["sample_offset"])
    model.train()
    logf = open(f"{out}/log.jsonl", "a") if rank == 0 else None
    t0, seen = time.time(), 0
    while state["step"] < total:
        f = lr_factor(state["step"], total, o.get("warmup_frac", 0.03))
        for g in optim.param_groups:
            g["lr"] = g["base_lr"] * f
        agg = {"loss": 0.0, "loss_text": 0.0, "loss_state": 0.0, "answer_tokens": 0.0}
        for _ in range(accum):
            while True:  # every rank does exactly one forward/backward per micro-step (FSDP collectives stay aligned)
                try:
                    batch = next(it)
                except StopIteration:
                    state["epoch"] += 1; state["sample_offset"] = 0
                    it = loader(state["epoch"], 0)
                    batch = next(it)
                state["sample_offset"] += bs
                for s in batch["skipped"]:
                    state["skipped"][s] = state["skipped"].get(s, 0) + 1
                if "video" in batch:
                    break
            res = model(batch["video"].to(device, vis_dtype, non_blocking=True), batch["frame_times"].to(device),
                        batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device),
                        batch["vis_mask"].to(device), lambda_state=lam, region_grid=batch["region_grid"])
            (res["loss"] / accum).backward()
            for k in agg:
                agg[k] += float(res[k].detach()) / (accum if k != "answer_tokens" else 1)
            seen += batch["video"].shape[0]
        gnorm = torch.nn.utils.clip_grad_norm_([p for g in optim.param_groups for p in g["params"]], o.get("clip", 1.0))
        gnorm = float(gnorm.full_tensor() if hasattr(gnorm, "full_tensor") else gnorm)
        optim.step(); optim.zero_grad(set_to_none=True)
        state["step"] += 1
        if rank == 0 and (state["step"] % run.get("log_every", 10) == 0 or state["step"] == 1):
            dt = time.time() - t0
            rec = {"step": state["step"], **{k: round(v, 5) for k, v in agg.items()}, "grad_norm": round(gnorm, 4),
                   "lr_llm": next(g["lr"] for g in optim.param_groups if g["name"].startswith("llm")), "samples_per_s_rank0": round(seen / dt, 2),
                   "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else 0,
                   "skipped": state["skipped"], "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
        if state["step"] % run.get("save_every", 1000) == 0 or state["step"] == total:
            save_ckpt(out, state["step"], trainable(model), optim, state, distributed)
    log0("[done]", state)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
