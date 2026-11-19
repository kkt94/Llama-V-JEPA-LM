"""Stage-2 / stage-3 trainer (spec 5.4, 5.5): two passes of the shared LLM, composer C, executed evidence, L_use.
Stage 3 = `model.train_vision: true`: online E/P train (FSDP-sharded), the frozen reference E0/P0 gives retention
targets, and a `native` batch family (unlabelled 4-fps clips, L_native) is mixed in; the family of each step is drawn
from a step-seeded RNG so every rank runs the same modules (spec 14.5).

    torchrun --nproc_per_node 8 -m jepa_llm.training.train_stage2 configs/stage2/<run>.yaml
    python -m jepa_llm.training.train_stage2 <cfg> --device cpu

Initialised from a stage-1 checkpoint (`init_from`): LLM, bridge (observed projection) and R (observed heads) carry
over; the predicted-evidence projection, the predicted R heads and C start fresh — any other missing key is an error.
Teacher forcing (2A -> 2B): p_gold decays linearly from 1 to `p_gold_min` over `forcing_frac` of the steps.
"""
import argparse
import functools
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from jepa_llm.data.native_dataset import NativeClipDataset, collate_native
from jepa_llm.data.stage1_dataset import ResumableShardSampler, Stage1Dataset, collate
from jepa_llm.data.stage2_dataset import PairBatchSampler, Stage2Dataset, collate2
from jepa_llm.losses.native import make_mask_generators, native_loss
from jepa_llm.models.stage2 import Stage2Model
from jepa_llm.models.vision_backbone import VJEPABackbone
from jepa_llm.training.train_stage1 import (build_llm, compute_dtype, load_ckpt, log0, lr_factor, param_groups, save_ckpt,
                                          setup_dist, shard)

EXPECTED_NEW = ("bridge.inp.1.", "retention.1.", "composer.", "bridge.geom_proj.")


def adapt_legacy(sd, target):
    """Checkpoints written before the bridge always had OBSERVED/PREDICTED kind rows carry a 1-row kind embedding:
    keep their OBSERVED row, take the PREDICTED row from the freshly initialised model."""
    k = "bridge.kind_emb.weight"
    if k in sd and k in target and sd[k].shape[0] == 1 and target[k].shape[0] == 2:
        sd = dict(sd)
        sd[k] = torch.cat([sd[k], target[k][1:].to(sd[k].dtype)], 0)
    return sd


def stage1_to_stage2_keys(sd):
    """stage-1 trainable keys -> stage-2 names: retention.* -> retention.0.*; bridge/core unchanged. Stage-2/3
    checkpoints (retention.0./retention.1. already present) pass through unchanged."""
    if any(k.startswith(("retention.0.", "retention.1.")) for k in sd):
        return dict(sd)
    return {("retention.0." + k[len("retention."):] if k.startswith("retention.") else k): v for k, v in sd.items()}


def load_init(model, path):
    """Initialise from a stage-1 or stage-2 checkpoint. Allowed to be missing: the modules new at this stage, and in
    stage 3 the online V-JEPA (it starts from the official weights already loaded). `path` may be a run's
    ckpt/latest pointer file."""
    if os.path.isfile(path):
        path = open(path).read().strip()
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    tmp = f"{path}/../../init_full_{os.path.basename(path)}.pt"
    if os.path.exists(f"{path}/state.pt"):                  # single-process checkpoint
        sd = torch.load(f"{path}/state.pt", map_location="cpu", weights_only=False)["model"]
    else:
        if not dist.is_initialized() or dist.get_rank() == 0:
            if not os.path.exists(tmp):
                dcp_to_torch_save(path, tmp)
        if dist.is_initialized():
            dist.barrier()
        sd = torch.load(tmp, map_location="cpu", weights_only=False)["model"]
    sd = stage1_to_stage2_keys(sd)
    target = {n: p for n, p in trainable(model).state_dict().items()}
    sd = adapt_legacy(sd, target)
    missing = [k for k in target if k not in sd]
    unexpected = [k for k in sd if k not in target]
    allowed = EXPECTED_NEW + (("vision.",) if model.train_vision else ())
    bad = [k for k in missing if not k.startswith(allowed)]
    assert not bad and not unexpected, f"stage-1 -> stage-2 mismatch: missing {bad[:5]} unexpected {unexpected[:5]}"
    trainable(model).load_state_dict({k: v for k, v in sd.items() if k in target}, strict=False)
    return len(sd), len(missing)


def trainable(model):
    mods = {"core": model.core, "bridge": model.bridge, "composer": model.composer}
    if model.train_vision:
        mods["vision"] = model.vision          # online E/P (the frozen reference is restored from its checkpoint)
    if model.retention is not None:
        mods["retention"] = model.retention
    return torch.nn.ModuleDict(mods)


def to_device(batch, device, vis_dtype):
    out = {"video": batch["video"].to(device, vis_dtype, non_blocking=True), "frame_times": batch["frame_times"].to(device)}
    for k in ("pass1", "pass2", "gold"):
        out[k] = {a: t.to(device) for a, t in batch[k].items()}
    out["pair_index"], out["pair_type"] = batch["pair_index"].to(device), batch["pair_type"].to(device)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--device", default="cuda"); ap.add_argument("--max-steps", type=int)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run, o, d, m = cfg["run"], cfg["optim"], cfg["data"], cfg["model"]
    distributed, rank, world, device, mesh = setup_dist(args.device)
    seed = run.get("seed", 0)
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed)
    out = run["out_dir"]; os.makedirs(out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(m["llm"])
    vision = VJEPABackbone(m["vision"]).eval()
    lam = cfg["loss"]
    model = Stage2Model(vision, build_llm(cfg, device), n_sem=m["n_sem"],
                        retention_layers=m.get("retention_layers") if lam.get("state", 0) > 0 else None,
                        n_time_bins=m.get("n_time_bins", 4), region_grid=m.get("region_grid", 3),
                        bridge_hidden=m.get("bridge_hidden", 2048), use_margin=lam.get("use_margin", 0.2),
                        train_vision=m.get("train_vision", False))
    if m.get("init_from"):
        n_loaded, n_new = load_init(model, m["init_from"])
        log0(f"[init] weights from {m['init_from']}: {n_loaded} tensors, {n_new} new / official tensors")
    if run.get("activation_checkpointing", device.type == "cuda"):   # default on for GPU; CPU smoke can force it
        model.core.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vis_dtype = compute_dtype(device)
    if model.train_vision:
        model.vision.to(device)                       # fp32 masters; FSDP casts compute to bf16
        model.reference.to(device=device, dtype=vis_dtype)
    else:
        model.vision.to(device=device, dtype=vis_dtype)
    if distributed:
        for mod in (model.core, model.bridge, model.composer) + ((model.retention,) if model.retention is not None else ()):
            mod.to(device)
        for layer in model.core.llm.model.layers:
            shard(layer, mesh, device)
        # R is a ModuleDict (one head set per evidence kind) without forward(): shard each head set, which has one
        for mod in (model.core, model.bridge, model.composer) + (tuple(model.retention.values()) if model.retention is not None else ()):
            shard(mod, mesh, device)
        if model.train_vision:
            for blk in list(model.vision.encoder.blocks) + list(model.vision.predictor.predictor_blocks):
                shard(blk, mesh, device)
            shard(model.vision.encoder, mesh, device)
            shard(model.vision.predictor, mesh, device)
    else:
        for mod in (model.core, model.bridge, model.composer) + ((model.retention,) if model.retention is not None else ()):
            mod.to(device)
    optim = torch.optim.AdamW(param_groups(model, cfg), betas=(0.9, 0.95), eps=1e-8, fused=device.type == "cuda")
    llm_ids = {id(p) for p in model.core.llm.parameters()}
    opt_ids = [id(p) for g in optim.param_groups for p in g["params"]]
    assert llm_ids <= set(opt_ids) and len(opt_ids) == len(set(opt_ids)), "full fine-tuning coverage violated"
    if model.train_vision:
        vis_ids = {id(p) for p in model.vision.parameters()}
        assert vis_ids <= set(opt_ids), "stage 3: every online E/P parameter must be in the optimizer"
        assert not any(p.requires_grad for p in model.reference.parameters()) and not (
            {id(p) for p in model.reference.parameters()} & set(opt_ids)), "reference E0/P0 must stay frozen"
    else:
        assert not any(p.requires_grad for p in model.vision.parameters()), "V-JEPA must be frozen in stage 2"

    n_tub = d["num_frames"] // 2
    ds = Stage2Dataset(d["sources"], tok, d["num_frames"], ev_tubelets=n_tub // m.get("n_time_bins", 4),
                       max_len=d["max_len"], chat_kwargs=d.get("chat_kwargs"))
    bs = o["batch_per_gpu"]
    total = args.max_steps or o.get("max_steps") or len(PairBatchSampler(ds.df, bs, rank, world, seed)) * o.get("epochs", 1)
    log0(f"[data] rows={len(ds)} pairs={(ds.df.pair_id >= 0).sum() // 2} evidence_tokens={ds.n_ev} overview={ds.n_ov} total_steps={total}")

    state = {"step": 0, "epoch": 0, "batch_offset": 0, "skipped": {}}
    latest = f"{out}/ckpt/latest"
    if os.path.exists(latest):
        state = load_ckpt(open(latest).read().strip(), trainable(model), optim, distributed); log0(f"[resume] {state}")

    def loader(epoch, start):
        smp = PairBatchSampler(ds.df, bs, rank, world, seed, epoch, start)
        return iter(DataLoader(ds, batch_sampler=smp, num_workers=d.get("workers", 8), pin_memory=True,
                               collate_fn=functools.partial(collate2, pad_id=tok.eos_token_id if tok.pad_token_id is None else tok.pad_token_id)))

    it = loader(state["epoch"], state["batch_offset"]); model.train()
    nat = cfg.get("native")
    gen_cfg = None
    if model.train_vision and nat:
        nds = NativeClipDataset(nat["sources"], d["num_frames"], nat.get("fps", 4.0), seed)
        gens = make_mask_generators(d["num_frames"])
        def native_iter(epoch):
            smp = torch.utils.data.DistributedSampler(nds, world, rank, shuffle=True, seed=seed + epoch) if distributed else None
            return iter(DataLoader(nds, batch_size=nat.get("batch_per_gpu", bs), sampler=smp, shuffle=smp is None,
                                   num_workers=d.get("workers", 8), collate_fn=collate_native, drop_last=True))
        nit, nepoch = native_iter(0), 0
        log0(f"[native] clips={len(nds)} family_weight={nat.get('weight', 0.25)} lambda={nat.get('lambda', 0.5)}")
    gen_cfg = cfg.get("general")
    if gen_cfg:
        gds = Stage1Dataset(gen_cfg["sources"], tok, d["num_frames"], 4, d["max_len"], d.get("chat_kwargs"))
        gw = {s.get("name", s["manifest"]): s["weight"] for s in gen_cfg["sources"] if "weight" in s}
        def general_iter(epoch):
            smp = ResumableShardSampler(len(gds), rank, world, seed, epoch, 0, source_ids=gds.df.source.tolist() if gw else None,
                                        weights=gw or None)
            return iter(DataLoader(gds, batch_size=gen_cfg.get("batch_per_gpu", bs), sampler=smp, num_workers=d.get("workers", 8),
                                   collate_fn=functools.partial(collate, pad_id=tok.eos_token_id if tok.pad_token_id is None else tok.pad_token_id)))
        git, gepoch = general_iter(0), 0
        log0(f"[general] samples={len(gds)} family_weight={gen_cfg.get('weight', 0.5)} per_source={gds.df.source.value_counts().to_dict()}")
    logf = open(f"{out}/log.jsonl", "a") if rank == 0 else None
    gen = torch.Generator().manual_seed(seed + 1000 * rank)
    t0, seen = time.time(), 0
    while state["step"] < total:
        f = lr_factor(state["step"], total, o.get("warmup_frac", 0.03))
        for g in optim.param_groups:
            g["lr"] = g["base_lr"] * f
        frac = state["step"] / max(1, total * o.get("forcing_frac", 0.5))
        p_gold = max(o.get("p_gold_min", 0.0), 1.0 - frac)
        # batch family drawn from a step-seeded RNG: every rank runs the same modules this step (spec 14.5)
        u = np.random.default_rng((seed, state["step"])).random()
        w_nat = nat.get("weight", 0.25) if (model.train_vision and nat) else 0.0
        w_gen = gen_cfg.get("weight", 0.5) if gen_cfg else 0.0
        fam = "native" if u < w_nat else ("general" if u < w_nat + w_gen else "language")
        if fam == "general":
            while True:
                try:
                    gb = next(git)
                except StopIteration:
                    gepoch += 1; git = general_iter(gepoch); gb = next(git)
                if "video" in gb:
                    break
            res = model.forward_general(gb["video"].to(device, vis_dtype), gb["frame_times"].to(device), gb["input_ids"].to(device),
                                        gb["attention_mask"].to(device), gb["labels"].to(device), gb["vis_mask"].to(device),
                                        lambda_state=lam.get("state", 0.0), region_grid=gb["region_grid"])
            res["loss"].backward()
            gnorm = torch.nn.utils.clip_grad_norm_([p for g in optim.param_groups for p in g["params"]], o.get("clip", 1.0))
            gnorm = float(gnorm.full_tensor() if hasattr(gnorm, "full_tensor") else gnorm)
            optim.step(); optim.zero_grad(set_to_none=True)
            state["step"] += 1
            if rank == 0 and (state["step"] % run.get("log_every", 10) == 0):
                rec = {"step": state["step"], "family": "general", **{k: round(float(v), 5) for k, v in res.items()},
                       "grad_norm": round(gnorm, 4), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
                print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
            if state["step"] % run.get("save_every", 1000) == 0 or state["step"] == total:
                save_ckpt(out, state["step"], trainable(model), optim, state, distributed)
            continue
        if fam == "native":
            while True:
                try:
                    nb = next(nit)
                except StopIteration:
                    nepoch += 1; nit = native_iter(nepoch); nb = next(nit)
                if nb["video"] is not None:
                    break
            res = native_loss(model.vision, model.reference, nb["video"].to(device, vis_dtype), gens,
                              nat.get("lambda_ctx", 0.5), nat.get("alpha", 0.1))
            (nat.get("lambda", 0.5) * res["loss"]).backward()
            res = {"loss_native": res["loss"].detach(), "native_dense": res["dense"], "native_ref": res["ref"]}
            gnorm = torch.nn.utils.clip_grad_norm_([p for g in optim.param_groups for p in g["params"]], o.get("clip", 1.0))
            gnorm = float(gnorm.full_tensor() if hasattr(gnorm, "full_tensor") else gnorm)
            optim.step(); optim.zero_grad(set_to_none=True)
            state["step"] += 1
            if rank == 0 and (state["step"] % run.get("log_every", 10) == 0):
                rec = {"step": state["step"], "family": "native", **{k: round(float(v), 5) for k, v in res.items()},
                       "grad_norm": round(gnorm, 4), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
                print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
            if state["step"] % run.get("save_every", 1000) == 0 or state["step"] == total:
                save_ckpt(out, state["step"], trainable(model), optim, state, distributed)
            continue
        while True:
            try:
                batch = next(it)
            except StopIteration:
                state["epoch"] += 1; state["batch_offset"] = 0; it = loader(state["epoch"], 0); batch = next(it)
            state["batch_offset"] += 1
            for s in batch["skipped"]:
                state["skipped"][s] = state["skipped"].get(s, 0) + 1
            if "video" in batch:
                break
        res = model(to_device(batch, device, vis_dtype), p_gold=p_gold, lambdas=lam, generator=gen)
        res["loss"].backward()
        gnorm = torch.nn.utils.clip_grad_norm_([p for g in optim.param_groups for p in g["params"]], o.get("clip", 1.0))
        gnorm = float(gnorm.full_tensor() if hasattr(gnorm, "full_tensor") else gnorm)
        optim.step(); optim.zero_grad(set_to_none=True)
        state["step"] += 1; seen += batch["video"].shape[0]
        if rank == 0 and (state["step"] % run.get("log_every", 10) == 0 or state["step"] == 1):
            rec = {"step": state["step"], **{k: round(float(v), 5) for k, v in res.items()}, "p_gold": round(p_gold, 3),
                   "grad_norm": round(gnorm, 4), "samples_per_s_rank0": round(seen / (time.time() - t0), 2),
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
