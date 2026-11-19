"""Single-stage JEPA-LLM training: language alignment of V-JEPA 2.1 + an LLM in ONE run, keeping V-JEPA's strengths.

The three ways V-JEPA's strengths can be lost (spec 1.1) and what each arm does about them:
  forgetting      model.train_vision: none (frozen) | encoder | full, and loss.native (V-JEPA's own masked-prediction
                  objective against the frozen reference E0/P0, added to the same optimiser step every `every` steps)
  transmission    loss.lambda_state: the LLM must keep the (reference) V-JEPA state at the evidence positions
  use             loss.use: every `every` steps one extra batch of whole pairs (same question, different video, different
                  answer). Every arm trains on these pairs with answer CE; arms with lambda > 0 also get the evidence-swap
                  margin (the answer must become `margin` nats less likely under the partner's evidence)
The staged recipe of the original spec / V-JEPA 2's VidQA (projector first, then the LLM, then the encoder) becomes a
per-group learning-rate schedule inside one run: optim.llm_start_frac / vision_start_frac delay those groups.
Every arm draws the same language batches in the same order (same sampler seed), so arms differ only in what is
trained and which losses are on.
usage: torchrun --nproc_per_node N -m jepa_llm.training.train_single <config.yaml> [--device cpu] [--max-steps K]
"""
import argparse
import functools
import json
import pandas as pd
import math
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
from jepa_llm.data.stage1_dataset import PairSampler, ResumableShardSampler, Stage1Dataset, collate, partner_positions
from jepa_llm.losses.native import make_mask_generators, native_loss
from jepa_llm.models.stage1 import Stage1Model
from jepa_llm.models.vision_backbone import VJEPABackbone, build_vision
from jepa_llm.training.train_stage1 import (build_llm, compute_dtype, load_ckpt, log0, param_groups, save_ckpt,
                                          setup_dist, shard)


def schedule(step, start, total, warm, floor=0.1):
    """0 before `start`, linear warm-up over `warm` steps, cosine to `floor` at `total`."""
    if step < start:
        return 0.0
    t = step - start
    if t < warm:
        return (t + 1) / warm
    u = (t - warm) / max(1, total - start - warm)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, u)))


def lr_factor(step, start, total, warm, extend=None, floor=0.1):
    """(10-19) schedule() of the run, or - with optim.extend {from_step S, orig_max_steps T0} - a run lengthened while training: up to S
    the original schedule (total T0), after S a cosine from the factor it had reached at S down to `floor` at the new total, so the
    learning rate does not jump when max_steps is raised at a resume."""
    if not extend or step < int(extend["from_step"]):
        return schedule(step, start, int(extend["orig_max_steps"]) if extend else total, warm, floor)
    s0 = int(extend["from_step"]); v0 = schedule(s0, start, int(extend["orig_max_steps"]), warm, floor)
    u = (step - s0) / max(1, total - s0)
    return floor + (v0 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, u)))


def trainable(model):
    mods = {"core": model.core, "bridge": model.bridge}
    if model.retention is not None:
        mods["retention"] = model.retention
    if getattr(model, "predictor_heads", None) is not None:
        mods["predictor_heads"] = model.predictor_heads
    if getattr(model, "geo", None) is not None:
        mods["geo"] = model.geo
    if getattr(model, "critic", None) is not None:
        mods["critic"] = model.critic
    if getattr(model, "eq_heads", None) is not None:
        mods["eq_heads"] = model.eq_heads
    if getattr(model, "corr_head", None) is not None:
        mods["corr_head"] = model.corr_head
    if getattr(model, "dyn_head", None) is not None:
        mods["dyn_head"] = model.dyn_head
    if getattr(model, "next_head", None) is not None:
        mods["next_head"] = model.next_head
    if getattr(model, "tr_head", None) is not None:
        mods["tr_head"] = model.tr_head
    if getattr(model, "mem_bridge", None) is not None:
        mods["mem_bridge"] = model.mem_bridge
    if getattr(model, "tmem", None) is not None:
        mods["tmem"] = model.tmem
    if model.train_vision != "none":
        mods["vision"] = model.vision               # the reference copy is never saved: it is the official weights
    return torch.nn.ModuleDict(mods)


def check_params(model, optim, partial_llm=False):
    llm = list(model.core.llm.parameters())
    assert partial_llm or all(p.requires_grad for p in llm), "LLM must be fully trainable (no partial / LoRA training)"
    enc = [p for n, p in model.vision.named_parameters() if n.startswith("encoder.")]
    pred = [p for n, p in model.vision.named_parameters() if n.startswith("predictor.")]
    mode = model.train_vision
    assert all(p.requires_grad == (mode != "none") for p in enc), f"encoder trainability does not match {mode}"
    assert all(p.requires_grad == (mode == "full") for p in pred), f"predictor trainability does not match {mode}"
    ids = [id(p) for g in optim.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "a parameter appears twice in the optimizer"
    want = {id(p) for p in model.parameters() if p.requires_grad}
    assert want == set(ids), "optimizer parameters != trainable parameters"
    if model.reference is not None:
        assert not any(p.requires_grad for p in model.reference.parameters()), "reference E0/P0 must stay frozen"
        assert not ({id(p) for p in model.reference.parameters()} & set(ids)), "reference in the optimizer"
    return sum(p.numel() for p in llm), sum(p.numel() for g in optim.param_groups for p in g["params"])


def shard_vision(module, mesh, device, dtype):
    """FSDP2 for the trainable V-JEPA modules with their own compute dtype (fp32 or bf16), fp32 gradient reduction."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    if device.type == "cuda" or os.environ.get("JEPA_EMULATE_BF16") == "1":
        fully_shard(module, mesh=mesh, mp_policy=MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32))
    else:
        fully_shard(module, mesh=mesh, mp_policy=MixedPrecisionPolicy())


def nonfinite_grads(params_named):
    """names of parameters whose (local shard of the) gradient has NaN / inf"""
    bad = []
    for n, p in params_named:
        g = p.grad
        if g is None:
            continue
        g = g.to_local() if hasattr(g, "to_local") else g
        if not torch.isfinite(g).all():
            bad.append(n)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--device", default="cuda"); ap.add_argument("--max-steps", type=int)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run, o, m, d, lo = cfg["run"], cfg["optim"], cfg["model"], cfg["data"], cfg.get("loss", {})
    os.environ["JEPA_VIS_LAST"] = "1" if d.get("vis_last") else "0"      # question before the video (chat_format.place_vis); inherited by the data workers
    os.environ["JEPA_STRIP_THINK"] = "1" if d.get("strip_think") else "0"   # (11-05) backbone without thinking mode (Llama): CoT rows "reasoning\n</think>\n\nanswer" -> "reasoning\n\nanswer" (chat_format)
    os.environ["JEPA_SAMPLING"] = d.get("sampling", "paired")          # (10-21) "uniform": num_frames distinct times (video.paired_uniform_times)
    os.environ["JEPA_MEMORY"] = "1" if (cfg["model"].get("memory") is True or (isinstance(cfg["model"].get("memory"), dict) and "cells" not in cfg["model"]["memory"])) else "0"   # video read as memory by every layer: no video slots (chat_format)
    if cfg["model"].get("time_groups"):                                     # video slots per clip = groups x cells^2 (Stage1Dataset)
        os.environ["JEPA_TIME_GROUPS"] = str(int(cfg["model"]["time_groups"]))
    distributed, rank, world, device, mesh = setup_dist(args.device)
    seed = run.get("seed", 0)
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed)
    out = run["out_dir"]; os.makedirs(out, exist_ok=True)
    if rank == 0:
        json.dump(cfg, open(f"{out}/config.json", "w"), indent=1)
        # (10-22) provenance of this launch (one file per launch, never overwritten): code fingerprints, data manifests, versions, layout
        import hashlib, glob as _glob, platform, sys, transformers as _tf
        _md5 = lambda f: hashlib.md5(open(f, "rb").read()).hexdigest()
        _src = sorted(_glob.glob("/data/llama_vjepa_lm/training/jepa_llm/**/*.py", recursive=True) + _glob.glob("/data/llama_vjepa_lm/training/scripts/**/*.py", recursive=True)
                      + _glob.glob("/data/llama_vjepa_lm/training/scripts/launch/*.sh") + _glob.glob("/data/llama_vjepa_lm/training/configs/templates/*"))
        _man = [s_["manifest"] for s_ in cfg["data"]["sources"]] + [s_["manifest"] for s_ in cfg.get("loss", {}).get("image_data", {}).get("sources", [])] \
            + ([cfg["loss"]["textonly_data"]["parquet"]] if cfg.get("loss", {}).get("textonly_data") else [])
        prov = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "world_size": world, "host_rank0": platform.node(), "argv": sys.argv,
                "torch": torch.__version__, "transformers": _tf.__version__, "env": {k: v for k, v in os.environ.items() if k.startswith(("JEPA_", "NCCL_", "PYTORCH_"))},
                "code_md5": {f.replace("/data/llama_vjepa_lm/training/", ""): _md5(f) for f in _src},
                "manifests": {m_: {"bytes": os.path.getsize(m_), "mtime": os.path.getmtime(m_)} for m_ in _man if os.path.exists(m_)}}
        json.dump(prov, open(f"{out}/provenance_{time.strftime('%Y%m%d_%H%M%S')}.json", "w"), indent=1)

    # ---- model ----------------------------------------------------------------------------------------------------
    from jepa_llm.data.chat_format import setup_tokenizer
    tok = setup_tokenizer(AutoTokenizer.from_pretrained(m["llm"]), m)
    vision = build_vision(m["vision"]).eval()
    llm = build_llm(cfg, device)
    lam = float(lo.get("lambda_state", 0.0))
    lam_pred = float(lo.get("lambda_pred", 0.0))       # latent prediction: V-JEPA's objective inside the LLM
    mode = m.get("train_vision", "none")
    lam_geo = float(lo.get("lambda_geo", 0.0))          # world-model geometry at the read position (GeometryHead)
    geo_kw = dict(lo.get("geo", {})); geo_kw["n_tubelets"] = d["num_frames"] // 2
    lam_info = float(lo.get("lambda_info", 0.0))        # ipa / upm (Stage1Model.info)
    lam_ws = float(lo.get("lambda_ws", 0.0))            # V-JEPA's structure in the LLM's word space (Stage1Model.ws_loss)
    lam_corr = float(lo.get("lambda_corr", 0.0))        # the video positions' own task: which cell of the moment before (Stage1Model.corr_loss)
    graft = m.get("graft")                               # finished V-JEPA grafted whole: encoder + its own predictor (models/graft.py)
    extra_tub = (graft.get("total_tubelets", 10) - d["num_frames"] // 2) if graft else 0
    if graft:
        from jepa_llm.models.graft import GraftModel
        model = GraftModel(vision, llm, evidence_cells=d.get("cells", 7), bridge_hidden=m.get("bridge_hidden", 2048), seed=seed, **graft)
    else:
        model = Stage1Model(vision, llm, m.get("retention_layers") if (lam > 0 or lam_pred > 0) else None,
                            evidence_cells=d.get("cells", 4), bridge_hidden=m.get("bridge_hidden", 2048),
                            surprise_segments=m.get("surprise_segments", 0), train_vision=mode, latent_predict=lam_pred > 0,
                            predict_layers=m.get("predict_layers"), mask_scale=m.get("mask_scale", "small"),
                            geo_layers=m.get("geo_layers") if lam_geo > 0 else None, geo_kw=geo_kw,
                            info=m.get("info") if lam_info > 0 else None, info_kw=lo.get("info_kw"),
                            n_tubelets=d["num_frames"] // 2, spectral=m.get("spectral"),
                            eq_state=m.get("eq_state"), token_order=m.get("token_order", "tyx"), channel=m.get("channel"),
                            expert=m.get("expert"), word_drop=float(m.get("word_drop", 0.0)),
                            corr=m.get("corr") if lam_corr > 0 else None, wstruct=m.get("wstruct") if lam_ws > 0 else None,
                            nextp=m.get("nextp") if float(lo.get("lambda_next", 0.0)) > 0 else None,
                            codes=dict(m["codes"], tok_ids=[tok.encode(str(i), add_special_tokens=False)[0] for i in range(256)]) if m.get("codes") else None,
                            textread=m.get("textread"), depw=m.get("depw"), vcode=m.get("vcode"), vtie=m.get("vtie"), entry_layer=m.get("entry_layer"), memory=m.get("memory", False),
                            dyncon=m.get("dyncon") if float(lo.get("lambda_dyn", 0.0)) > 0 else None)
    # (11-16) MoE backbone: router load-balancing loss weight (default: the backbone's own router_aux_loss_coef; 0 = off; dense: always off)
    if getattr(llm.config, "num_experts", None):
        model.core.router_coef = float(lo.get("lambda_router", getattr(llm.config, "router_aux_loss_coef", 0.0) or 0.0))
        log0(f"[moe] {llm.config.num_experts} experts, top-{llm.config.num_experts_per_tok}, experts kernel {llm.config._experts_implementation}, router loss weight {model.core.router_coef}")
    model.causal_win = int(m.get("causal_window", 0))   # V-JEPA tokens of moment t encoded from the frames up to t only
    model.ans_sd = float(lo.get("ans_sd", 0.0)); model.ans_ls = float(lo.get("ans_ls", 0.0))
    model.static_shrink = bool(m.get("static_shrink", False))
    model.vjepa_rsc = bool(m.get("vjepa_rsc", False))
    model.fw_online = bool((m.get("dyncon") or {}).get("fw_online", False))   # (11-03) dynamics loss in the answer-gradient metric
    model.flow_clean = bool(m.get("flow_clean", False))
    model.flow_global = bool(m.get("flow_global", False))   # (11-03) frame-level flow only   # (11-03) confidence-weighted, camera-relative flow
    if m.get("flow_lags"): model.flow_lags = tuple(int(x) for x in m["flow_lags"])   # (11-02 night) correspondence to t-1, t-2, ...
    if m.get("flow_feats"): model.enable_flow(9 * len(getattr(model, "flow_lags", (1,))))   # (11-02) V-JEPA latent flow per cell into the visual tokens (EvidenceBridge.flow_proj)   # (11-02) answer-dependence pull on V-JEPA's input in training (Stage1Model.forward / _rsc_hook)   # (11-02) V-JEPA appearance shortcut weakened per row in training (Stage1Model.forward)   # (11-02) answer-confidence regularisers (Stage1Model.forward)
    model.time_groups = int(m.get("time_groups", 0))     # tubelets averaged in this many groups (time traded for a finer grid)
    model.chunk_frames = int(m.get("chunk_frames", 0))   # long clips encoded in chunks of this many frames
    model.vjepa_last = m.get("vjepa_levels") == "last"   # V-JEPA final output only
    lam_swap, swap_margin = float(lo.get("lambda_swap", 0.0)), float(lo.get("swap_margin", 0.2))
    lam_eq = float(lo.get("lambda_eq", 0.0))            # preservation in V-JEPA's geometry (EqualisedStateHeads); 0 = watch only
    lam_vc = float(lo.get("lambda_vc", 0.0))            # collapse-free alignment (EvidenceBridge.collapse_stats)
    model.bridge.vc_weight = lam_vc
    if m.get("temporal_memory"):                         # (10-17) Gated DeltaNet memory over K V-JEPA clips (models/temporal_memory.py)
        model.enable_tmem(m["temporal_memory"])
        os.environ["JEPA_TMEM_Q"] = str(int(m["temporal_memory"].get("queries", 128)))
        os.environ["JEPA_TMEM_CLIPS"] = str(float(m["temporal_memory"].get("clip_s", 8.0)))   # seconds per memory clip (16 frames)
        if d.get("memory_store"):
            os.environ["JEPA_TMEM_STORE"] = d["memory_store"]
    latent_first = float(lo.get("latent_first_frac", 0.0))     # text losses off for the first fraction of steps
    ns_cfg, ns = o.get("nullspace"), None         # {"path", "energy"}: LLM updates outside its language directions (training/nullspace.py)
    if ns_cfg:
        from jepa_llm.training.nullspace import LanguageNullSpace, build_bases, freeze_rest
        n_frozen = freeze_rest(model.core.llm)
        ns_bases = build_bases(model.core.llm, ns_cfg)          # before sharding: reads the norm gains
        log0(f"[nullspace] LLM parameters that are not projected matrices frozen: {n_frozen / 1e6:.1f}M")
    margin_unit = lo.get("margin_unit", "token")              # "sum": margins on log p(answer | video) as a whole
    if m.get("init_from") and not os.path.exists(f"{out}/ckpt/latest"):
        # branch from a shared checkpoint (e.g. the frozen-encoder alignment phase): LLM + bridge weights; V-JEPA starts
        # from the official weights (it was frozen there); new heads (retention) and the optimizer start fresh
        from jepa_llm.eval.runner import load_weights
        init = {"core": model.core, "bridge": model.bridge}
        if m.get("init_corr_head") and getattr(model, "corr_head", None) is not None:
            init["corr_head"] = model.corr_head              # the read-out of the video positions' task continues from stage 1
        if m.get("init_tr_head") and getattr(model, "tr_head", None) is not None:
            init["tr_head"] = model.tr_head                  # the stage-1 read-out at the text positions continues
        if m.get("init_next_head") and getattr(model, "next_head", None) is not None:
            init["next_head"] = model.next_head              # the next-scene read-out continues from stage 1
        if m.get("init_dyn_head") and getattr(model, "dyn_head", None) is not None:
            init["dyn_head"] = model.dyn_head               # (11-01) the dynamics loss's decoder continues (rec2 continuation)
        load_weights(torch.nn.ModuleDict(init), m["init_from"])
        log0(f"[init_from] {m['init_from']}: {' + '.join(init)} loaded")
    gpu_like = device.type == "cuda"
    if run.get("activation_checkpointing", gpu_like):
        model.core.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        ck = int(run.get("checkpoint_every", 1))          # (10-21) recompute only every k-th decoder layer (memory vs speed)
        for i, layer in enumerate(model.core.llm.model.layers):
            layer.gradient_checkpointing = i % ck == 0
        if run.get("ckpt_mode") == "mlp":
            # (10-29) selective recompute: only each layer's MLP (the 3 x 12288-wide intermediates, most of a layer's activation memory) is
            # recomputed in backward; attention activations are kept, so the attention (its projections + SDPA) is not run twice. No
            # layer-level checkpointing then. The attention may be compiled (model.attn_compile): it is never inside a checkpoint, so its
            # forward and backward use the same compiled graph (layer compile + layer checkpoint broke: the recompute runs eager)
            for layer in model.core.llm.model.layers:
                layer.gradient_checkpointing = False
                _f = layer.mlp.forward
                layer.mlp.forward = (lambda x, _f=_f: torch.utils.checkpoint.checkpoint(_f, x, use_reentrant=False))
        if run.get("ckpt_mode") == "sac":
            # (10-29) selective activation checkpointing (torch SAC) on every layer: the outputs of the chosen expensive ops are SAVED in the
            # forward and only the rest is recomputed in backward. sac_save: "attn" = all matmuls except the two 12288-wide MLP up/gate ones
            # + the SDPA output (recompute: MLP up/gate + elementwise); "mm" = every matmul + SDPA (recompute: elementwise only). Same
            # math as plain checkpointing. The replay pass (rows up to 16k tokens) keeps plain full-layer recompute (memory), see below.
            import functools as _ft   # (not 'functools': a local import would shadow the module for the whole main())
            from torch.utils.checkpoint import checkpoint as _ckp, create_selective_checkpoint_contexts, CheckpointPolicy
            _A = torch.ops.aten; _inter = int(model.core.llm.config.intermediate_size); _how = run.get("sac_save", "attn")
            _SDPA = {getattr(_A, n).default for n in ("_scaled_dot_product_flash_attention", "_scaled_dot_product_efficient_attention",
                                                       "_scaled_dot_product_cudnn_attention") if hasattr(_A, n)}
            _MM = {_A.mm.default, _A.addmm.default, _A.bmm.default}
            def _policy(ctx, op, *a, **k):
                if op in _SDPA:
                    return CheckpointPolicy.MUST_SAVE
                if op in _MM:
                    if _how == "attn" and a[-1].shape[-1] == _inter:      # MLP gate / up projection (weight^T [hidden, intermediate])
                        return CheckpointPolicy.PREFER_RECOMPUTE
                    return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.PREFER_RECOMPUTE
            _sac = _ft.partial(_ckp, use_reentrant=False, context_fn=_ft.partial(create_selective_checkpoint_contexts, _policy))
            for layer in model.core.llm.model.layers:
                layer.gradient_checkpointing = True
                layer._gc_plain = layer._gradient_checkpointing_func; layer._gc_sac = _sac; layer._gradient_checkpointing_func = _sac
            log0(f"[ckpt] selective activation checkpointing (save: {_how}) on all {len(model.core.llm.model.layers)} LLM layers; replay pass: full recompute")
        if run.get("ckpt_mode") == "compile_ac":
            # (10-29) compile + activation checkpointing the supported way: each layer's forward = torch.compile(checkpoint(forward)),
            # so dynamo traces the checkpoint itself and the recompute happens inside the compiled graph (compiling the layer and letting
            # HF checkpoint it from outside recomputed eagerly and broke: CheckpointError). Fuses the elementwise ops (RMSNorm, RoPE, SiLU,
            # casts: ~1/3 of the eager GPU time in the nx16_Q8 profile). Every layer recomputed as before (same memory as checkpoint_every 1).
            from torch.utils.checkpoint import checkpoint as _ckp2
            for layer in model.core.llm.model.layers:
                layer.gradient_checkpointing = False
                _o = layer.forward
                layer.forward = torch.compile(lambda *a, _o=_o, **k: _ckp2(_o, *a, use_reentrant=False, **k), dynamic=True)
            torch._dynamo.config.recompile_limit = max(int(torch._dynamo.config.recompile_limit), 64)
            log0("[ckpt] compile_ac: every LLM layer = torch.compile(checkpoint(layer.forward))")
        log0(f"[ckpt] activation checkpointing on {sum(l.gradient_checkpointing for l in model.core.llm.model.layers)} of {len(model.core.llm.model.layers)} LLM layers")
        if mode != "none":
            model.vision.encoder.use_activation_checkpointing = True
            model.vision.predictor.use_activation_checkpointing = True
    # model.vision_dtype: fp32 runs V-JEPA (online and reference) in fp32 with TF32 matmuls. bf16 backward through
    # V-JEPA 2.1's RoPE attention gave NaN encoder gradients in 8-15 of 40 native-loss steps (block 6 attention; fp32
    # attention alone did not fix it); fp32 + TF32 gave 0/40 at ~1.3x the bf16 time. Same setting for every arm.
    vis_dtype = torch.float32 if m.get("vision_dtype", "bf16") == "fp32" else compute_dtype(device)
    if vis_dtype == torch.float32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    if mode == "none":
        model.vision.to(device=device, dtype=vis_dtype)
        if m.get("vision_compile"):                                # (10-17) frozen encoder compiled: 1.38x (logs/zz_vj_compile.py; outputs
            model.vision.encoder = torch.compile(model.vision.encoder, dynamic=True)   # differ from eager at bf16 rounding, rel 1.7e-2)
    else:
        model.vision.to(device)                                    # fp32 masters; FSDP computes in vis_dtype
        model.reference.to(device=device, dtype=vis_dtype)
        if mode == "encoder":
            model.vision.predictor.to(dtype=vis_dtype)              # frozen, unused by the encoder-only arm
    heads = (model.core, model.bridge) + ((model.retention,) if model.retention is not None else ()) + \
            ((model.predictor_heads,) if model.predictor_heads is not None else ()) + \
            ((model.geo,) if model.geo is not None else ()) + ((model.critic,) if model.critic is not None else ()) + \
            ((model.eq_heads,) if getattr(model, "eq_heads", None) is not None else ()) + \
            ((model.corr_head,) if getattr(model, "corr_head", None) is not None else ()) + \
            ((model.dyn_head,) if getattr(model, "dyn_head", None) is not None else ()) + \
            ((model.next_head,) if getattr(model, "next_head", None) is not None else ()) + \
            ((model.tr_head,) if getattr(model, "tr_head", None) is not None else ()) + \
            ((model.mem_bridge,) if getattr(model, "mem_bridge", None) is not None else ()) + \
            ((model.tmem,) if getattr(model, "tmem", None) is not None else ())
    # (11-16) an LLM whose fp32 copy does not fit on one GPU (Qwen3-30B-A3B: 122 GB) stays on the CPU layer by layer: everything but the
    # decoder layers goes to the GPU now, each layer is moved by fully_shard below (FSDP2 moves a CPU module's states to the mesh device
    # before sharding: one layer at a time). run.llm_shard_from_cpu overrides the size test; the result is the same sharded model.
    _fp32 = sum(p.numel() for p in model.core.parameters()) * 4
    _from_cpu = distributed and device.type == "cuda" and bool(run.get("llm_shard_from_cpu", _fp32 > 0.5 * torch.cuda.get_device_properties(device).total_memory))
    for mod in heads:
        if mod is model.core and _from_cpu:
            _lyr = model.core.llm.model.layers; model.core.llm.model.layers = torch.nn.ModuleList()
            mod.to(device); model.core.llm.model.layers = _lyr
            log0(f"[shard] LLM decoder layers ({_fp32 / 2**30:.0f} GB fp32) moved to the GPU layer by layer by fully_shard")
        else:
            mod.to(device)
    if model.critic is not None and model.info == "ipa":
        # the neutral instruction: identical text for every clip, so only the video can make the LLM states differ
        from jepa_llm.data.chat_format import VIS, build_generation_prompt
        npv = (d["num_frames"] // 2) * d.get("cells", 4) ** 2
        pr = build_generation_prompt(tok, f"{VIS}\n{lo.get('info_prompt', 'Describe the video.')}", npv, d.get("chat_kwargs"))
        ids = torch.tensor([pr["input_ids"]], device=device); vm = torch.tensor([pr["vis_mask"]], device=device).bool()
        tm = torch.zeros_like(vm); tm[0, int(vm[0].nonzero().max()) + 1:] = True
        model.neutral = (ids, vm, tm)
    if distributed:
        rs = bool(run.get("fsdp_reshard", True))          # false: small LLMs keep gathered params (speed, more memory)
        if m.get("llm_compile") or m.get("attn_compile"):     # (10-21) every layer instance recompiles the shared forward: the default limit (8) left 28 of 36 eager
            torch._dynamo.config.recompile_limit = int(m.get("llm_recompile_limit", 8 if m.get("llm_compile") else 256))
        _keep = int(run.get("fsdp_keep_last", 0))       # (11-17) the last K decoder layers keep their gathered params from forward to backward
        _nl = len(model.core.llm.model.layers)           # (no re-gather in backward for them; K x the bf16 layer size of memory, mx30a_Q)
        for _li, layer in enumerate(model.core.llm.model.layers):
            shard(layer, mesh, device, rs and _li < _nl - _keep)
            if m.get("llm_compile"):                    # (10-21) each decoder layer compiled after sharding (dynamic lengths)
                layer.compile(dynamic=True)
            elif m.get("attn_compile"):                 # (10-29) only the attention block compiled (see ckpt_mode mlp)
                layer.self_attn.compile(dynamic=True)
        for mod in heads:
            shard(mod, mesh, device, rs)
        vshard = functools.partial(shard_vision, mesh=mesh, device=device, dtype=vis_dtype)
        if mode != "none":
            for blk in model.vision.encoder.blocks:
                vshard(blk)
            vshard(model.vision.encoder)
        if mode == "full":
            for blk in model.vision.predictor.predictor_blocks:
                vshard(blk)
            vshard(model.vision.predictor)
    optim = torch.optim.AdamW(param_groups(model, cfg), betas=(0.9, 0.95), eps=1e-8, fused=device.type == "cuda")
    n_llm, n_train = check_params(model, optim, partial_llm=bool(ns_cfg) or bool(m.get("expert")))
    # (10-17) precision audit, logged once after the first optimizer step: fp32 master weights and optimizer states,
    # bf16 compute (dtype of an LLM layer's output and of the bridge output during the forward)
    _prec = {}
    def _prec_hook(name):
        def f(m_, i_, o_):                       # returns None: a forward hook that returns a value replaces the module output
            _prec.setdefault(name, str((o_[0] if isinstance(o_, tuple) else o_).dtype))
        return f
    _prec_h = [model.core.llm.model.layers[0].register_forward_hook(_prec_hook("llm_layer_out")),
               model.bridge.register_forward_hook(_prec_hook("bridge_out"))]
    if ns_cfg:
        ns = LanguageNullSpace(model.core.llm, ns_bases, device)
        log0(f"[nullspace] {len(ns.items)} matrices, energy {ns_cfg.get('energy', 0.99)}, protected share of the directions {ns.summary()}")
    log0(f"[init] world={world} train_vision={mode} llm={n_llm/1e6:.1f}M trainable={n_train/1e6:.1f}M "
         f"vision={m['vision']} lambda_state={lam} lambda_pred={lam_pred} lambda_swap={lam_swap} lambda_geo={lam_geo} info={m.get('info') if lam_info > 0 else None} lambda_info={lam_info} latent_first={latent_first} surprise={m.get('surprise_segments', 0)} groups="
         f"{[(g['name'], g['base_lr'], len(g['params'])) for g in optim.param_groups]}")

    # ---- data -----------------------------------------------------------------------------------------------------
    ds = Stage1Dataset(d["sources"], tok, d["num_frames"], d.get("cells", 4), d["max_len"], d.get("chat_kwargs"),
                       d.get("pair_gap_sec", 0.25), extra_tubelets=extra_tub, narrative=d.get("narrative"), augment=d.get("augment"))
    if float(lo.get("lambda_concept", 0.0)) > 0:         # (10-25) concept readout: option groups of every multiple-choice turn
        ds.concept = lo.get("concept") or {}
    bs, accum = o["batch_per_gpu"], o.get("grad_accum", 1)
    total = args.max_steps or o["max_steps"]
    warm = max(1, int(total * o.get("warmup_frac", 0.03)))
    starts = {"new": 0, "llm": int(total * o.get("llm_start_frac", 0.0)),
              "enc": int(total * o.get("vision_start_frac", 0.0)), "pred": int(total * o.get("vision_start_frac", 0.0))}
    w = {s.get("name", s["manifest"]): s["weight"] for s in d["sources"] if "weight" in s}
    log0(f"[data] samples={len(ds)} evidence_tokens={ds.n_vis} total_steps={total} global_batch={bs * accum * world} "
         f"starts={starts} per_source={ds.df.source.value_counts().to_dict()} weights={w}")

    nat = lo.get("native")
    use_native = bool(nat) and mode == "full"
    assert not nat or mode == "full", "the native loss trains the predictor too: needs train_vision: full"
    if use_native:
        nds = NativeClipDataset(nat["sources"], d["num_frames"], nat.get("fps", 4.0), seed)
        gens = make_mask_generators(d["num_frames"])

        def native_iter(epoch):
            smp = torch.utils.data.DistributedSampler(nds, world, rank, shuffle=True, seed=seed + epoch) if distributed else None
            return iter(DataLoader(nds, batch_size=nat.get("batch_per_gpu", bs), sampler=smp, shuffle=smp is None,
                                   num_workers=2, collate_fn=collate_native, drop_last=True))
        nit, nepoch = native_iter(0), 0
        log0(f"[native] clips={len(nds)} every={nat.get('every', 4)} lambda={nat.get('lambda', 1.0)}")

    use = lo.get("use")
    if use:
        pds = Stage1Dataset(use["sources"], tok, d["num_frames"], d.get("cells", 4), d["max_len"], d.get("chat_kwargs"),
                            d.get("pair_gap_sec", 0.25), extra_tubelets=extra_tub)
        pair_ids = pds.df.pair_id.to_numpy()
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        def pair_iter(epoch, start=0):
            smp = PairSampler(pds.df, use.get("pairs_per_gpu", 1), rank, world, seed, epoch, start)
            return iter(DataLoader(pds, batch_sampler=smp, num_workers=2, collate_fn=functools.partial(collate, pad_id=pad_id)))
        log0(f"[use] pairs={int((pair_ids >= 0).sum() // 2)} every={use.get('every', 4)} lambda={use.get('lambda', 0.0)} "
             f"margin={use.get('margin', 0.5)} per_source={pds.df.source.value_counts().to_dict()}")

    txd = lo.get("textonly_data")          # {"parquet" (input_ids, labels), "per_gpu", "weight"}: plain text batches, language kept
    if txd:
        _tx = pd.read_parquet(txd["parquet"]); tx_ids, tx_lab = list(_tx.input_ids), list(_tx.labels); del _tx
        if int(txd.get("max_len", 0)):              # (10-21) rows whose answer starts after the cut had no labelled token (CE over 0 tokens: NaN, 10.8%)
            _ml = int(txd["max_len"]); _keep = [i for i in range(len(tx_lab)) if any(int(x) != -100 for x in list(tx_lab[i])[:_ml])]
            tx_ids, tx_lab = [tx_ids[i] for i in _keep], [tx_lab[i] for i in _keep]
        _txg = {}
        def tx_batch():
            if "g" not in _txg:                     # (10-21) seeded by the step it starts from: a resume does not replay the same rows
                _txg["g"] = np.random.default_rng(seed * 1000 + rank + 100003 * int(state.get("step", 0)))
            txg = _txg["g"]
            j = txg.integers(0, len(tx_ids), txd.get("per_gpu", 4)); L_ = max(len(tx_ids[i]) for i in j); pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            ids = torch.full((len(j), L_), pad, dtype=torch.long); lab = torch.full((len(j), L_), -100, dtype=torch.long); am = torch.zeros((len(j), L_), dtype=torch.long)
            for r_, i in enumerate(j): n_ = len(tx_ids[i]); ids[r_, :n_] = torch.tensor(tx_ids[i]); lab[r_, :n_] = torch.tensor(tx_lab[i]); am[r_, :n_] = 1
            ml = int(txd.get("max_len", 0))             # (10-20) long UltraChat dialogues cost ~20% step time: cut
            if ml: ids, lab, am = ids[:, :ml], lab[:, :ml], am[:, :ml]
            return ids, am, lab
        log0(f"[textonly_data] rows={len(tx_ids)} per_gpu={txd.get('per_gpu', 4)} weight={txd.get('weight', 1.0)}")
    imd = lo.get("image_data")             # (10-21) {"sources", "per_gpu", "every", "size", "pool", "weight"}: still images (OCR / scene text)
    if imd:                                 #   at high resolution through Stage1Model.forward_images, one extra batch per `every` steps
        from jepa_llm.data.stage1_dataset import ImageQADataset
        img_ds = ImageQADataset(imd["sources"], tok, imd.get("size", 1152), imd.get("pool", 3), d["max_len"], d.get("chat_kwargs"))
        _img = {}
        def img_batch():
            if "it" not in _img:
                g_ = torch.Generator(); g_.manual_seed(seed * 1000 + rank + 7919 * int(state.get("step", 0)))
                n_draw = int(imd.get("per_gpu", 1)) * (-(-(total - int(state.get("step", 0))) // int(imd.get("every", 1))) + 1000)   # draws left in the run
                _img["it"] = iter(DataLoader(img_ds, batch_size=int(imd.get("per_gpu", 1)), num_workers=int(imd.get("workers", 4)),
                                             sampler=torch.utils.data.WeightedRandomSampler(img_ds.weights_by_source(imd["sources"]), n_draw, generator=g_),
                                             collate_fn=functools.partial(collate, pad_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)))
            while True:
                ib = next(_img["it"])
                if "video" in ib:
                    return ib
        log0(f"[image_data] rows={len(img_ds)} slots={img_ds.n_vis} size={img_ds.size} per_gpu={imd.get('per_gpu', 1)} every={imd.get('every', 1)} "
             f"per_source={img_ds.df.source.value_counts().to_dict()}")
    trd = lo.get("textread_data")          # {"sources", "per_gpu", "weight"}: the stage-1 task kept during stage 2 (10-14, plan B)
    if trd:
        tds = Stage1Dataset(trd["sources"], tok, d["num_frames"], d.get("cells", 4), d.get("textread_max_len", 512), d.get("chat_kwargs"), d.get("pair_gap_sec", 0.25))
        tpad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        def tr_iter(k):
            g = torch.Generator().manual_seed(seed * 1000 + rank * 17 + k)
            return iter(DataLoader(tds, batch_size=trd.get("per_gpu", 4), sampler=torch.utils.data.RandomSampler(tds, replacement=True, num_samples=10 ** 7, generator=g),
                                   num_workers=2, collate_fn=functools.partial(collate, pad_id=tpad)))
        tit, tk = tr_iter(0), 0
        log0(f"[textread_data] rows={len(tds)} per_gpu={trd.get('per_gpu', 4)} weight={trd.get('weight', 0.5)}")
    icd = lo.get("ic_data")                # {"sources", "per_gpu", "weight"}: in-context V-JEPA neighbours, one extra batch per step (10-15)
    if icd:
        ids_ = Stage1Dataset(icd["sources"], tok, d["num_frames"], d.get("cells", 4), d["max_len"], d.get("chat_kwargs"), d.get("pair_gap_sec", 0.25))
        ipad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        def ic_iter(k):
            g = torch.Generator().manual_seed(seed * 1000 + rank * 31 + k)
            return iter(DataLoader(ids_, batch_size=icd.get("per_gpu", 1), sampler=torch.utils.data.WeightedRandomSampler(ids_.weights_by_source(icd["sources"]), 10 ** 7, generator=g),
                                   num_workers=3, collate_fn=functools.partial(collate, pad_id=ipad)))
        iit = ic_iter(0)
        log0(f"[ic_data] rows={len(ids_)} per_gpu={icd.get('per_gpu', 1)} weight={icd.get('weight', 1.0)} per_source={ids_.df.source.value_counts().to_dict()}")
    state = {"step": 0, "epoch": 0, "sample_offset": 0, "skipped": {}, "native_epoch": 0, "pair_epoch": 0, "pair_batch": 0}
    latest = f"{out}/ckpt/latest"
    if os.path.exists(latest):
        state = load_ckpt(open(latest).read().strip(), trainable(model), optim, distributed)
        log0(f"[resume] {state}")
        if use_native:
            nepoch = state.get("native_epoch", 0); nit = native_iter(nepoch)

    # (10-20) data scheduler (data.schedule): every `every` steps the mix moves toward the sources the model is still learning from
    # (DynamixSFT-style: p = (1-gamma) * norm(p0 * exp(beta * Q)) + gamma * p0, Q = EMA of the relative drop of each source's own
    # training loss, min-max normalised), p0 = the configured mix x a video-dependence factor (gain_file: per-source own-vs-swapped
    # loss gap, measured on the evaluation GPU, clip 0.5..2 x its median) x an annealing factor for listed high-quality sources late in
    # training. Signals come from the training forward itself (no extra pass); the loader is rebuilt, training does not stop.
    sched = d.get("schedule")
    if sched and state.get("sched_qmode", "rel") != str(sched.get("qmode", "rel")):   # progress measure changed on resume: start the mix over
        state.pop("sched_Q", None); state.pop("sched_w", None); state["sched_qmode"] = str(sched.get("qmode", "rel"))
    cur_w = dict(state.get("sched_w") or w) if sched else w
    src_names = sorted(w) if w else []; src_ix = {k: i for i, k in enumerate(src_names)}
    ana = bool(run.get("analysis", True)); ana_every = int(run.get("analysis_every", 500))   # (10-22) per-source loss / counts / passes
    row_src = np.array([src_ix.get(x, -1) for x in ds.df.source.tolist()]) if (sched or ana) else None
    # (11-01 v3) dynamics counterfactual by substitution (_CfUse): code 1 = order-dependent multiple-choice rows (dyncon "sources"; every
    # answer of the row has listed options; log chance = -log(fewest options)), code 2 = appearance rows (dyncon "keep_sources"). Each step
    # draws code-1 rows with probability p1 and code-2 rows with p2; the others enter normally.
    _dc = m.get("dyncon") if float(lo.get("lambda_dyn", 0.0)) > 0 else None
    cf_code = cf_logc_all = None
    if _dc and _dc.get("mode") == "cf":
        import re as _re
        _src = ds.df.source.astype(str).tolist(); cf_code = np.zeros(len(_src), np.int8); cf_logc_all = np.zeros(len(_src), np.float32)
        _o1, _o2 = tuple(_dc["sources"]), tuple(_dc["keep_sources"])
        _kind = {nm_: (1 if any(k_ in nm_ for k_ in _o1) else 2 if any(k_ in nm_ for k_ in _o2) else 0) for nm_ in set(_src)}
        _k = np.array([_kind[nm_] for nm_ in _src]); cf_code[_k == 2] = 2
        for i_ in np.nonzero(_k == 1)[0]:
            us_ = [x_["content"] for x_ in json.loads(ds.df.conversations.iat[i_]) if x_["role"] == "user"]
            no_ = [len(_re.findall(r"(?m)^[A-J]\. ", u_)) for u_ in us_]
            if us_ and min(no_) >= 2: cf_code[i_] = 1; cf_logc_all[i_] = -math.log(min(no_))
        _p = {1: float(_dc.get("p1", 0.2)), 2: float(_dc.get("p2", 0.5))}
        _cfrng = np.random.default_rng(1101 + rank)
        def cf_draw(ix_):
            c_ = cf_code[ix_].astype(np.int64); u_ = _cfrng.random(len(ix_))
            c_ = np.where(((c_ == 1) & (u_ < _p[1])) | ((c_ == 2) & (u_ < _p[2])), c_, 0)
            return {"cf_rows": torch.as_tensor(c_, device=device), "cf_logc": torch.as_tensor(cf_logc_all[ix_], device=device)}
        log0(f"[cf] code 1 rows {int((cf_code == 1).sum())} (order-dependent MC), code 2 rows {int((cf_code == 2).sum())} (appearance), of {len(_src)}; "
             f"p1 {_p[1]} p2 {_p[2]}; code-1 sources {sorted(set(np.asarray(_src)[cf_code == 1].tolist()))}")
    src_acc = torch.zeros(2, max(1, len(src_names)), device=device, dtype=torch.float64)
    ana_acc = torch.zeros(2, max(1, len(src_names)), device=device, dtype=torch.float64)
    ana_rows = np.bincount(row_src[row_src >= 0], minlength=len(src_names)).astype(float) if ana and row_src is not None else None
    if ana and "ana_seen" not in state:     # samples seen per source; a run started without it estimates the past from the mix
        zw_ = sum(w.values()) if w else 1.0; state["ana_seen"] = {k: w[k] / zw_ * state["step"] * bs * accum * world for k in src_names}
    # (10-20) max_passes: no source is shown more than `max_passes` times over the whole run (repeats beyond ~4 add little; small human
    # sets were heading for 6-7 passes at 128k). Samples seen per source are counted from the scheduler windows; a run that starts the
    # count mid-way estimates the past from the base mix. A source at its cap gives its share to the others in proportion.
    src_rows = np.bincount(row_src[row_src >= 0], minlength=len(src_names)).astype(float) if sched else None
    gb = bs * accum * world
    if sched and sched.get("max_passes") and "sched_seen" not in state:
        zw = sum(w.values()); state["sched_seen"] = {k: w[k] / zw * state["step"] * gb for k in src_names}
    def cap_mix(p, step_):
        mp = sched.get("max_passes")
        if not mp: return p, []
        rem = max(1.0, (total - step_) * gb); seen = state["sched_seen"]
        m = {k: max(float(mp) * src_rows[src_ix[k]] - seen.get(k, 0.0), 0.0) / rem for k in src_names}
        p = dict(p); fixed = set()
        for _ in range(len(src_names)):
            over = [k for k in src_names if k not in fixed and p[k] > m[k]]
            if not over: break
            for k in over: p[k] = m[k]; fixed.add(k)
            free = [k for k in src_names if k not in fixed]; left = 1.0 - sum(p[k] for k in fixed); zf = sum(p[k] for k in free)
            if not free or zf <= 0: break
            for k in free: p[k] = p[k] * left / zf
        return p, sorted(fixed)
    if sched and sched.get("max_passes"):
        cur_w, _ = cap_mix({k: v / sum(cur_w.values()) for k, v in cur_w.items()}, state["step"])
    def sched_weights(step_):
        import json as _json, os as _os
        prev = state.get("sched_prevL"); acc = src_acc.clone()
        if distributed: dist.all_reduce(acc)
        L = {src_names[i]: float(acc[0, i] / acc[1, i]) for i in range(len(src_names)) if acc[1, i] >= int(sched.get("min_count", 8))}
        Q = dict(state.get("sched_Q") or {}); ema = float(sched.get("ema", 0.7))
        # (10-20) qmode abs: drop in loss over the mean loss of the sources (a relative drop makes near-zero losses look like fast learning)
        qmode = str(sched.get("qmode", "rel")); scale = float(np.mean([prev[k] for k in L if prev and k in prev])) if prev and qmode == "abs" else 1.0
        for k, v in L.items():
            if prev and k in prev and prev[k] > 0:
                Q[k] = ema * Q.get(k, 0.0) + (1 - ema) * (prev[k] - v) / (max(scale, 1e-6) if qmode == "abs" else prev[k])
        if "sched_seen" in state:
            for i, k in enumerate(src_names): state["sched_seen"][k] = state["sched_seen"].get(k, 0.0) + float(acc[1, i])
        state["sched_prevL"] = L; state["sched_Q"] = Q; src_acc.zero_()
        g = {}
        gf = sched.get("gain_file")
        if gf and _os.path.exists(gf):
            G = {r["source"]: max(r["swap"] - r["own"], 0.0) for r in map(_json.loads, open(gf)) if r.get("n", 0) >= 16}
            if G:
                med = float(np.median(list(G.values())))
                g = {k: float(np.clip(v / max(med, 1e-6), 0.5, 2.0)) for k, v in G.items()}
        ann = sched.get("anneal") or {}
        late = step_ >= float(ann.get("from", 1.1)) * total
        p0 = {k: w[k] * g.get(k, 1.0) * (float(ann.get("factor", 1.0)) if late and k in ann.get("sources", []) else 1.0) for k in src_names}
        z = sum(p0.values()); p0 = {k: v / z for k, v in p0.items()}
        qv = np.array([Q.get(k, 0.0) for k in src_names]); qn = (qv - qv.min()) / max(qv.max() - qv.min(), 1e-9) if len(qv) else qv
        beta, gamma = float(sched.get("beta", 2.0)), float(sched.get("gamma", 0.5))
        e = np.array([p0[k] for k in src_names]) * np.exp(beta * qn); e = e / e.sum()
        p = {k: float((1 - gamma) * e[i] + gamma * p0[k]) for i, k in enumerate(src_names)}
        p, capped = cap_mix(p, step_); state["sched_capped"] = capped
        return p, L, Q, g, late

    def loader(epoch, start):
        smp = ResumableShardSampler(len(ds), rank, world, seed + 7919 * int(state.get("sched_round", 0)), epoch, start,
                                    source_ids=ds.df.source.tolist() if w else None, weights=cur_w or None)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        return iter(DataLoader(ds, batch_size=bs, sampler=smp, num_workers=d.get("workers", 8), pin_memory=True,
                               collate_fn=functools.partial(collate, pad_id=pad),
                               prefetch_factor=4 if d.get("workers", 8) else None))

    it = loader(state["epoch"], state["sample_offset"])
    if use:
        pit = pair_iter(state.get("pair_epoch", 0), state.get("pair_batch", 0))
    model.train()
    logf = open(f"{out}/log.jsonl", "a") if rank == 0 else None
    t0, seen = time.time(), 0
    params = [p for g in optim.param_groups for p in g["params"]]
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    # (11-02) loss.dyn_gate: the dynamics loss only while its update agrees with the answer loss's. Measured (logs/zz_gradalign.py): the
    # relational loss's gradient was aligned with the answer loss's when it raised the motion categories (ncg 4000: cos +0.05, 68% of
    # batches helped) and not when the gain was gone (ncg 6000: ~0, 43%), while it kept pushing just as hard (its own direction 0.96-0.97
    # consistent, half the answer gradient's size). One backward per step: even steps carry +lambda, odd steps 0 (aligned) or -lambda
    # (not aligned: the pair's net dynamics update cancels); EMAs of the two step types' gradients give the dynamics direction
    # d = (A - B) / (1 - mean B scale) and the answer direction c = A - d; aligned = cos(d, c) > 0 over the shared weights (LLM + bridge;
    # the dynamics head itself never gets the -lambda update)
    dg = lo.get("dyn_gate")
    if dg:
        loc_ = lambda t_: t_.to_local() if hasattr(t_, "to_local") else t_
        g_named = [(n, p) for n, p in named if "dyn_head." not in n]; g_head = [p for n, p in named if "dyn_head." in n]
        g_ema = {k: [torch.zeros_like(loc_(p).detach(), dtype=torch.float32) for _, p in g_named] for k in ("A", "B")}
        g_st = {"nA": 0, "nB": 0, "sB": 0.0, "aligned": True, "cos": float("nan"), "beta": float(dg.get("beta", 0.99)), "warm": int(dg.get("warm", 50)),
                "start": warm, "gn": None}
        # (11-02 smoke) statistics only after the learning-rate warm-up (the first steps' gradients reach 1e12 and would dominate the averages
        # for thousands of steps); every step's gradient divided by ONE running gradient norm (shared by both step types, so their relative
        # size - which carries the dynamics term - is kept); a step more than 4x the running norm is left out of the averages
        log0(f"[dyn_gate] shared tensors {len(g_named)}, head tensors {len(g_head)}, beta {g_st['beta']}, warm {g_st['warm']} updates per step type")
    debug = os.environ.get("JEPA_DEBUG_NONFINITE") == "1"
    state.setdefault("skipped_steps", 0); bad_streak = 0
    by_m = {}
    while state["step"] < total:
        step = state["step"]
        for g in optim.param_groups:
            g["lr"] = g["base_lr"] * lr_factor(step, starts[g["name"].split("_")[0]], total, warm, o.get("extend"))
        tw = 0.0 if step < int(latent_first * total) else float(lo.get("text_weight", 1.0))   # loss.text_weight 0: video-only step (no text loss)
        agg = {"loss": 0.0, "loss_text": 0.0, "loss_state": 0.0, "loss_pred": 0.0, "loss_swap": 0.0, "swap_gap": 0.0,
               "loss_geo": 0.0, "geo_acc": 0.0, "loss_info": 0.0, "info_acc": 0.0, "loss_vc": 0.0, "loss_router": 0.0, "loss_eq": 0.0, "answer_tokens": 0.0,
               "loss_corr": 0.0, "corr_acc": 0.0, "corr_stay": 0.0, "loss_ws": 0.0, "ws_cka": 0.0, "loss_next": 0.0, "loss_tr": 0.0, "dep_gain": 0.0, "loss_vcode": 0.0, "loss_iso": 0.0, "loss_dyn": 0.0, "loss_concept": 0.0, "concept_acc": 0.0, "dynw_mean": 0.0, "dynw_max": 0.0, "dynw_min": 0.0, "loss_areg": 0.0}
        _TM = os.environ.get("JEPA_TIMING") == "1"     # (10-29) per-phase timing (cuda-synchronised) on rank 0 -> <out>/timing.jsonl
        _tt = {}
        def _tm(k):
            if _TM: torch.cuda.synchronize(); _tt[k] = time.time()
        _tm("t0")
        _PS = int(os.environ.get("JEPA_PROFILE_STEP", "0") or 0)   # (10-29) torch profiler over one step on rank 0 -> <out>/profile.txt
        _prof = None
        if _PS and step == _PS and rank == 0:
            _prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]); _prof.start()
        for _ in range(accum):
            while True:  # every rank does exactly one forward/backward per micro-step (FSDP collectives stay aligned)
                try:
                    batch = next(it)
                except StopIteration:
                    state["epoch"] += 1; state["sample_offset"] = 0
                    it = loader(state["epoch"], 0); batch = next(it)
                state["sample_offset"] += bs
                for s in batch["skipped"]:
                    state["skipped"][s] = state["skipped"].get(s, 0) + 1
                if "video" in batch:
                    break
            _tm("data")
            dsc_ = (1.0 if step % 2 == 0 else (0.0 if g_st["aligned"] else -1.0)) if dg else 1.0
            res = model(batch["video"].to(device, vis_dtype, non_blocking=True), batch["frame_times"].to(device),
                        batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device),
                        batch["vis_mask"].to(device), lambda_state=lam, region_grid=batch["region_grid"], lambda_pred=lam_pred,
                        text_weight=tw, lambda_swap=lam_swap, swap_margin=swap_margin, margin_unit=margin_unit, lambda_geo=lam_geo,
                        lambda_info=lam_info, lambda_vc=lam_vc, lambda_eq=lam_eq, n_tub=batch["n_tub"].to(device), lambda_corr=lam_corr, lambda_ws=lam_ws, lambda_next=float(lo.get("lambda_next", 0.0)), lambda_iso=float(lo.get("lambda_iso", 0.0)),
                        mem=batch.get("mem"), lambda_dyn=float(lo.get("lambda_dyn", 0.0)),
                        seg=batch["seg"].to(device) if "seg" in batch and float(lo.get("seg_norm", 0.0)) > 0 else None,
                        seg_norm=float(lo.get("seg_norm", 0.0)),      # (10-23) packed rows: loss per original row
                        lambda_concept=float(lo.get("lambda_concept", 0.0)), ans_weight=str(lo.get("ans_weight", "row")),
                        cg={k_: v_.to(device) for k_, v_ in batch["cg"].items()} if "cg" in batch and float(lo.get("lambda_concept", 0.0)) > 0 else None,
                        **(cf_draw(batch["index"].numpy()) if cf_code is not None else {}),
                        **({"ce_dynw": lo["ce_dynw"]} if lo.get("ce_dynw") else {}),   # (11-01) dynamics-weighted answer loss (main batch only)
                        **({"dyn_scale": dsc_} if dg else {}))                         # (11-02) alignment gate
            _tm("fwd")
            (res["loss"] / accum).backward()
            _tm("bwd")
            if (sched or ana) and "per_ex" in res:
                with torch.no_grad():
                    si = torch.as_tensor(row_src[batch["index"].numpy()], device=device)
                    ok = si >= 0
                    for acc_ in ((src_acc,) if sched else ()) + ((ana_acc,) if ana else ()):
                        acc_[0].index_add_(0, si[ok], res["per_ex"].double()[ok]); acc_[1].index_add_(0, si[ok], torch.ones_like(si[ok], dtype=torch.float64))
            if debug and nonfinite_grads(named):
                print(f"[nonfinite] rank {rank} step {step} after LANGUAGE backward: loss {float(res['loss'])} "
                      f"first {nonfinite_grads(named)[:6]}", flush=True)
            for k in agg:
                if k in res:
                    agg[k] += float(res[k].detach()) / (accum if k != "answer_tokens" else 1)
            if "vcode_acc" in res and (getattr(model, "vcode", None) is not None or getattr(model, "vtie", None) is not None):
                vc_acc = [round(float(x), 4) for x in res["vcode_acc"]]
            if "tr_r2" in res:
                tr_r2 = [round(float(x), 4) for x in res["tr_r2"]]
            if "dyn_acc" in res and getattr(model, "dyn_head", None) is not None:
                dyn_acc = [round(float(x), 4) for x in res["dyn_acc"]]
            if "code_acc" in res and getattr(model, "codes", None) is not None:
                code_acc = [round(float(x), 4) for x in res["code_acc"]]
            if "next_err" in res and getattr(model, "next_head", None) is not None:
                next_err = [round(float(x), 3) for x in res["next_err"]]
            if "eq_r2" in res and getattr(model, "eq_heads", None) is not None:
                eq_r2 = [round(float(x), 3) for x in res["eq_r2"]]
            if "obs_tubelets" in res:                    # graft: text loss by how many tubelets the encoder saw
                e = by_m.setdefault(int(res["obs_tubelets"]), [0.0, 0])
                e[0] += float(res["loss_text"]); e[1] += 1
            seen += batch["video"].shape[0]
        _tm("pre_rp")
        tx_rec = state.get("_tx_last", {})               # (10-26) the latest replay loss in every record ("every" > 1 skipped it at the logged steps)
        if txd and step % int(txd.get("every", 1)) == 0:   # (10-26) "every": a replay batch every k steps (fixed 53k-row set: ~16 passes in nx12)
            ids, am, lab = tx_batch()
            if run.get("ckpt_mode") == "sac":               # (10-29) replay rows: plain full-layer recompute (SAC would keep 16k-token activations)
                for l in model.core.llm.model.layers:
                    l._gradient_checkpointing_func = l._gc_plain
            _rck = run.get("replay_checkpoint_every")      # (10-29) recompute density for the replay pass only (long single rows, up to 16k tokens):
            if _rck and run.get("activation_checkpointing", True):   # the video steps keep checkpoint_every; same on every rank (step-based)
                _lyr = model.core.llm.model.layers; _ck_saved = [l.gradient_checkpointing for l in _lyr]
                for i, l in enumerate(_lyr):
                    l.gradient_checkpointing = i % int(_rck) == 0
            if ids.shape[0] > 1 and txd.get("row_split", True):
                # (10-28) replay rows one at a time, padding dropped (nxs14_Q OOM at step ~32525: two rows padded to ~16k tokens on top of the
                # video step). Same objective: each row's mean CE weighted by its share of the labelled tokens = the batch token mean.
                # ALWAYS split (not only long batches): FSDP needs the same number of forward/backward passes on every rank - a length test
                # differs between ranks and deadlocked the collectives (22:53 relaunch hung at its first replay step)
                n_r = (lab[:, 1:] != -100).sum(1).float(); tot = n_r.sum().clamp_min(1.0); lt = 0.0
                for r in range(ids.shape[0]):
                    on = am[r].nonzero().flatten(); s0, s1 = int(on.min()), int(on.max()) + 1
                    lr = model.core(ids[r:r + 1, s0:s1].to(device), am[r:r + 1, s0:s1].to(device), None, None, labels=lab[r:r + 1, s0:s1].to(device))
                    (lr * (n_r[r] / tot) * float(txd.get("weight", 1.0))).backward(); lt += float(lr) * float(n_r[r] / tot)
                tx_rec = {"text_only": round(float(lt), 5)}; state["_tx_last"] = tx_rec
            else:
                lt = model.core(ids.to(device), am.to(device), None, None, labels=lab.to(device))
                (lt * float(txd.get("weight", 1.0))).backward(); tx_rec = {"text_only": round(float(lt), 5)}; state["_tx_last"] = tx_rec
            if _rck and run.get("activation_checkpointing", True):
                for l, v in zip(_lyr, _ck_saved):
                    l.gradient_checkpointing = v
            if run.get("ckpt_mode") == "sac":
                for l in model.core.llm.model.layers:
                    l._gradient_checkpointing_func = l._gc_sac
        _tm("rp")
        im_rec = {"image_text": _img["last"]} if imd and "last" in _img else {}       # (10-21) the latest image loss in every record
        if imd and step % int(imd.get("every", 1)) == 0:
            ib = img_batch()
            ri = model.forward_images(ib["video"].to(device, vis_dtype, non_blocking=True), ib["input_ids"].to(device), ib["attention_mask"].to(device),
                                      ib["labels"].to(device), ib["vis_mask"].to(device), pool=int(imd.get("pool", 3)))
            # one image row weighs as much as one video row of the language batch
            (ri["loss"] * float(imd.get("weight", 1.0)) * ib["video"].shape[0] / (bs * accum)).backward()
            _img["last"] = round(float(ri["loss_text"]), 5); im_rec = {"image_text": _img["last"]}
        tr_rec = {}
        if trd:
            while True:
                tb = next(tit)
                if "video" in tb: break
            rt = model(tb["video"].to(device, vis_dtype, non_blocking=True), tb["frame_times"].to(device), tb["input_ids"].to(device),
                       tb["attention_mask"].to(device), tb["labels"].to(device), tb["vis_mask"].to(device), n_tub=tb["n_tub"].to(device))
            (rt["loss"] * float(trd.get("weight", 0.5))).backward()
            tr_rec = {"tr_loss": round(float(rt["loss"]), 5), "tr_r2": [round(float(x), 4) for x in rt["tr_r2"]]}
        ic_rec = {}
        if icd:
            while True:
                ib = next(iit)
                if "video" in ib: break
            ri = model.forward_ic(ib["video"].to(device, vis_dtype, non_blocking=True), ib["frame_times"].to(device), ib["input_ids"].to(device),
                                  ib["attention_mask"].to(device), ib["labels"].to(device), ib["vis_mask"].to(device), ib["region_grid"])
            # one sequence (two answers) weighs as much as two language rows, as a pair does
            (ri["loss"] * float(icd.get("weight", 1.0)) * 2 * icd.get("per_gpu", 1) / (bs * accum)).backward()
            ic_rec = {"ic_loss": round(float(ri["loss"]), 5), "ic_tgt": round(float(ri["loss_tgt"]), 5)}
        use_rec = {}
        if use and step % use.get("every", 4) == 0:
            while True:
                try:
                    pb = next(pit)
                except StopIteration:
                    state["pair_epoch"] = state.get("pair_epoch", 0) + 1; state["pair_batch"] = 0
                    pit = pair_iter(state["pair_epoch"]); pb = next(pit)
                state["pair_batch"] = state.get("pair_batch", 0) + 1
                if "video" in pb:
                    break
            part = partner_positions(pb["index"], pair_ids)
            r = model.forward_pairs(pb["video"].to(device, vis_dtype, non_blocking=True), pb["frame_times"].to(device),
                                    pb["input_ids"].to(device), pb["attention_mask"].to(device), pb["labels"].to(device),
                                    pb["vis_mask"].to(device), part, lambda_state=lam, lambda_use=use.get("lambda", 0.0),
                                    margin=use.get("margin", 0.5), region_grid=pb["region_grid"], text_weight=tw, margin_unit=margin_unit,
                                    lambda_geo=lam_geo, lambda_info=lam_info, lambda_vc=lam_vc)
            # one pair row weighs as much as one language row (1 / global language batch) whatever the GPU count / accum
            (r["loss"] * (2 * use.get("pairs_per_gpu", 1)) / (bs * accum)).backward()
            use_rec = {"pair_text": round(float(r["loss_text"]), 5), "loss_use": round(float(r["loss_use"]), 5),
                       "use_gap": round(float(r["use_gap"]), 4), "pairs_kept": int((part >= 0).sum()) // 2,
                       **({"pair_geo": round(float(r["loss_geo"]), 5)} if lam_geo > 0 else {}),
                       **({"pair_info": round(float(r["loss_info"]), 5), "pair_info_acc": round(float(r["info_acc"]), 4)} if lam_info > 0 else {})}
        nat_rec = {}
        if use_native and step >= starts["enc"] and step % nat.get("every", 4) == 0:
            while True:
                try:
                    nb = next(nit)
                except StopIteration:
                    nepoch += 1; state["native_epoch"] = nepoch; nit = native_iter(nepoch); nb = next(nit)
                if nb["video"] is not None:
                    break
            r = native_loss(model.vision, model.reference, nb["video"].to(device, vis_dtype), gens,
                            nat.get("lambda_ctx", 0.5), nat.get("alpha", 0.1))
            (nat.get("lambda", 1.0) * r["loss"]).backward()
            if debug and nonfinite_grads(named):
                print(f"[nonfinite] rank {rank} step {step} after NATIVE backward: loss {float(r['loss'])} dense "
                      f"{float(r['dense'])} ref {float(r['ref'])} first {nonfinite_grads(named)[:6]}", flush=True)
            nat_rec = {"loss_native": round(float(r["loss"].detach()), 5), "native_dense": round(float(r["dense"]), 5),
                       "native_ref": round(float(r["ref"]), 5)}
        dg_rec = {}
        if dg:
            with torch.no_grad():
                k_ = "A" if step % 2 == 0 else "B"; b_ = g_st["beta"]
                if dsc_ < 0:
                    for p_ in g_head:
                        if p_.grad is not None: p_.grad.zero_()
                ss_ = torch.zeros((), device=device, dtype=torch.float64)
                for _, p_ in g_named:
                    if p_.grad is not None: ss_ += loc_(p_.grad).double().pow(2).sum()
                if distributed: dist.all_reduce(ss_)
                gn_ = float(ss_) ** 0.5
                if math.isfinite(gn_) and step >= g_st["start"] and (g_st["gn"] is None or gn_ <= 4 * g_st["gn"]):   # non-finite batches are skipped below
                    g_st["gn"] = gn_ if g_st["gn"] is None else b_ * g_st["gn"] + (1 - b_) * gn_
                    sc_ = (1 - b_) / g_st["gn"]
                    for e_, (_, p_) in zip(g_ema[k_], g_named):
                        if p_.grad is not None: e_.mul_(b_).add_(loc_(p_.grad).float(), alpha=sc_)
                    g_st["n" + k_] += 1
                    if k_ == "B": g_st["sB"] = b_ * g_st["sB"] + (1 - b_) * dsc_
                if k_ == "B" and min(g_st["nA"], g_st["nB"]) >= g_st["warm"]:
                    den_ = 1.0 - g_st["sB"] / (1 - b_ ** g_st["nB"])      # 1 - bias-corrected mean scale of the B steps (1 or 2)
                    v_ = torch.zeros(3, device=device, dtype=torch.float64)
                    for a_, bb_ in zip(g_ema["A"], g_ema["B"]):
                        d_ = (a_ - bb_) / den_; c_ = a_ - d_
                        v_ += torch.stack([(d_ * c_).sum().double(), d_.pow(2).sum().double(), c_.pow(2).sum().double()])
                    if distributed: dist.all_reduce(v_)          # every rank: the same cosine, the same gate
                    g_st["cos"] = float(v_[0] / (v_[1] * v_[2]).sqrt().clamp_min(1e-30)); g_st["aligned"] = g_st["cos"] > 0
                    g_st["ratio"] = float((v_[1] / v_[2].clamp_min(1e-30)).sqrt())
            dg_rec = {"dyn_cos": round(g_st["cos"], 5) if g_st["cos"] == g_st["cos"] else None, "dyn_gate": int(g_st["aligned"]),
                      "dyn_ratio": round(g_st.get("ratio", float("nan")), 4) if "ratio" in g_st else None, "dyn_scale": dsc_}
        gn_rec = {}
        if ana and (state["step"] + 1) % run.get("log_every", 10) == 0:      # (10-22) gradient norm per optimizer group, before clipping
            with torch.no_grad():
                gss = {}
                for g_ in optim.param_groups:
                    k_ = g_["name"].replace("_nd", "")
                    v_ = gss.get(k_, torch.zeros((), device=device, dtype=torch.float64))
                    for p_ in g_["params"]:
                        if p_.grad is not None:
                            gl_ = p_.grad.to_local() if hasattr(p_.grad, "to_local") else p_.grad
                            v_ = v_ + gl_.double().pow(2).sum()
                    gss[k_] = v_
                ks_ = sorted(gss)
                we_ = model.core.llm.get_input_embeddings().weight; we_ = we_.to_local() if hasattr(we_, "to_local") else we_
                sc_ = torch.stack([torch.as_tensor(getattr(model, "_ev_ss", 0.0), device=device, dtype=torch.float64),
                                   torch.as_tensor(float(getattr(model, "_ev_n", 0)), device=device, dtype=torch.float64),
                                   we_.detach().double().pow(2).sum(), torch.as_tensor(float(we_.numel()), device=device, dtype=torch.float64)])
                vv_ = torch.cat([torch.stack([gss[k] for k in ks_]), sc_])
                if distributed: dist.all_reduce(vv_)       # every rank: same condition, same order
                gn_rec = {f"gn_{k}": round(float(x) ** 0.5, 4) for k, x in zip(ks_, vv_[: len(ks_)].tolist())}
                e_ss, e_n, w_ss, w_n = vv_[len(ks_):].tolist()
                if e_n > 0 and w_n > 0:                     # rms of the visual tokens (all ranks' last batch) / rms of the whole word table
                    gn_rec.update({"ev_rms": round((e_ss / e_n) ** 0.5, 5), "word_rms": round((w_ss / w_n) ** 0.5, 5)})
        gnorm = torch.nn.utils.clip_grad_norm_(params, o.get("clip", 1.0))
        gnorm = float(gnorm.full_tensor() if hasattr(gnorm, "full_tensor") else gnorm)
        if math.isfinite(gnorm):          # the norm is global, so every rank takes the same branch
            if ns is not None:
                ns.snapshot()
            _tm("pre_opt"); optim.step(); bad_streak = 0; _tm("opt")
            if _prof is not None:
                torch.cuda.synchronize(); _prof.stop()
                with open(os.path.join(out, "profile.txt"), "w") as _f:
                    _f.write(_prof.key_averages().table(sort_by="cuda_time_total", row_limit=60, max_name_column_width=90))
            if _TM and rank == 0:
                with open(os.path.join(out, "timing.jsonl"), "a") as _f:
                    _f.write(json.dumps({"step": step, **{k: round(v - _tt["t0"], 3) for k, v in _tt.items()}, "ntok": int(batch["attention_mask"].sum())}) + "\n")
            if _prec_h:
                for h_ in _prec_h: h_.remove()
                _prec_h = []
                loc = lambda p_: p_.to_local() if hasattr(p_, "to_local") else p_
                pd_ = sorted({str(loc(p_).dtype) for g_ in optim.param_groups for p_ in g_["params"]})
                sd_ = sorted({str(v_.dtype) for st_ in optim.state.values() for k_, v_ in st_.items() if k_ in ("exp_avg", "exp_avg_sq")})
                log0(f"[precision] master weights {pd_} | optimizer states {sd_} | compute: LLM layer out {_prec.get('llm_layer_out')}, "
                     f"bridge out {_prec.get('bridge_out')} | frozen V-JEPA {next(model.vision.parameters()).dtype}")
            if ns is not None:
                ns.project()
        else:                              # never let one bad batch turn the weights into NaN: skip and record
            state["skipped_steps"] += 1; bad_streak += 1
            if rank == 0:
                rec = {"step": step + 1, "nonfinite_grad": True, "loss_text": agg["loss_text"], **use_rec, **nat_rec,
                       "first_bad": nonfinite_grads(named)[:6], "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
                print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
            assert bad_streak < 20, "20 consecutive non-finite steps: stopping"
        optim.zero_grad(set_to_none=True)
        state["step"] += 1
        sched_rec = {}
        if sched and state["step"] % int(sched.get("every", 1000)) == 0:
            p, Lw, Qw, gw, late = sched_weights(state["step"])
            if state.get("sched_prevL") is not None and Lw:
                cur_w = p; state["sched_w"] = p; state["sched_round"] = int(state.get("sched_round", 0)) + 1
                state["sample_offset"] = 0; it = loader(state["epoch"], 0)
                sched_rec = {"sched_mix": {k: round(v, 5) for k, v in p.items()}, "sched_loss": {k: round(v, 4) for k, v in Lw.items()},
                             "sched_Q": {k: round(v, 5) for k, v in Qw.items()}, "sched_gain": {k: round(v, 3) for k, v in gw.items()}, "sched_late": late,
                             **({"sched_capped": state.get("sched_capped", []), "sched_passes": {k: round(state["sched_seen"][k] / max(src_rows[src_ix[k]], 1), 3) for k in src_names}} if "sched_seen" in state else {})}
        src_rec = {}
        if ana and ana_rows is not None and state["step"] % ana_every == 0:
            a_ = ana_acc.clone()
            if distributed: dist.all_reduce(a_)
            for i_, k_ in enumerate(src_names): state["ana_seen"][k_] = state["ana_seen"].get(k_, 0.0) + float(a_[1, i_])
            src_rec = {"src_loss": {k_: round(float(a_[0, i_] / a_[1, i_]), 4) for i_, k_ in enumerate(src_names) if a_[1, i_] > 0},
                       "src_n": {k_: int(a_[1, i_]) for i_, k_ in enumerate(src_names)},
                       "src_passes": {k_: round(state["ana_seen"][k_] / max(ana_rows[i_], 1.0), 4) for i_, k_ in enumerate(src_names)}}
            ana_acc.zero_()
        if rank == 0 and (state["step"] % run.get("log_every", 10) == 0 or state["step"] == 1 or (nat_rec or use_rec) and state["step"] < 50 or src_rec):
            dt = time.time() - t0
            lrs = {g["name"]: g["lr"] for g in optim.param_groups if not g["name"].endswith("_nd")}
            rec = {"step": state["step"], **{k: round(v, 5) for k, v in agg.items()}, **use_rec, **nat_rec, **tr_rec, **tx_rec, **im_rec, **ic_rec, **sched_rec, **src_rec, **gn_rec, **dg_rec, **({"rsc_pr": round(model._rsc_pr, 2)} if getattr(model, "_rsc_pr", None) else {}), "grad_norm": round(gnorm, 4),
                   **{f"lr_{k}": v for k, v in lrs.items()}, "samples_per_s_rank0": round(seen / dt, 2),
                   "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else 0,
                   "skipped": state["skipped"], "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if getattr(model, "dyn_head", None) is not None:
                rec["dyn_acc"] = dyn_acc       # own clip picked among batch clips + its moment-shuffled copies; real beats all its shuffles
            if getattr(model, "vtie", None) is not None:
                rec["vtie_acc"] = vc_acc       # next moment's token picked among word table + every video token of the batch: right, own token (copy)
            if getattr(model, "vcode", None) is not None:
                rec["vcode_acc"] = vc_acc      # next-moment V-JEPA codes predicted right at the video positions: level 1, all levels (chance 1/1024)
            if getattr(model, "textread", None) is not None and not trd:
                rec["tr_r2"] = tr_r2           # explained share of V-JEPA's region states read at the text positions: seen moments, moments after them
            if getattr(model, "codes", None) is not None:
                rec["code_acc"] = code_acc     # V-JEPA code numbers written right by the LLM, quantiser level 1..4 (chance 0.0039)
            if getattr(model, "next_head", None) is not None:
                rec["next_err"] = next_err     # next moment, error / variance: prediction for directions 0-16 / 16-256 / 256-1024 / 1024+, then the same for copying
            if getattr(model, "eq_heads", None) is not None:
                rec["eq_r2"] = eq_r2                                     # explained fraction of the equalised state, directions 0-16 / 16-256 / 256+
            if ns is not None:
                rec["ns_removed"] = round(ns.removed, 4)                 # share of the LLM step that lay in the language directions (rank 0's shard)
            if model.bridge.last_erank is not None:
                rec["token_erank"] = round(model.bridge.last_erank, 2)   # effective rank of the visual tokens the LLM receives

            if getattr(model, "tmem_stats", None) is not None:              # (10-17) memory token rms, overview token rms, clips per row
                rec["mem_rms"], rec["ev_rms"], rec["mem_k"] = [round(x, 4) for x in model.tmem_stats]
            if by_m:
                rec["text_by_obs"] = {str(k): round(a / n, 4) for k, (a, n) in sorted(by_m.items())}; by_m.clear()
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
        if state["step"] % run.get("save_every", 1000) == 0 or state["step"] == total:
            save_ckpt(out, state["step"], trainable(model), optim, state, distributed)
    log0("[done]", state)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
