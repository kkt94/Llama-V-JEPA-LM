"""V-JEPA-LM (JEPA-LLM): frozen V-JEPA 2.1 video encoder -> spectral bridge (MLP) -> LLM (Qwen3 or Llama 3).

Inference path (the same computation as the evaluation code the model was measured with); L levels of width D, top k
principal directions and LLM width H come from the config (ViT-g: L 4, D 1408, k 2048; distilled ViT-L: L 1, D 1024,
k 1024; H = the LLM's hidden size):
  video [1, 3, 32, 384, 384] -> V-JEPA 2.1 encoder, L levels [1, 16 * 24 * 24, L, D]
  -> per-level parameter-free LayerNorm (fp32) -> 3 x 3 patch average pooling to 8 x 8 cells per tubelet
     ([1, 16 * 64 = 1024, L, D], stored in bf16)
  -> equalisation (fp32): centre, project on the top k principal directions of frozen V-JEPA, unit variance
  -> MLP bridge (LayerNorm, Linear k -> 2048, GELU, Linear 2048 -> H) + learned (tubelet, row, column) position,
     real-time (seconds) and cell-geometry codes -> 1024 video tokens
  -> spliced into the prompt at the evidence positions of the LLM's input embeddings.
Multiple choice: the logits of the option letters at the first answer position (no generation).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from .configuration_jepa_llm import JepaLlmConfig
from .processing_jepa_llm import (CHAT_TEMPLATES, IGNORE, VIS, build_generation_prompt, build_scoring_example, letter_ids,
                                  load_video, mc_prompt)
from .vjepa2_1_encoder import VARIANTS as VISION_VARIANTS

OBSERVED = 0
GRID = 24  # 384 / 16 patches per side


def pool_regions(states: torch.Tensor, n_tubelets: int, cells: int, regions: torch.Tensor, grid: int = GRID):
    """[B, n_tub * grid * grid, L, D] -> (pooled [B, n_tub * cells^2, L, D], coords [K, 3] (t, y, x), geometry [B, K, 4]
    = (cy, cx, h, w) of each cell as a fraction of the frame). regions [B, 4] = patch-grid (y0, y1, x0, x1)."""
    b, n, l, d = states.shape
    x = states.view(b, n_tubelets, grid, grid, l * d)
    out, geo = [], []
    for i in range(b):
        y0, y1, x0, x1 = [int(v) for v in regions[i]]
        sub = x[i, :, y0:y1, x0:x1].permute(0, 3, 1, 2)
        p = F.adaptive_avg_pool2d(sub, cells)
        out.append(p.permute(0, 2, 3, 1).reshape(n_tubelets * cells * cells, l, d))
        ch, cw = (y1 - y0) / cells, (x1 - x0) / cells
        cy = (y0 + (torch.arange(cells) + 0.5) * ch) / grid
        cx = (x0 + (torch.arange(cells) + 0.5) * cw) / grid
        g = torch.stack(torch.meshgrid(cy, cx, indexing="ij"), -1).reshape(-1, 2)
        g = torch.cat([g, torch.tensor([[ch / grid, cw / grid]]).expand(len(g), 2)], -1)
        geo.append(g.repeat(n_tubelets, 1))
    tt, yy, xx = torch.meshgrid(torch.arange(n_tubelets), torch.arange(cells), torch.arange(cells), indexing="ij")
    return torch.stack(out), torch.stack([tt, yy, xx], -1).reshape(-1, 3), torch.stack(geo).to(states.device)


def time_features(seconds: torch.Tensor, dim: int) -> torch.Tensor:
    """sinusoidal encoding of real timestamps (seconds), periods from 0.1 s to ~1000 s"""
    half = dim // 2
    freqs = torch.exp(-math.log(1e4) * torch.arange(half, device=seconds.device, dtype=torch.float32) / half) * 10.0
    ang = seconds.float().unsqueeze(-1) * freqs
    return torch.cat([ang.sin(), ang.cos()], -1)


class SpectralStats(nn.Module):
    """Equalisation statistics of frozen V-JEPA (pooled cell states, levels concatenated: L * D): mean [L * D], top-k
    principal directions [L * D, k], per-direction scale [k] (eigenvalue^-1/2). Always fp32: the small directions are scaled up by ~100x, so bf16
    statistics would not reproduce the model. Casting the model (`.to(torch.bfloat16)`, `.half()`) moves these buffers to
    the new device but keeps them fp32."""

    def __init__(self, in_dim: int, k: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(in_dim, dtype=torch.float32))
        self.register_buffer("basis", torch.zeros(in_dim, k, dtype=torch.float32))
        self.register_buffer("scale", torch.zeros(k, dtype=torch.float32))

    def _apply(self, fn, recurse=True):
        for name, buf in self._buffers.items():
            if buf is None:
                continue
            new = fn(buf)
            if new.dtype != buf.dtype and buf.dtype == torch.float32:
                new = buf.to(device=new.device)
            self._buffers[name] = new
        return self

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        b, n = pooled.shape[:2]
        x = pooled.reshape(b, n, -1).float() - self.mean.float()
        return (x @ self.basis.float()) * self.scale.float()


class SpectralBridge(nn.Module):
    """equalised V-JEPA cell states -> LLM-width video tokens (MLP mode) + position / time / geometry codes"""

    def __init__(self, in_dim: int, k: int, hidden: int, out_dim: int, max_tubelets: int = 64, max_cells: int = 8, time_dim: int = 64):
        super().__init__()
        self.spectral = SpectralStats(in_dim, k)
        self.inp = nn.ModuleDict({str(OBSERVED): nn.Sequential(nn.LayerNorm(k), nn.Linear(k, hidden))})
        self.out = nn.Sequential(nn.GELU(), nn.Linear(hidden, out_dim))
        self.t_emb = nn.Embedding(max_tubelets, out_dim)
        self.y_emb = nn.Embedding(max_cells, out_dim)
        self.x_emb = nn.Embedding(max_cells, out_dim)
        self.kind_emb = nn.Embedding(2, out_dim)
        self.time_proj = nn.Linear(time_dim, out_dim)
        self.geom_proj = nn.Linear(4, out_dim)
        self.time_dim = time_dim

    def forward(self, pooled, coords, seconds, geometry):
        """pooled [B, K, L, D], coords [K, 3], seconds [B, K], geometry [B, K, 4] -> [B, K, out_dim]"""
        inp = self.inp[str(OBSERVED)]
        x = self.spectral(pooled)
        h = self.out(inp(x.to(inp[1].weight.dtype)))
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        pos = self.t_emb(coords[..., 0]) + self.y_emb(coords[..., 1]) + self.x_emb(coords[..., 2])
        code = pos.to(h.dtype) + self.kind_emb.weight[OBSERVED].to(h.dtype) \
            + self.time_proj(time_features(seconds, self.time_dim).to(self.time_proj.weight.dtype)).to(h.dtype)
        code = code + self.geom_proj(geometry.to(self.geom_proj.weight.dtype)).to(h.dtype)
        return h + code


class JepaLlmPreTrainedModel(PreTrainedModel):
    config_class = JepaLlmConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = False
    _no_split_modules = ["Block", "Qwen3DecoderLayer", "LlamaDecoderLayer"]
    _supports_sdpa = True
    _keep_in_fp32_modules_strict = ["spectral"]


class JepaLlmModel(JepaLlmPreTrainedModel):
    """Video question answering with a frozen V-JEPA 2.1 encoder and a Qwen3 or Llama 3 language model.

        model = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
        model.chat("clip.mp4", "Describe what happens.")
        model.choose("clip.mp4", "What does the person do?", ["opens a door", "closes a door"])   # -> (index, scores)
    """
    # lm_head shares the input embeddings only when config.tie_word_embeddings is true (Qwen3-1.7B / 4B, Llama-3.2-3B);
    # transformers skips this mapping otherwise (Qwen3-8B, Llama-3.1-8B: lm_head stored in the checkpoint)
    _tied_weights_keys = {"language_model.lm_head.weight": "language_model.model.embed_tokens.weight"}

    def __init__(self, config: JepaLlmConfig):
        super().__init__(config)
        self.vision_encoder = VISION_VARIANTS[getattr(config, "vision_variant", "vit_giant_384")](**(config.vision_config or {}))
        hidden = config.text_config.hidden_size
        self.bridge = SpectralBridge(config.vision_levels * config.vision_level_dim, config.spectral_k, config.bridge_hidden, hidden,
                                     max_tubelets=config.max_tubelets, max_cells=max(4, config.evidence_cells), time_dim=config.time_dim)
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self._tokenizer = None
        self._rope_mode = None
        self.post_init()

    def _prepare_rope(self):
        """rope_inv_freq_bf16 (config): round the LLM's rotary frequencies to bf16, as casting the whole model to bf16
        does (the evaluation setup); otherwise restore fp32 frequencies computed from the config. Idempotent."""
        want = bool(getattr(self.config, "rope_inv_freq_bf16", True))
        if self._rope_mode == want:
            return
        for mod in self.language_model.modules():
            if type(mod).__name__.endswith("RotaryEmbedding") and isinstance(getattr(mod, "inv_freq", None), torch.Tensor):
                if not hasattr(mod, "_inv_freq_fp32"):
                    rope_type = getattr(mod, "rope_type", "default")
                    if rope_type == "default" and hasattr(mod, "compute_default_rope_parameters"):
                        inv, _ = mod.compute_default_rope_parameters(mod.config, torch.device("cpu"))   # fp32, as at init
                    else:                                       # e.g. Llama 3 "llama3" frequency scaling, recomputed in fp32 as at init
                        try:
                            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
                            inv, _ = ROPE_INIT_FUNCTIONS[rope_type](mod.config, torch.device("cpu"))
                        except (ImportError, KeyError):
                            inv = mod.inv_freq.detach().float().cpu()
                    mod._inv_freq_fp32 = inv.float()
                for name in ("inv_freq", "original_inv_freq"):
                    if isinstance(mod._buffers.get(name), torch.Tensor):
                        dev = mod._buffers[name].device
                        src = mod._inv_freq_fp32.to(dev)
                        mod._buffers[name] = src.to(torch.bfloat16) if want else src.clone()
        self._rope_mode = want

    # ---- tokenizer --------------------------------------------------------------------------------------------------
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self.set_tokenizer(AutoTokenizer.from_pretrained(self.name_or_path, trust_remote_code=True))
        return self._tokenizer

    def set_tokenizer(self, tok):
        tok.chat_template = CHAT_TEMPLATES[getattr(self.config, "chat_template_name", "qwen3_instruct")]   # the template the model was trained with
        self._tokenizer = tok
        return tok

    # ---- video -> tokens ----------------------------------------------------------------------------------------------
    def load_video(self, video_path: str, start_sec=None, end_sec=None):
        """-> (frames [3, 32, 384, 384] float32 normalised, frame times [32] seconds) on the CPU"""
        return load_video(video_path, self.config.num_frames, self.config.image_size, start_sec, end_sec)

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor, frame_times: torch.Tensor) -> torch.Tensor:
        """video [B, 3, T, 384, 384] (normalised), frame_times [B, T] seconds -> video tokens [B, T/2 * 64, LLM hidden]"""
        if video.dim() == 4:
            video, frame_times = video.unsqueeze(0), frame_times.unsqueeze(0)
        dev, dt = self.device, self.vision_encoder.patch_embed.proj.weight.dtype
        video, frame_times = video.to(dev, dt), frame_times.to(dev)
        levels = self.config.vision_levels
        # ViT-g: the 4 trained levels concatenated [B, N, 4 * 1408]; distilled ViT-L: its last (only trained) level [B, N, 1024]
        z = self.vision_encoder(video, training=True) if levels > 1 else self.vision_encoder(video)
        b, n = z.shape[:2]
        z = z.view(b, n, levels, -1)
        zn = F.layer_norm(z.float(), z.shape[-1:])                                     # each level, no affine
        n_tub = frame_times.shape[-1] // 2
        cells = self.config.evidence_cells
        pooled, coords, geom = pool_regions(zn, n_tub, cells, torch.tensor([[0, GRID, 0, GRID]] * b))
        pooled = pooled.to(torch.bfloat16).to(dev, self.bridge.inp[str(OBSERVED)][1].weight.dtype)   # as the evaluation cache
        sec = frame_times.float().view(b, n_tub, 2).mean(-1).repeat_interleave(cells * cells, dim=1)
        return self.bridge(pooled, coords.to(dev), sec, geom)

    def _evidence(self, video, start_sec=None, end_sec=None, evidence=None):
        if evidence is not None:
            return evidence.to(self.device)
        if isinstance(video, str):
            frames, times = self.load_video(video, start_sec, end_sec)
        else:
            frames, times = video
        return self.encode_video(frames, times)

    def _embed(self, input_ids, vis_mask, ev):
        emb = self.language_model.get_input_embeddings()(input_ids)
        return emb.masked_scatter(vis_mask.unsqueeze(-1), ev.to(emb.dtype))

    # ---- multiple choice ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def choose(self, video, question: str, options, start_sec=None, end_sec=None, evidence=None):
        """Multiple choice by option-letter likelihood. video: a file path, or (frames, frame_times) from load_video.
        -> (index of the chosen option, [score per option]) where the scores are the LLM's logits of "A", "B", ... at the
        first answer position for the prompt "{question}\\nA. ...\\nB. ...\\nAnswer with the option's letter from the
        given choices directly." """
        tok = self.tokenizer
        self._prepare_rope()
        ev = self._evidence(video, start_sec, end_sec, evidence)
        ex = build_scoring_example(tok, f"{VIS}\n{mc_prompt(question, options)}", ev.shape[1])
        dev = self.device
        ids = torch.tensor([ex["input_ids"]], device=dev)
        lab = torch.tensor([ex["labels"]], device=dev)
        vis = torch.tensor([ex["vis_mask"]], device=dev).bool()
        last = self.language_model.model(inputs_embeds=self._embed(ids, vis, ev), attention_mask=torch.ones_like(ids),
                                         use_cache=False).last_hidden_state
        sel = lab[:, 1:] != IGNORE
        logits = self.language_model.lm_head(last[:, :-1][sel]).float()           # every answer position, as in training
        scores = logits[0, letter_ids(tok, len(options))]                        # the first answer position
        return int(scores.argmax()), scores.tolist()

    # ---- free generation ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def chat(self, video, question: str, max_new_tokens: int = 256, start_sec=None, end_sec=None, evidence=None) -> str:
        """Greedy answer to a free-form question about the video (single turn)."""
        tok = self.tokenizer
        self._prepare_rope()
        ev = self._evidence(video, start_sec, end_sec, evidence)
        p = build_generation_prompt(tok, f"{VIS}\n{question}", ev.shape[1])
        dev = self.device
        ids = torch.tensor([p["input_ids"]], device=dev)
        vis = torch.tensor([p["vis_mask"]], device=dev).bool()
        llm = self.language_model
        out = llm(inputs_embeds=self._embed(ids, vis, ev), use_cache=True)
        stop = {tok.eos_token_id, *(tok.convert_tokens_to_ids(t) for t in ("<|eot_id|>", "<|im_end|>") if t in tok.get_vocab())}
        gen = []
        for _ in range(max_new_tokens):
            nxt = int(out.logits[0, -1].argmax())
            if nxt in stop:
                break
            gen.append(nxt)
            out = llm(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=out.past_key_values, use_cache=True)
        return tok.decode(gen, skip_special_tokens=True)

    def forward(self, input_ids, vis_mask=None, video=None, frame_times=None, evidence=None, attention_mask=None, **kw):
        """Plain forward of the LLM with the video tokens spliced in at `vis_mask` positions (returns the CausalLM output)."""
        self._prepare_rope()
        if evidence is None and video is not None:
            evidence = self.encode_video(video, frame_times)
        if evidence is None:
            return self.language_model(input_ids=input_ids, attention_mask=attention_mask, **kw)
        emb = self._embed(input_ids, vis_mask.bool(), evidence)
        return self.language_model(inputs_embeds=emb, attention_mask=attention_mask, **kw)
