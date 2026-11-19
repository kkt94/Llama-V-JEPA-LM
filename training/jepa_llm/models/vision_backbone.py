"""V-JEPA 2.1 encoder/predictor wrapper with a canonical tensor contract.

Outside this module every visual state is [B, N, L, D] (tokens x levels x dim). The upstream API differs per variant:
- g1B returns the 4 trained levels concatenated (5632 = 4 x 1408) when called with training=True, and its predictor
  consumes and predicts that 4-level concat.
- The distilled B/L models train only the last level (the other three level norms are untouched initial values), their
  predictor consumes the last level only and predicts in the 2B teacher space (1664).

Upstream facts this wrapper relies on (checked against code/vjepa2 @ 204698b):
- `masks` passed to the encoder are indices to KEEP (apply_masks gathers), applied after patch embedding and before any
  transformer block, despite the docstring saying "remove".
- The predictor returns target predictions in the order of `masks_y` and context predictions in the order of `masks_x`.
"""
import copy
import os
import sys
from dataclasses import dataclass

import torch
from torch import nn

VJEPA_CODE_DIR = "/data/llama_vjepa_lm/code/vjepa2"
CKPT_DIR = "/data/llama_vjepa_lm/models/vjepa2_1"


@dataclass(frozen=True)
class VariantSpec:
    constructor: str
    checkpoint: str
    encoder_key: str
    level_dim: int
    n_levels: int          # trained output levels exposed by encode_observation
    pred_levels: int       # levels in predictor output
    pred_dim: int          # per-level predictor output dim


VARIANTS = {
    "vitb_dist": VariantSpec("vjepa2_1_vit_base_384", "vjepa2_1_vitb_dist_vitG_384.pt", "ema_encoder", 768, 1, 1, 1664),
    "vitl_dist": VariantSpec("vjepa2_1_vit_large_384", "vjepa2_1_vitl_dist_vitG_384.pt", "ema_encoder", 1024, 1, 1, 1664),
    "vitg_1b": VariantSpec("vjepa2_1_vit_giant_384", "vjepa2_1_vitg_384.pt", "target_encoder", 1408, 4, 4, 1408),
    # (11-16) V-JEPA 2.1 ViT-G/16 (1.85B, 48 blocks, width 1664): 4 trained levels -> 6656, predictor in the same 4 x 1664 space
    "vitG_2b": VariantSpec("vjepa2_1_vit_gigantic_384", "vjepa2_1_vitG_384.pt", "target_encoder", 1664, 4, 4, 1664),
}

PATCH, TUBELET, GRID = 16, 2, 24  # 384 / 16 = 24 patches per side


class _SameDtypeSDPA:
    """Stand-in for torch.nn.functional inside the upstream attention module. The 2.1 RoPE attention returns q, k in
    float32 (rotation in fp32) while v stays in the compute dtype; upstream only ever runs under autocast, which hides
    this. Here bf16 weights run without autocast (first GPU run: 'query float, key float, value BFloat16'), so q and k are
    cast to v's dtype - what autocast would do - and everything else is torch.nn.functional unchanged."""

    def __init__(self, f):
        self._f = f

    def __getattr__(self, name):
        return getattr(self._f, name)

    fp32 = os.environ.get("JEPA_VJEPA_ATTN_FP32", "0") == "1"   # tried: does NOT remove the NaN (12/40, 9/40)

    def scaled_dot_product_attention(self, q, k, v, *a, **kw):
        if self.fp32 and v.dtype != torch.float32:
            # bf16 attention backward through the RoPE path gave NaN gradients in ~25% of native-loss steps (encoder
            # block 6, while fp32 gave 0/30); attention runs in fp32, linear / MLP layers stay bf16
            return self._f.scaled_dot_product_attention(q.float(), k.float(), v.float(), *a, **kw).to(v.dtype)
        return self._f.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v, *a, **kw)


def _backbones():
    if VJEPA_CODE_DIR not in sys.path:
        sys.path.insert(0, VJEPA_CODE_DIR)
    from src.hub import backbones
    import importlib
    mods = importlib.import_module("app.vjepa_2_1.models.utils.modules")
    if not isinstance(mods.F, _SameDtypeSDPA):
        mods.F = _SameDtypeSDPA(mods.F)
    return backbones


class VJEPABackbone(nn.Module):
    def __init__(self, variant: str, ckpt_dir: str = CKPT_DIR):
        super().__init__()
        self.variant, self.spec = variant, VARIANTS[variant]
        bb = _backbones()
        self.encoder, self.predictor = getattr(bb, self.spec.constructor)(pretrained=False)
        sd = torch.load(f"{ckpt_dir}/{self.spec.checkpoint}", map_location="cpu", mmap=True, weights_only=False)
        # strict=True: a failed load must never be hidden behind random init
        self.encoder.load_state_dict(bb._clean_backbone_key(dict(sd[self.spec.encoder_key])), strict=True)
        self.predictor.load_state_dict(bb._clean_backbone_key(dict(sd["predictor"])), strict=True)

    # ---- geometry -------------------------------------------------------------------------------------------------
    @staticmethod
    def grid_shape(num_frames: int, height: int = 384, width: int = 384):
        assert num_frames % TUBELET == 0, f"num_frames={num_frames} must be a multiple of tubelet {TUBELET}"
        assert height % PATCH == 0 and width % PATCH == 0
        return num_frames // TUBELET, height // PATCH, width // PATCH

    @classmethod
    def token_coords(cls, num_frames: int, height: int = 384, width: int = 384) -> torch.Tensor:
        """[N, 3] (tubelet t, patch row y, patch col x) in the upstream flattening order t-major, then y, then x."""
        t, h, w = cls.grid_shape(num_frames, height, width)
        tt, yy, xx = torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")
        return torch.stack([tt, yy, xx], -1).reshape(-1, 3)

    @classmethod
    def tokens_before_frame(cls, frame_cutoff: int, num_frames: int, height: int = 384, width: int = 384):
        """Token indices whose tubelet lies entirely before `frame_cutoff` (exclusive). The cutoff must sit on a tubelet
        boundary so that no tubelet mixes observed and future frames (spec 7.2)."""
        if frame_cutoff % TUBELET:
            raise ValueError(f"frame_cutoff={frame_cutoff} splits a tubelet of {TUBELET} frames")
        c = cls.token_coords(num_frames, height, width)
        return torch.nonzero(c[:, 0] < frame_cutoff // TUBELET).squeeze(1)

    # ---- forward paths ----------------------------------------------------------------------------------------------
    def _split_levels(self, x: torch.Tensor, levels: int) -> torch.Tensor:
        b, n, d = x.shape
        return x.view(b, n, levels, d // levels)

    def encode_observation(self, video: torch.Tensor) -> torch.Tensor:
        """video [B,3,T,H,W] (already only the observable frames) -> [B, N, n_levels, level_dim]."""
        if self.spec.n_levels == 4:
            return self._split_levels(self.encoder(video, training=True), 4)
        return self.encoder(video).unsqueeze(2)

    def encode_masked_context(self, video: torch.Tensor, visible_idx: torch.Tensor) -> torch.Tensor:
        """Encoder attention only over `visible_idx` [B, K] (same K for every row). Returns the predictor input format
        [B, K, D_in]: 4-level concat for g1B, last level for the distilled models."""
        assert visible_idx.dim() == 2 and visible_idx.shape[0] == video.shape[0]
        return self.encoder(video, masks=visible_idx, training=self.spec.n_levels == 4)

    def predict(self, context: torch.Tensor, visible_idx: torch.Tensor, target_idx: torch.Tensor):
        """-> (pred_target [B, M, pred_levels, pred_dim] in target_idx order, pred_context [B, K, ...] or None)."""
        p_t, p_c = self.predictor(context, visible_idx, target_idx, mod="video")
        p_t = self._split_levels(p_t, self.spec.pred_levels)
        p_c = None if p_c is None else self._split_levels(p_c, self.spec.pred_levels)
        return p_t, p_c

    def frozen_reference(self) -> "VJEPABackbone":
        """Independent, frozen copy (E0/P0). Never shares storage with the online model."""
        ref = copy.deepcopy(self)
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
        ref.train = lambda mode=True: ref  # stays in eval even if a parent calls .train()
        return ref


SIGLIP_HUB = "/data/llama_vjepa_lm/hf_home/hub"


class SigLIPBackbone(nn.Module):
    """Control encoder (10-13): a language-aligned image encoder (SigLIP so400m-patch14-384, frozen) behind the same
    tensor contract, so that a run differs from the V-JEPA runs in the encoder only. Every frame is encoded on its own
    (penultimate layer, 27x27 patches, resampled to the 24x24 grid the pooling expects); the two frames of a V-JEPA
    tubelet become the two 'levels' of one token, [B, (T/2)*24*24, 2, 1152], so the token layout (moments x cells), the
    frames seen and the number of LLM tokens are those of the V-JEPA runs and the short time step inside a pair is kept
    (not averaged away). No predictor."""

    def __init__(self, variant: str = "siglip_so400m"):
        super().__init__()
        from transformers import AutoModel
        self.variant, self.spec = variant, VariantSpec("", "", "", 1152, 2, 2, 1152)
        self.encoder = AutoModel.from_pretrained("google/siglip-so400m-patch14-384", cache_dir=SIGLIP_HUB, local_files_only=True).vision_model
        self.predictor = nn.Identity()
        self.register_buffer("in_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("in_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def encode_observation(self, video: torch.Tensor) -> torch.Tensor:
        """video [B,3,T,384,384] ImageNet-normalised (the loader's output) -> [B, (T/2)*576, 2, 1152]."""
        b, _, t, h, w = video.shape
        x = video.transpose(1, 2).reshape(b * t, 3, h, w).float()
        x = (x * self.in_std.float() + self.in_mean.float() - 0.5) / 0.5                    # SigLIP's own normalisation
        hs = self.encoder(pixel_values=x.to(next(self.encoder.parameters()).dtype), output_hidden_states=True).hidden_states[-2]
        g = int(round(hs.shape[1] ** 0.5))
        hs = torch.nn.functional.adaptive_avg_pool2d(hs.float().transpose(1, 2).reshape(b * t, -1, g, g), GRID)   # 27 -> 24
        hs = hs.reshape(b, t // 2, 2, -1, GRID * GRID).permute(0, 1, 4, 2, 3)                # [B, T/2, 576, 2, D]
        return hs.reshape(b, (t // 2) * GRID * GRID, 2, -1).to(video.dtype)

    def frozen_reference(self) -> "SigLIPBackbone":
        ref = copy.deepcopy(self)
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
        ref.train = lambda mode=True: ref
        return ref


def build_vision(name: str) -> nn.Module:
    """model.vision of a config -> backbone. 'siglip_so400m' = the control encoder, anything else a V-JEPA 2.1 variant."""
    return SigLIPBackbone(name) if name.startswith("siglip") else VJEPABackbone(name)
