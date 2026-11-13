# V-JEPA 2.1 video encoders (ViT-g/16 and the distilled ViT-L/16, 384 px, RoPE), inference-only copy.
#
# Adapted from facebookresearch/vjepa2 @ 204698b:
#   app/vjepa_2_1/models/vision_transformer.py, app/vjepa_2_1/models/utils/{modules,patch_embed}.py, src/masks/utils.py
# Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the MIT License (see LICENSE_VJEPA2 in this folder).
#
# Changes from upstream (numerics of the forward pass are unchanged):
#  - only the pieces used by the video encoder are kept; the timm / einops imports are removed;
#  - scaled_dot_product_attention receives q and k cast to v's dtype. Upstream's RoPE rotation returns q, k in fp32
#    while v stays in the compute dtype; upstream always ran under autocast, which does this cast implicitly. This is
#    the same cast the JEPA-LLM training / evaluation code applies when the encoder runs in bf16 without autocast;
#  - the deprecated `torch.backends.cuda.sdp_kernel()` context (no arguments = all kernels enabled = default) is dropped.
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_masks(x, masks, concat=True):
    """x [B, N, D]; masks: list of [B, K] index tensors of the patches to KEEP."""
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    return x * x.new_empty(shape).bernoulli_(keep) / keep


def rotate_queries_or_keys(x, pos, n_registers, has_cls_first):
    B, num_heads, N, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be a multiple of 2 for block matrix rotation"

    n_cls = 1 if has_cls_first else 0
    start_ctx = n_cls
    end_ctx = N - n_registers

    x_cls = x[..., :n_cls, :] if n_cls else None
    x_ctx = x[..., start_ctx:end_ctx, :]
    x_reg = x[..., end_ctx:, :] if n_registers > 0 else None

    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega
    freq = torch.einsum("..., f -> ... f", pos, omega)

    emb_sin = freq.sin()
    emb_cos = freq.cos()

    emb_sin = emb_sin.repeat_interleave(2, dim=-1)
    emb_cos = emb_cos.repeat_interleave(2, dim=-1)

    y = x_ctx.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)

    out_ctx = (x_ctx * emb_cos) + (y * emb_sin)

    parts = []
    if n_cls:
        parts.append(x_cls)
    parts.append(out_ctx)
    if n_registers:
        parts.append(x_reg)
    return torch.cat(parts, dim=-2)


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0, wide_silu=True):
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


class RoPEAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, use_sdpa=True,
                 grid_size=14, is_causal=False, n_registers=0, has_cls_first=False, interpolate_rope=False, patch_size=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal
        self.n_registers = n_registers
        self.has_cls_first = has_cls_first
        self.interpolate_rope = interpolate_rope
        self.pretrained_patch_size = patch_size
        if patch_size == 14:
            self.pretrained_grid_size = int(252 / patch_size)
        elif patch_size == 16:
            self.pretrained_grid_size = int(256 / patch_size)

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
        else:
            tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def forward(self, x, mask=None, T=None, H_patches=None, W_patches=None, return_attn=False):
        B, N, C = x.size()
        N_ctx = N - self.n_registers
        grid_depth = int(N_ctx // (self.grid_size * self.grid_size))

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        else:
            if T is None or H_patches is None or W_patches is None:
                mask = torch.arange(int(grid_depth * self.grid_size * self.grid_size), device=x.device)
            else:
                mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)

        if self.interpolate_rope:
            if H_patches is None:
                H_patches = int(self.grid_size)
            if W_patches is None:
                W_patches = int(self.grid_size)
            h_mask = h_mask * (self.pretrained_grid_size - 1) / (H_patches - 1)
            w_mask = w_mask * (self.pretrained_grid_size - 1) / (W_patches - 1)

        s = 0
        qd = rotate_queries_or_keys(q[..., s:s + self.d_dim], pos=d_mask, n_registers=self.n_registers, has_cls_first=self.has_cls_first)
        kd = rotate_queries_or_keys(k[..., s:s + self.d_dim], pos=d_mask, n_registers=self.n_registers, has_cls_first=self.has_cls_first)
        s += self.d_dim
        qh = rotate_queries_or_keys(q[..., s:s + self.h_dim], pos=h_mask, n_registers=self.n_registers, has_cls_first=self.has_cls_first)
        kh = rotate_queries_or_keys(k[..., s:s + self.h_dim], pos=h_mask, n_registers=self.n_registers, has_cls_first=self.has_cls_first)
        s += self.h_dim
        qw = rotate_queries_or_keys(q[..., s:s + self.w_dim], pos=w_mask, n_registers=self.n_registers, has_cls_first=self.has_cls_first)
        kw = rotate_queries_or_keys(k[..., s:s + self.w_dim], pos=w_mask, n_registers=self.n_registers, has_cls_first=self.has_cls_first)
        s += self.w_dim

        if s < self.head_dim:
            q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
            k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if self.use_sdpa:
            # q, k come back from the fp32 rotation; cast to v's dtype (what autocast does upstream)
            x = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal)
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn.to(v.dtype) @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return (x, attn) if return_attn else (x, None)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, wide_silu=True, norm_layer=nn.LayerNorm, use_sdpa=True, is_causal=False, grid_size=16,
                 use_rope=False, n_registers=0, has_cls_first=False, interpolate_rope=False, patch_size=16, **kwargs):
        super().__init__()
        assert use_rope, "this copy only supports the RoPE encoder of V-JEPA 2.1"
        self.norm1 = norm_layer(dim)
        self.use_rope = use_rope
        self.attn = RoPEAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                  use_sdpa=use_sdpa, is_causal=is_causal, grid_size=grid_size, proj_drop=drop,
                                  n_registers=n_registers, has_cls_first=has_cls_first, interpolate_rope=interpolate_rope,
                                  patch_size=patch_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop)
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, T=None, H_patches=None, W_patches=None, return_attn=False, mode="video"):
        y, attn = self.attn(self.norm1(x), mask=mask, T=T, H_patches=H_patches, W_patches=W_patches, return_attn=return_attn)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return (x, attn) if return_attn else (x, None)


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=16, tubelet_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, kernel_size=(tubelet_size, patch_size, patch_size),
                              stride=(tubelet_size, patch_size, patch_size))

    def forward(self, x, **kwargs):
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    """V-JEPA 2.1 encoder. forward(x [B, 3, T, H, W], training=True) returns the hierarchical levels (ViT-g: outputs of blocks
    9 / 19 / 29 / 39, each through its own LayerNorm) concatenated on the channel axis: [B, T/2 * H/16 * W/16, 4 * 1408];
    forward(x) (training=False) returns the last block through the last LayerNorm: [B, T/2 * H/16 * W/16, embed_dim]."""

    def __init__(self, img_size=(224, 224), patch_size=16, num_frames=1, tubelet_size=2, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, init_std=0.02, out_layers=None, uniform_power=False, use_silu=False, wide_silu=True,
                 use_sdpa=True, use_activation_checkpointing=False, is_causal=False, use_rope=False, init_type: str = "default",
                 handle_nonsquare_inputs=True, img_temporal_dim_size=None, n_registers=0, has_cls_first=False,
                 interpolate_rope=False, modality_embedding=True, n_output_distillation=4, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.init_type = init_type
        self.handle_nonsquare_inputs = handle_nonsquare_inputs
        self.img_temporal_dim_size = img_temporal_dim_size
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1
        assert self.is_video and use_rope
        self.use_activation_checkpointing = use_activation_checkpointing
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        self.patch_embed = PatchEmbed3D(patch_size=patch_size, tubelet_size=tubelet_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = (num_frames // tubelet_size) * (img_size[0] // patch_size) * (img_size[1] // patch_size)
        # still-image patch embedding of the checkpoint (kept so the official weights load strictly; unused for video)
        self.patch_embed_img = (PatchEmbed3D(patch_size=patch_size, tubelet_size=1, in_chans=in_chans, embed_dim=embed_dim)
                                if img_temporal_dim_size is not None else None)
        self.uniform_power = uniform_power
        self.use_rope = use_rope
        self.blocks = nn.ModuleList([
            Block(use_rope=use_rope, grid_size=img_size[0] // patch_size, grid_depth=num_frames // tubelet_size, dim=embed_dim,
                  num_heads=num_heads, mlp_ratio=mlp_ratio, use_sdpa=use_sdpa, is_causal=is_causal, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, act_layer=nn.SiLU if use_silu else nn.GELU, wide_silu=wide_silu,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, n_registers=n_registers,
                  has_cls_first=has_cls_first, interpolate_rope=interpolate_rope, patch_size=patch_size)
            for i in range(depth)])
        self.attn_out = False
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()
        levels = {12: [2, 5, 8, 11], 24: [5, 11, 17, 23], 40: [9, 19, 29, 39], 48: [11, 23, 37, 47]}[depth]
        self.hierarchical_layers = levels
        self.out_layers_distillation = levels if n_output_distillation == 4 else [levels[-1]]
        self.norms_block = nn.ModuleList([norm_layer(embed_dim) for _ in range(len(self.hierarchical_layers))])
        self.cls_token = None
        self.return_hierarchical = False
        self.modality_embedding = False
        if modality_embedding:
            self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.modality_embedding = True

    def _init_weights(self, m):
        # only matters for a model built without weights; the released weights overwrite everything
        if isinstance(m, nn.LayerNorm):
            if m.weight is not None and not m.weight.is_meta:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            return
        if isinstance(m, (nn.Linear, nn.Conv3d)) and not m.weight.is_meta:
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        for layer_id, layer in enumerate(self.blocks):
            if not layer.attn.proj.weight.is_meta:
                layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
                layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def check_temporal_dim(self, shape) -> bool:
        return self.img_temporal_dim_size is not None and shape[2] == self.img_temporal_dim_size

    def forward(self, x, masks=None, training=False):
        if masks is not None and not isinstance(masks, list):
            masks = [masks]
        _, _, T, H, W = x.shape
        T = T if self.check_temporal_dim(x.shape) else T // self.tubelet_size
        H_patches, W_patches = H // self.patch_size, W // self.patch_size
        if not self.handle_nonsquare_inputs:
            T = H_patches = W_patches = None
        if self.check_temporal_dim(x.shape):
            x = self.patch_embed_img(x)
            mode = "img"
            if self.modality_embedding:
                x += self.img_mod_embed.repeat(x.shape[0], 1, 1)
        else:
            x = self.patch_embed(x)
            mode = "video"
            if self.modality_embedding:
                x += self.video_mod_embed.repeat(x.shape[0], 1, 1)
        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)
        outs, hier = [], []
        for i, blk in enumerate(self.blocks):
            x, _ = blk(x, mask=masks, T=T, H_patches=H_patches, W_patches=W_patches, return_attn=self.attn_out, mode=mode)
            if self.out_layers is not None and i in self.out_layers:
                outs.append(self.norms_block[self.hierarchical_layers.index(i)](x))
            if i in self.out_layers_distillation:
                hier.append(self.norms_block[self.hierarchical_layers.index(i)](x))
        if self.out_layers is not None:
            return outs
        if training or self.return_hierarchical:
            return torch.cat(hier, dim=2)
        return self.norms_block[-1](x)


def vjepa2_1_vit_giant_384(**kw):
    """The encoder of `vjepa2_1_vit_giant_384` (checkpoint vjepa2_1_vitg_384.pt, key target_encoder): ViT-g/16 with 22
    heads ("xformers" layout), RoPE with interpolation, 384 x 384 input, tubelet 2, 4 output levels."""
    args = dict(patch_size=16, img_size=(384, 384), num_frames=64, tubelet_size=2, use_sdpa=True, uniform_power=False,
                use_rope=True, img_temporal_dim_size=1, interpolate_rope=True, embed_dim=1408, depth=40, num_heads=22,
                mlp_ratio=48 / 11, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), n_output_distillation=4)
    args.update(kw)
    return VisionTransformer(**args)


def vjepa2_1_vit_large_384(**kw):
    """The encoder of `vjepa2_1_vit_large_384` (checkpoint vjepa2_1_vitl_dist_vitG_384.pt, key ema_encoder): ViT-L/16
    distilled from V-JEPA 2.1 ViT-G, 16 heads, RoPE with interpolation, 384 x 384 input, tubelet 2. Only its last level is
    trained (the output of block 23 through norms_block[3]); upstream builds it with the default n_output_distillation
    = 4, but JEPA-LLM reads it with training=False (the last level only), so n_output_distillation does not matter here."""
    args = dict(patch_size=16, img_size=(384, 384), num_frames=64, tubelet_size=2, use_sdpa=True, uniform_power=False,
                use_rope=True, img_temporal_dim_size=1, interpolate_rope=True, embed_dim=1024, depth=24, num_heads=16,
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), n_output_distillation=4)
    args.update(kw)
    return VisionTransformer(**args)


VARIANTS = {"vit_giant_384": vjepa2_1_vit_giant_384, "vit_large_384": vjepa2_1_vit_large_384}
