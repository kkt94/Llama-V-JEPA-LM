"""Native predictive loss for stage 3 (spec 6.6), following the official V-JEPA 2.1 cooldown recipe where it applies.

Masks: the official multi-block 3D mask generator (src.masks.multiseq_multiblock3d._MaskGenerator) with the cooldown
config: 8 blocks of spatial scale 0.15 and 2 blocks of 0.7, aspect 0.75-1.5, full temporal extent.
L_dense (g1B, where predictor and encoder share the 4-level space): target = per-level LayerNorm of the frozen reference
encoder E0 on the full clip; L1 on masked tokens + 0.5 x L1 on context tokens (official: loss_exp 1, lambda_value_vid 0.5,
weight_distance_loss false, normalize_predictor false at cooldown).
L_ref: L1 between the online predictor and the frozen reference predictor P0(E0(.)) on the same masks (target outputs).
Distilled B/L predict in the 2B teacher space and their distillation recipe is not public, so for them L_native = L_ref.
"""
import torch
import torch.nn.functional as F

from jepa_llm.models.vision_backbone import VJEPA_CODE_DIR, _backbones  # noqa: F401  (puts vjepa2 on sys.path)

COOLDOWN_MASKS = [dict(spatial=(0.15, 0.15), npred=8), dict(spatial=(0.7, 0.7), npred=2)]


def make_mask_generators(num_frames: int, crop: int = 384):
    _backbones()
    from src.masks.multiseq_multiblock3d import _MaskGenerator
    return [_MaskGenerator(crop_size=crop, num_frames=num_frames, spatial_patch_size=16, temporal_patch_size=2,
                           spatial_pred_mask_scale=m["spatial"], temporal_pred_mask_scale=(1.0, 1.0),
                           aspect_ratio=(0.75, 1.5), npred=m["npred"], max_context_frames_ratio=1.0)
            for m in COOLDOWN_MASKS]


def _levels_ln(x, levels):
    b, n, d = x.shape
    return F.layer_norm(x.view(b, n, levels, d // levels), (d // levels,)).view(b, n, d)


def _gather(x, idx):
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def native_loss(online, reference, video, generators, lambda_ctx: float = 0.5, alpha: float = 0.1):
    """online / reference: VJEPABackbone. Returns dict(loss, dense, ref). Gradients reach only the online E/P."""
    spec = online.spec
    dense_ok = spec.n_levels == spec.pred_levels and spec.level_dim == spec.pred_dim   # same space (g1B)
    b = video.shape[0]
    if dense_ok:
        with torch.no_grad():
            target = _levels_ln(reference.encoder(video, training=True).float(), spec.n_levels)
    l_dense, l_ref, n = video.new_zeros(()).float(), video.new_zeros(()).float(), 0
    for gen in generators:
        enc, pred = gen(b)
        enc = torch.stack(enc).to(video.device) if isinstance(enc, list) else enc.to(video.device)
        pred = torch.stack(pred).to(video.device) if isinstance(pred, list) else pred.to(video.device)
        z = online.encode_masked_context(video, enc)
        zp, zc = online.predictor(z, enc, pred, mod="video")
        with torch.no_grad():
            p0, _ = reference.predictor(reference.encode_masked_context(video, enc), enc, pred, mod="video")
        l_ref = l_ref + (zp.float() - p0.float()).abs().mean()
        if dense_ok:
            l_dense = l_dense + (zp.float() - _gather(target, pred)).abs().mean()
            if zc is not None:
                l_dense = l_dense + lambda_ctx * (zc.float() - _gather(target, enc)).abs().mean()
        n += 1
    l_dense, l_ref = l_dense / n, l_ref / n
    total = (l_dense + alpha * l_ref) if dense_ok else l_ref
    return {"loss": total, "dense": l_dense.detach(), "ref": l_ref.detach(), "dense_used": dense_ok}
