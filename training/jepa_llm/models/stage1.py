"""Stage 1 (visual evidence readout, spec 5.3): frozen V-JEPA E/P, full fine-tuning of the LLM, bridge B and retention R.

    video -> E (frozen, no grad) -> level-normalised, 4x4-pooled evidence -> B -> evidence embeddings spliced into the
    LLM input at vis_mask positions -> answer CE  (+ lambda_state * R loss on evidence positions)

In stage 1 the online encoder is frozen and identical to the reference E0, so the retention target is the same pooled
evidence under stop-gradient (no second encoder pass). The B0 baseline is this model with lambda_state = 0 and no R.

FSDP2 note: parameters are gathered and gradients reduced only around a module's forward(). Embedding lookup, evidence
splice, decoder and the answer-position LM head therefore all run inside LanguageCore.forward, which matters for tied
embeddings (Llama 3.2 1B/3B). Only answer positions go through the LM head, so [B, S, vocab] logits never exist.
"""
import os
import math
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn

from .evidence_bridge import (OBSERVED, ContrastiveCritic, EqualisedStateHeads, Equaliser, EvidenceBridge, GeometryHead, RetentionHeads,
                              SpectralBridge, normalize_levels, pool_regions)


class VisGate:
    """Which positions of the sequence being processed are video tokens ([B, S] bool, or None = text only)."""
    mask = None
    span = None        # (start, end) when the video tokens are the same contiguous block in every row (set with the mask)

    def set(self, mask):
        self.mask, self.span = mask, None
        if mask is not None and mask.dim() == 2 and bool((mask == mask[:1]).all()):
            idx = mask[0].nonzero()[:, 0]
            if idx.numel() and int(idx[-1] - idx[0]) + 1 == idx.numel():
                self.span = (int(idx[0]), int(idx[-1]) + 1)


class GatedLinear(nn.Linear):
    """A linear map of the LLM whose weights can only be taught by the video positions.

    Use failure, as the project defines it: the answer follows the language prior instead of the video. A text loss
    lowers itself first by teaching the LLM the data's wording (91% of the LLM's update lay in the directions text
    uses; at 3000 steps removing the video changed the caption loss by 0.05 nats). Here no weight of the LLM can be
    updated by what happens at a TEXT position:
      mode "copy":   text positions run the original, frozen weight; video positions run their own trainable copy
                     (initialised to it). Without video the LLM is exactly the pretrained LLM, so the loss can only go
                     down through what the video positions compute.
      mode "shared": one weight; text positions use it without contributing to its gradient.
    Same number of matrix products as a plain linear map."""

    def __init__(self, base: nn.Linear, gate: VisGate, mode: str):
        super().__init__(base.in_features, base.out_features, bias=False, device=base.weight.device, dtype=base.weight.dtype)
        assert base.bias is None and mode in ("copy", "shared")
        self.weight = base.weight
        self._gate, self.mode = [gate], mode                   # in a list: not a submodule
        if mode == "copy":
            self.weight_v = nn.Parameter(base.weight.detach().clone())
            self.weight.requires_grad_(False)

    def forward(self, x):
        m = self._gate[0].mask
        if m is None or m.shape != x.shape[:-1]:
            return F.linear(x, self.weight)
        wv, wt = (self.weight_v, self.weight) if self.mode == "copy" else (self.weight, self.weight.detach())
        g = self._gate[0]
        if g.span is not None:                                 # the video tokens are one block at the same place in every row: slices, no copies
            a, b = g.span
            return torch.cat([F.linear(x[:, :a], wt), F.linear(x[:, a:b], wv), F.linear(x[:, b:], wt)], 1)
        yv, yt = F.linear(x[m], wv), F.linear(x[~m], wt)
        y = yv.new_empty(*x.shape[:-1], self.out_features)
        y[m] = yv
        y[~m] = yt.to(yv.dtype)
        return y


class _DynRecon(nn.Module):
    """(10-20) dynamics reconstruction: one linear map from the LLM's answer-time state (mean final state over the answer positions)
    to the clip's FIXED V-JEPA state - every moment x 2 x 2 screen regions x the top `dirs` equalised directions (unit variance each).
    The contrast (dc_L / dc2_L) learned its own target and shrank it to what was easiest (moment order): solved at 100%, dive information in
    the LLM state unchanged (Diving48 16.8 -> 18). A fixed target cannot shrink and a regression does not saturate."""
    def __init__(self, hidden, n_tub, dirs, regions=2, target="top", kin=2048, scale=1.0):
        super().__init__(); self.n_tub, self.dirs, self.reg = n_tub, dirs, regions
        # (10-23) target "randproj": a FIXED random orthonormal projection of ALL kin equalised directions to `dirs` (seed 0, not a parameter,
        # identical on every rank / resume) instead of the top `dirs` directions. The top directions carry the least task information
        # (10-11: dropping the top 256 barely changes the answer loss; action / physics QA information lies outside them); in the
        # equalised space every direction has unit variance, so a random projection weighs all bands alike (a sketch of reconstructing
        # all of them). target "both": the top `dirs` directions AND the projection (2 x dirs per region): the reconstruction that built
        # the current strengths is kept, the projection adds the bands the answer state does not hold (nx2_Q@32000 answer state, ridge
        # R^2 held out: top 0.52, projection 0.09; logs/zz_dyn_target_recov.py). Training only; dropped at inference.
        self.target, self.scale = target, float(scale)      # scale: the projection's variance matched to the top target's (logs/zz_dyn_target_var.py)
        self._proj = None
        if target in ("randproj", "both"):
            q, _ = torch.linalg.qr(torch.randn(kin, dirs, generator=torch.Generator().manual_seed(0)))
            self._proj = q.contiguous()
        self.d_out = 2 * dirs if target == "both" else dirs
        self.rec = nn.Linear(hidden, n_tub * regions * regions * self.d_out)
        nn.init.zeros_(self.rec.weight); nn.init.zeros_(self.rec.bias)

    def forward(self, x, ans, n_tub):
        """x [B, K, kin] equalised tokens (moment, row, column of 8 x 8 cells), ans [B, H] -> (mean squared error, R^2 [top or all, projection
        part for "both" else 0])"""
        b = x.shape[0]; g = int(round((x.shape[1] // n_tub) ** 0.5)); r = self.reg
        parts = []
        if self.target in ("top", "both"):
            parts.append(x[..., : self.dirs].float())
        if self._proj is not None:
            parts.append((x.float() @ self._proj.to(x.device)) * self.scale)
        xs = parts[0] if len(parts) == 1 else torch.cat(parts, -1)
        t = xs.reshape(b, n_tub, r, g // r, r, g // r, self.d_out).mean((3, 5)).float()          # [B, moments, r, r, d_out]
        pred = self.rec(ans.to(self.rec.weight.dtype)).float().view_as(t)
        err = (pred - t).pow(2)
        mse = err.mean()
        with torch.no_grad():
            if self.target == "both":
                r2 = torch.stack([1 - err[..., k].mean() / t[..., k].var(0, unbiased=False).mean().clamp_min(1e-6)
                                  for k in (slice(0, self.dirs), slice(self.dirs, None))])
            else:
                r2v = 1 - mse / t.var(0, unbiased=False).mean().clamp_min(1e-6)
                r2 = torch.stack([r2v, torch.zeros_like(r2v)])
        return mse, r2


class _DynMask(nn.Module):
    """(10-25) masked-moment dynamics prediction, the JEPA objective inside the LLM. The reconstruction (_DynRecon) was measured on nx5_Q@20000
    (logs/zz_dyn_diag_an.py): 65 % of its target is each clip's STATIC content (R^2 0.71) and the dynamic part (deviation over moments) is
    reconstructed at R^2 0.15 with 41 % of its amplitude - it copied what the LLM already reads, appearance first. Here, in the same forward,
    `moments` whole moments (all their cells) of a fraction `p_row` of the rows enter as the bridge's mask token (positions / time codes kept);
    the LLM's layer-`layer` state at each masked cell predicts that cell's equalised V-JEPA state MINUS the cell's mean over the clip (dynamics
    only, every direction unit-scaled by equalisation). Masked moments are drawn from 1..T-1: the causal LLM must infer them from the moments
    before. One linear map, zero at start (= predicts 'no change'). Training only: no mask and no head at inference."""
    def __init__(self, hidden, kin=2048, layer=18, moments=2, p_row=0.5):
        super().__init__(); self.layer, self.moments, self.p_row = int(layer), int(moments), float(p_row)
        self.rec = nn.Linear(hidden, kin)
        nn.init.zeros_(self.rec.weight); nn.init.zeros_(self.rec.bias)

    def draw(self, b, n_tub, cells, device):
        """-> content mask [B, n_tub * cells] (moment-major token order, as pool_regions)"""
        m = torch.zeros(b, n_tub, dtype=torch.bool)
        for i in range(b):
            if torch.rand(()) < self.p_row:
                m[i, 1 + torch.randperm(n_tub - 1)[: self.moments]] = True
        if not bool(m.any()):                    # at least one row per batch: the head always runs through its own forward (FSDP unshards there)
            m[int(torch.randint(b, ())), 1 + torch.randperm(n_tub - 1)[: self.moments]] = True
        return m.repeat_interleave(cells, dim=1).to(device)

    def forward(self, x, h, mask, n_tub):
        """x [B, K, kin] equalised tokens (fixed), h [B, K, H] LLM layer states at the evidence positions, mask [B, K] -> (mse, [R^2 vs 'no change', masked share])"""
        b, k, d = x.shape
        xd = (x.float().view(b, n_tub, k // n_tub, d) - x.float().view(b, n_tub, k // n_tub, d).mean(1, keepdim=True)).view(b, k, d)
        t = xd[mask]; p = self.rec(h[mask].to(self.rec.weight.dtype)).float()
        mse = (p - t).pow(2).mean()
        with torch.no_grad():
            r2 = 1 - mse / t.pow(2).mean().clamp_min(1e-6)
        return mse, torch.stack([r2, mask.float().mean()])


class _DynTime(nn.Module):
    """(10-25) dynamics losses whose answers ARE in the input but need the time axis (after _DynMask: a masked moment's dynamic state is not
    predictable even from V-JEPA's own other moments, held-out linear R^2 <= 0.01, logs/zz_dyn_diag). Targets: the equalised V-JEPA state on the
    top `dirs` directions + a fixed random projection of all directions (as _DynRecon "both"), fixed, no gradient.
      A  change reading - at every evidence position of moments 1..T-1, LLM layer `layer` predicts its cell's change from the previous moment
         (z_t - z_{t-1}). Held-out linear R^2 of that change: 0.14 from the current moment alone, 1.00 from both -> it pays to attend back
         64 tokens to the same cell one moment earlier (the LLM gave such positions ~0.1 % of its attention).
      B  dynamic summary - from the mean layer-`layer` state over the answer positions, the clip's states over moments x regions x regions
         MINUS each region's clip mean (the old reconstruction spent 65 % of its target on static content; dynamic part R^2 0.15).
    Each term is its MSE over the batch target variance (= 1 - R^2), loss = their mean. Training only."""
    def __init__(self, hidden, n_tub, kin=2048, layer=18, dirs=128, regions=4, scale=1.0, summary=True):
        super().__init__(); self.n_tub, self.layer, self.dirs, self.reg, self.scale = n_tub, int(layer), int(dirs), int(regions), float(scale)
        self.summary = bool(summary)    # (10-25) B off (nx8_Q): on nx7_Q its R^2 stayed at 0.03 for 1500 steps while the answer loss ran 0.02-0.035 above nx5
        q, _ = torch.linalg.qr(torch.randn(kin, dirs, generator=torch.Generator().manual_seed(0))); self._proj = q.contiguous()
        self.rec_a = nn.Linear(hidden, 2 * dirs); self.rec_b = nn.Linear(hidden, n_tub * regions * regions * 2 * dirs) if self.summary else None
        for l_ in [m_ for m_ in (self.rec_a, self.rec_b) if m_ is not None]:
            nn.init.zeros_(l_.weight); nn.init.zeros_(l_.bias)

    def forward(self, x, h, ans, n_tub):
        """x [B, K, kin] equalised tokens (moment-major), h [B, K, H] layer states at the evidence positions, ans [B, H] mean layer state over the
        answer positions -> (loss, [R^2 change, R^2 dynamic summary])"""
        b, k, _ = x.shape; g = int(round((k // n_tub) ** 0.5)); r = self.reg
        xs = torch.cat([x[..., : self.dirs].float(), (x.float() @ self._proj.to(x.device)) * self.scale], -1).view(b, n_tub, k // n_tub, -1)
        ta = xs[:, 1:] - xs[:, :-1]
        pa = self.rec_a(h.view(b, n_tub, k // n_tub, -1)[:, 1:].to(self.rec_a.weight.dtype)).float()
        ua = (pa - ta).pow(2).mean() / ta.pow(2).mean().detach().clamp_min(1e-6)
        if not self.summary:
            return ua, torch.stack([1 - ua.detach(), torch.zeros_like(ua).detach()])
        tb = xs.view(b, n_tub, r, g // r, r, g // r, -1).mean((3, 5)); tb = tb - tb.mean(1, keepdim=True)
        pb = self.rec_b(ans.to(self.rec_b.weight.dtype)).float().view_as(tb)
        ua = (pa - ta).pow(2).mean() / ta.pow(2).mean().detach().clamp_min(1e-6)
        ub = (pb - tb).pow(2).mean() / tb.pow(2).mean().detach().clamp_min(1e-6)
        return 0.5 * (ua + ub), torch.stack([1 - ua.detach(), 1 - ub.detach()])


class _DynCorr(nn.Module):
    """(10-25) causal correspondence distillation. Dynamics are relations between tokens, not token values: the dynamic part of a V-JEPA state is
    high-dimensional and not predictable from the other moments (held-out linear R^2 <= 0.01), whereas V-JEPA's own temporal correspondence tells
    where content moved - on SSv2 the mean displacement from cosine matches separates object left/right pushes at 88 % and camera pans at 98-99 %
    (logs/zz_corr_valid.py). Teacher (fixed): for every evidence token (t, c) and EVERY earlier moment s < t, the distribution over the cells c' of
    moment s of softmax(cos(z_tc, z_sc') / tau), z = equalised V-JEPA tokens minus the clip mean. Student: from the LLM's FINAL state at that token
    (positions with no language target), one linear map gives, per lag t - s, logits over the cells. Loss = mean KL(teacher || student). One token's
    own V-JEPA state explains ~25 % of its displacement; the rest needs content-based attention to earlier moments (a tracker inside the LLM).
    No layer, token subset or lag is chosen: final representation, all tokens, all earlier moments (what the causal LLM can see). Training only."""
    def __init__(self, hidden, n_tub, cells, tau=0.05, prior=None):
        super().__init__(); self.n_tub, self.cells, self.tau = int(n_tub), int(cells), float(tau)
        # (logging only) the content-free baseline: the mean teacher distribution per (t, s, cell) over training clips (logs/zz_corr_prior.py).
        # A student that ignores the video reaches it from position codes alone (70 % of the KL vs uniform at lag 1), so progress is reported
        # against it: explained = 1 - KL / KL(teacher || position prior).
        self._logprior = torch.load(prior, map_location="cpu")["prior"].clamp_min(1e-8).log() if prior else None
        self.rec = nn.Linear(hidden, (self.n_tub - 1) * self.cells)          # per lag 1..T-1: logits over the cells of that earlier moment
        nn.init.zeros_(self.rec.weight); nn.init.zeros_(self.rec.bias)          # starts uniform
        t_, s_ = torch.tril_indices(self.n_tub, self.n_tub, -1)                  # all pairs s < t
        self.register_buffer("_t", t_, persistent=False); self.register_buffer("_s", s_, persistent=False)

    def forward(self, x, h, n_tub):
        """x [B, K, kin] equalised tokens (moment-major), h [B, K, H] final LLM states at the evidence positions -> (mean KL, [explained lag 1,
        explained all lags]) where explained = 1 - KL / KL(teacher || uniform)"""
        b, k, d = x.shape; T, C = n_tub, k // n_tub
        with torch.no_grad():
            f = F.normalize(x.float() - x.float().mean(1, keepdim=True), dim=-1).view(b, T, C, d)
            S = torch.einsum("btcd,bsed->btsce", f, f)[:, self._t, self._s]       # [B, P, C, C] pairs (t, s<t)
            A = (S / self.tau).softmax(-1); logA = (S / self.tau).log_softmax(-1)
            ku = (A * (logA + math.log(C))).sum(-1)                            # KL(teacher || uniform)
        lg = self.rec(h.to(self.rec.weight.dtype)).float().view(b, T, C, T - 1, C).permute(0, 1, 3, 2, 4)   # [B, t, lag, c, c']
        lp = lg[:, self._t, self._t - self._s - 1].log_softmax(-1)                 # [B, P, C, C] (adjacent index arrays keep the batch axis first)
        kl = (A * (logA - lp)).sum(-1)                                          # [B, P, C]
        with torch.no_grad():
            l1 = (self._t - self._s) == 1
            if self._logprior is not None:
                if self._logprior.device != A.device: self._logprior = self._logprior.to(A.device)
                ku = (A * (logA - self._logprior.unsqueeze(0))).sum(-1)              # KL(teacher || position prior)
            ex1 = 1 - kl[:, l1].mean() / ku[:, l1].mean().clamp_min(1e-6); exa = 1 - kl.mean() / ku.mean().clamp_min(1e-6)
        return kl.mean(), torch.stack([ex1, exa])


class _RecLang(nn.Module):
    """(10-25) reconstruction v2: the LANGUAGE stream must hold V-JEPA's whole output, at every depth. Measured on nx5_Q@20000 (logs/zz_rec_where*):
    the old head (mean final answer state -> 16 moments x 4x4 regions, top + projected directions) was satisfied by a last-minute lookup - text
    positions hold the clip at R^2 0.05-0.09 up to layer 31 and 0.23-0.25 only at layers 34-36 - and a single mean vector cannot carry per-moment
    detail at all (even the evidence tokens' own mean: dynamic R^2 0.01). Here: target = every evidence token, all equalised directions (equal
    weight); readout = the text positions after the video (question + answer), never the evidence positions; one layer drawn uniformly per step
    (an unbiased estimate of the mean over all layers - no layer is chosen, and a last-minute lookup does not satisfy it); a small cross-attention
    decoder (one learned query per evidence token + a layer embedding) reads that layer's text states. Loss = MSE / target variance (1 - R^2).
    Training only: the decoder and the extra layer hooks are not used at inference."""
    def __init__(self, hidden, n_tok, n_tub, n_layers, kin=2048, d=512, heads=8, layer=0, pyramid=False, norm="ln", linear=False):
        super().__init__(); self.n_tok, self.n_tub, self.n_layers, self.heads, self.d = int(n_tok), int(n_tub), int(n_layers), int(heads), int(d)
        # (10-25) layer: 0 = one layer drawn uniformly per step (nx10_Q: did not learn - R^2 < 0 at 1400 steps; the text positions take in the video only
        # in the late layers, so the early ones cannot hold it); n_layers = the FINAL layer only (where the old reconstruction learned steadily)
        self.fixed = int(layer)
        # (10-25) pyramid: the loss is taken on every dyadic space-time pooling of prediction and target (time 16/8/4/2/1 x space 8x8/4x4/2x2/1x1),
        # each over its own variance, averaged. nx10/nx11: full resolution only (2M values per clip from a few dozen text positions) - the decoder
        # settled on the mean (R^2 identical to 3 decimals whatever it read). Coarse levels are within what the language stream can hold (the old
        # reconstruction learned at 16 x 4x4), fine levels keep asking for detail; no single resolution or direction subset is chosen.
        self.pyramid = bool(pyramid)
        self.q = nn.Parameter(torch.randn(self.n_tok, d) * 0.02); self.lemb = nn.Embedding(self.n_layers + 1, d); nn.init.normal_(self.lemb.weight, std=0.02)
        # (10-25) input normalisation of the LLM states: "ln" (nx10/nx11: LayerNorm - the few huge channels of Qwen's hidden states set its scale and
        # flatten the informative ones; on frozen nx5 states the decoder stayed below R^2 0 while a linear head reached 0.26), "none", or "bn" (each
        # channel standardised over the text positions of the batch, running statistics at evaluation - as the ridge probes that read the clip)
        self.norm_mode = str(norm)
        self.norm = nn.LayerNorm(hidden) if self.norm_mode == "ln" else nn.BatchNorm1d(hidden, affine=False, momentum=0.01) if self.norm_mode == "bn" else nn.Identity()
        self.kv = nn.Linear(hidden, 2 * d)
        if linear:
            # a LINEAR read-out instead of
            # the 2-layer MLP: the MLP could decode information the LLM kept in a tangled, non-linear form that lm_head - a linear read-out of
            # the same states - cannot use; with one linear map the preserved information has to be linearly explicit, as the answer reads it
            self.out = nn.Sequential(nn.Linear(d, kin))
            nn.init.zeros_(self.out[0].weight); nn.init.zeros_(self.out[0].bias)
        else:
            self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, kin))
            nn.init.zeros_(self.out[3].weight); nn.init.zeros_(self.out[3].bias)   # starts at the mean (loss 1)
        self.want = 0
        self.vp = None                                 # (10-27) video-position preservation maps (Stage1Model dyncon.vpres), set by the model

    def pick(self):
        self.want = self.fixed if self.fixed else int(torch.randint(1, self.n_layers + 1, ())); return self.want

    def forward(self, x, h, tmask, layer, vp=None):
        """x [B, K, kin] equalised tokens (fixed target), h [B, S, H] states of decoder layer `layer` (1-based), tmask [B, S] text positions after
        the video -> (1 - R^2 over all, [R^2 all, R^2 static (moment mean), R^2 dynamic (deviation over moments)])
        vp (10-27): (target [B, K, kin], [states at the video positions [B, K, H] per preservation map]) -> also (preservation loss, [mean R^2,
        finest R^2] per map); inside this forward so the sharded maps are gathered with the head"""
        if vp is not None:
            loss, acc = self.forward(x, h, tmask, layer)
            # (10-27) recomputed in backward: per read point only its input state is kept (the fp32 prediction and the pyramid's
            # intermediates were ~0.18 GB each; 36 read points ran out of memory)
            r = [torch.utils.checkpoint.checkpoint(lambda hh, m=m: _pyramid(vp[0].float(), m(hh.to(m.weight.dtype)).float(), self.n_tub), hv, use_reentrant=False)
                 for m, hv in zip(self.vp, vp[1])]
            return loss, acc, torch.stack([l_ for l_, _ in r]).mean(), torch.cat([a_[[0, 3]] for _, a_ in r])
        b, s, _ = h.shape; k = x.shape[1]; hd = self.d // self.heads
        if self.norm_mode == "bn":                     # statistics from the text positions only (padding / evidence positions excluded)
            hn = torch.zeros(b, s, h.shape[-1], device=h.device, dtype=torch.float32); hn[tmask] = self.norm(h[tmask].float())
        else:
            hn = self.norm(h.to(self.kv.weight.dtype))
        kv = self.kv(hn.to(self.kv.weight.dtype)); kk, vv = kv.view(b, s, 2, self.heads, hd).permute(2, 0, 3, 1, 4)
        q = (self.q + self.lemb.weight[layer - 1]).view(1, k, self.heads, hd).permute(0, 2, 1, 3).expand(b, -1, -1, -1).to(kk.dtype)
        o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=tmask[:, None, None, :].to(kk.device))
        pred = self.out(o.transpose(1, 2).reshape(b, k, self.d)).float()
        t = x.float(); kl_extra = None
        # (11-02) fw: per-direction weights from the answer loss's sensitivity to V-JEPA's input (sqrt of the mean squared answer gradient per
        # principal direction, mean 1): prediction and target compared in the answer loss's own (first-order) metric, so the dynamics loss asks
        # for the parts of V-JEPA's output that change the answer. None = every direction equal (as before)
        fw_ = getattr(self, "fw", None)
        if fw_ is not None:
            fw_ = fw_.to(pred.device, torch.float32)
            if fw_.dim() == 2:                                   # full metric: M^(1/2) of the answer-gradient covariance (symmetric) - rotated directions
                pred = pred @ fw_; t = t @ fw_
            else:
                pred = pred * fw_; t = t * fw_
        # (11-02) roww [b]: per-row weights of the dynamics loss (the row's own answer loss / batch mean, set by the model when dyncon.cew):
        # the dynamics loss pressed where the answer is still wrong - what the answers still lack - and released where it is right
        rw_ = getattr(self, "roww", None); self.roww = None
        wmean = (lambda z: z.mean()) if rw_ is None else (lambda z: (z.reshape(z.shape[0], -1).mean(1) * rw_.to(z.device, z.dtype)).mean())
        if getattr(self, "objective", "mse") == "nce_euclid" and getattr(self, "_tall", None) is not None:
            # (11-02) the same candidates and soft V-JEPA target as nce_global, scored by EUCLIDEAN distance instead of cosine. Across the
            # four cosine runs (nce, nce_soft, ncg, ncf; 10 run x checkpoint pairs) the categories that fell were the ones carried by what all
            # moments share - object existence 9/10 below the controls, PerceptionTest semantics 9/10, abstraction 9/10 - and the ones that
            # rose were relational (antonym 0/10 below, character order 1/10, MotionBench 1/10). Cosine discards the norm and the shared
            # component; rec2's MSE (Euclidean) kept that content and did not fall there. logit(k) = -||y - x_k||^2 / (2 tau n), n = mean
            # ||x||^2 of the batch's targets (detached): for unit vectors this is exactly cos / tau (+ const), so only the metric changes.
            # -log p(self) = ||y - x_self||^2 / (2 tau n) + log-partition: the MSE (content) and the contrast (relations) are one likelihood.
            T_ = self.n_tub; C_ = k // T_; tau = float(getattr(self, "tau", 0.05))
            P_ = pred.view(b, T_, C_, -1); X_ = t.view(b, T_, C_, -1)
            ta = self._tall.view(self._tall.shape[0], T_, C_, -1).float(); own = self._own; self._tall = None
            n_ = X_.pow(2).sum(-1).mean().detach().clamp_min(1e-6); s_ = 1.0 / (2 * tau * n_)
            def dist_(A, B_):                                                     # A [b, T, C, D] vs own clip [b, U, C, D] and all clips [J, T, C, D]
                a2 = A.pow(2).sum(-1)
                dm = a2[..., None] + B_.pow(2).sum(-1).transpose(1, 2)[:, None] - 2 * torch.einsum("btcd,bucd->btcu", A, B_)
                dc = a2[..., None] + ta.pow(2).sum(-1).permute(1, 2, 0)[None] - 2 * torch.einsum("btcd,jtcd->btcj", A, ta)
                return dm, dc
            dm_s, dc_s = dist_(P_, X_)
            with torch.no_grad():
                dm_t, dc_t = dist_(X_, X_)
                m_ = torch.zeros(dc_s.shape[-1], dtype=torch.bool, device=dc_s.device); m_[own: own + b] = True
            logits = torch.cat([-dm_s, (-dc_s).masked_fill(m_, float("-inf"))], -1) * s_
            with torch.no_grad():
                q_ = (torch.cat([-dm_t, (-dc_t).masked_fill(m_, float("-inf"))], -1) * s_).softmax(-1)
            lp_ = logits.log_softmax(-1).masked_fill(q_ == 0, 0.0)
            loss = (q_ * (q_.clamp_min(1e-12).log() - lp_)).sum(-1).mean()
            # (11-02 v2) the softmax over candidates is blind to what all candidates share (||y||^2 and any component common to them
            # cancel; smoke test: R^2 -0.07 -> -2.2 in 30 steps), and that shared part is the content that fell in the cosine runs. The
            # SAME Gaussian (mean y, variance sigma^2 = tau n) also gives the density of the true token: ||y - x_self||^2 / (2 sigma^2) per
            # token. Identity (which candidate) + density (where exactly) = the joint likelihood of one model, same units, no weight.
            lab = torch.arange(T_, device=pred.device).view(1, T_, 1).expand(b, T_, C_)
            dens_ = (dm_s.gather(-1, lab.unsqueeze(-1)).squeeze(-1) * s_).mean()
            loss = loss + dens_
            with torch.no_grad():
                ht = (dm_s.argmin(-1) == lab).float().mean(); pos_ = dm_s.gather(-1, lab.unsqueeze(-1))
                hc = (pos_ < dc_s.masked_fill(m_, float("inf")).min(-1, keepdim=True).values).float().mean()
                r2 = 1 - (pred - t).pow(2).mean() / (t - t.mean((0, 1), keepdim=True)).pow(2).mean().clamp_min(1e-6)
            return loss, torch.stack([loss.detach() - dens_.detach(), dens_.detach(), ht, hc, r2])   # [KL, density, acc moments, acc clips, R^2]
        if getattr(self, "objective", "mse") in ("nce_global", "mse_ncf") and getattr(self, "_tall", None) is not None:
            # (11-02) ONE loss, no weights: the LLM's reconstruction must reproduce V-JEPA's own similarity distribution of each token over
            # (a) the same cell at every moment of its clip (relations: order, direction, counts) and (b) the same cell and moment of EVERY
            # clip of the global batch (content: what is there - with the 5 local clips of nce_soft that was too easy, acc 0.99, and what all
            # moments share was lost: PT memory -8, object existence -4.5). Target = softmax(cos(x, x') / tau), student = softmax(cos(pred,
            # x') / tau), same tau, so an exact copy is the optimum; KL(target || student). The balance between (a) and (b) is V-JEPA's own.
            T_ = self.n_tub; C_ = k // T_; tau = float(getattr(self, "tau", 0.05))
            pn = F.normalize(pred, dim=-1).view(b, T_, C_, -1); tn = F.normalize(t, dim=-1).view(b, T_, C_, -1)
            ta = self._tall.view(self._tall.shape[0], T_, C_, -1); own = self._own; self._tall = None
            if fw_ is not None:                                  # the batch's unit vectors in the same weighted metric (normalising is per vector)
                ta = F.normalize((ta.float() @ fw_) if fw_.dim() == 2 else (ta.float() * fw_), dim=-1).to(ta.dtype)
            st_ = torch.einsum("btcd,bucd->btcu", pn, tn); gt_ = torch.einsum("btcd,bucd->btcu", tn, tn)          # [b, T, C, T]
            sc_ = torch.einsum("btcd,jtcd->btcj", pn.to(ta.dtype), ta).float()                                    # [b, T, C, B_all]
            with torch.no_grad():
                gc_ = torch.einsum("btcd,jtcd->btcj", tn.to(ta.dtype), ta).float()
                m_ = torch.zeros(sc_.shape[-1], dtype=torch.bool, device=sc_.device); m_[own: own + b] = True    # own clips: in the moment block
            sc_ = sc_.masked_fill(m_, float("-inf")); gc_ = gc_.masked_fill(m_, float("-inf"))
            logits = torch.cat([st_, sc_], -1) / tau
            with torch.no_grad():
                q_ = (torch.cat([gt_, gc_], -1) / tau).softmax(-1)
            lp_ = logits.log_softmax(-1).masked_fill(q_ == 0, 0.0)
            if getattr(self, "floor", False):
                # (11-02 v2) ONE-SIDED: the LLM must identify each token (its own moment of its own clip) at least as surely as V-JEPA does
                # among the same candidates - a loss only where the LLM has LOST information V-JEPA had; zero once it matches or exceeds it,
                # so V-JEPA's geometry is never imposed (the matching KL (v1) helped to ~4000 steps, then held the LLM to V-JEPA's
                # similarity structure and fell behind: 8000-step motion pairs -2.3 vs no dynamics loss). Where V-JEPA itself cannot tell
                # moments apart (repeats, stills) the requirement is low by construction.
                lab_ = torch.arange(T_, device=pred.device).view(1, T_, 1, 1).expand(b, T_, C_, 1)
                lq_ = q_.gather(-1, lab_).clamp_min(1e-12).log().squeeze(-1); lpp_ = logits.log_softmax(-1).gather(-1, lab_).squeeze(-1)
                loss = wmean(F.relu(lq_ - lpp_))
            else:
                loss = wmean((q_ * (q_.clamp_min(1e-12).log() - lp_)).sum(-1))
            with torch.no_grad():
                lab = torch.arange(T_, device=pred.device).view(1, T_, 1).expand(b, T_, C_)
                ht = (st_.argmax(-1) == lab).float().mean(); pos_ = st_.gather(-1, lab.unsqueeze(-1))
                hc = (pos_ > sc_.max(-1, keepdim=True).values).float().mean()
            if getattr(self, "objective", "") != "mse_ncf":
                return loss, torch.stack([loss.detach(), ht, hc, torch.tensor(float(sc_.shape[-1]), device=pred.device)])   # [KL, acc moments, acc vs all clips, clips]
            # (11-03) mse_ncf: the relational term (above, one-sided with floor) ADDED to the reconstruction MSE (below), each at the run's lambda
            kl_extra, kl_acc = loss, torch.stack([ht, ht, hc])
        if getattr(self, "objective", "mse") in ("nce", "nce_soft", "mse_nce_soft"):
            # (11-01) the same reconstruction scored by DISCRIMINATION instead of mean squared error: every predicted token must pick its own
            # V-JEPA token among (a) the same cell at every other moment of the same clip and (b) the same cell and moment of every other
            # clip of the batch (cosine / tau, cross-entropy). MSE spends itself on the large shared directions (the clip's static appearance,
            # ~76 %); here the shared part cancels and what separates moments and clips - what the benchmarks ask - carries the loss.
            T_ = self.n_tub; C_ = k // T_; tau = float(getattr(self, "tau", 0.1))
            pn = F.normalize(pred, dim=-1).view(b, T_, C_, -1); tn = F.normalize(t, dim=-1).view(b, T_, C_, -1)
            st_ = torch.einsum("btcd,bucd->btcu", pn, tn)                                   # [b, T, C, T]: other moments, positive at u = t
            if b > 1:
                sc_ = torch.einsum("btcd,jtcd->btcj", pn, tn)                               # [b, T, C, b]: other clips, own clip excluded
                own = torch.eye(b, dtype=torch.bool, device=sc_.device)[:, None, None, :].expand_as(sc_)
                sc_ = sc_.masked_fill(own, float("-inf"))
                logits = torch.cat([st_, sc_], -1) / tau
            else:
                logits = st_ / tau
            lab = torch.arange(T_, device=pred.device).view(1, T_, 1).expand(b, T_, C_)
            if self.objective in ("nce_soft", "mse_nce_soft"):
                # (11-01 v2) soft targets from V-JEPA's own similarity among the same candidates (cosine / tau_t): moments that V-JEPA sees as
                # (near) identical - a repeated action, a still stretch - share the target instead of being pushed apart (the hard version cost
                # the counting categories: MVBench action_count -12.5 at 2000 steps), while moments that V-JEPA tells apart stay negatives
                tt_ = float(getattr(self, "tau_t", 0.05))
                with torch.no_grad():
                    gt_ = torch.einsum("btcd,bucd->btcu", tn, tn)
                    if b > 1:
                        gc_ = torch.einsum("btcd,jtcd->btcj", tn, tn).masked_fill(own, float("-inf"))
                        q_ = (torch.cat([gt_, gc_], -1) / tt_).softmax(-1)
                    else:
                        q_ = (gt_ / tt_).softmax(-1)
                loss = -(q_ * logits.log_softmax(-1).masked_fill(q_ == 0, 0.0)).sum(-1).mean() + (q_ * q_.clamp_min(1e-12).log()).sum(-1).mean()   # KL(q || p)
            else:
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1))
            with torch.no_grad():
                hit = (logits.argmax(-1) == lab).float().mean(); ht = (st_.argmax(-1) == lab).float().mean()
                pos_ = st_.gather(-1, lab.unsqueeze(-1))
                hc = (pos_ > sc_.max(-1, keepdim=True).values).float().mean() if b > 1 else torch.ones((), device=pred.device)
            if self.objective != "mse_nce_soft":
                return loss, torch.stack([hit, ht, hc])
            # (11-01 v3) content + relations: the relation term (above) is ADDED to the reconstruction MSE below. nce_soft alone (which
            # replaced the MSE) gained the relational categories (counts, direction, order) but lost what all moments share (PerceptionTest
            # memory -8, object existence -4.5 at 5000 steps); the MSE carries that shared content. With tau == tau_t an exact copy of
            # V-JEPA's tokens is the optimum of both terms (no pull towards exaggerated contrasts).
            kl_extra, kl_acc = loss, torch.stack([hit, ht, hc])
        if self.pyramid:
            T_ = self.n_tub; G = int(round((k // T_) ** 0.5)); D_ = t.shape[-1]
            tg, pg = t.view(b, T_, G, G, D_), pred.view(b, T_, G, G, D_); us, r2l = [], {}
            # (11-02) bands: the time axis of the pyramid as an orthogonal (Haar) decomposition - level tt = 1 is the clip's time mean (what all
            # moments share), every finer level only its DETAIL, i.e. the level minus its parent (tt / 2) repeated over the two halves (the
            # change at that time scale). Each band over its own variance, averaged, as the pyramid's levels already are. In the plain pyramid
            # every finer level's variance is dominated by the shared content (clip mean; ~65% of the target), so the loss was mostly met
            # by keeping content - the moment-to-moment part the relational losses showed moves the motion categories was a small share of it
            bands = bool(getattr(self, "bands", False)); par = {}
            for tt in (1, 2, 4, 8, 16):
                for ss in (1, 2, 4, 8):
                    if tt > T_ or ss > G: continue
                    a_ = tg.view(b, tt, T_ // tt, ss, G // ss, ss, G // ss, D_).mean((2, 4, 6))
                    p_ = pg.view(b, tt, T_ // tt, ss, G // ss, ss, G // ss, D_).mean((2, 4, 6))
                    if bands:
                        par[(tt, ss)] = (a_, p_)
                        if tt > 1:
                            a0, p0 = par[(tt // 2, ss)]
                            a_, p_ = a_ - a0.repeat_interleave(2, dim=1), p_ - p0.repeat_interleave(2, dim=1)
                    v_ = (a_ - a_.mean((0, 1, 2, 3), keepdim=True)).pow(2).mean().detach().clamp_min(1e-6)
                    u_ = wmean((p_ - a_).pow(2)) / v_; us.append(u_); r2l[(tt, ss)] = 1 - u_.detach()
            loss = torch.stack(us).mean()
            if kl_extra is not None:
                w_ = torch.ones((), device=loss.device)
                if getattr(self, "balance", False) == "init" and pred.requires_grad:
                    # (11-02 v6) each term divided by its OWN early scale (mean over the first 200 calls), then fixed: w = MSE_0 / KL_0. The
                    # running ratio (v5) fed back - KL fell faster, w rose 0.47 -> 1.06 and the content R^2 slid back (0.39 -> 0.36 while MSE
                    # alone reached 0.62). Fixed weights keep the 400-800-step balance where both terms learned.
                    n_ = getattr(self, "_n0", 0)
                    if n_ < 200:
                        self._s0 = (getattr(self, "_s0", 0.0) * n_ + torch.stack([loss.detach(), kl_extra.detach()])) / (n_ + 1); self._n0 = n_ + 1
                    w_ = (self._s0[0].clamp_min(1e-3) / self._s0[1].clamp_min(1e-3)).detach()
                elif getattr(self, "balance", False) == "loss" and pred.requires_grad:
                    # (11-02 v5) loss-scale balance: the KL weighted by the MSE / KL loss ratio (exponential moving averages, detached), so each
                    # term's RELATIVE decrease counts the same. Gradient-norm balance (v4) held the relation term at ~0.07 (moments 0.15 vs 0.54
                    # of nce_soft at 700 steps); no balance (v3) held the content at R^2 0.19 vs 0.45. This sits between (gradient ratio ~3-4).
                    lm_, lk_ = loss.detach(), kl_extra.detach()
                    self._em = lm_ if getattr(self, "_em", None) is None else 0.98 * self._em + 0.02 * lm_
                    self._ek = lk_ if getattr(self, "_ek", None) is None else 0.98 * self._ek + 0.02 * lk_
                    w_ = (self._em.clamp_min(1e-3) / self._ek.clamp_min(1e-3)).detach()
                elif getattr(self, "balance", False) and pred.requires_grad:
                    # (11-01 v4) the two terms pull the decoder output (and through it the LLM) with EQUAL gradient norms: the relation term,
                    # with logits at cos / 0.05, out-pulled the MSE (content R^2 0.19 vs 0.45 of MSE alone at 700 steps). The KL is weighted
                    # by ||d MSE / d pred|| / ||d KL / d pred|| (detached, exponential moving average); gradients w.r.t. the output only.
                    gm_ = torch.autograd.grad(loss, pred, retain_graph=True)[0]; gk_ = torch.autograd.grad(kl_extra, pred, retain_graph=True)[0]
                    r_ = (gm_.norm() / gk_.norm().clamp_min(1e-12)).detach()
                    self._bal = r_ if getattr(self, "_bal", None) is None else 0.98 * self._bal + 0.02 * r_
                    w_ = self._bal
                return loss + w_ * kl_extra, torch.cat([torch.stack([1 - loss.detach(), kl_extra.detach(), w_.detach().float()]), kl_acc])   # [R^2, KL, w, acc all / moments / clips]
            return loss, torch.stack([1 - loss.detach(), r2l[(1, 1)], r2l[(min(16, T_), 4)], r2l[(min(16, T_), G)]])
        var = (t - t.mean((0, 1), keepdim=True)).pow(2).mean().detach().clamp_min(1e-6)
        loss = (pred - t).pow(2).mean() / var
        with torch.no_grad():
            T_ = self.n_tub; ts = t.view(b, T_, k // T_, -1).mean(1); ps = pred.view(b, T_, k // T_, -1).mean(1)
            r2s = 1 - (ps - ts).pow(2).mean() / (ts - ts.mean(0, keepdim=True)).pow(2).mean().clamp_min(1e-6)
            td = t.view(b, T_, k // T_, -1) - ts[:, None]; pd = pred.view(b, T_, k // T_, -1) - ps[:, None]
            r2d = 1 - (pd - td).pow(2).mean() / td.pow(2).mean().clamp_min(1e-6)
        return loss, torch.stack([1 - loss.detach(), r2s, r2d])


class _StructKeep(nn.Module):
    """
    Head-free structure preservation at the video positions. At each listed LLM layer (and the final one), per clip, the linear CKA between
      the LLM states H [K, H] and V-JEPA's tokens X [K, k]                       (what is where: the clip's content geometry), and
      the change over one moment dH = H[t+1] - H[t] and dX (every direction of dX scaled by its own std, so the motion's small directions
      count as much as its large ones)                                             (what moves where: the clip's dynamics geometry)
    must be 1. No parameters: nothing can read the information out for the loss except the LLM's own representation, so it has to keep
    V-JEPA's relational structure (and its motion structure) itself, at the depths the question / answer positions read through attention.
    loss = mean over layers of ((1 - CKA_state) + (1 - CKA_motion)) / 2; acc = [CKA_state per layer..., CKA_motion per layer...]."""
    def __init__(self, cells):
        super().__init__(); self.cells = int(cells)
    @staticmethod
    def _cka(a, b):
        a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
        ga = a @ a.transpose(1, 2); gb = b @ b.transpose(1, 2)
        return (ga * gb).sum((1, 2)) / (ga.flatten(1).norm(dim=1) * gb.flatten(1).norm(dim=1)).clamp_min(1e-8)
    def forward(self, states, x):
        c = self.cells
        with torch.autocast(x.device.type, enabled=False):
            x = x.float(); dx = x[:, c:] - x[:, :-c]
            dx = dx / dx.flatten(0, 1).std(0).detach().clamp_min(1e-6)
            cs, cm = [], []
            for h in states:
                h = h.float(); cs.append(self._cka(h, x)); cm.append(self._cka(h[:, c:] - h[:, :-c], dx))
            cs, cm = torch.stack(cs), torch.stack(cm)                       # [layers, B]
            loss = ((1 - cs).mean() + (1 - cm).mean()) / 2
        return loss, torch.cat([cs.mean(1).detach(), cm.mean(1).detach()])


class _ReadStruct(nn.Module):
    """Head-free READ-OUT of the clip's
    dynamics by the language stream. _StructKeep (v1) held V-JEPA's geometry at the video positions of every layer (CKA 0.96 vs 0.63 without
    it) but cost -6 points at 1000 steps (MVP 14 vs 30, Vinoground text 27 vs 51): pinning the video positions to the input geometry stops
    the LLM from turning them into something its text can use. Here the video positions are free; the constraint is on the FIRST TEXT
    POSITION AFTER THE VIDEO (the same template token in every row, causal: it has seen the prompt and the video only, no question), the
    position every question / answer token reads after the video, at the listed layers and the final one:
      across all clips of the global batch, the debiased linear CKA (unbiased HSIC) between those states [N, H] and the clips' V-JEPA MOTION FIELDS
      dX = X[t+1] - X[t] at every cell and moment (every direction of dX scaled by its own std; each token projected by a fixed random
      Gaussian to `proj` dims, which keeps inner products: the clip-by-clip similarity of where / how / when things move, cell-aligned)
    must be 1. Clips whose fine motion fields are alike (same rotation, same object swap, same count of events) must be alike in the text
    stream; clips that differ only in their motion must differ there. Nothing but the LLM's own attention can move the motion into that
    position. No parameters. The global batch (all ranks; local rows carry the gradient, x world size = the exact gradient of the global
    loss after the data-parallel mean). loss = mean over layers of (1 - CKA); acc = [CKA per layer..., N]."""
    def __init__(self, cells, hidden, proj=32):
        super().__init__(); self.cells, self.hidden, self.proj = int(cells), int(hidden), int(proj)
    @staticmethod
    def _hsic(k, l):
        # unbiased HSIC (Song et al. 2012): diagonals removed. The plain (biased) linear CKA is ~1 for ANY two feature sets whose dimension
        # exceeds the number of clips (both Grams ~ diagonal; smoke test: 0.78 at step 20 with 24 clips) - it would measure nothing.
        n = k.shape[0]; k = k - torch.diag(torch.diag(k)); l = l - torch.diag(torch.diag(l)); one = torch.ones(n, 1, device=k.device)
        return ((k * l).sum() + (k.sum() * l.sum()) / ((n - 1) * (n - 2)) - 2.0 / (n - 2) * (one.T @ k @ l @ one).squeeze()) / (n * (n - 3))
    @classmethod
    def _cka(cls, a, b):
        ga = a @ a.T; gb = b @ b.T
        return cls._hsic(ga, gb) / (cls._hsic(ga, ga).clamp_min(1e-12) * cls._hsic(gb, gb).clamp_min(1e-12)).sqrt()
    def forward(self, states, x, n_layers):
        dist = torch.distributed; on = dist.is_available() and dist.is_initialized()
        dev = torch.device("cuda", torch.cuda.current_device()) if states is None else states[0].device
        with torch.autocast(dev.type, enabled=False):
            if states is not None:
                c = self.cells; x = x.float(); dx = x[:, c:] - x[:, :-c]
                dx = dx / dx.flatten(0, 1).std(0).clamp_min(1e-6)
                g = torch.Generator(device=dev); g.manual_seed(1101)
                P = torch.randn(dx.shape[-1], self.proj, generator=g, device=dev) / self.proj ** 0.5
                f = (dx @ P).flatten(1)                                          # [b, moments x cells x proj]
                hs = torch.stack([h.float() for h in states])                     # [layers, b, H]
            else:
                f = torch.zeros(0, 0, device=dev); hs = torch.zeros(n_layers, 0, self.hidden, device=dev)
            b, fd = f.shape
            if on:
                W, r = dist.get_world_size(), dist.get_rank()
                sz = torch.tensor([b, fd], device=dev); szs = [torch.zeros_like(sz) for _ in range(W)]; dist.all_gather(szs, sz)
                szs = torch.stack(szs); mb, mf = int(szs[:, 0].max()), int(szs[:, 1].max())
                fp = F.pad(f, (0, mf - fd, 0, mb - b)); hp = F.pad(hs, (0, 0, 0, mb - b))
                fo = [torch.zeros_like(fp) for _ in range(W)]; ho = [torch.zeros_like(hp) for _ in range(W)]
                dist.all_gather(fo, fp.detach().contiguous()); dist.all_gather(ho, hp.detach().contiguous())
                fo[r], ho[r] = fp, hp                                             # the local rows keep their gradient
                keep = torch.cat([torch.arange(mb, device=dev) < int(n) for n in szs[:, 0]])
                fa, ha = torch.cat(fo)[keep], torch.cat(ho, 1)[:, keep]
            else:
                W, fa, ha = 1, f, hs
            n = fa.shape[0]
            if n < 8 or b == 0:
                z = hs.sum() * 0.0
                return z, torch.cat([torch.zeros(n_layers, device=dev), torch.tensor([float(n)], device=dev)])
            ck = torch.stack([self._cka(h, fa) for h in ha])
            loss = (1 - ck).mean()
        return loss * W, torch.cat([ck.detach(), torch.tensor([float(n)], device=dev)])


class _CfUse(nn.Module):
    """
    Dynamics counterfactual inside the ONE forward pass, by substitution. A few rows of the batch enter with their clip's V-JEPA moments
    shuffled (every token kept, its time stamp kept, the moments' contents permuted: identical appearance, no coherent dynamics); no row is
    added. What such a row must answer depends on its question:
      code 1 - generated dynamics QA whose answer is decided by the ORDER of what happens (order / before-after, direction over time, state
               change, tracking, when, physical order, done / success), multiple choice: the answer may be no more likely than chance,
               loss = mean over its answers of max(0, log p(answer | shuffled) - log(1 / options)); its normal CE is dropped;
      code 2 - appearance questions (objects, colour, scene, action class): the usual CE, unchanged (the answer survives the shuffle).
    Lowering both needs the clip's dynamics to be READ: whether this question needs them and whether this clip has them. "Shuffled -> be
    unsure" fails on code 2; "answer from appearance" fails on code 1. No parameters. acc = [mean log p - log chance on code 1, share of
    code-1 answers above chance, code-1 answers, code-2 rows]."""
    def __init__(self):
        super().__init__()


class _DynAns(nn.Module):
    """The dynamics loss moved to WHERE THE ANSWER IS PRODUCED. rec2 read V-JEPA back from any text position after the
    video through a cross-attention + MLP decoder: the LLM could park the clip in question tokens or in directions the output never reads, and
    76 % of its target was static appearance. Here:
      readout = the final-layer state at the answer positions (mean over the row's answer tokens) - the very vector lm_head maps to the answer;
      target  = the clip's DYNAMICS only: the equalised V-JEPA state minus its mean over moments (top `dirs` principal directions), at coarse
                space-time resolution (every moment x regions x regions) - what one answer state can hold;
      map     = ONE linear layer (simpler than rec2's decoder; zero-initialised: loss starts at 1).
    To satisfy it, the computation that produces the answer state has to fetch and process the clip's motion (attention from the answer
    positions into the video at every layer) and leave it linearly readable where lm_head reads - a path the answer CE can then use.
    loss = sum of squared error / sum of squared target over rows with answers (1 - R^2); acc = [R^2]."""
    def __init__(self, hidden, n_tub, cells, dirs=64, regions=2):
        super().__init__()
        self.n_tub, self.side, self.dirs, self.reg = int(n_tub), int(round(cells ** 0.5)), int(dirs), int(regions)
        self.lin = nn.Linear(hidden, self.n_tub * self.reg * self.reg * self.dirs)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)
    def forward(self, x, h, has):
        if x is None:                                    # FSDP: every rank runs the head; weight 0 when there is nothing to read
            z = self.lin(self.lin.weight.new_zeros(1, self.lin.in_features)).float().sum() * 0.0
            return z, z.new_zeros(1)
        b, s, r, d = x.shape[0], self.side, self.reg, self.dirs
        with torch.autocast(x.device.type, enabled=False):
            t = x[..., :d].float().view(b, self.n_tub, s, s, d)
            t = t - t.mean(1, keepdim=True)                                       # dynamics: deviation over moments
            t = t.view(b, self.n_tub, r, s // r, r, s // r, d).mean((3, 5)).flatten(1)
        pred = self.lin(h.to(self.lin.weight.dtype)).float()
        m = has.float()
        loss = (((pred - t) ** 2).sum(1) * m).sum() / ((t ** 2).sum(1) * m).sum().clamp_min(1e-8)
        return loss, (1 - loss.detach()).view(1)


class _RouteKeep(nn.Module):
    """The dynamics loss on HOW THE LLM READS THE VIDEO, not on what a state holds (holding it -
    rec2, struct, ans - never raised the benchmarks; ans even cost the fine paired ones). Following an object through a clip (shuffle,
    rotation, counting, order) is a routing problem: a token of moment t+1 must read the tokens of moment t where ITS content was.
    V-JEPA knows that correspondence: teacher = softmax over the 64 cells of moment t of cos(x[t+1, c], x[t, c']) / tau (equalised states).
    Student = the LLM's own attention (q . k / sqrt(d), every head of the layer, its keys renormalised over the cells of moment t; head mean of
    the probabilities), at EVERY layer: no layer, head or subset chosen. loss = mean over layers of KL(teacher || student). Conditional on
    attending to the previous moment: how much a token reads the previous moment, and everything text positions do, is left to the answer loss.
    Pre-rotary scores (content routing; the rotary offset of a cell pair is fixed). No parameters. acc = [top-1 agreement, mean over layers]."""
    def __init__(self, cells, n_layers, tau=0.05):
        super().__init__(); self.cells, self.n_layers, self.tau = int(cells), int(n_layers), float(tau)
    def forward(self, qs, ks, x):
        c = self.cells; b, K = x.shape[:2]; nt = K // c
        with torch.autocast(x.device.type, enabled=False):
            xn = F.normalize(x.float(), dim=-1).view(b, nt, c, -1)
            tl = (torch.einsum("btcd,bted->btce", xn[:, 1:], xn[:, :-1]) / self.tau).log_softmax(-1)     # [b, nt-1, c, c]
            tp = tl.exp(); ent = (tp * tl).sum(-1)
            losses, agree = [], []
            for q, k in zip(qs, ks):
                hq, d = q.shape[-2], q.shape[-1]; hk = k.shape[-2]
                q = q.float().view(b, nt, c, hq, d); k = k.float().view(b, nt, c, hk, d).repeat_interleave(hq // hk, dim=3)
                sc = torch.einsum("btchd,btehd->bthce", q[:, 1:], k[:, :-1]) / math.sqrt(d)            # [b, nt-1, heads, c, c]
                sp = sc.softmax(-1).mean(2).clamp_min(1e-8)                                            # head mean of the probabilities
                losses.append((ent - (tp * sp.log()).sum(-1)).mean())
                agree.append((sp.argmax(-1) == tl.argmax(-1)).float().mean())
            loss = torch.stack(losses).mean()
        return loss, torch.stack([torch.stack(agree).mean().detach()])


class _VFloor(nn.Module):
    """The answers read the video
    through the video TOKENS at every layer (attention), not through a copy in the text stream (rec2 kept such a copy: preserved, never used).
    In a model without any dynamics loss the V-JEPA structure of the video tokens decays with depth (state CKA 0.77 -> 0.63, motion 0.59 ->
    0.40). This loss keeps the INFORMATION at every layer without fixing the geometry (struct, which pinned the geometry, and the relational
    losses that matched it, helped early and then held the LLM back):
      at every layer l, each video token's state must still identify its OWN input token (the bridge output e, detached) among candidates -
      the same cell at the clip's other moments and the same cell and moment of the other clips of the GPU's batch - at least as surely as
      V-JEPA identifies that token among the same candidates. ONE-SIDED: loss = mean max(0, log q_self - log p_self); zero where the layer
      keeps what V-JEPA had, so the layer is free to re-encode. Where V-JEPA itself cannot tell moments apart (repeats, stills) q_self is low.
    p: softmax over candidates of cos(h_l, e) / tau after each side is standardised per channel over the batch's video tokens (detached
    statistics: Qwen's few huge channels would otherwise set every cosine); q: softmax of cos(x, x) / tau of V-JEPA's tokens. No parameters.
    acc = [loss, p_self >= q_self share at the first / middle / last layer, mean over layers]."""
    def __init__(self, cells, tau=0.05):
        super().__init__(); self.cells, self.tau = int(cells), float(tau)
    @staticmethod
    def _std(z):
        m = z.mean((0, 1), keepdim=True).detach(); s = z.std((0, 1), keepdim=True).detach().clamp_min(1e-4)
        return (z - m) / s
    def _cand(self, a, bref, b, T, C):
        """cos of a [b, T, C, D] with the candidates of bref [b, T, C, D]: same cell other moments + other clips same (t, c) -> [b, T, C, T + b - 1]"""
        st = torch.einsum("btcd,bucd->btcu", a, bref)
        if b > 1:
            sc = torch.einsum("btcd,jtcd->btcj", a, bref)
            own = torch.eye(b, dtype=torch.bool, device=sc.device)[:, None, None, :].expand_as(sc)
            sc = sc[~own].view(b, T, C, b - 1)
            return torch.cat([st, sc], -1)
        return st
    def forward(self, states, e, x, nt):
        b, K = e.shape[:2]; T, C = nt, K // nt
        with torch.autocast(e.device.type, enabled=False):
            lab = torch.arange(T, device=e.device).view(1, T, 1, 1).expand(b, T, C, 1)
            with torch.no_grad():
                xn = F.normalize(x.float(), dim=-1).view(b, T, C, -1)
                lq = (self._cand(xn, xn, b, T, C) / self.tau).log_softmax(-1).gather(-1, lab).squeeze(-1)      # V-JEPA's own identifiability
                en = F.normalize(self._std(e.float()), dim=-1).view(b, T, C, -1)
            losses, oks = [], []
            for h in states:
                hn = F.normalize(self._std(h.float()), dim=-1).view(b, T, C, -1)
                lp = (self._cand(hn, en, b, T, C) / self.tau).log_softmax(-1).gather(-1, lab).squeeze(-1)
                losses.append(F.relu(lq - lp).mean()); oks.append((lp >= lq).float().mean().detach())
            loss = torch.stack(losses).mean(); oks = torch.stack(oks)
        return loss, torch.stack([loss.detach(), oks[0], oks[len(oks) // 2], oks[-1], oks.mean()])


class _DynNext(nn.Module):
    """(10-31) dynamics as NEXT-MOMENT PREDICTION inside the LLM . The LLM is causal: its state
    at the video position of moment t, cell c has seen moments <= t only. From that state a training-only head predicts the CHANGE of
    V-JEPA's state at the same cell from t to t+1 (x[t+1,c] - x[t,c]), every direction normalised by its own variance of change (the
    clip-constant appearance, 76 % of the state's variance, is not a target; small and large directions count the same).
    Why not the reconstruction (_RecLang): it read the text positions after the whole video (copying is enough) and 76 % of its target was
    static appearance; it preserved V-JEPA's state in the LLM (+0.1-0.2 R^2) but nothing used it. Here the target is what moves, and it is
    produced at the video positions, which the question / answer positions read through attention in every layer.
    Loss = mean over directions of squared error / variance of change (1 - R^2; predicting no change = 1). acc = [1 - loss, R^2 on
    directions 0-16, 16-256, 256-end]."""
    def __init__(self, hidden, kin, cells, d=1024, horizons=(1,)):
        super().__init__(); self.cells = int(cells); self.hz = tuple(int(x) for x in horizons); self.kin = int(kin)
        # (11-01 v2 option) horizons (1, 2, 4): the change to t+1, t+2 and t+4 from the same state (one output block per horizon)
        self.net = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, d), nn.GELU(), nn.Linear(d, kin * len(self.hz)))
        nn.init.zeros_(self.net[3].weight); nn.init.zeros_(self.net[3].bias)       # (smoke fix) starts at 'no change': loss exactly 1
    def forward(self, h, x):
        """h [B, K, H] final states at the video positions (time-major), x [B, K, k] equalised V-JEPA states (detached)"""
        c = self.cells; out = self.net(h.to(self.net[1].weight.dtype)).float(); es = []
        for j, hz in enumerate(self.hz):
            s_ = c * hz
            if s_ >= x.shape[1]: continue
            dx = (x[:, s_:] - x[:, :-s_]).float(); pred = out[:, :-s_, j * self.kin:(j + 1) * self.kin]
            var = (dx - dx.mean((0, 1))).pow(2).mean((0, 1)).detach()
            var = var.clamp_min(0.01 * var.mean() + 1e-8)   # (smoke fix) directions that hardly move must not dominate (loss reached 137 at step 5)
            es.append((pred - dx).pow(2).mean((0, 1)) / var)
        e = torch.stack(es).mean(0)
        k = e.shape[0]; bands = [(0, 16), (16, min(256, k)), (min(256, k), k)]
        return e.mean(), torch.stack([1 - e.mean().detach()] + [1 - e[a:z].mean().detach() for a, z in bands])


def _pyramid(t, pred, T_):
    """(10-26) the reconstruction loss of _RecLang (pyramid=True) as a function: MSE over every dyadic space-time pooling (time 16..1 x space
    8x8..1x1), each over its own target variance, averaged -> (loss, [mean R^2, clip (1,1), (16,4), finest])"""
    b, k, D_ = t.shape; G = int(round((k // T_) ** 0.5)); tg, pg = t.view(b, T_, G, G, D_), pred.view(b, T_, G, G, D_); us, r2l = [], {}
    for tt in (1, 2, 4, 8, 16):
        for ss in (1, 2, 4, 8):
            if tt > T_ or ss > G: continue
            a_ = tg.view(b, tt, T_ // tt, ss, G // ss, ss, G // ss, D_).mean((2, 4, 6)); p_ = pg.view(b, tt, T_ // tt, ss, G // ss, ss, G // ss, D_).mean((2, 4, 6))
            v_ = (a_ - a_.mean((0, 1, 2, 3), keepdim=True)).pow(2).mean().detach().clamp_min(1e-6)
            u_ = (p_ - a_).pow(2).mean() / v_; us.append(u_); r2l[(tt, ss)] = 1 - u_.detach()
    loss = torch.stack(us).mean()
    return loss, torch.stack([1 - loss.detach(), r2l[(1, 1)], r2l[(min(16, T_), min(4, G))], r2l[(min(16, T_), G)]])


class _FlowPred(nn.Module):
    """(11-02 night) teach the LLM the computation instead of handing it the result: from the final state at every video position, one linear
    map must give V-JEPA's latent flow of that cell and moment (Stage1Model.latent_flow: displacement, local curl / divergence, confidence,
    per-moment global curl and mean motion). The LLM is causal, so the state at moment t sees the moments before it: getting the flow right
    means computing the cross-moment correspondence itself. Loss = per-feature MSE / that feature's variance, averaged (1 - R^2)."""
    def __init__(self, hidden, n_feat: int = 9):
        super().__init__(); self.lin = nn.Linear(hidden, n_feat)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)
        self.want = 0

    def forward(self, f, h):
        p = self.lin(h.to(self.lin.weight.dtype)).float(); t = f.float().to(p.device)
        v = (t - t.mean((0, 1), keepdim=True)).pow(2).mean((0, 1)).clamp_min(1e-6)
        u = (p - t).pow(2).mean((0, 1)) / v
        return u.mean(), (1 - u.detach())


class _RecSelf(nn.Module):
    """(10-26) reconstruction INSIDE the LLM, no decoder: the final state at every video position, through one linear map (units only: LLM
    width -> V-JEPA's 2048 principal axes), must give back that position's own V-JEPA token (original geometry), scored with the same space-time
    pyramid as _RecLang. Keeps the LLM from washing V-JEPA's content out of the video positions on the way through its layers - the positions
    every later token reads. Training only (the linear map is dropped at inference)."""
    def __init__(self, hidden, n_tub, kin=2048):
        super().__init__(); self.n_tub = int(n_tub); self.lin = nn.Linear(hidden, kin)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)          # starts at the mean (loss 1)
        self.want = 0

    def forward(self, x, h):
        """x [B, K, kin] target tokens (fixed), h [B, K, H] final states at the video positions (same order)"""
        return _pyramid(x.float(), self.lin(h.to(self.lin.weight.dtype)).float(), self.n_tub)


class _DynHead(nn.Module):
    """(10-19) dynamics contrast (Stage1Model dyncon): a small attentive reader over the equalised V-JEPA tokens and a projection of the
    LLM's answer-time state; forward = the contrastive loss (one module with a forward, so FSDP can shard it)"""
    def __init__(self, kin, hidden, dd, nk, perms, tau):
        super().__init__()
        # positions start as fixed sinusoids of (moment, cell) - with zeros the reader is blind to order, a shuffled clip reads exactly like the
        # real one and the contrast never leaves ln(perms + 1) (dc_L first try: 1.39 for 1800 steps)
        g = 64; t = torch.arange(nk) // g; c = torch.arange(nk) % g; half = dd // 2
        fr = torch.exp(-torch.arange(0, half, 2).float() * (torch.log(torch.tensor(1000.0)) / half))
        pe = torch.cat([torch.sin(t[:, None] * fr), torch.cos(t[:, None] * fr), torch.sin(c[:, None] * fr), torch.cos(c[:, None] * fr)], 1)[:, :dd]
        self.inp = nn.Linear(kin, dd); self.pos = nn.Parameter(0.5 * pe)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dd, 8, 4 * dd, dropout=0.0, batch_first=True, norm_first=True), 2)
        self.norm = nn.LayerNorm(dd); self.lp = nn.Linear(hidden, dd)        # (10-20) no attention pool / query: unused since the ordered read
        self.n_t = nk // 64; self.seq = nn.Linear(self.n_t * dd, dd)        # moments kept in order: per-moment summaries concatenated
        self.perms, self.tau = perms, tau

    def read(self, x):
        """x [N, K, kin] equalised V-JEPA tokens (time-major) -> [N, d] unit summary"""
        z = self.enc(self.inp(x.to(self.inp.weight.dtype)) + self.pos.to(self.inp.weight.dtype))
        # order-sensitive by construction: a mean / attention pool over all tokens reads a shuffled clip like the real one at the start
        # (cos 0.99997), so the contrast had nothing to learn from; the moments' summaries in their order
        zt = z.view(len(z), self.n_t, -1, z.shape[-1]).mean(2).flatten(1)
        return F.normalize(self.norm(self.seq(zt)).float(), dim=-1)

    def forward(self, x, ans, n_tub):
        """x [B, K, kin] equalised tokens, ans [B, H] mean final state over the answer positions -> (CE, [acc over all candidates,
        share where the real clip beats all its own moment-shuffled copies])"""
        b, k, kin = x.shape; P = self.perms; g = k // n_tub; ar = torch.arange(n_tub, device=x.device)
        xs = [x]
        for j in range(P):
            if getattr(self, "neg", "shuffle") == "hard":
                # (10-20) the random shuffles were solved by step 1200 (100%): the order of 8 moments is easy to tell from a full shuffle.
                # Hard copies differ from the real clip in one place only: two neighbouring moments swapped (first half of the copies) or a
                # span of 2-4 moments played backwards (second half)
                perm = ar.repeat(b, 1)
                for r in range(b):
                    if j < P // 2:
                        i = int(torch.randint(0, n_tub - 1, (1,))); perm[r, i], perm[r, i + 1] = ar[i + 1], ar[i]
                    else:
                        L_ = int(torch.randint(2, min(4, n_tub) + 1, (1,))); i = int(torch.randint(0, n_tub - L_ + 1, (1,)))
                        perm[r, i:i + L_] = ar[i:i + L_].flip(0)
            else:                                            # each row's moments in a random non-identity order
                perm = torch.stack([torch.randperm(n_tub, device=x.device) for _ in range(b)])
                perm[(perm == ar).all(1)] = ar.flip(0)
            xs.append(x.view(b, n_tub, g, kin)[torch.arange(b, device=x.device)[:, None], perm].reshape(b, k, kin))
        t = self.read(torch.cat(xs)).view(P + 1, b, -1)                   # [P+1, B, d]
        u = F.normalize(self.lp(ans.to(self.lp.weight.dtype)).float(), dim=-1)    # [B, d]
        real = u @ t[0].T                                                  # [B, B] own clip on the diagonal, other clips off it
        own_sh = torch.einsum("bd,pbd->bp", u, t[1:])                     # [B, P] own clip, moments shuffled
        logits = torch.cat([real, own_sh], 1) / self.tau
        tgt = torch.arange(b, device=x.device); loss = F.cross_entropy(logits, tgt)
        with torch.no_grad():
            acc = torch.stack([(logits.argmax(1) == tgt).float().mean(), (real.diagonal()[:, None] > own_sh).all(1).float().mean()])
        return loss, acc


def moe_balance_loss(router_logits, num_experts: int, top_k: int, attention_mask):
    """(11-17) transformers' load_balancing_loss_func (Switch loss over all layers' router logits, padding masked), computed layer by
    layer: the library version materialises an int64 one-hot of [layers x tokens, top_k, experts] (+ float mask copies), ~30 GB at
    32 rows x 2560 tokens x 48 layers, which ran the mx30a_Q main run out of memory at step 1. Same value (softmax in fp32)."""
    m = attention_mask.reshape(-1).float()                                   # [T]
    cnt = torch.zeros(top_k, num_experts, device=m.device)
    prob = 0.0
    for g in router_logits:                                                  # [T, E] per layer
        p = torch.softmax(g.float(), dim=-1)
        sel = p.detach().topk(top_k, dim=-1).indices                         # [T, k]
        cnt.scatter_add_(1, sel.t(), m.expand(top_k, -1))
        prob = prob + (p * m[:, None]).sum(0)
    den = len(router_logits) * m.sum().clamp_min(1.0)
    return (cnt / den * (prob / den)[None]).sum() * num_experts


class LanguageCore(nn.Module):
    """HF causal LM + evidence splice + selective LM head, with forward hooks capturing chosen decoder-layer outputs."""

    def __init__(self, llm, capture_layers=()):
        super().__init__()
        self.llm = llm
        self.captured = {}
        self.gate = VisGate()
        self.router_coef, self.router_aux = 0.0, None   # (11-16) MoE backbone: Switch load-balancing loss weight (loss.lambda_router) / last value
        self.text_take = 0          # > 0: also return the captured states at up to this many text positions per example
        self.read_take, self.read_upto, self._read = 0, (), None
        # context dropout (training): each answer token is, with this probability, replaced AS AN INPUT by a blank
        # embedding (the mean token embedding); it stays a target. See forward().
        self.word_drop, self.blank = 0.0, None
        for b in capture_layers:  # 1-based: output of decoder layer b
            self.llm.model.layers[b - 1].register_forward_hook(self._hook(b))

    def _body(self, attention_mask, **kw):
        """(11-16) decoder stack -> final hidden states. MoE backbone with router_coef > 0: also the Switch load-balancing loss of this pass
        (transformers' load_balancing_loss_func over every layer's router logits, padding masked) in self.router_aux, as in the backbone's
        own pretraining (Qwen3MoeForCausalLM adds router_aux_loss_coef x the same term); the video tokens are a new input distribution."""
        if self.router_coef > 0:
            out = self.llm.model(attention_mask=attention_mask, use_cache=False, output_router_logits=True, **kw)
            c = self.llm.config
            self.router_aux = moe_balance_loss(out.router_logits, c.num_experts, c.num_experts_per_tok, attention_mask)
            return out.last_hidden_state
        return self.llm.model(attention_mask=attention_mask, use_cache=False, **kw).last_hidden_state

    def add_expert(self, mode: str):
        """GatedLinear on every linear map of the decoder layers; everything else of the LLM (embeddings, norms, head) is
        frozen: the LLM is taught only through the video positions."""
        for p in self.llm.parameters():
            p.requires_grad_(False)
        for layer in self.llm.model.layers:
            for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")), (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
                for n in names:
                    lin = getattr(parent, n)
                    lin.weight.requires_grad_(mode == "shared")
                    setattr(parent, n, GatedLinear(lin, self.gate, mode))

    def enable_read(self, take: int, upto):
        """Complete read (EqualisedStateHeads.read): for `take` random text positions per example, record how much
        attention they give to every video cell (all heads, layers 1 .. L, for each L in `upto`), with the model's own
        weights and no gradient. Extra work: the key projection of these layers once more and take x S attention rows."""
        self.read_take, self.read_upto = int(take), tuple(sorted(upto))
        for i in range(max(self.read_upto)):
            self.llm.model.layers[i].self_attn.register_forward_pre_hook(self._read_hook(i + 1), with_kwargs=True)

    def _read_hook(self, layer):
        from transformers.models.llama.modeling_llama import rotate_half

        def hook(mod, args, kwargs):
            r = self._read
            if r is None:
                return
            hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            cos, sin = kwargs["position_embeddings"]
            with torch.no_grad():
                b, s, _ = hs.shape
                idx, hd = r["idx"], mod.head_dim                                           # [B, n]
                n = idx.shape[1]
                g = idx.unsqueeze(-1)
                q = mod.q_proj(hs.gather(1, g.expand(-1, -1, hs.shape[-1]))).view(b, n, -1, hd).transpose(1, 2).float()
                k = mod.k_proj(hs).view(b, s, -1, hd).transpose(1, 2).float()             # [B, n_kv, S, hd]
                cos, sin = cos.float().expand(b, -1, -1), sin.float().expand(b, -1, -1)
                cq, sq = cos.gather(1, g.expand(-1, -1, hd)).unsqueeze(1), sin.gather(1, g.expand(-1, -1, hd)).unsqueeze(1)
                q = q * cq + rotate_half(q) * sq
                k = k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1)
                k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)                   # grouped-query attention
                lg = q @ k.transpose(-1, -2) * mod.scaling                                 # [B, heads, n, S]
                ok = r["mask"][:, None, None, :] & (torch.arange(s, device=hs.device)[None, None, None, :] <= idx[:, None, :, None])
                a = lg.masked_fill(~ok, float("-inf")).softmax(-1).mean(1)                 # [B, n, S] mean over heads
                m = a.gather(2, r["vis_idx"][:, None, :].expand(-1, n, -1))                # mass on each video cell [B, n, K]
                r["acc"] = m if r["acc"] is None else r["acc"] + m
                if layer in self.read_upto:
                    r["snap"][layer] = r["acc"] / layer
        return hook

    entry_layer, _entry, _eg = 0, None, None
    memory, _mem, hybrid, mem_input = False, None, False, None

    def enable_memory(self, layers=None):
        """(10-15/16) The video as memory that decoder layers read, instead of positions in the sequence. A V-JEPA token has
        already seen the whole clip (V-JEPA attends over all of it): in the sequence its position barely works (77% of its
        attention on the first token). Only the text runs through the LLM; in the memory layers (`layers`, 0-based; None = all)
        the text attends, besides the text before it, to the clip's tokens, whose keys / values are made by THAT layer's own
        input norm and k / v projections from the projected V-JEPA states (no RoPE: read by content; the bridge's place / time
        codes are in the states). Other layers are untouched.
        Kernel: memory keys go in front of the text keys and the causal mask is aligned to the lower right
        (torch causal_lower_right): every text query sees all memory keys and the text up to itself - no materialised mask,
        grouped-query heads native (enable_gqa), the flash / efficient SDPA kernels (an explicit boolean mask had forced the
        slow path). Right padding only: padded keys are seen by padded queries only, whose outputs are ignored."""
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        from torch.nn.attention.bias import causal_lower_right
        self.memory = True
        which = range(len(self.llm.model.layers)) if layers is None else [int(i) for i in layers]
        self.memory_layers = list(which)
        for li in which:
            layer = self.llm.model.layers[li]; attn, orig = layer.self_attn, layer.self_attn.forward

            def fwd(hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, _a=attn, _l=layer, _o=orig, **kw):
                m = self._mem
                if m is None:
                    return _o(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, past_key_values=past_key_values, **kw)
                mem = m[0]                                                        # [B, K, H] projected V-JEPA states
                b, sq, _ = hidden_states.shape; hd = _a.head_dim
                q = _a.q_proj(hidden_states).view(b, sq, -1, hd).transpose(1, 2)
                k = _a.k_proj(hidden_states).view(b, sq, -1, hd).transpose(1, 2)
                v = _a.v_proj(hidden_states).view(b, sq, -1, hd).transpose(1, 2)
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                if past_key_values is not None:
                    k, v = past_key_values.update(k, v, _a.layer_idx)
                hm = _l.input_layernorm(mem.to(hidden_states.dtype))
                mk = _a.k_proj(hm).view(b, hm.shape[1], -1, hd).transpose(1, 2)
                mv = _a.v_proj(hm).view(b, hm.shape[1], -1, hd).transpose(1, 2)
                kk, vv = torch.cat([mk, k], 2), torch.cat([mv, v], 2)
                o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=causal_lower_right(sq, kk.shape[2]), scale=_a.scaling, enable_gqa=True)
                return _a.o_proj(o.transpose(1, 2).reshape(b, sq, -1)), None
            attn.forward = fwd

    def enable_entry(self, layer: int, gain):
        """(10-15) V-JEPA tokens enter at decoder layer `layer` (0-based input of that layer) instead of the word-embedding
        layer. A V-JEPA token has already seen the whole clip - it is a contextualised state, like the LLM's own middle-layer
        states, not a context-free word (SigLIP's tokens, one frame each, are the latter: the LLM's early layers must relate
        them). The video positions carry a zero vector up to `layer`; there the projected states are written in, scaled to
        the size of the text states of that layer (median token RMS) times a learned gain (gain: a parameter of the bridge)."""
        self.entry_layer, self._eg = int(layer), [gain]
        self.llm.model.layers[self.entry_layer].register_forward_pre_hook(self._entry_hook, with_kwargs=True)

    def _entry_hook(self, mod, args, kwargs):
        e = self._entry
        if e is None:
            return None
        vis, ev = e
        hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        if hs.shape[1] != vis.shape[1]:
            return None
        with torch.no_grad():
            rt = hs[~vis].float().pow(2).mean(-1).sqrt().median()
            re_ = ev.float().pow(2).mean(-1).sqrt().median().clamp_min(1e-6)
        new = hs.masked_scatter(vis.unsqueeze(-1), (ev * (self._eg[0].to(ev.dtype) * (rt / re_).to(ev.dtype))).to(hs.dtype))
        if "hidden_states" in kwargs:
            return args, {**kwargs, "hidden_states": new}
        return (new,) + tuple(args[1:]), kwargs

    def _hook(self, b):
        def hook(_m, _i, out):
            if isinstance(b, str) and b in self.captured:   # (10-27) vpres read points: keep the forward's state; the activation-checkpoint
                return                                     # recomputation in backward fired the hook again and held a second copy
            self.captured[b] = out[0] if isinstance(out, tuple) else out
        return hook

    def forward(self, input_ids, attention_mask, evidence, vis_mask, labels=None, read_mask=None, extra=None, vwords=None):
        """Answer mode (labels given) -> (answer logits [n_answer_tokens, V] fp32, answer targets, example index per token,
        {layer: hidden at vis}). Read mode (labels None, read_mask [B, S] bool) -> final-layer hidden at read_mask
        positions [n, H] (pass 1 of stage 2: SEM / LOC positions). Both go through this forward for FSDP."""
        self.captured.clear()
        if evidence is None:            # text only (language kept while the LLM learns the V-JEPA words): mean CE of the labelled tokens
            last = self._body(attention_mask, input_ids=input_ids)
            tgt = labels[:, 1:]; sel = tgt != -100
            # (10-26) chunked + recomputed in backward: the self-generated replay rows reach 16k tokens and the fp32 logits of all answer
            # tokens at once (n x 152k vocab) did not fit (8.9 GB, OOM); same mean CE, at most 2048 tokens' logits alive at a time
            h, t = last[:, :-1][sel], tgt[sel]
            ce = lambda hh, tt: F.cross_entropy(self.llm.lm_head(hh).float(), tt, reduction="sum")
            tot = sum(torch.utils.checkpoint.checkpoint(ce, h[i:i + 512], t[i:i + 512], use_reentrant=False) for i in range(0, len(t), 512))   # (10-28) 2048 -> 512: OOM by 0.12 GB at nxs14_Q step ~32525
            return tot / max(len(t), 1) + (self.router_coef * self.router_aux.to(tot.dtype) if self.router_coef > 0 else 0.0)
        emb = self.llm.get_input_embeddings()(input_ids)
        vis = vis_mask.bool()
        if vwords is not None:          # V-JEPA words added to the projected states (inside this forward: the sharded weights are gathered here)
            we = self.llm.get_input_embeddings()(vwords).to(evidence.dtype)
            sc = (evidence.detach().float().pow(2).mean().sqrt() / we.detach().float().pow(2).mean().sqrt().clamp_min(1e-6)).to(evidence.dtype)
            evidence = evidence + self.vword_gain.to(evidence.dtype) * sc * we
        if self.training and self.word_drop > 0 and labels is not None:
            # With teacher forcing a caption explains itself: on main1b_SPEC @8000 swapping the video for another clip's
            # costs 0.4-0.7 nats on the first 10 answer tokens but only 0.13 after the 80th - most caption tokens are
            # predicted from the caption text before them and teach the wording, not the reading of the video (the
            # decoder ignoring its conditioning, as in posterior collapse). Hiding part of the preceding answer text
            # from the input makes the same caption ask for the video at every position. Targets are unchanged; no
            # token, loss or computation is added; inference is untouched.
            drop = (labels != -100) & (torch.rand(labels.shape, device=labels.device) < self.word_drop)
            if self.blank.device != emb.device:
                self.blank = self.blank.to(emb.device)
            emb = torch.where(drop.unsqueeze(-1), self.blank.to(emb.dtype).view(1, 1, -1), emb)
        if self.memory and not self.hybrid:   # the video is read as memory by every layer (enable_memory): no video positions
            self._mem = (evidence if evidence.dim() == 3 else evidence.unsqueeze(0), attention_mask)
        elif self.hybrid:               # pooled tokens in the sequence AND the full grid as memory (mem_input, set by Stage1Model)
            emb = emb.masked_scatter(vis.unsqueeze(-1), evidence.to(emb.dtype))
            if self.mem_input is not None:
                self._mem = (self.mem_input, attention_mask)
        elif self.entry_layer:          # V-JEPA tokens enter at a middle layer (enable_entry); zeros at the video positions until then
            emb = emb.masked_fill(vis.unsqueeze(-1), 0.0)
            self._entry = (vis, evidence.view(-1, evidence.shape[-1]) if evidence.dim() == 3 else evidence)
        else:
            emb = emb.masked_scatter(vis.unsqueeze(-1), evidence.to(emb.dtype))
        s_len, keys = emb.shape[1], None
        self._read = None
        if self.read_take > 0 and labels is not None and self.training and extra is None:
            k_vis = evidence.shape[1]
            txt = attention_mask.bool() & ~vis & (vis.cumsum(1) == k_vis)                  # text after the whole video
            val, ridx = torch.rand(txt.shape, device=txt.device).masked_fill(~txt, -1.0).topk(min(self.read_take, s_len), dim=1)
            self._read = {"idx": ridx, "ok": val >= 0, "vis_idx": vis.nonzero()[:, 1].view(-1, k_vis),
                          "mask": attention_mask.bool(), "acc": None, "snap": {}}
        if extra is not None:                                  # tokens appended after the whole sequence (training only)
            emb = torch.cat([emb, extra.to(emb.dtype)], 1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(extra.shape[:2])], 1)
        self.gate.set(vis if extra is None else None)
        try:
            last = self._body(attention_mask, inputs_embeds=emb)
        finally:
            self.gate.set(None); self._entry = None; self._mem = None; self.mem_input = None
        if extra is not None:
            keys = {layer: t[:, s_len:] for layer, t in self.captured.items()}
            self.captured = {layer: t[:, :s_len] for layer, t in self.captured.items()}
            last, attention_mask = last[:, :s_len], attention_mask[:, :s_len]
        if labels is None:
            return last[read_mask.bool()]
        tgt = labels[:, 1:]
        sel = tgt != -100
        logits = self.llm.lm_head(last[:, :-1][sel]).float()
        ex_id = torch.arange(labels.shape[0], device=labels.device).unsqueeze(1).expand_as(tgt)[sel]
        if evidence.dim() == 2 or (self.memory and not self.hybrid):   # different numbers of visual tokens / video only as memory: no evidence positions
            return logits, tgt[sel], ex_id, {}
        b, k = evidence.shape[:2]
        hid = {layer: t[vis].view(b, k, -1) for layer, t in self.captured.items()}
        hid["last"] = last[vis].view(b, k, -1)                 # final hidden state at the evidence positions
        hid["last_full"] = last                                 # (10-25) the final representation at every position (reconstruction v2, no hook)
        if getattr(self, "want_vlog", False):
            # (10-25) concept readout: the clip's mean final state over the video positions read through the LLM's own unembedding (weights
            # detached: the LLM's word space is the reference, the video is moved to it). Video positions precede the question (causal mask):
            # question-independent by construction.
            hid["vlog"] = F.linear(hid["last"].mean(1), self.llm.lm_head.weight.detach()).float()
        sv = last[:, :-1][sel]                                 # (10-19) mean final state over the answer positions (what is held while writing)
        hid["ans"] = torch.zeros(b, sv.shape[-1], device=sv.device, dtype=torch.float32).index_add_(0, ex_id, sv.float()) \
            / torch.zeros(b, device=sv.device).index_add_(0, ex_id, torch.ones(len(ex_id), device=sv.device)).clamp_min(1).unsqueeze(1)
        if getattr(self, "ws", False):
            # what the LLM is about to say while it reads the video: its mean next-word distribution over the video
            # positions, as a unit vector (square roots: the dot product of two of them is the overlap of the distributions)
            hid["ws"] = torch.stack([self.llm.lm_head(hid["last"][i]).float().softmax(-1).mean(0).clamp_min(1e-12).sqrt() for i in range(b)])
        if keys is not None:
            hid["keys"] = keys                                 # {layer: [B, n, H]} states at the appended tokens
        if self._read is not None:
            r, self._read = self._read, None
            g = r["idx"].unsqueeze(-1)
            # {L: (state after layer L at the sampled text positions [B, n, H], attention mass on the video cells [B, n, K])}
            hid["read"] = {L: (self.captured[L].gather(1, g.expand(-1, -1, self.captured[L].shape[-1])), r["snap"][L])
                           for L in self.read_upto}
            hid["read_ok"] = r["ok"]
        if self.text_take > 0:                                 # the language stream: random positions after the evidence
            txt = attention_mask.bool() & ~vis & (vis.cumsum(1) == k)
            score = torch.rand(txt.shape, device=txt.device).masked_fill(~txt, -1.0)
            val, idx = score.topk(min(self.text_take, txt.shape[1]), dim=1)
            ok = val >= 0
            hid["text"] = {layer: t.gather(1, idx.unsqueeze(-1).expand(-1, -1, t.shape[-1]))[ok]
                           for layer, t in self.captured.items()}
            hid["text_ex"] = torch.arange(b, device=txt.device).unsqueeze(1).expand_as(idx)[ok]
            hid["text_set"] = ({layer: t.gather(1, idx.unsqueeze(-1).expand(-1, -1, t.shape[-1]))
                                for layer, t in self.captured.items()}, ok)        # the same positions, per example
        return logits, tgt[sel], ex_id, hid


def answer_ce(logits, targets, ex_id, batch_size):
    """Per-example mean CE over answer tokens, then mean over examples that have answer tokens (spec 6.2)."""
    tok = F.cross_entropy(logits, targets, reduction="none")
    per_ex = torch.zeros(batch_size, device=tok.device).index_add_(0, ex_id, tok)
    n = torch.zeros(batch_size, device=tok.device).index_add_(0, ex_id, torch.ones_like(tok))
    has = n > 0
    return (per_ex[has] / n[has]).mean(), n.sum()


def answer_nll(logits, targets, ex_id, batch_size, unit: str = "token"):
    """-> (per-example answer NLL [B], has-answer mask [B]); unit "token" = mean over answer tokens, "sum" = total
    (log p(answer | video) as a whole: the same margin means the same evidence gain for short and long answers)"""
    tok = F.cross_entropy(logits.float(), targets, reduction="none")
    per_ex = torch.zeros(batch_size, device=tok.device).index_add_(0, ex_id, tok)
    n = torch.zeros(batch_size, device=tok.device).index_add_(0, ex_id, torch.ones_like(tok))
    return (per_ex if unit == "sum" else per_ex / n.clamp_min(1)), n > 0


def latent_mask(b, n_tub, cells, device, gen=None, scale="small"):
    """V-JEPA-style masks on the pooled evidence grid (t-major, then y, x) -> [B, n_tub * cells * cells] bool.
    half 'future' (the last 1/4 - 1/2 of the tubelets, all cells): predict what comes next;
    half 'tube' (a spatial block of 1/2 - 3/4 side, all tubelets): predict what is hidden in space."""
    m = torch.zeros(b, n_tub, cells, cells, dtype=torch.bool)
    r = lambda lo, hi: int(torch.randint(lo, hi + 1, (1,), generator=gen))
    for i in range(b):
        big = scale == "big"                 # V-JEPA-like large targets: local interpolation cannot solve them
        if r(0, 1) == 0:
            lo_k, hi_k = (max(1, n_tub // 3), max(1, n_tub // 2)) if big else (max(1, n_tub // 4), max(1, n_tub // 2))
            k = r(lo_k, hi_k); m[i, n_tub - k:] = True
        else:
            lo_s, hi_s = (max(1, (3 * cells) // 5), max(1, (4 * cells) // 5)) if big else (max(1, cells // 2), max(1, 3 * cells // 4))
            h, w = r(lo_s, hi_s), r(lo_s, hi_s)
            y, x = r(0, cells - h), r(0, cells - w); m[i, :, y:y + h, x:x + w] = True
    return m.view(b, -1).to(device)


class Stage1Model(nn.Module):
    def __init__(self, vision, llm, retention_layers=None, evidence_cells: int = 4, bridge_hidden: int = 2048,
                 surprise_segments: int = 0, train_vision: str = "none", latent_predict: bool = False,
                 predict_layers=None, mask_scale: str = "small", geo_layers=None, geo_kw=None,
                 info: str = None, info_kw=None, n_tubelets: int = 8, spectral=None, eq_state=None,
                 token_order: str = "tyx", channel=None, expert: str = None, word_drop: float = 0.0, corr=None, wstruct=None,
                 nextp=None, codes=None, textread=None, depw=None, vcode=None, vtie=None, entry_layer=None, memory=False, dyncon=None):
        """surprise_segments = S > 0: the clip's tubelets are cut into S + 1 equal segments; segment j >= 1 is predicted
        by the frozen predictor from everything before it (tokens removed before the encoder), and the evidence of
        those tubelets also carries observed - predicted (the world model's surprise). Segment 0 has no expectation
        (residual 0). Needs a model whose predictor works in its own encoder space (g1B).
        train_vision: "none" (frozen V-JEPA), "encoder" (encoder fine-tuned, predictor frozen) or "full" (encoder and
        predictor fine-tuned; the predictor only receives gradients from the native loss). When V-JEPA is trained, a
        frozen copy (reference E0/P0) supplies the retention targets and the surprise expectations, so these stay
        anchored to the original world model.
        latent_predict: the JEPA objective inside the LLM (alignment target = V-JEPA's latent space, not only text): a
        second LLM pass sees the evidence with some cells replaced by a mask token, and heads on the LLM hidden states
        at those cells must predict the (frozen-reference) V-JEPA states there. Needs retention_layers (hidden taps)."""
        super().__init__()
        assert train_vision in ("none", "encoder", "full"), train_vision
        self.vision, self.train_vision = vision, train_vision
        self.reference = vision.frozen_reference() if train_vision != "none" else None
        for n, p in self.vision.named_parameters():
            p.requires_grad_(train_vision == "full" or (train_vision == "encoder" and n.startswith("encoder.")))
        spec, hidden = vision.spec, llm.config.hidden_size
        self.cells, self.surprise_segments = evidence_cells, surprise_segments
        if surprise_segments:
            assert (spec.pred_levels, spec.pred_dim) == (spec.n_levels, spec.level_dim), \
                f"{vision.variant}: the predictor does not predict its own encoder space; surprise needs g1B"
        pred_layers = list(predict_layers or retention_layers or [])
        self.mask_scale = mask_scale
        # eq_state: {"layers": [...], "stats": path, "k": 2048} -> EqualisedStateHeads (preservation in V-JEPA's geometry)
        self.core = LanguageCore(llm, sorted(set(retention_layers or ()) | set(pred_layers if latent_predict else ()) | set(geo_layers or ())
                                             | set((eq_state or {}).get("layers", ()))))
        self.bridge = EvidenceBridge({OBSERVED: spec.n_levels * spec.level_dim}, hidden, hidden=bridge_hidden,
                                     max_cells=max(4, evidence_cells), mask_token=latent_predict,
                                     residual_dim=spec.n_levels * spec.level_dim if surprise_segments else 0)
        self.eq_heads, self.equaliser = None, None
        if eq_state:
            self.equaliser = Equaliser(eq_state["stats"], eq_state.get("k", 2048))
            # "stream": weight of the same constraint at the text positions (0 = evidence positions only)
            self.stream_weight = float(eq_state.get("stream", 0.0))
            # "read": {"weight", "take"} -> EqualisedStateHeads.read (what a word looks at must arrive whole)
            self.read = dict(eq_state["read"]) if eq_state.get("read") else None
            if self.read:
                self.core.enable_read(int(self.read.get("take", 32)), eq_state["layers"])
            # "bind": {"n": keys per example, "keys": coarse directions in a key, "weight"} -> EqualisedStateHeads.bind
            self.bind = dict(eq_state["bind"]) if eq_state.get("bind") else None
            self.eq_heads = EqualisedStateHeads(hidden, eq_state["layers"], self.equaliser.k,
                                                stream_tubelets=n_tubelets if self.stream_weight > 0 else 0,
                                                bind_keys=int(self.bind["keys"]) if self.bind else 0, read=bool(self.read),
                                                stream_set=eq_state.get("stream_mode", "each") == "set")
            if self.stream_weight > 0:
                self.core.text_take = int(eq_state.get("stream_take", 32))
        spectral_args = dict(spectral) if spectral else None
        if spectral:                                   # equalised interface (SpectralBridge): {"stats", "k", "alpha", "mode"}
            spectral = dict(spectral)
            tw = spectral.pop("time_words", None)      # gain: tokens carry the LLM's embedding of their integer second
            vocab = spectral.pop("vocab", None)        # (10-19) tokens as mixtures of the LLM's own input embeddings (SpectralBridge.to_words)
            emb_rms = float(llm.get_input_embeddings().weight.float().pow(2).mean().sqrt())
            self.bridge = SpectralBridge(out_dim=hidden, hidden=bridge_hidden, max_cells=max(4, evidence_cells),
                                         mask_token=latent_predict, token_rms=spectral.pop("gain", emb_rms), **spectral)
            if vocab:
                emb_ = llm.get_input_embeddings().weight.detach()
                if isinstance(vocab, dict) and vocab.get("filter") == "words":     # only whole English words (" " + 3+ lowercase letters)
                    import re as _re
                    from transformers import AutoTokenizer
                    t_ = AutoTokenizer.from_pretrained(llm.config._name_or_path)
                    keep = [i for i in range(emb_.shape[0]) if _re.fullmatch(r" [a-z]{3,}", t_.decode([i]) or "")]
                    emb_ = emb_[keep]
                self.bridge.vocab_emb = emb_.to(torch.bfloat16).clone()
                self.bridge.vocab_heads = int(vocab.get("heads", 24)) if isinstance(vocab, dict) else 24
                self.bridge.vocab_scale = nn.Parameter(torch.full((1,), __import__("math").log(float(vocab.get("scale", 30.0)) if isinstance(vocab, dict) else 30.0)))
            if tw:
                from transformers import AutoTokenizer
                t_ = AutoTokenizer.from_pretrained(llm.config._name_or_path)
                ids = [t_(str(n), add_special_tokens=False)["input_ids"] for n in range(1000)]
                assert all(len(i) == 1 for i in ids), "integers 0..999 must be single tokens"
                self.bridge.time_words = llm.get_input_embeddings().weight.detach().float()[[i[0] for i in ids]].clone()
                self.bridge.tw_gain = float(tw)
        self.core.word_drop = float(word_drop)
        self.core.blank = llm.get_input_embeddings().weight.detach().float().mean(0)      # fixed; read before the LLM is sharded
        if expert:                                     # "copy" | "shared": the LLM is taught only through the video positions
            self.core.add_expert(expert)
        self.bridge.order = token_order
        if channel:                                    # {"joint": path, "energy"}: the tokens enter in the LLM's perception channel
            from jepa_llm.training.nullspace import channel_basis
            U, V = channel_basis(channel["joint"], float(channel.get("energy", 0.95)), hidden)
            self.bridge.channel = ("drop", U) if U.shape[1] <= V.shape[1] else ("keep", V)
        self.retention = RetentionHeads(hidden, retention_layers, spec.n_levels, spec.level_dim) if retention_layers else None
        assert not latent_predict or pred_layers, "latent prediction needs LLM layers to read (predict_layers)"
        self.predictor_heads = RetentionHeads(hidden, pred_layers, spec.n_levels, spec.level_dim) if latent_predict else None
        # info: one objective that changes what alignment asks of the LLM (both InfoNCE against frozen V-JEPA)
        #   "ipa" information-preserving alignment: the LLM's final states over a neutral instruction ("Describe the
        #         video.", the same text for every clip, so only the video can make them differ) must identify the clip's
        #         V-JEPA state among a large bank: alignment = text likelihood + I(LLM ; V-JEPA)
        #   "upm" one predictive model: the final state of every video step t (the state the LM head reads for the next
        #         token) must predict the V-JEPA state of steps t+1..t+K against the clip's other moments and other clips
        #         (CPC): the LLM models the video stream in V-JEPA's space as it models text
        self.info, self.n_tub = info, n_tubelets
        self.critic, self.neutral = None, None
        if info:
            assert info in ("ipa", "upm"), info
            kw = dict(info_kw or {})
            self.horizons = int(kw.pop("horizons", 3)) if info == "upm" else 0
            tdim = spec.n_levels * spec.level_dim * (n_tubelets if info == "ipa" else 1)
            self.critic = ContrastiveCritic(hidden, tdim, n_heads=max(1, self.horizons), **kw)
        # geo_layers: world-model geometry at the read position (GeometryHead); teacher = per-tubelet clip state
        self.geo = None
        if geo_layers:
            n_tub = 8 if not geo_kw else geo_kw.pop("n_tubelets", 8)
            self.geo = GeometryHead(hidden, geo_layers, n_tub * spec.n_levels * spec.level_dim, **(geo_kw or {}))
        # corr = {"bases": path}: the video positions get a task of their own. They carry no loss otherwise and stay as
        # the connector made them (main1b_SPEC @8000: a layer changes a video position by 4-22% of its size, a text
        # position by 60-200%; video tokens never look at each other). Every video token must say, from its own final
        # state alone, which cell of the moment BEFORE it was. The answer is V-JEPA's own patch correspondence (nearest
        # patch in its feature space, kept when the 9 patches of a cell agree and come back: 96-99% right against human
        # masks, logs/zz_corr_gt2.py). Copying its own place cannot solve it for what moved: the layers have to find the
        # match among the earlier tokens and write the result at this position, where text can then read it.
        self.corr_head, self._zn, self._corr_geo = None, None, None
        # wstruct = {"bank": n}: V-JEPA's structure planted in the LLM's word space, without any text. The fixed LLM reads
        # the video tokens; "what it is about to say" = its next-word distribution over its whole vocabulary (mean over the
        # video positions). Videos that are alike / different in V-JEPA's own space (clip mean of the equalised tokens) must
        # get alike / different word distributions: the two similarity structures are aligned (linear CKA over the batch and
        # a bank of recent videos; no temperature, no chosen words, no read-out head: nothing can hide outside the words).
        # After a caption stage 1 the V-JEPA neighbours of a video are 23-26% of its neighbours in word space, untrained 10%
        # (logs/zz_lang_struct.py).
        self.wstruct, self._ws_bank = wstruct, None
        if wstruct:
            self.core.ws = True
        if corr:
            assert evidence_cells == 8, "corr: 8x8 cells of 3x3 patches"
            pb = torch.load(corr["bases"], map_location="cpu")["patch"]
            self._corr_basis = (pb["mean"].float(), pb["basis"].float())
            self.corr_head = nn.Linear(hidden, evidence_cells * evidence_cells)
        # nextp = {"k": 2048} (10-13): the LLM as V-JEPA's predictor, video only. V-JEPA predicts the next frames in its own
        # space and an LLM predicts the next token; here the LLM, reading the video tokens moment by moment, predicts the
        # NEXT MOMENT's tokens in V-JEPA's equalised space: the final state at the position of cell (t, y, x) must give the
        # equalised state of the same cell at moment t + 1 (causal attention: it has seen the earlier moments and the cells
        # of moment t before it). The error is measured in equalised units, every direction weighs the same, so the quiet
        # directions - which V-JEPA's own predictor cannot express (384-wide bottleneck) and where the old raw-unit test
        # saw only 'copy' - have to be predicted too. No text is involved in this loss.
        self.next_head = nn.Linear(hidden, int(nextp.get("k", 2048))) if nextp else None
        # codes = {"book": path, "tok_ids": [256 token ids], "mode": "all" | "next"} (10-14): the same video-only idea but
        # THROUGH THE LANGUAGE HEAD. Every video-only stage 1 whose output left the LLM through a separate head (REL, WS, the
        # read, next-scene regression: pair loss in stage 2 = no stage 1) did not carry over; the one that made the LLM WRITE
        # tokens that depend on the video did (RC, +8.6 MVP with ~10 bits per clip). Here the LLM writes V-JEPA's own state:
        # the equalised region states (moments x 2x2 regions) are residual-quantised (logs/zz_rvq.py: 4 levels x 256, error
        # in equalised units) and the answer tokens of the row are replaced by those code numbers ("0".."255", one token
        # each), region after region in time order. mode "all": the codes of the video it sees. mode "next": the row shows
        # only its first moments (dataset column `see`) and the LLM writes the codes of the moments AFTER them - next-scene
        # prediction as next-token prediction. No human text anywhere.
        # textread = {"see": 6, "k": 256, "pos": 4} (10-14): stage 1, video only. The LLM gets the first `see` moments (encoded on
        # their own: no leak) followed by a fixed instruction; at the last `pos` text positions before the answer - where the
        # LLM starts to write when it answers - a single linear read-out must give the spatiotemporal content of the clip in
        # V-JEPA's equalised space: the state of every moment x 2x2 region (top `k` directions), the seen moments AND the
        # ones after them. The text positions hold no video token: the LLM has to gather the video through attention and
        # compare moments to get the changes and the future, and the result has to sit where answers are produced.
        # depw = {"floor": 0.2} (10-14): stage 2 trained on WHAT THE VIDEO DECIDES. With plain next-token CE every answer token
        # counts the same, and most of them (the wording, the parts the question or the previous words already give away)
        # are predicted without the video: the cheapest solution - video positions idle, a little gist pulled into the
        # text - wins, and it erased the stage-1 reading twice (text->video attention 49.6% -> 6.6%). Here every answer token
        # is weighted by how much the video decides it: gain = NLL(token | another clip of the batch) - NLL(token | its clip),
        # measured by the model itself in the same step (one extra forward without gradient); weight = floor + gain / mean
        # gain, normalised to mean 1. Same data, same targets, full fine-tuning; tokens no video decides keep a small weight.
        self.depw = dict(depw) if depw else None
        # vcode = {"book": rvq16.pt, "weight": 1.0} (10-14): the video becomes part of the sequence the LLM models. An LLM learns one
        # thing, predicting the next token, and positions without a loss sit idle (video positions: ~80% of their attention on
        # the first token, unchanged over training). V-JEPA's space is predictive (the next moment can be predicted from the
        # past), so the video positions get the LLM's own job: each predicts the next moment's V-JEPA state, written as
        # discrete codes (equalised states of 4x4 places, residual-quantised 4 x 1024; runs/tok1/rvq16.pt) that are NEW TOKENS OF
        # THE LLM'S VOCABULARY - same output layer, same cross-entropy as the text, at every step of every stage. Position
        # (t, y, x) predicts, for the place containing its cell, the code of moment t + 1 at level (y % 2) * 2 + x % 2.
        self.vcode = None
        if vcode:
            st = torch.load(vcode["book"], map_location="cpu"); L, K = st["books"].shape[:2]; base = llm.get_input_embeddings().weight.shape[0]
            llm.resize_token_embeddings(base + L * K, mean_resizing=False)
            self.core.vword_gain = nn.Parameter(torch.ones(1))       # in the LLM core: trained, saved and loaded with it
            self.vcode = {"input": bool(vcode.get("input", False)), "mode": vcode.get("mode", "state"), "weight": float(vcode.get("weight", 1.0)), "base": base, "L": L, "K": K, "mu": st["mu"].float(), "sd": st["sd"].float(), "books": st["books"].float()}
        # vtie = {"weight": 1.0} (10-14, after vcode): the same job without a dictionary. The codes keep 40% of the variance and
        # read SSv2 at 23.6 against 43.1 for the states (runs/tok1), and the LLM that learned them held LESS of the video (SSv2
        # read-out from its states 19 against 22-25). Llama 3.2 ties its input and output word tables: the logit of a word is
        # the dot product of the final state with the word's input vector. A video token's input vector is its projected
        # V-JEPA state, so it is a word of its own: position (t, cell) predicts the token (t + 1, same cell) with logits =
        # final state . [word table ; projected states of every video token of the batch] and one CE over both (the text
        # words, the same cell now, the other cells and the other clips are the alternatives). Targets carry no gradient
        # (as V-JEPA's own target encoder), so the projector cannot make the video easier to predict by dropping information.
        self.vtie = dict(vtie) if vtie else None
        # dyncon (10-19): caption / QA training also carries V-JEPA's dynamics into what the LLM reads. Diving48 (47 dives, same pool, same
        # divers: dynamics only): V-JEPA's tokens read with a temporal reader 31.0%, the caption/QA-trained LLM's state after reading 20-22% -
        # a third of the dynamics never reaches the language stream, because the answer text only asks for what it says. The mean final
        # state over the answer positions (what the LLM holds while it writes) must pick its own clip's V-JEPA summary (a small attentive
        # reader over the equalised tokens) against the same clip with its moments shuffled (identical appearance, other dynamics) and the
        # other clips of the batch. Training only: the reader and projection are dropped at inference.
        self.dyn_head, self.dyncon = None, (dict(dyncon) if dyncon else None)
        if dyncon and dyncon.get("mode") == "mask":       # (10-25) masked-moment dynamics prediction (_DynMask)
            self.dyn_head = _DynMask(hidden, int((spectral or {}).get("k", 2048)), int(dyncon.get("layer", 18)), int(dyncon.get("moments", 2)), float(dyncon.get("p_row", 0.5)))
            if self.bridge.mask_token is None:
                self.bridge.mask_token = nn.Parameter(torch.zeros(hidden))
            self.core.llm.model.layers[self.dyn_head.layer - 1].register_forward_hook(self.core._hook(self.dyn_head.layer))
        elif dyncon and dyncon.get("mode") == "flowpred": # (11-02 night) the LLM computes V-JEPA's latent flow at its video positions (_FlowPred)
            self.dyn_head = _FlowPred(hidden); self.flow_target = True
        elif dyncon and dyncon.get("mode") == "self":     # (10-26) reconstruction inside the LLM at the video positions (_RecSelf)
            self.dyn_head = _RecSelf(hidden, n_tubelets, int((spectral or {}).get("k", 2048)))
        elif dyncon and dyncon.get("mode") == "struct":   # (11-01) head-free structure preservation at the video positions (_StructKeep)
            self.dyn_head = _StructKeep(evidence_cells * evidence_cells)
            self.struct_layers = [int(a) for a in dyncon.get("layers", (7, 14, 21))]
            for a in self.struct_layers:
                self.core.llm.model.layers[a - 1].register_forward_hook(self.core._hook(f"sk{a}"))
        elif dyncon and dyncon.get("mode") == "cf":       # (11-01 v3) dynamics counterfactual by substitution (_CfUse)
            self.dyn_head = _CfUse()
        elif dyncon and dyncon.get("mode") == "vfloor":   # (11-02) every layer's video tokens keep V-JEPA's information, one-sided (_VFloor)
            self.dyn_head = _VFloor(evidence_cells * evidence_cells, float(dyncon.get("tau", 0.05)))
            self.vf_layers = list(range(1, len(self.core.llm.model.layers) + 1))
            for a_ in self.vf_layers:
                self.core.llm.model.layers[a_ - 1].register_forward_hook(self.core._hook(f"vf{a_}"))
        elif dyncon and dyncon.get("mode") == "route":    # (11-01 v5) the LLM's attention follows V-JEPA's temporal correspondence (_RouteKeep)
            nl_ = len(self.core.llm.model.layers)
            self.dyn_head = _RouteKeep(evidence_cells * evidence_cells, nl_, float(dyncon.get("tau", 0.05)))
            for i_, lay_ in enumerate(self.core.llm.model.layers):
                lay_.self_attn.q_norm.register_forward_hook(self.core._hook(f"rq{i_}"))
                lay_.self_attn.k_norm.register_forward_hook(self.core._hook(f"rk{i_}"))
        elif dyncon and dyncon.get("mode") == "ans":      # (11-01 v4) the dynamics loss at the answer state (_DynAns)
            self.dyn_head = _DynAns(hidden, int(dyncon.get("n_tub", 16)), evidence_cells * evidence_cells, int(dyncon.get("dirs", 64)), int(dyncon.get("regions", 2)))
        elif dyncon and dyncon.get("mode") == "read":     # (11-01 v2) head-free read-out of the motion field by the text stream (_ReadStruct)
            self.dyn_head = _ReadStruct(evidence_cells * evidence_cells, hidden, int(dyncon.get("proj", 32)))
            self.read_layers = [int(a) for a in dyncon.get("layers", (14, 21))]
            for a in self.read_layers:
                self.core.llm.model.layers[a - 1].register_forward_hook(self.core._hook(f"rs{a}"))
        elif dyncon and dyncon.get("mode") == "next":     # (10-31) next-moment change prediction at the video positions (_DynNext)
            self.dyn_head = _DynNext(hidden, int((spectral or {}).get("k", 2048)), evidence_cells * evidence_cells, int(dyncon.get("dim", 1024)),
                                    tuple(dyncon.get("horizons", (1,))))
        elif dyncon and dyncon.get("mode") == "rec2":     # (10-25) reconstruction v2 from the language stream at every depth (_RecLang)
            nl_ = len(self.core.llm.model.layers)
            self.dyn_head = _RecLang(hidden, n_tubelets * evidence_cells * evidence_cells, n_tubelets, nl_, int((spectral or {}).get("k", 2048)),
                                     int(dyncon.get("dim", 512)), int(dyncon.get("heads", 8)), nl_ + 1 if dyncon.get("layer") == "final" else int(dyncon.get("layer", 0)), bool(dyncon.get("pyramid", False)), str(dyncon.get("norm", "ln")),
                                     linear=dyncon.get("decoder") == "linear")
            self.dyn_head.objective, self.dyn_head.tau = str(dyncon.get("objective", "mse")), float(dyncon.get("tau", 0.1))   # (11-01) "nce": discrimination
            self.dyn_head.tau_t = float(dyncon.get("tau_t", 0.05))
            self.dyn_head.bands = bool(dyncon.get("bands", False))                                                            # (11-02) Haar time bands in the pyramid
            self.dyn_head.balance = dyncon.get("balance", False)
            self.dyn_head.floor = bool(dyncon.get("floor", False))                                                              # (11-02) one-sided loss                                                         # (11-01 v4) gradient-norm balance                                                               # (11-01) nce_soft target temperature
            self._rec_cap = None
            def _rec_hook(j):
                def hook(_m, _i, out):
                    if self.dyn_head.want == j:
                        self._rec_cap = out[0] if isinstance(out, tuple) else out
                return hook
            for j_, lay_ in enumerate(self.core.llm.model.layers):
                lay_.register_forward_hook(_rec_hook(j_ + 1))
            # "final" = the LLM's final representation: the output of its final norm (what the LM head reads). The last decoder layer's raw residual
            # output carries a few huge channels that swamp a LayerNorm-ed readout (nx11 smoke: R^2 flat at -0.03 for 200 steps reading it).
            self.core.llm.model.norm.register_forward_hook(_rec_hook(nl_ + 1))
            # (10-27) vpres: the video positions themselves keep V-JEPA's tokens, all equalised directions at equal weight, through one linear map per
            # read point (zero init, space-time pyramid as above), weight dyncon.vpres.weight, IN ADDITION to the language-stream reconstruction.
            # Measured (logs/zz_presv13.py, nx13 @8k): the finest bands survive the bridge (R^2 0.58) but only 0.27 at the final video positions.
            # Read points = where that was measured: "mid" (output of layer index n/2, as zz_presv13) and "final" (final norm output). The 10-26
            # _RecSelf trial REPLACED this reconstruction; here both run. Training only.
            vp_ = dyncon.get("vpres")
            self.vpres_at, self.vpres_w = [], 0.0
            if vp_:
                self.vpres_at = [nl_ // 2 if a == "mid" else a for a in vp_.get("at", ["mid", "final"])]
                if vp_.get("every"):                   # every k-th layer's output + the final representation
                    k_ = int(vp_["every"]); self.vpres_at = list(range(k_ - 1, nl_ - 1, k_)) + ["final"]
                self.vpres_w = float(vp_.get("weight", 0.1))
                self.dyn_head.vp = nn.ModuleList([nn.Linear(hidden, int((spectral or {}).get("k", 2048))) for _ in self.vpres_at])
                for m_ in self.dyn_head.vp:
                    nn.init.zeros_(m_.weight); nn.init.zeros_(m_.bias)
                for a in self.vpres_at:
                    if a != "final":
                        self.core.llm.model.layers[a].register_forward_hook(self.core._hook(f"vp{a}"))
        elif dyncon and dyncon.get("mode") == "corr":     # (10-25) causal correspondence distillation (_DynCorr)
            self.dyn_head = _DynCorr(hidden, n_tubelets, evidence_cells * evidence_cells, float(dyncon.get("tau", 0.05)), dyncon.get("prior"))
        elif dyncon and dyncon.get("mode") == "time":     # (10-25) change reading + dynamic summary (_DynTime)
            self.dyn_head = _DynTime(hidden, n_tubelets, int((spectral or {}).get("k", 2048)), int(dyncon.get("layer", 18)), int(dyncon.get("dirs", 128)),
                                     int(dyncon.get("regions", 4)), float(dyncon.get("target_scale", 1.0)), bool(dyncon.get("summary", True)))
            self.core.llm.model.layers[self.dyn_head.layer - 1].register_forward_hook(self.core._hook(self.dyn_head.layer))
        elif dyncon:
            dd = int(dyncon.get("dim", 512)); kin = int((spectral or {}).get("k", 2048)); nk = n_tubelets * evidence_cells * evidence_cells
            self.dyn_head = (_DynRecon(hidden, n_tubelets, int(dyncon.get("dirs", 256)), int(dyncon.get("regions", 2)),
                                       target=str(dyncon.get("target", "top")), kin=kin, scale=float(dyncon.get("target_scale", 1.0))) if dyncon.get("mode") == "recon"
                             else _DynHead(kin, hidden, dd, nk, int(dyncon.get("perms", 3)), float(dyncon.get("tau", 0.07))))
            self.dyn_head.neg = str(dyncon.get("neg", "shuffle"))     # "hard" (10-20): adjacent swaps + short reversed spans
        self.mem_cells, self._mem_args = 0, None
        if isinstance(memory, dict) and "cells" not in memory:        # (10-16) video-only memory read by the listed layers
            self.core.enable_memory(memory.get("layers"))
        elif isinstance(memory, dict):  # (10-15) hybrid: the pooled tokens stay in the sequence (the LLM relates them, as before) and the
            # full grid (memory["cells"] per side, no pooling) is ALSO read as memory by every layer, through its own projector (equalised
            # with the full grid's statistics, memory["stats"]). Video-only memory (memory: true) described captions better than the
            # sequence (caption loss lower from step 2500) but separated MVP pairs worse (pair loss stuck at 0.17, baseline 0.12).
            sa = {k_: v_ for k_, v_ in spectral_args.items() if k_ not in ("time_words",)}; sa["stats"] = memory["stats"]
            self.mem_bridge = SpectralBridge(out_dim=hidden, hidden=bridge_hidden, max_cells=int(memory["cells"]),
                                             token_rms=sa.pop("gain", float(llm.get_input_embeddings().weight.float().pow(2).mean().sqrt())), **sa)
            self.mem_cells = int(memory["cells"]); self.core.enable_memory(); self.core.hybrid = True
        elif memory:                    # the video as memory read by every decoder layer (LanguageCore.enable_memory)
            self.core.enable_memory()
        if entry_layer:                 # V-JEPA tokens enter at a middle decoder layer (LanguageCore.enable_entry); gain trained with the bridge
            self.bridge.entry_gain = nn.Parameter(torch.ones(1))
            self.core.enable_entry(int(entry_layer), self.bridge.entry_gain)
        self.textread = None
        if textread:
            self.textread = dict(textread); kk = int(textread.get("k", 256))
            self.tr_head = nn.Linear(hidden, n_tubelets * 4 * kk)
        self.codes = None
        if codes:
            st = torch.load(codes["book"], map_location="cpu")
            self.codes = {"moments": int(codes.get("moments", 1)), "mode": codes.get("mode", "all"), "mu": st["mu"].float(), "sd": st["sd"].float(), "books": st["books"].float(),
                          "tok": torch.tensor(codes["tok_ids"], dtype=torch.long)}

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision.eval()  # V-JEPA runs deterministically (no drop-path) whether frozen or fine-tuned
        return self

    def anchor(self):
        """the frozen world model: the reference copy when V-JEPA is trained, otherwise V-JEPA itself"""
        return self.reference if self.reference is not None else self.vision

    def evidence(self, video: torch.Tensor, frame_times: torch.Tensor, region_grid: torch.Tensor = None):
        """video [B,3,T,H,W], region_grid [B,4] token-grid (y0,y1,x0,x1) or None (whole frame)
        -> (pooled states [B,K,L,D] fp32, coords [K,3], seconds [B,K], cell geometry [B,K,4][, pooled surprise])."""
        out, _ = self._evidence(video, frame_times, region_grid, need_anchor=False)
        return out

    def enable_flow(self, n_feat: int = 9):
        """(11-02) V-JEPA latent flow into the visual tokens: a small MLP (zero-initialised output, so training starts from the plain model)"""
        H = self.core.llm.config.hidden_size
        self.bridge.flow_proj = nn.Sequential(nn.Linear(n_feat, 256), nn.GELU(), nn.Linear(256, H))
        nn.init.zeros_(self.bridge.flow_proj[2].weight); nn.init.zeros_(self.bridge.flow_proj[2].bias)
        self.flow_feats = True

    @torch.no_grad()
    def latent_flow(self, zn, n_tub, region_grid, tau: float = 0.05):
        """(11-02 night) flow_lags (default (1,)): the 9 features for each lag (source = moment t - lag), concatenated -> 9 x len(lags)"""
        lags = tuple(getattr(self, "flow_lags", (1,)))
        if lags == (1,): return self._latent_flow1(zn, n_tub, region_grid, tau, 1)
        return torch.cat([self._latent_flow1(zn, n_tub, region_grid, tau, L) for L in lags], -1)

    @torch.no_grad()
    def _latent_flow1(self, zn, n_tub, region_grid, tau: float = 0.05, lag: int = 1):
        """(11-02) zn [B, n_tub*576, L, D] (V-JEPA patch states, 24 x 24 per moment) -> per pooled cell [B, n_tub*cells^2, 9]: for moment t >= 1,
        each patch's source at t-1 = softmax over the patches at t-1 of cos / tau (as zz_vjepa_rotation.py), displacement d = position - source;
        per patch: dx, dy, |d|, local curl and divergence of the d field, correspondence confidence (max probability); per moment (every patch):
        curl about the frame centre, mean dx, mean dy. Moment 0: zeros. Pooled to the cells like the states (pool_regions, crops included)."""
        b = zn.shape[0]; G = 24; T = n_tub
        z = F.normalize(zn.float().reshape(b, T, G * G, -1), dim=-1)
        yy, xx = torch.meshgrid(torch.arange(G, device=zn.device, dtype=torch.float32), torch.arange(G, device=zn.device, dtype=torch.float32), indexing="ij")
        pos = torch.stack([xx, yy], -1).view(G * G, 2) - (G - 1) / 2
        f = torch.zeros(b, T, G * G, 9, device=zn.device)
        for t in range(lag, T):
            a = (torch.einsum("bpd,bqd->bpq", z[:, t], z[:, t - lag]) / tau).softmax(-1)        # [b, 576 at t, 576 at t-lag]
            d = pos - a @ pos                                                                       # [b, 576, 2]
            if getattr(self, "flow_clean", False):
                # (11-03 night) a diffuse correspondence puts the source at the frame centre and d = the patch's own position (a fake outward
                # flow on most patches - mean max-probability 0.1 - 0.3; flow2 lost moving-direction 26 points): d scaled by the
                # correspondence confidence (max probability, 1/576 -> 0, 1 -> 1); the frame's mean motion (camera) taken out of the cells
                c_ = ((a.max(-1).values - 1.0 / (G * G)) / (1 - 1.0 / (G * G))).clamp(0, 1).unsqueeze(-1)
                d = d * c_; cam_ = d.mean(1, keepdim=True)
                d = d - cam_
            dg = d.view(b, G, G, 2)
            ddx_dy, ddx_dx = torch.gradient(dg[..., 0], dim=(1, 2)); ddy_dy, ddy_dx = torch.gradient(dg[..., 1], dim=(1, 2))
            curl_l = (ddy_dx - ddx_dy).reshape(b, G * G); div_l = (ddx_dx + ddy_dy).reshape(b, G * G)
            curl_g = (pos[:, 0] * d[..., 1] - pos[:, 1] * d[..., 0]).mean(-1, keepdim=True).expand(b, G * G)
            if getattr(self, "flow_clean", False):        # camera motion (taken out of d above) as the per-moment global features
                gx_, gy_ = cam_[..., 0].expand(b, G * G), cam_[..., 1].expand(b, G * G)
            else:
                gx_, gy_ = d[..., 0].mean(-1, keepdim=True).expand(b, G * G), d[..., 1].mean(-1, keepdim=True).expand(b, G * G)
            if getattr(self, "flow_global", False):
                # (11-03 night) frame-level flow only: per-cell displacement is noise for small objects (a rule on it: 22 - 27 % on TVBench
                # moving direction, chance 25 %; every per-cell flow input lost 10 - 26 points there) while the frame-level motion gained
                # in every version (speed +16 - 20, scene transition +13 - 20, action count +9 - 13). Per moment, broadcast to every cell:
                # global curl, camera dx / dy, mean |d|, mean confidence, mean |local curl|, mean |local divergence|, 0, 0
                gm_ = lambda x_: x_.mean(-1, keepdim=True).expand(b, G * G)
                f[:, t] = torch.stack([curl_g, gx_, gy_, gm_(d.norm(dim=-1)), gm_(a.max(-1).values), gm_(curl_l.abs()), gm_(div_l.abs()),
                                       torch.zeros_like(curl_g), torch.zeros_like(curl_g)], -1)
            else:
                f[:, t] = torch.stack([d[..., 0], d[..., 1], d.norm(dim=-1), curl_l, div_l, a.max(-1).values, curl_g, gx_, gy_], -1)
        pf, _, _ = pool_regions(f.view(b, T * G * G, 1, 9), n_tub, self.cells, region_grid)
        return pf.reshape(b, pf.shape[1], 9)

    def _evidence(self, video, frame_times, region_grid, need_anchor):
        """-> (evidence tuple as in evidence(), pooled anchor states or None). The frozen world model is run at most
        once and shared by the surprise residual and the retention target."""
        if video.shape[-1] != video.shape[-2] or video.dim() == 4:
            # (10-15) precomputed features (data feature store): video is already the pooled, level-normalised V-JEPA grid
            # [B, n_tub * cells^2, L, D] - the encoder is frozen, so encoding once and reading it back is the same computation
            b, n_tub = video.shape[0], frame_times.shape[-1] // 2
            if region_grid is None:
                region_grid = torch.tensor([[0, 24, 0, 24]] * b)
            pooled = video.float()
            _, coords, geom = pool_regions(torch.zeros(b, n_tub * 576, 1, 1), n_tub, self.cells, region_grid)
            tub_sec = frame_times.view(b, n_tub, 2).mean(-1)
            return (pooled, coords.to(video.device), tub_sec.repeat_interleave(self.cells * self.cells, dim=1), geom.to(video.device)), None
        with torch.set_grad_enabled(self.train_vision != "none" and torch.is_grad_enabled()):
            if getattr(self, "causal_win", 0):
                z = self.causal_encode(video)
            elif getattr(self, "chunk_frames", 0) and video.shape[2] > self.chunk_frames:
                # (10-16) long clips in chunks of chunk_frames (what V-JEPA was trained on, and what the equalisation statistics were
                # taken from): each chunk encoded alone, its tokens kept in time order -> the same layout as one long encoding
                b_, c_, t_, h_, w_ = video.shape; n_ = t_ // self.chunk_frames
                vv = video.view(b_, c_, n_, self.chunk_frames, h_, w_).permute(0, 2, 1, 3, 4, 5).reshape(b_ * n_, c_, self.chunk_frames, h_, w_)
                z = self.vision.encode_observation(vv)
                z = z.reshape(b_, n_ * z.shape[1], *z.shape[2:])
            else:
                z = self.vision.encode_observation(video)
        b, n_tub = video.shape[0], frame_times.shape[-1] // 2
        if region_grid is None:
            region_grid = torch.tensor([[0, 24, 0, 24]] * b)
        zn = normalize_levels(z.float())
        self._zn = zn if getattr(self, "corr_head", None) is not None else None
        if getattr(self, "tmem", None) is not None:                # (10-17) rows without memory clips: the overview clip itself, pooled from
            self._live_ov = pool_regions(zn[:, :, -1:], n_tub, 4, torch.tensor([[0, 24, 0, 24]] * b))[0].reshape(b, -1, zn.shape[-1])   # this pass
        if getattr(self.bridge, "split", 0):
            # (10-17) band-split tokens (SpectralBridge.band_tokens): the patch grid itself goes to the bridge; a region crop is
            # resampled to the 24 x 24 patch grid first (identity for the whole frame), cells / geometry as pool_regions
            assert self.cells == 8 and not getattr(self, "vjepa_last", False) and not getattr(self, "time_groups", 0)
            g = zn.reshape(b, n_tub, 24, 24, -1)
            full = (region_grid == torch.tensor([0, 24, 0, 24])).all(1)
            if not bool(full.all()):
                g = g.clone()
                for i in (~full).nonzero().flatten().tolist():
                    y0, y1, x0, x1 = [int(v_) for v_ in region_grid[i]]
                    g[i] = F.adaptive_avg_pool2d(g[i, :, y0:y1, x0:x1].permute(0, 3, 1, 2), 24).permute(0, 2, 3, 1)
            pooled = self.bridge.band_tokens(g)
            _, coords, geom = pool_regions(zn[:, :, :1, :1], n_tub, self.cells, region_grid)
        else:
            pooled, coords, geom = pool_regions(zn, n_tub, self.cells, region_grid)
        if getattr(self, "flow_feats", False) or getattr(self, "flow_target", False):   # (11-02) V-JEPA's latent flow per cell
            fl_ = self.latent_flow(zn, n_tub, region_grid)
            if getattr(self, "flow_feats", False): self.bridge._flow = fl_          # as input (EvidenceBridge.flow_proj)
            if getattr(self, "flow_target", False): self._flow_t = fl_             # as the target of _FlowPred
        if getattr(self, "vjepa_last", False):        # (10-16) V-JEPA's final output only (level 4 of 4: block 39), statistics of that level
            pooled = pooled[:, :, -1:]
        tub_sec = frame_times.view(b, n_tub, 2).mean(-1)
        if getattr(self, "time_groups", 0):
            # (10-15) model.time_groups = G: time positions traded for space. A V-JEPA token has seen the whole clip, so the
            # time steps hold the clip's motion inside every token (a SigLIP frame token does not): the tubelets are averaged
            # in G groups and the freed positions go to a finer grid (data.cells 16 instead of 8: 2 x 256 = 512 tokens,
            # the same count as 8 x 64). Equalisation statistics of this layout: runs/whiten_c16g2.
            G, c2 = int(self.time_groups), self.cells * self.cells
            pooled = pooled.view(b, G, n_tub // G, c2, *pooled.shape[2:]).mean(2).reshape(b, G * c2, *pooled.shape[2:])
            geom = geom.view(b, G, n_tub // G, c2, -1)[:, :, 0].reshape(b, G * c2, -1)
            coords = coords.view(G, n_tub // G, c2, 3)[:, 0].clone(); coords[..., 0] = torch.arange(G).view(G, 1); coords = coords.reshape(G * c2, 3)
            tub_sec, n_tub = tub_sec.view(b, G, n_tub // G).mean(-1), G
        out = (pooled, coords.to(video.device), tub_sec.repeat_interleave(self.cells * self.cells, dim=1), geom)
        if getattr(self, "mem_cells", 0):              # hybrid memory: the full grid of the same encoding
            p_m, c_m, g_m = pool_regions(zn, n_tub, self.mem_cells, region_grid)
            self._mem_args = (p_m, c_m.to(video.device), tub_sec.repeat_interleave(self.mem_cells ** 2, dim=1), g_m)
        za = self.anchor_states(video, zn) if (self.surprise_segments or need_anchor) else None
        if self.surprise_segments:
            out = out + (pool_regions(self.surprise(video, za), n_tub, self.cells, region_grid)[0],)
        anchor_pooled = None
        if need_anchor:
            anchor_pooled = pooled.detach() if self.reference is None else pool_regions(za, n_tub, self.cells, region_grid)[0]
        return out, anchor_pooled

    causal_win, time_groups, chunk_frames, vjepa_last = 0, 0, 0, False   # (no class-level tmem: nn.Module keeps submodules in _modules)

    def mem_tokens(self):
        """hybrid memory: the projected full grid of the clip(s) last encoded by _evidence -> [B, mem_cells^2 * T, H] (or None)"""
        if not getattr(self, "mem_cells", 0) or self._mem_args is None:
            return None
        return self.mem_bridge(*self._mem_args[:2], self._mem_args[2], OBSERVED, self._mem_args[3])

    def causal_encode(self, video, chunk: int = 16):
        """(10-14) model.causal_window = W tubelets. V-JEPA encodes a clip all at once, so the tokens of moment t already hold
        moments after t: on vt_S1 the video positions predicted the next moment 80-89% right while giving 0.2% of their
        attention to the same place earlier (the answer was inside their own token). Here the tokens of moment t come from
        an encoding of the W tubelets ending at t (the start padded with the first frame): nothing after t is in them,
        so the next moment can only be predicted from the sequence. -> [B, n_tub * 576, L, D] like encode_observation."""
        b, c, t, hh, ww = video.shape; nt, w = t // 2, int(self.causal_win)
        vp = torch.cat([video[:, :, :1].expand(-1, -1, 2 * (w - 1), -1, -1), video], 2)
        wins = torch.stack([vp[:, :, 2 * i: 2 * i + 2 * w] for i in range(nt)], 1).flatten(0, 1)     # [B*nt, 3, 2W, H, W]
        z = torch.cat([self.vision.encode_observation(wins[i: i + chunk]) for i in range(0, len(wins), chunk)])
        n = z.shape[1] // w
        return z.view(b, nt, w, n, *z.shape[2:])[:, :, -1].reshape(b, nt * n, *z.shape[2:])

    def anchor_states(self, video, zn=None):
        """normalised states of the frozen world model (reuses zn when V-JEPA itself is frozen)"""
        if self.reference is None and zn is not None:
            return zn.detach()
        with torch.no_grad():
            return normalize_levels(self.anchor().encode_observation(video).float())

    def surprise(self, video: torch.Tensor, zn: torch.Tensor) -> torch.Tensor:
        """zn [B,N,L,D] normalised observed states -> residual [B,N,L,D] = observed - causal prediction (0 in segment 0)."""
        b, n_tub = video.shape[0], video.shape[2] // 2
        g = zn.shape[1] // n_tub
        k = min(self.surprise_segments + 1, n_tub)
        bounds = [round(j * n_tub / k) for j in range(k + 1)]                   # tubelet boundaries of the k segments
        res = torch.zeros_like(zn)
        for s, e in zip(bounds[1:-1], bounds[2:]):
            ctx = torch.arange(s * g, device=video.device).unsqueeze(0).expand(b, -1)
            tgt = torch.arange(s * g, e * g, device=video.device).unsqueeze(0).expand(b, -1)
            with torch.no_grad():
                wm = self.anchor()
                p, _ = wm.predict(wm.encode_masked_context(video, ctx), ctx, tgt)
            res[:, s * g:e * g] = zn[:, s * g:e * g] - normalize_levels(p.float())
        return res

    def info_loss(self, target, hid, ev):
        """target [B, K, L, D] frozen V-JEPA pooled states; hid from the answer pass; ev the evidence embeddings"""
        b, nt = target.shape[0], self.n_tub
        cells = target.shape[1] // nt
        tub = target.float().view(b, nt, cells, -1).mean(2)                      # [B, n_tub, L*D] per-step state
        if self.info == "ipa":
            ids, vm, tm = self.neutral                                            # [1, S] each, built from the tokenizer
            ids, vm, tm = ids.expand(b, -1), vm.expand(b, -1), tm.expand(b, -1)
            h = self.core(ids, torch.ones_like(ids), ev, vm, labels=None, read_mask=tm)
            q = h.view(b, -1, h.shape[-1]).mean(1)                                 # mean final state over the instruction
            return self.critic([q], [torch.arange(b, device=q.device)], tub.flatten(1))
        c = hid["last"].view(b, nt, cells, -1).mean(2)                             # [B, n_tub, H] state of each step
        qs, ps = [], []
        for k in range(1, self.horizons + 1):                                     # horizons past the clip end: no query
            kk = min(k, nt)
            qs.append(c[:, : nt - kk].reshape(-1, c.shape[-1]))
            ps.append((torch.arange(b, device=c.device)[:, None] * nt + torch.arange(kk, nt, device=c.device)[None]).reshape(-1))
        return self.critic(qs, ps, tub.reshape(b * nt, -1))

    def geometry(self, target, vis_mask, n_tub):
        """loss of GeometryHead on the hidden states captured by the last core call (call right after it)"""
        b = target.shape[0]
        p0 = vis_mask.long().cumsum(1).argmax(1) + 1                            # first token after the evidence
        rows = torch.arange(b, device=p0.device)
        read = {layer: self.core.captured[layer][rows, p0] for layer in self.geo.layers}
        teacher = target.float().view(b, n_tub, -1, target.shape[-2] * target.shape[-1]).mean(2).flatten(1)
        return self.geo(read, teacher)

    def ws_loss(self, x, c):
        """x [B, V] unit vectors (square root of the mean next-word distribution over the video positions), c [B, k] V-JEPA
        clip vectors -> (1 - CKA, CKA) between the two similarity structures over this batch + the bank of recent videos."""
        bank = self._ws_bank
        X = x if bank is None else torch.cat([x, bank[0].to(x.dtype)]); C = c if bank is None else torch.cat([c, bank[1]])
        n = X.shape[0]
        Cn = (C - C.mean(0)) / C.std(0).clamp_min(1e-6)
        kl, kv = X @ X.T, Cn @ Cn.T / Cn.shape[1]
        hm = torch.eye(n, device=X.device, dtype=kl.dtype) - 1.0 / n
        klc, kvc = hm @ kl @ hm, hm @ kv @ hm
        cka = (klc * kvc).sum() / (klc.norm() * kvc.norm()).clamp_min(1e-12)
        with torch.no_grad():
            size = int(self.wstruct.get("bank", 1024)); nb = (x.detach().half(), c.detach())
            self._ws_bank = nb if bank is None else (torch.cat([nb[0], bank[0]])[:size], torch.cat([nb[1], bank[1]])[:size])
        return 1 - cka, cka.detach()

    def code_targets(self, pooled, input_ids, labels, n_tub):
        """-> (input_ids, labels) with the first answer tokens of every row replaced by its V-JEPA code numbers, and the
        number of code tokens per row [B]. Rows whose placeholder is shorter than their code are left as they are."""
        c = self.codes
        if c["mu"].device != pooled.device:
            for k_ in ("mu", "sd", "books", "tok"): c[k_] = c[k_].to(pooled.device)
        e = self.bridge.equalise(pooled).float(); b, k, d = e.shape; nt = k // (self.cells * self.cells); g = self.cells // 2
        res = (e.view(b, nt, 2, g, 2, g, d).mean((3, 5)).reshape(b, nt * 4, d) - c["mu"]) / c["sd"]; idx = []
        for book in c["books"]:
            i = torch.cdist(res, book.unsqueeze(0).expand(b, -1, -1)).argmin(-1); idx.append(i); res = res - book[i]
        ids = c["tok"][torch.stack(idx, -1)].view(b, nt, -1)                              # [B, moments, 4 regions x levels]
        input_ids, labels = input_ids.clone(), labels.clone(); n_code = torch.zeros(b, dtype=torch.long, device=pooled.device)
        for r in range(b):
            seen = int(n_tub[r]) if n_tub is not None and 0 < int(n_tub[r]) < nt else nt
            # "next": only the moment right after the seen ones (writing several lets the LLM copy the codes it has just
            # written, moment to moment - 63% right after 100 steps without needing the video); "all": the seen moments
            # "last": the last moment it sees (reading without predicting)
            w = ids[r, seen:seen + int(c.get("moments", 1))].reshape(-1) if c["mode"] == "next" else \
                (ids[r, seen - 1:seen].reshape(-1) if c["mode"] == "last" else ids[r, :seen].reshape(-1))
            pos = (labels[r] != -100).nonzero().flatten()
            if len(w) == 0 or len(pos) < len(w): continue
            input_ids[r, pos[: len(w)]] = w; labels[r, pos[: len(w)]] = w; n_code[r] = len(w)
        return input_ids, labels, n_code

    def textread_loss(self, pooled, input_ids, attention_mask, ev_in, vis_mask, labels, n_tub):
        tr = self.textread; kk = int(tr.get("k", 256)); npos = int(tr.get("pos", 4))
        b = input_ids.shape[0]; e = self.bridge.equalise(pooled).float().detach(); nt = e.shape[1] // (self.cells * self.cells); g = self.cells // 2
        tgt = e.view(b, nt, 2, g, 2, g, -1).mean((3, 5))[..., :kk].reshape(b, nt * 4 * kk)          # [B, moments x regions x k]
        first = (labels != -100).float().argmax(1)                                                   # first answer token
        ar = torch.arange(labels.shape[1], device=labels.device)[None]
        rm = (ar < first[:, None]) & (ar >= first[:, None] - npos)
        h = self.core(input_ids, attention_mask, ev_in, vis_mask, labels=None, read_mask=rm).view(b, npos, -1)
        pred = self.tr_head(h.to(self.tr_head.weight.dtype)).float()                                 # [B, pos, ...]
        err = (pred - tgt[:, None]).pow(2).mean((0, 1))                                              # [moments x regions x k]
        with torch.no_grad():
            seen = int(n_tub[0]) if n_tub is not None else nt
            var = (tgt - tgt.mean(0)).pow(2).mean(0).clamp_min(1e-8); per = (err.detach() / var).view(nt, 4 * kk)
            r2 = torch.stack([1 - per[:seen].mean(), 1 - per[seen:].mean() if seen < nt else per.new_zeros(())])
        loss = err.mean(); z = loss.new_zeros(())
        return {"loss": loss, "loss_text": z.detach(), "loss_tr": loss.detach(), "tr_r2": r2.detach()}

    def vcode_words(self, pooled):
        """-> token ids [B, T*g*g] of the V-JEPA word at every video position (moment t, the code of its place at the level
        assigned to the cell) - the same words the positions of moment t - 1 predict. State codes only."""
        c = self.vcode
        if c["mu"].device != pooled.device:
            for k_ in ("mu", "sd", "books"): c[k_] = c[k_].to(pooled.device)
        with torch.no_grad():
            e = self.bridge.equalise(pooled).float(); b, k, d = e.shape; g = self.cells; nt = k // (g * g)
            r = (e.view(b, nt, g // 2, 2, g // 2, 2, d).mean((3, 5)).reshape(b, nt, -1, d) - c["mu"]) / c["sd"]; idx = []
            for book in c["books"]:
                i = torch.cdist(r.reshape(b, -1, d), book.unsqueeze(0).expand(b, -1, -1)).argmin(-1).view(b, nt, -1); idx.append(i); r = r - book[i]
            idx = torch.stack(idx, -1)
            yy, xx = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij"); place = ((yy // 2) * (g // 2) + xx // 2).reshape(-1).to(e.device)
            level = ((yy % 2) * 2 + xx % 2).reshape(-1).to(e.device) % c["L"]
            return (idx[:, :, place, level] + c["base"] + level * c["K"]).reshape(b, -1)

    def vcode_loss(self, pooled, h, n_tub):
        """pooled: V-JEPA cell states of the clip; h [B, K, H] final LLM states at the video positions (t, y, x order).
        -> (mean CE of the next-moment codes over the video positions of moments 0..T-2, accuracy at level 1, at all levels)"""
        c = self.vcode
        if c["mu"].device != h.device:
            for k_ in ("mu", "sd", "books"): c[k_] = c[k_].to(h.device)
        with torch.no_grad():
            e = self.bridge.equalise(pooled).float(); b, k, d = e.shape; nt = n_tub; g = self.cells
            pl = e.view(b, nt, g // 2, 2, g // 2, 2, d).mean((3, 5)).reshape(b, nt, -1, d)          # [B, T, 16 places, d]
            if c["mode"] == "delta":     # the CHANGE from moment t to t + 1 (copying the present cannot predict it)
                r = (torch.cat([pl[:, 1:] - pl[:, :-1], pl[:, -1:] * 0], 1) - c["mu"]) / c["sd"]
            else:
                r = (pl - c["mu"]) / c["sd"]
            idx = []
            for book in c["books"]:
                i = torch.cdist(r.reshape(b, -1, d), book.unsqueeze(0).expand(b, -1, -1)).argmin(-1).view(b, nt, -1); idx.append(i); r = r - book[i]
            idx = torch.stack(idx, -1)                                                    # [B, T, places, L]
            yy, xx = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij"); place = ((yy // 2) * (g // 2) + xx // 2).reshape(-1).to(h.device)
            level = ((yy % 2) * 2 + xx % 2).reshape(-1).to(h.device) % c["L"]
            nxt = idx[:, :-1] if c["mode"] == "delta" else idx[:, 1:]                     # delta: entry t = change t -> t+1
            tgt = nxt[:, :, place, level] + c["base"] + level * c["K"]                    # [B, T-1, g*g]
        hv = h.view(b, nt, g * g, -1)[:, :-1].reshape(-1, h.shape[-1])
        logits = self.core.llm.lm_head(hv.to(self.core.llm.lm_head.weight.dtype)).float()
        t = tgt.reshape(-1); loss = F.cross_entropy(logits, t)
        with torch.no_grad():
            ok = (logits.argmax(-1) == t).float().view(b, nt - 1, g * g); lv = level.view(1, 1, -1).expand_as(ok)
            acc = torch.stack([ok[lv == 0].mean(), ok.mean()])
        return loss, acc

    def vtie_loss(self, ev, h, n_tub):
        """ev [B, K, H] projected states (the LLM's input at the video positions), h [B, K, H] final states there (t, y, x
        order). -> (CE of the next moment's token over word table + every video token of the batch, [accuracy, rate of
        picking the position's own token (copying the present)])"""
        b, k, hd = h.shape; g2 = k // n_tub; head = self.core.llm.lm_head
        hv = h.view(b, n_tub, g2, hd)[:, :-1].reshape(-1, hd)                          # positions of moments 0..T-2
        words = head(hv.to(head.weight.dtype)).float()                                  # [N, V] text vocabulary (tied table)
        # targets = the projected CONTENT only (before the bridge adds the moment / place / time codes): with the codes in, the
        # next token is told apart by its labels alone (vt_S1 first try: 98.5% right by step 500, nothing predicted)
        cand = ev.detach().reshape(-1, hd).float()                                      # [B*K, H] every video token of the batch
        vid = hv.float() @ cand.T
        i = torch.arange(b * k, device=h.device).view(b, n_tub, g2)
        tgt, own = i[:, 1:].reshape(-1), i[:, :-1].reshape(-1)
        logits = torch.cat([words, vid], 1); loss = F.cross_entropy(logits, tgt + words.shape[1])
        with torch.no_grad():
            pick = logits.argmax(-1) - words.shape[1]
            acc = torch.stack([(pick == tgt).float().mean(), (pick == own).float().mean()])
        return loss, acc

    def iso_loss(self, pooled, n: int = 1024):
        """(10-15) The projector keeps V-JEPA's geometry. On m3b_A @6000 the two videos of an MVP pair have cosine 0.07-0.39 in
        equalised V-JEPA space but 0.49-0.67 after the projector (the nearest two words of Llama's table: 0.63): the
        projector, taught only by what the text asks, folds together what V-JEPA tells apart, and the LLM - trained on
        discrete, far-apart words - reads near inputs as one. Loss: squared difference between the cosine matrix of n random
        tokens of the batch after the projector (content, before the position / time codes) and in equalised space."""
        e = self.bridge.equalise(pooled); b, k, d = e.shape
        idx = torch.randperm(b * k, device=e.device)[:n]
        x = e.reshape(b * k, d)[idx]
        inp = self.bridge.inp[str(OBSERVED)]; wd = inp[1].weight.dtype if self.bridge.mode == "mlp" else inp.weight.dtype
        y = self.bridge.out(inp(x.to(wd))).float()
        cx = F.normalize(x.float(), dim=-1); cy = F.normalize(y, dim=-1)
        off = ~torch.eye(len(idx), dtype=torch.bool, device=e.device)
        return ((cy @ cy.T - (cx @ cx.T).detach())[off]).pow(2).mean()

    NEXT_BANDS = ((0, 16), (16, 256), (256, 1024), (1024, 2048))

    def next_loss(self, h, e, n_tub):
        """h [B, K, H] final states at the video positions (time-major), e [B, K, k] equalised V-JEPA states.
        -> (mean squared error of the next moment in equalised units,
            [error / variance per band of directions for the prediction (4), for copying the current moment (4)])"""
        b, k, _ = h.shape; c = k // n_tub
        pred = self.next_head(h[:, :-c].to(self.next_head.weight.dtype)).float()
        tgt = e[:, c:]; err = (pred - tgt).pow(2).mean((0, 1))
        with torch.no_grad():
            var = (tgt - tgt.mean((0, 1))).pow(2).mean((0, 1)).clamp_min(1e-8); cp = (e[:, :-c] - tgt).pow(2).mean((0, 1))
            r = torch.stack([x[a:z].sum() / var[a:z].sum() for x in (err.detach(), cp) for a, z in self.NEXT_BANDS])
        return err.mean(), r

    def corr_loss(self, zn, h, n_tub):
        """zn [B, n_tub*576, L, D] V-JEPA patch states (level-normalised), h [B, n_tub*64, H] final states at the video
        positions (time-major, cells row-major) -> (loss, accuracy on cells that moved, accuracy on cells that stayed).
        Target of the token (moment t, cell c): the cell of moment t-1 it came from."""
        b, dev = h.shape[0], zn.device
        with torch.no_grad():
            if self._corr_geo is None or self._corr_geo[0].device != dev:
                yy, xx = torch.meshgrid(torch.arange(24., device=dev), torch.arange(24., device=dev), indexing="ij")
                pos = torch.stack([yy.flatten(), xx.flatten()], 1); cid = (pos[:, 0] // 3 * 8 + pos[:, 1] // 3).long()
                ci = torch.stack([(cid == c).nonzero().flatten() for c in range(64)])
                self._corr_geo = (pos, ci, pos[ci].mean(1), self._corr_basis[0].to(dev), self._corr_basis[1].to(dev))
            pos, ci, ctr0, mean, basis = self._corr_geo
            p = F.normalize((zn.reshape(b, n_tub * 576, -1).float() - mean) @ basis, dim=-1).view(b, n_tub, 576, -1)
            s = p[:, 1:] @ p[:, :-1].transpose(-1, -2)                       # [B, T-1, patches at t, patches at t-1]
            back, fwd = s.argmax(-1), s.argmax(-2)                           # where a patch of t came from / where a patch of t-1 goes
            cyc = ((pos[fwd.gather(-1, back)] - pos).abs().amax(-1) <= 1).float()
            dc = (pos[back] - pos)[:, :, ci]                                 # [B, T-1, 64, 9, 2] moves of the 9 patches of a cell
            med = dc.median(3).values
            coh = ((dc - med.unsqueeze(3)).abs().amax(-1) <= 1).float().mean(-1)
            ctr = ctr0 + med
            src = (ctr[..., 0].clamp(0, 23) // 3 * 8 + ctr[..., 1].clamp(0, 23) // 3).long()
            ok = (coh >= 0.75) & (cyc[:, :, ci].mean(-1) >= 0.75) & ((ctr >= 0) & (ctr <= 23)).all(-1)
            mv = med.abs().amax(-1)
            moved = ok & (mv >= 3) & (src != torch.arange(64, device=dev)); stay = ok & (mv < 1.5)
        logits = self.corr_head(h.view(b, n_tub, 64, -1)[:, 1:].to(self.corr_head.weight.dtype)).float()
        ce = F.cross_entropy(logits.flatten(0, 2), src.flatten(), reduction="none").view_as(src)
        hit = (logits.argmax(-1) == src).float()
        zero = ce.sum() * 0
        return ((ce[moved].mean() if moved.any() else zero) + (ce[stay].mean() if stay.any() else zero),
                hit[moved].mean() if moved.any() else zero.detach(), hit[stay].mean() if stay.any() else zero.detach())

    def enable_tmem(self, cfg):
        """(10-17) temporal memory (models/temporal_memory.py): M extra LLM input tokens read from a Gated DeltaNet over K V-JEPA clips"""
        from .temporal_memory import TemporalMemory
        kw = {k: v for k, v in cfg.items() if k in ("stats", "d", "layers", "heads", "head_dim", "queries", "k_eq", "token_rms", "horizon")}
        self.tmem = TemporalMemory(out_dim=self.core.llm.config.hidden_size, **kw)
        self.tmem_res = int(cfg.get("res", 256)); os.environ["JEPA_TMEM_RES"] = str(self.tmem_res)
        # training: memory clips encoded per GPU per step (0 = no cap). Every rank waits for the slowest one at each FSDP step, so an
        # uneven clip count (0..24 per rank) costs the maximum, not the mean: rows beyond the budget use the overview pass instead
        self.tmem_budget = int(cfg.get("train_clip_budget", 0))

    def append_memory(self, ev, frame_times, mem=None, log=True):
        """ev [B, n, H] overview tokens -> [B, n + M, H]. mem: {"frames" uint8 [B, Kmax*16, 384, 384, 3] (live memory clips, encoded here at
        tmem_res = 384) or "feat" [B, Kmax, 128, 1408] (encoded), "k" [B], "sec" [B, Kmax]} or None. Rows with k = 0 (short videos, composed clips, the pair stream): ONE memory clip =
        the overview clip's own final level, pooled from the overview pass (identical to encoding it again at 384, without the second pass)."""
        if getattr(self, "tmem", None) is None:
            return ev
        b = ev.shape[0]; dev_ = ev.device
        if mem is None:
            k = torch.zeros(b, dtype=torch.long, device=dev_); km = 1; sec = torch.zeros(b, 1, device=dev_)
        else:
            k = mem["k"].to(dev_).clone(); km = max(int(k.max()), 1); sec = mem["sec"].to(dev_).float()[:, :km].clone()   # (k all 0: one clip)
        if mem is not None and "feat" in mem:              # already-encoded memory clips (evaluation: cached / encoded by the runner)
            feat = mem["feat"].to(dev_).float(); rows = []
        else:
            feat = torch.zeros(b, km, 128, 1408, device=dev_); rows = (k > 0).nonzero().squeeze(1).tolist()
            if self.training and getattr(self, "tmem_budget", 0) and rows:   # greedy fill of the per-rank clip budget, batch order
                left, keep = self.tmem_budget, []
                for r_ in rows:
                    if int(k[r_]) <= left: keep.append(r_); left -= int(k[r_])
                    else: k[r_] = 0
                rows = keep
        if rows:
            from jepa_llm.data.video import MEAN, STD
            fr = mem["frames"][rows].to(dev_, non_blocking=True); R_, _, H_, W_, _ = fr.shape
            x = fr.view(R_ * km, 16, H_, W_, 3).permute(0, 4, 1, 2, 3).float().div_(255)
            x = (x - MEAN.view(1, 3, 1, 1, 1).to(dev_)) / STD.view(1, 3, 1, 1, 1).to(dev_)
            valid = (torch.arange(km, device=dev_).view(1, km) < k[rows].view(-1, 1)).reshape(-1)
            f_ = torch.zeros(R_ * km, 128, 1408, device=dev_); xs = x[valid].to(torch.bfloat16)
            f_[valid] = torch.cat([self.live_memory(xs[i:i + 24]) for i in range(0, len(xs), 24)])
            feat[rows] = f_.view(R_, km, 128, 1408)
        lv = k == 0
        if lv.any():
            feat[lv, 0] = self._live_ov.to(dev_)[lv]; sec[lv, 0] = frame_times.float().mean(-1).to(dev_)[lv]; k[lv] = 1
        self._live_ov = None
        out = self.tmem(feat, k, sec)
        if not self.training:
            self.tmem_last = out.detach()                  # (10-17) diagnostics (logs/zz_gm_diag.py)
        if log:                                             # (10-17) logged by train_single (main batch only): memory / overview token rms, clips per row
          with torch.no_grad():
            self.tmem_stats = (float(out.float().pow(2).mean().sqrt()), float(ev.float().pow(2).mean().sqrt()), float(k.float().mean()))
        return torch.cat([ev, out.to(ev.dtype)], 1)

    @torch.no_grad()
    def live_memory(self, video):
        """[B, 3, 16, 384, 384] normalised clips -> [B, 128, 1408]: frozen V-JEPA at the memory resolution (tmem_res, default 256; the
        store was built the same way: logs/zz_mfeat_build.py), final level, 4 x 4 cells x 8 steps"""
        import torch.nn.functional as F_
        r = int(getattr(self, "tmem_res", 256)); B, C, T = video.shape[:3]
        x = video if r == video.shape[-1] else F_.interpolate(video.float().transpose(1, 2).reshape(B * T, C, *video.shape[-2:]), size=(r, r), mode="bicubic",
                                                              antialias=True).view(B, T, C, r, r).transpose(1, 2).to(video.dtype)
        st = normalize_levels(self.vision.encode_observation(x)[:, :, -1:].float()); g = r // 16
        return pool_regions(st, T // 2, 4, torch.tensor([[0, g, 0, g]] * B), grid=g)[0].reshape(B, 128, -1)

    def _rsc_hook(self, g):
        """(11-02) vjepa_rsc: EMA (0.99) of the answer-gradient covariance in V-JEPA's principal coordinates; the operator M is rebuilt
        every 50 calls once 100 have been seen (before that: no pull)"""
        with torch.no_grad():
            Bm = self.bridge._w["basis"].to(g.device).float(); gp = g.reshape(g.shape[0] * g.shape[1], -1).float() @ Bm
            A = gp.T @ gp / gp.shape[0]
            self._rsc_C = A if getattr(self, "_rsc_C", None) is None else 0.99 * self._rsc_C + 0.01 * A
            self._rsc_n = getattr(self, "_rsc_n", 0) + 1
            if self._rsc_n >= 100 and self._rsc_n % 50 == 0:
                ev, V = torch.linalg.eigh(self._rsc_C.double()); ev = ev.clamp_min(0)
                S = (V * (ev / ev.max().clamp_min(1e-30))) @ V.T
                self._rsc_M = (Bm.double() @ S @ Bm.double().T).float()
                self._rsc_pr = float(ev.sum() ** 2 / ev.pow(2).sum().clamp_min(1e-30))
                if getattr(self, "fw_online", False) and getattr(self, "dyn_head", None) is not None:
                    # (11-03) the dynamics loss in the answer's own metric: fw = C^(1/2) (mean eigenvalue of fw^2 = 1), C = the answer-gradient
                    # covariance at V-JEPA's input in its principal coordinates (same backward; the dynamics loss's share there 0.01 %)
                    e2 = ev / ev.mean().clamp_min(1e-30)
                    new_ = ((V * e2.sqrt()) @ V.T).float()
                    # (11-03 smoke) the metric moves slowly: it starts as the identity (the plain reconstruction) and takes 10 % of the new
                    # answer metric at every rebuild (every 50 steps -> ~500 steps), so the decoder's target never jumps (the direct switch
                    # put R^2 at -0.33 and kept it near 0 for 200 steps)
                    old_ = getattr(self.dyn_head, "fw", None)
                    if old_ is None: old_ = torch.eye(new_.shape[0], device=new_.device)
                    self.dyn_head.fw = 0.9 * old_.to(new_.device) + 0.1 * new_
        return g

    def forward(self, video, frame_times, input_ids, attention_mask, labels, vis_mask, lambda_state: float = 0.0,
                region_grid=None, lambda_pred: float = 0.0, text_weight: float = 1.0, lambda_swap: float = 0.0,
                swap_margin: float = 0.2, margin_unit: str = "token", lambda_geo: float = 0.0, lambda_info: float = 0.0,
                lambda_vc: float = 0.0, lambda_eq: float = 0.0, n_tub=None, lambda_corr: float = 0.0,
                lambda_ws: float = 0.0, lambda_next: float = 0.0, lambda_iso: float = 0.0, mem=None, lambda_dyn: float = 0.0,
                seg=None, seg_norm: float = 0.0, lambda_concept: float = 0.0, cg=None, ans_weight: str = "row", cf_rows=None, cf_logc=None, ce_dynw=None, dyn_scale: float = 1.0):
        """lambda_swap: in-batch evidence swap on every sample (general visual dependence): each answer must be at least
        `swap_margin` nats (per answer token) less likely when the row is given another row's evidence (roll by one)."""
        want_state = self.retention is not None and lambda_state > 0
        want_pred = self.predictor_heads is not None and lambda_pred > 0
        want_geo = self.geo is not None and lambda_geo > 0
        want_info = self.critic is not None and lambda_info > 0
        (pooled, coords, seconds, geom, *res), target = self._evidence(video, frame_times, region_grid,
                                                                      want_state or want_pred or want_geo or want_info)
        cf_on = isinstance(self.dyn_head, _CfUse) and lambda_dyn > 0 and self.training and cf_rows is not None
        if cf_on:                                        # (11-01 v3) _CfUse: the drawn rows enter with their moments shuffled
            nt_ = frame_times.shape[-1] // 2; K_ = pooled.shape[1]; c_ = K_ // nt_
            cf_rows = cf_rows.to(pooled.device).long()
            if n_tub is not None:                        # rows showing only their first moments are never substituted
                cf_rows = torch.where((n_tub.to(pooled.device) > 0) & (n_tub.to(pooled.device) < nt_), torch.zeros_like(cf_rows), cf_rows)
            sw_ = cf_rows > 0
            if bool(sw_.any()):
                perm = torch.rand(pooled.shape[0], nt_, device=pooled.device).argsort(1)
                perm = torch.where(sw_[:, None], perm, torch.arange(nt_, device=pooled.device)[None])
                tok_ = (perm.unsqueeze(-1) * c_ + torch.arange(c_, device=pooled.device)).flatten(1)      # [B, K] token order
                shuf_ = lambda t: t.gather(1, tok_.unsqueeze(-1).expand(-1, -1, t.shape[-1])) if torch.is_tensor(t) and t.dim() == 3 and t.shape[:2] == (pooled.shape[0], K_) else t
                pooled = shuf_(pooled); res = [shuf_(r_) for r_ in res]
        extra, value = None, None
        if getattr(self, "bind", None) and self.training:
            # key = coarse part of random cells of this clip (equalised coordinates), value = their fine part
            e = self.equaliser(pooled)                                                     # [B, K, k]
            idx = torch.rand(e.shape[:2], device=e.device).argsort(1)[:, : int(self.bind.get("n", 32))]
            cell = e.gather(1, idx.unsqueeze(-1).expand(-1, -1, e.shape[-1]))
            r = int(self.bind["keys"])
            key = cell.clone(); key[..., r:] = 0
            value = cell[..., r:]
            ev, extra = self.bridge(pooled, coords, seconds, OBSERVED, geom, keys=key)
        elif (getattr(self, "textread", None) is not None or (getattr(self, "codes", None) is not None and self.codes["mode"] == "next")) and n_tub is not None \
                and int(n_tub.min()) == int(n_tub.max()) and 0 < int(n_tub[0]) < frame_times.shape[-1] // 2:
            # next-scene codes: V-JEPA attends over the whole clip, so the tokens of the first moments would already hold
            # the later ones. The LLM is given tokens from an encoding of the SEEN frames only; the targets (code_targets)
            # come from the full clip's states.
            s_ = int(n_tub[0])
            (p2, c2, s2, g2, *_), _ = self._evidence(video[:, :, : 2 * s_], frame_times[:, : 2 * s_], region_grid, False)
            ev = self.bridge(p2, c2, s2, OBSERVED, g2); seen_only = True
        else:
            dmask = None                                 # (10-25) _DynMask: whole moments of some rows enter as the mask token (training only)
            if isinstance(self.dyn_head, _DynMask) and self.training and lambda_dyn > 0:
                nt_ = frame_times.shape[-1] // 2
                if n_tub is None or not bool(((n_tub > 0) & (n_tub < nt_)).any()):
                    dmask = self.dyn_head.draw(pooled.shape[0], nt_, pooled.shape[1] // nt_, pooled.device)
                else:                                    # rows showing only their first moments: nothing masked, the mask token still in the graph
                    dmask = torch.zeros(pooled.shape[:2], dtype=torch.bool, device=pooled.device)
            # (11-02) static_shrink (training only): the answer reads ~10 directions of V-JEPA's input - the appearance shortcut - while the
            # dynamics loss keeps V-JEPA's dynamics in the LLM unused. Per row, V-JEPA's appearance part (each cell's mean over the moments) is
            # pulled toward V-JEPA's dataset mean by u ~ U(0, 1); what changes between moments (x_t - cell mean) is left as it is. The answer
            # loss of a row then falls only as far as the answer reads V-JEPA's dynamics. The dynamics loss's target stays the full V-JEPA
            # output (pooled, unchanged). Inference: no change
            pin_ = pooled
            if self.training and getattr(self, "static_shrink", False) and pooled.dim() >= 3:
                nt_ = frame_times.shape[-1] // 2; B_, K_ = pooled.shape[:2]
                if K_ % nt_ == 0 and (n_tub is None or not bool(((n_tub > 0) & (n_tub < nt_)).any())):
                    P_ = pooled.float().reshape(B_, nt_, K_ // nt_, -1); m_ = P_.mean(1, keepdim=True)
                    mu_ = self.bridge._w["mean"].to(P_.device).float().view(1, 1, 1, -1)
                    u_ = torch.rand(B_, 1, 1, 1, device=P_.device)
                    pin_ = (P_ + u_ * (mu_ - m_)).reshape(pooled.shape).to(pooled.dtype)
                    self._shrink_u = u_.flatten().detach()
            # (11-02) vjepa_rsc (training only; representation self-challenging, Huang et al. 2020, in V-JEPA's space): the answer reads
            # only a few of V-JEPA's directions (effective 2.5 - 5 on the dynamics questions) while the dynamics loss keeps many more alive
            # in the LLM. Per row, V-JEPA's input is pulled down along the directions the answer currently depends on, in proportion to that
            # dependence: x <- x - u (x - mean) M, M = B S B^T, S = C / lambda_max(C), C = EMA of the answer-gradient covariance at this
            # input in V-JEPA's principal coordinates B (from the same backward, hook below), u ~ U(0, 1) per row. The answer loss of a row
            # then falls only as far as the answer reads other V-JEPA directions - the ones the dynamics loss preserved. The dynamics
            # loss's target stays the full V-JEPA output. Inference: no change
            if self.training and (getattr(self, "vjepa_rsc", False) or getattr(self, "fw_online", False)) and pin_.dim() >= 3:
                if getattr(self, "vjepa_rsc", False) and getattr(self, "_rsc_M", None) is not None:           # pooled [B, K, (levels,) D]: flattened per token as the bridge's equalise does
                    sh_ = pin_.shape; x2_ = pin_.float().reshape(sh_[0], sh_[1], -1)
                    u_ = torch.rand(sh_[0], 1, 1, device=pin_.device)
                    xc_ = x2_ - self.bridge._w["mean"].to(pin_.device).float()
                    pin_ = (x2_ - u_ * (xc_ @ self._rsc_M)).reshape(sh_).to(pooled.dtype)
                pin_ = pin_.detach().requires_grad_(True); pin_.register_hook(self._rsc_hook)
            ev = self.bridge(pin_, coords, seconds, OBSERVED, geom, residual=res[0] if res else None, content_mask=dmask)
            ev = self.append_memory(ev, frame_times, mem)
        reg_vc = self.bridge.last_reg                    # anti-collapse statistics of this call's visual tokens (or None)
        b, k, _ = ev.shape
        if self.training:                                # (10-22) analysis: sum of squares / count of this batch's visual tokens (reduced over ranks
            with torch.no_grad():                        #   in the training loop, next to the word-embedding scale)
                self._ev_ss, self._ev_n = ev.detach().double().pow(2).sum(), ev.numel()
        ev_in, nt = ev, frame_times.shape[-1] // 2
        if locals().get("seen_only"):
            assert self.core.memory or int(vis_mask.sum()) == b * k, f"evidence slots {int(vis_mask.sum())} != {b}x{k}"
        elif n_tub is not None and bool(((n_tub > 0) & (n_tub < nt)).any()):          # some rows use only their first time steps
            use = torch.where(n_tub > 0, n_tub, torch.full_like(n_tub, nt)).to(ev.device) * (k // nt)
            ev_in = ev[torch.arange(k, device=ev.device)[None] < use[:, None]]       # [N, H], row by row, time-major
            assert int(vis_mask.sum()) == ev_in.shape[0], f"evidence slots {int(vis_mask.sum())} != {ev_in.shape[0]}"
        else:
            assert self.core.memory or int(vis_mask.sum()) == b * k, f"evidence slots {int(vis_mask.sum())} != {b}x{k}"
        if getattr(self, "textread", None) is not None and locals().get("seen_only"):     # rows showing only their first moments
            return self.textread_loss(pooled, input_ids, attention_mask, ev_in, vis_mask, labels, n_tub)
        vwords = None
        if getattr(self, "vcode", None) is not None and self.vcode["input"] and ev_in.dim() == 3:
            # the V-JEPA words go IN as well (as words go in and out of an LLM): word embedding added to the projected state,
            # at the scale of the projected tokens (a learned gain, starting at 1) - done inside LanguageCore.forward
            vwords = self.vcode_words(pooled)
        n_code = None
        if getattr(self, "codes", None) is not None:
            input_ids, labels, n_code = self.code_targets(pooled, input_ids, labels, n_tub)
        self.core.mem_input = self.mem_tokens()
        if isinstance(self.dyn_head, _RecLang):
            self._rec_cap = None
            if self.training and lambda_dyn > 0: self.dyn_head.pick()
            else: self.dyn_head.want = 0
        self.core.want_vlog = lambda_concept > 0 and cg is not None
        try:
            logits, targets, ex_id, hid = self.core(input_ids, attention_mask, ev_in, vis_mask, labels, extra=extra, vwords=vwords)
        finally:
            self.core.want_vlog = False
        router_aux = self.core.router_aux if self.core.router_coef > 0 else None    # (11-16) MoE: the main pass's load-balancing loss
        loss_text, n_tok = answer_ce(logits, targets, ex_id, b)
        w_dyn, dynw_stat = None, loss_text.new_zeros(3)
        if ce_dynw is not None and self.training:
            # (11-01) the answer loss weighted by V-JEPA's dynamics: per clip r = mean ||x[t+1,c] - x[t,c]||^2 / (2 mean ||x[t,c] - clip mean||^2)
            # (equalised V-JEPA tokens; the share of the clip's variation that is change between moments, scale-free), divided by its mean
            # over the clips of the GLOBAL batch (one all-reduce on every rank), capped at ce_dynw["cap"]. Rows whose clips carry their
            # information in what moves weigh more in the answer loss; the mean weight is 1. No parameters, no extra pass, weights detached.
            with torch.no_grad():
                nt_ = frame_times.shape[-1] // 2; b_, K_ = pooled.shape[:2]
                if K_ % nt_ == 0:
                    x_ = self.bridge.equalise(pooled).float().view(b_, nt_, K_ // nt_, -1)
                    d_ = (x_[:, 1:] - x_[:, :-1]).pow(2).sum(-1).mean((1, 2)); v_ = (x_ - x_.mean((1, 2), keepdim=True)).pow(2).sum(-1).mean((1, 2))
                    r_ = d_ / (2 * v_).clamp_min(1e-8); ok_ = v_ > 1e-6
                else:
                    r_ = torch.zeros(b_, device=pooled.device); ok_ = torch.zeros(b_, dtype=torch.bool, device=pooled.device)
                tot_ = torch.stack([(r_ * ok_).sum(), ok_.sum().float()])
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(tot_)
                w_dyn = torch.where(ok_, r_ / (tot_[0] / tot_[1].clamp_min(1.0)).clamp_min(1e-8), torch.ones_like(r_))
                w_dyn = w_dyn.clamp(max=float(ce_dynw.get("cap", 1e9)))
                dynw_stat = torch.stack([w_dyn.mean(), w_dyn.max(), w_dyn.min()])
            if seg is None or seg_norm <= 0:
                tok_w = F.cross_entropy(logits.float(), targets, reduction="none")
                pe_ = torch.zeros(b, device=tok_w.device).index_add_(0, ex_id, tok_w); n_w = torch.zeros(b, device=tok_w.device).index_add_(0, ex_id, torch.ones_like(tok_w))
                has_w = n_w > 0
                loss_text = (w_dyn[has_w] * pe_[has_w] / n_w[has_w]).sum() / w_dyn[has_w].sum().clamp_min(1e-8)
        loss_cf, cf_acc, cf_w = loss_text.new_zeros(()), None, None
        if cf_on:
            c1_ = (cf_rows == 1)[ex_id]                                      # answer tokens of code-1 rows
            cf_w = (~c1_).float()
            sgc_ = seg[:, 1:][labels[:, 1:] != -100].to(ex_id.device).clamp_min(0) if seg is not None else torch.zeros_like(ex_id)
            if bool(c1_.any()):
                tk1 = F.cross_entropy(logits[c1_].float(), targets[c1_], reduction="none")
                u1, i1 = torch.unique(ex_id[c1_] * 4096 + sgc_[c1_], return_inverse=True)
                lp = -torch.zeros(len(u1), device=tk1.device).index_add_(0, i1, tk1)                    # log p(answer) per answer
                lc = cf_logc.to(lp.device).float()[torch.div(u1, 4096, rounding_mode="floor")]
                over = lp - lc
                loss_cf = F.relu(over).mean()
                cf_acc = torch.stack([over.mean().detach(), (over > 0).float().mean().detach(), torch.tensor(float(len(u1)), device=lp.device),
                                      (cf_rows == 2).sum().float()])
                if seg is None or seg_norm <= 0:
                    keep_ = (cf_rows != 1); loss_text, n_tok = answer_ce(logits[~c1_], targets[~c1_], ex_id[~c1_], b) if bool((~c1_).any()) else (loss_text * 0.0, n_tok)
            else:
                cf_acc = torch.stack([loss_text.new_zeros(()), loss_text.new_zeros(()), loss_text.new_zeros(()), (cf_rows == 2).sum().float()])
        if seg is not None and seg_norm > 0:
            # (10-23) packed rows (several original rows of one video in one sequence): every ORIGINAL row keeps the weight it had
            # unpacked - token mean within it, summed, divided by (rows x the mix's mean originals per row) so that the expected
            # weight of an original row is the same on every rank and equals the unpacked per-row mean
            sg = seg[:, 1:][labels[:, 1:] != -100].to(ex_id.device)
            unit = ex_id * 4096 + sg.clamp_min(0)
            uniq, inv = torch.unique(unit, return_inverse=True)
            tok = F.cross_entropy(logits, targets, reduction="none")
            if cf_w is not None: tok = tok * cf_w                # (11-01 v3) code-1 rows: no CE (their hinge is loss_cf)
            s_ = torch.zeros(len(uniq), device=tok.device).index_add_(0, inv, tok)
            n_ = torch.zeros(len(uniq), device=tok.device).index_add_(0, inv, torch.ones_like(tok))
            if w_dyn is not None:                                # (11-01) each original row weighted by its clip's dynamics weight
                s_ = s_ * w_dyn.to(s_.device)[torch.div(uniq, 4096, rounding_mode="floor")]
            loss_text = (s_ / n_).sum() / (b * float(seg_norm))
            if ans_weight == "sqrt":                           # (10-26) between row mean and token mean: every original row weighted by sqrt(its answer tokens)
                w_ = n_.sqrt(); loss_text = (w_ * s_ / n_).sum() / w_.sum()
        elif ans_weight == "sqrt":
            tok_ = F.cross_entropy(logits, targets, reduction="none")
            s_ = torch.zeros(b, device=tok_.device).index_add_(0, ex_id, tok_); n_ = torch.zeros(b, device=tok_.device).index_add_(0, ex_id, torch.ones_like(tok_))
            w_ = n_.sqrt(); loss_text = (w_ * s_ / n_.clamp_min(1)).sum() / w_.sum().clamp_min(1e-6)
        loss_vcd, vc_acc = loss_text.new_zeros(()), loss_text.new_zeros(2)      # (loss_vc is the anti-collapse term below)
        if getattr(self, "vcode", None) is not None and "last" in hid and ev_in.dim() == 3:
            loss_vcd, vc_acc = self.vcode_loss(pooled, hid["last"], frame_times.shape[-1] // 2)
        elif getattr(self, "vtie", None) is not None and "last" in hid and ev_in.dim() == 3:
            with torch.no_grad():
                br = self.bridge; inp = br.inp[str(OBSERVED)]; wd = inp[1].weight.dtype if br.mode == "mlp" else inp.weight.dtype
                cont = br.out(inp(br.equalise(pooled).to(wd)))
                cont = cont * br.gain.to(cont.dtype) if br.mode == "linear" else cont
            loss_vcd, vc_acc = self.vtie_loss(cont, hid["last"], frame_times.shape[-1] // 2)
        loss_iso = loss_text.new_zeros(())
        if lambda_iso > 0:
            loss_iso = self.iso_loss(pooled)
        dep_gain = loss_text.new_zeros(())
        if getattr(self, "depw", None) and self.training and b > 1 and ev_in.dim() == 3:
            with torch.no_grad():
                lg2, tg2, ex2, _ = self.core(input_ids, attention_mask, ev_in.roll(1, dims=0), vis_mask, labels)
                tok1 = F.cross_entropy(logits.float(), targets, reduction="none"); tok2 = F.cross_entropy(lg2.float(), tg2, reduction="none")
                gain = (tok2 - tok1).clamp_min(0); w = float(self.depw.get("floor", 0.2)) + gain / gain.mean().clamp_min(1e-6); w = w / w.mean()
                dep_gain = gain.mean()
            tok = F.cross_entropy(logits.float(), targets, reduction="none") * w
            per_ex = torch.zeros(b, device=tok.device).index_add_(0, ex_id, tok); nn_ = torch.zeros(b, device=tok.device).index_add_(0, ex_id, torch.ones_like(tok))
            loss_text = (per_ex[nn_ > 0] / nn_[nn_ > 0]).mean()
        code_acc = loss_text.new_zeros(5)
        if n_code is not None:
            with torch.no_grad():
                # share of the code numbers written right (chance 1/256): levels 1..4 of the FIRST region written - nothing
                # written before it to copy from, so this is what is read from the video - then all code tokens together
                # (the other regions of a moment often repeat the first one's coarse codes)
                hit = (logits.argmax(-1) == targets).float(); accs = []
                for r in range(b):
                    m = (ex_id == r).nonzero().flatten()[: int(n_code[r])]
                    if len(m) >= 4: accs.append(torch.cat([hit[m[:4]], hit[m].mean().view(1)]))
                if accs: code_acc = torch.stack(accs).mean(0)
        loss_geo, geo_acc = loss_text.new_zeros(()), loss_text.new_zeros(())
        if want_geo:
            loss_geo, geo_acc = self.geometry(target, vis_mask, frame_times.shape[-1] // 2)
        loss_info, info_acc = loss_text.new_zeros(()), loss_text.new_zeros(())
        if want_info:
            loss_info, info_acc = self.info_loss(target, hid, ev)
        loss_state = loss_text.new_zeros(())
        if want_state:
            loss_state = self.retention(hid, target, torch.ones(b, k, dtype=torch.bool, device=ev.device))
        loss_swap, swap_gap = loss_text.new_zeros(()), loss_text.new_zeros(())
        if lambda_swap > 0 and b > 1:
            nll, has = answer_nll(logits, targets, ex_id, b, margin_unit)
            lg2, tg2, ex2, _ = self.core(input_ids, attention_mask, ev.roll(1, dims=0), vis_mask, labels)
            nll_sw, _ = answer_nll(lg2, tg2, ex2, b, margin_unit)
            d = nll_sw - nll
            if has.any():
                loss_swap = F.relu(swap_margin - d[has]).mean(); swap_gap = d[has].mean().detach()
        loss_pred = loss_text.new_zeros(())
        if want_pred:
            n_tub = frame_times.shape[-1] // 2
            mask = latent_mask(b, n_tub, self.cells, ev.device, scale=self.mask_scale)
            ev_m = self.bridge(pooled, coords, seconds, OBSERVED, geom, residual=res[0] if res else None, content_mask=mask)
            _, _, _, hid_m = self.core(input_ids, attention_mask, ev_m, vis_mask, labels)
            loss_pred = self.predictor_heads(hid_m, target, mask)
        # (11-02) answer-confidence regularisers (gradient starvation, Pezeshki et al. 2021): the answer loss locks onto ~10 directions of
        # V-JEPA's input and stops learning from the rest even when the dynamics loss has kept them. ans_sd: spectral decoupling - penalty on
        # the answer tokens' centred logits (mean over the vocabulary of (z - mean z)^2, mean over answer tokens); ans_ls: label smoothing
        # (eps * (uniform NLL - CE), mean over answer tokens). Both added to the loss; loss_text itself is unchanged (logged as before)
        loss_areg = loss_text.new_zeros(())
        if self.training and (getattr(self, "ans_sd", 0.0) > 0 or getattr(self, "ans_ls", 0.0) > 0) and logits.numel() > 0:
            zf_ = logits.float()
            if getattr(self, "ans_sd", 0.0) > 0:
                zc_ = zf_ - zf_.mean(-1, keepdim=True); loss_areg = loss_areg + float(self.ans_sd) * zc_.pow(2).mean(-1).mean()
            if getattr(self, "ans_ls", 0.0) > 0:
                lp_ = zf_.log_softmax(-1); ce_ = -lp_.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                loss_areg = loss_areg + float(self.ans_ls) * ((-lp_.mean(-1)) - ce_).mean()
        loss = loss_areg.to(loss_text.dtype) + text_weight * (loss_text + lambda_swap * loss_swap) + lambda_state * loss_state + lambda_pred * loss_pred \
            + lambda_geo * loss_geo + lambda_info * loss_info \
            + (self.vcode["weight"] * loss_vcd if getattr(self, "vcode", None) is not None else 0.0) \
            + (float(self.vtie.get("weight", 1.0)) * loss_vcd if getattr(self, "vtie", None) is not None else 0.0) \
            + lambda_iso * loss_iso
        loss_router = loss_text.new_zeros(())
        if router_aux is not None:
            loss_router = router_aux.detach(); loss = loss + self.core.router_coef * router_aux.to(loss.dtype)
        loss_vc = loss_text.new_zeros(())
        if lambda_vc > 0 and reg_vc is not None:
            loss_vc = reg_vc.to(loss.dtype); loss = loss + lambda_vc * loss_vc
        loss_eq, eq_r2 = loss_text.new_zeros(()), loss_text.new_zeros(3)
        if self.eq_heads is not None:
            # lambda_eq = 0: the heads only watch (trained on detached states); > 0: the loss also shapes connector + LLM
            l_eq, eq_r2 = self.eq_heads(hid, self.equaliser(pooled).detach(), detach=lambda_eq <= 0)
            loss_eq = l_eq.detach(); loss = loss + (lambda_eq if lambda_eq > 0 else 1.0) * l_eq.to(loss.dtype)
            if self.stream_weight > 0 and hid["text_ex"].numel() > 0:
                nt = frame_times.shape[-1] // 2
                per_step = self.equaliser(pooled).detach().view(b, nt, k // nt, -1).mean(2)            # [B, n_tub, k]
                if self.eq_heads.set_query is not None:                # the positions carry the clip together
                    l_st, st_r2 = self.eq_heads.stream_whole(hid["text_set"][0], hid["text_set"][1], per_step, detach=lambda_eq <= 0)
                else:
                    l_st, st_r2 = self.eq_heads.stream(hid["text"], hid["text_ex"], per_step, detach=lambda_eq <= 0)
                loss = loss + (self.stream_weight if lambda_eq > 0 else 1.0) * l_st.to(loss.dtype)
                eq_r2 = torch.cat([eq_r2, st_r2])              # explained fraction: evidence positions (3 bands), text positions (3)
        if "read" in hid:
            l_rd, rd = self.eq_heads.read(hid["read"], hid["read_ok"], self.equaliser(pooled).detach())
            loss = loss + float(self.read.get("weight", 1.0)) * l_rd.to(loss.dtype)
            eq_r2 = torch.cat([eq_r2, rd])        # ... then the read: explained fraction (3 bands), attention mass on the video, cells looked at
        if extra is not None:
            l_bd, bd_r2 = self.eq_heads.bind(hid["keys"], value)
            loss = loss + float(self.bind.get("weight", 1.0)) * l_bd.to(loss.dtype)
            eq_r2 = torch.cat([eq_r2, bd_r2])                  # ... then binding: explained fraction of the value (2 bands)
        loss_corr, corr_acc, corr_stay = loss_text.new_zeros(()), loss_text.new_zeros(()), loss_text.new_zeros(())
        if self.corr_head is not None and lambda_corr > 0 and "last" in hid and self._zn is not None:
            loss_corr, corr_acc, corr_stay = self.corr_loss(self._zn, hid["last"], frame_times.shape[-1] // 2)
            loss = loss + lambda_corr * loss_corr.to(loss.dtype)
        self._zn = None
        loss_ws, ws_cka = loss_text.new_zeros(()), loss_text.new_zeros(())
        if self.wstruct and lambda_ws > 0 and "ws" in hid:
            loss_ws, ws_cka = self.ws_loss(hid["ws"], self.bridge.equalise(pooled).float().mean(1).detach())
            loss = loss + lambda_ws * loss_ws.to(loss.dtype)
        loss_next, next_err = loss_text.new_zeros(()), loss_text.new_zeros(8)
        if getattr(self, "next_head", None) is not None and lambda_next > 0 and "last" in hid:
            loss_next, next_err = self.next_loss(hid["last"], self.bridge.equalise(pooled).float().detach(), frame_times.shape[-1] // 2)
            loss = loss + lambda_next * loss_next.to(loss.dtype)
        loss_dyn, dyn_acc = loss_text.new_zeros(()), loss_text.new_zeros(2)
        if isinstance(self.dyn_head, _RecLang) and getattr(self.dyn_head, "objective", "") in ("nce_global", "nce_euclid", "mse_ncf") and self.training and lambda_dyn > 0:
            # (11-02) the global batch's V-JEPA tokens (unit vectors, no gradient) for the candidate set of the single relational loss; every
            # rank takes part in the all-gather on every training forward (sizes gathered first: rows per rank may differ)
            with torch.no_grad():
                xg_ = self.bridge.equalise(pooled).detach()
                if (self.dyncon or {}).get("target") == "orig":
                    xg_ = xg_ / self.bridge._w["scale"].to(xg_.device)
                xg_ = (xg_.float() if self.dyn_head.objective == "nce_euclid" else F.normalize(xg_.float(), dim=-1)).to(torch.bfloat16)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    W_, r_ = torch.distributed.get_world_size(), torch.distributed.get_rank()
                    sz_ = torch.tensor([xg_.shape[0]], device=xg_.device); szs_ = [torch.zeros_like(sz_) for _ in range(W_)]
                    torch.distributed.all_gather(szs_, sz_); szs_ = [int(z) for z in szs_]; mb_ = max(szs_)
                    xp_ = torch.zeros((mb_,) + tuple(xg_.shape[1:]), dtype=xg_.dtype, device=xg_.device); xp_[: xg_.shape[0]] = xg_
                    out_ = [torch.zeros_like(xp_) for _ in range(W_)]; torch.distributed.all_gather(out_, xp_)
                    self.dyn_head._tall = torch.cat([o_[:n_] for o_, n_ in zip(out_, szs_)]); self.dyn_head._own = sum(szs_[:r_])
                else:
                    self.dyn_head._tall, self.dyn_head._own = xg_, 0
        if cf_acc is not None:                           # (11-01 v3) _CfUse: computed with the answer CE above
            loss_dyn, dyn_acc = loss_cf, cf_acc; loss = loss + lambda_dyn * loss_cf.to(loss.dtype)
        if isinstance(self.dyn_head, _CfUse):
            pass                                         # (11-01 v3) done above
        elif isinstance(self.dyn_head, _DynMask):
            dm_ = locals().get("dmask")
            if dm_ is not None and self.dyn_head.layer in hid:
                if bool(dm_.any()):
                    loss_dyn, dyn_acc = self.dyn_head(self.bridge.equalise(pooled).detach(), hid[self.dyn_head.layer], dm_, frame_times.shape[-1] // 2)
                    loss = loss + lambda_dyn * loss_dyn.to(loss.dtype)
                else:                                    # nothing could be masked: the head still runs (its own forward) on one cell, weight 0 (FSDP)
                    one = torch.zeros_like(dm_); one[0, -1] = True
                    z_, _ = self.dyn_head(self.bridge.equalise(pooled).detach(), hid[self.dyn_head.layer], one, frame_times.shape[-1] // 2)
                    loss = loss + 0.0 * z_.to(loss.dtype)
        elif isinstance(self.dyn_head, _StructKeep) and lambda_dyn > 0 and ev_in.dim() == 3 and "last" in hid:
            st_ = [hid[f"sk{a}"] for a in self.struct_layers if f"sk{a}" in hid] + [hid["last"]]
            loss_dyn, dyn_acc = self.dyn_head(st_, self.bridge.equalise(pooled).detach())
            loss = loss + lambda_dyn * loss_dyn.to(loss.dtype)
        elif isinstance(self.dyn_head, _ReadStruct) and lambda_dyn > 0 and self.training:
            # every rank takes part (all-gather inside), with or without video rows
            if ev_in.dim() == 3 and "last_full" in hid:
                vm_ = vis_mask.bool(); pos_ = torch.arange(vm_.shape[1], device=vm_.device)[None]
                p_ = (torch.where(vm_, pos_, torch.full_like(pos_, -1)).max(1).values + 1).clamp_max(vm_.shape[1] - 1)
                r_ = torch.arange(vm_.shape[0], device=vm_.device)
                st_ = [self.core.captured[f"rs{a}"][r_, p_] for a in self.read_layers] + [hid["last_full"][r_, p_]]
                loss_dyn, dyn_acc = self.dyn_head(st_, self.bridge.equalise(pooled).detach(), len(st_))
            else:
                loss_dyn, dyn_acc = self.dyn_head(None, None, len(self.read_layers) + 1)
            loss = loss + lambda_dyn * loss_dyn.to(loss.dtype)
        elif isinstance(self.dyn_head, _VFloor) and lambda_dyn > 0 and self.training and ev_in.dim() == 3 and f"vf{self.vf_layers[0]}" in hid:
            st_ = [hid[f"vf{a}"] for a in self.vf_layers]
            loss_dyn, dyn_acc = self.dyn_head(st_, ev_in.detach(), self.bridge.equalise(pooled).detach(), frame_times.shape[-1] // 2)
            loss = loss + lambda_dyn * loss_dyn.to(loss.dtype)
        elif isinstance(self.dyn_head, _RouteKeep) and lambda_dyn > 0 and self.training and ev_in.dim() == 3 and "rq0" in hid:
            nl_ = self.dyn_head.n_layers; b_, k_ = ev_in.shape[:2]
            qs_ = [hid[f"rq{i}"].view(b_, k_, -1, self.core.llm.config.head_dim) for i in range(nl_)]
            ks_ = [hid[f"rk{i}"].view(b_, k_, -1, self.core.llm.config.head_dim) for i in range(nl_)]
            loss_dyn, dyn_acc = self.dyn_head(qs_, ks_, self.bridge.equalise(pooled).detach())
            loss = loss + lambda_dyn * loss_dyn.to(loss.dtype)
        elif isinstance(self.dyn_head, _DynAns) and lambda_dyn > 0 and self.training:
            if ev_in.dim() == 3 and "ans" in hid and hid["ans"].shape[0] == b:
                has_ = torch.bincount(ex_id, minlength=b)[:b] > 0
                loss_dyn, dyn_acc = self.dyn_head(self.bridge.equalise(pooled).detach(), hid["ans"], has_)
            else:
                loss_dyn, dyn_acc = self.dyn_head(None, None, None)
            loss = loss + lambda_dyn * loss_dyn.to(loss.dtype)
        elif isinstance(self.dyn_head, _DynNext) and lambda_dyn > 0 and ev_in.dim() == 3 and "last" in hid:
            loss_dyn, dyn_acc = self.dyn_head(hid["last"], self.bridge.equalise(pooled).detach())
            loss = loss + lambda_dyn * loss_dyn.to(loss.dtype)
        elif isinstance(self.dyn_head, _RecLang) and lambda_dyn > 0 and ev_in.dim() == 3 and (self._rec_cap is not None or "last_full" in hid):
            vm_ = vis_mask.bool(); pos_ = torch.arange(vm_.shape[1], device=vm_.device)[None]
            last_ = torch.where(vm_, pos_, torch.full_like(pos_, -1)).max(1).values             # last evidence position per row
            tmask = attention_mask.bool() & (pos_ > last_[:, None])                              # text after the video (question + answer)
            if (self.dyncon or {}).get("read_at") == "answer" and labels is not None:
                # (11-01) read where the answer is decided: the positions that predict an answer token (the last prompt token and the answer
                # tokens but the last), so the preserved clip is wired into the answer computation; rows without answer keep the text mask
                am_ = torch.zeros_like(tmask); am_[:, :-1] = labels[:, 1:] != -100; am_ = am_ & tmask
                tmask = torch.where(am_.any(1, keepdim=True), am_, tmask)
            h_rec = hid["last_full"] if self.dyn_head.want == len(self.core.llm.model.layers) + 1 else self._rec_cap   # final: no hook (compiled LLM)
            x_rec = self.bridge.equalise(pooled).detach()
            if (self.dyncon or {}).get("target") == "orig":   # (10-25) V-JEPA's own geometry (equalised / scale: top-2048 principal axes, natural variances).
                x_rec = x_rec / self.bridge._w["scale"].to(x_rec.device)   # All directions at equal weight learned ~10x slower (frozen nx5 probe:
            if (self.dyncon or {}).get("rec_target") == "delta":
                # (11-01) the reconstruction target = V-JEPA's DYNAMICS: every token's change from the previous moment (first moment 0), each
                # direction divided by its own std of change in the batch (the small directions' motion counts as much as the large ones')
                nt_ = frame_times.shape[-1] // 2; b_, K_, d_ = x_rec.shape; c_ = K_ // nt_
                xv_ = x_rec.view(b_, nt_, c_, d_); dd_ = torch.zeros_like(xv_); dd_[:, 1:] = xv_[:, 1:] - xv_[:, :-1]
                x_rec = (dd_ / dd_[:, 1:].flatten(0, 2).std(0).clamp_min(1e-6)).view(b_, K_, d_)
            if self.vpres_w > 0:                          # (10-27) + preservation at the video positions (equalised target, all directions equal)
                vp_h = [hid["last"] if a == "final" else hid[f"vp{a}"] for a in self.vpres_at]
                loss_dyn, dyn_acc, loss_vp, vp_acc = self.dyn_head(x_rec, h_rec, tmask, self.dyn_head.want, vp=(self.bridge.equalise(pooled).detach(), vp_h))
                loss = loss + self.vpres_w * loss_vp.to(loss.dtype); dyn_acc = torch.cat([dyn_acc, vp_acc])
            else:
                if (self.dyncon or {}).get("cew") and self.training:      # (11-02) per-row answer loss (detached) -> dynamics-loss row weights, mean 1
                    with torch.no_grad():
                        tk_ = F.cross_entropy(logits.float(), targets, reduction="none")
                        rs_ = torch.zeros(b, device=tk_.device).index_add_(0, ex_id, tk_); rn_ = torch.zeros(b, device=tk_.device).index_add_(0, ex_id, torch.ones_like(tk_))
                        rc_ = torch.where(rn_ > 0, rs_ / rn_.clamp_min(1), torch.zeros_like(rs_))
                        self.dyn_head.roww = rc_ / rc_[rn_ > 0].mean().clamp_min(1e-8)
                loss_dyn, dyn_acc = self.dyn_head(x_rec, h_rec, tmask, self.dyn_head.want)   # clip R^2 0.03 vs 0.21 at 200 steps; logs/zz_rec2_probe4/5.py)
            # (11-02) dyn_scale: +1 / 0 / -1 per step from the training loop's alignment gate (loss.dyn_gate, train_single.py)
            loss = loss + lambda_dyn * dyn_scale * loss_dyn.to(loss.dtype); self._rec_cap = None
        elif isinstance(self.dyn_head, _FlowPred) and lambda_dyn > 0 and "last" in hid and getattr(self, "_flow_t", None) is not None \
                and self._flow_t.shape[:2] == hid["last"].shape[:2]:
            loss_dyn, dyn_acc = self.dyn_head(self._flow_t, hid["last"]); self._flow_t = None
            loss = loss + lambda_dyn * loss_dyn.to(loss.dtype)
        elif isinstance(self.dyn_head, _RecSelf) and lambda_dyn > 0 and ev_in.dim() == 3 and "last" in hid:
            x_rec = self.bridge.equalise(pooled).detach()
            if (self.dyncon or {}).get("target") == "orig":
                x_rec = x_rec / self.bridge._w["scale"].to(x_rec.device)
            loss_dyn, dyn_acc = self.dyn_head(x_rec, hid["last"])
            loss = loss + lambda_dyn * loss_dyn.to(loss.dtype)
        elif isinstance(self.dyn_head, _DynCorr) and lambda_dyn > 0 and ev_in.dim() == 3 and "last" in hid:
            loss_dyn, dyn_acc = self.dyn_head(self.bridge.equalise(pooled).detach(), hid["last"], frame_times.shape[-1] // 2)
            loss = loss + lambda_dyn * loss_dyn.to(loss.dtype)
        elif isinstance(self.dyn_head, _DynTime) and lambda_dyn > 0 and ev_in.dim() == 3 and self.dyn_head.layer in hid:
            L_ = self.dyn_head.layer; full = self.core.captured[L_]                        # [B, S, H] layer output of this call
            sel_ = labels[:, 1:] != -100; sv_ = full[:, :-1][sel_].float()                 # the positions that predict answer tokens (core's ex_id order)
            ans_m = torch.zeros(b, sv_.shape[-1], device=sv_.device).index_add_(0, ex_id, sv_) \
                / torch.zeros(b, device=sv_.device).index_add_(0, ex_id, torch.ones(len(ex_id), device=sv_.device)).clamp_min(1).unsqueeze(1)
            loss_dyn, dyn_acc = self.dyn_head(self.bridge.equalise(pooled).detach(), hid[L_], ans_m, frame_times.shape[-1] // 2)
            loss = loss + lambda_dyn * loss_dyn.to(loss.dtype)
        elif self.dyn_head is not None and lambda_dyn > 0 and "ans" in hid and ev_in.dim() == 3:
            loss_dyn, dyn_acc = self.dyn_head(self.bridge.equalise(pooled).detach(), hid["ans"], frame_times.shape[-1] // 2)
            loss = loss + lambda_dyn * loss_dyn.to(loss.dtype)
        loss_cc = torch.zeros((), device=loss.device); cc_acc = torch.zeros((), device=loss.device)
        if lambda_concept > 0 and cg is not None and "vlog" in hid:
            # (10-25) concept readout loss: every multiple-choice turn of the row (human label / measurement / known transformation) - its
            # options' content words scored by the clip's readout (mean log-probability of each option's words), CE towards the gold option
            lp = hid["vlog"].log_softmax(-1); tk = cg["tok"]                                   # [B, V], [G, O, W] (-1 = pad)
            v_ = lp[cg["row"]].gather(1, tk.clamp_min(0).flatten(1)).view(tk.shape); m_ = (tk >= 0).float()
            sc_ = ((v_ * m_).sum(-1) / m_.sum(-1).clamp_min(1)).masked_fill(m_.sum(-1) == 0, float("-inf"))
            loss_cc = F.cross_entropy(sc_, cg["gold"]); cc_acc = (sc_.argmax(1) == cg["gold"]).float().mean()
            loss = loss + lambda_concept * loss_cc.to(loss.dtype)
        per_ex, _has = answer_nll(logits, targets, ex_id, b)        # (10-20) per-example answer loss, for the data scheduler (no extra pass)
        return {"dynw_mean": dynw_stat[0], "dynw_max": dynw_stat[1], "dynw_min": dynw_stat[2], "per_ex": per_ex.detach(), "loss_concept": loss_cc.detach(), "concept_acc": cc_acc.detach(), "loss_dyn": loss_dyn.detach(), "dyn_acc": dyn_acc.detach(), "loss_iso": loss_iso.detach(), "loss_vcode": loss_vcd.detach(), "vcode_acc": vc_acc.detach(), "dep_gain": dep_gain.detach(), "code_acc": code_acc.detach(), "loss_next": loss_next.detach(), "next_err": next_err.detach(), "loss_ws": loss_ws.detach(), "ws_cka": ws_cka, "loss": loss, "loss_text": loss_text.detach(), "loss_areg": loss_areg.detach(), "loss_state": loss_state.detach(), "loss_pred": loss_pred.detach(),
                "loss_swap": loss_swap.detach(), "swap_gap": swap_gap, "answer_tokens": n_tok.detach(), "loss_vc": loss_vc.detach(), "loss_router": loss_router,
                "loss_corr": loss_corr.detach(), "corr_acc": corr_acc, "corr_stay": corr_stay,
                "loss_eq": loss_eq, "eq_r2": eq_r2.detach(),
                "loss_geo": loss_geo.detach(), "geo_acc": geo_acc, "loss_info": loss_info.detach(), "info_acc": info_acc}

    def forward_ic(self, video, frame_times, input_ids, attention_mask, labels, vis_mask, region_grid):
        """(10-15) in-context V-JEPA neighbours: video [B, n, 3, T, H, W] (n clips per sequence, in sequence order), vis_mask
        holding n * K slots per row. -> {"loss": answer CE over every answer, "loss_tgt": CE of the LAST answer only (the
        target, read with the neighbour and its answer in context)}"""
        b, n = video.shape[:2]
        (pooled, coords, seconds, geom, *res), _ = self._evidence(video.flatten(0, 1), frame_times.flatten(0, 1),
                                                                  region_grid.flatten(0, 1), False)
        ev = self.bridge(pooled, coords, seconds, OBSERVED, geom, residual=res[0] if res else None)
        ev = ev.view(b, n * ev.shape[1], -1)
        logits, targets, ex_id, _ = self.core(input_ids, attention_mask, ev, vis_mask, labels)
        loss, _ = answer_ce(logits, targets, ex_id, b)
        with torch.no_grad():                         # labelled tokens of the last assistant turn = after the last video
            last_vis = torch.where(vis_mask.bool(), torch.arange(vis_mask.shape[1], device=vis_mask.device), -1).max(1).values
            pos = torch.arange(labels.shape[1], device=labels.device)[None].expand_as(labels)[:, 1:][labels[:, 1:] != -100]
            tgt = pos > last_vis[ex_id]
            tok = F.cross_entropy(logits.float(), targets, reduction="none")
            loss_tgt = tok[tgt].mean() if tgt.any() else tok.new_zeros(())
        return {"loss": loss, "loss_tgt": loss_tgt}

    def forward_images(self, image, input_ids, attention_mask, labels, vis_mask, pool: int = 3):
        """(10-21) still images (OCR / scene text). A still has no time, so its token budget goes to space: one tubelet (two copies of
        the image) at high resolution, e.g. 1152 px -> 72 x 72 V-JEPA patches -> 3 x 3 pooling (as for video) -> 24 x 24 = 576 cells
        (video: 8 moments x 8 x 8 = 512). Same frozen encoder, level normalisation, equalisation statistics and bridge (V-JEPA's
        1152-px states pooled 3 x 3 agree with its 384-px states at cosine 0.89 and have the same equalised energy, logs/zz_vjepa_res.py).
        Positions: the bridge's 8 x 8 cell embeddings at row // 3, column // 3, plus the exact cell geometry (cy, cx, h, w).
        image [B, 3, 2, S, S] -> {"loss", "loss_text"} (answer CE; no dynamics term)."""
        ev = self.image_tokens(image, pool)
        b, k = ev.shape[:2]
        assert int(vis_mask.sum()) == b * k, f"image slots {int(vis_mask.sum())} != {b}x{k}"
        logits, targets, ex_id, _ = self.core(input_ids, attention_mask, ev, vis_mask, labels)
        loss, _ = answer_ce(logits, targets, ex_id, b)
        return {"loss": loss, "loss_text": loss.detach()}

    def image_tokens(self, image, pool: int = 3):
        """the bridge tokens of still images (see forward_images): image [B, 3, 2, S, S] -> [B, (S/16/pool)^2, H]"""
        with torch.set_grad_enabled(self.train_vision != "none" and torch.is_grad_enabled()):
            z = self.vision.encode_observation(image)
        zn = normalize_levels(z.float())
        b, n, L, D = zn.shape
        g = int(round(n ** 0.5)); c = g // pool
        pooled = F.avg_pool2d(zn.view(b, g, g, L * D).permute(0, 3, 1, 2), pool).permute(0, 2, 3, 1).reshape(b, c * c, L, D)
        yy, xx = torch.meshgrid(torch.arange(c), torch.arange(c), indexing="ij")
        coords = torch.stack([torch.zeros_like(yy), yy * self.cells // c, xx * self.cells // c], -1).reshape(-1, 3).to(image.device)
        geom = torch.stack([(yy + 0.5) / c, (xx + 0.5) / c, torch.full_like(yy, 1, dtype=torch.float) / c,
                            torch.full_like(yy, 1, dtype=torch.float) / c], -1).reshape(1, -1, 4).float().expand(b, -1, -1).to(image.device)
        seconds = torch.zeros(b, c * c, device=image.device)
        return self.bridge(pooled, coords, seconds, OBSERVED, geom)

    def forward_pairs(self, video, frame_times, input_ids, attention_mask, labels, vis_mask, partner,
                      lambda_state: float = 0.0, lambda_use: float = 0.0, margin: float = 0.5, region_grid=None,
                      text_weight: float = 1.0, margin_unit: str = "token", lambda_geo: float = 0.0,
                      lambda_info: float = 0.0, lambda_vc: float = 0.0):
        """Paired batch (spec 6.5, use failure). Answer CE on every row (+ retention) as in forward(); with lambda_use > 0
        also the evidence-swap margin: each row's own answer must be at least `margin` nats less likely when the row is
        given its partner's evidence (same question, different video, different answer). The LLM runs a second time
        on the swapped evidence; V-JEPA runs once. partner [B]: position of the partner, -1 = none."""
        want_state = self.retention is not None and lambda_state > 0
        want_geo = self.geo is not None and lambda_geo > 0
        want_info = self.critic is not None and lambda_info > 0
        (pooled, coords, seconds, geom, *res), target = self._evidence(video, frame_times, region_grid, want_state or want_geo or want_info)
        ev = self.bridge(pooled, coords, seconds, OBSERVED, geom, residual=res[0] if res else None)
        ev = self.append_memory(ev, frame_times, None, log=False)
        reg_vc = self.bridge.last_reg
        b, k, _ = ev.shape
        self.core.mem_input = self.mem_tokens()
        logits, targets, ex_id, hid = self.core(input_ids, attention_mask, ev, vis_mask, labels)
        loss_geo, geo_acc = logits.new_zeros(()), logits.new_zeros(())
        if want_geo:
            loss_geo, geo_acc = self.geometry(target, vis_mask, frame_times.shape[-1] // 2)
        loss_info, info_acc = logits.new_zeros(()), logits.new_zeros(())
        if want_info:
            loss_info, info_acc = self.info_loss(target, hid, ev)
        nll, has = answer_nll(logits, targets, ex_id, b)
        loss_text = nll[has].mean()
        loss_state = loss_text.new_zeros(())
        if want_state:
            loss_state = self.retention(hid, target, torch.ones(b, k, dtype=torch.bool, device=ev.device))
        loss_use, gap = loss_text.new_zeros(()), loss_text.new_zeros(())
        if lambda_use > 0:
            ok = (partner >= 0).to(ev.device) & has
            src = torch.where(partner >= 0, partner, torch.arange(b)).to(ev.device)
            lg2, tg2, ex2, _ = self.core(input_ids, attention_mask, ev[src], vis_mask, labels)
            nll_sw, _ = answer_nll(lg2, tg2, ex2, b, margin_unit)
            d = nll_sw - (nll if margin_unit == "token" else answer_nll(logits, targets, ex_id, b, "sum")[0])                                      # > 0: the answer depends on this video's evidence
            if ok.any():
                loss_use = F.relu(margin - d[ok]).mean()
                gap = d[ok].mean().detach()
        loss = text_weight * (loss_text + lambda_use * loss_use) + lambda_state * loss_state + lambda_geo * loss_geo \
            + lambda_info * loss_info
        if lambda_vc > 0 and reg_vc is not None:
            loss = loss + lambda_vc * reg_vc.to(loss.dtype)
        return {"loss": loss, "loss_text": loss_text.detach(), "loss_state": loss_state.detach(),
                "loss_use": loss_use.detach(), "use_gap": gap, "answer_tokens": has.sum().detach().float(),
                "loss_geo": loss_geo.detach(), "geo_acc": geo_acc, "loss_info": loss_info.detach(), "info_acc": info_acc}
