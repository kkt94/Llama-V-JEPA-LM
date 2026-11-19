import os
"""Shared visual-language bridge B, time-preserving spatial pooling and training-only retention heads R (spec 3.3, 3.7).

Pooling keeps every tubelet (no temporal mean) and pools the 24x24 patch grid to a fixed cell grid (evidence 4x4,
overview 2x2). Levels are normalised separately (parameter-free LayerNorm) before pooling so that the 4 g1B levels or
the single distilled level enter B on a comparable scale.

The first projection is per evidence kind because observed and predicted states can live in different spaces
(distilled B/L: observed 768/1024 vs predicted 1664 in the teacher space). Everything after it is shared.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

OBSERVED, PREDICTED = 0, 1


def normalize_levels(states: torch.Tensor) -> torch.Tensor:
    """[B, N, L, D] -> same shape, each level LayerNorm-ed over D without affine parameters."""
    return F.layer_norm(states, states.shape[-1:])


def pool_grid(states: torch.Tensor, n_tubelets: int, cells: int, grid: int = 24):
    """[B, n_tubelets*grid*grid, L, D] -> ([B, n_tubelets*cells*cells, L, D], coords [n_tubelets*cells*cells, 3] (t, y, x))."""
    b, n, l, d = states.shape
    assert n == n_tubelets * grid * grid, (n, n_tubelets, grid)
    x = states.view(b, n_tubelets, grid, grid, l * d).permute(0, 1, 4, 2, 3).reshape(b * n_tubelets, l * d, grid, grid)
    x = F.adaptive_avg_pool2d(x, cells)  # 24 -> 4 or 2: exact 6x6 / 12x12 averages
    x = x.view(b, n_tubelets, l, d, cells, cells).permute(0, 1, 4, 5, 2, 3).reshape(b, n_tubelets * cells * cells, l, d)
    tt, yy, xx = torch.meshgrid(torch.arange(n_tubelets), torch.arange(cells), torch.arange(cells), indexing="ij")
    return x, torch.stack([tt, yy, xx], -1).reshape(-1, 3)


def pool_regions(states: torch.Tensor, n_tubelets: int, cells: int, regions: torch.Tensor, grid: int = 24):
    """Per-sample region pooling. regions [B, 4] = token-grid (y0, y1, x0, x1). Every tubelet keeps its own cells.
    -> (pooled [B, n_tubelets*cells*cells, L, D], coords [K, 3], geometry [B, K, 4] = (cy, cx, h, w) of each cell as a
    fraction of the full frame, so B still knows where the region is after cropping to it)."""
    b, n, l, d = states.shape
    x = states.view(b, n_tubelets, grid, grid, l * d)
    out, geo = [], []
    for i in range(b):
        y0, y1, x0, x1 = [int(v) for v in regions[i]]
        sub = x[i, :, y0:y1, x0:x1].permute(0, 3, 1, 2)                      # [n_tub, L*D, h, w]
        p = F.adaptive_avg_pool2d(sub, cells)                                  # [n_tub, L*D, c, c]
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
    """Sinusoidal encoding of real timestamps (seconds), periods from 0.1 s to ~1000 s."""
    half = dim // 2
    freqs = torch.exp(-math.log(1e4) * torch.arange(half, device=seconds.device, dtype=torch.float32) / half) * 10.0
    ang = seconds.float().unsqueeze(-1) * freqs
    return torch.cat([ang.sin(), ang.cos()], -1)


class EvidenceBridge(nn.Module):
    def __init__(self, in_dims: dict, out_dim: int, hidden: int = 2048, max_tubelets: int = 64, max_cells: int = 4,
                 time_dim: int = 64, residual_dim: int = 0, mask_token: bool = False):
        """in_dims: {kind_id: levels * dim of that evidence kind}, e.g. {OBSERVED: 5632, PREDICTED: 5632}.
        residual_dim > 0 adds the surprise input: (observed - world-model expectation) per cell, projected and added to
        the evidence token; its last layer starts at zero, so at initialisation the bridge equals the plain one."""
        super().__init__()
        # latent prediction (JEPA objective inside the LLM): a hidden cell keeps its position / time / geometry
        # embeddings but its content is replaced by this learned token
        self.mask_token = nn.Parameter(torch.zeros(out_dim)) if mask_token else None
        self.res = None
        if residual_dim:
            self.res = nn.Sequential(nn.LayerNorm(residual_dim), nn.Linear(residual_dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, out_dim))
            nn.init.zeros_(self.res[3].weight)
            nn.init.zeros_(self.res[3].bias)
        self.inp = nn.ModuleDict({str(k): nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden)) for k, d in in_dims.items()})
        self.out = nn.Sequential(nn.GELU(), nn.Linear(hidden, out_dim))
        self.t_emb = nn.Embedding(max_tubelets, out_dim)
        self.y_emb = nn.Embedding(max_cells, out_dim)
        self.x_emb = nn.Embedding(max_cells, out_dim)
        self.kind_emb = nn.Embedding(2, out_dim)  # OBSERVED, PREDICTED: same shape in stage 1 and 2
        self.time_proj = nn.Linear(time_dim, out_dim)
        self.geom_proj = nn.Linear(4, out_dim)  # absolute (cy, cx, h, w) of each pooled cell in the full frame
        self.time_dim = time_dim
        for e in (self.t_emb, self.y_emb, self.x_emb, self.kind_emb):
            nn.init.normal_(e.weight, std=0.02)
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)
        nn.init.zeros_(self.geom_proj.weight)
        nn.init.zeros_(self.geom_proj.bias)

    # ---- collapse-free alignment -----------------------------------------------------------------------------------
    # Text-loss alignment collapses the visual tokens the LLM receives (effective rank ~2.5 of 2048 after 3000 steps of
    # the standard recipe; 37 for the V-JEPA states of the same clips). JEPA training itself rests on preventing
    # collapse; the same principle is kept during language alignment: over the content tokens of a batch (before
    # position embeddings, scale-free), every dimension must keep unit standard deviation (hinge) and dimensions must
    # be decorrelated (off-diagonal covariance), the variance / covariance terms of VICReg. vc_weight = 0: only the
    # effective rank is monitored (every 25 calls).
    vc_weight, vc_cov, last_reg, last_erank, _calls = 0.0, 0.04, None, None, 0

    def collapse_stats(self, h: torch.Tensor):
        self.last_reg = None
        if not self.training:
            return
        self._calls += 1
        monitor = self._calls % 25 == 1
        if self.vc_weight <= 0 and not monitor:
            return
        x = h.reshape(-1, h.shape[-1]).float()
        if x.shape[0] < 4:
            return
        x = x - x.mean(0, keepdim=True)
        x = x / x.pow(2).mean().sqrt().clamp_min(1e-8)                     # scale-free: only the shape of the spectrum counts
        cov = x.T @ x / (x.shape[0] - 1)
        if self.vc_weight > 0:
            std = (cov.diagonal() + 1e-4).sqrt()
            l_var = F.relu(1.0 - std).mean()
            l_cov = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / cov.shape[0]
            self.last_reg = l_var + self.vc_cov * l_cov
        if monitor:
            with torch.no_grad():
                lam = torch.linalg.eigvalsh(cov.detach().double()).clamp_min(0)
                pr = lam / lam.sum()
                self.last_erank = float(torch.exp(-(pr * (pr + 1e-30).log()).sum()))

    order = "tyx"       # "yxt": tokens leave the bridge trajectory by trajectory (all time steps of a cell in a row)
    channel = None      # ("drop", U) | ("keep", V): the tokens leave the bridge inside the LLM's perception channel

    def to_channel(self, h):
        """The visual tokens are not allowed to look like words: what leaves the bridge is projected off the directions
        of the LLM's residual stream that language uses (training/nullspace.py: channel_basis), so the video enters
        the LLM in the directions where the LLM is free to learn and its language is not touched. fp32 on purpose."""
        if self.channel is None:
            return h
        mode, m = self.channel
        if m.device != h.device:
            m = m.to(h.device); self.channel = (mode, m)
        x = h.float()
        x = x - (x @ m) @ m.T if mode == "drop" else (x @ m) @ m.T
        return x.to(h.dtype)

    def perm(self, coords):
        """Order of the output tokens. Frame by frame ("tyx", as V-JEPA's cells are pooled) puts the same place one time
        step earlier 64 tokens back, and the LLM's video positions give it 0.1% of their attention (they look at the
        neighbours in their own frame). "yxt" lays the clip out along time: the 8 steps of one cell are consecutive
        tokens, so the LLM's sequential machinery runs along V-JEPA's time axis. None = keep the order."""
        if self.order != "yxt":
            return None
        c = coords if coords.dim() == 2 else coords[0]
        return (c[:, 1] * (int(c[:, 2].max()) + 1) + c[:, 2]).mul(int(c[:, 0].max()) + 1).add(c[:, 0]).argsort()

    def forward(self, pooled: torch.Tensor, coords: torch.Tensor, seconds: torch.Tensor, kind: int,
                geometry: torch.Tensor = None, residual: torch.Tensor = None, content_mask: torch.Tensor = None) -> torch.Tensor:
        """pooled [B, K, L, D], coords [K, 3] or [B, K, 3] (t, y, x cell), seconds [B, K], geometry [B, K, 4] -> [B, K, out].
        content_mask [B, K] bool: those cells carry the mask token instead of their content (positions are kept)."""
        b, k, l, d = pooled.shape
        inp = self.inp[str(kind)]
        h = self.out(inp(pooled.reshape(b, k, l * d).to(inp[1].weight.dtype)))  # eval runs the bridge in bf16
        self.collapse_stats(h)
        if content_mask is not None:
            h = torch.where(content_mask.unsqueeze(-1).to(h.device), self.mask_token.to(h.dtype).view(1, 1, -1), h)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        pos = self.t_emb(coords[..., 0]) + self.y_emb(coords[..., 1]) + self.x_emb(coords[..., 2])
        kind_vec = self.kind_emb.weight[kind]
        h = h + pos.to(h.dtype) + kind_vec.to(h.dtype) + self.time_proj(time_features(seconds, self.time_dim).to(self.time_proj.weight.dtype)).to(h.dtype)
        if geometry is not None:
            h = h + self.geom_proj(geometry.to(self.geom_proj.weight.dtype)).to(h.dtype)
        if self.res is not None:
            assert residual is not None, "bridge built with a surprise input but called without residual"
            r = self.res(residual.reshape(b, k, -1).to(self.res[1].weight.dtype)).to(h.dtype)
            h = h + (r if content_mask is None else r * (~content_mask.to(h.device)).unsqueeze(-1).to(h.dtype))
        h = self.to_channel(h)
        p = self.perm(coords)
        return h if p is None else h[:, p]


class RetentionHeads(nn.Module):
    """R: LLM hidden state at evidence positions -> reference visual state, one linear map per (LLM layer, level).
    Training only; dropped at inference (spec 3.7). forward() returns the loss so FSDP hooks see the call."""

    def __init__(self, llm_hidden: int, layers: list, n_levels: int, level_dim: int):
        super().__init__()
        self.layers, self.n_levels = list(layers), n_levels
        self.heads = nn.ModuleDict({str(b): nn.Linear(llm_hidden, n_levels * level_dim) for b in self.layers})

    def forward(self, hidden_by_layer: dict, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """hidden_by_layer[b]: [B, K, H] at evidence positions; target [B, K, L, D] (already stop-grad and normalised);
        valid [B, K] bool. Mean L1 over valid cells, levels, dims and measured layers."""
        b, k, l, d = target.shape
        m = valid.unsqueeze(-1).unsqueeze(-1).to(target.dtype)
        denom = m.sum().clamp_min(1.0) * l * d
        total = target.new_zeros(())
        for layer in self.layers:
            pred = self.heads[str(layer)](hidden_by_layer[layer]).view(b, k, l, d).float()
            total = total + ((pred - target.float()).abs() * m).sum() / denom
        return total / len(self.layers)


class GeometryHead(nn.Module):
    """World-model geometry kept on the language side (relational distillation from frozen V-JEPA).

    Read position: the first token after the evidence (always "\n" in our prompts). Under causal attention its state
    holds what the LLM has taken from the video before any question is read; every answer can attend to it.
    Teacher: the clip's V-JEPA state (per-tubelet mean of the pooled cells, all levels, time order kept), a fixed random
    projection to `dim` (cosines kept to ~0.03), centred on the queue. Student: LLM hidden state at the read position of
    each chosen layer -> LayerNorm -> linear -> `dim`. Loss: KL(teacher neighbour distribution || student) over this
    batch + a FIFO queue of past clips (SEED-style): the LLM must keep apart what V-JEPA keeps apart and keep together
    what it keeps together, whatever the text says. Training only; no question or answer label is involved.
    forward() returns (loss, self-retrieval accuracy) so FSDP hooks see the call."""

    def __init__(self, llm_hidden: int, layers: list, teacher_dim: int, dim: int = 1024, queue: int = 4096,
                 tau_t: float = 0.05, tau_s: float = 0.1, seed: int = 0):
        super().__init__()
        self.layers, self.queue_size, self.tau_t, self.tau_s = list(layers), queue, tau_t, tau_s
        self.heads = nn.ModuleDict({str(b): nn.Sequential(nn.LayerNorm(llm_hidden), nn.Linear(llm_hidden, dim))
                                    for b in self.layers})
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("proj", torch.randn(teacher_dim, dim, generator=g) / dim ** 0.5, persistent=False)
        self.queue = None                                  # [Q, dim] projected teacher states, newest first (per rank)

    def forward(self, read_by_layer: dict, teacher: torch.Tensor):
        """read_by_layer[b]: [B, H] LLM hidden at the read position; teacher [B, T] clip states (no grad)."""
        b = teacher.shape[0]
        with torch.no_grad():
            t = teacher.float() @ self.proj.float()
            bank = t if self.queue is None else torch.cat([t, self.queue])
            cand = F.normalize(bank - bank.mean(0, keepdim=True), dim=-1)          # [B + Q, dim], rows 0..B-1 = batch
            target = F.softmax(cand[:b] @ cand.T / self.tau_t, dim=-1)
            self.queue = bank[: self.queue_size].detach()
        loss, acc = 0.0, 0.0
        for layer in self.layers:
            s = F.normalize(self.heads[str(layer)](read_by_layer[layer]).float(), dim=-1)
            logits = s @ cand.T / self.tau_s
            loss = loss + (target * (target.clamp_min(1e-12).log() - F.log_softmax(logits, -1))).sum(-1).mean()
            acc = acc + (logits.argmax(-1) == torch.arange(b, device=logits.device)).float().mean()
        return loss / len(self.layers), (acc / len(self.layers)).detach()


class ContrastiveCritic(nn.Module):
    """InfoNCE critic against frozen V-JEPA states (the MI lower bound log N - L). Queries come from the LLM, each with
    the index of its positive among this batch's teacher rows; candidates = this batch's teacher rows + a FIFO queue of
    past teacher rows (frozen teacher, so the queue is never stale). Teacher rows go through a fixed random projection
    to `dim` (cosines kept to ~0.03) and are centred on the bank. Candidates whose teacher cosine to the positive is
    >= `dup` count as positives too (the same clip / an unchanged moment seen twice cannot be a negative).
    One linear head per query group (e.g. per prediction horizon). Returns (loss, top-1 accuracy)."""

    def __init__(self, in_dim: int, teacher_dim: int, n_heads: int = 1, dim: int = 1024, queue: int = 4096,
                 tau: float = 0.1, dup: float = 0.98, seed: int = 0):
        super().__init__()
        self.queue_size, self.tau, self.dup = queue, tau, dup
        self.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim)) for _ in range(n_heads)])
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("proj", torch.randn(teacher_dim, dim, generator=g) / dim ** 0.5, persistent=False)
        self.queue = None

    def forward(self, queries: list, positives: list, teacher: torch.Tensor):
        """queries[h] [N_h, in_dim]; positives[h] [N_h] row of `teacher` [M, teacher_dim] that query n must pick."""
        with torch.no_grad():
            t = teacher.float() @ self.proj.float()
            bank = t if self.queue is None else torch.cat([t, self.queue])
            cand = F.normalize(bank - bank.mean(0, keepdim=True), dim=-1)
            self.queue = bank[: self.queue_size].detach()
        loss, acc, n = 0.0, 0.0, 0
        for head, q, pos in zip(self.heads, queries, positives):
            if q.shape[0] == 0:
                continue
            with torch.no_grad():
                pm = (cand[pos] @ cand.T) >= self.dup
                pm[torch.arange(len(pos), device=pos.device), pos] = True
                target = pm.float() / pm.sum(1, keepdim=True)
            lg = F.normalize(head(q).float(), dim=-1) @ cand.T / self.tau
            loss = loss + -(target * F.log_softmax(lg, -1)).sum(-1).mean()
            acc = acc + pm[torch.arange(len(pos), device=pos.device), lg.argmax(-1)].float().mean().detach()
            n += 1
        return loss / max(n, 1), acc / max(n, 1)


class SpectralBridge(EvidenceBridge):
    """Equalised interface. The pooled V-JEPA cell states are extremely anisotropic (over the training mix the top 16 of
    5632 directions hold 48% of the variance, the 1st / 2048th eigenvalue ratio is ~1.6e4), and the fine, non-verbal
    information sits in the small directions (a physics-violation probe reads 'teleport' at 76% from directions that
    hold 0.2% of the variance). A projector and an LLM trained by gradient descent on a text loss learn the large
    directions first and the small ones orders of magnitude more slowly - which looks like information being lost or
    unused. Here the states are first rotated to their principal axes and rescaled (eigenvalue^(-alpha/2); alpha = 1:
    every direction has unit variance), with statistics computed once from frozen V-JEPA: a fixed, invertible linear
    map on the kept directions, no extra computation. Then
      mode "mlp":    the standard projector on the equalised states (only the conditioning changes)
      mode "linear": one linear map, orthogonal at initialisation, times a scalar gain (alignment as a rotation of an
                     isotropic representation; `k` = LLM width keeps it square)."""

    def __init__(self, stats: str, out_dim: int, hidden: int = 2048, k: int = 2048, alpha: float = 1.0,
                 mode: str = "mlp", token_rms: float = 0.02, rotate: bool = True, match: str = None,
                 freeze_map: bool = False, split: int = 0, patch_stats: str = None, patch_gain: float = 1.0, **kw):
        """rotate False = control: per-coordinate standardisation only (every one of the 5632 coordinates to unit
        variance, no rotation to the principal axes), i.e. what a plain feature normalisation would do.
        split S > 0 (10-17, band-split tokens, band_tokens()): a cell token = its 3x3-patch MEAN on the top S directions
        + the 9 patches' OWN coordinates on directions S..k (input width S + 9 (k - S)); patch_stats = per-direction
        variance of a patch's coordinate (logs/zz_patch_stats.py), which scales those patch coordinates to unit variance."""
        st = torch.load(stats, map_location="cpu")
        if not rotate:
            k = st["mean"].numel()
        self_split = int(split)
        # split -1 (10-18, concatenation merge as InternVL's pixel shuffle / Qwen-VL's patch merger): no block mean at all, every
        # direction carried by each of the cell's 9 patches (input width 9 k); the MLP learns the 3x3 -> 1 compression
        super().__init__({OBSERVED: (9 * k) if self_split == -1 else (self_split + 9 * (k - self_split)) if self_split else k}, out_dim, hidden=hidden, **kw)
        self.split = self_split
        assert mode in ("mlp", "linear")
        self.mode, self.k = mode, k
        lam = st["eigvals"][:k].float().clamp_min(1e-8)
        # direction i ends with variance eigenvalue_i^(1 - alpha), rescaled so the mean variance over directions is 1.
        # Kept out of the module's buffers on purpose: they must stay fp32 when the model is cast to bf16 for evaluation
        # (the small directions are multiplied by up to ~100, so bf16 statistics would not reproduce training).
        self._w = {"mean": st["mean"].float(), "basis": st["eigvecs"][:, :k].float().contiguous(),
                   "scale": lam.pow(-alpha / 2) / lam.pow(1 - alpha).mean().sqrt()}
        if self.split:
            assert rotate and alpha == 1.0 and mode == "mlp" and (self.split == -1 or 0 < self.split < k), "band split: rotated, alpha 1, mlp"
            pv = torch.load(patch_stats, map_location="cpu")["var"][:k].float().clamp_min(1e-8)
            # patch_gain 1/3 (10-17, bs2_L): a small direction enters 9 times (once per patch), the top ones once. Unit variance per
            # patch coordinate gave those directions 9x the energy of a top direction (98% of the input); 1/sqrt(9) restores
            # equalisation's rule - every DIRECTION carries the same energy per token.
            self._w["scale_patch"] = pv[max(self.split, 0):].rsqrt() * float(patch_gain); self.patch_gain = float(patch_gain)
        if not rotate:
            var = (st["eigvecs"].double() ** 2) @ st["eigvals"].double().clamp_min(0)        # diagonal of the covariance
            self._w = {"mean": st["mean"].float(), "basis": None, "scale": var.clamp_min(1e-8).rsqrt().float()}
        if mode == "linear":
            lin = nn.Linear(k, out_dim, bias=True)
            nn.init.orthogonal_(lin.weight); nn.init.zeros_(lin.bias)
            self.inp = nn.ModuleDict({str(OBSERVED): lin})
            self.out = nn.Identity()
            self.gain = nn.Parameter(torch.full((1,), float(token_rms)))   # unit-variance directions -> LLM embedding scale (1-D: FSDP)
            if freeze_map:                           # no translator at all (10-12): the equalised V-JEPA state enters as it is
                assert k == out_dim, "the fixed interface needs equal widths"   # (identity; one learned scalar sets its size).
                lin.weight.data.copy_(torch.eye(k)); lin.weight.requires_grad_(False)   # Nothing here can choose what to keep:
                lin.bias.requires_grad_(False)                                   # reading V-JEPA is left entirely to the LLM.
            if match:                                # level-2 alignment without captions (logs/zz_match_stats.py): the map starts as
                st2 = torch.load(match, map_location="cpu")    # mu_w + U diag(s) R x: mean / covariance of the LLM's own input embeddings
                assert st2["W"].shape == lin.weight.shape, (st2["W"].shape, lin.weight.shape)
                lin.weight.data.copy_(st2["W"]); lin.bias.data.copy_(st2["b"]); self.gain.data.fill_(1.0)

    def band_tokens(self, grid):
        """(10-17) band-split tokens. Pooling a 3x3 patch block into one cell keeps 73% of the equalised energy of the top 16
        V-JEPA directions but 31% of directions 1024-2048 (logs/zz_pool_bands.out): the small directions vary from patch to
        patch (adjacent-patch correlation 0.79 for the top 16, 0.29-0.34 beyond 256) and averaging erases them, while
        V-JEPA's information is spread evenly over the directions. So a cell keeps the block mean only where the state is
        spatially smooth (top `split` directions) and carries every patch's own coordinates elsewhere.
        grid [B, T, 24, 24, L*D] level-normalised states (fp32) -> [B, T*64, split + 9*(k - split)] equalised (fp32),
        cells in the order of pool_regions (tubelet, row, column), the 9 patches of a cell in raster order."""
        if self._w["mean"].device != grid.device:
            self._w = {k: (v.to(grid.device) if v is not None else None) for k, v in self._w.items()}
        b, t = grid.shape[:2]; S = max(self.split, 0)                                                 # -1: concatenation only
        with torch.autocast(grid.device.type, enabled=False):                                         # fp32 even under autocast: the small
            x = (grid.reshape(-1, grid.shape[-1]).float() - self._w["mean"]) @ self._w["basis"]       # directions are ~1e-2 of the large ones
        x = x.view(b, t, 8, 3, 8, 3, -1)
        top = x[..., :S].mean((3, 5)) * self._w["scale"][:S]                                          # [B, T, 8, 8, S]
        low = (x[..., S:] * self._w["scale_patch"]).permute(0, 1, 2, 4, 3, 5, 6)                      # [B, T, 8, 8, 3, 3, k-S]
        return torch.cat([top.reshape(b, t * 64, S), low.reshape(b, t * 64, -1)], -1)

    # (10-19) Video in the LLM's own words. Measured: the trained model holds the video's content in a private code - a fitted linear map
    # reads unseen Diving48 dives from it (25.8%, chance 6.2), but with no map it lies nowhere near the directions the same LLM uses for those
    # concepts read as text (2.2% vs 27.7% when the dive is given as text); concepts in the training text do get there (SSv2 19.4%), one by
    # one, in the middle layers. So each token leaving the bridge is made a mixture of the LLM's own input embeddings (the pretrained
    # vocabulary, fixed copy): the bridge's output is a query, the vocabulary is keys and values (vocab_heads heads of width H / heads, cosine
    # logits x a learned scale), and the token enters at word scale and in word directions, where the LLM's language machinery operates.
    vocab_emb, vocab_heads = None, 24

    def to_words(self, h):
        E = self.vocab_emb
        if E.device != h.device:
            E = self.vocab_emb = E.to(h.device)
        b, k, d = h.shape; nh = self.vocab_heads; hd = d // nh
        if getattr(self, "_vk", None) is None or self._vk.device != h.device:                       # keys / values once: [1, nh, V, hd]
            self._vv = E.view(-1, nh, hd).transpose(0, 1).contiguous().unsqueeze(0)
            self._vk = F.normalize(self._vv.float(), dim=-1).to(E.dtype)
        q = F.normalize(h.reshape(1, b * k, nh, hd).transpose(1, 2).to(E.dtype), dim=-1)            # batch folded into the queries
        o = F.scaled_dot_product_attention(q * self.vocab_scale.exp().to(q.dtype), self._vk, self._vv, scale=1.0)   # flash: no [K, V] logits
        o = o.transpose(1, 2).reshape(b, k, d).float()
        if getattr(self, "_wn", None) is None:
            self._wn = float(E.float().norm(dim=1).mean())
        return (o * (self._wn / o.norm(dim=-1, keepdim=True).clamp_min(1e-6))).to(h.dtype)      # a mixture shrinks: back to word size

    def equalise(self, pooled):
        if self._w["mean"].device != pooled.device:
            self._w = {k: (v.to(pooled.device) if v is not None else None) for k, v in self._w.items()}
        b, n = pooled.shape[:2]
        x = pooled.reshape(b, n, -1).float() - self._w["mean"]
        if self._w["basis"] is not None:
            x = x @ self._w["basis"]
        x = x * self._w["scale"]
        if os.environ.get("JEPA_ABLATE_U"):              # (11-12, review W3) measurement only: remove a subspace U [k, r] (orthonormal) from every token
            if getattr(self, "_ablU", None) is None or self._ablU.device != x.device:
                self._ablU = torch.load(os.environ["JEPA_ABLATE_U"], map_location="cpu")["U"].float().to(x.device)
            x = x - (x @ self._ablU) @ self._ablU.T
        return x

    def forward(self, pooled, coords, seconds, kind, geometry=None, residual=None, content_mask=None, keys=None):
        """keys [B, n, k] (equalised coordinates, see EqualisedStateHeads.bind): also returns their tokens, made by the
        same projector but WITHOUT position / time, so a key can only be matched to its cell by content."""
        b, n = pooled.shape[:2]
        inp = self.inp[str(kind)]
        wd = inp[1].weight.dtype if self.mode == "mlp" else inp.weight.dtype
        x = pooled if (self.split and pooled.dim() == 3) else self.equalise(pooled)                   # band split: already equalised
        h = self.out(inp(x.to(wd)))
        # (11-02) flow_proj: V-JEPA's latent flow per cell (Stage1Model.latent_flow, set in _evidence as self._flow [B, n, F]) added to the
        # visual tokens - the cross-moment correspondence V-JEPA already has (a curl rule on it alone gets TOMATO rotation 59.8 % where every
        # video LLM is at 19 - 29 %), computed at full patch resolution before pooling, so the LLM reads it instead of having to compute it
        fl_ = getattr(self, "_flow", None)
        if fl_ is not None and getattr(self, "flow_proj", None) is not None and fl_.shape[:2] == h.shape[:2]:
            if os.environ.get("JEPA_FLOW_ZERO"): fl_ = torch.zeros_like(fl_)          # (11-03) measurement only: is the flow input used?
            h = h + self.flow_proj(fl_.to(h.device, self.flow_proj[0].weight.dtype)).to(h.dtype)
        self._flow = None
        if self.mode == "linear":
            h = h * self.gain.to(h.dtype)
        if self.vocab_emb is not None:
            h = self.to_words(h)
        self.collapse_stats(h)
        if keys is not None:
            kt = self.out(inp(keys.to(wd)))
            return self._place(h, coords, seconds, kind, geometry, content_mask), (kt * self.gain.to(kt.dtype) if self.mode == "linear" else kt)
        return self._place(h, coords, seconds, kind, geometry, content_mask)

    # Time written in the LLM's own words (10-12). A question says "at 5.4 s" with the LLM's token "5"; a video token
    # says when it is with a learned code the LLM has to decipher - and does not (plm_rtloc: answers barely depend on
    # the order of the clip). time_words [K, H] = the LLM's input embeddings of the integers 0..K-1 (read once from the
    # pretrained LLM, fixed, no new parameter): each token gets the embedding of the integer second it shows added to
    # it, so the same number in the question and on the video is the same vector. tw_gain is relative to the token's own
    # size (trained content tokens are ~40x larger than word embeddings; the learned position / time codes are 3-15% of
    # a token, which is why they are hard to read): added vector rms = tw_gain * rms of the token.
    time_words, tw_gain = None, 0.5

    def _place(self, h, coords, seconds, kind, geometry, content_mask):
        if content_mask is not None:
            h = torch.where(content_mask.unsqueeze(-1).to(h.device), self.mask_token.to(h.dtype).view(1, 1, -1), h)
        if self.time_words is not None:
            if self.time_words.device != h.device:
                self.time_words = self.time_words.to(h.device)
            sec_i = seconds.floor().long().clamp(0, self.time_words.shape[0] - 1)
            w = self.time_words[sec_i]
            w = w / w.pow(2).mean(-1, keepdim=True).sqrt()
            h = h + (self.tw_gain * h.detach().float().pow(2).mean(-1, keepdim=True).sqrt() * w).to(h.dtype)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        pos = self.t_emb(coords[..., 0]) + self.y_emb(coords[..., 1]) + self.x_emb(coords[..., 2])
        code = pos.to(h.dtype) + self.kind_emb.weight[kind].to(h.dtype) \
            + self.time_proj(time_features(seconds, self.time_dim).to(self.time_proj.weight.dtype)).to(h.dtype)
        if geometry is not None:
            code = code + self.geom_proj(geometry.to(self.geom_proj.weight.dtype)).to(h.dtype)
        h = h + (code * 0.2 if self.vocab_emb is not None else code)     # (10-19) word-size content: the codes must not drown it
        h = self.to_channel(h)
        p = self.perm(coords)
        return h if p is None else h[:, p]


class Equaliser:
    """V-JEPA's own geometry: pooled cell states -> coordinates on the principal axes with unit variance (fixed statistics
    of frozen V-JEPA, kept fp32; not a module, so casting the model to bf16 leaves it untouched)."""

    def __init__(self, stats: str, k: int = 2048):
        st = torch.load(stats, map_location="cpu")
        self.w = {"mean": st["mean"].float(), "basis": st["eigvecs"][:, :k].float().contiguous(),
                  "scale": st["eigvals"][:k].float().clamp_min(1e-8).rsqrt()}
        self.k = k

    def __call__(self, pooled):
        if self.w["mean"].device != pooled.device:
            self.w = {k: v.to(pooled.device) for k, v in self.w.items()}
        b, n = pooled.shape[:2]
        return ((pooled.reshape(b, n, -1).float() - self.w["mean"]) @ self.w["basis"]) * self.w["scale"]


class EqualisedStateHeads(nn.Module):
    """State preservation measured in the world model's geometry. The original recipe's preservation loss compared LLM
    states with V-JEPA states in raw units, where a handful of large directions carry almost all of the error: it protects
    the coarse content and puts no pressure on the fine, low-variance directions (and did nothing in our runs). Here a
    linear head per chosen LLM layer must give back the EQUALISED state of the same cell, every direction weighing the
    same. With `detach` the heads only watch (a probe that trains on the side); otherwise the gradient also shapes the
    connector and the LLM. Returns (mean squared error, explained fraction for directions 0-16 / 16-256 / 256-k)."""

    BANDS = ((0, 16), (16, 256), (256, None))

    def __init__(self, llm_hidden: int, layers: list, k: int = 2048, stream_tubelets: int = 0, bind_keys: int = 0,
                 read: bool = False, stream_set: bool = False):
        super().__init__()
        self.layers = list(layers)
        self.heads = nn.ModuleDict({str(b): nn.Linear(llm_hidden, k) for b in self.layers})
        # the language stream (stream_tubelets > 0): the LLM states at the TEXT positions must give back the equalised
        # state of every time step of the clip (mean over space), see stream()
        self.read_heads = nn.ModuleDict({str(b): nn.Linear(llm_hidden, k) for b in self.layers}) if read else None
        # binding (bind_keys > 0): key = the first bind_keys directions of a cell, value = the rest, see bind()
        self.bind_keys = bind_keys
        self.bind_heads = nn.ModuleDict({str(b): nn.Linear(llm_hidden, k - bind_keys) for b in self.layers}) if bind_keys else None
        self.n_tub = stream_tubelets
        self.stream_heads = nn.ModuleDict({str(b): nn.Linear(llm_hidden, stream_tubelets * k) for b in self.layers}) \
            if stream_tubelets and not stream_set else None
        # stream_set: the language stream AS A WHOLE carries the clip (stream_whole()); per layer a key map, a value map
        # and one query per time step
        self.set_dim = 256
        self.set_keys = nn.ModuleDict({str(b): nn.Linear(llm_hidden, self.set_dim) for b in self.layers}) if stream_tubelets and stream_set else None
        self.set_vals = nn.ModuleDict({str(b): nn.Linear(llm_hidden, k) for b in self.layers}) if stream_tubelets and stream_set else None
        self.set_query = nn.Parameter(torch.randn(len(self.layers), max(1, stream_tubelets), self.set_dim) * self.set_dim ** -0.5) \
            if stream_tubelets and stream_set else None

    def stream_whole(self, text_by_layer: dict, ok: torch.Tensor, target: torch.Tensor, detach: bool):
        """stream() asks every text position for the state of all time steps at once: 8 x k numbers from one LLM state of
        width ~k, so at most ~1/8 of the uncorrelated (fine) part can fit, and it competes with the position's own
        language. Here the sampled text positions of an example carry the clip TOGETHER: each time step is read back
        from a mixture of the positions (one query per step picks the positions, a shared linear map reads them), so a
        position can hold one step in full and the positions divide the clip among themselves - which position holds
        what is learned, not assigned. Training-only readout. text_by_layer {layer: [B, n, H]}, ok [B, n] valid
        positions, target [B, n_tub, k]. Same error measure as stream()."""
        t = target.float()
        mu = t.mean((0, 1), keepdim=True)
        var = (t - mu).pow(2).mean((0, 1)).clamp_min(1e-6)
        has = ok.any(1)                                                                  # examples with text positions
        tn = (t - mu)[has]
        loss, r2 = 0.0, torch.zeros(len(self.BANDS), device=t.device)
        for li, layer in enumerate(self.layers):
            h = text_by_layer[layer][has]
            h = h.detach() if detach else h
            att = torch.einsum("td,bnd->btn", self.set_query[li].to(h.dtype), self.set_keys[str(layer)](h)).float() / self.set_dim ** 0.5
            att = att.masked_fill(~ok[has].unsqueeze(1), float("-inf")).softmax(-1)      # [B, n_tub, n]
            out = torch.einsum("btn,bnk->btk", att, self.set_vals[str(layer)](h).float())
            per_dir = (out - tn).pow(2).mean((0, 1)) / var
            loss = loss + per_dir.mean()
            with torch.no_grad():
                r2 += torch.stack([1 - per_dir[a:b].mean() for a, b in self.BANDS]) / len(self.layers)
        return loss / len(self.layers), r2

    def read(self, by_layer: dict, ok: torch.Tensor, cells: torch.Tensor):
        """Complete read. Language decides WHERE a word looks in the video (its attention, shaped by the text loss);
        it must not decide WHAT comes across. A caption only ever rewards bringing the coarse, nameable part of what is
        looked at; here the state of every text position must give back the whole equalised V-JEPA state of the cells
        it attended to, fine directions included.
            target_i = sum_j a_ij e_j / sum_j a_ij     a_ij: attention of text position i on video cell j (mean over the
                                                       heads and the layers up to L, the model's own, no gradient)
        measured in equalised units (a sharp look asks for the fine part in full, a diffuse look for little of it) and
        weighted by how much the position looks at the video at all. No token is added and nothing is constructed:
        every word of every caption / QA sample is used as it is.
        by_layer {L: (states [B, n, H], mass [B, n, K])}, ok [B, n], cells [B, K, k]
        -> (loss, [explained fraction for directions 0-16 / 16-256 / 256-k, attention mass on the video, cells looked at])"""
        e = cells.float()
        loss, out = 0.0, torch.zeros(len(self.BANDS) + 2, device=e.device)
        for layer in self.layers:
            h, m = by_layer[layer]
            mass = m.sum(-1)                                                            # [B, n]
            p = m / mass.clamp_min(1e-8).unsqueeze(-1)
            t = p @ e                                                                   # [B, n, k]
            w = (mass * ok).float()
            w = (w / w.sum().clamp_min(1e-8)).unsqueeze(-1)
            err = (w * (self.read_heads[str(layer)](h).float() - t).pow(2)).sum((0, 1))  # [k]
            loss = loss + err.mean()
            with torch.no_grad():
                var = (w * (t - (w * t).sum((0, 1))).pow(2)).sum((0, 1)).clamp_min(1e-8)
                n_eff = torch.exp(-(p * (p + 1e-12).log()).sum(-1))                     # cells effectively looked at
                okf = ok.float()
                out += torch.stack([*[1 - err[a:b].sum() / var[a:b].sum() for a, b in self.BANDS],
                                    (mass * okf).sum() / okf.sum().clamp_min(1), (n_eff * okf).sum() / okf.sum().clamp_min(1)]) / len(self.layers)
        return loss / len(self.layers), out

    def bind(self, key_states: dict, value: torch.Tensor):
        """Key -> value. Language can only point at what it is aligned with: the coarse, large directions of V-JEPA's
        state. The fine directions are what a caption never asks for. Here a token that holds only the coarse part of
        one cell (the key; no position, no time) is appended after the sequence, and the LLM state at that token must
        give back the FINE part of that same cell (the value): the LLM has to find the cell in the video by its coarse
        content and bring its fine content into the stream - the read that non-verbal questions need, taught by the
        video alone. On the principal axes the two parts are linearly uncorrelated, so the value cannot be guessed from
        the key. key_states {layer: [B, n, H]}, value [B, n, k - bind_keys] (unit variance per direction)
        -> (mean squared error, explained fraction for directions bind_keys-1024 / 1024-k)."""
        v = value.float()
        cut = max(1, 1024 - self.bind_keys)
        loss, r2 = 0.0, torch.zeros(2, device=v.device)
        for layer in self.layers:
            err = (self.bind_heads[str(layer)](key_states[layer]).float() - v).pow(2)
            loss = loss + err.mean()
            with torch.no_grad():
                per_dir = err.mean((0, 1))
                r2 += torch.stack([1 - per_dir[:cut].mean(), 1 - per_dir[cut:].mean()]) / len(self.layers)
        return loss / len(self.layers), r2

    def stream(self, text_by_layer: dict, ex: torch.Tensor, target: torch.Tensor, detach: bool):
        """The same constraint carried from the evidence positions into the language stream: nothing asks a standard
        video LLM to move what V-JEPA saw to the positions where language is read and written, unless a caption happens
        to need it. text_by_layer {layer: [N, H]} states at sampled text positions, ex [N] their example, target
        [B, n_tub, k] equalised state per time step (mean over the cells of the step). Every direction weighs the same:
        the error of each coordinate is divided by its variance over the batch. -> (loss, explained fraction per band)"""
        t = target.float()
        mu = t.mean((0, 1), keepdim=True)
        var = (t - mu).pow(2).mean((0, 1)).clamp_min(1e-6)                              # [k]
        tn = (t - mu)[ex]                                                               # [N, n_tub, k]
        loss, r2 = 0.0, torch.zeros(len(self.BANDS), device=t.device)
        for layer in self.layers:
            h = text_by_layer[layer]
            out = self.stream_heads[str(layer)](h.detach() if detach else h).float().view(h.shape[0], self.n_tub, -1)
            per_dir = (out - tn).pow(2).mean((0, 1)) / var
            loss = loss + per_dir.mean()
            with torch.no_grad():
                r2 += torch.stack([1 - per_dir[a:b].mean() for a, b in self.BANDS]) / len(self.layers)
        return loss / len(self.layers), r2

    def forward(self, hidden_by_layer: dict, target: torch.Tensor, detach: bool):
        loss, r2 = 0.0, torch.zeros(len(self.BANDS), device=target.device)
        t = target.float()
        for layer in self.layers:
            h = hidden_by_layer[layer]
            err = (self.heads[str(layer)](h.detach() if detach else h).float() - t).pow(2)       # target has unit variance
            loss = loss + err.mean()
            with torch.no_grad():
                per_dir = err.mean((0, 1))
                r2 += torch.stack([1 - per_dir[a:b].mean() for a, b in self.BANDS]) / len(self.layers)
        return loss / len(self.layers), r2
