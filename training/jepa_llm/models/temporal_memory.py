"""(10-17) Temporal memory over many V-JEPA clips (Gated DeltaNet).

The overview path (16 frames over the whole window -> 512 equalised tokens) sees a long video at one frame every few seconds. Here the
window is cut into K clips of 16 frames (one per ~8 s), each encoded by frozen V-JEPA (final level, 4 x 4 cells x 8 steps = 128 tokens),
and the clips' tokens are read IN TIME ORDER by Gated DeltaNet layers (linear in length, a fixed-size state per head:
S_t = a_t S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T, delta-rule overwriting + gated forgetting). M learned query tokens appended after
the last clip read the state (o = S q); their outputs become M extra LLM input tokens, whatever the video length.
Probe before building this (logs/zz_gdn_probe.py, 8 clips, unseen clips): which clip held an event 67.1 (chance 12.5), how often the
first clip recurs 71.1 (25), which of two events came first 69.6 (50); a mean / one attention read / 2-layer transformer stayed near chance.

Inputs: mem [B, Kmax, 128, 1408] fp32 (pooled, level-normalised, final level), k [B] (clips per row, >= 1), clip_sec [B, Kmax] (clip
centre, seconds). Output: [B, M, out_dim]. Weights are fp32 masters; the caller runs it under bf16 autocast."""
import math
import torch
import torch.nn as nn
from fla.layers import GatedDeltaNet

from .evidence_bridge import time_features


class TemporalMemory(nn.Module):
    def __init__(self, stats: str, out_dim: int, d: int = 1024, layers: int = 3, heads: int = 8, head_dim: int = 128, queries: int = 128,
                 k_eq: int = 1408, token_rms: float = 0.02, horizon: tuple = None):
        super().__init__()
        def load(f):
            st = torch.load(f, map_location="cpu"); lam = st["eigvals"][:k_eq].float().clamp_min(1e-8)
            return {"mean": st["mean"].float(), "basis": st["eigvecs"][:, :k_eq].float().contiguous(), "scale": lam.rsqrt()}
        # whitening of the final level (cells 4) with frozen-V-JEPA statistics (384, as the overview path); fp32, outside the buffers
        self._w = load(stats)
        self.inp = nn.Sequential(nn.Linear(k_eq, d), nn.GELU(), nn.Linear(d, d))
        self.cell = nn.Parameter(torch.zeros(16, d)); self.step = nn.Parameter(torch.zeros(8, d))   # position inside a clip
        self.time_dim = 64; self.time_proj = nn.Linear(self.time_dim, d)                              # absolute time of the clip
        self.q = nn.Parameter(torch.randn(queries, d) * 0.02)
        self.mix = nn.ModuleList([GatedDeltaNet(hidden_size=d, head_dim=head_dim, num_heads=heads, mode="chunk", layer_idx=i) for i in range(layers)])
        if horizon:
            # (10-18) fla's default decay a = exp(-exp(A_log) softplus(a_proj x + dt_bias)) starts near 0.875 (A_log ~2.5, dt_bias ~-4.5):
            # a memory of ~8 tokens, while a clip is 128 tokens - gm_L's queries read the last clip's last cells (measured horizon 4-14).
            # Here every head starts with its own horizon H, log-spaced over horizon = (H_min, H_max) tokens (as Mamba's dt): A_log 0,
            # dt_bias = softplus^-1(1 / H), so a = exp(-1 / H) at a_proj x = 0
            import math
            for mx in self.mix:
                H = torch.logspace(math.log10(horizon[0]), math.log10(horizon[1]), heads)
                with torch.no_grad():
                    mx.A_log.zero_(); mx.dt_bias.copy_(torch.log(torch.expm1(1.0 / H)))
        self.n1 = nn.ModuleList([nn.RMSNorm(d) for _ in range(layers)]); self.n2 = nn.ModuleList([nn.RMSNorm(d) for _ in range(layers)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(layers)])
        self.out = nn.Sequential(nn.RMSNorm(d), nn.Linear(d, out_dim))
        self.gain = nn.Parameter(torch.full((1,), float(token_rms)))   # tokens at the scale of the LLM's input embeddings
        self.queries, self.d = queries, d

    def equalise(self, x, w):
        return ((x.float() - w["mean"]) @ w["basis"]) * w["scale"]

    def forward(self, mem, k, clip_sec):
        B, Kmax, T, D = mem.shape                                                     # T = 128 = 8 steps x 16 cells (step-major)
        if self._w["mean"].device != mem.device:
            self._w = {a: b.to(mem.device) for a, b in self._w.items()}
        h = self.inp(self.equalise(mem, self._w).to(self.inp[0].weight.dtype))         # [B, Kmax, 128, d]
        pos = (self.step.view(8, 1, -1) + self.cell.view(1, 16, -1)).view(1, 1, T, -1)
        tf = self.time_proj(time_features(clip_sec, self.time_dim).to(self.time_proj.weight.dtype)).unsqueeze(2)
        h = (h + pos + tf).reshape(B, Kmax * T, -1)
        # left-pad: row b's valid tokens are its first k[b] clips; put them right before the queries so the queries read the
        # state after the LAST valid clip. Padding positions are masked out of the recurrence (fla unpads them).
        L = Kmax * T; n_valid = k.to(h.device) * T
        src = torch.arange(L, device=h.device).view(1, L) - (L - n_valid).view(B, 1)     # target position j holds source j - (L - n)
        keep = src >= 0
        h = torch.where(keep.unsqueeze(-1), h.gather(1, src.clamp_min(0).unsqueeze(-1).expand(-1, -1, h.shape[-1])), torch.zeros_like(h))
        h = torch.cat([h, self.q.unsqueeze(0).expand(B, -1, -1).to(h.dtype)], 1)
        mask = torch.cat([keep, torch.ones(B, self.queries, dtype=torch.bool, device=h.device)], 1).long()
        for mix, n1, n2, mlp in zip(self.mix, self.n1, self.n2, self.mlp):
            h = h + mix(n1(h), attention_mask=mask)[0].to(h.dtype)
            h = h + mlp(n2(h))
        return self.out(h[:, -self.queries:]) * self.gain
