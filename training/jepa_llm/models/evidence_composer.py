"""EvidenceSpec and the composer head C (spec 3.4, 3.5, 4.1, 6.4).

Pass 1 of the shared LLM reads   [system] [question] [SEM] [overview tokens] [LOC]   (causal), and C reads:
  - SEM hidden  -> semantic operation id         (cannot see the overview: depends on the question only)
  - LOC hidden  -> evidence kind, time bin, region cell (sees question + overview: binds the operation to this video)
Everything is discrete and type-checked; pass 2 never receives pass-1 hidden states or KV cache, only the executed
evidence. Hard choices do not pass gradient: C and pass 1 learn from L_compose only (spec 6.7).

Address space (first version, spec 3.5): time = `n_time_bins` equal bins over the observed clip; region = a
`region_grid` x `region_grid` cell of the frame, or the whole frame (index 0).
"""
from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn.functional as F
from torch import nn


class EvidenceKind(IntEnum):
    OBSERVED = 0
    MASKED_PREDICTION = 1
    SHORT_FUTURE = 2  # disabled until a capability report passes (spec 5.2)


@dataclass(frozen=True)
class EvidenceSpec:
    kind: int
    semantic_id: int
    time_bin: int          # 0 .. n_time_bins-1
    region: int            # 0 = whole frame, 1 .. g*g = cell (row-major)

    def validate(self, n_sem: int, n_time_bins: int, region_grid: int, allow_future: bool = False):
        if self.kind not in (EvidenceKind.OBSERVED, EvidenceKind.MASKED_PREDICTION) and not (
                allow_future and self.kind == EvidenceKind.SHORT_FUTURE):
            raise ValueError(f"evidence kind {self.kind} not enabled")
        if not (0 <= self.semantic_id < n_sem and 0 <= self.time_bin < n_time_bins and 0 <= self.region <= region_grid ** 2):
            raise ValueError(f"spec out of range: {self}")
        if self.kind == EvidenceKind.MASKED_PREDICTION and self.region == 0:
            # masking a whole time bin is frame prediction (SHORT_FUTURE territory); v1 masks one cell so every masked
            # prediction has the same token count and runs as one batched call (equal collectives on every rank)
            raise ValueError(f"masked prediction needs a region cell, got the whole frame: {self}")
        return self


def region_cell_to_grid(region: int, region_grid: int = 3, token_grid: int = 24):
    """region index -> token-grid (y0, y1, x0, x1); 0 = whole frame."""
    if region == 0:
        return [0, token_grid, 0, token_grid]
    r, c = divmod(region - 1, region_grid)
    s = token_grid // region_grid
    return [r * s, (r + 1) * s, c * s, (c + 1) * s]


def time_bin_to_tubelets(time_bin: int, n_time_bins: int, n_tubelets: int):
    """time bin -> [first, last) tubelet indices covered by that bin of the observed clip."""
    a = (time_bin * n_tubelets) // n_time_bins
    b = max(a + 1, ((time_bin + 1) * n_tubelets) // n_time_bins)
    return a, b


class EvidenceComposer(nn.Module):
    """C: small heads on top of pass-1 hidden states at the SEM and LOC positions."""

    def __init__(self, hidden: int, n_sem: int, n_time_bins: int = 4, region_grid: int = 3, n_kinds: int = 2):
        super().__init__()
        self.n_sem, self.n_time_bins, self.region_grid, self.n_kinds = n_sem, n_time_bins, region_grid, n_kinds
        self.sem = nn.Linear(hidden, n_sem)
        self.kind = nn.Linear(hidden, n_kinds)
        self.time = nn.Linear(hidden, n_time_bins)
        self.region = nn.Linear(hidden, 1 + region_grid ** 2)

    def forward(self, h_sem: torch.Tensor, h_loc: torch.Tensor, gold: dict = None, eta: float = 1.0):
        """h_sem, h_loc [B, H]. Returns logits dict and, if `gold` (tensors kind/semantic_id/time_bin/region [B]) is given,
        L_compose = L_sem + eta * L_loc where L_loc averages the kind/time/region CEs (spec 6.4)."""
        out = {"sem": self.sem(h_sem).float(), "kind": self.kind(h_loc).float(), "time": self.time(h_loc).float(),
               "region": self.region(h_loc).float()}
        if gold is None:
            return out, None
        l_sem = F.cross_entropy(out["sem"], gold["semantic_id"])
        l_loc = (F.cross_entropy(out["kind"], gold["kind"]) + F.cross_entropy(out["time"], gold["time_bin"])
                 + F.cross_entropy(out["region"], gold["region"])) / 3
        return out, l_sem + eta * l_loc

    @torch.no_grad()
    def decode(self, logits: dict):
        """argmax -> list[EvidenceSpec] (validated)."""
        k, s, t = (logits[x].argmax(-1).tolist() for x in ("kind", "sem", "time"))
        r_any = logits["region"].argmax(-1).tolist()
        r_cell = (logits["region"][:, 1:].argmax(-1) + 1).tolist()     # masked prediction must pick a cell
        r = [rc if kk == EvidenceKind.MASKED_PREDICTION else ra for kk, ra, rc in zip(k, r_any, r_cell)]
        return [EvidenceSpec(int(a), int(b), int(c), int(d)).validate(self.n_sem, self.n_time_bins, self.region_grid)
                for a, b, c, d in zip(k, s, t, r)]


def choose_specs(model_specs, gold_specs, p_gold: float, generator=None):
    """Teacher forcing for pass 2 (spec 5.4, 2A): use the gold spec with probability p_gold, else the model's choice."""
    pick = torch.rand(len(gold_specs), generator=generator) < p_gold
    return [g if use_gold else m for g, m, use_gold in zip(gold_specs, model_specs, pick.tolist())], pick
