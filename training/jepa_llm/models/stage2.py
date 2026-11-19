"""Stage 2 (spec 5.4): shared LLM run twice, composer C, executed observed / masked-prediction evidence, L_use.

    pass 1: [question][SEM][overview 2x2/tubelet][LOC] -> C -> EvidenceSpec (teacher-forced towards gold in 2A)
    execute: OBSERVED          -> observed tokens of the chosen time bin x region, pooled to cells
             MASKED_PREDICTION -> those tokens are removed BEFORE the encoder (visible = the rest of the clip), the
                                  predictor estimates them, the estimate is pooled the same way (spec 7.1-7.3)
    pass 2: [evidence][question][answer] -> L_text; R -> L_state; paired worlds -> L_use (spec 6.5)

E/P are frozen in stage 2, so the reference evidence equals the executed evidence (stop-gradient). Pass 2 gets a fresh
forward: no pass-1 hidden state or KV cache crosses over; only the executed evidence does. Hard spec choices carry no
gradient; C and pass 1 learn from L_compose. Observed and predicted evidence can live in different spaces (distilled
B/L: 768/1024 vs 1664), so B has one input projection and R one head set per evidence kind.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .evidence_bridge import OBSERVED, PREDICTED, EvidenceBridge, RetentionHeads, normalize_levels, pool_regions
from .evidence_composer import (EvidenceComposer, EvidenceKind, EvidenceSpec, choose_specs, region_cell_to_grid,
                                time_bin_to_tubelets)
from .stage1 import LanguageCore, answer_ce

GRID = 24


def per_example_loglik(logits, targets, ex_id, batch_size):
    """Length-normalised answer log-likelihood per example [B] (examples without answer tokens -> 0)."""
    tok = -F.cross_entropy(logits, targets, reduction="none")
    s = torch.zeros(batch_size, device=tok.device).index_add_(0, ex_id, tok)
    n = torch.zeros(batch_size, device=tok.device).index_add_(0, ex_id, torch.ones_like(tok))
    return s / n.clamp_min(1)


class Stage2Model(nn.Module):
    def __init__(self, vision, llm, n_sem, retention_layers=None, n_time_bins=4, region_grid=3, overview_cells=2,
                 evidence_cells=4, bridge_hidden=2048, eta=1.0, use_margin=0.2, invariant_weight=1.0,
                 train_vision: bool = False):
        """train_vision=True is stage 3: online E/P are trained, evidence carries gradient into them, and the retention
        target is the same spec executed by the frozen reference copy E0/P0 (spec 5.5, 6.7)."""
        super().__init__()
        self.vision = vision
        self.train_vision = train_vision
        self.reference = vision.frozen_reference() if train_vision else None
        for p in self.vision.parameters():
            p.requires_grad_(train_vision)
        spec, hidden = vision.spec, llm.config.hidden_size
        self.ov_cells, self.ev_cells, self.n_bins, self.rgrid = overview_cells, evidence_cells, n_time_bins, region_grid
        self.eta, self.mu, self.beta = eta, use_margin, invariant_weight
        self.core = LanguageCore(llm, retention_layers or ())
        self.bridge = EvidenceBridge({OBSERVED: spec.n_levels * spec.level_dim, PREDICTED: spec.pred_levels * spec.pred_dim},
                                     hidden, hidden=bridge_hidden)
        self.composer = EvidenceComposer(hidden, n_sem, n_time_bins, region_grid)
        if retention_layers:
            self.retention = nn.ModuleDict({
                str(OBSERVED): RetentionHeads(hidden, retention_layers, spec.n_levels, spec.level_dim),
                str(PREDICTED): RetentionHeads(hidden, retention_layers, spec.pred_levels, spec.pred_dim)})
        else:
            self.retention = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision.eval()   # no stochastic layers are used; trainability is controlled by requires_grad
        if self.reference is not None:
            self.reference.eval()
        return self

    # ---- evidence ---------------------------------------------------------------------------------------------------
    def observed(self, video, backbone=None):
        bb = backbone or self.vision
        with torch.set_grad_enabled(self.train_vision and bb is self.vision and torch.is_grad_enabled()):
            return normalize_levels(bb.encode_observation(video).float())

    def overview(self, z, frame_times):
        b, n_tub = z.shape[0], z.shape[1] // (GRID * GRID)
        pooled, coords, geom = pool_regions(z, n_tub, self.ov_cells, torch.tensor([[0, GRID, 0, GRID]] * b))
        sec = frame_times.view(b, n_tub, 2).mean(-1).repeat_interleave(self.ov_cells ** 2, dim=1)
        return pooled, coords.to(z.device), sec, geom

    def _target_tokens(self, a, b, region):
        y0, y1, x0, x1 = region_cell_to_grid(region, self.rgrid, GRID)
        t = torch.arange(a, b).view(-1, 1, 1) * GRID * GRID
        yy = torch.arange(y0, y1).view(1, -1, 1) * GRID
        xx = torch.arange(x0, x1).view(1, 1, -1)
        return (t + yy + xx).reshape(-1), (y1 - y0, x1 - x0)

    def execute(self, video, z, frame_times, specs, backbone=None):
        """-> list over samples of (kind, pooled [K, L, D], coords [K, 3], seconds [K], geometry [K, 4]).
        backbone: which E/P runs the masked prediction (default: the online one).
        All masked-prediction samples run as ONE batched encoder+predictor call (every masked spec masks one region
        cell of one time bin, so index lengths match). When the online E/P is trained (stage 3, FSDP-sharded) that
        call happens on every rank every step - on a zero-weight dummy row if the rank has no masked sample - so the
        collectives stay aligned (spec 14.5); the dummy term is kept in self.align_term and added to the loss."""
        bb = backbone or self.vision
        grad = self.train_vision and bb is self.vision and torch.is_grad_enabled()
        b, n_tub = z.shape[0], z.shape[1] // (GRID * GRID)
        tub_sec = frame_times.view(b, n_tub, 2).mean(-1)
        out = [None] * b
        masked = [i for i, s in enumerate(specs) if s.kind == EvidenceKind.MASKED_PREDICTION]
        for i, s in enumerate(specs):
            if s.kind not in (EvidenceKind.OBSERVED, EvidenceKind.MASKED_PREDICTION):
                raise ValueError(f"evidence kind {s.kind} is not enabled")
            if s.kind == EvidenceKind.OBSERVED:
                a, e = time_bin_to_tubelets(s.time_bin, self.n_bins, n_tub)
                reg = torch.tensor([region_cell_to_grid(s.region, self.rgrid, GRID)])
                sub = z[i:i + 1].view(1, n_tub, GRID * GRID, *z.shape[2:])[:, a:e].reshape(1, -1, *z.shape[2:])
                pooled, coords, geom = pool_regions(sub, e - a, self.ev_cells, reg)
                out[i] = (OBSERVED, pooled[0], (coords + torch.tensor([a, 0, 0])).to(z.device),
                          tub_sec[i, a:e].repeat_interleave(self.ev_cells ** 2), geom[0])
        align = self.train_vision and bb is self.vision and torch.distributed.is_initialized()
        rows = masked if masked else ([0] if align else [])
        if bb is self.vision:          # the reference execution (stage-3 retention target) must not reset this
            self.align_term = None
        if rows:
            run_specs = [specs[i] if masked else EvidenceSpec(EvidenceKind.MASKED_PREDICTION, 0, 0, 1) for i in rows]
            tgts, viss, spans = [], [], []
            for s in run_specs:
                a, e = time_bin_to_tubelets(s.time_bin, self.n_bins, n_tub)
                tgt, _ = self._target_tokens(a, e, s.region)
                keep = torch.ones(n_tub * GRID * GRID, dtype=torch.bool)
                keep[tgt] = False
                tgts.append(tgt); viss.append(torch.nonzero(keep).squeeze(1)); spans.append((a, e))
            assert len({len(t) for t in tgts}) == 1, "masked specs must have equal target sizes"
            vis_t, tgt_t = torch.stack(viss).to(video.device), torch.stack(tgts).to(video.device)
            with torch.set_grad_enabled(grad):
                ctx = bb.encode_masked_context(video[rows], vis_t)
                pred, _ = bb.predict(ctx, vis_t, tgt_t)
            pred = normalize_levels(pred.float())
            if not masked:
                self.align_term = pred.sum() * 0.0 if bb is self.vision else None
            else:
                for j, (i, s) in enumerate(zip(rows, run_specs)):
                    a, e = spans[j]
                    full = pred.new_zeros(1, (e - a) * GRID * GRID, *pred.shape[2:])
                    local = self._target_tokens(0, e - a, s.region)[0].to(full.device)
                    full[:, local] = pred[j:j + 1]
                    reg = torch.tensor([region_cell_to_grid(s.region, self.rgrid, GRID)])
                    pooled, coords, geom = pool_regions(full, e - a, self.ev_cells, reg)
                    out[i] = (PREDICTED, pooled[0], (coords + torch.tensor([a, 0, 0])).to(z.device),
                              tub_sec[i, a:e].repeat_interleave(self.ev_cells ** 2), geom[0])
        k = {len(o[1]) for o in out}
        assert len(k) == 1, f"evidence token counts differ within the batch: {k}"
        return out

    def embed_evidence(self, ex):
        """One bridge call per evidence kind (both kinds every step; zero-weight dummy for an absent kind) so the
        sharded bridge runs the same number of forwards on every rank regardless of batch size or kind mix."""
        out = [None] * len(ex)
        dims = {OBSERVED: (self.vision.spec.n_levels, self.vision.spec.level_dim),
                PREDICTED: (self.vision.spec.pred_levels, self.vision.spec.pred_dim)}
        k = len(ex[0][1])
        self.bridge_align = None
        for kind in (OBSERVED, PREDICTED):
            idx = [i for i, e in enumerate(ex) if e[0] == kind]
            if idx:
                h = self.bridge(torch.stack([ex[i][1] for i in idx]), torch.stack([ex[i][2] for i in idx]),
                                torch.stack([ex[i][3] for i in idx]), kind, torch.stack([ex[i][4] for i in idx]))
                for j, i in enumerate(idx):
                    out[i] = h[j]
            else:
                l, d = dims[kind]
                dev = ex[0][1].device
                h = self.bridge(torch.zeros(1, k, l, d, device=dev), ex[0][2].unsqueeze(0), ex[0][3].unsqueeze(0), kind,
                                ex[0][4].unsqueeze(0))
                t = h.sum() * 0.0
                self.bridge_align = t if self.bridge_align is None else self.bridge_align + t
        return torch.stack(out)

    def state_loss(self, hid, ex):
        """ex: the retention TARGET evidence (stage 2: the executed evidence itself; stage 3: the reference execution).
        Both kind heads run every step (zero-weight dummy if a kind is absent) so FSDP collectives match across ranks."""
        if self.retention is None:
            return None
        total = None
        for kind in (OBSERVED, PREDICTED):
            idx = [i for i, e in enumerate(ex) if e[0] == kind]
            head = self.retention[str(kind)]
            if idx:
                tgt = torch.stack([ex[i][1] for i in idx]).detach()
                h = {layer: t[idx] for layer, t in hid.items()}
                term = head(h, tgt, torch.ones(tgt.shape[:2], dtype=torch.bool, device=tgt.device)) * len(idx)
            else:
                h = {layer: t[:1] for layer, t in hid.items()}
                lv, dim = head.n_levels, head.heads[str(head.layers[0])].out_features // head.n_levels
                k = next(iter(h.values())).shape[1]
                dummy_t = torch.zeros(1, k, lv, dim, device=next(iter(h.values())).device)
                term = head(h, dummy_t, torch.ones(1, k, dtype=torch.bool, device=dummy_t.device)) * 0.0
            total = term if total is None else total + term
        return total / len(ex)

    def forward_general(self, video, frame_times, input_ids, attention_mask, labels, vis_mask, lambda_state=0.0,
                        region_grid=None, cells=4):
        """General video-language batches inside stage 2/3 (spec 9.5: 50% of stage-2 updates): the stage-1 readout path
        (whole clip / given region as observed evidence, single pass). In stage 3 the evidence carries gradient into the
        online encoder and the retention target is the reference encoder's evidence."""
        z = self.observed(video)
        b, n_tub = video.shape[0], video.shape[2] // 2
        rg = region_grid if region_grid is not None else torch.tensor([[0, GRID, 0, GRID]] * b)
        pooled, coords, geom = pool_regions(z, n_tub, cells, rg)
        sec = frame_times.view(b, n_tub, 2).mean(-1).repeat_interleave(cells * cells, dim=1)
        ev = self.bridge(pooled, coords.to(video.device), sec, OBSERVED, geom)
        lg, tg, ei, hid = self.core(input_ids, attention_mask, ev, vis_mask, labels=labels)
        l_text, n_tok = answer_ce(lg, tg, ei, b)
        l_state = l_text.new_zeros(())
        if self.retention is not None and lambda_state > 0:
            if self.train_vision:
                with torch.no_grad():
                    target, _, _ = pool_regions(self.observed(video, self.reference), n_tub, cells, rg)
            else:
                target = pooled
            l_state = self.retention[str(OBSERVED)](hid, target.detach(), torch.ones(b, pooled.shape[1], dtype=torch.bool,
                                                                                     device=video.device))
        return {"loss": l_text + lambda_state * l_state, "loss_text": l_text.detach(), "loss_state": l_state.detach(),
                "answer_tokens": n_tok.detach()}

    # ---- forward ----------------------------------------------------------------------------------------------------
    def forward(self, batch, p_gold: float = 1.0, lambdas=None, generator=None):
        """batch: video, frame_times, pass1 {input_ids, attention_mask, vis_mask, sem_mask, loc_mask},
        pass2 {input_ids, attention_mask, vis_mask, labels}, gold {kind, semantic_id, time_bin, region} [B],
        optional pair_index [B] (partner row or -1) and pair_type [B] (1 = answer must change, 0 = invariant)."""
        lam = {"state": 0.1, "compose": 1.0, "use": 0.1, **(lambdas or {})}
        z = self.observed(batch["video"])
        ov_p, ov_c, ov_s, ov_g = self.overview(z, batch["frame_times"])
        ov = self.bridge(ov_p, ov_c, ov_s, OBSERVED, ov_g)
        p1 = batch["pass1"]
        read = (p1["sem_mask"] + p1["loc_mask"]).bool()
        h = self.core(p1["input_ids"], p1["attention_mask"], ov, p1["vis_mask"], labels=None, read_mask=read)
        h = h.view(read.shape[0], 2, -1)  # SEM precedes LOC in every row
        logits, l_compose = self.composer(h[:, 0], h[:, 1], batch["gold"], self.eta)
        gold = [EvidenceSpec(*map(int, t)) for t in zip(batch["gold"]["kind"], batch["gold"]["semantic_id"],
                                                         batch["gold"]["time_bin"], batch["gold"]["region"])]
        model_specs = self.composer.decode(logits)   # always decoded: spec accuracy is reported even when forced
        specs, used_gold = choose_specs(model_specs, gold, p_gold, generator)

        ex = self.execute(batch["video"], z, batch["frame_times"], specs)
        ev = self.embed_evidence(ex)
        p2 = batch["pass2"]
        lg, tg, ei, hid = self.core(p2["input_ids"], p2["attention_mask"], ev, p2["vis_mask"], labels=p2["labels"])
        b = ev.shape[0]
        l_text, n_tok = answer_ce(lg, tg, ei, b)
        l_state = None
        if lam["state"] > 0 and self.retention is not None:
            if self.train_vision:  # stage 3: target = same specs executed by the frozen reference E0/P0
                with torch.no_grad():
                    ref_ex = self.execute(batch["video"], self.observed(batch["video"], self.reference),
                                          batch["frame_times"], specs, backbone=self.reference)
            else:
                ref_ex = ex
            l_state = self.state_loss(hid, ref_ex)
        l_use = None
        pair = batch.get("pair_index")
        if pair is not None and lam["use"] > 0:
            # the swapped pass runs on every rank every step (row 0 with its own evidence and weight 0 if this rank has
            # no pair), so the sharded LLM sees the same number of forwards on all ranks
            has_pair = bool((pair >= 0).any())
            rows = torch.nonzero(pair >= 0).squeeze(1) if has_pair else torch.zeros(1, dtype=torch.long, device=ev.device)
            sw = ev[pair[rows]] if has_pair else ev[rows]                             # partner's evidence, own question
            lg2, tg2, ei2, _ = self.core(p2["input_ids"][rows], p2["attention_mask"][rows], sw, p2["vis_mask"][rows],
                                         labels=p2["labels"][rows])
            other = per_example_loglik(lg2, tg2, ei2, len(rows))
            if has_pair:
                own = per_example_loglik(lg, tg, ei, b)[rows]
                change = batch["pair_type"][rows].bool()
                l_change = F.relu(self.mu - own + other)[change].sum()
                l_inv = (-other)[~change].sum()                                       # same answer must hold under partner
                l_use = (l_change + self.beta * l_inv) / len(rows)
            else:
                l_use = other.sum() * 0.0
        total = l_text + lam["compose"] * l_compose
        if l_state is not None:
            total = total + lam["state"] * l_state
        if l_use is not None:
            total = total + lam["use"] * l_use
        for t in (getattr(self, "align_term", None), getattr(self, "bridge_align", None)):
            if t is not None:
                total = total + t
        z0 = l_text.new_zeros(())
        return {"loss": total, "loss_text": l_text.detach(), "loss_compose": l_compose.detach(),
                "loss_state": (l_state if l_state is not None else z0).detach(),
                "loss_use": (l_use if l_use is not None else z0).detach(), "answer_tokens": n_tok.detach(),
                "gold_used": used_gold.float().mean(),
                "spec_acc": torch.tensor([float(m == g) for m, g in zip(model_specs, gold)]).mean()}
