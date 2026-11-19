"""The finished V-JEPA grafted as a whole: its encoder (what was seen) AND its own predictor (what comes next).

Standard video LLMs keep V-JEPA's encoder and throw its predictor away. Here both run, frozen and unchanged, the way
V-JEPA's own anticipation evaluation uses them (encoder on the observed frames, predictor queried at the positions of
later tubelets), and the LLM reads the two together:

    training   the encoder sees only the first m tubelets of the clip (m drawn per step); the predictor fills the
               remaining tubelets of the clip and the ones after it; the answer is still the caption / QA of the whole
               clip, so the text loss can only be lowered by reading V-JEPA's predictions (no constructed task: the
               data as it is)
    inference  the encoder sees the whole clip, the predictor adds the next tubelets

Cost: the encoder processes at most as many frames as before (fewer when m < n), the predictor gets a sparse recent
context (last `ctx_recent` tubelets, every `ctx_stride`-th patch) and sparse targets (every `tgt_stride`-th patch):
about 5% of an encoder pass at batch 16. The LLM reads total_tubelets * cells^2 tokens (10 * 49 = 490 < 512).

mode "wm": predicted cells carry the predictor's states; mode "mask": the same slots carry a learned mask token (the
control that tells how much of the result is V-JEPA's prediction rather than the partial-observation training)."""
import numpy as np
import torch

from .evidence_bridge import OBSERVED, PREDICTED, EvidenceBridge, normalize_levels, pool_regions
from .stage1 import Stage1Model, answer_ce, answer_nll

GRID = 24                                    # patches per side at 384 px


class GraftModel(Stage1Model):
    def __init__(self, vision, llm, evidence_cells: int = 7, bridge_hidden: int = 2048, total_tubelets: int = 10,
                 ctx_recent: int = 2, ctx_stride: int = 2, tgt_stride: int = 2, mode: str = "wm",
                 obs_choices=(4, 5, 6, 7, 8), obs_probs=(0.15, 0.15, 0.15, 0.15, 0.4), seed: int = 0):
        super().__init__(vision, llm, None, evidence_cells=evidence_cells, bridge_hidden=bridge_hidden)
        assert mode in ("wm", "mask")
        spec = vision.spec
        d = spec.n_levels * spec.level_dim
        self.bridge = EvidenceBridge({OBSERVED: d, PREDICTED: d}, llm.config.hidden_size, hidden=bridge_hidden,
                                     max_cells=max(8, evidence_cells), mask_token=True)
        self.total, self.ctx_recent, self.ctx_stride, self.tgt_stride, self.mode = total_tubelets, ctx_recent, ctx_stride, tgt_stride, mode
        self.obs_choices, self.obs_probs = list(obs_choices), list(obs_probs)
        self.rng = np.random.default_rng(seed)       # same stream on every rank: the same m per forward everywhere
        self.graft = True

    def draw(self, n_tub):
        if not self.training:
            return n_tub
        return min(n_tub, int(self.rng.choice(self.obs_choices, p=self.obs_probs)))

    def _index(self, tubelets, stride, device):
        t = torch.as_tensor(tubelets, device=device)
        s = torch.arange(0, GRID, stride, device=device)
        return (t[:, None, None] * GRID * GRID + s[None, :, None] * GRID + s[None, None, :]).reshape(-1)

    @torch.no_grad()
    def states(self, video, frame_times, region_grid, m):
        """-> pooled states / coords / seconds / geometry of the observed cells and of the predicted cells"""
        b, n_tub = video.shape[0], video.shape[2] // 2
        spec, c = self.vision.spec, self.cells
        if region_grid is None:
            region_grid = torch.tensor([[0, GRID, 0, GRID]] * b)
        z = self.vision.encoder(video[:, :, : 2 * m], training=True)                 # [B, m*576, L*D], only frames seen
        zn = normalize_levels(z.float().view(b, m * GRID * GRID, spec.n_levels, spec.level_dim))
        po, co, go = pool_regions(zn, m, c, region_grid)
        n_tgt = self.total - m
        if self.mode == "wm":
            ctx = self._index(range(max(0, m - self.ctx_recent), m), self.ctx_stride, video.device)
            tgt = self._index(range(m, self.total), self.tgt_stride, video.device)
            p, _ = self.vision.predictor(z[:, ctx], ctx.unsqueeze(0).expand(b, -1), tgt.unsqueeze(0).expand(b, -1), mod="video")
            g = GRID // self.tgt_stride
            pn = normalize_levels(p.float().view(b, n_tgt * g * g, spec.n_levels, spec.level_dim))
            pn = pn.view(b, n_tgt, g, g, -1).repeat_interleave(self.tgt_stride, 2).repeat_interleave(self.tgt_stride, 3)
            pp, cp, gp = pool_regions(pn.reshape(b, n_tgt * GRID * GRID, spec.n_levels, spec.level_dim), n_tgt, c, region_grid)
        else:
            _, cp, gp = pool_regions(torch.zeros(b, n_tgt * GRID * GRID, 1, 1, device=video.device), n_tgt, c, region_grid)
            pp = torch.zeros(b, n_tgt * c * c, spec.n_levels, spec.level_dim, device=video.device)
        cp = cp.clone(); cp[:, 0] += m
        # real time of every tubelet: known for the clip's own tubelets (seen or not), extrapolated at the clip's stride after it
        ts = frame_times.view(b, n_tub, 2).mean(-1).float()
        step = (ts[:, -1] - ts[:, 0]) / max(1, n_tub - 1)
        ext = ts[:, -1:] + step[:, None] * torch.arange(1, self.total - n_tub + 1, device=ts.device)[None]
        ts = torch.cat([ts, ext], 1)
        so, sp = ts[:, :m].repeat_interleave(c * c, 1), ts[:, m:].repeat_interleave(c * c, 1)
        return (po, co.to(video.device), so, go), (pp, cp.to(video.device), sp, gp)

    def embed(self, video, frame_times, region_grid=None, m=None):
        """-> evidence embeddings [B, total * cells^2, H]: observed cells then predicted cells"""
        m = self.draw(video.shape[2] // 2) if m is None else m
        (po, co, so, go), (pp, cp, sp, gp) = self.states(video, frame_times, region_grid, m)
        eo = self.bridge(po, co, so, OBSERVED, go)
        mask = torch.ones(pp.shape[:2], dtype=torch.bool, device=pp.device) if self.mode == "mask" else None
        ep = self.bridge(pp, cp, sp, PREDICTED, gp, content_mask=mask)
        return torch.cat([eo, ep], 1), m

    def forward(self, video, frame_times, input_ids, attention_mask, labels, vis_mask, region_grid=None,
                text_weight: float = 1.0, **_):
        ev, m = self.embed(video, frame_times, region_grid)
        b, k, _ = ev.shape
        assert int(vis_mask.sum()) == b * k, f"evidence slots {int(vis_mask.sum())} != {b}x{k}"
        logits, targets, ex_id, _ = self.core(input_ids, attention_mask, ev, vis_mask, labels)
        loss_text, n_tok = answer_ce(logits, targets, ex_id, b)
        z = loss_text.new_zeros(())
        return {"loss": text_weight * loss_text, "loss_text": loss_text.detach(), "loss_state": z, "loss_pred": z, "loss_swap": z,
                "swap_gap": z, "loss_geo": z, "geo_acc": z, "loss_info": z, "info_acc": z, "answer_tokens": n_tok.detach(),
                "loss_vc": z, "loss_eq": z, "obs_tubelets": m}

    def forward_pairs(self, video, frame_times, input_ids, attention_mask, labels, vis_mask, partner, region_grid=None,
                      text_weight: float = 1.0, **_):
        ev, m = self.embed(video, frame_times, region_grid)
        b = ev.shape[0]
        logits, targets, ex_id, _ = self.core(input_ids, attention_mask, ev, vis_mask, labels)
        nll, has = answer_nll(logits, targets, ex_id, b)
        loss_text = nll[has].mean()
        z = loss_text.new_zeros(())
        return {"loss": text_weight * loss_text, "loss_text": loss_text.detach(), "loss_state": z, "loss_use": z, "use_gap": z,
                "answer_tokens": has.sum().detach().float(), "loss_geo": z, "geo_acc": z, "loss_info": z, "info_acc": z}
