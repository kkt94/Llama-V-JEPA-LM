"""The LLM learns to perceive in the directions its language does not use.

Standard video-LLM recipes were made for language-aligned encoders (CLIP, SigLIP): the features are already close to
words, a small projector does the alignment and the LLM is kept (almost) frozen. V-JEPA is not language-aligned: most
of what it encodes has no word, so the reader - the LLM itself - has to learn to read it. Training the LLM hard on
video, however, overwrites its language. Here every update of an LLM weight matrix W (input x) is restricted to the
directions of x that TEXT does not use:

    delta_W  <-  delta_W (I - U U^T)          U: the top directions of the uncentred second moment of x over
                                              text-only inputs of the pretrained LLM, holding `energy` of it

so, to first order, W x stays what it was for anything in the span of the language directions, while the LLM is free
to build new computation on the part of the activations that only the video tokens bring (25-55% of the activation
energy at the video positions lies outside the 90% language directions, logs/zz_ns_stats.py). The projection is applied
to the optimiser's actual step (after Adam's scaling), so it holds exactly for every step. LLM parameters that are not
such matrices (embeddings, norms) are frozen. No parameter, token or loss is added; inference is unchanged."""
import re

import torch

SPACE = {"q": "attn_in", "k": "attn_in", "v": "attn_in", "o": "o_in", "gate": "mlp_in", "up": "mlp_in", "down": "down_in"}
PAT = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj\.weight$")


def _local(p):
    t = p.data
    return t.to_local() if hasattr(t, "to_local") else t


def freeze_rest(llm):
    """before the optimiser is built: only the projected matrices of the LLM stay trainable"""
    n = 0
    for name, p in llm.named_parameters():
        if not PAT.search(name):
            p.requires_grad_(False); n += p.numel()
    return n


def channel_rank(lam, energy: float) -> int:
    lam = lam.double()
    return int(((lam.cumsum(0) / lam.sum()) < energy).sum()) + 1


def channel_basis(joint: str, energy: float, d: int = None):
    """The language channel of the residual stream (logs/zz_ns_joint.py): -> (U [d, r] most-used directions, V [d, d - r]
    the rest = the perception channel). joint "random": mechanics test."""
    if joint == "random":
        vec = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(0)))[0]; r = d // 2
    else:
        st = torch.load(joint, map_location="cpu"); vec = st["eigvecs"].float(); r = channel_rank(st["eigvals"], energy)
    return vec[:, :r].contiguous(), vec[:, r:].contiguous()


def build_bases(llm, cfg: dict):
    """Call BEFORE the LLM is sharded (it reads the norm gains). cfg: {"path": per-matrix text statistics, "energy",
    "joint": optional residual-stream channel}. -> {(layer, space): ("drop", U) | ("keep", V)}, {key: (rank, dim)}.
    With "joint", the matrices that read the residual stream (q / k / v through the first norm, gate / up through the
    second) are protected on ONE channel shared by all layers: in a layer's post-norm coordinates the language channel
    is span(diag(gain) U) and its complement span(diag(1 / gain) V), so the same residual directions are language (or
    perception) for every layer and for the connector."""
    energy, path, joint = float(cfg.get("energy", 0.99)), cfg.get("path"), cfg.get("joint")
    st = None if path in (None, "random") else torch.load(path, map_location="cpu")
    mats, rank = {}, {}
    for name, p in llm.named_parameters():
        m = PAT.search(name)
        if not m:
            continue
        key = (int(m[1]), SPACE[m[2]]); d = p.shape[-1]
        if key in mats:
            continue
        if joint and key[1] in ("attn_in", "mlp_in"):
            U, V = channel_basis(joint, energy, d)
            layer = llm.model.layers[key[0]]
            g = (layer.input_layernorm if key[1] == "attn_in" else layer.post_attention_layernorm).weight.detach().float().cpu()
            r = U.shape[1]
            mats[key] = ("drop", torch.linalg.qr(U * g[:, None])[0].contiguous()) if r <= d - r else \
                        ("keep", torch.linalg.qr(V / g[:, None])[0].contiguous())
        else:
            if st is None:                                      # mechanics test only: a random half of the directions
                vec = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(key[0])))[0]; r = d // 2
            else:
                vec = st[key]["eigvecs"]; r = channel_rank(st[key]["eigvals"], energy)
                assert vec.shape == (d, d), f"{name}: statistics {tuple(vec.shape)} vs weight input {d}"
            mats[key] = ("drop", vec[:, :r].float().contiguous()) if r <= d - r else ("keep", vec[:, r:].float().contiguous())
        rank[key] = (r, d)
    return mats, rank


class LanguageNullSpace:
    def __init__(self, llm, bases, device):
        mats, self.rank = bases
        self.mats = {k: (mode, m.to(device)) for k, (mode, m) in mats.items()}
        self.items = [(p, (int(m[1]), SPACE[m[2]])) for name, p in llm.named_parameters() for m in [PAT.search(name)] if m and p.requires_grad]
        self.old, self.removed = None, 0.0

    def summary(self):
        by = {}
        for (i, s), (r, d) in self.rank.items():
            by.setdefault(s, []).append(r / d)
        return {s: round(sum(v) / len(v), 3) for s, v in by.items()}       # mean share of the directions that is protected

    @torch.no_grad()
    def snapshot(self):
        self.old = [_local(p).clone() for p, _ in self.items]

    @torch.no_grad()
    def project(self):
        """call right after optimizer.step(): keeps only the part of each step outside the language directions"""
        num = den = 0.0
        for (p, key), old in zip(self.items, self.old):
            w = _local(p)
            if w.numel() == 0:
                continue
            delta = (w - old).float()
            mode, m = self.mats[key]
            kept = delta - (delta @ m) @ m.T if mode == "drop" else (delta @ m) @ m.T
            num += float((delta - kept).pow(2).sum()); den += float(delta.pow(2).sum())
            w.copy_(old + kept.to(w.dtype))
        self.old = None
        self.removed = num / max(den, 1e-30)                                # share of the step that was in the language directions
