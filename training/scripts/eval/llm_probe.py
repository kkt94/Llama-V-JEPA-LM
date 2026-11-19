"""How much of V-JEPA's non-verbal information survives inside the LLM? (the thesis' compression measure)

Diving48 (fine-grained dive classes, never used for training). For each clip the model sees the evidence with a fixed
neutral prompt; features = LLM hidden states at the evidence positions (middle and final layer), averaged over cells
per tubelet (time order kept), and a linear probe (multinomial logistic regression, L2 picked on a 20% split of train)
predicts the class. `--raw` probes the frozen V-JEPA pooled states themselves (the upper bound the LLM can keep).
usage: llm_probe.py extract --config <cfg>[,<cfg>..] --ckpt <ckpt>[,..] --out <dir>[,..] [--raw --raw_out <dir>] [--n_train 4000]
       llm_probe.py probe <dir> [<dir> ...]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/data/llama_vjepa_lm/training")
R = "/data/llama_vjepa_lm"
PROMPT = "Describe the video."


class _Clips(torch.utils.data.Dataset):
    def __init__(self, rows, nf):
        self.rows, self.nf = rows, nf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from jepa_llm.data.video import load_clip
        v, t, _ = load_clip(self.rows[i]["path"], self.nf)
        return v, t, self.rows[i]["label"]


@torch.no_grad()
def extract(a):
    """One pass over the clips for several checkpoints: decoding (DataLoader workers) and the frozen V-JEPA run once
    per clip, then each checkpoint's bridge + LLM on the same evidence (same prompt for every clip, so clips batch
    without padding). --config/--ckpt/--out take comma lists of equal length; --raw writes the V-JEPA features to
    --raw_out as well."""
    import yaml
    from transformers import AutoTokenizer
    import jepa_llm.eval.runner as RN
    from jepa_llm.data.chat_format import VIS, build_generation_prompt
    cfgs, ckpts, outs = a.config.split(","), (a.ckpt or "").split(","), (a.out or "").split(",")
    if a.raw and not a.ckpt:
        ckpts, outs = [], []
    assert len(ckpts) == len(outs), "one --out per --ckpt"
    if len(cfgs) == 1:
        cfgs = cfgs * max(1, len(ckpts))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    c0 = yaml.safe_load(open(cfgs[0]))
    nf = c0["data"]["num_frames"]; n_tub = nf // 2; cells = c0["data"].get("cells", 4)
    tok = AutoTokenizer.from_pretrained(c0["model"]["llm"])
    models, vision = [], None
    for cfg_p, ck in zip(cfgs, ckpts):
        cfg = yaml.safe_load(open(cfg_p))
        assert (cfg["data"]["num_frames"], cfg["data"].get("cells", 4)) == (nf, cells), cfg_p
        assert cfg["model"].get("train_vision", "none") == "none", "shared V-JEPA needs frozen vision"
        model, mods = RN.build_model(cfg, 1, dev)
        RN.load_weights(mods, ck)
        if vision is None:
            vision = model.vision
        else:
            model.vision = vision                      # frozen V-JEPA is identical for every checkpoint
        model.to(device=dev, dtype=dt).eval()
        layers = model.core.llm.model.layers
        cap = {}
        layers[len(layers) // 2 - 1].register_forward_hook(lambda m, i, o, cap=cap: cap.__setitem__("mid", o[0] if isinstance(o, tuple) else o))
        models.append((model, cap, len(layers) // 2))
    if not models:                                     # raw only
        cfg = c0
        model, _ = RN.build_model(cfg, 1, dev)
        model.to(device=dev, dtype=dt).eval()
        models_raw = model
    ev_model = models[0][0] if models else models_raw
    p = build_generation_prompt(tok, f"{VIS}\n{PROMPT}", n_tub * cells * cells)
    ids1 = torch.tensor([p["input_ids"]], device=dev); vm1 = torch.tensor([p["vis_mask"]], device=dev).bool()
    p0 = int(vm1[0].nonzero().max()) + 1                 # first token after the evidence: "\n", before any question text
    for o in outs + ([a.raw_out] if a.raw else []):
        os.makedirs(o, exist_ok=True)
    for split, n in (("train", a.n_train), ("test", a.n_test)):
        rows = [json.loads(l) for l in open(f"{R}/bench/diving48/{split}.jsonl")]
        if n:
            rows = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[:n]]
        rows = rows[a.shard::a.nshard]
        dl = torch.utils.data.DataLoader(_Clips(rows, nf), batch_size=a.batch, num_workers=a.workers, prefetch_factor=4)
        X = [[] for _ in models]; XT = [[] for _ in models]; XN = [[] for _ in models]; XR, Y = [], []
        for v, t, y in dl:
            v, t = v.to(dev, dt), t.to(dev)
            b = v.shape[0]
            pooled, coords, sec, geom, *res = ev_model.evidence(v, t)
            if a.raw:
                XR.append(pooled.float().view(b, n_tub, cells * cells, -1).mean(2).flatten(1).cpu().numpy().astype(np.float16))
            ids, vm = ids1.expand(b, -1), vm1.expand(b, -1)
            for j, (model, cap, _) in enumerate(models):
                ev = model.bridge(pooled, coords, sec, 0, geom, residual=res[0] if res else None)
                rd = vm.clone(); rd[:, p0] = True; rd[:, -1] = True     # evidence, first token after it ("\n"), last prompt token
                both = model.core(ids, torch.ones_like(ids), ev, vm, labels=None, read_mask=rd)
                both = both.view(b, int(rd[0].sum()), -1)                                    # [b, K+2, H], positions in order
                last = both[:, :-2].reshape(-1, both.shape[-1])                               # [b*K, H] evidence positions
                midh = cap["mid"][vm]
                XN[j].append(torch.cat([cap["mid"][:, p0].float(), both[:, -2].float()], -1).cpu().numpy().astype(np.float16))
                XT[j].append(torch.cat([cap["mid"][:, -1].float(), both[:, -1].float()], -1).cpu().numpy().astype(np.float16))
                f = torch.cat([midh.float().view(b, n_tub, cells * cells, -1).mean(2),
                               last.float().view(b, n_tub, cells * cells, -1).mean(2)], -1)
                X[j].append(f.flatten(1).cpu().numpy().astype(np.float16))
            Y.append(y.numpy())
        name = f"{split}_{a.shard:02d}.npz" if a.nshard > 1 else f"{split}.npz"
        for j, o in enumerate(outs):
            np.savez(f"{o}/{name}", x=np.concatenate(X[j]), xt=np.concatenate(XT[j]), xn=np.concatenate(XN[j]), y=np.concatenate(Y))
        if a.raw:
            np.savez(f"{a.raw_out}/{name}", x=np.concatenate(XR), y=np.concatenate(Y))
        print(split, sum(len(y) for y in Y), flush=True)
    for (model, cap, mid), cfg_p, ck, o in zip(models, cfgs, ckpts, outs):
        json.dump({"config": cfg_p, "ckpt": ck, "raw": False, "prompt": PROMPT, "mid_layer": mid}, open(f"{o}/meta.json", "w"), indent=1)
    if a.raw:
        json.dump({"config": cfgs[0], "ckpt": None, "raw": True, "prompt": PROMPT}, open(f"{a.raw_out}/meta.json", "w"), indent=1)


def fit_logreg(x, y, l2, n_cls=48, iters=200):
    w = torch.zeros(x.shape[1], n_cls, requires_grad=True, device=x.device); b = torch.zeros(n_cls, requires_grad=True, device=x.device)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(x @ w + b, y) + l2 * (w ** 2).sum(); loss.backward(); return loss
    opt.step(closure)
    return w.detach(), b.detach()


def probe(dirs, key="x"):
    torch.set_num_threads(32)
    for d in dirs:
        import glob
        ld = lambda sp: [np.load(f) for f in sorted(glob.glob(f"{d}/{sp}.npz") + glob.glob(f"{d}/{sp}_[0-9][0-9].npz"))]
        trs, tes = ld("train"), ld("test")
        xtr = np.concatenate([t[key] for t in trs]).astype(np.float32); xte = np.concatenate([t[key] for t in tes]).astype(np.float32)
        tr = {"y": np.concatenate([t["y"] for t in trs])}; te = {"y": np.concatenate([t["y"] for t in tes])}
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")     # same fit, seconds instead of minutes
        xtr, xte = torch.tensor((xtr - mu) / sd, device=dev), torch.tensor((xte - mu) / sd, device=dev)
        ytr, yte = torch.tensor(tr["y"]).long().to(dev), torch.tensor(te["y"]).long().to(dev)
        perm = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(0)).to(dev); nv = len(xtr) // 5
        va, trn = perm[:nv], perm[nv:]
        best = max(((l2, float(((xtr[va] @ w + b).argmax(1) == ytr[va]).float().mean()))
                    for l2 in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1) for w, b in [fit_logreg(xtr[trn], ytr[trn], l2)]), key=lambda t: t[1])
        w, b = fit_logreg(xtr, ytr, best[0])
        acc = float(((xte @ w + b).argmax(1) == yte).float().mean())
        res = {"dir": d, "key": key, "dim": xtr.shape[1], "n_train": len(xtr), "n_test": len(xte), "l2": best[0], "val": round(best[1], 4),
               "test_top1": round(acc, 4)}
        print(json.dumps(res), flush=True); json.dump(res, open(f"{d}/probe.json" if key == "x" else f"{d}/probe_{key}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "probe"]); ap.add_argument("dirs", nargs="*")
    ap.add_argument("--config"); ap.add_argument("--ckpt"); ap.add_argument("--raw", action="store_true")
    ap.add_argument("--out"); ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--raw_out"); ap.add_argument("--key", default="x", help="probe: x = evidence positions, xn = first token after the evidence, xt = last prompt position"); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    extract(a) if a.cmd == "extract" else probe(a.dirs, a.key)
