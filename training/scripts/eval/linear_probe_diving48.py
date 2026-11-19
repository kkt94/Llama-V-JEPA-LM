"""Low-variance companion to the attentive probe: mean-pool the saved Diving48 features over tokens (per level),
standardise, multinomial logistic regression (L2 chosen on a 20% split of train only; torch L-BFGS), test top-1. Deterministic.
usage: linear_probe_diving48.py mean|time <feat_dir> [<feat_dir> ...]   (time keeps the 8 tubelets of the last level)"""
import glob, json, sys
import numpy as np, torch
torch.set_num_threads(32)


def fit_logreg(x, y, l2, n_cls=48, iters=200):
    """multinomial logistic regression, full-batch L-BFGS, L2 on weights"""
    w = torch.zeros(x.shape[1], n_cls, requires_grad=True); b = torch.zeros(n_cls, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(x @ w + b, y) + l2 * (w ** 2).sum(); loss.backward(); return loss
    opt.step(closure)
    return w.detach(), b.detach()


MODE = "mean"


def load(d, split):
    xs, ys = [], []
    for f in sorted(glob.glob(f"{d}/{split}/*")):
        z = torch.load(f, map_location="cpu", weights_only=False) if f.endswith(".pt") else dict(np.load(f, allow_pickle=True))
        x = z["feats"] if "feats" in z else z["x"]; y = z["labels"] if "labels" in z else z["y"]
        x = torch.as_tensor(np.asarray(x)).float()              # [n, tokens, levels, dim]
        if MODE == "mean":                                       # all tokens, 4 levels -> 5632
            xs.append(x.mean(1).flatten(1).numpy())
        else:                                                    # keep time: 8 tubelets x 36 cells, last level -> 8 x 1408
            xs.append(x.view(x.shape[0], 8, 36, *x.shape[2:])[:, :, :, -1].mean(2).flatten(1).numpy())
        ys.append(np.asarray(y))
    return np.concatenate(xs), np.concatenate(ys)


MODE = sys.argv[1]
for d in sys.argv[2:]:
    xtr, ytr = load(d, "train"); xte, yte = load(d, "test")
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr = torch.tensor((xtr - mu) / sd).float(); xte = torch.tensor((xte - mu) / sd).float()
    ytr = torch.tensor(ytr).long(); yte = torch.tensor(yte).long()
    perm = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(0)); nv = len(xtr) // 5
    va, tr = perm[:nv], perm[nv:]
    best = None
    for l2 in (1e-4, 3e-4, 1e-3, 3e-3):
        w, b = fit_logreg(xtr[tr], ytr[tr], l2)
        v = float(((xtr[va] @ w + b).argmax(1) == ytr[va]).float().mean())
        if best is None or v > best[1]:
            best = (l2, v)
    w, b = fit_logreg(xtr, ytr, best[0])
    acc = float(((xte @ w + b).argmax(1) == yte).float().mean())
    print(json.dumps({"feat": d, "mode": MODE, "dim": xtr.shape[1], "l2": best[0], "val": round(best[1], 4), "test_top1": round(acc, 4)}), flush=True)
