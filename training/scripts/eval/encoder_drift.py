"""How far did a fine-tuned V-JEPA encoder move from the official one? Per-level cosine of encoder states and
masked-prediction self-consistency (official cooldown masks) on held-out clips never used for training.
usage: encoder_drift.py --ckpt runs/<run>/ckpt/step_XXXXXXX [--clips 16] --out <json>"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch, torch.nn.functional as F
sys.path.insert(0, "/data/llama_vjepa_lm/training"); sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/eval")
from jepa_health import load_vision, native_consistency  # noqa: E402
from jepa_llm.data.video import load_clip  # noqa: E402
R = "/data/llama_vjepa_lm"
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--clips", type=int, default=16)
ap.add_argument("--out", required=True); a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.set_num_threads(int(os.environ.get("THREADS", 32)))
paths = []
for mf, root in ((f"{R}/manifests/stage1/nvidia_state_heldout.parquet", f"{R}/prep/nvidia_rgb_384"),
                 (f"{R}/manifests/stage2/sav_tracks_heldout.parquet", f"{R}/prep/sav_384_24fps")):
    v = pd.read_parquet(mf, columns=["video"]).video.drop_duplicates().sample(frac=1, random_state=1)
    paths += [f"{root}/{x}" for x in v[: a.clips // 2]]
official, tuned = load_vision("vitg_1b", None, dev), load_vision("vitg_1b", a.ckpt, dev)
cos = []
with torch.no_grad():
    for p in paths:
        v = load_clip(p, 16)[0].unsqueeze(0).to(dev, next(official.parameters()).dtype)
        z0, z1 = official.encode_observation(v).float(), tuned.encode_observation(v).float()
        cos.append(F.cosine_similarity(z0, z1, dim=-1).mean(dim=(0, 1)).tolist())        # per level
nat0 = [r["native_l1"] for r in native_consistency(official, paths, dev, next(official.parameters()).dtype)]
nat1 = [r["native_l1"] for r in native_consistency(tuned, paths, dev, next(tuned.parameters()).dtype)]
res = {"ckpt": a.ckpt, "clips": len(paths), "cos_per_level_mean": np.mean(cos, 0).round(4).tolist(),
       "cos_per_level_min": np.min(cos, 0).round(4).tolist(), "native_l1_official": round(float(np.mean(nat0)), 4),
       "native_l1_tuned": round(float(np.mean(nat1)), 4), "native_l1_ratio": round(float(np.mean(nat1) / np.mean(nat0)), 3)}
print(json.dumps(res)); json.dump(res, open(a.out, "w"), indent=1)
