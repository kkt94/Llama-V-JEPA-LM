"""Synthetic stage-2 manifest for pipeline smoke tests only (real clips, made-up questions/specs, pairs of rows)."""
import json, random
import pandas as pd
src = pd.read_parquet("/data/llama_vjepa_lm/manifests/stage1/llava_smoke64.parquet")
rng = random.Random(0); rows = []
for i, r in enumerate(src.itertuples()):
    kind = i % 2                                                        # alternate observed / masked prediction
    rows.append({"video": r.video, "prep_exists": True, "t0": 0.0, "t1": 4.0,
                 "question": rng.choice(["Is the person left of the door?", "What is behind the table?"]),
                 "answer": rng.choice(["Yes.", "No.", "A chair."]), "gold_kind": kind, "gold_sem": rng.randrange(3),
                 "gold_time_bin": rng.randrange(4), "gold_region": rng.randrange(1, 10) if kind == 1 else rng.randrange(10), "group_id": i // 4,
                 "pair_id": i // 2 if i < 48 else -1, "pair_type": (i // 2) % 2})
pd.DataFrame(rows).to_parquet("/data/llama_vjepa_lm/manifests/stage2/smoke64.parquet", index=False) if __import__("os").makedirs("/data/llama_vjepa_lm/manifests/stage2", exist_ok=True) is None else None
print(len(rows))
