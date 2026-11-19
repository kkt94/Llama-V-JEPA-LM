"""Unlabelled clips for stage-3 native replay (spec 5.5, 9.5): contiguous clips at 4 fps, like V-JEPA 2.1 pretraining.
Videos come from the `video` column of existing manifests (only rows whose prep clip exists), de-duplicated."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .video import load_clip


class NativeClipDataset(Dataset):
    def __init__(self, sources, num_frames=16, fps=4.0, seed=0):
        frames = []
        for s in sources:
            df = pd.read_parquet(s["manifest"], columns=["video", "prep_exists"])
            df = df[df.prep_exists].drop_duplicates("video")
            frames.append(pd.DataFrame({"path": s["prep_root"] + "/" + df.video}))
        self.paths = pd.concat(frames, ignore_index=True).path.drop_duplicates().tolist()
        self.num_frames, self.fps, self.seed = num_frames, fps, seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import decord
        try:
            vr = decord.VideoReader(self.paths[i], num_threads=1)
            dur = len(vr) / float(vr.get_avg_fps())
            span = self.num_frames / self.fps
            t0 = np.random.default_rng((self.seed, i)).uniform(0, max(dur - span, 0.0))
            times = np.minimum(t0 + np.arange(self.num_frames) / self.fps, max(dur - 1e-3, 0.0))
            v, t, _ = load_clip(self.paths[i], self.num_frames, times=times)
            return {"video": v, "index": i}
        except Exception as e:
            return {"skip": f"decode:{type(e).__name__}", "index": i}


def collate_native(batch):
    kept = [b for b in batch if "skip" not in b]
    return {"video": torch.stack([b["video"] for b in kept]) if kept else None, "skipped": [b["skip"] for b in batch if "skip" in b]}
