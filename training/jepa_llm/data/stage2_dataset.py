"""Stage-2 dataset: question + gold EvidenceSpec + (optional) paired world, rendered for both LLM passes.

Manifest (parquet) columns:
  video, prep_exists, t0, t1            observed clip (seconds); the model never sees frames after t1
  question, answer                      pass-2 answer is the only text label
  gold_kind, gold_sem, gold_time_bin, gold_region   EvidenceSpec supervision (spec 6.4)
  group_id                              rows sharing a question meaning across worlds (spec 8.1)
  pair_id, pair_type                    rows with the same pair_id (exactly 2) form a corresponding pair; pair_type 1 =
                                        the answer must change between them, 0 = it must stay (spec 6.5); pair_id -1 = none
The pair sampler always puts both members of a pair in the same batch, so L_use can swap their evidence.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from .chat_format import IGNORE, VIS, build_example, build_pass1
from .video import load_clip

COLS = ["video", "prep_exists", "t0", "t1", "question", "answer", "gold_kind", "gold_sem", "gold_time_bin",
        "gold_region", "group_id", "pair_id", "pair_type"]
OPTIONAL = ["crop_cx", "crop_cy", "mask_bin", "mask_cell"]   # crop centre (fractions of width/height); blanked bin x cell


def blank_cell(video, time_bin, cell, n_bins=4, rgrid=3):
    """Occlude one time bin x 3x3 cell in pixel space (value 0 = the normalised mean colour). Frame and pixel spans match
    the tubelets / patches the masked-prediction executor targets for the same spec."""
    from jepa_llm.models.evidence_composer import region_cell_to_grid, time_bin_to_tubelets
    n_tub = video.shape[1] // 2
    a, e = time_bin_to_tubelets(time_bin, n_bins, n_tub)
    y0, y1, x0, x1 = region_cell_to_grid(cell, rgrid, 24)
    v = video.clone()
    v[:, 2 * a:2 * e, y0 * 16:y1 * 16, x0 * 16:x1 * 16] = 0.0
    return v


class Stage2Dataset(Dataset):
    def __init__(self, sources, tokenizer, num_frames=16, ev_tubelets=2, ev_cells=4, ov_cells=2, max_len=2048,
                 chat_kwargs=None, pair_gap_sec=0.25):
        frames, next_pid = [], 0
        for s in sources:
            import pyarrow.parquet as pq
            have = pq.ParquetFile(s["manifest"]).schema_arrow.names
            df = pd.read_parquet(s["manifest"], columns=COLS + [c for c in OPTIONAL if c in have])
            df = df[df.prep_exists].copy()
            df["root"] = s["prep_root"]
            # optional `repeat: k` (integer) keeps k copies of a source, e.g. to hold the share of paired rows when a
            # large unpaired source is added. Pair ids are renumbered per source and per copy so they never collide.
            paired = df.pair_id >= 0
            codes = pd.factorize(df.loc[paired, "pair_id"])[0]
            for _ in range(int(s.get("repeat", 1))):
                c = df.copy()
                c.loc[paired, "pair_id"] = codes + next_pid
                next_pid += len(set(codes))
                frames.append(c)
        self.df = pd.concat(frames, ignore_index=True)
        # a pair is usable only if both members survived the prep filter
        cnt = self.df[self.df.pair_id >= 0].groupby("pair_id").size()
        broken = set(cnt[cnt != 2].index)
        self.df.loc[self.df.pair_id.isin(broken), "pair_id"] = -1
        self.tok, self.num_frames, self.max_len, self.pair_gap = tokenizer, num_frames, max_len, pair_gap_sec
        self.n_ov = (num_frames // 2) * ov_cells * ov_cells
        self.n_ev = ev_tubelets * ev_cells * ev_cells
        self.chat_kwargs = chat_kwargs or {}
        self.n_bins, self.rgrid = num_frames // 2 // ev_tubelets, 3

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        p2 = build_example(self.tok, [{"role": "user", "content": f"{VIS}\n{r.question}"},
                                      {"role": "assistant", "content": r.answer}], self.n_ev, self.max_len, self.chat_kwargs)
        if p2 is None:
            return {"skip": "too_long", "index": i}
        p1 = build_pass1(self.tok, r.question, self.n_ov, self.chat_kwargs)
        try:
            cx, cy = getattr(r, "crop_cx", None), getattr(r, "crop_cy", None)
            cx = 0.5 if cx is None or pd.isna(cx) else float(cx)
            cy = 0.5 if cy is None or pd.isna(cy) else float(cy)
            video, times, _ = load_clip(f"{r.root}/{r.video}", self.num_frames, pair_gap_sec=self.pair_gap,
                                        window=(float(r.t0), float(r.t1)), crop_center=(cx, cy))
            mb = getattr(r, "mask_bin", -1)
            if mb is not None and not pd.isna(mb) and int(mb) >= 0:
                video = blank_cell(video, int(mb), int(r.mask_cell), self.n_bins, self.rgrid)
        except Exception as e:
            return {"skip": f"decode:{type(e).__name__}", "index": i}
        return {"video": video, "frame_times": times, "index": i, "pass1": p1, "pass2": p2,
                "gold": (int(r.gold_kind), int(r.gold_sem), int(r.gold_time_bin), int(r.gold_region)),
                "pair_id": int(r.pair_id), "pair_type": int(r.pair_type)}


def _pad(rows, keys, pads):
    L = max(len(r["input_ids"]) for r in rows)
    out = {k: torch.tensor([r[k] + [pads.get(k, 0)] * (L - len(r[k])) for r in rows]) for k in keys}
    out["attention_mask"] = torch.tensor([[1] * len(r["input_ids"]) + [0] * (L - len(r["input_ids"])) for r in rows])
    return out


def collate2(batch, pad_id):
    kept = [b for b in batch if "skip" not in b]
    skipped = [b["skip"] for b in batch if "skip" in b]
    # a pair whose partner was skipped loses its pair link
    pids = [b["pair_id"] for b in kept]
    pair_index = [next((j for j, q in enumerate(pids) if q == p and j != i), -1) if p >= 0 else -1
                  for i, p in enumerate(pids)]
    if not kept:
        return {"skipped": skipped}
    g = torch.tensor([b["gold"] for b in kept])
    return {
        "video": torch.stack([b["video"] for b in kept]),
        "frame_times": torch.stack([b["frame_times"] for b in kept]),
        "pass1": _pad([b["pass1"] for b in kept], ["input_ids", "vis_mask", "sem_mask", "loc_mask"], {"input_ids": pad_id}),
        "pass2": _pad([b["pass2"] for b in kept], ["input_ids", "vis_mask", "labels"], {"input_ids": pad_id, "labels": IGNORE}),
        "gold": {"kind": g[:, 0], "semantic_id": g[:, 1], "time_bin": g[:, 2], "region": g[:, 3]},
        "pair_index": torch.tensor(pair_index), "pair_type": torch.tensor([b["pair_type"] for b in kept]),
        "index": torch.tensor([b["index"] for b in kept]), "skipped": skipped,
    }


class PairBatchSampler(Sampler):
    """Batches of `batch_size` row indices in which every pair appears whole. Units (a pair = 2 rows, a single = 1 row)
    are shuffled per epoch and dealt to ranks; resumable by unit offset."""

    def __init__(self, df, batch_size, rank=0, world=1, seed=0, epoch=0, start_batch=0):
        assert batch_size % 2 == 0, "batch size must be even so pairs fit"
        pairs = df[df.pair_id >= 0].groupby("pair_id").groups          # index labels (= dataset rows), not positions
        singles = df.index[df.pair_id < 0].tolist()
        self.units = [[int(i) for i in v] for v in pairs.values()] + [[i] for i in singles]
        self.bs, self.rank, self.world, self.seed, self.epoch, self.start = batch_size, rank, world, seed, epoch, start_batch

    def _batches(self):
        order = np.random.default_rng(self.seed + self.epoch).permutation(len(self.units))
        batches, cur = [], []
        for u in order:
            unit = self.units[u]
            if len(cur) + len(unit) > self.bs:
                batches.append(cur); cur = []
            cur = cur + unit
        if cur:
            batches.append(cur)
        n = (len(batches) // self.world) * self.world                     # equal batch count on every rank
        return batches[:n][self.rank::self.world]

    def __iter__(self):
        return iter(self._batches()[self.start:])

    def __len__(self):
        return len(self._batches()) - self.start
