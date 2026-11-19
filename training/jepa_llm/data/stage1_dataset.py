"""Stage-1 dataset: manifest rows -> (clip, chat tokens with an evidence block, answer labels).

Each manifest is a parquet written by scripts/data/build_stage1_*.py with at least: video (path relative to its prep
root), prep_exists, conversations (JSON list of {"role", "content"} with one evidence placeholder).
Rows whose clip cannot be decoded or whose first QA pair does not fit are dropped by the collator (counted, not hidden).
"""
import json
import os
import re
import random
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from .chat_format import IGNORE, build_example, build_example_multi
from .video import load_clip, region_to_grid, compose_clip, memory_frames, seq_clip, seq_memory_frames


def _columns(path):
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).schema_arrow.names


def _window(r):
    t0, t1 = getattr(r, "t0", None), getattr(r, "t1", None)
    return None if t0 is None or t1 is None or not (t1 > t0) else (float(t0), float(t1))


def _region(r):
    """Sources without a region column get NaN after the manifests are concatenated: no region (whole frame)."""
    reg = getattr(r, "region", None)
    if reg is None or not hasattr(reg, "__len__") or len(reg) != 4:
        return None
    return [float(v) for v in reg]


def _crop(r):
    """Optional square-crop centre (fractions of width / height) for questions that name screen positions in the crop the
    question generator chose; NaN / missing -> centre."""
    import math
    cx, cy = getattr(r, "crop_cx", None), getattr(r, "crop_cy", None)
    ok = lambda v: v is not None and not (isinstance(v, float) and math.isnan(v))
    return None if not (ok(cx) or ok(cy)) else (float(cx) if ok(cx) else 0.5, float(cy) if ok(cy) else 0.5)


class Stage1Dataset(Dataset):
    def __init__(self, sources, tokenizer, num_frames=16, cells=4, max_len=2048, chat_kwargs=None, pair_gap_sec=0.25,
                 extra_tubelets=0, narrative=None, augment=None):
        """sources: list of {"manifest": parquet path, "prep_root": dir, "weight": float (optional, unused here)}."""
        frames, next_pid = [], 0
        for s in sources:
            have = _columns(s["manifest"])
            cols = [c for c in ("video", "prep_exists", "conversations", "t0", "t1", "region", "crop_cx", "crop_cy", "pair_id", "see", "ctx_video", "aug", "ctx_aug", "feat_file", "feat_row", "tq", "root2", "mem_row")
                    if c in have]
            if s.get("twin"):                # corresponding worlds built from this source's clips (see _twin): the rows
                df = pd.read_parquet(s["manifest"], columns=["video", "prep_exists", "conversations", "kind"])   # that name the action
                df = df[df.kind == "open"].drop(columns="kind")
            else:
                df = pd.read_parquet(s["manifest"], columns=cols)
            df = df[df.prep_exists].copy()
            df["root"] = s["prep_root"]
            df["source"] = s.get("name", s["manifest"])
            if "pair_id" in df:             # renumber per source so ids never collide; a pair that lost a member is no pair
                ok = df.groupby("pair_id").pair_id.transform("size") == 2
                codes = pd.factorize(df.loc[ok, "pair_id"])[0]
                df["pair_id"] = -1
                df.loc[ok, "pair_id"] = codes + next_pid
                next_pid += len(set(codes))
            frames.append(df)
        self.df = pd.concat(frames, ignore_index=True)
        if "pair_id" in self.df:
            self.df["pair_id"] = self.df.pair_id.fillna(-1).astype(int)
        self.tok, self.num_frames, self.max_len = tokenizer, num_frames, max_len
        # augment (10-19) {"jitter": 1.0}: plain clips get a new view each time they are drawn (frame times inside their slots, crop position);
        # frozen V-JEPA turns a fixed clip into fixed tokens, so repeated rows were otherwise memorised as the very same input
        self.jitter = float((augment or {}).get("jitter", 0.0))
        # (10-23) a source can opt out with its own "jitter": 0 (synthetic sources whose answers name a screen cell of the square view)
        self.jit = {s.get("name", s["manifest"]): float(s.get("jitter", self.jitter)) for s in sources}
        # (10-19) augment {"frame": "squash" | "pad"}: plain clips (no region / crop point) are given as the whole frame instead of the centre
        # square (a 16:9 frame loses 44% to the square). A source can keep the square with its own "frame": "crop" (dynamics questions whose
        # regions were computed on the square view)
        g = (augment or {}).get("frame", "crop")
        self.frame = {s.get("name", s["manifest"]): s.get("frame", g) for s in sources}
        # Corresponding worlds (spec H2; 10-12): the model reads from V-JEPA what the training text asks for and nothing
        # else (direction, asked by reversed-video pairs, is read; speed and order, asked by nothing, are not: answers
        # are the same on a frozen frame). A source with "twin" makes, from its own labelled clips, videos that differ
        # in ONE aspect only, with the answer that the transformation itself fixes:
        #   order: two clips one after the other, in either order -> which action comes first / the two in order
        #   speed: one clip sampled densely then sparsely, or the reverse, or evenly -> speeds up / slows down / constant
        # Time stamps are nominal (even), so the aspect can only be read from what is seen.
        self.twin = {s.get("name", s["manifest"]): s["twin"] for s in sources if s.get("twin")}
        self.twin_idx = {k: np.flatnonzero((self.df.source == k).values) for k in self.twin}
        self.n_vis = (num_frames // 2 + extra_tubelets) * cells * cells   # extra: tubelets predicted after the clip (graft)
        if os.environ.get("JEPA_TIME_GROUPS"):                          # (10-15) model.time_groups: tubelets averaged in groups
            self.n_vis = int(os.environ["JEPA_TIME_GROUPS"]) * cells * cells
        self.tmem_q = int(os.environ.get("JEPA_TMEM_Q", 0))              # (10-17) temporal memory: M more evidence slots per row
        self.n_vis += self.tmem_q
        self.chat_kwargs, self.pair_gap = chat_kwargs or {}, pair_gap_sec
        # narrative: {"p", "sources", "min_sent"}. A detailed caption tells the video in order ("opens with ... then ...
        # ends with"), and the usual training throws that order away (the whole clip against the whole caption; a model
        # trained that way gives the same caption loss when the clip's time steps are reversed). With probability p such
        # a sample becomes: the first m of the clip's time steps -> the first m / n of the caption's sentences. The start
        # of the story can then only be told from the start of the clip. Nothing is annotated or constructed.
        # mode "reverse" (10-12): {"mode": "reverse", "p", "sources", "min_sent", "require" (regex, default: words of
        # temporal progression)}. The order of events in ordinary data can be guessed from the set of moments and world
        # knowledge, so nothing makes the model read it (a trained model answers 'order' questions equally well on a
        # frozen frame). With probability p a chronological caption sample becomes: the clip's moments in reverse order
        # (each frame pair keeps its own direction, time stamps stay increasing) -> the caption's sentences in reverse
        # order. Video and text are transformed together, so the pair stays true, and the order can no longer be guessed.
        self.narrative, self.cells2, self.n_tub = narrative, cells * cells, num_frames // 2
        self.nv_re = re.compile((narrative or {}).get("require", r"\b(then|next|after|afterwards|later|subsequently|finally|begins|"
                                                                 r"opens|starts|transitions|followed|ends|concludes)\b"), re.I)

    def __len__(self):
        return len(self.df)

    MC = "Answer with the option's letter from the given choices directly."

    def _twin(self, r, mode):
        """-> (messages, video [3, T, H, W], nominal frame times [T])"""
        from .chat_format import VIS
        T = self.num_frames
        act = lambda row: json.loads(row.conversations)[1]["content"].strip()
        if mode == "order":
            a = act(r)
            for _ in range(20):
                q = self.df.iloc[int(random.choice(self.twin_idx[r.source]))]
                if q.video != r.video and act(q) != a:
                    break
            b = act(q)
            va, _, _ = load_clip(f"{r.root}/{r.video}", T // 2, pair_gap_sec=self.pair_gap)
            vb, _, _ = load_clip(f"{q.root}/{q.video}", T // 2, pair_gap_sec=self.pair_gap)
            if random.random() < 0.5:
                a, b, va, vb = b, a, vb, va                                   # the other world: same two clips, other order
            video, form = torch.cat([va, vb], 1), random.random()
            if form < 0.4:
                msgs = [f"{VIS}\nThe video shows two actions, one after the other. Describe them in order.", f"First: {a} Then: {b}"]
            else:
                which, flip = ("first", a) if form < 0.7 else ("last", b), random.random() < 0.5
                x, y = (b, a) if flip else (a, b)
                ans = "A." if (x == which[1]) else "B."
                msgs = [f"{VIS}\nThe video shows two actions, one after the other. Which action is shown {which[0]}?\nA. {x}\nB. {y}\n{self.MC}", ans]
        else:                                                                  # speed
            import decord
            vr = decord.VideoReader(f"{r.root}/{r.video}", num_threads=1); n, fps, h = len(vr), float(vr.get_avg_fps()), T // 2
            kind = random.choice(("up", "down", "const")) if n >= 3 * h else "const"
            st = max(2, min(6, (n - h - 1) // (h - 1))) if kind != "const" else max(1, (n - 1) // (T - 1))
            if kind == "up":      # dense (slow) then sparse (fast)
                idx = list(range(h)) + [h + st * k for k in range(h)]
            elif kind == "down":
                idx = [st * k for k in range(h)] + [st * (h - 1) + 1 + k for k in range(h)]
            else:
                idx = [st * k for k in range(T)]
            idx = np.clip(np.asarray(idx), 0, n - 1)
            video, tt, _ = load_clip(f"{r.root}/{r.video}", T, times=(idx + 0.01) / fps)
            said = {"up": "speeds up", "down": "slows down", "const": "stays at a constant speed"}
            if random.random() < 0.5:
                msgs = [f"{VIS}\nDoes the motion in the video speed up, slow down, or stay at a constant speed?", f"It {said[kind]}."]
            else:
                opts = random.sample(list(said), 3)
                msgs = [f"{VIS}\nHow does the speed of the video change?\n" + "\n".join(f"{'ABC'[j]}. It {said[o]}" for j, o in enumerate(opts)) + f"\n{self.MC}",
                        f"{'ABC'[opts.index(kind)]}."]
        times = (torch.arange(T, dtype=torch.float32) + 0.5) * 0.25               # nominal: 4 frames per second, whatever was done
        return [{"role": "user", "content": msgs[0]}, {"role": "assistant", "content": msgs[1]}], video, times

    def __getitem__(self, i):
        r = self.df.iloc[i]
        if r.source in self.twin:
            try:
                msgs, video, times = self._twin(r, self.twin[r.source])
            except Exception as e:
                return {"skip": f"decode:{type(e).__name__}", "index": i}
            ex = build_example(self.tok, msgs, self.n_vis, self.max_len, self.chat_kwargs)
            if ex is None:
                return {"skip": "too_long", "index": i}
            return {"video": video, "frame_times": times, "region_grid": region_to_grid(None, (0.0, 0.0, 1.0, 1.0)), "index": i, "n_tub": self.n_tub, **ex}
        msgs, n_vis, n_tub = json.loads(r.conversations), self.n_vis, self.n_tub
        ctx = getattr(r, "ctx_video", None)
        if isinstance(ctx, str):                      # (10-15) in-context neighbour: [neighbour clip, Q, A][target clip, Q, A]
            ex = build_example_multi(self.tok, msgs, n_vis, self.max_len, self.chat_kwargs)
            if ex is None:
                return {"skip": "too_long", "index": i}
            ft = _feat(r)
            if isinstance(getattr(r, "feat_file", None), str):
                if ft is None:
                    return {"skip": "no_feature", "index": i}
                return {"video": ft[0], "frame_times": ft[1], "region_grid": ft[2], "index": i, "n_tub": 0, **ex}
            try:
                clips = [load_clip(f"{r.root}/{v}", self.num_frames, pair_gap_sec=self.pair_gap, window=_window(r)) for v in (ctx, r.video)]
            except Exception as e:
                return {"skip": f"decode:{type(e).__name__}", "index": i}
            # (10-15) counterfactual pairs: the clip mirrored ("flip", left <-> right) or played backwards ("rev"); video [3, T, H, W]
            augs = [getattr(r, "ctx_aug", "") or "", getattr(r, "aug", "") or ""]
            clips = [((c[0].flip(-1) if a == "flip" else c[0].flip(1) if a == "rev" else c[0]),) + tuple(c[1:]) for c, a in zip(clips, augs)]
            return {"video": torch.stack([c[0] for c in clips]), "frame_times": torch.stack([c[1] for c in clips]),
                    "region_grid": [region_to_grid(None, c[2]) for c in clips], "index": i, "n_tub": 0, **ex}
        nv, rev = self.narrative, False
        if nv and nv.get("mode") == "reverse":
            if r.source in nv["sources"] and len(msgs) == 2 and random.random() < nv.get("p", 0.5):
                sents = re.split(r"(?<=[.!?])\s+", msgs[1]["content"].strip())
                if len(sents) >= nv.get("min_sent", 3) and self.nv_re.search(msgs[1]["content"]):
                    msgs, rev = [msgs[0], {**msgs[1], "content": " ".join(sents[::-1])}], True
        elif nv and self.n_tub >= 3 and r.source in nv["sources"] and len(msgs) == 2 and random.random() < nv.get("p", 0.5):
            sents = re.split(r"(?<=[.!?])\s+", msgs[1]["content"].strip())
            if len(sents) >= nv.get("min_sent", 3):
                n_tub = random.randint(2, self.n_tub - 1)
                keep = max(1, round(len(sents) * n_tub / self.n_tub))
                msgs = [msgs[0], {**msgs[1], "content": " ".join(sents[:keep])}]
                n_vis = n_tub * self.cells2
        see = getattr(r, "see", None)                 # column `see`: the row shows only its first `see` moments (video-only next-scene codes)
        if see is not None and see == see and 0 < int(see) < self.n_tub:
            n_tub = int(see); n_vis = n_tub * self.cells2
        ex = build_example(self.tok, msgs, n_vis, self.max_len, self.chat_kwargs)
        if ex is None:
            return {"skip": "too_long", "index": i}
        if getattr(self, "concept", None) is not None and r.source not in self.concept.get("exclude", ()):
            ex["cg"] = concept_groups(self, msgs)
        if isinstance(getattr(r, "feat_file", None), str):             # feature store: the pooled grid, no decoding
            ft = _feat(r)
            if ft is None or rev:
                return {"skip": "no_feature", "index": i}
            return {"video": ft[0][0], "frame_times": ft[1][0], "region_grid": ft[2][0], "index": i, "n_tub": n_tub, **ex}
        if isinstance(getattr(r, "tq", None), str):           # transformation question: the clip is composed (_compose)
            sp_ = json.loads(r.tq)
            if sp_.get("kind") == "seq":                     # (10-18) long video built from segments: overview over the joined timeline (+ memory)
                try:
                    video, times = seq_clip(sp_["segs"], r.root, self.num_frames, self.pair_gap)
                except Exception as e:
                    return {"skip": f"decode:{type(e).__name__}", "index": i}
                out = {"video": video, "frame_times": times, "region_grid": [0, 24, 0, 24], "index": i, "n_tub": n_tub, **ex}
                if self.tmem_q:
                    try:
                        fr, msec = seq_memory_frames(sp_["segs"], r.root, kmax=int(os.environ.get("JEPA_TMEM_KMAX", 4)),
                                                     clip_s=float(os.environ.get("JEPA_TMEM_CLIPS", 16.0)), pair_gap_sec=self.pair_gap)
                        if len(msec) > 1:
                            out["memfr"], out["mem_sec"] = fr, msec
                    except Exception:
                        pass
                return out
            try:
                video, times = _compose(r, r.root, self.num_frames, self.pair_gap)
            except Exception as e:
                return {"skip": f"decode:{type(e).__name__}", "index": i}
            a_ = getattr(r, "aug", ""); a_ = a_ if isinstance(a_, str) else ""
            video = video.flip(-1) if a_ == "flip" else video.flip(1) if a_ == "rev" else video
            return {"video": video, "frame_times": times, "region_grid": [0, 24, 0, 24], "index": i, "n_tub": n_tub, **ex}
        try:
            reg = _region(r)
            center = ((reg[0] + reg[2]) / 2, (reg[1] + reg[3]) / 2) if reg is not None else _crop(r)
            video, times, box = load_clip(f"{r.root}/{r.video}", self.num_frames, pair_gap_sec=self.pair_gap,
                                          window=_window(r), crop_center=center, jitter=self.jit.get(r.source, self.jitter) if reg is None else 0.0,
                                          frame=self.frame.get(r.source, "crop") if center is None else "crop")
        except Exception as e:  # corrupt or unreadable clip
            return {"skip": f"decode:{type(e).__name__}", "index": i}
        if rev:
            video = video.view(3, self.n_tub, 2, *video.shape[-2:]).flip(1).reshape(video.shape)
        a_ = getattr(r, "aug", "")                       # (10-16) rows built on transformed clips: mirrored ("flip") / played backwards ("rev")
        a_ = a_ if isinstance(a_, str) else ""            # (10-17) sources without the column carry NaN after the concat (NaN is truthy)
        video = video.flip(-1) if a_ == "flip" else video.flip(1) if a_ == "rev" else video
        out = {"video": video, "frame_times": times, "region_grid": region_to_grid(reg, box), "index": i, "n_tub": n_tub, **ex}
        if self.tmem_q and not a_ and not rev:          # (10-17) memory clips decoded live (<= 4 clips of ~8 s); one clip: the overview itself
            try:
                fr, msec = memory_frames(f"{r.root}/{r.video}", _window(r), center, kmax=int(os.environ.get("JEPA_TMEM_KMAX", 4)),
                                         clip_s=float(os.environ.get("JEPA_TMEM_CLIPS", 8.0)), pair_gap_sec=self.pair_gap)
                if len(msec) > 1:
                    out["memfr"], out["mem_sec"] = fr, msec
            except Exception:
                pass
        return out


def _weights_by_source(self, sources):
    """per-row sampling weights giving each source its share `weight` (default: proportional to its rows)"""
    n = self.df.source.value_counts()
    w = {s.get("name", s["manifest"]): float(s.get("weight", n.get(s.get("name", s["manifest"]), 0))) for s in sources}
    tot = sum(w.values())
    return torch.tensor([w[s] / tot / n[s] for s in self.df.source], dtype=torch.double)


Stage1Dataset.weights_by_source = _weights_by_source


def _compose(r, root, nf, pair_gap):
    """(10-16) transformation questions: column `tq` (JSON spec, video.compose_clip); a concat's second clip is spec["v2"] under `root2`"""
    sp = json.loads(r.tq); w2 = sp.get("w2")
    return compose_clip(sp, f"{root}/{r.video}", nf, pair_gap, window=_window(r),
                        path2=f"{r.root2}/{sp['v2']}" if "v2" in sp else None, window2=tuple(w2) if w2 else None)


_MEM = {}


def _mem(row):
    """(10-17) memory store (logs/zz_mfeat_build.py): entry `row` -> ([K, 128, 1408] fp16 V-JEPA final-level clips, [K] clip centres in
    seconds) or None when the entry was not encoded. Shard files are opened lazily per worker (read-only memmaps)."""
    d = os.environ.get("JEPA_TMEM_STORE")
    if not d:
        return None
    if "idx" not in _MEM:
        ix = np.load(f"{d}/mem.index.npz"); E = pd.read_parquet(f"{d}/entries.parquet", columns=["t0", "t1", "K"])
        _MEM["idx"] = (ix["shard"], ix["off"], ix["K"], ix["ok"], E.t0.values, E.t1.values)
    sh, off, K, ok, t0, t1 = _MEM["idx"]
    if not ok[row]:
        return None
    s_, k = int(sh[row]), int(K[row])
    if s_ not in _MEM:
        _MEM[s_] = np.memmap(f"{d}/mem_{s_}.bin", dtype=np.float16, mode="r")
    n = k * 128 * 1408; o = int(off[row]) // 2
    x = torch.from_numpy(np.array(_MEM[s_][o:o + n])).view(k, 128, 1408)
    sec = torch.tensor(t0[row] + (np.arange(k) + 0.5) * (t1[row] - t0[row]) / k, dtype=torch.float32)
    return x, sec


_FEAT = {}


def _feat(r):
    """(10-15) feature store (logs/zz_feat_build.py): the pooled V-JEPA grid of this row, computed once -> (states [clips, K, L, D] fp32,
    frame times [clips, T], token-grid region [clips, 4]); None when the row was not encoded."""
    f = getattr(r, "feat_file", None)
    if not isinstance(f, str):
        return None
    if f not in _FEAT:
        _FEAT[f] = (np.load(f, mmap_mode="r"), dict(np.load(f.replace(".npy", ".meta.npz"))))
    mm, meta = _FEAT[f]; k = int(r.feat_row)
    if not meta["ok"][k]:
        return None
    x = torch.from_numpy(np.ascontiguousarray(mm[k])).float()
    return x.view(x.shape[0], x.shape[1], 4, -1), torch.from_numpy(meta["times"][k]), meta["grid"][k].tolist()


_CC_STOP = set("something a an the with your of to up down in on into onto from off out over under and or is are it its his her their them they "
               "at by for as be being been this that these those then there here camera wearer person video yes no not".split())   # yes / no / numbers: their meaning is the question's


def concept_groups(ds, msgs):
    """(10-25) concept readout: per multiple-choice turn (user lines 'A. ...', assistant answer 'X' / 'X.'), the first token of every content
    word of every option (lower case, letters only, stop words / yes / no dropped: what the question gives them their meaning) and the gold index. Turns whose options have identical word sets
    (permutations, e.g. order questions) carry no word-level signal and are left out."""
    cache = ds.__dict__.setdefault("_cc_cache", {}); out = []
    for u, a in zip(msgs, msgs[1:]):
        if u.get("role") != "user" or a.get("role") != "assistant":
            continue
        opts = re.findall(r"^\s*([A-H])[.)]\s+(.+?)\s*$", u["content"], flags=re.M); ans = re.match(r"\s*([A-H])\b", a["content"])
        if len(opts) < 2 or not ans or ans.group(1) not in [o[0] for o in opts]:
            continue
        toks = []
        for _, t in opts:
            if t not in cache:
                ids = []
                for w in re.findall(r"[a-z]+", t.lower()):
                    if w not in _CC_STOP:
                        ids += ds.tok.encode(" " + w, add_special_tokens=False)[:1]
                cache[t] = ids[:12]
            toks.append(cache[t])
        if len({tuple(sorted(x)) for x in toks}) < 2 or not toks[[o[0] for o in opts].index(ans.group(1))]:
            continue
        out.append((toks, [o[0] for o in opts].index(ans.group(1))))
    return out


def _collate_cg(kept):
    G = [(j, t, g) for j, b in enumerate(kept) for t, g in b.get("cg", [])]
    if not G:
        return {}
    O = max(len(t) for _, t, _ in G); W = max(max((len(x) for x in t), default=1) for _, t, _ in G); W = max(W, 1)
    tok = torch.full((len(G), O, W), -1, dtype=torch.long)
    for n, (_, t, _) in enumerate(G):
        for o, x in enumerate(t):
            if x:
                tok[n, o, :len(x)] = torch.tensor(x)
    return {"cg": {"row": torch.tensor([j for j, _, _ in G]), "tok": tok, "gold": torch.tensor([g for _, _, g in G])}}


def collate(batch, pad_id):
    kept = [b for b in batch if "skip" not in b]
    skipped = [b["skip"] for b in batch if "skip" in b]
    if not kept:
        return {"skipped": skipped}
    L = max(len(b["input_ids"]) for b in kept)

    def pad(key, value):
        return torch.tensor([b[key] + [value] * (L - len(b[key])) for b in kept])

    return {
        "video": torch.stack([b["video"] for b in kept]),
        "frame_times": torch.stack([b["frame_times"] for b in kept]),
        "region_grid": torch.tensor([b["region_grid"] for b in kept]),
        "input_ids": pad("input_ids", pad_id),
        "labels": pad("labels", IGNORE),
        "vis_mask": pad("vis_mask", 0),
        "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (L - len(b["input_ids"])) for b in kept]),
        "index": torch.tensor([b["index"] for b in kept]),
        "n_tub": torch.tensor([b.get("n_tub", 0) for b in kept]),          # time steps of the clip the example uses (0 = all)
        **({"seg": pad("seg", -1)} if all("seg" in b for b in kept) else {}),    # (10-23) original row of each answer token (packed rows)
        "skipped": skipped,
        **({"mem": _collate_mem(kept)} if any("mem" in b for b in kept) else {}),
        **({"mem": _collate_memfr(kept)} if any("memfr" in b for b in kept) else {}),
        **_collate_cg(kept),
    }


def _collate_memfr(kept):
    """(10-17) live memory clips: uint8 frames padded to the batch's largest K; k = 0 = the row's overview clip is its only memory clip"""
    km = max(b["memfr"].shape[0] // 16 for b in kept if "memfr" in b); H = next(b["memfr"] for b in kept if "memfr" in b).shape[1]
    fr = torch.zeros(len(kept), km * 16, H, H, 3, dtype=torch.uint8); sec = torch.zeros(len(kept), km); k = torch.zeros(len(kept), dtype=torch.long)
    for j, b in enumerate(kept):
        if "memfr" in b:
            n = b["memfr"].shape[0]; fr[j, :n] = b["memfr"]; sec[j, :n // 16] = b["mem_sec"]; k[j] = n // 16
    return {"frames": fr, "k": k, "sec": sec}


def _collate_mem(kept):
    """(10-17) stored memory clips padded to the batch's largest K; k = 0 marks rows without a stored memory (the model pools their own
    clip live)"""
    km = max(b["mem"].shape[0] for b in kept if "mem" in b)
    feat = torch.zeros(len(kept), km, 128, 1408, dtype=torch.float16); sec = torch.zeros(len(kept), km); k = torch.zeros(len(kept), dtype=torch.long)
    for j, b in enumerate(kept):
        if "mem" in b:
            n = b["mem"].shape[0]; feat[j, :n] = b["mem"]; sec[j, :n] = b["mem_sec"]; k[j] = n
    return {"feat": feat, "k": k, "sec": sec}


class ResumableShardSampler(Sampler):
    """Deterministic per-epoch permutation, sharded by rank, that can start from a sample offset (resume without decoding
    the skipped batches). With `source_ids` and `weights` the epoch is a weighted mixture (spec 9.5): an epoch has n
    draws, source s contributes round(n * w_s / sum w) of them, sampled without replacement while its rows last."""

    def __init__(self, n, rank=0, world=1, seed=0, epoch=0, start=0, source_ids=None, weights=None):
        self.n, self.rank, self.world, self.seed, self.epoch, self.start = n, rank, world, seed, epoch, start
        self.per_rank = math.ceil(n / world)
        self.source_ids, self.weights = source_ids, weights

    def _epoch_order(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.source_ids is None or not self.weights:
            return rng.permutation(self.n)
        src = np.asarray(self.source_ids)
        names = sorted(self.weights)
        tot = sum(self.weights[k] for k in names)
        parts = []
        for k in names:
            rows = np.nonzero(src == k)[0]
            if len(rows) == 0:
                continue
            want = int(round(self.n * self.weights[k] / tot))
            reps = [rng.permutation(rows) for _ in range(want // len(rows) + 1)]
            parts.append(np.concatenate(reps)[:want])
        return rng.permutation(np.concatenate(parts))[: self.n] if parts else rng.permutation(self.n)

    def __iter__(self):
        perm = self._epoch_order()
        perm = np.concatenate([perm, perm[: self.per_rank * self.world - len(perm)]])  # pad to equal shards
        mine = perm[self.rank::self.world]
        return iter(mine[self.start:].tolist())

    def __len__(self):
        return self.per_rank - self.start


class PairSampler(Sampler):
    """Batches of `pairs` whole pairs (2 rows each, consecutive), from rows with pair_id >= 0. Per-epoch permutation of
    the pairs, sharded by rank, resumable by batch offset."""

    def __init__(self, df, pairs, rank=0, world=1, seed=0, epoch=0, start_batch=0):
        groups = df[df.pair_id >= 0].groupby("pair_id").groups      # index labels (= dataset rows), not positions
        self.units = [sorted(int(i) for i in v) for v in groups.values() if len(v) == 2]
        self.pairs, self.rank, self.world, self.seed, self.epoch, self.start = pairs, rank, world, seed, epoch, start_batch

    def _batches(self):
        order = np.random.default_rng((self.seed, self.epoch, 7)).permutation(len(self.units))
        mine = order[self.rank::self.world]
        n = len(order) // self.world // self.pairs                       # equal batch count on every rank
        return [[i for u in mine[b * self.pairs:(b + 1) * self.pairs] for i in self.units[u]] for b in range(n)]

    def __iter__(self):
        return iter(self._batches()[self.start:])

    def __len__(self):
        return len(self._batches()) - self.start


def partner_positions(index, pair_ids):
    """index [B] dataset rows of a collated batch (skipped rows already dropped), pair_ids: array over dataset rows ->
    LongTensor [B] position of each row's partner in the batch, -1 if the partner did not survive collation."""
    pos = {int(i): k for k, i in enumerate(index.tolist())}
    by_pid = {}
    for k, i in enumerate(index.tolist()):
        by_pid.setdefault(int(pair_ids[i]), []).append(k)
    out = torch.full((len(pos),), -1, dtype=torch.long)
    for ks in by_pid.values():
        if len(ks) == 2:
            out[ks[0]], out[ks[1]] = ks[1], ks[0]
    return out

class ImageQADataset(Dataset):
    """(10-21) still images with QA / captions (OCR, scene text) for Stage1Model.forward_images. Rows of the listed manifests: `video` = the
    image file under the source's prep_root, `conversations` as in Stage1Dataset. An image is the whole picture (long side to `size`, padded
    square, no crop: text at the edges is kept) as one tubelet of two identical frames, [3, 2, size, size]; its slots = (size/16/pool)^2."""

    def __init__(self, sources, tokenizer, size=1152, pool=3, max_len=2048, chat_kwargs=None):
        parts = []
        for s in sources:
            d = pd.read_parquet(s["manifest"], columns=["video", "conversations"])
            d["root"], d["source"] = s["prep_root"], s.get("name", s["manifest"])
            parts.append(d)
        self.df = pd.concat(parts, ignore_index=True)
        self.tok, self.size, self.max_len, self.chat_kwargs = tokenizer, int(size), max_len, chat_kwargs or {}
        self.n_vis = (self.size // 16 // int(pool)) ** 2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        ex = build_example(self.tok, json.loads(r.conversations), self.n_vis, self.max_len, self.chat_kwargs)
        if ex is None:
            return {"skip": "too_long", "index": i}
        try:
            img, times, _ = load_clip(f"{r.root}/{r.video}", 2, size=self.size, frame="pad")
        except Exception as e:
            return {"skip": f"decode:{type(e).__name__}", "index": i}
        return {"video": img, "frame_times": times, "region_grid": region_to_grid(None, (0.0, 0.0, 1.0, 1.0)), "index": i, **ex}


ImageQADataset.weights_by_source = _weights_by_source
