"""Stage-1 data for MVP `robot_object_interactions` from Language-Table REAL (TFDS language_table 0.0.1, train split).

Source episodes / videos: scripts/prep/prep_language_table.py -> prep/language_table_train/<gid>.mp4 (640x360, 5 fps,
one frame per step, h264) + per-shard metadata prep/language_table_train_meta/shard_*.jsonl for ALL 442,226 episodes.
<gid> is the GLOBAL episode index in TFDS enumeration order (default interleave cycle 16 / block 16), i.e. the same id
MVP uses in `language_table/video_<gid>.mp4`.

Exclusions: every MVP language_table id in manifests/allowlist/benchmark_ids_mvp.txt (25,768) - never encoded, never a
manifest row, never used as a negative; episodes with < 8 frames or an empty instruction. Up to 200,000 episodes are
kept: the encoded episodes with the smallest hash u (md5 seed 0), i.e. a deterministic uniform sample of the dataset.

Text normalisation = MVP's option text: lowercase, whitespace-collapsed, and every whole word "push" / "place" /
"slide" replaced by "move" (VERB_TO_MOVE: learned by comparing the raw instructions of all MVP ids with MVP's correct
options - 100% agreement; re-checked at build time, nudge / put / drag / keep / separate ... are left as MVP left them).

Rows (conversations JSON: user "<|jepa_evidence|>\\n" + prompt, assistant answer):
  mc    every episode. "Which robot instruction best describes the actions in the video?" + A./B. options + MVP suffix,
        answer "A." / "B." (true option position random, balanced). Options = the true instruction and ONE hard
        negative: another REAL instruction of a non-MVP episode with the smallest word-level edit distance > 0,
        excluding: identical after lowercasing / punctuation stripping, changes that touch only function words, typo-
        like single-word changes, synonym swaps (up/top/above, ...). Changes of a direction / colour / shape word are
        preferred (a non-salient change costs +1 distance in the ranking). Ties broken per episode at random.
  open  30% of episodes. "What instruction is the robot following in the video?" -> "<Instruction>." (normalised text,
        first letter capitalised, final period).
Columns: video ('<gid>.mp4' under prep_root prep/language_table_train), prep_exists, kind, episode, conversations,
         plus neg_dist / neg_text for mc rows.
Split: 2% of episodes by md5(episode) -> manifests/stage1/language_table_heldout.parquet, rest -> language_table.parquet.
"""
import argparse
import collections
import difflib
import glob
import hashlib
import json
import os
import multiprocessing as mp
import re

import numpy as np
import pandas as pd

R = "/data/llama_vjepa_lm"
PREP = f"{R}/prep/language_table_train"
META = f"{R}/prep/language_table_train_meta"
DEST = f"{R}/manifests/stage1/language_table.parquet"
ALLOW = f"{R}/manifests/allowlist/benchmark_ids_mvp.txt"
MVP_EVAL = f"{R}/manifests/eval/mvp_full.jsonl"
VIS = "<|jepa_evidence|>"
Q_MC = "Which robot instruction best describes the actions in the video?"
Q_OPEN = "What instruction is the robot following in the video?"
SUFFIX = "Answer with the option's letter from the given choices directly."
LETTERS = "AB"
MAX_EPISODES = 200_000
MIN_FRAMES = 8
OPEN_FRAC = 0.30
HELDOUT_PCT = 2

# words MVP rewrote to "move" wherever they occur (learned by learn_verb_map over all MVP ids; re-verified at build
# time). MVP kept every other verb (move / separate / put / keep / drag / adjust / nudge / ...) unchanged.
VERB_TO_MOVE = {"place", "push", "slide"}

COLOURS = {"red", "blue", "green", "yellow", "purple", "orange", "white", "black", "pink"}
SHAPES = {"circle", "star", "cube", "triangle", "heart", "hexagon", "moon", "square", "block", "blocks", "pentagon",
          "circles", "stars", "cubes", "triangles", "hearts", "hexagons"}
DIR_GROUPS = [{"up", "top", "above", "upper", "upwards", "upward", "upside", "topmost", "north"},
              {"down", "bottom", "below", "lower", "downwards", "downward", "beneath", "under", "underneath", "south"},
              {"left", "leftwards", "leftward", "leftside", "west"},
              {"right", "rightwards", "rightward", "rightside", "east"},
              {"center", "centre", "middle", "central"},
              {"corner"}, {"side"}, {"edge"}, {"diagonal", "diagonally"}, {"near", "close", "closer", "next", "beside", "nearer"},
              {"away", "far", "farther", "further"}, {"between", "in-between"}]
DIRS = set().union(*DIR_GROUPS)
GROUP_OF = {w: i for i, g in enumerate(DIR_GROUPS) for w in g}
SALIENT = COLOURS | SHAPES | DIRS
STOP = {"the", "a", "an", "to", "of", "on", "in", "into", "at", "with", "from", "and", "is", "it", "its", "please", "then",
        "so", "that", "this", "just", "bit", "little", "slightly", "very", "some", "towards", "toward", "by", "for", "as",
        "much", "more", "small", "tiny", "few", "inch", "inches", "position", "place", "area", "board", "table", "spot",
        "part", "region", "object", "objects", "piece", "one", "all", "move", "push", "slide", "nudge", "put", "drag",
        "location", "now", "over", "there", "up"}


def norm_basic(s):
    return " ".join(s.strip().lower().split())


def key_of(s):
    """identity key: lowercase, punctuation stripped, whitespace collapsed."""
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


def mvp_norm(s):
    """MVP option text: lowercase, whitespace-collapsed, every whole word in VERB_TO_MOVE replaced by "move"."""
    return " ".join("move" if w in VERB_TO_MOVE else w for w in norm_basic(s).split())


def load_mvp_ids():
    ids = {int(l.split("\t")[1].replace("video_", "")) for l in open(ALLOW) if l.startswith("language_table\t")}
    assert ids, "no MVP language_table ids read: the exclusion would be vacuous"
    return ids


def load_meta():
    files = sorted(glob.glob(f"{META}/shard_*.jsonl"))
    assert files, "no metadata shards yet"
    rows = [json.loads(l) for f in files for l in open(f)]
    df = pd.DataFrame(rows)
    return df, len(files)


def learn_verb_map(meta):
    """Compare raw instructions of MVP episodes with MVP's correct option text -> words MVP rewrote to "move"."""
    truth = {}
    for l in open(MVP_EVAL):
        d = json.loads(l)
        if d.get("extra", {}).get("source") == "language_table":
            gid = int(d["extra"]["mvp_video_path"].rsplit("video_", 1)[1][:-4])
            truth[gid] = d["options"][d["answer_index"]]
    m = meta[meta.mvp & meta.gid.isin(truth.keys())]
    rew, kept, other = collections.Counter(), collections.Counter(), 0
    for gid, inst in zip(m.gid, m.inst):
        a, b = norm_basic(inst).split(), norm_basic(truth[gid]).split()
        if len(a) != len(b) or any(x != y and y != "move" for x, y in zip(a, b)):
            other += 1
            continue
        for x, y in zip(a, b):
            (rew if x != y else kept)[x] += 1
    return truth, m, rew, kept, other


def build_index(texts):
    """word-level symmetric-deletion index (0/1 deletions) over unique token tuples."""
    idx = collections.defaultdict(list)
    for i, t in enumerate(texts):
        idx[t].append(i)
        for k in range(len(t)):
            idx[t[:k] + t[k + 1:]].append(i)
    return idx


def build_index2(texts):
    idx = collections.defaultdict(list)
    for i, t in enumerate(texts):
        if len(t) < 4:
            continue
        seen = set()
        for a in range(len(t)):
            for b in range(a + 1, len(t)):
                v = t[:a] + t[a + 1:b] + t[b + 1:]
                if v not in seen:
                    seen.add(v)
                    idx[v].append(i)
    return idx


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def char_sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def judge(q, c):
    """-> (word edit distance, preference 0..3) ; pref 0 = not an acceptable negative."""
    d = lev(q, c)
    if d == 0:
        return d, 0
    ops = [o for o in difflib.SequenceMatcher(None, q, c, autojunk=False).get_opcodes() if o[0] != "equal"]
    changed = [w for o in ops for w in q[o[1]:o[2]] + c[o[3]:o[4]]]
    content = [w for w in changed if w not in STOP]
    if not content:
        return d, 0
    subs = [(q[o[1]:o[2]], c[o[3]:o[4]]) for o in ops if o[0] == "replace"]
    if len(ops) == 1 and subs and len(subs[0][0]) == 1 and len(subs[0][1]) == 1:
        x, y = subs[0][0][0], subs[0][1][0]
        if x in DIRS and y in DIRS and GROUP_OF[x] == GROUP_OF[y]:
            return d, 0                                   # synonym direction swap
        if not (x in SALIENT and y in SALIENT) and char_sim(x, y) >= 0.75:
            return d, 0                                   # typo / inflection
        same_cat = (x in COLOURS and y in COLOURS) or (x in SHAPES and y in SHAPES) or (x in DIRS and y in DIRS)
        if same_cat:
            return d, 3
    if any(w in SALIENT for w in content):
        return d, 2
    return d, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--max-episodes", type=int, default=MAX_EPISODES)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    mvp = load_mvp_ids()
    meta, nshards = load_meta()
    print(f"meta shards {nshards}/500 | episodes in meta {len(meta)} | MVP ids {len(mvp)}")
    assert (meta.mvp == meta.gid.isin(mvp)).all(), "meta mvp flag disagrees with the allowlist"

    truth, m, rew, kept, other = learn_verb_map(meta)
    learned = {v for v, n in rew.items() if n >= 3 and kept.get(v, 0) == 0}
    print(f"MVP comparison on {len(m)} MVP episodes: word-aligned {len(m) - other} | other {other}")
    print("  words rewritten to 'move':", dict(rew.most_common()), "| kept verbs:",
          {w: kept[w] for w in ("move", "nudge", "put", "drag", "keep", "separate", "adjust", "shift", "pull")})
    print("  learned verb set", sorted(learned), "| VERB_TO_MOVE", sorted(VERB_TO_MOVE))
    assert learned == VERB_TO_MOVE, "MVP verb rewrite differs from VERB_TO_MOVE"
    agree = np.mean([mvp_norm(i) == norm_basic(truth[g]) for g, i in zip(m.gid, m.inst)]) if len(m) else float("nan")
    print(f"  mvp_norm(raw) == MVP option for {agree:.4f} of MVP episodes")
    assert agree > 0.99, "text normalisation does not reproduce MVP options"
    bad = [(i, truth[g]) for g, i in zip(m.gid, m.inst) if mvp_norm(i) != norm_basic(truth[g])]
    print("  mismatches (raw | MVP):", bad[:8])

    meta["text"] = meta.inst.map(mvp_norm)
    meta["key"] = meta.text.map(key_of)
    elig = (~meta.mvp) & (meta.n >= MIN_FRAMES) & (meta.key.str.len() > 0)
    sel = meta[elig & meta.sel].copy()
    sel["prep_exists"] = [os.path.exists(f"{PREP}/{g}.mp4") for g in sel.gid]
    print(f"eligible non-MVP {int(elig.sum())} | encoded-selected {len(sel)} | prep_exists {int(sel.prep_exists.sum())}")
    sel = sel[sel.prep_exists].sort_values("u").head(a.max_episodes).sort_values("gid").reset_index(drop=True)
    assert not sel.gid.isin(mvp).any()

    # negative pool: instructions of ALL non-MVP episodes with a non-empty instruction (any length)
    pool_df = meta[(~meta.mvp) & (meta.key.str.len() > 0)]
    utext_of_key = {}
    for k, t in zip(pool_df.key, pool_df.text):
        utext_of_key.setdefault(k, t)
    for k, t in zip(sel.key, sel.text):
        utext_of_key.setdefault(k, t)
    ukeys = sorted(utext_of_key)
    kid = {k: i for i, k in enumerate(ukeys)}
    utoks = [tuple(k.split()) for k in ukeys]
    pool_mask = np.zeros(len(ukeys), bool)
    pool_mask[[kid[k] for k in set(pool_df.key)]] = True
    need = sorted({kid[k] for k in sel.key})
    print(f"unique instruction keys: pool {int(pool_mask.sum())} | needed {len(need)}")
    b, stats = search_negatives([utoks[i] for i in need], utoks, pool_mask, a.workers)
    best = {need[qi]: v for qi, v in b.items()}
    print("negative search:", dict(stats))

    rows = []
    rng = np.random.default_rng(0)
    ab = collections.Counter()
    for r in sel.itertuples():
        video = f"{r.gid}.mp4"
        q = kid[r.key]
        if q not in best:
            continue
        cands = best[q]
        nid, nd = cands[rng.integers(len(cands))]
        neg = utext_of_key[ukeys[nid]]
        assert key_of(neg) != r.key
        opts = [r.text, neg]
        if rng.random() < 0.5:
            opts = opts[::-1]
        letter = LETTERS[opts.index(r.text)]
        ab[letter] += 1
        prompt = f"{Q_MC}\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)) + f"\n{SUFFIX}"
        rows.append({"video": video, "prep_exists": True, "kind": "mc", "episode": int(r.gid),
                     "conversations": json.dumps([{"role": "user", "content": f"{VIS}\n{prompt}"},
                                                  {"role": "assistant", "content": f"{letter}."}]),
                     "neg_dist": int(nd), "neg_text": neg})
        if rng.random() < OPEN_FRAC:
            ans = r.text[0].upper() + r.text[1:]
            ans = ans if ans.endswith(".") else ans + "."
            rows.append({"video": video, "prep_exists": True, "kind": "open", "episode": int(r.gid),
                         "conversations": json.dumps([{"role": "user", "content": f"{VIS}\n{Q_OPEN}"},
                                                      {"role": "assistant", "content": ans}]),
                         "neg_dist": -1, "neg_text": ""})
    df = pd.DataFrame(rows)
    df["prep_exists"] = df.video.map(lambda v: os.path.exists(f"{PREP}/{v}"))
    assert not df.episode.isin(mvp).any(), "MVP episode leaked into the manifest"
    held = df.episode.map(lambda e: int(hashlib.md5(f"lt:{e}".encode()).hexdigest()[:8], 16) % 100 < HELDOUT_PCT)
    os.makedirs(os.path.dirname(a.dest), exist_ok=True)
    df[~held].reset_index(drop=True).to_parquet(a.dest, index=False)
    df[held].reset_index(drop=True).to_parquet(a.dest.replace(".parquet", "_heldout.parquet"), index=False)

    print(f"episodes {df.episode.nunique()} | rows {len(df)} (train {int((~held).sum())}, heldout {int(held.sum())},"
          f" heldout episodes {df[held].episode.nunique()}) | prep_exists {int(df.prep_exists.sum())}")
    print("kinds", df.kind.value_counts().to_dict(), "| true option at", dict(ab))
    print("negative edit distance:", df[df.kind == "mc"].neg_dist.value_counts().sort_index().to_dict())
    print("MVP ids in manifest:", int(df.episode.isin(mvp).sum()))
    ex = pd.concat([df[df.kind == "mc"].sample(2, random_state=1), df[df.kind == "open"].sample(1, random_state=1)])
    for _, x in ex.iterrows():
        print(json.dumps({"video": x.video, "episode": int(x.episode), "kind": x.kind,
                          "conversations": json.loads(x.conversations)}, ensure_ascii=False))


_S = {}


def _score(q, qkey, js, out):
    pool, keys = _S["pool"], _S["keys"]
    for j in js:
        if keys[j] == qkey:
            continue
        d, p = judge(q, pool[j])
        if p:
            out.append((d + (0 if p >= 2 else 1), -p, d, int(j)))


def _search_chunk(chunk):
    idx1, idx2, by_len, pool_ids = _S["idx1"], _S["idx2"], _S["by_len"], _S["pool_ids"]
    best, stats = {}, collections.Counter()
    for qi, q in chunk:
        rng = np.random.default_rng(qi)
        qkey = " ".join(q)
        cand = set(idx1.get(q, ()))
        for k in range(len(q)):
            cand.update(idx1.get(q[:k] + q[k + 1:], ()))
        scored = []
        _score(q, qkey, cand, scored)
        if not any(s[0] <= 2 for s in scored):
            stats["deep2"] += 1
            c2 = set()
            for a in range(len(q)):
                for b in range(a + 1, len(q)):
                    c2.update(idx2.get(q[:a] + q[a + 1:b] + q[b + 1:], ()))
            _score(q, qkey, c2 - cand, scored)
        if not scored:
            stats["fallback"] += 1
            cands = [j for L in range(len(q) - 2, len(q) + 3) for j in by_len.get(L, ())]
            if cands:
                _score(q, qkey, rng.choice(cands, size=min(3000, len(cands)), replace=False), scored)
        if not scored:
            stats["none"] += 1
            continue
        scored.sort()
        top = [s for s in scored if s[:2] == scored[0][:2]][:16]
        best[qi] = [(int(pool_ids[s[3]]), s[2]) for s in top]
    return best, stats


def search_negatives(queries, utexts, pool_mask, workers):
    """for each query token tuple -> best candidates [(index into utexts, word distance)] (ties kept, <= 16).
    Ranking key: (distance + 1 if the change touches no direction/colour/shape word, -preference).
    Candidates: 0/1-deletion index (all pool texts within distance <= 2); if nothing acceptable with key <= 2, the
    2-deletion index (adds distance 3-4 neighbours); if still none, a random same-length-ish sample of 3000."""
    pool_ids = np.flatnonzero(pool_mask)
    pool = [utexts[i] for i in pool_ids]
    by_len = collections.defaultdict(list)
    for j, t in enumerate(pool):
        by_len[len(t)].append(j)
    _S.update(pool=pool, keys=[" ".join(t) for t in pool], pool_ids=pool_ids, by_len=by_len,
              idx1=build_index(pool), idx2=build_index2(pool))
    items = list(enumerate(queries))
    chunks = [items[i::workers * 8] for i in range(workers * 8)]
    best, stats = {}, collections.Counter()
    with mp.get_context("fork").Pool(workers) as pool_:
        for b, st in pool_.imap_unordered(_search_chunk, chunks):
            best.update(b)
            stats.update(st)
    return best, stats


if __name__ == "__main__":
    main()
