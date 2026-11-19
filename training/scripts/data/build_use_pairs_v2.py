"""Real-video pairs for L_use (v2), built the way MVP (Minimal Video Pairs) builds its evaluation pairs, TRAIN splits only.

A pair = two DIFFERENT videos, the SAME user prompt (byte-identical: question + the same options in the same order +
suffix) and DIFFERENT correct answers. Options are the two videos' own answers (MVP style, 2 options).

  use_pairs_ssv2.parquet     SSv2 TRAIN. Two videos whose templates are a minimal contrast. Template pairs: tokens of
                             the lowercased template ("[something]" -> "something", other placeholders unbracketed);
                             kept if word-level edit distance <= 2 OR the content-word set difference is <= 2 words on
                             each side (catches 'pulling .. from left to right' / 'pushing .. from right to left');
                             dropped if one template's word set contains the other's (ambiguous: 'holding something' vs
                             'holding something over something'). Tier 1 = contrast pairs (only spatial / outcome words
                             or an antonym verb change, e.g. opening/closing, up/down, into/onto, without/then, plus the
                             template contrasts MVP itself evaluates - labels only, never MVP videos); tier 2 = the other
                             1-2 word changes (another verb in the same frame). Question "Which action is being performed
                             in the video?" (MVP). Each video <= 2 uses, pairs spread evenly over template pairs.
  use_pairs_lt.parquet       Language-Table non-MVP, non-heldout episodes of stage1/language_table.parquet (instruction =
                             correct option of the episode's mc row, MVP-normalised). Two episodes whose instructions
                             differ by 1-2 word SUBSTITUTIONS, every substituted word a colour<->colour, shape<->shape or
                             direction<->direction (different direction group) change, or a swap of two salient words
                             ('red circle into red star' / 'red star into red circle'). Question "Which robot instruction
                             best describes the actions in the video?" (MVP). Each episode <= 2 uses, <= CAP per text pair.
  use_pairs_clevrer.parquet  CLEVRER TRAIN (video_00000-09999; MVP evaluates on validation). desc: the SAME descriptive
                             question text on two videos with DIFFERENT answers, options = the two answers (['3', '2']).
                             choice: explanatory / predictive / counterfactual question + choice text identical in two
                             videos with opposite correctness -> "<question> Is this one of them: <choice>?" ['yes','no'].
                             Each video <= 7 uses (mean ~6), pairs spread over (question, answer pair) groups.

Allocation: round robin over groups (one pair per group per round, deterministic seed), a video picked = the least used
eligible one (random tie-break). Held-out: pass 1 builds all pairs; pairs with md5(pair_id) % 100 < 2 are held out;
pass 2 rebuilds the train pairs with every held-out video removed, so held-out videos never appear in train.
Outputs: manifests/stage1/use_pairs_<src>.parquet + _heldout.parquet (video, prep_exists, pair_id, conversations, kind,
gkey, answer_text) and manifests/eval/heldout/use_pairs_<src>_heldout.jsonl (group_id = pair_id, for pair accuracy).
"""
import argparse
import collections
import difflib
import glob
import hashlib
import importlib.util
import itertools
import json
import os
import re

import numpy as np
import pandas as pd

R = "/data/llama_vjepa_lm"
M = f"{R}/manifests"
VIS = "<|jepa_evidence|>"
SUFFIX = "Answer with the option's letter from the given choices directly."
LETTERS = "AB"
HELD_PCT = 2
SEED = 0

SSV2_LABELS = f"{R}/data/ssv2_labels/labels/train.json"
SSV2_ROOT = f"{R}/bench/mvp_sources/extracted/ssv2/20bn-something-something-v2"
SSV2_Q = "Which action is being performed in the video?"
LT_ROOT = f"{R}/prep/language_table_train"
LT_Q = "Which robot instruction best describes the actions in the video?"
CL_D = f"{R}/data/clevrer"
CL_ROOT = f"{CL_D}/video_train"

_spec = importlib.util.spec_from_file_location("lt", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                  "build_stage1_language_table.py"))
LT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LT)


# ----------------------------------------------------------------------------------------------------------- common
def md5i(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def is_held(pid):
    return md5i(f"use2:{pid}") % 100 < HELD_PCT


def prompt(question, opts):
    return f"{question}\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)) + f"\n{SUFFIX}"


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def load_benchmark_ids():
    ids = collections.defaultdict(set)
    for f in sorted(glob.glob(f"{M}/allowlist/benchmark_ids_*.txt")):
        for line in open(f):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                ids[p[0]].add(p[1])
    return ids


def allocate(groups, target, usage, vmax, rng, fam_cap=None, fam_count=None, seen=None):
    """groups: list of dicts {gkey, fam, A (int array), B (int array), cap}. Round robin: every round each active group
    adds one pair (least-used eligible video on each side, random tie-break) until its cap, a side runs out, its family
    reaches fam_cap, or the target is met. usage (int array over all videos) is updated in place."""
    fam_count = fam_count if fam_count is not None else collections.Counter()
    seen = seen if seen is not None else set()
    out, cnt = [], np.zeros(len(groups), int)
    active = list(range(len(groups)))

    def pick(pool, avoid=()):
        u = usage[pool]
        ok = u < vmax
        if avoid:
            ok &= ~np.isin(pool, list(avoid))
        if not ok.any():
            return None
        cand = pool[ok & (u == u[ok].min())]
        return int(cand[rng.integers(len(cand))])

    while active and len(out) < target:
        nxt = []
        for gi in rng.permutation(len(active)):
            k = active[gi]
            g = groups[k]
            if len(out) >= target:
                break
            if cnt[k] >= g["cap"] or (fam_cap and fam_count[g["fam"]] >= fam_cap):
                continue
            a = pick(g["A"])
            if a is None:
                continue
            avoid, b = {a}, None
            for _ in range(4):                     # never repeat the same two videos for the same group
                b = pick(g["B"], avoid)
                if b is None or (min(a, b), max(a, b), g["gkey"]) not in seen:
                    break
                avoid.add(b)
                b = None
            if b is None:
                continue
            seen.add((min(a, b), max(a, b), g["gkey"]))
            usage[a] += 1
            usage[b] += 1
            cnt[k] += 1
            fam_count[g["fam"]] += 1
            out.append((k, a, b))
            nxt.append(k)
        active = sorted(nxt)
    return out


def two_pass(build, name):
    """build(excluded_video_set) -> list of pair dicts. Held-out = pass-1 pairs by pair hash; train = pass 2 without
    any held-out video."""
    p1 = build(set())
    held = [p for p in p1 if is_held(p["pair_id"])]
    hv = {v for p in held for v in p["videos"]}
    train = build(hv)
    assert not hv & {v for p in train for v in p["videos"]}
    print(f"[{name}] pass1 {len(p1)} pairs | heldout {len(held)} pairs ({len(hv)} videos) | pass2 train {len(train)}")
    return train, held


def to_rows(pairs, rng, root):
    rows = []
    for p in pairs:
        opts = list(p["answers"])
        if rng.random() < 0.5:
            opts = opts[::-1]
        u = prompt(p["question"], opts)
        for v, ans in zip(p["videos"], p["answers"]):
            rows.append({"video": v, "prep_exists": os.path.exists(f"{root}/{v}"), "pair_id": p["pair_id"],
                         "conversations": json.dumps([{"role": "user", "content": f"{VIS}\n{u}"},
                                                      {"role": "assistant", "content": LETTERS[opts.index(ans)] + "."}]),
                         "kind": p["kind"], "gkey": p["gkey"], "answer_text": ans, "question": p["question"],
                         "options": json.dumps(opts)})
    return pd.DataFrame(rows)


def write(name, train_df, held_df, root):
    cols = ["video", "prep_exists", "pair_id", "conversations", "kind", "gkey", "answer_text"]
    train_df[cols].to_parquet(f"{M}/stage1/{name}.parquet", index=False)
    held_df[cols].to_parquet(f"{M}/stage1/{name}_heldout.parquet", index=False)
    bench = f"{name}_heldout"
    os.makedirs(f"{M}/eval/heldout", exist_ok=True)
    with open(f"{M}/eval/heldout/{bench}.jsonl", "w") as f:
        for i, r in enumerate(held_df.itertuples()):
            opts = json.loads(r.options)
            f.write(json.dumps({"bench": bench, "split": "heldout", "id": f"{bench}/{i}", "video": f"{root}/{r.video}",
                                "video_exists": bool(r.prep_exists), "start_sec": None, "end_sec": None,
                                "cut_sec": None, "question": r.question, "options": opts,
                                "answer_index": opts.index(r.answer_text), "answer_text": r.answer_text,
                                "group_id": r.pair_id, "extra": {"source": name, "kind": r.kind, "gkey": r.gkey,
                                                                 "pair_id": r.pair_id, "rel_video": r.video}}) + "\n")


def report(name, df, held_df, id_of_video, family, bench_ids):
    def users(d):
        return d.conversations.map(lambda c: json.loads(c)[0]["content"])

    def ans(d):
        return d.conversations.map(lambda c: json.loads(c)[1]["content"])
    for tag, d in (("train", df), ("heldout", held_df)):
        g = d.assign(u=users(d), a=ans(d)).groupby("pair_id")
        size2 = (g.size() == 2).mean()
        same_u = (g.u.nunique() == 1).mean()
        diff_a = (g.a.nunique() == 2).mean()
        diff_v = (g.video.nunique() == 2).mean()
        print(f"[{name}/{tag}] pairs {d.pair_id.nunique()} rows {len(d)} videos {d.video.nunique()} | prep_exists "
              f"{d.prep_exists.mean():.4f} | 2 rows {size2:.4f} same prompt {same_u:.4f} answers differ {diff_a:.4f} "
              f"videos differ {diff_v:.4f} | letters {ans(d).value_counts().to_dict()}")
    use = df.video.value_counts()
    print(f"[{name}] uses per video: max {use.max()} mean {use.mean():.2f} dist {use.value_counts().sort_index().to_dict()}")
    print(f"[{name}] kinds (pairs) {df.drop_duplicates('pair_id').kind.value_counts().to_dict()}")
    gk = df.drop_duplicates("pair_id").gkey.value_counts()
    print(f"[{name}] groups {len(gk)} | pairs per group max {gk.max()} median {gk.median():.0f} | top5 {gk.head(5).to_dict()}")
    print(f"[{name}] answer_text top12 {df.answer_text.value_counts().head(12).to_dict()}")
    ids = pd.concat([df.video, held_df.video]).map(id_of_video)
    own = int(ids.isin(bench_ids.get(family, set())).sum())
    print(f"[{name}] benchmark-id check (all benchmark_ids_*.txt, family '{family}', "
          f"{len(bench_ids.get(family, ()))} ids): hits {own} | held/train video overlap "
          f"{len(set(df.video) & set(held_df.video))}")
    assert own == 0, "benchmark video leaked"
    rng = np.random.default_rng(1)
    for pid in rng.choice(df.pair_id.unique(), 3, replace=False):
        for r in df[df.pair_id == pid].itertuples():
            print(json.dumps({"pair_id": pid, "video": r.video, "kind": r.kind,
                              "conversations": json.loads(r.conversations)}, ensure_ascii=False))


# -------------------------------------------------------------------------------------------------------------- SSv2
SP = {"left", "right", "up", "down", "upwards", "downwards", "into", "onto", "out", "of", "off", "behind", "front",
      "in", "next", "to", "over", "under", "underneath", "towards", "toward", "away", "closer", "from", "on", "across",
      "along", "top", "bottom", "beneath", "inside", "each", "other", "the", "camera", "with", "upside", "upright",
      "side"}
OUTCOME = {"without", "then", "until", "so", "that", "but", "not", "slightly", "almost", "nothing", "falls", "fall",
           "stops", "continues", "collide", "pass", "halt", "deflected", "moves", "doesn't", "it", "stack", "collapses",
           "collapsing", "rolls", "roll", "slides", "slide", "letting", "drop", "falling", "quickly", "spinning",
           "starts", "sliding"}
ANTONYM = [{"opening", "closing"}, {"folding", "unfolding"}, {"covering", "uncovering"}, {"pushing", "pulling"},
           {"putting", "taking"}, {"putting", "removing"}, {"dropping", "picking"}, {"dropping", "putting"},
           {"plugging", "pulling"}, {"lifting", "lowering"}, {"attaching", "detaching"}, {"pouring", "spilling"},
           {"approaching", "moving"}, {"showing", "holding"}, {"burying", "digging"}, {"put", "take"}, {"pick", "put"},
           {"pushing", "throwing"}, {"rolling", "sliding"}]


def ssv2_option(t):
    t = re.sub(r"\[something\]", "something", t, flags=re.I)
    t = re.sub(r"\[([^\]]*)\]", r"\1", t)
    return " ".join(t.lower().split())


def ssv2_template_pairs(templates):
    toks = {t: re.findall(r"[a-z']+", ssv2_option(t)) for t in templates}
    mvp_tp = set()
    for line in open(f"{M}/eval/mvp_full.jsonl"):
        x = json.loads(line)
        if x["extra"].get("source") == "ssv2":
            mvp_tp.add(tuple(sorted(x["options"])))
    opt_to_t = {ssv2_option(t): t for t in templates}
    out = {}
    for a, b in itertools.combinations(sorted(templates), 2):
        ta, tb = toks[a], toks[b]
        if ssv2_option(a) == ssv2_option(b):
            continue
        A, B = set(ta) - set(tb) - {"something"}, set(tb) - set(ta) - {"something"}
        if bool(A) != bool(B):                                   # one word set contains the other: ambiguous
            continue
        d = lev(ta, tb)
        if not (d <= 2 or max(len(A), len(B)) <= 2):
            continue
        tier = 1
        for op in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
            if op[0] == "equal":
                continue
            x, y = ta[op[1]:op[2]], tb[op[3]:op[4]]
            if set(x) | set(y) <= SP | OUTCOME:
                continue
            if len(x) == 1 and len(y) == 1 and {x[0], y[0]} in ANTONYM:
                continue
            tier = 2
        if tuple(sorted([ssv2_option(a), ssv2_option(b)])) in mvp_tp:
            tier = 1
        out[(a, b)] = tier
    n_in = len(out)
    for o1, o2 in mvp_tp:                                       # MVP template contrasts outside the 1-2 word rule
        if o1 in opt_to_t and o2 in opt_to_t:
            k = tuple(sorted([opt_to_t[o1], opt_to_t[o2]]))
            out.setdefault(k, 0)
    print(f"[ssv2] template pairs: 1-2 word rule {n_in} (tier1 {sum(v == 1 for v in out.values())}, tier2 "
          f"{sum(v == 2 for v in out.values())}) + MVP template contrasts outside the rule {sum(v == 0 for v in out.values())}"
          f" | MVP template contrasts {len(mvp_tp)}")
    return out


def build_ssv2(a, bench_ids):
    train = json.load(open(SSV2_LABELS))
    mvp = bench_ids["ssv2"]
    assert len(mvp) > 1000, "MVP ssv2 ids not read"
    on_disk = set(os.listdir(SSV2_ROOT))
    by_t = collections.defaultdict(list)
    excl = miss = 0
    for x in train:
        if x["id"] in mvp:
            excl += 1
            continue
        if f"{x['id']}.webm" not in on_disk:
            miss += 1
            continue
        by_t[x["template"]].append(f"{x['id']}.webm")
    print(f"[ssv2] train {len(train)} | excluded MVP {excl} | missing {miss} | templates {len(by_t)}")
    tpairs = ssv2_template_pairs(list(by_t))
    vids = sorted({v for vs in by_t.values() for v in vs})
    vid = {v: i for i, v in enumerate(vids)}

    def build(excluded):
        rng = np.random.default_rng(SEED)
        usage = np.zeros(len(vids), int)
        seen = set()
        pools = {t: np.array(sorted(vid[v] for v in vs if v not in excluded)) for t, vs in by_t.items()}
        allp = []
        for tiers, target, cap in (((0, 1), a.ssv2_t1, a.ssv2_cap1), ((2,), a.ssv2_target, a.ssv2_cap2)):
            gs = [{"gkey": f"{ssv2_option(x)} || {ssv2_option(y)}", "fam": x, "A": pools[x], "B": pools[y], "cap": cap,
                   "tier": t} for (x, y), t in sorted(tpairs.items()) if t in tiers]
            got = allocate(gs, target - len(allp), usage, 2, rng, seen=seen)
            for gi, i, j in got:
                g = gs[gi]
                (x, y) = g["gkey"].split(" || ")
                allp.append({"pair_id": f"ssv2__{vids[i][:-5]}__{vids[j][:-5]}", "videos": (vids[i], vids[j]),
                             "answers": (x, y), "question": SSV2_Q, "kind": f"tier{max(g['tier'], 1)}"
                             + ("_mvptemplate" if g["tier"] == 0 else ""), "gkey": g["gkey"]})
        return allp
    return build, lambda v: v[:-5], "ssv2", SSV2_ROOT


# ---------------------------------------------------------------------------------------------------- Language-Table
LT_DIR_OK = {0, 1, 2, 3, 4, 9, 10}           # up / down / left / right / center / near / away groups
LT_SHAPES = LT.SHAPES - {"block", "blocks"}


def lt_category(w):
    if w in LT.COLOURS:
        return "colour"
    if w in LT_SHAPES:
        return "shape"
    if w in LT.DIRS and LT.GROUP_OF[w] in LT_DIR_OK:
        return "direction"
    return None


def lt_contrast(q, c):
    """-> kind if q and c differ by 1-2 word substitutions that are all salient same-category changes or a swap."""
    if len(q) != len(c):
        return None
    diff = [(x, y) for x, y in zip(q, c) if x != y]
    if not 1 <= len(diff) <= 2:
        return None
    if len(diff) == 2 and diff[0] == diff[1][::-1] and lt_category(diff[0][0]) and lt_category(diff[0][1]):
        return "swap:2w"
    kinds = []
    for x, y in diff:
        cx, cy = lt_category(x), lt_category(y)
        if cx is None or cx != cy or x.rstrip("s") == y.rstrip("s"):
            return None
        if cx == "direction" and LT.GROUP_OF[x] == LT.GROUP_OF[y]:
            return None
        kinds.append(cx)
    return "+".join(sorted(set(kinds))) + f":{len(diff)}w"


def build_lt(a, bench_ids):
    d = pd.read_parquet(f"{M}/stage1/language_table.parquet", columns=["video", "kind", "episode", "conversations"])
    h = set(pd.read_parquet(f"{M}/stage1/language_table_heldout.parquet", columns=["episode"]).episode)
    mvp = {int(x.replace("video_", "")) for x in bench_ids["language_table"]}
    assert len(mvp) > 20000
    m = d[d.kind == "mc"]

    def true_text(c):
        c = json.loads(c)
        letter = c[1]["content"][0]
        return next(line[3:] for line in c[0]["content"].split("\n") if line.startswith(letter + ". "))
    m = m.assign(text=m.conversations.map(true_text))
    n0 = len(m)
    m = m[~m.episode.isin(h) & ~m.episode.isin(mvp)]
    m = m[[os.path.exists(f"{LT_ROOT}/{v}") for v in m.video]]
    m = m.assign(key=m.text.map(LT.key_of))
    print(f"[lt] mc episodes {n0} | after heldout / MVP / prep filter {len(m)} | unique keys {m.key.nunique()}")
    text_of = {}
    for k, t in zip(m.key, m.text):
        text_of.setdefault(k, t)
    keys = sorted(text_of)
    toks = [tuple(k.split()) for k in keys]
    # substitution-only neighbours: same length, share a variant with the same 1 or 2 positions masked
    idx = collections.defaultdict(list)
    for i, t in enumerate(toks):
        for p in range(len(t)):
            idx[t[:p] + ("*",) + t[p + 1:]].append(i)
        for p, q in itertools.combinations(range(len(t)), 2):
            if t[p] in LT.SALIENT and t[q] in LT.SALIENT:
                idx[t[:p] + ("*",) + t[p + 1:q] + ("*",) + t[q + 1:]].append(i)
    cand = set()
    for v in idx.values():
        if 1 < len(v) <= 2000:
            cand.update(itertools.combinations(sorted(v), 2))
    tp = {}
    for i, j in cand:
        k = lt_contrast(toks[i], toks[j])
        if k:
            tp[(i, j)] = k
    print(f"[lt] candidate key pairs {len(cand)} | accepted text pairs {len(tp)} | kinds "
          f"{dict(collections.Counter(tp.values()))}")
    eps = sorted(m.video)
    vid = {v: n for n, v in enumerate(eps)}
    by_key = collections.defaultdict(list)
    for k, v in zip(m.key, m.video):
        by_key[k].append(vid[v])

    def build(excluded):
        rng = np.random.default_rng(SEED)
        usage = np.zeros(len(eps), int)
        ex = {vid[v] for v in excluded}
        pools = {k: np.array(sorted(set(v) - ex)) for k, v in by_key.items()}
        gs = [{"gkey": f"{keys[i]} || {keys[j]}", "fam": keys[i], "A": pools[keys[i]], "B": pools[keys[j]],
               "cap": a.lt_cap, "kind": k, "ti": (text_of[keys[i]], text_of[keys[j]])} for (i, j), k in sorted(tp.items())]
        gs = [g for g in gs if len(g["A"]) and len(g["B"])]
        # phase 1: one-word changes and swaps (most minimal); phase 2: two-word changes (e.g. a new target object)
        g1 = [g for g in gs if g["kind"].endswith(":1w") or g["kind"].startswith("swap")]
        g2 = [g for g in gs if not (g["kind"].endswith(":1w") or g["kind"].startswith("swap"))]
        seen = set()
        got = allocate(g1, a.lt_t1, usage, 2, rng, seen=seen)
        got = [(g1[gi], i, j) for gi, i, j in got]
        got += [(g2[gi], i, j) for gi, i, j in allocate(g2, a.lt_target - len(got), usage, 2, rng, seen=seen)]
        gs = [g for g, _, _ in got]
        got = [(n, i, j) for n, (_, i, j) in enumerate(got)]
        return [{"pair_id": f"lt__{eps[i][:-4]}__{eps[j][:-4]}", "videos": (eps[i], eps[j]), "answers": gs[gi]["ti"],
                 "question": LT_Q, "kind": gs[gi]["kind"], "gkey": gs[gi]["gkey"]} for gi, i, j in got]
    return build, lambda v: f"video_{v[:-4]}", "language_table", LT_ROOT


# ----------------------------------------------------------------------------------------------------------- CLEVRER
def build_clevrer(a, bench_ids):
    paths = {os.path.basename(p): os.path.relpath(p, CL_ROOT) for p in glob.glob(f"{CL_ROOT}/**/*.mp4", recursive=True)}
    mvp = bench_ids["clevrer"]
    assert mvp
    desc = collections.defaultdict(lambda: collections.defaultdict(set))
    ch = collections.defaultdict(lambda: collections.defaultdict(set))
    excl = miss = 0
    for v in json.load(open(f"{CL_D}/train.json")):
        fn = v["video_filename"]
        if fn[:-4] in mvp:
            excl += 1
            continue
        if fn not in paths:
            miss += 1
            continue
        p = paths[fn]
        for x in v["questions"]:
            if x["question_type"] == "descriptive":
                desc[x["question"]][x["answer"]].add(p)
            else:
                for c in x["choices"]:
                    ch[(x["question_type"], x["question"], c["choice"])][c["answer"]].add(p)
    vids = sorted(paths[f] for f in paths if f[:-4] not in mvp)
    vid = {v: i for i, v in enumerate(vids)}
    # a video that answers the same text differently in two questions is ambiguous for that text: drop it there
    for d in list(desc.values()) + list(ch.values()):
        dup = set.union(*[set()] + [s1 & s2 for (k1, s1), (k2, s2) in itertools.combinations(d.items(), 2)])
        for k in d:
            d[k] -= dup
    print(f"[clevrer] excluded MVP {excl} | missing {miss} | videos {len(vids)} | desc texts {len(desc)} | choice "
          f"items with both labels {sum(len(d) == 2 for d in ch.values())}")

    def build(excluded):
        rng = np.random.default_rng(SEED)
        usage = np.zeros(len(vids), int)
        seen = set()
        ex = {vid[v] for v in excluded}

        def pool(s):
            return np.array(sorted({vid[v] for v in s} - ex))
        allp = []
        gs = []
        for (qt, q, c), d in sorted(ch.items()):
            if len(d) == 2:
                qq = f"{q} Is this one of them: {c[0].lower() + c[1:]}?"
                gs.append({"gkey": qq, "fam": qq, "A": pool(d["correct"]), "B": pool(d["wrong"]), "cap": a.cl_choice_cap,
                           "kind": f"choice_{qt}", "q": qq, "ans": ("yes", "no")})
        gs = [g for g in gs if len(g["A"]) and len(g["B"])]
        got = allocate(gs, a.cl_choice_target, usage, a.cl_vmax, rng, seen=seen)
        n_choice = len(got)
        gs2 = []
        for q, d in sorted(desc.items()):
            for x, y in itertools.combinations(sorted(d), 2):
                gs2.append({"gkey": f"{q} || {x} | {y}", "fam": q, "A": pool(d[x]), "B": pool(d[y]),
                            "cap": a.cl_desc_cap, "kind": "desc", "q": q, "ans": (x, y)})
        gs2 = [g for g in gs2 if len(g["A"]) and len(g["B"])]
        got += [(gi + len(gs), i, j) for gi, i, j in allocate(gs2, a.cl_target - n_choice, usage, a.cl_vmax, rng,
                                                               fam_cap=a.cl_fam_cap, seen=seen)]
        gs = gs + gs2
        for gi, i, j in got:
            g = gs[gi]
            h = hashlib.md5(g["q"].encode()).hexdigest()[:8]
            allp.append({"pair_id": f"clevrer__{os.path.basename(vids[i])[:-4]}__{os.path.basename(vids[j])[:-4]}__{h}",
                         "videos": (vids[i], vids[j]), "answers": g["ans"], "question": g["q"], "kind": g["kind"],
                         "gkey": g["gkey"]})
        return allp
    return build, lambda v: os.path.basename(v)[:-4], "clevrer", CL_ROOT


# -------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["ssv2", "lt", "clevrer"])
    ap.add_argument("--ssv2-target", type=int, default=40_000)
    ap.add_argument("--ssv2-t1", type=int, default=28_000, help="pairs from tier-1 (contrast) template pairs")
    ap.add_argument("--ssv2-cap1", type=int, default=400)
    ap.add_argument("--ssv2-cap2", type=int, default=40)
    ap.add_argument("--lt-target", type=int, default=40_000)
    ap.add_argument("--lt-cap", type=int, default=6)
    ap.add_argument("--lt-t1", type=int, default=24_000, help="pairs from one-word changes / swaps")
    ap.add_argument("--cl-target", type=int, default=30_000)
    ap.add_argument("--cl-choice-target", type=int, default=6_000)
    ap.add_argument("--cl-choice-cap", type=int, default=3)
    ap.add_argument("--cl-desc-cap", type=int, default=40)
    ap.add_argument("--cl-fam-cap", type=int, default=150)
    ap.add_argument("--cl-vmax", type=int, default=7)
    a = ap.parse_args()
    bench_ids = load_benchmark_ids()
    print("benchmark id families:", {k: len(v) for k, v in sorted(bench_ids.items())})
    for src, fn in (("ssv2", build_ssv2), ("lt", build_lt), ("clevrer", build_clevrer)):
        if src not in a.only:
            continue
        build, id_of, family, root = fn(a, bench_ids)
        train, held = two_pass(build, src)
        name = f"use_pairs_{src}"
        rng = np.random.default_rng(SEED + 1)
        tdf, hdf = to_rows(train, rng, root), to_rows(held, rng, root)
        write(name, tdf, hdf, root)
        report(name, tdf, hdf, id_of, family, bench_ids)


if __name__ == "__main__":
    main()
