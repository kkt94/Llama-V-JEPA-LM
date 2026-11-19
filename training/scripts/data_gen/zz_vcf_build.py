"""(10-19) VideoChat-Flash training subsets -> stage-1 manifests  manifests/stage1/vcf/<subset>.parquet
Annotations and payloads are untrusted data: only read (json / file names / decord metadata). Run with python -I.

Columns: video (relative to the subset's prep_root), prep_exists, kind (subset), source, task, n_turns, conversations (JSON
[{role, content}], first user turn starts with "<|jepa_evidence|>\n"), t0 / t1 (seconds; NaN = whole video; the loader
samples its 16 frames inside [t0, t1] when both are set), orig_id.

Time-aware subsets (the model sees 16 frames and no time stamps), rephrased without numbers:
  dense step / event captions ("a - b seconds, step. ...")  -> whole video: the steps / events in order ("First, ... Then, ...")
  query grounding ("query 'q' -> a - b seconds")            -> clip [a, b] (t0 / t1): "describe this clip" -> q
  point-only answers ("happens in 32.3 seconds"), unparsable rows, RTL answers whose text still has numbers -> dropped
Contamination: rows whose video id is in a local benchmark (bench/), in Perception Test or in the Kinetics id lists are dropped.
usage: python -I zz_vcf_build.py [subset ...]   (default: all)  -> also logs/zz_vcf_build_stats.json"""
import os, re, sys, json, random, collections
import pandas as pd

R = "/data/llama_vjepa_lm"; D = f"{R}/data/VideoChat-Flash"; A = f"{D}/annotations/video"
OUT = f"{R}/manifests/stage1/vcf"; STATS = f"{R}/logs/zz_vcf_build_stats.json"
VIS = "<|jepa_evidence|>"
YT = re.compile(r"^(?:v_)?([A-Za-z0-9_-]{11})(?=$|[_.])")
NUM = r"\d+(?:\.\d+)?"

SUBSETS = {  # name: annotation, prep_root, mode, source, key kind for the contamination check
    "vidln_oops":    ("caption_vidln_oops-vidln-oops_train_11k.json", f"{D}/x/vidln_oops", "plain", "oops", "stem"),
    "youcook2":      ("caption_youcook2-youcook2-train_debug_9k.json", f"{D}/shortvideo_subset/caption_youcook2-youcook2-train_debug_9k", "plain", "youcook2", "ytdir"),
    "ego_qa":        ("vqa_ego_qa-ego_qa-train_8k.jsonl", f"{D}/shortvideo_subset/vqa_ego_qa-ego_qa-train_8k", "plain", "ego4d", "dir"),
    "lsmdc":         ("lsmdc-lsmdc_297k.json", f"{D}/shortvideo_subset/lsmdc-lsmdc_297k", "plain", "lsmdc", "lsmdc"),
    "timeit_coin":   ("timeit_COIN-TimeIT-COIN_10k.json", f"{D}/x/coin", "timeit", "coin", "yt"),
    "timeit_anet":   ("timeit_ANet-TimeIT-Activitynet_Captions_11k.json", f"{D}/x/timeit_ANet", "timeit", "activitynet", "yt"),
    "timeit_hirest": ("timeit_HiREST-TimeIT-HiREST_1k.json", f"{D}/timeaware_subset/timeit_HiREST-TimeIT-HiREST_1k", "timeit", "hirest", "yt"),
    "timeit_didemo": ("timeit_DiDeMo-TimeIT-DiDeMo_33k.json", f"{D}/x/timeit_DiDeMo", "timeit", "didemo", "flickr"),
    "timeit_queryd": ("timeit_QuerYD-TimeIT-QuerYD_15k.json", f"{D}/timeaware_subset/timeit_QuerYD-TimeIT-QuerYD_15k", "timeit", "queryd", "yt"),
    "anet_rtl":      ("grounding_ANetRTL-ActivityNet-RTL-ANet_RTL_34k.json", f"{D}/x/timeit_ANet", "rtl", "activitynet", "yt"),
    "anet_hl":       ("grounding_ANetHL-ANet-HL-ANet_HL2_11k.json", f"{D}/timeaware_subset/grounding_ANetHL-ANet-HL-ANet_HL2_11k", "hl", "activitynet", "yt"),
}
STEPS_Q = {"coin": "What steps are carried out in this video? Describe them in the order they happen.",
           "hirest": "What steps are carried out in this video? Describe them in the order they happen.",
           "activitynet": "Describe the events in this video in the order they happen."}
SEG_Q = ["Describe what happens in this clip.", "What is happening in this clip?", "Briefly describe what is shown in this clip."]
TASK_Q = ["What task is being carried out in this clip?", "What is the person doing in this clip?"]
LOWER_OK = {"a", "an", "the", "he", "she", "they", "we", "it", "his", "her", "their", "its", "people", "two", "three", "four", "several",
            "some", "another", "one", "this", "these", "those", "that", "there", "after", "more", "many", "both", "all", "each", "other",
            "when", "while", "as", "once", "before", "later", "now", "at", "in", "on", "with", "during", "eventually", "soon", "again",
            "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "kid", "kids", "child", "children", "person", "someone",
            "everyone", "add", "put", "cut", "take", "a", "lady", "group", "camera", "close", "various", "different", "multiple"}


def load(fn):
    p = f"{A}/{fn}"
    return [json.loads(l) for l in open(p) if l.strip()] if fn.endswith(".jsonl") else json.load(open(p))


def clean(text):
    t = str(text).replace(VIS, " ")
    t = re.sub(r"<\s*(image|video)\s*>", " ", t, flags=re.I)
    t = re.sub(r"[ \t]+\n", "\n", t)
    return re.sub(r"[ \t]{2,}", " ", t).strip()


def cap(s):
    s = s.strip().rstrip("\n").strip()
    return s[:1].upper() + s[1:] if s else s


def sent(s):
    s = cap(s).rstrip()
    return s if not s or s[-1] in ".!?" else s + "."


def low(s):
    m = re.match(r"[A-Za-z]+", s); w = m.group(0) if m else ""
    return s[:1].lower() + s[1:] if w.lower() in LOWER_OK and not (w.isupper() and len(w) > 1) else s


def plain_conv(cv):
    out = []
    for m in cv:
        role = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}.get(m.get("from", m.get("role")))
        if role is None:
            continue
        c = clean(m.get("value", m.get("content", "")))
        if not c or (out and out[-1]["role"] == role):
            return None
        out.append({"role": role, "content": c})
    while out and out[-1]["role"] != "assistant":
        out.pop()
    if not out or out[0]["role"] != "user":
        return None
    out[0]["content"] = VIS + "\n" + out[0]["content"]
    return out


def numbers_only(a):
    return not re.sub(r"(seconds?|[\d.,:\-\s<>()]|to|and|from)", "", a, flags=re.I).strip()


def parse_dense(a):
    seg = re.compile(rf"({NUM})\s*-\s*({NUM})\s*seconds,\s*(.*?)(?=\s+{NUM}\s*-\s*{NUM}\s*seconds,|\s*$)", re.S)
    out = [(float(x), float(y), d.strip()) for x, y, d in seg.findall(a)]
    return out if out and all(d for *_, d in out) else None


def ordered(steps):
    descs = [re.sub(r"^(?:(?:then|next|finally|afterwards|after that|later|lastly|first(?:ly)?)\b,?|after,)\s*", "", d.strip(), flags=re.I).strip().rstrip(".").strip()
             for _, _, d in sorted(steps, key=lambda s: (s[0], s[1]))]
    descs = [d for d in descs if d]
    if len(descs) == 1:
        return sent(descs[0])
    parts = []
    for i, d in enumerate(descs):
        lead = "First, " if i == 0 else "Finally, " if i == len(descs) - 1 else "Then, "
        parts.append(lead + low(d) + ".")
    return " ".join(parts)


def query_of(prompt):
    p = prompt.split("possible transcribed speech,")[-1]
    m = re.search(r"query:?\s*'", p)
    if not m:
        return None
    q = p[m.end():p.rfind("'")] if p.rfind("'") > m.end() else None
    return q.strip().strip("\n").strip() if q else None


def convert(name, r, mode, rng):
    """-> (task, conversations, t0, t1) or (None, reason)"""
    cv = r.get("conversations") or []
    if len(cv) < 2:
        return None, "empty_conversation"
    if mode == "plain":
        c = plain_conv(cv)
        if not c:
            return None, "empty_conversation"
        if all(numbers_only(m["content"]) for m in c if m["role"] == "assistant"):
            return None, "timestamp_only_answer"
        task = "mc" if "Options:" in c[0]["content"] or "(A)" in c[0]["content"] else "caption" if name == "vidln_oops" else "qa"
        if name == "youcook2":
            task = "step_caption"
        return task, c, None, None
    q, a = str(cv[0].get("value", "")), str(cv[1].get("value", "")).strip()
    src = SUBSETS[name][3]
    if mode == "timeit":
        steps = parse_dense(a)
        if steps and not a.startswith("The given query"):
            return "steps_ordered", [{"role": "user", "content": VIS + "\n" + STEPS_Q[src]}, {"role": "assistant", "content": ordered(steps)}], None, None
        m = re.match(rf"^The given query happens in ({NUM})\s*-\s*({NUM}) seconds\.?\s*$", a)
        if not m:
            return None, ("grounding_point_only" if re.match(rf"^The given query happens in {NUM} seconds", a) else "unparsed")
        t0, t1 = float(m.group(1)), float(m.group(2)); qq = query_of(q)
        if not qq:
            return None, "unparsed"
        if not t1 > t0:
            return None, "grounding_empty_window"
        qs = TASK_Q if src == "hirest" else SEG_Q
        return "segment_caption", [{"role": "user", "content": VIS + "\n" + rng.choice(qs)}, {"role": "assistant", "content": sent(qq)}], t0, t1
    if mode == "rtl":
        m = re.match(rf"^The given query happens in ({NUM})\s*-\s*({NUM}) seconds\.(.*)$", a, re.S)
        if not m:
            return None, "unparsed"
        t0, t1, rest = float(m.group(1)), float(m.group(2)), m.group(3).strip()
        rest = re.sub(r"^During (this|that) (time|period|interval|segment),?\s*", "", rest, flags=re.I)
        if not rest:
            return None, "grounding_no_text"
        if re.search(r"\d|<|>|interval", rest):
            return None, "grounding_text_has_numbers"
        if not t1 > t0:
            return None, "grounding_empty_window"
        return "segment_caption", [{"role": "user", "content": VIS + "\n" + rng.choice(SEG_Q)}, {"role": "assistant", "content": sent(rest)}], t0, t1
    if mode == "hl":
        m = re.match(r"^The highlight timestamps are in the ([\d., ]+) seconds\.", a)
        qm = re.search(r"sentence query:\s*(.*?)\.?\s*Please return the query-based", q, re.S)
        if not m or not qm:
            return None, "unparsed"
        ts = [float(x) for x in re.findall(NUM, m.group(1))]
        if not ts:
            return None, "unparsed"
        t0, t1 = min(ts), max(ts) + 2.0          # highlight clips are 2 s long
        return "segment_caption", [{"role": "user", "content": VIS + "\n" + rng.choice(SEG_Q)}, {"role": "assistant", "content": sent(qm.group(1))}], t0, t1
    raise ValueError(mode)


def keys(kind, video, rel):
    stem = os.path.basename(video).split(".")[0]
    parent = os.path.basename(os.path.dirname(video))
    if kind == "ytdir":                          # youcook2: training/<recipe>/<ytid>/split_k.mp4
        ks = {parent} | ({YT.match(parent).group(1)} if YT.match(parent) else set())
    elif kind == "dir":                          # ego_qa: <ego4d uuid>/split_k.mp4
        ks = {parent}
    elif kind == "lsmdc":
        ks = {stem, parent}
    elif kind == "flickr":                       # DiDeMo: <user>_<flickr id>_<secret>
        parts = stem.split("_")
        ks = {stem} | ({parts[1]} if len(parts) >= 2 and parts[1].isdigit() else set())
    elif kind == "yt":
        ks = {stem} | ({YT.match(stem).group(1)} if YT.match(stem) else set())
    else:
        ks = {stem}
    return {k for k in ks if len(k) >= 5}


def file_index(root):
    rel, by_base = set(), collections.defaultdict(list)
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if not d.startswith(".")]
        for f in fn:
            if f.startswith(".") or f.endswith(".zzpart"):
                continue
            p = os.path.relpath(os.path.join(dp, f), root)
            rel.add(p); by_base[f].append(p)
    return rel, by_base


def contamination_index():
    idx = json.load(open(f"{R}/logs/zz_vcf_benchidx.json"))
    ban = collections.defaultdict(set)
    for b, ids in idx.items():
        for i in ids:
            ban[i].add(b)
    pt = f"{R}/data/perception_test_train/extracted"
    for dp, dn, fn in os.walk(pt):
        for f in fn:
            ban[f.split(".")[0]].add("PerceptionTest(train)")
    for fn in ["classification_k710-k710-train_40k.jsonl", "caption_sharegemini_k400_223k.json", "caption_vidln_kinetics-vidln-kinetics_train_28k.json"]:
        for r in load(fn):
            m = YT.match(os.path.basename(r["video"]).split(".")[0])
            if m:
                ban[m.group(1)].add("Kinetics")
    for fn in [f for f in os.listdir(A) if "next_qa" in f or "nextqa" in f]:
        for r in load(fn):
            ban[os.path.basename(str(r["video"])).split(".")[0]].add("NExT-QA")
    for fn in [f for f in os.listdir(A) if f.startswith(("reasoning_star", "vqa_rgbd", "vqa_tgif", "classification_ssv2", "reasoning_clevrer"))]:
        tag = fn.split("-")[0]
        for r in load(fn):
            ban[os.path.basename(str(r["video"])).split(".")[0]].add(tag)
    return ban


BAD_SRC = re.compile(r"(?:^|[/:])(charades\w*|kinetics\w*|k[4-7]00|vatex|ssv2|something\w*|clevrer|next_?qa|nextvideo|tgif\w*|nturgbd\w*|ntu_?rgb\w*|perception\w*|star|mvbench|video-?mme|tempcompass|tomato|motionbench|tvbench|vinoground|favor\w*|temporalbench)(?:[/_\-.:]|$)", re.I)


def build(name, ban):
    fn, root, mode, src, kkind = SUBSETS[name]
    data = load(fn)
    rel, by_base = file_index(root)
    rng = random.Random(0)
    rows, drop, contam = [], collections.Counter(), collections.Counter()
    for r in data:
        video = str(r.get("video", ""))
        if BAD_SRC.search(video):
            drop["contamination_source_name"] += 1; continue
        hit = set()
        for k in keys(kkind, video, None):
            hit |= ban.get(k, set())
        if hit:
            drop["contamination_id"] += 1
            for h in hit:
                contam[h] += 1
            continue
        res = convert(name, r, mode, rng)
        if res[0] is None:
            drop[res[1]] += 1; continue
        task, conv, t0, t1 = res
        if video in rel:
            v = video
        elif len(by_base.get(os.path.basename(video), [])) == 1:
            v = by_base[os.path.basename(video)][0]
        else:
            drop["video_missing"] += 1; continue
        if os.path.getsize(os.path.join(root, v)) < 4096:     # source ships some empty stubs (262 B) for missing clips
            drop["video_stub_file"] += 1; continue
        assert conv[0]["role"] == "user" and conv[0]["content"].startswith(VIS + "\n")
        assert sum(m["content"].count(VIS) for m in conv) == 1
        assert not any(re.search(r"<\s*(image|video)\s*>", m["content"], re.I) for m in conv)
        rows.append({"video": v, "prep_exists": True, "kind": name, "source": src, "task": task, "n_turns": len(conv) // 2,
                     "conversations": json.dumps(conv, ensure_ascii=False), "t0": t0, "t1": t1, "orig_id": str(r.get("id", ""))})
    df = pd.DataFrame(rows)
    if len(df):
        df["t0"] = df.t0.astype("float64"); df["t1"] = df.t1.astype("float64")
    return df, {"annotation": fn, "prep_root": root, "rows_in": len(data), "rows_kept": len(df), "dropped": dict(drop),
                "contamination_by_bench": dict(contam), "unique_videos": int(df.video.nunique()) if len(df) else 0,
                "tasks": df.task.value_counts().to_dict() if len(df) else {}}


def decord_check(df, root, n=20, seed=0):
    import decord
    vids = sorted(df.video.unique()); rng = random.Random(seed); pick = rng.sample(vids, min(n, len(vids)))
    ok, bad, win_out = 0, [], 0
    for v in pick:
        try:
            vr = decord.VideoReader(f"{root}/{v}", num_threads=1); nfr, fps = len(vr), float(vr.get_avg_fps())
            idx = [min(nfr - 1, int((i + 0.5) * nfr / 16)) for i in range(16)]   # 16 frames spread like the loader's sampling
            vr.get_batch(idx).asnumpy(); ok += 1
            dur = nfr / fps
            w = df[(df.video == v) & df.t1.notna()]
            win_out += int((w.t0 >= dur).sum())
        except Exception as e:
            bad.append((v, repr(e)[:120]))
    return {"tested": len(pick), "ok": ok, "failed": bad, "windows_starting_after_end": win_out}


if __name__ == "__main__":
    names = sys.argv[1:] or list(SUBSETS)
    os.makedirs(OUT, exist_ok=True)
    stats = json.load(open(STATS)) if os.path.exists(STATS) else {}
    ban = contamination_index()
    for name in names:
        df, st = build(name, ban)
        st["decord"] = decord_check(df, SUBSETS[name][1]) if len(df) else None
        for v, _ in (st["decord"] or {}).get("failed", []):
            df.loc[df.video == v, "prep_exists"] = False
        st["sample"] = json.loads(df.conversations.iloc[0]) if len(df) else None
        df.to_parquet(f"{OUT}/{name}.parquet", index=False)
        stats[name] = st
        print(name, json.dumps({k: st[k] for k in ("rows_in", "rows_kept", "dropped", "contamination_by_bench", "unique_videos", "tasks", "decord")}), flush=True)
    json.dump(stats, open(STATS, "w"), indent=1)
    json.dump({k: v["prep_root"] for k, v in stats.items()}, open(f"{OUT}/prep_roots.json", "w"), indent=1)
