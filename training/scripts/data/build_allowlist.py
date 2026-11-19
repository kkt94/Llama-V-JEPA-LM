"""Benchmark-contamination filter: keep only training items whose source video is not used by any evaluation benchmark.

Ids are normalized per source family (youtube 11-char, charades 5-char, perceptiontest video_N, ssv2 numeric, ...)
and matched within the family. ActivityNet/COIN/FineGym/YouCook2/HD-VILA/Vinoground/Video-MME ids are all YouTube ids,
so they share the `youtube` family. Ambiguous or unmappable refs are removed (conservative) and listed in the report.
usage: build_allowlist.py [--root /data/llama_vjepa_lm] [--out /data/llama_vjepa_lm/manifests/allowlist]"""
import argparse, ast, collections, csv, glob, json, os, re, sys, tarfile, zipfile
import pyarrow.parquet as pq

YT = r"[A-Za-z0-9_-]{11}"
# Training folders excluded by policy (spec §9.3): first run excludes eval-related LLaVA folders wholesale.
# nextqa and perceptiontest folders were restored on 2026-10-09: every video id is in the official train split
# (NExT-QA OE/train, PerceptionTest train) and none is in an evaluation split. activitynetqa stays out: no train list.
FOLDER_POLICY = ("activitynetqa",)


# ---------------------------------------------------------------- benchmark id extraction
class Bench:
    """family -> id -> {label: first raw reference}."""
    def __init__(self, name):
        self.name, self.ids = name, collections.defaultdict(dict)

    def add(self, family, vid, label, raw):
        self.ids[family].setdefault(vid, {}).setdefault(label, raw)

    def counts(self):
        c = collections.Counter()
        for fam, d in self.ids.items():
            for vid, labels in d.items():
                for lab in labels:
                    c[(lab, fam)] += 1
        return c


def must(m, what):
    if not m:
        raise SystemExit(f"cannot parse {what!r}")
    return m


def bench_perception_test(root):
    b, P = Bench("perception_test"), f"{root}/bench/perception_test"
    valid = set(json.load(open(f"{P}/extracted/all_valid.json")))
    valid |= {f[:-4] for f in os.listdir(f"{P}/extracted/videos") if f.endswith(".mp4")}
    valid |= set(json.load(open(f"{P}/cut_frame_mapping_valid.json")))
    valid |= {x.strip() for x in open(f"{P}/mc_question_valid_id_list.csv").read().replace("\n", ",").split(",") if x.strip()}
    for v in valid:
        must(re.fullmatch(r"video_\d+", v), v)
        b.add("perceptiontest", v, "perception_test/valid", v)
    train = {k for k in json.load(open(f"{P}/cut_frame_mapping_train.json")) if re.fullmatch(r"video_\d+", k)}
    return b, train


def bench_mvp(root):
    b = Bench("mvp")
    snap = f"{root}/bench/minimal_video_pairs"
    rows = []
    for f in sorted(glob.glob(f"{snap}/*/*.parquet")):
        rows += pq.read_table(f, columns=["video_path", "source"]).to_pylist()
    vino = {r["index"]: r for r in csv.DictReader(open(f"{root}/bench/mvp_sources/raw/hf_vinoground/vinoground.csv"))}
    for r in rows:
        src, p = r["source"], r["video_path"].lstrip("/")
        rel = p.split("/", 1)[1]
        lab = f"mvp/{src}"
        if src == "pt":
            b.add("perceptiontest", must(re.fullmatch(r"(video_\d+)\.mp4", rel), p).group(1), lab, p)
        elif src == "ssv2":
            b.add("ssv2", str(int(must(re.fullmatch(r"(\d+)\.mp4", rel), p).group(1))), lab, p)
        elif src == "star":  # STAR is built on Charades
            b.add("charades", must(re.fullmatch(r"([A-Z0-9]{5})\.mp4", rel), p).group(1), lab, p)
        elif src == "vinoground":  # N_pos/N_neg -> YouTube id via vinoground.csv
            i, side = must(re.fullmatch(r"(\d+)_(pos|neg)\.mp4", rel), p).groups()
            b.add("youtube", must(re.fullmatch(YT, vino[i][f"{side}_vid"]), p).group(0), lab, f"{p} -> {vino[i][f'{side}_vid']}")
        elif src in ("clevrer", "intphys", "inflevel", "grasp", "language_table"):  # synthetic / lab-recorded
            b.add(src, rel[:-4], lab, p)
        else:
            raise SystemExit(f"unknown MVP source {src}")
    for i, r in vino.items():  # whole Vinoground is test-only; exclude all of it, not only MVP's subset
        for side in ("pos", "neg"):
            b.add("youtube", must(re.fullmatch(YT, r[f"{side}_vid"]), r[f"{side}_vid"]).group(0), "vinoground/all", f"vinoground.csv[{i}].{side}_vid")
    return b


def bench_video_mme_v2(root):
    b = Bench("video_mme_v2")
    for r in pq.read_table(f"{root}/bench/Video-MME-v2/test.parquet", columns=["video_id", "url"]).to_pylist():
        m = must(re.search(r"(?:v=|youtu\.be/|shorts/)(" + YT + r")(?![A-Za-z0-9_-])", r["url"]), r["url"])
        b.add("youtube", m.group(1), "video_mme_v2", f"{r['video_id']}: {r['url']}")
    return b


def bench_video_mme_logical(root):
    b = Bench("video_mme_logical")  # programmatically rendered; no real-video family
    for r in json.load(open(f"{root}/bench/video-mme-logical/extracted/video-mme-logical/three_level_testset.json")):
        b.add("vmml_synthetic", r["id"], "video_mme_logical", r["video_path"])
    return b


def bench_nextqa(root):
    """lmms-lab/NExTQA annotations. MC/test, OE/test, OE/validation are evaluated by lmms-eval; OE/train is allowed."""
    b, P = Bench("nextqa"), f"{root}/bench/NExTQA"
    train = set()
    for split in ("MC/test", "OE/test", "OE/validation", "OE/train"):
        col = pq.read_table(f"{P}/{split}-00000-of-00001.parquet", columns=["video"]).column("video").to_pylist()
        ids = {str(int(must(re.fullmatch(r"\d+", str(v)), v).group(0))) for v in col}
        if split == "OE/train":
            train = ids
        else:
            for v in ids:
                b.add("nextqa", v, f"nextqa/{split.replace('/', '_')}", f"{split}:{v}")
    return b, train


def bench_activitynetqa(root):
    """lmms-lab/ActivityNetQA test split (the split lmms-eval evaluates); video_name is a YouTube id without `v_`."""
    b = Bench("activitynetqa")
    for v in set(pq.read_table(f"{root}/bench/ActivityNetQA/data/test-00000-of-00001.parquet", columns=["video_name"]).column("video_name").to_pylist()):
        b.add("youtube", must(re.fullmatch(YT, v), v).group(0), "activitynetqa/test", f"test:{v}")
    return b


def bench_intphys2(root):
    b = Bench("intphys2")  # Unreal-rendered; no real-video family
    P = f"{root}/bench/IntPhys2"
    for split in ("Main", "Debug"):
        for r in csv.DictReader(open(f"{P}/{split}/metadata.csv")):
            b.add("intphys2_synthetic", r["name"], f"intphys2/{split}", r["file_name"])
    for f in sorted(os.listdir(f"{P}/HeldOut/Videos")):
        b.add("intphys2_synthetic", f.rsplit(".", 1)[0], "intphys2/HeldOut", f)
    return b


def temporalbench_key(ds, fn):
    """TemporalBench clip name -> (family, id). Clip names are <source id>_start_<s>_end_<e>.mp4 (FineGym differs)."""
    if ds in ("ActivityNet", "COIN"):
        return "youtube", must(re.fullmatch(r"(" + YT + r")_start_.*", fn), fn).group(1)
    if ds == "FineGym":
        m = re.fullmatch(r"Finegym_(" + YT + r")_start_.*", fn) or re.fullmatch(r"(" + YT + r")_E_\d+_\d+_A_.*", fn)
        return "youtube", must(m, fn).group(1)
    if ds == "Charades":
        return "charades", must(re.fullmatch(r"([A-Z0-9]{5})_start_.*", fn), fn).group(1)
    if ds == "EgoExo4D":  # take_name, e.g. georgiatech_bike_06_10
        return "egoexo4d_take_name", must(re.fullmatch(r"(.+?)_start_[\d.]+_end_[\d.]+\.mp4", fn), fn).group(1)
    if ds == "Movie_Description":  # LSMDC <movie>_<timestamps>
        return "lsmdc_movie", must(re.fullmatch(r"(\d+_.+?)_\d\d\.\d\d\.\d\d\.\d+-.*", fn), fn).group(1)
    if ds == "Oops":  # file names are YouTube compilation titles, not YouTube ids
        return "oops_title", must(re.fullmatch(r"(.+?)\d+_mute\.mp4", fn), fn).group(1)
    raise SystemExit(f"unknown TemporalBench dataset {ds}")


def bench_temporalbench(root):
    b, P = Bench("temporalbench"), f"{root}/bench/TemporalBench"
    names = set()
    for f in sorted(glob.glob(f"{P}/temporalbench_*.json")):
        names |= {r["video_name"] for r in json.load(open(f))}
    for f in sorted(glob.glob(f"{P}/data/*.parquet")):
        names |= set(pq.read_table(f, columns=["video_name"]).column("video_name").to_pylist())
    for f in sorted(glob.glob(f"{P}/*.zip")):  # union with archive contents (central directory only)
        names |= {n for n in zipfile.ZipFile(f).namelist() if n.endswith(".mp4") and "__MACOSX" not in n and not os.path.basename(n).startswith("._")}
    for n in sorted(names):
        parts = n.split("/")
        must(len(parts) >= 3 and parts[0] in ("short_video", "long_video"), n)
        fam, vid = temporalbench_key(parts[-2], parts[-1])
        b.add(fam, vid, f"temporalbench/{parts[-2]}", n)
    return b


def bench_perception_test_test(root):
    """PerceptionTest test split (MC questions; mc_question_test.json + all_test.json + id list)."""
    b, P = Bench("perception_test_test"), f"{root}/bench/perception_test_test"
    ids = set(json.load(open(f"{P}/extracted/mc_question_test.json"))) | set(json.load(open(f"{P}/extracted/all_test.json")))
    ids |= {x.strip() for x in open(f"{P}/raw/mc_question_test_id_list.csv").read().replace("\n", ",").split(",") if x.strip()}
    for v in ids:
        must(re.fullmatch(r"video_\d+", v), v)
        b.add("perceptiontest", v, "perception_test/test", v)
    return b


def bench_tempcompass(root):
    """TempCompass videos are Shutterstock clips (utils/download_video.py: ak.picdn.net/shutterstock/videos/<id>/...);
    WebVid ids are Shutterstock ids, so the raw source ids go in the `webvid` family. Constructed videos
    (<id>_reverse, <id>_concat_<k>, <id1>_<id2>_<k>) map to their source ids (all four task files, not only MC)."""
    b, P = Bench("tempcompass"), f"{root}/bench/TempCompass"
    vids = set(json.load(open(f"{P}/github_e1b4631/meta_info.json")))
    for f in sorted(glob.glob(f"{P}/raw/hf_lmms-lab_TempCompass/*/test-*.parquet")):
        vids |= set(pq.read_table(f, columns=["video_id"]).column("video_id").to_pylist())
    vids |= {os.path.basename(f)[:-4] for f in glob.glob(f"{P}/extracted/videos/*.mp4")}
    for v in vids:
        parts = v.split("_")
        src = parts[:1] if len(parts) == 1 or parts[1] in ("reverse", "concat") else parts[:2]
        for s in src:
            b.add("webvid", str(int(must(re.fullmatch(r"\d+", s), v).group(0))), "tempcompass", v)
    return b


def bench_tomato(root):
    """TOMATO: every video under `tomato_video` (<demonstration_type>/<key>); YouTube source URLs also as `youtube`.
    The 50 CLEVRER-sourced questions carry no CLEVRER id (url field is just 'CLEVRER'); their CLEVRER ids are taken
    from bench/TOMATO/clevrer_content_match.tsv (frame-content match against CLEVRER train+validation) when present."""
    b, P = Bench("tomato"), f"{root}/bench/TOMATO"
    for f in sorted(glob.glob(f"{P}/github_fe2025f/data/*.json")):
        for qid, x in json.load(open(f)).items():
            b.add("tomato_video", f"{x['demonstration_type']}/{x['key']}", f"tomato/{x['demonstration_type']}", f"{os.path.basename(f)}:{qid}")
            u = x["video_source_url"]
            m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)(" + YT + r")(?![A-Za-z0-9_-])", u)
            if m:
                b.add("youtube", m.group(1), "tomato/youtube", u)
    cm = f"{P}/clevrer_content_match.tsv"
    if os.path.exists(cm):
        for r in csv.DictReader((l for l in open(cm) if not l.startswith("#")), delimiter="\t"):
            if r["match"] == "1":
                b.add("clevrer", r["clevrer_video"], "tomato/clevrer", r["tomato_video"])
    return b


# ---------------------------------------------------------------- training ref normalization
def norm_llava(v):
    """LLaVA-Video-178K video path -> (family, id, provenance) or None."""
    for pat, fam, prov in (
        (r"academic_source/Charades/([A-Z0-9]{5})\.mp4", "charades", "charades"),
        (r"academic_source/activitynet/(?:v1-3/train_val/)?v_(" + YT + r")\.(?:mp4|mkv|webm)", "youtube", "activitynet"),
        (r"ActivityNet-QA/activitynet/train/(?:v1-3/train_val/)?v_(" + YT + r")\.(?:mp4|mkv|webm)", "youtube", "activitynet"),
        (r"academic_source/youcook2/\d+/(" + YT + r")/split_\d+\.mp4", "youtube", "youcook2"),
        (r"academic_source/ego4d/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.mp4", "ego4d", "ego4d"),
        (r"(?:academic_source/NextQA|NextQA/NExTVideo)/\d+/(\d+)\.mp4", "nextqa", "nextqa"),
        (r"perception_test/videos/(video_\d+)\.mp4", "perceptiontest", "perceptiontest"),
        (r"liwei_youtube_videos/videos/(?:youtube_video_\d+/)?ytb_(" + YT + r")(?:\.mp4)?", "youtube", "youtube"),
        (r"liwei_youtube_videos/videos/hdvila/(" + YT + r")\.(?:mp4|mkv)", "youtube", "hdvila"),
        (r"liwei_youtube_videos/videos/([0-9a-f]{32})", "hashed_name", "youtube"),  # no recoverable source id
    ):
        m = re.fullmatch(pat, v)
        if m:
            return fam, m.group(1), prov
    if v.startswith("shareVideoGPTV/frames/all_frames/"):
        return norm_sharegpt(v.rsplit("/", 1)[1])
    return None


def norm_sharegpt(v):
    """ShareGPTVideo id: v_<yt>-Scene-NNN (ActivityNet scenes), <yt> (YouTube), digits (WebVid)."""
    m = re.fullmatch(r"v_(" + YT + r")-Scene-\d+", v)
    if m:
        return "youtube", m.group(1), "activitynet"
    if re.fullmatch(YT, v):
        return "youtube", v, "youtube"
    if re.fullmatch(r"\d{1,10}", v):
        return "webvid", v, "webvid"
    return None


# ---------------------------------------------------------------- matching
class Matcher:
    def __init__(self, benches, pt_train, nextqa_train):
        self.idx = collections.defaultdict(dict)  # (family, id) -> {label: raw}
        for b in benches:
            for fam, d in b.ids.items():
                for vid, labels in d.items():
                    for lab, raw in labels.items():
                        self.idx[(fam, vid)].setdefault(lab, raw)
        self.pt_train, self.nextqa_train = pt_train, nextqa_train
        self.ssv2 = {vid for (fam, vid) in self.idx if fam == "ssv2"}

    def check(self, key):
        """(family, id, provenance) -> reasons. Empty = keep."""
        if key is None:
            return ["unparsed_video_ref"]
        fam, vid, prov = key
        reasons = [f"{lab}:{fam}:{vid}" for lab in sorted(self.idx.get((fam, vid), {}))]
        if fam == "perceptiontest" and vid not in self.pt_train:
            reasons.append(f"perceptiontest_not_in_train_split:{vid}")  # valid or (absent) test split
        if fam == "egoexo4d_take_uid":
            reasons.append(f"ambiguous:egoexo4d_take_uid_unmappable_to_temporalbench_take_name:{vid}")
        if fam == "hashed_name":
            reasons.append(f"ambiguous:hashed_filename_no_source_id:{vid}")
        if fam == "webvid" and vid in self.ssv2:
            reasons.append(f"ambiguous:numeric_id_equals_mvp_ssv2_id:{vid}")
        if fam == "nextqa" and vid not in self.nextqa_train and not reasons:
            reasons.append(f"ambiguous:nextqa_id_not_in_any_split:{vid}")
        return reasons


class Tally:
    """per source group: records/unique videos kept and removed, per reason label."""
    def __init__(self):
        self.rec = collections.Counter()
        self.vids = collections.defaultdict(set)
        self.examples = collections.defaultdict(list)
        self.resolved = collections.Counter()  # (group, state) for records flagged `unchecked` before the NExT-QA/ANet-QA lists existed

    def add(self, group, key, raw, reasons, matcher):
        vk = key[:2] if key else ("unparsed", raw)
        state = "removed" if reasons else "kept"
        self.rec[(group, state)] += 1
        self.vids[(group, state)].add(vk)
        labels = set()
        for r in reasons:
            lab = r.split(":", 1)[0] if not r.startswith(("ambiguous:", "folder_policy:")) else ":".join(r.split(":")[:2])
            labels.add(lab)
        for lab in labels:
            self.rec[(group, "removed", lab)] += 1
            self.vids[(group, "removed", lab)].add(vk)
            ex = self.examples[(group, lab)]
            if len(ex) < 3 and all(e[0] != vk for e in ex):
                bench_raw = matcher.idx.get(vk, {}).get(lab, "")
                ex.append((vk, raw, bench_raw))


def write_jsonl(fh, obj):
    fh.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def run_records(group, items, matcher, tally, keep_fh, rm_fh, extra_reasons=()):
    """items: iterable of (record locator dict, raw video ref, normalized key)."""
    for loc, raw, key in items:
        reasons = list(extra_reasons) + matcher.check(key)
        tally.add(group, key, raw, reasons, matcher)
        # records that carried `unchecked` before: nextqa / activitynet provenance and not removed for any other reason
        new_list_reasons = [r for r in reasons if r.startswith(("nextqa/", "activitynetqa/", "ambiguous:nextqa_id_not_in_any_split"))]
        if key and (key[0] == "nextqa" or key[2] == "activitynet") and len(new_list_reasons) == len(reasons):
            tally.resolved[(group, "removed" if reasons else "kept")] += 1
        out = dict(loc, video=raw, key=f"{key[0]}:{key[1]}" if key else None)
        if reasons:
            out["reasons"] = reasons
            write_jsonl(rm_fh, dict(out, source=group))
        else:
            write_jsonl(keep_fh, out)


# ---------------------------------------------------------------- training sources
def llava_files(root):
    R = f"{root}/data/LLaVA-Video-178K"
    fs = sorted(f for f in glob.glob(f"{R}/*/*.json") if "/gpt4o_" not in f)
    return R, fs


def llava_group(rel):
    folder = rel.split("/")[0]
    if folder == "llava_hound":
        return "llava/llava_hound"
    return "llava/" + re.sub(r"^\d+_\d+_[sm]_", "", folder).replace("_v0_1", "")


def iter_llava(R, f):
    rel = os.path.relpath(f, R)
    for i, r in enumerate(json.load(open(f))):
        v = r.get("video", "")
        yield {"file": rel, "idx": i, "id": r.get("id")}, v, norm_llava(v)


def iter_sharegpt(R, f):
    rel = os.path.relpath(f, R)
    with open(f) as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            v = r["video"]
            yield {"file": rel, "idx": i, "id": r.get("id")}, v, norm_sharegpt(v)


def plm_key(ds, vid):
    if ds in ("coin", "youcook2", "crosstask", "ht100m"):
        return ("youtube", vid, ds) if re.fullmatch(YT, vid) else None
    if ds == "ego4d":
        return ("ego4d", vid, ds) if re.fullmatch(r"(grp-)?[0-9a-f-]{36}", vid) else None
    if ds == "egoexo4d":
        return ("egoexo4d_take_uid", vid, ds)
    return None


def iter_plm_fgqa(R, f):
    rel = os.path.relpath(f, R)
    t = pq.read_table(f, columns=["qa_id", "metadata"])
    for i, (qa, md) in enumerate(zip(t.column("qa_id").to_pylist(), t.column("metadata").to_pylist())):
        md = md if isinstance(md, dict) else ast.literal_eval(md)
        raw = f"{md['source_dataset']}:{md['source_video_id']}"
        yield {"file": rel, "idx": i, "id": qa}, raw, plm_key(md["source_dataset"], md["source_video_id"])


def iter_plm_sav(R, f, sav_train):
    rel = os.path.relpath(f, R)
    t = pq.read_table(f, columns=["uid", "video"])
    for i, (uid, v) in enumerate(zip(t.column("uid").to_pylist(), t.column("video").to_pylist())):
        m = re.fullmatch(r"(sav_\d{6})\.mp4", v)
        loc = {"file": rel, "idx": i, "id": uid}
        if m and m.group(1) not in sav_train:
            loc["sav_train_membership"] = "not_in_scanned_train_tars"
        yield loc, v, ("sav", m.group(1), "sav") if m else None


def scan_sav(root):
    """SA-V train tars with a .ok marker -> {video id: tar}; returns (ids, done tars, pending tars)."""
    S = f"{root}/data/SA-V"
    listed = sorted(l.split("\t")[0] for l in open(f"{S}/links.tsv").read().splitlines()[1:] if re.match(r"sav_\d{3}\.tar\t", l))
    ids, done, pending = {}, [], []
    for tname in listed:
        if not os.path.exists(f"{S}/{tname}.ok"):
            pending.append(tname)
            continue
        done.append(tname)
        with tarfile.open(f"{S}/raw/{tname}") as t:
            for m in t:
                mm = re.fullmatch(r"sav_train/sav_\d{3}/(sav_\d{6})\.mp4", m.name)
                if mm:
                    ids[mm.group(1)] = tname
    return ids, done, pending


# ---------------------------------------------------------------- report
def fmt_table(tally, groups):
    lines = ["| source | records kept | records removed | videos kept | videos removed |", "|---|---:|---:|---:|---:|"]
    for g in groups:
        lines.append(f"| {g} | {tally.rec[(g, 'kept')]:,} | {tally.rec[(g, 'removed')]:,} | {len(tally.vids[(g, 'kept')]):,} | {len(tally.vids[(g, 'removed')]):,} |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/llama_vjepa_lm")
    ap.add_argument("--out", default="/data/llama_vjepa_lm/manifests/allowlist")
    a = ap.parse_args()
    root, out = a.root, a.out
    os.makedirs(out, exist_ok=True)

    # 1. benchmark ids, counted before any matching. A zero count makes the comparison vacuous -> abort.
    pt, pt_train = bench_perception_test(root)
    nq, nq_train = bench_nextqa(root)
    benches = [pt, bench_mvp(root), bench_video_mme_v2(root), bench_video_mme_logical(root), bench_temporalbench(root), bench_intphys2(root),
               nq, bench_activitynetqa(root), bench_perception_test_test(root), bench_tempcompass(root), bench_tomato(root)]
    print("== benchmark ids extracted (label, family): unique ids")
    bench_counts = []
    for b in benches:
        c = b.counts()
        for (lab, fam), n in sorted(c.items()):
            print(f"  {b.name:18s} {lab:32s} {fam:20s} {n}")
            bench_counts.append((b.name, lab, fam, n))
            if n == 0:
                raise SystemExit(f"0 ids extracted for {lab}/{fam}: vacuous comparison")
        with open(f"{out}/benchmark_ids_{b.name}.txt", "w") as fh:
            for fam in sorted(b.ids):
                for vid in sorted(b.ids[fam]):
                    fh.write(f"{fam}\t{vid}\t{','.join(sorted(b.ids[fam][vid]))}\n")
    print(f"  perception_test train ids (allowed): {len(pt_train)}")
    if not pt_train:
        raise SystemExit("0 PerceptionTest train ids")
    pt_mvp = {v for v, labs in benches[1].ids["perceptiontest"].items()}
    print(f"  mvp/pt ids: in PT train {len(pt_mvp & pt_train)}, in PT valid {len(pt_mvp & set(pt.ids['perceptiontest']))}, neither {len(pt_mvp - pt_train - set(pt.ids['perceptiontest']))}")
    print(f"  nextqa OE/train ids (allowed): {len(nq_train)}, train & eval: {len(nq_train & set(nq.ids['nextqa']))}")
    if not nq_train:
        raise SystemExit("0 NExT-QA train ids")
    matcher = Matcher(benches, pt_train, nq_train)

    tally, groups = Tally(), []
    rm_fh = open(f"{out}/removed.jsonl.part", "w")

    # 2. LLaVA-Video-178K (+ llava_hound)
    R, fs = llava_files(root)
    with open(f"{out}/llava_video_allowlist.jsonl.part", "w") as keep:
        for f in fs:
            rel = os.path.relpath(f, R)
            g = llava_group(rel)
            groups += [g] if g not in groups else []
            extra = [f"folder_policy:{p}" for p in FOLDER_POLICY if g == f"llava/{p}"]
            run_records(g, iter_llava(R, f), matcher, tally, keep, rm_fh, extra)
    # id-level result for policy-excluded folders (what an id-only filter would have kept)
    restorable = collections.Counter()
    for f in fs:
        g = llava_group(os.path.relpath(f, R))
        if g.split("/")[1] in FOLDER_POLICY:
            for loc, raw, key in iter_llava(R, f):
                restorable[(g, "id_clean" if not matcher.check(key) else "id_match")] += 1

    # 3. ShareGPTVideo instruction files (video only; image_* files carry no video)
    S = f"{root}/data/ShareGPTVideo/video_instruction/train"
    sfs = sorted(f for f in glob.glob(f"{S}/*/*.jsonl") if not os.path.basename(f).startswith("image_"))
    skipped_img = sorted(os.path.relpath(f, S) for f in glob.glob(f"{S}/*/image_*.jsonl"))
    with open(f"{out}/sharegptvideo_allowlist.jsonl.part", "w") as keep:
        for f in sfs:
            g = "sharegptvideo/" + os.path.relpath(f, S)
            groups.append(g)
            run_records(g, iter_sharegpt(S, f), matcher, tally, keep, rm_fh)

    # 4. SA-V (train tars that finished downloading)
    sav_ids, sav_done, sav_pending = scan_sav(root)
    with open(f"{out}/sav_allowlist.jsonl.part", "w") as keep:
        groups.append("sav/train")
        items = (({"tar": t, "member": f"{v}.mp4"}, f"{t}:{v}.mp4", ("sav", v, "sav")) for v, t in sorted(sav_ids.items()))
        run_records("sav/train", items, matcher, tally, keep, rm_fh)

    # 5. PLM-Video-Human (annotations only; FGQA -> external sources, regional -> SA-V)
    P = f"{root}/data/PLM-Video-Human"
    with open(f"{out}/plm_allowlist.jsonl.part", "w") as keep:
        groups.append("plm/fgqa")
        run_records("plm/fgqa", iter_plm_fgqa(P, f"{P}/fgqa/plm_fgqa_train.parquet"), matcher, tally, keep, rm_fh)
        for sub in ("rcap", "rdcap", "rtloc"):
            groups.append(f"plm/{sub}")
            run_records(f"plm/{sub}", iter_plm_sav(P, f"{P}/{sub}/plm_{sub}_train.parquet", sav_ids), matcher, tally, keep, rm_fh)

    # 6. NVIDIA synthetic (Isaac Sim): no real-video family, kept at shard level
    N = f"{root}/data/nvidia_physical_interaction"
    nv = sorted(os.path.relpath(f, N) for f in glob.glob(f"{N}/videos/*/*.tar"))
    with open(f"{out}/nvidia_allowlist.jsonl.part", "w") as keep:
        groups.append("nvidia/videos")
        items = (({"shard": s}, s, ("nvidia_synthetic", s, "nvidia")) for s in nv)
        run_records("nvidia/videos", items, matcher, tally, keep, rm_fh)

    rm_fh.close()
    for f in glob.glob(f"{out}/*.part"):
        os.replace(f, f[:-5])

    # 7. report + sanity examples
    print("\n== matched examples (training ref <-> benchmark ref), up to 3 per (source, reason)")
    ex_lines = []
    for (g, lab), exs in sorted(tally.examples.items()):
        for vk, raw, braw in exs:
            s = f"  {g:42s} {lab:40s} {vk[0]}:{vk[1]}  train={raw}  bench={braw}"
            print(s)
            ex_lines.append(f"| {g} | {lab} | `{vk[0]}:{vk[1]}` | `{raw}` | `{braw}` |")

    plm_not_in_sav = sum(v[:-4] not in sav_ids for x in ("rcap", "rdcap", "rtloc")
                         for v in pq.read_table(f"{P}/{x}/plm_{x}_train.parquet", columns=["video"]).column("video").to_pylist())
    rep = ["# Training-data contamination filter", "",
           "Generated by `repo/scripts/data/build_allowlist.py`. Ids are matched within a source family; "
           "ActivityNet / COIN / FineGym / YouCook2 / CrossTask / HT100M / HD-VILA / Vinoground / Video-MME ids are all YouTube ids and share the `youtube` family.", "",
           "## Benchmark ids extracted (before matching)", "", "| benchmark | label | family | unique ids |", "|---|---|---|---:|"]
    rep += [f"| {b} | {l} | {f} | {n:,} |" for b, l, f, n in bench_counts]
    rep += [f"| perception_test | train (allowed, not a benchmark) | perceptiontest | {len(pt_train):,} |",
            f"| nextqa | OE/train (allowed, not a benchmark) | nextqa | {len(nq_train):,} |", "",
            f"MVP `pt` ids: {len(pt_mvp & pt_train)} in PerceptionTest train, {len(pt_mvp & set(pt.ids['perceptiontest']))} in valid, "
            f"{len(pt_mvp - pt_train - set(pt.ids['perceptiontest']))} in neither. MVP-used train ids are removed from training.", "",
            "## Kept / removed per training source", ""]
    rep += fmt_table(tally, groups)
    rep += ["", "### Formerly unchecked records (NExT-QA / ActivityNet provenance, outside policy folders)", "",
            "Before the NExT-QA and ActivityNet-QA annotation lists were on disk these records were kept with an `unchecked` flag. "
            "`now removed` counts only removals by the NExT-QA / ActivityNet-QA lists (records removed for other reasons were never flagged).", "",
            "| source | now kept | now removed |", "|---|---:|---:|"]
    rep += [f"| {g} | {tally.resolved[(g, 'kept')]:,} | {tally.resolved[(g, 'removed')]:,} |" for g in groups if tally.resolved[(g, 'kept')] + tally.resolved[(g, 'removed')]]
    rep += ["", "## Removed per source and reason", "", "| source | reason | records | unique videos |", "|---|---|---:|---:|"]
    for k in sorted(k for k in tally.rec if len(k) == 3):
        rep.append(f"| {k[0]} | {k[2]} | {tally.rec[k]:,} | {len(tally.vids[k]):,} |")
    rep += ["", "Reason labels: `<benchmark label>` = source video id is in that benchmark; "
            "`perceptiontest_not_in_train_split` = PerceptionTest id not in the official train list (valid, or test which is not on disk); "
            "`folder_policy:*` = spec §9.3 first-run folder exclusion; `ambiguous:*` = id format/mapping cannot be resolved, removed conservatively; "
            "`unparsed_video_ref` = path did not match any known format.", "",
            "### Policy-excluded LLaVA folders, id-level view", "",
            "Records in these folders are removed by folder policy. Id-level view of the same records (`id-clean` = no hit against the on-disk benchmark/split lists). "
            "For perceptiontest, id-clean means the id is in the official train list and not used by MVP, so it is restorable under spec §9.3. "
            "For nextqa, id-clean means the id is in NExT-QA OE/train and in no evaluated split. "
            "For activitynetqa, id-clean means not in the ActivityNet-QA test split or any other benchmark (ActivityNet-QA train/val lists are not on disk).", "",
            "| folder | records with id-level hit | records id-clean |", "|---|---:|---:|"]
    for p in FOLDER_POLICY:
        g = f"llava/{p}"
        rep.append(f"| {g} | {restorable[(g, 'id_match')]:,} | {restorable[(g, 'id_clean')]:,} |")
    rep += ["", "## Matched examples (3 per overlapping source x reason)", "", "| source | reason | key | training ref | benchmark ref |", "|---|---|---|---|---|"] + ex_lines
    rep += ["", "## Could not check", "",
            "- ActivityNet-QA: only the test split is on disk (lmms-lab/ActivityNetQA has only `test`, the split lmms-eval evaluates). ActivityNet-QA train/val lists are not on disk; val is not an evaluated split here.",
            "- PerceptionTest: only ids in `cut_frame_mapping_train.json` (2184 train ids) are allowed and everything else is removed; the test split ids (bench/perception_test_test) are also listed as a benchmark.",
            "- TempCompass: Shutterstock ids matched in the `webvid` family (WebVid ids are Shutterstock ids). TOMATO: YouTube source ids in `youtube`; CLEVRER-sourced clips only through bench/TOMATO/clevrer_content_match.tsv (content match); tgif / self-recorded clips have no training family.",
            "- TemporalBench `Oops` clips are named by YouTube compilation title, not YouTube id; `Movie_Description` clips are LSMDC movies. Neither can be matched against YouTube-id training videos (LLaVA youtube, ShareGPTVideo, HT100M). Not checked.",
            "- TemporalBench `EgoExo4D` uses take names; PLM FGQA `egoexo4d` uses take UUIDs. The take_name <-> take_uid map (Ego-Exo4D takes.json) is not on disk, so all PLM FGQA egoexo4d records are removed as ambiguous.",
            "- Ego4D (LLaVA academic, PLM FGQA): no evaluation benchmark here uses Ego4D ids (TemporalBench uses Ego-Exo4D, a separate recording set). Kept.",
            "- LLaVA youtube files named by a 32-hex hash (`liwei_youtube_videos/videos/<md5>`) carry no YouTube id; removed as ambiguous.",
            "- WebVid numeric ids (ShareGPTVideo): no benchmark uses WebVid. Numeric ids equal to an MVP SSv2 id are removed as ambiguous (see reason table).",
            f"- SA-V: {len(sav_done)} of {len(sav_done) + len(sav_pending)} train tars finished downloading and were scanned ({', '.join(sav_done[:1])} .. {sav_done[-1] if sav_done else '-'}); pending: {len(sav_pending)}. "
            f"No benchmark uses SA-V videos. PLM regional records whose SA-V id is not in a scanned train tar carry `sav_train_membership: not_in_scanned_train_tars` "
            f"({plm_not_in_sav:,} records); "
            "kept, since this is an SA-V split question, not benchmark contamination. Spec requires SA-V official train only, so filter on this field before use.",
            "- NVIDIA physical interaction: fully synthetic (Isaac Sim), cannot overlap real-video benchmarks; kept at shard level without member scan.",
            "- Synthetic/lab benchmark families (MVP clevrer/intphys/inflevel/grasp/language_table, IntPhys2, Video-MME-Logical): ids extracted (counts above) but no training source carries these families, so 0 overlap is by construction.",
            f"- ShareGPTVideo image-only files not processed (no video): {', '.join(skipped_img)}.",
            "- Video content duplicated under a different id (re-uploads, trims of the same footage) is not detectable by id matching.", "",
            "## Id-format assumptions", "",
            "- YouTube id = 11 chars `[A-Za-z0-9_-]`, case-sensitive; `v_` prefix (ActivityNet), `ytb_` prefix (LLaVA youtube) and `-Scene-NNN` suffix (ShareGPTVideo) are stripped.",
            "- LLaVA `hdvila/<11 chars>.{mkv,mp4}` and YouCook2 `<recipe>/<11 chars>/split_k.mp4` are treated as YouTube ids.",
            "- ShareGPTVideo all-digit ids (1-10 digits) are WebVid ids; 11-char ids are YouTube.",
            "- Charades id = 5 chars `[A-Z0-9]`; MVP `star` videos are Charades ids.",
            "- MVP `vinoground` N_pos/N_neg are mapped to YouTube ids through `vinoground.csv`; all Vinoground rows are excluded, not only MVP's subset.",
            "- TemporalBench clip names `<id>_start_<s>_end_<e>.mp4`; FineGym `<yt>_E_..._A_...` (short) and `Finegym_<yt>_start_...` (long).",
            "- NExT-QA id = trailing numeric file stem (VidOR id), folder prefix dropped.",
            "- SSv2 id = integer file stem (leading zeros dropped)."]
    open(f"{out}/report.md", "w").write("\n".join(rep) + "\n")

    print("\n== kept / removed")
    print("\n".join(fmt_table(tally, groups)))
    print("\n== formerly unchecked (nextqa family / activitynet provenance, outside policy folders): kept / removed")
    for g in groups:
        if tally.resolved[(g, "kept")] + tally.resolved[(g, "removed")]:
            print(f"  {g:52s} {tally.resolved[(g, 'kept')]:>9,} {tally.resolved[(g, 'removed')]:>7,}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
