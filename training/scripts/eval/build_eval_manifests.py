"""Build /data/llama_vjepa_lm/manifests/eval/<bench>.jsonl and README.md from jepa_llm.eval.adapters.

Counts are printed for every benchmark/split first; a benchmark with 0 items aborts before anything is written.
A small random sample of resolvable videos per benchmark is opened with PyAV (first frame decoded) as a smoke check.

usage: python -I build_eval_manifests.py [--out /data/llama_vjepa_lm/manifests/eval] [--probe 20] [--only bench ...]
With --only, README.md keeps every other benchmark's row (md5 included) and only the rebuilt rows are replaced/added.
"""
import argparse
import hashlib
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from jepa_llm.eval.adapters import ADAPTERS, load_items, write_jsonl  # noqa

NOTES = {
    "perception_test": "valid split, mc_question_valid (19,140 Q / 5,260 videos). cut_frame_mapping_valid.json applied: "
                       "for cut videos only frames [0, cut_frame) may be decoded; cut_sec = cut_frame / metadata fps "
                       "(extra.cut_frame kept). Metric: top-1 + calc_top1_by_cat (area via TAG_AREA, reasoning, tag).",
    "videomme_v2": "test.parquet, 800 groups x 4 Q, options A-H (2-8 per Q). RGB only: no subtitles/audio; Q with "
                   "second_head 'Frames & Audio' flagged extra.audio_category. Metric: official grouped non-linear "
                   "score (relevance quadratic, logic first-error truncation) + question accuracy.",
    "videomme_logical": "three_level_testset.json. Official protocol = generation with <answer></answer>, normalised "
                        "exact match (eval.py); question = official prompt verbatim; options only where the prompt lists "
                        "'A. ...' lines. extra.requires_image (450 items: hidden_container, 3d_maze_route, "
                        "rotation_center); extra.video_only marks the 3,300-item video-only subset.",
    "mvp_mini": "MVP mini (lmms-eval mvp_mini). Video paths mapped per MVP Makefile (see mvp_videos/path_map.tsv). "
                "Metric: paired accuracy per subset, overall = unweighted mean of the 4 subsets; single accuracy too. "
                "ssv2 -> original .webm (setup_ssv2.py's lossy mp4 re-encode skipped); intphys -> scene PNGs rendered "
                "at 24 fps (setup_intphys.py) and language_table -> jpg frames at 5 fps (merge_lt.py, id-parsing bug "
                "fixed; the official script writes 0 videos) under bench/mvp_videos/generated. temporal_reasoning has "
                "pairs whose two candidates are identical (extra.duplicate_candidates; official answer index = 0).",
    "mvp_full": "MVP full (lmms-eval mvp). 28 exact duplicate rows (14 robot pairs listed twice) dropped. Same metric.",
    "temporalbench_short_qa": "temporalbench_short_qa.json; videos extracted from short_video.zip. Metric: binary "
                              "accuracy + multiple binary accuracy (group = video_name), per dataset/category.",
    "temporalbench_long_qa": "temporalbench_long_qa.json; videos from long_video_*.zip. Same metrics (no categories).",
    "intphys2": "Main (1,012) + Debug (60), one item per video, options [Yes, No] (Yes = possible), official MLLM "
                "prompt 0. Metric: single-video accuracy per split + notebook subsets (difficulty env lists, camera "
                "game lists, property; the notebook's Hard list has 'None', which never matches env 'None_0', so 72 videos "
                "are in no difficulty subset). group_id = scene; extra.pair_id for surprise-pair scoring (auxiliary).",
    "intphys2_heldout": "HeldOut 344 videos, labels/metadata not public: answer_index null, leaderboard only.",
    "tempcompass_mc": "TempCompass multi-choice: questions/multi-choice.json of github.com/llyx97/TempCompass e1b4631 "
                      "(identical, same order, to the lmms-lab/TempCompass 'multi-choice' parquet lmms-eval loads), "
                      "1,580 Q / 410 videos from tempcompass_videos.zip (204 original, 95 reversed, 54 speed-concat, "
                      "57 two-clip concat; extra.video_kind). 2-4 options, letters stripped (3 released questions have "
                      "malformed option lines ' D. ..' / 'C...', parsed; extra.option_lines_irregular). Metric: accuracy "
                      "over all questions ('avg' = the multi-choice accuracy V-JEPA 2 reports, 76.9) + 5 temporal aspects "
                      "(extra.dim) + 11 fine-grained aspects (meta_info, extra.fine_dim). Official scorer: hand-crafted "
                      "rules (exact answer string, bare letter, 'X.' / 'X)' prefix), unmatched answers judged by "
                      "gpt-3.5-turbo-1106. The likelihood runner always yields a letter, so only the rule path applies "
                      "(identical to --disable_llm); free text is rated by the same rules with unmatched = 0 "
                      "(n_unmatched), the LLM judge is not reproduced. Official prompt = question_raw + "
                      "'\\nPlease directly give the best option:' (lmms-eval); the runner uses its common suffix. "
                      "Contamination: the 207 Shutterstock (= WebVid) source ids are in no stage-1 training manifest.",
    "tomato": "TOMATO: data/<reasoning type>.json of github.com/yale-nlp/TOMATO fe2025f (same 1,484 Q as the yale-nlp/TOMATO "
              "and lmms-eval/TOMATO parquets), 1,397 distinct videos videos/<demonstration_type>/<key>.mp4 from the official "
              "Google Drive videos.zip (1,417 clips, 20 unused; every member CRC/size-identical to the lmms-eval/TOMATO zip "
              "parts). 3-7 options. Metric: accuracy over all questions (get_categorized_score: summed Score / Count; the "
              "number V-JEPA 2 reports, 40.3) + 6 reasoning types + 3 demonstration types + cells. Official inference: 16 "
              "uniformly sampled frames, generation parsed by regex pre_parser then a GPT-4o-mini parser, unparsable -> "
              "random letter; here the runner's option index is scored directly, None counts wrong (n_invalid). The 50 "
              "'CLEVRER' clips are CLEVRER validation video_10000-10049 (frame-identical; bench/TOMATO/clevrer_content_match.tsv). "
              "Contamination: YouTube 8VPiqCCOPWg and P6UwokP8BEg (ActivityNet scene clips) are in stage-1 sharegpt "
              "(84 rows), llava (36) and capcontrast_sharegpt (6): 4 questions (rotation/54, rotation/79, rotation/194, "
              "velocity&frequency/161; videos object/0285-00, object/0261-01, object/0261-02). CLEVRER clips are "
              "validation (training uses train): 0. Other stage-1 manifests: 0.",
    "perception_test_test": "PerceptionTest TEST split, mc_question_test.json (mc_question_test_annotations.zip): 11,528 Q / "
                            "3,146 videos (test_videos.zip), 3 options each. Answers, areas, reasoning and tags are not "
                            "released: answer_index / answer_text null, no local metric. Cut: no cut mapping is released and "
                            "the released test files are NOT cut; for 222 videos (1,448 Q, all occlusion/cup games) the file "
                            "is longer than metadata.num_frames by >5 frames (median ratio 0.52; on valid the metadata equals "
                            "the file length), so metadata.num_frames is used as the cut: cut_sec = num_frames / frame_rate, "
                            "extra.cut_frame (file lengths in bench/perception_test_test/file_num_frames.json). lmms-eval "
                            "applies no cut. Scoring only through the eval server: "
                            "repo/scripts/eval/write_pt_test_submission.py writes {video_id: [{id, answer_id, answer}]} for "
                            "eval.ai challenge 2091 (Test Phase; closed 2026-01-31 per the eval.ai API on 2026-10-10). "
                            "Contamination: no test id in any stage-1 training manifest.",
}


def probe(path):
    import av
    try:
        with av.open(path) as c:
            for _ in c.decode(video=0):
                return True
    except Exception:
        return False
    return False


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/llama_vjepa_lm/manifests/eval")
    ap.add_argument("--probe", type=int, default=20)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    benches = a.only or list(ADAPTERS)
    built, rows = {}, []
    for b in benches:
        items = load_items(b, check_files=True)
        built[b] = items
        per_split = Counter(it["split"] for it in items)
        vids = {it["video"] for it in items}
        ok_vids = {it["video"] for it in items if it["video_exists"]}
        n_ok = sum(bool(it["video_exists"]) for it in items)
        print(f"[count] {b}: items={len(items)} splits={dict(per_split)} unique_videos={len(vids)} "
              f"resolvable_items={n_ok} resolvable_videos={len(ok_vids)} missing_items={len(items) - n_ok}", flush=True)
        if b == "mvp_mini" or b == "mvp_full":
            sub = Counter((it["extra"]["subset"], it["extra"]["source"], bool(it["video_exists"])) for it in items)
            for k in sorted(sub):
                print(f"         {k}: {sub[k]}", flush=True)
        if b == "videomme_logical":
            print(f"         requires_image={sum(it['extra']['requires_image'] for it in items)} "
                  f"video_only={sum(it['extra']['video_only'] for it in items)} "
                  f"images_exist={sum(bool(it['extra']['images_exist']) for it in items if it['extra']['requires_image'])}",
                  flush=True)
    empty = [b for b, its in built.items() if not its]
    if empty:
        raise SystemExit(f"BUG: 0 items for {empty}; nothing written")

    out = Path(a.out)
    rng = random.Random(0)
    for b, items in built.items():
        p = out / f"{b}.jsonl"
        write_jsonl(items, p)
        missing = sorted({it["video"] for it in items if not it["video_exists"]})
        mp = out / f"{b}.missing_videos.txt"
        if missing:
            mp.write_text("\n".join(missing) + "\n")
        elif mp.exists():
            mp.unlink()
        ok_vids = sorted({it["video"] for it in items if it["video_exists"]})
        sample = rng.sample(ok_vids, min(a.probe, len(ok_vids)))
        n_dec = sum(probe(v) for v in sample)
        n_ok = sum(bool(it["video_exists"]) for it in items)
        rows.append((b, len(items), dict(Counter(it["split"] for it in items)), len({it["video"] for it in items}),
                     n_ok, len(ok_vids), len(items) - n_ok, len(missing), f"{n_dec}/{len(sample)}", md5(p)))
        print(f"[write] {p} items={len(items)} decode_probe={n_dec}/{len(sample)}", flush=True)

    lines = [
        "# Evaluation manifests",
        "",
        f"Built {datetime.now().astimezone().isoformat(timespec='seconds')} by `repo/scripts/eval/build_eval_manifests.py` "
        "from `jepa_llm.eval.adapters`; metrics in `jepa_llm.eval.metrics` (tests: `repo/tests/test_eval_metrics.py`).",
        "One JSON object per line, deterministic order. Fields: bench, split, id, video (absolute), video_exists, "
        "start_sec, end_sec, cut_sec, question, options (letters stripped), answer_index, answer_text, group_id, extra.",
        "`resolvable` = file exists and is non-empty at build time; `decode probe` = random resolvable videos whose first "
        "frame PyAV decoded. Missing videos are listed in `<bench>.missing_videos.txt`. Do not report a score on the "
        "resolvable part as the full-benchmark score (spec §10.3).",
        "",
        "| bench | items | splits | unique videos | resolvable items | resolvable videos | missing items | missing videos | decode probe | md5 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        split_s = ", ".join(f"{k}: {v}" for k, v in r[2].items())
        lines.append(f"| {r[0]} | {r[1]} | {split_s} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} | {r[8]} | `{r[9][:12]}` |")
    lines += ["", "## Protocol notes", ""]
    lines += [f"- **{b}**: {NOTES[b]}" for b in built]
    readme = out / "README.md"
    if not a.only:
        readme.write_text("\n".join(lines) + "\n")
        print(f"[write] {readme}", flush=True)
    elif readme.exists():
        # --only: replace / add the rows of the rebuilt benchmarks, keep every other row and its md5 verbatim
        old = readme.read_text().splitlines()
        hdr = next(i for i, l in enumerate(old) if l.startswith("| bench |"))
        end = next(i for i in range(hdr + 2, len(old) + 1) if i == len(old) or not old[i].startswith("| "))
        table = {l.split(" | ")[0][2:]: l for l in old[hdr + 2:end]}
        for l in lines[lines.index(old[hdr]) + 2:]:
            if not l.startswith("| "):
                break
            table[l.split(" | ")[0][2:]] = l
        order = [b for b in ADAPTERS if b in table] + [b for b in table if b not in ADAPTERS]
        stamp = (f"Rows {', '.join(built)} rebuilt {datetime.now().astimezone().isoformat(timespec='seconds')} "
                 f"with `--only` (other rows unchanged).")
        head = list(old[:hdr])
        i_last = max(i for i, l in enumerate(head) if l.startswith(("Built ", "Rows ")))
        head.insert(i_last + 1, stamp)  # earlier "Rows ... rebuilt" stamps are kept
        notes = ["", "## Protocol notes", ""] + [f"- **{b}**: {NOTES[b]}" for b in order if b in NOTES]
        readme.write_text("\n".join(head + old[hdr:hdr + 2] + [table[b] for b in order] + notes) + "\n")
        print(f"[write] {readme} (rows {list(built)} replaced/added)", flush=True)


if __name__ == "__main__":
    main()
