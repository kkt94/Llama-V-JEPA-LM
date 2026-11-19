"""Benchmark adapters: each benchmark -> a deterministic list of unified evaluation items.

Unified item (one JSON object per question / per classified video):

    bench        manifest name (perception_test, videomme_v2, videomme_logical, mvp_mini, mvp_full,
                 temporalbench_short_qa, temporalbench_long_qa, intphys2, intphys2_heldout, tempcompass_mc,
                 tomato, perception_test_test)
    split        benchmark split / subset
    id           unique within the manifest
    video        absolute path of the file to decode (RGB only; no subtitles / audio are attached)
    video_exists the file exists and is non-empty (checked when the item list is built)
    start_sec, end_sec
                 visible span when the protocol restricts it (null = whole file)
    cut_sec      PerceptionTest cut point: only frames [0, cut_frame) may be shown (null = no cut)
    question     question text (for open-answer benchmarks: the official prompt, verbatim)
    options      option strings with letters stripped, in official order (null for open answers)
    answer_index index into options (null when not a fixed-option item or labels are not public)
    answer_text  ground-truth string as released (letter, option text, integer, JSON ...)
    group_id     unit for grouped / paired metrics and for bootstrap resampling (spec §10.4)
    extra        benchmark-specific fields needed by metrics.py and for reporting

The official prompts, answer extraction and scoring live in metrics.py; adapters only describe the data.
Nothing here writes into the benchmark directories.
"""
from __future__ import annotations

import ast
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable

BENCH = Path(os.environ.get("JEPA_LLM_BENCH", "/data/llama_vjepa_lm/bench"))
MVP_SRC = BENCH / "mvp_sources" / "extracted"
MVP_VIDEOS = BENCH / "mvp_videos"  # derived files only (generated mp4 + symlink tree); never the sources


def _exists(p: str | Path) -> bool:
    try:
        return os.path.getsize(p) > 0
    except OSError:
        return False


def make_item(bench: str, split: str, id: str, video: str | Path, question: str, options: list[str] | None,
              answer_index: int | None, answer_text: str | None, group_id: str, *, start_sec: float | None = None,
              end_sec: float | None = None, cut_sec: float | None = None, extra: dict | None = None,
              check_files: bool = True) -> dict[str, Any]:
    video = str(video)
    return {
        "bench": bench, "split": split, "id": id, "video": video,
        "video_exists": _exists(video) if check_files else None,
        "start_sec": start_sec, "end_sec": end_sec, "cut_sec": cut_sec,
        "question": question, "options": options, "answer_index": answer_index, "answer_text": answer_text,
        "group_id": group_id, "extra": extra or {},
    }


_LETTER_LINE = re.compile(r"^([A-Z])\. (.*)$")


def split_lettered_options(lines: list[str]) -> list[str]:
    """['A. foo', 'B. bar'] -> ['foo', 'bar']; letters must run A, B, C, ... in order."""
    out = []
    for i, line in enumerate(lines):
        m = _LETTER_LINE.match(line)
        if not m or m.group(1) != chr(65 + i):
            raise ValueError(f"option line {i} is not '{chr(65 + i)}. ...': {line!r}")
        out.append(m.group(2))
    return out


# ---------------------------------------------------------------------------------------------------------------
# PerceptionTest validation, multiple-choice video QA
# ---------------------------------------------------------------------------------------------------------------

def perception_test(root: Path = BENCH / "perception_test", check_files: bool = True) -> list[dict]:
    """19,140 questions over 5,260 validation videos. cut_frame_mapping_valid.json is applied per video:
    -1 means no cut; otherwise only frames [0, cut_frame) may be decoded (cut_sec = cut_frame / metadata fps)."""
    ann = json.loads((root / "extracted" / "mc_question_valid.json").read_text())
    cut = json.loads((root / "cut_frame_mapping_valid.json").read_text())
    items = []
    for vid in sorted(ann, key=lambda v: int(v.split("_")[1])):
        meta = ann[vid]["metadata"]
        cf = int(cut[vid])  # KeyError here is a real error: every mc video must have a cut entry
        fps = float(meta["frame_rate"])
        for qi, q in enumerate(ann[vid]["mc_question"]):
            assert q["id"] == qi, (vid, qi)  # official scorer indexes mc_question by 'id'
            items.append(make_item(
                "perception_test", "valid", f"{vid}#{qi}", root / "extracted" / "videos" / f"{vid}.mp4",
                q["question"], list(q["options"]), int(q["answer_id"]), q["options"][q["answer_id"]], vid,
                cut_sec=(cf / fps) if cf >= 0 else None, check_files=check_files,
                extra={"video_id": vid, "question_id": qi, "cut_frame": cf, "frame_rate": fps,
                       "num_frames": int(meta["num_frames"]), "area": q["area"], "reasoning": q["reasoning"],
                       "tags": list(q["tag"]), "is_cup_game": int(meta["is_cup_game"])}))
    return items


# ---------------------------------------------------------------------------------------------------------------
# Video-MME-v2 (grouped non-linear score)
# ---------------------------------------------------------------------------------------------------------------

def videomme_v2(root: Path = BENCH / "Video-MME-v2", check_files: bool = True) -> list[dict]:
    """3,200 questions = 800 groups (one per video) of 4. Order is the parquet order, which the official scorer
    relies on (it groups consecutive rows by 4). RGB-only: subtitles are never attached; questions whose
    second_head is 'Frames & Audio' are flagged because they were written for audio+frames input."""
    import pandas as pd

    df = pd.read_parquet(root / "test.parquet")
    items = []
    for i, r in enumerate(df.itertuples(index=False)):
        pos = int(r.question_id.split("-")[-1])
        assert r.question_id == f"{r.video_id}-{i % 4 + 1}", (i, r.question_id)
        options = split_lettered_options(r.options.split("\n"))
        ai = ord(r.answer) - 65
        assert 0 <= ai < len(options), (r.question_id, r.answer)
        items.append(make_item(
            "videomme_v2", "test", r.question_id, root / "extracted" / f"{r.video_id}.mp4", r.question, options, ai,
            r.answer, r.video_id, check_files=check_files,
            extra={"video_id": r.video_id, "question_pos": pos, "group_type": r.group_type,
                   "group_structure": r.group_structure, "level": int(r.level), "second_head": r.second_head,
                   "third_head": r.third_head, "url": r.url, "options_raw": r.options,
                   "audio_category": r.second_head == "Frames & Audio"}))
    return items


# ---------------------------------------------------------------------------------------------------------------
# Video-MME-Logical (open answers, exact match)
# ---------------------------------------------------------------------------------------------------------------

def _logical_answer_format(answer: str, options: list[str] | None, requires_image: bool) -> str:
    a = answer.strip()
    if a.startswith("{") or a.startswith("["):
        return "json"
    if re.fullmatch(r"-?\d+", a):
        return "integer"
    if re.fullmatch(r"[A-Z]", a):
        if options:
            return "letter_text_options"
        return "letter_image_options" if requires_image else "letter_position_label"
    return "string"


def _logical_path_values(value: object) -> list[str]:
    """image_paths is a list for hidden_container but a plain string for 3d_maze_route / rotation_center; same
    handling as eval.py iter_path_values."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v or "").strip() for v in value if str(v or "").strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                return _logical_path_values(json.loads(s))
            except json.JSONDecodeError:
                return [s]
        return [s]
    return [str(value).strip()] if str(value).strip() else []


def videomme_logical(root: Path = BENCH / "video-mme-logical" / "extracted" / "video-mme-logical",
                     check_files: bool = True) -> list[dict]:
    """3,750 test items. The official protocol is free generation with <answer>...</answer> and normalised exact
    match (eval.py), so `question` is the official prompt verbatim and `answer_text` is the target. `options` is
    filled only when the prompt lists 'A. ...' lines (then answer_index is set too). 450 items (hidden_container,
    3d_maze_route, rotation_center) need reference images: extra.requires_image / extra.video_only mark them."""
    data = json.loads((root / "three_level_testset.json").read_text())
    items = []
    for x in data:
        lines = x["question"].split("\n")
        opt_lines = [l for l in lines if _LETTER_LINE.match(l)]
        options = split_lettered_options(opt_lines) if opt_lines else None
        ans = str(x["answer"])
        ai = None
        if options is not None and re.fullmatch(r"[A-Z]", ans.strip()):
            ai = ord(ans.strip()) - 65
            assert ai < len(options), (x["id"], ans)
        imgs = [str(root / p) for p in _logical_path_values(x.get("image_paths"))]
        req = bool(imgs)
        items.append(make_item(
            "videomme_logical", "test", x["id"], root / x["video_path"], x["question"], options, ai, ans, x["id"],
            check_files=check_files,
            extra={"parent_major": x["parent_major"], "major": x["major"], "difficulty": x["difficulty"],
                   "image_paths": imgs, "images_exist": all(_exists(p) for p in imgs) if check_files else None,
                   "requires_image": req, "video_only": not req,
                   "answer_format": _logical_answer_format(ans, options, req)}))
    return items


# ---------------------------------------------------------------------------------------------------------------
# MVP (minimal video pairs, paired accuracy)
# ---------------------------------------------------------------------------------------------------------------

MVP_SUBSETS = ("human_object_interactions", "intuitive_physics", "robot_object_interactions", "temporal_reasoning")


def mvp_intphys_frames_dir(rel: str) -> Path:
    """'_O1_01_1.mp4' -> intphys_dev/dev/O1/01/1/scene (inverse of setup_intphys.py's naming)."""
    prop, scene, vid = rel[1:-len(".mp4")].split("_")
    return MVP_SRC / "intphys_dev" / "dev" / prop / scene / vid / "scene"


def mvp_resolve(video_path: str) -> tuple[Path, str]:
    """MVP video_path (as in the HF parquet, with or without leading '/') -> (file to decode, how it was mapped).

    Follows the MVP Makefile without touching the extracted sources:
      pt          valid_videos.zip 'videos/' renamed to pt/           -> perception_test/extracted/videos
      clevrer     validation zip, 5 sub-folders merged into clevrer/  -> clevrer/video_<k000>-<k000+1000>/
      grasp       videos/ renamed to grasp/                           -> grasp/videos/
      inflevel    inflevel_lab/ renamed to inflevel/                  -> inflevel_lab/inflevel_lab/
      star        Charades_v1_480/ renamed to star/                   -> charades/Charades_v1_480/
      vinoground  vinoground_videos/ renamed to vinoground/           -> vinoground/vinoground_videos/
      ssv2        setup_ssv2.py re-encodes <id>.webm to <id>.mp4      -> the original .webm (no lossy re-encode)
      intphys     setup_intphys.py renders scene PNGs at 24 fps       -> mvp_videos/generated/intphys/ (see script)
      language_table  merge_lt.py renders jpg frames at 5 fps         -> extracted/language_table_slow/ if present,
                                                                         else mvp_videos/generated/language_table/
    """
    vp = video_path.lstrip("/")
    src, rel = vp.split("/", 1)
    if src == "pt":
        return BENCH / "perception_test" / "extracted" / "videos" / rel, "pt_valid_videos"
    if src == "clevrer":
        n = int(re.fullmatch(r"video_(\d+)\.mp4", rel).group(1))
        lo = n // 1000 * 1000
        return MVP_SRC / "clevrer" / f"video_{lo}-{lo + 1000}" / rel, "clevrer_validation"
    if src == "grasp":
        return MVP_SRC / "grasp" / "videos" / rel, "grasp_videos"
    if src == "inflevel":
        return MVP_SRC / "inflevel_lab" / "inflevel_lab" / rel, "inflevel_lab"
    if src == "star":
        return MVP_SRC / "charades" / "Charades_v1_480" / rel, "charades_480"
    if src == "vinoground":
        return MVP_SRC / "vinoground" / "vinoground_videos" / rel, "vinoground_videos"
    if src == "ssv2":
        return MVP_SRC / "ssv2" / "20bn-something-something-v2" / (rel[: -len(".mp4")] + ".webm"), "ssv2_webm_original"
    if src == "intphys":
        return MVP_VIDEOS / "generated" / "intphys" / rel, "intphys_png_to_mp4_24fps"
    if src == "language_table":
        official = MVP_SRC / "language_table_slow" / rel
        if _exists(official):
            return official, "language_table_slow"
        return MVP_VIDEOS / "generated" / "language_table" / rel, "language_table_jpg_to_mp4_5fps"
    raise ValueError(f"unknown MVP source in {video_path!r}")


def mvp(split: str, root: Path = BENCH / "minimal_video_pairs", check_files: bool = True) -> list[dict]:
    """split in {'mini', 'full'}. Items keep parquet order per subset (subsets in MVP_SUBSETS order). Exact duplicate
    rows (28 in robot_object_interactions/full, 14 pairs listed twice) are dropped: the official compute_metrics
    unpacks exactly two items per pair and would fail on them. group_id = subset/<video_id without trailing _1/_2>."""
    import pandas as pd

    items = []
    for subset in MVP_SUBSETS:
        df = pd.read_parquet(root / subset / f"{split}-00000-of-00001.parquet")
        n0 = len(df)
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
        dropped = n0 - len(df)
        assert not df.video_id.duplicated().any(), (subset, split, "non-identical rows share a video_id")
        for r in df.itertuples(index=False):
            cands = ast.literal_eval(r.candidates) if isinstance(r.candidates, str) else list(r.candidates)
            ai = cands.index(str(r.answer))  # official: cand.index(str(doc['answer']))
            path, method = mvp_resolve(r.video_path)
            stem, pos = r.video_id.rsplit("_", 1)
            items.append(make_item(
                f"mvp_{split}", split, f"{subset}/{r.video_id}", path, r.question, [str(c) for c in cands], ai,
                str(r.answer), f"{subset}/{stem}", check_files=check_files,
                extra={"subset": subset, "source": r.source, "video_id": r.video_id, "pair_pos": int(pos),
                       "mvp_video_path": r.video_path, "resolve": method,
                       "duplicate_candidates": len(set(cands)) < len(cands),
                       "subset_rows_dropped_as_exact_duplicates": dropped}))
    return items


# ---------------------------------------------------------------------------------------------------------------
# TemporalBench (binary QA; binary accuracy + multiple binary accuracy)
# ---------------------------------------------------------------------------------------------------------------

TB_INSTR = "Answer with the option's letter from the given choices directly."


def temporalbench(kind: str, root: Path = BENCH / "TemporalBench", check_files: bool = True) -> list[dict]:
    """kind in {'short_qa', 'long_qa'}. Videos come from the zips extracted to <root>/extracted/. Each question is
    'stem\\nA. caption\\nB. caption\\n<instruction>'; group_id = video_name, the official multiple-binary unit."""
    data = json.loads((root / f"temporalbench_{kind}.json").read_text())
    items = []
    for x in data:
        lines = x["question"].split("\n")
        assert len(lines) == 4 and lines[-1] == TB_INSTR, x["idx"]
        options = split_lettered_options(lines[1:3])
        items.append(make_item(
            f"temporalbench_{kind}", kind, x["idx"], root / "extracted" / x["video_name"], lines[0], options,
            ord(x["GT"]) - 65, x["GT"], x["video_name"], check_files=check_files,
            extra={"dataset": x["dataset"], "category": x.get("category"), "video_name": x["video_name"],
                   "question_raw": x["question"]}))
    return items


# ---------------------------------------------------------------------------------------------------------------
# IntPhys2 (possible / impossible, evaluation only)
# ---------------------------------------------------------------------------------------------------------------

# Prompt 0 of the official IntPhys2_transformers.py; scored by label_pred_yes_no (yes -> possible).
INTPHYS2_PROMPT = ("The video you’re seeing was generated by a simulator. Given how objects behave on earth, is the "
                   "behaviour of the objects plausible? Your answer should be based on the events in the video and "
                   "ignore the quality of the simulation. You should also ignore the framerate of the video, as it may "
                   "be slowed down. Finish your answer by Yes or No")
INTPHYS2_OPTIONS = ["Yes", "No"]  # index 0 = possible (label 1), index 1 = impossible (label 0)


def intphys2(root: Path = BENCH / "IntPhys2", check_files: bool = True) -> list[dict]:
    """Main (1,012 videos) and Debug (60) with public labels. One item per video (the official MLLM metric is
    single-video accuracy). group_id = split/SceneIndex (4 videos per scene); extra.pair_id groups each
    possible/impossible pair (1_*, 2_*) for surprise-based relative scoring."""
    items = []
    for split in ("Main", "Debug"):
        with open(root / split / "metadata.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            num, kind = r["type"].split("_")
            assert kind in ("Possible", "Impossible"), r
            label = 1 if kind == "Possible" else 0
            items.append(make_item(
                "intphys2", split.lower(), f"{split}/{r['name']}", root / split / r["file_name"], INTPHYS2_PROMPT,
                list(INTPHYS2_OPTIONS), 0 if label == 1 else 1, "Yes" if label == 1 else "No",
                f"{split}/{r['SceneIndex']}", check_files=check_files,
                extra={"name": r["name"], "label_possible": label, "type": r["type"],
                       "pair_id": f"{split}/{r['SceneIndex']}/{num}", "scene_index": r["SceneIndex"],
                       "game_name": r["game_name"], "condition": r["condition"], "env": r["env"],
                       "occluder": r["occluder"], "difficulty": r["Difficulty"], "camera": r["Camera"],
                       "labels_public": True}))
    return items


def intphys2_heldout(root: Path = BENCH / "IntPhys2", check_files: bool = True) -> list[dict]:
    """HeldOut: 344 videos, labels and metadata NOT public (leaderboard only). answer_index is null."""
    items = []
    for fn in sorted(os.listdir(root / "HeldOut" / "Videos")):
        if not fn.endswith(".mp4"):
            continue
        name = fn[:-4]
        items.append(make_item(
            "intphys2_heldout", "heldout", f"HeldOut/{name}", root / "HeldOut" / "Videos" / fn, INTPHYS2_PROMPT,
            list(INTPHYS2_OPTIONS), None, None, f"HeldOut/{name}", check_files=check_files,
            extra={"name": name, "labels_public": False}))
    return items


# ---------------------------------------------------------------------------------------------------------------
# PerceptionTest TEST split, multiple-choice video QA (labels hidden; eval.ai submission only)
# ---------------------------------------------------------------------------------------------------------------

PT_TEST_CUT_TOLERANCE = 5  # frames; container nb_frames and metadata num_frames differ by 1-5 on ~2% of uncut videos


def perception_test_test(root: Path = BENCH / "perception_test_test", check_files: bool = True) -> list[dict]:
    """11,528 questions over 3,146 test videos (mc_question_test.json from mc_question_test_annotations.zip). Answers,
    areas, reasoning types and tags are not released: answer_index / answer_text are null and the split is scored
    only by the eval server (scripts/eval/write_pt_test_submission.py).

    Cut: no cut_frame_mapping is released for test and the README says test videos are cut where the end gives the
    answer away, but the files in test_videos.zip are NOT cut: for 222 videos (1,448 questions, all occlusion /
    cup games) the file has far more frames than metadata.num_frames (median ratio 0.52), while on valid the metadata
    equals the file length even for the 362 cut videos. So metadata.num_frames is the cut length here: when the file
    (file_num_frames.json, ffprobe nb_frames) exceeds it by more than PT_TEST_CUT_TOLERANCE frames, only frames
    [0, num_frames) may be shown: cut_sec = num_frames / frame_rate, extra.cut_frame = num_frames (else -1)."""
    ann = json.loads((root / "extracted" / "mc_question_test.json").read_text())
    file_frames = json.loads((root / "file_num_frames.json").read_text())["nb_frames"]
    items = []
    for vid in sorted(ann, key=lambda v: int(v.split("_")[1])):
        meta = ann[vid]["metadata"]
        assert meta["split"] == "test" and meta["video_id"] == vid, vid
        fps, nf = float(meta["frame_rate"]), int(meta["num_frames"])
        cf = nf if int(file_frames[vid]) - nf > PT_TEST_CUT_TOLERANCE else -1
        for qi, q in enumerate(ann[vid]["mc_question"]):
            assert q["id"] == qi, (vid, qi)  # the submission is keyed by video id + question 'id'
            items.append(make_item(
                "perception_test_test", "test", f"{vid}#{qi}", root / "extracted" / "videos" / f"{vid}.mp4",
                q["question"], list(q["options"]), None, None, vid, cut_sec=(cf / fps) if cf >= 0 else None,
                check_files=check_files,
                extra={"video_id": vid, "question_id": qi, "frame_rate": fps, "num_frames": nf,
                       "file_num_frames": int(file_frames[vid]), "cut_frame": cf,
                       "cut_source": "metadata_num_frames" if cf >= 0 else None, "labels_public": False}))
    return items


# ---------------------------------------------------------------------------------------------------------------
# TempCompass multi-choice QA
# ---------------------------------------------------------------------------------------------------------------

TEMPCOMPASS_DIMS = ("action", "speed", "direction", "order", "attribute_change")  # lmms-eval metric order


def tempcompass_source_ids(video_id: str) -> list[str]:
    """TempCompass video id -> Shutterstock (= WebVid) ids of the raw clips it was built from, following
    utils/download_video.py: '<id>', '<id>_reverse', '<id>_concat_<k>' -> [id]; '<id1>_<id2>_<k>' -> [id1, id2]."""
    parts = video_id.split("_")
    if len(parts) == 1:
        out = parts
    elif parts[1] in ("reverse", "concat"):
        out = parts[:1]
    else:
        assert len(parts) == 3, video_id
        out = parts[:2]
    assert all(re.fullmatch(r"\d+", p) for p in out), video_id
    return out


def _tempcompass_kind(video_id: str) -> str:
    parts = video_id.split("_")
    if len(parts) == 1:
        return "original"
    return {"reverse": "reverse", "concat": "speed_concat"}.get(parts[1], "pair_concat")


def tempcompass_mc(root: Path = BENCH / "TempCompass", check_files: bool = True) -> list[dict]:
    """1,580 multi-choice questions over 410 videos (official questions/multi-choice.json, GitHub llyx97/TempCompass
    e1b4631; identical, in the same order, to the lmms-lab/TempCompass 'multi-choice' parquet that lmms-eval loads).
    question = 'stem\\nA. ..\\nB. ..' in the release: the stem goes to `question`, the lettered lines to `options`
    (extra.question_raw keeps the official string). answer 'C. text' -> answer_index 2. id = video/dim/k with k the
    position within (video, dim). extra.fine_dim is the fine-grained aspect of eval_utils.print_result (meta_info)."""
    gh = root / "github_e1b4631"
    qs = json.loads((gh / "questions" / "multi-choice.json").read_text())
    meta = json.loads((gh / "meta_info.json").read_text())
    items = []
    for vid, dims in qs.items():
        for dim, lst in dims.items():
            assert dim in TEMPCOMPASS_DIMS, (vid, dim)
            fine = "order" if dim == "order" else meta[vid]["eval_dim"][dim]["type"]
            for k, q in enumerate(lst):
                lines = q["question"].split("\n")
                # 3 released questions have ' D. x' or 'C.x' option lines; letters are still in A, B, C order
                opt = [re.fullmatch(r"\s*([A-D])\.\s*(\S.*)", l) for l in lines[1:]]
                assert all(m and m.group(1) == chr(65 + i) for i, m in enumerate(opt)), (vid, dim, k)
                options = [m.group(2) for m in opt]
                irregular = any(not _LETTER_LINE.match(l) for l in lines[1:])
                ai = ord(q["answer"][0]) - 65
                assert q["answer"] == f"{chr(65 + ai)}. {options[ai]}", (vid, dim, k)
                items.append(make_item(
                    "tempcompass_mc", "test", f"{vid}/{dim}/{k}", root / "extracted" / "videos" / f"{vid}.mp4",
                    lines[0], options, ai, q["answer"], vid, check_files=check_files,
                    extra={"video_id": vid, "dim": dim, "fine_dim": fine, "question_raw": q["question"],
                           "video_kind": _tempcompass_kind(vid), "source_ids": tempcompass_source_ids(vid),
                           "option_lines_irregular": irregular,
                           "content": meta[vid].get("content")}))
    return items


# ---------------------------------------------------------------------------------------------------------------
# TOMATO (visual temporal reasoning, multiple choice)
# ---------------------------------------------------------------------------------------------------------------

TOMATO_REASONING_TYPES = ("count", "direction", "rotation", "shape&trend", "velocity&frequency", "visual_cues")
TOMATO_DEMONSTRATION_TYPES = ("human", "object", "simulated")  # src/config.json order


def tomato_source(url: str) -> tuple[str, str] | None:
    """video_source_url -> (family, id) for the contamination check: YouTube watch/shorts/youtu.be URLs -> youtube id;
    'CLEVRER' / 'tgif' are dataset names without a clip id; self-recorded videos have an empty URL."""
    if not url:
        return None
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])", url)
    if m:
        return "youtube", m.group(1)
    return "url", url


def tomato(root: Path = BENCH / "TOMATO", check_files: bool = True) -> list[dict]:
    """1,484 questions over 1,417 videos (GitHub yale-nlp/TOMATO fe2025f data/<reasoning type>.json; the order is
    the official config order of reasoning types, then the json order). Video = videos/<demonstration_type>/<key>.mp4
    (src/generate_lib/*.py), from the official videos.zip (Google Drive). id = '<reasoning type>/<json key>'.
    The same (demonstration_type, key) video can carry several questions: group_id is that video."""
    data = root / "github_fe2025f" / "data"
    items = []
    for rt in TOMATO_REASONING_TYPES:
        for qid, x in json.loads((data / f"{rt}.json").read_text()).items():
            dt = x["demonstration_type"]
            assert dt in TOMATO_DEMONSTRATION_TYPES, (rt, qid, dt)
            ai = int(x["answer"])
            assert 0 <= ai < len(x["options"]), (rt, qid)
            src = tomato_source(x["video_source_url"])
            items.append(make_item(
                "tomato", "test", f"{rt}/{qid}", root / "extracted" / "videos" / dt / f"{x['key']}.mp4",
                x["question"], list(x["options"]), ai, x["options"][ai], f"{dt}/{x['key']}", check_files=check_files,
                extra={"reasoning_type": rt, "motion_type": x["motion_type"], "demonstration_type": dt, "key": x["key"],
                       "question_id": qid, "variation": dict(x["variation"]), "video_source_url": x["video_source_url"],
                       "source_family": src[0] if src else None, "source_id": src[1] if src else None,
                       "note": x["note"]}))
    return items


# ---------------------------------------------------------------------------------------------------------------
# Registry and writer
# ---------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------
# MVBench (OpenGVLab/MVBench) and Video-MME (lmms-lab/Video-MME), added 10-16
# ---------------------------------------------------------------------------------------------------------------

MVBENCH_TASKS = {  # json -> video dir (the official data_list); bound = the json's start / end seconds restrict the clip
    "action_sequence": ("star/Charades_v1_480", True), "action_prediction": ("star/Charades_v1_480", True),
    "action_antonym": ("ssv2_video", False), "fine_grained_action": ("Moments_in_Time_Raw/videos", False),
    "unexpected_action": ("FunQA_test/test", False), "object_existence": ("clevrer/video_validation", False),
    "object_interaction": ("star/Charades_v1_480", True), "object_shuffle": ("perception/videos", False),
    "moving_direction": ("clevrer/video_validation", False), "action_localization": ("sta/sta_video", True),
    "scene_transition": ("scene_qa/video", False), "action_count": ("perception/videos", False),
    "moving_count": ("clevrer/video_validation", False), "moving_attribute": ("clevrer/video_validation", False),
    "state_change": ("perception/videos", False), "fine_grained_pose": ("nturgbd", False),
    "character_order": ("perception/videos", False), "egocentric_navigation": ("vlnqa", False),
    "episodic_reasoning": ("tvqa/mp4", True), "counterfactual_inference": ("clevrer/video_validation", False),
}


def mvbench(root: Path = BENCH / "MVBench", check_files: bool = True) -> list[dict]:
    """4,000 questions, 20 tasks x 200; options in the given candidate order (as the official evaluation). fine_grained_pose needs NTU
    RGB+D (licence; not distributed) and is left out when its videos are missing; a few STAR / CLEVRER clips sit in video/data0613/.
    episodic_reasoning: TVQA frame folders written as 3 fps mp4 (logs/zz_tvqa_frames2mp4.py)."""
    import json
    items = []
    for task, (vdir, bound) in MVBENCH_TASKS.items():
        for k, r in enumerate(json.load(open(root / "json" / f"{task}.json"))):
            v = root / "video" / vdir / (r["video"] + (".mp4" if task == "episodic_reasoning" else ""))
            if not _exists(v):
                alt = root / "video" / "data0613" / vdir / r["video"]
                v = alt if _exists(alt) else v
            if task == "fine_grained_pose" and not _exists(v):
                continue
            opts = list(r["candidates"]); ai = opts.index(r["answer"])
            st, en = (float(r["start"]), float(r["end"])) if bound and r.get("start") is not None else (None, None)
            items.append(make_item("mvbench", "test", f"{task}_{k}", v, r["question"], opts, ai, r["answer"], f"{task}_{k}",
                                   start_sec=st, end_sec=en, check_files=check_files, extra={"task": task}))
    return items


def videomme(root: Path = BENCH / "Video-MME", check_files: bool = True) -> list[dict]:
    """2,700 questions on 900 videos (short / medium / long); RGB only (no subtitles)"""
    import pandas as pd
    df = pd.read_parquet(root / "videomme" / "test-00000-of-00001.parquet"); items = []
    for r in df.itertuples(index=False):
        opts = [re.sub(r"^[A-D]\.\s*", "", o) for o in r.options]; ai = ord(r.answer) - 65
        items.append(make_item("videomme", "test", r.question_id, root / "extracted" / "data" / f"{r.videoID}.mp4", r.question, opts, ai,
                               opts[ai], r.video_id, check_files=check_files,
                               extra={"duration": r.duration, "domain": r.domain, "sub_category": r.sub_category, "task_type": r.task_type}))
    return items


ADAPTERS: dict[str, Callable[..., list[dict]]] = {
    "perception_test": perception_test,
    "videomme_v2": videomme_v2,
    "videomme_logical": videomme_logical,
    "mvp_mini": lambda check_files=True: mvp("mini", check_files=check_files),
    "mvp_full": lambda check_files=True: mvp("full", check_files=check_files),
    "temporalbench_short_qa": lambda check_files=True: temporalbench("short_qa", check_files=check_files),
    "temporalbench_long_qa": lambda check_files=True: temporalbench("long_qa", check_files=check_files),
    "intphys2": intphys2,
    "intphys2_heldout": intphys2_heldout,
    "tempcompass_mc": tempcompass_mc,
    "tomato": tomato,
    "perception_test_test": perception_test_test,
    "mvbench": mvbench,
    "videomme": videomme,
}


def load_items(bench: str, check_files: bool = True) -> list[dict]:
    items = ADAPTERS[bench](check_files=check_files)
    ids = [it["id"] for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{bench}: duplicate item ids")
    return items


def write_jsonl(items: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
