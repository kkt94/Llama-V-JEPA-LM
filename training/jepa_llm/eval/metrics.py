"""Official metrics for the evaluation benchmarks, computed from per-item predictions.

Input: the unified items of adapters.py and ``preds: Mapping[item_id, prediction]``.
  * fixed-option benchmarks: prediction = predicted option index (0-based) or None (no / unparseable answer);
  * Video-MME-Logical: prediction = raw model response text (official <answer> extraction + normalisation is
    applied here), or an option index for items that list options in the prompt.
A missing id counts as an incorrect answer and is reported in n_missing; None is reported in n_invalid. Both are
scored as wrong, as in the official scorers (they never drop questions).

All accuracies are returned in percent (0-100), whatever scale the official script prints.

Sources reproduced (the tests in tests/test_eval_metrics.py compare against them where they can be called):
  PerceptionTest   perception_test/baselines/mc_vqa.ipynb  calc_top1, calc_top1_by_cat (AREA/REASONING/TAG_AREA)
  Video-MME-v2     Video-MME-v2/evaluation/test_video_mme_v2.py  cal_relevance_rating, cal_logic_rating,
                   get_final_rating, get_final_acc  (identical rules in lmms-eval tasks/videomme_v2/utils.py)
  MVP              minimal_video_pairs/tasks/mvp/utils.py  compute_metrics; group score = unweighted mean of subsets
                   (mvp.yaml / mvp_mini.yaml: aggregation mean, weight_by_size false)
  TemporalBench    TemporalBench/get_qa_acc.py (= lmms-eval temporalbench_aggregate_results): binary accuracy,
                   multiple binary accuracy (all questions of a video_name correct), per dataset / category
  Video-MME-Logical  video-mme-logical/eval.py  extract_answer, normalize_answer, is_correct  (exact match)
  IntPhys2         IntPhys2/ComputeScoreIntPhys2.ipynb  label_target, calculate_accuracy, get_values_singlevideo
                   subsets (env lists for difficulty, game lists for camera, condition lists)
  TempCompass MC   TempCompass/eval_multi-choice.py (rule matching, --disable_llm) + utils/eval_utils.py print_result;
                   lmms-eval tasks/tempcompass/utils.py tempcompass_process_results_multi_choice / aggregate_rating
  TOMATO           TOMATO/src/get_categorized_score.py (get_single_score, grouped sums) and parse_result.pre_parser;
                   lmms-eval tasks/tomato/utils.py tomato_process_results / tomato_aggregate_results
  PerceptionTest test  no labels: perception_test_official_answers() builds the eval.ai submission
                   (scripts/eval/write_pt_test_submission.py)
"""
from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from typing import Any, Mapping


def _pct(c: float, n: int) -> float:
    return 100.0 * c / n if n else float("nan")


def _option_correct(items: list[dict], preds: Mapping[str, Any]) -> tuple[list[int], int, int]:
    """Per-item 0/1 correctness for fixed-option items + (n_missing, n_invalid)."""
    out, miss, inv = [], 0, 0
    for it in items:
        if it["answer_index"] is None:
            raise ValueError(f"{it['id']}: no public answer; this item cannot be scored")
        if it["id"] not in preds:
            miss += 1
            out.append(0)
            continue
        p = preds[it["id"]]
        if p is None:
            inv += 1
            out.append(0)
            continue
        p = int(p)
        if not 0 <= p < len(it["options"]):
            raise ValueError(f"{it['id']}: predicted index {p} outside 0..{len(it['options']) - 1}")
        out.append(int(p == it["answer_index"]))
    return out, miss, inv


def _acc_by(items, correct, key) -> dict[str, float]:
    acc = defaultdict(lambda: [0, 0])
    for it, c in zip(items, correct):
        k = key(it)
        acc[k][0] += c
        acc[k][1] += 1
    return {k: _pct(*v) for k, v in sorted(acc.items(), key=lambda kv: str(kv[0]))}


# ---------------------------------------------------------------------------------------------------------------
# PerceptionTest
# ---------------------------------------------------------------------------------------------------------------

PT_AREA = ["physics", "semantics", "abstraction", "memory"]
PT_REASONING = ["descriptive", "counterfactual", "explanatory", "predictive"]
PT_TAG_AREA = {
    "change detection": "memory", "event recall": "memory", "sequencing": "memory", "visual discrimination": "memory",
    "action counting": "abstraction", "event counting": "abstraction", "feature matching": "abstraction",
    "object counting": "abstraction", "pattern breaking": "abstraction", "pattern discovery": "abstraction",
    "collision": "physics", "conservation": "physics", "containment": "physics", "material": "physics",
    "motion": "physics", "occlusion": "physics", "object attributes": "physics", "object permanence": "physics",
    "quantity": "physics", "solidity": "physics", "spatial relations": "physics", "stability": "physics",
    "action recognition": "semantics", "adversarial action": "semantics", "colour recognition": "semantics",
    "distractor action": "semantics", "distractor object": "semantics", "general knowledge": "semantics",
    "language": "semantics", "object recognition": "semantics", "part recognition": "semantics",
    "place recognition": "semantics", "state recognition": "semantics", "task completion": "semantics",
}


def perception_test_metrics(items: list[dict], preds: Mapping[str, Any]) -> dict:
    """top-1 accuracy, plus calc_top1_by_cat: area is derived from the tags through TAG_AREA (each area counted once
    per question), reasoning from the question, and every tag separately."""
    correct, miss, inv = _option_correct(items, preds)
    cat = defaultdict(lambda: [0, 0])
    for it, c in zip(items, correct):
        ex = it["extra"]
        for a in dict.fromkeys(PT_TAG_AREA[t] for t in ex["tags"]):
            cat[("area", a)][0] += c
            cat[("area", a)][1] += 1
        cat[("reasoning", ex["reasoning"])][0] += c
        cat[("reasoning", ex["reasoning"])][1] += 1
        for t in ex["tags"]:
            cat[("tag", t)][0] += c
            cat[("tag", t)][1] += 1
    by = lambda kind: {k[1]: _pct(*v) for k, v in sorted(cat.items()) if k[0] == kind}
    return {"top1": _pct(sum(correct), len(items)), "n": len(items), "n_missing": miss, "n_invalid": inv,
            "n_cut_items": sum(it["cut_sec"] is not None for it in items),
            "by_area": by("area"), "by_reasoning": by("reasoning"), "by_tag": by("tag")}


def perception_test_official_answers(items: list[dict], preds: Mapping[str, int]) -> dict:
    """Predictions in the official results format {video_id: [{'id', 'answer_id', 'answer'}]} (calc_top1 input)."""
    out = defaultdict(list)
    for it in items:
        if it["id"] in preds and preds[it["id"]] is not None:
            p = int(preds[it["id"]])
            out[it["extra"]["video_id"]].append({"id": it["extra"]["question_id"], "answer_id": p,
                                                 "answer": it["options"][p]})
    return dict(out)


# ---------------------------------------------------------------------------------------------------------------
# Video-MME-v2
# ---------------------------------------------------------------------------------------------------------------

def vmme2_relevance_rating(scores: list[int]) -> float:
    score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    return score_map.get(sum(scores), 0.0)


def vmme2_logic_rating(scores: list[int], group_structure: str) -> float:
    """First-error truncation: the chain length counts consecutive correct answers from Q1, with the official
    exceptions for the parallel structures [1,[2,3],4] and [[1,2],3,4]."""
    gs = ast.literal_eval(group_structure)
    last = -1
    for i, v in enumerate(scores):
        if v:
            last = i
        else:
            break
    if gs == [1, 2, 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    elif gs == [1, [2, 3], 4]:
        score_map = {0: 0.0, 1: 100.0 / 12, 2: 100.0 * 4 / 12, 3: 100.0 * 7 / 12, 4: 100.0}
        if last == 0 and scores[2]:
            last += 1
    elif gs == [[1, 2], 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 10, 2: 100.0 * 2 / 10, 3: 100.0 * 5 / 10, 4: 100.0}
        if last == -1 and scores[1]:
            last += 1
    else:
        raise ValueError(f"Unknown group_structure: {group_structure}")
    return score_map.get(last + 1, 0.0)


def videomme_v2_metrics(items: list[dict], preds: Mapping[str, Any]) -> dict:
    """Official grouped non-linear score ('rating') and question accuracy ('acc'), both in percent. A group is the
    4 questions of one video (Q1..Q4); its type/structure/level/heads are taken from Q4, as in get_final_rating."""
    correct, miss, inv = _option_correct(items, preds)
    groups: dict[str, list] = defaultdict(list)
    for it, c in zip(items, correct):
        groups[it["group_id"]].append((it["extra"]["question_pos"], it, c))
    ratings = []
    for gid, g in groups.items():
        g.sort(key=lambda t: t[0])
        if [t[0] for t in g] != [1, 2, 3, 4]:
            raise ValueError(f"group {gid}: expected questions 1..4, got {[t[0] for t in g]}")
        last = g[-1][1]["extra"]
        scores = [t[2] for t in g]
        if last["group_type"] == "relevance":
            r = vmme2_relevance_rating(scores)
        elif last["group_type"] == "logic":
            r = vmme2_logic_rating(scores, last["group_structure"])
        else:
            raise ValueError(last["group_type"])
        ratings.append((last, r))
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0

    def rate_by(key):
        d = defaultdict(list)
        for ex, r in ratings:
            d[key(ex)].append(r)
        return {k: mean(v) for k, v in sorted(d.items())}

    rating = {"overall": mean([r for _, r in ratings]),
              "group_type": {k: mean([r for ex, r in ratings if ex["group_type"] == k]) for k in ("relevance", "logic")},
              "level": {f"level_{k}": v for k, v in rate_by(lambda ex: ex["level"]).items()},
              "second_head": rate_by(lambda ex: ex["second_head"]),
              "third_head": rate_by(lambda ex: ex["third_head"])}
    acc = {"overall": _pct(sum(correct), len(items)),
           "group_type": _acc_by(items, correct, lambda it: it["extra"]["group_type"]),
           "level": {f"level_{k}": v for k, v in _acc_by(items, correct, lambda it: it["extra"]["level"]).items()},
           "second_head": _acc_by(items, correct, lambda it: it["extra"]["second_head"]),
           "third_head": _acc_by(items, correct, lambda it: it["extra"]["third_head"])}
    return {"score": rating["overall"], "rating": rating, "acc": acc, "n": len(items), "n_groups": len(ratings),
            "n_missing": miss, "n_invalid": inv,
            "n_audio_category_items": sum(bool(it["extra"]["audio_category"]) for it in items)}


# ---------------------------------------------------------------------------------------------------------------
# MVP
# ---------------------------------------------------------------------------------------------------------------

def mvp_metrics(items: list[dict], preds: Mapping[str, Any]) -> dict:
    """Per subset: single accuracy and paired accuracy (a pair counts only if both of its items are correct).
    Overall = unweighted mean over the subsets present (the lmms-eval group aggregation for mvp / mvp_mini)."""
    correct, miss, inv = _option_correct(items, preds)
    per = defaultdict(lambda: {"items": [], "pairs": defaultdict(list)})
    for it, c in zip(items, correct):
        s = per[it["extra"]["subset"]]
        s["items"].append(c)
        s["pairs"][it["group_id"]].append(c)
    subsets = {}
    for name, s in sorted(per.items()):
        bad = [g for g, v in s["pairs"].items() if len(v) != 2]
        if bad:
            raise ValueError(f"{name}: {len(bad)} pairs without exactly 2 items, e.g. {bad[0]}")
        pc = sum(all(v) for v in s["pairs"].values())
        subsets[name] = {"pair_accuracy": _pct(pc, len(s["pairs"])), "single_accuracy": _pct(sum(s["items"]), len(s["items"])),
                         "n_items": len(s["items"]), "n_pairs": len(s["pairs"])}
    m = lambda k: sum(v[k] for v in subsets.values()) / len(subsets) if subsets else float("nan")
    return {"pair_accuracy": m("pair_accuracy"), "single_accuracy": m("single_accuracy"), "subsets": subsets,
            "n": len(items), "n_missing": miss, "n_invalid": inv}


# ---------------------------------------------------------------------------------------------------------------
# TemporalBench
# ---------------------------------------------------------------------------------------------------------------

TB_CATEGORY_ORDER = ["Action Order", "Action Frequency", "Action Type", "Motion Magnitude",
                     "Motion Direction/Orientation", "Action Effector", "Event Order", "Others"]


def temporalbench_metrics(items: list[dict], preds: Mapping[str, Any]) -> dict:
    """Binary accuracy (per question) and multiple binary accuracy (a video_name is correct only if every question
    on it is), overall, per source dataset and (short QA) per category."""
    correct, miss, inv = _option_correct(items, preds)

    def both(sel):
        qs = [(it, c) for it, c in zip(items, correct) if sel(it)]
        vid = {}
        for it, c in qs:
            vid[it["group_id"]] = vid.get(it["group_id"], True) and bool(c)
        return {"binary_accuracy": _pct(sum(c for _, c in qs), len(qs)),
                "multiple_binary_accuracy": _pct(sum(vid.values()), len(vid)), "n_questions": len(qs),
                "n_videos": len(vid)}

    out = both(lambda it: True)
    out["by_dataset"] = {d: both(lambda it, d=d: it["extra"]["dataset"] == d)
                         for d in sorted({it["extra"]["dataset"] for it in items})}
    cats = {it["extra"].get("category") for it in items} - {None, ""}
    if cats:
        order = [c for c in TB_CATEGORY_ORDER if c in cats] + sorted(cats - set(TB_CATEGORY_ORDER))
        out["by_category"] = {c: both(lambda it, c=c: it["extra"].get("category") == c) for c in order}
    out.update(n=len(items), n_missing=miss, n_invalid=inv)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Video-MME-Logical (verbatim answer handling of eval.py)
# ---------------------------------------------------------------------------------------------------------------

def logical_extract_answer(text: object) -> str:
    response = str(text or "").strip()
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def logical_normalize_answer(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.strip("`").strip()
    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return re.sub(r"\s+", "", text).casefold()
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")).casefold()
    explicit = re.match(r"^\s*([A-Z])\s*(?:[.)。,:：;；-].*)?$", text, flags=re.IGNORECASE)
    if explicit:
        return explicit.group(1).casefold()
    text = re.sub(r"^\s*(?:final\s+answer|answer|option|choice)\s*(?:is|:|：)\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("`*").rstrip("。.,;:!?！？")
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def logical_is_correct(answer: object, response_text: object) -> bool:
    pred = logical_normalize_answer(logical_extract_answer(response_text))
    gt = logical_normalize_answer(answer)
    return bool(gt) and bool(pred) and pred == gt


def videomme_logical_metrics(items: list[dict], preds: Mapping[str, Any]) -> dict:
    """Exact-match accuracy overall / per difficulty / per task, and separately on the video-only subset (items
    without reference images). An int prediction is converted to its option letter (only for items with options)."""
    correct, miss, inv = [], 0, 0
    for it in items:
        if it["id"] not in preds:
            miss += 1
            correct.append(0)
            continue
        p = preds[it["id"]]
        if p is None:
            inv += 1
            correct.append(0)
            continue
        if isinstance(p, int):
            if it["options"] is None:
                raise ValueError(f"{it['id']}: option index given for an open-answer item")
            p = chr(65 + p)
        correct.append(int(logical_is_correct(it["answer_text"], p)))

    def block(sel):
        sub = [(it, c) for it, c in zip(items, correct) if sel(it)]
        its = [it for it, _ in sub]
        cs = [c for _, c in sub]
        return {"accuracy": _pct(sum(cs), len(cs)), "n": len(cs),
                "by_difficulty": _acc_by(its, cs, lambda it: it["extra"]["difficulty"])}

    out = block(lambda it: True)
    out["by_parent_major"] = _acc_by(items, correct, lambda it: it["extra"]["parent_major"])
    out["by_major"] = _acc_by(items, correct, lambda it: it["extra"]["major"])
    out["video_only_subset"] = block(lambda it: it["extra"]["video_only"])
    out["image_required_subset"] = block(lambda it: not it["extra"]["video_only"])
    out.update(n_missing=miss, n_invalid=inv)
    return out


# ---------------------------------------------------------------------------------------------------------------
# IntPhys2
# ---------------------------------------------------------------------------------------------------------------

# ComputeScoreIntPhys2.ipynb subsets for source == 'Main' (env / game_name / condition lists copied verbatim;
# note 'None' never matches the metadata value 'None_0', so those 72 videos are in no difficulty subset there either).
INTPHYS2_DIFFICULTY_ENVS = {
    "Easy": ["BasicLevel_0"],
    "Medium": ["SaltFlats_0", "DesertMap_0", "RaceTrack_0", "TropicalIsland_0"],
    "Hard": ["PLVDaylight_0", "Egypt_0", "RuralAustralia03_0", "ParkingGarage_0", "None"],
}
INTPHYS2_CAMERA_GAMES = {
    "FixedCamera": ["FixedMarryPoppins", "FixedJumpSolidity", "RotatingCup", "HotAirBallon", "SphereFallingDown",
                    "SolidityFallingFlat"],
    "MovingCamera": ["SphereFallingDownSoldity", "BoxSoldity", "Scaffoling", "CameraSolidity", "JumpSolidity", "Box",
                     "MovingAroundOccluder", "JailStone", "PrisonCell", "Restaurant"],
}
INTPHYS2_CONDITIONS = {"Permanence": ["permanence"], "Immutability": ["immutability", "immutability_texture"],
                       "Continuity": ["continuity", "continuity_swap"], "Solidity": ["solidity"]}


def intphys2_metrics(items: list[dict], preds: Mapping[str, Any]) -> dict:
    """Single-video possible/impossible accuracy (prediction index 0 = 'Yes' = possible, 1 = impossible; None =
    no yes/no found, the notebook's label 2, counted wrong). Denominator = all ground-truth videos of the subset.
    Reported per split; for Main also the notebook's difficulty, camera and property subsets."""
    if any(it["answer_index"] is None for it in items):
        raise ValueError("IntPhys2 HeldOut labels are not public; score it on the leaderboard")
    correct, miss, inv = _option_correct(items, preds)
    out = {"n": len(items), "n_missing": miss, "n_invalid": inv}
    for split in sorted({it["split"] for it in items}):
        sub = [(it, c) for it, c in zip(items, correct) if it["split"] == split]

        def acc(sel):
            s = [c for it, c in sub if sel(it["extra"])]
            return {"accuracy": _pct(sum(s), len(s)), "n": len(s)}

        r = {"all": acc(lambda ex: True)}
        if split == "main":
            r["difficulty"] = {k: acc(lambda ex, v=v: ex["env"] in v) for k, v in INTPHYS2_DIFFICULTY_ENVS.items()}
            r["camera"] = {k: acc(lambda ex, v=v: ex["game_name"] in v) for k, v in INTPHYS2_CAMERA_GAMES.items()}
            r["condition"] = {k: acc(lambda ex, v=v: ex["condition"] in v) for k, v in INTPHYS2_CONDITIONS.items()}
            r["condition_x_camera"] = {
                f"{k}/{cam}": acc(lambda ex, v=v, g=g: ex["condition"] in v and ex["game_name"] in g)
                for k, v in INTPHYS2_CONDITIONS.items() for cam, g in INTPHYS2_CAMERA_GAMES.items()}
        out[split] = r
    return out


def intphys2_pairwise_accuracy(items: list[dict], surprise: Mapping[str, float]) -> dict:
    """Auxiliary, for prediction-based (surprise) models: within each possible/impossible pair (extra.pair_id), the
    pair is correct when the impossible video gets the strictly higher surprise. This is the relative protocol of
    the IntPhys/V-JEPA papers; the released IntPhys2 repo only extracts surprises and ships no scoring function,
    so this is NOT an official-code reproduction. Pairs with a missing score count as wrong."""
    pairs = defaultdict(dict)
    for it in items:
        pairs[(it["split"], it["extra"]["pair_id"])][it["extra"]["label_possible"]] = it["id"]
    res = defaultdict(lambda: [0, 0])
    for (split, pid), d in sorted(pairs.items()):
        if set(d) != {0, 1}:
            raise ValueError(f"pair {pid}: {d}")
        ok = d[0] in surprise and d[1] in surprise and surprise[d[0]] > surprise[d[1]]
        res[split][0] += int(ok)
        res[split][1] += 1
    return {s: {"accuracy": _pct(*v), "n_pairs": v[1]} for s, v in sorted(res.items())}


# ---------------------------------------------------------------------------------------------------------------
# TempCompass multi-choice
# ---------------------------------------------------------------------------------------------------------------

TEMPCOMPASS_DIMS = ["action", "direction", "speed", "order", "attribute_change"]  # eval_utils.print_result order
TEMPCOMPASS_FINE_DIMS = ["fine-grained action", "coarse-grained action", "object motion", "camera motion",
                         "absolute speed", "relative speed", "order", "color & light change", "size & shape change",
                         "combined change", "other change"]


def tempcompass_mc_rule_rating(prediction: str | None, answer: str) -> int | None:
    """The hand-crafted matching rules of eval_multi-choice.py (= lmms-eval tempcompass_process_results_multi_choice)
    before the ChatGPT fallback: 1 / 0, or None when no rule matches (official: ChatGPT judges it; with
    --disable_llm the rating is 0). A None prediction is rated 0, as in the official script."""
    if prediction is None:
        return 0
    p = prediction.replace("</s>", "").strip()
    if p == answer:
        return 1
    if p in ["A", "B", "C", "D"]:
        return int(p == answer[0])
    if any(p.startswith(x) for x in ["A.", "B.", "C.", "D."]):
        return int(p.split(".")[0] == answer[0])
    if any(p.startswith(x) for x in ["A)", "B)", "C)", "D)"]):
        return int(p.split(")")[0] == answer[0])
    return None


def tempcompass_mc_metrics(items: list[dict], preds: Mapping[str, Any]) -> dict:
    """Accuracy over all questions ('avg', the number reported as TempCompass multi-choice accuracy), per temporal
    aspect (5) and per fine-grained aspect (11), as eval_utils.print_result (unrounded, percent). A prediction is an
    option index (likelihood runner), None (invalid), or raw generated text, which is rated with the official rules
    and the --disable_llm path (unmatched text = 0, counted in n_unmatched). The official default sends unmatched
    text to gpt-3.5-turbo-1106 instead; that judge is not reproduced here."""
    correct, miss, inv, unmatched = [], 0, 0, 0
    for it in items:
        if it["id"] not in preds:
            miss += 1
            correct.append(0)
            continue
        p = preds[it["id"]]
        if p is None:
            inv += 1
            correct.append(0)
        elif isinstance(p, str):
            r = tempcompass_mc_rule_rating(p, it["answer_text"])
            unmatched += r is None
            correct.append(r or 0)
        else:
            p = int(p)
            if not 0 <= p < len(it["options"]):
                raise ValueError(f"{it['id']}: predicted index {p} outside 0..{len(it['options']) - 1}")
            correct.append(int(p == it["answer_index"]))
    ok = {it["id"]: c for it, c in zip(items, correct)}               # (10-18) per-item 0/1 (rule ratings included), not raw predictions
    by_dim = _rate_by(items, ok, lambda it: it["extra"]["dim"])
    by_fine = _rate_by(items, ok, lambda it: it["extra"]["fine_dim"])
    return {"avg": _pct(sum(correct), len(items)),
            "by_dim": {k: by_dim[k] for k in TEMPCOMPASS_DIMS if k in by_dim},
            "by_fine_dim": {k: by_fine[k] for k in TEMPCOMPASS_FINE_DIMS if k in by_fine},
            "n": len(items), "n_missing": miss, "n_invalid": inv, "n_unmatched": unmatched}


# ---------------------------------------------------------------------------------------------------------------
# TOMATO
# ---------------------------------------------------------------------------------------------------------------

def tomato_metrics(items: list[dict], preds: Mapping[str, Any]) -> dict:
    """src/get_categorized_score.py: accuracy = correct / count summed over all (reasoning type, demonstration type)
    result files, i.e. over all 1,484 questions; also per reasoning type (main table), per demonstration type and
    per (reasoning, demonstration) cell. Prediction = option index. The official parser maps an unparsable response
    to a random letter (parse_result.py); here None is scored wrong and reported in n_invalid."""
    correct, miss, inv = _option_correct(items, preds)
    return {"accuracy": _pct(sum(correct), len(items)),
            "by_reasoning_type": _acc_by(items, correct, lambda it: it["extra"]["reasoning_type"]),
            "by_demonstration_type": _acc_by(items, correct, lambda it: it["extra"]["demonstration_type"]),
            "by_reasoning_x_demonstration": _acc_by(
                items, correct, lambda it: f"{it['extra']['reasoning_type']}/{it['extra']['demonstration_type']}"),
            "n": len(items), "n_missing": miss, "n_invalid": inv}


# ---------------------------------------------------------------------------------------------------------------

def _rate_by(items, ok, key):
    """mean of per-item 0/1 correctness `ok` {id: 0/1} within each group"""
    g = defaultdict(list)
    for x in items:
        g[key(x)].append(int(ok.get(x["id"], 0)))
    return {k: 100.0 * sum(v) / len(v) for k, v in sorted(g.items())}


def _acc_by_preds(items, preds, key):
    """(10-18 renamed: a second _acc_by here shadowed the (items, correct, key) one and broke TOMATO and two other scorers)"""
    g = defaultdict(list)
    for x in items:
        p = preds.get(x["id"]); g[key(x)].append(int(p is not None and int(p) == x["answer_index"]))
    return {k: 100.0 * sum(v) / len(v) for k, v in sorted(g.items())}


def mvbench_metrics(items: list[dict], preds: Mapping[str, Any]) -> dict:
    """official: mean of the per-task accuracies"""
    t = _acc_by_preds(items, preds, lambda x: x["extra"]["task"])
    return {"avg": sum(t.values()) / len(t), "n_tasks": len(t), "per_task": t, "n": len(items),
            "n_missing": sum(1 for x in items if preds.get(x["id"]) is None)}


def videomme_metrics(items: list[dict], preds: Mapping[str, Any]) -> dict:
    d = _acc_by_preds(items, preds, lambda x: x["extra"]["duration"]); a = _acc_by_preds(items, preds, lambda x: "all")["all"]
    return {"overall": a, **d, "n": len(items), "n_missing": sum(1 for x in items if preds.get(x["id"]) is None)}


METRICS = {
    "perception_test": perception_test_metrics,
    "videomme_v2": videomme_v2_metrics,
    "videomme_logical": videomme_logical_metrics,
    "mvp_mini": mvp_metrics,
    "mvp_full": mvp_metrics,
    "temporalbench_short_qa": temporalbench_metrics,
    "temporalbench_long_qa": temporalbench_metrics,
    "intphys2": intphys2_metrics,
    "tempcompass_mc": tempcompass_mc_metrics,
    "tomato": tomato_metrics,
    "mvbench": mvbench_metrics,
    "videomme": videomme_metrics,
}


def score(bench: str, items: list[dict], preds: Mapping[str, Any]) -> dict:
    if bench == "perception_test_test":
        raise ValueError(f"{bench}: labels are not public; submit with scripts/eval/write_pt_test_submission.py")
    return METRICS[bench](items, preds)

