"""Convert runner predictions on the PerceptionTest TEST split into the official eval-server submission JSON.

usage: python -I write_pt_test_submission.py --preds <runner_out.jsonl> --out <submission.json>
           [--items /data/llama_vjepa_lm/manifests/eval/perception_test_test.jsonl]

Input  : runner output (jepa_llm.eval.runner), one JSON object per line with at least {"id": "<video_id>#<q>",
         "pred_index": int | null}; ids must be those of perception_test_test.jsonl.
Output : the format of the PerceptionTest multiple-choice videoQA challenge (eval.ai "Submission format", identical to
         the results dict of baselines/mc_vqa.ipynb / calc_top1):

             {"video_1009": [{"id": 0, "answer_id": 2, "answer": "static or shaking"},
                             {"id": 1, "answer_id": 0, "answer": "moving"}], ...}

         one entry per test video (3,146), one answer per question (11,528), questions in 'id' order; answer_id is the
         0-based option index in the released option order and answer is that option string. The server requires
         answers to ALL questions of the split, so a missing id or a null pred_index is an error (no silent fill);
         pass --allow-fill-random only for a format smoke test (never for a reported number).

Eval server (nothing is submitted by this script):
  * Perception Test Challenge 2023 - Task 1 - Multiple-choice videoQA, https://eval.ai/web/challenges/challenge-page/2091
    "Test Phase" (phase id 4156) takes exactly this split (mc_question_test_annotations.zip, 11,528 questions). Metric:
    top-1 accuracy. Linked from github.com/google-deepmind/perception_test README ("Challenges and evaluation servers").
    Queried 2026-10-10 via https://eval.ai/api/challenges/challenge/2091/challenge_phase: end_date 2026-01-31,
    is_active false, i.e. the phase was closed at the time of writing.
  * Perception Test Challenge 2026 (https://eval.ai/web/challenges/challenge-page/2702, ECCV 2026) uses a different,
    "unified" question set (original questions + extra point-tracking / action-segmentation questions,
    mc_vqa_unified_inputs.json) with the same JSON format; its test phase also ended (2026-08-27). Do not submit this
    file there: the question set differs.
  V-JEPA 2's 84.0 is a test-server number; check the challenge page for an open phase before submitting.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from jepa_llm.eval.adapters import read_jsonl  # noqa: E402

N_QUESTIONS, N_VIDEOS = 11528, 3146


def build_submission(items: list[dict], preds: dict, fill_random: bool = False) -> dict:
    """items: perception_test_test manifest rows; preds: id -> option index (or None). Returns the submission dict."""
    known = {it["id"] for it in items}
    extra = sorted(set(preds) - known)
    if extra:
        raise ValueError(f"{len(extra)} predicted ids are not test items, e.g. {extra[:3]}")
    out, bad = defaultdict(list), []
    rng = random.Random(0)
    for it in sorted(items, key=lambda x: (int(x["extra"]["video_id"].split("_")[1]), x["extra"]["question_id"])):
        p = preds.get(it["id"])
        if p is None:
            if not fill_random:
                bad.append(it["id"])
                continue
            p = rng.randrange(len(it["options"]))
        p = int(p)
        if not 0 <= p < len(it["options"]):
            raise ValueError(f"{it['id']}: pred_index {p} outside 0..{len(it['options']) - 1}")
        out[it["extra"]["video_id"]].append({"id": it["extra"]["question_id"], "answer_id": p, "answer": it["options"][p]})
    if bad:
        raise ValueError(f"{len(bad)} of {len(items)} questions have no prediction (missing id or null pred_index), "
                         f"e.g. {bad[:3]}; the server requires all questions")
    return dict(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", default="/data/llama_vjepa_lm/manifests/eval/perception_test_test.jsonl")
    ap.add_argument("--allow-fill-random", action="store_true", help="format smoke test only")
    a = ap.parse_args()
    items = read_jsonl(a.items)
    if len(items) != N_QUESTIONS or len({it["group_id"] for it in items}) != N_VIDEOS:
        raise SystemExit(f"{a.items}: expected {N_QUESTIONS} questions / {N_VIDEOS} videos")
    preds = {}
    for r in read_jsonl(a.preds):
        if r["id"] in preds:
            raise SystemExit(f"duplicate prediction for {r['id']}")
        preds[r["id"]] = r.get("pred_index")
    sub = build_submission(items, preds, fill_random=a.allow_fill_random)
    n = sum(len(v) for v in sub.values())
    assert n == N_QUESTIONS and len(sub) == N_VIDEOS, (n, len(sub))
    Path(a.out).write_text(json.dumps(sub))
    print(f"wrote {a.out}: {len(sub)} videos, {n} answers"
          + (" (RANDOM FILL USED - not a valid result)" if a.allow_fill_random else ""))


if __name__ == "__main__":
    main()
