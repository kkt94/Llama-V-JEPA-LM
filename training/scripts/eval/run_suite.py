"""Pilot evaluation suite: fixed, group-preserving subsets of every benchmark, run -> merge ranks -> official metrics.

  subsets   writes manifests/eval/pilot/<bench>.jsonl once (deterministic; whole groups are kept, because the metrics
            pair or group items: MVP pairs, TemporalBench videos, Video-MME-v2 question groups) and a md5 table
  run       for one checkpoint: torchrun the runner per benchmark on the subset, merge rank files, score with
            jepa_llm.eval.metrics.score, write <out>/scores.json (and keep per-item predictions)
Subsets are for comparing arms of the same pilot, not for reporting benchmark numbers.
usage: run_suite.py subsets
       run_suite.py run --config <cfg> --ckpt <ckpt> --out <dir> --gpus 0,1,2,3 [--benches a,b] [--stage 1]
"""
import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/data/llama_vjepa_lm/training")
R = "/data/llama_vjepa_lm"
E = f"{R}/manifests/eval"
SUB = f"{E}/pilot"
# bench -> number of GROUPS kept (None = all)
PLAN = {"mvp_mini": 1500, "perception_test": 900, "temporalbench_short_qa": 600, "videomme_v2": 200,
        "intphys2": None, "tempcompass_mc": None, "tomato": None}


def subsets():
    os.makedirs(SUB, exist_ok=True)
    table = []
    for bench, n in PLAN.items():
        items = [json.loads(l) for l in open(f"{E}/{bench}.jsonl")]
        if bench == "perception_test":                       # group by video (several questions per video)
            key = lambda x: x["video"]
        else:
            key = lambda x: str(x.get("group_id"))
        groups = collections.OrderedDict()
        for x in items:
            groups.setdefault(key(x), []).append(x)
        names = sorted(groups)
        if n is not None and n < len(names):
            names = sorted(np.random.default_rng(0).choice(names, n, replace=False).tolist())
        keep = [x for g in names for x in groups[g]]
        path = f"{SUB}/{bench}.jsonl"
        with open(path, "w") as f:
            for x in keep:
                f.write(json.dumps(x) + "\n")
        md5 = hashlib.md5(open(path, "rb").read()).hexdigest()[:12]
        table.append((bench, len(names), len(keep), md5))
        print(bench, "groups", len(names), "items", len(keep), md5)
    with open(f"{SUB}/README.md", "w") as f:
        f.write("# Pilot evaluation subsets (run_suite.py subsets; seed 0, whole groups kept)\n\n| bench | groups | items | md5 |\n|---|---:|---:|---|\n")
        for b, g, i, m in table:
            f.write(f"| {b} | {g} | {i} | `{m}` |\n")


def generic_pairs(items, preds):
    """single accuracy + pair accuracy over groups of exactly two items (both right) for our held-out pair sets"""
    ok = {x["id"]: int(preds.get(x["id"]) is not None and int(preds[x["id"]]) == x["answer_index"]) for x in items}
    groups = collections.defaultdict(list)
    for x in items:
        groups[x.get("group_id", x["id"])].append(ok[x["id"]])   # (10-18) items without group_id (longq) count as singles
    pairs = [g for g in groups.values() if len(g) == 2]
    return {"single_accuracy": 100.0 * np.mean(list(ok.values())), "pair_accuracy": 100.0 * np.mean([all(g) for g in pairs]) if pairs else None,
            "n": len(items), "n_pairs": len(pairs), "n_missing": sum(1 for x in items if preds.get(x["id"]) is None)}


def items_path(bench):
    """<bench> = pilot subset; heldout:<name> = our held-out pair sets; light:<name> = light per-checkpoint set"""
    if bench.startswith("heldout:"):
        return f"{E}/heldout/{bench.split(':', 1)[1]}.jsonl"
    if bench.startswith("light:"):
        return f"{E}/light/{bench.split(':', 1)[1]}.jsonl"
    if bench.startswith("full:"):                         # (10-19) the whole benchmark manifest, not the pilot subset
        return f"{E}/{bench.split(':', 1)[1]}.jsonl"
    return f"{SUB}/{bench}.jsonl"


def scorer(bench):
    name = bench.split(":", 1)[-1]
    from jepa_llm.eval.metrics import METRICS
    return "generic" if (bench.startswith("heldout:") or "heldout" in name or name not in METRICS) else name   # (10-18) new benches: generic


def special_score(bench, items, recs, preds):
    """(10-22) open-answer benchmarks shared by run_suite and baseline_eval: VSI-Bench (MC accuracy + numeric MRA, mean of tasks) and
    Charades-STA (temporal grounding). None for every other benchmark."""
    name = bench.split(":", 1)[-1]
    if name == "vsibench":                # (10-22) VSI-Bench: MC accuracy / numeric MRA per task, overall = mean of tasks
        import re as _re
        resp = {r["id"]: r.get("response") for r in recs}
        _W = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                                            "sixteen seventeen eighteen nineteen twenty".split())}
        _W.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100})
        def _num(s):                                      # a digit number, else a number word ("Two." for a count)
            s = str(s or "").replace(",", "")
            m = _re.search(r"-?\d+(?:\.\d+)?", s)
            if m:
                return float(m.group())
            w = [_W[t] for t in _re.findall(r"[a-z]+", s.lower()) if t in _W]
            return float(w[0]) if w else None
        def _mra(p, g):                                   # mean relative accuracy over thresholds 0.50 .. 0.95
            if p is None:
                return 0.0
            return float(np.mean([abs(p - g) / max(abs(g), 1e-9) < 1 - t for t in np.arange(0.5, 1.0, 0.05)]))
        per = collections.defaultdict(list)
        for x in items:
            task = x["extra"]["task"]
            if x.get("options"):
                per[task].append(float(preds.get(x["id"]) is not None and int(preds[x["id"]]) == x["answer_index"]))
            else:
                per[task].append(_mra(_num(resp.get(x["id"])), float(x["answer_text"])))
        res = {"avg": 100.0 * float(np.mean([np.mean(v) for v in per.values()])), "n": len(items),
               "n_missing": sum(1 for x in items if preds.get(x["id"]) is None and resp.get(x["id"]) is None),
               "per_task": {k: 100.0 * float(np.mean(v)) for k, v in sorted(per.items())}}
    elif name == "charades_sta":            # (10-22) temporal grounding: "From Xs to Ys." -> R@{0.3,0.5,0.7}, mIoU
        import re as _re
        resp = {r["id"]: r.get("response") for r in recs}
        ious = []
        for x in items:
            nums = [float(v) for v in _re.findall(r"\d+(?:\.\d+)?", str(resp.get(x["id"]) or ""))[:2]]
            g0, g1 = x["extra"]["timestamp"]
            if len(nums) < 2:
                ious.append(0.0); continue
            p0, p1 = min(nums), max(nums)
            inter = max(0.0, min(p1, g1) - max(p0, g0)); union = max(p1, g1) - min(p0, g0)
            ious.append(inter / union if union > 0 else 0.0)
        ious = np.array(ious)
        res = {"R@0.3": 100.0 * float((ious >= 0.3).mean()), "R@0.5": 100.0 * float((ious >= 0.5).mean()), "R@0.7": 100.0 * float((ious >= 0.7).mean()),
               "mIoU": 100.0 * float(ious.mean()), "n": len(items), "n_missing": sum(1 for x in items if resp.get(x["id"]) is None)}
    elif name == "timeblind":               # (10-28) TimeBlind: instances of 4 (2 videos x 2 questions; official utils.get_scores)
        ok = {x["id"]: int(preds.get(x["id"]) is not None and int(preds[x["id"]]) == x["answer_index"]) for x in items}
        inst = collections.defaultdict(dict)
        for x in items:
            inst[x["group_id"]][x["extra"]["role"]] = (ok[x["id"]], x["extra"]["type"])   # role: 0 q0_i0, 1 q0_i1, 2 q1_i0, 3 q1_i1
        def _m(sel):
            g = [r for r in inst.values() if len(r) == 4 and r[0][1] in sel]
            c = [[r[k][0] for k in range(4)] for r in g]
            return {"I_Acc": 100.0 * float(np.mean([all(v) for v in c])),                                  # all four right
                    "Q_Acc": 100.0 * float(np.mean([v[0] and v[1] for v in c] + [v[2] and v[3] for v in c])),  # a question right on both videos
                    "V_Acc": 100.0 * float(np.mean([v[0] and v[2] for v in c] + [v[1] and v[3] for v in c])),  # a video right on both questions
                    "Acc": 100.0 * float(np.mean([s for v in c for s in v])), "n_instances": len(g)}
        res = {**_m({"yes_no", "multiple_choice"}), "n": len(items), "n_missing": sum(1 for x in items if preds.get(x["id"]) is None),
               "n_incomplete_instances": sum(1 for r in inst.values() if len(r) != 4),
               "per_type": {t: _m({t}) for t in ("yes_no", "multiple_choice")}}
        res["instance_accuracy"], res["single_accuracy"] = res["I_Acc"], res["Acc"]
    elif name == "mvp_full":                # (10-28) official MVP numbers (mvp_metrics: per subset, mean of subsets) + per source + mini / non-mini
        from jepa_llm.eval.metrics import mvp_metrics
        res = mvp_metrics(items, preds)
        ok = {x["id"]: int(preds.get(x["id"]) is not None and int(preds[x["id"]]) == x["answer_index"]) for x in items}
        def _pairs(sel):
            g = collections.defaultdict(list)
            for x in items:
                if sel(x):
                    g[x["group_id"]].append(ok[x["id"]])
            v = [all(p) for p in g.values() if len(p) == 2]
            return {"pair_accuracy": 100.0 * float(np.mean(v)) if v else None,
                    "single_accuracy": 100.0 * float(np.mean([s for p in g.values() for s in p])) if g else None, "n_pairs": len(v)}
        res["per_source"] = {s: _pairs(lambda x, s=s: x["extra"]["source"] == s) for s in sorted({x["extra"]["source"] for x in items})}
        res["by_mini"] = {k: _pairs(lambda x, k=k: bool(x["extra"].get("in_mvp_mini")) == (k == "in_mvp_mini")) for k in ("in_mvp_mini", "not_in_mvp_mini")}
    elif name in ("vlm4d", "stibench", "physgame", "wmabench") or name.startswith("paibench_"):   # (10-28) accuracy + per-category accuracy
        key = {"vlm4d": "source", "stibench": "task", "physgame": "class", "wmabench": "task"}.get(name, "subcategory")
        ok = {x["id"]: float(preds.get(x["id"]) is not None and int(preds[x["id"]]) == x["answer_index"]) for x in items}
        per = collections.defaultdict(list)
        for x in items:
            per[x["extra"][key]].append(ok[x["id"]])
        res = {"single_accuracy": 100.0 * float(np.mean(list(ok.values()))), "n": len(items),
               "n_missing": sum(1 for x in items if preds.get(x["id"]) is None), f"per_{key}": {k: 100.0 * float(np.mean(v)) for k, v in sorted(per.items())}}
        if name == "vlm4d":                 # the paper's real / synthetic split
            per2 = collections.defaultdict(list)
            for x in items:
                per2[x["extra"]["part"]].append(ok[x["id"]])
            res["per_part"] = {k: 100.0 * float(np.mean(v)) for k, v in sorted(per2.items())}
        if name == "stibench":              # official overall = accuracy over all items; also the mean of the 8 tasks
            res["task_mean"] = float(np.mean(list(res["per_task"].values())))
        if name in ("physgame", "wmabench"):  # (10-28) PhysGame per subclass; WM-ABench per dimension and the mean of the task configs
            k2 = {"physgame": "subclass", "wmabench": "dimension"}[name]
            per2 = collections.defaultdict(list)
            for x in items:
                per2[x["extra"][k2]].append(ok[x["id"]])
            res[f"per_{k2}"] = {k: 100.0 * float(np.mean(v)) for k, v in sorted(per2.items())}
            if name == "wmabench":
                res["task_mean"] = float(np.mean(list(res["per_task"].values())))
    else:
        return None
    return res


def run(a):
    from jepa_llm.eval.metrics import score
    os.makedirs(a.out, exist_ok=True)
    benches = a.benches.split(",") if a.benches else list(PLAN)
    n = len(a.gpus.split(","))
    results = json.load(open(f"{a.out}/scores.json")) if os.path.exists(f"{a.out}/scores.json") else {}
    todo = [b for b in benches if b not in results]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": a.gpus, "PYTHONPATH": f"{R}/training"}
    if a.limit:
        n = 1
    def runner(bl):
        ips, outs = [items_path(b) for b in bl], [f"{a.out}/{b.replace(':', '_')}.jsonl" for b in bl]
        cmd = ["torchrun", "--standalone", f"--nproc_per_node={n}", "-m", "jepa_llm.eval.runner",
               "--config", a.config, "--ckpt", a.ckpt, "--items", ",".join(ips), "--out", ",".join(outs), "--stage", str(a.stage), "--batch", str(a.batch), "--workers", str(a.workers)] + (["--cache_dir", a.cache_dir] if a.cache_dir else []) + (["--limit", str(a.limit), "--device", "cpu"] if a.limit else [])
        rc = subprocess.run(cmd, env=env, cwd=f"{R}/training", stdout=open(f"{outs[0]}.log", "w"), stderr=subprocess.STDOUT).returncode
        assert rc == 0, f"{bl}: runner rc {rc}, see {outs[0]}.log"
    if a.one_load and len(todo) > 1:          # (10-19) all benchmarks on one loaded model (loading ~4 min each otherwise)
        runner(todo)
    for bench in todo:
        ipath = items_path(bench)
        out = f"{a.out}/{bench.replace(':', '_')}.jsonl"
        if not (a.one_load and len(todo) > 1):
            runner([bench])
        import glob as _g
        parts = sorted(_g.glob(f"{out}.rank*"))                 # torchrun makes the runner write rank files even for 1 GPU
        recs = [json.loads(l) for p in parts for l in open(p)] if parts else [json.loads(l) for l in open(out)]
        with open(out, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        for p in parts:
            os.remove(p)
        items = [json.loads(l) for l in open(ipath)][: a.limit]
        preds = {r["id"]: r.get("pred_index") for r in recs}
        sc = scorer(bench)
        if bench.split(":", 1)[-1] == "textvqa_ocr":         # (10-21) open answers: normalised exact match against the reference
            import re as _re
            norm = lambda x: " ".join(w for w in _re.sub(r"[^a-z0-9 ]", " ", str(x).lower()).split() if w not in ("a", "an", "the"))
            resp = {r["id"]: r.get("response") for r in recs}
            ok = [int(resp.get(x["id"]) is not None and norm(resp[x["id"]]) == norm(x["answer_text"])) for x in items]
            res = {"exact_match": 100.0 * float(np.mean(ok)), "n": len(items), "n_missing": sum(1 for x in items if resp.get(x["id"]) is None)}
        elif special_score(bench, items, recs, preds) is not None:
            res = special_score(bench, items, recs, preds)
        else:
            res = generic_pairs(items, preds) if sc == "generic" else score(sc, items, preds)
        res["n_errors"] = sum(1 for r in recs if r.get("error"))
        results[bench] = res
        json.dump(results, open(f"{a.out}/scores.json", "w"), indent=1)
        print(bench, json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)})[:400], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["subsets", "run"])
    ap.add_argument("--config"); ap.add_argument("--ckpt"); ap.add_argument("--out"); ap.add_argument("--gpus", default="0")
    ap.add_argument("--benches"); ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--limit", type=int, help="CPU smoke only: first N items, runner on CPU")
    ap.add_argument("--one_load", action="store_true", help="(10-19) one runner process (one model load) for all the benchmarks")
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache_dir", default="/data/llama_vjepa_lm/runs/evcache", help="frozen V-JEPA evidence cache ('' = off)")
    a = ap.parse_args()
    subsets() if a.cmd == "subsets" else run(a)
