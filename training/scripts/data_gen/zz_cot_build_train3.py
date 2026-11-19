"""(10-30 11:05, ROUND 3 = dual mode: template qwen3_dual.jinja puts "<think>\n" on the PROMPT side of thinking turns, the no-think
turns are exactly the original format; CoT assistant content = "{reasoning}\n</think>\n\n{letter}."; thinking text replay tokenised with
enable_thinking=True; output runs/cot/train3)
(10-30 07:40, ROUND 2: CoT rows get their own instruction "Think step by step about what happens in the video, then answer with the
option's letter." instead of "...directly" (round 1 used the direct suffix and the model left <think> empty on MC); output runs/cot/train2)
(10-30) CoT-stage training manifests for the 2B (and later 8B): video rows in the trained MC format, assistant turn
"<think>\\n{reasoning}\\n</think>\\n\\n{letter}." (template configs/templates/qwen3_hybrid.jinja: a turn that starts with <think> is the thinking
mode; any other assistant turn gets the empty block, which is now LABELLED, so the bare header learns think / no-think; the evaluation keeps
qwen3_nothink.jinja = the no-think header, unchanged benchmark protocol).
  A  runs/cot/A_items.jsonl + A_rationales_*.jsonl (ok)            teacher Qwen3-8B from our annotations
  B  runs/cot/B_*.jsonl (letter after the reasoning == gold, 20-319 tokens, no loops)   the 4B's own video-grounded reasoning
  C  Video-R1-COT-165k + VideoRFT-CoT (video, multiple choice), clips matched to our local copies (llava / clevrer / PT train),
     eval videos excluded (basename match against every eval manifest), deduplicated on (clip, question)
  T  text replay: the backbone's own no-think answers (existing parquet) + its thinking-mode answers (selfgen_think_<tag>), think share 0.4
usage: python -I zz_cot_build_train.py <backbone tag: q3_1p7b|q3_8b> <backbone dir>  -> runs/cot/train/*.parquet + summary.json"""
import json, re, os, glob, sys, collections, random
import numpy as np, pandas as pd
from transformers import AutoTokenizer
R = "/data/llama_vjepa_lm"; O = f"{R}/runs/cot/train3"; os.makedirs(O, exist_ok=True); TAG, BB = sys.argv[1], sys.argv[2]; L = "ABCDEFGHIJ"; rng = random.Random(0)
SUF = "Answer with the option's letter from the given choices directly."
THINK_SUF = "Think step by step about what happens in the video, then answer with the option's letter."
def user(q, opts): return "<|jepa_evidence|>\n" + q.strip() + "\n" + "\n".join(f"{L[j]}. {o}" for j, o in enumerate(opts)) + "\n" + THINK_SUF
def think(r, g): return r.strip() + "\n</think>\n\n" + L[g] + "."
def rep4(t):
    w = t.split(); g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]; return 0.0 if not g else 1 - len(set(g)) / len(g)
rows = collections.defaultdict(list); summ = collections.Counter()
items = [json.loads(l) for l in open(f"{R}/runs/cot/A_items.jsonl")]
# (00:25) A re-scored here (the generator's novelty test also rejected plain reasoning words and inflections of option words):
# answer letter = gold, no annotation wording, and at most 3 content-word STEMS that appear neither in the question / options / facts
# nor in a list of generic reasoning / motion vocabulary
_gsrc = open(f"{R}/logs/zz_cot_a_gen.py").read(); GEN = set(_gsrc[_gsrc.index('GEN = set("""') + 13:_gsrc.index('""".split())')].split())
GEN |= set("""followed follow follows observed observe observing directly align aligns aligned incorrect incorrectly correctly reverses reverse
  reversed suggests suggest immediately subsequently afterward afterwards preceding precedes following clear clearly indicate indicated
  confirms confirm consistent inconsistent describe matches fits contradicts contradict implies indicates specifically shown option's
  entirely completely fully mentioned stated given context sequence's correct correctly accurately accurate reflect reflects""".split())
def stem(w):
    for suf in ("ingly", "edly", "ing", "ed", "es", "ly", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3: w = w[:-len(suf)]; break
    return w[:-1] if len(w) > 4 and w[-1] == w[-2] else w
BADW = re.compile(r"ground truth|annotation|label|\bclip|metadata|was told|i was given", re.I)
def a_ok(it, t):
    m = re.search(r"Answer:\s*([A-J])\s*$", t)
    if not m or m.group(1) != L[it["gold"]] or BADW.search(t): return False
    allowed = {stem(w) for w in re.findall(r"[a-z]{4,}", (it["question"] + " " + " ".join(it["options"]) + " " + " ".join(it["facts"])).lower()) + list(GEN)}
    return len({stem(w) for w in re.findall(r"[a-z]{4,}", t.lower())} - allowed) <= 3
for f in glob.glob(f"{R}/runs/cot/A_rationales_*of*.jsonl"):
    for l in open(f):
        r = json.loads(l); it = items[r["i"]]
        if not a_ok(it, r["text"]): summ["A_rejected"] += 1; continue
        txt = re.sub(r"\s*Answer:\s*[A-J]\s*$", "", r["text"]).strip()
        rows[("A", it["prep_root"])].append({"video": it["video"], "prep_exists": True, "t0": it["t0"], "t1": it["t1"],
                                            "conversations": json.dumps([{"role": "user", "content": user(it["question"], it["options"])}, {"role": "assistant", "content": think(txt, it["gold"])}])}); summ["A"] += 1
for f in glob.glob(f"{R}/runs/cot/B_*of*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if "pred" not in r or r["pred"] != r["gold"] or not (20 <= r["ntok"] < 320) or rep4(r["reason"]) > 0.2: summ["B_rejected"] += "pred" in r; continue
        it = items[r["i"]]
        rows[("B", it["prep_root"])].append({"video": it["video"], "prep_exists": True, "t0": it["t0"], "t1": it["t1"],
                                            "conversations": json.dumps([{"role": "user", "content": user(it["question"], it["options"])}, {"role": "assistant", "content": think(r["reason"], it["gold"])}])}); summ["B"] += 1
# ---- C ----
ll = pd.read_parquet(f"{R}/manifests/stage1/clean1027/llava.parquet", columns=["video", "prep_exists"]); llset = set(ll.video[ll.prep_exists.astype(bool)])
cl = {os.path.basename(v): v for v in pd.read_parquet(f"{R}/manifests/stage1/clevrer.parquet", columns=["video"]).video}
pt = {os.path.basename(v): v for v in pd.read_parquet(f"{R}/manifests/stage1/pt_train_mc.parquet", columns=["video"]).video}
ev = {os.path.splitext(os.path.basename(json.loads(l)["video"]))[0] for f in glob.glob(f"{R}/manifests/eval/pilot/*.jsonl") for l in open(f) if json.loads(l).get("video")}
seen = set()
src = [a for a in json.load(open(f"{R}/ext/cot_public/Video-R1-data/Video-R1-COT-165k.json")) if a["data_type"] == "video"] + \
      [b for b in map(json.loads, open(f"{R}/ext/cot_public/VideoRFT-Data/VideoRFT-CoT-Data.jsonl")) if b["data_type"] == "video"]
for a in src:
    if a.get("problem_type") != "multiple choice": continue
    p = a["path"]; parts = p.split("/", 2)
    if len(parts) < 3 or os.path.splitext(os.path.basename(p))[0] in ev: summ["C_eval_or_bad"] += 1; continue
    s, rest, b = parts[1], parts[2], os.path.basename(p)
    if s in ("LLaVA-Video-178K", "NeXT-QA") and rest in llset: key, vid = ("C", f"{R}/prep/llava_384_15fps"), rest
    elif s == "CLEVRER" and b in cl: key, vid = ("C", f"{R}/data/clevrer/video_train"), cl[b]
    elif s == "PerceptionTest" and b in pt: key, vid = ("C", f"{R}/prep/pt_train_384"), pt[b]
    else: summ["C_no_local_clip"] += 1; continue
    opts = a.get("options") or []
    if not opts:                                   # VideoRFT: options inside the problem text ("question\nA. ...\nB. ...")
        mm = re.findall(r"^\s*([A-J])[\.\)]\s*(.+)$", a["problem"], re.M)
        if mm and [x for x, _ in mm] == list(L[:len(mm)]): opts = [t for _, t in mm]; a = {**a, "problem": a["problem"][:a["problem"].find("\n" + mm[0][0])]}
    opts = [re.sub(r"^\s*[A-J][\.\)]\s*", "", o) for o in opts]
    m = re.search(r"<answer>\s*([A-J])", a.get("solution", "")); pr = re.search(r"<think>(.*?)</think>", a.get("process", ""), re.S)
    if not (opts and m and pr) or L.index(m.group(1)) >= len(opts) or len(opts) > 10: summ["C_unparsed"] += 1; continue
    q = re.sub(r"\n?Options?:.*$", "", a["problem"], flags=re.S).strip()
    if (vid, q) in seen: summ["C_dup"] += 1; continue
    seen.add((vid, q))
    rows[key].append({"video": vid, "prep_exists": True, "t0": None, "t1": None,
                      "conversations": json.dumps([{"role": "user", "content": user(q, opts)}, {"role": "assistant", "content": think(pr.group(1), L.index(m.group(1)))}])}); summ["C"] += 1
man = []
for (kind, root), rs in sorted(rows.items(), key=lambda x: (x[0][0], str(x[0][1]))):
    name = f"cot{kind}_" + re.sub(r"[^a-z0-9]+", "_", os.path.basename(str(root).rstrip("/")).lower() or "root")
    path = f"{O}/{name}.parquet"; pd.DataFrame(rs).to_parquet(path); man.append({"name": name, "kind": kind, "manifest": path, "prep_root": root, "rows": len(rs)})
# ---- text replay: no-think (existing) + thinking-mode selfgen ----
tok = AutoTokenizer.from_pretrained(BB); tok.chat_template = open(f"{R}/training/configs/templates/qwen3_dual.jinja").read()
base = pd.read_parquet(f"{R}/manifests/textonly/ultrachat_selfgen_" + {"q3_1p7b": "qwen3_1p7b", "q3_8b": "qwen3_8b"}[TAG] + ".parquet")
th = [json.loads(l) for f in glob.glob(f"{R}/manifests/textonly/selfgen_think_{TAG}/*.jsonl") for l in open(f)]
th = list({r["prompt_id"]: r for r in th if r["finish"] == "stop" and "</think>" in r["answer"] and r["answer"].lstrip().startswith("<think>")}.values())
ids, labs = [], []
for r in th:
    ans = r["answer"].lstrip()[len("<think>"):].lstrip("\n")          # the template supplies "<think>\n" on the prompt side
    p = tok(tok.apply_chat_template([{"role": "user", "content": r["user"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
    full = tok(tok.apply_chat_template([{"role": "user", "content": r["user"]}, {"role": "assistant", "content": ans}], tokenize=False, enable_thinking=True), add_special_tokens=False).input_ids
    if full[:len(p)] != p: continue
    ids.append(np.array(full, dtype=np.int64)); labs.append(np.array([-100] * len(p) + full[len(p):], dtype=np.int64))
n_base = int(round(len(ids) * 0.6 / 0.4)); bsel = base.sample(min(n_base, len(base)), random_state=0)
tx = pd.concat([bsel[["input_ids", "labels"]], pd.DataFrame({"input_ids": ids, "labels": labs})], ignore_index=True).sample(frac=1, random_state=0)
tpath = f"{O}/text_{TAG}_nothink_think.parquet"; tx.to_parquet(tpath)
summ["T_think"] = len(ids); summ["T_nothink"] = len(bsel)
json.dump({"counts": summ, "manifests": man, "text": tpath}, open(f"{O}/summary_{TAG}.json", "w"), indent=1, default=str)
print(json.dumps(summ), "\n", "\n".join(f"{m['name']} {m['rows']} {m['prep_root']}" for m in man))
