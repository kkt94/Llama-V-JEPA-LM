"""(10-30 17:20, user option 2 data, v2) full-sentence answers written by a VIDEO teacher: Qwen3-VL-8B-Instruct (vLLM) watches the clip
(16 frames over the item's window), is told the question and its verified correct answer, and writes the answer a helpful assistant gives:
the direct answer as a complete sentence, then 1-2 sentences of what is visible in the video that supports it. Same item selection as
zz_fullans_gen.py (A_items sources <= 2500 each + 6 types of the verified GRPO pool <= 3000 each; no eval videos).
Kept if 10-90 words, no mention of options / letters / being told, and consistent with the gold answer (content-word stems of the gold
option, or yes / no polarity).
usage: <vllm python> -I zz_fullans_vl.py <k> <n> [limit]   -> runs/cot/fullans_vl_<k>of<n>.jsonl"""
import sys, os, json, re, random, collections
import numpy as np

STOP = set("a an the of to in on at by for with from and or is are was were be been it its this that these those his her their there".split())
BAN = re.compile(r"\b(option|options|letter|ground truth|annotation|label|choices?|i was told|you told|given answer|correct answer)\b", re.I)
NEG = re.compile(r"\b(no|not|never|n't|cannot|isn't|doesn't|didn't|wasn't|aren't)\b", re.I)


def content_words(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


def select(R):
    items = [json.loads(l) for l in open(f"{R}/runs/cot/A_items.jsonl")]
    for it in (json.loads(l) for l in open(f"{R}/runs/cot/grpo_pool.jsonl")):
        if it["type"] in ("general", "physics", "predict", "speed", "count", "narrative"):
            it = dict(it); it["src"] = "pool_" + it["type"]; items.append(it)
    by = collections.defaultdict(list)
    for i, it in enumerate(items): by[it["src"]].append(i)
    rng = random.Random(7); sel = []
    for s, v in sorted(by.items()): rng.shuffle(v); sel += v[:3000 if s.startswith("pool_") else 2500]
    rng.shuffle(sel); return items, sel


def frames(path, t0, t1, nf=16):
    import decord
    vr = decord.VideoReader(path, num_threads=2); fps, n = float(vr.get_avg_fps()), len(vr)
    a = 0.0 if t0 is None or (isinstance(t0, float) and np.isnan(t0)) else max(0.0, float(t0))
    b = n / fps if t1 is None or (isinstance(t1, float) and np.isnan(t1)) else min(n / fps, float(t1))
    if b - a < 0.2: a, b = 0.0, n / fps
    idx = np.clip(np.round((a + (np.arange(nf) + 0.5) * (b - a) / nf) * fps).astype(int), 0, n - 1)
    x = vr.get_batch(idx.tolist()).asnumpy()
    import cv2                       # (17:12) resize here (<= 384 x 384 pixels, sides multiple of 32): the processor's max_pixels is not applied to video
    h, w = x.shape[1:3]; sc = min(1.0, (384 * 384 / (h * w)) ** 0.5); nh, nw = max(32, int(h * sc) // 32 * 32), max(32, int(w * sc) // 32 * 32)
    if (nh, nw) != (h, w): x = np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA) for f in x])
    return x, {"fps": fps, "frames_indices": idx.tolist(), "total_num_frames": n, "video_backend": "decord", "duration": n / fps, "do_sample_frames": False}


def main():
    R = "/data/llama_vjepa_lm"; k, n = int(sys.argv[1]), int(sys.argv[2]); lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
    items, sel = select(R); idx = sel[k::n][:lim]
    out = f"{R}/runs/cot/fullans_vl_{k}of{n}.jsonl"
    import glob as _g      # (17:30) items done by ANY earlier shard file (so the run can be re-sharded over more GPUs)
    done = {json.loads(l)["i"] for f_ in _g.glob(f"{R}/runs/cot/fullans_vl_*of*.jsonl") for l in open(f_) if l.strip()}
    idx = [i for i in idx if i not in done]
    M = f"{R}/models/Qwen3-VL-8B-Instruct"
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(M)

    def prompt(it):
        gold = it["options"][it["gold"]]
        u = (f"Question about this video: {it['question']}\n"
             f"The correct answer is: {gold}\n\n"
             "Answer the question as a helpful assistant who has watched the video. Start with the direct answer as one complete, natural "
             "sentence (not a single word). Then add one or two short sentences describing what is visible in the video that supports this "
             "answer (the relevant objects, motions, order of events). Only describe what you can actually see; do not guess causes, "
             "intentions or anything not shown. Do not mention options, letters, or that you were given the answer. Write only the answer.")
        msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": u}]}]
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def keep(it, a):
        a = a.strip(); nw = len(a.split()); gold = it["options"][it["gold"]].strip().rstrip(".").lower()
        if not (10 <= nw <= 90) or BAN.search(a): return False
        if gold in ("yes", "no"):
            first = a.split(".")[0]
            return bool(NEG.search(first)) == (gold == "no")
        gw = content_words(gold); aw = {w[:4] for w in content_words(a)}
        return not gw or sum(w[:4] in aw for w in gw) / len(gw) >= 0.6

    print(f"fullans_vl {k}/{n}: {len(idx)} items to write ({len(done)} done)", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model=M, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85, seed=k, limit_mm_per_prompt={"video": 1},
              mm_processor_kwargs={"max_pixels": 384 * 384})
    sp = SamplingParams(temperature=0.3, top_p=0.8, top_k=20, max_tokens=160, seed=k)
    f = open(out, "a"); nk = 0
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(10)
    def load(i):
        it = items[i]
        try: return frames(os.path.join(it["prep_root"], it["video"]), it.get("t0"), it.get("t1"))
        except Exception as e: return None
    chunks = [idx[c:c + 512] for c in range(0, len(idx), 512)]
    nxt = [ex.submit(load, i) for i in chunks[0]] if chunks else []      # (17:18) decode the next chunk while the GPU generates
    for ci, ch in enumerate(chunks):
        c = ci * 512; vids = [x.result() for x in nxt]
        nxt = [ex.submit(load, i) for i in chunks[ci + 1]] if ci + 1 < len(chunks) else []
        reqs, keepi = [], []
        for i, v in zip(ch, vids):
            if v is None: f.write(json.dumps({"i": i, "ok": False, "err": "video"}) + "\n"); continue
            reqs.append({"prompt": prompt(items[i]), "multi_modal_data": {"video": [v]}}); keepi.append(i)
        try: res = llm.generate(reqs, sp) if reqs else []
        except Exception as e:                 # one bad request fails the whole batch: retry one by one, skip the failing ones
            res, ki = [], []
            for rq, i in zip(reqs, keepi):
                try: res.append(llm.generate([rq], sp)[0]); ki.append(i)
                except Exception: f.write(json.dumps({"i": i, "ok": False, "err": "gen"}) + "\n")
            keepi = ki
        for i, r in zip(keepi, res):
            it = items[i]; a = r.outputs[0].text.strip(); ok = keep(it, a)
            rec = {"i": i, "src": it["src"], "video": it["video"], "prep_root": it["prep_root"], "t0": it.get("t0"), "t1": it.get("t1"),
                   "question": it["question"], "gold_option": it["options"][it["gold"]], "answer": a, "ok": ok}
            nk += ok; f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        f.flush(); print(f"{c + len(ch)}/{len(idx)} kept {nk}", flush=True)
    print("FULLANS_VL DONE", flush=True)


if __name__ == "__main__":
    main()
