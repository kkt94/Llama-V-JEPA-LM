"""(10-21) Public VideoLLMs (Qwen3-VL-4B, InternVL3.5-4B, PLM-3B; 10-22 Molmo2-4B, Qwen2.5-VL-3B; 10-24 Gemma 4 E4B) on OUR benchmark item files under OUR protocol, for a same-condition table:
the same items (manifests/eval/pilot/<bench>.jsonl), the same frame count (16, uniform over the item's window / whole video), the same prompt
(question + lettered options + "Answer with the option's letter from the given choices directly.") and the same scoring (the letter with the
highest next-token logit among the item's options). Resolution / tokens per frame: each model's own processor defaults.
Resumable: items already in <out>/<bench>.jsonl are skipped. Scores: run_suite's scorer (python baseline_eval.py score ...).

  python baseline_eval.py run --model qwen3vl --path /data/llama_vjepa_lm/models/Qwen3-VL-4B-Instruct --benches tomato,mvbench --out DIR
  python baseline_eval.py score --benches tomato,mvbench --out DIR
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, "/data/llama_vjepa_lm/training")
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/eval")
PILOT = os.environ.get("BASE_EVAL_PILOT", "/data/llama_vjepa_lm/manifests/eval/pilot")   # (10-28) BASE_EVAL_PILOT: another item folder (analysis subsets)
LETTERS = "ABCDEFGHIJ"
SUFFIX = "Answer with the option's letter from the given choices directly."


def prompt(item):
    if not item.get("options"):                      # (10-22) open answers (VSI-Bench numeric, Charades-STA): the item's own instruction
        return item["question"]
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(item["options"]))
    return f"{item['question']}\n{opts}\n{SUFFIX}"


def window(item):
    if item.get("cut_sec") is not None:
        return 0.0, float(item["cut_sec"])
    if item.get("start_sec") is not None and item.get("end_sec") is not None:
        return float(item["start_sec"]), float(item["end_sec"])
    return None


def shuffle_index(item_id, nf, unit=2):
    """(10-28) fixed per-item permutation of the frame order in units of `unit` consecutive frames (2: a V-JEPA / Qwen3-VL tubelet keeps
    its own short motion, the order of the units is shuffled); seeded by the item id (md5), identical in runner.py"""
    import hashlib
    perm = np.random.RandomState(int(hashlib.md5(str(item_id).encode()).hexdigest()[:8], 16)).permutation(nf // unit)
    return (perm[:, None] * unit + np.arange(unit)[None]).reshape(-1)


def frames(item, nf):
    """16 frames uniform over the window (segment centres) -> uint8 [T, H, W, 3], their indices, fps, total frames"""
    if item.get("tq"):                                # (11-04) composed / transformed clips (T4 generalisation sets): the SAME pixels as our runner
        import sys as _s                              # (jepa_llm.data.video compose_clip / seq_clip, 384 x 384), then the item's aug (flip / rev)
        if "/data/llama_vjepa_lm/training" not in _s.path: _s.path.insert(0, "/data/llama_vjepa_lm/training")
        from jepa_llm.data.video import compose_clip, seq_clip, MEAN, STD
        sp = item["tq"]
        if sp.get("kind") == "seq":
            v, t = seq_clip(sp["segs"], "/", nf)
        else:
            v, t = compose_clip(sp, item["video"], nf, window=window(item), path2=sp.get("v2"), window2=tuple(sp["w2"]) if sp.get("w2") else None)[:2]
        f = ((v * STD + MEAN).clamp(0, 1) * 255).round().byte().permute(1, 2, 3, 0).numpy()   # [T, H, W, 3]
        if item.get("aug") == "flip": f = np.ascontiguousarray(f[:, :, ::-1])
        elif item.get("aug") == "rev": f = np.ascontiguousarray(f[::-1])
        t = np.asarray(t, dtype=np.float64); dur = max(float(t[-1] - t[0]) * len(t) / max(len(t) - 1, 1), 1e-3)
        return np.ascontiguousarray(f), np.arange(len(f)), len(f) / dur, len(f)
    import decord
    vr = decord.VideoReader(item["video"], num_threads=1)
    fps, n = float(vr.get_avg_fps()), len(vr)
    w = window(item)
    t0, t1 = (0.0, n / fps) if w is None else (max(0.0, w[0]), min(n / fps, w[1]))
    if t1 <= t0:
        t0, t1 = 0.0, n / fps
    spd = os.environ.get("BASE_EVAL_SPEED", "")           # (10-28) measurement only: playback speed s (0.5 | 1 | 2). The frames cover a centred
    if spd:                                               # sub-window of min(1, 0.5 s) of the span (s = 1: the middle half, s = 2: all of it,
        s = float(spd); c, L = (t0 + t1) / 2, (t1 - t0) * min(1.0, 0.5 * s)  # s = 0.5: the middle quarter) and the reported fps is fps x s,
        t0, t1 = c - L / 2, c + L / 2                     # so every condition shows 32 frames over the same apparent duration (half the span)
        fps_rep = fps * s                                 # and only the real motion per frame step changes (s = 2: twice as fast)
    times = t0 + (np.arange(nf) + 0.5) * (t1 - t0) / nf
    idx = np.clip(np.round(times * fps).astype(int), 0, n - 1)
    f = vr.get_batch(idx.tolist()).asnumpy()
    if item.get("aug") == "flip":                     # (10-23) left/right diagnostic items: the clip mirrored (as runner._aug)
        f = np.ascontiguousarray(f[:, :, ::-1])
    elif item.get("aug") == "rev":
        f = np.ascontiguousarray(f[::-1])
    mode = os.environ.get("BASE_EVAL_VIDEO", "")          # (10-28) video controls for the baselines, as runner.py's JEPA_EVAL_VIDEO:
    if mode == "held":                                    # the middle frame repeated (no motion), time stamps unchanged
        f = np.ascontiguousarray(np.repeat(f[len(f) // 2:len(f) // 2 + 1], len(f), 0))
    elif mode == "backwards":                             # frames in reverse order (time reversed), time stamps unchanged
        f = np.ascontiguousarray(f[::-1])
    elif mode == "hflip":                                 # (11-04) the clip mirrored left <-> right (as runner.py's JEPA_EVAL_VIDEO=hflip)
        f = np.ascontiguousarray(f[:, :, ::-1])
    elif mode == "blank":                                 # no visual content
        f = np.zeros_like(f)
    elif mode == "shuffle":                               # (10-28) frame order shuffled in 2-frame units (fixed per item), time stamps unchanged
        f = np.ascontiguousarray(f[shuffle_index(item["id"], len(f))])
    if spd:
        return f, idx, fps_rep, n
    return f, idx, fps, n


class Items(torch.utils.data.Dataset):
    def __init__(self, items, nf, prep=None):
        self.items, self.nf, self.prep = items, nf, prep     # (10-22) prep: the model's processor run in the loader workers (CPU tensors)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        try:
            if it.get("image"):                      # (10-28) image QA items: the image itself (PIL RGB), fed through the model's image path
                from PIL import Image
                ex = {"item": it, "image": Image.open(it["image"]).convert("RGB")}
            else:
                f, idx, fps, n = frames(it, self.nf)
                ex = {"item": it, "frames": f, "idx": idx, "fps": fps, "n": n}
            if self.prep is not None:
                torch.set_num_threads(1)
                ex["x"], ex["text"] = self.prep(ex)
                ex.pop("frames", None); ex.pop("image", None)
            return ex
        except Exception as e:
            return {"item": it, "error": f"{type(e).__name__}: {e}"[:200]}


def load(model, path, device):
    from transformers import AutoProcessor
    if model == "molmo2":                              # (10-22) no native transformers class: the model folder's own code (read 10-22:
        os.environ["HF_HUB_OFFLINE"] = "1"             # no network or process calls except for URL videos, which we never pass), offline
        from transformers import AutoModelForImageTextToText
        proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        m = AutoModelForImageTextToText.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16).to(device).eval()
        return proc, m
    if model == "keye":                                # (11-19) Keye-VL 2.0 30B-A3B: the model folder's own code (read 11-19: no network / process calls), transformers 4.57 site
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        from transformers import AutoModelForCausalLM
        proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16).to(device).eval()
        import importlib.util as iu                    # (11-19) its dense attention matrix does not fit next to the model at 8-16k tokens: same arithmetic by groups of heads
        sp = iu.spec_from_file_location("zz_keye_attn", "/data/llama_vjepa_lm/training/scripts/data_gen/zz_keye_attn.py"); ka = iu.module_from_spec(sp); sp.loader.exec_module(ka); ka.patch(m)
        return proc, m
    proc = AutoProcessor.from_pretrained(path)
    if model == "gemma4":                              # (10-24) Gemma 4 E4B-it: native transformers class, thinking off (E4B emits no thought block)
        from transformers import Gemma4ForConditionalGeneration as M
    elif model == "qwen25vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as M
    elif model == "qwen3vl":
        from transformers import AutoConfig                # (11-18) Qwen3-VL-30B-A3B is the MoE variant of the same family (same processor and prompts)
        if AutoConfig.from_pretrained(path).model_type == "qwen3_vl_moe":
            from transformers import Qwen3VLMoeForConditionalGeneration as M
        else:
            from transformers import Qwen3VLForConditionalGeneration as M
    elif model == "minicpm":                           # (11-19) MiniCPM-V 4.7 35B-A3B: native class, transformers >= 5.18 site
        from transformers import MiniCPMV4_7ForConditionalGeneration as M
    elif model == "internvl":
        from transformers import InternVLForConditionalGeneration as M
    elif model == "plm":
        from transformers import PerceptionLMForConditionalGeneration as M
    elif model == "videollama3":                       # (10-27) VideoLLaMA3-2B (video model) weights renamed to the native class's layout
        from transformers import VideoLlama3ForConditionalGeneration as M   # (models/VideoLLaMA3-2B-HF: no remote code)
    elif model == "varco":                             # (11-04) NC AI VARCO-VISION-2.0 (LLaVA-OneVision: SigLIP2 + Qwen3), native class, no remote code
        from transformers import LlavaOnevisionForConditionalGeneration as M
    else:
        raise ValueError(model)
    m = M.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
    return proc, m


def inputs(model, proc, ex, device=None):
    if "image" in ex:                                  # (10-28) image QA items: the image natively as ONE image (not a video), each model's own
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt(ex["item"])}]}]   # image processor defaults
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **({"enable_thinking": False} if model == "gemma4" else {}))
        x = proc(text=[text], images=[ex["image"]], return_tensors="pt")
        return ({k: v.to(device) for k, v in x.items()} if device is not None else dict(x)), text
    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt(ex["item"])}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **({"enable_thinking": False} if model in ("gemma4", "minicpm") else {}))
    if os.environ.get("BASE_EVAL_NOTHINK") == "1":     # (11-19) templates without a thinking switch: open the answer with an empty thinking block
        text += "<think>\n\n</think>\n\n"
    vid = ex["frames"]
    if model == "gemma4":                              # (10-24) our 32 frames, processor defaults (70 soft tokens / frame); real timestamps from the metadata
        meta = {"fps": ex["fps"], "frames_indices": [int(i) for i in ex["idx"]], "total_num_frames": ex["n"], "duration": ex["n"] / ex["fps"]}
        x = proc(text=[text], videos=[vid], video_metadata=[meta], do_sample_frames=False, return_tensors="pt")
    elif model == "qwen3vl":
        meta = {"fps": ex["fps"], "frames_indices": [int(i) for i in ex["idx"]], "total_num_frames": ex["n"], "duration": ex["n"] / ex["fps"]}
        x = proc(text=[text], videos=[vid], video_metadata=[meta], do_sample_frames=False, return_tensors="pt")
    elif model == "keye":                              # (11-19) our 32 frames, per-frame size range of the model card's video example (128-512 x 28^2), real time stamps
        sr = sys.modules[type(proc.image_processor).__module__].smart_resize
        H, W = sr(vid.shape[1], vid.shape[2], factor=28, min_pixels=128 * 28 * 28, max_pixels=512 * 28 * 28)
        x = proc(text=[text], videos=[torch.from_numpy(np.ascontiguousarray(vid)).permute(0, 3, 1, 2)], width=[W], height=[H],
                 timestamps=[torch.as_tensor(np.asarray(ex["idx"]) / ex["fps"], dtype=torch.float32)], return_tensors="pt")
    elif model == "minicpm":                           # (11-19) our 32 frames, the model card's video settings (16x downsampling, one tile per frame)
        x = proc(text=[text], videos=[vid], do_sample_frames=False, downsample_mode="16x", max_slice_nums=1, stack_frames=1, return_tensors="pt")
    elif model == "plm":
        x = proc(text=[text], videos=[vid], do_sample_frames=False, return_tensors="pt")
    elif model == "varco":                             # (11-04) our 32 frames, the processor's own video defaults (384 px, LLaVA-OneVision frame pooling)
        x = proc(text=[text], videos=[vid], do_sample_frames=False, return_tensors="pt")
    elif model == "videollama3":                       # (10-27) our 32 frames; the video model's own budget (processor_config max_tokens 16384 = patches
        meta = {"fps": ex["fps"], "frames_indices": [int(i) for i in ex["idx"]], "total_num_frames": ex["n"], "duration": ex["n"] / ex["fps"]}
        px = 16384 * 14 * 14                           # of the WHOLE video; size.longest_edge is the whole video's pixel budget: 32 x 16x30 patches ->
                                                       # ~3840 LLM tokens after 2x2 merge); token compression on (as shipped)
        x = proc(text=[text], videos=[vid], video_metadata=[meta], do_sample_frames=False, size={"shortest_edge": 12544, "longest_edge": px}, return_tensors="pt")
    elif model == "molmo2":                            # our frames (32 uniform), the model's own sampling off; timestamps from the metadata
        h, w = vid.shape[1], vid.shape[2]
        meta = {"fps": ex["fps"], "frames_indices": [int(i) for i in ex["idx"]], "total_num_frames": ex["n"], "duration": ex["n"] / ex["fps"],
                "width": w, "height": h}
        x = proc(text=[text], videos=[vid], video_metadata=[meta], do_sample_frames=False, return_tensors="pt")
    elif model == "qwen25vl":                          # frames as given; fps of the sampled frames so the time encoding is in real seconds
        dur = ex["n"] / ex["fps"]; w_ = window(ex["item"]); span = (w_[1] - w_[0]) if w_ else dur
        # token budget like the other baselines (~256 per frame: <= 512 per 2-frame group, ~8k for 32 frames; the default gave ~43k)
        x = proc(text=[text], videos=[vid], fps=len(ex["idx"]) / max(span, 1e-3), do_sample_frames=False,
                 size={"shortest_edge": 56 * 56, "longest_edge": 512 * 28 * 28}, return_tensors="pt")
    else:
        # InternVL3.5-HF: the shipped video_preprocessor_config says 384x384, but the vision tower is 448 (32x32 patches, 256 tokens after pixel shuffle)
        x = proc(text=[text], videos=[vid], do_sample_frames=False, size={"height": 448, "width": 448}, return_tensors="pt")
    return ({k: (v.to(device) if hasattr(v, "to") else v) for k, v in x.items()} if device is not None else dict(x)), text


@torch.no_grad()
def run(a):
    device = torch.device(a.device)
    proc, m = load(a.model, a.path, device)
    tok = proc.tokenizer
    lid = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    os.makedirs(a.out, exist_ok=True)
    for b in a.benches.split(","):
        items = [json.loads(l) for l in open(f"{PILOT}/{b}.jsonl")][: a.limit]
        of = f"{a.out}/{b}.jsonl"
        done = set()
        if a.shard:                                      # (10-24) --shard i/n: items idx % n == i, written to <bench>.s<i>of<n>.jsonl (merge later); the main file counts as done
            si, sn = map(int, a.shard.split("/"))
            if os.path.exists(of):
                for l in open(of):
                    try: done.add(json.loads(l)["id"])
                    except Exception: pass
            items = [x for j, x in enumerate(items) if j % sn == si]; of = f"{a.out}/{b}.s{si}of{sn}.jsonl"
        if os.path.exists(of):                           # (10-24) a run killed mid-write may leave a partial last line: drop it before appending
            lines = open(of).read().split("\n")
            good = []
            for l in lines:
                try: done.add(json.loads(l)["id"]); good.append(l)
                except Exception: pass
            if len(good) != len([l for l in lines if l.strip()]):
                open(of, "w").write("".join(l + "\n" for l in good)); print(b, "dropped", len([l for l in lines if l.strip()]) - len(good), "partial line(s)", flush=True)
        todo = [x for x in items if x["id"] not in done]
        print(b, "items", len(items), "done", len(done), "todo", len(todo), flush=True)
        # (10-22) Molmo2's processor (no fast version: per-frame crops in Python, >10 s per 32-frame item) runs in the loader workers
        prep = (lambda ex: inputs(a.model, proc, ex)) if a.model in ("molmo2", "keye", "minicpm") and a.workers else None
        dl = torch.utils.data.DataLoader(Items(todo, a.nf, prep), batch_size=None, num_workers=a.workers, prefetch_factor=2 if a.workers else None)
        with open(of, "a") as f:
            for k, ex in enumerate(dl):
                it = ex["item"]
                if "error" in ex:
                    f.write(json.dumps({"id": it["id"], "pred_index": None, "error": ex["error"]}) + "\n"); continue
                if "x" in ex:
                    x, text = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in ex["x"].items()}, ex["text"]
                else:
                    x, text = inputs(a.model, proc, ex, device)
                if k == 0 and not done:
                    print("prompt tail:", repr(text[-160:]), "| input tokens", int(x["input_ids"].shape[1]), flush=True)
                if it.get("options") and os.environ.get("BASE_EVAL_GEN") == "1":   # (11-12, review Q3) generation + parsing instead of letter logits
                    from jepa_llm.eval.runner import parse_option
                    g = m.generate(**x, max_new_tokens=16, do_sample=False)
                    resp = tok.decode(g[0, x["input_ids"].shape[1]:], skip_special_tokens=True)
                    f.write(json.dumps({"id": it["id"], "pred_index": parse_option(resp, it["options"]), "response": resp}) + "\n")
                    continue
                if not it.get("options"):                # (10-22) open answer: greedy decoding, same instruction as our model gets
                    g = m.generate(**x, max_new_tokens=32, do_sample=False)
                    f.write(json.dumps({"id": it["id"], "pred_index": None, "response": tok.decode(g[0, x["input_ids"].shape[1]:], skip_special_tokens=True)}) + "\n")
                    continue
                logits = m(**x).logits[0, -1].float()
                sc = [float(logits[lid[i]]) for i in range(len(it["options"]))]
                f.write(json.dumps({"id": it["id"], "pred_index": int(np.argmax(sc)), "scores": [round(s, 4) for s in sc]}) + "\n")
                if k % 200 == 0:
                    f.flush(); print(b, k, "/", len(todo), flush=True)
        print(b, "finished", flush=True)


def score(a):
    from run_suite import generic_pairs, scorer, special_score
    from jepa_llm.eval.metrics import score as sc_fn
    res = json.load(open(f"{a.out}/scores.json")) if os.path.exists(f"{a.out}/scores.json") else {}
    for b in a.benches.split(","):
        items = [json.loads(l) for l in open(f"{PILOT}/{b}.jsonl")]
        of = f"{a.out}/{b}.jsonl"
        if not os.path.exists(of):
            continue
        recs = [json.loads(l) for l in open(of)]
        preds = {r["id"]: r.get("pred_index") for r in recs}
        if sum(1 for x in items if x["id"] in preds) < len(items):
            print(b, "incomplete", len(preds), "/", len(items)); continue
        s = scorer(b)
        sp = special_score(b, items, recs, preds)
        res[b] = sp if sp is not None else generic_pairs(items, preds) if s == "generic" else sc_fn(s, items, preds)
        print(b, json.dumps({k: v for k, v in res[b].items() if not isinstance(v, dict)})[:300])
    json.dump(res, open(f"{a.out}/scores.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"]); ap.add_argument("--model"); ap.add_argument("--path")
    ap.add_argument("--benches", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--nf", type=int, default=16); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--limit", type=int); ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard")
    a = ap.parse_args()
    # (10-28) decoding (1080p clips: ~3 s per item on one core) left the GPUs idle with 4 loader workers while the nodes still had free
    # cores: at least 8 workers unless BASE_EVAL_MIN_WORKERS says otherwise (only the loading parallelism changes; outputs are keyed by id)
    a.workers = max(a.workers, int(os.environ.get("BASE_EVAL_MIN_WORKERS", 8)))
    run(a) if a.cmd == "run" else score(a)
