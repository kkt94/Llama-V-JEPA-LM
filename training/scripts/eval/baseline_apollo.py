"""(10-28) Apollo-3B-t32 (baseline video LMM; models/Apollo-3B-t32 = ModelScope mirror of Apollo-LMMs/Apollo-3B-t32) on OUR benchmark
item files under OUR protocol (baseline_eval.py: same items, 32 frames uniform over the item's window / whole video, same prompt, letter-logit
scoring, same output rows {id, pred_index, scores} in <out>/<bench>.jsonl, item-level resume, --shard). Score with the usual
`baseline_eval.py score --benches ... --out runs/eval/base32_apollo3b` (jepa-llm env).

Apollo has no native transformers class: this runs the model folder's own code (modeling_apollo.py, vision_tower.py, mm_connector.py,
internvideo2/*; read in full 10-28: no network / process / exec calls; only local from_pretrained paths, which resolve to sub-folders of
the model dir; eval() only on the config's "torch.bfloat16" string; the torch.load / save paths are never reached at inference)
under transformers 4.44.2 in its own venv (/path/to/venv), offline (HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1).

How Apollo is fed (its README's ApolloMMLoader + conv_templates["qwen_2"]; the `apollo` package itself is not in the model folder, so its
video loader / prompt are re-implemented here from the README pipeline):
  * vision input: clips of frames_per_clip = 4 frames; every clip goes to BOTH encoders: SigLIP-SO400M (per frame, 384 px, the shipped
    SiglipImageProcessor) and InternVideo2 (4-frame clip, 224 px, the shipped transform: resize, centre crop, /255, ImageNet norm). Each
    frame is first padded to a square with the processor's mean colour (expand2square, as Apollo's process_video), so nothing is cropped
    away. Features -> Perceiver connector -> 128 LLM tokens per clip (t32 = 32 tokens per frame).
  * OUR frames as the frame source: the 32 uniformly sampled frames (baseline_eval.frames, segment centres over the window) are cut, in
    order, into 8 consecutive clips of 4 frames -> 8 x 128 = 1024 visual tokens. Apollo's own loader samples clips of 4 frames at 2 fps
    (one 2 s clip per ~3 s of video, clip_sampling_ratio 0.65); our clips have the frame spacing span/32 (equal to Apollo's 0.5 s for a
    16 s video, sparser for longer ones) and the same count for every item (the baseline protocol: equal frames for all models).
  * prompt: Apollo's timestamped video prompt, then the question:
      "You are provided the following series of eight, 2 second clips from a 0:00:16.016667 [H:MM:SS] video.\n"
      "0.3-1.8 seconds: <128 video tokens>,\n2.3-3.8 seconds: <128 video tokens>, ...\n\n<question + lettered options + suffix>"
    with each clip's real start / end time (its first / last frame, relative to the window start), the window length as the video length
    and "2 second" = the config's clip_duration (the constant text Apollo always writes), inside the qwen_2 chat format with Apollo's
    system line ("You are a helpful assistant."). Video tokens are the X_TOKEN_INDEX (-200) placeholders the model replaces with features.
  * scoring: next-token logits at the assistant start, the option letters (as baseline_eval).

  python baseline_apollo.py [--model /data/llama_vjepa_lm/models/Apollo-1.5B-t32] --benches tempcompass_mc,mvbench --out /data/llama_vjepa_lm/runs/eval/base32_apollo3b [--limit 40]
"""
import argparse, datetime, json, os, sys
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", "/path/to/venv/hf_home")        # remote-code module copies + hub stub live with the venv
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/eval")
from baseline_eval import PILOT, LETTERS, prompt, window, Items           # (imports only numpy / torch / decord: works under 4.44)

MODEL = "/data/llama_vjepa_lm/models/Apollo-3B-t32"
X_TOKEN_INDEX = -200                    # modeling_apollo.X_TOKEN_INDEX
VT = "<|apollo_video_clip|>"            # text placeholder, split out before tokenising (never reaches the tokenizer)
FPC = 4                                 # frames per clip (README: frames_per_clip=4; InternVideo2 num_frames 4)
SYSTEM = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"


def num_words(n):
    from num2words import num2words
    return num2words(n)


def expand2square(img, bg):
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    out = Image.new(img.mode, (s, s), bg)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


class Prep:
    """item frames -> (model inputs on CPU, prompt text); runs in the DataLoader workers"""
    def __init__(self, tok, processors, clip_duration, n_tok):
        self.tok, self.processors, self.clip_duration, self.n_tok = tok, processors, clip_duration, n_tok

    def video(self, frames):
        imgs = [Image.fromarray(f) for f in frames]
        out = []
        for p in self.processors:                                          # [SigLIP, InternVideo2] (config vision_towers order)
            bg = tuple(int(x * 255) for x in p.image_mean)
            sq = [expand2square(im, bg) for im in imgs]
            clips = []
            for c in range(len(sq) // FPC):
                x = p.preprocess(sq[c * FPC:(c + 1) * FPC], return_tensors="pt")["pixel_values"]
                if isinstance(x, list):
                    assert len(x) == 1
                    x = x[0]
                clips.append(torch.as_tensor(x))
            out.append(torch.stack(clips).to(torch.bfloat16))               # SigLIP [Nc, 4, 3, 384, 384]; InternVideo2 [Nc, 3, 4, 224, 224]
        return out

    def text(self, ex):
        it, fps, n = ex["item"], ex["fps"], ex["n"]
        w = window(it)                                                      # same window as baseline_eval.frames
        t0, t1 = (0.0, n / fps) if w is None else (max(0.0, w[0]), min(n / fps, w[1]))
        if t1 <= t0:
            t0, t1 = 0.0, n / fps
        ts = np.asarray(ex["idx"], dtype=np.float64) / fps - t0
        nc = len(ts) // FPC
        head = (f"You are provided the following series of {num_words(nc)}, {self.clip_duration} second clips from a "
                f"{datetime.timedelta(seconds=float(t1 - t0))} [H:MM:SS] video.\n")
        clips = ",\n".join(f"{round(float(ts[c * FPC]), 1)}-{round(float(ts[c * FPC + FPC - 1]), 1)} seconds: {VT}" for c in range(nc))
        user = head + clips + "\n\n" + prompt(it)
        return f"{SYSTEM}<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

    def ids(self, text):                                                    # Apollo's tokenizer_mm_token: text pieces + n_tok placeholders per clip
        parts = text.split(VT)
        ids = []
        for j, p in enumerate(parts):
            if p:
                ids += self.tok(p).input_ids
            if j < len(parts) - 1:
                ids += [X_TOKEN_INDEX] * self.n_tok
        return torch.tensor([ids], dtype=torch.long)

    def __call__(self, ex):
        torch.set_num_threads(1)
        assert len(ex["frames"]) % FPC == 0
        text = self.text(ex)
        return {"input_ids": self.ids(text), "video": self.video(ex["frames"])}, text


def load(device, path=MODEL):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, low_cpu_mem_usage=True)
    m = m.to(device=device, dtype=torch.bfloat16).eval()
    return m


@torch.no_grad()
def run(a):
    device = torch.device(a.device)
    m = load(device, a.model)
    tok = m.tokenizer
    prep = Prep(tok, m.vision_tower.vision_processor, m.config.clip_duration, m.config.mm_connector_cfg["num_output_tokens"])
    lid = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    os.makedirs(a.out, exist_ok=True)
    for b in a.benches.split(","):
        items = [json.loads(l) for l in open(f"{PILOT}/{b}.jsonl")][: a.limit]
        of = f"{a.out}/{b}.jsonl"
        done = set()
        if a.shard:                                      # --shard i/n: items idx % n == i -> <bench>.s<i>of<n>.jsonl; the main file counts as done
            si, sn = map(int, a.shard.split("/"))
            if os.path.exists(of):
                for l in open(of):
                    try: done.add(json.loads(l)["id"])
                    except Exception: pass
            items = [x for j, x in enumerate(items) if j % sn == si]; of = f"{a.out}/{b}.s{si}of{sn}.jsonl"
        if os.path.exists(of):                           # a run killed mid-write may leave a partial last line: drop it before appending
            lines = open(of).read().split("\n")
            good = []
            for l in lines:
                try: done.add(json.loads(l)["id"]); good.append(l)
                except Exception: pass
            if len(good) != len([l for l in lines if l.strip()]):
                open(of, "w").write("".join(l + "\n" for l in good)); print(b, "dropped", len([l for l in lines if l.strip()]) - len(good), "partial line(s)", flush=True)
        todo = [x for x in items if x["id"] not in done]
        print(b, "items", len(items), "done", len(done), "todo", len(todo), flush=True)
        dl = torch.utils.data.DataLoader(Items(todo, a.nf, prep), batch_size=None, num_workers=a.workers, prefetch_factor=4 if a.workers else None)
        ntok = []
        with open(of, "a") as f:
            for k, ex in enumerate(dl):
                it = ex["item"]
                if "error" in ex:
                    f.write(json.dumps({"id": it["id"], "pred_index": None, "error": ex["error"]}) + "\n"); continue
                x, text = ex["x"], ex["text"]
                ids = x["input_ids"].to(device)
                vid = [[v.to(device, non_blocking=True) for v in x["video"]]]
                ntok.append(int(ids.shape[1]))
                if k == 0 and not done:
                    print("prompt head:", repr(text[:300].replace(VT, "<V>")), "\nprompt tail:", repr(text[-160:]), "| input tokens", ntok[-1],
                          "| video", [tuple(v.shape) for v in vid[0]], flush=True)
                logits = m(input_ids=ids, vision_input=vid, data_types=["video"]).logits[0, -1].float()
                sc = [float(logits[lid[i]]) for i in range(len(it["options"]))]
                f.write(json.dumps({"id": it["id"], "pred_index": int(np.argmax(sc)), "scores": [round(s, 4) for s in sc]}) + "\n")
                f.flush()                                # every row: a preempted run (killed for a main evaluation) loses nothing
                if k % 200 == 0:
                    print(b, k, "/", len(todo), flush=True)
        if ntok:
            print(b, "input tokens mean", round(float(np.mean(ntok)), 1), "min", min(ntok), "max", max(ntok), flush=True)
        print(b, "finished", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--nf", type=int, default=32); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--shard")
    ap.add_argument("--model", default=MODEL)   # (10-28) also models/Apollo-1_5B-t32 (symlink to Apollo-1.5B-t32: a "." in the folder name breaks the remote-code module import; same .py files byte for byte, only the LLM size differs)
    run(ap.parse_args())
