"""(10-30, CoT stage data B: video-grounded reasoning written by a JEPA-LLM that SEES the clip) the 4B (34k, best perception of our models)
reads the training clip and writes its own step-by-step reasoning: no-think template, the MC question with the CoT instruction, the assistant
turn prefilled with "Let's think step by step about what happens in the video.", greedy, <= 320 new tokens; then "\\nAnswer:" is appended and
the option letter read from the logits. Kept later (build step) only if that letter is the gold one and the reasoning is not degenerate.
Items: every 9th of runs/cot/A_items.jsonl (sources with descriptive ground truth), clip = prep_root/video (the training copy), t0/t1.
usage: python -I zz_cot_b_gen.py <export dir> <k> <n>   -> runs/cot/B_<k>of<n>.jsonl (resumable)"""
import json, sys, os, time, math
import torch
from transformers import AutoModel
P, k, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
R = "/data/llama_vjepa_lm"; OUT = f"{R}/runs/cot/B_{k}of{n}.jsonl"
items = [json.loads(l) for l in open(f"{R}/runs/cot/A_items.jsonl")]
idx = list(range(0, len(items), 9))[k::n]
done = {json.loads(l)["i"] for l in open(OUT)} if os.path.exists(OUT) else set(); idx = [i for i in idx if i not in done]
model = AutoModel.from_pretrained(P, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval(); tok = model.tokenizer
proc = sys.modules[[m for m in sys.modules if m.endswith("processing_jepa_llm")][0]]
model._prepare_rope(); llm = model.language_model; eos = tok.convert_tokens_to_ids("<|im_end|>")
COT = "Explain your reasoning step by step, then answer as 'Answer: <letter>'."
PRE = "Let's think step by step about what happens in the video."
def fnum(x):
    try: x = float(x); return None if math.isnan(x) else x
    except Exception: return None
def emb(ids, vis, ev): return model._embed(torch.tensor([ids], device="cuda"), torch.tensor([vis], device="cuda").bool(), ev)
print(f"B {k}/{n}: {len(idx)} items ({len(done)} done)", flush=True); f = open(OUT, "a"); t0 = time.time(); B = 16
for s in range(0, len(idx), B):
    E1, base, recs = [], [], []
    for i in idx[s:s + B]:
        it = items[i]; path = os.path.join(it["prep_root"], it["video"])
        try: ev = model._evidence(path, fnum(it.get("t0")), fnum(it.get("t1")))
        except Exception as e: f.write(json.dumps({"i": i, "error": repr(e)[:160]}) + "\n"); continue
        user = proc.mc_prompt(it["question"], it["options"]).replace(proc.MC_SUFFIX, COT)
        p = proc.build_generation_prompt(tok, f"{proc.VIS}\n{user}", ev.shape[1])
        ids = p["input_ids"] + tok(PRE, add_special_tokens=False)["input_ids"]; vis = p["vis_mask"] + [0] * (len(ids) - len(p["input_ids"]))
        E1.append(emb(ids, vis, ev)[0]); base.append((ids, vis, ev, len(it["options"]))); recs.append({"i": i, "src": it["src"], "gold": it["gold"]})
    if not E1: continue
    L = max(e.shape[0] for e in E1); E = torch.zeros(len(E1), L, E1[0].shape[1], dtype=E1[0].dtype, device="cuda"); M = torch.zeros(len(E1), L, dtype=torch.long, device="cuda")
    for j, e in enumerate(E1): E[j, L - e.shape[0]:] = e; M[j, L - e.shape[0]:] = 1
    with torch.no_grad(): g = llm.generate(inputs_embeds=E, attention_mask=M, max_new_tokens=320, do_sample=False, eos_token_id=eos, pad_token_id=eos)
    for r, (ids, vis, ev, no), x in zip(recs, base, g):
        x = [t for t in x.tolist() if t != eos]; full = ids + x + tok("\nAnswer:", add_special_tokens=False)["input_ids"]
        with torch.no_grad(): last = llm(inputs_embeds=emb(full, vis + [0] * (len(full) - len(vis)), ev), use_cache=False).logits[0, -1].float()
        sc = torch.maximum(last[proc.letter_ids(tok, no)], last[[tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in "ABCDEFGHIJ"[:no]]])
        r["pred"] = int(sc.argmax()); r["reason"] = PRE + tok.decode(x, skip_special_tokens=True); r["ntok"] = len(x)
        f.write(json.dumps(r) + "\n")
    f.flush(); print(f"B {k}: {s + B}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
print("B DONE", flush=True)
