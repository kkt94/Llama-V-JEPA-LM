"""(10-28) lcap step 3 (GPU, one card): text-only prior of every gold and candidate caption under the BASE Qwen3-1.7B (no video):
log p(caption | 'Video caption: ') summed over the caption tokens, exactly as logs/zz_tb_lmprior.py (prefix and caption tokenized apart,
no special tokens) but bf16 on GPU and batched by length.  Sanity: the 676 TemporalBench items of zz_tb_lmprior.json (fp32 CPU) are
re-scored here and the sign agreement of the option difference is printed.
-> logs/zz_lcap_lp.parquet (text, lp, ntok)   usage: CUDA_VISIBLE_DEVICES=<g> python -I zz_lcap_lmscore.py"""
import os, json, time, numpy as np, pandas as pd, torch
SMOKE = os.environ.get("LCAP_SMOKE"); DEV = "cpu" if SMOKE else "cuda"
from transformers import AutoTokenizer, AutoModelForCausalLM
R = "/data/llama_vjepa_lm"; M = f"{R}/models/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(M)
lm = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
G = pd.read_parquet(f"{R}/logs/zz_lcap_golds.parquet"); C = pd.read_parquet(f"{R}/logs/zz_lcap_cands.parquet")
G = G[G.gid.isin(set(C.gid))]
tb = json.load(open(f"{R}/logs/zz_tb_lmprior.json"))
TB = {json.loads(l)["id"]: json.loads(l)["options"] for l in open(f"{R}/manifests/eval/pilot/temporalbench_short_qa.jsonl")}
tb_texts = [o for k in tb for o in TB[k]]
texts = list(dict.fromkeys(list(G.cap) + list(C.neg) + tb_texts))
if SMOKE: texts = texts[:6] + tb_texts[:6]; tb = {k: tb[k] for k in list(tb)[:3]}
pre = tok("Video caption: ", add_special_tokens=False).input_ids
ids = tok(texts, add_special_tokens=False).input_ids
order = np.argsort([len(x) for x in ids])[::-1]
print("texts", len(texts), "tokens", sum(len(x) for x in ids), flush=True)
lp = np.zeros(len(texts)); BUD = 12000; t = time.time(); i = 0; nb = 0
with torch.no_grad():
    while i < len(order):
        L = len(pre) + len(ids[order[i]]); B = max(1, BUD // L); sel = order[i:i + B]; i += B
        x = torch.full((len(sel), L), tok.pad_token_id or 0, dtype=torch.long); am = torch.zeros((len(sel), L), dtype=torch.long)
        for j, k in enumerate(sel):
            s = pre + ids[k]; x[j, :len(s)] = torch.tensor(s); am[j, :len(s)] = 1
        x, am = x.to(DEV), am.to(DEV)
        logits = lm(input_ids=x, attention_mask=am).logits[:, len(pre) - 1:-1]          # predicts caption tokens
        tgt = x[:, len(pre):]; m = am[:, len(pre):].bool()
        lsm = torch.log_softmax(logits.float(), -1).gather(2, tgt[..., None])[..., 0]
        out = (lsm * m).sum(1).cpu().numpy()
        for j, k in enumerate(sel): lp[k] = out[j]
        nb += 1
        if nb % 200 == 0: print(f"{i}/{len(order)} {time.time() - t:.0f}s", flush=True)
D = pd.DataFrame(dict(text=texts, lp=lp, ntok=[len(x) for x in ids]))
D.to_parquet(f"{R}/logs/zz_lcap_lp{'_smoke' if SMOKE else ''}.parquet")
mp = dict(zip(texts, lp)); agree = []
for k, v in tb.items():
    o = TB[k]; d_gpu = mp[o[0]] - mp[o[1]]; d_cpu = v[0] - v[1]; agree.append((np.sign(d_gpu) == np.sign(d_cpu), abs(d_gpu - d_cpu)))
a = np.array(agree)
print(f"TB sanity vs fp32 CPU: sign agreement {a[:, 0].mean():.4f} over {len(a)}; |diff err| median {np.median(a[:, 1]):.3f} p95 {np.percentile(a[:, 1], 95):.3f}", flush=True)
print(f"LMSCORE DONE {time.time() - t:.0f}s", flush=True)
