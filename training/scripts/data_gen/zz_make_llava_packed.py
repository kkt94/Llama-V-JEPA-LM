"""(10-23, user: go) LLaVA-Video rows packed per video: the rows of ONE video from the SAME source file (same question generator; the oe
and mc files of NExT-QA / PT / ANetQA hold the same questions, so they are never packed together) go into one sequence, so the clip is
decoded and encoded once for all of them. Caption rows stay single (a caption can answer the questions after it). Rows are shuffled per
video (seeded) and filled greedily up to a text budget (Qwen3 tokens of the contents + 8 per message, <= BUDGET; the 1024 video slots and
the template fit in max_len 2560). Every assistant message keeps "seg" = its original row inside the pack (loss per original row,
config loss.seg_norm). -> manifests/stage1/llava_packed.parquet (columns of llava_clean + n_orig)
usage: zz_make_llava_packed.py [--budget 1250] [--src llava_clean] [--dst llava_packed]"""
import hashlib, json, random, sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
R = "/data/llama_vjepa_lm"; VIS = "<|jepa_evidence|>"
BUDGET = int(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 1250
ARG = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
SRC, DST = ARG("--src", "llava_clean"), ARG("--dst", "llava_packed")
_tok = None


def ntok(conv):
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(f"{R}/models/Qwen3-4B-Instruct-2507")
    msgs = json.loads(conv)
    return sum(len(_tok(m["content"].replace(VIS, ""), add_special_tokens=False)["input_ids"]) + 8 for m in msgs)


def pack(args):
    (video, file), rows = args
    rng = random.Random(int(hashlib.md5(f"{video}|{file}".encode()).hexdigest()[:8], 16)); rng.shuffle(rows)
    out, cur, used = [], [], 0
    def flush():
        if not cur:
            return
        msgs = []
        for j, r in enumerate(cur):
            for m in json.loads(r["conversations"]):
                m = dict(m)
                if m["role"] == "user" and j > 0:
                    m["content"] = m["content"].replace(VIS + "\n", "").replace(VIS, "")
                if m["role"] == "assistant":
                    m["seg"] = j
                msgs.append(m)
        base = dict(cur[0]); base["conversations"] = json.dumps(msgs); base["n_orig"] = len(cur)
        base["n_turns"] = sum(m["role"] == "assistant" for m in msgs); out.append(base)
    for r in rows:
        if cur and used + r["ntok"] > BUDGET:
            flush(); cur, used = [], 0
        cur.append(r); used += r["ntok"]
    flush()
    return out


if __name__ == "__main__":
    df = pd.read_parquet(f"{R}/manifests/stage1/{SRC}.parquet")
    df = df[df.prep_exists.astype(str).isin(["True", "true", "1"])].reset_index(drop=True)
    with ProcessPoolExecutor(48) as ex:
        df["ntok"] = list(ex.map(ntok, df.conversations.tolist(), chunksize=2000))
    cap = df[df.kind == "cap"].copy(); cap["n_orig"] = 1
    qa = df[df.kind != "cap"]
    groups = [(k, g.drop(columns=[]).to_dict("records")) for k, g in qa.groupby(["video", "file"])]
    rows = []
    with ProcessPoolExecutor(48) as ex:
        for o in ex.map(pack, groups, chunksize=256):
            rows += o
    out = pd.concat([pd.DataFrame(rows), cap], ignore_index=True).drop(columns=["ntok"])
    out.to_parquet(f"{R}/manifests/stage1/{DST}.parquet")
    q = pd.DataFrame(rows)
    print(f"input rows {len(df)} (qa {len(qa)}, cap {len(cap)}) -> packed qa rows {len(q)} + cap {len(cap)} = {len(out)}")
    print("originals per packed qa row: mean %.2f, median %d, max %d" % (q.n_orig.mean(), q.n_orig.median(), q.n_orig.max()))
    print("text tokens per original row: median %d, p99 %d, max %d" % (df.ntok.median(), df.ntok.quantile(0.99), df.ntok.max()))
    print("seg_norm for llava alone (originals per row incl. captions): %.4f" % (len(df) / len(out)))
