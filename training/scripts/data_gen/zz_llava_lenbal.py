"""(10-24, new-benchmark failure review: letter choice follows option LENGTH) LLaVA-Video MC: in the GPT-written files the gold is the longest
option 67 % of the time (shortest 8 %), so the model learned "longest = correct" (MMWorld / PhysBench / RoboFail failures with a right free answer).
(v2) Per TURN (an MC row holds ~4 question turns): per (source file, option count), gold-longest turns are kept with probability min(1, t / c_0),
c_0 = their count, t = median turn count over the length ranks; other turns stay. A dropped first turn hands the video marker to the next
question; a row with no turn left is dropped. Nothing is written, turns are only removed. -> manifests/stage1/llava_clean_lb.parquet"""
import pandas as pd, json, re, collections, numpy as np
R = "/data/llama_vjepa_lm"; OPT = re.compile(r"^\(?([A-H])[\.\)]\s*(.+)$", re.M); VIS = "<|jepa_evidence|>"
df = pd.read_parquet(f"{R}/manifests/stage1/llava_clean.parquet")
def ranks(conv):
    m = json.loads(conv); out = []
    for u, a in zip(m[0::2], m[1::2]):
        opts = OPT.findall(u["content"]); lm = re.match(r"^\(?([A-H])\b", a["content"].strip())
        if u["role"] != "user" or len(opts) < 2 or not lm or lm.group(1) not in dict(opts): out.append(None); continue
        L = sorted(((len(t), k) for k, t in opts), reverse=True); out.append((len(opts), [k for _, k in L].index(lm.group(1))))
    return out
CONV = df.conversations.tolist(); FILE = df.file.tolist(); mc = [i for i, k in enumerate(df.kind.tolist()) if k == "mc"]; R_ = {i: ranks(CONV[i]) for i in mc}   # (v2b) lists: per-cell pandas writes copied the column
cnt = collections.defaultdict(collections.Counter)
for i in mc:
    for r in R_[i]:
        if r: cnt[(FILE[i], r[0])][r[1]] += 1
P = {}
for (f, n), c in cnt.items():
    t = float(np.median([c[k] for k in range(n)])); P[(f, n)] = min(1.0, t / c[0]) if c[0] and sum(c.values()) >= 500 else 1.0
rng = np.random.default_rng(0); new = {}; before = collections.Counter(); after = collections.Counter()
for i in mc:
    m = json.loads(CONV[i]); f = FILE[i]; keep = []
    for j, r in enumerate(R_[i]):
        pair = m[2 * j: 2 * j + 2]
        if r: before[(f, "n")] += 1; before[(f, "L")] += r[1] == 0
        if r and r[1] == 0 and rng.random() >= P[(f, r[0])]: continue
        if r: after[(f, "n")] += 1; after[(f, "L")] += r[1] == 0
        keep.append([dict(x) for x in pair])
    if not keep: new[i] = None; continue
    had = VIS in m[0]["content"]
    for k, (u, a) in enumerate(keep):
        u["content"] = u["content"].replace(VIS + "\n", "").replace(VIS, "")
        if k == 0 and had: u["content"] = VIS + "\n" + u["content"]
    new[i] = json.dumps([x for p in keep for x in p])
drop = [i for i, v in new.items() if v is None]
for i, v in new.items():
    if v is not None: CONV[i] = v
df["conversations"] = CONV; out = df.drop(index=drop).reset_index(drop=True)
for f in sorted({f for f, _ in before}):
    b, a = before[(f, "n")], after[(f, "n")]; print(f"{f[:58]:58s} turns {b:6d} -> {a:6d}  gold-longest {before[(f,'L')]/b:.2f} -> {after[(f,'L')]/max(a,1):.2f}")
print("rows", len(df), "->", len(out), "| mc rows", len(mc), "->", int((out.kind == "mc").sum()), "| mc turns", sum(v for (f, k), v in before.items() if k == "n"), "->", sum(v for (f, k), v in after.items() if k == "n"))
out.to_parquet(f"{R}/manifests/stage1/llava_clean_lb.parquet")
