"""(10-16) Event order INSIDE one natural video (TempCompass 'order / original': tq2_A fell 48 -> 37 there; our order questions so far joined
two different clips). PLM-Video region-time captions on SA-V (manifests/stage1/plm_rtloc.parquet): two described events of the same video
that do not overlap (second starts >= 0.5 s after the first ends). Variants: the video itself over [start of A, end of B] (natural order,
from the annotation), the two segments joined A then B, and joined B then A (order fixed by construction; the cut appears in both joined
variants, so a cut says nothing about the order). Descriptions cut to their first sentence (<= 25 words). 2 options (no 'together' here: it
would never be true). -> manifests/stage1/tq3/order_nat.parquet, held-out manifests/eval/pilot/ordn_heldout.jsonl (videos left out)"""
import json, re, numpy as np, pandas as pd, collections
R = "/data/llama_vjepa_lm"; SV = f"{R}/prep/sav_384_24fps"; rng = np.random.default_rng(5)
TAIL = "\nAnswer with the option's letter from the given choices directly."
d = pd.read_parquet(f"{R}/manifests/stage1/plm_rtloc.parquet"); ev = []
for v, c in zip(d.video, d.conversations):
    cv = json.loads(c); m = re.search(r'"(.*)"\?', cv[0]["content"], re.S); a = re.search(r"From ([\d.]+)s to ([\d.]+)s", cv[1]["content"])
    if m and a: ev.append((v, m.group(1).strip(), float(a.group(1)), float(a.group(2))))
E = pd.DataFrame(ev, columns=["video", "desc", "s", "e"])
import os; ex = {v: os.path.exists(f"{SV}/{v}") for v in E.video.unique()}; E = E[E.video.map(ex)]      # prep_exists is False for some
def short(t):
    t = re.split(r"(?<=[.!?])\s", t.strip())[0].rstrip(".").strip(); w = t.split()
    return t[0].upper() + t[1:] if 3 <= len(w) <= 25 else None
pairs = []
for v, g in E.groupby("video"):
    g = g.sort_values("s").values.tolist(); found = None
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            a, b = short(g[i][1]), short(g[j][1])
            if g[j][2] >= g[i][3] + 0.5 and a and b and a.lower() != b.lower() and g[i][3] - g[i][2] >= 0.8 and g[j][3] - g[j][2] >= 0.8:
                found = (v, a, g[i][2], g[i][3], b, g[j][2], g[j][3]); break
        if found: break
    if found: pairs.append(found)
P = pd.DataFrame(pairs, columns=["video", "a", "as_", "ae", "b", "bs", "be"]).sample(frac=1, random_state=0).reset_index(drop=True)
print("pairs", len(P))
L = lambda s: s[0].lower() + s[1:]
QS = [("What is the order of events in the video?", lambda f, s: f"{f} and then {L(s)}"), ("What order do the events occur in?", lambda f, s: f"First, {L(f)}; then, {L(s)}"),
      ("Which event happens first?", None), ("What is happening at the end of the video?", "last")]
def qa(a, b, a_first, k):
    q, f_ = QS[k]; f, s = (a, b) if a_first else (b, a)
    if callable(f_): opts, ans = [f_(a, b), f_(b, a)], f_(f, s)
    elif f_ is None: opts, ans = [a, b], f
    else: opts, ans = [a, b], s
    o = list(opts); rng.shuffle(o); return q, o, o.index(ans)
def row(r, variant, k):
    if variant == "nat": tq, t0, t1, af = None, r.as_, r.be, True
    else:
        af = variant == "ab"; x, y = ((r.as_, r.ae), (r.bs, r.be)) if af else ((r.bs, r.be), (r.as_, r.ae))
        tq = {"kind": "concat", "v2": r.video, "w2": list(y)}; t0, t1 = x
    q, o, k_ = qa(r.a, r.b, af, k)
    return tq, t0, t1, q, o, k_
H = P.iloc[:400]; T = P.iloc[400:]; rows = []
for i, r in enumerate(T.itertuples()):
    tq, t0, t1, q, o, k = row(r, ["nat", "ab", "ba"][i % 3], int(rng.integers(len(QS))))
    u = f"<|jepa_evidence|>\n{q}\n" + "\n".join(f"{'AB'[j]}. {x}" for j, x in enumerate(o)) + TAIL
    rows.append({"video": r.video, "prep_exists": True, "root2": SV if tq else None, "tq": json.dumps(tq) if tq else None, "aug": "", "t0": t0, "t1": t1,
                 "conversations": json.dumps([{"role": "user", "content": u}, {"role": "assistant", "content": "AB"[k] + "."}])})
pd.DataFrame(rows).to_parquet(f"{R}/manifests/stage1/tq3/order_nat.parquet"); print("train", len(rows))
with open(f"{R}/manifests/eval/pilot/ordn_heldout.jsonl", "w") as f:
    for i, r in enumerate(H.itertuples()):
        k = int(rng.integers(len(QS))); st = rng.bit_generator.state
        for n, var in enumerate(("nat", "ab", "ba")):
            rng.bit_generator.state = st; tq, t0, t1, q, o, ki = row(r, var, k)
            it = {"bench": "ordn_heldout", "id": f"ordn_{i}_{var}", "video": f"{SV}/{r.video}", "question": q, "options": o, "answer_index": ki, "answer_text": o[ki],
                  "group_id": f"ordn_{i}", "aug": "", "start_sec": float(t0), "end_sec": float(t1), "extra": {"kind": "order_nat" if var == "nat" else "order_seg"}}
            if tq: tq["v2"] = f"{SV}/{r.video}"; it["tq"] = tq
            f.write(json.dumps(it) + "\n")
print("held-out", 3 * len(H))
