"""(10-30 21:40, user approved) brevity instruction for terse answers in the base mix: every user turn that is an open question (no options,
no instruction about the answer's length) and whose answer has <= 3 words gets "\\nAnswer with a single word or phrase." appended (LLaVA's
convention), so the model gives one-word answers only when asked; together with the full-sentence answer data, un-instructed open questions
get full answers. Answers are unchanged. Writes changed copies of the affected manifests to manifests/stage1/mx2b_brief/<name>.parquet
(all columns kept) and runs/cot/brief_map.json {original manifest: copy, rows changed, turns changed}.
usage: python -I zz_brief_instr.py <config with the base sources>"""
import sys, os, json, re
import numpy as np, pandas as pd, yaml
R = "/data/llama_vjepa_lm"; O = f"{R}/manifests/stage1/mx2b_brief"; os.makedirs(O, exist_ok=True)
SUF = "\nAnswer with a single word or phrase."
BRIEF = re.compile(r"single word|few words|one word|short phrase|word or phrase|briefly|concise|short answer|in one sentence|yes or no|answer (?:with )?(?:a )?(?:number|left|right|up|down|yes|no)\b", re.I)
OPT = re.compile(r"(^|\n)\s*\(?[A-J][\.\):]\s|Options:|option's letter|letter of the|from the given choices", re.I)
c = yaml.safe_load(open(sys.argv[1])); out = {}
for s in c["data"]["sources"]:
    m = s["manifest"]
    if m in out or not os.path.exists(m): continue
    try: d = pd.read_parquet(m)
    except Exception: continue
    if "conversations" not in d.columns: continue
    as_str = isinstance(d.conversations.iloc[0], str); nrow = nturn = 0; new = []
    for x in d.conversations:
        msgs = json.loads(x) if as_str else [dict(t) for t in x]; ch = False
        for j in range(0, len(msgs) - 1):
            u, a = msgs[j], msgs[j + 1]
            if u.get("role") != "user" or a.get("role") != "assistant": continue
            q = str(u["content"]); ans = str(a["content"]).strip()
            if not ans or len(ans.split()) > 3 or OPT.search(q) or BRIEF.search(q): continue
            u["content"] = q.rstrip() + SUF; ch = True; nturn += 1
        nrow += ch; new.append(json.dumps(msgs, ensure_ascii=False) if as_str else msgs)
    if nturn == 0: continue
    d = d.copy(); d["conversations"] = new
    p = f"{O}/{os.path.basename(os.path.dirname(m))}__{os.path.basename(m)}"; d.to_parquet(p)
    out[m] = {"copy": p, "rows": len(d), "rows_changed": nrow, "turns_changed": nturn}
    print(f"{os.path.basename(m):40s} rows {len(d):7d} changed rows {nrow:7d} ({100 * nrow / len(d):.1f}%) turns {nturn}", flush=True)
json.dump(out, open(f"{R}/runs/cot/brief_map.json", "w"), indent=1)
print("manifests changed", len(out))
