"""(10-25 night, nx5 audit) the frozen Countix / SSv2-count manifests nx5_Q reads, with ONLY the options redrawn by count_options (gold at a
uniform rank): same rows in the same order (video, window, aug, question, gold count), so a resume draws the same rows. Every turn of a row
(first and the reordered second turn) gets the new option set, each in its own random order.
-> manifests/stage1/countfix_1025/{countix,ssv2_count}.parquet   usage: zz_fix_count_opts.py"""
import sys, json, re, collections
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import count_options
import numpy as np, pandas as pd
R = "/data/llama_vjepa_lm"; OUT = f"{R}/manifests/stage1/countfix_1025"; import os; os.makedirs(OUT, exist_ok=True)
SRC = {"countix": f"{R}/manifests/stage1/opendl_frozen_1024_1923/countix.parquet", "ssv2_count": f"{R}/manifests/stage1/opendl2_frozen_1024_1923/ssv2_count.parquet"}
OPT = re.compile(r"((?:^[A-H]\. .+\n?)+)", re.M); L = "ABCDEFGH"; rng = np.random.default_rng(1025)
for name, p in SRC.items():
    df = pd.read_parquet(p); new = []; rk = collections.Counter(); gold_same = 0
    for conv in df.conversations:
        m = json.loads(conv); u0, a0 = m[0]["content"], m[1]["content"].strip()
        o = re.findall(r"^([A-H])\. (.+)$", u0, re.M); gold = int(dict(o)[a0[0]])
        opts = count_options(gold, rng)
        for j in range(0, len(m), 2):
            perm = list(rng.permutation(len(opts))); shown = [opts[i] for i in perm]
            block = "".join(f"{L[i]}. {t}\n" for i, t in enumerate(shown))
            u = m[j]["content"]; mm = OPT.search(u); tail = "" if u[mm.end() - 1] == "\n" else ""
            m[j]["content"] = u[:mm.start()] + block + u[mm.end():] if u[mm.end() - 1] == "\n" else u[:mm.start()] + block.rstrip("\n") + u[mm.end():]
            m[j + 1]["content"] = f"{L[shown.index(str(gold))]}."
        chk = re.findall(r"^([A-H])\. (.+)$", m[0]["content"], re.M); gold_same += int(dict(chk)[m[1]["content"][0]]) == gold
        v = sorted(int(t) for _, t in chk); rk[v.index(gold)] += 1
        new.append(json.dumps(m))
    out = df.copy(); out["conversations"] = new; out.to_parquet(f"{OUT}/{name}.parquet")
    same_rows = all((df[c].astype(str) == out[c].astype(str)).all() for c in df.columns if c != "conversations")
    print(f"{name}: rows {len(df)} -> {len(out)} | other columns identical {same_rows} | gold kept {gold_same}/{len(df)} | gold rank among options {dict(sorted(rk.items()))}")
    print("   before:", json.loads(df.conversations.iloc[0])[0]["content"].replace("\n", " | ")[:150], "->", json.loads(df.conversations.iloc[0])[1]["content"])
    print("   after: ", json.loads(new[0])[0]["content"].replace("\n", " | ")[:150], "->", json.loads(new[0])[1]["content"])
    if len(json.loads(new[0])) > 2: print("   after2:", json.loads(new[0])[2]["content"].replace("\n", " | ")[:150], "->", json.loads(new[0])[3]["content"])
