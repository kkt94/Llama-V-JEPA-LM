"""(10-30 18:25, user: round-4 SFT = round-3 recipe + full-sentence answer data) training manifests from runs/cot/fullans_vl_*of*.jsonl
(video teacher, kept rows only, one row per item): normal-mode turn, user = "<|jepa_evidence|>\\n{question}" (no "few words" instruction, no
options), assistant = the full answer. One parquet per prep_root in runs/cot/train4/, plus repo/configs/single/cot2b4_Q.yaml (+ _eval):
cot2b3_Q with every source weight x (1 - F_W) and the F sources at F_W total (by rows); 24 GPUs x 12 = 288 / step, 1000 steps (the same
number of samples as round 3's 8 GPUs x 3000), checkpoint every 250, out runs/cot2b4_Q.
usage: python -I zz_fullans_build.py [F_W=0.12]"""
import json, glob, os, re, sys, collections, copy
import pandas as pd, yaml
R = "/data/llama_vjepa_lm"; O = f"{R}/runs/cot/train4"; os.makedirs(O, exist_ok=True); FW = float(sys.argv[1]) if len(sys.argv) > 1 else 0.12
rows = collections.defaultdict(list); seen = set(); n_all = 0
NW = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
def num_ok(r):
    """(18:40) count answers were rejected because the teacher spells numbers out ("four times" for gold 4): accept when the first sentence
    has the gold number as digits or word and no other number"""
    g = str(r.get("gold_option", "")).strip().rstrip(".")
    if not re.fullmatch(r"[0-9]+", g) or not r.get("answer") or not (10 <= len(r["answer"].split()) <= 90): return False
    first = r["answer"].split(".")[0].lower(); ws = re.findall(r"[a-z]+|\d+", first)
    nums = {w if w.isdigit() else str(NW.index(w)) for w in ws if w.isdigit() or w in NW}
    return nums == {str(int(g))}
for f in sorted(glob.glob(f"{R}/runs/cot/fullans_vl_*of*.jsonl")):
    for l in open(f):
        r = json.loads(l); n_all += 1
        if not r.get("ok") and not num_ok(r): continue
        if r["i"] in seen: continue
        seen.add(r["i"])
        conv = [{"role": "user", "content": "<|jepa_evidence|>\n" + r["question"].strip()}, {"role": "assistant", "content": r["answer"].strip()}]
        rows[r["prep_root"] or "/"].append({"video": r["video"], "prep_exists": True, "t0": r.get("t0"), "t1": r.get("t1"), "conversations": json.dumps(conv)})
man = []
for root, rs in sorted(rows.items()):
    name = "fullans_" + (re.sub(r"[^a-z0-9]+", "_", os.path.basename(str(root).rstrip("/")).lower()) or "abs")
    path = f"{O}/{name}.parquet"; pd.DataFrame(rs).to_parquet(path); man.append({"name": name, "manifest": path, "prep_root": root, "rows": len(rs)})
tot = sum(m["rows"] for m in man)
c = yaml.safe_load(open(f"{R}/training/configs/single/cot2b3_Q.yaml"))
for s in c["data"]["sources"]: s["weight"] = round(s["weight"] * (1 - FW), 6)
for m in man: c["data"]["sources"].append({"name": m["name"], "manifest": m["manifest"], "prep_root": m["prep_root"], "weight": round(FW * m["rows"] / tot, 6)})
c["optim"].update({"max_steps": 1000}); c["run"].update({"out_dir": f"{R}/runs/cot2b4_Q", "save_every": 250})
hdr = f"# (10-30) cot2b4_Q: round-3 recipe (cot2b3_Q) + full-sentence answers {tot} rows at {FW} (logs/zz_fullans_build.py), 24 GPUs x 1000 steps\n"
open(f"{R}/training/configs/single/cot2b4_Q.yaml", "w").write(hdr + yaml.safe_dump(c, sort_keys=False))
e = copy.deepcopy(c); e["model"]["chat_template"] = f"{R}/training/configs/templates/qwen3_nothink.jinja"
open(f"{R}/training/configs/single/cot2b4_Q_eval.yaml", "w").write(hdr.replace("cot2b4_Q:", "cot2b4_Q (EVAL, no-think template):", 1) + yaml.safe_dump(e, sort_keys=False))
json.dump({"generated": n_all, "kept_unique": tot, "manifests": man, "F_W": FW}, open(f"{O}/summary.json", "w"), indent=1)
print("generated", n_all, "kept", tot, {m["name"]: m["rows"] for m in man}, "weights sum", round(sum(s["weight"] for s in c["data"]["sources"]), 4))
