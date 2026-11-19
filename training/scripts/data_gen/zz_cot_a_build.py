"""(10-29, rev 10-30 00:10: only sources whose ground truth DESCRIBES what is seen) (CoT stage data A: annotation-grounded rationales) collect multiple-choice items from OUR training manifests whose ground truth
gives verifiable facts about the motion / order / count, with those facts in plain words. A text LLM later writes the rationale from them
(zz_cot_a_gen.py). First turn of each row only (later turns re-ask with reordered options). Videos already in training -> no eval overlap.
  facts per source:
    tq4 order_ssv2 / order_llava / order_nat (two clips concatenated): the action of the first and of the second clip (SSv2 labels)
    tq4 odir/cdir/dir2/state/rspeed/dspeed ssv2, ssv2_obj/count/color/real: the clip's SSv2 action label + the gold option
    holo/epic/chego seq|order, chego_rev/epic_rev, fmt_before/after_*: the gold option (the true order / step)
    countix / ssv2_count: the action + the gold count;  sav_tracks_qa (mc): the gold direction of the tracked object
    clevrer (mc): the gold option + the CLEVRER question family
-> runs/cot/A_items.jsonl  {src, video, prep, t0, t1, question, options, gold, facts}"""
import json, re, random, collections, sys
import pandas as pd, yaml
R = "/data/llama_vjepa_lm"; cfg = yaml.safe_load(open(f"{R}/training/configs/single/nx16_Q8.yaml"))
DS = {d["name"]: d for d in cfg["data"]["sources"]} if "sources" in cfg.get("data", {}) else {d["name"]: d for d in cfg["data"]}
lab = {}
ss = pd.read_parquet(f"{R}/manifests/stage1/clean1027/ssv2.parquet", columns=["video", "kind", "conversations"])
for v, k, c in zip(ss.video, ss.kind, ss.conversations):
    if k == "open" and v not in lab:
        lab[v] = json.loads(c)[1]["content"].rstrip(".")
print("ssv2 labels", len(lab), flush=True)
MC = re.compile(r"^(.*?)\n((?:[A-J]\. .*\n?)+)Answer with the option's letter", re.S)
def first_mc(conv):
    c = json.loads(conv) if isinstance(conv, str) else conv
    u = c[0]["content"].replace("<|jepa_evidence|>\n", ""); a = c[1]["content"].strip()
    m = MC.match(u)
    if not m or not re.match(r"^[A-J]\.?$", a): return None
    opts = [re.sub(r"^[A-J]\. ", "", l) for l in m.group(2).strip().split("\n")]
    g = "ABCDEFGHIJ".index(a[0])
    return (m.group(1).strip(), opts, g) if g < len(opts) else None
OUT = []; stat = collections.Counter(); rng = random.Random(0)
def add(src, row, facts, q, opts, g):
    OUT.append({"src": src, "video": row["video"], "prep_root": DS[src].get("prep_root"), "t0": row.get("t0"), "t1": row.get("t1"),
                "question": q, "options": opts, "gold": g, "facts": facts}); stat[src] += 1
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 6000      # items per source (sampled)
for src, d in DS.items():
    fam = None
    if re.match(r"tq4_order_", src): fam = "order"
    elif re.match(r"tq4_(odir|cdir|dir2|state)_ssv2|ssv2_(obj|count|color|real)$", src): fam = "ssv2"   # (00:10) rspeed/dspeed dropped: the label says nothing about speed
    elif re.match(r"(holo|epic|chego)_(seq|order|rev)$|fmt_(before|after)_|next_(action|turn)_", src): fam = "gold"
    elif src in ("countix", "sav_tracks_qa"): fam = "gold"   # (10-30 00:10) clevrer / star_train / plausibility dropped: the gold option alone is no
                                                             # description of the video and the teacher invented the visual details
    if fam is None: continue
    m = pd.read_parquet(d["manifest"]); rows = m.to_dict("records"); rng.shuffle(rows); n0 = len(OUT)
    for r in rows:
        if len(OUT) - n0 >= CAP: break
        x = first_mc(r["conversations"])
        if not x: continue
        q, opts, g = x; facts = []
        if fam == "order":
            tq = json.loads(r["tq"]) if isinstance(r.get("tq"), str) else r.get("tq")
            tq = tq if isinstance(tq, dict) else {}
            a, b = lab.get(r["video"]), lab.get(tq.get("v2"))
            if not (a and b): continue
            facts = [f"First: {a}.", f"Then: {b}."]
        elif fam == "ssv2":
            a = lab.get(r["video"])
            if not a: continue
            facts = [f"The action in the video: {a}.", f"Correct option: {opts[g]}."]
        else:
            facts = [f"Correct option: {opts[g]}."]
            if src == "clevrer" and r.get("subtype"): facts.append(f"Question type: {r['subtype']}.")
        add(src, r, facts, q, opts, g)
pd.DataFrame(OUT).to_json(f"{R}/runs/cot/A_items.jsonl", orient="records", lines=True)
print("items", len(OUT)); [print(" ", k, v) for k, v in stat.most_common()]
