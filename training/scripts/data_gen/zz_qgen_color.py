"""(10-24, after the colour probe: frozen V-JEPA features give 75 % on 8 SSv2 colours, chance 12.5 %, pixel HSV histogram 41.8 %) SSv2 train
clips whose crowd-written object name carries exactly one colour word: "What colour is the <object>?" with the labelled colour and 3 other
colours (never a near-synonym: silver/grey, gold/yellow, white/silver). Answer = the human label only.
-> manifests/stage1/opendl2/ssv2_color.parquet   usage: zz_qgen_color.py"""
import sys, json, re, collections
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
import pandas as pd
R = "/data/llama_vjepa_lm"; g = Gen(67)
COLS = ["white", "red", "black", "blue", "green", "orange", "yellow", "silver", "pink", "purple", "brown", "grey", "gold"]
NEAR = {"silver": {"grey", "white", "gold"}, "grey": {"silver", "black", "white"}, "gold": {"yellow", "orange", "silver", "brown"}, "yellow": {"gold", "orange"},
        "orange": {"gold", "yellow", "red"}, "white": {"silver", "grey"}, "pink": {"purple", "red"}, "purple": {"pink", "blue"}, "red": {"pink", "orange"},
        "brown": {"gold", "orange"}, "black": {"grey"}, "blue": {"purple"}, "green": set()}
CW = re.compile(r"\b(" + "|".join(COLS + ["gray"]) + r")\b")
DROP = re.compile(r"\b(colou?r(ed)?|light|dark|bright|a|an|the|one|some|my|this|that|his|her)\b")
L = json.load(open(f"{R}/data/ssv2_labels/labels/train.json"))
QS = ["What colour is the {o}?", "What color is the {o} in the video?", "Which colour is the {o} that the person handles?"]
rows = []; per = collections.Counter()
for x in L:
    hits = [(p, CW.findall(p.lower())) for p in x["placeholders"]]
    cs = {c.replace("gray", "grey") for _, h in hits for c in h}
    if len(cs) != 1: continue
    c = cs.pop(); p = next(p for p, h in hits if h).lower()
    if re.search(r"white ?out|red ?bull|white ?board|black ?board|green ?tea|red ?wine|white ?wine", p): continue   # names, not colours
    o = re.sub(r"\s+", " ", DROP.sub(" ", CW.sub(" ", p))).strip(" ,.-")
    if len(o) < 3 or not re.search(r"[a-z]{3}", o) or " and " in o or "&" in o: continue
    if per[c] >= 1200: continue
    wrong = [k for k in COLS if k != c and k not in NEAR[c]]
    opts = [c.capitalize()] + [w.capitalize() for w in g.rng.permutation(wrong)[:3]]
    rows.append(g.row(f"{x['id']}.webm", 0, 0, g.pick(QS).format(o=o), opts, 0)); per[c] += 1
for r in rows: r["t0"] = r["t1"] = None
print("per colour", dict(per)); report("ssv2_color", rows); pd.DataFrame(rows).to_parquet(f"{R}/manifests/stage1/opendl2/ssv2_color.parquet")
print(json.loads(rows[0]["conversations"]), "\n", json.loads(rows[7]["conversations"])[:2])
