"""(10-24, user: keep it video-only, but put recognition into the questions we build) Recognition questions on videos, answers from human labels:
  ssv2_obj    : SSv2 train (crowd-written object names in the label placeholders): which object is handled. Distractors: frequent object names whose
                head word and words differ from every object named for that video
  chego_scene : Charades-Ego (first- and third-person): where the video was recorded (16 human-labelled room types)
  chego_obj   : Charades-Ego object list of the script: which object appears (3 distractors from objects NOT listed for that video) or, for 40%,
                which object does NOT appear (3 listed objects + 1 not listed); whole video up to 30 s
-> manifests/stage1/opendl/{ssv2_obj,chego_scene,chego_obj}.parquet"""
import sys, io, csv, json, zipfile, os, re, collections
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
import pandas as pd
R = "/data/llama_vjepa_lm"; OUT = f"{R}/manifests/stage1/opendl"; os.makedirs(OUT, exist_ok=True)
g = Gen(53)
ART = re.compile(r"^(a|an|the|some|one|two|three|a pair of|a piece of|a bunch of|a few|few|my|his|her|their)\s+")
norm = lambda s: ART.sub("", s.strip().lower()).strip(" .,")
# ---- SSv2 objects ----
L = json.load(open(f"{R}/data/ssv2_labels/labels/train.json"))
cnt = collections.Counter(norm(p) for x in L for p in x["placeholders"])
vocab = [w for w, c in cnt.most_common(3000) if c >= 15 and 1 <= len(w.split()) <= 3]
Q_OBJ = ["Which object does the person handle in the video?", "What object is being manipulated?", "Which object is used in this video?",
         "What is the object in the person's hand?"]
rows = []
for x in L:
    if not x["placeholders"]: continue
    names = [norm(p) for p in x["placeholders"]]; t = g.pick(names)
    if not t or len(t.split()) > 4: continue
    words = {w for n in names for w in n.split()}
    dis = [w for w in g.rng.permutation(vocab)[:60] if not (set(w.split()) & words)][:3]
    if len(dis) < 3: continue
    rows.append(g.row(f"{x['id']}.webm", 0, 0, g.pick(Q_OBJ), [t] + dis, 0))
for r in rows: r["t0"] = r["t1"] = None          # whole clip
report("ssv2_obj", rows); pd.DataFrame(rows).to_parquet(f"{OUT}/ssv2_obj.parquet")
# ---- Charades-Ego scene / objects ----
D = f"{R}/data/opendl/charades_ego"; VID = f"{D}/CharadesEgo_v1_480"
z = zipfile.ZipFile(f"{D}/CharadesEgo.zip")
C = list(csv.DictReader(io.TextIOWrapper(z.open("CharadesEgo/CharadesEgo_v1_train.csv"), encoding="utf-8")))
scene = lambda s: re.sub(r"\s*\(.*\)", "", s).split(" / ")[0].strip()
scenes = sorted({scene(r["scene"]) for r in C} - {"Other"})
CONF = [{"Entryway", "Hallway", "Stairs"}, {"Living room", "Recreation room", "Dining room"}, {"Closet", "Bedroom"}, {"Kitchen", "Pantry", "Dining room"},
        {"Basement", "Garage", "Laundry room"}, {"Home Office", "Bedroom"}]
ocnt = collections.Counter(o for r in C for o in r["objects"].split(";") if o)
GENERIC = {"floor", "doorway", "wall", "room", "house", "home", "doorknob", "something", "someone", "somewhere", "stuff", "thing", "things", "object"}
ovocab = [o for o, c in ocnt.most_common(200) if c >= 20 and o not in GENERIC]
nice = lambda o: o.replace("/", " or ")
Q_SC = ["Where was this video recorded?", "In which room does this video take place?", "What kind of room is shown?"]
Q_IN = ["Which of these objects can be seen in the video?", "Which object appears in the video?"]
Q_OUT = ["Which of these objects does NOT appear in the video?", "Which object is not in the video?"]
sc_rows, ob_rows = [], []
for r in C:
    v = f"{r['id']}.mp4"; ex = os.path.exists(f"{VID}/{v}"); t1 = min(float(r["length"] or 30), 30.0)
    s = scene(r["scene"])
    if s in scenes:
        near = set().union({s}, *[grp for grp in CONF if s in grp])   # (v2) look-alike rooms (all groups containing s) are never distractors
        sc_rows.append(g.row(v, 0, t1, g.pick(Q_SC) + (" (first-person video)" if r["egocentric"] == "Yes" else ""), [s] + list(g.rng.permutation([x for x in scenes if x not in near])[:3]), 0, "", ex))
    objs = [o for o in r["objects"].split(";") if o and o not in GENERIC]
    toks = {t for o in r["objects"].split(";") for t in o.split("/")}
    absent = [o for o in ovocab if o not in objs and not (set(o.split("/")) & toks)]
    if objs and len(absent) >= 3:
        if len(objs) >= 3 and g.rng.random() < 0.4:
            ob_rows.append(g.row(v, 0, t1, g.pick(Q_OUT), [nice(g.pick(absent))] + [nice(o) for o in g.rng.permutation(objs)[:3]], 0, "", ex))
        else:
            ob_rows.append(g.row(v, 0, t1, g.pick(Q_IN), [nice(g.pick(objs))] + [nice(o) for o in g.rng.permutation(absent)[:3]], 0, "", ex))
report("chego_scene", sc_rows); pd.DataFrame(sc_rows).to_parquet(f"{OUT}/chego_scene.parquet")
report("chego_obj", ob_rows); pd.DataFrame(ob_rows).to_parquet(f"{OUT}/chego_obj.parquet")
