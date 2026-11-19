"""(10-24, after the new-benchmark analysis: static appearance / object state / success-vs-failure / counting are the gaps) Video questions whose answers
come from human labels or from how the clip is cut (no model writes anything):
  ssv2_real   SSv2 train "pretending / failing" templates and their real counterparts (same object names): did the person really do it or only
              pretend / fail. Options: the clip's own label, its counterpart filled with the same objects, one other real action with the same
              object. Real and pretend clips balanced per pair.                                                   (RoboFail / RoboVQA, object state)
  chego_done  Charades-Ego state-changing actions >= 2.5 s: the whole labelled span -> finished; the span cut at 30-45 % -> not finished
              (truncated by construction)                                                                          (task completion)
  chego_state Charades-Ego open / close and light on / off spans + 0.5 s: the state at the end; played backwards ("rev") the other state
  ssv2_count  SSv2 "[number of]" templates (crowd-written count 2..6): how many objects                                  (static counting)
-> manifests/stage1/opendl2/{ssv2_real,chego_done,chego_state,ssv2_count}.parquet"""
import sys, io, csv, json, zipfile, os, re, collections
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
import pandas as pd
R = "/data/llama_vjepa_lm"; OUT = f"{R}/manifests/stage1/opendl2"; os.makedirs(OUT, exist_ok=True); g = Gen(61)
# ---------------- SSv2 real vs pretend / fail ----------------
L = json.load(open(f"{R}/data/ssv2_labels/labels/train.json")); T = collections.defaultdict(list)
for x in L: T[x["template"]].append(x)
PAIRS = [("Pretending to pick [something] up", "Picking [something] up"), ("Pretending to open [something] without actually opening it", "Opening [something]"),
         ("Pretending to close [something] without actually closing it", "Closing [something]"), ("Pretending to put [something] on a surface", "Putting [something] on a surface"),
         ("Pretending to take [something] from [somewhere]", "Taking [something] from [somewhere]"), ("Pretending to put [something] next to [something]", "Putting [something] next to [something]"),
         ("Pretending to put [something] into [something]", "Putting [something] into [something]"), ("Pretending to take [something] out of [something]", "Taking [something] out of [something]"),
         ("Pretending to throw [something]", "Throwing [something]"), ("Pretending to turn [something] upside down", "Turning [something] upside down"),
         ("Pretending to squeeze [something]", "Squeezing [something]"), ("Pretending to put [something] behind [something]", "Putting [something] behind [something]"),
         ("Pretending to put [something] onto [something]", "Putting [something] onto [something]"), ("Pretending to put [something] underneath [something]", "Putting [something] underneath [something]"),
         ("Pretending to scoop [something] up with [something]", "Scooping [something] up with [something]"), ("Pretending to poke [something]", "Poking [something] so it slightly moves"),
         ("Pushing [something] so that it almost falls off but doesn't", "Pushing [something] off of [something]"),
         ("Tilting [something] with [something] on it slightly so it doesn't fall down", "Tilting [something] with [something] on it until it falls off"),
         ("Putting [something] onto a slanted surface but it doesn't glide down", "Putting [something] that can't roll onto a slanted surface, so it slides down"),
         ("Trying but failing to attach [something] to [something] because it doesn't stick", "Attaching [something] to [something]"),
         ("Failing to put [something] into [something] because [something] does not fit", "Putting [something] into [something]")]
SLOT = re.compile(r"\[[^\]]+\]")
def fill(tpl, objs):
    k = len(SLOT.findall(tpl))
    if k > len(objs): objs = objs + ["something"] * (k - len(objs))          # (v2) a missing slot is "something", never the same object twice
    it = iter(objs); return SLOT.sub(lambda m: next(it), tpl)
reals = sorted({r for _, r in PAIRS if r in T and len(SLOT.findall(r)) == 1})   # (v2) third option from one-slot actions only
BADOBJ = re.compile(r"^(up|down|off|out|in|on|into|onto)\b")
Q_REAL = ["What does the person actually do?", "Which description matches what really happens in the video?", "What happens in the video?"]
rows = []; used = collections.Counter()
for fake, real in PAIRS:
    if fake not in T or real not in T: continue
    n = min(len(T[fake]), len(T[real]))
    for tpl, other, xs in ((fake, real, g.rng.permutation(len(T[fake]))[:n]), (real, fake, g.rng.permutation(len(T[real]))[:n])):
        for j in xs:
            x = T[tpl][int(j)]; objs = [p.strip() for p in x["placeholders"]] or ["something"]
            if any(BADOBJ.match(o.lower()) for o in objs): continue
            third = fill(g.pick([r for r in reals if r not in (real,)]), objs)
            opts = [fill(tpl, objs), fill(other, objs), third]
            if len(set(o.lower() for o in opts)) < 3: continue
            rows.append(g.row(f"{x['id']}.webm", 0, 0, g.pick(Q_REAL), opts, 0)); used[tpl] += 1
for r in rows: r["t0"] = r["t1"] = None
report("ssv2_real", rows); print("   pairs used", len([p for p in PAIRS if p[0] in T and p[1] in T]), "of", len(PAIRS)); pd.DataFrame(rows).to_parquet(f"{OUT}/ssv2_real.parquet")
# ---------------- SSv2 counts ----------------
W = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
rows = []
for x in L:
    if "[number of]" not in x["template"]: continue
    p = x["placeholders"]; c0 = p[0].lower().strip(); n = int(c0) if c0.isdigit() else W.get(c0)
    if n is None or not 2 <= n <= 6 or len(p) < 2: continue
    obj = p[1].strip(); verb = x["template"].split(" [")[0].lower()
    q = f"How many {obj} does the person {verb.replace('stacking', 'stack').replace('putting', 'put').replace('taking', 'take').replace('moving', 'move').replace('pushing', 'push')}?" if verb in ("stacking", "putting", "taking", "moving", "pushing") else f"How many {obj} are involved in the video?"
    wrong = [str(k) for k in range(1, 9) if k != n and abs(k - n) <= 3]
    rows.append(g.row(f"{x['id']}.webm", 0, 0, q, [str(n)] + list(g.rng.permutation(wrong)[:3]), 0))
for r in rows: r["t0"] = r["t1"] = None
report("ssv2_count", rows); pd.DataFrame(rows).to_parquet(f"{OUT}/ssv2_count.parquet")
# ---------------- Charades-Ego completion / end state ----------------
D = f"{R}/data/opendl/charades_ego"; VID = f"{D}/CharadesEgo_v1_480"
z = zipfile.ZipFile(f"{D}/CharadesEgo.zip"); rd = lambda n: io.TextIOWrapper(z.open("CharadesEgo/" + n), encoding="utf-8")
NAME = {l.split()[0]: l.strip()[5:] for l in rd("Charades_v1_classes.txt") if l.strip()}
CHANGE = ["c006", "c008", "c025", "c027", "c039", "c041", "c046", "c048", "c089", "c090", "c104", "c105", "c112", "c113", "c142", "c143", "c055", "c057",
          "c001", "c002", "c017", "c018", "c022", "c023", "c028", "c030", "c042", "c043", "c049", "c050", "c062", "c063", "c109", "c110", "c119", "c120", "c151", "c154"]
STATE = {"c006": ("door", "closed"), "c008": ("door", "open"), "c025": ("book", "closed"), "c027": ("book", "open"), "c039": ("box", "closed"), "c041": ("box", "open"),
         "c046": ("laptop", "closed"), "c048": ("laptop", "open"), "c089": ("window", "closed"), "c090": ("window", "open"), "c112": ("closet / cabinet", "closed"),
         "c113": ("closet / cabinet", "open"), "c142": ("refrigerator", "closed"), "c143": ("refrigerator", "open"), "c104": ("light", "on"), "c105": ("light", "off")}
OPP = {"open": "closed", "closed": "open", "on": "off", "off": "on"}
lo = lambda c: NAME[c][0].lower() + NAME[c][1:]
done, state = [], []
for r in csv.DictReader(rd("CharadesEgo_v1_train.csv")):
    v = f"{r['id']}.mp4"; ex = os.path.exists(f"{VID}/{v}"); who = "the camera wearer" if r["egocentric"] == "Yes" else "the person"
    for a in (r["actions"] or "").split(";"):
        p = a.split()
        if len(p) != 3: continue
        c, s, e = p[0], float(p[1]), float(p[2])
        if c in CHANGE and 2.5 <= e - s <= 15:
            full = g.rng.random() < 0.5; t1 = e if full else s + (e - s) * float(g.rng.uniform(0.30, 0.45))
            q = g.pick([f"Does {who} finish {lo(c)}?", f"Is the action '{lo(c)}' completed in this clip?", f"By the end of the clip, has {who} finished {lo(c)}?"])
            done.append(g.row(v, s, t1, q, ["Yes, it is completed", "No, the clip ends before it is completed"], 0 if full else 1, "", ex))
        if c in STATE and e - s <= 12:
            obj, st_ = STATE[c]; aug = g.pick(["", "rev"]); ans = st_ if aug == "" else OPP[st_]
            pair = ["on", "off"] if st_ in ("on", "off") else ["open", "closed"]
            q = g.pick([f"At the end of the clip, is the {obj} {pair[0]} or {pair[1]}?", f"What state is the {obj} in at the end?"])
            state.append(g.row(v, s, min(e + 0.5, float(r["length"] or e + 0.5)), q, [ans.capitalize(), OPP[ans].capitalize()], 0, aug, ex))
report("chego_done", done); pd.DataFrame(done).to_parquet(f"{OUT}/chego_done.parquet")
report("chego_state", state); pd.DataFrame(state).to_parquet(f"{OUT}/chego_state.parquet")
