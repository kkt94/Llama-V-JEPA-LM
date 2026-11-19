"""(10-26, RoboVQA / RoboFail - Cosmos-Reason splits - were never won: every item is one of two yes / no templates (task success, task
affordance) that nothing in our mix uses) the same templates from HUMAN labels: HoloAssist coarse / fine actions with Action Correctness,
EgoDex single-task episodes (task folder names, no llm_* text). Exact wording of the benchmark templates; yes / no exactly balanced per kind.
 robo_success   "The agent in the video was given the instruction - X. Was the agent successful in completing the task ...?"
                X = a coarse action (HoloAssist) or the episode task (EgoDex); yes: the LAST 30-60 % of the span (ends done); no: the FIRST
                30-60 % (not done yet) - equal clip lengths, so duration cannot answer
 robo_done      RoboFail affordance wording "... It has made some progress towards the instruction. Has the agent completed the task ...?"
                same yes / no rule as robo_success
 robo_subtask   RoboFail success wording "... attempted to complete the following subtask - "Y". Was it successful in completing this subtask?"
                Y = a fine action, yes = Correct Action, no = Wrong Action (human label), paired by the same action phrase
 robo_afford    "... given the instruction - Z. Is it possible for the agent to execute the task specified in the instruction?"  clip = 6 s
                before a fine action; yes: Z = that next action; no: Z = an action of another task type whose object never appears in this video
-> manifests/stage1/opendl3/robo_*__{holo,egodex}.parquet   usage: zz_qgen_robo.py"""
import sys, os, json, glob, collections
import numpy as np, pandas as pd
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
R = "/data/llama_vjepa_lm"; O3 = f"{R}/manifests/stage1/opendl3"; g = Gen(103)
HP = f"{R}/prep/holoassist_384_15fps"; EP = f"{R}/prep/egodex_384_15fps"
out = collections.defaultdict(list)
def add(kind, path, t0, t1, conv, key): out[kind].append(dict(video=path, prep_exists=True, root2=None, tq=None, aug="", t0=float(t0), t1=float(t1), conversations=conv, _key=key))
def yn(kind, path, t0, t1, q, yes):
    k = "yes" if yes else "no"
    if g.rng.random() < 0.5: add(kind, path, t0, t1, g.row(path, t0, t1, q, ["yes", "no"] if yes else ["no", "yes"], 0)["conversations"], k)
    else: add(kind, path, t0, t1, json.dumps([{"role": "user", "content": f"<|jepa_evidence|>\n{q}\nAnswer yes or no."}, {"role": "assistant", "content": "Yes." if yes else "No."}]), k)
AG = lambda: g.pick(["agent", "agent", "robot"])
SUC = "The {a} in the video was given the instruction - {x}. Was the {a} successful in completing the task specified by the instruction?"
DONE = "The {a} in the video was given the instruction - {x}. It has made some progress towards the instruction. Has the {a} completed the task specified in the instruction?"
SUB = 'The {a} in the video was given the instruction - {x}. To make progress towards the instruction, the {a} attempted to complete the following subtask - "{y}". Was it successful in completing this subtask?'
AFF = "The {a} in the video was given the instruction - {z}. Is it possible for the {a} to execute the task specified in the instruction?"
cl = lambda s: str(s).replace("_", " ").replace("-", " ").strip().lower()
# ---- HoloAssist
D = json.load(open(f"{R}/data/opendl/holoassist/data-annotation-trainval-v1_1.json")); fine_all = collections.defaultdict(set)   # task type -> fine phrases
vids = []
for v in D:
    path = f"{v['video_name']}.mp4"
    if not os.path.exists(f"{HP}/{path}"): continue
    tk = cl(v.get("taskType") or ""); C, F = [], []
    for e in v["events"]:
        a = e.get("attributes") or {}
        if "end" not in e or cl(a.get("Verb", "none")) == "none" or cl(a.get("Noun", "none")) == "none": continue
        adj = cl(a.get("Adjective", "none")); ph = f"{cl(a['Verb'])} the {'' if adj == 'none' else adj + ' '}{cl(a['Noun'])}"
        if e["label"] == "Coarse grained action": C.append((float(e["start"]), float(e["end"]), ph))
        elif e["label"] == "Fine grained action": F.append((float(e["start"]), float(e["end"]), ph, cl(a["Noun"]), str(a.get("Action Correctness", ""))))
    vids.append((path, tk, sorted(C), sorted(F))); [fine_all[tk].add((f[2], f[3])) for f in F]
FREQ = collections.Counter(f[2] for _, _, _, F in vids for f in F)             # how often each fine-action phrase occurs
wrong, right = [], collections.defaultdict(list)
for path, tk, C, F in vids:
    for s, e, ph in C:
        if not (4.0 <= e - s <= 120.0): continue
        kind = "robo_success" if g.rng.random() < 0.6 else "robo_done"; tmpl = SUC if kind == "robo_success" else DONE; a = AG()
        f = g.rng.uniform(0.3, 0.6)                   # same clip length either way: the LAST f of the span (done) or the FIRST f (not yet)
        if g.rng.random() < 0.5: yn(kind + "__holo", path, e - (e - s) * f, e, tmpl.format(a=a, x=ph), True)
        else: yn(kind + "__holo", path, s, s + (e - s) * f, tmpl.format(a=a, x=ph), False)
    nouns = {f[3] for f in F}
    for i, f in enumerate(F):
        if 0.6 <= f[1] - f[0] <= 10.0:
            if f[4].startswith("Wrong Action"): wrong.append((path, tk, f))
            elif f[4] == "Correct Action": right[f[2]].append((path, tk, f))
        if i > 0 and f[0] >= 6.0 and g.rng.random() < 0.25:                    # robo_afford
            a = AG()
            if g.rng.random() < 0.5: yn("robo_afford__holo", path, f[0] - 6.0, f[0], AFF.format(a=a, z=f[2]), True)
            else:
                cand = [ph for t2, S in fine_all.items() if t2 != tk for ph, nn in S if nn not in nouns]
                if cand:                                  # (v2) drawn by how often each phrase occurs as an action, like the positives
                    w_ = np.array([FREQ[ph] for ph in cand], float); yn("robo_afford__holo", path, f[0] - 6.0, f[0], AFF.format(a=a, z=cand[int(g.rng.choice(len(cand), p=w_ / w_.sum()))]), False)
used = set()
for j in g.rng.permutation(len(wrong)):                                       # robo_subtask, paired by phrase
    path, tk, f = wrong[j]; c = [x for x in right[f[2]] if id(x) not in used]
    if not c: continue
    pr = min(c, key=lambda x: abs((x[2][1] - x[2][0]) - (f[1] - f[0]))); used.add(id(pr))
    for (p_, tk_, f_), yes in (((path, tk, f), False), (pr, True)):
        yn("robo_subtask__holo", p_, f_[0], f_[1], SUB.format(a=AG(), x=tk_, y=f_[2]), yes)
# ---- EgoDex: whole episode vs first 30-60 %, single-task folders only (paired names like add_remove are ambiguous as instructions)
PAIRV = {"add", "remove", "assemble", "disassemble", "open", "close", "insert", "stack", "unstack", "fold", "unfold", "charge", "uncharge", "clip",
         "unclip", "crumple", "flatten", "deal", "gather", "screw", "unscrew", "sleeve", "unsleeve", "thread", "unthread", "tie", "untie", "zip", "unzip",
         "wrap", "unwrap", "lock", "unlock", "throw", "catch", "collect", "braid", "unbraid", "build", "load", "dispense", "scoop", "dump", "peel", "place",
         "pick", "set", "put", "take", "stock", "unstock", "fry", "boil", "serve", "rake", "smooth", "push", "pop", "play", "reset", "paint", "clean", "wash"}
for split in sorted(os.path.basename(x) for x in glob.glob(f"{EP}/*") if os.path.isdir(x)):     # (10-26) every transcoded EgoDex split
    for mp4 in sorted(glob.glob(f"{EP}/{split}/*/*.mp4")):
        task = mp4.split("/")[-2]; w = task.split("_")
        if sum(x in PAIRV for x in w) >= 2 or task.startswith("basic"): continue
        if g.rng.random() > 0.35: continue
        rel = os.path.relpath(mp4, EP); n = int(os.path.basename(mp4)[:-4]); h5 = f"{R}/data/opendl/egodex/{split}/{task}/{n}.hdf5"
        try:
            import h5py; T = h5py.File(h5, "r")["transforms/camera"].shape[0] / 30.0
        except Exception: continue
        if T < 4.0: continue
        x = task.replace("_", " "); kind = "robo_success" if g.rng.random() < 0.6 else "robo_done"; tmpl = SUC if kind == "robo_success" else DONE
        f = g.rng.uniform(0.3, 0.6)
        if g.rng.random() < 0.5: yn(kind + "__egodex", rel, T * (1 - f), T, tmpl.format(a=AG(), x=x), True)
        else: yn(kind + "__egodex", rel, 0, T * f, tmpl.format(a=AG(), x=x), False)
for k, rows in sorted(out.items()):
    d = pd.DataFrame(rows); m = d._key.value_counts().min(); d = pd.concat([gg.sample(m, random_state=0) for _, gg in d.groupby("_key")]).sort_index()
    d = d.drop(columns="_key").reset_index(drop=True); report(k, d.to_dict("records")); d.to_parquet(f"{O3}/{k}.parquet")
