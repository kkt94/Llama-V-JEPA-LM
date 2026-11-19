"""(10-25) HoloAssist (Microsoft, CDLA-Permissive-2.0; HoloLens 2 first-person, people doing 20 assembly / setup tasks while an instructor
helps). All answers from the HUMAN annotations (data-annotation-trainval-v1_1.json; label times = seconds of Video_pitchshift.mp4, checked
against the mp4 length). The Narration / Action sentence / Transcription free text is NOT used, only the structured fields.
 hfine  - fine-grained action (verb + adjective + noun) of a labelled span; distractors from the same video: same noun other verb, same verb
          other noun, any other
 hseq   - 3-4 consecutive non-overlapping fine actions (<= 20 s), options = permutations (as zz_qgen_seq.py; no reversal)
 hmist  - "is this step done correctly": fine actions labelled Wrong Action (corrected by the student or the instructor), each paired with a
          Correct Action span of the SAME action phrase and the nearest duration (any video, used once): without the pairing the phrase in the
          question alone gave 67.6 % on held-out videos
 htask  - which of the tasks (taskType) a 16 s window shows
The head pose (Pose_sync.txt) is NOT used: no reading of it matched the video's optical flow (logs/zz_holo_pose_check*.py, |corr| <= 0.44).
-> manifests/stage1/opendl3/holo_{fine,seq,mist,task}.parquet (prep root prep/holoassist_384_15fps, <video_name>.mp4)  usage: zz_qgen_holoassist.py"""
import sys, json, os, itertools, collections
sys.path.insert(0, "/data/llama_vjepa_lm/training/scripts/data_gen"); from zz_qgen_common import Gen, report
import pandas as pd
R = "/data/llama_vjepa_lm"; OUT = f"{R}/manifests/stage1/opendl3"; PREP = f"{R}/prep/holoassist_384_15fps"; g = Gen(89)
D = json.load(open(f"{R}/data/opendl/holoassist/data-annotation-trainval-v1_1.json"))
cl = lambda s: str(s).replace("_", " ").replace("-", " ").strip().lower()
def phrase(a):
    v, adj, n = cl(a.get("Verb", "")), cl(a.get("Adjective", "none")), cl(a.get("Noun", ""))
    return f"{v} the {'' if adj in ('none', '') else adj + ' '}{n}"
Q_F = ["What does the person do in this clip?", "Which action is shown?", "What am I doing in this video?"]
Q_S = ["What is the sequence of actions shown in the video?", "In what order are these actions performed?", "What do I do in this video, in order?"]
Q_T = ["Which task is the person working on?", "What is the person trying to do in this video?", "Which task am I doing?"]
TASKS = sorted({cl(v["taskType"]) for v in D if v.get("taskType")})
out = {k: [] for k in ("fine", "seq", "mist", "task")}; kinds = collections.Counter(); WR = []; RI = collections.defaultdict(list)
for v in D:
    name = v["video_name"]; path = f"{name}.mp4"; ex = os.path.exists(f"{PREP}/{path}"); dur = float(v["videoMetadata"]["duration"]["seconds"])
    F = []
    for e in v["events"]:
        a = e.get("attributes") or {}
        if e["label"] != "Fine grained action" or "end" not in e or not a.get("Verb") or not a.get("Noun") or cl(a["Verb"]) == "none" or cl(a["Noun"]) == "none": continue
        F.append((float(e["start"]), float(e["end"]), phrase(a), cl(a["Verb"]), cl(a["Noun"]), str(a.get("Action Correctness", ""))))
    F.sort()
    for s, e, ph, vb, nn, corr in F:                                                  # hfine
        if not (0.6 <= e - s <= 10.0): continue
        pool = {x[2]: x for x in F if x[2] != ph}
        if len(pool) < 3: continue
        same_n = [p for p, x in pool.items() if x[4] == nn and x[3] != vb]; same_v = [p for p, x in pool.items() if x[3] == vb and x[4] != nn]
        ds = []
        for cand in (same_n, same_v, list(pool)):
            c_ = [p for p in cand if p not in ds]
            if c_: ds.append(c_[int(g.rng.integers(len(c_)))])
        while len(ds) < 3:
            c_ = [p for p in pool if p not in ds]; ds.append(c_[int(g.rng.integers(len(c_)))])
        out["fine"].append(g.row(path, s, e, g.pick(Q_F), [ph] + ds[:3], 0, "", ex))
    i = 0                                                                             # hseq (as zz_qgen_seq.build)
    sp = [(s, e, ph) for s, e, ph, *_ in F if e - s >= 0.6]
    while i < len(sp):
        k = int(g.rng.choice([3, 4])); seq = [sp[i]]
        for x in sp[i + 1:]:
            if len(seq) == k: break
            if x[0] >= seq[-1][1] + 0.2 and x[2] not in {y[2] for y in seq}: seq.append(x)
        if len(seq) == k and seq[-1][1] - seq[0][0] <= 20.0:
            true = [x[2] for x in seq]; perms = [list(p) for p in itertools.permutations(true) if list(p) != true]
            others = [perms[j] for j in g.rng.permutation(len(perms))[:3]]
            out["seq"].append(g.row(path, seq[0][0], seq[-1][1], g.pick(Q_S), [", then ".join(true)] + [", then ".join(p) for p in others], 0, "", ex)); i += 2
        else: i += 1
    WR += [(path, ex) + x for x in F if x[5].startswith("Wrong Action") and 0.6 <= x[1] - x[0] <= 10.0]   # hmist, paired after the loop
    for x in F:
        if x[5] == "Correct Action" and 0.6 <= x[1] - x[0] <= 10.0: RI[x[2]].append((path, ex) + x)
    tk = cl(v.get("taskType") or "")                                                  # htask
    if tk in TASKS and dur > 20:
        for t0 in g.rng.uniform(0, dur - 16.0, size=max(1, int(dur // 60))):
            others = [TASKS[j] for j in g.rng.permutation(len(TASKS)) if TASKS[j] != tk][:3]
            out["task"].append(g.row(path, float(t0), float(t0) + 16.0, g.pick(Q_T), [tk] + others, 0, "", ex))
used = set()
for j in g.rng.permutation(len(WR)):
    w = WR[j]; cand = [c for c in RI[w[4]] if id(c) not in used]
    if not cand: kinds["mist_unpaired"] += 1; continue
    c = min(cand, key=lambda c: abs((c[3] - c[2]) - (w[3] - w[2]))); used.add(id(c))
    for x, lab in ((w, 1), (c, 0)):
        opts = ["No, it is a mistake", "Yes, it is done correctly"] if lab else ["Yes, it is done correctly", "No, it is a mistake"]
        out["mist"].append(g.row(x[0], x[2], x[3], f"The person tries to {x[4]}. Is this step done correctly?", opts, 0, "", x[1])); kinds["mist_" + str(lab)] += 1
for k, rr in out.items():
    report(f"holo_{k}", rr); pd.DataFrame(rr).to_parquet(f"{OUT}/holo_{k}.parquet")
print(dict(kinds), "| tasks", len(TASKS), TASKS[:6], "| videos", len(D))
