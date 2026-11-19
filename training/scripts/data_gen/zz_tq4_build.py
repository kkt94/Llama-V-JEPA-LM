"""(10-16) tq4 = tq3's transformation questions with the option TEXT made uninformative, + event order inside one natural video.
Why: tq2_A shifted answer POSITIONS on other benchmarks (Perception Test: the third option chosen 1335 -> 594 times of 3465; with the
per-letter bias removed PT is 55.2 vs 52.9, i.e. the drop is mostly position habit), and our options carried length cues (relative speed:
the right option was the SHORTEST in 66%; order: the LONGEST in 53%, because the 'together' option was short and usually wrong).
 1) every option phrase that names a category (faster / slower / same speed, accelerating / decelerating / not moving, together, the
    direction words) is redrawn per option from a pool of phrasings of different lengths, so the answer's length rank is ~uniform
 2) 30% of items get a second turn: the same question with the options in another order (the answer must follow the content, not a slot)
 3) + order_nat (logs/zz_order_nat_build.py: two events of one SA-V video, the video itself or its segments joined either way)
reads manifests/stage1/tq3/*.parquet -> manifests/stage1/tq4/*.parquet; prints, per kind, where the answer ranks by length (before / after)"""
import json, re, glob, os, collections, numpy as np, pandas as pd
R = "/data/llama_vjepa_lm"; IN, OUT = f"{R}/manifests/stage1/tq3", f"{R}/manifests/stage1/tq4"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0); TAIL = "\nAnswer with the option's letter from the given choices directly."
POOLS = {
 "faster": ["faster", "quicker", "faster than the one below", "moving more quickly than the one below", "at a faster pace"],
 "slower": ["slower", "more slowly", "slower than the one below", "moving more slowly than the one below", "at a slower pace"],
 "same":   ["same speed", "equally fast", "the same speed", "at the same speed as the one below", "at an equal pace"],
 "acc":    ["Accelerating.", "Speeding up.", "It speeds up.", "It gets faster and faster.", "Accelerating over time."],
 "dec":    ["Decelerating.", "Slowing down.", "It slows down.", "It gets slower and slower.", "Decelerating over time."],
 "none":   ["Not moving at all.", "Still.", "It does not move.", "No movement.", "Not moving."],
 "tog":    ["Both events happen at the same time", "Both events happen simultaneously", "Both events happen together", "Simultaneously",
            "At the same time", "Both at once", "The two events happen at the same time, one above or beside the other"],
}
CAT = {p: c for c, ps in POOLS.items() for p in ps}
CAT.update({"the same speed": "same", "It stays about the same": "same", "It accelerates": "acc", "It decelerates": "dec", "It does not change": "same",
            "faster than the one below": "faster", "slower than the one below": "slower", "at the same speed as the one below": "same",
            "Both events happen at the same time": "tog", "Both events happen simultaneously": "tog", "Both events happen together": "tog"})
DIRP = {  # direction words: each category gets several phrasings
 "left": ["to the left", "going left", "leftward", "moving towards the left side"], "right": ["to the right", "going right", "rightward", "moving towards the right side"],
 "up": ["upwards", "going up", "up", "moving towards the top"], "down": ["downwards", "going down", "down", "moving towards the bottom"],
 "towards": ["towards the camera", "coming towards the screen", "closer to the camera", "approaching the camera"],
 "away": ["away from the camera", "going away from the screen", "further from the camera", "receding from the camera"],
 "cleft": ["panning left", "turning leftward", "the camera turns left", "rotating to the left"], "cright": ["panning right", "turning rightward", "the camera turns right", "rotating to the right"],
 "cup": ["tilting up", "turning upward", "the camera tilts up", "pointing higher"], "cdown": ["tilting down", "turning downward", "the camera tilts down", "pointing lower"],
 "fwd": ["moving forward", "moving towards the scene", "the camera moves forward", "advancing"], "bwd": ["moving backward", "moving away from the scene", "the camera moves back", "retreating"],
}
for c, ps in DIRP.items(): POOLS[c] = ps
DIRC = {"to the left": "left", "going left": "left", "to the right": "right", "going right": "right", "upwards": "up", "going up": "up", "downwards": "down",
        "going down": "down", "towards the camera": "towards", "coming towards the screen": "towards", "away from the camera": "away",
        "going away from the screen": "away", "panning left": "cleft", "turning leftward": "cleft", "panning right": "cright", "turning rightward": "cright",
        "tilting up": "cup", "turning upward": "cup", "tilting down": "cdown", "turning downward": "cdown", "moving forward": "fwd",
        "moving towards the scene": "fwd", "moving backward": "bwd", "moving away from the scene": "bwd"}
CAT.update(DIRC)


def parse(cv):
    x = json.loads(cv); u = x[0]["content"]; q = u.split("\n")[1]
    opts = [o for _, o in re.findall(r"\n([A-G])\. (.*)", u)]; k = "ABCDEFG".index(x[1]["content"][0]); return q, opts, k


def render(q, opts, k, second):
    u = f"<|jepa_evidence|>\n{q}\n" + "\n".join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(opts)) + TAIL
    conv = [{"role": "user", "content": u}, {"role": "assistant", "content": "ABCDEFG"[k] + "."}]
    if second:
        p = rng.permutation(len(opts))
        while len(opts) > 1 and (p == np.arange(len(opts))).all(): p = rng.permutation(len(opts))
        o2 = [opts[i] for i in p]; k2 = int(np.nonzero(p == k)[0][0])
        conv += [{"role": "user", "content": pick(["Same question, options in another order:", "Again, with the options reordered:", "Once more:"]) + f"\n{q}\n"
                  + "\n".join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(o2)) + TAIL}, {"role": "assistant", "content": "ABCDEFG"[k2] + "."}]
    return json.dumps(conv)


pick = lambda xs: xs[int(rng.integers(len(xs)))]


def redraw1(opts):
    out = []
    for o in opts:
        c = CAT.get(o.rstrip()) or CAT.get(o)
        out.append(pick(POOLS[c]) if c else o)
    return out if len(set(out)) == len(out) else opts


def redraw(opts, k):
    """redraw until the answer's length rank equals a target drawn uniformly (longest / middle / shortest; up to 40 tries)"""
    if not any(CAT.get(o.rstrip()) or CAT.get(o) for o in opts): return opts
    tgt = pick(["longest", "shortest"] + (["middle"] if len(opts) > 2 else [])); best = None
    for _ in range(40):
        o2 = redraw1(opts); best = best or o2
        if lrank(o2, k) == tgt: return o2
    return best


def lrank(opts, k):
    L = [len(o) for o in opts]; return "longest" if L[k] == max(L) and L.count(max(L)) == 1 else "shortest" if L[k] == min(L) and L.count(min(L)) == 1 else "middle"   # ties: middle


for f in sorted(glob.glob(f"{IN}/*.parquet")):
    name = os.path.basename(f); d = pd.read_parquet(f); before, after, convs = collections.Counter(), collections.Counter(), []
    for cv in d.conversations:
        q, opts, k = parse(cv); before[lrank(opts, k)] += 1
        o2 = redraw(opts, k); after[lrank(o2, k)] += 1
        convs.append(render(q, o2, k, rng.random() < 0.3))
    d = d.assign(conversations=convs); d.to_parquet(f"{OUT}/{name}")
    n = len(d); fmt = lambda c: " ".join(f"{k} {100*c[k]/n:.0f}" for k in ("longest", "middle", "shortest"))
    print(f"{name:22s} n {n:6d} | answer by length before: {fmt(before)} | after: {fmt(after)}", flush=True)
