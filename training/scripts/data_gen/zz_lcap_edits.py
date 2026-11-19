"""(10-28) lcap step 2: RULE-BASED single edits of each human gold caption (no model text). One edit per negative; each candidate records its
edit type (TemporalBench-like categories), the direction of the change (word a -> word b, used for mirror balancing), and the char span.
  dir       left<->right (+ 'from left to right' as one phrase, leftward/rightward), up<->down after motion verbs, upward/downward, upstairs,
            forward<->backward, clockwise<->counterclockwise, towards X<->away from X
  effector  hand<->foot, hands<->feet, arm<->leg (noun after a determiner), left hand<->right hand (any left/right + body part)
  count     once<->twice<->three times, 'N times' -> N+-1, number word + plural noun -> N+-1 (2..10)
  order     before<->after; two adjacent sentences swapped (only where the sentences are known to be in time order: ActivityNet / PLM dense
            runs with their segment times ordered, PLM single captions narrating one subject)
  action    verb antonyms with their inflections: open/close, push/pull, picks up/puts down, sit/stand (sits down/stands up), walk/run,
            throw/catch, enter/exit, start/stop, appear/disappear, turn on/off, put on/take off, raise/lower, ascend/descend, fold/unfold-type
            un- verbs, fill/empty, speeds up/slows down, smile/frown, attach/detach, connect/disconnect
  speed     quickly/fast/rapidly<->slowly, faster<->slower
VidLN captions (kin/oops/uvo/ovis): edits only before 'In the background' (the background list is not an event); no sentence swaps (per-actor
narratives, their order is not time).  A candidate identical to the gold or with a doubled word is dropped.  Max 8 candidates per gold
(one per type first, then random).  -> logs/zz_lcap_cands.parquet (gid, typ, sub, dirn, neg, span)  usage: python -I zz_lcap_edits.py"""
import re, json, collections
import numpy as np, pandas as pd
R = "/data/llama_vjepa_lm"; rng = np.random.default_rng(2028)
G = pd.read_parquet(f"{R}/logs/zz_lcap_golds.parquet")
WRE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
VIDLN = {"kin", "oops", "uvo", "ovis"}

def cap_like(src, dst):
    return dst[:1].upper() + dst[1:] if src[:1].isupper() else dst

def infl(base, s3, past, ing):
    return dict(base=base, s3=s3, past=past, ing=ing)

# verb antonym pairs: (type, sub, forms A, forms B); base form only swapped after 'to'
VA = [("action", "open_close", infl("open", "opens", "opened", "opening"), infl("close", "closes", "closed", "closing")),
      ("action", "push_pull", infl("push", "pushes", "pushed", "pushing"), infl("pull", "pulls", "pulled", "pulling")),
      ("action", "sit_stand", infl("sit", "sits", "sat", "sitting"), infl("stand", "stands", "stood", "standing")),
      ("action", "walk_run", infl("walk", "walks", "walked", "walking"), infl("run", "runs", "ran", "running")),
      ("action", "throw_catch", infl("throw", "throws", "threw", "throwing"), infl("catch", "catches", "caught", "catching")),
      ("action", "enter_exit", infl("enter", "enters", "entered", "entering"), infl("exit", "exits", "exited", "exiting")),
      ("action", "start_stop", infl("start", "starts", "started", "starting"), infl("stop", "stops", "stopped", "stopping")),
      ("action", "appear", infl("appear", "appears", "appeared", "appearing"), infl("disappear", "disappears", "disappeared", "disappearing")),
      ("action", "raise_lower", infl(None, "raises", "raised", "raising"), infl(None, "lowers", "lowered", "lowering")),
      ("action", "ascend", infl("ascend", "ascends", "ascended", "ascending"), infl("descend", "descends", "descended", "descending")),
      ("action", "fill_empty", infl("fill", "fills", "filled", "filling"), infl("empty", "empties", "emptied", "emptying")),
      ("action", "smile_frown", infl("smile", "smiles", "smiled", "smiling"), infl("frown", "frowns", "frowned", "frowning")),
      ("action", "attach", infl("attach", "attaches", "attached", "attaching"), infl("detach", "detaches", "detached", "detaching")),
      ("action", "connect", infl("connect", "connects", "connected", "connecting"), infl("disconnect", "disconnects", "disconnected", "disconnecting")),
      ("action", "inflate", infl("inflate", "inflates", "inflated", "inflating"), infl("deflate", "deflates", "deflated", "deflating")),
      ("action", "tighten", infl("tighten", "tightens", "tightened", "tightening"), infl("loosen", "loosens", "loosened", "loosening"))]
for v, s3, pa, ing in (("fold", "folds", "folded", "folding"), ("lock", "locks", "locked", "locking"), ("wrap", "wraps", "wrapped", "wrapping"),
                       ("zip", "zips", "zipped", "zipping"), ("plug", "plugs", "plugged", "plugging"), ("pack", "packs", "packed", "packing"),
                       ("load", "loads", "loaded", "loading"), ("tie", "ties", "tied", "tying"), ("screw", "screws", "screwed", "screwing"),
                       ("dress", "dresses", "dressed", "dressing"), ("button", "buttons", "buttoned", "buttoning")):
    VA.append(("action", "un_" + v, infl(v, s3, pa, ing), infl("un" + v, "un" + s3, "un" + pa, "un" + ing)))
VMAP = {}                                   # word -> (type, sub, replacement, form)
for typ, sub, A, B in VA:
    for f in ("base", "s3", "past", "ing"):
        if A[f] and B[f]:
            VMAP[A[f]] = (typ, sub, B[f], f); VMAP[B[f]] = (typ, sub, A[f], f)
VMAP.pop("buttons", None); VMAP.pop("button", None); VMAP.pop("dress", None); VMAP.pop("dresses", None)   # mostly nouns
# phrase pairs (whole phrase swapped; checked before single words)
PH = [("action", "sit_stand", [("sits down", "stands up"), ("sit down", "stand up"), ("sat down", "stood up"), ("sitting down", "standing up")]),
      ("action", "pickup_putdown", [("picks up", "puts down"), ("picked up", "put down"), ("picking up", "putting down")]),
      ("action", "turn_on_off", [("turns on", "turns off"), ("turned on", "turned off"), ("turning on", "turning off"), ("switches on", "switches off"),
                                 ("switched on", "switched off")]),
      ("action", "put_on_take_off", [("puts on", "takes off"), ("putting on", "taking off")]),
      ("action", "speed_up_down", [("speeds up", "slows down"), ("speeding up", "slowing down"), ("sped up", "slowed down")]),
      ("dir", "lr_phrase", [("from left to right", "from right to left"), ("left to right", "right to left")]),
      ("dir", "clockwise", [("counter clockwise", "clockwise"), ("anti clockwise", "clockwise")])]
PHL = []
for typ, sub, prs in PH:
    for a, b in prs: PHL += [(a, b, typ, sub), (b, a, typ, sub)]
PHL.sort(key=lambda x: -len(x[0]))

WORDSWAP = {"clockwise": ("dir", "clockwise", "counterclockwise"), "counterclockwise": ("dir", "clockwise", "clockwise"),
            "counter-clockwise": ("dir", "clockwise", "clockwise"), "anticlockwise": ("dir", "clockwise", "clockwise"),
            "anti-clockwise": ("dir", "clockwise", "clockwise"),
            "upward": ("dir", "up_down", "downward"), "downward": ("dir", "up_down", "upward"), "upwards": ("dir", "up_down", "downwards"),
            "downwards": ("dir", "up_down", "upwards"), "upstairs": ("dir", "up_down", "downstairs"), "downstairs": ("dir", "up_down", "upstairs"),
            "uphill": ("dir", "up_down", "downhill"), "downhill": ("dir", "up_down", "uphill"),
            "leftward": ("dir", "left_right", "rightward"), "rightward": ("dir", "left_right", "leftward"), "leftwards": ("dir", "left_right", "rightwards"),
            "rightwards": ("dir", "left_right", "leftwards"), "left-hand": ("dir", "left_right", "right-hand"), "right-hand": ("dir", "left_right", "left-hand"),
            "forward": ("dir", "fwd_back", "backward"), "backward": ("dir", "fwd_back", "forward"), "forwards": ("dir", "fwd_back", "backwards"),
            "backwards": ("dir", "fwd_back", "forwards"),
            "once": ("count", "times", "twice"), "twice": ("count", "times", None),
            "before": ("order", "before_after", "after"), "after": ("order", "before_after", "before"),
            "quickly": ("speed", "speed", "slowly"), "rapidly": ("speed", "speed", "slowly"), "slowly": ("speed", "speed", "quickly"),
            "faster": ("speed", "speed", "slower"), "slower": ("speed", "speed", "faster")}
DET = {"his", "her", "their", "its", "my", "your", "the", "both", "one", "a", "an", "two", "bare", "left", "right", "own", "other", "each", "our"}
BODY = {"hand": "foot", "foot": "hand", "hands": "feet", "feet": "hands", "arm": "leg", "leg": "arm", "arms": "legs", "legs": "arms"}
BODYN = set(BODY) | {"knee", "knees", "shoulder", "shoulders", "elbow", "elbows", "eye", "ear", "wrist", "palm", "thumb", "finger", "fingers", "cheek",
                     "heel", "toe", "toes", "hip", "side"}
SIDEN = BODYN | {"corner", "edge", "lane", "wing", "half", "part", "direction", "bank", "end", "of", "turn", "bottom", "top", "front", "back",
                 "rear", "border", "portion", "area", "sides", "hand-side"}
DIRP = {"the", "his", "her", "their", "its", "my", "your", "to", "towards", "toward", "from", "turns", "turn", "turned", "turning", "moves", "move",
        "moved", "moving", "on", "at", "far", "bottom", "top", "upper", "lower", "lean", "leans", "leaning", "leaned", "swerves", "steers", "veers",
        "goes", "go", "going", "went", "looks", "look", "looking", "looked", "glances", "glancing", "rolls", "rolling", "slides", "sliding", "swings",
        "swinging", "shifts", "shifting", "pans", "panning", "tilts", "tilting", "facing", "faces", "faced", "heads", "heading", "walks", "walking",
        "runs", "running", "rotates", "rotating", "bends", "bending", "steps", "stepping", "jumps", "jumping", "hops", "drifts", "swerving", "and"}
RIGHT_ADV = {"now", "away", "after", "before", "behind", "into", "there", "here", "next", "back", "in", "on", "at", "through", "under", "above", "below",
             "beside", "over", "up", "down", "onto", "by", "then", "across", "around", "off", "out", "where", "as", "when", "near"}
POSS = {"the", "his", "her", "their", "its", "my", "your", "our", "far", "upper", "lower", "bottom", "top"}
UDV = set()
for b in ("look", "glance", "move", "climb", "walk", "ride", "roll", "swim", "scroll", "head", "swipe", "pan", "tilt", "float", "crawl", "point",
          "drift", "travel", "skate", "ski", "hop", "slide", "jump", "bounce", "pull", "push", "zoom", "stare", "gaze", "peer", "scroll", "rush",
          "race", "march", "hike", "paddle", "sail", "drive", "flow", "fly", "go", "come", "run", "step", "swing"):
    UDV |= {b, b + "s", b + "ed", b + "ing", b + "es"}
UDV |= {"went", "came", "ran", "rode", "slid", "flew", "flies", "flying", "swam", "swung", "stepped", "stepping", "running", "swimming", "sliding",
        "riding", "coming", "moving", "hopping", "hopped", "goes", "drove", "driving", "climbs", "moves", "rolled",
        "scrolled", "panned", "tilted", "panning", "tilting"}
NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven"]
IRR_PL = {"people", "children", "men", "women", "feet", "teeth", "fish", "sheep", "deer", "geese", "mice", "persons", "times"}
SENT_RE = re.compile(r"[^.!?]+[.!?]+(?:\s+|$)|[^.!?]+$")


def cands(row):
    s = row.cap; ex = json.loads(row.extra); out = []
    lim = len(s)
    if row.src in VIDLN:
        m = re.search(r"\bIn the (back|fore)ground\b", s)
        if m: lim = m.start()
    W = [(m.start(), m.end(), m.group(0)) for m in WRE.finditer(s)]
    lw = [w[2].lower() for w in W]
    used = set()                                            # word indices consumed by a phrase match

    def emit(typ, sub, a, b, i0, i1, repl):
        st, en = W[i0][0], W[i1][1]
        if en > lim: return
        neg = s[:st] + repl + s[en:]
        if neg != s: out.append((typ, sub, f"{a}>{b}", neg, st))
    # phrases
    for i in range(len(W)):
        for a, b, typ, sub in PHL:
            ws = a.split(); k = len(ws)
            if lw[i:i + k] == ws and not (set(range(i, i + k)) & used):
                if sub == "lr_phrase" and k == 3 and i > 0 and lw[i - 1] == "from": continue
                emit(typ, sub, a, b, i, i + k - 1, cap_like(W[i][2], b)); used |= set(range(i, i + k))
    for i, (st, en, w) in enumerate(W):
        if i in used: continue
        lo = lw[i]; p = lw[i - 1] if i else ""; n = lw[i + 1] if i + 1 < len(W) else ""; n2 = lw[i + 2] if i + 2 < len(W) else ""
        pp = lw[i - 2] if i > 1 else ""
        # left / right
        if lo in ("left", "right"):
            if n in ("and", "or", "to") and n2 in ("left", "right"): continue
            if p in ("and", "or", "to") and pp in ("left", "right"): continue
            other = "right" if lo == "left" else "left"
            ok = (p in DIRP or n in SIDEN)
            if lo == "right" and n in RIGHT_ADV and p not in POSS: ok = False
            if lo == "left" and p in ("and",): ok = n in SIDEN
            if lo == "right" and p in ("all", "just", "and"): ok = n in SIDEN and p != "all"
            if ok:
                if n in BODYN - {"side"}: emit("effector", "body_side", lo, other, i, i, cap_like(w, other))
                else: emit("dir", "left_right", lo, other, i, i, cap_like(w, other))
            continue
        if lo in ("up", "down"):
            if p in UDV and not (n in ("and", "or") and n2 in ("up", "down")) and not (p in ("and", "or") and pp in ("up", "down")):
                emit("dir", "up_down", lo, "down" if lo == "up" else "up", i, i, cap_like(w, "down" if lo == "up" else "up"))
            continue
        if lo in ("towards", "toward"):
            if n == "the" and n2 in ("end", "beginning", "start", "middle", "conclusion", "close"): continue
            if n in ("each", "one"): continue
            emit("dir", "towards_away", "towards", "away from", i, i, cap_like(w, "away from")); continue
        if lo == "away" and n == "from":
            emit("dir", "towards_away", "away from", "towards", i, i + 1, cap_like(w, "towards")); continue
        if lo in WORDSWAP:
            typ, sub, b = WORDSWAP[lo]
            if lo in ("forward", "forwards") and (p in ("look", "looks", "looking", "straight") or n == "to"): continue
            if lo in ("before", "after") and (n in ("all", "which") or p in ("look", "looks", "looked", "looking", "ever", "day", "night", "year",
                                                                            "soon", "shortly", "right")): continue
            if lo == "after" and n in ("him", "her", "them", "it") and p in ("runs", "run", "running", "chases", "chasing", "goes", "go"): continue
            if lo == "slower" and n == "motion": continue
            if lo in ("once", "twice") and (w[0].isupper() or n in ("the", "a", "an", "he", "she", "they", "it", "we", "you", "his", "her", "their",
                                                                   "this", "that", "there", "inside", "all", "both", "in", "on", "at", "upon")): continue
            if lo == "twice":
                b = "once" if rng.random() < 0.5 else "three times"
            emit(typ, sub, lo, b, i, i, cap_like(w, b)); continue
        if lo in BODY and p in DET and not (lo in ("hand", "hands") and p in ("a", "an") and n in ("to", "in")):
            if lo in ("hands", "hand") and p == "both" and False: pass
            b = BODY[lo]
            if p in ("a", "an", "one", "each") and b in ("arm",) : continue
            if p == "a" and b == "arm": continue
            if p == "an" and b in ("foot", "hand", "leg"): continue
            emit("effector", "hand_foot" if lo in ("hand", "hands", "foot", "feet") else "arm_leg", lo, b, i, i, cap_like(w, b)); continue
        if lo in NUM[2:] and (n in IRR_PL or (n.endswith("s") and len(n) > 3 and not n.endswith("ss"))) and p not in ("at", "by", "of") :
            k = NUM.index(lo); opts = [x for x in (k - 1, k + 1) if 2 <= x <= 10]
            if n == "times": opts = [x for x in (k - 1, k + 1) if 1 <= x <= 10]
            b = NUM[int(rng.choice(opts))]
            if n == "times" and b == "one":
                emit("count", "times", f"{lo} times", "once", i, i + 1, cap_like(w, "once")); continue
            emit("count", "times" if n == "times" else "objects", lo, b, i, i, cap_like(w, b)); continue
        if lo in VMAP:
            typ, sub, b, f = VMAP[lo]
            if f == "base" and p != "to": continue
            if sub == "sit_stand" and (n in ("up", "down") or p in ("a", "the", "his", "her", "their")): continue
            if sub == "push_pull" and n in ("up", "out", "off", "over", "on", "apart", "ups", "through", "in", "down", "back", "away", "along"): continue
            if sub == "walk_run" and (n in ("out", "into", "over", "off", "through", "water", "shoes", "track", "away") or p in ("a", "the", "his", "her")): continue
            if sub == "start_stop" and (n in ("to", "at", "from", "by", "with", "the", "a", "his", "her") or p in ("a", "the", "bus")): continue
            if sub == "open_close" and (n in ("to", "by", "up", "in") or lo in ("closed", "opened") and n in ("with", "eyes")): continue
            if sub == "fill_empty" and (n == "with" or lo in ("empty", "filled", "emptied") and p in ("is", "are", "was", "an", "a", "the")): continue
            if sub in ("raise_lower",) and n in ("hand",): pass
            if sub.startswith("un_") and (n in ("of", "with") or p in ("a", "the", "his", "her", "their", "of")): continue
            if sub == "enter_exit" and p in ("the", "an", "a"): continue
            if sub in ("attach", "connect") and (n == "to" or f == "past"): continue
            if sub == "throw_catch" and n in ("up",) and lo.startswith("catch"): continue
            emit(typ, sub, lo, b, i, i, cap_like(w, b)); continue
    # sentence swaps (time-ordered sources only)
    if row.src in ("anet", "plmd", "plm"):
        if row.src in ("anet", "plmd"):
            sents, ts = ex["sents"], ex["ts"]
            if " ".join(sents) != s: sents = None
        else:
            sents = [x.strip() for x in SENT_RE.findall(s) if x.strip()]; ts = None
            if " ".join(sents) != s: sents = None
        if sents and len(sents) >= 2:
            for k in range(len(sents) - 1):
                a, b = sents[k], sents[k + 1]
                if ts is not None and ts[k + 1][0] < ts[k][1] - 1.0: continue          # overlapping segments: order not defined
                if ts is None:
                    bad = re.compile(r"^(There|In the|The video|It is|This)\b|\bwear|\bbackground\b|\bwhile\b|\bcontinues?\b", re.I)
                    if bad.search(a) or bad.search(b) or len(a.split()) < 4 or len(b.split()) < 4: continue
                if a.strip(" .") == b.strip(" .") : continue
                neg = " ".join(sents[:k] + [b, a] + sents[k + 2:])
                if neg != s: out.append(("order", "sent_swap", "swap", neg, len(" ".join(sents[:k]))))
    # dedupe, drop bad
    res, seen = [], set()
    for typ, sub, dirn, neg, st in out:
        if neg in seen or neg == s or re.search(r"\b(\w+) \1\b", neg, re.I) and not re.search(r"\b(\w+) \1\b", s, re.I): continue
        seen.add(neg); res.append((typ, sub, dirn, neg, st))
    return res


rows = []
for r in G.itertuples():
    c = cands(r)
    if not c: continue
    # at most 8: one per type first (random within type), then random others
    rng.shuffle(c); bytype = collections.OrderedDict()
    for x in c: bytype.setdefault(x[0], []).append(x)
    pick = [v[0] for v in bytype.values()]; rest = [x for v in bytype.values() for x in v[1:]]
    pick += rest[: max(0, 12 - len(pick))]
    for typ, sub, dirn, neg, st in pick: rows.append(dict(gid=r.gid, typ=typ, sub=sub, dirn=dirn, neg=neg, span=st))
C = pd.DataFrame(rows)
C.to_parquet(f"{R}/logs/zz_lcap_cands.parquet")
print("golds", len(G), "with >= 1 candidate", C.gid.nunique(), "candidates", len(C), flush=True)
print(pd.crosstab(C.merge(G[["gid", "src"]], on="gid").src, C.typ), flush=True)
print(C.groupby(["typ", "sub"]).size().to_string(), flush=True)
for typ, x in C.groupby("sub"):
    for q in x.sample(min(2, len(x)), random_state=1).itertuples():
        g = G.cap[q.gid]; st = max(0, q.span - 60)
        print(f"[{typ} {q.dirn}]\n   G: ...{g[st:st + 160]}\n   N: ...{q.neg[st:st + 160]}", flush=True)
