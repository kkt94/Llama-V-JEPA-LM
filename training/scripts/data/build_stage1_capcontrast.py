"""Stage-1 minimal-contrast caption pairs for fine-grained temporal understanding (TemporalBench / Vinoground style).

Each row is a 2-option multiple-choice item "Which caption best describes this video?": one option is an excerpt of the
video's own detailed caption (GPT-4V ShareGPTVideo captions, LLaVA-Video-178K detailed captions; TRAIN videos already
contamination-filtered in manifests/stage1/{sharegpt,llava}.parquet), the other is the SAME excerpt with one small
rule-based edit:

  order      two temporally ordered events swap places while connectives stay put, so the text stays fluent:
               - adjacent sentences "S1 VP1. Then, S2 VP2." / "S2 then VP2." -> predicates swapped (subjects kept,
                 only when S2 is S1 or a matching pronoun, so who-does-what does not change)
               - "S VP1, (and) then VP2." -> "S VP2, (and) then VP1."
               - "... with NP1, followed by NP2" -> "... with NP2, followed by NP1"
               - "S VP1 before/after VPing2" <-> "S VP2 before/after VPing1", "After VPing1, S VP2" <-> ...
                 (3sg <-> gerund forms taken from a corpus-validated lexicon, no guessing)
             moved pieces may not contain object pronouns or order-presupposing words (again, continues, returns,
             another, finally, ...), which would make the wrong order readable from the text alone.
  direction  one reversible word/phrase in one sentence: left<->right (only with a spatial cue: 'the left side',
             'turns left'), up<->down / forward<->backward / toward(s)<->away from (only after a motion verb),
             into<->out of (idioms excluded), opens<->closes, picks up<->puts down, enters<->exits/leaves (a place),
             raises/lifts<->lowers, pushes<->pulls, starts<->stops (+Ving), appears<->disappears, increases<->decreases,
             ascends<->descends, clockwise<->counterclockwise, fills<->empties, ties<->unties ..., and whole phrases
             'from left to right' <-> 'from right to left'. Inflection by table, case preserved, a/an fixed.
  count      a number 2..10 (word or digit) followed by a counted plural noun (units, ages, frames, ... excluded)
             changed by +-1 (2..10 kept, so the noun's plural form stays correct).

Text-only shortcut control: for every direction/count surface pair (e.g. 'into|out of', 'two|three') the selection
keeps exactly as many items whose TRUE caption has form a as items whose true caption has form b (detailed balance),
so word identity carries no information about which option is true. Order edits are permutations of the same words.
Balance is done per stratum = pair + previous word + next word (count: pair + counted noun), which removes the
context priors ("pointing towards", "into the") a text-only n-gram classifier otherwise exploits (65% -> ~50%).
Order swaps keep the article class of the moved pieces equal (a/the), so given-new article order gives nothing away.
Order items are balanced the same way on the first word where the two options diverge ('view' vs 'close-up',
'adjusts' vs 'gestures', 'day' vs 'night'): as many items have that word first in the TRUE caption as in the negative
(order n-gram classifier 62% -> 57%, at the price of ~70% of the order candidates). Selection order: order, count,
direction (scarcest first). Text-only check (hashed word 1-2gram + char 3-5gram LR, 80/20 by video): order 0.57,
count 0.52, direction 0.49; length/punctuation features 0.50.

Both options are the same excerpt span (2-4 sentences containing the edit, <=120 words; 1 sentence if nothing longer
fits). One item per video. Target volume: ShareGPTVideo 150k, LLaVA 100k, edit mix ~ order 40 / direction 40 / count 20
subject to availability. Answer letters are exactly balanced. Prompt = evaluator MC format.

Output: manifests/stage1/capcontrast_{sharegpt,llava}.parquet (+ _heldout: 2% by md5(video)), columns
  video, prep_exists (copied from the source manifest), kind (order|direction|count), conversations (JSON string).
Run:  nice -n 10 python scripts/data/build_stage1_capcontrast.py [--src sharegpt llava] [--workers 16] [--meta_out DIR]
"""
import argparse
import collections
import hashlib
import json
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd

R = "/data/llama_vjepa_lm"
VIS = "<|jepa_evidence|>"
SUFFIX = "Answer with the option's letter from the given choices directly."
QUESTIONS = ["Which caption best describes this video?", "What is the best caption for this video?",
             "Which description matches the video?", "Which caption is correct for this video?",
             "Which of the following descriptions fits the video?", "Which caption accurately describes the video?"]
SOURCES = {"sharegpt": dict(kinds=("cap", "cap6"), target=150000),
           "llava": dict(kinds=("cap",), target=100000)}
MIX = {"order": 0.4, "direction": 0.4, "count": 0.2}
MAXW = 120
MAX_KEY_SHARE = 0.25          # one surface pair at most 25% of its type

# ----------------------------------------------------------------------------------------------- text utilities
TOK = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:[.,:/]\d+)*|\S")
ABBR = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|etc|No|approx|e\.g|i\.e|U\.S)\.|\b[A-Z]\.)$")


def split_sents(text):
    text = re.sub(r"(?:\\n|\n)+\s*\d{1,2}\.\s+(?=[A-Z])", " ", text)     # numbered list markers
    text = re.sub(r"(?:\\n)+", " ", text)                                # literal backslash-n in some captions
    text = re.sub(r"(^|(?<=[.!?:]))\s*\d{1,2}\.\s+(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+(?=[\"'(]?[A-Z0-9])", text)
    out = []
    for p in parts:
        if out and ABBR.search(out[-1]):
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return [p for p in out if p]


def toks(s):
    return [(m.group(), m.start(), m.end()) for m in TOK.finditer(s)]


def nwords(s):
    return len(s.split())


def match_case(src, new):
    if src[:1].isupper() and not src.isupper():
        return new[:1].upper() + new[1:]
    if src.isupper() and len(src) > 1:
        return new.upper()
    return new


def an_for(word):
    w = word.lower()
    if w.startswith(("one", "uni", "use", "usu", "eu", "ur")):
        return "a"
    return "an" if w[:1] in "aeiou" else "a"


def replace_span(s, lo, hi, new):
    """Replace s[lo:hi] with new (case of the first letter preserved) and fix a preceding a/an."""
    new = match_case(s[lo:hi], new)
    out = s[:lo] + new + s[hi:]
    m = re.search(r"\b(a|an|A|An)\s+$", out[:lo])
    if m:
        art = an_for(new)
        art = art.capitalize() if m.group(1)[0].isupper() else art
        out = out[:m.start(1)] + art + out[m.end(1):]
    return out


# ----------------------------------------------------------------------------------------------- word lists
OBJ = set("the a an his her their its it them one both each this that these those my your our some two three several "
          "another him".split())
DET_POSS = set("the his her their its my your our".split())
SUBJ_PRON = {"he", "she", "they", "it"}
ANAPH = set("it its them they him this these those itself themselves himself herself".split())
PRESUP = set("again continues continue continuing still remaining rest also too back returns return returning finishes "
             "finish finished completes complete completed resumes resume another other further same then finally "
             "after before afterwards afterward later next once first second third last final previous previously "
             "initially eventually already yet more meanwhile simultaneously while repeat repeats repeating ends begins "
             "starts".split())
FRAME_HEADS = set("begins starts opens ends concludes commences culminates transitions shifts cuts changes switches "
                  "returns continues resumes finishes completes repeats proceeds fades wraps closes".split())
HUMAN_M = set("man boy guy gentleman father dad husband king son brother grandfather".split())
HUMAN_F = set("woman girl lady mother mom wife queen daughter sister grandmother".split())
HUMAN_N = set("person individual child kid baby toddler teenager chef cook player athlete worker artist presenter host "
              "speaker driver rider skier surfer dancer singer musician performer student teacher customer bartender "
              "mechanic user climber runner swimmer cyclist skateboarder boxer fighter wrestler gymnast barber "
              "stylist craftsman carpenter farmer doctor nurse officer soldier vendor demonstrator instructor "
              "magician juggler golfer pitcher batter goalkeeper diver hiker figure character".split())
IT_HEADS = set("camera video clip footage dog cat car vehicle bird horse animal robot machine drone truck "
               "puppy kitten".split())
TEMPORAL_NEXT = set("end beginning conclusion middle latter close start final last finish".split())
NOUN_ING = set("something nothing anything everything morning evening building ceiling clothing string thing king "
               "ring wing sibling lightning wedding pudding stuffing frosting icing seasoning dressing filling topping "
               "bedding ending beginning opening meeting offering setting during spring swing awning painting drawing "
               "earring railing ceiling siding landing parking packaging".split())
PLACES = set("room frame scene house building car stage area kitchen water pool office bathroom bedroom vehicle "
             "elevator restaurant garage ring track shop store field court hallway tent cave tunnel bus train "
             "classroom studio church arena cage gym".split())
UNITS = set("years months weeks days hours minutes seconds decades centuries percent dollars cents pounds euros inches "
            "feet foot meters metres miles kilometers yards degrees kilograms kg grams liters litres ounces "
            "tablespoons teaspoons cups ml mm cm stars episodes seasons levels ways types kinds reasons options "
            "aspects elements methods versions varieties categories stages phases parts sections segments scenes "
            "frames shots clips images screenshots pictures photos photographs stills panels things words letters "
            "lines points digits numbers pages chapters steps tips ideas questions styles colors "
            "colours shades items products features angles views perspectives moments instances examples".split())
PAIRED = set("hands eyes ears arms legs feet wings shoes sneakers boots socks gloves cheeks lips nostrils wheels halves "
             "sides ends thumbs knees shoulders elbows wrists ankles hips parents twins pair pairs earrings "
             "eyebrows lenses headlights".split())
IRREG_PL = set("people men women children teeth mice geese sheep fish deer cattle police".split())
NUMW = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
NUM_PREV_BAD = set("number no no. # player jersey page step level chapter episode part version season round lap "
                   "channel route gate room platform model iphone windows ps xbox galaxy out of the top".split())


def stems_re(stems):
    return re.compile(r"^(?:" + "|".join(sorted(stems, key=len, reverse=True)) + r")(?:s|es|ed|d|ing|ning|ping|ming|ting|ging|ling)?$")


MOTION_UD = stems_re("walk run ran climb go goes went gone move look glance gaze swim swam scroll pan tilt fly flies flew "
                     "float jump bounce slid slide rid ride rode drive drove roll hike march crawl travel head swipe point "
                     "glide ski skat sprint jog hop step swing bob rush hurri hurry race dash wander stroll paddl paddle "
                     "cycl cycle sail drift dive dove leap lept leapt sprint scurr scurry".split())
MOTION_LR = stems_re("turn move pan shift swipe slid slide lean look glance steer veer rotat rotate tilt swerv swerve "
                     "scroll step swing sway roll drift point gestur gesture head walk run ran driv drive".split())
MOTION_TW = stems_re("walk run ran move swim swam drive drove head ride rode fly flies flew turn face lean look point "
                     "swing roll step pull push throw threw kick glance crawl jog sprint dash rush race travel sail paddl "
                     "paddle gaze stare slid slide drift hurri hurry wander march skat ski cycl cycle bik bike jump leap "
                     "dive dove climb reach gestur gesture hop dart charg charge pan zoom tilt back float glid glide "
                     "approach mov".split())
MOTION_FB = stems_re("move lean step bend bent push pull tilt rock swing walk jump fall fell roll scroll lunge "
                     "stumble slid slide drive drove rid ride rode run ran crawl skat skate swim swam fly flew hop "
                     "shuffle shift lurch jerk thrust pedal paddle".split())
INTO_BAD = stems_re("turn transform chang change morph convert look peer zoom break broke burst divid divide split cut "
                    "slic slice chop merg merge fad fade blend translat translate form shap shape roll fold mold mould "
                    "crash bump tuck delve run ran bang slam evolv evolve develop grow grew dic dice shred tear tore "
                    "mash crumbl crumble knead weav weave twist spun spin made mak make carv carve craft built build "
                    "sculpt fashion get got gett".split())
INTO_BAD_NEXT = set("account consideration detail details action existence play perspective laughter tears smiles "
                    "pieces slices halves strips chunks cubes rounds balls shape shapes focus view frame sight "
                    "position place a an".split())
OUTOF_BAD_NEXT = set("focus breath control reach curiosity nowhere character bounds proportion style date season stock "
                     "order line place touch shape sync tune range business work luck love fear respect boredom "
                     "frustration excitement habit view sight frame the a an".split())
FILL_BAD = set("frame screen scene air sky space view background image picture room area shot composition entire "
               "whole".split())
APPEAR_NEXT = set(", . and again briefly suddenly behind momentarily completely gradually slowly quickly abruptly".split())
ADV_SKIP = re.compile(r"^[a-z]+ly$")


def forms(*rows):
    """rows of equal-length tuples (a-forms, b-forms) -> list of (a, b) phrase pairs."""
    out = []
    for a, b in rows:
        out.append((a, b))
    return out


# (a, b, condition-id); a and b are phrases, the occurrence of either is replaced by the other
DIR_PAIRS = []


def _add(cond, *pairs):
    for a, b in pairs:
        DIR_PAIRS.append((a, b, cond))


_add("phrase", ("from left to right", "from right to left"), ("left to right", "right to left"),
     ("from top to bottom", "from bottom to top"), ("top to bottom", "bottom to top"),
     ("front to back", "back to front"), ("clockwise", "counterclockwise"), ("clockwise", "counter-clockwise"),
     ("clockwise", "anticlockwise"))
_add("lr", ("left", "right"), ("left-hand", "right-hand"), ("left-side", "right-side"))
_add("ud", ("up", "down"), ("upward", "downward"), ("upwards", "downwards"))
_add("into", ("into", "out of"))
_add("tw", ("towards", "away from"), ("toward", "away from"))
_add("fb", ("forward", "backward"), ("forwards", "backwards"))
_add("obj", ("opens", "closes"), ("opened", "closed"), ("opening", "closing"),
     ("picks up", "puts down"), ("picking up", "putting down"),
     ("raises", "lowers"), ("raising", "lowering"), ("raised", "lowered"),
     ("lifts", "lowers"), ("lifting", "lowering"), ("lifted", "lowered"),
     ("pushes", "pulls"), ("pushing", "pulling"), ("pushed", "pulled"),
     ("ties", "unties"), ("tying", "untying"), ("tied", "untied"),
     ("locks", "unlocks"), ("locking", "unlocking"), ("locked", "unlocked"),
     ("wraps", "unwraps"), ("wrapping", "unwrapping"), ("wrapped", "unwrapped"),
     ("folds", "unfolds"), ("folding", "unfolding"), ("folded", "unfolded"),
     ("plugs", "unplugs"), ("plugging", "unplugging"), ("plugged", "unplugged"),
     ("zips", "unzips"), ("zipping", "unzipping"), ("zipped", "unzipped"),
     ("packs", "unpacks"), ("packing", "unpacking"), ("packed", "unpacked"),
     ("loads", "unloads"), ("loading", "unloading"), ("loaded", "unloaded"),
     ("screws", "unscrews"), ("screwing", "unscrewing"), ("screwed", "unscrewed"),
     ("buttons", "unbuttons"), ("buttoning", "unbuttoning"), ("buttoned", "unbuttoned"),
     ("attaches", "detaches"), ("attaching", "detaching"), ("attached", "detached"),
     ("connects", "disconnects"), ("connecting", "disconnecting"), ("connected", "disconnected"),
     ("inflates", "deflates"), ("inflating", "deflating"), ("inflated", "deflated"),
     ("assembles", "disassembles"), ("assembling", "disassembling"), ("assembled", "disassembled"),
     ("fastens", "unfastens"), ("fastening", "unfastening"), ("buckles", "unbuckles"))
_add("fill", ("fills", "empties"), ("filling", "emptying"))
_add("fillbase", ("fill", "empty"))
_add("objbase", ("open", "close"), ("pick up", "put down"), ("push", "pull"), ("raise", "lower"), ("lift", "lower"),
     ("tie", "untie"), ("lock", "unlock"), ("wrap", "unwrap"), ("fold", "unfold"),
     ("attach", "detach"), ("connect", "disconnect"))
_add("enter", ("enters", "exits"), ("entering", "exiting"), ("entered", "exited"))
_add("enterbase", ("enter", "exit"))
_add("leave", ("enters", "leaves"), ("entering", "leaving"))
_add("ving", ("starts", "stops"), ("starting", "stopping"), ("started", "stopped"))
_add("vingbase", ("start", "stop"))
_add("appear", ("appears", "disappears"), ("appearing", "disappearing"), ("appeared", "disappeared"))
_add("free", ("increases", "decreases"), ("increasing", "decreasing"), ("increased", "decreased"),
     ("increase", "decrease"), ("ascends", "descends"), ("ascending", "descending"), ("ascended", "descended"))
DIR_PATTERNS = []   # (phrase tokens, replacement, key, dir, cond)
for a, b, cond in DIR_PAIRS:
    key = f"{a}|{b}"
    DIR_PATTERNS.append((a.split(), b, key, 0, cond))
    DIR_PATTERNS.append((b.split(), a, key, 1, cond))
DIR_BY_FIRST = collections.defaultdict(list)
for p in DIR_PATTERNS:
    DIR_BY_FIRST[p[0][0]].append(p)
BASE_PREV = set("to and then , will can they you we".split())


def _stem_ok(rx, w):
    return bool(w) and bool(rx.match(w))


def dir_cond(cond, low, i, j):
    """low: lowercased token texts; phrase occupies [i, j)."""
    prev = low[i - 1] if i > 0 else ""
    prev2 = low[i - 2] if i > 1 else ""
    nxt = low[j] if j < len(low) else ""
    nxt2 = low[j + 1] if j + 1 < len(low) else ""
    if cond == "phrase":
        return True
    if cond == "lr":
        win = low[max(0, i - 4):i] + low[j:j + 4]
        if "left" in win or "right" in win:
            return False
        if low[i] in ("left-hand", "right-hand", "left-side", "right-side"):
            return True
        if prev in DET_POSS:
            return nxt in set("side sides hand hands arm arms leg legs foot shoulder corner edge of , . ; and while as "
                              "with lane wrist knee ear eye palm fist half part portion end ankle elbow cheek hip "
                              "thigh wing column panel bank".split())
        mot = prev if not ADV_SKIP.match(prev) else prev2
        return _stem_ok(MOTION_LR, mot) and nxt not in ("now", "away", "after", "before", "there", "here", "-", "into")
    if cond == "ud":
        mot = prev if not ADV_SKIP.match(prev) else prev2
        return _stem_ok(MOTION_UD, mot) and nxt not in ("to", "and", "-", "with", "of") and "and" not in (prev, nxt)
    if cond == "into":
        if any(_stem_ok(INTO_BAD, w) for w in low[max(0, i - 5):i]):
            return False
        if low[i] == "into":
            return nxt not in INTO_BAD_NEXT and nxt2 not in INTO_BAD_NEXT
        return nxt not in OUTOF_BAD_NEXT and nxt2 not in OUTOF_BAD_NEXT and not (prev in NUMW or prev.isdigit() or prev == "one")
    if cond == "tw":
        mot = prev if not ADV_SKIP.match(prev) else prev2
        return _stem_ok(MOTION_TW, mot) and nxt2 not in TEMPORAL_NEXT and prev not in ("far", "away")
    if cond == "fb":
        mot = prev if not ADV_SKIP.match(prev) else prev2
        return _stem_ok(MOTION_FB, mot) and nxt not in ("to", "-") and prev != "fast"
    near = low[max(0, i - 3):i]
    if any(w in ("video", "clip", "scene", "footage", "segment", "film", "episode", "sequence", "shot") for w in near):
        return False
    if cond == "obj":
        return nxt in OBJ and prev not in ("the", "a", "an")
    if cond in ("fill", "fillbase"):
        if nxt not in OBJ or prev in ("the", "a", "an") or (cond == "fillbase" and prev not in BASE_PREV):
            return False
        return not any(w in FILL_BAD for w in low[j:j + 4])
    if cond == "objbase":
        return nxt in OBJ and prev in BASE_PREV
    if cond == "enter":
        return (nxt in OBJ or nxt in ("from", "through", ".", ",")) and prev not in ("the", "a", "an")
    if cond == "enterbase":
        return (nxt in OBJ or nxt in ("from", "through")) and prev in BASE_PREV
    if cond == "leave":
        return nxt == "the" and nxt2 in PLACES and prev not in ("the", "a", "an")
    if cond in ("ving", "vingbase"):
        ok = nxt.endswith("ing") and len(nxt) > 5 and nxt not in NOUN_ING and prev not in ("the", "a", "an", "its", "his", "her")
        return ok and (cond == "ving" or prev in BASE_PREV)
    if cond == "appear":
        return nxt in APPEAR_NEXT and prev not in ("the", "a", "an")
    if cond == "free":
        return True
    raise KeyError(cond)


def direction_cands(sent):
    tk = toks(sent)
    low = [t[0].lower() for t in tk]
    out, used = [], set()
    for i, w in enumerate(low):
        for phrase, repl, key, d, cond in DIR_BY_FIRST.get(w, ()):
            j = i + len(phrase)
            if low[i:j] != phrase or any(k in used for k in range(i, j)):
                continue
            if not dir_cond(cond, low, i, j):
                continue
            new = replace_span(sent, tk[i][1], tk[j - 1][2], repl)
            prev = low[i - 1] if i > 0 else "^"
            nxt = low[j] if j < len(low) else "$"
            out.append(("direction", key, d, new, (" ".join(phrase), repl.lower(), f"{prev}\t{nxt}")))
            if cond == "phrase":
                used.update(range(i, j))
    return out


def count_cands(sent):
    tk = toks(sent)
    low = [t[0].lower() for t in tk]
    out = []
    for i, w in enumerate(low):
        if w in NUMW:
            n, word = NUMW.index(w) + 2, True
        elif w.isascii() and w.isdigit() and 2 <= int(w) <= 10 and not w.startswith("0"):
            n, word = int(w), False
        else:
            continue
        prev = low[i - 1] if i > 0 else ""
        if prev in NUM_PREV_BAD or prev in ("$", "#", "of", "-", "–", "/", "x", "×", "+", "=") or prev[:1].isdigit() or prev in NUMW:
            continue
        if i + 1 < len(low) and low[i + 1] in ("-", "/", ":", ".", ",", "%", "of", "or", "to", "and"):
            continue
        # counted plural noun right after, or after one adjective
        k = None
        for jj in (i + 1, i + 2):
            if jj >= len(low):
                break
            x = low[jj]
            if not re.fullmatch(r"[a-z]+", x):
                break
            if x in UNITS or x in PAIRED:
                break
            if x in IRREG_PL or (x.endswith("s") and not x.endswith(("ss", "us", "is", "ous")) and len(x) > 3):
                k = jj
                break
            if jj == i + 1 and x not in ("more", "other", "different", "of", "and", "or", "to"):
                continue
            break
        if k is None:
            continue
        for m in (n - 1, n + 1):
            if not 2 <= m <= 10:
                continue
            rep = NUMW[m - 2] if word else str(m)
            new = replace_span(sent, tk[i][1], tk[i][2], rep)
            a, b = sorted([w, rep.lower()], key=lambda x: NUMW.index(x) if word else int(x))
            out.append(("count", f"{a}|{b}", 0 if w == a else 1, new, (w, rep.lower(), low[k])))
    return out


# ----------------------------------------------------------------------------------------------- order edits
CONN = re.compile(r"^(Then|Next|After that|Afterwards|Afterward|Later|Later on|Subsequently|Following this|"
                  r"Following that|Eventually|Soon after|Shortly after|Shortly afterwards|Finally)(,)?\s+")
LEX = {}     # filled by init_lex: verbs (set), s2ing, ing2s


def vclass(w):
    return "s" if w.endswith("s") else ("ed" if w.endswith("ed") else "o")


def parse_clause(s, start_char=0):
    """[connective] Subject [then] VP <final punct>.  Returns dict or None."""
    body = s[start_char:]
    m = CONN.match(body)
    off = start_char + (m.end() if m else 0)
    tk = [t for t in toks(s) if t[1] >= off]
    if len(tk) < 3:
        return None
    low = [t[0].lower() for t in tk]
    verbs = LEX["verbs"]
    if low[0] in SUBJ_PRON:
        k = 1
    elif low[0] in ("the", "a", "an"):
        k = None
        for j in range(1, min(8, len(low))):
            w = low[j]
            if j >= 2 and w in verbs and not low[j - 1].endswith("'s"):
                k = j
                break
            if not re.fullmatch(r"[a-z][a-z'-]*", w) or w in ("and", "or", "who", "which", "that", "while", "as",
                                                               "then", "is", "are", "was", "were", "has", "have"):
                return None
        if k is None:
            return None
    else:
        return None
    subj = " ".join(low[:k])
    head = subj.split()[-1]
    has_then = False
    if k < len(low) and low[k] == "then":
        has_then, k = True, k + 1
    if k >= len(low) or low[k] not in verbs:
        return None
    if not s.rstrip().endswith((".", "!")):
        return None
    vp_lo, vp_hi = tk[k][1], len(s.rstrip()) - 1
    return dict(conn=bool(m), subj=subj, head=head, then=has_then, vlo=vp_lo, vhi=vp_hi, verb=low[k],
                vp=s[vp_lo:vp_hi])


def clean_piece(p, maxw=20, allow_pron=False):
    w = [x[0].lower() for x in toks(p)]
    if not 1 <= len(w) <= maxw:
        return False
    bad = PRESUP | (set() if allow_pron else ANAPH)
    if any(x in bad for x in w):
        return False
    return True


def coref(a, b):
    """subject of the later sentence (b) refers to subject a."""
    if a == b:
        return True
    ha = a.split()[-1]
    if b == "he":
        return ha in HUMAN_M or ha in HUMAN_N
    if b == "she":
        return ha in HUMAN_F or ha in HUMAN_N
    if b == "it":
        return ha in IT_HEADS
    return False


def order_pair_sents(s1, s2):
    c1, c2 = parse_clause(s1), parse_clause(s2)
    if not c1 or not c2:
        return None
    if not (c2["conn"] or c2["then"]):
        return None
    if c1["verb"] in FRAME_HEADS or c2["verb"] in FRAME_HEADS or vclass(c1["verb"]) != vclass(c2["verb"]):
        return None
    if not coref(c1["subj"], c2["subj"]):
        return None
    if c1["subj"] in ("he", "she", "it", "they") and c2["subj"] != c1["subj"]:
        return None
    if not (clean_piece(c1["vp"]) and clean_piece(c2["vp"])):
        return None
    if c1["vp"].lower() == c2["vp"].lower():
        return None
    n1 = s1[:c1["vlo"]] + c2["vp"] + s1[c1["vhi"]:]
    n2 = s2[:c2["vlo"]] + c1["vp"] + s2[c2["vhi"]:]
    return n1, n2


SEP = re.compile(r"(, and then |, then | and then )")


def order_then_inside(s):
    m = SEP.search(s)
    if not m:
        return None
    c = parse_clause(s)
    if not c or c["vlo"] >= m.start():
        return None
    vp1 = s[c["vlo"]:m.start()]
    end = len(s.rstrip()) - 1
    vp2 = s[m.end():end]
    w2 = toks(vp2)
    if not w2 or w2[0][0].lower() not in LEX["verbs"]:
        return None
    if vclass(w2[0][0].lower()) != vclass(c["verb"]) or c["verb"] in FRAME_HEADS or w2[0][0].lower() in FRAME_HEADS:
        return None
    if "," in vp1 or not (clean_piece(vp1) and clean_piece(vp2)) or vp1.lower() == vp2.lower():
        return None
    return s[:c["vlo"]] + vp2 + m.group(1) + vp1 + s[end:]


NP_TRIG = set("with shows showing features featuring displays displaying depicts depicting captures capturing "
              "presents presenting includes including sees".split())
NP_DET = set("a an the some several two three four five".split())


def order_followed_by(s):
    m = re.search(r", followed by ", s)
    if not m:
        return None
    left = toks(s[:m.start()])
    j = len(left) - 1
    while j >= 0 and left[j][0].lower() not in NP_TRIG:
        if not re.fullmatch(r"[A-Za-z][A-Za-z'-]*", left[j][0]):
            return None
        j -= 1
    lo = left[j + 1][1] if j + 1 < len(left) else None
    if lo is None or left[j + 1][0].lower() not in NP_DET or len(left) - 1 - j > 12:
        return None
    if j < 0 and lo != 0:
        return None
    np1 = s[lo:m.start()]
    rest = s[m.end():]
    mm = re.search(r",|[.!?]\s*$", rest)
    if not mm:
        return None
    np2 = rest[:mm.start()]
    if re.match(r",\s*(which|who|whom|whose|where|that)\b", rest[mm.start():]):
        return None
    t2 = toks(np2)
    if not t2 or t2[0][0].lower() not in NP_DET or len(t2) > 12 or any(not re.fullmatch(r"[A-Za-z][A-Za-z'-]*", t[0]) for t in t2):
        return None
    if not (clean_piece(np1, 12) and clean_piece(np2, 12)) or np1.lower() == np2.lower():
        return None
    if any(t[0].lower() in LEX["verbs"] for t in toks(np1) + t2) or not given_new_ok(np1, np2):
        return None
    if det_class(np1) != det_class(np2):
        return None
    if lo == 0:      # NP1 is sentence-initial: move the capital letter, not the words' case
        a, b = np2[:1].upper() + np2[1:], np1[:1].lower() + np1[1:]
    else:
        a, b = np2, np1
    return s[:lo] + a + ", followed by " + b + rest[mm.start():]


def to_ing(vp):
    t = toks(vp)
    w = t[0][0]
    g = LEX["s2ing"].get(w.lower())
    return None if g is None else g + vp[t[0][2]:]


def to_s(vp):
    t = toks(vp)
    w = t[0][0]
    g = LEX["ing2s"].get(w.lower())
    return None if g is None else g + vp[t[0][2]:]


def gerund_ok(p):
    return "," not in p and " and " not in p and clean_piece(p, 15)


def order_before_after(s):
    end = len(s.rstrip()) - 1
    if end < 0 or s[end] not in ".!":
        return None
    # "After VPing1, S VP2."  /  "Before VPing1, S VP2."
    m = re.match(r"(After|Before) ([a-z]+ing\b[^,]*), ", s)
    if m:
        c = parse_clause(s, m.end())
        if not c or c["then"] or c["verb"] in FRAME_HEADS:
            return None
        g1, vp2 = m.group(2), s[c["vlo"]:end]
        if not (gerund_ok(g1) and gerund_ok(vp2)):
            return None
        a, b = to_ing(vp2), to_s(g1)
        if not a or not b:
            return None
        return s[:m.start(2)] + a + s[m.end(2):c["vlo"]] + b + s[end:]
    # "S VP1 before/after VPing2."
    c = parse_clause(s)
    if not c or c["verb"] in FRAME_HEADS:
        return None
    mm = re.search(r" (before|after) ([a-z]+ing\b)", s[c["vlo"]:end])
    if not mm:
        return None
    blo = c["vlo"] + mm.start()
    vp1 = s[c["vlo"]:blo]
    comma = vp1.endswith(",")
    vp1 = vp1.rstrip(",")
    g2 = s[c["vlo"] + mm.start(2):end]
    if not (gerund_ok(vp1) and gerund_ok(g2)):
        return None
    a, b = to_s(g2), to_ing(vp1)
    if not a or not b:
        return None
    return s[:c["vlo"]] + a + ("," if comma else "") + " " + mm.group(1) + " " + b + s[end:]


PRON_ALL = ANAPH | set("he she his her hers him they their theirs we our you your".split())
POSITIONAL = set("following subsequent subsequently preceding initial initially beginning end ending start starting "
                 "opening closing concluding conclusion meanwhile now finished complete completed ready done".split())
DET_LOWER = set("the a an some several two three four five six seven eight nine ten each both many various multiple "
                "two".split())
META_START = re.compile(r"^(The|This|These|A|An)? ?(video|clip|footage|sequence|scene|scenes|frame|frames|image|images|"
                        r"shot|shots|segment|camera|view|screen|text|title|caption|narrative|content|visuals?)\b", re.I)
EVENT_PASSIVE = re.compile(r"\b(?:is|are|gets|get) (?:then |now |carefully |gently |slowly |quickly )?[a-z]+(?:ed|en|wn|ut|ung)\b")
INNER_THEN = re.compile(r"\b(is|are|was|were|gets|get) then ")


STATIVE = set("is are has have appears appear seems seem remains remain stands stand sits sit lies lie features feature "
              "shows show displays display contains contain includes include wears wear looks look varies vary belongs "
              "consists depicts depict presents present showcases showcase highlights highlight captures capture "
              "reveals reveal provides provide offers offer illustrates illustrate becomes become gets get "
              "shown seen displayed depicted featured captured presented located positioned arranged attached dressed "
              "surrounded covered made decorated adorned set framed illuminated lit highlighted labeled labelled "
              "written printed situated accompanied characterized filled equipped mounted organized organised "
              "scattered visible focused centered centred".split())


def event_body(b, maxw=30):
    t = [x[0].lower() for x in toks(b)]
    if not 3 <= len(t) <= maxw:
        return False
    if any(x in PRESUP or x in PRON_ALL or x in POSITIONAL for x in t if x != "then"):
        return False
    if META_START.match(b) or b.lower().startswith(("there ", "in ", "on ", "at ", "as ", "with ", "while ")):
        return False
    if any(x in LEX["verbs"] and x not in STATIVE for x in t):
        return True
    m = EVENT_PASSIVE.search(b)
    return bool(m) and m.group().split()[-1] not in STATIVE


NEW_STOP = set("the a an of in on with and to at by for from one other some small large big white black red blue "
               "green yellow brown gray grey dark light".split())


def det_class(piece):
    w = piece.split(" ", 1)[0].lower()
    return "def" if w in ("the", "this", "these", "that", "those") else (
        "indef" if w in ("a", "an", "some", "several", "two", "three", "four", "five", "various", "multiple") else "other")


def given_new_ok(earlier, later):
    """False if the later piece introduces with 'a X' what the earlier piece already calls 'the X' (or vice versa
    after the swap): moving them would invert the given/new article order and give the order away."""
    def after(text, dets):
        t = [x[0].lower() for x in toks(text)]
        return {t[k] for i, w in enumerate(t) if w in dets for k in range(i + 1, min(len(t), i + 4))} - NEW_STOP
    return not (after(earlier, ("a", "an")) & after(later, ("the",)))


def place_body(body, initial):
    """body moved into a slot: sentence-initial (capitalise) or after a connective (lowercase a determiner)."""
    if initial:
        return body[:1].upper() + body[1:]
    first = body.split(" ", 1)[0]
    if first.lower() in DET_LOWER:
        return first.lower() + body[len(first):]
    if first[:1].islower() or first == "I" or first.isupper() or first[:1].isdigit():
        return body
    return None      # capitalised word: proper noun or not cannot be decided safely


def order_sent_swap(s1, s2):
    """[C1] B1. [C2] B2.  ->  [C1] B2. [C2] B1.  (connectives stay; an inner 'is then' stays in the 2nd slot)"""
    m1, m2 = CONN.match(s1), CONN.match(s2)
    inner2 = INNER_THEN.search(s2)
    if not m2 and not inner2:
        return None
    if INNER_THEN.search(s1) or " then " in (s1[m1.end():] if m1 else s1):
        return None
    e1, e2 = len(s1.rstrip()) - 1, len(s2.rstrip()) - 1
    if s1[e1:] not in (".", "!") or s2[e2:] not in (".", "!"):
        return None
    b1 = s1[m1.end() if m1 else 0:e1]
    b2 = s2[m2.end() if m2 else 0:e2]
    if not m2:
        # move the inner 'then' from B2 into B1 (after its first is/are) so the 2nd slot keeps it
        b2n = INNER_THEN.sub(lambda m: m.group(1) + " ", b2, count=1)
        mm = re.search(r"\b(is|are|was|were|gets|get) ", b1)
        if not mm:
            return None
        b1n = b1[:mm.end()] + "then " + b1[mm.end():]
        b1, b2 = b1n, b2n
    if " then " in b2 and m2:
        return None
    if not (event_body(b1) and event_body(b2)) or b1.lower() == b2.lower() or not given_new_ok(b1, b2):
        return None
    if det_class(b1) != det_class(b2):
        return None
    nb2 = place_body(b2, m1 is None)
    nb1 = place_body(b1, m2 is None)
    if nb1 is None or nb2 is None:
        return None
    n1 = (s1[:m1.end()] if m1 else "") + nb2 + s1[e1:]
    n2 = (s2[:m2.end()] if m2 else "") + nb1 + s2[e2:]
    return n1, n2


SLOT = re.compile(r"\b(?:(?:begins|starts|opens|commences|ends|concludes|finishes|culminates) with|"
                  r"(?:shows?|showing|features?|featuring|displays?|displaying|depicts?|depicting|presents?|presenting|"
                  r"captures?|capturing|reveals?|revealing|showcases?|showcasing|introduces|highlights|focuses on|"
                  r"zooms in on|transitions to|shifts to|cuts to|moves to|switches to|changes to)|followed by) ")
SLOT_PRE = re.compile(r"^(?:(?:Then|Next|Finally|Later|Subsequently|Afterwards|Afterward|Following this|After that|"
                      r"Eventually|Initially|First|Lastly|Soon after),? )?(?:[Tt]he |[Aa]n? |[Tt]his |[Tt]hese |"
                      r"[Aa]nother |[Ss]everal |[Ss]ubsequent |[Ff]ollowing |[Nn]ext |[Ll]ater |[Ff]inal |[Ll]ast |"
                      r"[Ff]irst |[Ss]econd |[Tt]hird |[Ii]nitial |[Oo]pening |[Cc]losing |[Ss]ubsequently |[Ff]inally )*"
                      r"(?:video|clip|footage|sequence|scene|scenes|frame|frames|image|images|shot|shots|segment|segments|"
                      r"camera|view|It|it|Video|Clip|Footage|Sequence|Scene|Scenes|Frame|Frames|Image|Images|Shot|Shots|"
                      r"Segment|Segments|Camera)(?: then| later| next| finally| subsequently| now)? $")
ORDER_WORDS = re.compile(r"\b(begins|starts|opens|commences|ends|concludes|finishes|culminates|then|next|subsequent|"
                         r"subsequently|following|followed|later|final|finally|last|first|initially|afterwards|"
                         r"progresses|after|before|transitions|cuts|shifts|switches|changes|moves)\b", re.I)
META_WORD = re.compile(r"\b(video|clip|footage|sequence|scene|scenes|frame|frames|image|images|shot|shots|segment|"
                       r"camera|view|it|then|next|later|finally|subsequent|following|followed)\b", re.I)


def np_slot(s):
    """(lo, hi) of a scene-content NP that runs from a presentational trigger to the end of the sentence."""
    m = SLOT.search(s)
    if m is None:
        return None
    pre = s[:m.start()]
    if not (SLOT_PRE.match(pre) or (pre == "This is " and m.group().startswith("followed"))):
        return None
    if not (META_WORD.search(pre) or m.group().startswith("followed")):
        return None
    hi = len(s.rstrip()) - 1
    if s[hi:] != ".":
        return None
    np_ = s[m.end():hi]
    if re.search(r"[\"“”:;!?()]", np_):
        return None
    t = toks(np_)
    if not t or t[0][0].lower() not in NP_DET or not 2 <= len(t) <= 25:
        return None
    if any(x[0].lower() in PRESUP or x[0].lower() in ANAPH or x[0].lower() in POSITIONAL for x in t):
        return None
    if any(x[0].lower() in ("he", "she", "they", "him", "them") for x in t):
        return None
    return m.end(), hi, bool(ORDER_WORDS.search(pre + m.group()))


def order_np_slots(sents):
    slots = {i: np_slot(s) for i, s in enumerate(sents)}
    out = []
    for i in range(len(sents)):
        if not slots[i]:
            continue
        for j in range(i + 1, min(len(sents), i + 3)):
            if not slots[j]:
                continue
            a, b = slots[i], slots[j]
            if not (a[2] and b[2]):
                continue
            np1, np2 = sents[i][a[0]:a[1]], sents[j][b[0]:b[1]]
            if np1.lower() == np2.lower() or not given_new_ok(" ".join(sents[i:j]), sents[j]):
                continue
            if det_class(np1) != det_class(np2):
                continue
            n1 = sents[i][:a[0]] + np2 + sents[i][a[1]:]
            n2 = sents[j][:b[0]] + np1 + sents[j][b[1]:]
            out.append({i: n1, j: n2})
            break
    return out


CHANGE = stems_re("chang change transition shift transform turn go goes went fade switch evolv evolve progress morph "
                  "mov move walk run ran driv drive drove travel fly flew swim swam ride rode carri carry jump cycl "
                  "cycle pan glid glide slid slide fall fell grow grew alternat alternate").pattern
CHANGE_RE = re.compile(CHANGE)
FT_STOP = set(", . ; : as while when with in on at and before after during over for then where which who that by "
              "into onto through until within under".split())
FT_BAD = set("left right top bottom side one another time place end beginning start finish front back".split())


def order_from_to(s):
    tk = toks(s)
    low = [t[0].lower() for t in tk]
    for i, w in enumerate(low):
        if w != "from" or not any(CHANGE_RE.match(x) for x in low[max(0, i - 3):i]):
            continue
        if any(x in ("range", "ranging", "ranges", "vary", "varies", "varying") for x in low[max(0, i - 3):i]):
            continue
        k = next((k for k in range(i + 2, min(len(low), i + 7)) if low[k] == "to"), None)
        if k is None or any(not re.fullmatch(r"[a-z][a-z'-]*", x) for x in low[i + 1:k]):
            continue
        e = k + 1
        while e < len(low) and e - k - 1 < 5 and low[e] not in FT_STOP and re.fullmatch(r"[a-z][a-z'-]*", low[e]):
            e += 1
        if e == k + 1 or e - k - 1 > 4 or (e < len(low) and low[e] not in FT_STOP):
            continue
        np1, np2 = low[i + 1:k], low[k + 1:e]
        if np1 == np2 or any(x in FT_BAD or x in ANAPH or x in PRESUP for x in np1 + np2):
            continue
        if any(x in LEX["verbs"] for x in np1 + np2):
            continue
        a_lo, a_hi = tk[i + 1][1], tk[k - 1][2]
        b_lo, b_hi = tk[k + 1][1], tk[e - 1][2]
        return s[:a_lo] + s[b_lo:b_hi] + s[a_hi:b_lo] + s[a_lo:a_hi] + s[b_hi:]
    return None


def order_cands(sents):
    out = []
    for i, s in enumerate(sents):
        for kind, fn in (("then_in", order_then_inside), ("followed_by", order_followed_by),
                         ("before_after", order_before_after), ("from_to", order_from_to)):
            n = fn(s)
            if n and n != s:
                out.append(("order", kind, 0, {i: n}, None))
        if i + 1 < len(sents):
            r = order_pair_sents(s, sents[i + 1])
            if r:
                out.append(("order", "sent_pair", 0, {i: r[0], i + 1: r[1]}, None))
            else:
                r = order_sent_swap(s, sents[i + 1])
                if r:
                    out.append(("order", "sent_swap", 0, {i: r[0], i + 1: r[1]}, None))
    for e in order_np_slots(sents):
        out.append(("order", "np_slots", 0, e, None))
    return out


# ----------------------------------------------------------------------------------------------- lexicon
def build_lex(texts):
    txt = "\n".join(texts)
    vocab = collections.Counter(re.findall(r"\b[a-z]+\b", txt))
    after_then = collections.Counter(re.findall(r"\bthen ([a-z]+)\b", txt))
    after_pron = collections.Counter(re.findall(r"\b(?:he|she) ([a-z]+s)\b", txt))
    stop = set("the a an his her their its this that these those he she they it we you i there to in into on at of "
               "from with by for as is was are were be been being has have had also still always perhaps sometimes "
               "afterwards towards across carefully gently quickly slowly gradually fully again back once finally "
               "immediately briefly suddenly seen shown placed added used displayed introduced transferred "
               "standing moving walking placing transitioning returning turning".split())
    verbs = {w for w, c in after_then.items() if c >= 10 and w not in stop and not w.endswith("ly")}
    verbs |= {w for w, c in after_pron.items() if c >= 20 and w not in stop and not w.endswith("ly")}
    s2ing = {}
    for v in sorted(verbs):
        if not v.endswith("s") or v.endswith("ss") or v in ("is", "has", "does", "was"):
            continue
        bases = {v[:-1]}
        if v.endswith("ies"):
            bases.add(v[:-3] + "y")
        if v.endswith("es"):
            bases.add(v[:-2])
        cands = []
        for b in bases:
            cands += [b + "ing"]
            if b.endswith("e") and not b.endswith(("ee", "ye", "oe")):
                cands.append(b[:-1] + "ing")
            if b.endswith("ie"):
                cands.append(b[:-2] + "ying")
            if len(b) >= 3 and b[-1] not in "aeiouwxy" and b[-2] in "aeiou" and b[-3] not in "aeiou":
                cands.append(b + b[-1] + "ing")
        cands = [c for c in cands if vocab.get(c, 0) >= 5]
        if cands:
            s2ing[v] = max(cands, key=lambda c: vocab[c])
    ing2s = {}
    for v, g in s2ing.items():
        if g not in ing2s or vocab[v] > vocab[ing2s[g]]:
            ing2s[g] = v
    return {"verbs": verbs, "s2ing": s2ing, "ing2s": ing2s}


def init_worker(lex):
    LEX.update(lex)


# ----------------------------------------------------------------------------------------------- per video
def h32(s, salt=""):
    return int(hashlib.md5((salt + s).encode()).hexdigest()[:8], 16)


START_BAD = re.compile(r"^(Then|Next|After|Afterwards|Later|Finally|Subsequently|Following|He|She|They|It|This|These|"
                       r"His|Her|Their|Its|The same|Meanwhile|Eventually|Also|Additionally|However)\b")


def excerpt(sents, lo, hi, rng):
    """window [a, b) of 2-4 sentences containing [lo, hi], <= MAXW words; prefers a clean first sentence."""
    nw = [nwords(s) for s in sents]
    wins = []
    for L in (2, 3, 4):
        for a in range(hi - L + 1, lo + 1):
            b = a + L
            if a < 0 or b > len(sents) or b - 1 < hi:
                continue
            if sum(nw[a:b]) <= MAXW:
                wins.append((a, b))
    if not wins:
        if hi == lo and nw[lo] <= MAXW:
            return lo, lo + 1
        return None
    good = [w for w in wins if w[0] == 0 or not START_BAD.match(sents[w[0]])]
    pool = good or wins
    return pool[rng.integers(len(pool))]


def process(args):
    video, caption = args
    sents = split_sents(caption)
    rng = np.random.default_rng(h32(video, "capc"))
    raw = order_cands(sents)
    for i, s in enumerate(sents):
        for typ, key, d, new, meta in direction_cands(s) + count_cands(s):
            raw.append((typ, key, d, {i: new}, meta))
    # one candidate per (type, stratum, dir)
    by = collections.defaultdict(list)
    for c in raw:
        by[(c[0], c[1], c[2], c[4][2] if c[4] else "")].append(c)
    out = []
    for (typ, key, d, ctx), cs in by.items():
        typ, key, d, edits, meta = cs[rng.integers(len(cs))]
        lo, hi = min(edits), max(edits)
        w = excerpt(sents, lo, hi, rng)
        if w is None:
            continue
        true = " ".join(sents[w[0]:w[1]])
        neg = " ".join(edits.get(k, sents[k]) for k in range(w[0], w[1]))
        if re.sub(r"\W+", " ", true.lower()).strip() == re.sub(r"\W+", " ", neg.lower()).strip():
            continue
        if meta and not (n_phrase(true, meta[0]) == 1 and n_phrase(true, meta[1]) == 0):
            continue      # the edited word must be the only one of its pair in the excerpt (no 'left ... left')
        if typ == "order":
            # balance unit for order: the first word where the two options diverge (e.g. 'view' vs 'close-up',
            # 'adjusts' vs 'gestures', 'day' vs 'night'); dir = which of the two the TRUE caption has first
            tw = [x[0].lower() for x in toks(true)]
            nw = [x[0].lower() for x in toks(neg)]
            i = next((i for i, (a, b) in enumerate(zip(tw, nw)) if a != b), None)
            if i is None:
                continue
            a, b = sorted((tw[i], nw[i]))
            d, ctx = (0 if tw[i] == a else 1), f"{a}\t{b}"
        out.append((typ, key, d, true, neg, ctx))
    return video, out


def n_phrase(text, phrase):
    return len(re.findall(r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])", text.lower()))


# ----------------------------------------------------------------------------------------------- selection
def stratum(c, strat):
    """balance unit: surface pair + context word(s), so that neither the pair nor its neighbours tell the true side."""
    typ, key, ctx = c[0], c[1], c[5]
    if typ == "order":
        return "order:" + ctx.replace("\t", "|")
    if strat == "key":
        return key
    if typ == "count":
        return f"{key}||{ctx}"
    prev, nxt = ctx.split("\t")
    return f"{key}||{ {'prev': prev, 'next': nxt, 'prevnext': prev + '|' + nxt}[strat] }"


def select(cands, target, seed, strat="prevnext"):
    """cands: dict video -> list of (typ, key, dir, true, neg, ctx). Returns list of (video, typ, key, dir, true, neg, ctx)."""
    rng = np.random.default_rng(seed)
    videos = list(cands)
    rng.shuffle(videos)
    used, chosen = set(), []
    tgt = {t: int(round(MIX[t] * target)) for t in MIX}
    # order first: it is the scarcest type
    for typ in ("order", "count", "direction"):
        # supply per (key, dir) among unused videos
        sup = collections.defaultdict(list)
        for v in videos:
            if v in used:
                continue
            for c in cands[v]:
                if c[0] == typ:
                    sup[(stratum(c, strat), c[2])].append(v)
        keys = sorted({k for k, _ in sup}, key=lambda k: min(len(sup[(k, 0)]), len(sup[(k, 1)])))
        cap = max(1, int(MAX_KEY_SHARE * tgt[typ]) // 2)      # per side, for one surface pair (all strata)
        if typ == "order":
            cap = max(1, int(0.02 * tgt[typ]))
        base_n = collections.Counter()
        picked = []
        for k in keys:
            got = {0: [], 1: []}
            base = k.split("||")[0]
            # take the minority side first so the majority side has to match it
            d_order = sorted((0, 1), key=lambda d: len(sup[(k, d)]))
            limit = max(0, cap - base_n[base])
            for d in d_order:
                for v in sup[(k, d)]:
                    if len(got[d]) >= limit:
                        break
                    if v in used:
                        continue
                    got[d].append(v)
                    used.add(v)
                if d == d_order[0]:
                    limit = len(got[d])
            n = min(len(got[0]), len(got[1]))
            for d in (0, 1):
                for v in got[d][n:]:
                    used.discard(v)
                got[d] = got[d][:n]
            base_n[base] += n
            picked.append((k, got))
        total = sum(2 * len(g[0]) for _, g in picked)
        # trim to the type target by dropping matched pairs proportionally
        keep_frac = min(1.0, tgt[typ] / max(1, total))
        for k, got in picked:
            n = len(got[0])
            m = int(round(n * keep_frac))
            for d in (0, 1):
                for v in got[d][m:]:
                    used.discard(v)
                for v in got[d][:m]:
                    c = next(c for c in cands[v] if c[0] == typ and stratum(c, strat) == k and c[2] == d)
                    chosen.append((v,) + c)
    return chosen


def to_rows(chosen, prep, seed):
    rng = np.random.default_rng(seed)
    n = len(chosen)
    ans = np.array([0] * (n // 2) + [1] * (n - n // 2))
    rng.shuffle(ans)
    rows = []
    for (v, typ, key, d, true, neg, ctx), a in zip(chosen, ans):
        opts = [true, neg] if a == 0 else [neg, true]
        q = QUESTIONS[rng.integers(len(QUESTIONS))]
        text = f"{VIS}\n{q}\nA. {opts[0]}\nB. {opts[1]}\n{SUFFIX}"
        conv = [{"role": "user", "content": text}, {"role": "assistant", "content": "AB"[a] + "."}]
        rows.append({"video": v, "prep_exists": bool(prep[v]), "kind": typ, "edit_key": key,
                     "stratum": stratum((typ, key, d, true, neg, ctx), "prevnext"),
                     "conversations": json.dumps(conv, ensure_ascii=False)})
    return pd.DataFrame(rows)


def load_source(name):
    d = pd.read_parquet(f"{R}/manifests/stage1/{name}.parquet", columns=["video", "prep_exists", "kind", "conversations"])
    d = d[d.kind.isin(SOURCES[name]["kinds"])]
    d = d.drop_duplicates("video").reset_index(drop=True)
    caps = []
    for c in d.conversations:
        m = json.loads(c)
        caps.append(next((x["content"] for x in m if x["role"] == "assistant"), ""))
    d["caption"] = caps
    return d.drop(columns=["conversations"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", default=list(SOURCES))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_dir", default=f"{R}/manifests/stage1")
    ap.add_argument("--meta_out", default="", help="dir for an analysis sidecar (video, kind, edit_key, stratum)")
    args = ap.parse_args()
    data = {n: load_source(n) for n in args.src}
    lex_txt = []
    for n, d in data.items():
        lex_txt += d.caption.sample(min(len(d), 150000), random_state=0).tolist()
    lex = build_lex(lex_txt)
    print(f"lexicon: {len(lex['verbs'])} verbs, {len(lex['s2ing'])} 3sg->gerund", flush=True)
    for n, d in data.items():
        if args.limit:
            d = d.sample(min(args.limit, len(d)), random_state=0).reset_index(drop=True)
        prep = dict(zip(d.video, d.prep_exists))
        with Pool(args.workers, initializer=init_worker, initargs=(lex,)) as pool:
            res = pool.map(process, zip(d.video, d.caption), chunksize=500)
        cands = {v: c for v, c in res if c}
        avail = collections.Counter()
        for v, cs in cands.items():
            for t in {c[0] for c in cs}:
                avail[t] += 1
        target = SOURCES[n]["target"] if not args.limit else int(SOURCES[n]["target"] * args.limit / 1e6) or 1000
        chosen = select(cands, target, seed=h32(n), strat="prevnext")
        df = to_rows(chosen, prep, seed=h32(n, "rows"))
        split = df.video.map(lambda v: int(hashlib.md5(v.encode()).hexdigest()[8:16], 16) % 100 < 2)
        cols = ["video", "prep_exists", "kind", "conversations"]
        if args.meta_out:
            df.assign(heldout=split.values)[["video", "kind", "edit_key", "stratum", "heldout"]].to_parquet(
                f"{args.meta_out}/capcontrast_{n}_meta.parquet", index=False)
        out = f"{args.out_dir}/capcontrast_{n}.parquet"
        df[~split][cols].reset_index(drop=True).to_parquet(out, index=False)
        df[split][cols].reset_index(drop=True).to_parquet(out.replace(".parquet", "_heldout.parquet"), index=False)
        ans = df.conversations.map(lambda c: json.loads(c)[1]["content"])
        print(f"{n}: videos {len(d)} | with candidates {len(cands)} | videos with a candidate per type {dict(avail)}")
        print(f"{n}: rows {len(df)} (train {int((~split).sum())}, heldout {int(split.sum())}) | kinds "
              f"{df.kind.value_counts().to_dict()} | answers {ans.value_counts().to_dict()} -> {out}", flush=True)


if __name__ == "__main__":
    main()
