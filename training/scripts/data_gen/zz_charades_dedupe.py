"""(10-23, user: STAR train approved, but never the evaluation videos themselves) which Charades videos appear in ANY evaluation set, by
content: a 64-bit difference hash of frames. Charades: every video at 4 fps; evaluation: every video of manifests/eval/**/*.jsonl at 5
points. An evaluation video is matched to a Charades video when >= 2 of its 5 frames are within 6 bits of that video's frames (4 x 16-bit
chunks: candidates share a chunk, so recall is complete up to 3 bits). Union with the ids visible in file names. -> logs/zz_charades_eval_ids_content.json
usage: zz_charades_dedupe.py"""
import glob, json, os, collections, sys
from multiprocessing import Pool
import numpy as np
R = "/data/llama_vjepa_lm"; CH = f"{R}/bench/CharadesSTA/videos"


def dhash(img):                                    # img: uint8 gray [H, W] -> 64-bit int
    from PIL import Image
    a = np.asarray(Image.fromarray(img).resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def charades(p):
    import av
    out = []
    try:
        with av.open(p) as c:
            s = c.streams.video[0]; s.thread_type = "AUTO"; nxt = 0.0
            for fr in c.decode(s):
                t = float(fr.time or 0)
                if t + 1e-6 >= nxt:
                    out.append(dhash(fr.to_ndarray(format="gray"))); nxt = t + 0.25
    except Exception:
        pass
    return os.path.basename(p)[:-4], out


def evalv(p):
    import av
    out = []
    try:
        with av.open(p) as c:
            s = c.streams.video[0]; dur = float(c.duration / 1e6) if c.duration else float(s.duration * s.time_base)
            for q in (0.1, 0.3, 0.5, 0.7, 0.9):
                c.seek(int(q * dur / s.time_base), stream=s, backward=True)
                for fr in c.decode(s):
                    if float(fr.time or 0) >= q * dur - 0.05:
                        out.append(dhash(fr.to_ndarray(format="gray"))); break
    except Exception:
        pass
    return p, out


if __name__ == "__main__":
    vids = sorted(glob.glob(f"{CH}/**/*.mp4", recursive=True))
    ev = set()
    for f in glob.glob(f"{R}/manifests/eval/**/*.jsonl", recursive=True):
        for l in open(f):
            v = json.loads(l).get("video")
            if isinstance(v, str) and v.endswith((".mp4", ".webm", ".mkv", ".avi", ".mov")) and os.path.exists(v):
                ev.add(v)
    ev = sorted(v for v in ev if not v.startswith(CH))     # Charades-STA items already are Charades by name
    print("charades", len(vids), "eval videos", len(ev), flush=True)
    cache = f"{R}/logs/zz_charades_hashes.json"
    if os.path.exists(cache):
        ch, eh = json.load(open(cache))
    else:
        with Pool(64) as pool:
            ch = dict(pool.map(charades, vids, chunksize=8))
            eh = dict(pool.map(evalv, ev, chunksize=16))
        json.dump([ch, eh], open(cache, "w"))
    # (v2) v1 matched 28k evaluation videos incl. Diving48 (impossible): flat frames (black / fades) share hashes. Drop near-flat hashes
    # (popcount <= 8 or >= 56) and hashes seen in >= 4 different Charades videos; a match needs >= 3 of the 5 frames on one video.
    flat = lambda h: not (8 < bin(h).count("1") < 56)
    seen = collections.defaultdict(set)
    for vid, hs in ch.items():
        for h in hs:
            seen[h].add(vid)
    ch = {vid: [h for h in hs if not flat(h) and len(seen[h]) < 4] for vid, hs in ch.items()}
    eh = {p: [h for h in hs if not flat(h)] for p, hs in eh.items()}
    idx = [collections.defaultdict(set) for _ in range(4)]          # 4 x 16-bit chunks: every hash within 3 bits shares a chunk
    for vid, hs in ch.items():
        for h in hs:
            for k in range(4):
                idx[k][(h >> (16 * k)) & 65535].add((vid, h))
    match = {}
    for p, hs in eh.items():
        votes = collections.Counter()
        for h in hs:
            best = set()
            for k in range(4):
                for vid, g in idx[k].get((h >> (16 * k)) & 65535, ()):
                    if vid not in best and bin(h ^ g).count("1") <= 6:
                        best.add(vid)
            votes.update(best)
        if votes:
            vid, n = votes.most_common(1)[0]
            if n >= 3:
                match[p] = vid
    named = set(json.load(open(f"{R}/logs/zz_charades_eval_ids.json")))
    ids = sorted(set(match.values()) | named)
    by = collections.Counter(p.split("/bench/")[1].split("/")[0] if "/bench/" in p else p.split("/")[3] for p in match)
    json.dump({"ids": ids, "matched_videos": match}, open(f"{R}/logs/zz_charades_eval_ids_content.json", "w"))
    print("content matches", len(match), "by folder", dict(by), "| ids by content", len(set(match.values())), "| by name", len(named), "| union", len(ids))
    print("charades videos without hashes", sum(1 for h in ch.values() if not h), "| eval videos without hashes", sum(1 for h in eh.values() if not h))
