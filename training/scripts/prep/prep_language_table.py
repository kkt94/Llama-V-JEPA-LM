"""Encode Language-Table REAL robot episodes (train only, NOT evaluated by MVP) as 5 fps mp4 for stage-1 training.

Source: the TFDS/RLDS `language_table` 0.0.1 release (gs://gresearch/robotics/language_table, local copy under
bench/mvp_sources/raw/language_table/0.0.1, 500 shards, 442,226 episodes). Each tf.Example holds one episode with
`steps/observation/rgb` (360x640 JPEG per step) and `steps/observation/instruction` (512 int32 UTF-8 bytes per step).

Episode id = GLOBAL index in `builder.as_dataset(split='train').enumerate()` order (the order MVP's
download_lt_local.py uses for `language_table/video_<id>.mp4`). NOTE: that order is NOT shard-sequential: TFDS's default
ReadConfig interleaves the 500 shard files with cycle_length=16, block_length=16 (deterministic). The exact order is
reproduced by running the same tf.data interleave over per-shard index ranges of length shardLengths
(global_order_c16b16.npy: row gid -> (shard, index in shard)); `--verify` checks it against MVP items.

Exclusions: every MVP language_table id (manifests/allowlist/benchmark_ids_mvp.txt, 25,768 ids) is never encoded;
episodes with < 8 frames or an empty instruction are skipped. Selection: u = md5("lt<seed>:<gid>") in [0,1); an
eligible episode is encoded iff u < P (default 0.49 -> ~204k; the manifest keeps the 200k smallest u).

Format (matches MVP's merge_lt.py videos: native 640x360, 5 fps, one frame per step, no resampling): the original JPEG
bytes are piped to ffmpeg (image2pipe/mjpeg) -> h264 yuv420p crf 20, -g 8, +faststart.
Output: prep/language_table_train/<gid>.mp4 (temp name in the same dir, then rename), and per-shard metadata for ALL
episodes (gid, episode_id, n_frames, instruction, mvp flag, u, selected) in prep/language_table_train_meta/.
Resumable: a shard whose meta exists and whose selected mp4s all exist is skipped; existing mp4s are never re-encoded.

Usage: python prep_language_table.py --node 0 --nodes 2 --workers 30      (shards node::nodes)
       python prep_language_table.py --verify 153659,268620,45898   (prints decoded instruction / n_frames)
"""
import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from multiprocessing import Pool

R = "/data/llama_vjepa_lm"
RAW = f"{R}/bench/mvp_sources/raw/language_table/0.0.1"
OUT = f"{R}/prep/language_table_train"
META = f"{R}/prep/language_table_train_meta"
FFMPEG = f"{R}/tools/ffmpeg"
ALLOW = f"{R}/manifests/allowlist/benchmark_ids_mvp.txt"
FPS = 5
MIN_FRAMES = 8
NSHARDS = 500


def shard_offsets():
    info = json.load(open(f"{RAW}/dataset_info.json"))
    lens = [int(x) for x in [s for s in info["splits"] if s["name"] == "train"][0]["shardLengths"]]
    assert len(lens) == NSHARDS and sum(lens) == 442226, (len(lens), sum(lens))
    offs = [0]
    for n in lens:
        offs.append(offs[-1] + n)
    return lens, offs


def global_order():
    """gid -> (shard, idx) under TFDS's default interleave (cycle 16, block 16) of the 500 files in order."""
    path = f"{META}/global_order_c16b16.npy"
    if not os.path.exists(path):
        import numpy as np
        import tensorflow as tf
        lens, _ = shard_offsets()
        L = tf.constant(lens, dtype=tf.int64)
        ds = tf.data.Dataset.range(len(lens)).interleave(
            lambda s: tf.data.Dataset.range(L[s]).map(lambda i: (s, i)), cycle_length=16, block_length=16)
        arr = np.array([(int(a), int(b)) for a, b in ds.as_numpy_iterator()], dtype=np.int32)
        np.save(path + ".tmp.npy", arr)
        os.replace(path + ".tmp.npy", path)
    import numpy as np
    arr = np.load(path)
    assert arr.shape == (442226, 2)
    gid_of = [None] * NSHARDS
    lens, _ = shard_offsets()
    for s in range(NSHARDS):
        gid_of[s] = np.full(lens[s], -1, dtype=np.int64)
    for g, (s, i) in enumerate(arr):
        gid_of[s][i] = g
    assert all((x >= 0).all() for x in gid_of)
    return arr, gid_of


def load_mvp_ids():
    ids = set()
    for line in open(ALLOW):
        f = line.rstrip("\n").split("\t")
        if f[0] == "language_table":
            ids.add(int(f[1].replace("video_", "")))
    assert ids, "no MVP language_table ids read: the exclusion would be vacuous"
    return ids


def u_of(gid, seed):
    return int(hashlib.md5(f"lt{seed}:{gid}".encode()).hexdigest()[:12], 16) / float(1 << 48)


def iter_records(path):
    """Raw TFRecord framing: uint64 len, uint32 crc(len), data, uint32 crc(data)."""
    with open(path, "rb", buffering=8 << 20) as fh:
        while True:
            hdr = fh.read(12)
            if not hdr:
                return
            (n,) = struct.unpack("<Q", hdr[:8])
            data = fh.read(n)
            fh.read(4)
            yield data


def parse(example_pb2, data):
    ex = example_pb2.Example()
    ex.ParseFromString(data)
    f = ex.features.feature
    jpgs = list(f["steps/observation/rgb"].bytes_list.value)
    inst = f["steps/observation/instruction"].int64_list.value
    n = len(jpgs)
    per = len(inst) // max(n, 1)

    def dec(chunk):
        return bytes(b for b in chunk if b != 0).decode("utf-8", errors="replace")

    last = dec(inst[(n - 1) * per:n * per]) if n else ""
    first = dec(inst[:per]) if n else ""
    eid = f["episode_id"].bytes_list.value[0].decode() if f["episode_id"].bytes_list.value else ""
    return jpgs, last, first == last, eid


def encode(jpgs, dst):
    tmp = os.path.join(os.path.dirname(dst), f".{os.path.basename(dst)}.tmp.mp4")
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "image2pipe", "-framerate", str(FPS),
           "-c:v", "mjpeg", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-g", "8",
           "-pix_fmt", "yuv420p", "-threads", "1", "-movflags", "+faststart", "-f", "mp4", tmp]
    p = subprocess.run(cmd, input=b"".join(jpgs), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if p.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) < 1000:
        if os.path.exists(tmp):
            os.remove(tmp)
        return p.stderr.decode()[-300:] or "empty output"
    os.replace(tmp, dst)
    return None


def init_worker():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    global EXPB2
    from tensorflow.core.example import example_pb2
    EXPB2 = example_pb2


def do_shard(args):
    shard, gids, mvp, seed, p_sel = args
    length = len(gids)
    mpath = f"{META}/shard_{shard:05d}.jsonl"
    if os.path.exists(mpath):
        rows = [json.loads(l) for l in open(mpath)]
        if all(os.path.exists(f"{OUT}/{r['gid']}.mp4") for r in rows if r["sel"]):
            return shard, 0, 0, 0, "skip"
    t0 = time.time()
    rows, enc, fail = [], 0, 0
    for i, data in enumerate(iter_records(f"{RAW}/language_table-train.tfrecord-{shard:05d}")):
        gid = int(gids[i])
        jpgs, inst, same, eid = parse(EXPB2, data)
        u = u_of(gid, seed)
        is_mvp = gid in mvp
        sel = (not is_mvp) and len(jpgs) >= MIN_FRAMES and bool(inst.strip()) and u < p_sel
        r = {"gid": gid, "shard": shard, "idx": i, "eid": eid, "n": len(jpgs), "inst": inst, "inst_const": same,
             "mvp": is_mvp, "u": u, "sel": sel}
        if sel:
            dst = f"{OUT}/{gid}.mp4"
            if not os.path.exists(dst):
                err = encode(jpgs, dst)
                if err:
                    fail += 1
                    r["sel"] = False
                    r["err"] = err
                else:
                    enc += 1
        rows.append(r)
    assert len(rows) == length, f"shard {shard}: {len(rows)} records != shardLength {length}"
    tmp = f"{mpath}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, mpath)
    return shard, len(rows), enc, fail, f"{time.time() - t0:.0f}s"


def verify(ids):
    init_worker()
    arr, _ = global_order()
    for gid in ids:
        s, idx = (int(x) for x in arr[gid])
        for i, data in enumerate(iter_records(f"{RAW}/language_table-train.tfrecord-{s:05d}")):
            if i == idx:
                jpgs, inst, same, eid = parse(EXPB2, data)
                print(json.dumps({"gid": gid, "shard": s, "idx": i, "n_frames": len(jpgs), "inst": inst,
                                  "inst_const": same}), flush=True)
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, default=0)
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=float, default=0.49)
    ap.add_argument("--shards", default="", help="comma list overriding node::nodes")
    ap.add_argument("--verify", default="")
    a = ap.parse_args()
    if a.verify:
        return verify([int(x) for x in a.verify.split(",")])
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(META, exist_ok=True)
    _, gid_of = global_order()
    mvp = load_mvp_ids()
    print(f"MVP ids {len(mvp)} | node {a.node}/{a.nodes} | P {a.p}", flush=True)
    shards = [int(x) for x in a.shards.split(",")] if a.shards else list(range(a.node, NSHARDS, a.nodes))
    jobs = [(s, gid_of[s], mvp, a.seed, a.p) for s in shards]
    t0, tot_enc = time.time(), 0
    with Pool(a.workers, initializer=init_worker) as pool:
        for k, (s, n, enc, fail, msg) in enumerate(pool.imap_unordered(do_shard, jobs)):
            tot_enc += enc
            el = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}] shard {s:03d} eps {n} enc {enc} fail {fail} {msg} | done {k + 1}/{len(jobs)}"
                  f" | enc total {tot_enc} | {tot_enc / max(el, 1):.1f} ep/s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
