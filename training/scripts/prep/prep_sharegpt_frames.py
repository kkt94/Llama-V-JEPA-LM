"""ShareGPTVideo frame chunks (2 fps jpeg folders, 336x252) -> one 2 fps mp4 per clip, streamed from the tar.gz.

The frames are never written to disk as files (the 900k clips hold tens of millions of jpegs): members are read in tar
order, grouped by clip folder, and piped to ffmpeg (image2pipe). 2 fps was measured, not assumed: frame k of an
ActivityNet scene matches the original video at t0 + k / 2.001 s (scripts/checks in the 2026-10-09 notes).
usage: prep_sharegpt_frames.py <chunk.tar.gz> <out_dir>   (writes <out_dir>/<clip>.mp4, skips clips already done)
"""
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

FPS = 2
FFMPEG = os.environ.get("FFMPEG", "/data/llama_vjepa_lm/tools/ffmpeg")   # static build on the shared disk (every node)


def encode(out, frames):
    if os.path.exists(out) or len(frames) < 2:
        return "skip"
    tmp = out[:-4] + f".part{os.getpid()}.mp4"
    cmd = [FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "mjpeg",
           "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-g", "8", "-pix_fmt", "yuv420p", "-threads", "1", "-an", tmp]
    p = subprocess.run(cmd, input=b"".join(b for _, b in sorted(frames)), capture_output=True)   # c01_0001, c01_0002, ...
    if p.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        return "fail"
    os.replace(tmp, out)
    return "ok"


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    stats = {"ok": 0, "skip": 0, "fail": 0}
    futs = []
    with ThreadPoolExecutor(int(os.environ.get("ENC_THREADS", 6))) as ex, tarfile.open(src, "r|gz") as tf:
        cur, frames, done = None, [], False
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith((".jpg", ".jpeg")):
                continue
            parts = m.name.lstrip("./").split("/")
            clip = parts[-2] if len(parts) >= 2 else None
            if clip != cur:
                if cur is not None:
                    futs.append(ex.submit(encode, f"{out_dir}/{cur}.mp4", frames))
                    while len(futs) > 48:                    # back-pressure: reading is faster than encoding
                        stats[futs.pop(0).result()] += 1
                cur, frames = clip, []
                done = os.path.exists(f"{out_dir}/{clip}.mp4")   # one stat per clip, not per frame
            if done:
                continue                                     # resume: still have to stream past it
            frames.append((parts[-1], tf.extractfile(m).read()))
        if cur is not None:
            futs.append(ex.submit(encode, f"{out_dir}/{cur}.mp4", frames))
        for f in futs:
            stats[f.result()] += 1
    print(os.path.basename(src), stats, flush=True)
    return 0 if stats["fail"] < max(10, 0.01 * sum(stats.values())) else 1


if __name__ == "__main__":
    sys.exit(main())
