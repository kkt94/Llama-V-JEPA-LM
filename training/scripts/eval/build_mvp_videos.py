"""Materialise MVP videos that the MVP setup scripts *generate* (they do not exist as files in any source archive),
plus a path-mapping table and an lmms-eval-style symlink tree. Sources under bench/mvp_sources/extracted are only read.

  intphys         setup_intphys.py: dev/<P>/<id>/<v>/scene/scene_001..100.png -> 24 fps libx264 mp4
                  (moviepy ImageSequenceClip -> write_videofile(codec='libx264'); here PyAV libx264 yuv420p, RGB input)
  language_table  merge_lt.py: language_table/video_<id>_frame_<n>.jpg (n = 0.. while present) -> 5 fps mp4v mp4 via cv2.
                  The official merge_lt.py parses the id as '_'.join(file.split('_')[3:4]) which, for the path
                  'language_table/video_<id>_frame_<n>.jpg', yields 'frame' (the folder name contains '_'), so it
                  writes no video (our lane log: lt_post complete mp4=0). We use the intended id field.

Outputs (all under /data/llama_vjepa_lm/bench/mvp_videos):
  generated/intphys/_<P>_<id>_<v>.mp4, generated/language_table/video_<id>.mp4
  videos/<mvp video_path>  -> symlink to the resolved file (same layout lmms-eval's mvp task expects; ssv2/<id>.mp4
                              points at the original <id>.webm)
  path_map.tsv              mvp_video_path, resolved_path, method, exists

usage: python -I build_mvp_videos.py [--workers 24] [--only intphys|language_table|links]
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from jepa_llm.eval.adapters import BENCH, MVP_SRC, MVP_SUBSETS, MVP_VIDEOS, mvp_intphys_frames_dir, mvp_resolve  # noqa

LT_FRAMES = MVP_SRC / "lt_work" / "language_table"


def all_mvp_video_paths() -> list[str]:
    import pandas as pd
    vps = set()
    for subset in MVP_SUBSETS:
        for split in ("mini", "full"):
            df = pd.read_parquet(BENCH / "minimal_video_pairs" / subset / f"{split}-00000-of-00001.parquet")
            vps.update(v.lstrip("/") for v in df.video_path)
    return sorted(vps)


def make_intphys(rel: str) -> str:
    import av
    import numpy as np
    from PIL import Image
    out = MVP_VIDEOS / "generated" / "intphys" / rel
    if out.exists() and out.stat().st_size > 0:
        return "skip"
    src = mvp_intphys_frames_dir(rel)
    frames = [src / f"scene_{i:03}.png" for i in range(1, 101) if (src / f"scene_{i:03}.png").exists()]
    if not frames:
        return f"missing {src}"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.mp4")
    with av.open(str(tmp), "w") as c:
        st = c.add_stream("libx264", rate=24)
        first = np.asarray(Image.open(frames[0]).convert("RGB"))
        st.height, st.width = first.shape[:2]
        st.pix_fmt = "yuv420p"
        for p in frames:
            fr = av.VideoFrame.from_ndarray(np.asarray(Image.open(p).convert("RGB")), format="rgb24")
            for pkt in st.encode(fr):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
    os.replace(tmp, out)
    return "ok"


def make_lt(rel: str) -> str:
    import cv2
    out = MVP_VIDEOS / "generated" / "language_table" / rel
    if out.exists() and out.stat().st_size > 0:
        return "skip"
    vid = rel[len("video_"):-len(".mp4")]
    frames, n = [], 0
    while (LT_FRAMES / f"video_{vid}_frame_{n}.jpg").exists():
        frames.append(LT_FRAMES / f"video_{vid}_frame_{n}.jpg")
        n += 1
    if not frames:
        return f"missing frames for {vid}"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.mp4")
    h, w = cv2.imread(str(frames[0])).shape[:2]
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
    for p in frames:
        vw.write(cv2.imread(str(p)))
    vw.release()
    os.replace(tmp, out)
    return "ok"


def run_pool(fn, rels, workers):
    from collections import Counter
    c = Counter()
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, rels, chunksize=8)):
            c[r if r in ("ok", "skip") else "fail"] += 1
            if r not in ("ok", "skip"):
                print("FAIL", r, flush=True)
            if (i + 1) % 2000 == 0:
                print(fn.__name__, i + 1, "/", len(rels), dict(c), flush=True)
    print(fn.__name__, "done", dict(c), flush=True)


def build_links(vps):
    rows = []
    for vp in vps:
        path, method = mvp_resolve(vp)
        ok = path.exists() and path.stat().st_size > 0
        rows.append((vp, str(path), method, int(ok)))
        if ok:
            link = MVP_VIDEOS / "videos" / vp
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() and os.readlink(link) != str(path):
                link.unlink()
            if not link.is_symlink():
                os.symlink(path, link)
    tmp = MVP_VIDEOS / "path_map.tsv.tmp"
    with open(tmp, "w") as f:
        f.write("mvp_video_path\tresolved_path\tmethod\texists\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    os.replace(tmp, MVP_VIDEOS / "path_map.tsv")
    print("links: unique video paths", len(rows), "resolvable", sum(r[3] for r in rows), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--only", choices=["intphys", "language_table", "links"])
    a = ap.parse_args()
    vps = all_mvp_video_paths()
    print("unique MVP video paths (mini+full):", len(vps), flush=True)
    if a.only in (None, "intphys"):
        run_pool(make_intphys, [v.split("/", 1)[1] for v in vps if v.startswith("intphys/")], a.workers)
    if a.only in (None, "language_table"):
        if (MVP_SRC / "language_table_slow").is_dir() and any(Path(MVP_SRC / "language_table_slow").glob("*.mp4")):
            print("language_table_slow has official mp4s; generation skipped", flush=True)
        else:
            run_pool(make_lt, [v.split("/", 1)[1] for v in vps if v.startswith("language_table/")], a.workers)
    build_links(vps)


if __name__ == "__main__":
    main()
