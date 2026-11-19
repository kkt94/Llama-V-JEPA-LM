"""(10-19) PLM-FGQA segments cut from LOCAL full videos (no YouTube): COIN videos from VideoChat-Flash TimeIT-COIN, HowTo100M videos from
VideoChat-Flash HT-Step. Same output format as logs/zz_fgqa_dl2.sh (short side 384, 15 fps, GOP 15) -> prep/fgqa_384_15fps/<source>/<segment>.mp4;
segments already there are skipped. usage: zz_fgqa_local_cut.py <source: coin|ht100m> <video root> [workers]"""
import sys, os, subprocess, collections
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
src, root = sys.argv[1], sys.argv[2]; W = int(sys.argv[3]) if len(sys.argv) > 3 else 24
R = "/data/llama_vjepa_lm"; OUT = f"{R}/prep/fgqa_384_15fps/{src}"; FF = f"{R}/tools/ffmpeg"; os.makedirs(OUT, exist_ok=True)
local = {}
for dp, _, fs in os.walk(root):
    for f in fs:
        if f.endswith((".mp4", ".mkv", ".webm")): local[os.path.splitext(f)[0]] = os.path.join(dp, f)
d = pd.read_parquet(f"{R}/data/PLM-Video-Human/fgqa/plm_fgqa_train.parquet", columns=["segment_id", "metadata"])
M = pd.DataFrame(list(d.metadata.map(dict))); M["segment_id"] = d.segment_id
M = M[(M.source_dataset == src) & M.source_video_id.isin(local)].drop_duplicates("segment_id")
jobs = [(local[r.source_video_id], r.source_start_time, r.source_end_time, f"{OUT}/{r.segment_id}.mp4") for r in M.itertuples()
        if not os.path.exists(f"{OUT}/{r.segment_id}.mp4")]
print(f"{src}: local videos {len(local)}, FGQA segments on them {len(M)}, to cut {len(jobs)}", flush=True)
def cut(j):
    v, t0, t1, out = j
    r = subprocess.run([FF, "-nostdin", "-v", "error", "-y", "-threads", "1", "-filter_threads", "1", "-ss", str(t0), "-to", str(t1), "-i", v, "-an",
                        "-vf", "scale='if(gt(iw,ih),-2,384)':'if(gt(iw,ih),384,-2)':flags=bicubic,fps=15", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "23", "-g", "15", "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", "-f", "mp4", out + ".part"], capture_output=True)
    if r.returncode == 0 and os.path.getsize(out + ".part") > 0:
        os.replace(out + ".part", out); return 1
    if os.path.exists(out + ".part"): os.remove(out + ".part")
    return 0
with ProcessPoolExecutor(W) as ex: ok = sum(ex.map(cut, jobs, chunksize=8))
print(f"{src}: cut {ok} of {len(jobs)}", flush=True)
