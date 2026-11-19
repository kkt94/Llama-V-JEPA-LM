"""(10-27) transcode the extracted Oops videos (logs/zz_oops_extract.py) -> prep/oops_384_15fps/<name>.mp4 (transcode_one.sh, 40 parallel)"""
import glob, os, subprocess
from concurrent.futures import ThreadPoolExecutor
R = "/data/llama_vjepa_lm"; I = sorted(glob.glob(f"{R}/data/opendl/oops_hf_x/video/*.mp4")); os.makedirs(f"{R}/prep/oops_384_15fps", exist_ok=True)
def one(p): subprocess.run(["bash", f"{R}/training/scripts/prep/transcode_one.sh", p, f"{R}/prep/oops_384_15fps/" + os.path.basename(p), "15", "15"])
with ThreadPoolExecutor(40) as ex: list(ex.map(one, I))
print("transcoded", len(glob.glob(f"{R}/prep/oops_384_15fps/*.mp4")), "of", len(I), "OOPS PREP DONE", flush=True)
