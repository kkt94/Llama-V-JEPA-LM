"""Resumable HF snapshot download pinned to the commit sha resolved at start.
usage: dl_hf.py <name> <repo_type> <repo_id> <local_dir> [--allow P ...] [--ignore P ...]"""
import sys, time, json, argparse, datetime
from huggingface_hub import HfApi, snapshot_download
ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("repo_type"); ap.add_argument("repo_id"); ap.add_argument("local_dir")
ap.add_argument("--allow", nargs="*", default=None); ap.add_argument("--ignore", nargs="*", default=None)
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()
STATUS = "/data/llama_vjepa_lm/training/scripts/data_gen/status.tsv"
def log(state, extra=""):
    with open(STATUS, "a") as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}\t{a.name}\t{state}\t{extra}\n")
api = HfApi()
info = api.repo_info(a.repo_id, repo_type=a.repo_type)
sha = info.sha
log("start", f"{a.repo_id}@{sha}")
for attempt in range(1, 31):
    try:
        snapshot_download(a.repo_id, repo_type=a.repo_type, revision=sha, local_dir=a.local_dir,
                          allow_patterns=a.allow, ignore_patterns=a.ignore, max_workers=a.workers)
        rec = {"name": a.name, "repo_id": a.repo_id, "repo_type": a.repo_type, "revision": sha,
               "local_dir": a.local_dir, "allow": a.allow, "ignore": a.ignore,
               "completed": datetime.datetime.now().isoformat(timespec="seconds")}
        json.dump(rec, open(f"/data/llama_vjepa_lm/manifests/{a.name}.source.json", "w"), indent=1)
        log("complete", f"{a.repo_id}@{sha} attempt={attempt}")
        sys.exit(0)
    except Exception as e:
        log("retry", f"attempt={attempt} {type(e).__name__}: {str(e)[:200]}")
        time.sleep(min(600, 30 * attempt))
log("FAILED", a.repo_id); sys.exit(1)
