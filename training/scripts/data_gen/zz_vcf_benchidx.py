"""(10-19) Benchmark video-id index for the VideoChat-Flash contamination check (read-only over /data/llama_vjepa_lm/bench).
Collects, per benchmark (top-level dir), every id-like token: file / zip-member stems and strings in annotation files
(json/jsonl/csv/tsv/txt/parquet). Each token adds variants: itself, without extension(s), without a 'v_' prefix, and the
11-char YouTube id when the token starts with one followed by '_' / '.' / end. -> logs/zz_vcf_benchidx.json {bench: [ids]}"""
import os, re, sys, json, zipfile, collections

B = "/data/llama_vjepa_lm/bench"
OUT = "/data/llama_vjepa_lm/training/scripts/data_gen/zz_vcf_benchidx.json"
TOK = re.compile(r"[A-Za-z0-9_\-@.]{5,}")
YT = re.compile(r"^(?:v_)?([A-Za-z0-9_-]{11})(?:[_.].*)?$")
TEXT = (".json", ".jsonl", ".csv", ".tsv", ".txt", ".parquet")
VID = (".mp4", ".avi", ".mkv", ".webm", ".mov", ".mpg", ".mpeg", ".m4v", ".gif", ".flv", ".3gp", ".wmv")


def variants(t):
    t = t.strip().strip(".")
    out = {t}
    b = t.rsplit("/", 1)[-1]
    out.add(b)
    s = b
    while True:
        r, e = os.path.splitext(s)
        if not e or len(e) > 6:
            break
        s = r
    out.add(s)
    if s.startswith("v_"):
        out.add(s[2:])
    m = YT.match(s)
    if m:
        out.add(m.group(1))
    return {x for x in out if len(x) >= 5}


def text_tokens(path):
    try:
        if path.endswith(".parquet"):
            import pyarrow as pa, pyarrow.parquet as pq     # string columns only (some parquets embed video bytes)
            sch = pq.ParquetFile(path).schema_arrow
            cols = [f.name for f in sch if pa.types.is_string(f.type) or pa.types.is_large_string(f.type)]
            if not cols:
                return set()
            t = pq.read_table(path, columns=cols)
            blob = "\n".join(" ".join(str(x) for x in t.column(c).to_pylist()) for c in cols)
        else:
            if os.path.getsize(path) > 2 << 30:
                return set()
            blob = open(path, errors="ignore").read()
    except Exception as e:
        print("skip", path, e, file=sys.stderr)
        return set()
    toks = set()
    for t in TOK.findall(blob):
        if "/" in t or any(t.lower().endswith(v) for v in VID) or YT.match(t) or re.search(r"\d", t):
            toks |= variants(t)
    # path-like strings with '/' (TOK excludes '/'): take basenames
    for t in re.findall(r"[A-Za-z0-9_\-@./]+\.(?:mp4|avi|mkv|webm|mov|mpg|gif|m4v)", blob, flags=re.I):
        toks |= variants(t)
    return toks


idx = collections.defaultdict(set)
for bench in sorted(os.listdir(B)):
    root = os.path.join(B, bench)
    n_files = 0
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d != ".cache"]
        for f in fn:
            n_files += 1
            p = os.path.join(dp, f)
            idx[bench] |= variants(f)
            fl = f.lower()
            if fl.endswith(".zip"):
                try:
                    for name in zipfile.ZipFile(p).namelist():
                        idx[bench] |= variants(name.rstrip("/"))
                except Exception as e:
                    print("badzip", p, e, file=sys.stderr)
            elif fl.endswith(TEXT):
                idx[bench] |= text_tokens(p)
        for d in dn:
            idx[bench] |= variants(d)
    print(bench, "files", n_files, "ids", len(idx[bench]), flush=True)
json.dump({k: sorted(v) for k, v in idx.items()}, open(OUT, "w"))
print("wrote", OUT)
