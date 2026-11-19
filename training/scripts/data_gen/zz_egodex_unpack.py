"""(10-25) safe unpack of one EgoDex zip (untrusted): members must be <split>/<task>/<int>.(mp4|hdf5), no absolute / '..' paths, written
byte-for-byte (never executed) to data/opendl/egodex/<split>/...  usage: python -I zz_egodex_unpack.py <zip>"""
import sys, zipfile, re, os
B = "/data/llama_vjepa_lm/data/opendl/egodex"; z = zipfile.ZipFile(sys.argv[1]); ok = bad = 0
pat = re.compile(r"^(test|part\d|extra)/[a-z0-9_]+/\d+\.(mp4|hdf5)$")
for m in z.infolist():
    if m.is_dir(): continue
    if not pat.match(m.filename): bad += 1; print("skip", m.filename); continue
    out = os.path.join(B, m.filename)
    if os.path.exists(out) and os.path.getsize(out) == m.file_size: ok += 1; continue
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with z.open(m) as s, open(out + ".part", "wb") as d:
        while (b := s.read(1 << 24)): d.write(b)
    os.replace(out + ".part", out); ok += 1
print(sys.argv[1], "ok", ok, "skipped", bad)
