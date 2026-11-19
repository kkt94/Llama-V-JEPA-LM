"""(10-21) frame-folder subsets of VideoChat-Flash (ViDLN OVIS, HT-Step event count / relationship / understanding) -> data/vcf_frames/<subset>/<video>/*.jpg"""
import zipfile, os, sys
for z in sys.argv[1:]:
    zf = zipfile.ZipFile(z); n = 0
    for m in zf.namelist():
        if m.endswith("/") or m.startswith("/") or ".." in m.split("/"): continue
        out = os.path.join("/data/llama_vjepa_lm/data/vcf_frames", m)
        if os.path.exists(out): continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with zf.open(m) as s, open(out, "wb") as d: d.write(s.read())
        n += 1
    print("done", z, n, flush=True)
