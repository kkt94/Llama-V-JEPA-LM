"""Extract a zip without the unzip binary (not installed on this host). Skips files already present with the same size.
usage: unzip_py.py <zip> <dest>"""
import sys, os, zipfile, shutil
zp, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zp) as z:
    for info in z.infolist():
        out = os.path.join(dest, info.filename)
        if not os.path.realpath(out).startswith(os.path.realpath(dest)):
            raise SystemExit(f"unsafe path in archive: {info.filename}")
        if info.is_dir():
            os.makedirs(out, exist_ok=True); continue
        if os.path.exists(out) and os.path.getsize(out) == info.file_size:
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with z.open(info) as src, open(out + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst, 16 << 20)
        os.replace(out + ".part", out)
