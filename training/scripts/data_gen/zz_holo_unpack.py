"""(10-25) safe unpack of HoloAssist video_pitch_shifted.tar (untrusted): only regular files <video>/Export_py/Video_pitchshift.mp4 (and the
Video/*.txt timing / pose files), names checked by regex, tarfile filter="data", written to data/opendl/holoassist/videos/; never executed.
usage: python -I zz_holo_unpack.py"""
import tarfile, re, os
B = "/data/llama_vjepa_lm/data/opendl/holoassist"; pat = re.compile(r"^[A-Za-z0-9_.-]+/Export_py/(Video_pitchshift\.mp4|Video/(Pose_sync|VideoMp4Timing)\.txt)$"); n = sk = 0
with tarfile.open(f"{B}/video_pitch_shifted.tar") as t:
    for m in t:
        if not m.isfile() or not pat.match(m.name): sk += 1; continue
        out = f"{B}/videos/{m.name}"
        if os.path.exists(out) and os.path.getsize(out) == m.size: n += 1; continue
        t.extract(m, f"{B}/videos", filter="data"); n += 1
print("holoassist unpacked", n, "skipped", sk)
