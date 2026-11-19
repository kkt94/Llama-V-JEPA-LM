"""(10-25) sign check of camera rotation from EPIC Fields poses against the video itself: for random 2 s windows, yaw / pitch of the camera
between the window ends (COLMAP world->camera [qw qx qy qz tx ty tz]; forward axis of the later camera expressed in the earlier camera frame:
x right, y down, z forward) vs the mean optical flow of the frames (Farneback on the 384 px transcode, summed over 0.2 s steps). A camera
turning right moves the content left (mean dx < 0); looking up moves it down (dy > 0). usage: zz_epicfields_check.py [n_windows]"""
import sys, json, glob, os, csv, random, numpy as np, cv2
R = "/data/llama_vjepa_lm"; F = f"{R}/data/opendl/epic_fields/json-format"; PREP = f"{R}/prep/epic_384_15fps"; N = int(sys.argv[1]) if len(sys.argv) > 1 else 160
FPS = {r["video_id"]: float(r["fps"]) for r in csv.DictReader(open(f"{R}/data/opendl/epic_kitchens_100/annotations/EPIC_100_video_info.csv"))}
def rot(q):
    w, x, y, z = q; return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
def yaw_pitch(p0, p1):
    R0, R1 = rot(p0[:4]), rot(p1[:4]); f = R0 @ R1.T @ np.array([0, 0, 1.0])
    return np.degrees(np.arctan2(f[0], f[2])), np.degrees(np.arctan2(-f[1], np.hypot(f[0], f[2])))
rng = random.Random(0); vids = [os.path.basename(p)[:-5] for p in glob.glob(f"{F}/*.json")]; rng.shuffle(vids); res = []
for vid in vids:
    path = f"{PREP}/{vid.split('_')[0]}/{vid}.mp4"
    if not os.path.exists(path) or vid not in FPS: continue
    imgs = json.load(open(f"{F}/{vid}.json"))["images"]; fr = {int(k[6:16]): v for k, v in imgs.items()}; fps = FPS[vid]; keys = np.array(sorted(fr))
    cap = cv2.VideoCapture(path); vfps = cap.get(cv2.CAP_PROP_FPS); nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for _ in range(8):
        t0 = rng.uniform(5, nfr / vfps - 5); t1 = t0 + 2.0
        f0, f1 = int(round(t0 * fps)) + 1, int(round(t1 * fps)) + 1
        k0, k1 = keys[np.abs(keys - f0).argmin()], keys[np.abs(keys - f1).argmin()]
        if abs(k0 - f0) > fps * 0.1 or abs(k1 - f1) > fps * 0.1: continue
        yaw, pitch = yaw_pitch(fr[k0], fr[k1])
        dx = dy = 0.0; prev = None
        for t in np.arange(t0, t1 + 1e-6, 0.2):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, im = cap.read()
            if not ok: break
            g = cv2.cvtColor(cv2.resize(im, (192, 192)), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0); dx += float(np.median(fl[..., 0])); dy += float(np.median(fl[..., 1]))
            prev = g
        res.append((yaw, pitch, dx, dy))
    if len(res) >= N: break
res = np.array(res); print("windows", len(res))
for name, a, b, thr in (("yaw (turn right +)", res[:, 0], res[:, 2], 10), ("pitch (look up +)", res[:, 1], res[:, 3], 8)):
    m = np.abs(a) > thr
    agree = np.mean(np.sign(a[m]) == (-np.sign(b[m]) if "yaw" in name else np.sign(b[m]))) if m.any() else float("nan")
    print(f"{name}: |angle| > {thr} deg in {int(m.sum())} windows; sign agrees with the expected content flow in {100*agree:.1f}% | corr(angle, flow) {np.corrcoef(a, b)[0,1]:+.2f}")
print("abs yaw percentiles 50/75/90:", np.round(np.percentile(np.abs(res[:, 0]), [50, 75, 90]), 1), "| abs pitch:", np.round(np.percentile(np.abs(res[:, 1]), [50, 75, 90]), 1))
