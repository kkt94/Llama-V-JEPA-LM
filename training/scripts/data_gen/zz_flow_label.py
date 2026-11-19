"""(10-16) Camera movement of natural clips from optical flow (the answer key measured from the pixels, not from a model): for a window,
9 evenly spaced frames (grey, 160 px wide), Farneback flow between neighbours; per pair a global fit flow(x, y) = (tx, ty) + s * (x - cx, y - cy)
(translation + expansion, least squares over all pixels). Content moving right (tx > 0) = the camera pans LEFT; content moving down = tilts UP;
expanding (s > 0) = moving forward / zooming in. A clip gets a label when one component dominates and keeps its sign in >= 7 of 8 pairs.
usage: zz_flow_label.py <n> <shard> <nshard> [pass] -> runs/info/flow_<pass>_<shard>.parquet (pass: another random window per video) (SA-V 6 s windows; video, t0, t1, tx, ty, s, agree_*)"""
import sys, numpy as np, pandas as pd, cv2, decord
N, SH, NS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); PS = int(sys.argv[4]) if len(sys.argv) > 4 else 0; SV = "/data/llama_vjepa_lm/prep/sav_384_24fps"
cv2.setNumThreads(1)
vids = pd.read_parquet("/data/llama_vjepa_lm/manifests/stage1/sav_tracks_qa.parquet", columns=["video"]).video.drop_duplicates().sample(N, random_state=31).tolist()[SH::NS]
rng = np.random.default_rng(1000 * PS + SH); out = []
for v in vids:
    try:
        vr = decord.VideoReader(f"{SV}/{v}", num_threads=1); fps = vr.get_avg_fps(); D = len(vr) / fps
        t0 = float(rng.uniform(0, max(D - 6, 0))); t1 = min(D, t0 + 6)
        idx = np.clip(np.round((t0 + (np.arange(9) + 0.5) * (t1 - t0) / 9) * fps).astype(int), 0, len(vr) - 1)
        fr = vr.get_batch(idx.tolist()).asnumpy(); del vr
    except Exception:
        continue
    h, w = fr.shape[1:3]; W = 160; H = int(round(h * W / w))
    g = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (W, H), interpolation=cv2.INTER_AREA) for f in fr]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32); xc, yc = xx - W / 2, yy - H / 2; r2 = (xc ** 2 + yc ** 2).sum()
    T = []
    for a, b in zip(g[:-1], g[1:]):
        f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        tx, ty = float(f[..., 0].mean()), float(f[..., 1].mean())
        s = float(((f[..., 0] - tx) * xc + (f[..., 1] - ty) * yc).sum() / r2)
        res = float(np.sqrt(((f[..., 0] - tx - s * xc) ** 2 + (f[..., 1] - ty - s * yc) ** 2).mean()))
        T.append((tx / W, ty / W, s, res / W))
    T = np.array(T)
    out.append(dict(video=v, t0=t0, t1=t1, tx=T[:, 0].mean(), ty=T[:, 1].mean(), s=T[:, 2].mean(), res=T[:, 3].mean(),
                    agree_x=max((T[:, 0] > 0).mean(), (T[:, 0] < 0).mean()), agree_y=max((T[:, 1] > 0).mean(), (T[:, 1] < 0).mean()),
                    agree_s=max((T[:, 2] > 0).mean(), (T[:, 2] < 0).mean())))
pd.DataFrame(out).to_parquet(f"/data/llama_vjepa_lm/runs/info/flow_{PS}_{SH}.parquet"); print("done", SH, len(out), flush=True)
