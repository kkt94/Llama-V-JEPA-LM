"""Clip loading for V-JEPA inputs.

Sampling ("paired uniform"): num_frames/2 anchor times spread uniformly over the video; each anchor contributes two
frames `pair_gap_sec` apart so every tubelet (2 frames) sees a short real time step, matching V-JEPA 2.1 pretraining
(4 fps -> 0.25 s). Plain uniform sampling over a long video would put seconds between the two frames of a tubelet.
Frames are resized (short side 384, only if needed), centre-cropped to 384x384 and ImageNet-normalised.
Returns real timestamps so evidence keeps its actual time (spec 7.5).
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


def paired_uniform_times(duration: float, num_frames: int, pair_gap_sec: float = 0.25, jitter: float = 0.0):
    """jitter (10-19, training views): each frame / pair anchor drawn uniformly inside its own 1/k slot (scaled by `jitter`) instead of the
    slot centre - the clip is still covered end to end and in order, so what a caption or answer says stays true; each pass sees new frames"""
    assert num_frames % 2 == 0
    if os.environ.get("JEPA_SAMPLING") == "uniform":
        # (10-21, data.sampling: uniform) the VideoLLM convention (V-JEPA 2 VideoLLM: "Sampling method Uniform", 32 frames; PLM: "32 frames
        # uniformly from the full video"): num_frames distinct times, slot centres over the whole span (each frame jittered inside its own
        # slot for training views); a V-JEPA tubelet then pairs two consecutive samples (span / num_frames apart)
        off = 0.5 + (np.random.rand(num_frames) - 0.5) * jitter if jitter else 0.5
        return (np.arange(num_frames) + off) * duration / num_frames
    k = num_frames // 2
    span = max(duration - pair_gap_sec, 0.0)
    if span / k < pair_gap_sec:
        # short clip: pairs would overlap (times going backwards across tubelets) -> evenly spaced, monotonic frames
        off = 0.5 + (np.random.rand(num_frames) - 0.5) * jitter if jitter else 0.5
        return (np.arange(num_frames) + off) * duration / num_frames
    off = 0.5 + (np.random.rand(k) - 0.5) * jitter if jitter else 0.5
    anchors = (np.arange(k) + off) * span / k
    return np.stack([anchors, anchors + pair_gap_sec], 1).reshape(-1)


def load_clip(path: str, num_frames: int, size: int = 384, pair_gap_sec: float = 0.25, times=None, window=None,
              crop_center=None, jitter: float = 0.0, frame=None):
    """-> (video [3, T, size, size] float32 normalised, frame_times [T] seconds, crop box (x0, y0, x1, y1) normalised in
    the source frame). `window` = (t0, t1) restricts sampling to that span; `crop_center` = normalised (cx, cy) moves the
    square crop so that point is as central as the frame allows (default: centre crop)."""
    if os.path.splitext(path)[1].lower() in (".jpg", ".jpeg", ".png", ".webp"):   # (10-21) a still image = a static clip (OCR / scene text)
        from PIL import Image
        im = torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy()).permute(2, 0, 1).float() / 255
        frames = im.unsqueeze(1).expand(-1, num_frames, -1, -1).contiguous()
        return _finish(frames, torch.as_tensor((np.arange(num_frames) + 0.5) * 0.25, dtype=torch.float32), size, frame, crop_center, jitter)
    if os.path.isdir(path):                 # (10-21) a video stored as a folder of frames (ViDLN OVIS, HT-Step): nominal 1 frame per second
        from PIL import Image
        files = sorted(f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        n, fps = len(files), float(os.environ.get("JEPA_DIR_FPS", 1.0))
        if times is None:
            times = paired_uniform_times(n / fps, num_frames, max(pair_gap_sec, 1.0 / fps), jitter=jitter)
        idx = np.clip(np.round(np.asarray(times) * fps).astype(int), 0, n - 1)
        frames = torch.stack([torch.from_numpy(np.asarray(Image.open(os.path.join(path, files[i])).convert("RGB")).copy()) for i in idx]).permute(3, 0, 1, 2).float() / 255
        return _finish(frames, torch.as_tensor(idx / fps, dtype=torch.float32), size, frame, crop_center, jitter)
    import decord
    vr = decord.VideoReader(path, num_threads=1)
    fps = float(vr.get_avg_fps())
    n = len(vr)
    pair_gap_sec = max(pair_gap_sec, 1.0 / fps)   # low-fps sources (2 fps ShareGPTVideo): pair = adjacent source frames
    if times is None:
        if window is not None:
            t0, t1 = max(0.0, float(window[0])), min(n / fps, float(window[1]))
            times = t0 + paired_uniform_times(max(t1 - t0, 0.0), num_frames, min(pair_gap_sec, max(t1 - t0, 1e-3) / 2), jitter=jitter)
        else:
            times = paired_uniform_times(n / fps, num_frames, pair_gap_sec, jitter=jitter)
    idx = np.clip(np.round(np.asarray(times) * fps).astype(int), 0, n - 1)
    frames = torch.from_numpy(vr.get_batch(idx.tolist()).asnumpy()).permute(3, 0, 1, 2).float() / 255  # 3,T,H,W
    return _finish(frames, torch.as_tensor(idx / fps, dtype=torch.float32), size, frame, crop_center, jitter)


def _finish(frames, ft, size, frame, crop_center, jitter):
    """resize / crop / normalise frames [3, T, H, W] in 0..1 (the second half of load_clip); ft = frame times [T]"""
    h, w = frames.shape[-2:]
    fm = os.environ.get("JEPA_EVAL_FRAME", "") if frame is None else frame   # (10-19) the WHOLE frame instead of the centre square
    fm = "" if fm == "crop" else fm                    #   (`frame` from the training config; evaluation: env JEPA_EVAL_FRAME)
    if fm == "squash" and (h, w) != (size, size):       # aspect ratio ignored
        frames = F.interpolate(frames.transpose(0, 1), size=(size, size), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1).transpose(0, 1)
        return (frames - MEAN) / STD, ft, (0.0, 0.0, 1.0, 1.0)
    if fm == "pad":                                     # long side to `size`, the short side padded with the mean colour (0 after normalising)
        s = size / max(h, w); nh, nw = round(h * s), round(w * s)
        frames = F.interpolate(frames.transpose(0, 1), size=(nh, nw), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1).transpose(0, 1)
        out = torch.zeros(frames.shape[0], frames.shape[1], size, size); y, x = (size - nh) // 2, (size - nw) // 2
        out[:, :, y:y + nh, x:x + nw] = (frames - MEAN) / STD
        return out, ft, (0.0, 0.0, 1.0, 1.0)
    if min(h, w) != size:
        s = size / min(h, w)
        frames = F.interpolate(frames.transpose(0, 1), size=(round(h * s), round(w * s)), mode="bicubic",
                               antialias=True, align_corners=False).clamp(0, 1).transpose(0, 1)
        h, w = frames.shape[-2:]
    cx, cy = crop_center if crop_center is not None else (0.5, 0.5)
    if jitter and crop_center is None:                   # (10-19) training views: the square slides around the centre (up to +-15% of the frame)
        cx, cy = cx + (np.random.rand() - 0.5) * 0.3 * jitter, cy + (np.random.rand() - 0.5) * 0.3 * jitter
    x = int(min(max(round(cx * w - size / 2), 0), w - size))
    y = int(min(max(round(cy * h - size / 2), 0), h - size))
    frames = frames[:, :, y:y + size, x:x + size]
    box = (x / w, y / h, (x + size) / w, (y + size) / h)
    return (frames - MEAN) / STD, ft, box


def compose_clip(spec, path, num_frames, pair_gap_sec=0.25, window=None, path2=None, window2=None):
    """(10-16) a clip BUILT so a question's answer is known exactly (transformation questions; the constructions of TempCompass / TOMATO).
    spec["kind"]:
     concat - first half of the frames from `path`, second half from `path2` (may be the same file); spec "aug1" / "aug2" ("flip" mirror,
              "rev" backwards) per part: which event comes first; direction A then direction B
     side   - `path` (left) and `path2` (right) at the same time, the middle vertical band of each: both events happen together
     stack  - one clip twice, above and below (the middle horizontal band of each), frames evenly over spec "top" / "bottom" = [a, b]
              fractions of the span: one plays faster than, slower than or as fast as the other; with `path2`: two clips above / below
              at the same time (both events happen together)
     speed  - the span split 1:3 ('up'), 3:1 ('down') or 1:1 ('same'); 8 evenly spaced frames in each part, so the second half moves
              3 times as fast / a third as fast / as fast as the first (3x: clips' own pace varies up to ~2x between halves)
     freeze - the middle frame held: nothing moves
     rotate - (10-17) a central disc turned by spec "deg" per frame (> 0 clockwise on screen) over a frozen first frame (still_bg) or the clip
     pingpong - (10-17) a short stretch played forward and back spec "n" times: n back-and-forth movements
     rate   - (10-17) 16 frames over spec "frac" of the window: slow motion (small) / normal / time-lapse (whole long window)
     jump   - 16 evenly spaced frames over the first half of the span ('cut' false), or over its first and last quarters ('cut' true: the
              middle half is missing, same spacing between frames)
    Frame times are replaced by even ones (0.25 s apart), so the answer cannot be read off the time codes. -> (video [3, T, H, W], times [T])"""
    import decord
    k, h = spec["kind"], num_frames // 2
    fx = lambda v, a: v.flip(-1) if a == "flip" else v.flip(1) if a == "rev" else v
    def span(p, w):
        vr = decord.VideoReader(p, num_threads=1); end = len(vr) / float(vr.get_avg_fps()); del vr
        return (0.0, end) if w is None else (max(0.0, float(w[0])), min(end, float(w[1])))
    even = lambda a, b, n: a + (np.arange(n) + 0.5) * (b - a) / n
    if k == "concat":
        a = fx(load_clip(path, h, pair_gap_sec=pair_gap_sec, window=window)[0], spec.get("aug1", ""))
        b = fx(load_clip(path2, h, pair_gap_sec=pair_gap_sec, window=window2)[0], spec.get("aug2", ""))
        video = torch.cat([a, b], 1)
    elif k == "concat_even":                            # (11-12, eval only) the clip played twice one after the other, each part's frames evenly
        t0, t1 = window if window is not None else span(path, None)          # spaced over its own span (playback rate = span length), like "stack"
        u0, u1 = window2
        video = torch.cat([load_clip(path, h, times=even(t0, t1, h))[0], load_clip(path2 or path, h, times=even(u0, u1, h))[0]], 1)
    elif k == "side":
        a = load_clip(path, num_frames, pair_gap_sec=pair_gap_sec, window=window)[0]
        b = load_clip(path2, num_frames, pair_gap_sec=pair_gap_sec, window=window2)[0]
        q = a.shape[-1] // 4; video = torch.cat([a[..., q:3 * q], b[..., q:3 * q]], -1)
    elif k == "stack":                                  # with path2: two different clips above / below, at the same time (TempCompass layout)
        pan = []
        for p_, w_, f in ((path, window, spec["top"]), (path2 or path, window2 if path2 else window, spec["bottom"])):
            t0, t1 = span(p_, w_); D = t1 - t0
            pan.append(load_clip(p_, num_frames, times=even(t0 + D * f[0], t0 + D * f[1], num_frames))[0])
        q = pan[0].shape[-2] // 4; video = torch.cat([x[..., q:3 * q, :] for x in pan], -2)
    elif k == "rotate":                                 # (10-17) a disc in the middle turns: spec "deg" degrees per frame (> 0 clockwise on screen)
        import torch.nn.functional as F_
        t0, t1 = span(path, window); video = load_clip(path, num_frames, times=even(t0, t1, num_frames))[0]
        T, H, W = video.shape[1:]; d1 = float(spec["deg"]); d2 = float(spec.get("deg2", d1))          # deg2: per-frame turn in the second half
        steps = torch.tensor([d1 if t < T // 2 else d2 for t in range(T)], dtype=torch.float32); steps[0] = 0
        ang = -torch.cumsum(steps, 0) * np.pi / 180                                          # the grid turns opposite to the image
        c, s_ = torch.cos(ang), torch.sin(ang)
        theta = torch.stack([torch.stack([c, -s_, torch.zeros(T)], 1), torch.stack([s_, c, torch.zeros(T)], 1)], 1)   # rotate sampling grid
        grid = F_.affine_grid(theta, (T, 3, H, W), align_corners=False)
        rot = F_.grid_sample(video.transpose(0, 1), grid, align_corners=False, padding_mode="border").transpose(0, 1)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"); disc = ((yy ** 2 + xx ** 2) < float(spec.get("r", 0.55)) ** 2).float()
        base = video[:, :1].expand(-1, T, -1, -1) if spec.get("still_bg", True) else video
        video = rot * disc + base * (1 - disc)
    elif k == "pingpong":                               # (10-17) a short stretch played forward and back spec "n" times (n back-and-forth cycles)
        t0, t1 = span(path, window); n = int(spec["n"]); a = float(spec.get("a", 0.3)); L = float(spec.get("len", 0.25))
        m0 = t0 + (t1 - t0) * a; m1 = min(t1, m0 + (t1 - t0) * L); per = num_frames / n; ph = np.arange(num_frames) / per % 1.0      # turning points not duplicated
        t = m0 + (m1 - m0) * (1 - np.abs(2 * ph - 1))     # triangle wave: forward then back, n times
        video = load_clip(path, num_frames, times=t)[0]
    elif k == "rate":                                   # (10-17) absolute playback rate: 16 frames over spec "frac" of the window
        t0, t1 = span(path, window); m = (t0 + t1) / 2; L = (t1 - t0) * float(spec["frac"])
        video = load_clip(path, num_frames, times=even(max(t0, m - L / 2), min(t1, m + L / 2), num_frames))[0]
    elif k == "camera":                                 # (10-25) a known camera move over a static-camera source: a square window of spec
        # "base" x the short side (default 0.7) slides / scales over the frames at source resolution, each crop resized to the output size.
        # spec "move": pan_left / pan_right (window centre moves left / right: scene content drifts the other way), tilt_up / tilt_down,
        # zoom_in (window shrinks to 60 %) / zoom_out, static; "amt" = share of the free range travelled (default 0.8)
        import torch.nn.functional as F_
        t0, t1 = span(path, window); vr = decord.VideoReader(path, num_threads=1); fps = float(vr.get_avg_fps()); n = len(vr)
        idx = np.clip(np.round(even(t0, t1, num_frames) * fps).astype(int), 0, n - 1)
        fr = torch.from_numpy(vr.get_batch(idx.tolist()).asnumpy()).permute(3, 0, 1, 2).float() / 255; del vr     # [3, T, H, W]
        H, W = fr.shape[-2:]; base = min(H, W) * float(spec.get("base", 0.7)); amt = float(spec.get("amt", 0.8)); mv = spec["move"]; T = fr.shape[1]
        out = []
        for i in range(T):
            f = i / max(T - 1, 1); sz = base; cx, cy = W / 2, H / 2
            if mv in ("pan_left", "pan_right"): rx = (W - base) / 2 * amt; cx = W / 2 + rx * ((1 - 2 * f) if mv == "pan_left" else (2 * f - 1))
            elif mv in ("tilt_up", "tilt_down"): ry = (H - base) / 2 * amt; cy = H / 2 + ry * ((1 - 2 * f) if mv == "tilt_up" else (2 * f - 1))
            elif mv == "zoom_in": sz = base * (1 - 0.4 * f)
            elif mv == "zoom_out": sz = base * (0.6 + 0.4 * f)
            x0 = int(round(min(max(cx - sz / 2, 0), W - sz))); y0 = int(round(min(max(cy - sz / 2, 0), H - sz))); s_ = int(round(sz))
            out.append(F_.interpolate(fr[:, i:i + 1, y0:y0 + s_, x0:x0 + s_].transpose(0, 1), size=(384, 384), mode="bilinear", antialias=True, align_corners=False))
        video = (torch.cat(out, 0).transpose(0, 1).clamp(0, 1) - MEAN) / STD
    elif k == "roll":                                   # (11-05) the view turns about its centre (camera roll): content rotates ON SCREEN by a known
        # angle schedule, spec "dir": cw / ccw (0 -> +-deg), cwccw / ccwcw (0 -> +-deg -> 0, turns back half way), none; "deg" total turn.
        # Window = centre square at 0.68 of the short side, so it stays inside the frame at any angle (0.68 * sqrt 2 < 1): no corner cue.
        # theta > 0 = clockwise on screen (y down); checked on a dot (logs/zz_rot_premise.py warp, same maths).
        import torch.nn.functional as F_, math
        t0, t1 = span(path, window); vr = decord.VideoReader(path, num_threads=1); fps = float(vr.get_avg_fps()); n = len(vr)
        idx = np.clip(np.round(even(t0, t1, num_frames) * fps).astype(int), 0, n - 1)
        fr = torch.from_numpy(vr.get_batch(idx.tolist()).asnumpy()).permute(0, 3, 1, 2).float() / 255; del vr     # [T, 3, H, W]
        H, W = fr.shape[-2:]; m = min(H, W); fr = fr[:, :, (H - m) // 2:(H - m) // 2 + m, (W - m) // 2:(W - m) // 2 + m]
        fr = F_.interpolate(fr, size=(384, 384), mode="bilinear", antialias=True, align_corners=False); T = fr.shape[0]
        f = np.linspace(0, 1, T); tri = 1 - np.abs(2 * f - 1)
        shape = {"cw": f, "ccw": -f, "cwccw": tri, "ccwcw": -tri, "none": 0 * f}[spec["dir"]]
        th = torch.tensor(float(spec.get("deg", 60.0)) * shape * math.pi / 180, dtype=torch.float32); c_, s_ = torch.cos(th), torch.sin(th); sc = 0.68
        A_ = torch.zeros(T, 2, 3); A_[:, 0, 0] = sc * c_; A_[:, 0, 1] = sc * s_; A_[:, 1, 0] = -sc * s_; A_[:, 1, 1] = sc * c_
        g_ = F_.affine_grid(A_, (T, 3, 384, 384), align_corners=False)
        out = F_.grid_sample(fr, g_, mode="bilinear", padding_mode="border", align_corners=False)
        video = (out.transpose(0, 1).clamp(0, 1) - MEAN) / STD
    elif k == "freeze":
        t0, t1 = span(path, window)
        video = load_clip(path, 1, times=np.array([(t0 + t1) / 2]))[0].expand(-1, num_frames, -1, -1).contiguous()
    else:
        t0, t1 = span(path, window); D = t1 - t0
        if k == "speed":
            m = t0 + D * {"up": 1 / 4, "down": 3 / 4, "same": 1 / 2}[spec["change"]]
            t = np.concatenate([even(t0, m, h), even(m, t1, h)])
        elif k == "jump":
            q = D / 4; t = np.concatenate([even(t0, t0 + q, h), even(t0 + 3 * q, t1, h) if spec["cut"] else even(t0 + q, t0 + 2 * q, h)])
        else:
            raise ValueError(k)
        video = load_clip(path, num_frames, times=t)[0]
    return video, (torch.arange(num_frames, dtype=torch.float32) + 0.5) * 0.25


def memory_frames(path, window=None, crop_center=None, kmax=4, clip_s=8.0, num_frames=16, size=384, pair_gap_sec=0.25):
    """(10-17) temporal-memory clips, decoded live: the window (whole video by default) cut into K = clamp(ceil(span / clip_s), 1, kmax)
    equal sub-windows, 16 paired-uniform frames each (as load_clip), the video opened ONCE, the overview's 384 crop -> (uint8
    [K * 16, size, size, 3], clip centres [K] seconds). Normalising and resizing to the memory resolution happen on the GPU."""
    import decord
    vr = decord.VideoReader(path, num_threads=1); fps = float(vr.get_avg_fps()); n = len(vr)
    t0, t1 = (0.0, n / fps) if window is None else (max(0.0, float(window[0])), min(n / fps, float(window[1])))
    K = int(min(max(int(np.ceil((t1 - t0) / clip_s)), 1), kmax)); gap = max(pair_gap_sec, 1.0 / fps); edges = np.linspace(t0, t1, K + 1); T = []
    for k in range(K):
        w = max(edges[k + 1] - edges[k], 1e-3); T.append(edges[k] + paired_uniform_times(w, num_frames, min(gap, w / 2)))
    idx = np.clip(np.round(np.concatenate(T) * fps).astype(int), 0, n - 1); u, inv = np.unique(idx, return_inverse=True)
    fr = torch.from_numpy(vr.get_batch(u.tolist()).asnumpy())[torch.from_numpy(inv)]          # [K*16, H, W, 3] uint8
    h, w_ = fr.shape[1:3]
    if min(h, w_) != size:                                                                     # (rare: low-resolution sources)
        sc = size / min(h, w_); x = fr.permute(0, 3, 1, 2).float()
        fr = F.interpolate(x, size=(round(h * sc), round(w_ * sc)), mode="bilinear", antialias=True, align_corners=False).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        h, w_ = fr.shape[1:3]
    cx, cy = crop_center if crop_center is not None else (0.5, 0.5)
    x0 = int(min(max(round(cx * w_ - size / 2), 0), w_ - size)); y0 = int(min(max(round(cy * h - size / 2), 0), h - size))
    return fr[:, y0:y0 + size, x0:x0 + size].contiguous(), torch.tensor((edges[:-1] + edges[1:]) / 2, dtype=torch.float32)


def _seq_decode(segs, root, gtimes, size=384):
    """(10-18) frames of a sequence of segments [[path, t0, t1], ...] (path absolute, or relative to `root`) laid end to end on one
    timeline, at the global times `gtimes` -> uint8 [n, size, size, 3] (short side to `size`, centre crop), each file opened once"""
    import decord
    dur = [max(float(t1) - float(t0), 1e-3) for _, t0, t1 in segs]; off = np.concatenate([[0.0], np.cumsum(dur)])
    gt = np.clip(np.asarray(gtimes, dtype=np.float64), 0.0, off[-1] - 1e-6); which = np.clip(np.searchsorted(off, gt, side="right") - 1, 0, len(segs) - 1)
    out = torch.zeros(len(gt), size, size, 3, dtype=torch.uint8)
    for j in np.unique(which):
        p, t0, _ = segs[j]; p = p if p.startswith("/") else f"{root}/{p}"
        vr = decord.VideoReader(p, num_threads=1); fps = float(vr.get_avg_fps()); n = len(vr); sel = np.nonzero(which == j)[0]
        idx = np.clip(np.round((float(t0) + gt[sel] - off[j]) * fps).astype(int), 0, n - 1)
        fr = torch.from_numpy(vr.get_batch(idx.tolist()).asnumpy()); h, w = fr.shape[1:3]
        if min(h, w) != size:
            sc = size / min(h, w); x = fr.permute(0, 3, 1, 2).float()
            fr = F.interpolate(x, size=(round(h * sc), round(w * sc)), mode="bilinear", antialias=True, align_corners=False).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
            h, w = fr.shape[1:3]
        y0, x0 = (h - size) // 2, (w - size) // 2; out[sel] = fr[:, y0:y0 + size, x0:x0 + size]
    return out


def seq_clip(segs, root, num_frames, pair_gap_sec=0.25, size=384):
    """(10-18) a long video BUILT from segments (long-video questions: where / when / order / how many times): `num_frames` paired-uniform
    frames over the joined timeline -> (video [3, T, size, size] normalised, times [T] = positions on the joined timeline, seconds)"""
    total = float(sum(max(float(b) - float(a), 1e-3) for _, a, b in segs)); t = paired_uniform_times(total, num_frames, pair_gap_sec)
    u8 = _seq_decode(segs, root, t, size)
    return (u8.permute(3, 0, 1, 2).float() / 255 - MEAN) / STD, torch.as_tensor(t, dtype=torch.float32)


def seq_memory_frames(segs, root, kmax=4, clip_s=16.0, num_frames=16, size=384, pair_gap_sec=0.25):
    """(10-18) memory clips of a joined sequence, as memory_frames on one timeline -> (uint8 [K*16, size, size, 3], clip centres [K])"""
    total = float(sum(max(float(b) - float(a), 1e-3) for _, a, b in segs))
    K = int(min(max(int(np.ceil(total / clip_s)), 1), kmax)); edges = np.linspace(0.0, total, K + 1); T = []
    for k in range(K):
        w = max(edges[k + 1] - edges[k], 1e-3); T.append(edges[k] + paired_uniform_times(w, num_frames, min(pair_gap_sec, w / 2)))
    return _seq_decode(segs, root, np.concatenate(T), size), torch.tensor((edges[:-1] + edges[1:]) / 2, dtype=torch.float32)


def region_to_grid(region, crop_box, grid: int = 24):
    """Normalised source-frame region (x0,y0,x1,y1) -> token-grid rows/cols (y0, y1, x0, x1) inside the crop, at least one
    patch wide. Empty/None region -> the whole grid."""
    if region is None or len(region) != 4:
        return [0, grid, 0, grid]
    cx0, cy0, cx1, cy1 = crop_box
    def to(v, a, b):
        return min(max((v - a) / (b - a), 0.0), 1.0) * grid
    x0, x1 = to(region[0], cx0, cx1), to(region[2], cx0, cx1)
    y0, y1 = to(region[1], cy0, cy1), to(region[3], cy0, cy1)
    gx0, gy0 = int(min(x0, grid - 1)), int(min(y0, grid - 1))
    gx1, gy1 = max(gx0 + 1, int(-(-x1 // 1))), max(gy0 + 1, int(-(-y1 // 1)))
    return [gy0, min(gy1, grid), gx0, min(gx1, grid)]
