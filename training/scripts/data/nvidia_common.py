"""Shared helpers for the NVIDIA PhysicalAI synthetic-physics manifests (nvidia_index / nvidia_geometry_check / nvidia_pairs).

Conventions established empirically in manifests/nvidia/report.md (all checked on real files, see that report):
  * physics `<Cam>_com.npz['data']` is the object position (not a displacement) in the camera frame of that camera, USD/OpenGL
    convention (x right, y up, camera looks along -z). `_velocity.npz['velocities']` (== ['data']) uses the same frame and row order.
  * `cameras/<world>/<Cam>.json['usd_transform']` is the camera-to-world 4x4 matrix, row-vector convention (p_w = p_c @ R + t,
    R = M[:3,:3], t = M[3,:3]). It was constant over frames in every clip checked by nvidia_index.py (column cam_static).
  * Pinhole projection: u = (0.5 + fxn * x / -z) * W, v = 0.5 * H - fxn * W * y / -z  (square pixels; the json's second focal
    value fyn is NOT used, it is inconsistent with the 16:9 render; see geometry report).
  * The npz row order is NOT the order of npz['segmentation_colors']; row identity is resolved from the scene USDA initial poses.
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # shared host: one BLAS thread per process
    _os.environ.setdefault(_v, "1")

import io, json, os, re, tarfile
import numpy as np

ROOT = "/data/llama_vjepa_lm"
DATA = f"{ROOT}/data/nvidia_physical_interaction"
OUT = f"{ROOT}/manifests/nvidia"
PREP = f"{ROOT}/prep/nvidia_rgb_384"
SCENARIOS = ["ball_mixer", "billiards", "bowling", "dominoes", "objects_falling", "obstruction",
             "rolling_ramp_objects", "rolling_ramp_obstruct", "towers", "wrecking_ball"]
MODALITIES = ["physics", "cameras", "scene", "captions", "segmentation", "videos"]
DERIVED_VIDEO = re.compile(r"_(com|rot|spin|velocity)_per_clip\.mp4$")
WORLD_RE = re.compile(r"^(?P<scenario>[a-z_]+?)_(?P<hash>[0-9a-f]{8})_(?P<variant>\d+)$")


def split_world(world):
    m = WORLD_RE.match(world)
    if not m:
        raise ValueError(f"unexpected world dir name {world!r}")
    return m["scenario"], m["hash"], int(m["variant"])


def tar_paths(modality, scenario):
    d = f"{DATA}/{modality}/{scenario}"
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".tar"))


def listing_path(tar):
    """Where `nvidia_index.py list` stores the member listing of a tar."""
    modality = os.path.basename(os.path.dirname(os.path.dirname(tar)))
    return f"{OUT}/tar_listing/{modality}/{os.path.basename(tar)[:-4]}.parquet"


_FH = {}


def read_member(tar, offset, size):
    """Random access read of one tar member (per-process cache of open read-only handles, at most 64)."""
    f = _FH.get(tar)
    if f is None:
        if len(_FH) >= 64:
            _FH.pop(next(iter(_FH))).close()
        f = _FH[tar] = open(tar, "rb")
    f.seek(offset)
    return f.read(size)


def load_npz_bytes(b):
    z = np.load(io.BytesIO(b), allow_pickle=False)
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ camera geometry
def cam_to_world(p_cam, M):
    """p_cam [..., 3] in USD camera frame -> world, with M the 4x4 usd_transform (row-vector convention)."""
    M = np.asarray(M, dtype=np.float64).reshape(4, 4)
    return p_cam @ M[:3, :3] + M[3, :3]


def world_to_cam(p_w, M):
    M = np.asarray(M, dtype=np.float64).reshape(4, 4)
    return (p_w - M[3, :3]) @ M[:3, :3].T


def project(p_cam, fxn, W=1920, H=1080, ppx=0.5, ppy=0.5):
    """USD camera frame -> pixel (u, v), square pixels from the horizontal normalised focal length. Returns u, v, in_front."""
    x, y, z = p_cam[..., 0], p_cam[..., 1], p_cam[..., 2]
    depth = -z
    with np.errstate(divide="ignore", invalid="ignore"):
        u = ppx * W + fxn * W * x / depth
        v = ppy * H - fxn * W * y / depth
    return u, v, depth > 1e-6


# ------------------------------------------------------------------ USDA top-level prims
_DEF = re.compile(r'\n    (?:def|over) (?:(\w+) )?"(\w+)"')
_TR = re.compile(r'\n        (?:double3|float3) xformOp:translate = \(([^)]*)\)')
_VEL = re.compile(r'\n        vector3f physics:velocity = \(([^)]*)\)')


def usda_top_prims(txt):
    """{'/World/<name>': {'type', 'translate', 'velocity'}} for direct children of /World that carry a translate op."""
    out = {}
    ms = list(_DEF.finditer(txt))
    for k, m in enumerate(ms):
        end = ms[k + 1].start() if k + 1 < len(ms) else len(txt)
        seg = txt[m.end():end]
        t = _TR.search(seg)
        if not t or m.group(1) == "Camera":
            continue
        v = _VEL.search(seg)
        out["/World/" + m.group(2)] = {
            "type": m.group(1) or "",
            "translate": np.array([float(x) for x in t.group(1).split(",")]),
            "velocity": np.array([float(x) for x in v.group(1).split(",")]) if v else None,
        }
    return out


def prim_top(path):
    """'/World/Ball_02_glass/sphere/geometry' -> '/World/Ball_02_glass'."""
    return "/".join(path.split("/")[:3])


MATERIALS = ("clear_plastic", "plastic", "glass", "metal", "wood", "rubber", "stone", "ceramic", "paper", "foam", "steel")


def slot_key(prim_path):
    """Variant-independent object slot inside one scene template: strips the randomised material token.
    '/World/Ball_02_glass' -> 'Ball_02', '/World/Pillar_L0_A0_Left_plastic' -> 'Pillar_L0_A0_Left', '/World/CueBall' -> 'CueBall'.
    Random-asset names ('Prop_lemon_02_30887', 'S0_Stone_Cube_012') are returned unchanged and are NOT stable slots."""
    name = path_name = prim_top(prim_path).split("/")[-1]
    for mat in MATERIALS:
        if name.lower().endswith("_" + mat):
            return name[: -len(mat) - 1]
    return path_name
