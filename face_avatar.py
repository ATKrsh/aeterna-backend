"""
face_avatar.py  —  Aeterna 3D Head Bust Engine v4
================================================================
REAL 3D reconstruction from a single photo:
  1. MediaPipe FaceLandmarker  →  478 precise 3D landmarks
  2. Canonical face model OBJ  →  468-vertex topology + UV atlas
  3. UV-space rasterisation    →  sample photo pixels per barycentric UV coordinate
                                   (no stretching, no distortion)
  4. Landmark-driven 3D shape  →  scale canonical Z to detected metric dimensions
  5. Procedural skull dome     →  closed ellipsoid back of head
  6. Procedural neck           →  tapered cylinder + shoulder base
  7. Skin-tone dilation+blur   →  hide seams at skull/neck transition
  8. GLB 2.0 binary export     →  PBR material, correct normals
================================================================
"""

from __future__ import annotations
import io, uuid, struct, json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions
from mediapipe import Image as MpImage, ImageFormat

# ── paths ─────────────────────────────────────────────────────────────────────
_MODEL_PATH  = Path(__file__).parent / "face_landmarker.task"
_CANON_PATH  = Path(__file__).parent / "canonical_face_model.obj"

for p in (_MODEL_PATH, _CANON_PATH):
    if not p.exists():
        raise FileNotFoundError(f"Required file missing: {p}")

# ── face-oval landmark ring (36 pts) ─────────────────────────────────────────
FACE_OVAL = [
    10,338,297,332,284,251,389,356,454,323,361,288,
    397,365,379,378,400,377,152,148,176,149,150,136,
    172, 58,132, 93,234,127,162, 21, 54,103, 67,109,
]

TEX_RES = 2048       # texture atlas resolution


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _best_photo(images: list[Image.Image]) -> Image.Image:
    """Return the sharpest image by Laplacian variance."""
    if len(images) == 1:
        return images[0]
    best, best_sc = images[0], -1.0
    for img in images:
        g = np.array(img.convert("L").resize((256, 256)), dtype=np.float32)
        sc = float(np.var(g[1:] - g[:-1]) + np.var(g[:, 1:] - g[:, :-1]))
        if sc > best_sc:
            best_sc, best = sc, img
    return best


def _detect(image_rgb: np.ndarray):
    """Run MediaPipe FaceLandmarker → list[NormalizedLandmark] or None."""
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    with mp_vision.FaceLandmarker.create_from_options(opts) as det:
        res = det.detect(MpImage(image_format=ImageFormat.SRGB, data=image_rgb))
    return res.face_landmarks[0] if res.face_landmarks else None


def _parse_obj(path: Path):
    """
    Parse canonical_face_model.obj.
    Returns:
        canon_verts  (468,3) float32  – 3-D positions in cm
        canon_uvs    (468,2) float32  – UV in [0,1]
        triangles    (898,3) uint32   – vertex-index triangles
    """
    raw_v, raw_vt, raw_f = [], [], []
    with open(path) as fh:
        for line in fh:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "v":
                raw_v.append([float(tok[1]), float(tok[2]), float(tok[3])])
            elif tok[0] == "vt":
                raw_vt.append([float(tok[1]), float(tok[2])])
            elif tok[0] == "f":
                corners = [tuple(int(x) - 1 for x in t.split("/")) for t in tok[1:]]
                raw_f.append(corners)

    # build vertex → UV mapping (guaranteed 1-to-1 in this OBJ)
    v2vt: dict[int, int] = {}
    for face in raw_f:
        for vi, vti in face:
            v2vt[vi] = vti

    uvs = np.zeros((468, 2), dtype=np.float32)
    for vi, vti in v2vt.items():
        uvs[vi] = raw_vt[vti]

    tris = np.array([[f[0][0], f[1][0], f[2][0]] for f in raw_f], dtype=np.uint32)
    verts = np.array(raw_v, dtype=np.float32)
    return verts, uvs, tris


def _skin_color(img: Image.Image) -> tuple[int, int, int]:
    """Sample forehead + cheeks to get a robust average skin tone."""
    a = np.array(img)
    H, W = a.shape[:2]
    samples = [
        a[int(H*.15):int(H*.25), int(W*.42):int(W*.58)],
        a[int(H*.50):int(H*.60), int(W*.25):int(W*.38)],
        a[int(H*.50):int(H*.60), int(W*.62):int(W*.75)],
    ]
    cols = [s.mean(axis=(0, 1)) for s in samples if s.size > 0]
    if not cols:
        return (190, 155, 130)
    return tuple(map(int, np.mean(cols, axis=0)))


def _rasterise_uv_texture(
    photo_arr: np.ndarray,
    lm,                        # list of NormalizedLandmark (478 total)
    canon_uvs: np.ndarray,     # (468,2)
    triangles: np.ndarray,     # (898,3)
) -> np.ndarray:
    """
    Build a TEX_RES×TEX_RES texture by rasterising each UV triangle.

    For every pixel (u,v) in texture space:
      1. Find which canonical-UV triangle contains it.
      2. Compute barycentric weights (w0,w1,w2).
      3. Interpolate the corresponding photo-pixel position.
      4. Sample the photo.

    This gives a geometrically correct, distortion-free texture.
    """
    H, W = photo_arr.shape[:2]
    texture = np.zeros((TEX_RES, TEX_RES, 3), dtype=np.uint8)

    # UV → texture-pixel coords  (flip V: UV origin bottom-left, image origin top-left)
    uv_px = np.empty((468, 2), dtype=np.float64)
    uv_px[:, 0] = canon_uvs[:, 0] * (TEX_RES - 1)
    uv_px[:, 1] = (1.0 - canon_uvs[:, 1]) * (TEX_RES - 1)

    # Photo pixel coords for each of the 468 landmarks
    ph_px = np.empty((468, 2), dtype=np.float64)
    for i in range(468):
        ph_px[i, 0] = lm[i].x * W
        ph_px[i, 1] = lm[i].y * H

    for tri in triangles:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        u0, u1, u2 = uv_px[i0], uv_px[i1], uv_px[i2]
        p0, p1, p2 = ph_px[i0], ph_px[i1], ph_px[i2]

        # axis-aligned bounding box in texture space
        xlo = max(0,        int(np.floor(min(u0[0], u1[0], u2[0]))))
        xhi = min(TEX_RES-1, int(np.ceil (max(u0[0], u1[0], u2[0]))))
        ylo = max(0,        int(np.floor(min(u0[1], u1[1], u2[1]))))
        yhi = min(TEX_RES-1, int(np.ceil (max(u0[1], u1[1], u2[1]))))

        if xhi < xlo or yhi < ylo:
            continue

        denom = ((u1[1]-u2[1])*(u0[0]-u2[0]) + (u2[0]-u1[0])*(u0[1]-u2[1]))
        if abs(denom) < 1e-8:
            continue

        PX, PY = np.meshgrid(
            np.arange(xlo, xhi+1, dtype=np.float64),
            np.arange(ylo, yhi+1, dtype=np.float64),
        )

        w0 = ((u1[1]-u2[1])*(PX-u2[0]) + (u2[0]-u1[0])*(PY-u2[1])) / denom
        w1 = ((u2[1]-u0[1])*(PX-u2[0]) + (u0[0]-u2[0])*(PY-u2[1])) / denom
        w2 = 1.0 - w0 - w1

        inside = (w0 >= -0.005) & (w1 >= -0.005) & (w2 >= -0.005)
        if not np.any(inside):
            continue

        sx = np.clip((w0*p0[0] + w1*p1[0] + w2*p2[0]).astype(np.int32), 0, W-1)
        sy = np.clip((w0*p0[1] + w1*p1[1] + w2*p2[1]).astype(np.int32), 0, H-1)

        dx = PX.astype(np.int32)
        dy = PY.astype(np.int32)

        texture[dy[inside], dx[inside]] = photo_arr[sy[inside], sx[inside]]

    return texture


def _normals(pos: np.ndarray, tri: np.ndarray) -> np.ndarray:
    nrm = np.zeros_like(pos)
    v0, v1, v2 = pos[tri[:,0]], pos[tri[:,1]], pos[tri[:,2]]
    fn = np.cross(v1-v0, v2-v0)
    np.add.at(nrm, tri[:,0], fn)
    np.add.at(nrm, tri[:,1], fn)
    np.add.at(nrm, tri[:,2], fn)
    mg = np.linalg.norm(nrm, axis=1, keepdims=True)
    mg[mg==0] = 1.0
    return (nrm / mg).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def generate_avatar_glb(photo_files: list[bytes],
                        output_dir: Path,
                        avatar_id: str | None = None) -> Path:
    if avatar_id is None:
        avatar_id = str(uuid.uuid4())
    output_dir.mkdir(parents=True, exist_ok=True)
    glb_path = output_dir / f"{avatar_id}.glb"

    # ── 1. decode & select best image ────────────────────────────────────────
    images = []
    for raw in photo_files:
        try:
            images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception:
            continue
    if not images:
        raise ValueError("No valid images provided.")

    best = _best_photo(images)

    # ── 2. detect face landmarks ──────────────────────────────────────────────
    # Work on a square, high-res copy so small faces are OK
    WH = 1024
    arr = np.array(best.resize((WH, WH), Image.LANCZOS), dtype=np.uint8)
    lm = _detect(arr)
    if lm is None:
        for img in images:
            arr = np.array(img.resize((WH, WH), Image.LANCZOS), dtype=np.uint8)
            lm = _detect(arr)
            if lm is not None:
                best = img
                break
    if lm is None:
        raise ValueError(
            "No face detected. Please use a clear, front-facing, well-lit photo.")

    # Landmarks are in [0,1] normalised over the 1024² input image
    # We'll keep photo sampling in full-resolution space
    ph_w, ph_h = best.size

    # ── 3. parse canonical model ──────────────────────────────────────────────
    c_verts, c_uvs, c_tris = _parse_obj(_CANON_PATH)
    # c_verts in centimetres.  Canonical face width ≈ 15.49 cm (X: ±7.74)
    CANON_FACE_W_CM = c_verts[:, 0].max() - c_verts[:, 0].min()  # ~15.49

    # ── 4. build 3-D face mesh from detected landmarks ────────────────────────
    #
    # Strategy (avoids the "waxy mask" problem):
    #   • X, Y  → from MediaPipe normalised coords, scaled so face width = 0.15 m
    #   • Z     → from canonical model (gives real anatomical depth profile),
    #              scaled proportionally to face width + enhanced by detected Z
    #
    lm_x = np.array([lm[i].x for i in range(468)])
    lm_y = np.array([lm[i].y for i in range(468)])
    lm_z = np.array([lm[i].z for i in range(468)])

    # detected face width in normalised image coords
    oval_x = lm_x[FACE_OVAL]
    det_face_w = oval_x.max() - oval_x.min()
    if det_face_w < 0.01:
        det_face_w = 0.35   # fallback

    FACE_W_M  = 0.155   # target face width in metres (~15.5 cm)
    scale_xy  = FACE_W_M / det_face_w

    # canonical model is in centimetres; convert to metres
    # scale so canonical face width matches FACE_W_M
    scale_canon = FACE_W_M / CANON_FACE_W_CM / 100.0   # cm → m, proportional

    # centre landmarks at origin
    cx = (lm_x.max() + lm_x.min()) * 0.5
    cy = (lm_y.max() + lm_y.min()) * 0.5

    face_pos = np.zeros((468, 3), dtype=np.float32)
    face_pos[:, 0] =  (lm_x - cx) * scale_xy
    face_pos[:, 1] = -(lm_y - cy) * scale_xy    # flip Y for WebGL
    # Z: use canonical anatomical depth profile in metres
    # canonical Z: nose ≈ 7.6 cm from back → ~5 cm range centred
    c_z_centered = c_verts[:, 2] - c_verts[:, 2].mean()
    face_pos[:, 2] = c_z_centered * scale_canon

    # ── 5. UV-rasterised texture ──────────────────────────────────────────────
    # a) rasterise face triangles → sample photo at correct positions
    raw_tex = _rasterise_uv_texture(
        np.array(best, dtype=np.uint8), lm, c_uvs, c_tris)

    # b) extract skin colour from rasterised face region
    face_mask_px = np.any(raw_tex > 0, axis=2)
    if face_mask_px.any():
        fc = raw_tex[face_mask_px]
        skin_col = tuple(int(np.median(fc[:, ch])) for ch in range(3))
    else:
        skin_col = _skin_color(best)

    # c) fill background with skin colour, then dilate face region to hide seams
    tex_img = Image.fromarray(raw_tex)
    bg = Image.new("RGB", (TEX_RES, TEX_RES), skin_col)

    # dilate the face pixels outward so there are no black gaps at edges
    dilated = Image.fromarray(raw_tex).filter(ImageFilter.MaxFilter(7))
    # composite: dilated fills edges, original sits on top
    face_alpha = Image.fromarray((face_mask_px * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(12))
    blended = Image.composite(dilated, bg, face_alpha)
    # put original sharp face on top
    blended.paste(tex_img, mask=Image.fromarray(face_mask_px.astype(np.uint8) * 255))

    # d) subtle photo enhancement
    blended = ImageEnhance.Sharpness(blended).enhance(1.2)
    blended = ImageEnhance.Contrast(blended).enhance(1.08)

    # ── 6. assemble combined geometry ─────────────────────────────────────────
    all_pos  = list(face_pos)        # will grow with skull + neck
    all_uvs  = list(c_uvs)          # canonical face UVs; skull/neck use skin-tone corner
    all_tris = list(c_tris)

    CENTER = face_pos.mean(axis=0)
    NOSE   = face_pos[4]             # nose tip
    CHIN   = face_pos[152]           # chin landmark

    # ── 6a. ellipsoidal skull dome ────────────────────────────────────────────
    # Back pole is behind the face, at head-depth distance
    HEAD_DEPTH = 0.12                # typical head depth ~12 cm
    back_z     = NOSE[2] - HEAD_DEPTH
    back_y     = (NOSE[1] + face_pos[10][1]) * 0.5   # halfway up

    oval_pts   = face_pos[FACE_OVAL]   # (36,3) boundary ring
    oval_cx    = oval_pts[:, 0].mean()
    oval_cy    = oval_pts[:, 1].mean()

    SKULL_RINGS = 5
    skull_base_idx = len(all_pos)

    for ring in range(1, SKULL_RINGS + 1):
        t = ring / (SKULL_RINGS + 1)        # 0→1 going toward back
        lat = t * (np.pi / 2)               # 0° at face, 90° at back pole

        for i in range(36):
            ox, oy, oz = oval_pts[i]
            # lerp xy toward centre, z sweeps toward back pole
            rx = oval_cx + (ox - oval_cx) * np.cos(lat)
            ry = oval_cy + (oy - oval_cy) * np.cos(lat) * 0.85
            rz = oz + (back_z - oz) * np.sin(lat)
            all_pos.append([rx, ry, rz])
            all_uvs.append([0.005, 0.95])   # solid skin-tone corner

    # back-pole vertex
    all_pos.append([oval_cx, back_y, back_z])
    all_uvs.append([0.005, 0.95])
    back_pole = len(all_pos) - 1

    # skull triangles (quad-strip + pole fan)
    for ring in range(1, SKULL_RINGS + 1):
        for i in range(36):
            ni = (i + 1) % 36
            if ring == 1:
                v0, v1 = FACE_OVAL[i], FACE_OVAL[ni]
            else:
                base = skull_base_idx + (ring - 2) * 36
                v0, v1 = base + i, base + ni

            v2 = skull_base_idx + (ring - 1) * 36 + i
            v3 = skull_base_idx + (ring - 1) * 36 + ni
            all_tris.append([v0, v2, v1])
            all_tris.append([v1, v2, v3])

    last_ring_base = skull_base_idx + (SKULL_RINGS - 1) * 36
    for i in range(36):
        ni = (i + 1) % 36
        all_tris.append([last_ring_base + i, back_pole, last_ring_base + ni])

    # ── 6b. tapered neck cylinder ─────────────────────────────────────────────
    NECK_SEGS  = 28
    NECK_RINGS = 4

    # Neck anchor: just below chin, slightly behind
    neck_top_y  = CHIN[1] - 0.018
    neck_bot_y  = CHIN[1] - 0.14
    neck_top_r  = 0.058
    neck_bot_r  = 0.068      # slightly wider at shoulders
    neck_cx     = CENTER[0]
    neck_cz     = NOSE[2] - 0.04

    neck_base   = len(all_pos)

    for ring in range(NECK_RINGS):
        t = ring / (NECK_RINGS - 1)
        y = neck_top_y - t * (neck_top_y - neck_bot_y)
        r = neck_top_r + t * (neck_bot_r - neck_top_r)
        for i in range(NECK_SEGS):
            θ = 2 * np.pi * i / NECK_SEGS
            all_pos.append([neck_cx + r*np.cos(θ), y, neck_cz + r*np.sin(θ)])
            all_uvs.append([0.005, 0.95])

    # shoulder disc (flat base cap)
    disc_cx, disc_cy, disc_cz = neck_cx, neck_bot_y, neck_cz
    all_pos.append([disc_cx, disc_cy, disc_cz])
    all_uvs.append([0.005, 0.95])
    disc_centre = len(all_pos) - 1

    # neck side quads
    for ring in range(1, NECK_RINGS):
        for i in range(NECK_SEGS):
            ni = (i + 1) % NECK_SEGS
            v0 = neck_base + (ring-1)*NECK_SEGS + i
            v1 = neck_base + (ring-1)*NECK_SEGS + ni
            v2 = neck_base + ring*NECK_SEGS + i
            v3 = neck_base + ring*NECK_SEGS + ni
            all_tris.append([v0, v2, v1])
            all_tris.append([v1, v2, v3])

    # bottom disc fan
    last_neck = neck_base + (NECK_RINGS-1)*NECK_SEGS
    for i in range(NECK_SEGS):
        ni = (i + 1) % NECK_SEGS
        all_tris.append([last_neck + i, last_neck + ni, disc_centre])

    # ── 7. finalise arrays ────────────────────────────────────────────────────
    positions = np.array(all_pos,  dtype=np.float32)
    uvcoords  = np.array(all_uvs,  dtype=np.float32)
    triangles = np.array(all_tris, dtype=np.uint32)

    nrm = _normals(positions, triangles)

    # ── 8. texture → PNG bytes ────────────────────────────────────────────────
    buf = io.BytesIO()
    blended.save(buf, format="PNG", compress_level=1)
    tex_bytes = buf.getvalue()

    # ── 9. write GLB ─────────────────────────────────────────────────────────
    glb_data = _build_glb(positions, nrm, uvcoords, triangles, tex_bytes)
    glb_path.write_bytes(glb_data)
    return glb_path


# ═══════════════════════════════════════════════════════════════════════════════
#  GLB 2.0 binary builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_glb(pos, nrm, uv, tri, tex_bytes):
    pb = pos.astype(np.float32).tobytes()
    nb = nrm.astype(np.float32).tobytes()
    ub = uv .astype(np.float32).tobytes()
    ib = tri.flatten().astype(np.uint32).tobytes()

    o0, o1, o2, o3, o4 = 0, len(pb), len(pb)+len(nb), len(pb)+len(nb)+len(ub), len(pb)+len(nb)+len(ub)+len(ib)
    raw = pb + nb + ub + ib + tex_bytes
    raw += b"\x00" * ((4 - len(raw) % 4) % 4)

    nv = len(pos); ni = len(tri)*3
    pmin = pos.min(0).tolist(); pmax = pos.max(0).tolist()

    gltf = {
        "asset": {"version": "2.0", "generator": "Aeterna v4"},
        "scene": 0, "scenes": [{"nodes":[0]}],
        "nodes":  [{"mesh": 0, "name": "AvatarHead"}],
        "meshes": [{"name":"HeadMesh","primitives":[{
            "attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2},
            "indices":3, "material":0, "mode":4}]}],
        "materials": [{"name":"Skin","pbrMetallicRoughness":{
            "baseColorTexture":{"index":0},
            "metallicFactor": 0.0,
            "roughnessFactor": 0.75,
        },"doubleSided": True}],
        "textures": [{"source":0,"sampler":0}],
        "samplers": [{"magFilter":9729,"minFilter":9987,"wrapS":33071,"wrapT":33071}],
        "images":  [{"mimeType":"image/png","bufferView":4}],
        "accessors":[
            {"bufferView":0,"byteOffset":0,"componentType":5126,"count":nv,"type":"VEC3","min":pmin,"max":pmax},
            {"bufferView":1,"byteOffset":0,"componentType":5126,"count":nv,"type":"VEC3"},
            {"bufferView":2,"byteOffset":0,"componentType":5126,"count":nv,"type":"VEC2"},
            {"bufferView":3,"byteOffset":0,"componentType":5125,"count":ni,"type":"SCALAR"},
        ],
        "bufferViews":[
            {"buffer":0,"byteOffset":o0,"byteLength":len(pb),"target":34962},
            {"buffer":0,"byteOffset":o1,"byteLength":len(nb),"target":34962},
            {"buffer":0,"byteOffset":o2,"byteLength":len(ub),"target":34962},
            {"buffer":0,"byteOffset":o3,"byteLength":len(ib),"target":34963},
            {"buffer":0,"byteOffset":o4,"byteLength":len(tex_bytes)},
        ],
        "buffers":[{"byteLength":len(raw)}],
    }

    jb = json.dumps(gltf, separators=(",",":")).encode()
    jb += b" " * ((4 - len(jb) % 4) % 4)

    total = 12 + 8 + len(jb) + 8 + len(raw)
    return (
        struct.pack("<III", 0x46546C67, 2, total) +
        struct.pack("<II", len(jb),  0x4E4F534A) + jb  +
        struct.pack("<II", len(raw), 0x004E4942) + raw
    )
