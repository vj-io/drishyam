"""Image intake, pyramid, saliency overview and the virtual-PTZ gaze.

Every pixel op on the hot path is a KleidiCV-accelerated OpenCV call (pyrDown,
cvtColor, GaussianBlur, Sobel, morphologyEx, inRange, warpPerspective, resize),
so the agent's own camera moves are the workload COOL speeds up on Graviton.
"""

from __future__ import annotations

import base64
import math
import struct
from dataclasses import dataclass

import cv2
import numpy as np

MAX_PIXELS = 25_000_000
MAX_DECODED_BYTES = 100_000_000  # what libpng/libjpeg may allocate; 16-bit RGBA at 25 MP would be 200 MB
MIN_SIDE = 64
LEVELS = 5  # L0 (full resolution) .. L4
OVERVIEW_LEVEL = 3
SHARPNESS_MIN = 12.0  # Laplacian variance below this is blur
GLARE_MAX = 0.20  # fraction of blown-out pixels
DARK_MAX = 0.50

_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


class ImageError(ValueError):
    """The upload is not an image we accept."""


_PNG_CHANNELS = {0: 1, 2: 3, 3: 3, 4: 2, 6: 4}  # colour type -> channels after decode (palette expands)


def sniff(data: bytes) -> tuple[int, int, float]:
    """Width, height and decoded bytes per pixel from the header, before any pixel is decoded."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        if len(data) < 26 or data[12:16] != b"IHDR":
            raise ImageError("corrupt PNG header")
        w, h = struct.unpack(">II", data[16:24])
        depth, colour = data[24], data[25]
        if colour not in _PNG_CHANNELS or depth not in (1, 2, 4, 8, 16):
            raise ImageError("unsupported PNG colour type or bit depth")
        return w, h, _PNG_CHANNELS[colour] * max(depth, 8) / 8
    if data[:3] == b"\xff\xd8\xff":
        i = 2
        while i + 9 <= len(data):
            if data[i] != 0xFF:
                raise ImageError("corrupt JPEG marker stream")
            marker = data[i + 1]
            if marker == 0xFF:  # fill byte
                i += 1
                continue
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            (seg_len,) = struct.unpack(">H", data[i + 2 : i + 4])
            if seg_len < 2:
                raise ImageError("corrupt JPEG segment")
            if marker in _SOF:
                precision = data[i + 4]
                h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                comps = data[i + 9] if i + 9 < len(data) else 3
                return w, h, max(comps, 3) * max(precision, 8) / 8
            i += 2 + seg_len
        raise ImageError("JPEG has no frame header")
    raise ImageError("only JPEG and PNG images are accepted")


def sniff_size(data: bytes) -> tuple[int, int]:
    return sniff(data)[:2]


def decode_image(data: bytes, max_pixels: int = MAX_PIXELS) -> np.ndarray:
    """Decode an upload to BGR, refusing decompression bombs before decoding."""
    if not data:
        raise ImageError("empty upload")
    w, h, bpp = sniff(data)
    if w < MIN_SIDE or h < MIN_SIDE:
        raise ImageError(f"image is {w}x{h}; the minimum side is {MIN_SIDE} px")
    if w * h > max_pixels:
        raise ImageError(f"image is {w}x{h} = {w * h / 1e6:.1f} MP; the limit is {max_pixels / 1e6:.0f} MP")
    if w * h * bpp > MAX_DECODED_BYTES:
        raise ImageError(f"image would decode to {w * h * bpp / 1e6:.0f} MB; the limit is "
                         f"{MAX_DECODED_BYTES / 1e6:.0f} MB (use 8-bit colour)")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    # EXIF orientation may swap the axes; anything else means the header lied.
    if img is None or sorted(img.shape[:2]) != sorted((h, w)):
        raise ImageError("image data could not be decoded")
    return img


def build_pyramid(img: np.ndarray, levels: int = LEVELS) -> list[np.ndarray]:
    pyr = [img]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _norm(x: np.ndarray) -> np.ndarray:
    hi = float(np.percentile(x, 99.5))
    return np.clip(x / hi, 0, 1) if hi > 1e-6 else np.zeros_like(x, np.float32)


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def saliency(img: np.ndarray) -> np.ndarray:
    """0-1 prior of 'something worth a closer look', built from imgproc ops only."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.GaussianBlur(_CLAHE.apply(gray), (3, 3), 0)
    gx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)
    edges = _norm(cv2.magnitude(gx, gy))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thin_dark = _norm(cv2.morphologyEx(eq, cv2.MORPH_BLACKHAT, kernel).astype(np.float32))
    local_dark = cv2.adaptiveThreshold(eq, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 12)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rust = cv2.inRange(hsv, (3, 70, 35), (25, 255, 235)).astype(np.float32) / 255
    white = cv2.inRange(hsv, (0, 0, 205), (180, 40, 255)).astype(np.float32) / 255
    s = 0.30 * edges + 0.40 * thin_dark + 0.10 * local_dark + 0.45 * rust + 0.15 * white
    s = cv2.GaussianBlur(s, (0, 0), 2.0)
    return _norm(s).astype(np.float32)


@dataclass
class Candidate:
    id: str
    cx: int  # L0 centre
    cy: int
    box: tuple[int, int, int, int]  # L0 window of the first look
    prior: float


def box_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    return inter / float(aw * ah + bw * bh - inter) if inter else 0.0


def _window_sums(m: np.ndarray, wy: int, wx: int) -> np.ndarray:
    """Sum of `m` under every wy x wx window that lies inside the map, indexed by its top-left corner."""
    ii = cv2.integral(m)
    h, w = m.shape
    return ii[wy:, wx:] - ii[: h - wy + 1, wx:] - ii[wy:, : w - wx + 1] + ii[: h - wy + 1, : w - wx + 1]


def find_candidates(
    sal: np.ndarray, level: int, window_l0: int, l0_size: tuple[int, int], max_n: int = 20, thresh: float = 0.5
) -> list[Candidate]:
    """Ranked first-look windows: a greedy maximum cover of the salient mass at the overview level.

    Each pick is the in-image window holding the most salient mass (saliency summed over pixels at or
    above `thresh`) that no earlier pick covers; among equals, the one overlapping earlier picks least.
    So the first k looks see the most distinct evidence, rather than k views of one strong blob. The
    prior is the mean saliency of the salient pixels the window newly covers.
    """
    scale = 2**level
    win = max(4, window_l0 // scale)
    h, w = sal.shape
    wy, wx = min(win, h), min(win, w)
    W, H = l0_size
    salient = sal >= thresh
    mass = np.where(salient, sal, 0).astype(np.float32)
    covered = np.zeros(sal.shape, np.float32)
    kept: list[Candidate] = []
    while len(kept) < max_n:
        sums = _window_sums(mass, wy, wx)
        best = float(sums.max())
        if best <= 0:
            break
        overlap = np.where(sums >= best * (1 - 1e-6), _window_sums(covered, wy, wx), np.inf)
        y0, x0 = np.unravel_index(np.argmin(overlap), overlap.shape)  # first index on ties: deterministic
        new = salient[y0 : y0 + wy, x0 : x0 + wx]
        prior = float(mass[y0 : y0 + wy, x0 : x0 + wx].sum()) / max(1, int(np.count_nonzero(new)))
        px = int(min(max((x0 + wx / 2) * scale, 0), W - 1))
        py = int(min(max((y0 + wy / 2) * scale, 0), H - 1))
        box = (px - window_l0 // 2, py - window_l0 // 2, window_l0, window_l0)
        kept.append(Candidate(f"r{len(kept) + 1}", px, py, box, round(min(prior, 1.0), 4)))
        mass[y0 : y0 + wy, x0 : x0 + wx] = 0
        salient[y0 : y0 + wy, x0 : x0 + wx] = False
        covered[y0 : y0 + wy, x0 : x0 + wx] = 1
    return kept


def _rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    return ry @ rx


def gaze(
    pyr: list[np.ndarray], x: float, y: float, level: int, yaw: float = 0.0, pitch: float = 0.0, size: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Render a virtual-PTZ view centred on L0 point (x, y) at a pyramid level.

    Returns the crop and the 3x3 matrix mapping crop pixels to L0 pixels. With
    yaw = pitch = 0 the view is a pure translation, so the mapping is exact.
    """
    if not 0 <= level < len(pyr):
        raise ValueError(f"level must be 0..{len(pyr) - 1}")
    s = 2.0**level
    f = float(size)
    k = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]])
    t = np.array([[1, 0, x / s - size / 2], [0, 1, y / s - size / 2], [0, 0, 1]])
    m = t @ k @ _rotation(yaw, pitch) @ np.linalg.inv(k)
    crop = cv2.warpPerspective(
        pyr[level], m, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT_101
    )
    to_l0 = np.diag([s, s, 1.0]) @ m
    return crop, to_l0


def map_points(to_l0: np.ndarray, pts) -> np.ndarray:
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, to_l0).reshape(-1, 2)


def quality(crop: np.ndarray) -> dict:
    """Sharpness (Laplacian variance) and exposure gate; a failed gate means 'do not trust this view'."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    glare = float(np.count_nonzero(gray >= 250)) / gray.size
    dark = float(np.count_nonzero(gray <= 5)) / gray.size
    reason = None
    if glare > GLARE_MAX:
        reason = "glare"
    elif dark > DARK_MAX:
        reason = "dark"
    elif sharp < SHARPNESS_MIN:
        reason = "blur"
    return {"ok": reason is None, "reason": reason, "sharpness": round(sharp, 1),
            "exposure": round(float(gray.mean()) / 255, 3)}


def enhance(crop: np.ndarray) -> np.ndarray:
    """CLAHE on the lightness channel: a cheaper second look before zooming further."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lab[..., 0] = _CLAHE.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def jpeg_b64(img: np.ndarray, max_side: int, quality_: int = 80) -> str:
    h, w = img.shape[:2]
    k = max_side / max(h, w)
    if k < 1:
        img = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality_])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def png_b64(gray: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", gray)
    return base64.b64encode(buf.tobytes()).decode() if ok else ""
