"""Procedural weathered-concrete photos with ground-truth defect boxes. Everything here is SYNTHETIC.

The background, the defects and the sensor noise each draw from their own RNG stream (all
`np.random.default_rng((seed, k))`), so changing a spec never changes the background. That is what lets
the tests isolate one defect's pixel signal by rendering the same seed with and without it.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np

# Bump whenever a change alters the images or ground truth a seed yields; evals.harness refuses a set built by
# another version. 2: GT crack boxes carry `hairline` (sets built before it have no version and no flag).
GENERATOR_VERSION = 2
CLASSES = ("crack", "spalling", "efflorescence", "exposed_rebar", "corrosion")  # CODEBRIM's five
DISTRACTORS = ("graffiti", "formwork_line", "water_stain", "blur_region", "glare")  # drawn in this order
CLEAN_DISTRACTORS = ("formwork_line", "water_stain", "graffiti", "glare")  # no GT box (blur_region hides a crack)
CLASS_WEIGHTS = (0.34, 0.20, 0.16, 0.08, 0.22)  # for random mixes, same order as CLASSES
MIN_PER_CLASS = 40
REF = 2048.0  # sizes below are pixels at this long side; defect sizes scale with the image, hairlines don't
SAMPLE_LIMIT = 4_000_000
RUST = (30, 70, 140)  # BGR


# ---------------------------------------------------------------- small drawing helpers
def value_noise(rng, h, w, top=None, gain=0.72, low=2):
    """Sum of upscaled random fields (octaves of low..top grid cells along the long side), zero mean, unit std."""
    long = max(h, w)
    top = top or max(4, long // 3)
    out, amp, c = np.zeros((h, w), np.float32), 1.0, low
    while c <= top:
        gh, gw = max(2, round(c * h / long)), max(2, round(c * w / long))
        field = rng.standard_normal((gh, gw), dtype=np.float32)
        out += amp * cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
        amp *= gain
        c *= 2
    return (out - out.mean()) / (out.std() + 1e-6)


def _stroke(canvas, pts, widths, colour=255, fade=None):
    """Anti-aliased variable-width polyline; sub-pixel coordinates via cv2's `shift`. LINE_AA is 8-bit only."""
    p = np.round(np.asarray(pts) * 16).astype(int).tolist()
    c = np.atleast_1d(np.asarray(colour, float))
    for i in range(len(p) - 1):
        col = tuple(float(v) for v in (c if fade is None else c * fade[i]))
        cv2.line(canvas, tuple(p[i]), tuple(p[i + 1]), col, max(1, round(widths[i])), cv2.LINE_AA, 4)


def _walk(rng, start, angle, n, step, jitter):
    """Random walk whose heading drifts (correlated), so it meanders instead of jittering."""
    ang = angle + np.cumsum(rng.normal(0, jitter, n))
    d = np.stack([np.cos(ang), np.sin(ang)], 1) * step
    return np.asarray(start, float) + np.vstack([[0.0, 0.0], np.cumsum(d, 0)])


def _wobble(rng, n, k=7):
    """Smooth 1-D noise with ~unit std, for width and darkness variation along a stroke."""
    return np.convolve(rng.standard_normal(n), np.ones(k) / k, "same") * np.sqrt(k)


def _blob(rng, cx, cy, rx, ry, rough, n=48):
    """Organic closed polygon: low-order Fourier radius around a rotated ellipse, plus fine jitter."""
    ang, k = np.linspace(0, 2 * np.pi, n, endpoint=False), np.arange(2, 9)
    coef = rough * rng.standard_normal(len(k)) / k ** 1.2
    r = 1 + coef @ np.cos(np.outer(k, ang) + rng.uniform(0, 2 * np.pi, (len(k), 1)))
    r = np.clip(r + rng.normal(0, 0.04, n), 0.4, 1.7)
    x, y, th = rx * r * np.cos(ang), ry * r * np.sin(ang), rng.uniform(0, np.pi)
    c, s = float(np.cos(th)), float(np.sin(th))
    return np.stack([cx + x * c - y * s, cy + x * s + y * c], 1)


def _bounds(pts, pad, w, h):
    lo, hi = np.floor(pts.min(0) - pad).astype(int), np.ceil(pts.max(0) + pad).astype(int)
    return max(lo[0], 0), max(lo[1], 0), min(hi[0], w), min(hi[1], h)


def _fill(shape, poly, origin):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round((poly - origin) * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    return mask.astype(np.float32) / 255


def _soft(mask, sigma):
    """uint8 0/255 or float mask -> blurred float32 in 0..1 (uint8 only; floats pass through)."""
    f = mask.astype(np.float32) / 255 if mask.dtype == np.uint8 else mask
    return cv2.GaussianBlur(f, (0, 0), sigma)


def _paint(img, x0, y0, mask, colour, strength=1.0):
    """Alpha-blend `colour` (BGR triple or HxWx3 array) into img under a float mask, in place."""
    roi = img[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
    a = (mask * strength)[..., None]
    roi[:] = roi * (1 - a) + a * np.asarray(colour, np.float32)


def _bbox(mask, x0, y0, thr):
    """Tight [x, y, w, h] around mask > thr (full-image pixels)."""
    ys, xs = np.nonzero(mask > thr)
    if not len(xs):
        return [int(x0), int(y0), 1, 1]
    return [int(x0 + xs.min()), int(y0 + ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def _centre(rng, w, h, rx, ry):
    return rng.uniform(rx, max(rx + 1, w - rx)), rng.uniform(ry, max(ry + 1, h - ry))


# ---------------------------------------------------------------- background
def _speckle(rng, h, w, S):
    """Aggregate: tiny light/dark dots plus rarer dark bug-holes. Impulses scaled so the blurred peak is `amp`."""
    out = np.zeros((h, w), np.float32)
    for density, sigma, lo, hi, dark_only in ((9e-4, 0.9, 10, 34, False), (1.2e-4, 2.2, 20, 55, True)):
        n, sigma = int(h * w * density), sigma * S
        amp = rng.uniform(lo, hi, n) * (-1 if dark_only else rng.choice((-1, 1), n))
        layer = np.zeros((h, w), np.float32)
        layer[rng.integers(0, h, n), rng.integers(0, w, n)] = amp
        out += cv2.GaussianBlur(layer, (0, 0), sigma) * (2 * np.pi * sigma ** 2)
    return out


def base_concrete(rng, w, h, S):
    grey = rng.uniform(140, 195)
    tint = np.array([-1, 0, 1], np.float32) * rng.uniform(-5, 5)  # BGR: positive = warm
    long = max(w, h)
    rain = cv2.resize(rng.standard_normal((3, 90), dtype=np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    lum = (grey + 4.5 * value_noise(rng, h, w, top=24)  # blotches, surface mottling, sand grain, rain streaks
           + 6 * value_noise(rng, h, w, top=long // 3, gain=0.85, low=24)
           + 6 * value_noise(rng, h, w, top=long // 2, gain=1.0, low=long // 4)
           + 2 * rain + _speckle(rng, h, w, S))
    # formwork panels: faint wobbly seams
    pw, ph = w / rng.uniform(2.2, 3.4), h / rng.uniform(1.7, 2.6)
    ox, oy = rng.uniform(0, pw), rng.uniform(0, ph)
    seam = np.zeros((h, w), np.uint8)
    for x in np.arange(ox, w, pw):
        ys = np.linspace(0, h, max(2, h // 90))
        _stroke(seam, np.column_stack([x + rng.normal(0, 0.8, len(ys)), ys]), np.full(len(ys), 1.5 * S))
    for y in np.arange(oy, h, ph):
        xs = np.linspace(0, w, max(2, w // 90))
        _stroke(seam, np.column_stack([xs, y + rng.normal(0, 0.8, len(xs))]), np.full(len(xs), 1.5 * S))
    lum *= 1 - 0.09 * _soft(seam, 1.1)
    # soft lighting gradient and vignette
    u = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    v = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    a = rng.uniform(0, 2 * np.pi)
    lum *= 1 + rng.uniform(0.06, 0.16) * (u * float(np.cos(a)) + v * float(np.sin(a))) \
        - rng.uniform(0.08, 0.2) * (u * u + v * v) / 2
    img = lum[..., None] + tint
    chroma = 2 * value_noise(rng, h, w, top=8)
    img[..., 0] -= chroma
    img[..., 2] += chroma
    return img.astype(np.float32, copy=False)


# ---------------------------------------------------------------- defects (each returns GT boxes, draws in place)
def _jag(rng, p, S, amp):
    """Resample a path every ~2.5 px and shake it sideways, so edges are ragged rather than pen-smooth."""
    t = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    tt = np.arange(0, t[-1], 2.5 * S)
    q = np.column_stack([np.interp(tt, t, p[:, 0]), np.interp(tt, t, p[:, 1])])
    d = np.gradient(q, axis=0)
    nrm = np.column_stack([-d[:, 1], d[:, 0]]) / (np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-9)
    return q + nrm * rng.normal(0, amp * S, (len(q), 1))


def crack(img, rng, S, thin=False):
    """Meandering ragged walk, optionally one branch, tapering and wobbling width. Hairlines are 1-2 px at any size."""
    h, w = img.shape[:2]
    n, step = int(rng.integers(35, 90)), rng.uniform(6, 11) * S
    start = [rng.uniform(0.06 * w, 0.94 * w), rng.uniform(0.06 * h, 0.94 * h)]
    main = _walk(rng, start, rng.uniform(0, 2 * np.pi), n, step, rng.uniform(0.18, 0.4))
    paths = [main]
    if rng.random() < 0.5:
        k = int(rng.integers(n // 4, 3 * n // 4))
        head = np.arctan2(*(main[k + 1] - main[k])[::-1]) + rng.choice((-1, 1)) * rng.uniform(0.4, 0.9)
        paths.append(_walk(rng, main[k], head, int(rng.integers(n // 5, n // 2)), step, 0.3))
    w0 = rng.uniform(1.0, 1.8) if thin else rng.uniform(2.5, 6.0)
    dark = rng.uniform(45, 90) if thin else rng.uniform(30, 90)
    paths = [_jag(rng, p, S, 0.35 if thin else 0.6) for p in paths]
    x0, y0, x1, y1 = _bounds(np.vstack(paths), w0 + 8, w, h)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for p in paths:
        rough = _wobble(rng, len(p))
        width = w0 * (1 - 0.85 * np.linspace(0, 1, len(p)) ** 1.5) * np.clip(1 + 0.3 * rough, 0.5, 1.1 if thin else 1.6)
        _stroke(mask, p - (x0, y0), width, 255, np.clip(0.8 + 0.25 * rough, 0.45, 1))
    m = _soft(mask, 0.5 if thin else 0.5 + 0.1 * w0)
    if not thin:  # soft shadow either side of an open crack
        m = np.clip(m + 0.3 * _soft(mask, 2.5 + w0), 0, 1)
    _paint(img, x0, y0, m, (dark, dark * 1.02, dark * 1.08), 0.92)
    return _bbox(m, x0, y0, 0.12), main


def spall(img, rng, S, rx=None):
    """Broken-off patch: rough chipped edge, lighter rough floor, shadowed upper edge, lit lower rim."""
    h, w = img.shape[:2]
    rx = rx or rng.uniform(50, 170) * S
    ry = rx * rng.uniform(0.55, 1.0)
    cx, cy = _centre(rng, w, h, rx * 1.5, ry * 1.5)
    poly, d = _blob(rng, cx, cy, rx, ry, 0.45), max(2, round(8 * S))
    x0, y0, x1, y1 = _bounds(poly, d + 14, w, h)
    tex, chip = value_noise(rng, y1 - y0, x1 - x0), value_noise(rng, y1 - y0, x1 - x0, low=16)
    m = np.clip((_soft(_fill(tex.shape, poly, (x0, y0)), 2 * S) + 0.1 * chip - 0.5) * 5 + 0.5, 0, 1)  # chipped edge
    relief = cv2.Sobel(_soft(tex, 2 * S), cv2.CV_32F, 0, 1, ksize=3)  # fake relief, light from above
    grit = _soft(rng.standard_normal(tex.shape, dtype=np.float32), 0.9)
    top, rim = np.clip(m - np.roll(m, d, 0), 0, 1), np.clip(m - np.roll(m, -d, 0), 0, 1)
    rough = 24 * relief / (relief.std() + 1e-6) + 45 * grit
    floor = img[y0:y1, x0:x1] * (0.62 + 0.1 * tex[..., None]) + rough[..., None]
    floor = floor * (1 - 0.55 * top[..., None]) + 40 * rim[..., None]
    _paint(img, x0, y0, m, floor)
    return _bbox(m, x0, y0, 0.4), m, x0, y0


def rebar(img, rng, S):
    """1-3 parallel rust bars as stacked-line cylinders inside a spall. Returns (spall box, bar box)."""
    thick, n, gap = rng.uniform(11, 17) * S, int(rng.integers(1, 4)), rng.uniform(2.4, 3.4)
    ang = rng.choice((0.0, np.pi / 2)) + rng.normal(0, 0.1)
    sbox, m, x0, y0 = spall(img, rng, S, rng.uniform(110, 190) * S)
    rh, rw = m.shape
    gap *= thick
    u = np.array([np.cos(ang), np.sin(ang)])
    nrm = np.array([-u[1], u[0]])
    if nrm[1] + 0.3 * nrm[0] > 0:
        nrm = -nrm  # toward the light (up-left)
    centre, long = np.array([sbox[0] + sbox[2] / 2 - x0, sbox[1] + sbox[3] / 2 - y0]), np.hypot(rw, rh)
    layer, alpha, shadow = np.zeros((rh, rw, 3), np.uint8), np.zeros((rh, rw), np.uint8), np.zeros((rh, rw), np.uint8)
    for i in range(n):
        c = centre + nrm * gap * (i - (n - 1) / 2)
        seg = np.array([c - u * long, c + u * long])
        for frac, col, off in ((1.0, (14, 34, 70), 0), (0.8, RUST, 0), (0.5, (50, 100, 170), 0.14),
                               (0.14, (80, 135, 195), 0.22)):
            _stroke(layer, seg + nrm * off * thick, [thick * frac], col)
        for t in np.arange(-long, long, thick * 1.4):  # rib marks of a deformed bar
            q = c + u * t
            rib = np.array([q - nrm * thick * 0.4, q + nrm * thick * 0.4 + u * thick * 0.3])
            _stroke(layer, rib, [S], (20, 45, 90))
        _stroke(alpha, seg, [thick])
        _stroke(shadow, seg + np.array([4, 5]) * S, [thick * 1.2])
    a = _soft(alpha, 0.6) * m
    sh = _soft(shadow, 2.5 * S) * m
    img[y0:y0 + rh, x0:x0 + rw] *= 1 - 0.5 * sh[..., None]
    _paint(img, x0, y0, _soft(alpha, 5 * S) * m, RUST, 0.35)  # rust halo
    scale = np.clip(0.8 + 0.22 * value_noise(rng, rh, rw, low=48), 0.5, 1.2)[..., None]  # weathered, scaly rust
    _paint(img, x0, y0, a, layer.astype(np.float32) * scale)
    return sbox, _bbox(a, x0, y0, 0.3)


def corrosion(img, rng, S):
    """Orange-brown mottled blotch with run-off streaks fading downward."""
    h, w = img.shape[:2]
    rx = rng.uniform(45, 120) * S
    ry = rx * rng.uniform(0.5, 1.0)
    cx, _ = _centre(rng, w, h, rx, ry)
    cy = rng.uniform(ry, max(ry + 1, 0.65 * h))
    poly = _blob(rng, cx, cy, rx, ry, 0.25)
    streaks = []
    for _ in range(int(rng.integers(5, 11))):
        s = [cx + rng.uniform(-0.8, 0.8) * rx, cy + ry * rng.uniform(-0.1, 0.8)]
        streaks.append(_walk(rng, s, np.pi / 2, max(2, int(rng.uniform(80, 460) / 8)), 8 * S, 0.04))
    x0, y0, x1, y1 = _bounds(np.vstack([poly] + streaks), 14 * S + 6, w, h)
    smask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for p in streaks:
        width = rng.uniform(3, 10) * S * np.linspace(1, 0.4, len(p))
        _stroke(smask, p - (x0, y0), width, 255, np.linspace(1, 0.15, len(p) - 1))
    b = _soft(_fill(smask.shape, poly, (x0, y0)), 0.12 * ry + 1)
    mottle = np.clip(0.7 + 0.3 * value_noise(rng, *b.shape, low=8), 0.1, 1.1)
    m = np.clip(b + 0.9 * _soft(smask, 2.5 * S), 0, 1) * mottle
    edge, core = np.array((30, 95, 175), np.float32), np.array((20, 55, 110), np.float32)
    colour = edge * (1 - b[..., None]) + core * b[..., None]
    _paint(img, x0, y0, m, colour, 0.7)
    return _bbox(m, x0, y0, 0.25)


def efflorescence(img, rng, S, anchor=None):
    """Chalky white crust with drips; deposited just under a crack when `anchor` (a crack path) is given."""
    h, w = img.shape[:2]
    if anchor is None:
        start = [rng.uniform(0.1 * w, 0.9 * w), rng.uniform(0.1 * h, 0.7 * h)]
        src = _walk(rng, start, rng.normal(0, 0.25), 14, 14 * S, 0.15)
    else:
        a = int(rng.integers(0, len(anchor) // 2))
        src = anchor[a:a + max(4, int(len(anchor) * rng.uniform(0.3, 0.6)))] + (0, 5 * S)
    drips = [_walk(rng, src[j] + (0, 2), np.pi / 2, max(2, int(rng.uniform(25, 160) * S / 6)), 6, 0.05)
             for j in rng.integers(0, len(src), int(rng.integers(4, 10)))]
    x0, y0, x1, y1 = _bounds(np.vstack([src] + drips), 12 * S + 8, w, h)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    _stroke(mask, src - (x0, y0), np.full(len(src), rng.uniform(5, 13) * S))
    for p in drips:
        width = np.linspace(rng.uniform(2.5, 5) * S, 1.2, len(p))
        _stroke(mask, p - (x0, y0), width, 255, np.linspace(1, 0.3, len(p) - 1))
    m = _soft(mask, 2 * S + 0.8)
    patchy = np.clip((m * 1.5 + 0.45 * value_noise(rng, *m.shape, low=12) - 0.35) * 1.8, 0, 1) * np.clip(m * 8, 0, 1)
    _paint(img, x0, y0, patchy, (224, 230, 234), 0.72)
    return _bbox(patchy, x0, y0, 0.2)


def graffiti(img, rng, S):
    """Saturated curly paint strokes with dark outlines, drips and overspray; deliberately not letterforms."""
    h, w = img.shape[:2]
    c = np.array([rng.uniform(0.2 * w, 0.8 * w), rng.uniform(0.2 * h, 0.8 * h)])
    strokes = [_walk(rng, c + rng.normal(0, 100 * S, 2), rng.uniform(0, 6.3), int(rng.integers(25, 60)), 9 * S, 0.4)
               for _ in range(int(rng.integers(4, 8)))]
    x0, y0, x1, y1 = _bounds(np.vstack(strokes), 24 * S + 8, w, h)
    layer, alpha = np.zeros((y1 - y0, x1 - x0, 3), np.uint8), np.zeros((y1 - y0, x1 - x0), np.uint8)
    hue = int(rng.integers(0, 180))
    for p in strokes:
        wd = rng.uniform(9, 18) * S * np.clip(1 + 0.25 * _wobble(rng, len(p), 5), 0.6, 1.4)
        col = cv2.cvtColor(np.uint8([[[(hue + int(rng.integers(0, 40))) % 180, 235, 235]]]), cv2.COLOR_HSV2BGR)[0, 0]
        _stroke(alpha, p - (x0, y0), wd + 3 * S)
        _stroke(layer, p - (x0, y0), wd + 3 * S, col // 2)
        _stroke(layer, p - (x0, y0), wd, col)
        for j in rng.integers(0, len(p), 3):
            drip = np.array([p[j], p[j] + (0, rng.uniform(15, 70) * S)]) - (x0, y0)
            _stroke(layer, drip, [2.5 * S], col)
            _stroke(alpha, drip, [3 * S])
    a = np.clip(_soft(alpha, 0.8) + 0.25 * _soft(alpha, 5 * S), 0, 1)  # overspray halo
    shade = np.clip(img[y0:y1, x0:x1].mean(2, keepdims=True) / 170, 0.75, 1.15)  # concrete grain shows through
    _paint(img, x0, y0, a, layer.astype(np.float32) * shade, 0.93)
    return _bbox(a, x0, y0, 0.3)


def formwork_line(img, rng, S):
    """Long, perfectly straight thin groove with a lit lower lip: looks like a crack, is not one."""
    h, w = img.shape[:2]
    horiz, tilt = rng.random() < 0.7, rng.uniform(-0.03, 0.03)
    length = rng.uniform(0.4, 0.8) * (w if horiz else h)
    a = tilt if horiz else np.pi / 2 + tilt
    p0 = np.array([rng.uniform(0.05, 0.95) * w * (0.5 if horiz else 1), rng.uniform(0.1, 0.9) * h])
    seg = np.array([p0, p0 + length * np.array([np.cos(a), np.sin(a)])])
    x0, y0, x1, y1 = _bounds(seg, 8, w, h)
    groove, lip = np.zeros((y1 - y0, x1 - x0), np.uint8), np.zeros((y1 - y0, x1 - x0), np.uint8)
    _stroke(groove, seg - (x0, y0), [rng.uniform(2, 3.2)])
    _stroke(lip, seg - (x0, y0) + (0, 2.5), [2])
    g = _soft(groove, 0.8)
    _paint(img, x0, y0, _soft(lip, 0.9), (235, 235, 235), 0.35)
    _paint(img, x0, y0, g, (45, 50, 55), 0.6)
    return _bbox(g, x0, y0, 0.12)


def water_stain(img, rng, S):
    """Dark wet patch with a darker tide-mark rim."""
    h, w = img.shape[:2]
    rx = rng.uniform(90, 260) * S
    ry = rx * rng.uniform(0.4, 0.9)
    poly = _blob(rng, *_centre(rng, w, h, rx * 1.3, ry * 1.3), rx, ry, 0.3)
    x0, y0, x1, y1 = _bounds(poly, 0.3 * rx, w, h)
    m = _soft(_fill((y1 - y0, x1 - x0), poly, (x0, y0)), 0.08 * rx)
    rim = np.clip(m - _soft(m, 0.25 * rx), 0, 1)
    dark = (0.25 * m + 0.5 * rim) * np.clip(0.85 + 0.25 * value_noise(rng, *m.shape), 0.4, 1.2)
    img[y0:y1, x0:x1] *= 1 - dark[..., None] * np.array([0.5, 0.5, 0.55], np.float32)
    return _bbox(dark, x0, y0, 0.08)


def glare(img, rng, S):
    """Blown-out specular highlight (values exceed 255 and clip at the end)."""
    h, w = img.shape[:2]
    rx = rng.uniform(0.05, 0.1) * w
    ry = rx * rng.uniform(0.5, 0.9)
    poly = _blob(rng, *_centre(rng, w, h, rx * 1.6, ry * 1.6), rx, ry, 0.25)
    x0, y0, x1, y1 = _bounds(poly, 0.6 * rx, w, h)
    m = _soft(_fill((y1 - y0, x1 - x0), poly, (x0, y0)), 0.2 * rx)
    img[y0:y1, x0:x1] += m[..., None] * np.array([205, 215, 225], np.float32)
    return _bbox(m, x0, y0, 0.35)


def blur_region(img, rng, S, box):
    """Strongly blur a feathered rectangle around `box` (a crack whose ground truth still counts)."""
    h, w = img.shape[:2]
    pad = int(0.35 * max(box[2], box[3]) + 30 * S)
    x0, y0 = max(0, box[0] - pad), max(0, box[1] - pad)
    x1, y1 = min(w, box[0] + box[2] + pad), min(h, box[1] + box[3] + pad)
    inset = np.zeros((y1 - y0, x1 - x0), np.float32)
    f = int(0.12 * min(inset.shape))
    inset[f:-f or None, f:-f or None] = 1
    m = _soft(inset, f / 2 + 1)[..., None]
    roi = img[y0:y1, x0:x1]
    roi[:] = roi * (1 - m) + cv2.GaussianBlur(roi, (0, 0), 5 * S + 2) * m
    return [x0, y0, x1 - x0, y1 - y0]


_SIMPLE_DISTRACTORS = {"graffiti": graffiti, "formwork_line": formwork_line, "water_stain": water_stain, "glare": glare}


# ---------------------------------------------------------------- rendering
def planned_counts(spec):
    """GT boxes a spec yields per class: each rebar brings its containing spall, blur_region its crack."""
    c = {k: int(spec.get(k, 0)) for k in CLASSES}
    c["spalling"] += c["exposed_rebar"]
    c["crack"] += list(spec.get("distractors", [])).count("blur_region")
    return c


def _random_spec(rng):
    spec = dict.fromkeys(CLASSES, 0)
    for cls in rng.choice(CLASSES, size=int(rng.integers(1, 6)), p=CLASS_WEIGHTS):
        spec[str(cls)] += 1
    spec["distractors"] = [str(rng.choice(DISTRACTORS))] if rng.random() < 0.25 else []
    return spec


def render(seed: int, width: int, height: int, spec: dict | None = None) -> tuple[np.ndarray, list[dict], list[dict]]:
    """(BGR uint8 image, GT boxes, distractor records). `generate` drops the last item."""
    sp = spec if spec is not None else _random_spec(np.random.default_rng((seed, 3)))
    S = max(width, height) / REF
    img = base_concrete(np.random.default_rng((seed, 0)), width, height, S)
    counter, gt, extra, paths = itertools.count(), [], [], []

    def rng():  # one independent stream per defect: editing one never reshuffles the others
        return np.random.default_rng((seed, 1, next(counter)))

    def add(cls, box, **flags):
        gt.append({"cls": cls, "box": box, **flags})

    for _ in range(sp.get("corrosion", 0)):
        add("corrosion", corrosion(img, rng(), S))
    for _ in range(sp.get("spalling", 0)):
        add("spalling", spall(img, rng(), S)[0])
    for _ in range(sp.get("exposed_rebar", 0)):
        sbox, bbox = rebar(img, rng(), S)
        add("spalling", sbox)
        add("exposed_rebar", bbox)
    for _ in range(sp.get("crack", 0)):
        r = rng()
        thin = bool(sp["hairline"] if "hairline" in sp else r.random() < 0.35)
        box, path = crack(img, r, S, thin)
        paths.append(path)
        add("crack", box, hairline=thin)
    for _ in range(sp.get("efflorescence", 0)):
        r = rng()
        anchor = paths[int(r.integers(len(paths)))] if paths and r.random() < 0.8 else None
        add("efflorescence", efflorescence(img, r, S, anchor))
    for kind in sorted(sp.get("distractors", []), key=DISTRACTORS.index):
        if kind == "blur_region":
            r = rng()
            box, _ = crack(img, r, S, True)
            add("crack", box, hairline=True)
            extra.append({"kind": kind, "box": blur_region(img, r, S, box)})
        else:
            extra.append({"kind": kind, "box": _SIMPLE_DISTRACTORS[kind](img, rng(), S)})
    post = np.random.default_rng((seed, 2))  # sensor noise (luma + chroma), then lens/JPEG-like softness
    img += 3 * post.standard_normal((height, width, 1), dtype=np.float32)
    img += 1.5 * post.standard_normal((height, width, 3), dtype=np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    return np.clip(img, 0, 255).astype(np.uint8), gt, extra


def generate(seed: int, width: int, height: int, spec: dict | None = None) -> tuple[np.ndarray, list[dict]]:
    img, gt, _ = render(seed, width, height, spec)
    return img, gt


# ---------------------------------------------------------------- dataset and samples
def stratify(n, seed, floor=MIN_PER_CLASS):
    """Per-image specs: a seeded random mix, topped up until every class has >= floor GT boxes (exposed_rebar
    is rare, so it is forced). The floor is capped at n // 5 so a small n is not crammed. Renders nothing."""
    floor, rng = min(floor, max(1, n // 5)), np.random.default_rng(seed)
    specs = [_random_spec(rng) for _ in range(n)]
    for cls in CLASSES:
        while sum(planned_counts(s)[cls] for s in specs) < floor:
            load = [sum(planned_counts(s).values()) for s in specs]
            specs[int(rng.choice(np.flatnonzero(np.array(load) == min(load))))][cls] += 1
    return specs


def clean_specs(n, seed):
    """Per-image specs with no damage: one or two distinct CLEAN_DISTRACTORS each."""
    rng = np.random.default_rng((seed, 4))
    return [{"distractors": [str(k) for k in rng.choice(CLEAN_DISTRACTORS, int(rng.integers(1, 3)), replace=False)]}
            for _ in range(n)]


def build_dataset(out_dir, n: int = 200, seed: int = 7, width: int = 2048, height: int = 1536, clean: bool = False):
    """The labelled benchmark; `clean`: n images with distractors and no damage, then the demo's clean sample
    bridge_08 (byte for byte as `build_samples` writes it, at 3000x2250) as the last record, for false alarms."""
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    images = []
    for i, spec in enumerate(clean_specs(n, seed) if clean else stratify(n, seed)):
        img, gt, extra = render(seed * 100_000 + i, width, height, spec)
        iid = f"s{i + 1:04d}"
        cv2.imwrite(str(out / "images" / f"{iid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        images.append({"id": iid, "file": f"images/{iid}.jpg", "width": width, "height": height,
                       "boxes": gt, "distractors": extra})
    if clean:
        k, (s, _, _, spec) = next((k, x) for k, x in enumerate(SAMPLES, 1) if x[1] == "Clean concrete")
        img, gt, extra = render(s, 3000, 2250, spec)
        (out / "images" / f"bridge_{k:02d}.jpg").write_bytes(_jpeg_under(img, SAMPLE_LIMIT)[0].tobytes())
        images.append({"id": f"bridge_{k:02d}", "file": f"images/bridge_{k:02d}.jpg", "width": 3000, "height": 2250,
                       "boxes": gt, "distractors": extra})
    name = f"synthetic-clean-{len(images)}" if clean else f"synthetic-{n}"
    (out / "annotations.json").write_text(json.dumps(
        {"dataset": name, "synthetic": True, "seed": seed, "generator_version": GENERATOR_VERSION, "clean": clean,
         "images": images}, indent=1))
    return images


# (seed, title, one honest sentence, spec)
SAMPLES = [
    (101, "Hairline cracks", "Five 1-2 px cracks that all but vanish in a thumbnail.", {"crack": 5, "hairline": True}),
    (102, "Crack with efflorescence", "A crack with chalky salt deposits leached out beneath it.",
     {"crack": 2, "efflorescence": 2}),
    (120, "Exposed rebar", "A broken-out spall exposing reinforcing bars: the critical finding.",
     {"crack": 1, "spalling": 1, "exposed_rebar": 1}),
    (104, "Corrosion run-off", "Rust staining bleeding down the face in streaks.", {"corrosion": 2, "crack": 1}),
    (105, "Mixed defects", "Cracks, spalling, exposed bars, rust and efflorescence on one wall.",
     {"crack": 3, "spalling": 1, "exposed_rebar": 1, "corrosion": 1, "efflorescence": 1}),
    (106, "Graffiti and one real crack", "Spray-paint tags that are not damage, plus one genuine crack.",
     {"crack": 1, "distractors": ["graffiti"]}),
    (107, "Glare and blur", "A blown-out highlight and a blurred patch hiding a thin crack.",
     {"distractors": ["glare", "blur_region"]}),
    (108, "Clean concrete", "Sound concrete with one straight formwork groove; nothing here is damage.",
     {"distractors": ["formwork_line"]}),
]


def _jpeg_under(img, limit):
    for q in range(92, 40, -4):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok and len(buf) < limit:
            return buf, q
    raise RuntimeError("cannot get under the size limit")


def build_samples(out_dir="public/samples"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for k, (seed, title, note, spec) in enumerate(SAMPLES, 1):
        img, gt, extra = render(seed, 3000, 2250, spec)
        buf, q = _jpeg_under(img, SAMPLE_LIMIT)
        name = f"bridge_{k:02d}.jpg"
        (out / name).write_bytes(buf.tobytes())
        (out / "thumbs").mkdir(exist_ok=True)
        thumb = cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / "thumbs" / name), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        manifest.append({"file": name, "thumb": f"thumbs/{name}", "title": title, "note": note, "synthetic": True,
                         "width": 3000, "height": 2250, "boxes": gt, "distractors": extra})
        print(f"{name}: {len(buf):,} bytes (q={q}), {len(gt)} boxes")
    (out / "index.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m evals.synth", description="Synthetic concrete photos with GT boxes.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset", help="labelled synthetic benchmark")
    d.add_argument("--out", default="data/synth200")
    for name, default in (("n", 200), ("seed", 7), ("width", 2048), ("height", 1536)):
        d.add_argument(f"--{name}", type=int, default=default)
    d.add_argument("--clean", action="store_true", help="no damage: n images of distractors only, plus bridge_08")
    sub.add_parser("samples", help="8 curated demo images").add_argument("--out", default="public/samples")
    a = ap.parse_args(argv)
    t = time.perf_counter()
    if a.cmd == "dataset":
        build_dataset(a.out, a.n, a.seed, a.width, a.height, a.clean)
    else:
        build_samples(a.out)
    print(f"done in {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
