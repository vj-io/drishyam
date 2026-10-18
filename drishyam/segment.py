"""Per-class defect segmentation of a gaze crop, with uncertainty.

Two engines share one output: `ClassicalSegmenter` (imgproc evidence maps, no
weights, runs anywhere) and `OnnxSegmenter` (a dacl10k-trained model on the
OpenCV 5 DNN `ENGINE_NEW`). The policy only reads probabilities, areas and
entropy, so swapping engines changes no other code.
"""

from __future__ import annotations

import functools
import os
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .measure import thin

CLASSES = ("background", "crack", "spalling", "efflorescence", "exposed_rebar", "corrosion")
DEFECTS = CLASSES[1:]
MIN_BLOB_PX = 12  # connected components smaller than this are noise
ENTROPY_MIN_PX = 64  # fewer ambiguous pixels than this is not uncertainty worth a closer look
DEFECT_PX_BG_MAX = 0.7  # a pixel counts toward entropy when p(background) is below this
CRACK_LINK_SIGMA = 2.5  # crack hysteresis low threshold: half the 5-sigma seed threshold (Canny's 2:1)
STRAIGHT_RMS_PX = 1.5  # crack centreline RMS residual from its cv2.fitLine (Huber) line, L0 px
STRAIGHT_ASPECT = 20.0  # length over mean width, and the minAreaRect long side over its short side
STRAIGHT_CAUSE = "straight line: formwork or joint"


@dataclass
class Segmentation:
    probs: np.ndarray  # (C, H, W) float32, sums to 1 over C
    labels: np.ndarray  # (H, W) uint8 class index
    entropy: float  # mean binary damage-vs-background entropy over defect-like pixels, 0..1
    areas: dict[str, int]  # crop pixels per defect class
    confidence: dict[str, float]  # mean probability over each class's pixels
    latency_ms: float
    model: str
    net_px: int = 0  # pixels the engine computed on: the net's input tensor (pad included), or the classical work image
    lines: list = field(default_factory=list)  # crack components dropped as straight lines (`drop_straight_lines`)

    def mask(self, cls: str) -> np.ndarray:
        return (self.labels == CLASSES.index(cls)).astype(np.uint8)

    @property
    def best(self) -> str | None:
        present = {c: a * self.confidence[c] for c, a in self.areas.items() if a}
        return max(present, key=present.get) if present else None


def _robust(x: np.ndarray) -> tuple[float, float]:
    """Median and MAD-sigma of a map, from a strided sample (the map is a texture, not a signal)."""
    sample = x[::4, ::4]
    med = float(np.median(sample))
    return med, max(1.0, 1.4826 * float(np.median(np.abs(sample - med))))


def _wide_blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Large-sigma Gaussian on a 1/8 copy: the same low-pass at a fraction of the cost."""
    h, w = x.shape[:2]
    small = cv2.resize(x, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / 8), (w, h), interpolation=cv2.INTER_LINEAR)


def _hysteresis(strong: np.ndarray, weak: np.ndarray, min_len: int) -> np.ndarray:
    """Canny-style linking for a linear defect: a pixel whose `weak` evidence passes the coin
    flip counts as crack when it is 8-connected through such pixels to one whose `strong`
    evidence does, and the linked trace spans at least `min_len` px (a line, not a speck).
    A faint stretch of a crack then stays part of the crack at every scale, while faint
    texture around an isolated dark speck stays background."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats((weak > 0.5).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[lab[strong > 0.5]] = True
    keep &= np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= min_len
    keep[0] = False
    return np.where(keep[lab], np.maximum(strong, weak), strong)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


def _clean(labels: np.ndarray, probs: np.ndarray, ids) -> tuple[dict[str, int], dict[str, float]]:
    """Drop specks of the classes `ids` from `labels` (in place); their areas and mean confidences."""
    areas, conf = {}, {}
    for ci in ids:
        m = (labels == ci).astype(np.uint8)
        _, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        small = stats[:, cv2.CC_STAT_AREA] < MIN_BLOB_PX
        small[0] = False
        if small.any():
            drop = small[lab]
            labels[drop] = 0
            m[drop] = 0
        a = int(np.count_nonzero(m))
        areas[CLASSES[ci]] = a
        conf[CLASSES[ci]] = round(float(probs[ci][m > 0].mean()), 4) if a else 0.0
    return areas, conf


def _entropy(probs: np.ndarray) -> float:
    """Uncertainty = binary entropy of "is this damage", averaged where damage is plausible.
    1.0 is a coin flip; a crisp mask scores ~0.2 (its anti-aliased rim)."""
    p_def = 1.0 - probs[0]
    defect_like = p_def > 1.0 - DEFECT_PX_BG_MAX
    if np.count_nonzero(defect_like) < ENTROPY_MIN_PX:
        return 0.0
    q = np.clip(p_def[defect_like], 1e-6, 1 - 1e-6)
    return round(float((-(q * np.log2(q) + (1 - q) * np.log2(1 - q))).mean()), 4)


def finish(probs: np.ndarray, t0: float, model: str, net_px: int | None = None) -> Segmentation:
    """Labels, areas, confidences and entropy from a (C, H, W) probability stack. `net_px`: the pixels the engine
    computed on (default: the stack's own H x W)."""
    labels = probs.argmax(axis=0).astype(np.uint8)
    areas, conf = _clean(labels, probs, range(1, len(CLASSES)))
    return Segmentation(probs, labels, _entropy(probs), areas, conf, (time.perf_counter() - t0) * 1000, model,
                        probs[0].size if net_px is None else net_px)


def restrict(seg: Segmentation, valid: np.ndarray) -> Segmentation:
    """`seg` with background forced wherever `valid` is False (crop pixels that are not the image, see
    `imaging.footprint`), so labels, areas, confidence, entropy and everything built on them see no
    mirrored defect. Equal to `finish` on the masked probabilities, at a third of its cost: inside `valid`
    the argmax is unchanged, so only classes labelled outside it need their specks re-checked."""
    if valid.all():
        return seg
    t0 = time.perf_counter()
    probs = seg.probs * valid
    probs[0] += ~valid
    labels = np.where(valid, seg.labels, 0).astype(np.uint8)
    areas, conf = dict(seg.areas), dict(seg.confidence)
    a, c = _clean(labels, probs, [int(ci) for ci in np.unique(seg.labels[~valid]) if ci])
    areas.update(a)
    conf.update(c)
    return Segmentation(probs, labels, _entropy(probs), areas, conf,
                        seg.latency_ms + (time.perf_counter() - t0) * 1000, seg.model, seg.net_px)


def drop_straight_lines(seg: Segmentation, scale: float = 1.0) -> Segmentation:
    """`seg` without the crack components that are one straight line, each listed in `.lines` as ((x, y, w, h) in
    the view, then length, mean width and residual in L0 px; `scale` is L0 px per view px). A centreline within
    STRAIGHT_RMS_PX L0 px (RMS) of its cv2.fitLine line over a length of at least STRAIGHT_ASPECT x its mean width,
    inside a minAreaRect at least that thin, is a formwork seam, joint or edge: real cracks meander. The residual
    is judged in L0 px because a coarse view smooths a crack's jitter away, so only a fine view can call a line
    straight. Its pixels become background (`restrict`): no finding, entropy or escalation comes from it."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(seg.mask("crack"), connectivity=8)
    lines, keep = [], np.ones(seg.labels.shape, bool)
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        comp = (lab[y : y + h, x : x + w] == i).astype(np.uint8)
        (_, _), (rw, rh), _ = cv2.minAreaRect(cv2.findNonZero(comp))
        length = max(rw, rh)
        if length < STRAIGHT_ASPECT * max(1.0, min(rw, rh)) or length * length < STRAIGHT_ASPECT * area:
            continue  # not long and thin, the common case: decided before any thinning
        c = cv2.findNonZero(thin(cv2.copyMakeBorder(comp, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)))
        c = c.reshape(-1, 2).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(c, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        # RMS over the 90 % of centreline points nearest the line: thinning leaves spurs where a speck touches it
        d = np.sort(np.abs((c[:, 0] - x0) * vy - (c[:, 1] - y0) * vx))[: max(1, len(c) * 9 // 10)]
        rms = scale * float(np.sqrt(np.mean(d * d)))
        if rms < STRAIGHT_RMS_PX:
            keep[lab == i] = False
            lines.append(((x, y, w, h), round(length * scale, 1), round(area / length * scale, 2), round(rms, 2)))
    if not lines:
        return seg
    out = restrict(seg, keep)
    out.lines = lines
    return out


def segment_view(segmenter, crop: np.ndarray, model: str, valid: np.ndarray, scale: float = 1.0) -> Segmentation:
    """Segment a gaze crop, keeping only what lies inside the image (`valid`, from `imaging.footprint`) and is not
    a straight line (`drop_straight_lines`; `scale`: L0 px per crop px, 2**level)."""
    return drop_straight_lines(restrict(segmenter.predict(crop, model), valid), scale)


class ClassicalSegmenter:
    """Hand-built evidence maps turned into calibrated class probabilities.

    No learned weights: this is the honest baseline that keeps the agent runnable
    with zero downloads. Gains map evidence 0.5 to a coin flip against background,
    which is where the entropy the policy escalates on comes from.
    """

    name = "classical-v1"
    gain = 9.0

    def evidence(self, crop: np.ndarray) -> np.ndarray:
        """Five 0-1 maps. Thresholds are relative to the crop's own texture, so a rough
        but healthy surface does not read as damage."""
        h, w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # A CLAHE per call: it keeps working buffers, and default_segmenter() is shared by request threads.
        g = cv2.GaussianBlur(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray), (3, 3), 0)
        k = max(7, (min(h, w) // 34) | 1)
        bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        bh = bh.astype(np.float32)
        med, noise = _robust(bh)
        crack = np.clip((bh - med - 3.0 * noise) / (4.0 * noise), 0, 1)  # coin flip at 5 sigma
        # The same ramp with its coin flip at CRACK_LINK_SIGMA: evidence a crack pixel may have
        # when it continues a confident one (hysteresis, below).
        linked = np.clip((bh - med - (CRACK_LINK_SIGMA - 2.0) * noise) / (4.0 * noise), 0, 1)

        gf = gray.astype(np.float32)
        sig = max(3.0, min(h, w) / 60)
        mu = cv2.GaussianBlur(gf, (0, 0), sig)
        rough = np.sqrt(np.maximum(cv2.GaussianBlur(gf * gf, (0, 0), sig) - mu * mu, 0))
        r_med, r_noise = _robust(rough)
        # Spalling is a wide, dark, rough patch. Darkness is read from a wide median (which
        # erases cracks up to ~k px wide) and the map is opened with a kernel wider than any
        # crack, so neither a crack's contrast nor the rim of a stain can pass as a spall.
        wide = 2 * k + 1
        cf = cv2.GaussianBlur(cv2.medianBlur(gray, wide).astype(np.float32), (0, 0), sig)
        darker = np.clip((_wide_blur(gf, sig * 6) - cf) / 25.0, 0, 1)
        spall_raw = np.clip((rough - r_med - 2.5 * r_noise) / (3.0 * r_noise), 0, 1) * np.clip(2.5 * darker, 0, 1)
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (wide, wide))  # rect: separable, fast

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        rust = cv2.GaussianBlur(
            cv2.inRange(hsv, (3, 70, 35), (25, 255, 235)).astype(np.float32) / 255, (0, 0), 2.0
        )
        spall = cv2.morphologyEx(spall_raw * (1 - rust), cv2.MORPH_OPEN, open_k)
        near_spall = cv2.dilate(spall, cv2.getStructuringElement(cv2.MORPH_RECT, (k * 3, k * 3)))
        rebar = rust * np.clip(2 * near_spall, 0, 1)
        corrosion = rust * (1 - np.clip(2 * near_spall, 0, 1))

        v = hsv[..., 2].astype(np.float32)
        lift = v - _wide_blur(v, sig * 4)
        white = (hsv[..., 1] < 40).astype(np.float32)
        efflo = np.clip((lift - 22.0) / 25.0, 0, 1) * white
        efflo = cv2.morphologyEx(efflo, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # A thin dark line inside spalling is the spall's own edge, and rust inside it is the bar.
        # Linking never runs through rust (a faint dark line there is the bar's or the stain's edge),
        # and a linked trace must be longer than k, the widest dark line the blackhat passes.
        not_spall = 1 - np.clip(1.5 * spall, 0, 1)
        crack = _hysteresis(crack * not_spall, linked * not_spall * (1 - rust), k)
        spall = spall * (1 - np.clip(1.5 * rebar, 0, 1))
        return np.stack([crack, spall, efflo, rebar, corrosion]).astype(np.float32)

    def predict(self, crop: np.ndarray, model: str = "accurate") -> Segmentation:
        t0 = time.perf_counter()
        h, w = crop.shape[:2]
        work = crop if model == "accurate" else cv2.resize(crop, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        ev = self.evidence(work)
        logits = np.concatenate([np.zeros((1, *ev.shape[1:]), np.float32), self.gain * (ev - 0.5)])
        probs = _softmax(logits)
        if work is not crop:
            probs = np.stack([cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) for p in probs])
        return finish(probs, t0, model, work.shape[0] * work.shape[1])


class OnnxSegmenter:
    """A segmentation ONNX (N, C, H, W logits) on the OpenCV 5 DNN new engine.

    Each crop is letterboxed into the square `input_size` input (`input_size // 2` for the fast model): one scale
    for both axes, the rest padded with the normalised mean (zero). The logits are cropped back to the letterbox's
    content before they are resized to the crop, so a non-square crop is never squashed.
    `class_map[i]` is the Drishyam class index that model channel i adds to, so a
    19-class dacl10k head collapses onto CODEBRIM's five defects plus background.
    """

    def __init__(self, path: str, fp16_path: str | None = None, input_size: int = 512,
                 class_map: list[int] | None = None, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.name = f"onnx:{path.rsplit('/', 1)[-1]}"
        self.files = (path, fp16_path)  # what the nets were read from (evals.harness hashes them)
        self.nets = {"accurate": self._load(path)}
        self.nets["fp16"] = self._load(fp16_path) if fp16_path else self.nets["accurate"]
        self.size = input_size
        self.class_map = class_map
        self.mean = np.array(mean, np.float32).reshape(3, 1, 1)
        self.std = np.array(std, np.float32).reshape(3, 1, 1)
        # A Net holds its input between setInput and forward, and default_segmenter() is shared by
        # request threads. ponytail: one lock for both nets; a lock per net if fp16 and accurate must overlap.
        self.lock = threading.Lock()

    @staticmethod
    def _load(path: str):
        return cv2.dnn.readNetFromONNX(path, engine=cv2.dnn.ENGINE_NEW)

    def predict(self, crop: np.ndarray, model: str = "accurate") -> Segmentation:
        t0 = time.perf_counter()
        h, w = crop.shape[:2]
        size = self.size if model == "accurate" else self.size // 2
        k = size / max(h, w)
        nw, nh = max(1, round(w * k)), max(1, round(h * k))
        rgb = cv2.cvtColor(cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        blob = np.zeros((1, 3, size, size), np.float32)
        blob[0, :, :nh, :nw] = (rgb.astype(np.float32).transpose(2, 0, 1) / 255.0 - self.mean) / self.std
        net = self.nets["fp16" if model in ("fp16", "fast") else "accurate"]
        with self.lock:
            net.setInput(blob)
            logits = np.asarray(net.forward())[0].astype(np.float32)
        oh, ow = logits.shape[1:]  # the output may be strided: crop the content at the same fraction
        probs = _softmax(logits[:, : max(1, round(nh * oh / size)), : max(1, round(nw * ow / size))])
        if self.class_map is not None:
            merged = np.zeros((len(CLASSES), *probs.shape[1:]), np.float32)
            for ch, target in enumerate(self.class_map):
                merged[target] += probs[ch]
            probs = merged
        if probs.shape[0] != len(CLASSES):
            raise ValueError(f"model has {probs.shape[0]} classes; pass class_map to reach {len(CLASSES)}")
        probs = np.stack([cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) for p in probs])
        return finish(probs, t0, model, size * size)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 1.0


@functools.cache
def default_segmenter():
    """The ONNX model named by DRISHYAM_ONNX (optional DRISHYAM_ONNX_FP16), else the classical engine."""
    path = os.environ.get("DRISHYAM_ONNX")
    return OnnxSegmenter(path, os.environ.get("DRISHYAM_ONNX_FP16")) if path else ClassicalSegmenter()
