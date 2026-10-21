"""Per-class defect segmentation of a gaze crop, with uncertainty.

Two engines share one output: `ClassicalSegmenter` (imgproc evidence maps, no
weights, runs anywhere) and `OnnxSegmenter` (a dacl10k-trained model on the
OpenCV 5 DNN `ENGINE_NEW`). The policy only reads probabilities, areas and
entropy, so swapping engines changes no other code.
"""

from __future__ import annotations

import functools
import hashlib
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import imaging
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
PORE_ASPECT = 8.0  # a crack is drafted only if its measured length is at least this many mean widths: not a blob
PORE_CAUSE = "not elongated: pore or bug hole"
# OpenCV calls in place of NumPy ones whose output is the same to the byte; False is the NumPy form. Each is True
# only if bench/opencv5_features.py found the agent's traces identical and the call measurably faster (docs/bench.md,
# "OpenCV 5 features in the pipeline"; tests/test_opencv5_features.py checks the values against its JSON).
CV_ARGMAX = True  # finish: cv2.reduceArgMax (OpenCV 4.6), the first maximum, as ndarray.argmax(axis=0) returns
CV_COMPARE = False  # _clean: cv2.compare(CMP_EQ), 255 where NumPy's == gives 1 (identical, not measurably faster)
CV_BOOL_MASKS = False  # restrict: cv2.copyTo with the bool footprint as a CV_Bool mask (new in OpenCV 5.0); identical,
# faster in 3 of its 4 timings, the fourth's interval reaching 1.02: not adopted under the rule
# The engine OnnxSegmenter reads its FP32 net with: "new" (ENGINE_NEW, the default) or "classic" (ENGINE_CLASSIC on
# a static-shape copy per input size; the FP16 net stays on ENGINE_NEW, which alone runs it). Not adopted: its output
# differs from ENGINE_NEW's by up to 2.5e-5 in probability (docs/bench.md, "OpenCV 5 DNN engines for v1").
DNN_ENGINE = os.environ.get("DRISHYAM_DNN_ENGINE", "new")


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
        m = cv2.compare(labels, ci, cv2.CMP_EQ) if CV_COMPARE else (labels == ci).astype(np.uint8)
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
    if CV_ARGMAX:
        labels = cv2.reduceArgMax(probs.reshape(len(probs), -1), 0).reshape(probs.shape[1:]).astype(np.uint8)
    else:
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
    if CV_BOOL_MASKS:  # copyTo leaves 0 outside the mask, as the product with False does
        probs = np.zeros_like(seg.probs)
        for p, out in zip(seg.probs, probs, strict=True):
            cv2.copyTo(p, valid, out)
        labels = cv2.copyTo(seg.labels, valid)
    else:
        probs = seg.probs * valid
        labels = np.where(valid, seg.labels, 0).astype(np.uint8)
    probs[0] += ~valid
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


def rejected_lines(seg: Segmentation, to_l0: np.ndarray, l0_size: tuple[int, int]) -> list[dict]:
    """`seg.lines` as trace records (the step's `rejected` list): class, cause, L0 box, length, width, residual."""
    return [{"cls": "crack", "cause": STRAIGHT_CAUSE, "box": imaging.l0_box(b, to_l0, l0_size), "length_px": ln,
             "width_px": wd, "residual_px": rms} for b, ln, wd, rms in seg.lines]


def elongated(m: dict) -> bool:
    """A crack `measure` that is a line, not a blob: length >= PORE_ASPECT x mean width (a pore or bug hole is not)."""
    return m["length_px"] >= PORE_ASPECT * m["width_px"]


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


@functools.lru_cache(maxsize=8)
def _sha256(path: str, mtime_ns: int, size: int) -> str:
    """sha256 of a weights file, once per (path, mtime, size): every job loads its own segmenter."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class OnnxSegmenter:
    """A segmentation ONNX (N, C, H, W logits) on the OpenCV 5 DNN new engine.

    Each crop is letterboxed into the square `input_size` input (`input_size // 2` for the fast model): one scale
    for both axes, the rest padded with the normalised mean (zero). The logits are cropped back to the letterbox's
    content before they are resized to the crop, so a non-square crop is never squashed.
    `class_map[i]` is the Drishyam class index that model channel i adds to, so a
    19-class dacl10k head collapses onto CODEBRIM's five defects plus background.
    `engine` (default DNN_ENGINE, from DRISHYAM_DNN_ENGINE): "classic" reads the FP32 net on ENGINE_CLASSIC instead.
    """

    def __init__(self, path: str, fp16_path: str | None = None, input_size: int = 512,
                 class_map: list[int] | None = None, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 engine: str | None = None):
        self.name = f"onnx:{path.rsplit('/', 1)[-1]}"
        self.files = (path, fp16_path)  # what the nets were read from (evals.harness hashes them)
        st = os.stat(path)
        self.sha256 = _sha256(path, st.st_mtime_ns, st.st_size)  # the weights a profile was chosen with (agent.WEIGHTS)
        self.engine = engine or DNN_ENGINE
        if self.engine == "new":
            self.nets = {"accurate": self._load(path)}
            self.nets["fp16"] = self._load(fp16_path) if fp16_path else self.nets["accurate"]
        elif self.engine == "classic":
            self.nets = {"accurate": _classic(path, input_size),
                         "fp16": self._load(fp16_path) if fp16_path else _classic(path, input_size // 2)}
        else:
            raise ValueError(f"engine {self.engine!r}: 'new' or 'classic'")
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


def _classic(path: str, size: int):
    """`path` on ENGINE_CLASSIC, its symbolic input H and W set to `size` first (the classic importer folds every
    Shape node into a constant and cannot import them: bench/repro/). The copy goes through a temporary file: in
    5.0.0 readNetFromONNX(buffer, engine=ENGINE_CLASSIC) ignores the engine and returns an ENGINE_NEW net."""
    import onnx  # a dev dependency; only this off-by-default engine needs it

    m = onnx.load(path)
    for d in m.graph.input[0].type.tensor_type.shape.dim[2:]:
        if d.dim_param:
            d.dim_value = size  # assigning the value clears dim_param (a oneof)
    with tempfile.TemporaryDirectory() as tmp:
        onnx.save(m, f"{tmp}/static{size}.onnx")
        return cv2.dnn.readNetFromONNX(f"{tmp}/static{size}.onnx", engine=cv2.dnn.ENGINE_CLASSIC)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 1.0


@functools.cache
def default_segmenter():
    """The ONNX model named by DRISHYAM_ONNX (optional DRISHYAM_ONNX_FP16), else the classical engine."""
    path = os.environ.get("DRISHYAM_ONNX")
    return OnnxSegmenter(path, os.environ.get("DRISHYAM_ONNX_FP16")) if path else ClassicalSegmenter()
