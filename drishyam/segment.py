"""Per-class defect segmentation of a gaze crop, with uncertainty.

Two engines share one output: `ClassicalSegmenter` (imgproc evidence maps, no
weights, runs anywhere) and `OnnxSegmenter` (a dacl10k-trained model on the
OpenCV 5 DNN `ENGINE_NEW`). The policy only reads probabilities, areas and
entropy, so swapping engines changes no other code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

CLASSES = ("background", "crack", "spalling", "efflorescence", "exposed_rebar", "corrosion")
DEFECTS = CLASSES[1:]
MIN_BLOB_PX = 12  # connected components smaller than this are noise
ENTROPY_MIN_PX = 64  # fewer ambiguous pixels than this is not uncertainty worth a closer look
DEFECT_PX_BG_MAX = 0.7  # a pixel counts toward entropy when p(background) is below this


@dataclass
class Segmentation:
    probs: np.ndarray  # (C, H, W) float32, sums to 1 over C
    labels: np.ndarray  # (H, W) uint8 class index
    entropy: float  # mean binary damage-vs-background entropy over defect-like pixels, 0..1
    areas: dict[str, int]  # crop pixels per defect class
    confidence: dict[str, float]  # mean probability over each class's pixels
    latency_ms: float
    model: str

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


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


def finish(probs: np.ndarray, t0: float, model: str) -> Segmentation:
    """Labels, areas, confidences and entropy from a (C, H, W) probability stack."""
    labels = probs.argmax(axis=0).astype(np.uint8)
    areas, conf = {}, {}
    for ci, name in enumerate(DEFECTS, start=1):
        m = (labels == ci).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        small = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < MIN_BLOB_PX]
        if small:
            drop = np.isin(lab, small)
            labels[drop] = 0
            m[drop] = 0
        a = int(np.count_nonzero(m))
        areas[name] = a
        conf[name] = round(float(probs[ci][m > 0].mean()), 4) if a else 0.0
    # Uncertainty = binary entropy of "is this damage", averaged where damage is plausible.
    # 1.0 is a coin flip; a crisp mask scores ~0.2 (its anti-aliased rim).
    p_def = 1.0 - probs[0]
    defect_like = p_def > 1.0 - DEFECT_PX_BG_MAX
    if np.count_nonzero(defect_like) >= ENTROPY_MIN_PX:
        q = np.clip(p_def[defect_like], 1e-6, 1 - 1e-6)
        ent = float((-(q * np.log2(q) + (1 - q) * np.log2(1 - q))).mean())
    else:
        ent = 0.0
    return Segmentation(probs, labels, round(ent, 4), areas, conf, (time.perf_counter() - t0) * 1000, model)


class ClassicalSegmenter:
    """Hand-built evidence maps turned into calibrated class probabilities.

    No learned weights: this is the honest baseline that keeps the agent runnable
    with zero downloads. Gains map evidence 0.5 to a coin flip against background,
    which is where the entropy the policy escalates on comes from.
    """

    name = "classical-v1"
    gain = 9.0

    def __init__(self) -> None:
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def evidence(self, crop: np.ndarray) -> np.ndarray:
        """Five 0-1 maps. Thresholds are relative to the crop's own texture, so a rough
        but healthy surface does not read as damage."""
        h, w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(self.clahe.apply(gray), (3, 3), 0)
        k = max(7, (min(h, w) // 34) | 1)
        bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        bh = bh.astype(np.float32)
        med, noise = _robust(bh)
        crack = np.clip((bh - med - 3.0 * noise) / (4.0 * noise), 0, 1)

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
        crack = crack * (1 - np.clip(1.5 * spall, 0, 1))
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
        return finish(probs, t0, model)


class OnnxSegmenter:
    """A segmentation ONNX (N, C, H, W logits, dynamic H x W) on the OpenCV 5 DNN new engine.

    `class_map[i]` is the Drishyam class index that model channel i adds to, so a
    19-class dacl10k head collapses onto CODEBRIM's five defects plus background.
    """

    def __init__(self, path: str, fp16_path: str | None = None, input_size: int = 512,
                 class_map: list[int] | None = None, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.name = f"onnx:{path.rsplit('/', 1)[-1]}"
        self.nets = {"accurate": self._load(path)}
        self.nets["fp16"] = self._load(fp16_path) if fp16_path else self.nets["accurate"]
        self.size = input_size
        self.class_map = class_map
        self.mean = np.array(mean, np.float32).reshape(3, 1, 1)
        self.std = np.array(std, np.float32).reshape(3, 1, 1)

    @staticmethod
    def _load(path: str):
        return cv2.dnn.readNetFromONNX(path, engine=cv2.dnn.ENGINE_NEW)

    def predict(self, crop: np.ndarray, model: str = "accurate") -> Segmentation:
        t0 = time.perf_counter()
        h, w = crop.shape[:2]
        size = self.size if model == "accurate" else self.size // 2
        rgb = cv2.cvtColor(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        blob = ((rgb.astype(np.float32).transpose(2, 0, 1) / 255.0 - self.mean) / self.std)[None]
        net = self.nets["fp16" if model in ("fp16", "fast") else "accurate"]
        net.setInput(np.ascontiguousarray(blob, np.float32))
        logits = np.asarray(net.forward())[0].astype(np.float32)
        probs = _softmax(logits)
        if self.class_map is not None:
            merged = np.zeros((len(CLASSES), *probs.shape[1:]), np.float32)
            for ch, target in enumerate(self.class_map):
                merged[target] += probs[ch]
            probs = merged
        if probs.shape[0] != len(CLASSES):
            raise ValueError(f"model has {probs.shape[0]} classes; pass class_map to reach {len(CLASSES)}")
        probs = np.stack([cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) for p in probs])
        return finish(probs, t0, model)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 1.0
