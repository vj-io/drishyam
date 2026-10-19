"""Train a dacl10k segmenter for Drishyam (FR-4): smp UNet / DeepLabV3+ on a timm encoder (MobileNetV3 by default).

This file is also the Kaggle notebook: training/make_notebook.py turns its `# %%` cells into one notebook per
VARIANTS key (training/drishyam_dacl10k_kaggle.ipynb is v2a). Run headless, it runs the same cells top to bottom
(train, export FP32 + FP16 + 19-class ONNX, parity, zip); imported (tests), it only defines things.
    python training/train_dacl10k.py --data /path/to/dacl10k --out runs/dacl10k
Smoke test, no dataset and no downloads:
    python training/train_dacl10k.py --synthetic --max-steps 3 --size 128 --batch-size 2 --out /tmp/run

The head is multi-label (19 sigmoid channels, dacl10k-toolkit TARGET_LIST order from class_map.json) and the
19 -> 6 merge for Drishyam happens at export time (export_onnx.py), not here. `--recipe v2` (the default):
scale-matched crops, polygons max-pooled from 4x the target, focal BCE with frequency weights + Dice, 60 epochs,
best.pt on the mean over view scales; `--recipe v1` is the first model's (README section 5).
"""
# %% [markdown]
# # Drishyam: train the dacl10k segmenter on Kaggle
#
# Generated from `training/train_dacl10k.py` by `training/make_notebook.py` (edit the script, not this notebook).
# It trains a UNet on dacl10k (recipe v2; the encoder is the settings cell's `VARIANT`), exports the FP32 and FP16
# ONNX that Drishyam loads and the 19-class logits ONNX, checks them against PyTorch on OpenCV 5 DNN, and zips
# `best.onnx`, `best_fp16.onnx`, `best_19.onnx`, `metrics.json` and `run_info.json`.
#
# 1. **Data**: *Add Input* > Datasets > search `dacl10k-v2-devphase` and add **winterqi131/dacl10k-v2-devphase**
#    (the official v2 devphase files: 6,935 train and 975 validation images with labelme JSON; the identical
#    **sristi29raj/dacl10k-dataset** also works). Both are third-party re-uploads; the data's own licence is
#    CC BY-NC 4.0 (its `LICENSE.md`). Off Kaggle (Colab, a laptop), `kagglehub.dataset_download(
#    "winterqi131/dacl10k-v2-devphase")` fetches the same files without a token; pass its path to `--data`.
#    (The official AWS zip answers "All access to this object has been disabled" since 2026-10.)
# 2. **Settings** (right panel): Accelerator **GPU T4 x2** (one GPU is used), Internet **on** (pip and the
#    ImageNet encoder weights; needs a phone-verified account).
# 3. **Save Version > Save & Run All (Commit)**. It runs in the background (you can close the tab) for up to
#    12 h; recipe v2 asks for 60 epochs (time not measured), and `--hours 11` stops early enough to export anyway.
#    From a terminal instead: `scripts/kaggle_train.sh push v2a` (training/README.md, "Kaggle from the command line").
# 4. When the version is done: its **Output** tab > `run/drishyam_dacl10k.zip` > download.
# 5. Stopped early ("resume to continue" in the log)? New version: *Add Input* > Your Work > this notebook's
#    output, then Save & Run All again; `--resume auto` carries on from its `last.pt`.
# 6. On your machine, in the repo: `scripts/install_model.sh ~/Downloads/drishyam_dacl10k.zip` (the path the
#    browser saved it to; it installs `models/drishyam_dacl10k_<variant>.onnx` beside v1, then checks, smoke jobs and
#    a before/after eval; see training/README.md).
#
# The weights are trained on CC BY-NC 4.0 data, so they are for non-commercial use.

# %% Imports
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# %% Setup: the repo files and packages the cells below need
# In the notebook make_notebook.py fills REPO_FILES with the training helpers and drishyam/segment.py (for the
# parity check), so it runs on Kaggle without the repo; run from the repo, the files beside this script are used.
REPO_FILES: dict[str, str] = {}
SOURCE_SHA256 = ""  # make_notebook.py writes the sha256 of the sources it built the notebook from
NOTEBOOK = "__file__" not in globals()
KAGGLE = Path("/kaggle/working").is_dir()
CV5 = Path(tempfile.gettempdir()) / "cv5"  # OpenCV 5 for the DNN checks, kept apart from Kaggle's OpenCV 4
if REPO_FILES:
    HERE = Path(tempfile.gettempdir()) / "drishyam" / "training"
    for rel, text in REPO_FILES.items():
        (HERE.parent / rel).parent.mkdir(parents=True, exist_ok=True)
        (HERE.parent / rel).write_text(text)
else:
    HERE = Path("training").resolve() if NOTEBOOK else Path(__file__).resolve().parent
if KAGGLE and __name__ == "__main__":
    # Pinned: what this pipeline adds; torch, torchvision, timm and numpy stay Kaggle's own (pinning them would swap
    # the CUDA build). onnx is pinned too: Kaggle forces protobuf 5.29.5, which recent onnx wheels cannot load, and
    # this one pulls protobuf >= 6.31 (only unused TensorFlow-side packages mind). OpenCV 5 goes to its own folder:
    # Kaggle holds cv2 at 4.x, and two OpenCV wheels in one site-packages overwrite each other's cv2/. Training
    # only needs 4.x; the DNN checks (export and parity, run as child processes) need 5.
    pins = ["segmentation-models-pytorch==0.5.0", "onnx==1.23.2", "onnxslim==0.1.98", "onnxconverter-common==1.16.0"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pins], check=True)
    if not (CV5 / "cv2").is_dir():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(CV5),
                        "opencv-python-headless==5.0.0.93"], check=True)
sys.path.insert(0, str(HERE))
from dacl_common import MEAN, STD, build_model, load_class_map


# %% Data
def find_root(path: str) -> Path:
    """The folder holding annotations/ and images/ (an unzip or a Kaggle mount adds wrapper folders)."""
    p = Path(path)
    for depth in range(5):
        for c in sorted(p.glob("/".join(["*"] * depth))) if depth else [p]:
            if (c / "annotations" / "train").is_dir():
                return c
    raise FileNotFoundError(f"no annotations/train under {path}: on Kaggle, Add Input > winterqi131/dacl10k-v2-devphase;"
                            " otherwise see training/README.md for the layout")


# Recipe v2 (the default): the agent's gaze windows are --size x 2**L source px (L0 = 512 px, about 13 % of a 4000 px
# photo's side), a scale v1's 30-100 % crops never showed the model. A recipe sets the defaults of the flags below
# (parse()); any flag given on the command line wins. `--recipe v1` reproduces the first model's training.
RECIPES = {
    "v2": {"epochs": 60, "crop_mode": "v2", "raster_k": 4, "focal_gamma": 2.0, "class_weights": "freq",
           "select": "scales"},
    "v1": {"epochs": 30, "crop_mode": "v1", "raster_k": 1, "focal_gamma": 0.0, "class_weights": "none",
           "select": "whole"},
}
CROP_P = 0.85  # v2: share of train samples that are a crop; the rest are the whole photo
CROP_MIN = 0.75  # v2: smallest crop side, as a fraction of --size (just under an L0 window)
CROP_LABEL_P = 0.6  # v2: share of crops centred on a pixel of a random labelled polygon (else a uniform position)
ASPECT_JITTER = math.log(1.2)  # v2: a crop's width / height is within 1/1.2 .. 1.2 (gaze views are square)
VIEW_LEVELS = (0, 1, 2)  # per-scale validation: square crops of --size x 2**L source px (L0, L1, L2), plus the whole
VAL_STREAM = 1 << 20  # rng stream of the fixed validation windows (train epochs use streams 0, 1, ...)


def labelled_point(shapes, index, rng) -> np.ndarray | None:
    """A random pixel (x, y) of a random labelled polygon, or None. Uniform over polygons, not over labelled pixels,
    so thin, numerous cracks get their share against large spalls; the polygon is rasterised on a 64 x 64 grid."""
    polys = [s["points"] for s in shapes if s["label"] in index and len(s["points"]) >= 3]
    if not polys:
        return None
    p = np.asarray(polys[int(rng.integers(len(polys)))], np.float64)
    lo = p.min(0)
    f = 63 / max(1.0, float((p.max(0) - lo).max()))
    grid = np.zeros((64, 64), np.uint8)
    cv2.fillPoly(grid, [np.round((p - lo) * f).astype(np.int32)], 1)
    ys, xs = np.nonzero(grid)
    if not len(xs):  # a degenerate polygon: one of its vertices
        return p[int(rng.integers(len(p)))]
    j = int(rng.integers(len(xs)))
    return lo + (np.array([xs[j], ys[j]]) + rng.uniform(-0.5, 0.5, 2)) / f


def window(w: int, h: int, cw: int, ch: int, shapes, index, rng, label_p: float) -> tuple[int, int, int, int]:
    """(x0, y0, cw, ch): a cw x ch window inside a w x h photo, with probability `label_p` centred (as far as the
    photo allows) on `labelled_point`, else at a uniform position."""
    pt = labelled_point(shapes, index, rng) if rng.random() < label_p else None
    if pt is None:
        return int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1)), cw, ch
    return int(np.clip(round(pt[0] - cw / 2), 0, w - cw)), int(np.clip(round(pt[1] - ch / 2), 0, h - ch)), cw, ch


def rasterise(shapes, index: dict, box, size: int, k: int = 1) -> np.ndarray:
    """(classes, size, size) uint8 masks of the labelme polygons inside the window `box` = (x0, y0, cw, ch) in source
    px, scaled to size x size, one channel per class (a pixel can carry several classes).

    k = 1 fills each polygon scaled straight to the target size (recipe v1): one thinner than a target pixel can
    vanish. k > 1 fills it, outline included, at k x the target and max-pools each k x k block: a target pixel is on
    when any of its sub-pixels is covered, so a 1 px crack in a 4000 px photo is still there in a 512 px whole-photo
    view (the outline marks the sub-pixels a polygon thinner than one of them passes through)."""
    x0, y0, cw, ch = box
    R = size * k
    out = np.zeros((len(index), size, size), np.uint8)
    polys: dict[int, list[np.ndarray]] = {}
    for s in shapes:
        c = index.get(s["label"])
        if c is not None and len(s["points"]) >= 3:
            pts = (np.asarray(s["points"], np.float64) - (x0, y0)) * (R / cw, R / ch)
            polys.setdefault(c, []).append(np.round(pts).astype(np.int32))
    canvas = np.zeros((R, R), np.uint8) if k > 1 else None
    for c, ps in polys.items():
        if canvas is None:
            for p in ps:  # one call per polygon: overlapping polygons in one fillPoly call cancel out
                cv2.fillPoly(out[c], [p], 1)
            continue
        canvas[:] = 0
        for p in ps:
            cv2.fillPoly(canvas, [p], 1)
        cv2.polylines(canvas, ps, True, 1)
        out[c] = canvas.reshape(size, k, size, k).max(axis=(1, 3))
    return out


class Dacl10k(Dataset):
    """Official split: annotations/{train,validation}/*.json (labelme polygons) + images/<split>/*.jpg.

    A window of the full-resolution photo is resized to size x size with INTER_AREA, as OnnxSegmenter resizes a
    gaze view (and like the dacl10k paper's 512 x 512 baselines). Train windows (`crop_box`): v2, with probability
    `crop_p` a square-ish crop whose side is log-uniform between `crop_min_px` and the photo's short side (every
    pyramid level the agent looks at), `crop_label_p` of them centred on a labelled pixel, else the whole photo;
    v1, 70 % of the time a crop of 30-100 % of each side. `views` (validation): every sample is a stack of fixed,
    seeded windows, one per crop side in source px (clamped to the short side; 0 = the whole photo), the same
    whatever the training flags. Masks: `rasterise`, at min(raster_k, the window's downscale factor rounded up) x
    the target size (raster_k 1 = recipe v1).
    """

    def __init__(self, root: Path, split: str, size: int, names: list[str], train: bool, seed: int,
                 crop_mode: str = "v2", crop_p: float = CROP_P, crop_min_px: float = 0.0,
                 crop_label_p: float = CROP_LABEL_P, raster_k: int = 4, views: tuple[int, ...] = ()):
        self.ann = sorted((root / "annotations" / split).glob("*.json"))
        if not self.ann:
            raise FileNotFoundError(f"no annotations in {root / 'annotations' / split}")
        self.img_dir = root / "images" / split
        self.size, self.train, self.seed, self.epoch = size, train, seed, 0
        self.index = {n: i for i, n in enumerate(names)}
        self.crop_mode, self.crop_p, self.crop_label_p, self.views = crop_mode, crop_p, crop_label_p, views
        self.crop_min_px = crop_min_px or CROP_MIN * size
        self.raster_k = raster_k

    def __len__(self) -> int:
        return len(self.ann)

    def meta(self, i: int) -> dict:
        return json.loads(self.ann[i].read_text())

    def load(self, i: int):
        d = self.meta(i)
        path = self.img_dir / d["imageName"]
        # EXIF orientation applied (cv2's default), as labelme applied it when the polygons were drawn: the 10 of 7,910
        # dacl10k photos with a 90-degree EXIF rotation are labelled in the rotated frame (their JSON size is transposed).
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        h, w = img.shape[:2]
        if (w, h) != (d["imageWidth"], d["imageHeight"]):
            raise ValueError(f"{path} decodes to {w}x{h} but {self.ann[i].name} says "
                             f"{d['imageWidth']}x{d['imageHeight']}: its polygons would be misplaced")
        return d, img

    def crop_box(self, w: int, h: int, shapes, rng) -> tuple[int, int, int, int]:
        """The train window (x0, y0, cw, ch) in source px."""
        if self.crop_mode == "v1":
            if rng.random() >= 0.7:
                return 0, 0, w, h
            S = self.size
            side = rng.uniform(0.3, 1.0)
            cw, ch = max(S // 4, int(w * side)), max(S // 4, int(h * side))
            cw, ch = min(cw, w), min(ch, h)
            return int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1)), cw, ch
        if rng.random() >= self.crop_p:
            return 0, 0, w, h
        short = min(w, h)
        side = math.exp(rng.uniform(math.log(min(self.crop_min_px, short)), math.log(short)))
        a = math.sqrt(math.exp(rng.uniform(-ASPECT_JITTER, ASPECT_JITTER)))
        cw, ch = min(w, max(1, round(side * a))), min(h, max(1, round(side / a)))
        return window(w, h, cw, ch, shapes, self.index, rng, self.crop_label_p)

    def render(self, img: np.ndarray, shapes, box, rng=None):
        """The window `box` as the net's (3, S, S) input and (classes, S, S) uint8 mask; `rng`: train augmentation."""
        S = self.size
        x0, y0, cw, ch = box
        x = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (S, S), interpolation=cv2.INTER_AREA)
        mask = rasterise(shapes, self.index, box, S, max(1, min(self.raster_k, math.ceil(max(cw, ch) / S))))
        if rng is not None:
            if rng.random() < 0.5:
                x, mask = x[:, ::-1], mask[:, :, ::-1]
            if rng.random() < 0.8:  # brightness / contrast
                x = np.clip(x.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25), 0, 255)
        rgb = cv2.cvtColor(np.ascontiguousarray(x).astype(np.uint8), cv2.COLOR_BGR2RGB)
        rgb = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        # uint8 mask (a quarter of float32 through worker shared memory and pinning); the loops convert on the device
        return torch.from_numpy(rgb.transpose(2, 0, 1).copy()), torch.from_numpy(np.ascontiguousarray(mask))

    def __getitem__(self, i: int):
        d, img = self.load(i)
        h, w = img.shape[:2]
        if self.views:  # the same windows every epoch: a fixed validation set per scale
            rng = np.random.default_rng((self.seed, VAL_STREAM, i))
            boxes = [window(w, h, min(s, w, h), min(s, w, h), d["shapes"], self.index, rng, CROP_LABEL_P) if s
                     else (0, 0, w, h) for s in self.views]
            xs, ms = zip(*(self.render(img, d["shapes"], b) for b in boxes))
            return torch.stack(xs), torch.stack(ms)
        if not self.train:
            return self.render(img, d["shapes"], (0, 0, w, h))
        rng = np.random.default_rng((self.seed, self.epoch, i))  # same sample + epoch => same augmentation
        return self.render(img, d["shapes"], self.crop_box(w, h, d["shapes"], rng), rng)


def make_synthetic(root: Path, names: list[str], size: int, seed: int, n_train: int = 16, n_val: int = 8) -> Path:
    """A fabricated dacl10k-layout folder for smoke runs: noise photos of 5 x 4 --size px with small random labelled
    quadrilaterals, so the real loader, the v2 crops and every validation scale run. It proves the pipeline runs,
    and says nothing about accuracy."""
    rng = np.random.default_rng(seed)
    W, H = 5 * size, 4 * size
    for split, n in (("train", n_train), ("validation", n_val)):
        (root / "annotations" / split).mkdir(parents=True, exist_ok=True)
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            name = f"{split}_{i}.jpg"
            cv2.imwrite(str(root / "images" / split / name), rng.integers(0, 256, (H, W, 3), dtype=np.uint8))
            shapes = [{"label": names[int(rng.integers(len(names)))], "shape_type": "polygon",
                       "points": np.clip(rng.random(2) * (W, H) + rng.uniform(-size / 2, size / 2, (4, 2)),
                                         0, (W - 1, H - 1)).tolist()} for _ in range(8)]
            (root / "annotations" / split / f"{split}_{i}.json").write_text(
                json.dumps({"imageName": name, "imageWidth": W, "imageHeight": H, "shapes": shapes}))
    return root


def class_weights(ds: Dacl10k, groups, power: float, wmax: float) -> tuple[np.ndarray, np.ndarray]:
    """(share of train pixels per channel, loss weight per channel), from the train JSONs alone (no image is read).

    Share: the channel's polygon area (cv2.contourArea) over all photos' area. Weight: a channel merged into a Drishyam
    defect (class_map.json) gets clip((median share / share) ** power, 1, wmax), so the rare safety-critical ones
    (Crack, ExposedRebars) count up to wmax times; a channel dropped to background keeps 1. A weight multiplies the
    channel's positive and negative terms alike: it changes how much the channel drives the shared features, not
    where its sigmoid crosses 0.5 (BCE's pos_weight would move that threshold, so the merged ONNX's labels too)."""
    area, total = np.zeros(len(ds.index)), 0.0
    for i in range(len(ds)):
        d = ds.meta(i)
        total += d["imageWidth"] * d["imageHeight"]
        for s in d["shapes"]:
            c = ds.index.get(s["label"])
            if c is not None and len(s["points"]) >= 3:
                area[c] += cv2.contourArea(np.asarray(s["points"], np.float32))
    share = area / max(total, 1.0)
    w = np.ones(len(share))
    defect = sorted({c for g in groups for c in g})
    w[defect] = np.clip((np.median(share) / np.maximum(share[defect], 1e-9)) ** power, 1.0, wmax)
    return share, w


def seg_loss(logits: torch.Tensor, y: torch.Tensor, dice, gamma: float, weights, bce_weight: float,
             dice_weight: float) -> torch.Tensor:
    """bce_weight x focal BCE (gamma 0 is plain BCE; `weights`: (1, C, 1, 1) per channel, or None) + dice_weight x
    Dice. Recipe v1 is gamma 0, no weights, 1 + 1: BCEWithLogitsLoss() + Dice. Focal down-weights the easy pixels
    (mostly clean background) so the hard ones drive the mean; it treats both labels alike, so its optimum still
    crosses 0.5 where the true probability does. Dice is per class already, so it is not weighted."""
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    if gamma:
        loss = loss * (1 - torch.exp(-loss)) ** gamma  # (1 - p_t) ** gamma, p_t = exp(-BCE)
    if weights is not None:
        loss = loss * weights
    return bce_weight * loss.mean() + dice_weight * dice(logits, y)


# %% Metrics: IoU per class accumulated over the whole validation set, as in the dacl10k paper
def _counts(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    dims = (0, 2, 3)
    return torch.stack([(pred & gt).sum(dims), (pred & ~gt).sum(dims), (~pred & gt).sum(dims)])  # tp, fp, fn


def _iou(c: torch.Tensor) -> torch.Tensor:
    tp, fp, fn = c.double()
    den = tp + fp + fn
    return torch.where(den > 0, tp / den.clamp(min=1), torch.full_like(den, float("nan")))  # nan: class never seen


def _merge(t: torch.Tensor, groups) -> torch.Tensor:
    return torch.stack([t[:, g].any(1) for g in groups], 1)  # a Drishyam class is on if any mapped channel is


@torch.no_grad()
def evaluate(model, loader, device, groups, amp, max_batches=0):
    """Per view of the loader's (B, views, ...) batches: (IoU for the 19 dacl10k classes, IoU for the 5 Drishyam
    defects after the class_map merge)."""
    model.eval()
    c19 = c5 = 0
    for b, (x, y) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        B, V = x.shape[:2]
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(x.flatten(0, 1).to(device))
        pred, gt = (logits.float() > 0).view(B, V, *logits.shape[1:]), y.to(device) > 0  # logit > 0 is sigmoid > 0.5
        c19 = c19 + torch.stack([_counts(pred[:, v], gt[:, v]) for v in range(V)])
        c5 = c5 + torch.stack([_counts(_merge(pred[:, v], groups), _merge(gt[:, v], groups)) for v in range(V)])
    return [(_iou(t19).cpu().numpy(), _iou(t5).cpu().numpy()) for t19, t5 in zip(c19, c5)]


# %% Training loop
def parse(argv=None):
    p = argparse.ArgumentParser(description="Train a dacl10k segmenter for Drishyam (see training/README.md)")
    p.add_argument("--data", help="dacl10k root (annotations/ + images/); required unless --synthetic")
    p.add_argument("--out", default="runs/dacl10k", help="checkpoints (last.pt, best.pt) and metrics.json")
    p.add_argument("--variant", default="", help="a name for these weights (the notebook's VARIANT): "
                   "scripts/install_model.sh installs a v2 run as models/drishyam_dacl10k_<variant>.onnx")
    p.add_argument("--recipe", choices=[*RECIPES], default="v2",
                   help="the defaults of --epochs, --crop-mode, --raster-k, --focal-gamma, --class-weights and "
                   "--select (RECIPES above); v1 = the first model's training")
    p.add_argument("--arch", choices=["unet", "deeplabv3plus"], default="unet")
    p.add_argument("--encoder", default="tu-mobilenetv3_large_100")
    p.add_argument("--encoder-weights", default="imagenet", help="'imagenet' (downloads) or 'none'")
    p.add_argument("--size", type=int, default=512, help="train/val resolution, a multiple of 32")
    p.add_argument("--epochs", type=int, help="v2 60, v1 30")
    p.add_argument("--crop-mode", choices=["v2", "v1"], help="v2: scale-matched crops (CROP_* above); v1: 30-100 %% "
                   "of each side, 70 %% of the time")
    p.add_argument("--crop-p", type=float, default=CROP_P, help="v2 crops: share of train samples that are a crop")
    p.add_argument("--crop-min-px", type=float, default=0.0,
                   help=f"v2 crops: smallest side in source px; 0 = {CROP_MIN} x --size (an L0 gaze window is --size)")
    p.add_argument("--crop-label-p", type=float, default=CROP_LABEL_P,
                   help="v2 crops: share centred on a pixel of a random labelled polygon")
    p.add_argument("--raster-k", type=int, help="masks of a downscaled window are drawn at up to this many times "
                   "the target size and max-pooled (rasterise()); v2 4, v1 1 (drawn at the target size)")
    p.add_argument("--focal-gamma", type=float, help="focal exponent of the BCE term; 0 = plain BCE (v1); v2 2")
    p.add_argument("--bce-weight", type=float, default=1.0, help="weight of the (focal) BCE term")
    p.add_argument("--dice-weight", type=float, default=1.0, help="weight of the Dice term")
    p.add_argument("--class-weights", choices=["freq", "none"],
                   help="freq: rare channels merged into a Drishyam defect weighted up from their train pixel share "
                   "(class_weights()); v2 freq, v1 none")
    p.add_argument("--class-weight-power", type=float, default=0.5, help="freq: (median share / share) ** this")
    p.add_argument("--class-weight-max", type=float, default=4.0, help="freq: the largest weight")
    p.add_argument("--select", choices=["scales", "whole"], help="best.pt: the mean val mIoU5 over the view scales "
                   "(v2), or the whole photo's alone (v1)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # macOS: DataLoader worker processes hang at interpreter exit here (torch 2.2.2 + Python 3.12.13, reproduced with
    # a 10-line script unrelated to this one), so default to in-process loading there.
    p.add_argument("--workers", type=int, default=0 if sys.platform == "darwin" else min(4, os.cpu_count() or 1))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0, help="stop after N optimiser steps (smoke runs); 0 = off")
    p.add_argument("--resume", help="a last.pt to continue from, or 'auto': <out>/last.pt if there is one, else a "
                   "last.pt under /kaggle/input (an earlier notebook version's output added as input). It refuses a "
                   "checkpoint trained with other RESUME_KEYS flags")
    p.add_argument("--hours", type=float, default=0, help="start no epoch that would end after this many hours "
                   "(a Kaggle session ends at 12); 0 = off. Resume to continue")
    p.add_argument("--synthetic", action="store_true", help="fabricated tiny dataset, no downloads (smoke test)")
    p.add_argument("--log-every", type=int, default=50)
    a = p.parse_args(argv)
    for k, v in RECIPES[a.recipe].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    if not a.synthetic and not a.data:
        p.error("--data is required unless --synthetic")
    if a.size % 32:
        p.error("--size must be a multiple of 32")
    if a.raster_k < 1:
        p.error("--raster-k must be at least 1")
    return a


# The flags that define what a checkpoint was trained to do: a resumed run must keep them all, or its schedule,
# data and selection would change mid-run (a v1 checkpoint, saved before --recipe existed, counts as --recipe v1).
RESUME_KEYS = ("recipe", "arch", "encoder", "size", "epochs", "batch_size", "lr", "weight_decay", "seed", "crop_mode",
               "crop_p", "crop_min_px", "crop_label_p", "raster_k", "focal_gamma", "bce_weight", "dice_weight",
               "class_weights", "class_weight_power", "class_weight_max", "select")


def resume_mismatch(saved: dict, a) -> list[str]:
    """The RESUME_KEYS on which this session's flags differ from the checkpoint's `args` (keys it lacks are skipped)."""
    saved = {"recipe": "v1", **saved}
    return [f"--{k.replace('_', '-')} {saved[k]} (now {getattr(a, k)})" for k in RESUME_KEYS
            if k in saved and saved[k] != getattr(a, k)]


def select_rule(a) -> str:
    """How best.pt is chosen (recorded in run_info.json)."""
    if a.select == "whole":
        return "best.pt: the epoch with the highest val_miou5 (mean IoU of Drishyam's 5 defects on whole photos)"
    sides = ", ".join(f"{a.size << lv}" for lv in VIEW_LEVELS)
    return (f"best.pt: the epoch with the highest scale_mean, the mean over 4 views (square crops of {sides} source "
            "px, each clamped to the photo's short side, 60 % centred on a labelled pixel, seeded and the same every "
            "epoch, and the whole photo) of the validation mean IoU of Drishyam's 5 defects")


def run_info(start_epoch: int, a=None, share=None, weights=None) -> dict:
    """This session's package versions, source and flags, one entry of <out>/run_info.json (zipped with the weights).
    On Kaggle there is no git checkout: `source_sha256` (make_notebook.source_sha256 at the commit the notebook was
    built from) identifies the code instead."""
    from importlib import metadata

    def version(dist: str, path: list[str] | None = None) -> str | None:
        found = next(iter(metadata.distributions(name=dist, **({"path": path} if path else {}))), None)
        return found.version if found else None

    names = ("torch", "torchvision", "timm", "segmentation-models-pytorch", "onnx", "onnxslim", "onnxconverter-common",
             "numpy")
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        git = ""
    if not SOURCE_SHA256:
        from make_notebook import source_sha256
    info = {"start_epoch": start_epoch, "python": sys.version.split()[0],
            "versions": {**{n: version(n) for n in names}, "cv2": cv2.__version__,
                         "cv2_export": version("opencv-python-headless", [str(CV5)]) if KAGGLE else cv2.__version__},
            "git_sha": git or None, "source_sha256": SOURCE_SHA256 or source_sha256()}
    if a is not None:
        info.update(flags=vars(a), select_rule=select_rule(a))
    if share is not None:
        info["train_pixel_share"] = dict(zip(load_class_map()[0], map(float, share)))
    if weights is not None:
        info["class_weights"] = dict(zip(load_class_map()[0], map(float, weights)))
    return info


def auto_resume(out: Path, search: Path = Path("/kaggle/input")) -> str | None:
    """`--resume auto`: <out>/last.pt, else the newest last.pt under `search`, copied into <out> with the
    best.pt, metrics.json and run_info.json beside it so the history carries on."""
    if (out / "last.pt").exists():
        return str(out / "last.pt")
    # glob.glob, not Path.glob: pathlib's ** does not enter symlinked folders, and an input mount may be one
    found = sorted(map(Path, glob.glob(str(search / "**" / "last.pt"), recursive=True)), key=lambda f: f.stat().st_mtime)
    if not found:
        print(f"--resume auto: no last.pt in {out} or under {search}, starting from epoch 0")
        return None
    for name in ("last.pt", "best.pt", "metrics.json", "run_info.json"):
        if (found[-1].parent / name).exists():
            shutil.copy2(found[-1].parent / name, out / name)
    print(f"resuming from {found[-1]}")
    return str(out / "last.pt")


def main(argv=None):
    import segmentation_models_pytorch as smp

    a = parse(argv)
    t_run = time.time()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    names, drishyam, groups = load_class_map()
    C = len(names)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.resume == "auto":
        a.resume = auto_resume(out)
    ck = torch.load(a.resume, map_location=device) if a.resume else None
    if ck is not None and (diff := resume_mismatch(ck.get("args", {}), a)):
        raise SystemExit(f"refusing to resume {a.resume}: it was trained with " + ", ".join(diff) + ". Resume it with "
                         "its own flags, or start afresh in another --out (or without --resume)")
    if a.synthetic:
        a.encoder_weights, a.workers = "none", 0
        root = make_synthetic(out / "synthetic", names, a.size, a.seed)
    else:
        root = find_root(a.data)
    views = (*(a.size << lv for lv in VIEW_LEVELS), 0)  # source px; 0 = the whole photo (v1's validation)
    scales = [str(v) if v else "whole" for v in views]
    train_ds = Dacl10k(root, "train", a.size, names, True, a.seed, a.crop_mode, a.crop_p, a.crop_min_px,
                       a.crop_label_p, a.raster_k)
    val_ds = Dacl10k(root, "validation", a.size, names, False, a.seed, raster_k=a.raster_k, views=views)
    kw = {"num_workers": a.workers, "pin_memory": amp}
    train_dl = DataLoader(train_ds, a.batch_size, shuffle=True, drop_last=True,
                          generator=torch.Generator().manual_seed(a.seed), **kw)
    val_dl = DataLoader(val_ds, max(1, a.batch_size // len(views)), shuffle=False, **kw)  # a sample is every view

    model = build_model(a.arch, a.encoder, None if a.encoder_weights == "none" else a.encoder_weights, C).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    total = a.max_steps or a.epochs * len(train_dl)
    warm = min(2 * len(train_dl), total // 10)  # the paper's baselines: cosine schedule, 2-epoch warm-up
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    # torch.amp.GradScaler is torch >= 2.3 (Kaggle); torch.cuda.amp's is deprecated there but is all 2.2.2 (the Mac) has
    scaler = (torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=amp))
    share = weights = None
    if a.class_weights == "freq":  # measured once per session from the train JSONs (deterministic), and logged
        share, weights = class_weights(train_ds, groups, a.class_weight_power, a.class_weight_max)
        to = {c: drishyam[i + 1] for i, g in enumerate(groups) for c in g}
        print("train pixel share and loss weight per dacl10k channel (--class-weights freq):")
        for c in np.argsort(share):
            print(f"  {names[c]:<14}{share[c] * 100:>8.3f} %  weight {weights[c]:.2f}  -> {to.get(c, 'background')}")
    w_loss = None if weights is None else torch.tensor(weights, dtype=torch.float32, device=device).view(1, C, 1, 1)
    dice = smp.losses.DiceLoss("multilabel", from_logits=True)

    key = {"scales": "scale_mean", "whole": "val_miou5"}[a.select]  # the metrics.json field best.pt is chosen on
    start, step, best, history = 0, 0, -1.0, []
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, step, best = ck["epoch"] + 1, ck["step"], ck.get("best", -1.0)
        history = json.loads((out / "metrics.json").read_text()) if (out / "metrics.json").exists() else []
    # one entry per session: a run resumed on another Kaggle image says which epochs which versions trained
    info = json.loads((out / "run_info.json").read_text()) if a.resume and (out / "run_info.json").exists() else []
    (out / "run_info.json").write_text(json.dumps([*info, run_info(start, a, share, weights)], indent=1))
    print(f"recipe {a.recipe}: crops {a.crop_mode}, raster k {a.raster_k}, focal gamma {a.focal_gamma}, class weights "
          f"{a.class_weights}, loss {a.bce_weight} x BCE + {a.dice_weight} x Dice; {select_rule(a)}")
    print(f"{a.arch}/{a.encoder}  device={device.type} amp={amp}  train={len(train_ds)} val={len(val_ds)}  "
          f"steps/epoch={len(train_dl)}  total_steps={total}  start epoch {start}")

    for epoch in range(start, a.epochs):
        train_ds.epoch = epoch
        model.train()
        t0, run = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).float()
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = seg_loss(logits.float(), y, dice, a.focal_gamma, w_loss, a.bce_weight, a.dice_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run = loss.item() if run == 0.0 else 0.98 * run + 0.02 * loss.item()
            if step % a.log_every == 0 or step == 1:
                print(f"epoch {epoch} step {step}/{total} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
            if a.max_steps and step >= a.max_steps:
                break

        t_val = time.time()
        per_view = evaluate(model, val_dl, device, groups, amp, max_batches=2 if a.max_steps else 0)
        iou19, iou5 = per_view[-1]  # the whole photo: v1's validation numbers
        miou, miou5 = float(np.nanmean(iou19)), float(np.nanmean(iou5))
        by_scale = {s: {"miou19": float(np.nanmean(v19)), "miou5": float(np.nanmean(v5)),
                        "iou19": dict(zip(names, map(float, v19))), "iou5": list(map(float, v5))}
                    for s, (v19, v5) in zip(scales, per_view)}
        scale_mean = float(np.nanmean([v["miou5"] for v in by_scale.values()]))
        print(f"epoch {epoch} done in {time.time() - t0:.0f}s (validation {time.time() - t_val:.0f}s)  "
              f"val mIoU(19) {miou:.4f}  Drishyam-5 mean {miou5:.4f}  over scales {scale_mean:.4f}")
        print(f"  {'view (source px)':<24}" + "".join(f"{s:>8}" for s in scales))
        for c, n in enumerate(names):
            print(f"  {n:<24}" + "".join(f"{v19[c]:>8.3f}" for v19, _ in per_view))
        for c, n in enumerate(drishyam[1:]):
            print(f"  [merged] {n:<15}" + "".join(f"{v5[c]:>8.3f}" for _, v5 in per_view))
        print(f"  {'Drishyam-5 mean':<24}" + "".join(f"{by_scale[s]['miou5']:>8.3f}" for s in scales))
        # best.pt is what ships: chosen on the five defects the exported graph outputs, not on all 19 classes, and
        # (v2) averaged over the scales the agent looks at, so it is the epoch that works across the pyramid
        score = scale_mean if a.select == "scales" else miou5
        best = max(best, score)
        history.append({"epoch": epoch, "step": step, "train_loss": run, "val_miou19": miou, "val_miou5": miou5,
                        "iou19": dict(zip(names, map(float, iou19))), "iou5": list(map(float, iou5)),
                        "scales": by_scale, "scale_mean": scale_mean, "select": a.select, "best": score >= best,
                        "epoch_s": round(time.time() - t0, 1), "val_s": round(time.time() - t_val, 1)})
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "scaler": scaler.state_dict(), "epoch": epoch, "step": step, "best": best, "arch": a.arch,
              "encoder": a.encoder, "classes": C, "names": names, "val_miou19": miou, "val_miou5": miou5,
              "scale_mean": scale_mean, "args": vars(a)}
        torch.save(ck, out / "last.pt.tmp")
        os.replace(out / "last.pt.tmp", out / "last.pt")  # every epoch; a killed session never leaves a torn file
        if score >= best:
            torch.save(ck, out / "best.pt")
        (out / "metrics.json").write_text(json.dumps(history, indent=1))
        if a.max_steps and step >= a.max_steps:
            break
        if a.hours and epoch + 1 < a.epochs and time.time() - t_run + (time.time() - t0) > a.hours * 3600:
            print(f"--hours {a.hours}: stopping after epoch {epoch}, the next would end too late; resume to continue")
            break
    print(f"checkpoints in {out}  (best.pt: {key} {best:.4f})")


def tool(script: str, *args) -> int:
    """Run a training/ script in a child process and show its output; on Kaggle with OpenCV 5 first on the path."""
    env = dict(os.environ)
    if KAGGLE:
        env["PYTHONPATH"] = os.pathsep.join([str(CV5), *filter(None, [env.get("PYTHONPATH")])])
    r = subprocess.run([sys.executable, str(HERE / script), *map(str, args)], check=False, env=env,
                       capture_output=True, text=True)
    print(r.stdout + r.stderr)
    return r.returncode


# %% Settings
# The notebook runs with these flags, then DACL_FLAGS if set (a flag given there wins: Colab sets --data and --out
# there, a smoke run --synthetic); a headless run takes them from its command line alone.
# Every flag is in parse() above. --hours 11 leaves time to export inside Kaggle's 12 h; --resume auto picks up
# a run that stopped (this session's /kaggle/working/run, or an earlier version's output added as input).
# make_notebook.py writes one notebook per VARIANTS key (Kaggle kernel vaibhav2105/drishyam-dacl10k-train-<key>,
# training/kaggle/<key>/); the VARIANT line is all that differs between them. Each variant is one flag line.
VARIANT = "v2a"
VARIANTS = {
    "v2a": "",  # recipe v2 on v1's UNet + MobileNetV3-Large: the one Drishyam can deploy today
    "v2b": "--encoder tu-efficientnet_b2",  # recipe v2 on a UNet + EfficientNet-B2: about 1.25x v2a's CPU per look (README 5b)
}
NOTEBOOK_FLAGS = f"--data /kaggle/input --out /kaggle/working/run --resume auto --hours 11 --variant {VARIANT} " \
                 + VARIANTS[VARIANT]
if __name__ == "__main__":  # true in a notebook too: importing this module (tests) skips every cell from here
    FLAGS = shlex.split(NOTEBOOK_FLAGS) + shlex.split(os.environ.get("DACL_FLAGS", "")) if NOTEBOOK else sys.argv[1:]
    A = parse(FLAGS)
    OUT, ONNX = Path(A.out), Path(A.out) / "onnx"

# %% Data and GPU check
if __name__ == "__main__":
    ROOT = None if A.synthetic else find_root(A.data)
    if ROOT:
        n_train, n_val = (len(list((ROOT / "annotations" / s).glob("*.json"))) for s in ("train", "validation"))
        full = (n_train, n_val) == (6935, 975)
        print(f"dacl10k at {ROOT}: {n_train} train, {n_val} validation"
              + ("" if full else "  (v2 devphase has 6935 and 975: is this the whole set?)"))
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 2**30:.0f} GB  (torch {torch.__version__}, cv2 {cv2.__version__})")
    elif KAGGLE and not A.synthetic:
        raise SystemExit("no GPU: Settings > Accelerator > GPU T4 x2, then run again")
    else:
        print(f"no GPU: training on CPU, fine for --synthetic smoke runs only (torch {torch.__version__})")

# %% Train: <out>/last.pt and metrics.json every epoch, best.pt at the best mean IoU of Drishyam's 5 defects
if __name__ == "__main__":
    main(FLAGS)

# %% Export: FP32 and FP16 ONNX with the 19 -> 6 merge in the graph, and the 19-class logits; OpenCV 5 DNN at 256, 512
if __name__ == "__main__":
    CKPT = OUT / "best.pt" if (OUT / "best.pt").exists() else OUT / "last.pt"
    if tool("export_onnx.py", "--ckpt", CKPT, "--out", ONNX):
        raise SystemExit("export failed: see the output above")
    FP32, FP16, F19 = (ONNX / f"drishyam_dacl10k{s}.onnx" for s in ("", ".fp16", "_19"))

# %% Parity: OnnxSegmenter on OpenCV 5 DNN vs PyTorch, 4 crops (validation images, or noise for --synthetic)
if __name__ == "__main__":
    samples = Path(tempfile.mkdtemp())
    val = sorted((ROOT / "images" / "validation").glob("*.jpg"))[:4] if ROOT else []
    noise = np.random.default_rng(0)
    for i in range(4):
        img = cv2.imread(str(val[i])) if i < len(val) else noise.integers(0, 256, (576, 768, 3), dtype=np.uint8)
        cv2.imwrite(str(samples / f"{i}.jpg"), cv2.resize(img, (768, 576), interpolation=cv2.INTER_AREA))
    PARITY = tool("parity.py", "--ckpt", CKPT, "--onnx", FP32, *(["--onnx-fp16", FP16] if FP16.exists() else []),
                  "--samples", samples, "--n", 4)
    PARITY_OK = PARITY in (0, 3)  # 3 is parity.FP16_OVER: FP32 within tolerance, only FP16 over (it is left out)

# %% Package: <out>/drishyam_dacl10k.zip: best.onnx, best_fp16.onnx (if its parity passed), best_19.onnx, metrics, run_info
if __name__ == "__main__":
    ZIP = OUT / "drishyam_dacl10k.zip"
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for src, name in ((FP32, "best.onnx"), (FP16, "best_fp16.onnx"), (F19, "best_19.onnx"),
                          (OUT / "metrics.json", "metrics.json"), (OUT / "run_info.json", "run_info.json")):
            if src.exists() and (src != FP16 or PARITY == 0):
                z.write(src, name)
    print(f"{ZIP}: {', '.join(z.namelist())}, {ZIP.stat().st_size / 1e6:.1f} MB")
    if PARITY == 0:
        print("parity ok")
    elif PARITY_OK:
        print("FP32 parity ok, FP16 over its tolerance (table above): best_fp16.onnx is left out of the zip, so "
              "Drishyam's fast looks will use the FP32 net")
    else:
        print("PARITY FAILED (table above): do not install these weights")
    print("next, on your machine in the repo, with the zip in ~/Downloads (or give its path, or the folder Safari unpacked it into):")
    print("  scripts/install_model.sh ~/Downloads/drishyam_dacl10k.zip"
          + (f"   (installs models/drishyam_dacl10k_{A.variant or 'v2'}.onnx beside v1)" if A.recipe == "v2" else ""))
    if not NOTEBOOK and not PARITY_OK:
        sys.exit(1)
