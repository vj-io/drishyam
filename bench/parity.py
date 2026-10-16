"""Mask parity between two OpenCV builds: segment K fixed crops, save the label maps, compare two saved files.

    python -m bench.parity --save results/A/parity.npz [--k 24] [--inputs results/A/parity.npz]
    python -m bench.parity --compare results/A/parity.npz results/B/parity.npz [--min-agreement 0.99]

The crops come from `evals.synth` (seeded) and are stored in the .npz, so a second arm can segment the very
same pixels with `--inputs`; without it each arm renders its own crops and the comparison also contains any
build-to-build difference in the renderer (`inputs_identical` says which case you are in).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from bench.micro import provenance
from drishyam.segment import CLASSES, ClassicalSegmenter
from evals import synth

SEED, SIZE, CROP = 9000, (1024, 768), 512


def make_crops(k: int) -> np.ndarray:
    """K 512x512 BGR crops centred on ground-truth boxes (NumPy slicing only, so the crop step cannot differ by build)."""
    crops = []
    for i in range(k):
        img, gt = synth.generate(SEED + i, *SIZE)
        x, y, w, h = gt[0]["box"] if gt else (SIZE[0] // 2, SIZE[1] // 2, 0, 0)
        x0 = int(np.clip(x + w // 2 - CROP // 2, 0, SIZE[0] - CROP))
        y0 = int(np.clip(y + h // 2 - CROP // 2, 0, SIZE[1] - CROP))
        crops.append(img[y0:y0 + CROP, x0:x0 + CROP].copy())
    return np.stack(crops)


def save(path: str, k: int, inputs: str | None) -> None:
    crops = np.load(inputs)["crops"] if inputs else make_crops(k)
    seg = ClassicalSegmenter()
    labels = np.stack([seg.predict(c, "accurate").labels for c in crops])
    meta = provenance()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, crops=crops, labels=labels, classes=np.array(CLASSES), meta=json.dumps(meta))
    print(f"wrote {path}: {len(crops)} crops, cv2 {meta['cv2']}, build {meta['build']}")


def compare(a: np.ndarray, b: np.ndarray, classes=CLASSES) -> dict:
    """Pixel agreement (overall and worst crop) and per-class IoU over all crops of two (K, H, W) label stacks."""
    if a.shape != b.shape:
        raise ValueError(f"label stacks differ in shape: {a.shape} vs {b.shape}")
    same = a == b
    iou = {}
    for i, name in enumerate(classes):
        inter, union = np.count_nonzero((a == i) & (b == i)), np.count_nonzero((a == i) | (b == i))
        iou[name] = round(inter / union, 6) if union else None
    return {"crops": int(a.shape[0]), "pixel_agreement": round(float(same.mean()), 6),
            "worst_crop_agreement": round(float(same.reshape(len(a), -1).mean(1).min()), 6), "class_iou": iou,
            "identical_crops": int(same.reshape(len(a), -1).all(1).sum())}


def compare_files(pa: str, pb: str) -> dict:
    a, b = np.load(pa), np.load(pb)
    out = compare(a["labels"], b["labels"])
    out["inputs_identical"] = bool(np.array_equal(a["crops"], b["crops"]))
    out["a"], out["b"] = ({k: json.loads(str(f["meta"]))[k] for k in ("cv2", "build", "cpu")} for f in (a, b))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bench.parity", description=__doc__.split("\n\n")[0])
    ap.add_argument("--save", metavar="ARM.npz")
    ap.add_argument("--compare", nargs=2, metavar=("A.npz", "B.npz"))
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--inputs", help="reuse the crops stored in this .npz (another arm's file)")
    ap.add_argument("--min-agreement", type=float, help="--compare exits 1 below this pixel agreement")
    ap.add_argument("--out", help="also write the --compare result here")
    a = ap.parse_args(argv)
    if bool(a.save) == bool(a.compare):
        ap.error("give exactly one of --save or --compare")
    if a.save:
        save(a.save, a.k, a.inputs)
        return
    res = compare_files(*a.compare)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    if a.min_agreement is not None and res["pixel_agreement"] < a.min_agreement:
        sys.exit(1)


if __name__ == "__main__":
    main()
