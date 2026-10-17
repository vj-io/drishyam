"""Near-duplicate check between dacl10k (what the segmenter trains on) and the CODEBRIM eval list (what it is tested on).

    python -m evals.leakage --dacl10k <dacl10k folder> --codebrim data/codebrim [--out runs/model-eval/leakage.json]
    python -m evals.leakage --codebrim data/codebrim     # the calibration alone

Two stages, both on the grey image decoded at 1/4 scale with its lighting removed (resized to 256x256, minus a
sigma-16 GaussianBlur: low-texture concrete otherwise hashes to its illumination gradient, and a 9x8 dHash matched
unrelated walls at distance 1). 1. Candidate: a 255-bit pHash (the 16x16 low-frequency DCT block of that image at
32x32, minus DC, against its median); each CODEBRIM image is hashed in its four 90-degree rotations, because two
datasets can store one photo under different EXIF orientations, and its nearest dacl10k hash is a candidate at
Hamming distance <= --max-dist. 2. Confirmed: the normalised cross-correlation of the two lighting-free 128x128
images, in the rotation that matched, is >= --min-ncc. The calibration (in the output) measures both on copies
(resized, rotated, JPEG-recompressed, re-exposed and cropped versions of the eval images) and on the nearest other
CODEBRIM photo of each eval image, which is not a copy. It finds copies of a photo; crops that drop 5 % or more of
the frame, and other shots of the same structure, are not found (the calibration's crop95 row). Only counts, file
names and distances are written, never an image. A missing or empty dacl10k folder is skipped with exit 0: fetch
the set with kagglehub as in training/README.md and pass its folder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from evals import codebrim

LIST = Path(__file__).with_name("codebrim_eval200.txt")
SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_DIST, MIN_NCC = 32, 0.8  # of 255 bits; on CODEBRIM (calibrate): copies <= 10 and >= 0.93, other photos >= 56 and <= 0.43
COPIES = {  # what a re-shared copy of a photo may have been through
    "resize0.5+rot90": lambda g: cv2.rotate(cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                                            cv2.ROTATE_90_CLOCKWISE),
    "resize0.25": lambda g: cv2.resize(g, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA),
    "jpeg50": lambda g: cv2.imdecode(cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 50])[1], cv2.IMREAD_GRAYSCALE),
    "brightness": lambda g: cv2.convertScaleAbs(g, alpha=0.8, beta=25),
    "crop95": lambda g: g[g.shape[0] // 40: g.shape[0] - g.shape[0] // 40, g.shape[1] // 40: g.shape[1] - g.shape[1] // 40],
}


def flat(gray: np.ndarray, n: int = 256) -> np.ndarray:
    """The image at n x n with its lighting removed: minus a GaussianBlur of sigma n/16."""
    g = cv2.resize(gray, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), n / 16)


def phash(gray: np.ndarray) -> np.ndarray:
    d = cv2.dct(cv2.resize(flat(gray), (32, 32), interpolation=cv2.INTER_AREA))[:16, :16].ravel()[1:]
    return np.packbits(d > np.median(d))


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.matchTemplate(flat(a, 128), flat(b, 128), cv2.TM_CCOEFF_NORMED)[0, 0])


def gray(path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)  # 1/4-scale decode: plenty for a 32x32 hash


def nearest(g: np.ndarray, th: np.ndarray) -> tuple[int, int, int]:
    """(rotation, index, Hamming distance) of the closest of the hashes `th` to any 90-degree rotation of `g`."""
    d = np.unpackbits(np.stack([phash(np.rot90(g, k)) for k in range(4)])[:, None] ^ th[None], axis=-1).sum(-1)
    k, j = np.unravel_index(int(d.argmin()), d.shape)
    return int(k), int(j), int(d[k, j])


def near_duplicates(eval_paths: list, train_paths: list, max_dist: int = MAX_DIST, min_ncc: float = MIN_NCC) -> dict:
    """For each eval image, its closest train image by pHash; a candidate at <= max_dist, confirmed at NCC >= min_ncc."""
    train = [(p, phash(g)) for p in train_paths if (g := gray(p)) is not None]
    if not train:
        return {"skipped": "no readable dacl10k images"}
    th = np.stack([h for _, h in train])
    pairs, dists = [], []
    for p in eval_paths:
        g = gray(p)
        k, j, d = nearest(g, th)
        dists.append(d)
        if d <= max_dist:
            pairs.append([Path(p).name, Path(train[j][0]).name, d, round(ncc(np.rot90(g, k), gray(train[j][0])), 3)])
    dist = np.array(dists)
    return {"eval_images": len(dists), "train_images": len(train), "max_dist": max_dist, "min_ncc": min_ncc,
            "candidates": len(pairs), "near_duplicates": sum(c >= min_ncc for *_, c in pairs),
            "hash_at_or_below": {str(t): int((dist <= t).sum()) for t in (8, 16, 32, 48, 64)},
            "pairs": sorted(pairs, key=lambda r: (-r[3], r[2]))}


def calibrate(eval_paths: list, other_paths: list, n_copies: int = 50) -> dict:
    """Both stages on pairs that are copies (COPIES of the first n eval images) and on pairs that are not (each eval
    image against its nearest other photo of the same dataset)."""
    copies = {}
    for name, make in COPIES.items():
        d, c = [], []
        for p in eval_paths[:n_copies]:
            g = gray(p)
            x = make(g)
            k, _, dk = nearest(g, phash(x)[None])
            d.append(dk)
            c.append(ncc(np.rot90(g, k), x))
        copies[name] = {"max_dist": max(d), "min_ncc": round(min(c), 3), "found": sum(a <= MAX_DIST and b >= MIN_NCC
                                                                                   for a, b in zip(d, c, strict=True))}
    pool = [(p, phash(g)) for p in other_paths if (g := gray(p)) is not None]
    th = np.stack([h for _, h in pool])
    near = [nearest(g, th) for g in map(gray, eval_paths)]
    others = [ncc(np.rot90(gray(p), k), gray(pool[j][0])) for p, (k, j, _) in zip(eval_paths, near, strict=True)]
    dist = [d for *_, d in near]
    found = [n for n in COPIES if n != "crop95"]
    return {"copies": {"n": min(n_copies, len(eval_paths)), **copies},
            "other_photos": {"n": len(near), "pool": len(th), "min_dist": min(dist), "max_ncc": round(max(others), 3),
                             "at_or_below_max_dist": sum(d <= MAX_DIST for d in dist),
                             "confirmed": sum(d <= MAX_DIST and c >= MIN_NCC for d, c in zip(dist, others, strict=True))},
            "separated": all(copies[n]["max_dist"] <= MAX_DIST and copies[n]["min_ncc"] >= MIN_NCC for n in found)
                         and min(dist) > MAX_DIST and max(others) < MIN_NCC,
            "not_found": "crops dropping 5 % or more of the frame (crop95), and other shots of the same structure"}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(prog="python -m evals.leakage", description=__doc__.split("\n\n")[0])
    ap.add_argument("--dacl10k", help="dacl10k folder (searched recursively for images); without it, the calibration alone")
    ap.add_argument("--codebrim", default="data/codebrim", help="CODEBRIM root (annotations and images)")
    ap.add_argument("--list", default=str(LIST), help="the eval image names")
    ap.add_argument("--max-dist", type=int, default=MAX_DIST)
    ap.add_argument("--min-ncc", type=float, default=MIN_NCC)
    ap.add_argument("--out", default="runs/model-eval/leakage.json")
    a = ap.parse_args(argv)
    root = Path(a.dacl10k).expanduser() if a.dacl10k else None
    train = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUFFIXES) if root and root.is_dir() else []
    if root and not train:
        report = {"skipped": f"no dacl10k images under {root}"}
    else:
        ev = [r["path"] for r in codebrim.load_eval(a.codebrim, a.list)]
        others = sorted({r["path"] for r in codebrim.load(a.codebrim)} - set(ev))
        report = {**(near_duplicates(ev, train, a.max_dist, a.min_ncc) if train else {}),
                  "calibration": calibrate(ev, others) if others else "no other CODEBRIM photos to calibrate on"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "pairs"}), f"(pairs in {a.out})" if "pairs" in report else "")
    return report


if __name__ == "__main__":
    main()
