"""Near-duplicate check between dacl10k (what the segmenter trains on) and the CODEBRIM eval list (what it is tested on).

    python -m evals.leakage --dacl10k <dacl10k folder> --codebrim data/codebrim [--max-dist 10] [--out runs/model-eval/leakage.json]

Difference hash: the grey image resized to 9x8 (cv2.INTER_AREA), one bit per horizontally adjacent pair (left brighter
than right), 64 bits. A CODEBRIM image counts as a likely duplicate of a dacl10k image when the Hamming distance of the
hashes is <= --max-dist (of 64). Each CODEBRIM image is hashed in its four 90-degree rotations, because two datasets can
store one photo under different EXIF orientations. Only counts and file names are written, never an image. A missing or
empty dacl10k folder is skipped with exit 0: fetch the set with kagglehub as in training/README.md and pass its folder.
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


def dhash(gray: np.ndarray) -> np.uint64:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    return np.packbits((small[:, :-1] > small[:, 1:]).ravel()).view(">u8")[0]


def hashes(path, rotations: bool = False) -> list[np.uint64]:
    gray = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)  # 1/4-scale decode: plenty for a 9x8 hash
    if gray is None:
        return []
    return [dhash(np.rot90(gray, k)) for k in range(4 if rotations else 1)]


def near_duplicates(eval_paths: list, train_paths: list, max_dist: int = 10) -> dict:
    """For each eval image, the closest train image by dHash Hamming distance (over the eval image's rotations)."""
    train = [(p, h) for p in train_paths for h in hashes(p)]
    if not train:
        return {"skipped": "no readable dacl10k images"}
    th = np.array([h for _, h in train], np.uint64)
    pairs = []
    for p in eval_paths:
        hs = np.array(hashes(p, rotations=True), np.uint64)
        d = np.bitwise_count(hs[:, None] ^ th[None, :]).min(0)
        j = int(d.argmin())
        pairs.append((Path(p).name, Path(train[j][0]).name, int(d[j])))
    dist = np.array([d for *_, d in pairs])
    return {"eval_images": len(pairs), "train_images": len(train), "max_dist": max_dist,
            "near_duplicates": int((dist <= max_dist).sum()),
            "at_or_below": {str(t): int((dist <= t).sum()) for t in (0, 5, 10, 15)},
            "pairs": sorted(([e, t, d] for e, t, d in pairs if d <= max_dist), key=lambda r: r[2])}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(prog="python -m evals.leakage", description=__doc__.split("\n\n")[0])
    ap.add_argument("--dacl10k", required=True, help="dacl10k folder (searched recursively for images)")
    ap.add_argument("--codebrim", default="data/codebrim", help="CODEBRIM root (annotations and images)")
    ap.add_argument("--list", default=str(LIST), help="the eval image names")
    ap.add_argument("--max-dist", type=int, default=10)
    ap.add_argument("--out", default="runs/model-eval/leakage.json")
    a = ap.parse_args(argv)
    root = Path(a.dacl10k).expanduser()
    train = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUFFIXES) if root.is_dir() else []
    if not train:
        report = {"skipped": f"no dacl10k images under {root}"}
    else:
        report = near_duplicates([r["path"] for r in codebrim.load_eval(a.codebrim, a.list)], train, a.max_dist)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "pairs"}), f"(pairs in {a.out})" if "pairs" in report else "")
    return report


if __name__ == "__main__":
    main()
