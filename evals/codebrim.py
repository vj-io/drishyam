"""CODEBRIM (Mundt et al., CVPR 2019) original-image loader and the seeded 200-image eval subset.

The original-image release is Pascal-VOC-style XML: one `<object>` per defect box, carrying one 0/1 flag
element per label (Background, Crack, Spallation, Efflorescence, ExposedBars, CorrosionStain). A box can
carry several flags; it becomes one ground-truth entry per flag, so a spall with exposed bars is scored as
a spalling box and an exposed_rebar box (the same convention `evals.synth` uses). Written from the dataset's
documented format and unit-tested on an inline sample; not yet run against the real 8.3 GB download.

    python -m evals.codebrim --root data/codebrim --out data/eval200.txt
"""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

CLASSES = ("crack", "spalling", "efflorescence", "exposed_rebar", "corrosion")  # same order as evals.synth
# CODEBRIM flag (lower-cased, letters only) -> our class. "Background" is deliberately absent.
FLAG_TO_CLASS = {"crack": "crack", "spallation": "spalling", "spalling": "spalling",
                 "efflorescence": "efflorescence", "exposedbars": "exposed_rebar", "exposedbar": "exposed_rebar",
                 "exposedrebar": "exposed_rebar", "corrosionstain": "corrosion", "corrosion": "corrosion"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _key(tag: str) -> str:
    return re.sub(r"[^a-z]", "", tag.lower())


def parse_xml(text: str) -> dict:
    """{"file", "width", "height", "boxes": [{"cls", "box": [x, y, w, h]}]} from one annotation XML."""
    root = ET.fromstring(text)
    size = root.find("size")
    boxes = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x0, y0, x1, y1 = (round(float(bb.findtext(k, "0"))) for k in ("xmin", "ymin", "xmax", "ymax"))
        box = [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]
        flagged = {FLAG_TO_CLASS[_key(c.tag)] for c in obj
                   if _key(c.tag) in FLAG_TO_CLASS and (c.text or "").strip() in ("1", "1.0", "true", "True")}
        if not flagged and _key(obj.findtext("name", "")) in FLAG_TO_CLASS:  # plain VOC style: <name>Crack</name>
            flagged = {FLAG_TO_CLASS[_key(obj.findtext("name", ""))]}
        boxes += [{"cls": c, "box": list(box)} for c in CLASSES if c in flagged]
    return {"file": (root.findtext("filename") or "").strip(),
            "width": int(float(size.findtext("width", "0"))) if size is not None else 0,
            "height": int(float(size.findtext("height", "0"))) if size is not None else 0,
            "boxes": boxes}


def load(root) -> list[dict]:
    """Every annotated original image under `root`, as harness records (id, path, width, height, boxes)."""
    root = Path(root)
    images = {p.name: p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
    records = []
    for xml in sorted(root.rglob("*.xml")):
        try:
            a = parse_xml(xml.read_text())
        except ET.ParseError:
            continue
        path = images.get(a["file"])
        if path is None:
            continue
        records.append({"id": path.stem, "path": str(path), "width": a["width"], "height": a["height"],
                        "boxes": a["boxes"], "distractors": []})
    return sorted(records, key=lambda r: r["id"])


def select_eval(records: list[dict], n: int = 200, seed: int = 7, min_per_class: int = 40):
    """n records chosen by a fixed seed so every class has >= min_per_class boxes where the data allows.

    Greedy: each pick maximises the still-needed boxes it adds, weighted toward the rarer classes; once
    every class is covered the rest is filled in seeded-random order. Returns (subset, shortfall) where
    shortfall maps any class that could not reach the floor to the count it reached.
    """
    order = np.random.default_rng(seed).permutation(len(records))
    cnt = np.array([[sum(b["cls"] == c for b in records[i]["boxes"]) for c in CLASSES] for i in range(len(records))])
    need, supply = np.full(len(CLASSES), min_per_class), np.maximum(cnt.sum(0), 1)
    chosen, left = [], list(order)
    while len(chosen) < n and left and (need > 0).any():
        gain = (np.minimum(cnt[left], need) / supply).sum(1)
        best = int(np.argmax(gain))  # first maximum in seeded order
        if gain[best] <= 0:
            break  # a class the data cannot supply
        i = left.pop(best)
        chosen.append(i)
        need = np.maximum(need - cnt[i], 0)
    chosen += left[: n - len(chosen)]
    got = cnt[chosen].sum(0)
    shortfall = {c: int(got[k]) for k, c in enumerate(CLASSES) if got[k] < min_per_class}
    return sorted((records[i] for i in chosen), key=lambda r: r["id"]), shortfall


def write_list(subset: list[dict], out) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("".join(Path(r["path"]).name + "\n" for r in subset))


def load_eval(root, list_file) -> list[dict]:
    """The records named in `data/eval200.txt` (image file names), in list order."""
    names = [ln.strip() for ln in Path(list_file).read_text().splitlines() if ln.strip()]
    by_name = {Path(r["path"]).name: r for r in load(root)}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images in {list_file} not found under {root}, e.g. {missing[0]}")
    return [by_name[n] for n in names]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evals.codebrim", description="Select the CODEBRIM eval subset.")
    ap.add_argument("--root", default="data/codebrim")
    ap.add_argument("--out", default="data/eval200.txt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-per-class", type=int, default=40)
    a = ap.parse_args(argv)
    records = load(a.root)
    if not records:
        raise SystemExit(f"no annotated images found under {a.root}; run `make data` first")
    subset, shortfall = select_eval(records, a.n, a.seed, a.min_per_class)
    write_list(subset, a.out)
    counts = {c: sum(b["cls"] == c for r in subset for b in r["boxes"]) for c in CLASSES}
    print(f"{len(records)} annotated images; wrote {len(subset)} to {a.out}; boxes per class: {counts}")
    if shortfall:
        print(f"WARNING: below the {a.min_per_class}-box floor: {shortfall}")


if __name__ == "__main__":
    main()
