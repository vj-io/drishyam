"""CODEBRIM (Mundt et al., CVPR 2019) original-image loader and the seeded 200-image eval subset.

The original-image release is Pascal-VOC-style XML: one `<object>` per defect box, whose `<Defect>` element
holds one 0/1 flag per label (Background, Crack, Spallation, Efflorescence, ExposedBars, CorrosionStain). A box
can carry several flags; it becomes one ground-truth entry per flag, so a spall with exposed bars is scored as
a spalling box and an exposed_rebar box (the same convention `evals.synth` uses). The zip also carries macOS
`__MACOSX/._*` resource forks named like the real files; they are skipped.

    python -m evals.codebrim --unpack data/CODEBRIM_original_images.zip --root data/codebrim --out data/codebrim/eval200.txt
"""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
import zipfile
import zlib
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
        flagged = {FLAG_TO_CLASS[_key(c.tag)] for c in obj.iter()  # flags sit in <Defect>, not on <object>
                   if _key(c.tag) in FLAG_TO_CLASS and (c.text or "").strip() in ("1", "1.0", "true", "True")}
        if not flagged and _key(obj.findtext("name", "")) in FLAG_TO_CLASS:  # plain VOC style: <name>Crack</name>
            flagged = {FLAG_TO_CLASS[_key(obj.findtext("name", ""))]}
        boxes += [{"cls": c, "box": list(box)} for c in CLASSES if c in flagged]
    return {"file": (root.findtext("filename") or "").strip(),
            "width": int(float(size.findtext("width", "0"))) if size is not None else 0,
            "height": int(float(size.findtext("height", "0"))) if size is not None else 0,
            "boxes": boxes}


def _members(z: zipfile.ZipFile, size: int):
    """(basename, bytes) for every real image and annotation in the release zip.

    The zip was written without zip64 records, so offsets past 4 GiB wrap at 2**32 and zipfile places each
    entry 2**32 off (one way or the other). Each entry is tried at every in-file candidate; zipfile checks the
    local header's file name and, on read, the CRC, so a wrong candidate cannot pass.
    """
    for info in z.infolist():
        name = Path(info.filename).name
        if info.is_dir() or "__MACOSX" in info.filename or name.startswith("._") \
                or Path(name).suffix.lower() not in IMAGE_SUFFIXES | {".xml"}:
            continue
        reported = info.header_offset
        for off in (reported, reported - 2**32, reported + 2**32):
            if not 0 <= off < size:
                continue
            info.header_offset, info._end_offset = off, None  # _end_offset: zipfile's overlap guard trusts the wrapped offsets
            try:
                yield name, z.read(info)
                break
            except (zipfile.BadZipFile, zlib.error):
                continue
        else:
            raise zipfile.BadZipFile(f"{info.filename}: no valid local header at any offset")


def unpack(zip_path, root) -> int:
    """Extract images to root/images and annotations to root/annotations (basenames only). Returns the count."""
    root, n = Path(root), 0
    with zipfile.ZipFile(zip_path) as z:
        for name, data in _members(z, Path(zip_path).stat().st_size):
            out = root / ("annotations" if name.endswith(".xml") else "images") / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            n += 1
    return n


def load(root) -> list[dict]:
    """Every annotated original image under `root`, as harness records (id, path, width, height, boxes)."""
    root = Path(root)
    def real(p):  # not a macOS resource fork (__MACOSX/._name)
        return not p.name.startswith("._") and "__MACOSX" not in p.parts

    images = {p.name: p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and real(p)}
    records = []
    for xml in sorted(p for p in root.rglob("*.xml") if real(p)):
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


def select_dev(records: list[dict], exclude, n: int = 100, seed: int = 4244) -> list[dict]:
    """n records drawn with a fixed seed from those whose image is not in `exclude` (the held-out eval list's file
    names): the dev split that tuning may read (evals/codebrim_dev100.txt)."""
    pool = [r for r in records if Path(r["path"]).name not in set(exclude)]
    pick = np.random.default_rng(seed).choice(len(pool), size=min(n, len(pool)), replace=False)
    return sorted((pool[i] for i in pick), key=lambda r: r["id"])


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
    ap.add_argument("--unpack", metavar="ZIP", help="first extract the release zip into --root")
    ap.add_argument("--root", default="data/codebrim")
    ap.add_argument("--out", default="data/eval200.txt")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-per-class", type=int, default=40)
    ap.add_argument("--dev", type=int, metavar="N", help="instead: N images (seed --seed) not in the committed eval list")
    a = ap.parse_args(argv)
    if a.unpack:
        print(f"unpacked {unpack(a.unpack, a.root)} images and annotations from {a.unpack} to {a.root}")
    records = load(a.root)
    if not records:
        raise SystemExit(f"no annotated images found under {a.root}; run `make data` first")
    if a.dev:
        held_out = Path(__file__).with_name("codebrim_eval200.txt").read_text().split()
        write_list(select_dev(records, held_out, a.dev, a.seed), a.out)
        print(f"wrote {a.dev} dev images (seed {a.seed}, none in codebrim_eval200.txt) to {a.out}")
        return
    subset, shortfall = select_eval(records, a.n, a.seed, a.min_per_class)
    write_list(subset, a.out)
    counts = {c: sum(b["cls"] == c for r in subset for b in r["boxes"]) for c in CLASSES}
    print(f"{len(records)} annotated images; wrote {len(subset)} to {a.out}; boxes per class: {counts}")
    if shortfall:
        print(f"WARNING: below the {a.min_per_class}-box floor: {shortfall}")
    committed = Path(__file__).with_name("codebrim_eval200.txt")  # the published run's list (PRD: committed)
    if (a.n, a.seed, a.min_per_class) == (200, 7, 40) and committed.read_text() != Path(a.out).read_text():
        print(f"WARNING: {a.out} differs from the committed {committed.name}; results will not match the published run")


if __name__ == "__main__":
    main()
