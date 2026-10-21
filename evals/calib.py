"""Calibration slice for the agent's classical-CV thresholds (saliency floors, straight-line and pore rules).

    python -m evals.calib [--out runs/calib] [--codebrim data/codebrim] [--set segment.PORE_ASPECT=0 ...]

Builds (or reuses) a seeded synthetic slice in <out>/slice: 20 defect images with one distractor each and 10
distractor-only ("clean") images, 2048x1536, evals.synth seed 11. Runs the agent once per image at the 30 % cap on
the fixed cost model (results/local-macos-x86 job_ops) on the slice, the 8 public samples and, with --codebrim, the
first 10 CODEBRIM eval200 images (read in place, never written). Per group it prints candidates per image, pixel
fraction, recall in finding boxes and in per-component boxes (the harness's matching rule), false findings (matching
no GT box of any class), false cracks and the trace's rejections by cause; <out>/calib.json keeps every image's row.
--set patches a drishyam module constant first (e.g. segment.PORE_ASPECT=0 turns that rule off), to rerun a
before/after pair."""
from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from drishyam.agent import JobConfig
from drishyam.segment import default_segmenter
from evals import codebrim, harness, synth

SEED, SIZE, CAP = 11, (2048, 1536), 0.3
CLEAN = ("formwork_line", "water_stain", "graffiti", "glare", "formwork_line")


def build_slice(out: Path) -> list[dict]:
    """The slice's annotation records; images are rendered once and reused."""
    ann = out / "annotations.json"
    if ann.exists():
        return json.loads(ann.read_text())["images"]
    out.mkdir(parents=True, exist_ok=True)
    specs = synth.stratify(20, SEED)
    for i, s in enumerate(specs):
        s["distractors"] = [synth.DISTRACTORS[i % 5]]  # every defect image carries one distractor
    specs += [{"distractors": [CLEAN[i % 5]] + (["formwork_line"] if i >= 5 else [])} for i in range(10)]
    images = []
    for i, spec in enumerate(specs):
        img, gt, extra = synth.render(SEED * 100_000 + i, *SIZE, spec)
        iid = f"c{i + 1:02d}"
        cv2.imwrite(str(out / f"{iid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        images.append({"id": iid, "file": f"{iid}.jpg", "width": SIZE[0], "height": SIZE[1], "clean": i >= 20,
                       "boxes": gt, "distractors": extra})
    ann.write_text(json.dumps({"seed": SEED, "images": images}, indent=1))
    return images


def run_one(path: Path, boxes: list[dict], rows: list[dict]) -> dict:
    img = cv2.imread(str(path))
    rec_seg = harness.RecordingSegmenter()
    costs = harness.op_costs(rows, img.shape[1], img.shape[0])
    job = harness.FixedCostJob(img, JobConfig(pixel_cap=CAP, thumbs=False), costs, segmenter=rec_seg)
    events = list(job.run())
    union = [{"cls": f.cls, "box": list(f.box)} for f in job.findings]
    comps = harness.finding_components(job, rec_seg.log)
    false = [p for p in union if not any(harness.hits(p["box"], g["box"]) for g in boxes)]
    return {"candidates": len(events[1]["candidates"]), "gt": len(boxes), "findings": len(union),
            "false": len(false), "false_crack": sum(p["cls"] == "crack" for p in false),
            "found_union": sum(harness.found(g, union) for g in boxes),
            "found_comp": sum(harness.found(g, comps) for g in boxes),
            "rejected": dict(Counter(x["cause"] for e in events if e["type"] == "step" for x in e.get("rejected", []))),
            # the areas the gate or the overview could not read (re-shoot), not those a halt left (check-by-eye)
            "not_assessed": sum(a["action"] == "re-shoot" for a in events[-1]["work_order"]["not_assessed"]),
            "pixels_frac": events[-1]["summary"]["pixels_frac"], "halt": events[-1]["cause"]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evals.calib", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="runs/calib")
    ap.add_argument("--codebrim", help="CODEBRIM root holding eval200.txt (e.g. data/codebrim)")
    ap.add_argument("--set", action="append", default=[], metavar="MODULE.NAME=VALUE")
    a = ap.parse_args(argv)
    for kv in a.set:
        name, val = kv.split("=")
        mod, attr = name.split(".")
        m = importlib.import_module(f"drishyam.{mod}")
        assert hasattr(m, attr), f"drishyam.{mod} has no {attr}"
        setattr(m, attr, float(val))
    out = Path(a.out)
    recs = [("clean" if r["clean"] else "defect", r["id"], out / "slice" / r["file"], r["boxes"])
            for r in build_slice(out / "slice")]
    samples = harness.REPO / "public" / "samples"
    recs += [("sample", r["file"][:-4], samples / r["file"], r["boxes"])
             for r in json.loads((samples / "index.json").read_text()) if r["synthetic"]]  # the real photos have no ground truth
    if a.codebrim:
        cb = Path(a.codebrim)
        recs += [("codebrim", Path(r["path"]).stem, Path(r["path"]), r["boxes"])
                 for r in codebrim.load_eval(cb, cb / "eval200.txt")[:10]]
    rows = harness.cost_rows("local-macos-x86", default_segmenter().name)
    per = {iid: {"group": g, **run_one(p, boxes, rows)} for g, iid, p, boxes in recs}
    summary = {}
    for g in dict.fromkeys(r[0] for r in recs):
        v = [x for x in per.values() if x["group"] == g]
        gt = sum(x["gt"] for x in v)
        summary[g] = {"images": len(v), "candidates": round(float(np.mean([x["candidates"] for x in v])), 2),
                      "pixels_frac": round(float(np.mean([x["pixels_frac"] for x in v])), 4),
                      "found_union": f"{sum(x['found_union'] for x in v)}/{gt}",
                      "found_comp": f"{sum(x['found_comp'] for x in v)}/{gt}",
                      "false": sum(x["false"] for x in v), "false_crack": sum(x["false_crack"] for x in v),
                      "rejected": dict(sum((Counter(x["rejected"]) for x in v), Counter())),
                      "not_assessed": sum(x["not_assessed"] for x in v)}
        print(g, json.dumps(summary[g]), flush=True)
    (out / "calib.json").write_text(json.dumps({"set": a.set, "summary": summary, "per_image": per}, indent=1))


if __name__ == "__main__":
    main()
