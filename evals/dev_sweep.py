"""Dev-only sweep of JobConfig values: Drishyam alone on the tuning splits, fixed linux-aarch64 costs, no baselines.

    python -m evals.dev_sweep synth_dev 0.3 "v0_0=v0:0;v0_2=v0:2" [--workers 6] [--codebrim data/codebrim]

Splits: synth_dev (60, seed 4242), clean (data/synth_clean_dev, 30 with no damage, seed 4243) and real (CODEBRIM
evals/codebrim_dev100.txt). It never reads a held-out set. Prints one JSON row per variant (recall with the
components boxes, pixels, false boxes, halt causes, looks). results/dev/v0_sweep.json collects the rows that chose
JobConfig.v0.
"""

import argparse
import collections
import functools
import json

import cv2

from drishyam.segment import default_segmenter
from evals import harness

SPLITS = {"synth_dev": ("data/synth_dev", 60, 4242), "clean": ("data/synth_clean_dev", 30, 4243)}


def parse(spec: str) -> dict:
    """'name=k:v,k:v;name2=...' -> {name: {k: v}}; v is a float or True/False."""
    out = {}
    for item in spec.split(";"):
        name, _, kv = item.partition("=")
        out[name] = {k: (v == "True" if v in ("True", "False") else float(v))
                     for k, v in (x.split(":") for x in kv.split(",") if x)}
    return out


def task(rec, variants, rows, cap):
    img = cv2.imread(rec["path"], cv2.IMREAD_COLOR)
    costs = harness.op_costs(rows, img.shape[1], img.shape[0])
    return {"id": rec["id"], "configs": {k: harness.run_drishyam(img, cap, costs, **sw) for k, sw in variants.items()}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evals.dev_sweep")
    ap.add_argument("split", choices=(*SPLITS, "real"))
    ap.add_argument("cap", type=float)
    ap.add_argument("variants")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--codebrim", default="data/codebrim")
    a = ap.parse_args(argv)
    variants = parse(a.variants)
    if a.split == "real":
        records, _ = harness.load_records(a.codebrim, 100, list_file=harness.REPO / "evals" / "codebrim_dev100.txt")
    else:
        d, n, seed = SPLITS[a.split]
        records, _ = harness.load_records(harness.REPO / d, n, seed=seed)
    rows = harness.cost_rows("linux-aarch64", default_segmenter().name)
    results = harness.pool_map(functools.partial(task, variants=variants, rows=rows, cap=a.cap), records, a.workers,
                               lambda m: print(m, flush=True), a.split)
    for k in variants:
        m = harness.score_config(k, records, results, "preds")
        cs = [r["configs"][k] for r in results]
        row = {"recall": m["recall"], "pixels_frac": m["pixels_frac"], "fp_per_image": m["fp_per_image"],
               "boxes_per_image": m["predictions_per_image"],
               "halt": dict(collections.Counter(c["halt_cause"] for c in cs)),
               "looks_mean": sum(sum(t["action"] != "halt" for t in c["trace"]) for c in cs) / len(cs),
               "images_with_any": sum(bool(c["preds"]) for c in cs) / len(cs)}
        print(k, json.dumps({x: (round(y, 4) if isinstance(y, float) else y) for x, y in row.items()}), flush=True)


if __name__ == "__main__":
    main()
