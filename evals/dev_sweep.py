"""Dev-only sweep of JobConfig values: Drishyam on the tuning splits, fixed costs, beside one downscale.

    python -m evals.dev_sweep synth_dev 0.3 "v0_0=v0:0;v0_2=v0:2" [--workers 6] [--codebrim data/codebrim] [--out f]
    python -m evals.dev_sweep real 0.3,1.0 "old=;p=first_level:2" --micro-arm local-macos-x86-dacl10k --baselines

Splits: synth_dev (60, seed 4242), clean (data/synth_clean_dev, 30 with no damage, seed 4243) and real (CODEBRIM
evals/codebrim_dev100.txt). It never reads a held-out set. `cap`: one pixel cap or several, comma-separated. Every
run also scores one 1024-px downscale with the same segmenter (one look per image), so each variant gets a paired
bootstrap recall ratio against it; --baselines adds L2/L1/L0 tiling (with an ONNX, the costly part). Costs: the
job_ops table of results/<--micro-arm>/micro.json (default linux-aarch64, which timed the classical engine; an ONNX
needs its own table). Prints one JSON row per variant and cap (recall, IoU-only recall, chance, precision, boxes,
pixels, halt causes, looks per level, the ratio to the downscale with its CI); `--out` also writes them, with the
split, caps, variants and each row's per-image counts (for paired comparisons across runs), as JSON. results/dev/v0_sweep.json holds the rows that chose JobConfig.v0.
"""

import argparse
import collections
import functools
import json
from pathlib import Path

import cv2

from drishyam.segment import default_segmenter
from evals import harness

SPLITS = {"synth_dev": ("data/synth_dev", 60, 4242), "clean": ("data/synth_clean_dev", 30, 4243)}


def parse(spec: str) -> dict:
    """'name=k:v,k:v;name2=...' -> {name: {k: v}}; v is an int, a float or True/False."""
    out = {}
    for item in spec.split(";"):
        name, _, kv = item.partition("=")
        out[name] = {k: (v == "True" if v in ("True", "False") else int(v) if v.lstrip("-").isdigit() else float(v))
                     for k, v in (x.split(":") for x in kv.split(",") if x)}
    return out


def task(rec, variants, rows, caps, baselines=False):
    img = cv2.imread(rec["path"], cv2.IMREAD_COLOR)
    costs, seg = harness.op_costs(rows, img.shape[1], img.shape[0]), default_segmenter()
    cfg = {"downscale": harness.run_downscale(img, seg, costs)}
    for lv in harness.TILE_LEVELS if baselines else ():
        cfg[f"tiling@L{lv}"] = harness.run_tiling(img, seg, lv, costs)
    for k, sw in variants.items():
        for cap in caps:
            cfg[f"{k}@{cap}"] = harness.run_drishyam(img, cap, costs, **sw)
    return {"id": rec["id"], "width": img.shape[1], "height": img.shape[0], "configs": cfg}


def row(key: str, records, results, base) -> dict:
    """One config's dev numbers; `base`: the downscale's scored metrics, for the paired ratio."""
    m, n = harness.score_config(key, records, results, "preds"), len(records)
    cs = [r["configs"][key] for r in results]
    out = {"recall": m["recall"], "recall_ci": harness.boot_ci(harness.recall_of(m), n),
           "recall_iou_only": m["recall_iou_only"], "chance_recall": m["chance_recall"],
           "chance_recall_iou_only": m["chance_recall_iou_only"], "precision": m["precision"],
           "fp_per_image": m["fp_per_image"], "boxes_per_image": m["predictions_per_image"],
           "pixels_frac": m["pixels_frac"], "net_pixels_frac": m["net_pixels_frac"],
           "images_with_any": sum(bool(c["preds"]) for c in cs) / len(cs)}
    if key != "downscale" and base["recall"]:
        out |= {"vs_downscale": m["recall"] / base["recall"], "vs_downscale_ci": harness.ratio_ci(m, base, n),
                "vs_downscale_iou_only_ci": harness.ratio_ci(m, base, n, col=1)}
    if "trace" in cs[0]:
        looks = [t for c in cs for t in c["trace"] if t["action"] != "halt"]
        out |= {"halt": dict(collections.Counter(c["halt_cause"] for c in cs)), "looks_mean": len(looks) / len(cs),
                "looks_by_level": dict(sorted(collections.Counter(f"L{t['level']}" for t in looks).items())),
                "looks_by_action": dict(collections.Counter(t["action"] for t in looks))}
    out["per_image"] = m["per_image"]  # [found, found IoU-only, chance, chance IoU-only, GT] per image: paired comparisons
    return {x: (round(y, 4) if isinstance(y, float) else y) for x, y in out.items()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evals.dev_sweep")
    ap.add_argument("split", choices=(*SPLITS, "real"))
    ap.add_argument("cap", help="pixel cap(s), comma-separated")
    ap.add_argument("variants")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--codebrim", default="data/codebrim")
    ap.add_argument("--micro-arm", default="linux-aarch64", help="results/<arm>/micro.json job_ops: the fixed costs")
    ap.add_argument("--baselines", action="store_true", help="also tiling at L2, L1 and L0 (the downscale always runs)")
    ap.add_argument("--out", help="also write the rows here, as JSON")
    a = ap.parse_args(argv)
    variants, caps = parse(a.variants), [float(c) for c in a.cap.split(",")]
    if a.split == "real":
        records, _ = harness.load_records(a.codebrim, 100, list_file=harness.REPO / "evals" / "codebrim_dev100.txt")
    else:
        d, n, seed = SPLITS[a.split]
        records, _ = harness.load_records(harness.REPO / d, n, seed=seed)
    seg = default_segmenter()
    rows = harness.cost_rows(a.micro_arm, seg.name)
    results = harness.pool_map(functools.partial(task, variants=variants, rows=rows, caps=caps, baselines=a.baselines),
                               records, a.workers, lambda m: print(m, flush=True), a.split)
    base = harness.score_config("downscale", records, results, "preds")
    out = {}
    for k in results[0]["configs"]:
        out[k] = row(k, records, results, base)
        print(k, json.dumps({x: y for x, y in out[k].items() if x != "per_image"}), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"split": a.split, "n": len(records), "caps": caps, "segmenter":
                                           harness.segmenter_id(seg), "micro_arm": a.micro_arm, "variants": variants,
                                           "rows": out}, indent=1) + "\n")


if __name__ == "__main__":
    main()
