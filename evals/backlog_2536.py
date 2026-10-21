"""A 2,536-photo backlog (the size of one real drone bridge inspection, docs/report.md section 1) from dev-safe photos
only, for an end-to-end `drishyam batch --router nova` run, and its summary against the labels.

    # 2,436 fresh synthetic photos at 4608x3456 (CODEBRIM's commonest size), seeds no evaluation uses:
    python -m evals.backlog_2536 damage     # 1,220 damage mixes, seed 2536 (evals.synth.stratify), 8 processes
    for s in 2537 2539 2541 2543; do python -m evals.synth dataset --clean --out data/backlog_2536/clean_$s \\
        --n 304 --seed $s --width 4608 --height 3456; done  # no damage (its bridge_08 demo copy is left out)
    python -m evals.backlog_2536 list       # + the 100 CODEBRIM dev photos -> data/backlog_2536/backlog.txt, truth.json
    drishyam batch data/backlog_2536/backlog.txt --router nova --cost-model fixed --micro-arm <arm> --usd-cap <total>
    python -m evals.backlog_2536 summary runs/batch-<id>   # -> results/backlog_2536/summary.json, summary.md

No held-out photo is used (CODEBRIM's come from evals/codebrim_dev100.txt), and no photo is written to results/.
`damage` is evals.synth.build_dataset's damage set made in parallel, skipping the mixes evals.synth cannot render at
this size (an efflorescence whose bounds leave the photo raises there: seeds 2536 and 2538 each died that way in
build_dataset after 136 to 199 photos), and recording them.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from drishyam import router
from evals import codebrim, harness, synth
from evals.router_eval import gt_label, wilson

ROOT = harness.REPO / "data" / "backlog_2536"
DAMAGE_SEED, DAMAGE_N, SPARE = 2536, 1220, 60
SETS = [(f"damage_{DAMAGE_SEED}", DAMAGE_SEED, DAMAGE_N, False)] + \
       [(f"clean_{s}", s, 304, True) for s in (2537, 2539, 2541, 2543)]
SHUFFLE_SEED = 2536
OUT = harness.REPO / "results" / "backlog_2536"
W, H = 4608, 3456


def _render(job):
    i, spec = job
    cv2.setNumThreads(1)
    try:
        img, gt, extra = synth.render(DAMAGE_SEED * 100_000 + i, W, H, spec)
    except Exception as exc:  # noqa: BLE001 -- see the module docstring
        return i, None, f"{type(exc).__name__}: {str(exc)[:120]}"
    iid = f"s{i + 1:04d}"
    cv2.imwrite(str(ROOT / f"damage_{DAMAGE_SEED}" / "images" / f"{iid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return i, {"id": iid, "file": f"images/{iid}.jpg", "width": W, "height": H, "boxes": gt, "distractors": extra}, None


def build_damage(workers: int = 8) -> dict:
    """The first DAMAGE_N mixes of stratify(DAMAGE_N + SPARE) that render, as build_dataset writes a set."""
    out = ROOT / f"damage_{DAMAGE_SEED}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    specs = synth.stratify(DAMAGE_N + SPARE, DAMAGE_SEED)
    with ProcessPoolExecutor(workers) as ex:
        done = sorted(ex.map(_render, enumerate(specs)), key=lambda r: r[0])
    kept = [rec for _, rec, _ in done if rec][:DAMAGE_N]
    for _, rec, _ in done:  # the spares past DAMAGE_N
        if rec and rec not in kept:
            os.remove(out / rec["file"])
    skipped = [{"index": i, "error": err} for i, _, err in done if err]
    assert len(kept) == DAMAGE_N, f"only {len(kept)} of {DAMAGE_N} rendered"
    (out / "annotations.json").write_text(json.dumps(
        {"dataset": f"backlog-damage-{DAMAGE_N}", "synthetic": True, "seed": DAMAGE_SEED,
         "generator_version": synth.GENERATOR_VERSION, "clean": False, "images": kept, "skipped": skipped}, indent=1))
    return {"kept": len(kept), "skipped": skipped}


def build_list() -> dict:
    truth = {}
    for name, seed, n, clean in SETS:
        ann = json.loads((ROOT / name / "annotations.json").read_text())
        assert (ann["seed"], ann["generator_version"], ann["clean"]) == (seed, synth.GENERATOR_VERSION, clean), name
        imgs = [r for r in ann["images"] if r["id"].startswith("s")]  # not the clean set's bridge_08 demo copy
        assert len(imgs) == n and all((r["width"], r["height"]) == (4608, 3456) for r in imgs), name
        truth.update({f"{name}/{r['file']}": {"split": "synthetic clean" if clean else "synthetic damage",
                                              "boxes": r["boxes"]} for r in imgs})
    for r in codebrim.load_eval(harness.REPO / "data" / "codebrim", harness.REPO / "evals" / "codebrim_dev100.txt"):
        truth[os.path.relpath(r["path"], ROOT)] = {"split": "codebrim dev100", "boxes": r["boxes"]}
    names = sorted(truth)
    order = [names[i] for i in np.random.default_rng(SHUFFLE_SEED).permutation(len(names))]
    (ROOT / "backlog.txt").write_text("\n".join(order) + "\n")
    (ROOT / "truth.json").write_text(json.dumps({k: {**truth[k], "gt_label": gt_label(truth[k]["boxes"])} for k in order}))
    return {"photos": len(order), "by_split": {s: sum(t["split"] == s for t in truth.values()) for s in
                                               ("synthetic damage", "synthetic clean", "codebrim dev100")}}


def summarise(bdir: Path) -> dict:
    rep = json.loads((bdir / "inspection.json").read_text())
    truth = json.loads((ROOT / "truth.json").read_text())
    key = {str((ROOT / k).resolve()): k for k in truth}
    photos = [{**p, **truth[key[str(Path(p["photo"]).resolve())]]} for p in rep["photos"]]
    labels, splits = router.LABELS, ("synthetic damage", "synthetic clean", "codebrim dev100")
    lab = lambda p: p.get("route", {}).get("label")
    crit = [p for p in photos if p["gt_label"] == "Critical"]
    cleared_crit = [p for p in crit if p.get("outcome") == "cleared"]
    queue_hits = 0  # labelled critical defects a queued critical finding hits (evals.harness.hits)
    by_photo = {}
    for q in rep["queue"]:
        by_photo.setdefault(q["photo"], []).append(q)
    for p in photos:
        qs = by_photo.get(p["photo"], [])
        queue_hits += sum(any(q["cls"] == g["cls"] and harness.hits(q["box"], g["box"]) for q in qs)
                          for g in p["boxes"] if harness.gt_critical(g))
    gt_crit = sum(harness.gt_critical(g) for p in photos for g in p["boxes"])
    ro, s = rep["router"], rep["spent"]
    out = {
        "batch": bdir.name, "created": rep["created"], "photos": len(photos),
        "by_split": {sp: sum(p["split"] == sp for p in photos) for sp in splits},
        "budget": rep["budget"], "stop": rep["stop"], "status": rep["status"], "counts": rep["counts"],
        "triaged": sum("triage" in p for p in photos), "routed": ro["photos"],
        "router_labels": ro["labels"], "router_status": ro["status"], "router_causes": ro["causes"],
        "run": sum("run_dir" in p for p in photos), "cleared": sum(p.get("outcome") == "cleared" for p in photos),
        "not_reached": sum(p.get("outcome") == "not-reached" for p in photos),
        "unusable": sum(p.get("outcome") in ("unreadable", "refused") for p in photos),
        "run_by_label": {k: sum("run_dir" in p and lab(p) == k for p in photos) for k in labels},
        "criticals_queued": rep["counts"]["critical"], "criticals_pending": rep["counts"]["pending"],
        "labelled_critical_defects": gt_crit, "labelled_critical_defects_queued": queue_hits,
        "photos_with_labelled_critical": len(crit),
        "photos_with_labelled_critical_run": sum("run_dir" in p for p in crit),
        "photos_with_labelled_critical_cleared": len(cleared_crit),
        "miss_rate": len(cleared_crit) / len(crit) if crit else None, "miss_rate_ci": wilson(len(cleared_crit), len(crit)),
        "agreement": {
            "accuracy": sum(lab(p) == p["gt_label"] for p in photos if lab(p)) / max(1, ro["photos"]),
            "confusion_gt_by_router": {g: {k: sum(p["gt_label"] == g and lab(p) == k for p in photos) for k in labels}
                                       for g in labels},
            "by_split": {sp: {g: {k: sum(p["split"] == sp and p["gt_label"] == g and lab(p) == k for p in photos)
                                  for k in labels} for g in labels} for sp in splits}},
        "usd": {"router_bedrock": ro["usd"], "compute_modelled": s["usd"] - s["router_usd"], "total_modelled": s["usd"]},
        "router_tokens": {"input": ro["input_tokens"], "output": ro["output_tokens"], "calls": ro["calls"],
                          "cached": ro["cached"]},
        "router_latency_ms": ro["latency_ms"],
        "seconds": {"charged": s["seconds"], "triage": s["triage_seconds"], "router_charged": s["router_seconds"],
                    "wall": s["wall_seconds"]},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evals.backlog_2536")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("damage")
    sub.add_parser("list")
    sub.add_parser("summary").add_argument("batch_dir")
    a = ap.parse_args(argv)
    run = {"damage": build_damage, "list": build_list, "summary": lambda: summarise(Path(a.batch_dir))}[a.cmd]
    print(json.dumps(run(), indent=1))


if __name__ == "__main__":
    main()
