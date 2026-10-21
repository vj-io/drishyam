"""Dev only: does `drishyam batch --router nova` queue the labelled critical defects for a person within a budget sooner
than the rule triage or a random order, and how many photos holding a critical defect does it clear unseen?

    DRISHYAM_ONNX=models/drishyam_dacl10k.onnx DRISHYAM_ONNX_FP16=models/drishyam_dacl10k.fp16.onnx AWS_PROFILE=drishyam \\
        python -m evals.router_eval [--workers 4] [--out results/dev/router_eval.json]

The backlog: synth_dev (60 photos with damage, seed 4242), synth_clean_dev (its first 30: no damage, seed 4243) and
CODEBRIM dev100 (evals/codebrim_dev100.txt), 190 photos in a seeded shuffle (seed 0), "file order". It never reads a
held-out set, and writes no CODEBRIM image: only ids, labels and numbers. Each photo's job runs once with v1 as
`drishyam batch` runs it (evals.batch_triage.photo_task: decoded, people blurred, default per-photo caps, governor on,
the fixed costs of results/local-macos-x86-dacl10k-retime/micro.json on a virtual clock), and any order and budget is
replayed from those records (evals.batch_triage.replay). Each photo is routed once through drishyam.router, the code
`drishyam batch --router nova` runs (cached by sha256: a rerun is free and charges each photo its first call's $). A real
`drishyam batch --router nova` run at the primary budget checks the replay.
"""

import argparse
import json
import math
import os
from functools import partial
from pathlib import Path

import numpy as np

from drishyam import batch, router
from drishyam.agent import JobConfig
from drishyam.segment import default_segmenter
from evals import batch_triage, harness, synth

RULE = (
    "Stated before any paid router call (committed with this file). Backlog: synth_dev (60), synth_clean_dev (first 30) "
    "and CODEBRIM dev100, 190 photos in a seeded shuffle; per-photo engine v1 (onnx:drishyam_dacl10k.onnx, its FP16 net "
    "as the fast model) on the fixed costs of results/local-macos-x86-dacl10k-retime/micro.json, default per-photo "
    "caps, governor on. Router: Nova Lite (amazon.nova-lite-v1:0, us-east-1, temperature 0) with the fixed prompt of "
    "drishyam/router.py, one call per photo, cached. Primary metric: critical recall within budget, the share of the "
    "backlog's labelled critical defects that a critical finding queues for a person before the batch budget runs out, "
    "at a total budget of half the seconds the 190 jobs take run in full. The rule-triage arm and the router arm are "
    "charged their triage seconds, and the router arm also its Bedrock $ as seconds at the agent's $0.339 per hour. "
    "Gain: the paired 95 % bootstrap CI (1000 resamples of the photos within each of the three splits) of router / rule "
    "triage lies above 1, and so does that of router / random order (mean of 1000 seeded permutations; 100 per "
    "resample). Harm: either CI lies below 1. Otherwise: no measurable difference. Safety, reported with the verdict: "
    "the router's miss rate, the share of photos holding a labelled critical defect that it clears, with a Wilson 95 % "
    "CI; clearing is called safe only if that CI's upper bound is at most 0.05. Reported, not part of the verdict: the "
    "other budgets (a quarter, three quarters), the router arm with its $ not charged, IoU-only recall, time to the "
    "first queued critical defect, label agreement with the labels (Critical: a labelled critical defect; Requires "
    "zoom: any other labelled defect; Clear: no labelled defect), cost, latency and fallbacks by cause."
)
SPLITS = (("synth_dev", "data/synth_dev", 60, 4242, None), ("clean", "data/synth_clean_dev", 30, 4243, None),
          ("codebrim_dev", "data/codebrim", 100, None, "evals/codebrim_dev100.txt"))
MICRO_ARM, ENGINE = "local-macos-x86-dacl10k-retime", "onnx:drishyam_dacl10k.onnx"
PRIMARY, BUDGETS = 0.5, (0.25, 0.5, 0.75)
RANDOM_N, RANDOM_N_BOOT = 1000, 100
SAFE_MISS = 0.05
USD_PER_HOUR = JobConfig().usd_per_hour


def make(img, cfg, **kw):
    return batch.make_job(img, cfg, cost_model="fixed", micro_arm=MICRO_ARM, **kw)


def gt_label(boxes: list[dict]) -> str:
    return "Critical" if any(harness.gt_critical(b) for b in boxes) else "Requires zoom" if boxes else "Clear"


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if not n:
        return [None, None]
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def route_all(records: list[dict], rt: router.Router) -> list[dict]:
    """Every photo through the router as run_batch sends it: decoded, people blurred, with its own triage numbers."""
    futs = []
    for r in records:
        img = batch.load(r["path"])[0]
        tri = batch.triage(img, JobConfig(thumbs=False), make)
        futs.append((tri, rt.submit(r["path"], img, tri)))
    out = []
    for tri, f in futs:
        r = f.result()
        billed = r["usage"] or r.get("first_usage", {})
        out.append({k: r[k] for k in ("label", "reason", "status", "cause", "cached", "latency_ms", "reply") if k in r}
                   | {"usage": billed, "usd": router.usd(billed), "seconds": router.usd(billed) * 3600 / USD_PER_HOUR,
                      "prior_sum": tri["prior_sum"]})
    return out


def order_of(photos: list[dict], arm: str) -> list[int]:
    idx = range(len(photos))
    if arm == "file":
        return sorted(idx, key=lambda i: photos[i]["file_index"])
    if arm == "triage":
        return sorted(idx, key=lambda i: (-photos[i]["triage"]["prior_sum"], photos[i]["file_index"]))
    return sorted((i for i in idx if photos[i]["route"]["label"] != "Clear"), key=lambda i: (
        router.RANK[photos[i]["route"]["label"]], -photos[i]["triage"]["prior_sum"], photos[i]["file_index"]))


def arms(photos: list[dict], frac: float, rng, n_random: int) -> dict:
    budget = frac * sum(batch_triage.full_seconds(p) for p in photos)
    route_s = sum(p["route"]["seconds"] for p in photos)
    rand = [batch_triage.replay(photos, rng.permutation(len(photos)), budget, False) for _ in range(n_random)]
    keys = ("recall", "recall_iou_only", "ttfc_s", "reached", "false_critical", "found")
    return {"budget_s": budget, "router_seconds": route_s,
            "router": batch_triage.replay(photos, order_of(photos, "router"), budget, True, route_s),
            "router_uncharged": batch_triage.replay(photos, order_of(photos, "router"), budget, True),
            "triage": batch_triage.replay(photos, order_of(photos, "triage"), budget, True),
            "file": batch_triage.replay(photos, order_of(photos, "file"), budget, False),
            "random": {k: float(np.mean([r[k] for r in rand])) for k in keys}}


def safety(photos: list[dict]) -> dict:
    crit = [p for p in photos if p["gt"]]
    cleared = [p for p in crit if p["route"]["label"] == "Clear"]
    clean = [p for p in photos if p["gt_label"] == "Clear"]
    return {"photos_with_gt_critical": len(crit), "cleared_with_gt_critical": len(cleared),
            "miss_rate": batch_triage.ratio(len(cleared), len(crit)), "miss_rate_ci": wilson(len(cleared), len(crit)),
            "gt_critical_in_cleared": sum(len(p["gt"]) for p in cleared), "gt_critical": sum(len(p["gt"]) for p in photos),
            "full_run_queued_in_cleared": sum(s is not None for p in cleared for s in p["gt"]),
            "cleared": sum(p["route"]["label"] == "Clear" for p in photos), "photos": len(photos),
            "clean_photos": len(clean), "clean_cleared": sum(p["route"]["label"] == "Clear" for p in clean),
            "cleared_ids": sorted(p["id"] for p in cleared)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evals.router_eval")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(harness.REPO / "results" / "dev" / "router_eval.json"))
    ap.add_argument("--check-dir", default=str(harness.REPO / "runs" / "eval-router"))
    ap.add_argument("--no-check", action="store_true", help="skip the real batch run that checks the replay")
    ap.add_argument("--route-only", action="store_true", help="only route the backlog (fills the cache) and print labels")
    a = ap.parse_args(argv)
    if default_segmenter().name != ENGINE or not os.environ.get("DRISHYAM_ONNX_FP16"):
        raise SystemExit(f"set DRISHYAM_ONNX=models/drishyam_dacl10k.onnx and DRISHYAM_ONNX_FP16 (the rule's engine is "
                         f"{ENGINE}), not {default_segmenter().name}")
    records = []
    for name, d, n, seed, listing in SPLITS:
        if seed == 4243 and not (harness.REPO / d / "annotations.json").exists():  # load_records would add damage
            synth.build_dataset(harness.REPO / d, n, seed, clean=True)
        recs = harness.load_records(harness.REPO / d, n, seed=seed or 7, list_file=listing and harness.REPO / listing)[0]
        records += [{**r, "split": name} for r in recs]
    shuffle = np.random.default_rng(0).permutation(len(records))
    records = [records[i] for i in shuffle]
    harness.cost_rows(MICRO_ARM, default_segmenter().name)
    rt = router.Router(router.Nova())
    routes = route_all(records, rt)
    if a.route_only:
        print(json.dumps([{"id": r["id"], "split": r["split"], "gt": gt_label(r["boxes"]), **ro}
                          for r, ro in zip(records, routes, strict=True)], indent=1))
        return
    photos = harness.pool_map(partial(batch_triage.photo_task, make=make), records, a.workers,
                              lambda m: print(m, flush=True), "photos")
    for i, (p, r, ro) in enumerate(zip(photos, records, routes, strict=True)):
        assert ro["prior_sum"] == p["triage"]["prior_sum"], (r["id"], ro["prior_sum"], p["triage"])  # same overview
        p.update(file_index=i, split=r["split"], route=ro, gt_label=gt_label(r["boxes"]))

    point = {f: arms(photos, f, np.random.default_rng(0), RANDOM_N) for f in BUDGETS}
    rng = np.random.default_rng(harness.BOOT_SEED)
    groups = [[i for i, p in enumerate(photos) if p["split"] == s] for s, *_ in SPLITS]
    pairs = {"recall_vs_triage": ("router", "triage", "recall"), "recall_vs_random": ("router", "random", "recall"),
             "uncharged_vs_triage": ("router_uncharged", "triage", "recall"),
             "uncharged_vs_random": ("router_uncharged", "random", "recall"),
             "iou_vs_triage": ("router", "triage", "recall_iou_only"), "iou_vs_random": ("router", "random",
                                                                                         "recall_iou_only"),
             "ttfc_vs_triage": ("router", "triage", "ttfc_s"), "ttfc_vs_random": ("router", "random", "ttfc_s")}
    boot = {f: {k: [] for k in pairs} for f in BUDGETS}
    miss_boot = []
    for _ in range(harness.BOOT_N):
        sample = [photos[g[j]] for g in groups for j in rng.integers(0, len(g), len(g))]
        for f in BUDGETS:
            r = arms(sample, f, rng, RANDOM_N_BOOT)
            for k, (x, y, m) in pairs.items():
                boot[f][k].append(batch_triage.ratio(r[x][m], r[y][m]))
        miss_boot.append(safety(sample)["miss_rate"])
    out = {}
    for f in BUDGETS:
        p = point[f]
        out[str(f)] = {**p, "ratios": {k: batch_triage.ratio(p[x][m], p[y][m]) for k, (x, y, m) in pairs.items()}
                       | {f"{k}_ci": batch_triage.ci(boot[f][k]) for k in pairs}}
    q = out[str(PRIMARY)]["ratios"]
    cis = (q["recall_vs_triage_ci"], q["recall_vs_random_ci"])
    verdict = ("not computable" if any(c[0] is None for c in cis) else "gain" if min(c[0] for c in cis) > 1
               else "harm" if min(c[1] for c in cis) < 1 else "no measurable difference")
    safe = {"all": safety(photos), **{s: safety([p for p in photos if p["split"] == s]) for s, *_ in SPLITS}}
    safe["all"]["miss_rate_boot_ci"] = batch_triage.ci(miss_boot)
    up = safe["all"]["miss_rate_ci"][1]
    clearing = "safe" if up is not None and up <= SAFE_MISS else "not shown safe"

    labels = router.LABELS
    confusion = {g: {lab: sum(p["gt_label"] == g and p["route"]["label"] == lab for p in photos) for lab in labels}
                 for g in labels}
    by_split = {s: {g: {lab: sum(p["split"] == s and p["gt_label"] == g and p["route"]["label"] == lab for p in photos)
                        for lab in labels} for g in labels} for s, *_ in SPLITS}
    live = [p["route"] for p in photos if not p["route"]["cached"]]
    lat = sorted(r["latency_ms"] for r in live)
    rs = [p["route"] for p in photos]
    jobs_usd = sum(batch_triage.full_seconds(p) for p in photos) * USD_PER_HOUR / 3600
    cost = {"calls_this_run": len(live), "cached_this_run": len(photos) - len(live),
            "usd": sum(r["usd"] for r in rs), "usd_per_photo": sum(r["usd"] for r in rs) / len(rs),
            "input_tokens": sum(r["usage"].get("inputTokens", 0) for r in rs),
            "output_tokens": sum(r["usage"].get("outputTokens", 0) for r in rs),
            "router_seconds_charged": sum(r["seconds"] for r in rs),
            "full_jobs_usd_modelled": jobs_usd, "full_job_usd_per_photo": jobs_usd / len(photos),
            "status": {k: sum(r["status"] == k for r in rs) for k in ("accepted", "rejected", "fallback")},
            "causes": {c: sum(r["cause"] == c for r in rs) for c in sorted({r["cause"] for r in rs if r["cause"]})},
            "latency_ms_this_run": {"p50": lat[len(lat) // 2] if lat else None,
                                    "p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
                                    "max": lat[-1] if lat else None},
            "price_per_1k_tokens": {"input": router.PRICE["input"] * 1000, "output": router.PRICE["output"] * 1000}}

    check = None
    if not a.no_check:  # the replay against a real `drishyam batch --router nova` run at the primary budget (cached)
        Path(a.check_dir).mkdir(parents=True, exist_ok=True)
        bdir = batch.run_batch([Path(r["path"]) for r in records], Path(a.check_dir), JobConfig(thumbs=False),
                               seconds_cap=out[str(PRIMARY)]["budget_s"], make=batch.make_job, key=b"k" * 32,
                               cost_model="fixed", micro_arm=MICRO_ARM, router=rt)
        real = json.loads((bdir / "inspection.json").read_text())
        rec_of = {r["path"]: r for r in records}
        found = 0
        for ph in real["photos"]:
            if "run_dir" in ph:
                wo = json.loads((bdir / ph["run_dir"] / "work_order.json").read_text())
                crit = [f for f in wo["findings"] if f["critical"]]
                found += sum(any(f["cls"] == g["cls"] and harness.hits(f["box"], g["box"]) for f in crit)
                             for g in rec_of[ph["photo"]]["boxes"] if harness.gt_critical(g))
        # a rerun's cached labels cost nothing, so a run whose every label came from the cache charges no router $
        arm = "router_uncharged" if real["router"]["cached"] == real["router"]["photos"] else "router"
        sim = out[str(PRIMARY)][arm]
        check = {"batch": str(bdir.relative_to(harness.REPO)) if bdir.is_relative_to(harness.REPO) else str(bdir),
                 "replay_arm": arm,
                 "reached": [sum("run_dir" in ph for ph in real["photos"]), sim["reached"]],
                 "found": [found, sim["found"]],
                 "router_seconds": [real["spent"]["router_seconds"], out[str(PRIMARY)]["router_seconds"]],
                 "cleared": [real["counts"]["cleared"], safe["all"]["cleared"]]}
        check["agrees"] = check["reached"][0] == check["reached"][1] and check["found"][0] == check["found"][1]

    result = {"rule": RULE, "verdict": verdict, "clearing": clearing, "primary_budget_frac": PRIMARY, "n": len(photos),
              "splits": {s: sum(p["split"] == s for p in photos) for s, *_ in SPLITS}, "file_order_seed": 0,
              **harness.dataset_sha256(records), "segmenter": harness.segmenter_id(default_segmenter()),
              "micro_arm": MICRO_ARM, "router": {"model": router.MODEL_ID, "region": router.REGION, "prompt_key": rt.key},
              "boot": {"n": harness.BOOT_N, "seed": harness.BOOT_SEED, "random_per_resample": RANDOM_N_BOOT},
              "gt_critical": sum(len(p["gt"]) for p in photos), "photos_with_gt_critical": sum(bool(p["gt"]) for p in photos),
              "full_seconds": sum(batch_triage.full_seconds(p) for p in photos),
              "triage_seconds": sum(p["triage"]["seconds"] for p in photos),
              "budgets": out, "safety": safe, "agreement": {
                  "accuracy": sum(p["gt_label"] == p["route"]["label"] for p in photos) / len(photos),
                  "confusion_gt_by_router": confusion, "by_split": by_split}, "cost": cost, "check": check,
              "per_photo": [{"id": p["id"], "split": p["split"], "file_index": p["file_index"],
                             "prior_sum": p["triage"]["prior_sum"], "seconds": batch_triage.full_seconds(p),
                             "gt_label": p["gt_label"], "gt_critical": len(p["gt"]),
                             "queued_full_run": sum(s is not None for s in p["gt"]),
                             "route": {k: v for k, v in p["route"].items() if k != "prior_sum"}} for p in photos]}
    Path(a.out).write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_photo", "rule")}, indent=1, default=float))


if __name__ == "__main__":
    main()
