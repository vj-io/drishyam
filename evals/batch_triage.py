"""Dev only: does `drishyam batch`'s triage queue the labelled critical defects for a person sooner than running the
same backlog in file order or in a random order?

    python -m evals.batch_triage [--workers 4] [--out results/dev/batch_triage.json]

The backlog: synth_dev (60 photos with damage, seed 4242) and synth_clean_dev (its first 30: distractors, no damage,
seed 4243), 90 photos listed in a seeded shuffle (seed 0) that stands in for capture order: "file order". It never
reads a held-out set. Each photo's job runs once as `drishyam batch` runs it (decoded, people blurred, the default
per-photo caps, thumbnails off, the published fixed costs of results/linux-aarch64/micro.json on a virtual clock),
recording the seconds it had used before each look. A job given a smaller seconds cap takes the same looks up to the
cap and halts there (on these costs the cap only halts: the stop rule and the survey read pixels), so the batch of any
order and total budget is replayed from those records; a photo cut short by the budget is credited with the critical
findings of the looks it ran, boxes as the full run left them. A real `drishyam batch` run of the triage arm at the
primary budget checks the replay (`check`).

A labelled critical defect (evals.harness.gt_critical: exposed rebar, or spalling whose box is at least the critical
area) counts as queued for a person once a critical finding of its class hits it (evals.harness.hits: IoU >= 0.3 or
containing its centre; IoU-only reported too). Arms: triage (the batch's order, its triage cost charged first), file
order, and random order (mean of 1000 seeded permutations; 100 per bootstrap resample), the last two charged no triage.
The paired bootstrap resamples photos within each split (1000 resamples, seed 0, as evals.harness.boot_ci); each
resample's budget is the same share of its own total job seconds.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from drishyam import batch
from drishyam.agent import JobConfig
from drishyam.segment import default_segmenter
from evals import harness, synth

RULE = (
    "Stated before the run (committed with this file). Primary metric: critical recall within budget, the share of the "
    "backlog's labelled critical defects that a critical finding queues for a person before the batch budget runs out, "
    "at a total budget of half the seconds the 90 jobs take run in full (triage's own cost charged to the triage arm). "
    "Gain: the paired 95 % bootstrap CI of triage / random order lies above 1, and so does that of triage / file order. "
    "Harm: either CI lies below 1. Otherwise: no measurable difference. Time to the first queued critical defect "
    "(censored at the budget), the IoU-only variant, the other budgets (a quarter, three quarters) and the queue's "
    "critical findings that match no labelled critical defect are reported, not part of the verdict."
)
SPLITS = (("synth_dev", "data/synth_dev", 60, 4242, False), ("clean", "data/synth_clean_dev", 30, 4243, True))
MICRO_ARM, SECONDS_CAP = "linux-aarch64", JobConfig().seconds_cap
PRIMARY, BUDGETS = 0.5, (0.25, 0.5, 0.75)
RANDOM_N, RANDOM_N_BOOT = 1000, 100


def make(img, cfg, **kw):
    return batch.make_job(img, cfg, cost_model="fixed", micro_arm=MICRO_ARM, **kw)


def photo_task(rec: dict, make=make) -> dict:
    """One photo's triage and full job, with what the replay needs: the seconds used after the overview and after
    each look, and for each labelled critical defect the first look after which a critical finding queues it.
    `make`: the Job factory (evals.router_eval passes one on another cost table)."""
    img, _ = batch.load(rec["path"])
    cfg = JobConfig(thumbs=False)
    tri = batch.triage(img, cfg, make)
    job, post, overview = make(img, cfg), [], None
    for ev in job.run():
        if ev["type"] == "overview":
            overview = job.budget.seconds_used
        elif ev["type"] == "step" and ev["decision"]["action"] != "halt":
            post.append(job.budget.seconds_used)
    crit = [f for f in job.findings if f.critical]
    gts = [g for g in rec["boxes"] if harness.gt_critical(g)]
    first = lambda g, centre: min((f.critical_step for f in crit if f.cls == g["cls"] and
                                   harness.hits(f.box, g["box"], centre)), default=None)
    return {"id": rec["id"], "triage": tri, "overview_s": overview, "post": post,
            "gt": [first(g, True) for g in gts], "gt_iou": [first(g, False) for g in gts],
            "false": [f.critical_step for f in crit if not any(
                g["cls"] == f.cls and harness.hits(f.box, g["box"]) for g in gts)]}


def full_seconds(p: dict) -> float:
    return p["post"][-1] if p["post"] else p["overview_s"]


def replay(photos: list[dict], order, budget: float, triage: bool, extra: float = 0.0) -> dict:
    """The batch run_batch would run: photos in `order` (indices into `photos`), each job's seconds cap what is left;
    `extra`: further seconds charged before the first job (a router's $)."""
    spent = (sum(p["triage"]["seconds"] for p in photos) if triage else 0.0) + extra
    found = found_iou = false = reached = 0
    first = None
    for i in order:
        p = photos[i]
        if spent >= budget:
            break
        cap, before, ran = min(SECONDS_CAP, budget - spent), p["overview_s"], 0
        for t in p["post"]:  # the governor's check before each look: seconds used so far < cap
            if before >= cap:
                break
            ran, before = ran + 1, t
        times = [spent + p["post"][s - 1] for s in p["gt"] if s is not None and s <= ran]
        found += len(times)
        found_iou += sum(s is not None and s <= ran for s in p["gt_iou"])
        false += sum(s <= ran for s in p["false"])
        first = min([first, *times], key=lambda x: np.inf if x is None else x)
        spent, reached = spent + before, reached + 1
    total = sum(len(p["gt"]) for p in photos)
    return {"recall": ratio(found, total), "recall_iou_only": ratio(found_iou, total), "found": found, "gt": total,
            "ttfc_s": budget if first is None else first, "ttfc_censored": first is None, "reached": reached,
            "false_critical": false, "spent_s": spent}


def arms(photos: list[dict], frac: float, rng, n_random: int) -> dict:
    budget = frac * sum(full_seconds(p) for p in photos)
    tri = sorted(range(len(photos)), key=lambda i: (-photos[i]["triage"]["prior_sum"], photos[i]["file_index"]))
    by_file = sorted(range(len(photos)), key=lambda i: photos[i]["file_index"])
    rand = [replay(photos, rng.permutation(len(photos)), budget, False) for _ in range(n_random)]
    mean = {k: float(np.mean([r[k] for r in rand])) for k in ("recall", "recall_iou_only", "ttfc_s", "reached",
                                                              "false_critical", "found")}
    return {"budget_s": budget, "triage": replay(photos, tri, budget, True), "file": replay(photos, by_file, budget, False),
            "random": {**mean, "ttfc_censored_share": float(np.mean([r["ttfc_censored"] for r in rand]))}}


def ci(vals) -> list:
    vals = np.asarray(vals, float)
    return [None, None] if np.isnan(vals).all() else [round(float(v), 4) for v in np.nanpercentile(vals, [2.5, 97.5])]


def ratio(a, b):
    return a / b if b else np.nan


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evals.batch_triage")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(harness.REPO / "results" / "dev" / "batch_triage.json"))
    ap.add_argument("--check-dir", default=str(harness.REPO / "runs" / "eval-batch"))
    a = ap.parse_args(argv)
    records = []
    for name, d, n, seed, clean in SPLITS:
        if clean and not (harness.REPO / d / "annotations.json").exists():  # load_records would add damage
            synth.build_dataset(harness.REPO / d, n, seed, clean=True)
        records += [{**r, "split": name} for r in harness.load_records(harness.REPO / d, n, seed=seed)[0]]
    shuffle = np.random.default_rng(0).permutation(len(records))  # file order: position in the listing
    records = [records[i] for i in shuffle]
    harness.cost_rows(MICRO_ARM, default_segmenter().name)  # refuses a table that timed another segmenter
    photos = harness.pool_map(photo_task, records, a.workers, lambda m: print(m, flush=True), "photos")
    for i, (p, r) in enumerate(zip(photos, records, strict=True)):
        p.update(file_index=i, split=r["split"])

    point = {f: arms(photos, f, np.random.default_rng(0), RANDOM_N) for f in BUDGETS}
    # paired bootstrap: the same resamples (within each split) for every arm
    rng = np.random.default_rng(harness.BOOT_SEED)
    groups = [[i for i, p in enumerate(photos) if p["split"] == s] for s, *_ in SPLITS]
    boot = {f: {k: [] for k in ("recall_vs_random", "recall_vs_file", "iou_vs_random", "iou_vs_file",
                                "ttfc_vs_random", "ttfc_vs_file")} for f in BUDGETS}
    for _ in range(harness.BOOT_N):
        idx = [g[j] for g in groups for j in rng.integers(0, len(g), len(g))]
        sample = [photos[i] for i in idx]
        for f in BUDGETS:
            r = arms(sample, f, rng, RANDOM_N_BOOT)
            t, fi, ra = r["triage"], r["file"], r["random"]
            b = boot[f]
            b["recall_vs_random"].append(ratio(t["recall"], ra["recall"]))
            b["recall_vs_file"].append(ratio(t["recall"], fi["recall"]))
            b["iou_vs_random"].append(ratio(t["recall_iou_only"], ra["recall_iou_only"]))
            b["iou_vs_file"].append(ratio(t["recall_iou_only"], fi["recall_iou_only"]))
            b["ttfc_vs_random"].append(ratio(t["ttfc_s"], ra["ttfc_s"]))
            b["ttfc_vs_file"].append(ratio(t["ttfc_s"], fi["ttfc_s"]))
    out = {}
    for f in BUDGETS:
        p = point[f]
        t, fi, ra = p["triage"], p["file"], p["random"]
        out[str(f)] = {**p, "ratios": {
            "recall_vs_random": ratio(t["recall"], ra["recall"]), "recall_vs_random_ci": ci(boot[f]["recall_vs_random"]),
            "recall_vs_file": ratio(t["recall"], fi["recall"]), "recall_vs_file_ci": ci(boot[f]["recall_vs_file"]),
            "iou_vs_random_ci": ci(boot[f]["iou_vs_random"]), "iou_vs_file_ci": ci(boot[f]["iou_vs_file"]),
            "ttfc_vs_random": ratio(t["ttfc_s"], ra["ttfc_s"]), "ttfc_vs_random_ci": ci(boot[f]["ttfc_vs_random"]),
            "ttfc_vs_file": ratio(t["ttfc_s"], fi["ttfc_s"]), "ttfc_vs_file_ci": ci(boot[f]["ttfc_vs_file"])}}
    q = out[str(PRIMARY)]["ratios"]
    lo = min(q["recall_vs_random_ci"][0], q["recall_vs_file_ci"][0])
    hi = min(q["recall_vs_random_ci"][1], q["recall_vs_file_ci"][1])
    verdict = "gain" if lo > 1 else "harm" if hi < 1 else "no measurable difference"

    # the replay against a real `drishyam batch` run of the triage arm at the primary budget
    Path(a.check_dir).mkdir(parents=True, exist_ok=True)
    bdir = batch.run_batch([Path(r["path"]) for r in records], Path(a.check_dir), JobConfig(thumbs=False),
                           seconds_cap=out[str(PRIMARY)]["budget_s"], make=batch.make_job, key=b"k" * 32,
                           cost_model="fixed", micro_arm=MICRO_ARM)
    real = json.loads((bdir / "inspection.json").read_text())
    rec_of = {r["path"]: r for r in records}
    found = 0
    for ph in real["photos"]:
        if "run_dir" not in ph:
            continue
        wo = json.loads((bdir / ph["run_dir"] / "work_order.json").read_text())
        crit = [f for f in wo["findings"] if f["critical"]]
        found += sum(any(f["cls"] == g["cls"] and harness.hits(f["box"], g["box"]) for f in crit)
                     for g in rec_of[ph["photo"]]["boxes"] if harness.gt_critical(g))
    sim = out[str(PRIMARY)]["triage"]
    check = {"batch": str(bdir.relative_to(harness.REPO)) if bdir.is_relative_to(harness.REPO) else str(bdir),
             "reached": [sum("run_dir" in ph for ph in real["photos"]), sim["reached"]],
             "found": [found, sim["found"]], "spent_s": [real["spent"]["seconds"], sim["spent_s"]],
             "order_matches": [Path(ph["photo"]).name for ph in sorted(
                 (ph for ph in real["photos"] if ph["rank"]), key=lambda ph: ph["rank"])] == [
                 Path(records[i]["path"]).name for i in sorted(
                     range(len(photos)), key=lambda i: (-photos[i]["triage"]["prior_sum"], i))]}
    check["agrees"] = check["reached"][0] == check["reached"][1] and check["found"][0] == check["found"][1]

    result = {"rule": RULE, "verdict": verdict, "primary_budget_frac": PRIMARY, "n": len(photos),
              "splits": {s: sum(p["split"] == s for p in photos) for s, *_ in SPLITS}, "file_order_seed": 0,
              **harness.dataset_sha256(records), "segmenter": harness.segmenter_id(default_segmenter()),
              "micro_arm": MICRO_ARM, "per_photo_caps": {"pixel_cap": JobConfig().pixel_cap,
                                                         "usd_cap": JobConfig().usd_cap, "seconds_cap": SECONDS_CAP},
              "boot": {"n": harness.BOOT_N, "seed": harness.BOOT_SEED, "random_per_resample": RANDOM_N_BOOT},
              "gt_critical": sum(len(p["gt"]) for p in photos),
              "photos_with_gt_critical": sum(bool(p["gt"]) for p in photos),
              "full_seconds": sum(full_seconds(p) for p in photos),
              "triage_seconds": sum(p["triage"]["seconds"] for p in photos),
              "budgets": out, "check": check,
              "per_photo": [{"id": p["id"], "split": p["split"], "file_index": p["file_index"],
                             "prior_sum": p["triage"]["prior_sum"], "candidates": p["triage"]["candidates"],
                             "seconds": full_seconds(p), "gt_critical": len(p["gt"]),
                             "queued_full_run": sum(s is not None for s in p["gt"]), "false_critical": len(p["false"])}
                            for p in photos]}
    Path(a.out).write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_photo", "rule")}, indent=1, default=float))


if __name__ == "__main__":
    main()
