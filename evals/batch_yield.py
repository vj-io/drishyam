"""Dev only: does `drishyam batch --batch-yield` (the patch-leaving rule, JobConfig.next_ev; Charnov's marginal value
theorem, 1976) queue more of the labelled critical defects for a person within a budget than the rule triage alone?

    DRISHYAM_ONNX=models/drishyam_dacl10k.onnx DRISHYAM_ONNX_FP16=models/drishyam_dacl10k.fp16.onnx \\
        python -m evals.batch_yield [--workers 4]

The rule, backlog, agent and arms are stated in results/dev/batch_yield.json, committed before this ran; the run writes
its numbers into that file beside them, and refuses to run once they are there. The backlog is evals/router_eval.py's
190 dev photos in its seeded shuffle, each job run once in full as `drishyam batch` runs it (v1, fixed costs on a virtual
clock). The rule only ends a job earlier: a job given a seconds cap or a next_ev takes the same looks, on the same fixed
cost table, up to its halt. So every arm and budget is replayed exactly from those runs, crediting each truncated job
with the work order it would leave: the critical findings as they stood after its last look, and its not-assessed list
(a halted job also lists every candidate it did not look at). A real `run_batch(batch_yield=True)` on the first
CHECK_N photos checks the replay. No CODEBRIM image is written: only ids, labels and numbers.
"""

import argparse
import bisect
import json
import os
import subprocess
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np

from drishyam import batch
from drishyam.agent import JobConfig
from drishyam.segment import default_segmenter
from evals import batch_triage, harness, router_eval

RULE_FILE = harness.REPO / "results" / "dev" / "batch_yield.json"
PRIMARY, BUDGETS = 0.5, (0.25, 0.5, 0.75)
RANDOM_N, RANDOM_N_BOOT = 1000, 100
SECONDS_CAP = JobConfig().seconds_cap
CHECK_N = 24
HALTED = ("cap-hit", "kill-switch", "batch-yield")


def photo_task(rec: dict, make=router_eval.make) -> dict:
    """One photo's triage and full job, with what an exact replay of any truncation needs: the seconds used after the
    overview and after each look (`t`), each best-score look's value, the job's own halt cause, and after k looks (`at`,
    k = 0 .. n) what a job halting there leaves: [labelled critical defects a critical finding hits, the same by IoU
    only, critical findings hitting none, critical findings, not-assessed entries if it ends there, if it halts there].
    Also batch_triage.photo_task's fields (post, gt, gt_iou, false: the published credit, final boxes), so the published
    replay can be checked on the same runs."""
    img, _ = batch.load(rec["path"])
    cfg = JobConfig(thumbs=False)
    tri = batch.triage(img, cfg, make)
    gts = [g for g in rec["boxes"] if harness.gt_critical(g)]
    zero = list(make(img, replace(cfg, seconds_cap=0.0)).run())[-1]  # halts before its first look: k = 0
    at = [[0, 0, 0, 0, None, len(zero["work_order"]["not_assessed"])]]
    job, t, best = make(img, cfg), [], []

    def state() -> list:
        crit = [(f.cls, f.box) for f in job.findings if f.critical]
        hit = lambda centre: sum(any(c == g["cls"] and harness.hits(b, g["box"], centre) for c, b in crit) for g in gts)
        false = sum(not any(g["cls"] == c and harness.hits(b, g["box"]) for g in gts) for c, b in crit)
        na = len(job.not_assessed) + sum(len(job.regions[rid].held) for rid in job.pending)
        unlooked = sum(not r.looks and r.box is not None for r in job.regions.values())
        return [hit(True), hit(False), false, len(crit), na, na + unlooked]

    for ev in job.run():
        if ev["type"] == "overview":
            t.append(job.budget.seconds_used)
        elif ev["type"] == "step" and ev["decision"]["action"] != "halt":
            if ev["cause"] == "best-score":
                best.append((len(t) - 1, ev["value"]))
            t.append(job.budget.seconds_used)
            at.append(state())
        done = ev
    n, na_done = len(t) - 1, len(done["work_order"]["not_assessed"])
    assert at[n][5 if done["cause"] in HALTED else 4] in (na_done, None), (rec["id"], at[n], na_done)
    at[n][4 if done["cause"] not in HALTED else 5] = na_done
    crit = [f for f in job.findings if f.critical]
    first = lambda g, centre: min((f.critical_step for f in crit if f.cls == g["cls"] and
                                   harness.hits(f.box, g["box"], centre)), default=None)
    return {"id": rec["id"], "triage": tri, "t": t, "best": best, "end": done["cause"], "at": at,
            "overview_s": t[0], "post": t[1:], "gt": [first(g, True) for g in gts],
            "gt_iou": [first(g, False) for g in gts],
            "false": [f.critical_step for f in crit if not any(g["cls"] == f.cls and harness.hits(f.box, g["box"])
                                                               for g in gts)]}


def job_replay(p: dict, cap: float, next_ev: float) -> tuple[int, str]:
    """(looks run, halt cause) of p's job under a seconds cap and a next_ev, from its full run: the governor's seconds
    check runs before each look (t[j] < cap), then the argmax's batch-yield check (a best-score look j >= 1 worth less
    than next_ev); after its last look the governor still halts it when no time is left."""
    t, n = p["t"], len(p["t"]) - 1
    k = bisect.bisect_left(t, cap, 0, n)
    y = next((j for j, v in p["best"] if 0 < j < k and v < next_ev), None) if next_ev > 0 else None
    if y is not None:
        return y, "batch-yield"
    return (k, "cap-hit") if k < n or t[n] >= cap else (n, p["end"])


def replay(photos: list[dict], order, budget: float, triage: bool, yield_on: bool, detail: bool = False) -> dict:
    """The batch run_batch would run: photos in `order` (indices into `photos`), each job's seconds cap what is left,
    and with `yield_on` its next_ev as run_batch sets it (the next photo's triage prior_max under budget pressure)."""
    spent = sum(p["triage"]["seconds"] for p in photos) if triage else 0.0
    found = found_iou = false = false_clean = crit = na = yields = 0
    first, jobs, ran = None, [], []
    for pos, i in enumerate(order):
        if spent >= budget:
            break
        p, left = photos[i], len(order) - pos
        pressed = yield_on and jobs and budget - spent < left * sum(jobs) / len(jobs)
        nev = photos[order[pos + 1]]["triage"]["prior_max"] if pressed and pos + 1 < len(order) else 0.0
        k, cause = job_replay(p, min(SECONDS_CAP, budget - spent), nev)
        h, h_iou, f, c, na_end, na_halt = p["at"][k]
        found, found_iou, false, crit = found + h, found_iou + h_iou, false + f, crit + c
        false_clean += c if p.get("split") == "clean" else 0
        na += na_halt if cause in HALTED else na_end
        yields += cause == "batch-yield"
        if h and (q := next(j for j in range(1, k + 1) if p["at"][j][0])):
            first = min(first if first is not None else np.inf, spent + p["t"][q])
        if detail:
            ran.append({"index": i, "looks": k, "cause": cause, "seconds": p["t"][k], "next_ev": nev})
        jobs.append(p["t"][k])
        spent += p["t"][k]
    total = sum(len(p["gt"]) for p in photos)
    out = {"recall": batch_triage.ratio(found, total), "recall_iou_only": batch_triage.ratio(found_iou, total),
           "found": found, "gt": total, "ttfc_s": budget if first is None else first, "ttfc_censored": first is None,
           "reached": len(jobs), "not_assessed": na + len(order) - len(jobs), "false_critical": false,
           "false_critical_clean": false_clean, "critical": crit, "critical_per_photo": crit / max(1, len(jobs)),
           "batch_yield_halts": yields, "spent_s": spent}
    return out | ({"jobs": ran} if detail else {})


def triage_order(photos: list[dict]) -> list[int]:
    return sorted(range(len(photos)), key=lambda i: (-photos[i]["triage"]["prior_sum"], photos[i]["file_index"]))


KEYS = ("recall", "recall_iou_only", "found", "ttfc_s", "reached", "not_assessed", "false_critical",
        "false_critical_clean", "critical", "critical_per_photo")


def arms(photos: list[dict], frac: float, rng, n_random: int) -> dict:
    budget = frac * sum(batch_triage.full_seconds(p) for p in photos)
    tri = triage_order(photos)
    rand = [replay(photos, rng.permutation(len(photos)), budget, False, False) for _ in range(n_random)]
    return {"budget_s": budget, "triage": replay(photos, tri, budget, True, False),
            "yield": replay(photos, tri, budget, True, True),
            "random": {k: float(np.mean([r[k] for r in rand])) for k in KEYS}}


PAIRS = {"recall_yield_vs_triage": ("yield", "triage", "recall"),
         "iou_yield_vs_triage": ("yield", "triage", "recall_iou_only"),
         "ttfc_yield_vs_triage": ("yield", "triage", "ttfc_s"),
         "recall_triage_vs_random": ("triage", "random", "recall"),
         "recall_yield_vs_random": ("yield", "random", "recall")}


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=harness.REPO, check=True, capture_output=True, text=True).stdout.strip()


def check_real(photos: list[dict], records: list[dict], out_dir: Path) -> dict:
    """The replay of the batch-yield arm against a real run_batch(batch_yield=True) of the first CHECK_N photos (file
    order) at half their full seconds: the same jobs, looks, causes, seconds and next_ev, critical defects queued,
    critical findings and not-assessed entries."""
    sub = [dict(p, file_index=i) for i, p in enumerate(photos[:CHECK_N])]
    budget = PRIMARY * sum(batch_triage.full_seconds(p) for p in sub)
    sim = replay(sub, triage_order(sub), budget, True, True, detail=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    bdir = batch.run_batch([Path(r["path"]) for r in records[:CHECK_N]], out_dir, JobConfig(thumbs=False),
                           seconds_cap=budget, make=batch.make_job, key=b"k" * 32, cost_model="fixed",
                           micro_arm=router_eval.MICRO_ARM, batch_yield=True)
    real = json.loads((bdir / "inspection.json").read_text())
    trace = [json.loads(line) for line in (bdir / "batch_trace.jsonl").read_text().splitlines()]
    rec_of = {str(Path(r["path"])): r for r in records[:CHECK_N]}
    by_photo = {ph["photo"]: ph for ph in real["photos"]}
    jobs, found = [], 0
    for j in sim["jobs"]:
        ph = by_photo[str(Path(records[j["index"]]["path"]))]
        jobs.append([ph.get("steps"), ph.get("cause"), ph.get("seconds"), j["looks"], j["cause"], j["seconds"]])
        wo = json.loads((bdir / ph["run_dir"] / "work_order.json").read_text())
        crit = [f for f in wo["findings"] if f["critical"]]
        found += sum(any(f["cls"] == g["cls"] and harness.hits(f["box"], g["box"]) for f in crit)
                     for g in rec_of[ph["photo"]]["boxes"] if harness.gt_critical(g))
    nev = [t["next_ev"] for t in trace if t["decision"] == "started"]
    out = {"batch": str(bdir.relative_to(harness.REPO)) if bdir.is_relative_to(harness.REPO) else str(bdir),
           "photos": CHECK_N, "budget_s": budget, "reached": [sum("run_dir" in ph for ph in real["photos"]),
                                                              sim["reached"]],
           "found": [found, sim["found"]], "critical": [real["counts"]["critical"], sim["critical"]],
           "not_assessed": [len(real["not_assessed"]), sim["not_assessed"]],
           "spent_s": [real["spent"]["seconds"], sim["spent_s"]],
           "next_ev": [nev, [j["next_ev"] for j in sim["jobs"]]],
           "batch_yield_halts": [sum(ph.get("cause") == "batch-yield" for ph in real["photos"]),
                                 sim["batch_yield_halts"]],
           "jobs_real_vs_replay": jobs}
    out["agrees"] = (all(a == b for a, b in (out[k] for k in ("reached", "found", "critical", "not_assessed",
                                                             "next_ev", "batch_yield_halts")))
                     and all(j[:3] == j[3:] for j in jobs) and abs(out["spent_s"][0] - out["spent_s"][1]) < 1e-9)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evals.batch_yield")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--check-dir", default=str(harness.REPO / "runs" / "eval-batch-yield"))
    a = ap.parse_args(argv)
    if default_segmenter().name != router_eval.ENGINE or not os.environ.get("DRISHYAM_ONNX_FP16"):
        raise SystemExit("set DRISHYAM_ONNX=models/drishyam_dacl10k.onnx and DRISHYAM_ONNX_FP16: the rule's engine is "
                         f"{router_eval.ENGINE}, not {default_segmenter().name}")
    doc = json.loads(RULE_FILE.read_text())
    rel = str(RULE_FILE.relative_to(harness.REPO))
    if "verdict" in doc or git("status", "--porcelain", "--", rel):
        raise SystemExit(f"{rel} already holds outputs or is not committed as it stands: the rule is run once")
    commit, committed = git("log", "-1", "--format=%H %cI", "--", rel).split()
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    assert datetime.fromisoformat(committed) < datetime.fromisoformat(started), (committed, started)

    records = []
    for name, d, n, seed, listing in router_eval.SPLITS:
        if listing is None and not (harness.REPO / d / "annotations.json").exists():  # load_records would build one
            raise SystemExit(f"{d} is missing: link or build router_eval's dev split first")
        recs = harness.load_records(harness.REPO / d, n, seed=seed or 7, list_file=listing and harness.REPO / listing)[0]
        assert len(recs) == n, (d, len(recs))  # never a smaller set left behind by a smoke run
        records += [{**r, "split": name} for r in recs]
    records = [records[i] for i in np.random.default_rng(0).permutation(len(records))]
    harness.cost_rows(router_eval.MICRO_ARM, default_segmenter().name)
    photos = harness.pool_map(partial(photo_task, make=router_eval.make), records, a.workers,
                              lambda m: print(m, flush=True), "photos")
    for i, (p, r) in enumerate(zip(photos, records, strict=True)):
        p.update(file_index=i, split=r["split"])

    # identity: at next_ev 0 these runs are router_eval's (published per-photo seconds, and its triage arm replayed)
    pub = json.loads((harness.REPO / "results" / "dev" / "router_eval.json").read_text())
    tri_pub = batch_triage.replay(photos, triage_order(photos), pub["budgets"][str(PRIMARY)]["budget_s"], True)
    keys = ("recall", "found", "reached", "false_critical", "spent_s")
    identity = {"per_photo_seconds_equal": [batch_triage.full_seconds(p) for p in photos] == [
                    q["seconds"] for q in pub["per_photo"]],
                "per_photo_queued_equal": [sum(s is not None for s in p["gt"]) for p in photos] == [
                    q["queued_full_run"] for q in pub["per_photo"]],
                "triage_arm_published_credit": {k: [tri_pub[k], pub["budgets"][str(PRIMARY)]["triage"][k]]
                                                for k in keys}}
    identity["agrees"] = (identity["per_photo_seconds_equal"] and identity["per_photo_queued_equal"]
                          and all(x == y for x, y in identity["triage_arm_published_credit"].values()))

    point = {f: arms(photos, f, np.random.default_rng(0), RANDOM_N) for f in BUDGETS}
    rng = np.random.default_rng(harness.BOOT_SEED)
    groups = [[i for i, p in enumerate(photos) if p["split"] == s] for s, *_ in router_eval.SPLITS]
    boot = {f: {k: [] for k in PAIRS} for f in BUDGETS}
    for _ in range(harness.BOOT_N):
        sample = [photos[g[j]] for g in groups for j in rng.integers(0, len(g), len(g))]
        for f in BUDGETS:
            r = arms(sample, f, rng, RANDOM_N_BOOT)
            for k, (x, y, m) in PAIRS.items():
                boot[f][k].append(batch_triage.ratio(r[x][m], r[y][m]))
    budgets = {str(f): {**point[f], "ratios": {k: batch_triage.ratio(point[f][x][m], point[f][y][m])
                                               for k, (x, y, m) in PAIRS.items()}
                        | {f"{k}_ci": batch_triage.ci(boot[f][k]) for k in PAIRS}} for f in BUDGETS}
    q, p = budgets[str(PRIMARY)]["ratios"], budgets[str(PRIMARY)]
    lo, hi = q["recall_yield_vs_triage_ci"]
    verdict = ("not computable" if lo is None else "gain" if lo > 1 else "harm" if hi < 1
               else "no measurable difference")
    adopt = verdict == "gain" and p["yield"]["false_critical_clean"] <= p["triage"]["false_critical_clean"]

    vals = [p["best"][0][1] / p["triage"]["prior_max"] for p in photos if p["best"] and p["triage"]["prior_max"]]
    proxy = {"photos": len(vals), "first_best_score_value_over_prior_max_quartiles": [
                 round(float(v), 4) for v in np.percentile(vals, [25, 50, 75])],
             "share_equal": float(np.mean([abs(v - 1) < 1e-9 for v in vals])),
             "share_above": float(np.mean([v > 1 + 1e-9 for v in vals]))}
    check = check_real(photos, records, Path(a.check_dir))

    doc.update({
        "verdict": verdict, "adopted": adopt, "rule_commit": commit, "rule_committed": committed,
        "run_started": started, "n": len(photos),
        "splits": {s: sum(p["split"] == s for p in photos) for s, *_ in router_eval.SPLITS},
        **harness.dataset_sha256(records), "segmenter": harness.segmenter_id(default_segmenter()),
        "micro_arm": router_eval.MICRO_ARM,
        "boot": {"n": harness.BOOT_N, "seed": harness.BOOT_SEED, "random_per_resample": RANDOM_N_BOOT},
        "gt_critical": sum(len(p["gt"]) for p in photos),
        "full_seconds": sum(batch_triage.full_seconds(p) for p in photos),
        "triage_seconds": sum(p["triage"]["seconds"] for p in photos),
        "identity_check": identity, "next_ev_proxy": proxy, "budgets": budgets, "check": check,
        "per_photo": [{"id": p["id"], "split": p["split"], "file_index": p["file_index"],
                       "prior_sum": p["triage"]["prior_sum"], "prior_max": p["triage"]["prior_max"],
                       "looks": len(p["t"]) - 1, "seconds": batch_triage.full_seconds(p), "end": p["end"],
                       "best_score_values": [round(v, 6) for _, v in p["best"]], "gt_critical": len(p["gt"])}
                      for p in photos]})
    RULE_FILE.write_text(json.dumps(doc, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k not in ("per_photo", "rule", "budgets")}, indent=1,
                     default=float))
    print(json.dumps({f: {k: v for k, v in b.items()} for f, b in budgets.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
