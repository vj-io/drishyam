"""Dev only: the both-gates blur gate (JobConfig.blur_gate "both": blur only if the Laplacian variance is under
SHARPNESS_MIN and the re-blur ratio, imaging.reblur after Crete-Roffet et al. 2007, is above REBLUR_MAX) against the
Laplacian gate, for v1 (v1-policy2 at its profile) at the 30 % pixel cap, on the fixed costs v1's dev runs were charged.

    DRISHYAM_ONNX=models/drishyam_dacl10k.onnx python -m evals.blur_gate calibrate [--workers 6]
    DRISHYAM_ONNX=models/drishyam_dacl10k.onnx python -m evals.blur_gate dev [--workers 6]
    DRISHYAM_ONNX=models/drishyam_dacl10k.onnx python -m evals.blur_gate real01

Each photo's job runs with the Laplacian gate (in `dev` with the both-gates gate too). Every look of the Laplacian job
is rendered again as its gate read it (the sharpness its step recorded must match) and scored by both gates as it is and
blurred by a Gaussian of SIGMA view px, the rule's 'too blurred to trust'.
`calibrate` does that on a pool outside every dev and held-out set (`pool_records`) and prints imaging.REBLUR_MAX: the
largest value, to STEP, at which the both-gates gate refuses at least as many of the pool's blurred views as the
Laplacian gate does. It writes the views to runs/blur_gate/calibration.json; results/dev/blur_gate.json quotes them.
`dev` runs the rule stated in results/dev/blur_gate.json, committed before it ran, on the dev splits, writes its numbers
into that file beside the rule, and refuses to run once they are there. `real01` then adds the report on real_01, which
no clause reads, once the verdict is committed. No image is written: ids and numbers only.
"""

import argparse
import collections
import gzip
import json
from datetime import datetime
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from drishyam import batch, imaging
from drishyam.segment import default_segmenter
from evals import codebrim, dev_sweep, harness, synth
from evals.batch_yield import git

RULE_FILE = harness.REPO / "results" / "dev" / "blur_gate.json"
OUT = harness.REPO / "runs" / "blur_gate"
ENGINE, MICRO_ARM = "onnx:drishyam_dacl10k.onnx", "local-macos-x86-dacl10k-retime"
CAP, SIGMA, STEP = 0.3, 2.0, 0.005
GATES = imaging.BLUR_GATES  # ("laplacian", "both"): the gate as published first, the reference
# The pool: synthetic photos of seeds no other set uses (a clean set's demo photo bridge_08, its last, left out) and
# CODEBRIM photos in neither the dev list nor the held-out list, drawn with POOL_SEED.
POOL_DIR = harness.REPO / "data" / "blur_pool"
POOL_SYNTH = (("synth_5150", 40, 5150, False), ("clean_5151", 20, 5151, True))  # (folder, n, seed, clean)
POOL_CODEBRIM, POOL_SEED = 100, 5150
SPLITS = {"synth_dev": ("data/synth_dev", 60, 4242, None), "clean": ("data/synth_clean_dev", 30, 4243, None),
          "real": ("data/codebrim", 100, None, "evals/codebrim_dev100.txt")}  # (folder, n, seed, list file)
REAL_01 = harness.REPO / "public" / "samples" / "real_01.jpg"


def views(img, trace, blur_regions=()) -> list[dict]:
    """Every look in a job's trace (harness.run_drishyam's) rendered again as the job's quality gate read it, scored by
    each gate as it is and blurred by SIGMA view px: [as is, blurred] sharpness, re-blur ratio and each gate's reason
    (None: passed), and whether its L0 window overlaps one of `blur_regions` (synthetic blur_region distractors)."""
    pyr, out = imaging.build_pyramid(img), []
    for t in trace:
        if t["action"] == "halt":
            continue
        size = t["window"][2] >> t["level"]  # the window spans size x 2^level L0 px
        crop = imaging.gaze(pyr, t["x"], t["y"], t["level"], size=size)[0]
        q = {g: [imaging.quality(c, g) for c in (crop, cv2.GaussianBlur(crop, (0, 0), SIGMA))] for g in GATES}
        assert q["laplacian"][0]["sharpness"] == t["sharpness"], (t["step"], q["laplacian"][0], t["sharpness"])
        out.append({"step": t["step"], "level": t["level"], "size": size, "cause": t["cause"],
                    "sharpness": [x["sharpness"] for x in q["laplacian"]], "reblur": [x["reblur"] for x in q["both"]],
                    "reason": {g: [x["reason"] for x in q[g]] for g in GATES},
                    "blur_region": any(imaging.box_iou(t["window"], b) > 0 for b in blur_regions)})
    return out


def photo_task(rec: dict, rows: list[dict], gates=GATES) -> dict:
    """One photo's jobs, one per gate (harness.run_drishyam: governor overrides off, fixed costs), and the views of the
    Laplacian job (`views`)."""
    img = cv2.imread(rec["path"], cv2.IMREAD_COLOR)
    costs = harness.op_costs(rows, img.shape[1], img.shape[0])
    configs = {f"{g}@{CAP}": harness.run_drishyam(img, CAP, costs, blur_gate=g) for g in gates}
    regions = [d["box"] for d in rec.get("distractors", ()) if d["kind"] == "blur_region"]
    return {"id": rec["id"], "width": img.shape[1], "height": img.shape[0], "configs": configs,
            "views": views(img, configs[f"laplacian@{CAP}"]["trace"], regions)}


def counts(photos: list[dict]) -> np.ndarray:
    """Per photo: views, then the views each gate (GATES order) refuses as they are, then blurred."""
    return np.array([[len(p["views"]), *(sum(v["reason"][g][k] is not None for v in p["views"])
                                         for k in (0, 1) for g in GATES)] for p in photos]).reshape(-1, 5)


def gate_row(c: np.ndarray) -> dict:
    """The gates' refusals over these photos' views, and the both-gates gate's share of views refused minus the
    Laplacian gate's, as they are and blurred, each with its paired 95 % bootstrap CI over photos (harness.boot_ci)."""
    def diff(new, old):
        return lambda i: (c[i, new].sum() - c[i, old].sum()) / c[i, 0].sum() if c[i, 0].sum() else np.nan

    whole = np.arange(len(c))
    return {"photos": len(c), "views": int(c[:, 0].sum()),
            "refused_as_is": dict(zip(GATES, map(int, c[:, 1:3].sum(0)), strict=True)),
            "refused_blurred": dict(zip(GATES, map(int, c[:, 3:5].sum(0)), strict=True)),
            "as_is_diff": harness._r(diff(2, 1)(whole)), "as_is_diff_ci": harness.boot_ci(diff(2, 1), len(c)),
            "blurred_diff": harness._r(diff(4, 3)(whole)), "blurred_diff_ci": harness.boot_ci(diff(4, 3), len(c))}


def verdict(g: dict) -> str:
    """Gate accuracy by the rule: Harm if the new gate refuses measurably more views as they are, or measurably fewer
    blurred ones; else Gain if measurably fewer as they are; else no measurable difference. (The both-gates gate can
    refuse no more than the Laplacian gate, so for it Harm can only come from the blurred copies.)"""
    (lo, hi), (blo, bhi) = g["as_is_diff_ci"], g["blurred_diff_ci"]
    if None in (lo, hi, blo, bhi):
        return "not computable"
    if lo > 0 or bhi < 0:
        return "Harm"
    return "Gain" if hi < 0 else "no measurable difference"


def clauses(splits: dict) -> dict:
    """The rule's agent-level clauses, both-gates against Laplacian on the same photos (evals.dev_sweep.row's CIs)."""
    up = lambda s: (splits[s]["agent"]["both"].get("vs_ref_ci") or [None, None])[1]
    clean = splits["clean"]["agent"]["both"]
    return {"recall_holds_synth_dev": up("synth_dev") is not None and up("synth_dev") >= 1,
            "recall_holds_real": up("real") is not None and up("real") >= 1,
            "clean_boxes_do_not_rise": clean["boxes_diff_vs_ref_ci"][0] <= 0,
            "clean_share_with_a_box_does_not_rise": clean["images_with_any_vs_ref_ci"][0] <= 0}


def described(photos: list[dict]) -> dict:
    """Reported, not in the rule: the views by level and over synthetic blur regions, and the re-blur quartiles."""
    v = [x for p in photos for x in p["views"]]
    by = lambda keep: {g: [sum(x["reason"][g][k] is not None for x in v if keep(x)) for k in (0, 1)]
                       for g in GATES} | {"views": sum(map(keep, v))}
    return {"refused_by_level": {f"L{lv}": by(lambda x, lv=lv: x["level"] == lv)
                                 for lv in sorted({x["level"] for x in v})},
            "over_blur_regions": by(lambda x: x["blur_region"]),
            "reblur_quartiles": [[round(float(q), 4) for q in np.percentile([x["reblur"][k] for x in v], [25, 50, 75])]
                                 for k in (0, 1)] if v else None,
            "photos_where_the_jobs_differ": sum(p["configs"][f"laplacian@{CAP}"]["trace"]
                                                != p["configs"][f"both@{CAP}"]["trace"] for p in photos
                                                if f"both@{CAP}" in p["configs"])}


def split_records(d, n, seed, listing) -> list[dict]:
    d = harness.REPO / d
    if listing is None and not (d / "annotations.json").exists():  # load_records would build one
        raise SystemExit(f"{d} is missing: link the dev split first")
    recs = harness.load_records(d, n, seed=seed or 7, list_file=listing and harness.REPO / listing)[0]
    assert len(recs) == n, (str(d), len(recs))  # never a smaller set left behind by a smoke run
    return recs


def run_dev(splits: dict, workers: int, expect: dict | None = None, log=print) -> dict:
    """The rule's numbers on `splits` ({name: (folder, n, seed, list file)}): per split its gate row, both gates'
    agent rows (evals.dev_sweep.row, the Laplacian one the reference) and `described`; the gate row over every photo
    ("pooled"). `expect`: per split the dataset sha256s it must match."""
    rows = harness.cost_rows(MICRO_ARM, default_segmenter().name)
    out, every = {}, []
    for name, spec in splits.items():
        recs = split_records(*spec)
        sha = harness.dataset_sha256(recs)
        if expect and sha != expect[name]:
            raise SystemExit(f"{name}: {sha} is not the split v1's dev rows scored ({expect[name]})")
        photos = harness.pool_map(partial(photo_task, rows=rows), recs, workers, log, name)
        for p in photos:
            for g in GATES:
                c = p["configs"][f"{g}@{CAP}"]
                assert (c["switches"]["blur_gate"], c["profile"]) == (g, ENGINE), (p["id"], g, c["switches"])
        scored = {f"{g}@{CAP}": harness.score_config(f"{g}@{CAP}", recs, photos, "preds") for g in GATES}
        out[name] = {"n": len(recs), **sha, "gate": gate_row(counts(photos)),
                     "agent": {g: dev_sweep.row(f"{g}@{CAP}", recs, photos, scored, f"{GATES[0]}@{CAP}") for g in GATES},
                     "described": described(photos)}
        every += [{"split": name, "id": p["id"], "views": p["views"]} for p in photos]
    return {"splits": out, "pooled": gate_row(counts(every)), "views": every}


def fallback_look(path, rows) -> dict:
    """Reported, not in the rule: a photo with no candidate (real_01) as `drishyam run` loads it, under each gate: its
    whole-photo fallback-l2 look's gate result, the job's looks and those refused as blur, and the boxes it drafted
    (per-component, by class) and the critical ones among them."""
    img, _ = batch.load(path)
    costs, out = harness.op_costs(rows, img.shape[1], img.shape[0]), {}
    for g in GATES:
        r = harness.run_drishyam(img, CAP, costs, blur_gate=g)
        (t,) = [t for t in r["trace"] if t["cause"] == "fallback-l2"]
        (v,) = [v for v in views(img, r["trace"]) if v["cause"] == "fallback-l2"]
        out[g] = {"level": t["level"], "size": v["size"], "sharpness": t["sharpness"], "reblur": v["reblur"][0],
                  "ok": t["q_ok"], "reason": t["q_reason"],
                  "looks": [[x["level"], x["cause"], x["q_reason"]] for x in r["trace"] if x["action"] != "halt"],
                  "looks_refused_blur": sum(x["q_reason"] == "blur" for x in r["trace"]), "boxes": len(r["preds"]),
                  "classes": dict(collections.Counter(p["cls"] for p in r["preds"])), "signoffs": r["signoffs"],
                  "needs_look": r["needs_look"], "status": r["status"], "halt": r["halt_cause"]}
    return out


def pool_records() -> list[dict]:
    """The calibration pool (POOL_SYNTH, built under data/blur_pool on first use, and POOL_CODEBRIM CODEBRIM photos
    drawn with POOL_SEED from those in neither evals/codebrim_dev100.txt nor evals/codebrim_eval200.txt)."""
    recs = []
    for name, n, seed, clean in POOL_SYNTH:
        if not (POOL_DIR / name / "annotations.json").exists():
            synth.build_dataset(POOL_DIR / name, n, seed, clean=clean)
        recs += [{**r, "id": f"{name}/{r['id']}"} for r in split_records(POOL_DIR / name, n, seed, None)]
    used = set().union(*((harness.REPO / "evals" / f).read_text().split()
                         for f in ("codebrim_dev100.txt", "codebrim_eval200.txt")))
    real = codebrim.select_dev(codebrim.load(harness.REPO / "data" / "codebrim"), used, POOL_CODEBRIM, POOL_SEED)
    assert len(real) == POOL_CODEBRIM and not used & {Path(r["path"]).name for r in real}
    return recs + real


def threshold(photos: list[dict]) -> dict:
    """REBLUR_MAX by the calibration procedure, and both gates' refusals of the pool's views at it. The both-gates gate
    refuses a view at t if it is glare or dark, or if the Laplacian gate calls it blur and its re-blur ratio is above t."""
    v = [x for p in photos for x in p["views"]]
    other = [[x["reason"]["both"][k] in ("glare", "dark") for x in v] for k in (0, 1)]  # refused before blur
    lap = [[x["reason"]["laplacian"][k] == "blur" for x in v] for k in (0, 1)]
    rb = np.array([x["reblur"] for x in v]).reshape(-1, 2)
    new = lambda k, t: int(sum(o or (b and r > t) for o, b, r in zip(other[k], lap[k], rb[:, k], strict=True)))
    old = [sum(x["reason"]["laplacian"][k] is not None for x in v) for k in (0, 1)]
    t = max(round(i * STEP, 3) for i in range(round(1 / STEP) + 1) if new(1, round(i * STEP, 3)) >= old[1])
    return {"views": len(v), "reblur_max": t,
            "refused_as_is": {"laplacian": old[0], "both": new(0, t)},
            "refused_blurred": {"laplacian": old[1], "both": new(1, t)},
            "reblur_quartiles": [[round(float(q), 4) for q in np.percentile(rb[:, k], [25, 50, 75])] for k in (0, 1)],
            "views_by_level": {f"L{lv}": sum(x["level"] == lv for x in v) for lv in sorted({x["level"] for x in v})}}


def calibrate(workers: int) -> dict:
    recs = pool_records()
    rows = harness.cost_rows(MICRO_ARM, default_segmenter().name)
    photos = harness.pool_map(partial(photo_task, rows=rows, gates=GATES[:1]), recs, workers,
                              lambda m: print(m, flush=True), "pool")
    groups = {"synthetic": [p for p in photos if "/" in p["id"]], "codebrim": [p for p in photos if "/" not in p["id"]]}
    out = {"pool": {"photos": len(photos), **{k: len(g) for k, g in groups.items()}, **harness.dataset_sha256(recs)},
           "sigma": SIGMA, "step": STEP, **threshold(photos),
           "by_source": {k: {x: y for x, y in threshold(g).items() if x != "reblur_max"} for k, g in groups.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration.json").write_text(json.dumps(
        {**out, "views": [{"id": p["id"], "views": p["views"]} for p in photos]}, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evals.blur_gate")
    ap.add_argument("command", choices=("calibrate", "dev", "real01"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    if default_segmenter().name != ENGINE:
        raise SystemExit(f"set DRISHYAM_ONNX=models/drishyam_dacl10k.onnx: the rule's engine is {ENGINE}")
    if a.command == "calibrate":
        calibrate(a.workers)
        return
    doc, rel = json.loads(RULE_FILE.read_text()), str(RULE_FILE.relative_to(harness.REPO))
    if git("status", "--porcelain", "--", rel):
        raise SystemExit(f"{rel} is not committed as it stands")
    if a.command == "real01":  # reported, never read by the rule: only once its verdict is committed
        if "verdict" not in doc or "real_01_fallback_look" in doc:
            raise SystemExit(f"{rel}: real_01 is reported once, after the dev run's verdict")
        doc["real_01_fallback_look"] = fallback_look(REAL_01, harness.cost_rows(MICRO_ARM, default_segmenter().name))
        RULE_FILE.write_text(json.dumps(doc, indent=1, default=float) + "\n")
        print(json.dumps(doc["real_01_fallback_look"], indent=1))
        return
    if "verdict" in doc:
        raise SystemExit(f"{rel} already holds outputs: the rule is run once")
    commit, committed = git("log", "-1", "--format=%H %cI", "--", rel).split()
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    assert datetime.fromisoformat(committed) < datetime.fromisoformat(started), (committed, started)
    if doc.get("arms") != list(GATES) or imaging.REBLUR_MAX != doc["calibration"]["reblur_max"]:
        raise SystemExit(f"the rule's arms {doc.get('arms')} and REBLUR_MAX {doc['calibration']['reblur_max']} are not "
                         f"the code's {list(GATES)} and {imaging.REBLUR_MAX}")
    pub = json.loads((harness.REPO / "results" / "dev" / "model_choice_dev.json").read_text())["datasets"]
    expect = {s: {"dataset_gt_sha256": d["gt_sha256"], "dataset_images_sha256": d["images_sha256"]}
              for s, d in pub.items()}
    code = {"commit": git("rev-parse", "HEAD"), "tracked_changes": git("status", "--porcelain", "--untracked-files=no")}
    res = run_dev(SPLITS, a.workers, expect, lambda m: print(m, flush=True))
    word, cl = verdict(res["pooled"]), clauses(res["splits"])
    adopted = word == "Gain" and all(cl.values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev_views.json.gz").write_bytes(gzip.compress(json.dumps(res["views"]).encode()))
    doc.update({"run": {"rule_commit": commit, "rule_committed": committed, "run_started": started, "code": code,
                        "segmenter": harness.segmenter_id(default_segmenter()), "micro_arm": MICRO_ARM,
                        "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "views_file": str((OUT / "dev_views.json.gz").relative_to(harness.REPO))},
                "gate_pooled": res["pooled"], "splits": res["splits"], "clauses": cl,
                "verdict": word, "adopted": adopted})
    RULE_FILE.write_text(json.dumps(doc, indent=1, default=float) + "\n")
    print(json.dumps({k: doc[k] for k in ("gate_pooled", "clauses", "verdict", "adopted")}, indent=1, default=float))


if __name__ == "__main__":
    main()
