"""v1-policy2 on the dev splits: its HARD subset, its metrics, its pre-listed grid and the rule that picks from it.

    python -m evals.policy2 spec all|eligible     # the grid as evals.dev_sweep variants (runs/policy2/sweep.sh)
    python -m evals.policy2 build                 # apply the rule to runs/policy2/ -> results/dev/policy2_sweep.json

The rule, the grid and the commands were committed in results/dev/policy2_sweep.json before any run of the grid;
`build` applies that rule as written and fills in the rows, the hard-subset rows and the verdict.

A photo is HARD when either holds:
  small-critical  a ground-truth box of a critical class (exposed_rebar or spalling, the classes findings.is_critical
                  can make critical) covers under HARD_AREA_FRAC (1 %) of the photo's pixels: the defect a person must
                  see is small at the scale the agent first looks at;
  no-candidate    the overview proposes no candidate region (drishyam.imaging.find_candidates on the L3 saliency map,
                  the agent's own call): the zero-candidate case the fallback exists for.
Both read only the labels and the image itself (its saliency, a fixed OpenCV function), never any policy's output, so
the same photos are hard for every policy compared on them. The 1 % cut was set from the dev splits' label sizes only
(the 50th percentile of a critical-class box is 1.3 % of a synth_dev photo and 3.3 % of a CODEBRIM dev100 one): 21 of
60 synth_dev photos and 20 of 100 dev100 photos hold such a box. Large cracks are not a critical class here
(findings.is_critical never makes a crack critical), so they do not make a photo hard.
"""
from __future__ import annotations

import re
import subprocess
from datetime import datetime

import cv2
import numpy as np

from drishyam import imaging
from drishyam.findings import is_critical
from evals import harness

CRITICAL_CLASSES = ("exposed_rebar", "spalling")
HARD_AREA_FRAC = 0.01
TIERS = ("crit", "signoffs", "reviews", "look", "needs_look", "all", "crit_merged")  # per-photo counts of `fields`


def small_critical(rec: dict) -> list[dict]:
    """The photo's ground-truth boxes of a critical class under HARD_AREA_FRAC of its pixels."""
    area = rec["width"] * rec["height"]
    return [g for g in rec["boxes"] if g["cls"] in CRITICAL_CLASSES and g["box"][2] * g["box"][3] < HARD_AREA_FRAC * area]


def no_candidate(img) -> bool:
    """The overview proposes nothing: the same saliency and proposal call as Job.run (the window size only splits a
    salient blob into windows, so it cannot change whether there is one)."""
    pyr = imaging.build_pyramid(img)
    return not imaging.find_candidates(imaging.saliency(pyr[imaging.OVERVIEW_LEVEL]), imaging.OVERVIEW_LEVEL, 1024,
                                       img.shape[1::-1])


def hard_reasons(rec: dict, img=None) -> list[str]:
    """Why a photo is hard (empty: not hard). `img`: the decoded photo, else read from rec['path']."""
    img = cv2.imread(rec["path"], cv2.IMREAD_COLOR) if img is None else img
    return ["small-critical"] * bool(small_critical(rec)) + ["no-candidate"] * no_candidate(img)


def hard_subset(records: list[dict]) -> dict[str, list[str]]:
    """{photo id: reasons} for the hard photos of `records`, in record order."""
    return {r["id"]: why for r in records if (why := hard_reasons(r))}


# ---------------------------------------------------------------- the metrics the policy2 rule reads
def crit_cols(m: dict, records: list[dict]) -> np.ndarray:
    """Per photo [critical-class ground-truth boxes found, critical-class ground-truth boxes] from a scored config's
    flags (evals.harness.score_config): exposed_rebar and spalling boxes of any size."""
    return np.array([[sum(f for f, g in zip(m["flags"][r["id"]], r["boxes"], strict=True) if g["cls"] in CRITICAL_CLASSES),
                      sum(g["cls"] in CRITICAL_CLASSES for g in r["boxes"])] for r in records], float).reshape(-1, 2)


def pooled(cols: np.ndarray):
    """stat(idx) for harness.boot_ci: the pooled recall of [found, total] rows on the resampled photos."""
    return lambda i: cols[i, 0].sum() / cols[i, 1].sum() if cols[i, 1].sum() else np.nan


def ratio_of(a: np.ndarray, b: np.ndarray, n: int) -> tuple:
    """(pooled recall of a / of b, its paired 95 % bootstrap CI), photos resampled together."""
    pa, pb = pooled(a), pooled(b)
    whole = np.arange(n)
    v = pa(whole) / pb(whole) if pb(whole) else None
    return v, harness.boot_ci(lambda i: pa(i) / pb(i) if pb(i) else np.nan, n)


def tiers(cs: list[dict]) -> dict:
    """Per photo: critical boxes in `preds` (what goes to sign-off, per component), critical findings (sign-offs),
    review requests, needs-a-look boxes and entries, every box a person is shown (`preds_all`), and the critical boxes
    merged as the tiling baseline's are (evals.harness.merge_boxes), a reporting convention only."""
    crit = lambda c: [p for p in c["preds"] if is_critical(p["cls"], p["area"])]
    return {"crit": np.array([len(crit(c)) for c in cs], float), "signoffs": np.array([c["signoffs"] for c in cs], float),
            "reviews": np.array([c["reviews"] for c in cs], float), "look": np.array([len(c["preds_look"]) for c in cs], float),
            "needs_look": np.array([c["needs_look"] for c in cs], float),
            "all": np.array([len(c["preds_all"]) for c in cs], float),
            "crit_merged": np.array([len(harness.merge_boxes(crit(c))) for c in cs], float)}


def fields(key: str, records: list[dict], results: list[dict], scored: dict, ref: str | None = None) -> dict:
    """A dev row's critical-class recall (any config; against L2 and L1 tiling and the downscale when they ran) and, for a
    Drishyam config, v1-policy2's two tiers: recall of everything a person is shown (`preds_all`) and the per-photo
    human-gate counts (`tiers`), each against `ref` with paired 95 % bootstrap CIs (ratios for recall, per-photo
    differences for counts)."""
    n, m, cs = len(records), scored[key], [r["configs"][key] for r in results]
    crit = crit_cols(m, records)
    whole = np.arange(n)
    out = {"crit_recall": None if np.isnan(v := pooled(crit)(whole)) else float(v),
           "crit_recall_ci": harness.boot_ci(pooled(crit), n),
           "per_class_gt": {c: int(v) for c, v in m["gt_per_class"].items()}}
    for name, k in (("tiling_L2", "tiling@L2"), ("tiling_L1", "tiling@L1"), ("downscale", "downscale"), ("ref", ref)):
        if k in scored and k != key and crit[:, 1].sum():
            out[f"crit_recall_vs_{name}"], out[f"crit_recall_vs_{name}_ci"] = ratio_of(crit, crit_cols(scored[k], records), n)
    if "preds_all" not in cs[0]:
        return out
    ma, x = harness.score_config(key, records, results, "preds_all"), tiers(cs)
    crit_all = crit_cols(ma, records)
    out |= {"recall_all": ma["recall"], "recall_all_ci": harness.boot_ci(harness.recall_of(ma), n),
            "crit_recall_all": None if np.isnan(v := pooled(crit_all)(whole)) else float(v),
            "crit_recall_all_ci": harness.boot_ci(pooled(crit_all), n),
            **{f"{k}_per_image": float(v.mean()) for k, v in x.items()},
            "images_with_any_all": float((x["all"] > 0).mean())}
    if ref in scored and ref != key and "preds_all" in results[0]["configs"][ref]:
        mra, xr = harness.score_config(ref, records, results, "preds_all"), tiers([r["configs"][ref] for r in results])
        if ma["recall"] is not None and mra["recall"]:
            out |= {"recall_all_vs_ref": ma["recall"] / mra["recall"], "recall_all_vs_ref_ci": harness.ratio_ci(ma, mra, n)}
        if crit[:, 1].sum():
            out["crit_recall_all_vs_ref"], out["crit_recall_all_vs_ref_ci"] = ratio_of(crit_all, crit_cols(mra, records), n)
        for k in TIERS:
            out[f"{k}_diff_vs_ref_ci"] = harness.boot_ci(lambda i, a=x[k], b=xr[k]: a[i].mean() - b[i].mean(), n)
        out["images_with_any_all_vs_ref_ci"] = harness.boot_ci(
            lambda i: (x["all"][i] > 0).mean() - (xr["all"][i] > 0).mean(), n)
    return out


# ---------------------------------------------------------------- the pre-listed grid and the rule
CAP, MATCHED_CAP = 0.3, 0.1  # the rule's cap (JobConfig's default); about L2 tiling's pixels on CODEBRIM dev100 (0.102)
BASE = {"crit_weight": 2.0, "merge_critical": True}
ELIGIBLE = {f"{'g' if grid else 'n'}_{c[0]}{round(f * 100)}": {**BASE, "grid": grid, "crit_confirm": c, "crit_floor": f}
            for grid in (True, False) for c in ("shift", "finer") for f in (0.6, 0.7)}
REF = "g_s60"  # the ablations switch one piece of this one off
ABLATIONS = {"g_s60_nomerge": {**ELIGIBLE[REF], "merge_critical": False},
             "g_s60_w1": {**ELIGIBLE[REF], "crit_weight": 1.0},
             "g_60_noconfirm": {**ELIGIBLE[REF], "crit_confirm": "off"},
             "g_nofloor": {**ELIGIBLE[REF], "crit_floor": 0.0, "crit_confirm": "off"},
             "grid_only": {"grid": True}}
SPLITS = ("synth_dev", "clean", "real")
RUNS = harness.REPO / "runs" / "policy2"
OUT = harness.REPO / "results" / "dev" / "policy2_sweep.json"


def spec(which: str = "all") -> str:
    """The grid as evals.dev_sweep variants: cur (v1 as it is, listed first: every CI is against it), then the
    eligible configurations, then (all) the ablations."""
    grid = {**ELIGIBLE, **(ABLATIONS if which == "all" else {})}
    return ";".join(["cur=", *(f"{k}=" + ",".join(f"{a}:{v}" for a, v in sw.items()) for k, sw in grid.items())])


def clauses(rows: dict, k: str) -> dict:
    """The rule's clauses for configuration k at CAP (results/dev/policy2_sweep.json `rule`), each True or False. A CI
    that could not be computed ([None, None]) fails its clause."""
    s, r, c = (rows[x][f"{k}@{CAP}"] for x in ("synth_dev", "real", "clean"))
    ok = lambda ci: None not in ci
    not_below_1 = lambda ci: ok(ci) and ci[1] >= 1  # a ratio: not measurably lower
    not_above_0 = lambda ci: ok(ci) and ci[0] <= 0  # a difference: not measurably higher
    below_0 = lambda ci: ok(ci) and ci[1] < 0  # a difference: measurably lower
    return {"recall_shown_holds_synth_dev": not_below_1(s["recall_all_vs_ref_ci"]),
            "recall_shown_holds_real": not_below_1(r["recall_all_vs_ref_ci"]),
            "critical_recall_holds_synth_dev": not_below_1(s["crit_recall_vs_ref_ci"]),
            "critical_recall_holds_real": not_below_1(r["crit_recall_vs_ref_ci"]),
            "critical_boxes_fall_real": below_0(r["crit_diff_vs_ref_ci"]),
            "critical_boxes_do_not_rise_synth_dev": not_above_0(s["crit_diff_vs_ref_ci"]),
            "signoffs_do_not_rise_synth_dev": not_above_0(s["signoffs_diff_vs_ref_ci"]),
            "signoffs_do_not_rise_real": not_above_0(r["signoffs_diff_vs_ref_ci"]),
            "clean_boxes_do_not_rise": not_above_0(c["all_diff_vs_ref_ci"]),
            "clean_share_with_a_box_does_not_rise": not_above_0(c["images_with_any_all_vs_ref_ci"])}


def _load(split: str, cap_tag: str = ""):
    """A run's rows (evals.dev_sweep --out) and its dump's results, with the split's records."""
    import gzip
    import json

    from evals import dev_sweep

    rows = json.loads((RUNS / f"{split}{cap_tag}.json").read_text())
    dump = json.loads(gzip.decompress((RUNS / f"{split}{cap_tag}.dump.json.gz").read_bytes()))
    if split == "real":
        recs, _ = harness.load_records(harness.REPO / "data" / "codebrim", 100,
                                       list_file=harness.REPO / "evals" / "codebrim_dev100.txt")
    else:
        d, n, seed = dev_sweep.SPLITS[split]
        recs, _ = harness.load_records(harness.REPO / d, n, seed=seed)
    if dump["ids"] != [r["id"] for r in recs] or {k: rows[k] for k in ("dataset_gt_sha256", "dataset_images_sha256")} != \
            harness.dataset_sha256(recs):
        raise SystemExit(f"runs/policy2/{split}{cap_tag}: other photos or ground truth than the dev split")
    return rows, dump["results"], recs


def subset_rows(recs, results, keep: list[int], keys: list[str], first: str) -> dict:
    """Rows (evals.dev_sweep.row, against `first` at its cap) on the photos `keep` only, plus the recall of the small
    critical-class boxes that made them hard."""
    from evals import dev_sweep

    sr, sx = [recs[i] for i in keep], [results[i] for i in keep]
    scored = {k: harness.score_config(k, sr, sx, "preds") for k in keys}
    out = {}
    for k in keys:
        ref = f"{first}@{k.rpartition('@')[2]}" if "@" in k else None
        row = {a: b for a, b in dev_sweep.row(k, sr, sx, scored, ref).items() if a != "per_image"}
        small = [(g, x["configs"][k]) for r, x in zip(sr, sx, strict=True) for g in small_critical(r)]
        row["small_critical_recall"] = round(sum(harness.found(g, c["preds"]) for g, c in small) / len(small), 4) if small else None
        row["small_critical_recall_shown"] = (round(sum(harness.found(g, c.get("preds_all", c["preds"])) for g, c in small)
                                              / len(small), 4) if small else None)
        out[k] = row
    return out


def build() -> dict:
    """Apply the stated rule to the dev runs in runs/policy2/ and add the rows, the hard-subset rows, the matched-budget
    rows and the verdict to results/dev/policy2_sweep.json, whose stated fields it never changes."""
    import collections
    import json

    stub = json.loads(OUT.read_text())
    stated = {k: stub[k] for k in ("rule", "variants", "choices", "splits", "commands", "costs")}
    assert stub["variants"]["spec_all"] == spec("all")  # the grid that ran is the grid stated
    rows, crit, px, looks, hard = {}, {}, {}, {}, {}
    for split in SPLITS:
        r, results, recs = _load(split)
        rows[split] = {k: {a: b for a, b in v.items() if a != "per_image"} for k, v in r["rows"].items()}
        keys = list(results[0]["configs"])
        scored = {k: harness.score_config(k, recs, results, "preds") for k in keys}
        crit[split] = {k: crit_cols(scored[k], recs).sum(0).tolist() for k in keys}  # [found, total] critical-class GT
        px[split] = {k: rows[split][k]["pixels_frac"] for k in keys}
        looks[split] = {k: dict(collections.Counter(f"{t['action']}:{t['cause']} L{t['level']}" for x in results
                                                    for t in x["configs"][k]["trace"] if t["action"] != "halt"))
                        for k in keys if "trace" in results[0]["configs"][k]}
        keep = [i for i, rec in enumerate(recs) if hard_reasons(rec)]
        hard[split] = {"photos": len(keep), "of": len(recs), "indices": keep,
                       "rows": subset_rows(recs, results, keep, keys, "cur") if keep else {}}
    ver = {k: clauses(rows, k) for k in ELIGIBLE}
    passed = [k for k in ELIGIBLE if all(ver[k].values())]
    found = {k: crit["synth_dev"][f"{k}@{CAP}"][0] + crit["real"][f"{k}@{CAP}"][0] for k in [*ELIGIBLE, "cur"]}
    total = crit["synth_dev"][f"cur@{CAP}"][1] + crit["real"][f"cur@{CAP}"][1]
    order = sorted(passed, key=lambda k: (-found[k], (px["synth_dev"][f"{k}@{CAP}"] + px["real"][f"{k}@{CAP}"]) / 2,
                                          rows["synth_dev"][f"{k}@{CAP}"]["critical_per_image"]
                                          + rows["real"][f"{k}@{CAP}"]["critical_per_image"], list(ELIGIBLE).index(k)))
    chosen = order[0] if order else None
    m_rows, m_res, m_recs = _load("real", f"_{MATCHED_CAP}")
    shown = {k: harness.score_config(k, m_recs, m_res, "preds_all" if "preds_all" in m_res[0]["configs"][k] else "preds")
             for k in m_res[0]["configs"]}
    matched = {k: {a: b for a, b in v.items() if a != "per_image"} for k, v in m_rows["rows"].items()}
    for k in matched:
        if k != "tiling@L2":
            t = shown["tiling@L2"]
            matched[k] |= {"recall_shown": shown[k]["recall"], "recall_shown_vs_tiling_L2": shown[k]["recall"] / t["recall"],
                           "recall_shown_vs_tiling_L2_ci": harness.ratio_ci(shown[k], t, len(m_recs)),
                           "pixels_vs_tiling_L2": matched[k]["pixels_frac"] / matched["tiling@L2"]["pixels_frac"]}
    share = lambda c, pick: round(sum(v for a, v in c.items() if pick(a)) / max(1, sum(c.values())), 4)
    rates = {s: {k: {"escalation_rate": share(c, lambda a: a.startswith("escalate:")),
                     "verify_rate": share(c, lambda a: a.startswith("verify:")),
                     "confirm_share": share(c, lambda a: ":confirm-critical " in a),
                     "confirm_looks_per_photo": round(sum(v for a, v in c.items() if ":confirm-critical " in a)
                                                      / sum(rows[s][k]["halt"].values()), 4)}
                 for k, c in looks[s].items()} for s in SPLITS}
    fb = json.loads((OUT.parent / "zero_candidate_fallback.json").read_text())["rows"]
    same_as_fb = {s: {f: rows[s][f"cur@{CAP}"][f] == fb[s]["fb"][f] for f in ("recall", "boxes_per_image", "ms_per_job",
                                                                             "critical_per_image", "pixels_frac")}
                  for s in SPLITS}
    log = (RUNS / "log.txt").read_text()
    rule_commit = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H %cI", "--", "results/dev/policy2_sweep.json"],
                                 cwd=harness.REPO, capture_output=True, text=True, check=True).stdout.split()  # the stub's commit
    if (first := re.search(r"^(\S+) start \|", log, re.MULTILINE)) is None:
        raise SystemExit("runs/policy2/log.txt has no start")
    ran_after = datetime.fromisoformat(rule_commit[1]) < datetime.fromisoformat(first.group(1))
    heads = set(re.findall(r"HEAD (\w+) \| tracked changes (\d+)", log))
    assert ran_after and len(heads) == 1 and next(iter(heads))[1] == "0", (rule_commit, first.group(1), heads)
    out = {**stub, "rows": rows, "critical_class_found": crit, "looks_by_action_cause_level": looks, "hard": hard,
           "rates": rates, "cur_reproduces_v1_fallback_in_zero_candidate_fallback": same_as_fb,
           "run": {"code_commit": next(iter(heads))[0], "rule_commit": rule_commit[0][:7], "rule_committed": rule_commit[1],
                   "first_output": first.group(1), "order_checked": "the stub's last commit time is before the run log's "
                   "start, and every run ran one HEAD with no tracked change (asserted)", "log": log.splitlines(),
                   "workers": 6, "files": {s: f"runs/policy2/{s}.json" for s in (*SPLITS, "real_0.1")}},
           "matched_budget": {"cap": MATCHED_CAP, "split": "real", "rows": matched},
           "verdict": {"clauses": ver, "passed": passed, "critical_class_found_both_splits": found,
                       "critical_class_ground_truth_both_splits": total, "ranked": order, "chosen": chosen,
                       "adopted": chosen is not None}}
    assert {k: out[k] for k in stated} == stated  # the stated fields are never changed
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    return out


if __name__ == "__main__":
    import json
    import sys

    if sys.argv[1:2] == ["spec"]:
        print(spec(sys.argv[2] if len(sys.argv) > 2 else "all"))
    elif sys.argv[1:2] == ["build"]:
        print(json.dumps(build()["verdict"], indent=1))
    else:
        raise SystemExit(__doc__)
