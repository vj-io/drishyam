"""The held-out evaluation of the trained models, assembled into results/heldout_models.json from the runs it re-scores.

    python -m evals.heldout [--out results/heldout_models.json]

Each model and the classical engine were run once per held-out set (runs/heldout/run_model.sh): CODEBRIM eval200 and
synth200 by `evals.model_eval --costs`, the clean set data/synth_clean and synth_large by evals.harness, each charged
the cost table the dev comparison charged it (results/dev/model_choice_dev.json). This reads every run's per-image
file (runs/model-eval/<model>-<set>/eval_per_image.json), checks it scored the right model on the right data and table,
and re-scores it with the harness's own scoring: each row as evals.dev_sweep reads a dev row, every pair of models on
the same photos with paired 95 % bootstrap CIs, and the verdicts by the dev comparison's criterion, unchanged.
Nothing is tuned here.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np

from drishyam.agent import POLICY2, POLICY2_OFF
from drishyam.findings import is_critical
from evals import dev_sweep, harness, policy2

REPO = harness.REPO
RUNS = REPO / "runs" / "model-eval"
MODELS = {"v1": ("onnx", "dacl10k"), "v2a": ("onnx-v2a", "dacl10k_v2a"), "v2b": ("onnx-v2b", "dacl10k_v2b"),
          "v2c": ("onnx-v2c", "dacl10k_v2c"), "classical-v1": ("classical", "dacl10k")}  # run folder prefix; the
# run_model.sh call that ran it
SETS = {"codebrim": ("data/codebrim", 200, 7), "synth": ("data/synth200", 200, 7), "clean": ("data/synth_clean", 51, 11),
        "large": ("data/synth_large", 100, 7)}
DEV_SETS = ("codebrim", "synth", "clean")  # held-out counterparts of the dev splits (real, synth_dev, clean) the choice read
KEYS, CAPS = ("downscale", "tiling@L0", "drishyam@0.3", "drishyam@1.0"), ("drishyam@0.3", "drishyam@1.0")
COMMANDS = [("runs/heldout/run_model.sh dacl10k local-macos-x86-dacl10k-retime (v1; also the classical clean and "
             "synth_large runs and v1 on synth_large)"), "runs/heldout/run_model.sh dacl10k_v2a local-macos-x86-dacl10k_v2a",
            "runs/heldout/run_model.sh dacl10k_v2b local-macos-x86-dacl10k_v2b",
            ("runs/heldout/run_model.sh dacl10k_v2c local-macos-x86-dacl10k_v2c (later, after v2c's dev calibration, from "
             "another worktree with the other models' run folders copied in unchanged)"),
            ("each: python -m evals.model_eval --publish --onnx models/drishyam_<tag>.onnx --costs <arm> --dacl10k <kagglehub "
            "dacl10k folder> --codebrim data/codebrim --workers 6, then python -m evals.harness --cost-model fixed --workers 6 "
            "--micro-arm <arm> --data data/synth_clean --n 51 --seed 11 --chaos 0,0,0 (as make eval-clean) with "
            "DRISHYAM_ONNX set; synth_large as make eval-large (--n 100 --seed 7 --size 4608x3456)")]
PRIOR = ("v1 had been scored on two of these sets before, once each, on 2026-10-18 and never published: CODEBRIM eval200 "
         "(run started 14:07Z) and synth200 (00:52Z), with the agent of that day, before its dev profile, on its first cost "
         "table (results/local-macos-x86-dacl10k). The runs are in the main checkout's ignored runs/model-eval/. So for v1 "
         "these are second scorings; v2a and v2b had never been scored on any held-out set (no other run under runs/ or a "
         "worktree carries these sets' sha256s with an ONNX segmenter). Nor had v2c before its run here: its only "
         "earlier jobs were the install's smoke jobs on three of the demo's bridge photos (the main checkout's "
         "runs/model-smoke/), and no run folder under runs/ or a worktree names it.")
NAMES = {"codebrim": "CODEBRIM eval200", "synth": "synth200", "large": "synth_large, 4608×3456", "clean": "clean"}
# A configuration of one model, scored beside it after its own dev rule: run folder prefix, run script tag, model.
CONFIGS = {"v1-policy2": ("onnx-p2", "dacl10k+policy2", "v1")}
ALL = {}  # every model, each followed by its configurations: (run folder prefix, run script tag, model)
for _m, (_p, _t) in MODELS.items():
    ALL |= {_m: (_p, _t, _m), **{c: v for c, v in CONFIGS.items() if v[2] == _m}}
POLICY2_COMMAND = ("runs/heldout/run_policy2.sh (v1-policy2, later, after its dev rule chose it): the leakage check, then "
                   "python -m evals.harness --cost-model fixed --workers 6 --micro-arm local-macos-x86-dacl10k-retime with "
                   "DRISHYAM_ONNX set on data/codebrim (--n 200) and data/synth200 (--n 200), as evals.model_eval runs them "
                   "but without --publish, and on data/synth_clean (--n 51 --seed 11 --chaos 0,0,0), with v1's profile "
                   "taking agent.POLICY2 (its default since the dev choice) and the zero-candidate fallback on")
PRIOR_POLICY2 = ("v1-policy2 is v1 with the configuration results/dev/policy2_sweep.json chose on the dev splits (g_s60: "
                 "first looks on the tiling grid, a confirm look at the same level for a critical mask under confidence 0.6, "
                 "which otherwise goes to a person as needs-a-look, critical classes weighted x2 in the look score, "
                 "overlapping critical findings of one class merged) and the zero-candidate fallback on. It is v1's third "
                 "read of CODEBRIM eval200 and synth200 and its second of the clean set: v1 was scored on CODEBRIM eval200 "
                 "and synth200 on 2026-10-18 (unpublished, see prior_exposure) and on all three in its published run here. "
                 "A held-out run of v1 with the fallback alone was started on 2026-10-21 in another worktree and stopped; "
                 "its output was never read or used. This configuration was chosen on dev after those reads, by a rule "
                 "stated before its dev run, and nothing was changed after this read.")
r4 = lambda x: round(float(x), 4) if isinstance(x, (float, np.floating)) else x


def load(prefix: str, s: str, records: list[dict], data_id: dict) -> dict | None:
    """One run's per-image results and summary, checked against the records (and their sha256s) it must have scored."""
    d = RUNS / f"{prefix}-{s}"
    if not (d / "eval_per_image.json").exists():
        return None
    cache, summary = (json.loads((d / f).read_text()) for f in ("eval_per_image.json", "summary.json"))
    if cache["ids"] != [r["id"] for r in records] or cache["dataset"] != data_id:
        raise SystemExit(f"{d} scored other photos or ground truth than {SETS[s][0]}")
    if cache["cost_model"] != "fixed" or cache["governor"] or summary["agent_config"]["governor"]:
        raise SystemExit(f"{d} is not a fixed-cost policy comparison with the governor off")
    return {"results": cache["results"], "summary": summary, "segmenter": cache["segmenter"]}


def per_image(records, results, k) -> dict:
    cs = [x["configs"][k] for x in results]
    return {"box": np.array([len(c["preds"]) for c in cs]), "ms": 1000 * np.array([c["wall_s"] for c in cs]),
            "crit": np.array([harness.critical_counts(c["preds"], rec["boxes"])[0] for c, rec in zip(cs, records, strict=True)])}


def row(k: str, records, results, scored) -> dict:
    """A dev row (evals.dev_sweep.row) plus the CIs the dev comparison added: boxes, photos with any box, critical
    boxes and modelled ms per job, and the ms ratios to the downscale and to L0 tiling."""
    n, x = len(records), per_image(records, results, k)
    r = {a: b for a, b in dev_sweep.row(k, records, results, scored).items() if a != "per_image"}
    mean_ci = lambda a: harness.boot_ci(lambda i: a[i].mean(), n)
    r |= {"boxes_per_image_ci": mean_ci(x["box"]), "images_with_any_ci": mean_ci(x["box"] > 0),
          "critical_per_image_ci": mean_ci(x["crit"]), "ms_per_job_ci": mean_ci(x["ms"])}
    for name, b in (("downscale", "downscale"), ("tiling_L0", "tiling@L0")):
        if b != k:
            ms, mb = x["ms"], per_image(records, results, b)["ms"]
            r |= {f"ms_vs_{name}": ms.mean() / mb.mean(), f"ms_vs_{name}_ci": harness.boot_ci(lambda i, ms=ms, mb=mb: ms[i].mean() / mb[i].mean(), n)}
    return {a: r4(b) for a, b in r.items()}


def compare(ma: dict, mb: dict, xa: dict, xb: dict, n: int) -> dict:
    """A against B on the same photos: [value, paired 95 % CI] of each ratio, and of each difference per photo."""
    ratio = lambda a, b: [r4(a.sum() / b.sum()) if b.sum() else None,
                          harness.boot_ci(lambda i: a[i].sum() / b[i].sum() if b[i].sum() else np.nan, n)]
    diff = lambda a, b: [r4(a.mean() - b.mean()), harness.boot_ci(lambda i: a[i].mean() - b[i].mean(), n)]
    out = {"boxes": ratio(xa["box"], xb["box"]), "critical": ratio(xa["crit"], xb["crit"]), "ms": ratio(xa["ms"], xb["ms"]),
           "boxes_per_photo_diff": diff(xa["box"], xb["box"]), "any_box_share_diff": diff(xa["box"] > 0, xb["box"] > 0)}
    if mb["recall"] is not None:  # the clean set has no ground truth
        out |= {name: [r4(ma[f] / mb[f]) if mb[f] else None, harness.ratio_ci(ma, mb, n, col)]
                for name, f, col in (("recall", "recall", 0), ("iou_only", "recall_iou_only", 1), ("chance", "chance_recall", 2))}
    return out


def side(ci: list, null: float) -> int:
    """+1 when the CI lies wholly above `null`, -1 wholly below, 0 when it touches or spans it."""
    return 0 if ci is None or None in ci else (ci[0] > null) - (ci[1] < null)


def losses(pairs: dict) -> tuple[dict, dict]:
    """Each model's measurable losses, by the dev criterion's standard (a CI that excludes no change): recall and
    IoU-only recall ratios, and on the clean set boxes per photo and the share of photos with a box (all false alarms);
    and, outside the criterion, more critical boxes for the human gate and more modelled ms per job."""
    crit, other = {m: [] for m in MODELS}, {m: [] for m in MODELS}
    inv = lambda v, ci: (1 / v, [1 / ci[1], 1 / ci[0]])
    for where, p in pairs.items():
        s, a, b = re.match(r"(\S+): (\S+) / (\S+)", where).groups()
        for f, what, out in (("recall", "recall", crit), ("iou_only", "IoU-only recall", crit),
                             ("critical", "critical boxes per photo", other), ("ms", "modelled ms per job", other)):
            if f not in p or (f == "critical" and s.startswith("clean")):
                continue
            (v, ci), d = p[f], side(p[f][1], 1)
            if not d:
                continue
            loser, winner, (v, ci) = (a, b, (v, ci)) if (d < 0) == (f in ("recall", "iou_only")) else (b, a, inv(v, ci))
            out[loser].append(f"{s} {what} x{v:.3f} (CI {ci[0]:.2f}-{ci[1]:.2f}) against {winner}")
        if s.startswith("clean"):
            for f, what in (("boxes_per_photo_diff", "boxes per clean photo"), ("any_box_share_diff", "share of clean photos with a box")):
                (v, (lo, hi)), d = p[f], side(p[f][1], 0)
                if d:
                    loser, winner, v, lo, hi = (a, b, v, lo, hi) if d > 0 else (b, a, -v, -hi, -lo)
                    crit[loser].append(f"{s} {v:+.3f} {what} (CI {lo:+.3f} to {hi:+.3f}) against {winner}")
    return crit, other


def verdicts(pairs: dict) -> dict:
    """The dev criterion, on the held-out sets: better / worse / same on recall at the 30 % cap on CODEBRIM eval200 and
    synth200, and a clean harm when either clean-set difference CI lies wholly above 0, for each ordered pair."""
    out = {}
    for a, b in itertools.permutations(MODELS, 2):
        key, sg = (f"{a} / {b}", 1) if f"codebrim@0.3: {a} / {b}" in pairs else (f"{b} / {a}", -1)
        if f"codebrim@0.3: {key}" not in pairs:
            continue
        v = {s: None if (p := pairs.get(f"{s}@0.3: {key}")) is None else ("worse", "same", "better")[1 + sg * side(p["recall"][1], 1)]
             for s in ("codebrim", "synth")}  # None: that set has not been run for both (yet)
        c = pairs.get(f"clean@0.3: {key}")
        v["clean_harm"] = None if c is None else any(sg * side(c[f][1], 0) > 0 for f in ("boxes_per_photo_diff", "any_box_share_diff"))
        out[f"{a} vs {b}"] = v
    return out


def verdicts_vs(pairs: dict, c: str, b: str) -> dict:
    """Configuration c against its model b at the 30 % cap, by the same standard: recall better / same / worse on
    CODEBRIM eval200 and synth200, a clean harm, and (outside the criterion) critical boxes and ms ratios."""
    out = {}
    for s in ("codebrim", "synth"):
        if p := pairs.get(f"{s}@0.3: {c} / {b}"):
            out[s] = ("worse", "same", "better")[1 + side(p["recall"][1], 1)]
            out[f"{s}_critical"], out[f"{s}_ms"] = p["critical"], p["ms"]
    if p := pairs.get(f"clean@0.3: {c} / {b}"):
        out["clean_harm"] = any(side(p[f][1], 0) > 0 for f in ("boxes_per_photo_diff", "any_box_share_diff"))
    return out


BASELINES = ("downscale", "tiling@L2", "tiling@L1", "tiling@L0")
MATRIX_CAPS = ("0.3", "0.1")  # the cap the rule read, and about L2 tiling's pixels on CODEBRIM


def _gate(c: dict) -> dict:
    """A Drishyam job's two tiers in v1-policy2's terms. v1's runs predate them: nothing was held (its shown boxes are its
    drafted ones) and sign-offs were not recorded, so both sides count critical findings by each finding's final measure
    (`signoffs`, a slight undercount: a finding made critical by a coarser look is missed) and v1-policy2's exact count
    is kept beside it (`signoffs_exact`)."""
    approx = sum(is_critical(f["cls"], f["area"]) for f in c["preds_union"])
    return {**c, "preds_look": c.get("preds_look", []), "preds_all": c.get("preds_all", c["preds"]),
            "needs_look": c.get("needs_look", 0), "signoffs": approx, "signoffs_exact": c.get("signoffs"), "reviews": 0}


def policy2_read(p2: list[dict], v1: list[dict], records: list[dict]) -> dict:
    """v1-policy2 against v1 on one held-out set, both at the matrix caps, with the baselines of v1-policy2's own run
    (checked against v1's run of them): rows by evals.dev_sweep.row (v1-policy2 paired with v1 at its cap) on every
    photo and on the hard subset (evals.policy2), and the compute-savings vs accuracy matrix: pixels and modelled ms
    against recall of what a person is shown and drafted critical-class recall, each policy against L2 tiling, L1
    tiling, the downscale and v1, with paired 95 % bootstrap CIs."""
    same = {k: all(a["configs"][k]["preds"] == b["configs"][k]["preds"] and a["configs"][k]["wall_s"] == b["configs"][k]["wall_s"]
                   for a, b in zip(p2, v1, strict=True)) for k in BASELINES}
    merged = [{**a, "configs": {**{k: a["configs"][k] for k in BASELINES},
                                **{f"{m}@{cap}": _gate(x["configs"][f"drishyam@{cap}"]) for cap in MATRIX_CAPS
                                   for m, x in (("v1", b), ("v1-policy2", a))}}} for a, b in zip(p2, v1, strict=True)]
    keys, n = list(merged[0]["configs"]), len(records)
    scored = {k: harness.score_config(k, records, merged, "preds") for k in keys}
    shown = {k: harness.score_config(k, records, merged, "preds" if k in BASELINES else "preds_all") for k in keys}
    ref = lambda k: f"v1@{k.rpartition('@')[2]}" if k.startswith("v1-policy2@") else None
    rows = {k: {a: r4(b) for a, b in dev_sweep.row(k, records, merged, scored, ref(k)).items() if a != "per_image"}
            for k in keys}
    matrix = {}
    for k in keys:
        x = rows[k]
        m = {"pixels_frac": x["pixels_frac"], "ms_per_job": x["ms_per_job"], "recall_shown": r4(shown[k]["recall"]),
             "recall_shown_ci": harness.boot_ci(harness.recall_of(shown[k]), n),
             **{f: x.get(f) for f in ("crit_recall", "crit_recall_ci", "critical_per_image", "signoffs_per_image",
                                      "needs_look_per_image", "look_per_image")}}
        exact = [r["configs"][k].get("signoffs_exact") for r in merged]
        if k not in BASELINES and None not in exact:
            m["signoffs_exact_per_image"] = r4(sum(exact) / n)
        for b in ("tiling@L2", "tiling@L1", "downscale", f"v1@{k.rpartition('@')[2]}"):
            if b in scored and b != k and shown[b]["recall"]:
                cv, cci = policy2.ratio_of(policy2.crit_cols(scored[k], records), policy2.crit_cols(scored[b], records), n)
                m[f"vs {b}"] = {"recall_shown": [r4(shown[k]["recall"] / shown[b]["recall"]), harness.ratio_ci(shown[k], shown[b], n)],
                                "crit_recall": [r4(cv) if cv is not None else None, cci],
                                "pixels": r4(x["pixels_frac"] / rows[b]["pixels_frac"]),
                                "ms": r4(x["ms_per_job"] / rows[b]["ms_per_job"])}
        matrix[k] = m
    keep = [i for i, rec in enumerate(records) if policy2.hard_reasons(rec)]
    hard = {"photos": len(keep), "of": n, "indices": keep,
            "reasons": dict(collections.Counter(why for i in keep for why in policy2.hard_reasons(records[i]))),
            "small_critical_boxes": sum(len(policy2.small_critical(records[i])) for i in keep),
            "rows": {k: {a: r4(b) for a, b in v.items()} for k, v in policy2.subset_rows(records, merged, keep, keys, "v1").items()}
            if keep else {}}
    looks = {k: dict(collections.Counter(f"{t['action']}:{t['cause']}" for r in merged for t in r["configs"][k]["trace"]
                                         if t["action"] != "halt")) for k in keys if "trace" in merged[0]["configs"][k]}
    return {"baselines_same_as_v1s_run": same, "rows": rows, "matrix": matrix, "hard": hard, "looks_by_action_cause": looks}


def markdown(out: dict) -> str:
    """The held-out tables docs/report.md quotes at the 30 % cap (tests/test_evals.py checks the report holds them)."""
    ci = lambda c: "–" if c is None or None in c else f"{c[0]:.2f}–{c[1]:.2f}"
    d = lambda v, c: f"{v:+.2f} ({c[0]:+.2f} to {c[1]:+.2f})"
    t = out["table"]
    rows = [("| Set | Model | recall (95 % CI) | IoU-only | chance | precision | boxes / photo | critical / photo | "
            "× L0 tiling (95 % CI) | × downscale (95 % CI) | modelled ms / job |"), "|" + " --- |" * 11]
    rows += [f"| {NAMES[s]} | {m} | {r['recall']:.3f} ({ci(r['recall_ci'])}) | {r['recall_iou_only']:.3f} | "
             f"{r['chance_recall']:.3f} | {r['precision']:.3f} | {r['boxes_per_image']:.1f} | {r['critical_per_image']:.2f} | "
             f"{r['vs_tiling_L0']:.2f} ({ci(r['vs_tiling_L0_ci'])}) | {r['vs_downscale']:.2f} ({ci(r['vs_downscale_ci'])}) | "
             f"{r['ms_per_job']:.0f} |" for s in ("codebrim", "synth", "large") for m, r in t.get(f"{s}@0.3", {}).items()]
    n = out["datasets"]["clean"]["n"]
    clean = ["| Model | boxes / clean photo (95 % CI) | photos with a box | critical boxes / photo | modelled ms / job |",
             "|" + " --- |" * 5]
    clean += [f"| {m} | {r['boxes_per_image']:.3f} ({ci(r['boxes_per_image_ci'])}) | {round(r['images_with_any'] * n)} of {n} | "
              f"{r['critical_per_image']:.3f} | {r['ms_per_job']:.0f} |" for m, r in t.get("clean@0.3", {}).items()]
    pairs = [("| A / B | CODEBRIM recall, A ÷ B (95 % CI) | synth200 recall, A ÷ B | clean boxes / photo, A − B | "
             "clean share of photos with a box, A − B |"), "|" + " --- |" * 5]
    for k, p in out["pairs"].items():
        if k.startswith("codebrim@0.3: "):
            ab = k.split(": ")[1]
            q, c = out["pairs"].get(f"synth@0.3: {ab}"), out["pairs"].get(f"clean@0.3: {ab}")
            pairs.append(f"| {ab} | {p['recall'][0]:.2f} ({ci(p['recall'][1])}) | "
                         + (f"{q['recall'][0]:.2f} ({ci(q['recall'][1])})" if q else "–") + " | "
                         + (f"{d(*c['boxes_per_photo_diff'])} | {d(*c['any_box_share_diff'])} |" if c else "– | – |"))
    return "\n\n".join("\n".join(x) for x in (rows, clean, pairs))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(prog="python -m evals.heldout", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="results/heldout_models.json")
    a = ap.parse_args(argv)
    dev = json.loads((REPO / "results" / "dev" / "model_choice_dev.json").read_text())
    log = (REPO / "runs" / "heldout" / "log.txt").read_text()
    starts = dict(re.findall(r"^\S+ (\S+) start \| .* \| HEAD (\w+)", log, re.MULTILINE))  # the last start of each call wins
    table, pairs, datasets, runs, acted = {}, {}, {}, {}, {}
    for s, (path, n, seed) in SETS.items():
        records, _ = harness.load_records(REPO / path, n, seed=seed)
        data_id = harness.dataset_sha256(records)
        datasets[s] = {"n": len(records), **data_id}
        got = {m: r for m, (prefix, _, _) in ALL.items() if (r := load(prefix, s, records, data_id))}
        for m, r in got.items():
            dm = dev["models"][ALL[m][2]]  # a configuration is charged and checked as its model
            cost = dm["cost_table"]["file"]
            if not r["summary"]["cost_model"]["source"].startswith(cost + " ") or r["segmenter"] != dm["segmenter"]:
                raise SystemExit(f"{m} on {s}: scored {r['segmenter']} on {r['summary']['cost_model']['source']}, not "
                                 f"{dm['segmenter']} on {cost}")
            profiles = {x["configs"][k]["profile"] for x in r["results"] for k in CAPS}
            if profiles != {dm["profile"]}:
                raise SystemExit(f"{m} on {s} ran profiles {profiles}, not {dm['profile']}")
            fallback = {x["configs"][k]["switches"].get("fallback_l2", False) for x in r["results"] for k in CAPS}
            if fallback != {m in CONFIGS}:  # runs from before the switch existed ran without the fallback
                raise SystemExit(f"{m} on {s} ran fallback_l2 {fallback}")
            knobs = {tuple(x["configs"][k]["switches"].get(n, v) for n, v in POLICY2_OFF.items()) for x in r["results"] for k in CAPS}
            if knobs != {tuple({**POLICY2_OFF, **(POLICY2[dm["profile"]] if m in CONFIGS else {})}.values())}:
                raise SystemExit(f"{m} on {s} ran v1-policy2's knobs {knobs}")  # the models ran before the knobs existed
            prov = r["summary"]["provenance"]
            runs[f"{m} {s}"] = {"folder": f"runs/model-eval/{ALL[m][0]}-{s}", "code_commit": starts.get(ALL[m][1]),
                                "finished": prov["timestamp"],  # the harness takes its provenance at the end of a run
                                **{k: prov.get(k) for k in ("load_avg_start", "load_avg_end", "machine")}}
        scored = {m: {k: harness.score_config(k, records, r["results"]) for k in KEYS} for m, r in got.items()}
        x = {m: {k: per_image(records, r["results"], k) for k in CAPS} for m, r in got.items()}
        for c in (c for c in got if c in CONFIGS):
            acted[f"{c} {s}"] = policy2_read(got[c]["results"], got[ALL[c][2]]["results"], records)
        for k in CAPS:
            table[f"{s}@{k[9:]}"] = {m: row(k, records, r["results"], scored[m]) for m, r in got.items()}
            for ma, mb in itertools.combinations([m for m in got if m in MODELS], 2):
                pairs[f"{s}@{k[9:]}: {ma} / {mb}"] = compare(scored[ma][k], scored[mb][k], x[ma][k], x[mb][k], len(records))
            for c, b in ((c, ALL[c][2]) for c in got if c in CONFIGS):
                pairs[f"{s}@{k[9:]}: {c} / {b}"] = compare(scored[c][k], scored[b][k], x[c][k], x[b][k], len(records))
        for base in ("downscale", "tiling@L0"):
            table[f"{s}:{base}"] = {m: row(base, records, r["results"], scored[m]) for m, r in got.items()}
    pub = {p.parent.name: json.loads(p.read_text()) for p in sorted((REPO / "public" / "results").glob("*_dacl10k*/summary.json"))}
    leaks = {json.dumps(v["leakage"], sort_keys=True) for v in pub.values() if "leakage" in v}
    if len(leaks) > 1:
        raise SystemExit("the published CODEBRIM summaries carry different leakage counts")
    models_only = {k: v for k, v in pairs.items() if not any(f" {c} / " in k for c in CONFIGS)}
    crit, other = losses({k: v for k, v in models_only.items() if k.split("@")[0] in DEV_SETS})
    large_crit, large_other = losses({k: v for k, v in models_only.items() if k.startswith("large")})
    ver = verdicts(pairs)
    best = [m for m in MODELS if all(v["clean_harm"] is False and "worse" not in (v["codebrim"], v["synth"])
                                     and "better" in (v["codebrim"], v["synth"]) for k, v in ver.items() if k.startswith(m + " vs"))
            and any(k.startswith(m + " vs") for k in ver)]
    clear = [m for m in MODELS if not crit[m] and any(k.startswith(m + " vs") for k in ver)]
    others = [m for m in clear if m != "v1"]
    statement = (("Held-out confirms the dev choice: v1 is again the only model with no measurable loss against any other "
                  "on CODEBRIM eval200, synth200 or the clean set." if not others else
                  f"Held-out does not contradict the dev choice but does not separate v1 from {', '.join(others)}: none of "
                  "them has a measurable loss against any other on CODEBRIM eval200, synth200 or the clean set.")
                 if "v1" in clear else "Held-out does not confirm the dev choice: v1 has measurable losses (where_each_loses.v1).")
    if missing := [f"{m} {s}" for m in MODELS for s in DEV_SETS if f"{m} {s}" not in runs]:
        statement = f"Incomplete, no verdict on the dev choice yet: not run yet: {', '.join(missing)}."
    out = {
        "what": ("The one held-out evaluation of the trained segmenters (dacl10k v1, v2a, v2b, v2c) and the classical engine, "
                 "each at its profile (drishyam.agent.PROFILES, unchanged) and charged the cost table the dev comparison "
                 "charged it, each held-out set scored once per model. Nothing was tuned on these numbers. v2c was "
                 "scored later than the others, after its own dev calibration; the others' runs were not repeated."),
        "criterion": ("The dev comparison's criterion (results/dev/model_choice_dev.json), unchanged, on the held-out "
                      "counterparts of its splits: recall at the 30 % cap on CODEBRIM eval200 (for dev100) and synth200 (for "
                      "synth_dev), and false alarms at the 30 % cap on the clean set synth_clean (for the clean dev split). "
                      "A is better than B on a set when the paired 95 % bootstrap CI of A's recall ratio to B lies wholly "
                      "above 1, worse when wholly below; A harms against B on the clean set when the CI of its boxes-per-photo "
                      "difference or of its difference in the share of photos with any box lies wholly above 0. The dev "
                      "choice of v1 rested on v1 being the only model with no measurable loss against any other "
                      "(where_each_loses), so that is what this checks."),
        "dev_choice": {"dev": "v1", "best_held_out": best, "no_measurable_loss": clear, "confirms": None if missing else "v1" in clear and not others,
                       "statement": statement},
        "verdicts": ver, "where_each_loses": crit, "other_measurable_costs": other,
        "synth_large": {"where_each_loses": large_crit, "other_measurable_costs": large_other,
                        "note": "v1 and the classical engine only; not a counterpart of a dev split, so not in the verdicts"},
        "cost_tables": {m: {**dev["models"][m]["cost_table"], "sha256": hashlib.sha256(
            (REPO / dev["models"][m]["cost_table"]["file"]).read_bytes()).hexdigest()} for m in MODELS},
        "cost_note": dev["cost_tables"],
        "models": {m: {k: dev["models"][m][k] for k in ("segmenter", "profile", "profile_values")} for m in MODELS},
        "datasets": datasets, "runs": runs, "commands": COMMANDS,
        "prior_exposure": PRIOR, "leakage": json.loads(leaks.pop()) if leaks else None, "published": sorted(f"public/results/{k}/summary.json" for k in pub),
        "table": table, "pairs": pairs,
    }
    for c, (_, tag, b) in CONFIGS.items():  # only once its runs exist, so the file without them is as it was
        if done := [s for s in SETS if f"{c} {s}" in runs]:
            out["what"] += (f" {c}, v1 with the configuration its dev rule chose, was scored once more, later, beside v1 "
                            f"(whose runs were not repeated): see the {c} block.")
            out["commands"] = [*COMMANDS, POLICY2_COMMAND]
            leak = json.loads((REPO / "runs" / "heldout" / f"leakage-{tag}.json").read_text())
            if out["leakage"] and {k: leak[k] for k in out["leakage"] if k != "calibration"} != {
                    k: v for k, v in out["leakage"].items() if k != "calibration"}:
                raise SystemExit(f"{c}'s leakage check counts differ from the published ones")
            out[c] = {"model": b, "sets": done, "prior_exposure": PRIOR_POLICY2,
                      "configuration": POLICY2[out["models"][b]["profile"]],
                      "dev_rule": "results/dev/policy2_sweep.json (rule stated in 54b5601, before its dev run)",
                      "verdict_vs_v1": verdicts_vs(pairs, c, b), "read": {s: acted[f"{c} {s}"] for s in done},
                      "leakage": f"the check ran again before this run (runs/heldout/leakage-{tag}.json): the same counts "
                                 "as the published leakage block"}
    Path(REPO / a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(markdown(out), statement, sep="\n\n")
    return out


if __name__ == "__main__":
    main()
