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
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np

from evals import dev_sweep, harness

REPO = harness.REPO
RUNS = REPO / "runs" / "model-eval"
MODELS = {"v1": ("onnx", "dacl10k"), "v2a": ("onnx-v2a", "dacl10k_v2a"), "v2b": ("onnx-v2b", "dacl10k_v2b"),
          "classical-v1": ("classical", "dacl10k")}  # run folder prefix; the run_model.sh call that ran it
SETS = {"codebrim": ("data/codebrim", 200, 7), "synth": ("data/synth200", 200, 7), "clean": ("data/synth_clean", 51, 11),
        "large": ("data/synth_large", 100, 7)}
DEV_SETS = ("codebrim", "synth", "clean")  # held-out counterparts of the dev splits (real, synth_dev, clean) the choice read
KEYS, CAPS = ("downscale", "tiling@L0", "drishyam@0.3", "drishyam@1.0"), ("drishyam@0.3", "drishyam@1.0")
COMMANDS = [("runs/heldout/run_model.sh dacl10k local-macos-x86-dacl10k-retime (v1; also the classical clean and "
             "synth_large runs and v1 on synth_large)"), "runs/heldout/run_model.sh dacl10k_v2a local-macos-x86-dacl10k_v2a",
            "runs/heldout/run_model.sh dacl10k_v2b local-macos-x86-dacl10k_v2b",
            ("each: python -m evals.model_eval --publish --onnx models/drishyam_<tag>.onnx --costs <arm> --dacl10k <kagglehub "
            "dacl10k folder> --codebrim data/codebrim --workers 6, then python -m evals.harness --cost-model fixed --workers 6 "
            "--micro-arm <arm> --data data/synth_clean --n 51 --seed 11 --chaos 0,0,0 (as make eval-clean) with "
            "DRISHYAM_ONNX set; synth_large as make eval-large (--n 100 --seed 7 --size 4608x3456)")]
PRIOR = ("v1 had been scored on two of these sets before, once each, on 2026-10-18 and never published: CODEBRIM eval200 "
         "(run started 14:07Z) and synth200 (00:52Z), with the agent of that day, before its dev profile, on its first cost "
         "table (results/local-macos-x86-dacl10k). The runs are in the main checkout's ignored runs/model-eval/. So for v1 "
         "these are second scorings; v2a and v2b had never been scored on any held-out set (no other run under runs/ or a "
         "worktree carries these sets' sha256s with an ONNX segmenter).")
NAMES = {"codebrim": "CODEBRIM eval200", "synth": "synth200", "large": "synth_large, 4608×3456", "clean": "clean"}
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
    table, pairs, datasets, runs = {}, {}, {}, {}
    for s, (path, n, seed) in SETS.items():
        records, _ = harness.load_records(REPO / path, n, seed=seed)
        data_id = harness.dataset_sha256(records)
        datasets[s] = {"n": len(records), **data_id}
        got = {m: r for m, (prefix, _) in MODELS.items() if (r := load(prefix, s, records, data_id))}
        for m, r in got.items():
            cost = dev["models"][m]["cost_table"]["file"]
            if not r["summary"]["cost_model"]["source"].startswith(cost + " ") or r["segmenter"] != dev["models"][m]["segmenter"]:
                raise SystemExit(f"{m} on {s}: scored {r['segmenter']} on {r['summary']['cost_model']['source']}, not "
                                 f"{dev['models'][m]['segmenter']} on {cost}")
            profiles = {x["configs"][k]["profile"] for x in r["results"] for k in CAPS}
            if profiles != {dev["models"][m]["profile"]}:
                raise SystemExit(f"{m} on {s} ran profiles {profiles}, not {dev['models'][m]['profile']}")
            prov = r["summary"]["provenance"]
            runs[f"{m} {s}"] = {"folder": f"runs/model-eval/{MODELS[m][0]}-{s}", "code_commit": starts.get(MODELS[m][1]),
                                "finished": prov["timestamp"],  # the harness takes its provenance at the end of a run
                                **{k: prov.get(k) for k in ("load_avg_start", "load_avg_end", "machine")}}
        scored = {m: {k: harness.score_config(k, records, r["results"]) for k in KEYS} for m, r in got.items()}
        x = {m: {k: per_image(records, r["results"], k) for k in CAPS} for m, r in got.items()}
        for k in CAPS:
            table[f"{s}@{k[9:]}"] = {m: row(k, records, r["results"], scored[m]) for m, r in got.items()}
            for ma, mb in itertools.combinations(got, 2):
                pairs[f"{s}@{k[9:]}: {ma} / {mb}"] = compare(scored[ma][k], scored[mb][k], x[ma][k], x[mb][k], len(records))
        for base in ("downscale", "tiling@L0"):
            table[f"{s}:{base}"] = {m: row(base, records, r["results"], scored[m]) for m, r in got.items()}
    pub = {p.parent.name: json.loads(p.read_text()) for p in sorted((REPO / "public" / "results").glob("*_dacl10k*/summary.json"))}
    leaks = {json.dumps(v["leakage"], sort_keys=True) for v in pub.values() if "leakage" in v}
    if len(leaks) > 1:
        raise SystemExit("the published CODEBRIM summaries carry different leakage counts")
    crit, other = losses({k: v for k, v in pairs.items() if k.split("@")[0] in DEV_SETS})
    large_crit, large_other = losses({k: v for k, v in pairs.items() if k.startswith("large")})
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
        "what": ("The one held-out evaluation of the trained segmenters (dacl10k v1, v2a, v2b) and the classical engine, each "
                 "at its profile (drishyam.agent.PROFILES, unchanged) and charged the cost table the dev comparison charged "
                 "it, each held-out set scored once per model. Nothing was tuned on these numbers."),
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
    Path(REPO / a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(markdown(out), statement, sep="\n\n")
    return out


if __name__ == "__main__":
    main()
