"""The trained segmenter against the classical engine, in one command, once scripts/install_model.sh has installed it.

    make eval-model [ONNX=models/drishyam_dacl10k.onnx] [DACL10K=<kagglehub dacl10k folder>] [CODEBRIM=data/codebrim]
    python -m evals.model_eval --onnx models/drishyam_dacl10k.onnx [--fp16 F] [--dacl10k D] [--codebrim C] [--n 200]

1. the dacl10k vs CODEBRIM leakage check (evals.leakage), when --dacl10k is given; --publish with CODEBRIM requires it;
2. times the ONNX's own job ops into results/<micro-arm>-dacl10k/micro.json, so the fixed cost model charges it;
3. runs evals.harness with DRISHYAM_ONNX set (a child process each, so the variable reaches every worker) on
   CODEBRIM (if present) and data/synth200, into runs/model-eval/onnx-*, with the governor's overrides off: the
   agent runs the same accurate net as the baselines (the governor otherwise swaps in the 256-px fp16 net for latency);
4. checks every summary before comparing: the segmenter label (the ONNX runs must say onnx:<file>, the classical ones
   classical-v1: an env-var A/B once scored the default engine twice), that it is not confounded (governor on,
   p95>SLA > 0, another net), and that both sides scored the same data (dataset sha256s);
5. prints classical vs ONNX and, with --publish (make eval-model passes it), writes the ONNX summaries (aggregate
   numbers only) to public/results/codebrim_dacl10k/ (with the leakage count and its calibration) and
   public/results/synth_dacl10k/, whatever they show.
Those paths are models/drishyam_dacl10k.onnx's (v1). Another model there (models/drishyam_dacl10k_v2a.onnx, ...)
gets its own: results/<micro-arm>-dacl10k_v2a/, runs/model-eval/onnx-v2a-*, public/results/*_dacl10k_v2a/, so
installing it beside v1 overwrites none of v1's cost table or runs.
The classical side is the published summary when it is comparable on the same data; otherwise it is run too.

A two-model cascade (drishyam.cascade), on the dev splits only, never a held-out set and never published:

    make eval-cascade SCOUT=models/drishyam_dacl10k_v2a.onnx CONFIRM=models/drishyam_dacl10k_v2b.onnx [LABEL=...]
    python -m evals.model_eval --onnx SCOUT --confirm CONFIRM [--label L] [--codebrim C] [--workers 4]

times the pair's job ops (segment:confirm too) into results/<micro-arm>-<pair>/micro.json, the pair named as the
single models are (dacl10k_v2a+v2b; an FP16 file keeps its .fp16), runs evals.dev_sweep on synth_dev, the clean split
and CODEBRIM dev100 (if present) with the scout alone, the confirm net alone and the cascade at caps 0.3 and 1.0,
checks that each arm ran the nets it is named for, and writes results/dev/cascade_<pair>.json with --label.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from evals.harness import BUDGETS, HEADLINE_BUDGET, REPO, TILE_LEVELS, confounded

CLASSICAL = "classical-v1"
KEYS = ["downscale", *(f"tiling@L{lv}" for lv in TILE_LEVELS), *(f"drishyam@{c}" for c in BUDGETS)]
DATA_KEYS = ("dataset_gt_sha256", "dataset_images_sha256")
LEAK_KEYS = ("eval_images", "train_images", "max_dist", "min_ncc", "candidates", "near_duplicates", "hash_at_or_below")


def check(summary: dict, label: str, where) -> dict:
    """The summary's segmenter must be `label`, and it must be a like-for-like policy comparison."""
    if summary["segmenter"] != label:
        raise SystemExit(f"{where} was scored with {summary['segmenter']}, not {label}: refusing to compare")
    if why := confounded(summary):
        raise SystemExit(f"{where} is not a like-for-like policy comparison ({why}): refusing to compare or publish it")
    return summary


def same_data(a: dict, b: dict) -> bool:
    """Both summaries scored the same ground truth on the same image bytes (and both say so)."""
    ka, kb = ({k: s.get("provenance", {}).get(k) for k in DATA_KEYS} for s in (a, b))
    return None not in ka.values() and ka == kb


def same_costs(summary: dict, micro_arm: str) -> bool:
    """The summary was charged `micro_arm`'s fixed costs. The ONNX side is timed on this machine, so a published
    classical summary charged another machine's table (the arm64 one, say) is not its baseline: it is re-run here."""
    return summary.get("cost_model", {}).get("source", "").startswith(f"results/{micro_arm}/")


def leakage_block(path: Path) -> dict:
    """The dacl10k-vs-CODEBRIM leakage counts with their calibration (copies and known-unrelated photos), which a
    published CODEBRIM number for a dacl10k-trained model must carry."""
    rep = json.loads(path.read_text())
    if "near_duplicates" not in rep or not isinstance(rep.get("calibration"), dict):
        raise SystemExit(f"{path} has no leakage count with a calibration ({rep.get('skipped', 'incomplete')}): "
                         "refusing to publish a CODEBRIM number for the dacl10k model")
    return {**{k: rep[k] for k in LEAK_KEYS}, "calibration": rep["calibration"]}


def harness(data, out: Path, n: int, env: dict, micro_arm: str, workers) -> dict:
    cmd = [sys.executable, "-m", "evals.harness", "--cost-model", "fixed", "--data", str(data), "--n", str(n),
           "--out", str(out), "--results-dir", str(out), "--micro-arm", micro_arm,
           *(["--workers", str(workers)] if workers else [])]
    subprocess.run(cmd, env=env, cwd=REPO, check=True)
    return json.loads((out / "summary.json").read_text())


def table(name: str, before: dict, after: dict) -> None:
    pts = lambda s: dict(zip(KEYS, [p for k in ("downscale", "tiling", "drishyam") for p in s["policies"][k]], strict=True))
    b, a = pts(before), pts(after)
    f = lambda v: "-" if v is None else f"{v:.3f}"
    print(f"\n{name}: {before['segmenter']} -> {after['segmenter']} ({after['dataset']})")
    print(f"{'policy':<15}{'pixels':>8}{'recall':>15}{'IoU-only':>15}{'chance':>15}{'precision':>15}{'boxes/img':>15}")
    for k in KEYS:
        cells = [f"{f(b[k][m])}>{f(a[k][m])}" for m in ("recall", "recall_iou_only", "chance_recall", "precision")]
        print(f"{k:<15}{a[k]['pixels_frac']:>8.3f}" + "".join(f"{c:>15}" for c in cells)
              + f"{b[k]['boxes_per_image']:>7}>{a[k]['boxes_per_image']:<7}")
    key = f"drishyam@{HEADLINE_BUDGET}"
    print(f"per class at {key}: " + ", ".join(f"{c} {f(v)}>{f(after['per_class'][key][c])}" for c, v in before["per_class"][key].items()))
    fa = lambda s: s["agent_metrics"]["false_accept_critical"][key]["rate"]
    print(f"critical false-accept rate at {key}: {f(fa(before))}>{f(fa(after))}")
    print(f"recall 95% CI at {key}: {b[key]['recall_ci']} > {a[key]['recall_ci']}")


# ---------------------------------------------------------------- a two-model cascade, dev splits only
CASCADE_ARMS = "scout=;confirm=net:confirm;cascade=net:cascade"  # evals.dev_sweep variants: scout-only is the reference
CASCADE_CAPS = "0.3,1.0"
DEV_SPLITS = ("synth_dev", "clean", "real")  # evals.dev_sweep.SPLITS and CODEBRIM dev100: no held-out set


def pair_tag(scout: Path, confirm: Path) -> str:
    """dacl10k_v2a+v2b: each file's name without drishyam_ and .onnx (an FP16 file keeps its .fp16), the confirm's
    without the dacl10k_ both share."""
    a, b = (p.name.removesuffix(".onnx").removeprefix("drishyam_") for p in (scout, confirm))
    return f"{a}+{b.removeprefix('dacl10k_')}"


def check_arms(split: str, doc: dict, scout: str, confirm: str) -> None:
    """Each arm ran the nets it is named for, all under one profile (an env-var A/B once scored one engine twice)."""
    want = {"scout": (scout, None), "confirm": (confirm, None), "cascade": (scout, confirm)}
    agents = {k: r for k, r in doc["rows"].items() if "@" in k}
    for k, r in agents.items():
        if (r["segmenter"], r["confirm"]) != want[k.partition("@")[0]]:
            raise SystemExit(f"{split}: {k} ran {r['segmenter']} + {r['confirm']}, not {want[k.partition('@')[0]]}")
    if len({r["profile"] for r in agents.values()}) != 1:
        raise SystemExit(f"{split}: the arms ran different profiles: {sorted({r['profile'] for r in agents.values()})}")


# metric: (its paired CI against scout-only, the value meaning no change, +1 if higher is better / -1 if worse / 0 neither)
READ = {"recall": ("vs_ref_ci", 1, 1), "recall_iou_only": ("vs_ref_iou_only_ci", 1, 1), "chance_recall": ("chance_vs_ref_ci", 1, 0),
        "boxes_per_image": ("boxes_diff_vs_ref_ci", 0, 0), "images_with_any": ("images_with_any_vs_ref_ci", 0, 0),
        "ms_per_job": ("ms_vs_ref_ci", 1, -1), "recall_per_s": ("recall_per_s_vs_ref_ci", 1, 1)}


def readings(split: str, r: dict) -> dict:
    """Each paired 95 % CI against scout-only, gains and harms read by one standard: a change only when the CI excludes
    no change, a gain or a harm where the direction says which (on the clean split every box is a false alarm), else
    higher or lower."""
    out = {}
    for m, (k, null, good) in READ.items():
        if (ci := r.get(k)) and None not in ci:
            good = -1 if split == "clean" and m in ("boxes_per_image", "images_with_any") else good
            d = (ci[0] > null) - (ci[1] < null)
            out[m] = "no clear change" if not d else ("lower", "higher")[d > 0] if not good else ("harm", "gain")[d == good]
    return out


def cascade_table(split: str, doc: dict) -> None:
    f = lambda v: "-" if v is None else f"{v:.3f}"
    rd = lambda r, m: f"[{readings(split, r)[m]}]" if m in readings(split, r) else ""
    ci = lambda r, k, m: f"x{f(r.get(k))} {r.get(k + '_ci', '')} {rd(r, m)}" if k + "_ci" in r else ""
    fa = " (all false alarms)" if split == "clean" else ""
    print(f"\n{split} ({doc['n']} images): against scout-only at the same cap, paired 95 % bootstrap CIs, read alike")
    for k, r in doc["rows"].items():
        if "@" in k:
            print(f"{k:<13} recall {f(r['recall'])} {ci(r, 'vs_ref', 'recall')}  IoU-only {f(r['recall_iou_only'])}"
                  f" {rd(r, 'recall_iou_only')}  chance {f(r['chance_recall'])} {rd(r, 'chance_recall')}"
                  f"  ms/job {r['ms_per_job']:.0f} {ci(r, 'ms_vs_ref', 'ms_per_job')}  recall/s {f(r['recall_per_s'])}"
                  f" {ci(r, 'recall_per_s_vs_ref', 'recall_per_s')}  boxes/photo{fa} {r['boxes_per_image']:.2f}"
                  f" {rd(r, 'boxes_per_image')}  any box {r['images_with_any']:.2f} {rd(r, 'images_with_any')}"
                  f"  confirms/job {r.get('confirms_per_job', 0):.2f}")


def cascade_eval(a, scout: Path, fp16: Path) -> None:
    if a.publish:
        raise SystemExit("--confirm compares on the dev splits only: nothing is published")
    confirm = (REPO / a.confirm).resolve()
    if not confirm.exists():
        raise SystemExit(f"{confirm} not found")
    tag = pair_tag(scout, confirm)
    arm, runs = f"{a.micro_arm}-{tag}", REPO / "runs" / "model-eval" / f"cascade-{tag}"
    plain = {k: v for k, v in os.environ.items() if k not in ("DRISHYAM_ONNX", "DRISHYAM_ONNX_FP16", "DRISHYAM_ONNX_CONFIRM")}
    env = {**plain, "DRISHYAM_ONNX": str(scout), **({"DRISHYAM_ONNX_FP16": str(fp16)} if fp16.exists() else {}),
           "DRISHYAM_ONNX_CONFIRM": str(confirm)}
    subprocess.run([sys.executable, "-m", "bench.micro", "--arm", arm, "--job-ops-only",
                    *(["--max-load", str(a.max_load)] if a.max_load is not None else [])], env=env, cwd=REPO, check=True)
    runs.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in DEV_SPLITS:
        if split == "real" and not (Path(a.codebrim) / "annotations").is_dir():
            print(f"no CODEBRIM under {a.codebrim}: the real dev split is skipped (run `make data`)")
            continue
        out = runs / f"{split}.json"
        subprocess.run([sys.executable, "-m", "evals.dev_sweep", split, CASCADE_CAPS, CASCADE_ARMS, "--micro-arm", arm,
                        "--workers", str(a.workers or 4), "--codebrim", str(Path(a.codebrim).resolve()), "--out", str(out)],
                       env=env, cwd=REPO, check=True)
        splits[split] = json.loads(out.read_text())
        check_arms(split, splits[split], f"onnx:{scout.name}", f"onnx:{confirm.name}")
        cascade_table(split, splits[split])
    dst = REPO / "results" / "dev" / f"cascade_{tag}.json"
    reads = {s: {k: readings(s, r) for k, r in d["rows"].items() if "ref" in r} for s, d in splits.items()}
    dst.write_text(json.dumps({"label": a.label, "micro_arm": arm, "arms": CASCADE_ARMS, "caps": CASCADE_CAPS,
                               "readings": reads, "splits": splits}, indent=1) + "\n")
    print(f"wrote {dst.relative_to(REPO)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evals.model_eval", description=__doc__.split("\n\n")[0])
    ap.add_argument("--onnx", default="models/drishyam_dacl10k.onnx")
    ap.add_argument("--fp16", help="default: the .fp16.onnx beside --onnx, if installed")
    ap.add_argument("--codebrim", default="data/codebrim")
    ap.add_argument("--dacl10k", help="the dacl10k folder (kagglehub), for the leakage check")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--max-load", type=float, help="bench.micro exits before timing the ONNX's job ops at or above this load")
    ap.add_argument("--micro-arm", default="local-macos-x86")
    ap.add_argument("--publish", action="store_true", help="copy the ONNX summaries to public/results/*_dacl10k/")
    ap.add_argument("--skip-codebrim", action="store_true", help="synthetic set only (scripts/install_model.sh's quick check)")
    ap.add_argument("--confirm", help="a cascade's confirm net: --onnx alone, it alone and the cascade on the dev splits")
    ap.add_argument("--label", default="", help="with --confirm: what this run is, written into its results file")
    a = ap.parse_args(argv)
    onnx = (REPO / a.onnx).resolve()
    fp16 = (REPO / a.fp16).resolve() if a.fp16 else onnx.with_name(onnx.name.replace(".onnx", ".fp16.onnx"))
    if not onnx.exists():
        raise SystemExit(f"{onnx} not found: run scripts/install_model.sh first")
    if a.confirm:
        return cascade_eval(a, onnx, fp16)
    tag = onnx.name.split(".")[0].removeprefix("drishyam_")  # dacl10k (v1), dacl10k_v2a, ...: see the docstring
    sub = "" if tag == "dacl10k" else "-" + tag.removeprefix("dacl10k_")
    label, runs, arm = f"onnx:{onnx.name}", REPO / "runs" / "model-eval", f"{a.micro_arm}-{tag}"
    plain = {k: v for k, v in os.environ.items() if k not in ("DRISHYAM_ONNX", "DRISHYAM_ONNX_FP16", "DRISHYAM_ONNX_CONFIRM")}
    env = {**plain, "DRISHYAM_ONNX": str(onnx), **({"DRISHYAM_ONNX_FP16": str(fp16)} if fp16.exists() else {})}
    sets = [("synth", REPO / "data" / "synth200", REPO / "public" / "results")]
    if not a.skip_codebrim and (Path(a.codebrim) / "eval200.txt").exists():
        sets.insert(0, ("codebrim", Path(a.codebrim).resolve(), REPO / "public" / "results" / "codebrim"))
    elif not a.skip_codebrim:
        print(f"no {a.codebrim}/eval200.txt: CODEBRIM skipped (run `make data`)")
    if a.publish and sets[0][0] == "codebrim" and not a.dacl10k:  # before any run: a CODEBRIM number needs the check
        raise SystemExit("--publish with CODEBRIM needs the dacl10k leakage check first: pass --dacl10k (or --skip-codebrim)")
    if a.dacl10k:
        subprocess.run([sys.executable, "-m", "evals.leakage", "--dacl10k", a.dacl10k, "--codebrim", a.codebrim,
                        "--out", str(runs / "leakage.json")], cwd=REPO, check=True)
    leak = leakage_block(runs / "leakage.json") if a.publish and sets[0][0] == "codebrim" else None
    subprocess.run([sys.executable, "-m", "bench.micro", "--arm", arm, "--job-ops-only",
                    *(["--max-load", str(a.max_load)] if a.max_load is not None else [])], env=env, cwd=REPO, check=True)
    for name, data, published in sets:
        after = check(harness(data, runs / f"onnx{sub}-{name}", a.n, env, arm, a.workers), label, runs / f"onnx{sub}-{name}")
        pub = json.loads((published / "summary.json").read_text()) if (published / "summary.json").exists() else None
        if (pub and pub["n_images"] == a.n and pub.get("cost_model", {}).get("kind") == "fixed"
                and same_costs(pub, a.micro_arm) and not confounded(pub) and same_data(pub, after)):
            before = check(pub, CLASSICAL, published / "summary.json")
        else:
            before = check(harness(data, runs / f"classical-{name}", a.n, plain, a.micro_arm, a.workers), CLASSICAL,
                           runs / f"classical-{name}")
        if not same_data(before, after):
            raise SystemExit(f"{name}: the classical and ONNX runs scored different data (dataset hashes differ): refusing to compare")
        table(name, before, after)
        if a.publish:
            dst = REPO / "public" / "results" / f"{name}_{tag}"
            dst.mkdir(parents=True, exist_ok=True)
            extra = {"leakage": leak} if name == "codebrim" else {}  # the count beside the number it qualifies
            (dst / "summary.json").write_text(json.dumps({**after, **extra}, indent=1, ensure_ascii=False))
            print(f"published {dst.relative_to(REPO)}/summary.json")


if __name__ == "__main__":
    main()
