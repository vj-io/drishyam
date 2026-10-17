"""The trained segmenter against the classical engine, in one command, once scripts/install_model.sh has installed it.

    make eval-model [ONNX=models/drishyam_dacl10k.onnx] [DACL10K=<kagglehub dacl10k folder>] [CODEBRIM=data/codebrim]
    python -m evals.model_eval --onnx models/drishyam_dacl10k.onnx [--fp16 F] [--dacl10k D] [--codebrim C] [--n 200]

1. the dacl10k vs CODEBRIM leakage check (evals.leakage), when --dacl10k is given;
2. times the ONNX's own job ops into results/<micro-arm>-dacl10k/micro.json, so the fixed cost model charges it;
3. runs evals.harness with DRISHYAM_ONNX set (a child process each, so the variable reaches every worker) on
   CODEBRIM (if present) and data/synth200, into runs/model-eval/onnx-*;
4. checks every summary's segmenter label before comparing: the ONNX runs must say onnx:<file>, the classical ones
   classical-v1 (an env-var A/B once scored the default engine twice);
5. prints classical vs ONNX and, with --publish (make eval-model passes it), copies the ONNX summaries (aggregate
   numbers only) to public/results/codebrim_dacl10k/ and public/results/synth_dacl10k/, whatever they show.
The classical side is the published summary when it covers the same n with fixed costs; otherwise it is run too.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from evals.harness import BUDGETS, HEADLINE_BUDGET, REPO, TILE_LEVELS

CLASSICAL = "classical-v1"
KEYS = ["downscale", *(f"tiling@L{lv}" for lv in TILE_LEVELS), *(f"drishyam@{c}" for c in BUDGETS)]


def check(summary: dict, label: str, where) -> dict:
    if summary["segmenter"] != label:
        raise SystemExit(f"{where} was scored with {summary['segmenter']}, not {label}: refusing to compare")
    return summary


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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evals.model_eval", description=__doc__.split("\n\n")[0])
    ap.add_argument("--onnx", default="models/drishyam_dacl10k.onnx")
    ap.add_argument("--fp16", help="default: the .fp16.onnx beside --onnx, if installed")
    ap.add_argument("--codebrim", default="data/codebrim")
    ap.add_argument("--dacl10k", help="the dacl10k folder (kagglehub), for the leakage check")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--micro-arm", default="local-macos-x86")
    ap.add_argument("--publish", action="store_true", help="copy the ONNX summaries to public/results/*_dacl10k/")
    ap.add_argument("--skip-codebrim", action="store_true", help="synthetic set only (scripts/install_model.sh's quick check)")
    a = ap.parse_args(argv)
    onnx = (REPO / a.onnx).resolve()
    fp16 = (REPO / a.fp16).resolve() if a.fp16 else onnx.with_name(onnx.name.replace(".onnx", ".fp16.onnx"))
    if not onnx.exists():
        raise SystemExit(f"{onnx} not found: run scripts/install_model.sh first")
    label, runs, arm = f"onnx:{onnx.name}", REPO / "runs" / "model-eval", f"{a.micro_arm}-dacl10k"
    plain = {k: v for k, v in os.environ.items() if k not in ("DRISHYAM_ONNX", "DRISHYAM_ONNX_FP16")}
    env = {**plain, "DRISHYAM_ONNX": str(onnx), **({"DRISHYAM_ONNX_FP16": str(fp16)} if fp16.exists() else {})}
    if a.dacl10k:
        subprocess.run([sys.executable, "-m", "evals.leakage", "--dacl10k", a.dacl10k, "--codebrim", a.codebrim,
                        "--out", str(runs / "leakage.json")], cwd=REPO, check=True)
    subprocess.run([sys.executable, "-m", "bench.micro", "--arm", arm, "--job-ops-only"], env=env, cwd=REPO, check=True)
    sets = [("synth", REPO / "data" / "synth200", REPO / "public" / "results")]
    if not a.skip_codebrim and (Path(a.codebrim) / "eval200.txt").exists():
        sets.insert(0, ("codebrim", Path(a.codebrim).resolve(), REPO / "public" / "results" / "codebrim"))
    elif not a.skip_codebrim:
        print(f"no {a.codebrim}/eval200.txt: CODEBRIM skipped (run `make data`)")
    for name, data, published in sets:
        after = check(harness(data, runs / f"onnx-{name}", a.n, env, arm, a.workers), label, runs / f"onnx-{name}")
        pub = json.loads((published / "summary.json").read_text()) if (published / "summary.json").exists() else None
        if pub and pub["n_images"] == a.n and pub.get("cost_model", {}).get("kind") == "fixed":
            before = check(pub, CLASSICAL, published / "summary.json")
        else:
            before = check(harness(data, runs / f"classical-{name}", a.n, plain, a.micro_arm, a.workers), CLASSICAL,
                           runs / f"classical-{name}")
        table(name, before, after)
        if a.publish:
            dst = REPO / "public" / "results" / f"{name}_dacl10k"
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(runs / f"onnx-{name}" / "summary.json", dst / "summary.json")
            print(f"published {dst.relative_to(REPO)}/summary.json")


if __name__ == "__main__":
    main()
