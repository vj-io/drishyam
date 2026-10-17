#!/usr/bin/env bash
# Fold the Kaggle-trained dacl10k segmenter back into the repo (training/README.md, "Kaggle in 6 steps"):
#
#   scripts/install_model.sh best.onnx [best_fp16.onnx] [--n 200]
#
# 1. checks both files on OpenCV 5 DNN ENGINE_NEW (6-class output at 256 and 512) and through OnnxSegmenter
#    (FP16 against FP32 too), before touching the repo;
# 2. copies them to models/drishyam_dacl10k.onnx and .fp16.onnx (and a metrics.json found beside best.onnx to
#    models/drishyam_dacl10k.metrics.json), and records sizes, sha256, validation mIoU and the licence in
#    models/README.md;
# 3. runs 3 sample images through `drishyam run` with DRISHYAM_ONNX set (runs/model-smoke/);
# 4. runs `make eval` on --n images of data/synth200 twice, classical then this model, into runs/model-eval/
#    (never public/results), and prints the before/after recall table. 200 is the published eval;
# 5. prints the Vercel commands that would make the live site use the model. It does not run them.
#
# PY defaults to .venv/bin/python (the repo venv: OpenCV 5, no torch needed).
set -euo pipefail

usage() { echo "usage: scripts/install_model.sh best.onnx [best_fp16.onnx] [--n N]" >&2; exit 2; }
abspath() { [ -f "$1" ] || { echo "no such file: $1" >&2; exit 2; }; echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }
N=200 MODEL="" FP16=""
while [ $# -gt 0 ]; do
  case $1 in
    --n) [ $# -ge 2 ] || usage; N=$2; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    -*) usage ;;
    *) if [ -z "$MODEL" ]; then MODEL=$(abspath "$1"); elif [ -z "$FP16" ]; then FP16=$(abspath "$1"); else usage; fi
       shift ;;
  esac
done
[ -n "$MODEL" ] || usage
[[ $N =~ ^[1-9][0-9]*$ ]] || usage

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
# This checkout's drishyam, even when the venv's editable install points at another checkout.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
DST=models/drishyam_dacl10k.onnx DST16=models/drishyam_dacl10k.fp16.onnx

echo "== 1/5 OpenCV DNN load and OnnxSegmenter"
"$PY" - "$MODEL" "$FP16" <<'EOF'
import os, sys
import cv2
import numpy as np
import drishyam
from drishyam.segment import CLASSES, OnnxSegmenter

fp32, fp16 = sys.argv[1], sys.argv[2] or None
for p in filter(None, (fp32, fp16)):
    net = cv2.dnn.readNetFromONNX(p, engine=cv2.dnn.ENGINE_NEW)
    for s in (256, 512):
        net.setInput(np.zeros((1, 3, s, s), np.float32))
        shape = tuple(np.asarray(net.forward()).shape)
        assert shape == (1, len(CLASSES), s, s), f"{p} at {s}: output {shape}, Drishyam needs (1, 6, {s}, {s})"
    print(f"OpenCV {cv2.__version__} DNN ENGINE_NEW loads {os.path.basename(p)}: (1, 6, H, W) at 256 and 512")
crop = cv2.imread("public/samples/bridge_01.jpg")[:512, :512].copy()
seg = OnnxSegmenter(fp32, fp16)
for mode in ("accurate", "fast"):
    s = seg.predict(crop, mode)
    assert s.probs.shape == (6, 512, 512) and np.allclose(s.probs.sum(0), 1, atol=1e-3), mode
    print(f"OnnxSegmenter {mode}: {s.latency_ms:.0f} ms, labelled px {({k: v for k, v in s.areas.items() if v})}")
if fp16:
    d = np.abs(OnnxSegmenter(fp16).predict(crop).probs - seg.predict(crop).probs).max()
    assert d <= 5e-2, f"FP16 vs FP32 max |dprob| {d:.3g} > 5e-2 (training/parity.py's FP16 tolerance)"
    print(f"FP16 vs FP32 at 512: max |dprob| {d:.2e} (tolerance 5e-2)")
print(f"drishyam imported from {os.path.dirname(drishyam.__file__)}")
EOF

echo "== 2/5 install into models/"
cp "$MODEL" "$DST"
if [ -n "$FP16" ]; then cp "$FP16" "$DST16"; else rm -f "$DST16"; fi
METRICS="$(dirname "$MODEL")/metrics.json"
if [ -f "$METRICS" ]; then cp "$METRICS" models/drishyam_dacl10k.metrics.json; else rm -f models/drishyam_dacl10k.metrics.json; fi
"$PY" - <<'EOF'
import datetime, hashlib, json
from pathlib import Path

readme, head = Path("models/README.md"), "## Trained segmenter: `drishyam_dacl10k`"
rows = "".join(f"| `{p.name}` | {p.stat().st_size / 1e6:.1f} MB | `{hashlib.sha256(p.read_bytes()).hexdigest()}` |\n"
               for p in (Path("models/drishyam_dacl10k.onnx"), Path("models/drishyam_dacl10k.fp16.onnx")) if p.exists())
mp = Path("models/drishyam_dacl10k.metrics.json")
score = "No `metrics.json` came with the weights."
if mp.exists():
    hist = json.loads(mp.read_text())
    b = max(hist, key=lambda r: r["val_miou19"])
    merged = ", ".join(f"{n} {v:.3f}" for n, v in zip(("crack", "spalling", "efflorescence", "exposed_rebar", "corrosion"), b["iou5"]))
    score = (f"Validation mIoU over the 19 dacl10k classes: **{b['val_miou19']:.3f}**, best of {len(hist)} epochs (epoch "
             f"{b['epoch']}; the paper's MobileNetV3 baseline: 0.320). After the merge to Drishyam's classes: {merged}. "
             "Per-epoch numbers: `drishyam_dacl10k.metrics.json`.")
section = f"""{head}

Fine-tuned on dacl10k with `training/drishyam_dacl10k_kaggle.ipynb` (UNet, timm MobileNetV3-Large encoder) and installed
{datetime.date.today()} by `scripts/install_model.sh`. Input `(1, 3, H, W)` ImageNet-normalised RGB, output `(1, 6, H, W)`
logits in `drishyam.segment.CLASSES` order (the 19 -> 6 merge is in the graph); `DRISHYAM_ONNX` / `DRISHYAM_ONNX_FP16`
select it (`training/README.md`).

| File | Size | sha256 |
| --- | --- | --- |
{rows}
{score}

**Licence: CC BY-NC 4.0, non-commercial; not the permissive licences above and not the repo's MIT.** The weights are
trained on dacl10k (Flotzinger et al., WACV 2024; CC BY-NC 4.0), so they carry its non-commercial terms and need its
attribution. The ImageNet starting point (timm `mobilenetv3_large_100`) is Apache 2.0.
"""
text = readme.read_text().split("\n" + head)[0].rstrip("\n")
readme.write_text(text + "\n\n" + section)
print(f"models/README.md: {head[3:]} section written")
EOF
export DRISHYAM_ONNX="$PWD/$DST"
if [ -n "$FP16" ]; then export DRISHYAM_ONNX_FP16="$PWD/$DST16"; else unset DRISHYAM_ONNX_FP16; fi
ls -l models/drishyam_dacl10k*

echo "== 3/5 smoke: 3 jobs with DRISHYAM_ONNX=$DST"
SMOKE=runs/model-smoke
mkdir -p "$SMOKE"
for i in 01 02 03; do
  "$PY" -m drishyam.cli run "public/samples/bridge_$i.jpg" --out "$SMOKE" 2>"$SMOKE/bridge_$i.log" \
    || { cat "$SMOKE/bridge_$i.log" >&2; echo "smoke job on bridge_$i.jpg failed" >&2; exit 1; }
done

echo "== 4/5 eval: $N images of data/synth200, classical then this model (runs/model-eval/)"
if [ ! -f data/synth200/annotations.json ]; then  # build all 200 first: the harness would build only N
  echo "building data/synth200 (200 images, about 4 minutes)"
  "$PY" -c 'from pathlib import Path; from evals import synth; synth.build_dataset(Path("data/synth200"), n=200, seed=7)'
fi
EVAL=runs/model-eval
env -u DRISHYAM_ONNX -u DRISHYAM_ONNX_FP16 \
  make eval PY="$PY" DATA=data/synth200 N="$N" OUT="$EVAL/classical" RAW="$EVAL/classical/raw"
make eval PY="$PY" DATA=data/synth200 N="$N" OUT="$EVAL/onnx" RAW="$EVAL/onnx/raw"
"$PY" - "$EVAL/classical/summary.json" "$EVAL/onnx/summary.json" <<'EOF'
import json, sys
from evals.harness import BUDGETS, HEADLINE_BUDGET, TILE_LEVELS

before, after = (json.load(open(p)) for p in sys.argv[1:3])
assert after["segmenter"].startswith("onnx:"), f"the model run used {after['segmenter']}, not the ONNX"
f = lambda v: "-" if v is None else f"{v:.3f}"
d = lambda a, b: "" if None in (a, b) else f"{a - b:+.3f}"
print(f"\nrecall on {after['dataset']}: {before['segmenter']} (before) -> {after['segmenter']} (after)")
print(f"{'policy':<16}{'pixels':>8}{'before':>9}{'after':>9}{'change':>9}")
names = ["downscale", *(f"tiling@L{lv}" for lv in TILE_LEVELS), *(f"drishyam@{c}" for c in BUDGETS)]
points = [p for k in ("downscale", "tiling", "drishyam") for p in zip(before["policies"][k], after["policies"][k])]
for name, (b, a) in zip(names, points, strict=True):
    print(f"{name:<16}{a['pixels_frac']:>8.3f}{f(b['recall']):>9}{f(a['recall']):>9}{d(a['recall'], b['recall']):>9}")
key = f"drishyam@{HEADLINE_BUDGET}"
print(f"\nper class, {key}:")
for c, b in before["per_class"][key].items():
    a = after["per_class"][key][c]
    print(f"  {c:<14}{f(b):>9}{f(a):>9}{d(a, b):>9}")
print(f"\nheadline after: {after['headline']}")
EOF

echo "== 5/5 to make the live site use it (not run here; the weights are CC BY-NC 4.0, see models/README.md)"
ADD="$DST models/README.md"
[ -n "$FP16" ] && ADD="$ADD $DST16"
[ -f models/drishyam_dacl10k.metrics.json ] && ADD="$ADD models/drishyam_dacl10k.metrics.json"
echo "  git add $ADD"
echo "  git commit -m \"models: dacl10k segmenter\""
echo "  printf %s $DST | vercel env add DRISHYAM_ONNX production"
if [ -n "$FP16" ]; then echo "  printf %s $DST16 | vercel env add DRISHYAM_ONNX_FP16 production"
else echo "  vercel env rm DRISHYAM_ONNX_FP16 production   # only if an earlier install set it: that file is gone now"; fi
echo "  vercel deploy --prod        # or push, if the project deploys from git"
echo "  curl -s https://drishyam-seven.vercel.app/api/health    # expect \"segmenter\":\"onnx:drishyam_dacl10k.onnx\""
echo "The paths are relative on purpose: a Vercel Python function runs from the project root, where models/ is bundled."
echo "Undo: vercel env rm DRISHYAM_ONNX production${FP16:+ (and DRISHYAM_ONNX_FP16)}, then redeploy."
