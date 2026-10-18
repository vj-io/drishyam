#!/usr/bin/env bash
# Fold the Kaggle-trained dacl10k segmenter back into the repo (training/README.md, "Kaggle in 6 steps"):
#
#   scripts/install_model.sh ~/Downloads/drishyam_dacl10k.zip [--n 200]
#   scripts/install_model.sh ~/Downloads/drishyam_dacl10k [--n 200]     (the folder Safari expands the zip into)
#   scripts/install_model.sh best.onnx [best_fp16.onnx] [--n 200]
#
# The zip (the notebook's output, wherever the browser saved it) is unpacked into a fresh temporary folder.
# 1. checks both files on OpenCV 5 DNN ENGINE_NEW (6-class output at 256 and 512) and through OnnxSegmenter
#    (FP16 against FP32 too: over 5e-2, the FP16 file is dropped and FP32 installed alone), before touching the repo;
# 2. copies them to models/drishyam_dacl10k.onnx and .fp16.onnx (and the metrics.json and run_info.json found beside
#    best.onnx to models/drishyam_dacl10k.metrics.json and .run_info.json), and records sizes, sha256, validation mIoU,
#    the training run's package versions and source (or, without run_info.json, the hand-recorded
#    models/drishyam_dacl10k.provenance.json when its sha256 matches), and the licence in models/README.md
#    (README.md's own Licence section is hand-written and left alone);
# 3. runs 3 sample images through `drishyam run` with DRISHYAM_ONNX set (runs/model-smoke/);
# 4. runs evals.model_eval on --n images of data/synth200: this model (fixed cost model, its own job_ops timing)
#    against the classical engine, into runs/model-eval/ (never public/results; `make eval-model` adds the
#    leakage check and CODEBRIM and publishes), and prints the before/after table. 200 is the published eval;
# 5. prints the git and scripts/aws.sh commands that would make the live site use the model. It does not run them.
#
# PY defaults to .venv/bin/python (the repo venv: OpenCV 5, no torch needed).
set -euo pipefail

usage() { echo "usage: scripts/install_model.sh drishyam_dacl10k.zip | its unpacked folder | best.onnx [best_fp16.onnx]  [--n N]" >&2; exit 2; }
abspath() { [ -e "$1" ] || { echo "no such file or folder: $1" >&2; exit 2; }; echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }
N=200 MODEL="" FP16=""
while [ $# -gt 0 ]; do
  case $1 in
    --n) [ $# -ge 2 ] || usage; N=$2; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    -*) usage ;;
    *) if [ -z "$MODEL" ]; then MODEL=$(abspath "$1"); elif [ -z "$FP16" ]; then FP16=$(abspath "$1"); else usage; fi
       shift ;;
  esac
done
[ -n "$MODEL" ] || usage
[[ $N =~ ^[1-9][0-9]*$ ]] || usage
if [[ $MODEL == *.zip ]]; then
  [ -z "$FP16" ] || usage
  UNZIPPED=$(mktemp -d "${TMPDIR:-/tmp}/drishyam-model.XXXXXX")
  trap 'rm -rf "$UNZIPPED"' EXIT
  unzip -o -q "$MODEL" -d "$UNZIPPED"
  MODEL=$UNZIPPED
fi
if [ -d "$MODEL" ]; then  # the unpacked zip, ours or the browser's
  [ -z "$FP16" ] || usage
  [ -f "$MODEL/best.onnx" ] || { echo "no best.onnx in $MODEL" >&2; exit 2; }
  if [ -f "$MODEL/best_fp16.onnx" ]; then FP16="$MODEL/best_fp16.onnx"; fi
  MODEL="$MODEL/best.onnx"
fi

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
# This checkout's drishyam, even when the venv's editable install points at another checkout.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
DST=models/drishyam_dacl10k.onnx DST16=models/drishyam_dacl10k.fp16.onnx

echo "== 1/5 OpenCV DNN load and OnnxSegmenter"
RC=0
"$PY" - "$MODEL" "$FP16" <<'EOF' || RC=$?
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
print(f"drishyam imported from {os.path.dirname(drishyam.__file__)}")
if fp16:
    d = np.abs(OnnxSegmenter(fp16).predict(crop).probs - seg.predict(crop).probs).max()
    print(f"FP16 vs FP32 at 512: max |dprob| {d:.2e} (tolerance 5e-2, training/parity.py's)")
    if d > 5e-2:
        print("FP16 over tolerance: installing the FP32 file alone (Drishyam's fast looks then use it too)")
        sys.exit(3)
EOF
if [ "$RC" -eq 3 ]; then FP16=""; elif [ "$RC" -ne 0 ]; then exit "$RC"; fi

echo "== 2/5 install into models/"
cp "$MODEL" "$DST"
if [ -n "$FP16" ]; then cp "$FP16" "$DST16"; else rm -f "$DST16"; fi
METRICS="$(dirname "$MODEL")/metrics.json"
if [ -f "$METRICS" ]; then cp "$METRICS" models/drishyam_dacl10k.metrics.json; else rm -f models/drishyam_dacl10k.metrics.json; fi
RUNINFO="$(dirname "$MODEL")/run_info.json"
if [ -f "$RUNINFO" ]; then cp "$RUNINFO" models/drishyam_dacl10k.run_info.json; else rm -f models/drishyam_dacl10k.run_info.json; fi
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
    b = max(hist, key=lambda r: r["val_miou5"])  # the epoch train_dacl10k.py saved as best.pt
    merged = ", ".join(f"{n} {v:.3f}" for n, v in zip(("crack", "spalling", "efflorescence", "exposed_rebar", "corrosion"), b["iou5"]))
    score = (f"Epoch {b['epoch']} of {len(hist)}, chosen on the mean validation IoU of Drishyam's five defect classes: "
             f"**{b['val_miou5']:.3f}** ({merged}). That is the split that chose the epoch, so the number is "
             "optimistic, and no held-out number exists (dacl10k's test labels are withheld). The merged classes are "
             "unions (`training/class_map.json`): crack is dacl10k Crack or ACrack (alligator crack), whose own IoUs at "
             f"that epoch are {b['iou19']['Crack']:.3f} and {b['iou19']['ACrack']:.3f}. Validation mIoU over the 19 "
             f"dacl10k classes at that epoch: {b['val_miou19']:.3f}. For reference, not as a like-for-like comparison: "
             "the dacl10k paper's DeepLabv3+ with a MobileNetV3-Large encoder and no auxiliary loss scores 0.320 on "
             "the validation split ([arXiv:2309.00460](https://arxiv.org/abs/2309.00460), Table 3). Per-epoch "
             "numbers: `drishyam_dacl10k.metrics.json`.")
rp = Path("models/drishyam_dacl10k.run_info.json")
run = "No `run_info.json` came with the weights (notebooks built before it was added do not write one)."
pp = Path("models/drishyam_dacl10k.provenance.json")  # hand-recorded, for weights that came without run_info.json
if not rp.exists() and pp.exists():
    prov = json.loads(pp.read_text())
    if prov["sha256"]["drishyam_dacl10k.onnx"] == hashlib.sha256(Path("models/drishyam_dacl10k.onnx").read_bytes()).hexdigest():
        run += "\n\n" + prov["known"]
if rp.exists():
    sessions = json.loads(rp.read_text())
    last = sessions[-1]  # the session that exported the weights
    v = ", ".join(f"{k} {val}" for k, val in last["versions"].items() if val)
    src = f"git `{last['git_sha']}`" if last.get("git_sha") else "no git checkout (Kaggle)"
    run = (f"Trained with Python {last['python']}, {v} ({len(sessions)} session(s); every one is in "
           f"`drishyam_dacl10k.run_info.json`). Source: {src}; notebook sources sha256 `{last['source_sha256']}` "
           "(`training/make_notebook.py`'s `source_sha256()` at the commit the notebook was built from).")
section = f"""{head}

Fine-tuned on dacl10k with `training/drishyam_dacl10k_kaggle.ipynb` (UNet, timm MobileNetV3-Large encoder) and installed
{datetime.date.today()} by `scripts/install_model.sh`. Input `(1, 3, H, W)` ImageNet-normalised RGB, output `(1, 6, H, W)`
logits in `drishyam.segment.CLASSES` order (the 19 -> 6 merge is in the graph); `DRISHYAM_ONNX` / `DRISHYAM_ONNX_FP16`
select it (`training/README.md`).

| File | Size | sha256 |
| --- | --- | --- |
{rows}
{score}

{run}

**Licence: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), non-commercial; not the permissive licences
above and not the repo's MIT.** The weights are adapted from (trained on) [dacl10k](https://github.com/phiyodr/dacl10k-toolkit)
(Flotzinger et al., WACV 2024, [arXiv:2309.00460](https://arxiv.org/abs/2309.00460); CC BY-NC 4.0), so they carry its
non-commercial terms and its attribution (`LICENSE.drishyam_dacl10k`). No dacl10k images or labels are included. The
ImageNet starting point (timm `mobilenetv3_large_100`) is Apache 2.0.
"""
text = readme.read_text().split("\n" + head)[0].rstrip("\n")
text = text.replace("# Bundled models (third-party, permissive licences)",
                    "# Bundled models (third-party; permissive licences, except the CC BY-NC 4.0 `drishyam_dacl10k`)")
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

echo "== 4/5 eval: $N images of data/synth200, this model vs classical (runs/model-eval/; the full run is make eval-model)"
if [ ! -f data/synth200/annotations.json ]; then  # build all 200 first: the harness would build only N
  echo "building data/synth200 (200 images, about 4 minutes)"
  "$PY" -m evals.synth dataset --out data/synth200 --n 200 --seed 7
fi
"$PY" -m evals.model_eval --onnx "$DST" --n "$N" --skip-codebrim

echo "== 5/5 to make the live site use it (not run here; the weights are CC BY-NC 4.0, see models/README.md)"
ADD="$DST models/README.md"
for f in "$DST16" models/drishyam_dacl10k.metrics.json models/drishyam_dacl10k.run_info.json; do  # written, or removed after an earlier commit
  if [ -f "$f" ] || git ls-files --error-unmatch "$f" >/dev/null 2>&1; then ADD="$ADD $f"; fi
done
echo "  git add $ADD"
echo "  git commit -m \"models: dacl10k segmenter\""
if [ -n "$FP16" ]; then echo "  ONNX=$DST ONNX_FP16=$DST16 scripts/aws.sh deploy"
else echo "  ONNX=$DST scripts/aws.sh deploy"; fi
echo "scripts/build_lambda.sh packages every models/*.onnx, and ONNX (ONNX_FP16) sets DRISHYAM_ONNX (DRISHYAM_ONNX_FP16)"
echo "on the function; a deploy without them puts the classical engine back. Then expect"
echo "\"segmenter\":\"onnx:drishyam_dacl10k.onnx\" from:"
echo "  curl -s https://d1xg58guluup8c.cloudfront.net/api/health"
echo
echo "To undo later: training/README.md, section 7 (not printed here, so pasting all of the above never runs it)."
