#!/usr/bin/env bash
# Fold the Kaggle-trained dacl10k segmenter back into the repo (training/README.md, "Kaggle in 6 steps"):
#
#   scripts/install_model.sh ~/Downloads/drishyam_dacl10k.zip [--n 200 | --no-eval]
#   scripts/install_model.sh ~/Downloads/drishyam_dacl10k [--n 200]     (the folder Safari expands the zip into)
#   scripts/install_model.sh best.onnx [best_fp16.onnx] [--n 200]
#
# The zip (the notebook's output, wherever the browser saved it) is unpacked into a fresh temporary folder.
# NAME: drishyam_dacl10k for a v1 run (or bare .onnx files), drishyam_dacl10k_<variant> for a recipe v2 run (its
# run_info.json says which), so v2a and v2b install beside v1 and the three can be compared.
# 1. checks the files on OpenCV 5 DNN ENGINE_NEW (6-class output at 256 and 512; best_19.onnx 19-class) and through
#    OnnxSegmenter (FP16 against FP32 too: over 5e-2, the FP16 file is dropped and FP32 installed alone), before
#    touching the repo;
# 2. copies them to models/NAME.onnx and .fp16.onnx (best_19.onnx to models/eval/NAME_19.onnx, which
#    scripts/build_lambda.sh does not package; the metrics.json and run_info.json found beside best.onnx to
#    models/NAME.metrics.json and .run_info.json), and records sizes, sha256, validation mIoU, the training run's
#    package versions, flags and source (or, without run_info.json, the hand-recorded models/NAME.provenance.json when
#    its sha256 matches), and the licence in models/README.md's NAME section (README.md's own Licence section is
#    hand-written and left alone);
# 3. runs 3 sample images through `drishyam run` with DRISHYAM_ONNX set (runs/model-smoke/NAME/);
# 4. runs evals.model_eval on --n images of data/synth200: this model (fixed cost model, its own job_ops timing)
#    against the classical engine, into runs/model-eval/ (a v2 model's own folders and cost table; never
#    public/results; `make eval-model` adds the leakage check and CODEBRIM and publishes), and prints the before/after
#    table. 200 is the published eval; --no-eval skips this step (installing before dev-only tuning, so the
#    held-out set is read once, at the end);
# 5. prints the git and scripts/aws.sh commands that would make the live site use the model. It does not run them.
#
# PY defaults to .venv/bin/python (the repo venv: OpenCV 5, no torch needed).
set -euo pipefail

usage() { echo "usage: scripts/install_model.sh drishyam_dacl10k.zip | its unpacked folder | best.onnx [best_fp16.onnx]  [--n N | --no-eval]" >&2; exit 2; }
abspath() { [ -e "$1" ] || { echo "no such file or folder: $1" >&2; exit 2; }; echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }
N=200 EVAL=1 MODEL="" FP16=""
while [ $# -gt 0 ]; do
  case $1 in
    --n) [ $# -ge 2 ] || usage; N=$2; shift 2 ;;
    --no-eval) EVAL=0; shift ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
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
F19="$(dirname "$MODEL")/best_19.onnx"
[ -f "$F19" ] || F19=""

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
# This checkout's drishyam, even when the venv's editable install points at another checkout.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# The name the files are installed under, from the training run's own record (the last session's flags).
MODEL_NAME=$("$PY" -c '
import json, re, sys
from pathlib import Path
p = Path(sys.argv[1])
flags = json.loads(p.read_text())[-1].get("flags", {}) if p.exists() else {}
variant = flags.get("variant") or "v2"
if flags.get("recipe") == "v2" and not re.fullmatch(r"[a-z0-9]+", variant):
    sys.exit(f"{p}: variant {variant!r} is not a plain name")
print(f"drishyam_dacl10k_{variant}" if flags.get("recipe") == "v2" else "drishyam_dacl10k")
' "$(dirname "$MODEL")/run_info.json")
export MODEL_NAME
echo "installing as models/$MODEL_NAME.onnx"
DST=models/$MODEL_NAME.onnx DST16=models/$MODEL_NAME.fp16.onnx DST19=models/eval/${MODEL_NAME}_19.onnx

echo "== 1/5 OpenCV DNN load and OnnxSegmenter"
RC=0
"$PY" - "$MODEL" "$FP16" "$F19" <<'EOF' || RC=$?
import os, sys
import cv2
import numpy as np
import drishyam
from drishyam.segment import CLASSES, OnnxSegmenter

fp32, fp16, f19 = sys.argv[1], sys.argv[2] or None, sys.argv[3] or None
for p, c in ((fp32, len(CLASSES)), (fp16, len(CLASSES)), (f19, 19)):
    if not p:
        continue
    net = cv2.dnn.readNetFromONNX(p, engine=cv2.dnn.ENGINE_NEW)
    for s in (256, 512):
        net.setInput(np.zeros((1, 3, s, s), np.float32))
        shape = tuple(np.asarray(net.forward()).shape)
        assert shape == (1, c, s, s), f"{p} at {s}: output {shape}, expected (1, {c}, {s}, {s})"
    print(f"OpenCV {cv2.__version__} DNN ENGINE_NEW loads {os.path.basename(p)}: (1, {c}, H, W) at 256 and 512")
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
if [ -n "$F19" ]; then mkdir -p models/eval && cp "$F19" "$DST19"; else rm -f "$DST19"; fi
METRICS="$(dirname "$MODEL")/metrics.json"
if [ -f "$METRICS" ]; then cp "$METRICS" "models/$MODEL_NAME.metrics.json"; else rm -f "models/$MODEL_NAME.metrics.json"; fi
RUNINFO="$(dirname "$MODEL")/run_info.json"
if [ -f "$RUNINFO" ]; then cp "$RUNINFO" "models/$MODEL_NAME.run_info.json"; else rm -f "models/$MODEL_NAME.run_info.json"; fi
"$PY" - <<'EOF'
import datetime, hashlib, json, os
from pathlib import Path

name = os.environ.get("MODEL_NAME", "drishyam_dacl10k")  # scripts/install_model.sh exports it
m = Path("models")
readme, head = m / "README.md", f"## Trained segmenter: `{name}`"
rows = "".join(f"| `{p.relative_to(m)}` | {p.stat().st_size / 1e6:.1f} MB | `{hashlib.sha256(p.read_bytes()).hexdigest()}` |\n"
               for p in (m / f"{name}.onnx", m / f"{name}.fp16.onnx", m / "eval" / f"{name}_19.onnx") if p.exists())
rp = m / f"{name}.run_info.json"
sessions = json.loads(rp.read_text()) if rp.exists() else []
flags = sessions[-1].get("flags", {}) if sessions else {}
mp = m / f"{name}.metrics.json"
score = "No `metrics.json` came with the weights."
if mp.exists():
    hist = json.loads(mp.read_text())
    key = "scale_mean" if hist[-1].get("select") == "scales" else "val_miou5"
    b = max(hist, key=lambda r: r[key])  # the epoch train_dacl10k.py saved as best.pt
    merged = ", ".join(f"{n} {v:.3f}" for n, v in zip(("crack", "spalling", "efflorescence", "exposed_rebar", "corrosion"), b["iou5"]))
    if key == "scale_mean":
        views = ", ".join(f"{s if s == 'whole' else s + ' px'} {v['miou5']:.3f}" for s, v in b["scales"].items())
        sides = ", ".join(k for k in b["scales"] if k != "whole")
        chosen = (f"chosen on the mean validation IoU of Drishyam's five defect classes over {len(b['scales'])} views "
                  f"(square crops of {sides} source px, and the whole photo): **{b['scale_mean']:.3f}** ({views}); on "
                  f"whole photos alone {b['val_miou5']:.3f} ({merged})")
    else:
        chosen = f"chosen on the mean validation IoU of Drishyam's five defect classes: **{b['val_miou5']:.3f}** ({merged})"
    score = (f"Epoch {b['epoch']} of {len(hist)}, {chosen}. That is the split that chose the epoch, so the number is "
             "optimistic, and no held-out number exists (dacl10k's test labels are withheld). The merged classes are "
             "unions (`training/class_map.json`): crack is dacl10k Crack or ACrack (alligator crack), whose own IoUs at "
             f"that epoch are {b['iou19']['Crack']:.3f} and {b['iou19']['ACrack']:.3f}. Validation mIoU over the 19 "
             f"dacl10k classes at that epoch: {b['val_miou19']:.3f}. For reference, not as a like-for-like comparison: "
             "the dacl10k paper's DeepLabv3+ with a MobileNetV3-Large encoder and no auxiliary loss scores 0.320 on "
             "the validation split ([arXiv:2309.00460](https://arxiv.org/abs/2309.00460), Table 3). Per-epoch "
             f"numbers: `{name}.metrics.json`.")
run = "No `run_info.json` came with the weights (notebooks built before it was added do not write one)."
pp = m / f"{name}.provenance.json"  # hand-recorded, for weights that came without run_info.json
if not rp.exists() and pp.exists():
    prov = json.loads(pp.read_text())
    if prov["sha256"][f"{name}.onnx"] == hashlib.sha256((m / f"{name}.onnx").read_bytes()).hexdigest():
        run += "\n\n" + prov["known"]
if sessions:
    last = sessions[-1]  # the session that exported the weights
    v = ", ".join(f"{k} {val}" for k, val in last["versions"].items() if val)
    host = "Colab" if last["flags"].get("out", "").startswith("/content/") else "Kaggle"  # Colab writes to its Drive mount
    src = f"git `{last['git_sha']}`" if last.get("git_sha") else f"no git checkout ({host})"
    run = (f"Trained with Python {last['python']}, {v} ({len(sessions)} session(s); every one is in "
           f"`{name}.run_info.json`). Source: {src}; notebook sources sha256 `{last['source_sha256']}` "
           "(`training/make_notebook.py`'s `source_sha256()` at the commit the notebook was built from).")
    if "select_rule" in last:
        run += f" Training flags: that file's `flags`; {last['select_rule']}."
enc = flags.get("encoder", "tu-mobilenetv3_large_100").removeprefix("tu-")
if flags.get("recipe") == "v2":
    variant = flags.get("variant") or "v2"
    nb = "drishyam_dacl10k_kaggle.ipynb" if variant in ("v2", "v2a") else f"drishyam_dacl10k_kaggle_{variant}.ipynb"
    made = (f"Fine-tuned on dacl10k with `training/{nb}` (recipe v2, variant {variant}: {flags['arch']}, timm `{enc}` "
            "encoder)")
else:
    made = "Fine-tuned on dacl10k with `training/drishyam_dacl10k_kaggle.ipynb` (UNet, timm MobileNetV3-Large encoder)"
section = f"""{head}

{made} and installed
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
ImageNet starting point (timm `{enc}`) is Apache 2.0.
"""
text = readme.read_text()
at = text.find("\n" + head)
if at >= 0:  # replace this model's own section, keep the others (v1 and the v2 variants live side by side)
    nxt = text.find("\n## ", at + 1)
    text = text[:at] + (text[nxt:] if nxt >= 0 else "")
text = text.rstrip("\n").replace("# Bundled models (third-party, permissive licences)",
                                 "# Bundled models (third-party; permissive licences, except the CC BY-NC 4.0 `drishyam_dacl10k`)")
readme.write_text(text + "\n\n" + section)
print(f"models/README.md: {head[3:]} section written")
EOF
export DRISHYAM_ONNX="$PWD/$DST"
if [ -n "$FP16" ]; then export DRISHYAM_ONNX_FP16="$PWD/$DST16"; else unset DRISHYAM_ONNX_FP16; fi
ls -l "models/$MODEL_NAME".* ${F19:+"$DST19"}

echo "== 3/5 smoke: 3 jobs with DRISHYAM_ONNX=$DST"
SMOKE=runs/model-smoke/$MODEL_NAME
mkdir -p "$SMOKE"
for i in 01 02 03; do
  "$PY" -m drishyam.cli run "public/samples/bridge_$i.jpg" --out "$SMOKE" 2>"$SMOKE/bridge_$i.log" \
    || { cat "$SMOKE/bridge_$i.log" >&2; echo "smoke job on bridge_$i.jpg failed" >&2; exit 1; }
done

if [ "$EVAL" = 0 ]; then
  echo "== 4/5 eval: skipped (--no-eval)"
else
echo "== 4/5 eval: $N images of data/synth200, this model vs classical (runs/model-eval/; the full run is make eval-model)"
if [ ! -f data/synth200/annotations.json ]; then  # build all 200 first: the harness would build only N
  echo "building data/synth200 (200 images, about 4 minutes)"
  "$PY" -m evals.synth dataset --out data/synth200 --n 200 --seed 7
fi
"$PY" -m evals.model_eval --onnx "$DST" --n "$N" --skip-codebrim
fi

echo "== 5/5 to make the live site use it (not run here; the weights are CC BY-NC 4.0, see models/README.md)"
ADD="$DST models/README.md"
for f in "$DST16" "models/$MODEL_NAME.metrics.json" "models/$MODEL_NAME.run_info.json"; do  # written, or removed after an earlier commit
  if [ -f "$f" ] || git ls-files --error-unmatch "$f" >/dev/null 2>&1; then ADD="$ADD $f"; fi
done
echo "  git add $ADD"
echo "  git commit -m \"models: dacl10k segmenter ($MODEL_NAME)\""
if [ -n "$FP16" ]; then echo "  ONNX=$DST ONNX_FP16=$DST16 scripts/aws.sh deploy"
else echo "  ONNX=$DST scripts/aws.sh deploy"; fi
echo "scripts/build_lambda.sh packages every models/*.onnx, and ONNX (ONNX_FP16) sets DRISHYAM_ONNX (DRISHYAM_ONNX_FP16)"
echo "on the function; a deploy without them puts the classical engine back. Then expect"
echo "\"segmenter\":\"onnx:$MODEL_NAME.onnx\" from:"
echo "  curl -s https://d1xg58guluup8c.cloudfront.net/api/health"
N_MODELS=$(find models -maxdepth 1 -name 'drishyam_dacl10k*.onnx' ! -name '*.fp16.onnx' | wc -l | tr -d ' ')
if [ "$N_MODELS" -gt 1 ]; then
  echo "models/ holds $N_MODELS dacl10k segmenters, and build_lambda.sh packages them all: move the ones you are not"
  echo "deploying out of models/ first, or the package can pass Lambda's 250 MB limit (build_lambda.sh then stops)."
fi
echo
echo "To undo later: training/README.md, section 7 (not printed here, so pasting all of the above never runs it)."
