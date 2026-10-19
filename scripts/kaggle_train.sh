#!/usr/bin/env bash
# Train the dacl10k segmenter on Kaggle from a terminal (training/README.md, "Kaggle from the command line").
# <variant> is a folder of training/kaggle/ (v2a, v2b): its kernel-metadata.json names the kernel and its notebook.
#
#   scripts/kaggle_train.sh push <variant>            a new version of the kernel (its committed notebook, GPU T4,
#                                                     dacl10k attached), which Kaggle then runs
#   scripts/kaggle_train.sh push <variant> --resume   the same with the kernel's own latest output attached as input,
#                                                     so `--resume auto` carries on a run that stopped at --hours 11
#   scripts/kaggle_train.sh status <variant>          the latest version's status (queued, running, complete, error)
#   scripts/kaggle_train.sh output <variant>          download its drishyam_dacl10k.zip and log into a fresh
#                                                     runs/kaggle/<variant>/<time>/ of the main checkout (not
#                                                     a git worktree's) and print the install command
#
# The kaggle CLI (the one on PATH, else `uvx kaggle==2.2.4`) reads the owner's own Kaggle credentials itself;
# this script never reads, prints or passes them. Uploaded: the notebook (it embeds training/'s helpers and
# drishyam/segment.py) and its metadata, nothing else.
set -euo pipefail
cd "$(dirname "$0")/.."
usage() { sed -n '2,13p' "$0" >&2; exit 2; }
[ $# -ge 2 ] || usage
CMD=$1 V=$2
shift 2
META=training/kaggle/$V/kernel-metadata.json
[[ $V =~ ^[a-z0-9]+$ && -f $META ]] || { echo "no variant '$V' (training/kaggle/ has: $(ls training/kaggle | tr '\n' ' '))" >&2; exit 2; }
PY="${PY:-.venv/bin/python}"
# The kernel id, after checking that the metadata's notebook is this variant's (its VARIANT line).
KERNEL=$("$PY" -c '
import json, sys
from pathlib import Path
meta, variant = Path(sys.argv[1]), sys.argv[2]
m = json.loads(meta.read_text())
nb = (meta.parent / m["code_file"]).read_text()
cf = m["code_file"]
if f"VARIANT = \\\"{variant}\\\"" not in nb:
    sys.exit(f"{cf} is not the {variant} notebook (its VARIANT line differs)")
print(m["id"])
' "$META" "$V")
kaggle() {  # type -P finds the binary only: `command -v` would find this function
  if type -P kaggle >/dev/null; then command kaggle "$@"; else uvx "kaggle==2.2.4" "$@"; fi
}

case $CMD in
  push)
    [ $# -le 1 ] || usage
    "$PY" training/make_notebook.py --check  # never push a notebook older than training/train_dacl10k.py
    DIR=training/kaggle/$V
    if [ $# -eq 1 ]; then
      [ "$1" = --resume ] || usage
      DIR=$(mktemp -d "${TMPDIR:-/tmp}/drishyam-kaggle.XXXXXX")
      trap 'rm -rf "$DIR"' EXIT
      "$PY" -c '
import json, shutil, sys
from pathlib import Path
meta, out = Path(sys.argv[1]), Path(sys.argv[2])
m = json.loads(meta.read_text())
shutil.copy(meta.parent / m["code_file"], out / Path(m["code_file"]).name)
m["code_file"] = Path(m["code_file"]).name  # copied beside this metadata
m["kernel_sources"] = [m["id"]]  # this kernel latest output (run/last.pt) under /kaggle/input
(out / "kernel-metadata.json").write_text(json.dumps(m, indent=2))
' "$META" "$DIR"
    fi
    kaggle kernels push -p "$DIR"
    echo "https://www.kaggle.com/code/$KERNEL  (then: scripts/kaggle_train.sh status $V; a full run takes hours)"
    ;;
  status)
    [ $# -eq 0 ] || usage
    kaggle kernels status "$KERNEL"
    ;;
  output)
    [ $# -eq 0 ] || usage
    # the main checkout: a worktree (.claude/worktrees/...) is removed with its runs/ when its branch merges
    MAIN=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
    mkdir -p "$MAIN/runs/kaggle/$V"
    OUT=$MAIN/runs/kaggle/$V/$(date +%Y%m%d-%H%M%S)
    mkdir "$OUT"  # fresh: never mixes two versions' files
    kaggle kernels output "$KERNEL" -p "$OUT" --file-pattern 'drishyam_dacl10k\.zip$'
    ZIP=$(find "$OUT" -name drishyam_dacl10k.zip | head -1)
    if [ -z "$ZIP" ]; then
      echo "no drishyam_dacl10k.zip in $KERNEL's output (still running, or failed? its log, if any, is in $OUT;" \
           "scripts/kaggle_train.sh status $V)" >&2
      exit 1
    fi
    echo "downloaded $ZIP; to install it beside the v1 model (as models/drishyam_dacl10k_$V.onnx):"
    echo "  scripts/install_model.sh $ZIP"
    ;;
  *) usage ;;
esac
