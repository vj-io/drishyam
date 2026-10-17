#!/usr/bin/env bash
# Download the CODEBRIM original images (Mundt et al., CVPR 2019; DOI 10.5281/zenodo.2620293) from Zenodo,
# verify the checksum Zenodo publishes for the file, unpack to data/codebrim and pick the 200-image eval subset.
#
# 8.3 GB, non-commercial licence. Datasets are never committed (data/ is git-ignored). Nothing is fetched until
# you confirm; pass --yes (or YES=1) to skip the prompt.
set -euo pipefail
cd "$(dirname "$0")/.."

RECORD=2620293
FILE=CODEBRIM_original_images.zip
ZIP="data/$FILE"
DEST=data/codebrim
PYTHON="${PYTHON:-.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3

if [[ "${1:-}" != "--yes" && "${YES:-}" != "1" ]]; then
  echo "About to download $FILE (about 8.3 GB) from Zenodo record $RECORD (CODEBRIM, non-commercial licence)."
  echo "It goes to $ZIP and is unpacked to $DEST; nothing is committed to git."
  read -r -p "Continue? [y/N] " answer
  [[ "$answer" == [yY]* ]] || { echo "Aborted."; exit 1; }
fi

mkdir -p data
# Zenodo's record API lists each file's download link and md5 (accepts the legacy "filename" and the current "key").
read -r url md5 < <(curl -fsSL "https://zenodo.org/api/records/$RECORD" | "$PYTHON" -c '
import json, sys
name = sys.argv[1]
for f in json.load(sys.stdin)["files"]:
    if (f.get("key") or f.get("filename")) == name:
        links = f.get("links") or {}
        print(links.get("self") or links.get("download") or "", (f.get("checksum") or "").split(":")[-1])
        break
else:
    sys.exit(name + " is not listed in the Zenodo record")
' "$FILE")
[ -n "$md5" ] || { echo "Zenodo published no checksum for $FILE; refusing to continue." >&2; exit 1; }
[ -n "$url" ] || url="https://zenodo.org/records/$RECORD/files/$FILE?download=1"

curl -fL -C - -o "$ZIP" "$url"   # resumable

actual=$("$PYTHON" -c '
import hashlib, sys
h = hashlib.md5()
with open(sys.argv[1], "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
print(h.hexdigest())
' "$ZIP")
if [ "$actual" != "$md5" ]; then
  echo "Checksum mismatch for $ZIP: expected md5 $md5, got $actual. Not unpacking." >&2
  exit 1
fi
echo "md5 verified: $md5"

mkdir -p "$DEST"
unzip -q -n "$ZIP" -d "$DEST"
"$PYTHON" -m evals.codebrim --root "$DEST" --out "$DEST/eval200.txt"
echo "Done. Evaluate with: make eval DATA=$DEST"
