#!/usr/bin/env bash
# Build build/lambda/, the unzipped Lambda package for Python 3.12 on arm64, without Docker.
#
#   scripts/build_lambda.sh
#
# Dependencies come from uv.lock (runtime only, plus uvicorn at its locked version, which the dev group
# holds), as aarch64 wheels: the python3.12 runtime is Amazon Linux 2023 (glibc 2.34), and numpy 2.5.4
# ships no manylinux2014 aarch64 wheel, so the target platform is manylinux_2_28. Then the app itself:
# drishyam/, every models/*.onnx (the two privacy models, and the dacl10k segmenter once installed) and run.sh.
# public/ is not in it: CloudFront serves the static site from S3 and sends only /api/* to the function.
# Fails if the result would pass Lambda's 250 MB unzipped limit.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=build/lambda
LIMIT=262144000

rm -rf "$OUT"
mkdir -p "$OUT/models"
uv export --frozen --no-dev --no-hashes --no-emit-project --quiet -o build/requirements.txt
uv export --frozen --all-groups --all-extras --no-hashes --no-emit-project --quiet -o build/constraints.txt
uv pip install --quiet --python-platform aarch64-manylinux_2_28 --python-version 3.12 --only-binary :all: \
  --target "$OUT" -r build/requirements.txt -c build/constraints.txt uvicorn

cp -R drishyam "$OUT/"
cp models/*.onnx models/LICENSE.* models/README.md "$OUT/models/"
install -m 755 infra/run.sh "$OUT/run.sh"
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +

BYTES=$(find "$OUT" -type f -exec cat {} + | wc -c | tr -d ' ')
echo "build/lambda: $(find "$OUT" -type f | wc -l | tr -d ' ') files, $BYTES bytes unzipped (limit $LIMIT)"
[ "$BYTES" -lt "$LIMIT" ] || { echo "over Lambda's 250 MB unzipped limit" >&2; exit 1; }
