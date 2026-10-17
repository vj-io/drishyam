#!/bin/bash
# One benchmark arm on an instance prepared by bench/aws_user_data.sh; bench/aws_runbook.md sends it through SSM.
#   bash aws_arm.sh <A|B|C> <bucket> <git ref> <usd per hour> [cool python]
# Pulls the shared inputs from s3://<bucket>/drishyam-bench/inputs/, runs bench/run_all.sh <arm> and uploads
# results/<arm>/. Arm A uses the first Python under /opt/cool that imports cv2 (or the one given); B and C `uv sync`.
set -euo pipefail
arm=$1 bucket=$2 ref=$3 usd=$4 py=${5:-}
s3="s3://$bucket/drishyam-bench"
export HOME="${HOME:-/root}" PATH="/usr/local/bin:/snap/bin:$PATH"
cloud-init status --wait >/dev/null || true
[ -f /var/lib/drishyam-ready ] || { echo "first-boot setup did not finish:"; tail -40 /var/log/drishyam-setup.log; exit 1; }
cd /opt/drishyam
git fetch -q origin && git checkout -q --detach "$ref"
aws s3 sync --only-show-errors "$s3/inputs/synth200" data/synth200
aws s3 cp --only-show-errors "$s3/inputs/parity.npz" /opt/parity-inputs.npz
if [ "$arm" = A ]; then
  if [ -z "$py" ]; then
    for p in $(find /opt/cool -maxdepth 4 -path '*/bin/python3' 2>/dev/null); do
      if "$p" -c 'import cv2' 2>/dev/null; then py=$p; break; fi
    done
  fi
  [ -n "$py" ] || { echo "no Python under /opt/cool imports cv2; pass one as the 5th argument. /opt/cool:"
                    ls -la /opt/cool || true; exit 1; }
  "$py" -c 'import numpy' 2>/dev/null || "$py" -m pip install -q "numpy>=2.2"  # never pip-installs OpenCV
else
  uv sync -q --frozen --no-dev
  py=.venv/bin/python
fi
"$py" -c 'import cv2, numpy, sys; print("python", sys.executable, "| cv2", cv2.__version__, cv2.__file__, "| numpy", numpy.__version__)'
DRISHYAM_ON_EC2=1 PYTHON="$py" USD_PER_HOUR="$usd" WORKERS="1 2 4 8" PARITY_INPUTS=/opt/parity-inputs.npz \
  bash bench/run_all.sh "$arm"
aws s3 cp --only-show-errors --recursive "results/$arm" "$s3/results/$arm"
echo "uploaded results/$arm to $s3/results/$arm; this instance terminates itself in 15 minutes"
shutdown -h +15
