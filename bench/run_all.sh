#!/usr/bin/env bash
# Micro + macro + parity-save for one benchmark arm, into results/<arm>/.
#   bench/run_all.sh <arm>
# Environment (all optional): PYTHON (interpreter; on the COOL arm the one from the /opt/cool venv), DATA (default
# data/synth200), N (images, 200), PIXEL_CAP (0.3), THREADS (cv2 threads per worker, 1), WORKERS (space-separated
# worker counts for the macro run, "1"; e.g. "1 2 4 8" for the scaling sweep), USD_PER_HOUR (0.339),
# PARITY_INPUTS (another arm's parity.npz, so both arms segment identical crops), DRISHYAM_ON_EC2=1 (record IMDSv2
# instance metadata in the provenance block; nothing else ever calls IMDS).
set -euo pipefail

arm="${1:?usage: bench/run_all.sh <arm-name>}"
cd "$(dirname "$0")/.."
py="${PYTHON:-.venv/bin/python}"
data="${DATA:-data/synth200}"
out="results/$arm"
mkdir -p "$out"

if [ ! -f "$data/annotations.json" ]; then
  echo "== building $data (about 4 minutes)"
  "$py" -m evals.synth dataset --out "$data" --n "${N:-200}" --seed 7
fi

echo "== micro ($arm)"
"$py" -m bench.micro --arm "$arm" --threads "${THREADS:-1}"

for w in ${WORKERS:-1}; do
  echo "== macro ($arm, $w worker(s))"
  "$py" -m bench.macro --arm "$arm" --data "$data" --n "${N:-200}" --pixel-cap "${PIXEL_CAP:-0.3}" \
    --workers "$w" --threads "${THREADS:-1}" --usd-per-hour "${USD_PER_HOUR:-0.339}"
done

echo "== parity ($arm)"
"$py" -m bench.parity --save "$out/parity.npz" ${PARITY_INPUTS:+--inputs "$PARITY_INPUTS"}

echo "done: $out"
