#!/usr/bin/env bash
# Micro + macro + parity-save + kernels (Metric A) + agent loop (Metric B) for one benchmark arm, into results/<arm>/
#   bench/run_all.sh <arm>
# Environment (all optional): PYTHON (interpreter; on the COOL arm the one from the /opt/cool venv), DATA (default
# data/synth200), N (images, 200), PIXEL_CAP (0.3), THREADS (cv2 threads per worker, 1), WORKERS (space-separated
# worker counts for the macro run, "1"; e.g. "1 2 4 8" for the scaling sweep), USD_PER_HOUR (0.339),
# PARITY_INPUTS (another arm's parity.npz, so both arms segment identical crops), DRISHYAM_ON_EC2=1 (record IMDSv2
# instance metadata in the provenance block; nothing else ever calls IMDS), MAX_LOAD (micro and macro exit before
# timing if the 1-min load average is at or above it), BENCH_LABEL (where this ran, in words: provenance.label),
# KERNEL_THREADS (Metric A thread counts, "1,all") and LOOP_THREADS (Metric B, "1,2,4"; "1,all" on macOS, whose GCD
# backend cannot set 2 or 4).
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
"$py" -m bench.micro --arm "$arm" --threads "${THREADS:-1}" ${MAX_LOAD:+--max-load "$MAX_LOAD"}

for w in ${WORKERS:-1}; do
  echo "== macro ($arm, $w worker(s))"
  "$py" -m bench.macro --arm "$arm" --data "$data" --n "${N:-200}" --pixel-cap "${PIXEL_CAP:-0.3}" \
    --workers "$w" --threads "${THREADS:-1}" --usd-per-hour "${USD_PER_HOUR:-0.339}" ${MAX_LOAD:+--max-load "$MAX_LOAD"}
done

echo "== parity ($arm)"
"$py" -m bench.parity --save "$out/parity.npz" ${PARITY_INPUTS:+--inputs "$PARITY_INPUTS"}

echo "== kernels, Metric A ($arm)"
"$py" -m bench.kernels --arm "$arm" --threads "${KERNEL_THREADS:-1,all}" ${MAX_LOAD:+--max-load "$MAX_LOAD"}

echo "== agent loop with and without prefetch, Metric B ($arm)"
if [ "$(uname)" = Darwin ]; then : "${LOOP_THREADS:=1,all}"; fi
"$py" -m bench.loop --arm "$arm" --threads "${LOOP_THREADS:-1,2,4}" ${MAX_LOAD:+--max-load "$MAX_LOAD"}

echo "done: $out"
