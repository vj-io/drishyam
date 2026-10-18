"""Macro benchmark: the full Drishyam job on N eval images at a fixed pixel budget, rule planner, thumbs off.

    python -m bench.macro --arm local --data data/synth200 --n 200 --pixel-cap 0.3 [--workers 1] [--threads 1] [--max-load 4]

Writes results/<arm>/macro.json (macro_w<N>.json when workers > 1): images/s, gazes/s, per-image p50/p95,
CPU use, $/1k images = usd_per_hour x hours / images, and an Amdahl split of the time into the imgproc ops
KleidiCV covers (pyramid, saliency, gaze warp, quality gate), the segmenter, image decode and Python overhead.
Throughput is wall time from the first job's start to the last job's end, so pool start-up is excluded.
Each worker's thread count is set and read back as in bench.micro (`set_threads`); config records both.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from bench.micro import check_load, provenance, set_threads
from drishyam import agent as agent_mod
from drishyam.agent import Job, JobConfig
from drishyam.governor import USD_PER_HOUR
from evals.harness import load_records

COVERED = ("pyrDown", "saliency", "warp", "quality")  # imgproc ops on the KleidiCV path
AMDAHL_SPEEDUPS = (1.2, 1.5, 1.8, 3.0)


_MEASURE_MS = [0.0]
_measure = agent_mod.measure


def _timed_measure(*args, **kw):
    """Benchmark-side shim (the core is untouched): time `measure()`, which RuntimeStats does not see."""
    t = time.perf_counter()
    try:
        return _measure(*args, **kw)
    finally:
        _MEASURE_MS[0] += (time.perf_counter() - t) * 1000


agent_mod.measure = _timed_measure


def _init(threads: int, warm_path: str, cap: float) -> None:
    set_threads(threads)
    list(Job(cv2.imread(warm_path), JobConfig(pixel_cap=cap, thumbs=False)).run())


def one(task: tuple[str, float]) -> dict:
    path, cap = task
    cpu0, t0 = time.process_time(), time.perf_counter()
    _MEASURE_MS[0] = 0.0
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    t_dec = time.perf_counter()
    job = Job(img, JobConfig(pixel_cap=cap, thumbs=False))
    done = list(job.run())[-1]
    t1 = time.perf_counter()
    samples = job.stats.samples
    ops = {op: float(sum(samples[op])) for op in COVERED}
    ops["segment"] = float(sum(sum(v) for k, v in samples.items() if k.startswith("segment:")))
    ops["measure"] = _MEASURE_MS[0]
    job_ms = (t1 - t_dec) * 1000
    return {"path": Path(path).name, "t0": t0, "t1": t1, "wall_ms": (t1 - t0) * 1000, "decode_ms": (t_dec - t0) * 1000,
            "job_ms": job_ms, "steps": done["summary"]["steps"], "pixels_frac": done["summary"]["pixels_frac"],
            "cpu_s": time.process_time() - cpu0, "ops_ms": ops,
            "python_ms": max(0.0, job_ms - sum(ops.values())), "cv2_threads": cv2.getNumThreads()}


def amdahl(rows: list[dict]) -> dict:
    total = sum(r["wall_ms"] for r in rows)
    parts = {"kleidicv_covered_imgproc": sum(sum(r["ops_ms"][o] for o in COVERED) for r in rows),
             "segmentation": sum(r["ops_ms"]["segment"] for r in rows),
             "measure_thinning": sum(r["ops_ms"]["measure"] for r in rows),
             "decode": sum(r["decode_ms"] for r in rows), "python_overhead": sum(r["python_ms"] for r in rows)}
    f = parts["kleidicv_covered_imgproc"] / total
    return {"fraction": {k: round(v / total, 4) for k, v in parts.items()},
            "ms_per_image": {k: round(v / len(rows), 2) for k, v in parts.items()},
            "end_to_end_speedup_if_covered_ops_get": {f"{s}x": round(1 / (1 - f + f / s), 4) for s in AMDAHL_SPEEDUPS},
            "note": ("segmentation is the classical segmenter (imgproc plus NumPy), so only part of it is KleidiCV-covered; "
                     "measure_thinning is drishyam.measure.measure (hit-or-miss thinning), timed by a shim here; "
                     "python_overhead is whatever the job does outside every timed op. The covered fraction is a lower bound")}


def run(arm: str, data: str, n: int, cap: float, workers: int, threads: int, usd_per_hour: float, out_dir="results",
        max_load: float | None = None) -> dict:
    check_load(max_load)
    prov = provenance()  # its load_avg is the load at the start
    records, label = load_records(data, n)
    tasks = [(r["path"], cap) for r in records]
    if workers <= 1:
        _init(threads, tasks[0][0], cap)
        rows = [one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init,
                                 initargs=(threads, tasks[0][0], cap)) as ex:
            rows = list(ex.map(one, tasks, chunksize=1))
    wall = max(r["t1"] for r in rows) - min(r["t0"] for r in rows)
    lat = [r["wall_ms"] for r in rows]
    cpu = sum(r["cpu_s"] for r in rows)
    summary = {
        "dataset": label, "images": len(rows), "pixel_cap": cap, "wall_s": round(wall, 3),
        "images_per_s": round(len(rows) / wall, 4), "gazes_per_s": round(sum(r["steps"] for r in rows) / wall, 4),
        "per_image_ms": {"p50": round(float(np.percentile(lat, 50)), 1), "p95": round(float(np.percentile(lat, 95)), 1),
                         "p99": round(float(np.percentile(lat, 99)), 1)},
        "mean_steps": round(float(np.mean([r["steps"] for r in rows])), 2),
        "mean_pixels_frac": round(float(np.mean([r["pixels_frac"] for r in rows])), 4),
        "cpu_s": round(cpu, 2), "cpu_pct_of_machine": round(100 * cpu / (wall * (os.cpu_count() or 1)), 1),
        "cpu_pct_per_worker_busy": round(100 * cpu / sum(r["wall_ms"] for r in rows) * 1000, 1),
        "usd_per_hour": usd_per_hour, "usd_per_1k_images": round(usd_per_hour * wall / 3600 / len(rows) * 1000, 6),
    }
    actual = sorted({r["cv2_threads"] for r in rows})  # read back in each worker, after set_threads
    doc = {"arm": arm, "config": {"workers": workers, "requested_threads": threads,
                                  "cv2_threads_per_worker": actual[0] if len(actual) == 1 else actual, "planner": "rule"},
           "provenance": prov, "load_avg_end": check_load(), "summary": summary, "amdahl": amdahl(rows), "per_image": rows}
    out = Path(out_dir) / arm / ("macro.json" if workers == 1 else f"macro_w{workers}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps({"summary": summary, "amdahl": doc["amdahl"]}, indent=1))
    print(f"wrote {out}")
    return doc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bench.macro", description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--data", default="data/synth200")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--pixel-cap", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1, help="cv2.setNumThreads per worker")
    ap.add_argument("--usd-per-hour", type=float, default=USD_PER_HOUR, help="instance + COOL fee, default c8g.2xlarge $0.339")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-load", type=float, help="exit before timing if the 1-min load average is at or above this")
    a = ap.parse_args(argv)
    run(a.arm, a.data, a.n, a.pixel_cap, a.workers, a.threads, a.usd_per_hour, a.out_dir, a.max_load)


if __name__ == "__main__":
    main()
