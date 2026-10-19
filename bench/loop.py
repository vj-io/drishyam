"""Metric B: the end-to-end agent loop, with and without asynchronous look prefetching (drishyam.prefetch).

    python -m bench.loop --arm local-macos-x86 [--threads 1,2,4] [--synth 8] [--rounds 1] [--max-load 4]

Writes results/<arm>/loop.json (loop_onnx.json when DRISHYAM_ONNX selects the dacl10k segmenter; the prefetch thread
then segments with a second instance, since one serialises its forwards on a lock). Photos: the eight demo samples
(public/samples) and `--synth` synthetic 2048x1536 photos (evals.synth seeds 9100 up, outside every eval split),
decoded before timing. Every photo runs once per variant per round, interleaved and rotated (off, warp, warp+segment;
then warp+segment, off, warp; ...), at a 30 % pixel cap with thumbnails off.

Decisions are held fixed, so every variant does the same looks: the governor's overrides are off, t-hat is a fixed
table (its value cancels out of the argmax and of the stop rule; only its being fixed matters), and the seconds cap is
raised to 120 s so a slow run cannot halt early. The run checks that every variant's gaze path and findings equal
prefetch-off's at the same thread count (`decisions_identical`).

Per thread count and variant: jobs/s (one process, sequential: photos / summed job wall time), p50/p95 job wall time,
p50/p95 step latency (wall time between consecutive step records: governor, choice, look, bookkeeping), process CPU
seconds per job (the prefetch thread's speculation included, wasted or not) and the prefetch counters.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from bench.micro import check_load, provenance, set_threads
from drishyam.agent import Job, JobConfig
from drishyam.prefetch import Prefetcher
from drishyam.segment import OnnxSegmenter, default_segmenter
from evals import synth

REPO = Path(__file__).resolve().parent.parent
ALL = cv2.getNumThreads()
VARIANTS = {"off": None, "warp": {"segment": False}, "warp+segment": {"segment": True}}
T_HAT = {"warp": 2.0, "segment": 60.0}
CFG = {"pixel_cap": 0.3, "thumbs": False, "governor": False, "t_hat_ms": T_HAT, "seconds_cap": 120.0}
K = 2


def photos(n_synth: int) -> dict:
    out = {p.stem: cv2.imread(str(p)) for p in sorted((REPO / "public" / "samples").glob("bridge_*.jpg"))}
    out.update({f"synth_{s}": synth.generate(s, 2048, 1536)[0] for s in range(9100, 9100 + n_synth)})
    return out


def one(img: np.ndarray, variant: str, bg_seg) -> dict:
    job = Job(img, JobConfig(**CFG))
    if VARIANTS[variant] is not None:
        job.prefetch = Prefetcher(K, segmenter=bg_seg, **VARIANTS[variant])
    cpu0, t0 = time.process_time(), time.perf_counter()
    marks = [time.perf_counter() for ev in job.run()
             if ev["type"] == "overview" or ev["type"] == "step" and ev["decision"]["action"] != "halt"]
    t1 = time.perf_counter()
    counts = {}
    if job.prefetch:
        job.prefetch.close()  # speculation still running is waited for, so its CPU lands in this job's cpu_s
        counts = dict(job.prefetch.counts)
    return {"job_ms": (t1 - t0) * 1000, "cpu_s": time.process_time() - cpu0, "step_ms": list(np.diff(marks) * 1000),
            "counts": counts, "decisions": json.dumps([job.gaze_path, [f.to_dict(crop=False) for f in job.findings]])}


def pct(xs, q) -> float:
    return round(float(np.percentile(xs, q)), 1)


def summarise(runs: list[dict], base: list[dict]) -> dict:
    jobs, steps = [r["job_ms"] for r in runs], [s for r in runs for s in r["step_ms"]]
    c = sum((Counter(r["counts"]) for r in runs), Counter())
    looks = c["warp_hit"] + c["warp_miss"]
    return {"jobs": len(runs), "steps": len(steps), "jobs_per_s": round(1000 * len(runs) / sum(jobs), 4),
            "job_ms": {"p50": pct(jobs, 50), "p95": pct(jobs, 95)}, "step_ms": {"p50": pct(steps, 50), "p95": pct(steps, 95)},
            "cpu_s_per_job": round(sum(r["cpu_s"] for r in runs) / len(runs), 3),
            "prefetch": {**dict(c), "warp_hit_rate": round(c["warp_hit"] / looks, 3) if looks else None},
            "decisions_identical": all(r["decisions"] == b["decisions"] for r, b in zip(runs, base, strict=True))}


def run(arm: str, threads: list[int], n_synth: int = 8, rounds: int = 1, out_dir="results",
        max_load: float | None = None) -> dict:
    check_load(max_load)
    prov = provenance()
    seg = default_segmenter()
    bg = OnnxSegmenter(*seg.files) if isinstance(seg, OnnxSegmenter) else None
    imgs = photos(n_synth)
    names = list(VARIANTS)
    table, decisions = [], {}
    for t in threads:
        got = set_threads(t)
        one(next(iter(imgs.values())), "warp+segment", bg)  # warm-up: nets, pools, caches
        res = {v: [] for v in names}
        for rnd in range(rounds):
            for i, img in enumerate(imgs.values()):
                k = (i + rnd) % len(names)
                for v in names[k:] + names[:k]:
                    res[v].append(one(img, v, bg))
        for v in names:
            table.append({"threads": got, "variant": v, **summarise(res[v], res["off"])})
            print(json.dumps(table[-1]), flush=True)
        decisions[got] = [r["decisions"] for r in res["off"]]
    same = all(d == decisions[threads[0]] for d in decisions.values())
    doc = {"arm": arm, "segmenter": seg.name, "config": {**CFG, "prefetch_k": K, "threads": threads, "all_threads": ALL,
                                                         "rounds": rounds, "photos": list(imgs)},
           "provenance": prov, "load_avg_end": check_load(), "decisions_identical_across_threads": same, "results": table}
    out = Path(out_dir) / arm / ("loop.json" if bg is None else "loop_onnx.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {out}")
    return doc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bench.loop", description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--threads", default="1,2,4", help="comma-separated cv2 thread counts; all = OpenCV's default")
    ap.add_argument("--synth", type=int, default=8, help="synthetic 2048x1536 photos besides the 8 samples")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-load", type=float, help="exit before timing if the 1-min load average is at or above this")
    a = ap.parse_args(argv)
    run(a.arm, [ALL if t == "all" else int(t) for t in a.threads.split(",")], a.synth, a.rounds, a.out_dir, a.max_load)


if __name__ == "__main__":
    main()
