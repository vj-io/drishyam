"""Per-op latency of the gaze and saliency path: p50/p95/p99 ms for each OpenCV call at three image sizes.

    python -m bench.micro --arm local-macos-x86 [--repeats 50] [--threads 1] [--job-ops-only] [--max-load 4]

Writes results/<arm>/micro.json with a provenance block. `results` holds single OpenCV calls; `job_ops` holds
the agent's own timed ops under the names its trace uses (segment:* on `default_segmenter()`, so DRISHYAM_ONNX
selects the engine), which `python -m evals.harness --cost-model fixed` charges per call. `--job-ops-only`
re-times just those, keeping `results`. `dnn` holds bare OpenCV DNN forwards: the NanoDet-Plus privacy detector
on its 416x416 input, and the segmentation ONNX that DRISHYAM_ONNX (FP32) and DRISHYAM_ONNX_FP16 name, read with
ENGINE_NEW at 256 and 512; every run times them, and `--dnn-only` re-times only them. Also home of `provenance()`, which the other benchmarks and the
evaluation harness import. Nothing here touches the network unless DRISHYAM_ON_EC2=1, in which case instance
type, AMI and id are read from IMDSv2 (never otherwise).

`--threads` is read back with getNumThreads() and the run exits if it did not take (`set_threads`); `--max-load`
refuses to start on a busy machine, and the file records the load at the start (provenance) and at the end.
Provenance quotes the Baseline, Dispatch, Parallel framework and Custom HAL lines of getBuildInformation(), and
BENCH_LABEL (if set) as `label`: where this ran, in words.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from drishyam import imaging, privacy
from drishyam.segment import default_segmenter, segment_view

SIZES = ((1024, 768), (2048, 1536), (4000, 3000))
WARMUP = 5
IMDS = "http://169.254.169.254/latest"
BUILD_LINES = {"Baseline": "cv2_baseline", "Dispatched code generation": "cv2_dispatch",  # getBuildInformation()
               "Parallel framework": "cv2_parallel_framework", "Custom HAL": "cv2_custom_hal"}  # -> provenance key


def cpu_model() -> str:
    try:
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False,
                                  timeout=5).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "hardware", "cpu model")):
                return line.split(":", 1)[1].strip()
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5, check=False).stdout  # Graviton has no model name
        return next((ln.split(":", 1)[1].strip() for ln in out.splitlines() if ln.startswith("Model name")), "unknown")
    except (OSError, subprocess.SubprocessError, IndexError):
        return platform.processor() or "unknown"


def ec2_metadata() -> dict:
    """Instance type, AMI and id via IMDSv2. Called only when DRISHYAM_ON_EC2=1; failures are recorded, not raised."""
    try:
        tok = urllib.request.urlopen(urllib.request.Request(
            f"{IMDS}/api/token", method="PUT", headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}), timeout=2).read().decode()
        get = lambda p: urllib.request.urlopen(urllib.request.Request(
            f"{IMDS}/meta-data/{p}", headers={"X-aws-ec2-metadata-token": tok}), timeout=2).read().decode()
        return {"instance_type": get("instance-type"), "ami_id": get("ami-id"), "instance_id": get("instance-id")}
    except Exception as exc:  # noqa: BLE001 -- provenance must never kill a benchmark
        return {"ec2_error": type(exc).__name__}


def build_lines() -> dict:
    """The getBuildInformation() lines that say which kernels run (SIMD baseline and dispatch, threading, HAL), verbatim."""
    out = {}
    for line in cv2.getBuildInformation().splitlines():
        key, _, val = line.strip().partition(":")
        if key in BUILD_LINES:
            out.setdefault(BUILD_LINES[key], val.strip())
    return out


def set_threads(n: int) -> int:
    """cv2.setNumThreads(n), read back with getNumThreads(); exits if OpenCV did not take it. On macOS's GCD framework
    getNumThreads() reports the CPU count for any n > 0, though setNumThreads(1) does run on one OS thread; one thread
    then uses setNumThreads(0) (also sequential), so the recorded value is the 1 that ran."""
    cv2.setNumThreads(n)
    if n == 1 and cv2.getNumThreads() != 1:
        cv2.setNumThreads(0)
    if (got := cv2.getNumThreads()) != n:
        raise SystemExit(f"cv2.setNumThreads({n}) left getNumThreads() = {got} (parallel framework "
                         f"{build_lines().get('cv2_parallel_framework')}): not recording a thread count that did not run")
    return got


def check_load(max_load: float | None = None) -> list[float]:
    """The 1/5/15-min load average now. With `max_load`, exit if the 1-min value is at or above it. The process that
    times calls it, so nothing can start between the check and the launch (a separate shell check raced other work)."""
    load = [round(v, 2) for v in os.getloadavg()]
    if max_load is not None and load[0] >= max_load:
        raise SystemExit(f"load average {load[0]} >= --max-load {max_load}: not timing on a busy machine")
    return load


def provenance() -> dict:
    where = os.path.realpath(cv2.__file__)
    cool = "/opt/cool" in where
    p = {
        "cv2": cv2.__version__,
        "build_info_sha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
        **build_lines(),
        "cv2_file": where,
        "under_opt_cool": cool,
        "build": "COOL (/opt/cool)" if cool else "stock (not COOL)",
        "machine": f"{platform.system()} {platform.release()} {platform.machine()}",
        "platform": platform.platform(),
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "cv2_threads": cv2.getNumThreads(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "load_avg": check_load(),  # 1/5/15 min: other work on the machine while this ran
    }
    if os.environ.get("BENCH_LABEL"):  # where this ran, in words (e.g. a CI runner, which is not the deployment)
        p["label"] = os.environ["BENCH_LABEL"]
    if os.environ.get("DRISHYAM_ON_EC2") == "1":
        p.update(ec2_metadata())
    return p


def test_image(w: int, h: int) -> np.ndarray:
    """Deterministic concrete-ish BGR frame: low-frequency mottling plus grain."""
    rng = np.random.default_rng(0)
    base = cv2.resize(rng.integers(60, 200, (h // 16 + 2, w // 16 + 2, 3), dtype=np.uint8), (w, h),
                      interpolation=cv2.INTER_CUBIC)
    return cv2.add(base, rng.integers(0, 24, (h, w, 3), dtype=np.uint8))


def ops(w: int, h: int) -> dict:
    """name -> zero-argument callable. Gaze ops render the 512x512 view the agent actually asks for."""
    img = test_image(w, h)
    gray, hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    view = 512
    m = np.array([[1, 0, w / 2 - view / 2], [0, 1, h / 2 - view / 2], [2e-6, 1e-6, 1]], np.float64)
    u, v = np.meshgrid(np.arange(view, dtype=np.float32), np.arange(view, dtype=np.float32))
    r2 = ((u - view / 2) ** 2 + (v - view / 2) ** 2) / view**2
    mx, my = u + w / 2 - view / 2 + 8 * r2, v + h / 2 - view / 2 + 8 * r2  # mild oblique/barrel de-warp
    warp_flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    return {
        "pyrDown": lambda: cv2.pyrDown(img),
        "warpPerspective": lambda: cv2.warpPerspective(img, m, (view, view), flags=warp_flags,
                                                       borderMode=cv2.BORDER_REFLECT_101),
        "remap": lambda: cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101),
        "resize": lambda: cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR),
        "resize_area": lambda: cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA),
        "GaussianBlur": lambda: cv2.GaussianBlur(gray, (5, 5), 0),
        "Sobel": lambda: cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        "adaptiveThreshold": lambda: cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                           cv2.THRESH_BINARY_INV, 21, 12),
        "morphologyEx": lambda: cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ellipse),
        "CLAHE": lambda: clahe.apply(gray),
        "inRange": lambda: cv2.inRange(hsv, (3, 70, 35), (25, 255, 235)),
        "cvtColor": lambda: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        "cvtColor_hsv": lambda: cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
    }


def job_ops(w: int, h: int, seg) -> dict:
    """The agent's timed ops on one w x h frame, keyed by the names in its trace (`Job._timed`), plus the
    downscale baseline's resize. Per-look ops read the 512x512 view the agent asks for."""
    img = test_image(w, h)
    pyr = imaging.build_pyramid(img)
    crop, _ = imaging.gaze(pyr, w // 2, h // 2, 0, size=512)
    valid = np.ones(crop.shape[:2], bool)
    k = 1024 / max(w, h)  # evals.harness.LONG_SIDE
    return {
        "pyrDown": lambda: imaging.build_pyramid(img),
        "saliency": lambda: imaging.saliency(pyr[imaging.OVERVIEW_LEVEL]),
        "warp": lambda: imaging.gaze(pyr, w // 2, h // 2, 0, size=512),
        "quality": lambda: imaging.quality(crop),
        "segment:accurate": lambda: segment_view(seg, crop, "accurate", valid),
        "segment:fast": lambda: segment_view(seg, crop, "fast", valid),
        "downscale": lambda: cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA),
    }


def dnn_ops() -> dict:
    """name -> (input size, forward). NanoDet always; the segmentation ONNX only when DRISHYAM_ONNX(_FP16) is set."""
    nano = cv2.dnn.readNet(str(privacy.NANODET))  # as privacy.py reads it
    names = nano.getUnconnectedOutLayersNames()
    nano.setInput(np.zeros((1, 3, privacy.NANO_SIZE, privacy.NANO_SIZE), np.float32))
    table = {"nanodet": (privacy.NANO_SIZE, lambda: nano.forward(names))}
    for label, var in (("fp32", "DRISHYAM_ONNX"), ("fp16", "DRISHYAM_ONNX_FP16")):
        if os.environ.get(var):
            for s in (256, 512):
                net = cv2.dnn.readNetFromONNX(os.environ[var], engine=cv2.dnn.ENGINE_NEW)
                net.setInput(np.random.default_rng(0).standard_normal((1, 3, s, s)).astype(np.float32))
                table[f"onnx:{label}@{s}"] = (s, net.forward)
    return table


def time_dnn(repeats: int) -> dict:
    rows = []
    for name, (s, fn) in dnn_ops().items():
        rows.append({"op": name, "size": f"{s}x{s}", **time_op(fn, repeats)})
        print(f"{name:18s} {s}x{s:<5d} p50 {rows[-1]['p50_ms']:9.3f} ms  p95 {rows[-1]['p95_ms']:9.3f} ms", flush=True)
    return {"onnx": os.environ.get("DRISHYAM_ONNX"), "onnx_fp16": os.environ.get("DRISHYAM_ONNX_FP16"),
            "provenance": provenance(), "results": rows}


def time_op(fn, repeats: int, warmup: int = WARMUP) -> dict:
    for _ in range(warmup):
        fn()
    ms = np.empty(repeats)
    for i in range(repeats):
        t = time.perf_counter_ns()
        fn()
        ms[i] = (time.perf_counter_ns() - t) / 1e6
    p50, p95, p99 = np.percentile(ms, (50, 95, 99))
    return {"n": repeats, "min_ms": round(float(ms.min()), 3), "p50_ms": round(float(p50), 3),
            "p95_ms": round(float(p95), 3), "p99_ms": round(float(p99), 3), "mean_ms": round(float(ms.mean()), 3)}


def _time_all(table, sizes, repeats) -> list[dict]:
    rows = []
    for w, h in sizes:
        for name, fn in table(w, h).items():
            rows.append({"op": name, "size": f"{w}x{h}", **time_op(fn, repeats)})
            print(f"{name:18s} {w}x{h:<5d} p50 {rows[-1]['p50_ms']:9.3f} ms  p95 {rows[-1]['p95_ms']:9.3f} ms", flush=True)
    return rows


def run(arm: str, repeats: int = 50, threads: int = 1, out_dir="results", sizes=SIZES, job_ops_only=False,
        dnn_only=False, max_load: float | None = None) -> dict:
    if repeats < 30:
        raise ValueError("at least 30 repeats")
    check_load(max_load)
    actual = set_threads(threads)
    out = Path(out_dir) / arm / "micro.json"
    if dnn_only:
        if not out.exists():
            raise SystemExit(f"--dnn-only adds to an existing {out}: run without it first")
        doc = {**json.loads(out.read_text()), "dnn": {**time_dnn(repeats), "load_avg_end": check_load()}}
        out.write_text(json.dumps(doc, indent=1))
        print(f"wrote {out}")
        return doc
    seg = default_segmenter()
    jobs = {"segmenter": seg.name, "cv2_threads": actual, "provenance": provenance(),
            "results": _time_all(lambda w, h: job_ops(w, h, seg), sizes, repeats)}
    if job_ops_only and out.exists():
        doc = {**json.loads(out.read_text()), "job_ops": jobs, "dnn": time_dnn(repeats)}
    else:
        doc = {"arm": arm, "config": {"repeats": repeats, "warmup": WARMUP, "cv2_threads": actual,
                                      "requested_threads": threads, "sizes": [f"{w}x{h}" for w, h in sizes],
                                      "gaze_view_px": 512},
               "provenance": provenance(), "results": [] if job_ops_only else _time_all(ops, sizes, repeats), "job_ops": jobs,
               "dnn": time_dnn(repeats)}
    doc["load_avg_end"] = check_load()  # the provenance blocks hold the load at the start of each section
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {out}")
    return doc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bench.micro", description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True, help="results/<arm>/micro.json")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--threads", type=int, default=1, help="cv2.setNumThreads; recorded in the output")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--job-ops-only", action="store_true", help="re-time only the agent's ops (and dnn); keep the rest")
    ap.add_argument("--dnn-only", action="store_true", help="re-time only the DNN forwards; keep the rest of micro.json")
    ap.add_argument("--max-load", type=float, help="exit before timing if the 1-min load average is at or above this")
    a = ap.parse_args(argv)
    run(a.arm, a.repeats, a.threads, a.out_dir, job_ops_only=a.job_ops_only, dnn_only=a.dnn_only, max_load=a.max_load)


if __name__ == "__main__":
    main()
