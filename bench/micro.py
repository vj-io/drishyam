"""Per-op latency of the gaze and saliency path: p50/p95/p99 ms for each OpenCV call at three image sizes.

    python -m bench.micro --arm local-macos-x86 [--repeats 50] [--threads 1]

Writes results/<arm>/micro.json with a provenance block. Also home of `provenance()`, which the other
benchmarks and the evaluation harness import. Nothing here touches the network unless DRISHYAM_ON_EC2=1,
in which case instance type, AMI and id are read from IMDSv2 (never otherwise).
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

SIZES = ((1024, 768), (2048, 1536), (4000, 3000))
WARMUP = 5
IMDS = "http://169.254.169.254/latest"


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


def provenance() -> dict:
    where = os.path.realpath(cv2.__file__)
    cool = "/opt/cool" in where
    p = {
        "cv2": cv2.__version__,
        "build_info_sha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
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
    }
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


def run(arm: str, repeats: int = 50, threads: int = 1, out_dir="results", sizes=SIZES) -> dict:
    if repeats < 30:
        raise ValueError("at least 30 repeats")
    cv2.setNumThreads(threads)
    rows = []
    for w, h in sizes:
        for name, fn in ops(w, h).items():
            rows.append({"op": name, "size": f"{w}x{h}", **time_op(fn, repeats)})
            print(f"{name:18s} {w}x{h:<5d} p50 {rows[-1]['p50_ms']:9.3f} ms  p95 {rows[-1]['p95_ms']:9.3f} ms", flush=True)
    doc = {"arm": arm, "config": {"repeats": repeats, "warmup": WARMUP, "cv2_threads": cv2.getNumThreads(),
                                  "requested_threads": threads, "sizes": [f"{w}x{h}" for w, h in sizes],
                                  "gaze_view_px": 512},
           "provenance": provenance(), "results": rows}
    out = Path(out_dir) / arm / "micro.json"
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
    a = ap.parse_args(argv)
    run(a.arm, a.repeats, a.threads, a.out_dir)


if __name__ == "__main__":
    main()
